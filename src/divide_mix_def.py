import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.mixture import GaussianMixture
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
import random 

class DivideMixTrainer:
    """
    Cleaned DivideMix implementation for graph classification with noisy labels.
    - Parameterized criterion, optimizers, scheduler
    - Simplified unlabeled sampling
    - Flexible pretrain and augmentation integration
    """
    def __init__(
        self,
        model_creator,
        model_kwargs: dict,
        criterion: nn.Module,
        optim_sched_creator,
        optim_sched_kwargs: dict,
        device: str = 'cuda',
        lambda_u: float = 75,
        T: float = 0.5,
        alpha: float = 4.0,
        warmup_epochs: int = 10,
        p_threshold: float = 0.7,
        num_models: int = 2,
        num_classes: int = 6,
        rampup_epochs: int = 80
    ):
        self.device = device
        self.num_models = num_models
        self.criterion = criterion
        self.lambda_u = lambda_u
        self.T = T
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.p_threshold = p_threshold
        self.num_classes = num_classes
        self.rampup_epochs = rampup_epochs 
        # Initialize models, optimizers, schedulers
        self.models = []
        self.optimizers = []
        self.schedulers = []
        for _ in range(num_models):
            model = model_creator(**model_kwargs).to(device)
            optim, sched = optim_sched_creator(model, **optim_sched_kwargs)
            self.models.append(model)
            self.optimizers.append(optim)
            self.schedulers.append(sched)

    def _fit_gmm(self, losses: np.ndarray) -> np.ndarray:
        gmm = GaussianMixture(n_components=2, max_iter=100, reg_covar=5e-4)
        gmm.fit(losses.reshape(-1,1))
        probs = gmm.predict_proba(losses.reshape(-1,1))
        clean_comp = np.argmin(gmm.means_.flatten())
        return probs[:, clean_comp]

    def warmup_phase(self, train_dataset, batch_size):
        print(f"Warmup phase ({self.warmup_epochs} Epochs)")
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        final_losses = [[] for _ in range(self.num_models)]
        lambda_entropy = 0.1  # Entropy regularization weight

        for epoch in range(self.warmup_epochs):
            for model in self.models:
                model.train()
            for data in tqdm(loader):
                data = data.to(self.device)
                for model_index, (model, optim) in enumerate(zip(self.models, self.optimizers)):
                    optim.zero_grad()

                    logits = model(data)
                
                    # Per-sample cross entropy loss
                    loss_per_sample = F.cross_entropy(logits, data.y, reduction='none')
                    ce_loss = loss_per_sample.mean()
                    
                    pred_probs = F.softmax(logits, dim=1)
                    # Negative Entropy loss (encourages less confident predictions)
                    entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=1)
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = ce_loss + lambda_entropy * entropy_loss

                    loss.backward()
                    optim.step()

                    if epoch == self.warmup_epochs - 1:
                        final_losses[model_index].extend(loss_per_sample.detach().cpu())

        prob_clean = [self._fit_gmm(np.array(losses)) for losses in final_losses]
        return prob_clean

    def align_edge_counts(self,batch1, batch2):
        """Truncate larger batch to match smaller batch's edge count."""
        min_edges = min(len(batch1.edge_attr), len(batch2.edge_attr))
        batch1.edge_index = batch1.edge_index[:, :min_edges]
        batch1.edge_attr = batch1.edge_attr[:min_edges]
        batch2.edge_index = batch2.edge_index[:, :min_edges]
        batch2.edge_attr = batch2.edge_attr[:min_edges]
        return batch1, batch2

    def graph_mixup_edge(self,clean_batch, unlabeled_batch, pseudo_labels, alpha=1.0):
        """Safe edge-level Mixup with alignment checks."""
        # Align edge counts (Option 1 or 2 above)
        clean_batch, unlabeled_batch = self.align_edge_counts(clean_batch, unlabeled_batch)
        
        # Verify alignment
        assert len(clean_batch.edge_attr) == len(unlabeled_batch.edge_attr), "Edges still misaligned!"
        assert torch.all(clean_batch.edge_index == unlabeled_batch.edge_index), "Topology mismatch!"

        # Mix edge features
        lam = np.random.beta(alpha, alpha)
        mixed_edge_attr = lam * clean_batch.edge_attr + (1 - lam) * unlabeled_batch.edge_attr

        # Mixed labels (graph/node-level)
        mixed_y = lam * clean_batch.y + (1 - lam) * pseudo_labels

        # Return mixed batch (preserve original topology)
        return Batch(
            x=clean_batch.x,
            edge_index=clean_batch.edge_index,
            edge_attr=mixed_edge_attr,
            y=mixed_y,
            batch=clean_batch.batch
        ), lam

    def train_dividemix_epoch(self, dataset, prob_clean, batch_size, current_epoch):
        # --- Step 1: Split into Clean/Noisy Subsets (Co-Teaching) ---
        clean_subsets = []
        unlabeled_subsets = []
        
        for mi in range(self.num_models):
            other = (mi + 1) % self.num_models  # Get the other model's indices
            mask = prob_clean[other] > self.p_threshold
            clean_idx = np.where(mask)[0].tolist()
            noisy_idx = np.where(~mask)[0].tolist()
            
            clean_subsets.append([dataset[i] for i in clean_idx])
            unlabeled_subsets.append([dataset[i] for i in noisy_idx])

        # --- Step 2: Initialize Trackers ---
        sup_loss_acc = 0.0
        unsup_loss_acc = 0.0
        n_batches = 0

        # --- Step 3: Ramp-up lambda_u (Dynamic Weight) ---
        lambda_u = min(current_epoch / self.rampup_epochs, 1.0) * self.lambda_u

        # --- Step 4: Training Loop ---
        for _ in tqdm(range(len(dataset) // batch_size)):
            for mi in range(self.num_models):
                self.models[mi].train()
                
                # --- (A) Sample Clean (Labeled) Batch ---
                clean_subset = clean_subsets[mi]
                if len(clean_subset) == 0:
                    continue  # Skip if no clean data
                
                clean_batch = Batch.from_data_list(
                    random.sample(clean_subset, min(batch_size, len(clean_subset)))
                )
                clean_batch = clean_batch.to(self.device)

                # --- (B) Sample Noisy (Unlabeled) Batch ---
                unlabeled_subset = unlabeled_subsets[mi]
                if len(unlabeled_subset) == 0:
                    continue
                
                unlabeled_batch = Batch.from_data_list(
                    random.sample(unlabeled_subset, min(batch_size, len(unlabeled_subset))))
                unlabeled_batch = unlabeled_batch.to(self.device)

                # --- (C) Generate Pseudo-Labels for Noisy Data ---
                logits_s = self.models[mi](clean_batch)
                logits_u = self.models[mi](unlabeled_batch)
                pseudo_labels = F.softmax(logits_u / self.T, dim=1)  # Temperature sharpening

                # --- (D) Mix ONLY LABELS (y) ---
                lam = np.random.beta(self.alpha, self.alpha)
                
                # --- (E) Forward Pass & Loss Computation ---
                self.optimizers[mi].zero_grad()
                
                mix_logits = lam * logits_u + (1 - lam) * torch.log(pseudo_labels + 1e-6)

                # Supervised loss (clean portion)
                sup_loss = F.cross_entropy(logits_s, clean_batch.y)
                
                # Unsupervised loss (noisy portion)
                unsup_loss = F.cross_entropy(mix_logits, pseudo_labels)

                # Total loss
                loss = sup_loss + lambda_u * unsup_loss
                loss.backward()
                self.optimizers[mi].step()

                # --- (F) Update Trackers ---
                sup_loss_acc += sup_loss.item()
                unsup_loss_acc += unsup_loss.item()
                total_loss += loss.item()
                n_batches += 1

        # --- Step 5: Return Average Losses ---
        avg_sup_loss = sup_loss_acc / n_batches if n_batches > 0 else 0.0
        avg_unsup_loss = unsup_loss_acc / n_batches if n_batches > 0 else 0.0
        total_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return total_loss, avg_sup_loss, avg_unsup_loss

    def evaluate(self, val_loader):
        correct = total = 0
        for model in self.models:
            model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                logits = [F.softmax(m(data), dim=1) for m in self.models]
                ens = sum(logits) / len(self.models)
                preds = ens.argmax(dim=1)
                correct += (preds == data.y).sum().item()
                total += data.num_graphs
        return correct / total

    def train_full_pipeline(self, train_dataset, val_dataset, epochs, batch_size, patience):
        # Warm-up
        prob_clean = self.warmup_phase(train_dataset, batch_size)
        best_acc = 0.0
        history = {'sup_loss': [], 'unsup_loss': [], 'val_acc': [], 'train_acc': []}
        best_models = []
        patience_counter = 0
        print("End of Warmup Stage")
        for epoch in range(1, epochs+1):
            # Update schedulers
            for sched in self.schedulers:
                sched.step()
            total, sup, unsup = self.train_dividemix_epoch(train_dataset, prob_clean, batch_size, epoch)

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            val_acc = self.evaluate(val_loader)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            train_acc = self.evaluate(train_loader)

            # Log metrics
            history['total_loss'].append(total)
            history['sup_loss'].append(sup)
            history['unsup_loss'].append(unsup)
            history['val_acc'].append(val_acc)
            history['train_acc'].append(train_acc)
            
            print(f"Epoch {epoch}/{epochs} | Total={total:.4f} | Sup={sup:.4f} | Unsup={unsup:.4f} | TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} (Best={best_acc:.4f})")
                # Early stopping check
            if val_acc > best_acc:
                best_acc = val_acc
                best_models = [m.state_dict() for m in self.models]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
            
        if best_models:
            for m, best_state in zip(self.models, best_models):
                m.load_state_dict(best_state)

        return self.models
    
    def predict_ensemble(models, dataloader, device):
        all_logits = []
        for i, model in enumerate(models):
            print(f"Evaluating model {i}")
            model.eval()
            model.to(device)
            logits = []
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = batch.to(device)
                    out = model(batch)
                    logits.append(out)
            all_logits.append(torch.cat(logits, dim=0))
        
        avg_logits = sum(all_logits) / len(models)
        preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
        return preds
