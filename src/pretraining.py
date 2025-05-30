# src/pretraining.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.transforms import EdgeDropout, NodeDropout, Compose

# Utility: assicurarsi che esistano feature nodali

def add_zeros(data):
    if data.x is None or data.x.numel() == 0:
        # crea un indice 0 per ogni nodo
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

class NTXentLoss(nn.Module):
    """
    NT-Xent loss per SimCLR.
    Usa temperature scaling e normalizzazione.
    """
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # maschera diagonale per escludere i casi i==j
        mask = (~torch.eye(2 * N, device=sim.device, dtype=torch.bool)).float()
        exp_sim = torch.exp(sim) * mask
        denom = exp_sim.sum(dim=1)  # [2N]

        # similarità positive
        pos = torch.exp((z1 * z2).sum(dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)  # [2N]

        loss = -torch.log(pos / denom)
        return loss.mean()

class GraphCLTrainer:
    """
    Trainer per pre-training contrastivo SimCLR-style su GNN.
    Applica due viste con Edge/NodeDropout e ottimizza NT-Xent loss.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        temperature: float = 0.2,
        p_edge: float = 0.2,
        p_node: float = 0.2,
    ):
        self.device = device
        self.model = model.to(device)

        # Rimuovi temporaneamente la testa di classificazione se esiste
        if hasattr(self.model, 'graph_pred_linear'):
            self._orig_head = self.model.graph_pred_linear
            self.model.graph_pred_linear = nn.Identity().to(device)

        # Dimensione dell'embedding dal modello (dopo readout)
        # Adesso model(g) restituisce direttamente l'embedding di dim emb_dim
        emb_dim = getattr(model, 'emb_dim', None)
        if emb_dim is None:
            raise AttributeError("Model must have attribute 'emb_dim' for GraphCL pretraining")

        # Proiettore MLP
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        ).to(device)

        self.criterion = NTXentLoss(temperature).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr
        )

        # augmentazioni
        self.augment = Compose([
            EdgeDropout(p=p_edge),
            NodeDropout(p=p_node),
        ])

    def pretrain(self, loader: DataLoader, epochs: int = 10):
        self.model.train()
        self.projector.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in loader:
                # due viste augmentate con feature
                batch = add_zeros(batch)
                g1 = add_zeros(self.augment(batch.clone())).to(self.device)
                g2 = add_zeros(self.augment(batch.clone())).to(self.device)

                # estrai embedding (il modello restituisce dim=[B, emb_dim])
                z1 = self.model(g1)
                z2 = self.model(g2)

                # proiezione
                p1 = self.projector(z1)
                p2 = self.projector(z2)

                # loss contrastiva
                loss = self.criterion(p1, p2)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"[GraphCL] Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

    def restore_head(self):
        """
        Ripristina la testa di classificazione originale sul modello.
        Chiamare dopo il pretraining se si vuole continuare il fine-tuning.
        """
        if hasattr(self, '_orig_head'):
            self.model.graph_pred_linear = self._orig_head.to(self.device)
