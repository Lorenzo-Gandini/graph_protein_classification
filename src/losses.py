import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy: L_q = (1 - p_y^q) / q
    per q in (0,1]; con q→0 diventa MAE su probabilità.
    """
    def __init__(self, q=0.7):
        super().__init__()
        assert 0 < q <= 1
        self.q = q

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        prob = F.softmax(logits, dim=1)
        p_y = prob[range(len(targets)), targets]
        loss = (1 - p_y**self.q) / self.q
        return loss.mean()


class SymmetricCELoss(nn.Module):
    """
    SCE = alpha * CE + beta * ReverseCE
    ReverseCE = -sum( p * log(q) ) where p = one-hot(target), q = softmax(logits)
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def reverse_ce(self, logits, targets, reduction="mean"):
        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=prob.size(1)).float()
        # - \sum_i q_i log(p_i)
        
        return -(prob * torch.log(one_hot + 1e-8)).sum(dim=1) if reduction=="none" else -(prob * torch.log(one_hot + 1e-8)).sum(dim=1).mean() 

    def forward(self, logits, targets, reduction="mean"):
         
        loss_ce  = nn.CrossEntropyLoss(reduction=reduction)(logits, targets)
        loss_rce = self.reverse_ce(logits, targets, reduction=reduction)
        return self.alpha * loss_ce + self.beta * loss_rce

def estimate_transition_matrix(model, loader, tau, device, num_class):
    """
    Stima la matrice di transizione T usando i sample con loss ≤ tau.
    T[i,j] = stima di P(noisy_label=j | true_label=i) su anchor set.
    """
    model.eval()
    counts = torch.zeros(num_class, device=device)
    T = torch.zeros(num_class, num_class, device=device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            losses = F.cross_entropy(logits, data.y, reduction='none')
            prob = F.softmax(logits, dim=1)

            mask = losses <= tau
            if mask.sum() == 0:
                continue

            sel_prob = prob[mask]         # [K, C]
            sel_labels = data.y[mask]     # [K]

            # Accumula
            for k, y in enumerate(sel_labels):
                T[y] += sel_prob[k]
                counts[y] += 1

    # Normalizza per riga (true class)
    for c in range(num_class):
        if counts[c] > 0:
            T[c] /= counts[c]
        else:
            # se non ci sono sample, fallback all'identità
            T[c, c] = 1.0

    return T


class ForwardCorrectionLoss(nn.Module):
    """
    Forward correction loss: 
    p = softmax(logits), p_corr = p @ T, NLL su p_corr.
    """
    def __init__(self, T: torch.Tensor):
        super().__init__()
        # T sarà registrata come buffer, si muove automaticamente su gpu/cpu con model
        self.register_buffer('T', T)

    def forward(self, logits, labels, reduction="mean"):
        prob = F.softmax(logits, dim=1)             # [B, C]
        corrected = torch.clamp(prob @ self.T, 1e-7) # [B, C]
        log_p = torch.log(corrected)
        loss = F.nll_loss(log_p, labels, reduction=reduction)
        return loss

class BootstrappingLoss(nn.Module):
    def __init__(self, beta: float = 0.8, reduction: str = 'mean'):
        """
        beta: peso del target ‘vero’ rispetto alla predizione corrente
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        prob = F.softmax(logits, dim=1)
        # one‐hot del target
        one_hot = torch.zeros_like(prob).scatter(1, target.unsqueeze(1), 1.0)
        # distribuzione mista
        q = self.beta * one_hot + (1 - self.beta) * prob
        loss = (- q * torch.log(prob + 1e-7)).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
    
    
class EarlyLearningRegularizationLoss(nn.Module):
    def __init__(self, num_samples, num_classes, beta=0.9, lambda_elr=1.0, device=None):
        super().__init__()
        self.beta = beta
        self.lambda_elr = lambda_elr
        self.num_classes = num_classes

        # Initialize targets with uniform distribution
        self.register_buffer('targets', torch.full((num_samples, num_classes),
                            1.0/num_classes,
                            device=device))

    def forward(self, logits, labels, indices):

        # 1. Compute standard cross-entropy
        ce_loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            
            # 2. Update targets (detached)
            current_preds = F.softmax(logits.detach(), dim=1)
            self.targets[indices] = (self.beta * self.targets[indices] +
                                   (1 - self.beta) * current_preds)

        # 3. Compute ELR term with numerical stability
        probs = F.softmax(logits, dim=1)
        dot_product = (probs * self.targets[indices]).sum(dim=1)

        # Clamp to avoid log(0) or log(negative)
        elr_term = -torch.log(torch.clamp(1.0 - dot_product, min=1e-8, max=1.0))
        elr_loss = elr_term.mean()

        return ce_loss + self.lambda_elr * elr_loss
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class GCODLoss(torch.nn.Module):
    def __init__(self, num_classes, batch_size, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, logits, uB, labels, train_acc):
        """
        logits: [B, C] - output from the model
        labels: [B] - ground-truth class indices
        train_acc: float between 0 and 1
        """
        B, C = logits.size()

        current_uB = uB

        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            yB = F.one_hot(labels, num_classes=C).float()  # [B, C]
        else:
            yB = labels.float()
        
        # 1. L1: Confidence-weighted CrossEntropy Loss
        # Paper uses diag(uB)·yB, implemented as element-wise multiplication
        diag_u_y = current_uB.unsqueeze(1) * yB  # [B, C]

        adjusted_logits = logits + train_acc * diag_u_y
        loss1 = F.cross_entropy(adjusted_logits, labels)

        # 2. L2: Consistency Loss (MSE normalized by 1/|C|)
        probs = F.softmax(logits, dim=1)  # ˆyB
        adjusted_probs = probs + current_uB.unsqueeze(1) * yB

        # Paper shows squared L2 norm divided by num_classes
        loss2 = torch.norm(adjusted_probs - yB, p=2)**2 / C

        # 3. L3: KL Divergence regularization
        # L = log(σ(diag(fθ(ZB)yB^T)))
        diag_elements = torch.sum(logits * yB, dim=1)  # [B]
        L_logprob = torch.log(torch.sigmoid(diag_elements).clamp(min=1e-7, max=1 - 1e-7))  # [B]

        # σ(-log(uB)) → Probabilities (target)
        uB_clamped = torch.clamp(current_uB, min=1e-7, max=1 - 1e-7)
        uB_reg = torch.sigmoid(-torch.log(uB_clamped))  # [B]

        # Now compute KL divergence
        loss3 = F.kl_div(
            input=L_logprob,       # Already log-probabilities
            target=uB_reg,         # Probabilities
            reduction='batchmean',
            log_target=False       # Because target is not log
        )

        # Weight with (1 - train_acc)
        loss3 = (1 - train_acc) * loss3

        # Combine losses
        total_loss = loss1 + loss2 + loss3

        return total_loss, loss1, loss2, loss3