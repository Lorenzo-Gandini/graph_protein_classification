import torch
from torch_geometric.data import Data
import numpy as np

class EdgeDropout:
    def __init__(self, p: float):
        """
        Rimuove casualmente una frazione p di archi.
        """
        assert 0.0 <= p < 1.0
        self.p = p

    def __call__(self, data: Data) -> Data:
        # numero di archi (colonne di edge_index)
        num_edges = data.edge_index.size(1)
        # mask sui singoli archi
        mask_edges = torch.rand(num_edges, device=data.edge_index.device) > self.p

        # applica la mask
        data.edge_index = data.edge_index[:, mask_edges]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask_edges]

        return data


class NodeDropout:
    def __init__(self, p: float):
        assert 0.0 <= p < 1.0
        self.p = p
    def __call__(self, data: Data) -> Data:
        
        num_nodes = data.num_nodes
        mask_nodes = torch.rand(num_nodes, device=data.edge_index.device) > self.p
        kept = mask_nodes.nonzero(as_tuple=False).view(-1)
        new_index = -torch.ones(num_nodes, dtype=torch.long, device=data.edge_index.device)
        new_index[kept] = torch.arange(kept.size(0), device=new_index.device)
        src, dst = data.edge_index
        keep_edge = mask_nodes[src] & mask_nodes[dst]
        src, dst = src[keep_edge], dst[keep_edge]
        
        data.edge_index = torch.stack([new_index[src], new_index[dst]], dim=0)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep_edge]
        if data.x is not None:
            data.x = data.x[kept]
        data.num_nodes = kept.size(0)
        return data


class GraphMixUp:
    def __init__(self, alpha: float = 1.0):
        """
        alpha: parametro della Beta(alpha, alpha)
        """
        assert alpha > 0, "alpha deve essere > 0"
        self.alpha = alpha

    def __call__(self, batch):
        # batch.y è [B], batch.batch delineia i grafi nel batch
        lam = float(np.random.beta(self.alpha, self.alpha))
        # Permutazione dei grafi all'interno del batch
        perm = torch.randperm(batch.y.size(0), device=batch.y.device)

        # Salvo gli y “originali” e quelli mixati
        batch.y_a = batch.y
        batch.y_b = batch.y[perm]
        batch.lam = lam
        batch.perm = perm

        return batch




class Compose:
    """
    Applica sequenzialmente una lista di trasformazioni.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data
