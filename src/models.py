import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.aggr import AttentionalAggregation
from src.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, 
                 num_class, 
                 num_layer = 5, 
                 emb_dim = 300, 
                gnn_type = 'gin', 
                virtual_node = True, 
                residual = False, 
                drop_ratio = 0.5, 
                JK = "last", 
                graph_pooling = "mean",
                ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        elif self.graph_pooling == "attention_mean":
            self.att_pool = AttentionalAggregation(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "max_sum":
            self.pool = None 
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        if(self.graph_pooling == "attention_mean"):
            h_graph = self.att_pool(h_node, batched_data.batch)+global_mean_pool(h_node, batched_data.batch)
        elif(self.graph_pooling == "max_sum"):
            h_graph = global_add_pool(h_node, batched_data.batch)+global_max_pool(h_node, batched_data.batch)
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        self.embeddings = h_graph.detach()

        return self.graph_pred_linear(h_graph)
    

class GCODWrapper(torch.nn.Module):
    def __init__(self, gnn_model, dataset_size):
        super().__init__()
        self.gnn = gnn_model
        self.uB = torch.nn.Parameter(torch.zeros(dataset_size))
    
    def forward(self, x):
        logits = self.gnn(x)
        return logits
