import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric import nn as gnn

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        #x = F.log_softmax(x, dim=1)
        
        return x      
   

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = gnn.GATConv(in_channels, hidden_channels, heads)
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.linear(x)
        
        return x


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.SAGEConv(in_channels, hidden_channels)
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear_out(x)

        return x

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, filter_size=3):
        super().__init__()
        self.conv1 = gnn.ChebConv(in_channels, hidden_channels, K=filter_size)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor,  edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x)

        return x