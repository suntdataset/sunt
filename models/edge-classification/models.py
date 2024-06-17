import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GATEdgeClassifier, self).__init__()
        self.conv1 = gnn.GATConv(in_channels, hidden_channels, heads)
        self.conv2 = gnn.GATConv(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


class SAGEEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGEEdgeClassifier, self).__init__()
        self.conv1 = gnn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = gnn.SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


class GCNEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEdgeClassifier, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        return self.classifier(edge_embeddings)


class ChebEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, filter_size=3):
        super(ChebEdgeClassifier, self).__init__()
        self.conv1 = gnn.ChebConv(in_channels, hidden_channels, K=filter_size)
        self.conv2 = gnn.ChebConv(hidden_channels, hidden_channels, K=filter_size)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)





