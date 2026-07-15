import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class SimpleGCNEncoder(nn.Module):
    """
    Minimal GCN encoder for producing node-level embeddings
    from a PyTorch Geometric graph.

    Input:
        data.x:          [num_nodes, num_node_features]
        data.edge_index: [2, num_edges]

    Output:
        node embeddings: [num_nodes, embedding_dim]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        embedding_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x