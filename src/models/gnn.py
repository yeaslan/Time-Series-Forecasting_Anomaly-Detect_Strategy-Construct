from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class GCNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        pooled = global_mean_pool(x, data.batch)
        return self.head(pooled)


class GATModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, heads: int = 2, dropout: float = 0.1, output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            in_dim = hidden_dim * heads
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        pooled = global_mean_pool(x, data.batch)
        return self.head(pooled)
