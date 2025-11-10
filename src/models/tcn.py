from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        res = self.downsample(x)
        out = out + res[:, :, -out.size(2):]
        return out


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            layers.append(TemporalConvBlock(in_channels, out_channels, kernel_size, dilation_size, dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # take last timestep
        out = self.head(out)
        return out
