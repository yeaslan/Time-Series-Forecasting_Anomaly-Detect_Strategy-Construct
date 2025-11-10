from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]) -> None:
        self.features = arrays["features"]  # (n_windows, lookback, n_features)
        self.targets = arrays["targets"]  # (n_windows, target_dim)
        self.masks = arrays.get("masks")  # (n_windows, lookback)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        feats = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32) if self.masks is not None else torch.ones(feats.shape[0])
        return feats, target, mask, {}


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    lookback: int,
    horizon: int,
    groupby: str = "ticker",
) -> Dict[str, np.ndarray]:
    window_features: List[np.ndarray] = []
    window_targets: List[np.ndarray] = []
    window_masks: List[np.ndarray] = []

    for ticker, subdf in df.groupby(groupby):
        subdf = subdf.sort_values("timestamp")
        feats = subdf[feature_cols].to_numpy(dtype=np.float32)
        targets = subdf[target_cols].to_numpy(dtype=np.float32)
        mask = (~np.isnan(feats)).astype(np.float32)

        n_obs = feats.shape[0]
        for start in range(0, n_obs - lookback - horizon + 1):
            end = start + lookback
            target_idx = end + horizon - 1
            window_features.append(feats[start:end])
            window_targets.append(targets[target_idx])
            mask_window = mask[start:end].all(axis=1).astype(np.float32)
            window_masks.append(mask_window)

    if not window_features:
        return {
            "features": np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            "targets": np.empty((0, len(target_cols)), dtype=np.float32),
            "masks": np.empty((0, lookback), dtype=np.float32),
        }
    return {
        "features": np.stack(window_features, axis=0),
        "targets": np.stack(window_targets, axis=0),
        "masks": np.stack(window_masks, axis=0),
    }
