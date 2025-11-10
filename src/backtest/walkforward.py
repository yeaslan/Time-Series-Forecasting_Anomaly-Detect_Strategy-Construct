from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def time_series_splits(dates: np.ndarray, train_size: float = 0.7, val_size: float = 0.15) -> TimeSplit:
    n = len(dates)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    return TimeSplit(
        train_indices=np.arange(0, train_end),
        val_indices=np.arange(train_end, val_end),
        test_indices=np.arange(val_end, n),
    )


def walk_forward(
    df: pd.DataFrame,
    date_col: str,
    fold_length: int,
    step: int,
) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.sort_values(date_col)
    dates = df[date_col].unique()
    for start in range(0, len(dates) - fold_length, step):
        end = start + fold_length
        train_dates = dates[start:end]
        test_dates = dates[end : end + step]
        train_df = df[df[date_col].isin(train_dates)]
        test_df = df[df[date_col].isin(test_dates)]
        if not test_df.empty:
            yield train_df, test_df
