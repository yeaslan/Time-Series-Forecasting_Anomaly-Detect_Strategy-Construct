from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(df: pd.DataFrame, return_col: str = "ret_1", horizon: int = 1, groupby: str = "ticker") -> pd.DataFrame:
    df = df.sort_values([groupby, "timestamp"]).copy()
    df[f"realized_vol_{horizon}"] = df.groupby(groupby)[return_col].transform(
        lambda x: np.sqrt(x.rolling(window=horizon).apply(lambda r: np.sum(np.square(r)), raw=True))
    )
    return df


def parkison_volatility(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", groupby: str = "ticker") -> pd.DataFrame:
    df = df.copy()
    df["park_vol"] = (
        (1 / (4 * np.log(2)))
        * np.square(np.log(df[high_col]) - np.log(df[low_col]))
    )
    df["park_vol"] = df.groupby(groupby)["park_vol"].rolling(window=1).sum().reset_index(level=0, drop=True)
    return df
