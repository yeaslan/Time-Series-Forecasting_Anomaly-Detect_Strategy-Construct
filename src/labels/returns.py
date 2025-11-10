from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(df: pd.DataFrame, horizon: int = 1, price_col: str = "adj_close") -> pd.DataFrame:
    df = df.sort_values(["ticker", "timestamp"]).copy()
    df[f"fwd_return_{horizon}"] = df.groupby("ticker")[price_col].transform(lambda x: np.log(x.shift(-horizon)) - np.log(x))
    return df


def classify_returns(
    df: pd.DataFrame,
    return_col: str,
    percentile: float = 60.0,
) -> pd.DataFrame:
    df = df.copy()
    def compute_thresholds(subdf: pd.DataFrame) -> pd.Series:
        tau = np.percentile(np.abs(subdf[return_col].dropna()), percentile)
        return pd.Series({"tau": tau})

    thresholds = df.groupby("ticker").apply(compute_thresholds)
    df = df.join(thresholds["tau"], on="ticker")
    tau = df["tau"]
    ret = df[return_col]
    df[f"{return_col}_class"] = np.where(ret >= tau, 1, np.where(ret <= -tau, -1, 0))
    df.drop(columns=["tau"], inplace=True)
    return df
