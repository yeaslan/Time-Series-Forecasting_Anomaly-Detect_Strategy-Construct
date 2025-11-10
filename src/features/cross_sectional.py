from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def rank_normalize(df: pd.DataFrame, features: list[str], groupby: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    for feature in features:
        df[f"{feature}_rank"] = df.groupby(groupby)[feature].transform(
            lambda x: (x.rank(method="first") - 0.5) / len(x.dropna()) - 0.5
        )
    return df


def industry_zscore(df: pd.DataFrame, features: list[str], industry_col: str = "industry", groupby: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    for feature in features:
        df[f"{feature}_z_ind"] = df.groupby([groupby, industry_col])[feature].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
    return df


def compute_residuals_vs_market(
    df: pd.DataFrame,
    return_col: str = "ret_1",
    market_col: str = "market_return",
    groupby: str = "timestamp",
) -> pd.DataFrame:
    df = df.copy()
    if market_col not in df:
        market = df.groupby(groupby)[return_col].mean()
        df = df.join(market.rename(market_col), on=groupby, how="left")

    def regression_residuals(subdf: pd.DataFrame) -> pd.Series:
        y = subdf[return_col]
        x = sm.add_constant(subdf[market_col])
        model = sm.OLS(y, x, missing="drop")
        try:
            result = model.fit()
            residuals = result.resid
        except Exception:
            residuals = y - y.mean()
        return residuals

    df["idiosyncratic_ret"] = df.groupby(groupby).apply(regression_residuals).reset_index(level=0, drop=True)
    return df


def cross_sectional_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = rank_normalize(df, feature_cols, groupby="timestamp")
    df = industry_zscore(df, feature_cols, industry_col="industry", groupby="timestamp")
    df = compute_residuals_vs_market(df, return_col="ret_1", market_col="market_return")
    return df
