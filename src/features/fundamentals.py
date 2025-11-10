from __future__ import annotations

import pandas as pd


def compute_fundamental_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = fundamentals.copy()
    df.sort_values(["ticker", "report_date"], inplace=True)

    df["sales_growth_qoq"] = df.groupby("ticker")["sales"].pct_change()
    df["sales_growth_yoy"] = df.groupby("ticker")["sales"].pct_change(4)
    df["net_margin"] = df["net_income"] / df["sales"].replace(0, pd.NA)
    df["roe"] = df["net_income"] / df["book_value"].replace(0, pd.NA)
    df["roa"] = df["net_income"] / df["assets"].replace(0, pd.NA)

    df["ttm_sales"] = df.groupby("ticker")["sales"].rolling(window=4, min_periods=1).sum().reset_index(level=0, drop=True)
    df["ttm_net_income"] = (
        df.groupby("ticker")["net_income"].rolling(window=4, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    df["quality_score"] = df["roe"].rank(pct=True) + df["net_margin"].rank(pct=True)
    df["value_score"] = (df["ttm_sales"] / df["book_value"]).replace([pd.NA, 0], pd.NA)
    return df


def forward_fill_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals["report_date"] = pd.to_datetime(fundamentals["report_date"], utc=True)
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    fundamentals.sort_values(["ticker", "report_date"], inplace=True)
    fundamentals["effective_date"] = fundamentals["report_date"] + pd.Timedelta(days=1)

    merged = pd.merge_asof(
        prices.sort_values("timestamp"),
        fundamentals.sort_values("effective_date"),
        left_on="timestamp",
        right_on="effective_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=365),
    )
    return merged
