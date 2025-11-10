from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, DonchianChannel


def add_price_features(df: pd.DataFrame, ticker_col: str = "ticker", ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.sort_values([ticker_col, ts_col]).copy()
    group = df.groupby(ticker_col, group_keys=False)

    df["ret_1"] = group["adj_close"].apply(lambda x: np.log(x) - np.log(x.shift(1)))
    df["ret_5"] = group["adj_close"].apply(lambda x: np.log(x) - np.log(x.shift(5)))
    df["ret_20"] = group["adj_close"].apply(lambda x: np.log(x) - np.log(x.shift(20)))

    for window in (5, 10, 20):
        df[f"rolling_mean_{window}"] = group["ret_1"].transform(lambda x: x.rolling(window).mean())
        df[f"rolling_std_{window}"] = group["ret_1"].transform(lambda x: x.rolling(window).std())
        df[f"rolling_skew_{window}"] = group["ret_1"].transform(lambda x: x.rolling(window).skew())
        df[f"rolling_kurt_{window}"] = group["ret_1"].transform(lambda x: x.rolling(window).kurt())

    return df


def add_technical_indicators(df: pd.DataFrame, ticker_col: str = "ticker", ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.sort_values([ticker_col, ts_col]).copy()

    def compute_indicators(subdf: pd.DataFrame) -> pd.DataFrame:
        close = subdf["adj_close"]
        high = subdf["high"]
        low = subdf["low"]
        volume = subdf["volume"].replace(0, np.nan)

        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        rsi = RSIIndicator(close=close, window=14)
        roc = ROCIndicator(close=close, window=10)
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        donchian = DonchianChannel(high=high, low=low, close=close, window=20)
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)

        subdf["macd"] = macd.macd()
        subdf["macd_signal"] = macd.macd_signal()
        subdf["macd_hist"] = macd.macd_diff()
        subdf["rsi_14"] = rsi.rsi()
        subdf["roc_10"] = roc.roc()
        subdf["adx_14"] = adx.adx()
        subdf["donchian_low"] = donchian.donchian_channel_lband()
        subdf["donchian_high"] = donchian.donchian_channel_hband()
        subdf["donchian_mid"] = donchian.donchian_channel_mband()
        subdf["atr_14"] = atr.average_true_range()

        subdf["obv"] = ((np.sign(close.diff()) * volume.fillna(0)).fillna(0)).cumsum()
        subdf["turnover"] = volume / subdf.get("shares_out", volume.mean())
        return subdf

    df = df.groupby(ticker_col, group_keys=False).apply(compute_indicators)
    return df


def add_seasonality_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True)
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    return df
