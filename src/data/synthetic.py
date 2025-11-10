from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticUniverseGenerator:
    """
    Generates synthetic time series data intended to mimic equities with stochastic
    volatility, regime shifts, and cross-sectional structure. Useful for offline
    demos when real data cannot be distributed.
    """

    num_assets: int = 200
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    freq: str = "B"
    seed: Optional[int] = 1337

    sectors: Tuple[str, ...] = (
        "Technology",
        "Financials",
        "Healthcare",
        "Energy",
        "Industrials",
        "Consumer",
        "Utilities",
        "Materials",
        "RealEstate",
        "Communication",
    )

    industries_per_sector: int = 3

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.dates = pd.date_range(self.start, self.end, freq=self.freq, tz="UTC")
        self.tickers = [f"SYN{i:04d}" for i in range(self.num_assets)]

    def _generate_regimes(self) -> pd.Series:
        regimes = ["bull", "sideways", "bear"]
        transition_prob = np.array(
            [
                [0.90, 0.08, 0.02],
                [0.10, 0.80, 0.10],
                [0.05, 0.10, 0.85],
            ]
        )
        state = 0
        states = []
        for _ in range(len(self.dates)):
            states.append(regimes[state])
            state = self.rng.choice(len(regimes), p=transition_prob[state])
        return pd.Series(states, index=self.dates, name="regime")

    def _regime_params(self) -> Dict[str, Dict[str, float]]:
        return {
            "bull": {"mu": 0.0005, "vol": 0.01},
            "sideways": {"mu": 0.0001, "vol": 0.012},
            "bear": {"mu": -0.0006, "vol": 0.02},
        }

    def generate_price_panel(self, freq: str = "daily") -> pd.DataFrame:
        regimes = self._generate_regimes()
        params = self._regime_params()

        values = []
        for ticker in self.tickers:
            sector = self.rng.choice(self.sectors)
            industry = f"{sector}_{self.rng.integers(0, self.industries_per_sector)}"

            base_price = self.rng.uniform(20, 200)
            adj_close = [base_price]
            volume = [self.rng.uniform(1e5, 5e6)]

            for date in self.dates[1:]:
                regime = regimes.loc[date]
                mu = params[regime]["mu"] + self.rng.normal(0, 0.0003)
                vol = params[regime]["vol"]
                shock = self.rng.normal(mu, vol)
                price = adj_close[-1] * np.exp(shock)
                adj_close.append(price)
                volume.append(max(0, volume[-1] * np.exp(self.rng.normal(0, 0.1))))

            adj_close = np.array(adj_close)
            close = adj_close * (1 + self.rng.normal(0, 0.001, size=adj_close.shape))
            open_price = close * np.exp(self.rng.normal(0, 0.006, size=close.shape))
            high = np.maximum(open_price, close) * (1 + np.abs(self.rng.normal(0, 0.01, size=close.shape)))
            low = np.minimum(open_price, close) * (1 - np.abs(self.rng.normal(0, 0.01, size=close.shape)))

            df = pd.DataFrame(
                {
                    "timestamp": self.dates,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": adj_close,
                    "volume": volume,
                    "vwap": (high + low + close) / 3.0,
                    "sector": sector,
                    "industry": industry,
                }
            )
            values.append(df)

        panel = pd.concat(values, ignore_index=True)
        # Add corporate actions
        panel["div_cash"] = np.where(self.rng.random(len(panel)) < 0.001, self.rng.uniform(0.1, 1.0), 0.0)
        panel["split_ratio"] = np.where(self.rng.random(len(panel)) < 0.0005, self.rng.choice([0.5, 2.0]), 1.0)

        return panel

    def generate_fundamentals(self) -> pd.DataFrame:
        quarters = pd.date_range(self.start, self.end, freq="Q")
        records = []
        for ticker in self.tickers:
            base_sales = self.rng.uniform(1e8, 5e9)
            profit_margin = self.rng.uniform(0.05, 0.2)
            assets = self.rng.uniform(5e8, 1e10)
            for fiscal_period in quarters:
                growth = self.rng.normal(0.05, 0.03)
                sales = base_sales * (1 + growth) ** ((fiscal_period.year - quarters[0].year) * 4 + fiscal_period.quarter)
                net_income = sales * profit_margin * (1 + self.rng.normal(0, 0.1))
                ebit = net_income * 1.2
                book_value = assets * (1 + self.rng.normal(0, 0.02))
                shares_out = self.rng.uniform(1e7, 1e9)

                records.append(
                    {
                        "ticker": ticker,
                        "fiscal_period": fiscal_period,
                        "report_date": fiscal_period + pd.Timedelta(days=30),
                        "sales": sales,
                        "ebit": ebit,
                        "net_income": net_income,
                        "assets": assets,
                        "book_value": book_value,
                        "shares_out": shares_out,
                    }
                )
        return pd.DataFrame(records)

    def generate_metadata(self) -> pd.DataFrame:
        listings = []
        for ticker in self.tickers:
            sector = self.rng.choice(self.sectors)
            industry = f"{sector}_{self.rng.integers(0, self.industries_per_sector)}"
            listings.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "primary_exchange": "NYSE",
                    "ipo_date": pd.Timestamp("2000-01-01"),
                    "is_active": True,
                }
            )
        return pd.DataFrame(listings)

    def generate_graph_edges(self, window: int = 60) -> pd.DataFrame:
        prices = self.generate_price_panel()
        pivot = prices.pivot(index="timestamp", columns="ticker", values="adj_close")
        returns = np.log(pivot).diff()
        rolling_corr = returns.rolling(window=window, min_periods=window // 2).corr().dropna()

        edges = []
        for (ts, ticker_i), row in rolling_corr.groupby(level=[0, 1]):
            corr_series = row.droplevel(0)
            top_pairs = corr_series.nlargest(5)
            for ticker_j, corr in top_pairs.items():
                if ticker_i == ticker_j:
                    continue
                edges.append(
                    {"timestamp": ts, "source": ticker_i, "target": ticker_j, "corr_60d": corr, "same_industry": float(corr > 0.5)}
                )
        return pd.DataFrame(edges)
