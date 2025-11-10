from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.calendars import TradingCalendar, align_to_calendar
from src.data.synthetic import SyntheticUniverseGenerator
from src.utils.io import ensure_dir, load_dataframe, load_pickle, save_dataframe, save_pickle

LOGGER = logging.getLogger(__name__)


class DataManager:
    """
    Responsible for loading raw data, aligning, cleaning, and persisting processed datasets.
    Supports synthetic data generation for fully offline reproducibility.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        processed_dir: Optional[str | Path] = None,
        synthetic: bool = False,
        calendar: Optional[TradingCalendar] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = Path(processed_dir) if processed_dir else self.data_dir / "processed"
        self.synthetic = synthetic
        self.calendar = calendar or TradingCalendar()
        ensure_dir(self.processed_dir)

    def load_prices(self, freq: str = "daily") -> pd.DataFrame:
        if self.synthetic:
            generator = SyntheticUniverseGenerator(seed=1337)
            prices = generator.generate_price_panel(freq=freq)
            LOGGER.info("Generated synthetic %s prices with shape %s", freq, prices.shape)
            return prices

        path = self.raw_dir / "prices" / f"{freq}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing price data at {path}")
        prices = load_dataframe(path)
        LOGGER.info("Loaded %s prices from %s", freq, path)
        return prices

    def load_fundamentals(self) -> pd.DataFrame:
        if self.synthetic:
            generator = SyntheticUniverseGenerator(seed=1337)
            fundamentals = generator.generate_fundamentals()
            LOGGER.info("Generated synthetic fundamentals with shape %s", fundamentals.shape)
            return fundamentals

        path = self.raw_dir / "fundamentals" / "fundamentals.parquet"
        if not path.exists():
            LOGGER.warning("Fundamentals file missing at %s; returning empty DataFrame", path)
            return pd.DataFrame()
        return load_dataframe(path)

    def load_metadata(self) -> pd.DataFrame:
        if self.synthetic:
            generator = SyntheticUniverseGenerator(seed=1337)
            metadata = generator.generate_metadata()
            LOGGER.info("Generated synthetic metadata for %d tickers", metadata.shape[0])
            return metadata

        path = self.raw_dir / "meta" / "universe.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata at {path}")
        meta = load_dataframe(path)
        return meta

    def align_and_clean(self, prices: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
        aligned = align_to_calendar(prices, self.calendar, freq=freq, on="timestamp", groupby="ticker")
        aligned.sort_values(["ticker", "timestamp"], inplace=True)

        # Remove obvious outliers using MAD on returns
        aligned["close_ffill"] = aligned.groupby("ticker")["adj_close"].ffill()
        aligned["return"] = aligned.groupby("ticker")["close_ffill"].apply(lambda x: np.log(x) - np.log(x.shift(1)))
        median = aligned.groupby("ticker")["return"].transform("median")
        mad = aligned.groupby("ticker")["return"].transform(lambda x: np.median(np.abs(x - np.median(x))))
        threshold = 5 * mad.replace(0, np.nan)
        mask = threshold.notna()
        aligned.loc[mask & ((aligned["return"] - median).abs() > threshold), ["open", "high", "low", "close", "adj_close", "volume"]] = np.nan

        aligned["open"] = aligned.groupby("ticker")["open"].fillna(method="ffill")
        aligned["high"] = aligned.groupby("ticker")["high"].fillna(method="ffill")
        aligned["low"] = aligned.groupby("ticker")["low"].fillna(method="ffill")
        aligned["close"] = aligned.groupby("ticker")["close"].fillna(method="ffill")
        aligned["adj_close"] = aligned.groupby("ticker")["adj_close"].fillna(method="ffill")
        aligned["volume"] = aligned.groupby("ticker")["volume"].fillna(0)

        aligned.drop(columns=["close_ffill"], inplace=True)
        return aligned

    def persist_processed(self, name: str, df: pd.DataFrame) -> Path:
        path = self.processed_dir / f"{name}.parquet"
        save_dataframe(df, path)
        return path

    def load_processed(self, name: str) -> pd.DataFrame:
        path = self.processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset {name} not found at {path}")
        return load_dataframe(path)

    def load_all(self, freq: str = "daily") -> Dict[str, pd.DataFrame]:
        prices = self.load_prices(freq=freq)
        fundamentals = self.load_fundamentals()
        metadata = self.load_metadata()

        if "timestamp" in prices and "ticker" in prices:
            prices = self.align_and_clean(prices, freq="B" if freq == "daily" else "1T")

        datasets = {"prices": prices, "fundamentals": fundamentals, "metadata": metadata}

        for key, df in datasets.items():
            self.persist_processed(key, df)
            LOGGER.info("Persisted %s to processed store", key)
        return datasets
