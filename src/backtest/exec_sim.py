from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class ExecutionSimulator:
    style: Literal["twap", "vwap"] = "vwap"
    participation: float = 0.1

    def simulate(self, orders: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        merged = orders.merge(market, on=["timestamp", "ticker"], how="left", suffixes=("", "_mkt"))
        if self.style == "twap":
            merged["fill_price"] = merged["vwap"].fillna(merged["close"])
        else:
            merged["fill_price"] = merged["vwap_mkt"].fillna(merged["close"])
        merged["executed"] = np.sign(merged["order_qty"]) * np.minimum(np.abs(merged["order_qty"]), merged["volume"] * self.participation)
        merged["slippage"] = merged["fill_price"] - merged["close"]
        return merged
