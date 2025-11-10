from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from src.backtest.costs import CostModel


def inverse_variance_weights(cov: np.ndarray, assets: list[str]) -> pd.Series:
    inv_var = 1 / np.diag(cov)
    weights = inv_var / inv_var.sum()
    return pd.Series(weights, index=assets)


@dataclass
class PortfolioOptimizer:
    vol_target: float = 0.10
    net_exposure: float = 0.0
    gross_max: float = 2.0
    weight_cap: float = 0.02

    def optimize(self, forecasts: pd.Series, cov: pd.DataFrame) -> pd.Series:
        cov_matrix = cov.values
        assets = list(cov.columns)
        weights = inverse_variance_weights(cov_matrix, assets)
        weights.index = assets
        forecasts_aligned = forecasts.reindex(assets).fillna(0.0)
        weights = weights * np.sign(forecasts_aligned)
        weights = weights / weights.abs().sum()
        weights = weights.clip(-self.weight_cap, self.weight_cap)
        gross = weights.abs().sum()
        if gross > self.gross_max:
            weights *= self.gross_max / gross
        return weights


def rebalance_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    optimizer: PortfolioOptimizer,
    cost_model: CostModel,
    vol_target: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    signals = signals.sort_values("timestamp")
    portfolio_history = []
    trades_history = []

    current_weights = pd.Series(0, index=cov_matrix.columns)
    prev_value = 1.0

    for ts, snapshot in signals.groupby("timestamp"):
        forecast = snapshot.set_index("ticker")["signal"]
        weights = optimizer.optimize(forecast, cov_matrix)
        turnover = (weights - current_weights).abs().sum()
        adv_ratio = 0.02
        cost = cost_model.estimate(turnover, volatility=vol_target / np.sqrt(252), adv_ratio=adv_ratio)
        returns = snapshot.set_index("ticker")["fwd_return"]
        pnl = (weights * returns).sum() - cost
        prev_value *= (1 + pnl)
        portfolio_history.append({"timestamp": ts, "portfolio_value": prev_value, "turnover": turnover, "cost": cost})
        trades_history.append({"timestamp": ts, "weights": weights})
        current_weights = weights

    return {
        "portfolio": pd.DataFrame(portfolio_history),
        "trades": pd.DataFrame(trades_history),
    }
