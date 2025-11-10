from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bps: float = 0.3
    impact_y: float = 0.2
    slippage_bps: float = 1.0

    def estimate(self, turnover: float, volatility: float, adv_ratio: float) -> float:
        commission = self.commission_bps * turnover / 10000
        impact = self.impact_y * volatility * (adv_ratio ** 0.5)
        slippage = self.slippage_bps / 10000
        return commission + impact + slippage
