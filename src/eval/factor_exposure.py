from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def factor_exposures(returns: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
    aligned = pd.concat([returns, factors], axis=1).dropna()
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1:])
    model = sm.OLS(y, X).fit()
    exposures = model.params.to_dict()
    exposures.update({f"{key}_tstat": model.tvalues[key] for key in model.params.index})
    return exposures
