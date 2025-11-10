from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def regime_labels_gmm(series: pd.Series, n_components: int = 3, random_state: int = 1337) -> pd.Series:
    clean = series.dropna().values.reshape(-1, 1)
    if len(clean) < n_components:
        return pd.Series(index=series.index, data="unknown")

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(clean)
    probs = gmm.predict_proba(clean)
    regimes_idx = probs.argmax(axis=1)

    means = gmm.means_.flatten()
    regime_map = {idx: label for idx, label in enumerate(np.argsort(means))}

    labels = ["bear", "sideways", "bull"]
    mapped_labels = [labels[regime_map[idx]] for idx in regimes_idx]
    result = pd.Series(index=series.dropna().index, data=mapped_labels)
    return result.reindex(series.index).fillna(method="ffill")


def attach_regime_labels(df: pd.DataFrame, market_col: str = "market_return") -> pd.DataFrame:
    if market_col not in df:
        raise ValueError(f"{market_col} not found in dataframe for regime labeling")
    regimes = regime_labels_gmm(df[market_col])
    df = df.copy()
    df["market_regime"] = regimes
    return df
