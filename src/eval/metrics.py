from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray | None = None) -> Dict[str, float]:
    if y_pred is None:
        threshold = 0.5 if y_score.ndim == 1 else None
        if threshold is not None:
            y_pred = (y_score >= threshold).astype(int)
        else:
            y_pred = y_score.argmax(axis=1)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = np.asarray(y_pred)

    if y_score.ndim == 1 or y_score.shape[1] == 2:
        proba = y_score if y_score.ndim == 1 else y_score[:, 1]
        auc_roc = metrics.roc_auc_score(y_true, proba)
        auc_pr = metrics.average_precision_score(y_true, proba)
        brier = metrics.brier_score_loss(y_true, proba)
    else:
        auc_roc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        auc_pr = float("nan")
        brier = float("nan")

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "f1": float(metrics.f1_score(y_true, y_pred, average="macro")),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "brier": float(brier),
    }


def information_coefficient(df: pd.DataFrame, prediction_col: str, target_col: str, groupby: str = "timestamp") -> Dict[str, float]:
    grouped = df.groupby(groupby).apply(lambda x: x[prediction_col].corr(x[target_col], method="spearman"))
    return {
        "ic_mean": float(grouped.mean()),
        "ic_std": float(grouped.std()),
        "ic_ir": float(grouped.mean() / (grouped.std() + 1e-8)),
    }


def drawdown_stats(returns: pd.Series) -> Dict[str, float]:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    mdd = drawdown.min()
    return {"max_drawdown": float(mdd), "calmar": float(returns.mean() * 252 / abs(mdd + 1e-8))}
