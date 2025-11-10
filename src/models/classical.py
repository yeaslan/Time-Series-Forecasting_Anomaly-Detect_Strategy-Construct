from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier, XGBRegressor


REG_MODELS = {
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
}

CLS_MODELS = {
    "logistic_l1": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

ANOMALY_MODELS = {
    "isolation_forest": IsolationForest,
}


def build_preprocessor(feature_cols: List[str], categorical_cols: Optional[List[str]] = None) -> ColumnTransformer:
    categorical_cols = categorical_cols or []
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", RobustScaler(with_centering=True, with_scaling=True), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    return ColumnTransformer(transformers=transformers)


def build_regression_pipeline(model_name: str, params: Dict[str, Any], feature_cols: List[str], categorical_cols: Optional[List[str]] = None) -> Pipeline:
    preprocessor = build_preprocessor(feature_cols, categorical_cols)
    model_cls = REG_MODELS[model_name]
    model = model_cls(**params)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline


def build_classification_pipeline(model_name: str, params: Dict[str, Any], feature_cols: List[str], categorical_cols: Optional[List[str]] = None) -> Pipeline:
    preprocessor = build_preprocessor(feature_cols, categorical_cols)
    model_cls = CLS_MODELS[model_name]
    if model_name == "logistic_l1":
        params.setdefault("penalty", "l1")
        params.setdefault("solver", "liblinear")
    model = model_cls(**params)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline


def tune_model(
    pipeline: Pipeline,
    param_grid: Dict[str, Iterable[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    cv: Any,
    n_jobs: int = -1,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={f"model__{key}": value for key, value in param_grid.items()},
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    params: Dict[str, Any],
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[Pipeline, float]:
    pipeline = build_regression_pipeline(model_name, params, feature_cols, categorical_cols)
    pipeline.fit(X_train[feature_cols], y_train)
    preds = pipeline.predict(X_val[feature_cols])
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return pipeline, rmse


def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    params: Dict[str, Any],
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    pipeline = build_classification_pipeline(model_name, params, feature_cols, categorical_cols)
    pipeline.fit(X_train[feature_cols], y_train)
    proba = pipeline.predict_proba(X_val[feature_cols])[:, 1] if hasattr(pipeline[-1], "predict_proba") else pipeline.predict(X_val[feature_cols])
    preds = pipeline.predict(X_val[feature_cols])
    metrics = {
        "accuracy": (preds == y_val).mean(),
    }
    return pipeline, metrics


def train_anomaly_model(
    X_train: pd.DataFrame,
    model_name: str,
    params: Dict[str, Any],
    feature_cols: List[str],
) -> BaseEstimator:
    model_cls = ANOMALY_MODELS[model_name]
    model = model_cls(**params)
    model.fit(X_train[feature_cols])
    return model
