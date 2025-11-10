#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from src.backtest.costs import CostModel
from src.backtest.portfolio import PortfolioOptimizer, rebalance_portfolio
from src.data.loader import DataManager
from src.features.cross_sectional import cross_sectional_features
from src.features.fundamentals import compute_fundamental_ratios, forward_fill_fundamentals
from src.features.graph import enrich_with_graph_features
from src.features.tech import add_price_features, add_seasonality_features, add_technical_indicators
from src.labels.regimes import attach_regime_labels
from src.labels.returns import classify_returns, compute_forward_returns
from src.labels.vol import realized_volatility
from src.models.ae import FeedForwardAutoencoder
from src.models.classical import train_classification_model, train_regression_model
from src.models.transformer import TransformerEncoderModel
from src.training.dataset import TimeSeriesDataset, build_sequences
from src.training.trainer import EarlyStopping, TorchTrainer
from src.utils.config import load_config, log_config
from src.utils.io import ensure_dir, save_artifact
from src.utils.logging import create_logger
from src.utils.seed import set_global_seed


LOGGER = create_logger(__name__)


def prepare_features(cfg: Dict, synthetic: bool = True) -> pd.DataFrame:
    data_manager = DataManager(synthetic=synthetic, processed_dir=cfg["project"]["data_dir"])
    datasets = data_manager.load_all(freq=cfg["data"]["freq"])
    prices = datasets["prices"]
    fundamentals = datasets["fundamentals"]
    metadata = datasets["metadata"]

    prices = add_price_features(prices)
    prices = add_technical_indicators(prices)
    prices = add_seasonality_features(prices)

    if not fundamentals.empty and cfg["features"]["fundamentals"]:
        fundamentals_enriched = compute_fundamental_ratios(fundamentals)
        prices = forward_fill_fundamentals(prices, fundamentals_enriched)

    if cfg["features"]["graph"]:
        if data_manager.synthetic:
            from src.data.synthetic import SyntheticUniverseGenerator

            edge_df = SyntheticUniverseGenerator(seed=cfg["project"]["seed"]).generate_graph_edges()
        else:
            edge_df = pd.DataFrame()
        if not edge_df.empty:
            latest_edges = edge_df.groupby(["source", "target"]).tail(1)
            prices = enrich_with_graph_features(prices, latest_edges)

    prices = prices.merge(metadata[["ticker", "sector", "industry"]], on="ticker", how="left")

    numeric_cols = [col for col in prices.columns if pd.api.types.is_numeric_dtype(prices[col]) and col not in {"timestamp"}]
    numeric_cols = [col for col in numeric_cols if not col.startswith("fwd_return")]
    prices = cross_sectional_features(prices, numeric_cols[:10] if numeric_cols else [])

    prices = compute_forward_returns(prices, horizon=cfg["data"]["horizon"])
    prices = realized_volatility(prices, horizon=cfg["data"]["horizon"])

    prices["market_return"] = prices.groupby("timestamp")["ret_1"].transform("mean")
    prices = attach_regime_labels(prices, market_col="market_return")
    prices = classify_returns(prices, return_col=f"fwd_return_{cfg['data']['horizon']}", percentile=cfg["targets"]["returns"]["threshold_percentile"])
    return prices


def split_dataset(df: pd.DataFrame, cfg: Dict) -> Dict[str, pd.DataFrame]:
    df = df.dropna(subset=[f"fwd_return_{cfg['data']['horizon']}"]).copy()
    df.sort_values("timestamp", inplace=True)
    unique_dates = df["timestamp"].unique()
    n = len(unique_dates)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    splits = {
        "train": df[df["timestamp"].isin(unique_dates[:train_end])],
        "val": df[df["timestamp"].isin(unique_dates[train_end:val_end])],
        "test": df[df["timestamp"].isin(unique_dates[val_end:])],
    }
    return splits


def scale_features(train: pd.DataFrame, others: Dict[str, pd.DataFrame], feature_cols: List[str]) -> Dict[str, pd.DataFrame]:
    scaler = RobustScaler()
    train_scaled = train.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_scaled[feature_cols].fillna(0))

    scaled = {"train": train_scaled}
    for split_name, df in others.items():
        scaled_df = df.copy()
        scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols].fillna(0))
        scaled[split_name] = scaled_df
    save_artifact(scaler, Path("models/artifacts") / "feature_scaler.pkl")
    return scaled


def train_classical_models(splits: Dict[str, pd.DataFrame], feature_cols: List[str], cfg: Dict) -> tuple[Dict[str, Dict], Dict[str, object]]:
    train_df = splits["train"]
    val_df = splits["val"]
    metrics: Dict[str, Dict] = {}
    models: Dict[str, object] = {}

    target_reg = f"fwd_return_{cfg['data']['horizon']}"
    model_reg, rmse = train_regression_model(
        train_df,
        train_df[target_reg],
        val_df,
        val_df[target_reg],
        model_name="xgboost",
        params={"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8},
        feature_cols=feature_cols,
    )
    save_artifact(model_reg, Path("models/artifacts") / "regression_model.pkl")
    metrics["regression"] = {"rmse": rmse}
    models["regression"] = model_reg

    cls_col = f"{target_reg}_class"
    model_cls, cls_metrics = train_classification_model(
        train_df,
        train_df[cls_col],
        val_df,
        val_df[cls_col],
        model_name="random_forest",
        params={"n_estimators": 300, "max_depth": 8},
        feature_cols=feature_cols,
    )
    save_artifact(model_cls, Path("models/artifacts") / "classification_model.pkl")
    metrics["classification"] = cls_metrics
    models["classification"] = model_cls
    return metrics, models


def train_transformer(splits: Dict[str, pd.DataFrame], feature_cols: List[str], cfg: Dict) -> Dict[str, List[float]]:
    lookback = cfg["data"]["lookback"]
    horizon = cfg["data"]["horizon"]
    target_col = f"fwd_return_{horizon}"

    sequences_train = build_sequences(splits["train"], feature_cols, [target_col], lookback, horizon)
    sequences_val = build_sequences(splits["val"], feature_cols, [target_col], lookback, horizon)

    dataset_train = TimeSeriesDataset(sequences_train)
    dataset_val = TimeSeriesDataset(sequences_val)

    if len(dataset_train) == 0 or len(dataset_val) == 0:
        LOGGER.warning("Insufficient sequence data for transformer training; skipping deep model fit.")
        return {"train_loss": [], "val_loss": []}

    loader_train = DataLoader(dataset_train, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "auto" else "cpu"
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    model = TransformerEncoderModel(
        input_dim=len(feature_cols),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        num_layers=model_cfg["n_layers"],
        dim_feedforward=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    trainer = TorchTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["max_epochs"]),
        device=device,
    )
    checkpoint_dir = ensure_dir(Path(cfg["project"]["artifacts_dir"]) / "transformer")
    history = trainer.fit(
        loader_train,
        val_loader=loader_val,
        epochs=cfg["training"]["max_epochs"],
        early_stopping=EarlyStopping(patience=cfg["training"]["early_stopping"]["patience"]),
        checkpoint_dir=checkpoint_dir,
    )
    torch.save(model.state_dict(), checkpoint_dir / "final.pt")
    return history


def train_autoencoder(train_df: pd.DataFrame, feature_cols: List[str], cfg: Dict) -> Dict[str, float]:
    params = next(model for model in cfg["models"] if model["name"] == "autoencoder")["params"]
    input_dim = len(feature_cols)
    model = FeedForwardAutoencoder(input_dim=input_dim, hidden_dims=params["hidden_dims"], latent_dim=params["latent_dim"], dropout=params["dropout"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)

    best_loss = float("inf")
    patience = params["patience"]
    counter = 0

    for epoch in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        recon = model(X_train)
        loss = loss_fn(recon, X_train)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            artifact_dir = ensure_dir(Path(cfg["project"]["artifacts_dir"]))
            torch.save(model.state_dict(), artifact_dir / "autoencoder.pt")
        else:
            counter += 1
            if counter >= patience:
                break
    return {"reconstruction_loss": best_loss}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    log_config(cfg)
    set_global_seed(cfg["project"]["seed"])

    data = prepare_features(cfg, synthetic=cfg["data"]["synthetic"])
    splits = split_dataset(data, cfg)

    feature_cols = [col for col in data.columns if col not in {"timestamp", "ticker", f"fwd_return_{cfg['data']['horizon']}", f"fwd_return_{cfg['data']['horizon']}_class"} and data[col].dtype != "object"]
    scaled = scale_features(splits["train"], {"val": splits["val"], "test": splits["test"]}, feature_cols)

    classical_metrics, classical_models = train_classical_models({"train": scaled["train"], "val": scaled["val"]}, feature_cols, cfg)
    LOGGER.info("Classical ML results: %s", classical_metrics)

    transformer_history = train_transformer({"train": scaled["train"], "val": scaled["val"]}, feature_cols, cfg)
    LOGGER.info("Transformer training history: %s", transformer_history)

    anomaly_cfg = load_config("configs/anomaly.yaml")
    anomaly_results = train_autoencoder(scaled["train"], feature_cols, anomaly_cfg)
    LOGGER.info("Autoencoder anomaly results: %s", anomaly_results)

    # Backtest on test set using regression predictions as signals
    regressor = classical_models["regression"]
    test_df_scaled = scaled["test"].copy()
    test_df_scaled["signal"] = regressor.predict(test_df_scaled[feature_cols])
    signals = test_df_scaled[["timestamp", "ticker", "signal", f"fwd_return_{cfg['data']['horizon']}"]].rename(columns={f"fwd_return_{cfg['data']['horizon']}": "fwd_return"})

    test_df_raw = splits["test"]
    cov_matrix = test_df_raw.pivot_table(index="timestamp", columns="ticker", values="ret_1").cov()
    optimizer = PortfolioOptimizer(
        vol_target=cfg["backtest"]["vol_target"],
        gross_max=cfg["backtest"].get("gross_max", 2.0),
        weight_cap=cfg["backtest"].get("weight_cap", 0.02),
    )
    cost_model = CostModel(**cfg["backtest"]["costs"])
    bt = rebalance_portfolio(signals, test_df_raw, cov_matrix, optimizer, cost_model, vol_target=cfg["backtest"]["vol_target"])
    save_artifact(bt["portfolio"], Path("reports") / "backtest_portfolio.parquet")
    save_artifact(bt["trades"], Path("reports") / "backtest_trades.parquet")

    LOGGER.info("Backtest summary: %s", bt["portfolio"].tail())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quant Research Pipeline")
    parser.add_argument("--config", type=str, default="configs/forecast.yaml")
    args = parser.parse_args()
    main(args.config)
