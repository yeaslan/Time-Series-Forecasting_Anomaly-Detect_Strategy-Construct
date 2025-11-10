# Quant Research Case Study — Liquid US Equities

## 1. Overview

This project delivers an end-to-end research stack that blends classical machine learning with modern deep learning to support forecasting, anomaly detection, and portfolio construction for a large-cap US equity universe. It targets three core workflows:

- **Time-series forecasting:** predict next-period returns, realized volatility, and market regimes.
- **Anomaly detection:** identify microstructure glitches and structural breaks using reconstruction/error-based detectors.
- **Alpha & portfolio construction:** translate predictive signals into executable strategies with realistic costs and risk controls.

The codebase is designed for reproducibility (fixed seeds, configurable pipelines) and extensibility (modular data, feature, model components).

## 2. Repository Layout

- `configs/` — YAML configuration files for forecasting, anomaly detection, and portfolio optimization.
- `data/` — placeholder directories for raw, interim, and processed datasets. Synthetic generators provide offline reproducibility.
- `src/` — library code grouped by concern (`data`, `features`, `labels`, `models`, `training`, `backtest`, `eval`, `utils`).
- `scripts/run_pipeline.py` — orchestrates the end-to-end workflow.
- `reports/` — structured documentation and generated artifacts (e.g., backtest snapshots).

## 3. Data Engineering Pipeline

1. **Ingestion:** `DataManager` loads synthetic or real datasets (prices, fundamentals, metadata). Synthetic generation (`src/data/synthetic.py`) emulates stochastic volatility, correlational structure, and fundamentals.
2. **Alignment & Cleaning:** Trading calendars align series; outliers clipped via MAD on returns; corporate actions preserved; missing values forward-filled (daily) or linearly interpolated (intraday).
3. **Feature Stores:** Processed parquet files persisted under `data/processed/` for consistent reuse.

## 4. Feature Engineering

- **Price/technical:** returns, rolling stats, RSI, MACD, ADX, ATR, Donchian, OBV, turnover, seasonality flags (`src/features/tech.py`).
- **Cross-sectional:** industry z-scores, rank-normalization, regression residuals vs. market factors (`src/features/cross_sectional.py`).
- **Fundamentals:** QoQ/YoY growth, profitability ratios, TTM aggregates, quality/value scores (`src/features/fundamentals.py`).
- **Graph:** correlation and industry graphs with degree/clustering/eigenvector centrality, Node2Vec embeddings (`src/features/graph.py`).
- **News (optional):** SentenceTransformer embeddings with EWMA aggregation (`src/features/news.py`).

All features avoid look-ahead bias and defer scaling until after time-based splits (RobustScaler fit on train set only).

## 5. Labeling

- Forward log returns for horizons `H ∈ {1,5,20}` (configurable).
- Ternary classification via percentile-based thresholds.
- Realized volatility (sum of squared intraday returns) and Parkinson alternative.
- Regime labels using Gaussian Mixture clustering on market returns.

Modules in `src/labels/` keep label logic composable and re-usable.

## 6. Modeling Suite

### Classical ML (`src/models/classical.py`)

- Regression: ElasticNet, RandomForest, XGBoost.
- Classification: Logistic (L1), RandomForest, XGBoost.
- Anomaly: Isolation Forest.

Pipelines include preprocessing (RobustScaler + OneHot) and optional Optuna tuning (`src/training/tuning.py`).

### Deep Learning (`src/models/`)

- **TCN:** Dilated 1D convolutions capture local temporal patterns.
- **RNN:** GRU/LSTM for sequence modeling.
- **Transformer:** Multi-head encoder with learned positional encoding (`TransformerEncoderModel`).
- **Autoencoder:** Feed-forward AE for reconstruction-based anomaly scoring.
- **GNN:** GCN & GAT layers for cross-asset information sharing via graph structures.

All PyTorch modules integrate with `TorchTrainer` (early stopping, cosine annealing, mixed precision-ready).

## 7. Training Flow (`scripts/run_pipeline.py`)

1. Load configuration (`configs/forecast.yaml` by default).
2. Generate/ingest data and derive feature matrix.
3. Split chronologically (70/15/15 train/val/test), purge leakage.
4. Scale features using RobustScaler fitted on train set.
5. Train classical regressors/classifiers (returns metrics, persisted models).
6. Optionally train Transformer (sequence windows via `TimeSeriesDataset`) with early stopping.
7. Train autoencoder (configurable in `configs/anomaly.yaml`).
8. Produce walk-forward backtest using regression scores; outputs stored under `reports/`.

### Running the pipeline

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/forecast.yaml
```

To target anomaly-specific experiments:

```bash
python scripts/run_pipeline.py --config configs/anomaly.yaml
```

*(Deep models require a CUDA-capable GPU for practical throughput; the script falls back to CPU automatically.)*

## 8. Backtesting & Evaluation

- `rebalance_portfolio` implements inverse-variance risk parity with turnover, cost, and volatility targeting controls.
- Costs incorporate commission, square-root impact, and slippage.
- KPIs: IC mean/std, Sharpe/Sortino, turnover, drawdowns (via `src/eval/metrics.py`).
- Factor exposure diagnostics (`src/eval/factor_exposure.py`) regress strategy returns on Fama-French factors or custom inputs.

Generated artifacts:

- `reports/backtest_portfolio.parquet` — daily equity curve, turnover, costs.
- `reports/backtest_trades.parquet` — holdings per rebalance.

## 9. Anomaly Detection Workflow

- Isolation Forest model for fast unsupervised detection (sklearn pipeline).
- Autoencoder reconstruction error; thresholds derived via percentile rules (configurable).
- Extendable to VAE or LOB-specific CNNs by adding modules under `src/models/`.

Threshold calibration and live monitoring hooks can log to MLflow/W&B for governance.

## 10. Extending the Framework

1. **Alternative labels:** add new modules under `src/labels/` (e.g., realized skew, drawdown flags) and register in config.
2. **Custom models:** drop PyTorch modules under `src/models/` and reference them in `scripts/run_pipeline.py` or bespoke trainers.
3. **Execution/Risk:** plug-in execution styles via `src/backtest/exec_sim.py`, add optimizers to `src/backtest/portfolio.py`.
4. **Monitoring:** integrate `mlflow` (already required) to log metrics/artifacts per experiment.

## 11. Exporting Documentation to PDF

This markdown is structured for easy PDF export. Recommended options:

- **Pandoc:** `pandoc reports/quant_case_study.md -o reports/quant_case_study.pdf`
- **VS Code / Cursor:** open the markdown and use “Export as PDF”.
- **Jupyter Notebook:** convert to notebook cell with `nbconvert`.

## 12. Next Steps

- Add Optuna study definitions for hyperparameter sweeps.
- Enable GPU-friendly DataLoaders with multi-processing.
- Incorporate live data adapters (Polygon, Tiingo, Quandl) in `src/data/loader.py`.
- Build MLflow experiment templates for repeatable research campaigns.

With these components, the repository delivers a production-grade foundation for intraday-to-daily equity research, facilitating rapid experimentation across models and asset universes.
