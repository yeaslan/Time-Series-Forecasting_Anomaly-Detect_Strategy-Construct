from __future__ import annotations

from typing import Any, Dict

import optuna


def optimize_hyperparameters(objective_fn, n_trials: int = 20, study_name: str = "optuna_study", direction: str = "minimize") -> optuna.study.Study:
    storage = "sqlite:///models/artifacts/optuna.db"
    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study
