from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    patience: int = 20
    mode: str = "min"
    best_score: Optional[float] = None
    counter: int = 0

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improvement = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improvement:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class TorchTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str | torch.device = "cpu",
        grad_clip: Optional[float] = 1.0,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.scaler = scaler

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> Dict[str, list[float]]:
        self.model.to(self.device)
        history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        checkpoint_dir = ensure_dir(checkpoint_dir or "models/artifacts/checkpoints")
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                history["val_loss"].append(val_loss)

            LOGGER.info("Epoch %d | train_loss=%.5f | val_loss=%s", epoch, train_loss, f"{val_loss:.5f}" if val_loss is not None else "N/A")

            if val_loss is not None and early_stopping is not None:
                if early_stopping.step(val_loss):
                    LOGGER.info("Early stopping triggered at epoch %d", epoch)
                    break
                if early_stopping.best_score == val_loss:
                    best_state = self.model.state_dict()
                    torch.save(best_state, Path(checkpoint_dir) / "best_model.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        mode = "train" if train else "eval"
        getattr(self.model, mode)()
        total_loss = 0.0
        batches = 0

        for features, targets, mask, _ in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device) if mask is not None else None

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = self.model(features)
                loss_raw = self.loss_fn(outputs.squeeze(-1), targets.squeeze(-1))
                if loss_raw.ndim > 1:
                    loss_raw = loss_raw.mean(dim=-1)
                if mask is not None:
                    weights = mask.mean(dim=-1)
                    loss = (loss_raw * weights).mean()
                else:
                    loss = loss_raw.mean()

            if train:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(batches, 1)

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for features, _, _, _ in loader:
                features = features.to(self.device)
                outputs = self.model(features)
                preds.append(outputs.cpu().numpy())
        return np.concatenate(preds, axis=0)
