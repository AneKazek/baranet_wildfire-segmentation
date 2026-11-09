"""Training loop with AMP, gradient accumulation, warmup+cosine and early stopping.

Batch contract (produced by the caller's dataloader):

    inputs           float32 [B, T=3, C=8, H, W]   required
    sp_map           int64   [B, H, W]             optional (SLIC precomputed)
    ba_target        float32 [B, 1, H, W]          required
    boundary_target  float32 [B, 1, H, W]          optional
    valid_mask       bool    [B, 1, H, W]          optional
    fire_id          list[str] of length B         optional

The dataset itself is intentionally not part of this package.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..losses import CombinedLoss
from ..metrics import SegmentationMetrics
from .checkpoint import save_checkpoint
from .reproducibility import environment_report, git_commit, seed_everything


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        out_dir: str | Path,
        device: str | torch.device | None = None,
        seed: int = 17,
        threshold: float = 0.5,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.threshold = float(threshold)
        self.seed = seed

        seed_everything(seed)
        self.model = model.to(self.device)
        self.criterion = CombinedLoss(cfg.loss)
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_factor
        )
        self.use_amp = bool(cfg.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.global_step = 0

        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_dir = self.out_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._write_provenance()

    # ---------------------------------------------------------------- setup

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim > 1 and "norm" not in name.lower():
                decay.append(param)
            else:
                no_decay.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.learning_rate,
        )

    def _lr_factor(self, epoch: int) -> float:
        warmup = max(1, self.cfg.warmup_epochs)
        if epoch < warmup:
            return float(epoch + 1) / warmup
        remaining = max(1, self.cfg.epochs - warmup)
        progress = min(1.0, (epoch - warmup) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _write_provenance(self) -> None:
        model_cfg = getattr(self.model, "cfg", None)
        payload = {
            "training": self.cfg.model_dump(),
            "model": model_cfg.model_dump() if model_cfg is not None else None,
            "seed": self.seed,
        }
        (self.out_dir / "config.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        (self.out_dir / "environment.txt").write_text(environment_report(), encoding="utf-8")
        (self.out_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
        (self.out_dir / "seed.txt").write_text(f"{self.seed}\n", encoding="utf-8")

    # ----------------------------------------------------------------- data

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict]:
        out = self.model(batch["inputs"], batch.get("sp_map"))
        losses = self.criterion(
            out["ba_logits"],
            batch["ba_target"],
            out.get("boundary_logits"),
            batch.get("boundary_target"),
            batch.get("valid_mask"),
        )
        return losses, out

    # -------------------------------------------------------------- training

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        accum = max(1, self.cfg.gradient_accumulation_steps)
        totals = {"loss": 0.0, "focal_tversky": 0.0, "dice": 0.0, "boundary": 0.0}
        batches = 0
        positives, pixels = 0.0, 0.0
        started = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(self.train_loader):
            batch = self._to_device(raw_batch)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                losses, _ = self._forward_loss(batch)
                scaled = losses["loss"] / accum
            self.scaler.scale(scaled).backward()

            last = step + 1 == len(self.train_loader)
            if (step + 1) % accum == 0 or last:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            for key in totals:
                totals[key] += float(losses[key].detach())
            batches += 1
            target = batch["ba_target"]
            if batch.get("valid_mask") is not None:
                mask = batch["valid_mask"].float()
                positives += float((target * mask).sum())
                pixels += float(mask.sum())
            else:
                positives += float(target.sum())
                pixels += float(target.numel())

        stats = {key: value / max(1, batches) for key, value in totals.items()}
        stats["seconds"] = time.time() - started
        stats["positive_fraction"] = positives / max(1.0, pixels)
        stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return stats

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict[str, float]:
        loader = loader or self.val_loader
        self.model.eval()
        metrics = SegmentationMetrics(self.threshold)
        total_loss, batches = 0.0, 0
        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                losses, out = self._forward_loss(batch)
            total_loss += float(losses["loss"].detach())
            batches += 1
            metrics.update(
                torch.sigmoid(out["ba_logits"]),
                batch["ba_target"],
                batch.get("valid_mask"),
                batch.get("fire_id"),
            )
        result = metrics.compute()
        result["loss"] = total_loss / max(1, batches)
        result["threshold"] = self.threshold
        return result

    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        best_score, best_epoch, patience = -1.0, -1, 0
        history: list[dict[str, Any]] = []

        for epoch in range(cfg.epochs):
            train_stats = self._train_epoch()
            val_stats = self.evaluate()
            self.scheduler.step()

            row = {
                "epoch": epoch,
                **train_stats,
                **{f"val_{key}": value for key, value in val_stats.items()},
            }
            history.append(row)
            train_row = {k: v for k, v in row.items() if not k.startswith("val_")}
            self._append_log("train.csv", train_row)
            self._append_log("validation.csv", {"epoch": epoch, **val_stats})

            meta = self._checkpoint_metadata()
            save_checkpoint(
                self.ckpt_dir / "latest.pt",
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                metadata=meta,
            )

            score = val_stats["macro_f1"]
            if score > best_score + 1e-6:
                best_score, best_epoch, patience = score, epoch, 0
                save_checkpoint(
                    self.ckpt_dir / "best.pt",
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    metadata=meta,
                )
            elif epoch >= cfg.warmup_epochs:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        return {"best_macro_f1": best_score, "best_epoch": best_epoch, "history": history}

    def _checkpoint_metadata(self) -> dict[str, Any]:
        model_cfg = getattr(self.model, "cfg", None)
        return {
            "model_config": None if model_cfg is None else model_cfg.model_dump(),
            "train_config": self.cfg.model_dump(),
            "threshold": self.threshold,
            "seed": self.seed,
            "git_commit": git_commit(),
        }

    def _append_log(self, name: str, row: dict[str, Any]) -> None:
        path = self.log_dir / name
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
