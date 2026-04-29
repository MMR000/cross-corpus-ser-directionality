"""Shared trainer for Phase 2 SER baselines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.datasets import ID_TO_EMOTION
from src.training.metrics import (
    compute_classification_metrics,
    export_analysis_figures,
    export_confusion_matrix,
    export_training_curves,
)
from src.training.utils import safe_torch_load


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: Optional[float]
    utterance_domain_loss_weight: float = 0.0
    chunk_domain_loss_weight: float = 0.0


class SERTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        config: TrainerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.logger = logger or logging.getLogger(__name__)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.history: list[dict] = []

    def _step_train(self, batch: dict) -> dict[str, float]:
        self.model.train()
        waveforms = batch["waveforms"].to(self.device)
        lengths = batch["lengths"].to(self.device)
        labels = batch["labels"].to(self.device)
        domain_ids = batch.get("domain_ids")
        if domain_ids is not None:
            domain_ids = domain_ids.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(waveforms, lengths, domain_ids=domain_ids)
        loss_emo = self.criterion(outputs["logits"], labels)
        loss = loss_emo
        loss_dom_tensor: Optional[torch.Tensor] = None
        loss_chunk_dom_tensor: Optional[torch.Tensor] = None
        if (
            self.cfg.utterance_domain_loss_weight > 0.0
            and domain_ids is not None
            and "domain_logits" in outputs
        ):
            loss_dom_tensor = self.criterion(outputs["domain_logits"], domain_ids)
            loss = loss + self.cfg.utterance_domain_loss_weight * loss_dom_tensor
        if (
            self.cfg.chunk_domain_loss_weight > 0.0
            and "chunk_domain_logits" in outputs
            and "chunk_domain_targets" in outputs
        ):
            loss_chunk_dom_tensor = self.criterion(
                outputs["chunk_domain_logits"],
                outputs["chunk_domain_targets"],
            )
            loss = loss + self.cfg.chunk_domain_loss_weight * loss_chunk_dom_tensor

        loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        out = {
            "loss": float(loss.detach().item()),
            "loss_emo": float(loss_emo.detach().item()),
        }
        if loss_dom_tensor is not None:
            out["loss_dom"] = float(loss_dom_tensor.detach().item())
        else:
            out["loss_dom"] = 0.0
        if loss_chunk_dom_tensor is not None:
            out["loss_dom_chunk"] = float(loss_chunk_dom_tensor.detach().item())
        else:
            out["loss_dom_chunk"] = 0.0
        return out

    def _step_train_uda(self, source_batch: dict, target_batch: dict) -> dict[str, float]:
        """
        UDA step for single-source cross-corpus adaptation.
        - Emotion loss uses labeled source batch only.
        - Domain loss uses both source and unlabeled target batches.
        """
        self.model.train()
        src_waveforms = source_batch["waveforms"].to(self.device)
        src_lengths = source_batch["lengths"].to(self.device)
        src_labels = source_batch["labels"].to(self.device)
        src_domain_ids = source_batch.get("domain_ids")
        if src_domain_ids is not None:
            src_domain_ids = src_domain_ids.to(self.device)

        tgt_waveforms = target_batch["waveforms"].to(self.device)
        tgt_lengths = target_batch["lengths"].to(self.device)
        tgt_domain_ids = target_batch.get("domain_ids")
        if tgt_domain_ids is not None:
            tgt_domain_ids = tgt_domain_ids.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        src_out = self.model(src_waveforms, src_lengths, domain_ids=src_domain_ids)
        tgt_out = self.model(tgt_waveforms, tgt_lengths, domain_ids=tgt_domain_ids)

        loss_emo = self.criterion(src_out["logits"], src_labels)
        loss = loss_emo
        loss_dom_tensor: Optional[torch.Tensor] = None
        loss_chunk_dom_tensor: Optional[torch.Tensor] = None

        if (
            self.cfg.utterance_domain_loss_weight > 0.0
            and src_domain_ids is not None
            and tgt_domain_ids is not None
            and "domain_logits" in src_out
            and "domain_logits" in tgt_out
        ):
            dom_logits = torch.cat([src_out["domain_logits"], tgt_out["domain_logits"]], dim=0)
            dom_labels = torch.cat([src_domain_ids, tgt_domain_ids], dim=0)
            loss_dom_tensor = self.criterion(dom_logits, dom_labels)
            loss = loss + self.cfg.utterance_domain_loss_weight * loss_dom_tensor
        if (
            self.cfg.chunk_domain_loss_weight > 0.0
            and "chunk_domain_logits" in src_out
            and "chunk_domain_targets" in src_out
            and "chunk_domain_logits" in tgt_out
            and "chunk_domain_targets" in tgt_out
        ):
            chunk_logits = torch.cat(
                [src_out["chunk_domain_logits"], tgt_out["chunk_domain_logits"]],
                dim=0,
            )
            chunk_labels = torch.cat(
                [src_out["chunk_domain_targets"], tgt_out["chunk_domain_targets"]],
                dim=0,
            )
            loss_chunk_dom_tensor = self.criterion(chunk_logits, chunk_labels)
            loss = loss + self.cfg.chunk_domain_loss_weight * loss_chunk_dom_tensor

        loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        out = {
            "loss": float(loss.detach().item()),
            "loss_emo": float(loss_emo.detach().item()),
            "src_batch_size": float(src_labels.numel()),
            "tgt_batch_size": float(tgt_waveforms.size(0)),
        }
        if loss_dom_tensor is not None:
            out["loss_dom"] = float(loss_dom_tensor.detach().item())
        else:
            out["loss_dom"] = 0.0
        if loss_chunk_dom_tensor is not None:
            out["loss_dom_chunk"] = float(loss_chunk_dom_tensor.detach().item())
        else:
            out["loss_dom_chunk"] = 0.0
        return out

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader, split_name: str, show_progress: bool = True) -> dict:
        self.model.eval()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        eval_iter = tqdm(
            loader,
            total=len(loader),
            desc=f"[{split_name}]",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for batch in eval_iter:
            waveforms = batch["waveforms"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(waveforms, lengths, domain_ids=None)["logits"]
            loss = self.criterion(logits, labels)
            losses.append(float(loss.item()))

            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            eval_iter.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
        metrics = compute_classification_metrics(y_true_np, y_pred_np)
        metrics["loss"] = float(np.mean(losses) if losses else 0.0)
        metrics["split"] = split_name
        metrics["num_samples"] = int(len(y_true))
        metrics["y_true"] = y_true_np
        metrics["y_pred"] = y_pred_np
        return metrics

    def _save_checkpoint(self, epoch: int, val_metrics: dict, config_dict: dict) -> Path:
        ckpt_path = self.output_dir / "best.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred"}},
                "config": config_dict,
                "id_to_emotion": ID_TO_EMOTION,
            },
            ckpt_path,
        )
        return ckpt_path

    def _export_confusion(self, metrics: dict, name: str) -> None:
        labels = sorted(ID_TO_EMOTION.keys())
        names = [ID_TO_EMOTION[i] for i in labels]
        export_confusion_matrix(
            y_true=metrics["y_true"],
            y_pred=metrics["y_pred"],
            labels=labels,
            label_names=names,
            output_prefix=self.output_dir / f"confusion_{name}",
        )

    def fit(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: Optional[DataLoader],
        config_dict: dict,
        target_unlabeled_loader: Optional[DataLoader] = None,
    ) -> dict:
        best_uar = -1.0
        best_epoch = -1
        best_ckpt: Optional[Path] = None
        training_start = time.time()

        epoch_iter = tqdm(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="epochs",
            leave=True,
            dynamic_ncols=True,
        )
        for epoch in epoch_iter:
            epoch_start = time.time()
            losses = []
            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"[epoch {epoch}/{self.cfg.epochs}] train",
                leave=False,
                dynamic_ncols=True,
            )
            train_dom_losses: list[float] = []
            train_chunk_dom_losses: list[float] = []
            target_iter = iter(target_unlabeled_loader) if target_unlabeled_loader is not None else None
            for batch_idx, batch in enumerate(train_iter, start=1):
                if target_iter is None:
                    step = self._step_train(batch)
                else:
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_unlabeled_loader)
                        target_batch = next(target_iter)
                    step = self._step_train_uda(batch, target_batch)
                losses.append(step["loss"])
                train_dom_losses.append(step.get("loss_dom", 0.0))
                train_chunk_dom_losses.append(step.get("loss_dom_chunk", 0.0))
                postfix: dict[str, str] = {
                    "batch": f"{batch_idx}/{len(train_loader)}",
                    "loss": f"{np.mean(losses):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if self.cfg.utterance_domain_loss_weight > 0.0:
                    postfix["dom_utt"] = f"{np.mean(train_dom_losses):.4f}"
                if self.cfg.chunk_domain_loss_weight > 0.0:
                    postfix["dom_chunk"] = f"{np.mean(train_chunk_dom_losses):.4f}"
                train_iter.set_postfix(postfix)
            train_loss = float(np.mean(losses) if losses else 0.0)
            train_loss_dom = float(np.mean(train_dom_losses) if train_dom_losses else 0.0)
            train_loss_dom_chunk = float(np.mean(train_chunk_dom_losses) if train_chunk_dom_losses else 0.0)

            val_metrics = self.evaluate_loader(dev_loader, "dev", show_progress=True)
            elapsed_sec = float(time.time() - training_start)
            epoch_elapsed = float(time.time() - epoch_start)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_uar": val_metrics["uar"],
                "val_wa": val_metrics["wa"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_sec": elapsed_sec,
            }
            if self.cfg.utterance_domain_loss_weight > 0.0:
                row["train_loss_dom"] = train_loss_dom
            if self.cfg.chunk_domain_loss_weight > 0.0:
                row["train_loss_dom_chunk"] = train_loss_dom_chunk
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(self.output_dir / "train_log.csv", index=False)
            eta_epoch = epoch_iter.format_dict.get("remaining", None)
            eta_text = f"{eta_epoch:.1f}s" if isinstance(eta_epoch, (int, float)) else "n/a"
            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_uar=%.4f | lr=%.2e | "
                "epoch_elapsed=%.1fs | eta=%s",
                epoch,
                self.cfg.epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["uar"],
                self.optimizer.param_groups[0]["lr"],
                epoch_elapsed,
                eta_text,
            )
            epoch_iter.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "val_uar": f"{val_metrics['uar']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if val_metrics["uar"] > best_uar:
                best_uar = float(val_metrics["uar"])
                best_epoch = epoch
                best_ckpt = self._save_checkpoint(epoch, val_metrics, config_dict)
                self._export_confusion(val_metrics, "dev_best")

        history_path = self.output_dir / "train_log.csv"
        hist_df = pd.DataFrame(self.history)
        hist_df.to_csv(history_path, index=False)
        export_training_curves(hist_df, self.output_dir / "plots")

        summary = {"best_epoch": best_epoch, "best_dev_uar": best_uar, "checkpoint": str(best_ckpt)}

        if best_ckpt is not None:
            ckpt = safe_torch_load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        if test_loader is not None:
            test_metrics = self.evaluate_loader(test_loader, "test", show_progress=True)
            self._export_confusion(test_metrics, "test")
            export_analysis_figures(
                y_true=test_metrics["y_true"],
                y_pred=test_metrics["y_pred"],
                labels=sorted(ID_TO_EMOTION.keys()),
                label_names=[ID_TO_EMOTION[i] for i in sorted(ID_TO_EMOTION.keys())],
                output_dir=self.output_dir / "analysis",
            )
            summary.update(
                {
                    "test_uar": test_metrics["uar"],
                    "test_wa": test_metrics["wa"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_loss": test_metrics["loss"],
                    "test_num_samples": test_metrics["num_samples"],
                }
            )

        pd.DataFrame([summary]).to_csv(self.output_dir / "summary.csv", index=False)
        self.logger.info(
            "Training complete | best_epoch=%s | best_dev_uar=%.4f | checkpoint=%s | "
            "test_uar=%s | test_wa=%s | test_macro_f1=%s",
            best_epoch,
            best_uar,
            best_ckpt,
            f"{summary.get('test_uar', 'n/a')}",
            f"{summary.get('test_wa', 'n/a')}",
            f"{summary.get('test_macro_f1', 'n/a')}",
        )
        return summary
