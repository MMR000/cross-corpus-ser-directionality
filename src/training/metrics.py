"""Metrics and reporting utilities for SER training/evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, recall_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    wa = float((y_true == y_pred).mean())
    uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"uar": uar, "wa": wa, "macro_f1": macro_f1}


def export_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_names: list[str],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = output_prefix.with_suffix(".csv")
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Annotate cells.
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    plt.close("all")


def export_training_curves(history_df, output_dir: Path) -> None:
    """Save static training curve plots from epoch history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if history_df is None or len(history_df) == 0:
        return

    plot_defs = [
        ("train_loss", "Train Loss vs Epoch", "train_loss_vs_epoch.png"),
        ("val_loss", "Validation Loss vs Epoch", "val_loss_vs_epoch.png"),
        ("val_uar", "Validation UAR vs Epoch", "val_uar_vs_epoch.png"),
        ("learning_rate", "Learning Rate vs Epoch", "learning_rate_vs_epoch.png"),
    ]

    x = history_df["epoch"].to_numpy()
    for key, title, filename in plot_defs:
        if key not in history_df.columns:
            continue
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, history_df[key].to_numpy(), marker="o", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    plt.close("all")


def export_analysis_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_names: list[str],
    output_dir: Path,
) -> None:
    """Export paper-useful static evaluation figures under analysis/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.clip(row_sums, 1e-12, None))

    # 1) raw confusion matrix
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # 2) normalized confusion matrix
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / "normalized_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # 3) classwise F1 barplot
    _, _, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    x = np.arange(len(label_names))
    ax.bar(x, f1_scores, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Emotion Class")
    ax.set_title("Class-wise F1")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "classwise_f1_barplot.png", dpi=180)
    plt.close(fig)
    plt.close("all")
