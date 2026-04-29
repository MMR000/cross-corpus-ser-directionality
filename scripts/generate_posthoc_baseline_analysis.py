#!/usr/bin/env python3
"""Generate post-hoc analysis artifacts from completed baseline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIORITY_RUNS = [
    "iemocap_in_corpus_meanpool",
    "iemocap_in_corpus_attnpool",
    "iemocap_in_corpus_chunk",
    "podcast_in_corpus_meanpool",
    "podcast_in_corpus_attnpool",
    "podcast_in_corpus_chunk",
    "iemocap_to_podcast_meanpool",
    "iemocap_to_podcast_attnpool",
    "iemocap_to_podcast_chunk",
    "podcast_to_iemocap_meanpool",
    "podcast_to_iemocap_attnpool",
    "podcast_to_iemocap_chunk",
]

DEFAULT_CLASS_NAMES = ["angry", "happy", "sad", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate post-hoc analysis from saved experiment confusion matrices.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="ser_project root directory.")
    parser.add_argument("--exp-dir", type=Path, default=Path("exp"), help="Experiment root directory.")
    parser.add_argument("--strict-priority-only", action="store_true", help="Process only the 12 priority runs.")
    return parser.parse_args()


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, np.clip(den, 1e-12, None))


def _plot_matrix(matrix: np.ndarray, class_names: list[str], title: str, output_path: Path, fmt: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    plt.close("all")


def _plot_f1_bar(class_names: list[str], f1_scores: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    x = np.arange(len(class_names))
    ax.bar(x, f1_scores, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Emotion Class")
    ax.set_title("Class-wise F1")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    plt.close("all")


def _class_names_for_matrix(cm: np.ndarray) -> list[str]:
    if cm.shape[0] == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return [f"class_{i}" for i in range(cm.shape[0])]


def generate_for_run(run_dir: Path) -> bool:
    input_cm_path = run_dir / "confusion_test.csv"
    if not input_cm_path.exists():
        return False

    cm = np.loadtxt(input_cm_path, delimiter=",")
    if cm.ndim == 1:
        cm = np.expand_dims(cm, 0)
    cm = cm.astype(np.float64)
    class_names = _class_names_for_matrix(cm)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Core matrices
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = _safe_div(cm, row_sum)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(analysis_dir / "confusion_matrix.csv")
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        analysis_dir / "normalized_confusion_matrix.csv"
    )

    # Class-wise metrics from confusion matrix
    tp = np.diag(cm)
    recall = _safe_div(tp, cm.sum(axis=1))
    precision = _safe_div(tp, cm.sum(axis=0))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    pd.DataFrame({"class": class_names, "f1": f1}).to_csv(analysis_dir / "classwise_f1.csv", index=False)
    pd.DataFrame({"class": class_names, "recall": recall}).to_csv(
        analysis_dir / "classwise_recall.csv", index=False
    )

    # Plots
    _plot_matrix(cm, class_names, "Confusion Matrix", analysis_dir / "confusion_matrix.png", ".0f")
    _plot_matrix(cm_norm, class_names, "Normalized Confusion Matrix", analysis_dir / "normalized_confusion_matrix.png", ".2f")
    _plot_f1_bar(class_names, f1, analysis_dir / "classwise_f1_barplot.png")
    return True


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    exp_dir = (project_root / args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    priority_done = []
    priority_missing = []
    for run in PRIORITY_RUNS:
        run_dir = exp_dir / run
        if generate_for_run(run_dir):
            priority_done.append(run)
        else:
            priority_missing.append(run)

    other_done = []
    if not args.strict_priority_only:
        for sub in sorted(exp_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name in PRIORITY_RUNS:
                continue
            if generate_for_run(sub):
                other_done.append(sub.name)

    print("Post-hoc analysis complete.")
    print(f"Priority runs processed: {len(priority_done)}")
    for name in priority_done:
        print(f"  - done: {name}")
    if priority_missing:
        print(f"Priority runs missing/unfinished: {len(priority_missing)}")
        for name in priority_missing:
            print(f"  - missing confusion_test.csv: {name}")
    if other_done:
        print(f"Additional runs processed: {len(other_done)}")
        for name in other_done:
            print(f"  - done: {name}")


if __name__ == "__main__":
    main()
