#!/usr/bin/env python3
"""Shared plotting style utilities for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}

FIGSIZE = {
    "heatmap": (9.0, 5.4),
    "paired_confusion": (10.5, 4.8),
    "slope": (8.4, 4.8),
    "line_ablation": (7.8, 4.6),
    "embedding_pair": (13.0, 5.2),
}

PUB_LABEL = {
    "iemocap_to_podcast": "IEMOCAP→PODCAST",
    "podcast_to_iemocap": "PODCAST→IEMOCAP",
    "iemocap_plus_cremad_to_podcast": "IEMOCAP+CREMA-D→PODCAST",
    "baseline": "Baseline",
    "stage1": "Stage 1",
    "stage2": "Stage 2",
}


def apply_shared_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(DEFAULT_STYLE)


def save_png_pdf(fig: plt.Figure, base_no_suffix: Path) -> None:
    base_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_no_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

