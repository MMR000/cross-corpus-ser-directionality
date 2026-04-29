#!/usr/bin/env python3
"""Generate Pareto performance-stability scatter from multi-seed summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_style import apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _resolve_csv() -> Path:
    preferred = Path("/mnt/data/plotdata_multiseed_robustness.csv")
    fallback = REPORTS / "plotdata_multiseed_robustness.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing input CSV:\n- {preferred}\n- {fallback}")


def _short_label(exp: str) -> str:
    m = {
        "iemocap_to_podcast_chunk": "i2p-chunk",
        "iemocap_to_podcast_chunk_domain_utt": "i2p-s1",
        "podcast_to_iemocap_chunk": "p2i-chunk",
        "podcast_to_iemocap_chunk_domain_utt": "p2i-s1",
    }
    return m.get(exp, exp.replace("iemocap_to_podcast", "i2p").replace("podcast_to_iemocap", "p2i"))


def _direction(exp: str) -> str:
    if exp.startswith("iemocap_to_podcast"):
        return "IEMOCAP→PODCAST"
    if exp.startswith("podcast_to_iemocap"):
        return "PODCAST→IEMOCAP"
    return "Other"


def _is_pareto_best(df: pd.DataFrame, i: int) -> bool:
    xi = float(df.iloc[i]["std_test_uar"])   # minimize
    yi = float(df.iloc[i]["mean_test_uar"])  # maximize
    for j in range(len(df)):
        if j == i:
            continue
        xj = float(df.iloc[j]["std_test_uar"])
        yj = float(df.iloc[j]["mean_test_uar"])
        no_worse = (xj <= xi) and (yj >= yi)
        strictly_better = (xj < xi) or (yj > yi)
        if no_worse and strictly_better:
            return False
    return True


def main() -> None:
    apply_shared_style()
    in_csv = _resolve_csv()
    df = pd.read_csv(in_csv).copy()

    required = {"experiment", "mean_test_uar", "std_test_uar", "mean_test_macro_f1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["short_label"] = df["experiment"].map(_short_label)
    df["direction"] = df["experiment"].map(_direction)
    df["size"] = 800 * (df["mean_test_macro_f1"] - df["mean_test_macro_f1"].min() + 0.02)
    df["pareto_best"] = [ _is_pareto_best(df, i) for i in range(len(df)) ]

    color_map = {
        "IEMOCAP→PODCAST": "#1f77b4",
        "PODCAST→IEMOCAP": "#d62728",
        "Other": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for direction in sorted(df["direction"].unique()):
        d = df[df["direction"] == direction]
        ax.scatter(
            d["std_test_uar"],
            d["mean_test_uar"],
            s=d["size"],
            c=color_map.get(direction, "#7f7f7f"),
            alpha=0.78,
            edgecolor="black",
            linewidth=0.7,
            label=direction,
            zorder=3,
        )
        # Optional error bars from std itself (visual uncertainty cue)
        ax.errorbar(
            d["std_test_uar"],
            d["mean_test_uar"],
            yerr=d["std_test_uar"],
            fmt="none",
            ecolor=color_map.get(direction, "#7f7f7f"),
            elinewidth=1.0,
            alpha=0.35,
            capsize=2,
            zorder=2,
        )

    # Label points with slight offsets for readability
    for _, r in df.iterrows():
        ax.annotate(
            r["short_label"],
            (r["std_test_uar"], r["mean_test_uar"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.25},
        )

    # Reference lines
    ax.axvline(df["std_test_uar"].median(), color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(df["mean_test_uar"].median(), color="black", linestyle="--", linewidth=1.0, alpha=0.5)

    # Highlight Pareto frontier points
    pareto = df[df["pareto_best"]].sort_values("std_test_uar")
    if len(pareto) > 1:
        ax.plot(
            pareto["std_test_uar"],
            pareto["mean_test_uar"],
            color="black",
            linestyle=":",
            linewidth=1.4,
            marker="o",
            markersize=4,
            zorder=4,
            label="Pareto frontier",
        )
    elif len(pareto) == 1:
        p = pareto.iloc[0]
        ax.scatter([p["std_test_uar"]], [p["mean_test_uar"]], s=180, facecolors="none", edgecolors="black", linewidth=1.6)

    ax.set_xlabel("UAR Std Across Seeds (lower is better)")
    ax.set_ylabel("Mean UAR Across Seeds (higher is better)")
    ax.set_title("Performance vs Stability (Pareto View)")
    ax.legend(frameon=True, title="", loc="lower right")
    ax.grid(alpha=0.25, zorder=0)

    save_png_pdf(fig, REPORTS / "fig_pareto_performance_stability")

    out_csv = REPORTS / "plotdata_pareto_performance_stability.csv"
    df.to_csv(out_csv, index=False)

    lines = [
        "## Pareto Performance-Stability Note",
        "",
        "Objective: maximize mean_test_uar and minimize std_test_uar.",
        "Pareto-best points are those not dominated on both objectives.",
        "",
        "Pareto-best models:",
    ]
    if pareto.empty:
        lines.append("- (none)")
    else:
        for _, r in pareto.iterrows():
            lines.append(
                f"- {r['short_label']} ({r['direction']}): "
                f"mean_test_uar={r['mean_test_uar']:.3f}, std_test_uar={r['std_test_uar']:.3f}, "
                f"mean_test_macro_f1={r['mean_test_macro_f1']:.3f}"
            )
    (REPORTS / "fig_pareto_performance_stability_note.md").write_text("\n".join(lines), encoding="utf-8")

    print("Generated:")
    print("- reports/fig_pareto_performance_stability.png")
    print("- reports/fig_pareto_performance_stability.pdf")
    print("- reports/plotdata_pareto_performance_stability.csv")
    print("- reports/fig_pareto_performance_stability_note.md")


if __name__ == "__main__":
    main()

