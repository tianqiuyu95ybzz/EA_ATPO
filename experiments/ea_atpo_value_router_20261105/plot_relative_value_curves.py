#!/usr/bin/env python3
"""Render accuracy-versus-token curves from frozen experiment summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs/relative_value_20261106"

for split in ("validation", "test"):
    folder = ROOT / split
    source = folder / f"{split}_accuracy_cost_curve.csv"
    if not source.exists():
        continue
    frame = pd.read_csv(source)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for policy, rows in frame.groupby("policy", sort=False):
        ax.plot(rows["mean_total_tokens"], rows["accuracy"], marker="o", linewidth=1.7,
                label=policy.replace("_", " "))
    ax.set_xlabel("Mean total model tokens per case")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{split.title()} accuracy–token cost curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(folder / f"{split}_accuracy_cost_curve.png", dpi=180)
    plt.close(fig)
