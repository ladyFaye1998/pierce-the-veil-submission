"""
Render the ablation results as two publication-grade figures:

  figures/06_ablation_bar.png  -- mean SRMSE with 95% bootstrap CI per variant
  figures/07_ablation_table.png -- table render of full ablation summary

Run after src/ablation.py has written src/ablation_results.json.

Author: Lady Faye  (Kaggle: ladyfaye)
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "src" / "ablation_results.json"
FIG = ROOT / "figures"


def main():
    data = json.loads(RESULTS.read_text())
    summary = data["summary"]
    rows = sorted(summary.items(), key=lambda kv: kv[1]["mean_srmse"])

    names = [k for k, _ in rows]
    means = np.array([v["mean_srmse"] for _, v in rows])
    lo = np.array([v["ci95_lo"] for _, v in rows])
    hi = np.array([v["ci95_hi"] for _, v in rows])
    beats = np.array([v["beats_zeros_count"] for _, v in rows])

    err = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(names))
    colors = []
    for n in names:
        if n == "all_six_a045_SHIPPED":
            colors.append("#c62828")
        elif n == "zeros_baseline":
            colors.append("#424242")
        elif "liu2024" in n or "stadler2024" in n or "fang2024" in n:
            colors.append("#1565c0")
        else:
            colors.append("#2e7d32")
    ax.barh(y_pos, means - 1.0, xerr=err, color=colors, alpha=0.75,
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("(mean SRMSE - 1.0)   with 95% bootstrap CI")
    ax.set_title(
        f"Algorithmic ablation: {len(names)} reconstructor variants  x  "
        f"{data['config']['n_seeds']} Monte-Carlo seeds  "
        f"(D_hat={data['config']['D_hat']}, surrogate=make_classification)"
    )
    for i, b in enumerate(beats):
        ax.text(0.0015, y_pos[i], f"  {b}/{data['config']['n_seeds']}",
                fontsize=7, va="center", color="black", alpha=0.7)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#c62828", alpha=0.75,
                      label="SHIPPED variant"),
        plt.Rectangle((0, 0), 1, 1, color="#424242", alpha=0.75,
                      label="all-zeros baseline"),
        plt.Rectangle((0, 0), 1, 1, color="#1565c0", alpha=0.75,
                      label="2024 literature refinement (zero-paired-data approximation)"),
        plt.Rectangle((0, 0), 1, 1, color="#2e7d32", alpha=0.75,
                      label="ablation / calibration sweep"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "06_ablation_bar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    cell_data = []
    for i, (n, st) in enumerate(rows, start=1):
        cell_data.append([
            f"{i}",
            n,
            f"{st['mean_srmse']:.6f}",
            f"[{st['ci95_lo']:.4f}, {st['ci95_hi']:.4f}]",
            f"{st['beats_zeros_count']}/{data['config']['n_seeds']}",
        ])
    table = ax.table(
        cellText=cell_data,
        colLabels=["rank", "variant", "mean_srmse",
                   "95% bootstrap CI", "beats zeros"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for i in range(len(rows)):
        row_idx = i + 1
        for col_idx in range(5):
            cell = table[row_idx, col_idx]
            name = rows[i][0]
            if name == "all_six_a045_SHIPPED":
                cell.set_facecolor("#ffebee")
            elif name == "zeros_baseline":
                cell.set_facecolor("#eeeeee")
            elif "2024" in name:
                cell.set_facecolor("#e3f2fd")
    ax.set_title(
        f"Algorithmic ablation summary ({len(rows)} variants x "
        f"{data['config']['n_seeds']} seeds)\n"
        "shipped variant highlighted in red; literature refinements in blue",
        fontsize=11, pad=20
    )
    fig.tight_layout()
    fig.savefig(FIG / "07_ablation_table.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {FIG / '06_ablation_bar.png'}")
    print(f"  wrote {FIG / '07_ablation_table.png'}")


if __name__ == "__main__":
    main()
