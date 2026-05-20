"""Generate the publication-grade figures for the master notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
DATA = ROOT / "data" / "intercepted_data.csv"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY = "#1f3a68"
CRIM = "#b3294e"
GOLD = "#c79f3e"
TEAL = "#2a8b8f"
GREY = "#7a8285"


def load_z():
    return pd.read_csv(DATA).iloc[:, 0].to_numpy(np.float64)


def fig_distribution(z):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ax = axes[0]
    ax.hist(z, bins=80, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Marginal histogram of Z  (N = {len(z)})")
    ax.set_xlabel("z"); ax.set_ylabel("count")
    ax.axvline(z.mean(), color=CRIM, lw=1.2, ls="--", label=f"mean = {z.mean():+.3f}")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    a, loc, scale = stats.skewnorm.fit(z)
    xs = np.linspace(z.min() - 0.5, z.max() + 0.5, 400)
    ax.plot(xs, stats.skewnorm.pdf(xs, a, loc, scale), color=CRIM, lw=2,
            label=f"skew-normal\n a={a:.2f}, loc={loc:.2f}, scale={scale:.2f}")
    ax.hist(z, bins=80, density=True, color=NAVY, alpha=0.55, edgecolor="white",
            linewidth=0.3, label="empirical")
    ax.set_title("Best-fit skew-normal (KS p = {:.3f})".format(
        stats.kstest(z, lambda x: stats.skewnorm.cdf(x, a, loc, scale)).pvalue))
    ax.set_xlabel("z"); ax.set_ylabel("density")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[2]
    sorted_z = np.sort(z)
    quantiles = stats.norm.ppf(np.arange(1, len(z) + 1) / (len(z) + 1))
    coeffs = np.polyfit(quantiles, sorted_z, deg=3)
    fit = np.polyval(coeffs, quantiles)
    ss_res = np.sum((sorted_z - fit) ** 2)
    ss_tot = np.sum((sorted_z - sorted_z.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.scatter(quantiles, sorted_z, color=NAVY, s=4, alpha=0.6, label="sorted Z")
    ax.plot(quantiles, fit, color=GOLD, lw=1.6, label=f"cubic fit (R² = {r2:.4f})")
    ax.set_title("Polynomial-quantile fit (Φ⁻¹)")
    ax.set_xlabel("Φ⁻¹(rank/(N+1))"); ax.set_ylabel("sorted Z")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "01_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_mixture_and_duplicates(z):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ks = list(range(1, 7))
    bics = []
    for k in ks:
        gm = GaussianMixture(n_components=k, random_state=0, n_init=4)
        gm.fit(z.reshape(-1, 1))
        bics.append(gm.bic(z.reshape(-1, 1)))
    ax = axes[0]
    ax.plot(ks, bics, marker="o", color=NAVY, lw=1.8)
    best_k = ks[int(np.argmin(bics))]
    ax.scatter([best_k], [min(bics)], color=CRIM, s=110, zorder=5,
               label=f"best k = {best_k}")
    ax.set_title("Gaussian-mixture BIC vs k")
    ax.set_xlabel("k (mixture components)"); ax.set_ylabel("BIC (lower = better)")
    ax.legend(frameon=False)

    vals, counts = np.unique(np.round(z, 7), return_counts=True)
    dups = vals[counts >= 2]
    ax = axes[1]
    ax.hist(z, bins=80, color=GREY, alpha=0.45, edgecolor="white", linewidth=0.4)
    for d in dups:
        ax.axvline(d, color=CRIM, alpha=0.65, lw=0.8)
    ax.set_title(f"Duplicate codes ({len(dups)} values, all in negative tail)")
    ax.set_xlabel("z"); ax.set_ylabel("count")
    ax.annotate(f"all {len(dups)} duplicates ∈ [{dups.min():+.2f}, {dups.max():+.2f}]",
                xy=(0.02, 0.94), xycoords="axes fraction", color=CRIM,
                fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG / "02_mixture_and_duplicates.png", bbox_inches="tight")
    plt.close(fig)


def fig_signature_match():
    sig_path = ROOT / "src" / "signature_results.json"
    import json
    with open(sig_path) as f:
        data = json.load(f)
    top = data["top_30"][:20]
    labels = [f"D={r['D']}, {r['model']}, w={r['class_balance'][0]:.1f}"
              if r['kind'] == 'classifier'
              else f"D={r['D']}, {r['model']}, σ={r['noise']:.0f}"
              for r in top]
    w1 = [r["w1_after_rescale"] for r in top]
    ks_p = [r["ks_p"] for r in top]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(top))[::-1]
    bars = ax.barh(y, w1, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.4)
    for i, (b, p) in enumerate(zip(bars, ks_p)):
        col = GOLD if p > 0.05 else GREY
        ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                f"KS p={p:.2g}", va="center", fontsize=8, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0.10, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("Wasserstein-1 distance to Z (after rescale)")
    ax.set_title("Top-20 synthetic encoder fingerprints (lower W1 = better match)")
    top1 = top[0]
    ax.annotate(
        f"BEST: D={top1['D']}, LogReg, balance={top1['class_balance']},\n"
        f"sep={top1['class_sep']}, W1={top1['w1_after_rescale']:.3f}",
        xy=(top1['w1_after_rescale'], len(top) - 1),
        xytext=(0.50, len(top) - 4),
        color=CRIM, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CRIM, lw=1.5),
    )
    fig.tight_layout()
    fig.savefig(FIG / "03_signature_match.png", bbox_inches="tight")
    plt.close(fig)


def fig_reconstruction_risk():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from reconstruct import reconstruct, D_HAT  # noqa
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    z_pub = load_z()
    seeds = list(range(2024, 2024 + 30))
    srmses = []
    base = []
    for s in seeds:
        X, y = make_classification(
            n_samples=4096, n_features=D_HAT, n_informative=int(D_HAT * 0.8),
            n_redundant=0, n_classes=2, n_clusters_per_class=1, class_sep=0.5,
            weights=[0.8, 0.2], flip_y=0.01, random_state=s,
        )
        X_std = (X - X.mean(0)) / X.std(0)
        clf = LogisticRegression(max_iter=1500, random_state=s).fit(X, y)
        z_raw = clf.decision_function(X)
        z_std = (z_raw - z_raw.mean()) / z_raw.std()
        z_hid = z_std * z_pub.std() + z_pub.mean()
        X_hat = reconstruct(z_pub, z_hid)
        e = X_hat - X_std
        srmses.append(np.sqrt((e ** 2).mean()))
        base.append(np.sqrt((X_std ** 2).mean()))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(srmses, marker="o", color=NAVY, lw=1.5, label="our reconstruct()")
    ax.plot(base, marker="x", color=GREY, lw=1.0, ls="--", label="zeros baseline")
    ax.set_xlabel("synthetic surrogate index")
    ax.set_ylabel("SRMSE")
    ax.set_title(f"Reconstruction SRMSE across {len(seeds)} synthetic encoders (D=16)")
    ax.legend(frameon=False)
    ax.axhline(1.0, color=CRIM, lw=0.5, ls=":", alpha=0.5)
    ours = np.array(srmses)
    ax.text(0.02, 0.94,
            f"ours: mean = {ours.mean():.4f} ± {ours.std():.4f}\n"
            f"max = {ours.max():.4f}    min = {ours.min():.4f}",
            transform=ax.transAxes, color=NAVY, fontsize=9, va="top",
            family="monospace")
    fig.tight_layout()
    fig.savefig(FIG / "04_reconstruction_risk.png", bbox_inches="tight")
    plt.close(fig)


def fig_theoretical_floor():
    Ds = np.arange(1, 50)
    floor = np.sqrt((Ds - 1) / Ds)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(Ds, floor, color=NAVY, lw=1.8, marker="o", markersize=4)
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":", label="zeros baseline (SRMSE = 1)")
    ax.fill_between(Ds, floor, 1.0, color=GOLD, alpha=0.18,
                    label="window of attainable improvement")
    for D, label in [(16, "D=16\n(our primary)"), (132, "D=132\n(paper §10.1)")]:
        if D <= Ds.max():
            ax.scatter([D], [np.sqrt((D - 1) / D)], color=CRIM, s=70, zorder=5)
            ax.annotate(label, (D, np.sqrt((D - 1) / D)),
                        xytext=(D, 0.86), color=CRIM, fontsize=9,
                        ha="center",
                        arrowprops=dict(arrowstyle="-", color=CRIM, lw=0.8))
    ax.set_xlabel("True dimensionality D")
    ax.set_ylabel("Information-theoretic SRMSE floor")
    ax.set_title("Cramér-Rao floor: SRMSE ≥ √((D−1)/D) for any 1-D → D-D reconstruction")
    ax.set_ylim(0.0, 1.1)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "05_theoretical_floor.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, "Raw X\n(D unknown)", NAVY),
        (2.4, "VEIL\nencoder\n(SCRAE)", CRIM),
        (4.7, "Latent Ψ\n(E-dim)", NAVY),
        (7.0, "Downstream\nmodel head", CRIM),
        (9.3, "Z (1-D)\n4096 × 1", GOLD),
        (11.5, "X_hat\n(N × D)", TEAL),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 1.0), 1.5, 1.2, edgecolor=color,
                                    facecolor="white", lw=2.0))
        ax.text(x + 0.75, 1.6, label, ha="center", va="center",
                color=color, fontsize=10, fontweight="bold")
    for x in [1.8, 4.1, 6.4, 8.7]:
        ax.annotate("", xy=(x + 0.55, 1.6), xytext=(x, 1.6),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.annotate("", xy=(11.5, 1.6), xytext=(10.85, 1.6),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(11.15, 0.55, "reconstruct()", ha="center", color=TEAL,
            fontsize=10, fontweight="bold")
    ax.text(6.0, 2.7, "ENCODING PIPELINE  (forward, observable to attacker = only Z)",
            ha="center", color=NAVY, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "00_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def main():
    z = load_z()
    fig_pipeline_diagram()
    fig_distribution(z)
    fig_mixture_and_duplicates(z)
    fig_signature_match()
    fig_reconstruction_risk()
    fig_theoretical_floor()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
