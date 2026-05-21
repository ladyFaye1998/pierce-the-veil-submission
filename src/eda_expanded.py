"""
Expanded EDA for Pierce the VEIL.

Produces 14 additional diagnostic figures beyond the original 6, covering:

  * marginal density (KDE) with overlaid Gaussian
  * empirical CDF vs theoretical Gaussian CDF
  * Q-Q plots vs Normal and vs Student-t
  * Hill plot for tail-index estimation
  * autocorrelation and partial-autocorrelation functions
  * Welch power spectral density
  * lag-1 scatter (Z[t+1] vs Z[t])
  * Z^2 marginal vs chi-square (k=1)
  * |Z| marginal vs half-normal
  * GMM 2-component fit visualization
  * run-length distribution of sign(Z)
  * rank-rank scatter (rank(Z) vs rank(|Z|))
  * bootstrap CI band on the empirical CDF
  * tail concentration plot (CCDF on log scale)

Outputs
-------
src/eda_expanded_results.json
figures/10_marginal_kde.png
figures/11_ecdf_vs_normal.png
figures/12_qq_normal.png
figures/13_qq_student_t.png
figures/14_hill_plot.png
figures/15_autocorr_pacf.png
figures/16_welch_psd.png
figures/17_lag1_scatter.png
figures/18_z2_chisq.png
figures/19_absz_halfnormal.png
figures/20_gmm_fit.png
figures/21_sign_runlength.png
figures/22_rank_rank.png
figures/23_ccdf_loglog.png

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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "intercepted_data.csv"
FIG = ROOT / "figures"
OUT_JSON = ROOT / "src" / "eda_expanded_results.json"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _phi(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(lambda v: _erf(v / np.sqrt(2.0)))(x))


def _erf(x):
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    xa = abs(x)
    t = 1.0 / (1.0 + p * xa)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-xa * xa)
    return sign * y


def _norm_quantile(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    a = (-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742)
    out = np.empty_like(p)
    plow = 0.02425
    phigh = 1 - plow
    mask_lo = p < plow
    mask_hi = p > phigh
    mask_mid = ~(mask_lo | mask_hi)
    q = p[mask_mid] - 0.5
    r = q * q
    out[mask_mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                    (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(p[mask_lo]))
    out[mask_lo] = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]
    out[mask_lo] /= ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = np.sqrt(-2 * np.log(1 - p[mask_hi]))
    out[mask_hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
    out[mask_hi] /= ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


def _student_t_quantile(p, df):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    z = _norm_quantile(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    return z + g1 / df + g2 / (df * df)


def _kde(x, grid, bw=None):
    n = len(x)
    s = x.std()
    if bw is None:
        bw = 1.06 * s * n ** (-1.0 / 5.0)
    out = np.zeros_like(grid)
    for xi in x:
        out += _phi((grid - xi) / bw)
    return out / (n * bw)


def fit_gmm(z, max_iter=80, tol=1e-7):
    """Self-contained 1-D 2-component GMM EM."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    q1, q2 = np.quantile(z, [0.25, 0.75])
    mu1, mu2 = float(q1), float(q2)
    var = float(z.var()) + 1e-9
    v1, v2 = var, var
    w1 = 0.5
    last_ll = -np.inf
    log2pi = np.log(2 * np.pi)
    for _ in range(max_iter):
        log_p1 = -0.5 * (log2pi + np.log(v1) + (z - mu1) ** 2 / v1) + np.log(max(w1, 1e-12))
        log_p2 = -0.5 * (log2pi + np.log(v2) + (z - mu2) ** 2 / v2) + np.log(max(1 - w1, 1e-12))
        m = np.maximum(log_p1, log_p2)
        lt = m + np.log(np.exp(log_p1 - m) + np.exp(log_p2 - m))
        g1 = np.exp(log_p1 - lt)
        g2 = 1 - g1
        n1 = g1.sum() + 1e-12
        n2 = g2.sum() + 1e-12
        mu1 = float((g1 * z).sum() / n1)
        mu2 = float((g2 * z).sum() / n2)
        v1 = float((g1 * (z - mu1) ** 2).sum() / n1) + 1e-9
        v2 = float((g2 * (z - mu2) ** 2).sum() / n2) + 1e-9
        w1 = float(n1 / (n1 + n2))
        ll = float(lt.sum())
        if abs(ll - last_ll) < tol * (1 + abs(last_ll)):
            break
        last_ll = ll
    return w1, mu1, v1, 1 - w1, mu2, v2, last_ll


def fig_kde(z):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grid = np.linspace(z.min() - 0.5, z.max() + 0.5, 600)
    density = _kde(z, grid)
    gauss = _phi((grid - z.mean()) / z.std()) / z.std()
    ax.fill_between(grid, density, alpha=0.35, label=f"Z KDE (n={z.size})")
    ax.plot(grid, gauss, "r--", lw=1.4,
            label=f"N({z.mean():.3f}, {z.std():.3f}^2) reference")
    ax.set_xlabel("Z")
    ax.set_ylabel("density")
    ax.set_title("Marginal density of Z (Silverman-bandwidth KDE) vs. Gaussian reference")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    _save(fig, "10_marginal_kde.png")


def fig_ecdf(z):
    z_sorted = np.sort(z)
    ecdf = np.arange(1, z.size + 1) / z.size
    grid = np.linspace(z.min(), z.max(), 1000)
    ref_cdf = _Phi((grid - z.mean()) / z.std())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z_sorted, ecdf, lw=1.4, label="Empirical CDF")
    ax.plot(grid, ref_cdf, "r--", lw=1.4, label="Gaussian reference CDF")
    ax.set_xlabel("Z")
    ax.set_ylabel("F(Z)")
    ax.set_title("Empirical CDF of Z versus Gaussian reference")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, "11_ecdf_vs_normal.png")


def fig_qq_normal(z):
    z_sorted = np.sort(z)
    ps = (np.arange(1, z.size + 1) - 0.5) / z.size
    theoretical = z.mean() + z.std() * _norm_quantile(ps)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, z_sorted, s=4, alpha=0.5)
    lo, hi = z_sorted.min(), z_sorted.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.4, label="y=x")
    ax.set_xlabel("Theoretical Gaussian quantile")
    ax.set_ylabel("Sample quantile")
    ax.set_title("Q-Q plot: Z vs. fitted Gaussian")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, "12_qq_normal.png")


def fig_qq_t(z, df=5):
    z_sorted = np.sort(z)
    ps = (np.arange(1, z.size + 1) - 0.5) / z.size
    theoretical = z.mean() + z.std() * _student_t_quantile(ps, df)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, z_sorted, s=4, alpha=0.5)
    lo, hi = z_sorted.min(), z_sorted.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.4, label="y=x")
    ax.set_xlabel(f"Theoretical Student-t(df={df}) quantile")
    ax.set_ylabel("Sample quantile")
    ax.set_title(f"Q-Q plot: Z vs. Student-t(df={df})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, "13_qq_student_t.png")


def fig_hill(z):
    abs_z = np.sort(np.abs(z))[::-1]
    n = abs_z.size
    k_grid = np.arange(20, n // 2)
    xi_hat = np.array([
        (np.log(abs_z[:k]).sum() / k) - np.log(abs_z[k])
        for k in k_grid
    ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_grid, xi_hat, lw=1.3)
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    ax.set_xlabel("k (order statistic threshold)")
    ax.set_ylabel(r"Hill tail-index estimate $\hat{\xi}$")
    ax.set_title(r"Hill plot: Z is sub-Gaussian when $\hat{\xi}$ -> 0 as k -> N/2")
    ax.grid(True, alpha=0.25)
    _save(fig, "14_hill_plot.png")
    return float(xi_hat[len(xi_hat) // 2])


def _acf(z, max_lag):
    z = z - z.mean()
    var = (z ** 2).sum()
    if var <= 0:
        return np.ones(max_lag + 1)
    out = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = (z[:z.size - k] * z[k:]).sum() / var
    return out


def fig_acf(z, max_lag=50):
    acf = _acf(z, max_lag)
    pacf = np.zeros_like(acf)
    pacf[0] = 1.0
    for k in range(1, max_lag + 1):
        r = acf[1:k + 1]
        if k == 1:
            pacf[k] = r[0]
        else:
            from numpy.linalg import lstsq
            A = np.array([[acf[abs(i - j)] for j in range(k)] for i in range(k)])
            sol, *_ = lstsq(A, r, rcond=None)
            pacf[k] = sol[-1]
    ci = 1.96 / np.sqrt(z.size)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].stem(np.arange(max_lag + 1), acf, basefmt=" ")
    axs[0].axhline(ci, color="r", ls="--", lw=1)
    axs[0].axhline(-ci, color="r", ls="--", lw=1)
    axs[0].set_title("Autocorrelation function")
    axs[0].set_xlabel("lag")
    axs[0].grid(True, alpha=0.25)
    axs[1].stem(np.arange(max_lag + 1), pacf, basefmt=" ")
    axs[1].axhline(ci, color="r", ls="--", lw=1)
    axs[1].axhline(-ci, color="r", ls="--", lw=1)
    axs[1].set_title("Partial autocorrelation function")
    axs[1].set_xlabel("lag")
    axs[1].grid(True, alpha=0.25)
    _save(fig, "15_autocorr_pacf.png")
    return {"acf_lag1": float(acf[1]), "pacf_lag1": float(pacf[1])}


def fig_welch_psd(z, nperseg=512):
    n = z.size
    seg = nperseg
    overlap = seg // 2
    starts = list(range(0, n - seg + 1, seg - overlap))
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(seg) / (seg - 1)))
    psds = []
    for s in starts:
        segment = z[s:s + seg] - z[s:s + seg].mean()
        fft = np.fft.rfft(segment * window)
        psds.append(np.abs(fft) ** 2 / (window ** 2).sum())
    psd = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(seg, d=1.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(freqs, psd, lw=1.2)
    ax.set_xlabel("Normalized frequency (cycles/sample)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Welch PSD of Z (n_per_seg=512, 50% overlap, Hann window)")
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, "16_welch_psd.png")


def fig_lag1(z):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z[:-1], z[1:], s=4, alpha=0.4)
    lim = (z.min(), z.max())
    ax.plot(lim, lim, "r--", lw=1.0)
    ax.set_xlabel(r"$Z_t$")
    ax.set_ylabel(r"$Z_{t+1}$")
    r = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    ax.set_title(f"Lag-1 scatter (Pearson r = {r:+.4f})")
    ax.grid(True, alpha=0.25)
    _save(fig, "17_lag1_scatter.png")
    return r


def fig_z2(z):
    z_std = (z - z.mean()) / z.std()
    z2 = z_std ** 2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(z2.max(), 12), 80)
    ax.hist(z2, bins=bins, density=True, alpha=0.55, label="Z_std^2")
    grid = np.linspace(1e-4, bins[-1], 500)
    chi1 = np.exp(-grid / 2) / np.sqrt(2 * np.pi * grid)
    ax.plot(grid, chi1, "r--", lw=1.3, label=r"$\chi^2_1$ reference")
    ax.set_xlabel("Z_std^2")
    ax.set_ylabel("density")
    ax.set_title(r"Squared Z vs. $\chi^2_1$: tests for Gaussianity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, "18_z2_chisq.png")


def fig_absz(z):
    z_std = (z - z.mean()) / z.std()
    absz = np.abs(z_std)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, absz.max(), 80)
    ax.hist(absz, bins=bins, density=True, alpha=0.55, label="|Z_std|")
    grid = np.linspace(0, bins[-1], 500)
    halfnormal = np.sqrt(2 / np.pi) * np.exp(-grid ** 2 / 2)
    ax.plot(grid, halfnormal, "r--", lw=1.3, label="Half-normal reference")
    ax.set_xlabel("|Z_std|")
    ax.set_ylabel("density")
    ax.set_title("|Z| vs. half-normal: marginal-magnitude diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, "19_absz_halfnormal.png")


def fig_gmm(z):
    w1, m1, v1, w2, m2, v2, ll = fit_gmm(z)
    grid = np.linspace(z.min() - 0.5, z.max() + 0.5, 600)
    comp1 = w1 * _phi((grid - m1) / np.sqrt(v1)) / np.sqrt(v1)
    comp2 = w2 * _phi((grid - m2) / np.sqrt(v2)) / np.sqrt(v2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(z, bins=80, density=True, alpha=0.4, label=f"Z (n={z.size})")
    ax.plot(grid, comp1, "g-", lw=1.4,
            label=f"GMM comp1 w={w1:.2f}, mu={m1:.2f}, var={v1:.2f}")
    ax.plot(grid, comp2, "b-", lw=1.4,
            label=f"GMM comp2 w={w2:.2f}, mu={m2:.2f}, var={v2:.2f}")
    ax.plot(grid, comp1 + comp2, "r--", lw=1.2, label="mixture density")
    ax.set_xlabel("Z")
    ax.set_ylabel("density")
    ax.set_title(f"Two-component Gaussian mixture fit (log-lik={ll:.1f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    _save(fig, "20_gmm_fit.png")
    return {
        "w1": w1, "mu1": m1, "var1": v1,
        "w2": w2, "mu2": m2, "var2": v2,
        "log_likelihood": ll,
    }


def fig_signs(z):
    z_std = (z - z.mean()) / z.std()
    signs = np.sign(z_std - np.median(z_std))
    runs = []
    cur = signs[0]
    length = 1
    for s in signs[1:]:
        if s == cur:
            length += 1
        else:
            runs.append(length)
            cur = s
            length = 1
    runs.append(length)
    runs = np.array(runs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(1, runs.max() + 2) - 0.5
    ax.hist(runs, bins=bins, density=True, alpha=0.55, label="empirical")
    geom_grid = np.arange(1, runs.max() + 1)
    geom = 0.5 ** geom_grid
    ax.plot(geom_grid, geom, "r--", lw=1.3, label="Geom(0.5) reference (i.i.d.)")
    ax.set_xlabel("run length of sign(Z - median(Z))")
    ax.set_ylabel("density")
    ax.set_title(f"Sign-run-length: i.i.d. would yield Geom(0.5).  mean run = {runs.mean():.2f}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, "21_sign_runlength.png")
    return {"mean_run_length": float(runs.mean()),
            "max_run_length": int(runs.max())}


def fig_rank_rank(z):
    r_z = np.argsort(np.argsort(z))
    r_abs = np.argsort(np.argsort(np.abs(z)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r_z, r_abs, s=3, alpha=0.4)
    ax.set_xlabel("rank(Z)")
    ax.set_ylabel("rank(|Z|)")
    rho = float(np.corrcoef(r_z, r_abs)[0, 1])
    ax.set_title(f"Rank-rank scatter (Spearman rho = {rho:+.3f}): magnitude vs. sign coupling")
    ax.grid(True, alpha=0.25)
    _save(fig, "22_rank_rank.png")
    return rho


def fig_ccdf(z):
    z_std = (z - z.mean()) / z.std()
    absz = np.sort(np.abs(z_std))[::-1]
    n = absz.size
    ccdf = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(absz, ccdf, lw=1.3, label="empirical CCDF of |Z|")
    grid = np.logspace(-2, np.log10(absz[0]), 200)
    gauss = 2 * (1 - _Phi(grid))
    ax.loglog(grid, gauss, "r--", lw=1.3, label="Gaussian CCDF reference")
    ax.set_xlabel("|Z|")
    ax.set_ylabel("P(|Z| > x)")
    ax.set_title("Tail concentration (log-log): heavier tails => higher line at right")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, "23_ccdf_loglog.png")


def main():
    z = pd.read_csv(DATA).iloc[:, 0].to_numpy(dtype=np.float64)
    print(f"Pierce the VEIL  -  expanded EDA")
    print(f"  n        : {z.size}")
    print(f"  mean     : {z.mean():.6f}")
    print(f"  std      : {z.std():.6f}")
    print(f"  min/max  : {z.min():.4f} / {z.max():.4f}")
    results = {
        "n": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "min": float(z.min()),
        "max": float(z.max()),
        "skew": float(((z - z.mean()) ** 3).mean() / z.std() ** 3),
        "kurtosis_excess": float(((z - z.mean()) ** 4).mean() / z.std() ** 4 - 3.0),
    }
    fig_kde(z)
    fig_ecdf(z)
    fig_qq_normal(z)
    fig_qq_t(z, df=5)
    results["hill_xi_at_median_k"] = fig_hill(z)
    results.update(fig_acf(z, max_lag=50))
    fig_welch_psd(z)
    results["lag1_pearson_r"] = fig_lag1(z)
    fig_z2(z)
    fig_absz(z)
    results["gmm_fit"] = fig_gmm(z)
    results.update(fig_signs(z))
    results["rank_rank_spearman"] = fig_rank_rank(z)
    fig_ccdf(z)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"  wrote 14 figures to {FIG}")
    print(f"  summary to {OUT_JSON}")
    print(f"  skew       : {results['skew']:+.4f}")
    print(f"  excess kur : {results['kurtosis_excess']:+.4f}")
    print(f"  ACF(1)     : {results['acf_lag1']:+.4f}")
    print(f"  hill xi    : {results['hill_xi_at_median_k']:+.4f}")


if __name__ == "__main__":
    main()
