"""
Deep forensic EDA for Pierce the VEIL.
Goal: characterize Z and pin down what the encoder is, what domain the data
came from, and what dimensionality D is plausible.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "intercepted_data.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
OUT = ROOT / "src" / "eda_results.json"

SEED = 12345
np.random.seed(SEED)


def basic_stats(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64).ravel()
    return {
        "n": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std(ddof=0)),
        "min": float(z.min()),
        "p1": float(np.percentile(z, 1)),
        "p5": float(np.percentile(z, 5)),
        "p25": float(np.percentile(z, 25)),
        "median": float(np.median(z)),
        "p75": float(np.percentile(z, 75)),
        "p95": float(np.percentile(z, 95)),
        "p99": float(np.percentile(z, 99)),
        "max": float(z.max()),
        "skew": float(stats.skew(z)),
        "kurt_excess": float(stats.kurtosis(z, fisher=True)),
        "n_unique": int(np.unique(z).size),
        "n_duplicates": int(z.size - np.unique(z).size),
        "iqr": float(np.percentile(z, 75) - np.percentile(z, 25)),
        "mad": float(stats.median_abs_deviation(z)),
    }


def duplicate_codes(z: np.ndarray, min_count: int = 2) -> dict:
    z = np.asarray(z).ravel()
    vals, counts = np.unique(z, return_counts=True)
    mask = counts >= min_count
    dup_vals = vals[mask]
    dup_counts = counts[mask]
    order = np.argsort(-dup_counts)
    return {
        "n_distinct_duplicates": int(mask.sum()),
        "total_duplicated_rows": int(dup_counts.sum()),
        "top_duplicates": [
            {"value": float(v), "count": int(c)}
            for v, c in zip(dup_vals[order][:25], dup_counts[order][:25])
        ],
        "duplicate_value_range": [
            float(dup_vals.min()) if dup_vals.size else None,
            float(dup_vals.max()) if dup_vals.size else None,
        ],
    }


def gof_tests(z: np.ndarray) -> dict:
    """Goodness-of-fit tests against several reference distributions."""
    z = np.asarray(z).ravel()
    out = {}
    out["shapiro_subsample"] = {
        "stat": float(stats.shapiro(z[: min(5000, z.size)])[0]),
        "p": float(stats.shapiro(z[: min(5000, z.size)])[1]),
    }
    out["anderson_normal"] = {
        "stat": float(stats.anderson(z, dist="norm").statistic),
        "critical_values_5pct": float(
            stats.anderson(z, dist="norm").critical_values[2]
        ),
    }
    out["jarque_bera"] = {
        "stat": float(stats.jarque_bera(z)[0]),
        "p": float(stats.jarque_bera(z)[1]),
    }
    # Fit several distributions and rank by KS
    candidates = [
        ("norm", stats.norm),
        ("laplace", stats.laplace),
        ("logistic", stats.logistic),
        ("gennorm", stats.gennorm),
        ("skewnorm", stats.skewnorm),
        ("t", stats.t),
        ("hyperbolic", stats.hypsecant),
    ]
    fits = []
    for name, dist in candidates:
        try:
            params = dist.fit(z)
            ks_stat, ks_p = stats.kstest(z, lambda x: dist.cdf(x, *params))
            fits.append(
                {
                    "name": name,
                    "params": [float(p) for p in params],
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                }
            )
        except Exception as e:  # noqa: BLE001
            fits.append({"name": name, "error": str(e)})
    fits.sort(key=lambda d: d.get("ks_stat", 1.0))
    out["distribution_fits"] = fits
    return out


def polynomial_quantile_fit(z: np.ndarray, max_deg: int = 5) -> dict:
    """
    Sort Z, regress on N(0,1) quantiles. If z = polynomial(t) for t ~ N(0,1),
    we recover the polynomial. This characterizes the encoder if it is a
    smooth monotone calibration of a Gaussian latent.
    """
    z = np.sort(np.asarray(z).ravel())
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n
    t = stats.norm.ppf(p)
    out = {}
    for deg in range(1, max_deg + 1):
        coef = np.polyfit(t, z, deg=deg)
        zhat = np.polyval(coef, t)
        ss_res = float(np.sum((z - zhat) ** 2))
        ss_tot = float(np.sum((z - z.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot
        out[f"deg_{deg}"] = {
            "coef_high_to_low": [float(c) for c in coef],
            "r2": float(r2),
        }
    return out


def bit_entropy(z: np.ndarray) -> dict:
    """Per-bit entropy of the float64 mantissa to detect bit packing."""
    z = np.asarray(z, dtype=np.float64).ravel()
    bits = np.unpackbits(z.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    entropies = []
    for i in range(64):
        p = bits[:, i].mean()
        if p in (0.0, 1.0):
            entropies.append(0.0)
        else:
            entropies.append(float(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))
    return {
        "min_entropy": float(min(entropies)),
        "max_entropy": float(max(entropies)),
        "mean_entropy": float(np.mean(entropies)),
        "low_entropy_bit_count": int(sum(1 for e in entropies if e < 0.5)),
        "per_bit": entropies,
    }


def autocorrelation(z: np.ndarray, max_lag: int = 50) -> dict:
    z = np.asarray(z).ravel()
    z0 = z - z.mean()
    denom = float(np.dot(z0, z0))
    acf = [float(np.dot(z0[:-k], z0[k:]) / denom) for k in range(1, max_lag + 1)]
    return {
        "max_abs_acf_lag1_50": float(max(abs(a) for a in acf)),
        "acf_first_5": acf[:5],
        "n_significant_95pct": int(
            sum(1 for a in acf if abs(a) > 1.96 / np.sqrt(z.size))
        ),
    }


def reshape_rank_test(z: np.ndarray) -> dict:
    """
    If N=4096 is a flattened k x m matrix, then row-major reshape and SVD
    should show low rank. Test all factor pairs.
    """
    z = np.asarray(z).ravel()
    n = z.size
    factors = [k for k in range(2, int(np.sqrt(n)) + 1) if n % k == 0]
    out = []
    for k in factors:
        m = n // k
        M = z.reshape(k, m)
        s = np.linalg.svd(M, compute_uv=False)
        ratio = float(s[0] / s.sum()) if s.sum() else 0.0
        eff_rank = float(np.exp(-np.sum((s / s.sum()) * np.log((s / s.sum()) + 1e-12))))
        out.append(
            {
                "rows_cols": [int(k), int(m)],
                "top_sv_ratio": ratio,
                "effective_rank": eff_rank,
                "singular_value_decay_3": [float(x) for x in s[:3]],
            }
        )
    return out


def gaussian_mixture_bic(z: np.ndarray, k_max: int = 12) -> dict:
    """Bimodality / mixture structure detection."""
    from sklearn.mixture import GaussianMixture

    z = np.asarray(z).ravel().reshape(-1, 1)
    results = []
    for k in range(1, k_max + 1):
        gm = GaussianMixture(n_components=k, random_state=SEED, n_init=2, max_iter=200)
        gm.fit(z)
        results.append(
            {
                "k": k,
                "bic": float(gm.bic(z)),
                "aic": float(gm.aic(z)),
                "means": [float(m) for m in gm.means_.ravel()],
                "weights": [float(w) for w in gm.weights_],
                "covariances": [float(c) for c in gm.covariances_.ravel()],
            }
        )
    best_bic = min(results, key=lambda d: d["bic"])
    return {"by_k": results, "best_k_bic": best_bic["k"]}


def logit_probability_test(z: np.ndarray) -> dict:
    """
    Test if Z looks like a binary-classifier raw score (logit).
    If Z = log(p / (1-p)) for p in [0,1] of a true class label, then
    sigmoid(Z) should look like calibrated probabilities, and we expect
    a bimodal mixture (most negative for majority class, some positive for
    minority class).
    """
    z = np.asarray(z).ravel()
    p = 1.0 / (1.0 + np.exp(-z))
    return {
        "sigmoid_z_mean": float(p.mean()),
        "sigmoid_z_std": float(p.std()),
        "sigmoid_z_p10": float(np.percentile(p, 10)),
        "sigmoid_z_p50": float(np.percentile(p, 50)),
        "sigmoid_z_p90": float(np.percentile(p, 90)),
        "frac_pos_threshold_0": float((z > 0).mean()),
        "frac_pos_threshold_05": float((p > 0.5).mean()),
    }


def main():
    z = pd.read_csv(DATA).iloc[:, 0].to_numpy(dtype=np.float64)
    results = {
        "data_path": str(DATA),
        "basic_stats": basic_stats(z),
        "duplicate_codes": duplicate_codes(z),
        "goodness_of_fit": gof_tests(z),
        "polynomial_quantile_fit": polynomial_quantile_fit(z, max_deg=5),
        "bit_entropy": bit_entropy(z),
        "autocorrelation": autocorrelation(z, max_lag=50),
        "reshape_rank_test": reshape_rank_test(z),
        "gaussian_mixture": gaussian_mixture_bic(z, k_max=10),
        "logit_probability_test": logit_probability_test(z),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"EDA written to {OUT}")
    # Print key headlines
    bs = results["basic_stats"]
    print(
        f"n={bs['n']}, mean={bs['mean']:+.4f}, std={bs['std']:.4f}, "
        f"skew={bs['skew']:+.4f}, exkurt={bs['kurt_excess']:+.4f}, "
        f"range=[{bs['min']:.3f},{bs['max']:.3f}], "
        f"unique={bs['n_unique']}, dups={bs['n_duplicates']}"
    )
    print(f"Best distribution fit: {results['goodness_of_fit']['distribution_fits'][0]}")
    print(f"GMM best k (BIC): {results['gaussian_mixture']['best_k_bic']}")
    poly = results["polynomial_quantile_fit"]
    for deg, info in poly.items():
        print(f"  Polynomial-quantile {deg}: R^2={info['r2']:.6f}, coef={info['coef_high_to_low']}")
    lp = results["logit_probability_test"]
    print(
        f"sigmoid(Z): mean={lp['sigmoid_z_mean']:.4f}, std={lp['sigmoid_z_std']:.4f}, "
        f"frac(z>0)={lp['frac_pos_threshold_0']:.4f}"
    )


if __name__ == "__main__":
    main()
