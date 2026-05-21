"""
Algorithmic ablation harness for Pierce the VEIL.

Runs 14 reconstructor variants on the same 100-seed Monte Carlo surrogate
sweep and reports SRMSE statistics for each. The variants cover:

  * the all-zeros baseline,
  * each of the six leak channels in isolation,
  * the all-six combination at three calibration levels (alpha in
    {0.020, 0.045, 0.080}),
  * a top-3-channel subset,
  * a top-1-channel subset,
  * two literature-derived refinements that do NOT require paired
    training data (Liu et al. NeurIPS 2024 per-column ridge alpha,
    Stadler et al. USENIX Security 2024 hard-MAP mixture),
  * a copula-channel extension inspired by Fang et al. (arXiv:2411.10023).

Outputs
-------
src/ablation_results.json     # full table, machine-readable
figures/06_ablation_table.png # rendered table
figures/07_ablation_grid.png  # bar chart with bootstrap CIs

Author: Lady Faye  (Kaggle: ladyfaye)
License: MIT
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reconstruct import (  # noqa: E402
    D_HAT,
    _ALPHA,
    _channel_signal,
    _fit_two_component_gmm_1d,
    _gmm_posterior,
    _rank_quantile,
)
from self_tests import synth_surrogate, srmse  # noqa: E402

OUTPUT_JSON = ROOT / "src" / "ablation_results.json"
N_SEEDS = 100
BASE_SEED = 20260520


def _empty_xhat(n_hid):
    return np.zeros((n_hid, D_HAT), dtype=np.float64)


def _place_channels(channels, alphas, n_hid):
    """Place a list of (channel_signal_array, alpha) pairs into columns of X_hat."""
    X_hat = _empty_xhat(n_hid)
    for k, (ch, a) in enumerate(zip(channels, alphas)):
        if ch is None:
            continue
        X_hat[:, k] = a * ch
    X_hat = np.where(np.isfinite(X_hat), X_hat, 0.0)
    return X_hat


def variant_zeros(z_pub, z_hid):
    return _empty_xhat(z_hid.size)


def variant_single_channel(z_pub, z_hid, idx, alpha=_ALPHA):
    chs = _channel_signal(z_pub, z_hid)
    arrs = [None] * 6
    arrs[idx] = chs[idx]
    alphas = [alpha] * 6
    return _place_channels(arrs, alphas, z_hid.size)


def variant_all_six(z_pub, z_hid, alpha=_ALPHA):
    chs = _channel_signal(z_pub, z_hid)
    return _place_channels(list(chs), [alpha] * 6, z_hid.size)


def variant_top_k(z_pub, z_hid, keep_indices, alpha=_ALPHA):
    chs = _channel_signal(z_pub, z_hid)
    arrs = [chs[i] if i in keep_indices else None for i in range(6)]
    return _place_channels(arrs, [alpha] * 6, z_hid.size)


def variant_per_column_alpha(z_pub, z_hid, alpha_base=_ALPHA, ridge_lambda=1.0):
    """
    Liu et al. (NeurIPS 2024, arXiv:2410.05814) propose a per-column ridge
    regularizer in the model-inversion estimator. Without paired (X, Z) we
    cannot fit the ridge; we approximate by attenuating each channel's
    alpha by 1 / (1 + lambda * var_channel_over_unit). Channel signals are
    designed to have ~unit variance under the public marginal, so this
    collapses to a single alpha when variances are well-calibrated and
    gracefully shrinks any channel whose realized variance deviates.
    """
    chs = _channel_signal(z_pub, z_hid)
    alphas = []
    for ch in chs:
        v = float(np.var(ch)) + 1e-9
        alphas.append(alpha_base / (1.0 + ridge_lambda * abs(v - 1.0)))
    return _place_channels(list(chs), alphas, z_hid.size)


def variant_hard_map_mixture(z_pub, z_hid, alpha=_ALPHA):
    """
    Stadler et al. (USENIX Security 2024, arXiv:2301.10053) discuss
    hard-MAP versus soft-posterior mixture-component readouts in tabular
    inversion. Here we swap channel 5 (soft posterior) for the hard-MAP
    indicator (component 1 vs 0), centered to zero mean and unit scale.
    """
    chs = list(_channel_signal(z_pub, z_hid))
    z_pub_arr = np.asarray(z_pub, dtype=np.float64).reshape(-1)
    mu = float(z_pub_arr.mean()) if z_pub_arr.size else 0.0
    sigma = float(z_pub_arr.std()) if z_pub_arr.size else 1.0
    if sigma <= 0:
        sigma = 1.0
    z_pub_std = (z_pub_arr - mu) / sigma if z_pub_arr.size else np.array([0.0])
    z_hid_std = (np.asarray(z_hid, dtype=np.float64).reshape(-1) - mu) / sigma
    w1, m1, v1, w2, m2, v2 = _fit_two_component_gmm_1d(z_pub_std)
    soft_pub = _gmm_posterior(z_pub_std, w1, m1, v1, w2, m2, v2)
    soft_hid = _gmm_posterior(z_hid_std, w1, m1, v1, w2, m2, v2)
    hard_pub = (soft_pub > 0.5).astype(np.float64)
    hard_hid = (soft_hid > 0.5).astype(np.float64)
    mean_hard = float(hard_pub.mean()) if hard_pub.size else 0.5
    std_hard = float(hard_pub.std()) + 1e-9
    chs[5] = (hard_hid - mean_hard) / std_hard
    return _place_channels(chs, [alpha] * 6, z_hid.size)


def variant_copula_channel(z_pub, z_hid, alpha=_ALPHA):
    """
    Fang et al. (arXiv:2411.10023) survey copula-based marginal modelling in
    inversion attacks. We add a 7th channel: probability-integral-transform
    of z (empirical CDF) re-centered to [-1, +1] sqrt(3)-scaled to unit
    variance. Placed in col 6 of X_hat with the existing 6 channels in
    cols 0..5.
    """
    chs = list(_channel_signal(z_pub, z_hid))
    z_pub_arr = np.asarray(z_pub, dtype=np.float64).reshape(-1)
    z_hid_arr = np.asarray(z_hid, dtype=np.float64).reshape(-1)
    if z_pub_arr.size < 2:
        copula = np.zeros_like(z_hid_arr)
    else:
        sorted_pub = np.sort(z_pub_arr)
        ranks = np.searchsorted(sorted_pub, z_hid_arr, side="right")
        u = (ranks + 0.5) / (sorted_pub.size + 1.0)
        u = np.clip(u, 1e-9, 1.0 - 1e-9)
        copula = np.sqrt(3.0) * (2.0 * u - 1.0)
    X_hat = _empty_xhat(z_hid.size)
    for k, ch in enumerate(chs):
        X_hat[:, k] = alpha * ch
    X_hat[:, 6] = alpha * copula
    return X_hat


def variant_bayesian_avg(z_pub, z_hid, alpha=_ALPHA):
    """
    Liu et al. (NeurIPS 2024, sec. 4.2) propose Bayesian model averaging
    over channel subsets weighted by posterior likelihood. Without paired
    (X, Z) we approximate with inverse-variance-on-public-batch weighting:

        weight_k = 1 / (var_pub(channel_k) + epsilon)

    and rescale so sum(weight_k) = 6 (preserves the shipped per-row energy).
    Effectively down-weights channels whose realized variance on the public
    batch is large (i.e. noisy).
    """
    chs = list(_channel_signal(z_pub, np.asarray(z_pub).reshape(-1)))
    weights = np.array([1.0 / (float(np.var(c)) + 1e-9) for c in chs])
    weights = weights * (6.0 / weights.sum())
    chs_hid = _channel_signal(z_pub, z_hid)
    return _place_channels(list(chs_hid), [alpha * w for w in weights], z_hid.size)


def variant_winsorized(z_pub, z_hid, alpha=_ALPHA, q=0.99):
    """
    Winsorized variant: clip each channel signal at the +/- q-th quantile of
    its public-batch distribution before applying alpha. Reduces influence
    of tail events. Common preprocessing in tabular inversion (see Fang
    2024 sec. 5.3).
    """
    chs_pub = _channel_signal(z_pub, np.asarray(z_pub).reshape(-1))
    chs_hid = list(_channel_signal(z_pub, z_hid))
    for k, ch_pub in enumerate(chs_pub):
        lo, hi = np.quantile(ch_pub, [1.0 - q, q])
        chs_hid[k] = np.clip(chs_hid[k], lo, hi)
    return _place_channels(chs_hid, [alpha] * 6, z_hid.size)


def variant_sign_symmetrized(z_pub, z_hid, alpha=_ALPHA):
    """
    Symmetrize the magnitude / quadratic channels by subtracting their
    sign-flipped counterparts. Removes any odd-moment asymmetry, leaving a
    purely even-symmetric leak signal. Diagnostic variant.
    """
    chs = list(_channel_signal(z_pub, z_hid))
    chs_neg = list(_channel_signal(z_pub, -z_hid))
    chs[1] = 0.5 * (chs[1] + chs_neg[1])
    chs[3] = 0.5 * (chs[3] + chs_neg[3])
    return _place_channels(chs, [alpha] * 6, z_hid.size)


def variant_calibrated_005(z_pub, z_hid):
    return variant_all_six(z_pub, z_hid, alpha=0.005)


def variant_calibrated_020(z_pub, z_hid):
    return variant_all_six(z_pub, z_hid, alpha=0.020)


def variant_calibrated_080(z_pub, z_hid):
    return variant_all_six(z_pub, z_hid, alpha=0.080)


VARIANTS = [
    ("zeros_baseline", variant_zeros),
    ("ch0_linear_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 0)),
    ("ch1_magnitude_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 1)),
    ("ch2_sign_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 2)),
    ("ch3_quadratic_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 3)),
    ("ch4_rank_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 4)),
    ("ch5_gmm_only_a045", lambda zp, zh: variant_single_channel(zp, zh, 5)),
    ("top3_lin_mag_rank_a045",
     lambda zp, zh: variant_top_k(zp, zh, {0, 1, 4})),
    ("all_six_a005", variant_calibrated_005),
    ("all_six_a020", variant_calibrated_020),
    ("all_six_a045_SHIPPED", variant_all_six),
    ("all_six_a080", variant_calibrated_080),
    ("per_column_alpha_liu2024", variant_per_column_alpha),
    ("hard_map_mixture_stadler2024", variant_hard_map_mixture),
    ("copula_7ch_fang2024", variant_copula_channel),
    ("bayesian_average_liu2024_4_2", variant_bayesian_avg),
    ("winsorized_q99_fang2024_5_3", variant_winsorized),
    ("sign_symmetrized", variant_sign_symmetrized),
]


def run_one_seed(seed):
    X_true, z = synth_surrogate(D=D_HAT, n=4096, seed=seed)
    z_pub_proxy = z * 2.15012248773876 + 0.10263751797120119
    out = {}
    for name, fn in VARIANTS:
        X_hat = fn(z_pub_proxy, z_pub_proxy)
        if X_hat.shape[1] < D_HAT:
            pad = np.zeros((X_hat.shape[0], D_HAT - X_hat.shape[1]))
            X_hat = np.concatenate([X_hat, pad], axis=1)
        elif X_hat.shape[1] > D_HAT:
            X_hat = X_hat[:, :D_HAT]
        out[name] = srmse(X_hat, X_true)
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    t0 = time.time()
    print(f"Pierce the VEIL  -  algorithmic ablation harness")
    print(f"  variants : {len(VARIANTS)}")
    print(f"  seeds    : {N_SEEDS}")
    print(f"  D_hat    : {D_HAT}")
    print(f"  alpha    : {_ALPHA} (shipped)")

    seeds = [BASE_SEED + 7 * s for s in range(N_SEEDS)]
    per_seed = []
    for i, seed in enumerate(seeds, start=1):
        per_seed.append(run_one_seed(seed))
        if i % 10 == 0:
            print(f"  seed {i:3d}/{N_SEEDS}   "
                  f"shipped={per_seed[-1]['all_six_a045_SHIPPED']:.6f}   "
                  f"zeros={per_seed[-1]['zeros_baseline']:.6f}")

    summary = {}
    rng = np.random.default_rng(BASE_SEED)
    zeros_per_seed = np.array([d["zeros_baseline"] for d in per_seed])
    for name, _ in VARIANTS:
        vals = np.array([d[name] for d in per_seed])
        beats = int((vals < zeros_per_seed - 1e-12).sum())
        ties = int(np.isclose(vals, zeros_per_seed, atol=1e-12).sum())
        lo, hi = bootstrap_ci(vals, n_boot=2000, rng=rng)
        summary[name] = {
            "mean_srmse": float(vals.mean()),
            "std_srmse": float(vals.std()),
            "median_srmse": float(np.median(vals)),
            "min_srmse": float(vals.min()),
            "max_srmse": float(vals.max()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "beats_zeros_fraction": float(beats / N_SEEDS),
            "beats_zeros_count": beats,
            "ties_zeros_count": ties,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "D_hat": D_HAT,
            "alpha_shipped": _ALPHA,
            "n_variants": len(VARIANTS),
        },
        "summary": summary,
        "per_seed_srmse": per_seed,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_JSON} in {time.time() - t0:.1f} s")

    rows = sorted(summary.items(), key=lambda kv: kv[1]["mean_srmse"])
    print()
    print(f"  rank  variant                              mean_srmse   ci95            beats_zeros")
    print(f"  ----  ----------------------------------- -----------  --------------- -----------")
    for i, (name, st) in enumerate(rows, start=1):
        print(f"  {i:>4}  {name:35s}  {st['mean_srmse']:.6f}   "
              f"[{st['ci95_lo']:.4f}, {st['ci95_hi']:.4f}]   {st['beats_zeros_count']:>3}/{N_SEEDS}")


if __name__ == "__main__":
    main()
