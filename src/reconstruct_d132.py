"""
Pierce the VEIL --- backup reconstruction algorithm with D_hat = 132.

Author: Lady Faye  (Kaggle: ladyfaye)
License: MIT

This is the hedging submission. It uses the same 6-channel leak stack as
the primary D=16 submission --- linear, magnitude, sign, quadratic,
rank-Gaussian quantile, GMM mixture-component --- placed in columns 0..5
of a D=132 reconstruction with the remaining 126 columns set to the
zero-baseline (per-feature mean of a standardized X).

Rationale for D=132
-------------------
Per arXiv:2603.15842 (Samuelson), Section 10.1 documents the reference
VEIL deployment as a 132-dimensional real-estate input compressed to a
16-dimensional latent for house-price regression. If the competition
uses this exact reference deployment (rather than a custom synthetic
encoder), then D=132 is the correct dimensionality.

This backup covers that hypothesis. The primary kernel covers D=16 (which
is the empirical best W1 signature match against the public Z marginal).

SRMSE drift analysis (D=132, alpha = 0.05)
------------------------------------------
With 6 active columns and 126 zero columns:
  worst case (all six r = -1):  SRMSE = sqrt((6*(1+alpha)^2 + 126) / 132)
                              = sqrt((6*1.1025 + 126)/132) = 1.00141
  neutral (r = 0):              SRMSE = sqrt((6*(1+alpha^2) + 126)/132)
                              = sqrt(1 + 6*alpha^2/132)    = 1.00057
  best case (all six r = +1):   SRMSE = sqrt((6*(1-alpha)^2 + 126)/132)
                              = sqrt((6*0.9025 + 126)/132) = 0.99772

So worst-case drift from the all-zeros baseline is +0.14% and best-case
is -0.23%. Tighter than the primary D=16 kernel because the variance
penalty is diluted across more inert columns.
"""

from __future__ import annotations

import numpy as np


D_HAT = 132


_REFERENCE_MEAN = 0.10263751797120119
_REFERENCE_STD = 2.15012248773876
_REFERENCE_ABS_Z_MEAN = 0.8048
_REFERENCE_SIGN_TILT = 0.0
_REFERENCE_GMM_W1 = 0.39898259754442317
_REFERENCE_GMM_MU1 = 1.466420244941862
_REFERENCE_GMM_VAR1 = 5.034097674579278
_REFERENCE_GMM_W2 = 0.6010174024555769
_REFERENCE_GMM_MU2 = -0.8027032802649854
_REFERENCE_GMM_VAR2 = 2.295810732750933

_ALPHA = 0.05
_QUAD_NORM = float(np.sqrt(2.0))


def _fit_two_component_gmm_1d(z, max_iter=50, tol=1e-6):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size < 8:
        return (
            _REFERENCE_GMM_W1, _REFERENCE_GMM_MU1, _REFERENCE_GMM_VAR1,
            _REFERENCE_GMM_W2, _REFERENCE_GMM_MU2, _REFERENCE_GMM_VAR2,
        )
    q1, q2 = np.quantile(z, [0.25, 0.75])
    mu1, mu2 = float(q1), float(q2)
    var = float(z.var()) + 1e-9
    var1, var2 = var, var
    w1 = 0.5
    log2pi = np.log(2.0 * np.pi)
    last_ll = -np.inf
    for _ in range(max_iter):
        log_p1 = -0.5 * (log2pi + np.log(var1) + (z - mu1) ** 2 / var1) + np.log(max(w1, 1e-12))
        log_p2 = -0.5 * (log2pi + np.log(var2) + (z - mu2) ** 2 / var2) + np.log(max(1.0 - w1, 1e-12))
        m = np.maximum(log_p1, log_p2)
        log_total = m + np.log(np.exp(log_p1 - m) + np.exp(log_p2 - m))
        gamma1 = np.exp(log_p1 - log_total)
        gamma2 = 1.0 - gamma1
        n1 = gamma1.sum() + 1e-12
        n2 = gamma2.sum() + 1e-12
        mu1 = float((gamma1 * z).sum() / n1)
        mu2 = float((gamma2 * z).sum() / n2)
        var1 = float((gamma1 * (z - mu1) ** 2).sum() / n1) + 1e-9
        var2 = float((gamma2 * (z - mu2) ** 2).sum() / n2) + 1e-9
        w1 = float(n1 / (n1 + n2))
        ll = float(log_total.sum())
        if abs(ll - last_ll) < tol * (1.0 + abs(last_ll)):
            break
        last_ll = ll
    return w1, mu1, var1, 1.0 - w1, mu2, var2


def _gmm_posterior(z, w1, mu1, var1, w2, mu2, var2):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    log2pi = np.log(2.0 * np.pi)
    log_p1 = -0.5 * (log2pi + np.log(var1) + (z - mu1) ** 2 / var1) + np.log(max(w1, 1e-12))
    log_p2 = -0.5 * (log2pi + np.log(var2) + (z - mu2) ** 2 / var2) + np.log(max(w2, 1e-12))
    m = np.maximum(log_p1, log_p2)
    return np.exp(log_p1 - m) / (np.exp(log_p1 - m) + np.exp(log_p2 - m))


def _erfinv(x):
    x = np.clip(x, -1.0 + 1e-12, 1.0 - 1e-12)
    a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    b = (-2.118377725, 1.442710462, -0.329097515, 0.012229801)
    c = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    d = (3.543889200, 1.637067800)
    abs_x = np.abs(x)
    out = np.empty_like(x)
    mask = abs_x <= 0.7
    y = x[mask] * x[mask]
    num = ((a[3] * y + a[2]) * y + a[1]) * y + a[0]
    den = (((b[3] * y + b[2]) * y + b[1]) * y + b[0]) * y + 1.0
    out[mask] = x[mask] * num / den
    y2 = np.sqrt(-np.log((1.0 - abs_x[~mask]) / 2.0))
    num2 = ((c[3] * y2 + c[2]) * y2 + c[1]) * y2 + c[0]
    den2 = (d[1] * y2 + d[0]) * y2 + 1.0
    out[~mask] = np.sign(x[~mask]) * num2 / den2
    return out


def _rank_quantile(z_hid, z_pub):
    z_hid = np.asarray(z_hid, dtype=np.float64).reshape(-1)
    z_pub = np.asarray(z_pub, dtype=np.float64).reshape(-1)
    if z_pub.size < 2:
        z_pub = z_hid
    sorted_pub = np.sort(z_pub)
    ranks = np.searchsorted(sorted_pub, z_hid, side="right")
    n = sorted_pub.size
    u = (ranks + 0.5) / (n + 1.0)
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    sqrt2 = np.sqrt(2.0)
    return sqrt2 * _erfinv(2.0 * u - 1.0)


def reconstruct(public_latents, hidden_latents, metadata=None):
    """
    Backup submission with D_hat = 132 (paper sec. 10.1 reference deployment).

    Six bounded leak channels are placed in cols 0..5 with alpha = 0.05;
    cols 6..131 are zero (per-feature mean baseline of standardized X).
    """
    z_pub = np.asarray(public_latents, dtype=np.float64).reshape(-1)
    z_hid = np.asarray(hidden_latents, dtype=np.float64).reshape(-1)
    n_hid = int(z_hid.shape[0])

    if z_pub.size > 0:
        mu = float(z_pub.mean())
        sigma = float(z_pub.std())
    else:
        mu, sigma = _REFERENCE_MEAN, _REFERENCE_STD
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = _REFERENCE_STD if _REFERENCE_STD > 0 else 1.0

    z_pub_std = (z_pub - mu) / sigma if z_pub.size > 0 else np.array([0.0])
    z_hid_std = (z_hid - mu) / sigma

    if z_pub.size > 0:
        abs_mean = float(np.abs(z_pub_std).mean())
        sign_tilt = float(np.median(z_pub_std))
    else:
        abs_mean = _REFERENCE_ABS_Z_MEAN
        sign_tilt = _REFERENCE_SIGN_TILT
    if not np.isfinite(abs_mean):
        abs_mean = _REFERENCE_ABS_Z_MEAN

    if z_pub.size >= 8:
        w1, m1, v1, w2, m2, v2 = _fit_two_component_gmm_1d(z_pub_std)
        gmm_mean_post = float(
            _gmm_posterior(z_pub_std, w1, m1, v1, w2, m2, v2).mean()
        )
    else:
        w1, m1, v1, w2, m2, v2 = (
            _REFERENCE_GMM_W1,
            _REFERENCE_GMM_MU1 / _REFERENCE_STD,
            _REFERENCE_GMM_VAR1 / (_REFERENCE_STD ** 2),
            _REFERENCE_GMM_W2,
            _REFERENCE_GMM_MU2 / _REFERENCE_STD,
            _REFERENCE_GMM_VAR2 / (_REFERENCE_STD ** 2),
        )
        gmm_mean_post = 0.5

    ch0 = z_hid_std
    ch1 = np.abs(z_hid_std) - abs_mean
    ch2 = np.sign(z_hid_std - sign_tilt) - (
        float(np.sign(z_pub_std - sign_tilt).mean()) if z_pub.size > 0 else 0.0
    )
    ch3 = (z_hid_std ** 2 - 1.0) / _QUAD_NORM
    ch4 = _rank_quantile(z_hid, z_pub if z_pub.size > 0 else z_hid)
    ch5 = 2.0 * (
        _gmm_posterior(z_hid_std, w1, m1, v1, w2, m2, v2) - gmm_mean_post
    )

    X_hat = np.zeros((n_hid, D_HAT), dtype=np.float64)
    X_hat[:, 0] = _ALPHA * ch0
    X_hat[:, 1] = _ALPHA * ch1
    X_hat[:, 2] = _ALPHA * ch2
    X_hat[:, 3] = _ALPHA * ch3
    X_hat[:, 4] = _ALPHA * ch4
    X_hat[:, 5] = _ALPHA * ch5

    X_hat = np.where(np.isfinite(X_hat), X_hat, 0.0)

    assert X_hat.shape == (n_hid, D_HAT)
    return X_hat


__all__ = ["reconstruct", "D_HAT"]
