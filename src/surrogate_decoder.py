"""
Train a surrogate decoder on the best-matching synthetic dataset
(LogReg, D=16, 80/20 imbalance, class_sep=0.5) and quantify the
per-feature SRMSE improvement over the all-zeros baseline.

This gives us the realistic upper bound for our submission and lets us
pick which columns to populate with non-zero predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "intercepted_data.csv"
OUT = ROOT / "src" / "surrogate_results.json"

SEED = 12345
np.random.seed(SEED)

D = 16
N = 4096


def build_surrogate(D, weights, sep, seed):
    """Generate synthetic surrogate matching the top signature match."""
    X, y = make_classification(
        n_samples=N,
        n_features=D,
        n_informative=int(D * 0.8),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=sep,
        weights=weights,
        flip_y=0.01,
        random_state=seed,
    )
    # Standardize X column-wise (so per-feature "scale" is 1)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    # Encoder = standardized logreg score
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    z_raw = clf.decision_function(X)
    z_std = (z_raw - z_raw.mean()) / z_raw.std()
    return X_std, y, z_std, clf


def per_feature_srmse(X_std, z_std, alpha_grid):
    """
    For each feature j and each alpha, compute SRMSE of prediction
        x̂_j = alpha * z_std
    against true X_std[:, j]. s_j = std(X_std[:, j]) = 1.
    """
    D = X_std.shape[1]
    results = []
    for j in range(D):
        x_j = X_std[:, j]
        r_j = float(np.corrcoef(z_std, x_j)[0, 1])
        # Optimal alpha for this column
        alpha_star = r_j  # since both have unit variance
        # SRMSE at the OLS-optimal alpha
        srmse_star = float(np.sqrt(((alpha_star * z_std - x_j) ** 2).mean()))
        # SRMSE at fixed alphas
        srmse_grid = {
            f"alpha_{a:.2f}": float(np.sqrt(((a * z_std - x_j) ** 2).mean()))
            for a in alpha_grid
        }
        # Baseline (zeros)
        srmse_zero = float(np.sqrt((x_j ** 2).mean()))
        results.append(
            {
                "j": j,
                "r_j": r_j,
                "alpha_star": alpha_star,
                "srmse_at_alpha_star": srmse_star,
                "srmse_zero": srmse_zero,
                "srmse_at_fixed_alphas": srmse_grid,
            }
        )
    return results


def rank_match_per_feature(X_std, z_std):
    """
    Rank-match approach: sort surrogate by z_std, give the corresponding
    sorted X_std[:,j] values. This is the Wasserstein-optimal column-wise
    coupling under the constraint that the ranking of predictions matches
    the ranking of z.
    """
    D = X_std.shape[1]
    order = np.argsort(z_std)
    inv_order = np.argsort(order)
    results = []
    for j in range(D):
        x_j = X_std[:, j]
        x_sorted_by_z = x_j[order]
        # Predict by sorted z-rank: same rank in z gives same x
        # So for hidden Z, sort it and assign sorted X values
        x_hat = x_sorted_by_z[inv_order]  # back to original order
        srmse = float(np.sqrt(((x_hat - x_j) ** 2).mean()))
        results.append({"j": j, "srmse_rank_match": srmse})
    return results


def ridge_surrogate(X_std, z_std):
    """
    Train Ridge regression Z → X_j for each j on this surrogate. This is
    what a competitor with paired (z, X) data would do (the "stronger
    attacker" the paper §10.1 considers).
    """
    D = X_std.shape[1]
    Z_feat = np.column_stack(
        [z_std, z_std ** 2, z_std ** 3, np.tanh(z_std), np.tanh(z_std / 2),
         1 / (1 + np.exp(-z_std)), np.sign(z_std), np.abs(z_std)]
    )
    results = []
    for j in range(D):
        x_j = X_std[:, j]
        reg = Ridge(alpha=1.0)
        reg.fit(Z_feat, x_j)
        x_hat = reg.predict(Z_feat)
        srmse = float(np.sqrt(((x_hat - x_j) ** 2).mean()))
        r_predtrue = float(np.corrcoef(x_hat, x_j)[0, 1])
        results.append(
            {
                "j": j,
                "srmse_ridge": srmse,
                "ridge_coeffs": [float(c) for c in reg.coef_],
                "ridge_intercept": float(reg.intercept_),
                "corr_pred_true": r_predtrue,
            }
        )
    return results


def main():
    # The top match from signature_results.json: D=16, balance=[0.8, 0.2], sep=0.5, logreg
    X_std, y, z_std, clf = build_surrogate(D=D, weights=[0.8, 0.2], sep=0.5, seed=SEED)

    alpha_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    per_feat = per_feature_srmse(X_std, z_std, alpha_grid)
    rank_res = rank_match_per_feature(X_std, z_std)
    ridge_res = ridge_surrogate(X_std, z_std)

    # Combine results
    combined = []
    for f, r, rr in zip(per_feat, rank_res, ridge_res):
        combined.append(
            {
                "j": f["j"],
                "r_j": f["r_j"],
                "srmse_zero": f["srmse_zero"],
                "srmse_alpha_star": f["srmse_at_alpha_star"],
                "srmse_rank_match": r["srmse_rank_match"],
                "srmse_ridge_nonlinear": rr["srmse_ridge"],
                "corr_ridge_pred_true": rr["corr_pred_true"],
            }
        )

    # Aggregate
    total_srmse = {
        "zeros": float(np.sqrt(np.mean([c["srmse_zero"] ** 2 for c in combined]))),
        "alpha_star_oracle": float(
            np.sqrt(np.mean([c["srmse_alpha_star"] ** 2 for c in combined]))
        ),
        "rank_match": float(
            np.sqrt(np.mean([c["srmse_rank_match"] ** 2 for c in combined]))
        ),
        "ridge_nonlinear": float(
            np.sqrt(np.mean([c["srmse_ridge_nonlinear"] ** 2 for c in combined]))
        ),
    }

    # Theoretical floor: sqrt((D-1)/D)
    theoretical_floor = float(np.sqrt((D - 1) / D))

    print(f"\nSurrogate: D={D}, balance=[0.8, 0.2], sep=0.5, LogReg")
    print(f"N samples: {N}")
    print(f"Theoretical SRMSE floor (linear): sqrt((D-1)/D) = {theoretical_floor:.6f}")
    print(f"  (i.e., even perfect knowledge of weights -> SRMSE >= {theoretical_floor:.4f})")
    print()
    print("Per-feature analysis:")
    print("=" * 100)
    print(f"{'j':>3s} {'r_j':>8s} {'SRMSE0':>8s} {'SRMSEopt':>8s} {'SRMSErank':>10s} {'SRMSE_ridge':>12s}")
    for c in combined:
        print(
            f"{c['j']:>3d} {c['r_j']:>+8.4f} {c['srmse_zero']:>8.4f} "
            f"{c['srmse_alpha_star']:>8.4f} {c['srmse_rank_match']:>10.4f} "
            f"{c['srmse_ridge_nonlinear']:>12.4f}"
        )

    print()
    print("Overall SRMSE (aggregated across all D features):")
    for k, v in total_srmse.items():
        improvement = (1.0 - v) * 100
        print(f"  {k:<22s} = {v:.6f}   ({improvement:+.2f}% vs baseline=1.0)")

    out_data = {
        "D": D,
        "N": N,
        "theoretical_floor": theoretical_floor,
        "per_feature": combined,
        "total_srmse": total_srmse,
    }
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
