"""
Local emulation of the 8-stage evaluation pipeline so we can certify the
reconstruct() function before submitting.

Stage 1 - Execution & Validity:        runs, finite output, no NaN/Inf
Stage 2 - Structural Validation:       correct (N_hid, D_hat) shape
Stage 3 - Record Alignment:            f acts row-wise; row i depends only on z_i
Stage 4 - Reconstruction Accuracy:     simulated SRMSE on surrogate
Stage 5 - Baseline Separation:         beats random and constant baselines on synth
Stage 6 - Latent Dependence:           f(PZ) == P f(Z)  (permutation equivariance)
                                       also: not constant in Z
Stage 7 - Generalization:              same SRMSE on a second held-out surrogate
Stage 8 - Code Review:                 deterministic, internet-free, library audit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconstruct import reconstruct, D_HAT  # noqa: E402

DATA = ROOT / "data" / "intercepted_data.csv"
SEED = 12345


def synth_surrogate(D=16, weights=(0.8, 0.2), sep=0.5, seed=SEED, n=4096):
    X, y = make_classification(
        n_samples=n,
        n_features=D,
        n_informative=int(D * 0.8),
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=sep,
        weights=list(weights),
        flip_y=0.01,
        random_state=seed,
    )
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    z_raw = clf.decision_function(X)
    z_std = (z_raw - z_raw.mean()) / z_raw.std()
    return X_std, z_std


def srmse(X_hat, X_true):
    e = X_hat - X_true
    return float(np.sqrt((e ** 2).mean()))


def per_feature_srmse(X_hat, X_true):
    e = X_hat - X_true
    return np.sqrt((e ** 2).mean(axis=0))


def stage1_execution(z_pub, z_hid):
    t0 = time.time()
    X = reconstruct(z_pub, z_hid)
    dt = time.time() - t0
    return {
        "ran_successfully": True,
        "elapsed_seconds": dt,
        "all_finite": bool(np.isfinite(X).all()),
        "no_nan": bool(not np.isnan(X).any()),
        "no_inf": bool(not np.isinf(X).any()),
        "shape": tuple(int(x) for x in X.shape),
    }


def stage2_structural(z_pub, z_hid, expected_D):
    X = reconstruct(z_pub, z_hid)
    return {
        "n_rows_match": bool(X.shape[0] == z_hid.shape[0]),
        "D_match_expected": bool(X.shape[1] == expected_D),
        "shape": (int(X.shape[0]), int(X.shape[1])),
        "expected_shape": (int(z_hid.shape[0]), int(expected_D)),
    }


def stage3_alignment(z_pub, z_hid):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(z_hid.shape[0])
    X1 = reconstruct(z_pub, z_hid)
    X2 = reconstruct(z_pub, z_hid[perm])
    aligned_ok = bool(np.allclose(X2, X1[perm], atol=1e-12))
    return {"row_aligned_under_permutation": aligned_ok}


def stage4_accuracy(z_pub, n=4096):
    X_true, z = synth_surrogate(D=D_HAT, n=n, seed=SEED + 1)
    X_hat = reconstruct(z_pub, z * z_pub.std() + z_pub.mean())
    s = srmse(X_hat, X_true)
    per_feat = per_feature_srmse(X_hat, X_true)
    s_zeros = srmse(np.zeros_like(X_true), X_true)
    rng = np.random.default_rng(SEED + 2)
    s_random = srmse(rng.standard_normal(X_true.shape), X_true)
    s_const = srmse(np.full_like(X_true, X_true.mean()), X_true)
    return {
        "ours_srmse": s,
        "zeros_baseline_srmse": s_zeros,
        "random_baseline_srmse": s_random,
        "constant_baseline_srmse": s_const,
        "beats_zeros": bool(s < s_zeros + 1e-6),
        "beats_random": bool(s < s_random),
        "beats_constant": bool(s < s_const),
        "per_feature_srmse": [float(x) for x in per_feat],
    }


def stage5_baseline_sep(z_pub):
    X_true, z = synth_surrogate(D=D_HAT, n=2048, seed=SEED + 3)
    X_hat = reconstruct(z_pub, z * z_pub.std() + z_pub.mean())
    s = srmse(X_hat, X_true)
    rng = np.random.default_rng(SEED + 4)
    margins = []
    for _ in range(20):
        rb = srmse(rng.standard_normal(X_true.shape), X_true)
        margins.append(s < rb)
    pvals = []
    for _ in range(10):
        perm = rng.permutation(X_true.shape[0])
        X_hat_perm = reconstruct(z_pub, (z[perm]) * z_pub.std() + z_pub.mean())
        s_perm = srmse(X_hat_perm, X_true)
        pvals.append(s < s_perm)
    return {
        "srmse": s,
        "frac_random_baselines_beaten": float(np.mean(margins)),
        "frac_permuted_z_baselines_beaten": float(np.mean(pvals)),
    }


def stage6_latent_dependence(z_pub, z_hid):
    X_orig = reconstruct(z_pub, z_hid)
    rng = np.random.default_rng(SEED + 5)
    perm = rng.permutation(z_hid.shape[0])
    X_perm = reconstruct(z_pub, z_hid[perm])
    equivariant = bool(np.allclose(X_perm, X_orig[perm], atol=1e-12))
    eps = 1e-2
    z_perturbed = z_hid + eps
    X_eps = reconstruct(z_pub, z_perturbed)
    dydx = float(np.linalg.norm(X_eps - X_orig) / eps)
    col_std = X_orig.std(axis=0)
    return {
        "permutation_equivariant": equivariant,
        "norm_dXdZ_over_eps": dydx,
        "col0_std": float(col_std[0]),
        "all_cols_constant": bool(np.allclose(col_std, 0.0)),
        "n_nonzero_cols": int((col_std > 1e-12).sum()),
    }


def stage7_generalization(z_pub):
    results = []
    for seed in [SEED + 10, SEED + 11, SEED + 12, SEED + 13, SEED + 14]:
        X_true, z = synth_surrogate(D=D_HAT, n=2048, seed=seed)
        X_hat = reconstruct(z_pub, z * z_pub.std() + z_pub.mean())
        results.append({"seed": seed, "srmse": srmse(X_hat, X_true)})
    srmses = [r["srmse"] for r in results]
    return {
        "per_run": results,
        "mean": float(np.mean(srmses)),
        "std": float(np.std(srmses)),
        "max_minus_min": float(max(srmses) - min(srmses)),
    }


def stage8_code_review():
    import reconstruct as r_mod

    src_path = Path(r_mod.__file__)
    src = src_path.read_text(encoding="utf-8")
    banned_keywords = [
        "requests.",
        "urllib.",
        "http.",
        "socket.",
        "subprocess.",
    ]
    hits = {kw: (kw in src) for kw in banned_keywords}
    import ast

    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return {
        "imports": imports,
        "banned_keyword_hits": hits,
        "internet_free": not any(hits.values()),
    }


def determinism_test(z_pub, z_hid, n_runs=5):
    outputs = [reconstruct(z_pub, z_hid) for _ in range(n_runs)]
    deltas = [float(np.abs(outputs[i] - outputs[0]).max()) for i in range(n_runs)]
    return {"max_delta_over_runs": max(deltas), "is_bit_identical": max(deltas) == 0.0}


def _pp(label, d):
    print(f"\n{label}")
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 8:
            v = f"{v[:4]} ... ({len(v)} items)"
        print(f"  {k:35s}: {v}")


def main():
    z_pub = pd.read_csv(DATA).iloc[:, 0].to_numpy(dtype=np.float64)
    z_hid = z_pub.copy()

    print("=" * 78)
    print("PIERCE THE VEIL - LOCAL SELF-TEST HARNESS")
    print("=" * 78)
    print(f"  Z public shape: {z_pub.shape}, D_HAT = {D_HAT}")

    _pp("Stage 1 - Execution & Validity:", stage1_execution(z_pub, z_hid))
    _pp("Stage 2 - Structural Validation:", stage2_structural(z_pub, z_hid, D_HAT))
    _pp("Stage 3 - Record Alignment:", stage3_alignment(z_pub, z_hid))
    _pp("Stage 4 - Reconstruction Accuracy (synthetic surrogate):",
        stage4_accuracy(z_pub))
    _pp("Stage 5 - Baseline Separation:", stage5_baseline_sep(z_pub))
    _pp("Stage 6 - Latent Dependence:", stage6_latent_dependence(z_pub, z_hid))
    _pp("Stage 7 - Generalization:", stage7_generalization(z_pub))
    _pp("Stage 8 - Code Review (static):", stage8_code_review())
    _pp("Determinism (bit-identical across 5 runs):",
        determinism_test(z_pub, z_hid))

    print("\n" + "=" * 78)
    print("All stages run. Review numeric values above to confirm pass.")
    print("=" * 78)


if __name__ == "__main__":
    main()
