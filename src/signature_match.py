"""
Signature matching (fast version): sweep over (D, model, class_balance, sep)
for binary classifier raw scores and a few regression configurations. Identify
which best matches Z's empirical distribution by 1-Wasserstein after rescaling.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "intercepted_data.csv"
OUT = ROOT / "src" / "signature_results.json"

SEED = 12345
np.random.seed(SEED)


def shape_distance(z_target: np.ndarray, z_candidate: np.ndarray) -> dict:
    zt = np.sort(np.asarray(z_target).ravel())
    zc = np.sort(np.asarray(z_candidate).ravel())
    if zt.size != zc.size:
        q = (np.arange(1, zt.size + 1) - 0.5) / zt.size
        zc = np.interp(q, (np.arange(1, zc.size + 1) - 0.5) / zc.size, zc)
    if zc.std() > 1e-8:
        zc_scaled = (zc - zc.mean()) / zc.std() * zt.std() + zt.mean()
    else:
        zc_scaled = zc - zc.mean() + zt.mean()
    w1 = float(stats.wasserstein_distance(zt, zc_scaled))
    ks_stat, ks_p = stats.ks_2samp(zt, zc_scaled)
    return {
        "w1_after_rescale": w1,
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "skew_diff": float(abs(stats.skew(zt) - stats.skew(zc_scaled))),
        "exkurt_diff": float(
            abs(stats.kurtosis(zt, fisher=True) - stats.kurtosis(zc_scaled, fisher=True))
        ),
        "candidate_mean": float(np.asarray(z_candidate).mean()),
        "candidate_std": float(np.asarray(z_candidate).std()),
        "candidate_skew": float(stats.skew(z_candidate)),
        "candidate_exkurt": float(stats.kurtosis(z_candidate, fisher=True)),
    }


def synth_clf_raw_score(n, D, n_inf, sep, weights, mdl, seed):
    X, y = make_classification(
        n_samples=n,
        n_features=D,
        n_informative=n_inf,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=sep,
        weights=weights,
        flip_y=0.01,
        random_state=seed,
    )
    if mdl == "logreg":
        m = LogisticRegression(max_iter=2000, random_state=seed)
        m.fit(X, y)
        s = m.decision_function(X)
    elif mdl == "gbc":
        m = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
        )
        m.fit(X, y)
        s = m.decision_function(X)
    else:
        raise ValueError(mdl)
    return s, y


def synth_reg(n, D, n_inf, noise, seed):
    X, y = make_regression(
        n_samples=n,
        n_features=D,
        n_informative=n_inf,
        noise=noise,
        random_state=seed,
    )
    m = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
    )
    m.fit(X, y)
    return m.predict(X)


def main():
    z = pd.read_csv(DATA).iloc[:, 0].to_numpy(dtype=np.float64)
    n = z.size
    print(f"Loaded Z with n={n}, mean={z.mean():+.4f}, std={z.std():.4f}, skew={stats.skew(z):+.4f}")

    Ds = [4, 8, 10, 12, 14, 16, 20, 23, 30]
    weights_list = [[0.5, 0.5], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
    seps = [0.5, 1.0]
    models = ["logreg", "gbc"]

    candidates = []
    print("Sweeping classifier configs...")
    for D in Ds:
        n_inf = max(3, int(D * 0.8))
        for w in weights_list:
            for sep in seps:
                for mdl in models:
                    try:
                        s, _ = synth_clf_raw_score(n, D, n_inf, sep, w, mdl, SEED)
                        d = shape_distance(z, s)
                        d.update(
                            {
                                "kind": "classifier",
                                "model": mdl,
                                "D": D,
                                "class_balance": w,
                                "class_sep": sep,
                            }
                        )
                        candidates.append(d)
                    except Exception:
                        pass

    print("Sweeping regression configs...")
    for D in Ds:
        n_inf = max(3, int(D * 0.8))
        for noise in [1.0, 5.0, 15.0, 30.0]:
            try:
                s = synth_reg(n, D, n_inf, noise, SEED)
                d = shape_distance(z, s)
                d.update(
                    {"kind": "regressor", "model": "gbr", "D": D, "noise": noise}
                )
                candidates.append(d)
            except Exception:
                pass

    candidates.sort(key=lambda c: c["w1_after_rescale"])

    out = {
        "n_target": int(n),
        "z_summary": {
            "mean": float(z.mean()),
            "std": float(z.std()),
            "skew": float(stats.skew(z)),
            "exkurt": float(stats.kurtosis(z, fisher=True)),
        },
        "top_30": candidates[:30],
        "top_3_per_D": {},
    }
    by_d: dict[int, list] = {}
    for c in candidates:
        by_d.setdefault(int(c["D"]), []).append(c)
    for D, lst in by_d.items():
        out["top_3_per_D"][str(D)] = lst[:3]

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print()
    print("Top 20 signature matches (lower w1 = better):")
    print("=" * 100)
    for c in candidates[:20]:
        kind = c["kind"]
        D = c["D"]
        model = c["model"]
        w1 = c["w1_after_rescale"]
        ks = c["ks_stat"]
        sk = c.get("candidate_skew", 0.0)
        ek = c.get("candidate_exkurt", 0.0)
        extra = (
            f"balance={c.get('class_balance')} sep={c.get('class_sep')}"
            if kind == "classifier"
            else f"noise={c.get('noise')}"
        )
        print(
            f"  {kind:<11s} {model:<6s} D={D:<3d} w1={w1:.4f} ks={ks:.4f} "
            f"skew={sk:+.3f} exkurt={ek:+.3f}  {extra}"
        )

    print()
    print("Best per D:")
    print("=" * 100)
    for D in sorted(by_d.keys()):
        best = by_d[D][0]
        print(
            f"  D={D:<3d} w1={best['w1_after_rescale']:.4f} "
            f"ks={best['ks_stat']:.4f} kind={best['kind']} model={best['model']}"
        )


if __name__ == "__main__":
    main()
