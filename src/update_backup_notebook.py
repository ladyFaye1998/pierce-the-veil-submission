"""Surgically update the backup notebook for the 6-channel D=132 reconstruct."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebook" / "pierce-the-veil-backup-d132.ipynb"
RECON_PATH = ROOT / "src" / "reconstruct_d132.py"


CELL0 = """\
# Pierce the VEIL --- Backup Submission (`D\u0302 = 132`)

**This is the second of our two Final Submissions.** It uses the same six-channel calibrated leak stack as our primary (`linear`, `magnitude`, `sign`, `quadratic`, `rank-Gaussian quantile`, `GMM mixture-component`) placed in columns 0..5 of a `D = 132` reconstruction, with the remaining 126 columns set to the zero-baseline (per-feature mean of a standardised `X`).

**Why hedge?**  Per the Evaluation rules, Stage 2 (Structural Validation) requires the *exact* `D\u0302` to match the unknown true `D`. Submissions with the wrong shape are rejected outright. With two Final Submissions allowed, the optimal play is to cover the two most defensible `D\u0302` hypotheses:

| `D\u0302`     | Justification                                                                                                                                    | Notebook                                                       |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| **16**  | Best W\u2081 signature match on Z (480-cell sweep), matches "bank's ML prediction API" tagline.                                                       | `pierce-the-veil-master-submission-d16` (primary)              |
| **132** | Paper §10.1 documented deployment (real-estate, 132-dim raw features → 16-dim latent).                                                          | This notebook.                                                 |

**Worst-case SRMSE drift analysis** (`D = 132`, `\u03b1 = 0.05`, 6 active channels):

```
worst case (all six r = -1):  SRMSE = sqrt((6*1.1025 + 126)/132) = 1.00141
neutral  (r = 0):              SRMSE = sqrt((6*1.0025 + 126)/132) = 1.00057
best case (all six r = +1):   SRMSE = sqrt((6*0.9025 + 126)/132) = 0.99772
```

So worst-case drift from the zeros baseline is **+0.14 %** and best-case is **-0.23 %** --- tighter than the primary D=16 kernel because the variance penalty is diluted across 126 inert columns.

See the primary notebook `pierce-the-veil-master-submission-d16` for the full attack-strategy writeup and the 8-stage rubric walkthrough.
"""


CELL2_PRELUDE = '''\
"""Self-test harness for the D=132 backup."""
import os
import time
import numpy as np
import pandas as pd

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

CANDIDATE_PATHS = [
    "/kaggle/input/competitions/pierce-the-veil/intercepted_data.csv",
    "/kaggle/input/pierce-the-veil/intercepted_data.csv",
    "../input/pierce-the-veil/intercepted_data.csv",
    "../input/competitions/pierce-the-veil/intercepted_data.csv",
    "../data/intercepted_data.csv",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("intercepted_data.csv not found in any candidate path")

Z_pub = pd.read_csv(DATA_PATH).iloc[:, 0].to_numpy(dtype=np.float64)
print(f"Loaded Z_pub: shape={Z_pub.shape}, mean={Z_pub.mean():+.4f}, std={Z_pub.std():.4f}")

t0 = time.time()
X_hat = reconstruct(Z_pub, Z_pub)
dt = time.time() - t0
assert X_hat.shape == (4096, 132), f"shape mismatch: {X_hat.shape}"
assert np.all(np.isfinite(X_hat)), "non-finite values"

rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(len(Z_pub))
X_perm = reconstruct(Z_pub, Z_pub[perm])
assert np.allclose(X_perm, X_hat[perm]), "latent-dependence test failed"

col_std = X_hat.std(axis=0)
print(f"Output shape:        {X_hat.shape}")
print(f"Output range:        [{X_hat.min():+.4f}, {X_hat.max():+.4f}]")
print(f"Active columns:      {int((col_std > 1e-12).sum())} / {X_hat.shape[1]}")
print(f"Per-column std (first 8): {[float(x) for x in col_std[:8]]}")
print(f"Elapsed time:        {dt*1000:.1f} ms on 4,096 rows")
print(f"Latent dependence:   PASS  (f(PZ) = Pf(Z) exactly)")
print(f"Determinism:         seed={RANDOM_SEED}; reconstruct() has no random calls")
'''


CELL3 = """\
pd.DataFrame(X_hat).to_csv("submission.csv", index=False, header=False)
print(f"Wrote submission.csv  shape={X_hat.shape}  "
      f"all_finite={bool(np.isfinite(X_hat).all())}")
"""


def _set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 4, f"unexpected backup notebook length: {len(cells)}"

    recon_code = RECON_PATH.read_text(encoding="utf-8")

    _set_source(cells[0], CELL0)
    _set_source(cells[1], recon_code)
    _set_source(cells[2], CELL2_PRELUDE)
    _set_source(cells[3], CELL3)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {NB_PATH} -- {len(cells)} cells, recon code length {len(recon_code)} chars.")


if __name__ == "__main__":
    main()
