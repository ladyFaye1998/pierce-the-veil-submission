# Pierce the VEIL — Submission Guide

This is the end-to-end recipe for getting our two Final Submissions onto
Kaggle and confirming they pass evaluation.

## Deadline

**2026-05-22 04:00:00 UTC** (~Friday morning). Submit by Thursday evening
to leave buffer for re-runs.

## Our final submissions (we ship both — Kaggle allows up to 2)

| File                                              | `D̂`  | Strategy                                            | Worst-case SRMSE drift |
|---------------------------------------------------|------|-----------------------------------------------------|------------------------|
| `notebook/pierce-the-veil-master.ipynb`           | 16   | Linear + magnitude in cols 0,1; zeros in cols 2..15 | ±0.5 %                |
| `notebook/pierce-the-veil-backup-d132.ipynb`      | 132  | Same allocation; zeros in cols 2..131 (paper §10.1) | ±0.1 %                |

Both share:
* `reconstruct(public_latents, hidden_latents, metadata=None)` signature
* Imports only `numpy`
* Deterministic (no `random`, no `seed`, no I/O)
* Permutation equivariant (`f(PZ) = P f(Z)` exactly)
* Per-row finite output, fully sanitised

## Step-by-step upload

1. **Verify locally:**
   ```powershell
   python src\self_tests.py             # primary (D=16)
   python src\make_figures.py           # regenerate figures (optional)
   ```
   All eight stages must print pass-equivalent values; see
   `notebook/pierce-the-veil-master.executed.ipynb` for reference numbers.

2. **Open Kaggle:**
   - https://www.kaggle.com/competitions/pierce-the-veil
   - Click **Code → New Notebook**

3. **Paste `pierce-the-veil-master.ipynb`:**
   - In the Kaggle notebook, *File → Upload Notebook* → select
     `notebook/pierce-the-veil-master.ipynb`.
   - Add the competition dataset as input
     (`Add Data → Competition Data`).
   - In the right-hand panel, set **Internet OFF**, **GPU OFF**.
   - Click **Save & Run All (Commit)**.
   - Once committed, click **Submit to Competition** on the committed
     version. Save the version number.

4. **Repeat for `pierce-the-veil-backup-d132.ipynb`** as a separate
   notebook (do not put both in the same notebook; the scorer imports a
   single `reconstruct`).

5. **Select Final Submissions:**
   - Go to *My Submissions* and pick both committed notebooks as your
     two Final Submissions (Kaggle's two-final-submission rule).

## What we are not doing (and why)

- **No Grand Prize attempt.** The host's own paper (arXiv:2603.15842
  §10.1) reports that even a stronger-than-deployment attacker, given
  paired `(Ψ, X)` examples, fails to reconstruct (advantage = −0.0003,
  p = 0.4706). We have less information than that attacker; we cannot
  win the $8,000 Grand Prize and we do not pretend otherwise.

- **No internet, no external data, no platform tricks.** Strictly
  numpy + a public scalar dataset.

- **No ensembling, no neural decoder, no Optuna tuning.** Every extra
  parameter is one more way to overfit to the public batch and
  catastrophically fail Stage 7 (generalisation).

## What we are doing (in priority order)

1. **Attack Strategy & Analysis ($1,200)** — the comprehensive forensic
   writeup in the master notebook, identifying the encoder family,
   estimating `D̂`, reconciling with the paper's published deployment,
   and quantifying the information-theoretic floor.

2. **Best Technical Write-Up ($200)** — same notebook, double-purposed:
   carefully sourced, fully reproducible, every claim tied to either an
   in-notebook computation or a paper citation.

3. **Partial Reconstruction ($600)** — the calibrated `α = 0.08` signal
   allocation across the linear and magnitude channels (per paper §10.2)
   is a *bona fide* partial recovery operationalisation, and it is
   precisely the strategy the paper documents as "useful signal already
   exposed by simple geometric properties".

4. **Full Reconstruction Grand Prize ($8,000)** — opportunistically: if
   the true `D` happens to be 16 (primary) or 132 (backup), and if the
   chosen 2 columns happen to align with high-correlation features, we
   may marginally beat Stage 4 baselines. The expected value of this
   is small; we maximise it by keeping the per-column allocation small
   enough that we never *fail* Stage 4 either.

## Compliance checklist (must be re-verified before submitting)

- [x] `reconstruct(public_latents, hidden_latents, metadata=None)` defined
- [x] Output is `np.ndarray` of shape `(N_hid, D_HAT)`
- [x] Output is float64, all finite
- [x] Internet disabled in notebook settings
- [x] Imports only `numpy` (and `pandas` for I/O at scoring scaffolding)
- [x] No `random`, no `seed`, no `time`-dependent behaviour
- [x] Five back-to-back runs produce bit-identical output
- [x] `f(PZ) = P f(Z)` exactly (atol = 1e-12)
- [x] Notebook runs end-to-end on a Kaggle CPU kernel in < 1 minute

## What if neither `D̂` is correct?

Both notebooks still target the three secondary tracks via the writeup.
The Strategy & Analysis and Best Write-Up tracks are judged on quality
of analysis, not specific SRMSE values. Even with `D̂` wrong (=> Stage 2
DQ for Grand Prize purposes), the secondary tracks remain in play
because they are juried, not metric-gated.
