# Pierce the VEIL — Submission Guide

This is the end-to-end recipe for getting the two Final Submissions onto
Kaggle and confirming they pass evaluation.

## Deadline

**2026-05-22 04:00:00 UTC**. Submit by Thursday evening to leave buffer
for re-runs.

## The two Final Submissions (Kaggle allows up to 2)

| File                                              | `D̂`  | Strategy                                                                       | Analytic worst-case SRMSE drift |
|---------------------------------------------------|------|---------------------------------------------------------------------------------|--------------------------------|
| `notebook/pierce-the-veil-master.ipynb`           | 16   | 6 calibrated leak channels in cols 0..5 (`α = 0.045`); zeros in cols 6..15      | ±1.7 %                         |
| `notebook/pierce-the-veil-backup-d132.ipynb`      | 132  | Same 6 channels in cols 0..5 (`α = 0.05`); zeros in cols 6..131                  | ±0.14 % / −0.23 %              |

The six channels are: linear `z_std`, magnitude `|z_std| − E|z_std|`,
sign `sign(z − median)`, quadratic `(z_std² − 1)/√2`, rank-Gaussian
quantile `Φ⁻¹(rank/(N+1))`, and GMM mixture-component posterior
`2·P(C₁|z) − 1`. See §6 of `notebook/pierce-the-veil-master.ipynb` for
the full math.

Both share:
* `reconstruct(public_latents, hidden_latents, metadata=None)` signature
* Imports only `numpy`
* Deterministic (no `random`, no `seed`, no I/O; bit-identical across 5 runs)
* Permutation equivariant (`f(PZ) = P f(Z)` exactly)
* Per-row finite output, defensively sanitised

## Live kernels (already uploaded)

| Kernel | URL |
|---|---|
| Primary `D̂ = 16` | <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-master-submission-d16> |
| Backup `D̂ = 132` | <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-backup-submission-d132> |

Both have `KernelWorkerStatus.COMPLETE`. Verify with:

```powershell
kaggle kernels status ladyfaye/pierce-the-veil-master-submission-d16
kaggle kernels status ladyfaye/pierce-the-veil-backup-submission-d132
```

## Re-pushing after changes (only if needed)

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
python src\build_notebooks.py
jupyter nbconvert --to notebook --execute notebook\pierce-the-veil-master.ipynb --output pierce-the-veil-master.executed.ipynb --ExecutePreprocessor.timeout=240
jupyter nbconvert --to notebook --execute notebook\pierce-the-veil-backup-d132.ipynb --output pierce-the-veil-backup-d132.executed.ipynb --ExecutePreprocessor.timeout=90
cp notebook\pierce-the-veil-master.ipynb kaggle_push\primary\
cp notebook\pierce-the-veil-backup-d132.ipynb kaggle_push\backup\
kaggle kernels push -p kaggle_push\primary
kaggle kernels push -p kaggle_push\backup
```

## Final-Submission selection (user-action-only, on Kaggle web UI)

Kaggle's "Submit" flow requires a manual click:

1. Go to <https://www.kaggle.com/competitions/pierce-the-veil/submissions>.
2. Click **Submit Prediction** (or **Select Submission**).
3. Pick **both** committed kernels as the two Final Submissions.
4. Confirm.

Without this step the kernels exist on Kaggle but are **not entered**.

## Tracks targeted (in priority order)

1. **Best Attack Strategy & Analysis** — the forensic writeup in
   `pierce-the-veil-master.ipynb`: 480-cell W₁ encoder sweep, three-pronged
   impossibility argument, Cramér-Rao + Fano floors, six-channel
   calibrated leak stack, D=132 refutation, head-to-head with 27 public
   competitor notebooks.
2. **Best Technical Write-Up** — same notebook, double-purposed: every
   claim tied to either an in-notebook computation or a primary-source
   citation; rubric-mapped walkthrough in §14; standalone
   `WRITEUP.md` mirrors the argument for judges who prefer markdown.
3. **Partial Reconstruction** — the six calibrated channels operationalise
   the paper §10.2 magnitude leak; 100-seed Monte Carlo (mean SRMSE
   `1.00015`, 95 % bootstrap CI `[0.99993, 1.00039]`) characterises the
   risk envelope.
4. **Full Reconstruction Grand Prize** — attempted opportunistically; the
   published §10.1 result reports `−0.0003` reconstruction advantage with
   `p = 0.4706` under a strictly stronger attacker than this competition
   affords, which bounds any honest attempt from above. We say so directly
   in §4.3.

## What we are not doing (and why)

- **No internet, no external data, no platform tricks.** Strictly
  `numpy` + the supplied `intercepted_data.csv`.
- **No ensembling, no neural decoder, no Optuna tuning.** Every extra
  hyperparameter is one more way to overfit to the public batch and
  fail Stage 7 (generalisation).

## Compliance checklist (re-verified at every build)

- [x] `reconstruct(public_latents, hidden_latents, metadata=None)` defined
- [x] Output is `np.ndarray` of shape `(N_hid, D_HAT)`
- [x] Output is float64, all finite
- [x] Internet disabled in kernel metadata
- [x] Imports only `numpy` (and `pandas` for the wrapper-cell `submission.csv` write)
- [x] No `random`, no `seed`, no time-dependent behaviour
- [x] Five back-to-back runs produce bit-identical output
- [x] `f(PZ) = P f(Z)` exactly (`atol = 1e-12`)
- [x] Notebook runs end-to-end on a Kaggle CPU kernel in < 30 s (master) / < 5 s (backup)

## What if neither `D̂` is correct?

Both notebooks still target the three juried tracks (Attack Strategy &
Analysis, Best Write-Up, Partial Reconstruction) via the writeup. Those
tracks are judged on quality of analysis, not specific SRMSE values.
Even if `D̂` is wrong for both submissions (and the Grand Prize is
out of reach regardless per §4.3), the juried tracks remain in play.
