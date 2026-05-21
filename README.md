<p align="center">
  <img src="assets/banner.png" alt="Pierce the VEIL banner" width="100%"/>
</p>

# Pierce the VEIL — Hack It and Crack It Simulation

End-to-end solution for the Kaggle competition
[Pierce the VEIL](https://www.kaggle.com/competitions/pierce-the-veil) by
Integrated Quantum Technologies. **Deadline: 2026-05-22 04:00 UTC.**

**Final Submissions (live Kaggle kernels):**
- Primary `D̂ = 16`: <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-master-submission-d16>
- Backup `D̂ = 132`: <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-backup-submission-d132>

**Standalone write-up:** [`WRITEUP.md`](WRITEUP.md) (mirrors the in-notebook argument for judges who prefer markdown).

## Strategy in one paragraph

We treat the competition as statistical cryptanalysis on the
*Vector-Encoded Information Layer* (VEIL) described in
[arXiv:2603.15842](https://arxiv.org/abs/2603.15842) by the competition
host. The paper proves (§9) and empirically demonstrates (§10.1) that
the encoder is non-invertible even when the attacker has *paired*
training data — which competition participants do not. We focus on the
three juried tracks (Attack Strategy & Analysis, Partial Reconstruction,
Best Write-Up) through a forensic encoder-identification analysis, a
Cramér-Rao + Fano-inequality SRMSE
floor derivation, and a `reconstruct()` that implements **six
calibrated leak channels** (linear, magnitude, sign, quadratic,
rank-Gaussian quantile, GMM mixture-component) at `α = 0.045` per
channel, with analytic worst-case drift bounded to ±1.7 % from the
zeros baseline (and empirically tighter — see Monte Carlo below).
A 100-seed Monte Carlo on synthetic surrogates gives mean SRMSE
`1.00015`, 95 % bootstrap CI `[0.99993, 1.00039]` — statistically
indistinguishable from the zeros baseline, which is consistent with
the published §10.1 result on a strictly-stronger attacker. We also
ship two notebooks (`D̂ = 16` primary, `D̂ = 132` backup) to hedge
the unknown true dimensionality.

## Repository layout

```
Pierce the VEIL Hack It and Crack It Simulation/
├── assets/
│   └── banner.png                        # notebook + README banner
├── data/
│   └── intercepted_data.csv              # 4096×1 leaked Z (not redistributed)
├── src/
│   ├── eda.py                            # 8-test forensic battery
│   ├── eda_results.json                  # cached EDA output
│   ├── signature_match.py                # Wasserstein-1 encoder sweep
│   ├── signature_results.json            # cached top-30 fingerprints
│   ├── surrogate_decoder.py              # per-feature SRMSE analytics
│   ├── reconstruct.py                    # primary algorithm (D=16)
│   ├── reconstruct_d132.py               # backup algorithm (D=132)
│   ├── self_tests.py                     # local 8-stage emulator
│   ├── eda_expanded.py                   # 14-figure expanded EDA pack
│   ├── eda_expanded_results.json         # cached expanded-EDA summary
│   ├── ablation.py                       # 18-variant algorithmic ablation
│   ├── ablation_results.json             # cached 100-seed ablation table
│   ├── ablation_figures.py               # render ablation bar + table
│   ├── make_figures.py                   # publication-grade plots
│   └── build_notebooks.py                # rebuild both notebooks from sources
├── notebook/
│   ├── pierce-the-veil-master.ipynb              # PRIMARY submission (executed)
│   └── pierce-the-veil-backup-d132.ipynb         # BACKUP submission (executed)
├── figures/                              # generated plots
├── research/                             # scraped competition pages (notebooks in .gitignore)
├── kaggle_push/                          # staging for kaggle CLI push (in .gitignore)
├── SUBMISSION_GUIDE.md                   # upload recipe + Final-Submission checklist
├── WRITEUP.md                            # standalone judge-friendly writeup
├── DISCUSSION_POST.md                    # draft Kaggle Discussion-forum thread body
├── LICENSE                               # MIT (code) / Kaggle rules (submission)
└── README.md                             # this file
```

## Reproduce locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Place intercepted_data.csv in data/

python src\eda.py
python src\signature_match.py
python src\surrogate_decoder.py
python src\self_tests.py
python src\make_figures.py
python src\build_notebooks.py
jupyter nbconvert --to notebook --execute notebook\pierce-the-veil-master.ipynb
```

## Reference local self-test output (D = 16 primary)

```
Stage 1  all_finite                       : True
Stage 2  shape                            : (4096, 16)
Stage 3  row_aligned_under_perm           : True
Stage 4  ours_srmse                       : 1.0020 (single seed)
         monte-carlo 100-seed mean        : 1.00015
         monte-carlo 95% bootstrap CI     : [0.99993, 1.00039]
         beats zeros baseline             : 49 / 100 seeds
Stage 5  frac_random_baselines_beaten     : 1.00
Stage 6  permutation_equivariant          : True
Stage 6  n_nonzero_cols                   : 6
Stage 7  generalisation std (100 seeds)    : < 0.001
Stage 8  imports_only_numpy               : True
Determinism (5 runs, bit-identical)       : Δ = 0.0
```

## What this submission includes

A descriptive list of the artifacts that ship in this repository and the
two Kaggle kernels. Nothing here is offered as a ranking against other
participants; their work is their own and judges will weigh it on its
own merits.

1. **480-cell empirical W₁ signature sweep** for `D̂ = 16`
   (`src/signature_match.py`, cached in `src/signature_results.json`).
2. **Six-channel calibrated leak stack** combining linear, magnitude,
   sign, quadratic, rank-Gaussian quantile, and GMM mixture-component
   leaks in a bounded-`α` framework (`src/reconstruct.py`).
3. **Cramér–Rao + Fano-inequality SRMSE floor** with measured
   `H(X|Z) ≈ 3.05` bits (master notebook §5).
4. **Three-pronged impossibility argument**: topological non-invertibility
   (paper §9), Fano lower bound, and the published §10.1 empirical
   result on a strictly-stronger attacker.
5. **Local 8-stage emulator** (`src/self_tests.py`) and 100-seed Monte
   Carlo characterisation of the SRMSE distribution
   (master notebook §9).
6. **Dual `D̂` hedge**: `D = 16` primary (`pierce-the-veil-master-submission-d16`)
   and `D = 132` backup (`pierce-the-veil-backup-submission-d132`) as
   Final Submissions.
7. **Compliance posture**: internet off in metadata, numpy-only
   imports, no `random` calls, bit-identical determinism across 5 runs.
8. **2024–2026 literature contextualization** in §15.1 of the master
   notebook and `WRITEUP.md`, citing
   [Fang et al. (arXiv:2411.10023)](https://arxiv.org/abs/2411.10023),
   [Liu et al. NeurIPS 2024 (arXiv:2410.05814)](https://arxiv.org/abs/2410.05814),
   [Stadler et al. USENIX Security 2024 (arXiv:2301.10053)](https://arxiv.org/abs/2301.10053),
   and 18 additional primary sources.
9. **Expanded EDA gallery** (22 figures total). Beyond the original
   marginal / mixture / duplicate analysis, an additional 14
   diagnostics ship in `src/eda_expanded.py`: KDE, ECDF, Q-Q vs Normal,
   Q-Q vs Student-t, Hill tail-index plot, autocorrelation, partial
   autocorrelation, Welch power spectral density, lag-1 scatter, `Z²`
   chi-square diagnostic, `|Z|` half-normal diagnostic, GMM
   2-component overlay, rank-rank scatter, sign-run-length, and a
   log-log tail-concentration plot. The eight most informative panels
   are embedded inline in §2.5 of the master notebook.
10. **18-variant algorithmic ablation** (`src/ablation.py` and §8.5 of
    the master notebook): single-channel ablation (6 variants),
    top-`k` subset, four-point `α` calibration sweep (0.005 / 0.020 /
    0.045 / 0.080), and five zero-paired-data approximations of
    published 2024 refinements (per-column ridge `α` from Liu 2024,
    hard-MAP mixture from Stadler 2024, copula channel and
    winsorisation from Fang 2024, Bayesian model averaging from Liu
    §4.2), plus a sign-symmetrised diagnostic. Each variant scored
    on the same 100-seed Monte Carlo with 95 % bootstrap CIs.
    Findings reported as measured: no variant statistically beats the
    all-zeros baseline on the synthetic surrogate, and the published
    refinements offer no advantage without paired training data.
11. **22-entry primary-source bibliography** (§15.3 of the master
    notebook and the References section of `WRITEUP.md`): host paper
    (§9 / §10.1 / §10.2), 5 entries from 2024–2026 model-inversion
    literature, 6 classical-MI references (Fredrikson 2015, Hidano
    2018, Zhang 2020, Shokri 2017, Carlini 2021, Zhu 2019), 6
    information-theory references (Cover–Thomas, Cramér 1946, Rao
    1945, Tishby 1999, Kraskov 2004, Berrett–Samworth–Yuan 2019),
    and 4 statistical-methodology references for the EDA (Sklar
    1959, Welch 1967, Hill 1975, Acklam 2003). Inline citations
    throughout sections 2, 5, 6, 8.5, and 15.1.

## License

Code is MIT-licensed. By Kaggle competition rules, submissions grant
the Competition Sponsor a non-exclusive, royalty-free, perpetual
license to use the submission.
