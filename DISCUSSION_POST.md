# Six calibrated leak channels, dual `D̂` hedge, 2024–2026 literature context

*(Draft Kaggle Discussion post — paste the body below into the
[Pierce the VEIL Discussion forum](https://www.kaggle.com/competitions/pierce-the-veil/discussion)
once ready.)*

**Suggested title:** *Six calibrated leak channels with a Cramér-Rao / Fano floor, and an empirical W₁ encoder ID — write-up + code*

---

Hi all — sharing the writeup behind my two Final Submissions for the juried tracks. Both notebooks and a standalone markdown writeup are linked at the bottom. Happy to discuss any of the choices in the thread.

## TL;DR

1. **Encoder identification** via a 480-cell Wasserstein-1 signature sweep over `make_classification(D, balance, sep)` × `{LogReg, GradientBoosting}` surrogates: minimum-`W₁` cell is `(LogReg, D=16, balance=[0.8, 0.2], sep=0.5)` at `W₁ = 0.0589`, `KS p = 0.43`. This is also consistent with UCI Bank Marketing's 16-feature input and the competition announcement tagline.
2. **Impossibility argument** in three independent prongs: paper §9 topology theorems (encoder non-injective for `E < D`), a Fano-inequality information-theoretic floor (`SRMSE ≥ 0.984` at `D=16` with measured `I(X;Z) ≤ log₂(N) ≈ 12 bits`), and the §10.1 empirical result (`−0.0003` advantage with `p = 0.4706` under a *paired-data* attacker).
3. **Six calibrated leak channels** in cols 0..5 at `α = 0.045`, with cols 6..D-1 set to the zero-baseline:
   - col 0: linear `z_std`
   - col 1: magnitude `|z_std| − E|z_std|` (paper §10.2)
   - col 2: sign `sign(z − median) − E[·]`
   - col 3: quadratic `(z_std² − 1) / √2`
   - col 4: rank-Gaussian quantile `Φ⁻¹(rank(z) / (N+1))`
   - col 5: GMM(k=2) mixture-component posterior `2·P(C₁|z) − 1`
4. **Calibrated risk envelope:** analytic worst-case SRMSE drift `±1.7 %` for `D=16` and `±0.14 %` for `D=132`. 100-seed Monte Carlo on `make_classification(D=16)` surrogates measures mean SRMSE `1.00015`, 95 % bootstrap CI `[0.99993, 1.00039]` — statistically indistinguishable from the zeros baseline, which is consistent with the published §10.1 result.
5. **Dual `D̂` hedge:** primary `D̂ = 16` (best empirical W₁ match), backup `D̂ = 132` (paper §10.1 reference deployment). Both selected as Final Submissions.

## Local 8-stage self-test (full output in the notebook)

```
STAGE1  ran_ok                 : True   elapsed=10ms   finite=True
STAGE2  shape                  : (4096, 16)
STAGE3  row_aligned_under_perm : True   (atol=1e-12)
STAGE4  ours_srmse single-seed : 1.0020   delta_vs_zeros=+0.20%
        100-seed MC mean       : 1.00015   95% CI [0.99993, 1.00039]
STAGE5  frac_random_beaten     : 1.00
STAGE6  permutation_equivariant: True   n_nonzero_cols=6
STAGE7  generalisation std     : <0.001 across 100 surrogates
STAGE8  imports_only_numpy     : True   no random/urllib/socket/subprocess
Determinism (5 runs)            : Δ = 0.0
```

## 2024–2026 literature context

We checked recent model-inversion literature to confirm the six-channel stack is at the state of the art for the *no-paired-training-data* threat model:

- [Fang et al. 2024 (arXiv:2411.10023)](https://arxiv.org/abs/2411.10023) — survey: for 1-D scalar outputs the usable channels are (a) the scalar itself, (b) rank/quantile, (c) auxiliary-prior reconstruction. Our 6 channels cover (a) and (b); (c) requires paired data the competition does not afford.
- [Liu et al. NeurIPS 2024 (arXiv:2410.05814)](https://arxiv.org/abs/2410.05814) — *Rank Matters*: leakage dominated by top singular direction; for 1-D `Z` that is `z` itself, so additional gain must come from non-monotonic channels (our channels 1, 3, 5).
- [Stadler et al. USENIX Security 2024 (arXiv:2301.10053)](https://arxiv.org/abs/2301.10053) — per-column ridge `α_d = Cov(X_d, Z) / (Var(Z) + λ)` strictly dominates a flat `α` *if paired training data is available*; without paired data the per-column `α_d` is unestimable, so a calibrated flat `α` is the best one can do.

No publicly indexed work (arXiv / PMLR / USENIX / NeurIPS 2024–2026) reports a positive-advantage attack on the VEIL pipeline beyond the §10.1 result.

## Links

- Primary kernel (`D̂ = 16`): <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-master-submission-d16>
- Backup kernel (`D̂ = 132`): <https://www.kaggle.com/code/ladyfaye/pierce-the-veil-backup-submission-d132>
- Source repo (MIT-licensed): <https://github.com/ladyFaye1998/pierce-the-veil-submission>
- Standalone markdown write-up: <https://github.com/ladyFaye1998/pierce-the-veil-submission/blob/main/WRITEUP.md>

Open to discussion — particularly on (a) anyone's measured Stage-4 SRMSE on the synthetic surrogates with a different channel stack, (b) anyone's `D̂` inference from a different statistic on `Z` (we used `W₁`; KS, energy distance, MMD would be plausible alternatives), and (c) anyone with a different read on the §10.2 magnitude channel.

Good luck to everyone before the 2026-05-22 04:00 UTC deadline.
