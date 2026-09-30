# Rung 3 — basis projection (Stochastic Force Inference)

The estimator is [`BasisProjection`](../../dfi/estimators/basis.py) (Frishman & Ronceray,
PRX 2020). These figures come from the same study script as the other local estimators,
on the **same walkers**, re-used from the trajectory cache (see
[`../local_study/`](../local_study/)):

```bash
python scripts/03_local_estimator.py --estimators sfi
```

## How it works

Instead of averaging increments near each point, expand the drift in a basis,
`b(x) = f(x) C`, and solve for the coefficients by least squares against every increment
at once: `C = (FᵀF)⁻¹ FᵀY`. Every transition informs every coefficient.

- **Basis.** On the torus this is Fourier, `[1, cos(k·x), sin(k·x), …]` over lattice
  vectors with `|n|² ≤ L`. Both field generators build the potential from lattice modes, so
  a large enough basis spans the truth *exactly*. The cutoff is a Euclidean ball, matching
  the fields' isotropic spectrum. On an open box it uses Legendre polynomials scaled to the
  data.
- **Only sufficient statistics.** `G = FᵀF`, `B = FᵀY` and `Σ|y|²` are accumulated in
  chunks on the GPU (~2×10¹² flops/s), so every transition is used in every dimension,
  without subsampling.
- **Size by cross-validation over walkers, for free.** The bases are *nested*, so each
  smaller basis's normal equations are a leading block of the largest. One pass over the
  data gives every candidate, and each held-out loss is
  `S − 2 tr(CᵀB) + tr(CᵀGC)`, with no refits.
- **The fit predicts its own error.** Noise of variance `σ² = 2D/Δt` per increment, spread
  over `p·d` coefficients, gives a mean squared error of `d·p·σ²/n`, with `σ²` taken from
  the residuals. That's a prediction computed without the ground truth. It covers noise
  only; the gap between it and the measured error is projection bias.
- **Correction to the original stub:** its formula `p·d·D/(nΔt)` was missing a factor
  of 2. The measurements below settle it.

## Validation

Checked against independent references before any figure was drawn:

| check | result |
|---|---|
| chunked accumulation vs direct `lstsq` | coefficients agree to 3e-7 (GPU float32 blocks), 4e-15 (CPU) |
| CV loss from sufficient statistics vs brute-force refit per fold | 1e-10 for every well-posed size (they diverge only where the basis has outrun the data, far above the minimum) |
| error prediction vs measured error, exact basis, 1D OU | **1.016 ± 0.034** and **1.041 ± 0.037** (800 seeds each) |
| same, 2D field, `p = 81`, 8 seeds | **1.05 ± 0.03** at 32k transitions, **0.96 ± 0.04** at 1.5M |
| pointwise error band: truth inside ±1.96 SE | **94.9 ± 0.7%** and **95.6 ± 0.6%** |

One subtlety found along the way: the prediction is an average over the *sampled* points.
Scored under ρ_ss with only 32 walkers the ratio is 1.40, because the sample hasn't yet
covered ρ_ss. That isn't bias: on the sampled points the same fits give 1.05. Two tests in
`tests/test_dfi.py` pin the prediction and the nested-basis identity.

## The figures

| figure | what it shows |
|---|---|
| `01_setup.png` | the test; **nrmse 0.097** with 21 functions (binned 0.293, LL 0.126) |
| `02_recovery_vs_noise.png` | `D = 0.05 … 1.0`: **0.025, 0.043, 0.108, 0.211** |
| `03_recovery_vs_walkers.png` | `N = 16 … 1024`: **0.392, 0.173, 0.107, 0.053** |
| `04_basis_size_and_bound.png` | error vs basis size with the prediction; error vs data for fixed sizes |
| `05_error_vs_dimension.png` | `d = 2, 3, 5, 10`, and measured error against prediction |
| `06_sweeps.png` | walkers, `D`, measurement noise; prediction overlaid |

**Figure 04 is the one to look at.** On the left, at 12.3M transitions, measured error and
prediction coincide for every basis above ~29 functions. They're the same line, so the
fit's error is known without the truth. Below that size the measured error lifts off the
prediction: 0.22 against 0.03 at 13 functions. That gap is the modes the small basis
leaves out. Cross-validation picks within 1% of the best size at both budgets. On the
right, using nested walker subsets of one run: with 261 functions measured/predicted runs
**1.13 → 0.95** from 20k to 12M transitions, falling at the predicted rate. With 13 it runs
**1.43 → 6.82**, because the variance keeps falling while the bias doesn't.

**Low noise is where SFI's global nature shows (figure 02).** At `D = 0.05` it scores
0.025 under ρ_ss, the best number any method has posted, while its error map is dark across
the regions no walker visited. A Fourier series fitted to data in the basins extrapolates
across the gaps with full confidence. A global basis has no notion of "no data here". The
kernels refuse or thin out there; SFI answers.

**Dimension (figure 05).** Cross-validation chose 37 / 81 / 51 / 1 functions at
`d = 2 / 3 / 5 / 10`.

| `d` | measured | predicted (noise only) |
|---|---|---|
| 2 | **0.082** | 0.079 |
| 3 | **0.195** | 0.181 |
| 5 | 0.946 | 0.311 |
| 10 | 1.002 | 0.077 |

Up to `d = 3` the basis spans the field and the prediction is the error. At `d = 5` the
field's modes sit around `|n|² ≈ 3`, where the lattice already holds 131 modes, and the
basis CV can afford at this budget reaches only part of them. At `d = 10` (modes near
`|n|² ≈ 6`, thousands of them) CV retreats to a single constant. The prediction then
honestly reports that the remaining noise is small, and the measured error shows that
almost everything is bias. **The gap between the two lines is how SFI, run on real data,
would tell you it has run out of basis.**

**Sweeps (figure 06).** 0.578 at 16 walkers → 0.100 at 1024, with the prediction tracking
from 1.43× below to 1.07×. Best `D` is 0.05 (0.042). At σ_obs = 0.02 it reaches 1.43: the
Tweedie target bias hits every method equally.

## Against the other estimators

On identical walkers (full tables in [`../local_study/`](../local_study/)):

- **SFI wins once there is data**: 4.8M transitions 0.097 vs LL 0.126; 1024 walkers 0.053
  vs 0.088; `d = 3` 0.195 vs 0.261.
- **Kernels win with very little**: 16 walkers, LL 0.358 vs SFI 0.392 (sweeps: 0.516 vs
  0.578). With few walkers CV can only afford 13 functions, and a small global basis
  pays in bias what a local method doesn't.
- **Nothing wins at `d ≥ 5`**: SFI 0.946 / 1.002, LL 0.925 / 1.004. This is the regime the
  neural rung is for.
