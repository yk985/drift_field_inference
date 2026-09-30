# The estimators, on identical data

One study, eight estimators, the **same walkers** — except rungs 5, which are fed a
different *kind* of data on purpose (see the note below the tables):

| folder | estimator | smoothing |
|---|---|---|
| [`../binned_km/`](../binned_km/) | Binned Kramers–Moyal (rung 1) | bin count from the `N^(1/(d+2))` rule |
| [`../kernel_nw/`](../kernel_nw/) | Kernel regression, Nadaraya–Watson | bandwidth by cross-validation over walkers |
| [`../kernel_ll/`](../kernel_ll/) | Kernel regression, local-linear | bandwidth by cross-validation over walkers |
| [`../sfi/`](../sfi/) | Basis projection (SFI, rung 3) | Fourier basis size by cross-validation over walkers |
| [`../nn_mlp/`](../nn_mlp/) | Neural network, MLP (rung 4) | training steps, by held-out walkers |
| [`../nn_cnn/`](../nn_cnn/) | Neural network, CNN (rung 4, `d ≤ 3`) | training steps and output grid, by held-out walkers |
| [`../cnn_amortised/`](../cnn_amortised/) | Amortised CNN (rung 4b, 2D) | none per dataset: trained once on 3000 other simulated fields |
| [`../score_kde/`](../score_kde/) | KDE score (rung 5, snapshots) | bandwidth by held-out score matching |
| [`../score_dsm/`](../score_dsm/) | Denoising score matching (rung 5, snapshots) | read-out noise level by held-out score matching |

Each folder has the same six figures (`01_setup` … `06_sweeps`), so any figure can be laid
beside its counterpart.

```bash
python scripts/03_local_estimator.py --estimators binned nw ll sfi nn cnn    # everything
python scripts/03_local_estimator.py --estimators ll --which noise    # one figure
```

## Same data, verified

Every simulation goes through `dfi.sweeps.simulate_config` with a trajectory cache
(default `~/.cache/dfi/trajectories`, ~1.7 GB for this study, overridable with
`DFI_TRAJ_CACHE`). It is kept out of the project folder so it doesn't sync. All
estimators, and any later rung, are fitted to the same saved paths. Rung 3 was added
afterwards and fitted entirely from the cache: no new simulation.

Nothing had been saved before the cache existed, so rung 1's scores came from earlier,
uncached simulations. These are seeded, so re-simulating reproduces them. Checked by
re-fitting binned KM on the cached trajectories and comparing with its stored scores:
**all 75 configs reproduce, 56 bit-for-bit and the rest within 2.2e-16** (float rounding
in summation order). The kernels' dimension scores here also match `../kernel/`, which
simulated the same configs independently, to every printed digit.

[`data/`](data/) holds the shared score table: one row per (config, estimator).

## Side by side

Scores are taken where each estimator claims support, with coverage beside them, which is
rung 1's convention. The kernels answer for ~100% of evaluation points everywhere below, so
for them it makes no difference. The exception is binned at `d = 10`. `../kernel/` redoes the
comparison with every-point scoring and CV-tuned bins.

**4.8M transitions, `D = 0.3`** (figure 01)

| | binned | NW | LL | SFI | MLP | CNN |
|---|---|---|---|---|---|---|
| nrmse | 0.293 | 0.147 | 0.126 | 0.097 | 0.100 | 0.115 |
| smoothing | 16 bins | h = 0.072 | h = 0.088 | 21 functions | 7125 steps | 16² grid, 250 steps |

| | amortised CNN | KDE score | DSM |
|---|---|---|---|
| nrmse | **0.060** | 0.130 | 0.072 |
| smoothing | pretrained | h = 0.076 | σ = 0.025 |

**Noise level**, 4.8M transitions (figure 02)

| `D` | 0.05 | 0.15 | 0.4 | 1.0 |
|---|---|---|---|---|
| binned | 0.369 | 0.251 | 0.362 | 0.785 |
| NW | 0.077 | 0.115 | 0.174 | 0.281 |
| LL | 0.067 | 0.080 | 0.160 | 0.280 |
| SFI | **0.025** | **0.043** | **0.108** | **0.211** |
| MLP | **0.025** | 0.057 | 0.129 | 0.319 |
| CNN | 0.031 | 0.057 | 0.132 | 0.333 |
| amortised | 0.031 | 0.038 | 0.074 | 0.134 |
| KDE score | 0.131 | 0.138 | 0.145 | 0.843 |
| DSM | 0.053 | 0.055 | 0.083 | 0.164 |

At `D = 0.05` read the scores with the coverage in each figure's titles. Walkers stay in
their basins: binned answers for 42% of the box, LL for 62%, and LL's answers across the
unexplored gap are visibly wrong. The scores are measured where the walkers actually go.
LL's lead over NW is largest at low `D` (1.44× at 0.15) and gone by `D = 1`. That is the
`1/D` density-gradient bias explained in [`../kernel/`](../kernel/).

**Walkers**, 12000 steps each (figure 03)

| `N` | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| binned | 0.648 | 0.476 | 0.325 | 0.224 |
| NW | 0.415 | 0.267 | 0.175 | 0.103 |
| LL | 0.358 | 0.231 | 0.144 | 0.088 |
| SFI | 0.392 | **0.173** | **0.107** | **0.053** |
| MLP | **0.340** | 0.199 | 0.111 | 0.080 |
| CNN | 0.380 | 0.208 | 0.166 | 0.068 |
| amortised | **0.257** | **0.128** | 0.074 | 0.043 |
| KDE score | 0.132 | 0.130 | 0.130 | 0.130 |
| DSM | 0.194 | 0.072 | 0.072 | 0.072 |

Both kernels with 64 walkers beat binning with 256 (0.267 and 0.231 against 0.325), and with 256 they beat binning with 1024 (0.175 and 0.144 against 0.224). That's about a 4× saving in data for the same accuracy.

**The smoothing dial** (figure 04): automatic choice vs the best point on the grid

| | 192k: best → chosen | 12.3M: best → chosen |
|---|---|---|
| binned (rule) | 0.654 → 0.724 | 0.216 → 0.224 |
| NW (CV) | 0.437 → 0.439 | 0.104 → **0.103** |
| LL (CV) | 0.393 → **0.389** | 0.090 → **0.088** |

Cross-validation lands on the optimum, and in three cases beats the grid by choosing
between grid points. Silverman's rule sits well left of the optimum in both kernel figures.

**Dimension**, 6.1M transitions (figure 05)

| `d` | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| binned | 0.268 | 0.564 | 1.62 | 41.3 (on 10% of points) |
| NW | 0.135 | 0.286 | 0.928 | 1.004 |
| LL | 0.116 | 0.261 | **0.925** | 1.004 |
| SFI | **0.082** | **0.195** | 0.946 | 1.002 |
| MLP | 0.093 | 0.223 | 0.927 | 1.019 |
| CNN | 0.088 | 0.224 | — | — |
| amortised | 0.061 | — | — | — |
| KDE score | 0.128 | 0.231 | 0.983 | 0.997 |
| DSM | **0.072** | **0.142** | **0.625** | 0.996 |

The kernels widen instead of refusing. By `d = 10` their bandwidth is 0.68 on a box of
side 2, an average over most of the space, which scores like predicting zero.

**Sweeps**, 4000 steps, 3 seeds (figure 06)

| | 16 → 1024 walkers | best `D` | σ_obs = 0.02 |
|---|---|---|---|
| binned | 0.909 → 0.326 | 0.1 (0.377) | 1.56 |
| NW | 0.560 → 0.157 | 0.05 (0.135) | 1.22 |
| LL | **0.516** → 0.139 | 0.05 (0.084) | 1.25 |
| SFI | 0.578 → **0.100** | 0.05 (**0.042**) | 1.43 |
| MLP | 0.541 → 0.117 | 0.05 (0.046) | 1.38 |
| CNN | 0.759 → 0.132 | 0.05 (0.057) | 1.44 |
| amortised | 0.424 → **0.068** | 0.05 (**0.040**) | 0.63 |
| KDE score | 0.157 → 0.135 | 0.2 (0.134) | 0.14 |
| DSM | 0.614 → 0.077 | 0.05 (0.050) | **0.08** |

At σ_obs = 0.02 all three are above 1.0. The bias `b(1 + σ²/(DΔt))` is in the target, and
no smoother removes it.

## Where each one wins

- **SFI (rung 3)** wins once there is data: fewest parameters, every transition used for
  every coefficient, and it predicts its own error. It loses with very little data (16
  walkers), and it extrapolates confidently where no walker went (`D = 0.05`, figure 02).
- **Local-linear kernel** is the best local method, and the best overall with very
  little data. It never extrapolates far, because its support is local.
- **Nadaraya–Watson** trails local-linear wherever density gradients are steep (low `D`).
- **Binned KM** trails everywhere; its piecewise-constant estimate is first-order biased.
- **The MLP (rung 4)** sits between SFI and the local-linear kernel. It is best of all
  with the least data at 12000 steps (0.340 at 16 walkers), and trails SFI once data is
  plentiful. Early stopping on held-out walkers is its smoothing choice, and at large
  budgets it stops ~5% short of the truth's best step.
- **The CNN (rung 4)** trails the MLP almost everywhere and is unstable with little data.
  It wins against the MLP only at the largest 2D budget (0.068 vs 0.080), and it cannot
  run past `d = 3`.
- **The amortised CNN (rung 4b)** is not fitted per dataset. It was trained once on 3000
  other simulated fields and is only applied. It beats every per-dataset method nearly
  everywhere on these walkers: 0.060 at 4.8M transitions against SFI's 0.097, and 0.257 at
  16 walkers against 0.340. It does so because it has learned what fields from this
  generator look like, a prior the other rungs don't have. It loses outside that prior
  (stronger drift, strong rotation, near noise-free data: 2–3.5× worse than SFI; fields from other generators such as sparse wells or cellular patterns: up to 1.9× worse) and on
  the lowest-noise study case (`D = 0.05`). See [`../cnn_amortised/`](../cnn_amortised/).
- **The score estimators (rung 5)** are not fed the same data: they get positions without
  time order (independent samples of `ρ_ss`, matched in count, capped at 500k) plus the
  known `D`. On a *gradient* field that data is richer than the increments -- DSM reaches
  0.072 at 4.8M against SFI's 0.097 -- and it is almost immune to measurement noise (0.08
  at `σ_obs` = 0.02, where every trajectory method is above 1.2). It is also completely
  blind to circulation: on a rotational field its error against the rotational part is
  exactly 1.0 at every `ω`. See [`../score/`](../score/).
- **At `d ≥ 5` almost nothing learns the field** at this budget. The exception is DSM at
  `d = 5` (0.625 against ~0.93 for every trajectory method), on a gradient field and with
  idealised snapshots. It isn't capacity: the MLP fits the exact 5D drift to 0.04. It's
  information: with ~100× the data, MLP and SFI reach 0.25 and 0.20. See
  [`../nn_mlp/`](../nn_mlp/).
