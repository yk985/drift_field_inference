# Rung 2 — kernel regression, against rung 1

Everything here comes from [`scripts/04_kernel_vs_binned.py`](../../scripts/04_kernel_vs_binned.py).
The estimator is [`KernelRegression`](../../dfi/estimators/local.py); its numerics
live in [`dfi/estimators/_kernel.py`](../../dfi/estimators/_kernel.py).

```bash
python scripts/04_kernel_vs_binned.py --quick          # layout check, ~2 min
python scripts/04_kernel_vs_binned.py --which explain  # one stage
python scripts/04_kernel_vs_binned.py                  # everything, ~1.5 h cold
```

**How scores are taken here.** Every method is scored on *every* evaluation point drawn
from `ρ_ss`, including points it declines to answer for, where it is scored on what it
actually predicts. Scoring only on supported points rewards abstention: a method graded
where it chose to answer can post almost any number, and cross-validation, which
predicts every held-out point, would then be optimising something other than the score.
Coverage is still measured and reported. Rung 1's own study
([`../binned_km/`](../binned_km/)) uses the supported-only convention with coverage
beside it, and its `d = 10` row shows why that matters.

---

## How the kernel works

Both rungs estimate the same thing — the average increment near a point,

    b(q) ≈ E[ Δx / Δt  |  x = q ]

— from targets `y_i = Δx_i/Δt` that are **enormously noisy**: each carries noise of
standard deviation `sqrt(2D/Δt)`, about 25× `|b|` at the default settings. The whole
game is how you average them.

**Binning** puts a fixed grid over the box and averages whatever lands in each cell.

**A kernel** centres a weight function *on the query point* and averages everything,
weighted by distance:

    b̂(q) = Σ w_i y_i / Σ w_i,        w_i = exp(-|x_i - q|² / 2h²)

That is **Nadaraya–Watson (NW)**. **Local-linear (LL)** uses the same weights but fits a
straight line `y ≈ b + B·(x − q)` through the neighbourhood by weighted least squares, and
keeps the intercept `b`.

### Why a kernel beats bins: first-order vs second-order bias

A bin returns **one number for its whole cell**. At a query a distance `δ` from the cell
centre that number is wrong by `∇b·δ`, which is *first order* in the cell width. A kernel is
centred on the query, so the linear term cancels by symmetry and the error is *second
order*.

Measured with noise-free targets, so only bias remains (figure 01d), as the log-log slope of
error vs smoothing scale:

| | slope |
|---|---|
| binned, at a random point | **0.97** |
| binned, at the cell centre | 1.84 |
| Nadaraya–Watson | 1.86 |
| local-linear | 1.90 |

The binned slope is ~1 everywhere except at the exact centre of a cell, which is
never where you query. You can see this directly in figure 03a: the binned error map is light
at cell centres and dark at the edges.

### Why local-linear beats Nadaraya–Watson: the density drags the average

The kernel is centred on `q`, but the samples it weighs are not uniform: they follow
`ρ_ss`. On a slope of the potential there is more data on the downhill side, so the
*effective* weight — kernel × density — has its centre of mass shifted towards the dense
region (figure 01a). NW therefore reports the drift from a point that is not `q`. The bias is

    h² [ ½ Δb  +  (∇b) ∇log ρ ]

and for a gradient field `∇log ρ_ss = b/D`, so the second term **grows like 1/D**. It is
worst at *low* noise, which is exactly where the variance is smallest and you would most
like to use a wide kernel.

For 1D Ornstein–Uhlenbeck (`b = −kx`, Gaussian `ρ`) this can be done exactly: NW's slope is
shrunk by `1/(1 + h²/σ²)` with `σ² = D/k`. Measured as the ratio of the NW and LL slopes (so
the noise they share cancels), the worst deviation from that law across seven bandwidths is
**0.0043** (figure 01c).

LL fits a slope inside the window, so lopsided weights tilt the line instead of moving the
intercept. It is exact for any linear `b`.

### How the bandwidth is chosen

Kernel *shape* barely matters; *width* is everything. `h` is the same bias–variance dial as
bin width. There are three options (`bandwidth=`):

- **a number** — use it.
- **`'silverman'`** — `h = σ n^(−1/(d+4))` from the spread of the positions. This is a
  density-estimation rule. It has the right exponent but cannot see the noise level `2D/Δt`
  or how fast `b` varies, which are the two things the regression optimum depends on.
  **Measured 30–56% worse than CV.**
- **`'cv'`** (default) — cross-validation. Split the *walkers* into 3 groups, fit on two,
  predict the raw `y` of the third, and repeat. Keep the `h` with the lowest squared
  prediction error.

Why that works without the ground truth: a held-out `y` is `b + noise`, with noise
independent of the fit, so

    E|y − b̂|² = (noise floor) + E|b − b̂|².

The floor is hundreds of times larger than the signal, **but it does not depend on `h`**, so
the `h` that minimises prediction error is the `h` that minimises true error (figure 02,
bottom row vs top row). Two details matter:

- **Split by walker, not by row.** For local-linear, a row's offset from its time-neighbours
  *is* its own increment, so row-level splits leak.
- **Rescale after.** CV tunes on 2/3 of the data, and the optimum scales as
  `n^(−1/(d+4))`, so the chosen `h` is multiplied by `(2/3)^(1/(d+4))`.

Against the ground-truth optimum (figure 02):

| `N·T` | method | CV's choice → error | best on the grid |
|---|---|---|---|
| 384k | binned | 0.512 | 0.507 |
| 384k | NW | 0.311 | 0.309 |
| 384k | LL | 0.262 | 0.254 |
| 3.07M | binned | 0.320 | 0.314 |
| 3.07M | NW | 0.154 | 0.158 |
| 3.07M | LL | 0.136 | 0.139 |

CV is never more than 3% from the best point on the grid, and sometimes beats it because it
refines between grid values. Binning gets **the same CV procedure** (`bins='cv'`), so every
comparison below is method against method, not a tuned kernel against a rule of thumb.

---

## The figures

### `01_how_the_kernel_works.png`
The four claims above, each checked: effective weight (a), three estimates at one smoothing
scale (b), the NW shrinkage law (c), bias order (d).

### `02_choosing_the_bandwidth.png`
True error vs smoothing scale (top) and what CV sees (bottom), at two budgets.

### `03a_recovery_well_sampled.png`
384k transitions, `D = 0.3`. Error under `ρ_ss`: **binned 0.512, NW 0.311, LL 0.262**.

### `03b_recovery_poorly_explored.png`
`D = 0.05`, so walkers stay in their basins. This one needs care, because the single-number
summary is misleading:

| | binned | NW | LL |
|---|---|---|---|
| error under `ρ_ss` | 0.350 | 0.159 | **0.106** |
| share of box answered | 32% | 35% | 84% |
| error inside own support, over the box | 0.98 | 0.86 | 1.20 |
| **error on the region all three answer for** (31% of the box) | 0.92 | 0.70 | **0.41** |

LL is genuinely the most accurate where a like-for-like comparison is possible (last row).
Its worse "own support" number comes from answering over far more of the box, including hard
regions the others decline. It can: CV tunes on held-out walkers, so it optimises where
walkers go, and LL's bias ignores the density gradients that force NW narrow.

"Support" here means **at least 8 transitions within 2h**, the kernel form of rung 1's
`min_count` (a 2h disc in 2D is almost exactly one bin cell at the same smoothing scale). It
certifies that data was *present*, not that the answer is *accurate*. With noise ~4|b| per
increment at this `D`, 8 samples still leave an error of order `|b|`. That caveat applies to
rung 1 equally.

### `04_sweeps.png` — walkers, `D`, measurement noise
3000 steps per walker, 2 seeds, band = min/max.

**Walkers.** Local-linear beats Nadaraya–Watson beats binning at every budget. Binning's
error is **1.61×** local-linear's with 16 walkers and **2.33×** with 1024. The gap
*widens* with data, because a first-order bias shrinks more slowly than a second-order
one as the smoothing tightens.

| walkers | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| binned | 0.934 | 0.657 | 0.472 | 0.343 |
| NW | 0.630 | 0.451 | 0.275 | 0.165 |
| LL | **0.579** | **0.402** | **0.231** | **0.147** |

**Noise level.** Local-linear's lead over Nadaraya–Watson depends on `D` exactly as the
bias formula says it should:

| `D` | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 | 1.5 |
|---|---|---|---|---|---|---|
| NW / LL | **1.70** | 1.70 | 1.39 | 1.11 | 1.01 | 0.99 |

NW's density-gradient bias scales as `1/D`. It decides the comparison at low noise and is
irrelevant once variance dominates.

**Measurement noise.** All three curves rise and converge. Position noise biases the
*target*: a noisy position sits, on average, downhill of the true one (Tweedie's
formula), so

    E[ Δx̃/Δt | x̃ ] = b + σ² ∇log ρ / Δt = b · (1 + σ²/(DΔt))   for a gradient field.

Measured with local-linear fits and divided by the σ = 0 fit: **1.080 / 1.320 / 2.265**
at σ = 0.005 / 0.01 / 0.02, against **1.079 / 1.317 / 2.268** predicted. No averaging
scheme removes it, and it gets worse as `Δt → 0`. It needs a different target (for
example, increments over two steps with the noise correlation modelled), not a better
smoother.

### `05_error_vs_dimension.png` — d = 2, 3, 5, 10

| `d` | binned (CV) | NW | LL | what CV chose |
|---|---|---|---|---|
| 2 | 0.260 | 0.135 | **0.116** | 14 bins; h ≈ 0.07–0.08 |
| 3 | 0.539 | 0.286 | **0.261** | 7 bins; h ≈ 0.12–0.13 |
| 5 | 1.001 | 0.928 | **0.925** | **1 bin**; h = 0.38 |
| 10 | 1.002 | 1.004 | 1.004 | **1 bin**; h at the ceiling |

(6.1M transitions, one seed.) The bottom-right panel is the mechanism. As `d` grows,
cross-validation widens every method's neighbourhood. By `d = 5` binning has chosen a
**single bin**, the global mean, and the kernels are averaging over a fifth of the box
side in every direction. An estimate averaged over most of the space *is* nearly the
global mean, which is why every error converges on 1.0, the score of predicting zero.
At this budget there is nothing local left to learn.

Kernels degrade more gracefully on the way. At `d = 3` they halve binning's error, and
at `d = 5` they still extract 7% where binning extracts nothing. But neither escapes.
That is the case for rungs 3 and 4, which trade locality for structure.

Two honest caveats. Above `d = 3` the kernel uses direct sums on a compressed copy of
the data (per-walker time blocks, then 200k rows), so its high-`d` numbers do not use
every transition. And the field backend switches from grid to mesh-free at `d = 4`.

---

## What was fixed along the way

- **Rung 1's bin rule had the wrong exponent.** It was derived from an O(h²) bias, but binned
  predictions are piecewise constant, so the bias is O(h) (slope 0.97 above). The correct
  balance gives `bins ∝ N^(1/(d+2))`, not `N^(1/(d+4))`. With it, a single constant (0.34)
  fits the measured optima in both 1D and 2D: implied constants 0.28–0.48, against 0.8–3.9
  with the old exponent. At `d = 3` and 6.1M transitions, that fix plus CV halves binning's
  error (1.16 → 0.54). All rung-1 outputs have been regenerated; see
  [`../binned_km/`](../binned_km/).
- **Kernel support.** The usual effective sample size `(Σw)²/Σw²` claimed support over 100%
  of the box at `D = 0.05`, and `Σw` over 93%. Both are fooled by a wide kernel's reach, and
  the error was above `|b|` across whole regions. Now: count within 2h.
- **`run_sweep` returned its whole cache file**, so a `--quick` run and a full run into the
  same CSV were averaged together. The first rung-1 sweeps figure mixed two budgets. It now
  returns only the configs requested.
- **Supported-only scoring rewarded abstention.** Binned CV at `d = 10` picked a
  million-cell grid that answers for 10% of points and scored 41 there. Comparisons now
  score everywhere (`SweepConfig(use_support=False)`). Binned CV may also choose a
  single bin, which it now does at `d ≥ 5`.
- **`n_eval` in result rows** was being overwritten by the scored-point count. That count
  is now `n_scored`.
