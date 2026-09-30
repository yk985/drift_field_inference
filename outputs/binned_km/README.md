# Rung 1 — binned Kramers-Moyal, on its own

Everything here comes from [`scripts/03_local_estimator.py`](../../scripts/03_local_estimator.py).
For how it compares with rung 2, see [`../kernel/`](../kernel/).

```bash
python scripts/03_local_estimator.py --quick            # ~40 s, layout check
python scripts/03_local_estimator.py --animate          # full, + the mp4
python scripts/03_local_estimator.py --which dims       # one stage
```

Sweep rows are cached in [`../local_study/data/`](../local_study/data/), shared with the kernel studies, so a re-run redraws the
figures without recomputing anything.

The closed-form error budget of this estimator -- variance, the two biases, and what
measurement noise does -- is derived and measured in
[`docs/binned_km_error_analysis.pdf`](../../docs/binned_km_error_analysis.pdf), from
[`scripts/09_binned_error_analysis.py`](../../scripts/09_binned_error_analysis.py)
(figure `07_error_budget.png`, numbers in [`data/error_analysis.json`](data/error_analysis.json)).

Fixed throughout unless a figure says otherwise: the same 2D random field (`seed = 0`,
`omega = 0`), a periodic box `[-1, 1]^2`, `D = 0.3`, `dt` from `suggest_dt`, and the
bin count from `auto_bins`. This study scores each fit **only where it claims support**
and reports coverage beside every score. Read the two together: see figure 05 for what
happens when they are read apart.

> **Regenerated 2026-09-15.** The first version of these figures had two defects.
> The bin rule used the wrong exponent (below), and the sweep figure mixed a `--quick`
> and a full run from one cache file. Every number here is from the corrected run.

---

## The bin-count rule, corrected

A binned estimate is **piecewise constant**. The average over a cell is second-order
accurate at the cell *centre*, but `predict` returns that one value everywhere in the
cell. So at a query a distance `δ` from the centre the error is `∇b·δ`, which is *first
order* in the cell width. With noise-free targets, the error vs cell width has a
log-log slope of **0.97** at random points and **1.84** at cell centres.

Balancing an O(h) bias against the O(1/n_cell) variance gives

    bins ∝ N^(1/(d+2)),

not the `N^(1/(d+4))` of the original rule, which assumed second-order bias. The wrong
exponent had shown up as a constant that would not stay put. Fitted to the measured
optima it drifted 0.8 → 3.9 across 1D OU and the 2D field, and rose steadily *within*
1D. With the right exponent the same seven optima imply 0.28–0.48, and the rule uses
their geometric mean, **0.34**.

| `N·T` | measured best bins | old rule | **new rule** |
|---|---|---|---|
| 192k | 6 | 12 | **7** |
| 3.07M | 14 | 19 | **14** |
| 12.3M | 16 | 24 | **20** |

`bins='cv'` also exists: walker-grouped cross-validation, the same procedure the kernel
estimator uses. It lands within 1% of the measured optimum.

---

## The figures

### `01_setup.png` — the test
Ground truth, walkers running on it, and what comes back. Truth is drawn at the
estimator's own resolution, so the outer panels differ only by estimation error.
4.8M transitions → **nrmse 0.293** at 16 bins per axis.

### `02_recovery_vs_noise.png` — against the noise level `D`

| `D` | 0.05 | 0.15 | 0.4 | 1.0 |
|---|---|---|---|---|
| nrmse (on its support) | 0.369 | **0.251** | 0.362 | 0.785 |
| box covered | **42%** | 100% | 100% | 100% |

Both ends lose, for opposite reasons. At `D = 1` the per-increment noise `sqrt(2D/dt)`
swamps the cell average. A stable `dt` also shrinks as `D` grows, so the noise grows
faster than `sqrt(D)`. At `D = 0.05` the walkers never leave their basins, and the
estimator declines to answer for most of the box. Its score is earned only where it
did answer.

### `03_recovery_vs_walkers.png` — against `N`
`N = 16, 64, 256, 1024` at 12000 steps each: nrmse **0.648 → 0.476 → 0.325 → 0.224**,
with the bin count rising 7 → 20. With fewer walkers the estimate coarsens rather than
getting noisier.

### `04_bins_and_density.png` — the dial behind both
Left: the bias–variance U-curve in bin count at two budgets, with the rule (dashed)
landing next to the measured minimum (circle). Right: the same estimate resolved against
local sample density, against the `n^-1/2` rate of a local average.

### `05_error_vs_dimension.png` — where rung 1 ends

| `d` | nrmse at 6.1M | bins/axis | cells | coverage |
|---|---|---|---|---|
| 2 | 0.268 | 17 | 289 | 100% |
| 3 | 0.564 | 8 | 512 | 100% |
| 5 | 1.62 | 4 | 1.0k | 100% |
| 10 | 41.3 | 4 | 1.0M | **10%** |

Two things grow with `d`. The step size `suggest_dt` falls 6.5× from `d = 2` to
`d = 10`, and since a cell average's variance goes like `2D/(time in cell)`, that alone
inflates the error by **2.5×**. The measured rise is **154×**. The rest is `bins^d`.

**Read the `d = 10` row with its coverage.** The rule's floor of 4 bins per axis gives a
million cells for six million transitions. The estimator answers only for the 10% of
evaluation points in cells a walker happened to linger in, and those are the noisiest
averages of all. Scoring on the supported points is what makes 41.3 possible: a method
graded only where it chose to answer can score arbitrarily badly *or* well. The
comparison in [`../kernel/`](../kernel/) scores every method on every point, which
removes the reward for abstaining. There, cross-validated binning at `d = 10` sits at
the level of predicting zero, which is the honest version of the same conclusion.

Measured slopes `d(log nrmse)/d(log N·T)`: **−0.25, −0.25, −0.57, −0.01** at
`d = 2, 3, 5, 10`.

### `06_sweeps.png` — walkers, `D`, measurement noise
4000 steps per walker, 3 seeds, band = min/max across seeds.

- **Walkers:** 0.909 at 16 walkers → 0.326 at 1024. That's shallower than `n^-1/2`,
  because the bin count is re-chosen at every point.
- **`D`:** lowest near `D = 0.1–0.2` (0.377, 0.385); 1.93 at `D = 1.5`.
- **Measurement noise:** flat to `σ = 0.002` (0.47), then 0.66 at 0.01 and 1.56 at 0.02.
  This is a **bias in the target**, not extra variance. Noisy positions sit, on average,
  downhill of the true ones, so every method that averages `dx/dt` converges to
  `b (1 + σ²/(D dt))`. That law is verified in [`../kernel/`](../kernel/) to three
  decimals. It is quadratic in σ, and it gets worse as `dt → 0`.

### `animations/walkers.mp4`
400 walkers over the quiver field, with a live density panel. 10 s, 30 fps. Unaffected
by the corrections: it shows the data, not an estimate.

---

## Scaffolding fixed while producing this

- `metrics._local_density` allocated a `48^d` histogram (a 694 PiB `MemoryError` at
  `d = 10`). It now falls back to a k-nearest-neighbour density above a cell budget.
- Evaluation points are reused across configs sharing a field, measure and `D`. The
  dimension sweep went from ~35 min to ~10.
- Resolved hyperparameters are recorded as `param_*` columns.
- `run_sweep` returns only the configs requested, not its whole cache file.
- A result row's `n_eval` was being overwritten by the scored-point count. That count is
  now `n_scored`.
