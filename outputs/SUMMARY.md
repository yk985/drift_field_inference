# Drift field inference — what the whole ladder found

Recovering a drift field `b(x)` from Langevin trajectories `dx = b dt + √(2D) dW`, with six
estimators on the *same* simulated walkers. Each has its own folder of figures; this file
is the comparison, what the numbers mean, and what is worth testing next.

## The methods

| rung | method | what it consumes | what it assumes |
|---|---|---|---|
| 1 | binned Kramers–Moyal | increments | local smoothness (piecewise constant) |
| 2 | kernel regression (NW, local-linear) | increments | local smoothness, one bandwidth |
| 3 | basis projection (SFI) | increments | the field lies in a Fourier basis |
| 4 | neural network (MLP, CNN) | increments | smoothness implied by architecture + early stopping |
| 4b | amortised CNN | increments, binned | **what fields from this generator look like** |
| 5 | KDE score, denoising score matching | positions only (+ known `D`) | **equilibrium** (gradient drift) |

Every method regresses the same target for rungs 1–4b: `Δx/Δt = b(x) + noise`, where the
noise is ~1000× the drift's power in these settings. Rung 5 instead estimates
`∇log ρ_ss = b/D`, valid only at equilibrium.

## Headline numbers (2D random field, identical walkers)

| | 4.8M transitions | 16 walkers | 1024 walkers | `D` = 1.0 | σ_obs = 0.02 | `d` = 3 | `d` = 5 |
|---|---|---|---|---|---|---|---|
| binned KM | 0.293 | 0.648 | 0.224 | 0.785 | 1.56 | 0.564 | 1.62 |
| LL kernel | 0.126 | 0.358 | 0.088 | 0.280 | 1.25 | 0.261 | 0.925 |
| SFI | 0.097 | 0.392 | 0.053 | 0.211 | 1.43 | 0.195 | 0.946 |
| MLP | 0.100 | 0.340 | 0.080 | 0.319 | 1.38 | 0.223 | 0.927 |
| CNN (per dataset) | 0.115 | 0.380 | 0.068 | 0.333 | 1.44 | 0.224 | — |
| amortised CNN | **0.060** | **0.257** | **0.043** | **0.134** | 0.63 | — | — |
| KDE score (snapshots) | 0.130 | 0.132 | 0.130 | 0.843 | 0.14 | 0.231 | 0.983 |
| DSM (snapshots) | 0.072 | 0.194 | 0.072 | 0.164 | **0.08** | 0.142 | **0.625** |

Snapshot rows are a different kind of data on the same systems: independent positions
matched in count (capped at 500k) plus the known `D`, and no time ordering.

## What the results say

**1. More smoothing structure beats more flexibility, until the structure is wrong.**
Ranked at 4.8M transitions, the order is binned (0.293) → kernel (0.126) → network (0.100)
→ SFI (0.097). Binning is first-order biased because a cell answers with one value;
local-linear removes that; SFI removes it globally because the fields *are* sums of Fourier
modes. The advantage disappears the moment the assumption does: on fields built from other
generators (sparse wells, cellular patterns), SFI drops behind the local kernel.

**2. A network is a competent middle, not a breakthrough.** The MLP sits between the kernel
and SFI nearly everywhere, and wins where data is scarcest (0.340 at 16 walkers). Its
smoothing dial is the number of training steps, chosen on held-out walkers. Its one
structural weakness is that this choice is noisy: with lots of data the held-out curve is
flat, and stopping there costs ~5% of the error (0.082 against 0.076 at the best step).

**3. A CNN fitted to one dataset is the wrong tool; the same CNN trained across fields is
the best tool.** Per dataset the convolutional net has one image to learn from and loses to
the MLP almost everywhere. Trained once across 3000 simulated fields, it wins nearly every
comparison here — and the reason is prior knowledge, not a better estimator.

**4. That prior is measurable, and so is its cost.** The amortised CNN's equivalent
smoothing kernel keeps a fixed width as data grows and instead raises its gain from 0.60 to
0.93 — the signature of a Wiener filter matched to the field statistics, not of a bandwidth.
On fields from other generators it loses on five of six, by up to 1.9×.

**5. Nothing escapes dimension by being flexible.** At `d = 5`, with 6.1M transitions,
every trajectory method is near 0.93. It is not capacity (the MLP fits the exact 5D drift
to 0.04) but information: each increment carries noise 48× the drift, and the field has
~1700 independent correlation volumes. With 100× the data both MLP and SFI reach ~0.2.

**6. At equilibrium, positions beat increments.** This is the result I did not expect.
Δx/Δt is dominated by noise `√(2D/Δt)`; a position is not. With independent snapshots,
denoising score matching reaches 0.072 where SFI on the same-sized set of increments gets
0.097, and **at `d = 5` it is the only method that recovers anything (0.625 against ~0.93)**.
Even using the very same walkers' positions, it beats every trajectory method at 16 and 64
walkers.

**7. Measurement noise separates the two families completely.** Position noise biases the
increment target to `b(1 + σ²/(DΔt))` — at σ_obs = 0.02 that is 2.3×, and every trajectory
method scores above 1.2. On snapshots the same noise only blurs the density slightly: the
KDE score goes 0.134 → 0.140. If positions are measured badly and the system is at
equilibrium, use snapshots.

**8. But snapshots are blind to circulation, exactly and measurably.** Adding a rotational
part `ω A∇U` leaves `ρ_ss` bit-identical, so both snapshot estimators return *bit-identical*
predictions for `ω` = 0 … 4 (maximum difference 0.0). Their error against the rotational
part is 1.00 at every `ω`, while trajectory methods track it. This is information, not
method quality.

**9. The two families together measure dissipation.** Entropy production
`σ = ⟨|b − D∇log ρ|²⟩/D` from one dataset — drift from the increments, score from the same
walkers' positions — lands within 2% of the exact value from `ω` ≥ 1, with a detection floor
of ~0.1 at `ω` = 0 set by squared estimation error (halved by subtracting SFI's own
predicted error).

**10. Every method that tunes itself can be fooled by its own criterion.** Cross-validation
picked a 1M-cell grid that answered for 10% of the box (rung 1); early stopping quit 30%
early on a flat curve (rung 4); the KDE's bandwidth search ran to the bottom of its grid on
correlated positions (1.06 against 0.35 at a fixed bandwidth) and to the top at `D` = 1
(0.84). Each failure is visible only against the truth — which is why every rung reports
what it chose, not just what it scored.

## Honest limitations

- **Comparisons across families are not apples to apples.** The amortised CNN knows the
  field distribution; the snapshot methods get independent samples and the true `D`. Both
  advantages are stated wherever their numbers appear, but they are advantages of the
  *setting*, not of the estimator.
- **One field family, one box.** Almost every number comes from Gaussian-spectrum random
  fields on `[-1,1]²`. The other-generator test (`cnn_amortised/14`) is the only breadth
  check, and it is 2D only.
- **`D` is known** to the snapshot methods, and `dt` is chosen by the simulator's stability
  rule rather than by an experiment.
- **The KDE and DSM receive positions with no walker labels**, so their cross-validation
  cannot group by walker the way the trajectory rungs do. That is a real bug in the
  correlated-positions setting, not just a caveat.
- **Rung 4b is 2D-only** by construction, and its output grid (64²) caps its accuracy on
  nearly noise-free data.

## What is worth testing next

**Cheap, and likely to pay off**
1. **Hybrid estimator.** Fit the gradient part from positions (rung 5) and only the
   rotational remainder from increments. At 16 walkers that should beat both families;
   the pieces already exist in `08_score_study.py`.
2. **Walker-grouped CV for the snapshot methods.** Pass walker ids and split by walker.
   This should fix the 1.06 failure directly.
3. **Estimate `D` from the data** in rung 5 (quadratic variation), and report how the
   drift scale degrades — it is the one input the snapshot methods still get for free.
4. **Train the amortised CNN on a mix of generators** (wells, spectra, Perlin) and re-run
   `14_other_generators`. It should trade in-family accuracy for robustness; how much is
   the interesting number.

**More work, more interesting**
5. **The calibrated-posterior version** the guide suggests: a conditional diffusion model
   over drift fields, giving a *distribution* rather than a point estimate, with coverage
   checked against the true field. The infrastructure (data generation, scoring, the
   entropy-production check) is already here.
6. **Amortised inference in 3D**, to test whether the prior's benefit survives when the
   lattice gets expensive — the last dimension where a grid CNN is possible.
7. **Non-equilibrium beyond `ω A∇U`.** All rotational fields here share one structure, and
   `ρ_ss` stays exactly Boltzmann. A field where `ρ_ss` is *not* Boltzmann would test
   whether the snapshot methods fail gracefully or confidently.
8. **Time-resolution sweep.** `Δt` is fixed by the integrator here; sweeping it would show
   the O(Δt) discretisation bias trading against the `1/Δt` noise, and where each method's
   optimum sits.
9. **Real data.** Single-particle tracking or an ecological time series would expose what
   simulation hides: unknown `D`, non-stationary boundaries, heterogeneous noise.

## Where everything lives

| folder | contents |
|---|---|
| [`binned_km/`](binned_km/), [`kernel_nw/`](kernel_nw/), [`kernel_ll/`](kernel_ll/), [`sfi/`](sfi/) | rungs 1–3, six figures each |
| [`kernel/`](kernel/) | how the kernel and its bandwidth work, rung 2 vs rung 1 |
| [`nn_mlp/`](nn_mlp/), [`nn_cnn/`](nn_cnn/) | rung 4, with the training-statistics figure and the dimension argument |
| [`cnn_amortised/`](cnn_amortised/) | rung 4b: training, held-out fields, out-of-distribution, other generators, learned smoothing |
| [`score/`](score/), [`score_kde/`](score_kde/), [`score_dsm/`](score_dsm/) | rung 5: identifiability, entropy production, snapshots vs positions |
| [`local_study/`](local_study/) | the side-by-side tables |
| [`../models/`](../models/) | the amortised CNN's weights |
