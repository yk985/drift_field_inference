# Project 1 — Recovering Drift Fields from Random Walks

## One-sentence pitch

Given trajectories of particles undergoing Langevin dynamics `dx = b(x) dt + sqrt(2D) dW`,
recover the drift field `b(x)` — first with classical statistical estimators, then compare
against learned (NN / score-based) estimators, and characterize exactly where each one wins.

## The thesis of this project

The classical estimator is the protagonist. The neural network has to earn its place against
it. Every result should say not just "here is the error" but "here is *why* the error looks
like that, and here is the regime where the other method would have been the better choice."

## Core physics

Overdamped Langevin / Fokker-Planck system:

```
dx/dt = b(x) + sqrt(2D) * xi(t)        xi = white noise, <xi_i(t) xi_j(t')> = delta_ij delta(t-t')
```

Corresponding Fokker-Planck equation for the density rho(x,t):

```
d(rho)/dt = -div( b * rho ) + D * Laplacian(rho)
```

Two structurally different regimes to study, and you should be explicit in your writeup about
which one each experiment is in:

- **Gradient / equilibrium case**: `b(x) = -grad U(x)`. Stationary density is Boltzmann:
  `rho_ss(x) = exp(-U(x)/D) / Z`. This means **U(x) = -D * log(rho_ss(x))**, exactly, up to
  a constant. You can recover the potential from the stationary density alone.
- **Non-equilibrium case**: `b(x) = -grad U(x) + b_rot(x)` where `b_rot` is divergence-free
  (a rotational / solenoidal component, e.g. `b_rot = omega * (-y, x)` in 2D). The rotational
  part produces a persistent probability current at steady state but leaves `rho_ss`
  unchanged — it is **invisible in the density alone**. You need dynamics (trajectories with
  a time axis), not snapshots, to see it. This identifiability gap is the most important
  qualitative result to demonstrate in the whole project — make sure at least one experiment
  makes it visible.

## Two families of inverse problem — keep them explicitly separate

### (a) From trajectories (dynamics visible)

Sample positions at spacing `dt`. Kramers-Moyal estimator:

```
b(x) ~= E[ x(t+dt) - x(t) | x(t) = x ] / dt
D(x) ~= E[ (x(t+dt) - x(t))^2 | x(t) = x ] / (2 * dt)
```

This is a **local conditional average**: bin or kernel-weight trajectory points near `x`, look
at where they go next, divide by `dt`.

Known pathologies to reproduce and quantify (this is most of the "traditional estimator"
content of the project):

- **Discretization bias**: the estimator is only exact as `dt -> 0`. At finite `dt` there is
  a systematic bias, roughly O(dt) for the drift estimate. Sweep `dt` and show the bias curve.
- **Measurement noise on positions**: if you observe `x_obs(t) = x(t) + noise`, the increments
  `x_obs(t+dt) - x_obs(t)` pick up an extra variance term that does not vanish as `dt -> 0`
  and can swamp the true drift signal, especially at fine sampling. This is the actual
  limiting factor in real single-particle-tracking data (microscopy noise), not an
  academic add-on — treat it as a first-class parameter in your sweep, not an afterthought.
- **Finite-sample / binning bias**: sparse bins near the tails of the visited region give
  noisy or undefined estimates. Report estimator variance vs. local sample density, not just
  a single global error number.

### (b) From snapshots (density only, no time-ordering)

You are handed independent samples from `rho_ss`, no trajectory information. You can only
ever recover the **gradient part** of the drift, via `U = -D * log(rho_ss)` (kernel density
estimate the density, then take the log and differentiate, or fit directly). Demonstrate
explicitly that this fails to recover `b_rot` even in a system where `b_rot` is large — this
is the identifiability result from above made concrete.

## Estimator ladder (build in this order, each is a real baseline for the next)

1. **Binned Kramers-Moyal.** Divide space into bins, average increments per bin. Cheapest,
   noisiest, good for validating the pipeline against a known analytic `b(x)` (start with a
   linear / Ornstein-Uhlenbeck drift where everything is closed-form).
2. **Kernel regression (local-linear / Nadaraya-Watson).** Smooths across bins, handles
   uneven sample density better, still fully classical and interpretable.
3. **Basis projection (Stochastic Force Inference).** Project `b(x)` onto a finite basis
   (polynomials up to some degree, or Fourier modes) and solve a **linear least-squares**
   problem for the coefficients using the same Kramers-Moyal increments as the regression
   target. Reference: Frishman & Ronceray, "Learning force fields from stochastic
   trajectories," Physical Review X 10, 021009 (2020). This is fast, has closed-form
   coefficient estimates, and comes with an explicit theoretical error / information bound —
   implement that bound and plot empirical error against it. This is likely your strongest
   baseline in low dimension; expect it to beat the neural net there, and say so.
4. **Neural network regression.** MLP or small ResNet taking `x` (and optionally local
   trajectory statistics) to `b(x)`, trained by regressing predicted drift against the
   Kramers-Moyal target, or directly by a trajectory likelihood loss. This is where you scale
   to moderate dimension (d = 4-10) and where basis projection starts to break down
   (curse of dimensionality in the number of basis functions).
5. **Score-based / diffusion approach.** For the equilibrium case, note the identity:

   ```
   grad log(rho_ss(x)) = b(x) / D          (equilibrium, gradient drift only)
   ```

   A denoising score matching model trained on snapshots from `rho_ss` is literally learning
   an estimate of `b/D`. This is the "diffusion model" hook for this project: you are not
   using a diffusion model as a black box generator, you are using score matching because the
   score IS the physical quantity you want. Compare this score-based estimate against your
   Kramers-Moyal / basis-projection estimates on gradient-only systems, and show it fails (as
   it must, by the identifiability argument) on systems with a rotational component.

## Suggested amortized / "fancy DL" extension (optional, do after 1-5 work)

Instead of a diffusion model that outputs a point estimate of `b(x)`, build a **set-encoder +
conditional diffusion model** that outputs a *distribution over plausible drift fields* given
a batch of trajectories:

- Encode each trajectory (or the whole batch) into a fixed-size summary with a
  permutation-invariant set encoder (DeepSets-style mean/attention pooling over per-trajectory
  features).
- Condition a diffusion model on that summary; sample multiple drift-field hypotheses.
- Evaluate **calibration**: does the ensemble's spread actually track the true estimation
  error as you vary N (number of walkers), T (trajectory length), noise level? Run coverage
  tests (e.g. is the true field inside the model's 90% credible band 90% of the time?).

This "point estimate vs. calibrated posterior" framing is a much stronger deliverable than
"I also trained a diffusion model."

## Systems to test on (increasing difficulty)

1. **1D Ornstein-Uhlenbeck**: `b(x) = -k*x`. Fully closed-form stationary density and
   autocorrelation. Use this to unit-test your entire pipeline before trusting anything else.
2. **1D double well**: `b(x) = -grad U(x)`, `U(x) = a*x^4 - b*x^2`. Metastable, non-Gaussian
   stationary density, still gradient-only.
3. **2D rotational + gradient mix**: `b(x,y) = -grad U(x,y) + omega * (-y, x)`. This is your
   identifiability-gap demonstrator — sweep `omega` from 0 upward and show snapshot-only
   recovery degrades while trajectory-based recovery does not.
4. **Moderate-d (4-10) system**: e.g. a chain of coupled OU processes, or a random quadratic
   potential. This is where basis projection should start to lose to the NN — find and report
   the crossover.

## The sweep and the headline figure

Vary, independently where possible:

- **N**: number of independent walkers (or one walker run N times)
- **T / dt**: trajectory length and sampling interval (equivalently, total transitions
  observed)
- **d**: dimension of the system
- **noise**: measurement noise added to observed positions
- **omega**: rotational strength (for the identifiability experiment)

Produce a **phase diagram**: estimation error vs. (N*T, d), for each method, with a marked
crossover line where the NN starts beating basis projection. This single figure should be
able to carry most of the "what did you find" part of an interview conversation.

## Extra credit: non-equilibrium thermodynamics tie-in

For the rotational system, estimate the **entropy production rate**:

```
sigma = <b_rot . b_rot> / D     (schematically; look up exact form for your convention)
```

from the estimated drift field, and compare against the true value. Optionally check the
estimate against a thermodynamic uncertainty relation bound relating dissipation to the
precision of any current-like observable. This connects directly to the Jarzynski /
stochastic thermodynamics material in the optimal-protocol project — same mathematical
language, different physical question.

## Deliverables checklist

- [ ] Clean SDE simulator (Euler-Maruyama) for 1D/2D/d-D systems with pluggable `b(x)`
- [ ] Kramers-Moyal estimator, validated against OU closed form
- [ ] Kernel regression estimator
- [ ] Basis projection (Stochastic Force Inference) estimator with theoretical error bound
- [ ] NN regression estimator
- [ ] Score-matching estimator on snapshots, equilibrium systems only
- [ ] Identifiability experiment: gradient+rotational system, snapshot vs trajectory recovery
- [ ] Phase diagram figure: error vs (N*T, d) per method, with crossover
- [ ] (Optional) amortized diffusion posterior + calibration/coverage plots
- [ ] (Optional) entropy production estimate from recovered rotational field
- [ ] One-paragraph write-up per system stating: pitch, headline figure, and the negative
      result (where the fancy method lost, and why)

## Key references

- Frishman, A. & Ronceray, P. "Learning force fields from stochastic trajectories." Physical
  Review X 10, 021009 (2020). [Stochastic Force Inference / basis projection method]
- Risken, H. "The Fokker-Planck Equation" — standard reference for Kramers-Moyal expansion
- Song, Y. et al. "Score-Based Generative Modeling through Stochastic Differential
  Equations" (2021) — background on score matching / SDEs, useful for the diffusion extension
