# Neural network, CNN — on its own

The same six figures as the other estimators, from the same script, on the **same walkers**:

```bash
python scripts/03_local_estimator.py --estimators cnn
```

A convolutional decoder turns a learned coarse latent into the drift on a 16², 32² or 64²
grid, trained from the trajectories alone. It is fitted per dataset, like every other
rung. The loss over all transitions is computed exactly from lattice statistics, so
every step is full-batch. The step count and the grid are both chosen on held-out
walkers. It runs only for `d ≤ 3`.

| figure | what it shows |
|---|---|
| `01_setup.png` | the test: truth, walkers running, the recovered field (nrmse 0.115) |
| `02_recovery_vs_noise.png` | `D = 0.05 … 1.0`: 0.031, 0.057, 0.132, 0.333 |
| `03_recovery_vs_walkers.png` | `N = 16 … 1024`: 0.380, 0.208, 0.166, 0.068 |
| `04_training.png` | training statistics: raw loss, gain over predicting zero per grid, error against the truth |
| `05_error_vs_dimension.png` | `d = 2, 3` against budget (0.088, 0.224 at 6.1M), and held-out gain during training |
| `06_sweeps.png` | walkers (0.759 → 0.132), `D`, measurement noise; 3 seeds |

How the networks learn, the training statistics explained, every table against the other
rungs, and the question of dimensions: [`../nn_mlp/README.md`](../nn_mlp/README.md).

In short, the CNN trails the MLP nearly everywhere and is unstable with little data. The
grid choice flips between seeds, and 128 walkers scored worse than 64. The exception is the
largest budget, where it beat the MLP (0.068 vs 0.080 at 1024 walkers). A per-dataset fit
gives convolution one image to share its weights across. Its natural use is a network
trained across many fields.
