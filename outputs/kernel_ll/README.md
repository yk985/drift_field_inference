# Kernel regression, local-linear — on its own

The same six figures as [`../binned_km/`](../binned_km/), from the same script, on the
**same walkers** (verified; see [`../local_study/`](../local_study/)):

```bash
python scripts/03_local_estimator.py --estimators ll
```

| figure | what it shows |
|---|---|
| `01_setup.png` | the test: truth, walkers running, the recovered field |
| `02_recovery_vs_noise.png` | recovered field and error map at `D = 0.05, 0.15, 0.4, 1.0` |
| `03_recovery_vs_walkers.png` | the same at `N = 16, 64, 256, 1024` |
| `04_bandwidth_and_density.png` | true error across bandwidths (CV's choice dashed, Silverman dotted, optimum circled); error against local data density |
| `05_error_vs_dimension.png` | `d = 2, 3, 5, 10` against budget, and the bandwidth CV chose |
| `06_sweeps.png` | walkers, `D`, measurement noise; 3 seeds |

Numbers for all three estimators side by side:
[`../local_study/README.md`](../local_study/README.md). How the kernel and its bandwidth
work: [`../kernel/README.md`](../kernel/README.md). Companion variant:
[`../kernel_nw/`](../kernel_nw/).
