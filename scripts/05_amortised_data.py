"""Simulate the training, validation and test fields for the amortised CNN.

    python scripts/05_amortised_data.py                  # 3000 / 200 / 100 fields
    python scripts/05_amortised_data.py --train 200 --val 20 --test 10   # smoke

Every field is a fresh random 2D field on the same torus as the study,
``[-1, 1]^2``, drawn from a *range* of settings so the network cannot learn one
configuration by heart:

=====================  =====================================================
correlation length      uniform in [0.3, 0.7]      (the study uses 0.45)
rotation ``omega``      0 for half the fields, else uniform in [-1.5, 1.5]
noise level ``D``       log-uniform in [0.05, 1.5]
walkers                 256 or 1024
steps per walker        1000, 2000 or 4000         (``dt`` from ``suggest_dt``)
=====================  =====================================================

so 256k to 4.1M transitions per field. Walkers are split into 16 groups and the
lattice statistics are stored *per group*: summing a random subset of groups
turns one simulation into datasets from 1/16 of its budget up to all of it, so
training sees budgets from ~16k to 4.1M transitions.

Stored per field (~0.9 MB, in ``~/.cache/dfi/amortised/v1/<split>/``, outside
the synced folder): per-group ``N``, ``Y``, ``S``, ``n`` on the 64x64 node lattice;
the true drift and ``rho_ss`` on the same nodes (training targets only); and the
parameters, which re-create the trajectories exactly -- the simulation is seeded
-- when a test needs the raw walkers for SFI or a kernel.

Seeds are 100000+ (train), 200000+ (val), 300000+ (test). The study's fields use
seeds 0-2, so no field the study scores was ever seen in training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M = 64
N_GROUPS = 16
SEED_BASE = {"train": 100_000, "val": 200_000, "test": 300_000}


def data_root() -> Path:
    base = os.environ.get("DFI_AMORTISED_DATA")
    return Path(base) if base else Path.home() / ".cache" / "dfi" / "amortised" / "v1"


def field_params(split: str, i: int) -> dict:
    seed = SEED_BASE[split] + i
    rng = np.random.default_rng(seed)
    return dict(
        seed=seed,
        correlation_length=float(rng.uniform(0.3, 0.7)),
        omega=float(0.0 if rng.random() < 0.5 else rng.uniform(-1.5, 1.5)),
        D=float(np.exp(rng.uniform(np.log(0.05), np.log(1.5)))),
        n_walkers=int(rng.choice([256, 1024])),
        n_steps=int(rng.choice([1000, 2000, 4000])),
    )


def simulate_field(p: dict):
    """Field and trajectories for one parameter set (deterministic)."""
    from dfi.simulate import simulate, suggest_dt
    from dfi.sweeps import SweepConfig, _burn_in_time, build_field
    cfg = SweepConfig(d=2, n_walkers=p["n_walkers"], n_steps=p["n_steps"],
                      D=p["D"], omega=p["omega"], seed=p["seed"],
                      correlation_length=p["correlation_length"])
    field = build_field(cfg)
    dt = suggest_dt(field, cfg.D)
    traj = simulate(field, n_walkers=cfg.n_walkers, n_steps=cfg.n_steps, dt=dt,
                    D=cfg.D, burn_in=_burn_in_time(field, cfg), seed=cfg.seed,
                    check=False)
    return field, traj, dt


def targets(field, D):
    nodes = field.box.grid_points((M, M))
    b = field.drift(nodes).T.reshape(2, M, M)
    step = field.U_grid.shape[0] // M
    rho = field.rho_ss_on_grid(D)[::step, ::step]
    return b.astype(np.float32), (rho / rho.mean()).astype(np.float32)


def make_one(job):
    split, i, out = job
    path = Path(out) / split / f"field_{i:05d}.npz"
    if path.exists():
        return 0.0
    from dfi.estimators.amortised import splat_groups
    t0 = time.time()
    p = field_params(split, i)
    field, traj, dt = simulate_field(p)
    groups = np.random.default_rng(p["seed"] + 1).permutation(traj.n_walkers) % N_GROUPS
    N, Y, S, n = splat_groups(traj, M, groups=groups, n_groups=N_GROUPS)
    b, rho = targets(field, p["D"])
    p["dt"] = float(dt)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, N=N, Y=Y, S=S, n=n, b=b, rho=rho, params=json.dumps(p))
    tmp.replace(path)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    out = data_root()
    for split in SEED_BASE:
        (out / split).mkdir(parents=True, exist_ok=True)
    jobs = ([("train", i, str(out)) for i in range(args.train)]
            + [("val", i, str(out)) for i in range(args.val)]
            + [("test", i, str(out)) for i in range(args.test)])
    print(f"{len(jobs)} fields -> {out}  ({args.workers} workers)")
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for secs in pool.imap_unordered(make_one, jobs, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
