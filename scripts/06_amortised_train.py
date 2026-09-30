"""Train the amortised CNN on the simulated fields; save the weights.

    python scripts/05_amortised_data.py        # first: the fields
    python scripts/06_amortised_train.py       # ~1 h on an RTX 2060
    python scripts/06_amortised_train.py --steps 300 --out models/smoke.pt   # smoke

Each training sample is built on the fly:

1. pick a training field;
2. pick a budget -- ``k`` of its 16 walker groups, ``k`` log-uniform in 1..16 --
   and sum those groups' lattice statistics;
3. turn them into input channels (``dfi.estimators.amortised.make_features``);
4. apply one of the 8 symmetries of the square lattice to input, target and
   weights together (the field distribution is invariant under all 8: reflection
   sends ``omega`` to ``-omega``, and ``omega`` is drawn symmetrically).

So a field is seen at a different budget, a different subset of its walkers and
a different orientation every time. Loss: density-weighted squared relative
error on the 64x64 nodes (module docstring).

Validation, every ``--val-every`` steps, on 200 held-out fields at three fixed
budgets (1, 4 and all 16 groups), without augmentation, scored the way the study
scores: nrmse under ``rho_ss`` on the nodes. The weights kept are an exponential
moving average of the trained ones, and the checkpoint written to
``models/amortised_cnn_2d.pt`` is the one with the best mean validation nrmse.
The full history goes to ``outputs/cnn_amortised/data/train_history.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dfi.estimators.amortised import (TAU, UNet, default_checkpoint, dihedral,
                                      make_features)

DATA_MODULE = __import__("05_amortised_data")
VAL_BUDGETS = (1, 4, 16)


def load_split(split: str, limit: int | None = None):
    files = sorted((DATA_MODULE.data_root() / split).glob("field_*.npz"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no {split} fields in {DATA_MODULE.data_root()}; "
                         f"run scripts/05_amortised_data.py first")
    keys = ("N", "Y", "S", "n", "b", "rho")
    buf = {k: [] for k in keys}
    params = []
    for f in files:
        z = np.load(f)
        for k in keys:
            buf[k].append(z[k])
        params.append(json.loads(str(z["params"])))
    out = {k: torch.from_numpy(np.stack(v)) for k, v in buf.items()}
    out["S"] = out["S"].double()
    out["n"] = out["n"].double()
    out["dt"] = torch.tensor([p["dt"] for p in params], dtype=torch.float64)
    out["params"] = params
    return out


def batch_from(data, idx, masks, dev):
    """Sum the selected groups and build (features, target b, weights)."""
    where = data["N"].device
    idx, masks = idx.to(where), masks.to(where)
    m = masks.float()                                          # (B, G)
    N = torch.einsum("bg,bgij->bij", m, data["N"][idx])
    Y = torch.einsum("bg,bgcij->bcij", m, data["Y"][idx])
    S = (masks.double() * data["S"][idx]).sum(1)
    n = (masks.double() * data["n"][idx]).sum(1)
    feats = make_features(N.to(dev), Y.to(dev), S.to(dev), n.to(dev),
                          data["dt"][idx].to(dev), tau=TAU)
    return feats, data["b"][idx].to(dev), data["rho"][idx].to(dev)


def to_device(data, dev):
    """Move the tensors to the GPU when they fit: summing groups there is the
    difference between ~4 and ~15 steps per second."""
    if dev.type != "cuda":
        return data
    need = sum(v.numel() * v.element_size() for v in data.values()
               if torch.is_tensor(v))
    free, _ = torch.cuda.mem_get_info()
    if need > free - 2.5e9:
        print(f"  (keeping {need / 1e9:.1f} GB of data on the CPU)")
        return data
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data.items()}


def weighted_rel_error(pred, b, w):
    num = (w[:, None] * (pred - b) ** 2).sum((1, 2, 3))
    den = (w[:, None] * b ** 2).sum((1, 2, 3))
    return num / den


@torch.no_grad()
def validate(net, data, dev, batch=50):
    """Mean nrmse under rho_ss per budget; also the shrunk-mean input alone."""
    F = data["N"].shape[0]
    res = {}
    for k in VAL_BUDGETS:
        errs, raw = [], []
        for s in range(0, F, batch):
            idx = torch.arange(s, min(F, s + batch))
            masks = torch.zeros(len(idx), data["N"].shape[1], dtype=torch.bool)
            masks[:, :k] = True
            feats, b, rho = batch_from(data, idx, masks, dev)
            pred = net(feats)
            errs.append(weighted_rel_error(pred, b, rho).sqrt().cpu())
            raw.append(weighted_rel_error(feats[:, 2:4], b, rho).sqrt().cpu())
        res[k] = (float(torch.cat(errs).mean()), float(torch.cat(raw).mean()))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 96, 192, 256])
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--out", type=Path, default=default_checkpoint())
    ap.add_argument("--history", type=Path,
                    default=ROOT / "outputs" / "cnn_amortised" / "data" / "train_history.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    train = load_split("train", args.limit_train)
    val = load_split("val")
    train, val = to_device(train, dev), to_device(val, dev)
    F, G = train["N"].shape[:2]
    print(f"loaded {F} training and {val['N'].shape[0]} validation fields "
          f"[{time.time() - t0:.0f}s]", flush=True)

    arch = dict(in_ch=5, out_ch=2, widths=tuple(args.widths))
    net = UNet(**arch).to(dev)
    ema = copy.deepcopy(net).eval()
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    warm = 500

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        frac = (step - warm) / max(1, args.steps - warm)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * frac))

    history = {"step": [], "train_loss": [], "val": [], "lr": [], "seconds": []}
    best = (math.inf, None)
    run_loss, run_n = 0.0, 0
    vec = [(0, 1), (2, 3), (5, 6)]          # features z, m; then target b
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.from_numpy(rng.integers(0, F, args.batch))
        k = np.clip(np.rint(2.0 ** rng.uniform(0, math.log2(G), args.batch)), 1, G)
        masks = torch.zeros(args.batch, G, dtype=torch.bool)
        for i, ki in enumerate(k.astype(int)):
            masks[i, rng.permutation(G)[:ki]] = True
        feats, b, rho = batch_from(train, idx, masks, dev)
        stack = torch.cat([feats, b, rho[:, None]], dim=1)
        sym = rng.integers(0, 8, args.batch)
        parts = []
        for s_ in range(8):
            sel = np.flatnonzero(sym == s_)
            if len(sel):
                parts.append(dihedral(stack[sel], int(s_), vec))
        stack = torch.cat(parts)
        feats, b, rho = stack[:, :5], stack[:, 5:7], stack[:, 7]
        w = 0.8 * rho + 0.2
        loss = weighted_rel_error(net(feats), b, w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - args.ema)
        run_loss += float(loss)
        run_n += 1

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            res = validate(ema, val, dev)
            mean_nrmse = float(np.mean([v[0] for v in res.values()]))
            history["step"].append(step + 1)
            history["train_loss"].append(run_loss / run_n)
            history["val"].append({str(k): v for k, v in res.items()})
            history["lr"].append(lr_at(step))
            history["seconds"].append(time.time() - t0)
            run_loss, run_n = 0.0, 0
            flag = ""
            if mean_nrmse < best[0]:
                best = (mean_nrmse, step + 1)
                flag = "  *"
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": ema.state_dict(), "arch": arch,
                            "grid": int(feats.shape[-1]), "tau": TAU,
                            "n_groups": int(G), "data": "amortised/v1",
                            "train_fields": int(F), "step": step + 1,
                            "val": {str(k): v for k, v in res.items()},
                            "n_params": n_params, "args": {k: str(v) for k, v in vars(args).items()}},
                           args.out)
            print(f"step {step + 1:6d}  loss {history['train_loss'][-1]:.4f}  val nrmse "
                  + "  ".join(f"k={k}: {v[0]:.3f} (input {v[1]:.2f})" for k, v in res.items())
                  + f"  [{time.time() - t0:.0f}s]{flag}", flush=True)
            args.history.parent.mkdir(parents=True, exist_ok=True)
            args.history.write_text(json.dumps(
                {"history": history, "best_step": best[1], "best_val_nrmse": best[0],
                 "n_params": n_params, "arch": {k: list(v) if isinstance(v, tuple) else v
                                                for k, v in arch.items()},
                 "train_fields": int(F)}, indent=1))
    last = args.out.with_name(args.out.stem + "_last.pt")
    torch.save({"state_dict": ema.state_dict(), "arch": arch, "grid": int(feats.shape[-1]),
                "tau": TAU, "n_groups": int(G), "step": args.steps}, last)
    print(f"best mean val nrmse {best[0]:.4f} at step {best[1]}; weights {args.out}")


if __name__ == "__main__":
    main()
