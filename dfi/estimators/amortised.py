"""Rung 4b: an amortised CNN -- trained once across many fields, then applied.

Rungs 1-4 fit each dataset from scratch. This estimator is trained *once*, on
thousands of simulated fields, to map a dataset to its drift; given new
trajectories it runs one forward pass and never trains again. The true drift is
used only for the simulated training fields, never for the dataset being
estimated.

What the network sees
---------------------
A dataset is reduced to lattice statistics on the ``M x M`` nodes of the
periodic box, with bilinear weights ``w_ic`` (the same splatting as
``NeuralDrift(arch='cnn')``):

    N_c = sum_i w_ic          Y_c = sum_i w_ic dx_i/dt

plus ``sum |dx|^2`` for the noise level. With ``sigma^2 = 2D/dt`` estimated from
the quadratic variation (``D`` is not taken on trust), each node has an
unbiased but noisy mean ``Y_c / N_c`` of noise variance ``~ sigma^2 / N_c``. The
input channels are three views of that, each O(1) whatever the budget or noise:

``z``   ``Y_c / (sigma sqrt(N_c))`` -- a unit-noise z-score (2 channels)
``m``   ``Y_c / (N_c + lambda)``, ``lambda = sigma^2 / tau^2`` -- the posterior mean
        of ``b_c`` under a ``N(0, tau^2)`` prior, shrunk to 0 where data is thin
``p``   ``N_c / (N_c + lambda)`` -- how much of ``m`` is data and how much prior

``tau = 1`` is the fields' typical drift, so outputs are in physical units.
Nothing else: no positions, no box coordinates -- the network is translation
equivariant on the torus, as the field distribution is.

What it outputs, and how it is trained
--------------------------------------
The drift on the same ``M x M`` nodes; bilinear interpolation between them. The
training loss for one sample is the density-weighted relative error

.. math::  \\frac{\\sum_c w_c |\\hat b_c - b_c|^2}{\\sum_c w_c |b_c|^2},
           \\qquad w_c = 0.8\\, \\rho_{ss}(c)/\\overline{\\rho_{ss}} + 0.2 ,

i.e. the squared nrmse the study scores, mostly under ``rho_ss`` but with a
floor so the network still has to say something where walkers rarely go.

See ``scripts/05_amortised_data.py`` (simulation), ``06_amortised_train.py``
(training; weights saved to ``models/``) and ``07_amortised_study.py``.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .base import TrajectoryEstimator, register

__all__ = ["AmortizedDrift", "splat_groups", "make_features", "UNet",
           "default_checkpoint"]

TAU = 1.0


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "amortised_cnn_2d.pt"


# --------------------------------------------------------------------------
# data -> lattice statistics -> features
# --------------------------------------------------------------------------

def splat_groups(traj, M: int, groups=None, n_groups: int = 1):
    """Bilinear lattice statistics per walker group (numpy, 2D periodic).

    Returns ``N (G, M, M)``, ``Y (G, 2, M, M)``, ``S (G,)`` = sum of |dx|^2 and
    ``n (G,)`` transitions. ``groups`` assigns each walker a group id.
    """
    box = traj.box
    d = traj.d
    if d != 2 or not box.periodic:
        raise ValueError("the amortised model is built for 2D periodic boxes")
    if groups is None:
        groups = np.zeros(traj.n_walkers, dtype=np.int64)
    cells = M * M
    N = np.zeros(n_groups * cells)
    Y = np.zeros((2, n_groups * cells))
    S = np.zeros(n_groups)
    n = np.zeros(n_groups, dtype=np.int64)
    h = np.asarray(box.length, float) / M
    lo = np.asarray(box.lo, float)
    T = traj.n_frames - 1
    per = max(1, 2_000_000 // T)
    for w0 in range(0, traj.n_walkers, per):
        xs = traj.x[w0:w0 + per].astype(np.float64)
        g = np.repeat(np.asarray(groups[w0:w0 + per]), T)
        x0 = xs[:, :-1].reshape(-1, d)
        dx = box.displacement(xs[:, :-1], xs[:, 1:]).reshape(-1, d)
        y = dx / traj.dt
        S += np.bincount(g, weights=np.sum(dx * dx, axis=1), minlength=n_groups)
        n += np.bincount(g, minlength=n_groups)
        u = (x0 - lo) / h
        i0 = np.floor(u)
        t = u - i0
        i0 = i0.astype(np.int64)
        base = g * cells
        for e in itertools.product((0, 1), repeat=2):
            w = ((t[:, 0] if e[0] else 1 - t[:, 0])
                 * (t[:, 1] if e[1] else 1 - t[:, 1]))
            flat = base + ((i0[:, 0] + e[0]) % M) * M + (i0[:, 1] + e[1]) % M
            N += np.bincount(flat, weights=w, minlength=n_groups * cells)
            for k in range(2):
                Y[k] += np.bincount(flat, weights=w * y[:, k],
                                    minlength=n_groups * cells)
    return (N.reshape(n_groups, M, M).astype(np.float32),
            Y.reshape(2, n_groups, M, M).transpose(1, 0, 2, 3).astype(np.float32),
            S, n)


def make_features(N, Y, S, n, dt, tau: float = TAU):
    """Input channels from (possibly summed) statistics; torch, batched.

    ``N (B, M, M)``, ``Y (B, 2, M, M)``, ``S (B,)``, ``n (B,)``, ``dt (B,)`` ->
    ``(B, 5, M, M)``: z-score (2), shrunk mean (2), data weight (1).
    """
    import torch
    d = 2
    sigma2 = (S / (d * n * dt ** 2)).to(N.dtype)[:, None, None]     # 2D/dt
    lam = sigma2 / tau ** 2
    Nc = N.clamp_min(0.0)
    z = torch.where(Nc[:, None] > 1e-6,
                    Y / (torch.sqrt(sigma2 * Nc)[:, None] + 1e-12),
                    torch.zeros_like(Y))
    m = Y / (Nc + lam)[:, None]
    p = Nc / (Nc + lam)
    return torch.cat([z, m, p[:, None]], dim=1)


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

def UNet(in_ch: int = 5, out_ch: int = 2, widths=(48, 96, 192, 256)):
    """Periodic U-Net: circular padding everywhere, so the torus has no edges.

    No normalisation layers -- they would rescale each sample, and the output
    is a physical drift whose amplitude the network must preserve.
    """
    import torch
    nn, F = torch.nn, torch.nn.functional

    def conv(i, o):
        return nn.Conv2d(i, o, 3, padding=1, padding_mode="circular")

    class Res(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.a, self.b = conv(i, o), conv(o, o)
            self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.b(F.gelu(self.a(F.gelu(x))))

    def up(x):
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x[..., 2:-2, 2:-2]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = conv(in_ch, widths[0])
            self.down = nn.ModuleList([Res(widths[k], widths[k + 1])
                                       for k in range(len(widths) - 1)])
            self.mid = Res(widths[-1], widths[-1])
            self.up = nn.ModuleList([Res(widths[k + 1] + widths[k], widths[k])
                                     for k in reversed(range(len(widths) - 1))])
            self.enc0 = Res(widths[0], widths[0])
            self.out = nn.Conv2d(widths[0], out_ch, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, x):
            x = self.enc0(self.inc(x))
            skips = []
            for blk in self.down:
                skips.append(x)
                x = blk(F.avg_pool2d(x, 2))
            x = self.mid(x)
            for blk in self.up:
                x = blk(torch.cat([up(x), skips.pop()], dim=1))
            return self.out(F.gelu(x))

    return Net()


def dihedral(x, k: int, vector_channels):
    """Apply one of the 8 symmetries of the square lattice to ``(B, C, M, M)``.

    Spatial maps act on node indices (a reflection sends node ``j`` to
    ``-j mod M``, not to ``M-1-j``, so nodes stay nodes), and the vector
    channels transform with them: a transpose swaps the two components, a
    reflection flips the sign of one. ``vector_channels`` lists ``(cx, cy)``
    index pairs.
    """
    import torch
    out = x
    if k & 1:                                   # transpose
        out = out.transpose(-1, -2).clone()
        for cx, cy in vector_channels:
            out[:, [cx, cy]] = out[:, [cy, cx]]
    if k & 2:                                   # reflect axis 0
        out = torch.roll(torch.flip(out, dims=[-2]), 1, dims=-2).clone()
        for cx, _ in vector_channels:
            out[:, cx] = -out[:, cx]
    if k & 4:                                   # reflect axis 1
        out = torch.roll(torch.flip(out, dims=[-1]), 1, dims=-1).clone()
        for _, cy in vector_channels:
            out[:, cy] = -out[:, cy]
    return out


def grid_interpolate(grid, x, box):
    """Bilinear, periodic: ``grid (2, M, M)`` at points ``x (n, 2)``."""
    M = grid.shape[-1]
    h = np.asarray(box.length, float) / M
    u = (np.asarray(x, float) - np.asarray(box.lo, float)) / h
    i0 = np.floor(u).astype(np.int64)
    t = u - i0
    out = np.zeros((len(u), grid.shape[0]))
    for e in itertools.product((0, 1), repeat=2):
        w = ((t[:, 0] if e[0] else 1 - t[:, 0])
             * (t[:, 1] if e[1] else 1 - t[:, 1]))
        out += w[:, None] * grid[:, (i0[:, 0] + e[0]) % M, (i0[:, 1] + e[1]) % M].T
    return out


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

@register
class AmortizedDrift(TrajectoryEstimator):
    """Pretrained CNN: trajectories in, drift out, no training at fit time.

    ``fit`` computes the lattice statistics and runs one forward pass. With
    ``tta=True`` (default) the prediction is averaged over the 8 lattice
    symmetries -- the network was trained with them as augmentation, and the
    average is exactly equivariant.
    """

    key = "cnn_amortised"
    label = "Amortised CNN"
    color_slot = 4

    def __init__(self, checkpoint=None, device=None, tta: bool = True, **params):
        super().__init__(**params)
        self.checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()
        self.device = device
        self.tta = bool(tta)
        self.net = None
        self.grid_ = None
        self.box = None
        self._meta = None

    def _load(self):
        import torch
        if not self.checkpoint.exists():
            raise NotImplementedError(
                f"no trained weights at {self.checkpoint}; run "
                f"scripts/06_amortised_train.py")
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available()
                                           else "cpu"))
        ck = torch.load(self.checkpoint, map_location=dev, weights_only=False)
        net = UNet(**ck["arch"]).to(dev)
        net.load_state_dict(ck["state_dict"])
        self.net, self._meta, self._dev = net.eval(), ck, dev

    def fit(self, traj):
        import torch
        if traj.d != 2 or not traj.box.periodic:
            # Trained for 2D tori only; skipped (not failed) by the test suite.
            raise NotImplementedError(
                "the amortised CNN was trained for 2D periodic boxes only")
        if self.net is None:
            self._load()
        M = self._meta["grid"]
        self.box = traj.box
        N, Y, S, n = splat_groups(traj, M)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self._dev)
        feats = make_features(t(N), t(Y), t(S), t(n).float(),
                              torch.tensor([traj.dt], device=self._dev))
        vec = [(0, 1), (2, 3)]
        ks = range(8) if self.tta else [0]
        out = 0.0
        with torch.no_grad():
            for k in ks:
                pred = self.net(dihedral(feats, k, vec))
                out = out + _inverse_dihedral(pred, k, [(0, 1)])
        self.grid_ = (out / len(ks))[0].double().cpu().numpy()
        self.params["grid"] = M
        self._fitted = True
        return self

    def predict(self, x):
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        return grid_interpolate(self.grid_, x.reshape(-1, 2), self.box).reshape(lead + (2,))


def _inverse_dihedral(x, k: int, vector_channels):
    """Inverse of :func:`dihedral`: reflections first, then the transpose."""
    import torch
    out = x
    if k & 4:
        out = torch.roll(torch.flip(out, dims=[-1]), 1, dims=-1).clone()
        for _, cy in vector_channels:
            out[:, cy] = -out[:, cy]
    if k & 2:
        out = torch.roll(torch.flip(out, dims=[-2]), 1, dims=-2).clone()
        for cx, _ in vector_channels:
            out[:, cx] = -out[:, cx]
    if k & 1:
        out = out.transpose(-1, -2).clone()
        for cx, cy in vector_channels:
            out[:, [cx, cy]] = out[:, [cy, cx]]
    return out
