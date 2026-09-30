"""Rung 4: neural network regression of the drift.

The network regresses the same Kramers-Moyal target every other rung uses,
``y = dx/dt``, with the squared loss. The minimiser of that loss is the
conditional mean, which is the drift, so there is no trick: the question is
only which function class, and which regulariser, averages the noise best.

Two architectures
-----------------
``arch='mlp'`` (default, any ``d``)
    A multilayer perceptron on the points. On a torus the input is
    ``[sin(2 pi x/L), cos(2 pi x/L)]`` -- ``2d`` numbers, periodic and smooth
    across the box edge, so there is no seam. Trained by minibatch Adam.

``arch='cnn'`` (``d <= 3``)
    A convolutional decoder that outputs the drift on a regular grid of
    ``M^d`` nodes: a learned coarse latent, upsampled level by level through
    3x3 (x3) convolutions with periodic padding. The estimate between nodes is
    multilinear interpolation. Its training uses a device the MLP cannot:
    for a multilinear grid the loss over *all* transitions is a quadratic form
    in the node values,

    .. math::  \\sum_i |b(x_i) - y_i|^2
               = \\sum_{c, o} G_o(c)\\, b_c \\cdot b_{c+o}
                 - 2 \\sum_c b_c \\cdot Y_c + \\sum_i |y_i|^2 ,

    with ``G_o(c) = sum_i w_ic w_i,c+o`` over the ``3^d`` neighbour offsets and
    ``Y_c = sum_i w_ic y_i``. Those lattice statistics are accumulated once, so
    every optimisation step sees the exact full-data gradient in ``O(M^d)``,
    whatever the number of transitions.

Why the loss curve lies, and what to plot instead
-------------------------------------------------
Each target carries noise of variance ``2D/dt`` per component; in the 2D study
that is ~1000 times ``|b|^2``. The loss a perfect estimator reaches, the noise
floor ``2 d D/dt``, is ~99.9% of the loss at initialisation, so the raw curve is
flat from the first step. Subtracting the nominal floor does not help either:
the *realised* noise in a few million targets differs from its expectation by
more than the whole drift signal (and the O(dt) discretisation shifts it too),
so "loss minus floor" is routinely negative.

What is measurable is a *difference* of losses on the same targets, where the
shared noise cancels. The history records the **gain over the null model**,

.. math::  g = \\overline{|y|^2} - \\overline{|y - \\hat b(x)|^2}
             = \\overline{|b|^2} - \\overline{|\\hat b - b|^2}
               + 2\\, \\overline{\\hat b \\cdot \\xi} ,

on a fixed training subsample and on held-out walkers. The first term does not
depend on the fit, so ``g`` rises exactly as the mean squared error falls; the
noise term is ``O(sigma |b| / sqrt(n))`` -- small. On the training data ``g``
keeps rising as the network starts fitting noise; on held-out walkers it
peaks and falls. That peak is where training stops.

Regularisation, and the refit
-----------------------------
With one noisy field and no second look at it, the number of optimisation
steps *is* the smoothing parameter, just as bin width, bandwidth and basis size
were for rungs 1-3. It is chosen the same way: by held-out *walkers* (not
held-out timepoints -- consecutive frames of one walker are correlated). An
exponential moving average of the weights stands in for a decaying learning
rate, so the validated weights are not a noisy snapshot of one minibatch step.
Once the step count is known, the network is retrained on every walker for the
same number of steps per transition (``refit=True``), so the final fit, like the
cross-validated classical fits, uses all the data.

Measured on the study's walkers (``outputs/nn_mlp/README.md``): the MLP sits
between SFI and the local-linear kernel -- best of all with the least data, behind
SFI once data is plentiful -- and the CNN trails the MLP nearly everywhere except
the largest 2D budget. At ``d = 5`` neither network nor any classical method
learns the field at 6M transitions. That is not capacity (the MLP fits the exact
5D drift to nrmse 0.04); it is information: with ~100x the data MLP and SFI reach
0.25 and 0.20.

The truth never enters ``fit``: it receives a ``Trajectories`` object (positions,
dt, D, box), and ``monitor`` only records a diagnostic. A fit with and without a
ground-truth monitor gives bit-identical predictions.
"""
from __future__ import annotations

import copy
import itertools
import time

import numpy as np

from .base import TrajectoryEstimator, register

__all__ = ["NeuralDrift"]


def _torch():
    import torch
    return torch


@register
class NeuralDrift(TrajectoryEstimator):
    """Neural-network regression of ``b(x)`` on Kramers-Moyal targets.

    Parameters
    ----------
    arch:
        ``'mlp'`` (any ``d``) or ``'cnn'`` (``d <= 3``).
    hidden:
        MLP layer widths.
    periodic_features:
        MLP only: encode a periodic box as sines and cosines of the coordinate.
    channels, grid:
        CNN only: feature channels, and nodes per axis of the output grid.
        The grid is the CNN's smoothing dial -- a finer grid gives the decoder
        more freedom to fit noise -- so ``'cv'`` (default) picks it from
        ``CNN_GRIDS`` by the held-out gain, as rungs 1-3 pick theirs. The
        learned latent the grid is decoded from has a quarter of its nodes per
        axis (at least 4).
    max_steps, batch_size, lr, ema:
        Optimisation. ``batch_size='auto'`` uses ``min(16384, n/4)`` (MLP; the
        CNN is always full-batch). ``ema`` is the decay of the weight average;
        ``'auto'`` is 0.995 for the MLP, whose minibatch steps are noisy, and
        off for the full-batch CNN, where an average only lags.
    val_walkers:
        Fraction of walkers held out to choose the step count (at least two).
    eval_every:
        Steps between validation checks (``'auto'``: 100 for the MLP, 25 for
        the CNN); training stops once the held-out gain
        has not improved for ``max(patience, best_step / 2)`` steps
        (``patience='auto'``: 1000 for the MLP, 300 for the CNN, whose best
        step is typically a few hundred).
    refit:
        Retrain on all walkers for the validated number of steps.
    monitor:
        Optional ``callable(predict) -> float`` evaluated at every check and
        stored as ``history['monitor']``. Meant for a ground-truth error in a
        study; it never influences training.
    """

    key = "nn"
    label = "Neural network"
    color_slot = 3

    CNN_GRIDS = {1: (64, 128, 256), 2: (16, 32, 64), 3: (8, 16, 32)}
    #: Rows used to measure the training gain at each check, and the cap on
    #: held-out rows. The held-out gain is what selects the step, and near a
    #: plateau its noise decides the pick: with 1M rows the 12.3M-transition fit
    #: refit to nrmse 0.085, with all 2.5M to 0.080. (Stopping at the truth's
    #: best step would give 0.0765. Two one-standard-error variants of the rule
    #: were tried and each lost elsewhere -- 0.42 against 0.34 with 16 walkers,
    #: or no gain at all -- so the plain arg-max stays.)
    EVAL_ROWS = 1_000_000
    VAL_ROWS = 4_000_000

    def __init__(self, arch: str = "mlp", hidden=(128, 128, 128),
                 periodic_features: bool = True, channels: int = 32,
                 grid="cv", max_steps: int = 20000,
                 batch_size="auto", lr: float = 1e-3, weight_decay: float = 0.0,
                 ema="auto", val_walkers: float = 0.2,
                 eval_every="auto", patience="auto", refit: bool = True,
                 monitor=None, device=None, seed: int = 0, **params):
        if arch not in ("mlp", "cnn"):
            raise ValueError(f"arch must be 'mlp' or 'cnn', not {arch!r}")
        super().__init__(arch=arch, **params)
        self.arch = arch
        self.label = ("Neural network, MLP" if arch == "mlp"
                      else "Neural network, CNN")
        self.hidden = tuple(hidden)
        self.periodic_features = bool(periodic_features)
        self.channels = int(channels)
        self.grid = grid
        self.max_steps = int(max_steps)
        self.batch_size = batch_size
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.ema = (0.995 if arch == "mlp" else 0.0) if ema == "auto" else float(ema)
        self.val_walkers = float(val_walkers)
        self.eval_every = ((100 if arch == "mlp" else 25) if eval_every == "auto"
                           else int(eval_every))
        self.patience = ((1000 if arch == "mlp" else 300) if patience == "auto"
                         else int(patience))
        self.refit = bool(refit)
        self.monitor = monitor
        self.device = device
        self.seed = int(seed)
        self.net = None
        self.box = None
        self.history: dict = {}

    # ------------------------------------------------------------------
    # shared plumbing
    # ------------------------------------------------------------------
    def _dev(self):
        torch = _torch()
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def noise_floor(self, traj) -> float:
        """``2 d D / dt`` -- the expected loss of a perfect estimator."""
        return 2.0 * traj.d * traj.D / traj.dt

    def _split_walkers(self, n_walkers):
        n_val = int(round(self.val_walkers * n_walkers))
        n_val = min(max(n_val, 2), n_walkers - 1)
        perm = np.random.default_rng(self.seed + 101).permutation(n_walkers)
        is_val = np.zeros(n_walkers, bool)
        is_val[perm[:n_val]] = True
        return is_val

    def _stop(self, step, best_step):
        return step - best_step > max(self.patience, 0.5 * best_step)

    @staticmethod
    def _ema_update(avg, net, decay):
        torch = _torch()
        with torch.no_grad():
            for pa, pn in zip(avg.parameters(), net.parameters()):
                pa.lerp_(pn, 1.0 - decay)

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------
    def build_network(self, d_in: int, d_out: int):
        """The MLP: GELU layers, last layer zero-initialised.

        Starting from ``b = 0`` means the first steps learn the drift rather
        than first unlearning a random initial field.
        """
        nn = _torch().nn
        layers, w_in = [], d_in
        for w in self.hidden:
            layers += [nn.Linear(w_in, w), nn.GELU()]
            w_in = w
        out = nn.Linear(w_in, d_out)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        return nn.Sequential(*layers, out)

    def encode(self, x):
        """Network input for points ``x`` (a torch tensor on the fit's device)."""
        torch = _torch()
        if self.periodic_features and self.box.periodic:
            ph = 2.0 * np.pi * (x - self._lo) / self._L
            return torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)
        return (x - self._mu) / self._sd

    def _mlp_forward(self, net, x):
        return net(self.encode(x)) * self._ystd

    def _fit_mlp(self, traj):
        torch = _torch()
        dev = self._dev()
        d = traj.d
        x, y = self.regression_data(traj)
        T = traj.n_frames - 1
        walker = np.repeat(np.arange(traj.n_walkers), T)
        is_val_w = self._split_walkers(traj.n_walkers)
        is_val = is_val_w[walker]

        f32 = torch.float32
        self._lo = torch.as_tensor(traj.box.lo, dtype=f32, device=dev)
        self._L = torch.as_tensor(traj.box.length, dtype=f32, device=dev)
        self._mu = torch.as_tensor(x.mean(0), dtype=f32, device=dev)
        self._sd = torch.as_tensor(x.std(0) + 1e-12, dtype=f32, device=dev)
        # Targets standardised by their total RMS, which is dominated by the
        # noise: the drift then lives well inside (-1, 1) at any noise level.
        self._ystd = float(np.sqrt(np.mean(y ** 2)))
        d_in = 2 * d if (self.periodic_features and traj.box.periodic) else d

        X = torch.as_tensor(x, dtype=f32, device=dev)
        Y = torch.as_tensor(y, dtype=f32, device=dev)
        tr_idx = torch.as_tensor(np.flatnonzero(~is_val), device=dev)
        va_idx = torch.as_tensor(np.flatnonzero(is_val), device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(self.seed)

        def fixed_subset(idx, cap=self.EVAL_ROWS):
            if len(idx) <= cap:
                return idx
            pick = torch.randperm(len(idx), device=dev, generator=gen)[:cap]
            return idx[pick]

        def row_gain(net, idx):
            """Per-row ``|y|^2 - |y - b_hat|^2 = 2 b_hat.y - |b_hat|^2``."""
            out = torch.empty(len(idx), dtype=torch.float64, device=dev)
            with torch.no_grad():
                for s in range(0, len(idx), 500_000):
                    r = idx[s:s + 500_000]
                    b = self._mlp_forward(net, X[r])
                    out[s:s + 500_000] = (2.0 * (b * Y[r]).sum(1)
                                          - (b * b).sum(1)).double()
            return out

        def gain(net, idx):
            return float(row_gain(net, idx).mean())

        def train(rows, steps, val_rows=None):
            torch.manual_seed(self.seed)
            net = self.build_network(d_in, d).to(dev)
            avg = copy.deepcopy(net)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
            bs = (min(16384, max(256, len(rows) // 4))
                  if self.batch_size == "auto" else int(self.batch_size))
            tr_eval = fixed_subset(rows)
            h = {"step": [], "gain_train": [], "gain_val": [], "monitor": []}
            best = (-np.inf, 0, None)
            if val_rows is not None:
                val_rows = fixed_subset(val_rows, self.VAL_ROWS)
                h["y2_val"] = float(Y[val_rows].double().pow(2).sum(1).mean())
            h["y2_train"] = float(Y[tr_eval].double().pow(2).sum(1).mean())
            Yn = Y / self._ystd
            for step in range(1, steps + 1):
                r = rows[torch.randint(len(rows), (bs,), device=dev, generator=gen)]
                loss = (net(self.encode(X[r])) - Yn[r]).pow(2).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self._ema_update(avg, net, self.ema)
                if step % self.eval_every and step != steps:
                    continue
                h["step"].append(step)
                h["gain_train"].append(gain(avg, tr_eval))
                if val_rows is not None:
                    g = gain(avg, val_rows)
                    h["gain_val"].append(g)
                    if g > best[0]:
                        best = (g, step, copy.deepcopy(avg.state_dict()))
                if self.monitor is not None:
                    self.net = avg
                    h["monitor"].append(float(self.monitor(self.predict_unchecked)))
                if val_rows is not None and self._stop(step, best[1]):
                    break
            return avg, h, best

        t0 = time.time()
        net, h1, best = train(tr_idx, self.max_steps, va_idx)
        net.load_state_dict(best[2])
        h1["best_step"] = best[1]
        self.history = {"select": h1}
        if self.refit:
            steps = int(round(best[1] * len(X) / len(tr_idx)))
            net, h2, _ = train(torch.arange(len(X), device=dev), max(steps, 1))
            self.history["refit"] = h2
            self.history["refit_steps"] = steps
        self.net = net.eval()
        self.history.update(
            arch="mlp", noise_floor=self.noise_floor(traj),
            mean_y2=float(np.mean(np.sum(y ** 2, 1))), fit_seconds=time.time() - t0,
            n_params=sum(p.numel() for p in net.parameters()),
            n_val_walkers=int(is_val_w.sum()))

    # ------------------------------------------------------------------
    # CNN
    # ------------------------------------------------------------------
    def _lattice(self, traj, M):
        """Grid geometry: node spacing, origin, periodic or not."""
        d = traj.d
        if traj.box.periodic:
            origin = np.asarray(traj.box.lo, float)
            h = np.asarray(traj.box.length, float) / M
        else:
            # Nodes spanning where the data is, with a margin, not the box.
            xs = traj.x.reshape(-1, d)
            lo, hi = np.percentile(xs, [0.1, 99.9], axis=0)
            pad = 0.1 * (hi - lo)
            origin = lo - pad
            h = (hi - lo + 2 * pad) / (M - 1)
        self._M, self._origin, self._h = M, origin, h

    def _corners(self, x):
        """Base node index and fractional position, per axis (numpy or torch)."""
        u = (x - self._origin) / self._h
        M = self._M
        if self.box.periodic:
            i0 = np.floor(u) if isinstance(u, np.ndarray) else u.floor()
            t = u - i0
            return i0, t
        top = M - 2
        if isinstance(u, np.ndarray):
            i0 = np.clip(np.floor(u), 0, top)
            t = np.clip(u - i0, 0.0, 1.0)
        else:
            i0 = u.floor().clamp(0, top)
            t = (u - i0).clamp(0.0, 1.0)
        return i0, t

    def _lattice_stats(self, traj, groups, n_groups):
        """Per-group ``(G, Y, S, n)`` for the multilinear quadratic loss."""
        torch = _torch()
        dev = self._dev()
        d, M = traj.d, self._M
        cells = M ** d
        corners = list(itertools.product((0, 1), repeat=d))
        offsets = list(itertools.product((-1, 0, 1), repeat=d))
        oid = {o: k for k, o in enumerate(offsets)}
        f64 = torch.float64
        G = torch.zeros(n_groups * len(offsets) * cells, dtype=f64, device=dev)
        Yc = torch.zeros(n_groups * cells * d, dtype=f64, device=dev)
        S = torch.zeros(n_groups, dtype=f64, device=dev)
        cnt = torch.zeros(n_groups, dtype=f64, device=dev)
        stride = [M ** (d - 1 - k) for k in range(d)]
        T = traj.n_frames - 1
        per = max(1, 2_000_000 // T)
        origin = torch.as_tensor(self._origin, dtype=f64, device=dev)
        h = torch.as_tensor(self._h, dtype=f64, device=dev)
        for w0 in range(0, traj.n_walkers, per):
            xs = traj.x[w0:w0 + per]
            g = torch.as_tensor(np.repeat(groups[w0:w0 + per], T), device=dev)
            x0 = torch.as_tensor(xs[:, :-1].reshape(-1, d), dtype=f64, device=dev)
            y = torch.as_tensor(traj.box.displacement(xs[:, :-1], xs[:, 1:])
                                .reshape(-1, d), dtype=f64, device=dev) / traj.dt
            u = (x0 - origin) / h
            if traj.box.periodic:
                i0 = u.floor()
                t = u - i0
            else:
                i0 = u.floor().clamp(0, M - 2)
                t = (u - i0).clamp(0.0, 1.0)
            i0 = i0.long()
            S.index_add_(0, g, (y * y).sum(1))
            cnt.index_add_(0, g, torch.ones(len(g), dtype=f64, device=dev))
            ws, flat = [], []
            for e in corners:
                w = torch.ones(len(g), dtype=f64, device=dev)
                c = torch.zeros(len(g), dtype=torch.long, device=dev)
                for k in range(d):
                    w = w * (t[:, k] if e[k] else 1.0 - t[:, k])
                    c = c + ((i0[:, k] + e[k]) % M) * stride[k]
                ws.append(w)
                flat.append(c)
            for w, c in zip(ws, flat):
                base = (g * cells + c) * d
                for k in range(d):
                    Yc.index_add_(0, base + k, w * y[:, k])
            for a, e in enumerate(corners):
                for b, e2 in enumerate(corners):
                    o = tuple(e2[k] - e[k] for k in range(d))
                    gi = (g * len(offsets) + oid[o]) * cells + flat[a]
                    G.index_add_(0, gi, ws[a] * ws[b])
        shape = (M,) * d
        G = G.reshape(n_groups, len(offsets), *shape)
        Yc = Yc.reshape(n_groups, cells, d).permute(0, 2, 1).reshape(n_groups, d, *shape)
        return G, Yc, S, cnt, offsets

    @staticmethod
    def _quad(b, G, Y, offsets):
        """``sum_i |b(x_i)|^2 - 2 b(x_i).y_i`` from lattice statistics."""
        torch = _torch()
        dims = tuple(range(1, b.dim()))
        q = 0.0
        for k, o in enumerate(offsets):
            bo = torch.roll(b, shifts=tuple(-v for v in o), dims=dims) if any(o) else b
            q = q + (G[k] * (b * bo).sum(0)).sum()
        return q - 2.0 * (b * Y).sum()

    def build_cnn(self, d: int):
        """Decoder from a learned ``latent^d`` tensor to the ``M^d`` drift grid."""
        torch = _torch()
        nn, F = torch.nn, torch.nn.functional
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[d]
        mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[d]
        periodic = self.box.periodic
        M, C = self._M, self.channels
        m0 = max(4, M // 4)
        if periodic:
            n_up = int(round(np.log2(M / m0)))
            if m0 * 2 ** n_up != M:
                raise ValueError(f"grid {M} must be latent {m0} times a power of 2")
        else:
            n_up = max(1, int(np.ceil(np.log2((M - 1) / (m0 - 1)))))
        pad_mode = "circular" if periodic else "replicate"

        class Up(nn.Module):
            def forward(self, x):
                if periodic:
                    x = F.pad(x, (1, 1) * d, mode="circular")
                    x = F.interpolate(x, scale_factor=2, mode=mode,
                                      align_corners=False)
                    sl = (slice(None), slice(None)) + (slice(2, -2),) * d
                    return x[sl]
                size = (2 * (x.shape[-1] - 1) + 1,) * d
                return F.interpolate(x, size=size, mode=mode, align_corners=True)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = conv(C, C, 3, padding=1, padding_mode=pad_mode)
                self.b = conv(C, C, 3, padding=1, padding_mode=pad_mode)

            def forward(self, x):
                return F.gelu(self.b(F.gelu(self.a(x))))

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.z = nn.Parameter(0.1 * torch.randn((1, C) + (m0,) * d))
                self.up = Up()
                self.blocks = nn.ModuleList([Block() for _ in range(n_up)])
                self.out = conv(C, d, 1)
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

            def forward(self):
                x = self.z
                for blk in self.blocks:
                    x = blk(self.up(x))
                if not periodic and x.shape[-1] != M:
                    x = F.interpolate(x, size=(M,) * d, mode=mode,
                                      align_corners=True)
                return self.out(x)[0]

        return Decoder()

    def _cnn_scale(self, G, Y):
        """Output scale: the drift's RMS on a coarse average of the lattice
        targets -- the grid version of standardising the targets. Weighted
        node counts are the neighbour Gram rows summed over offsets."""
        torch = _torch()
        d = Y.shape[0]
        wsum = G.sum(0)
        k = max(1, self._M // 8)
        F = torch.nn.functional
        pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[d]
        Yp = pool(Y[None], k)[0]
        Np = pool(wsum[None, None], k)[0, 0].clamp_min(1e-12)
        rms = torch.sqrt(((Yp / Np).pow(2).sum(0) * Np).sum() / Np.sum())
        return max(float(rms), 1e-6)

    def _train_cnn(self, d, offsets, stats, steps, val=None):
        torch = _torch()
        dev = self._dev()
        Gs, Ys, ns = stats
        torch.manual_seed(self.seed)
        net = self.build_cnn(d).to(dev)
        avg = copy.deepcopy(net) if self.ema > 0 else net
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        h = {"step": [], "gain_train": [], "gain_val": [], "monitor": []}
        best = (-np.inf, 0, None)
        scale2 = self._ystd ** 2
        for step in range(1, steps + 1):
            b = net().double() * self._ystd
            loss = self._quad(b, Gs, Ys, offsets) / ns / scale2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if self.ema > 0:
                self._ema_update(avg, net, self.ema)
            if step % self.eval_every and step != steps:
                continue
            with torch.no_grad():
                bb = avg().double() * self._ystd
                h["step"].append(step)
                h["gain_train"].append(-float(self._quad(bb, Gs, Ys, offsets) / ns))
                if val is not None:
                    g = -float(self._quad(bb, val[0], val[1], offsets) / val[2])
                    h["gain_val"].append(g)
                    if g > best[0]:
                        best = (g, step, copy.deepcopy(avg.state_dict()))
            if self.monitor is not None:
                self._grid = bb.float().cpu().numpy()
                h["monitor"].append(float(self.monitor(self.predict_unchecked)))
            if val is not None and self._stop(step, best[1]):
                break
        return avg, h, best

    def _fit_cnn(self, traj):
        d = traj.d
        if d > 3:
            raise ValueError(
                f"arch='cnn' needs a grid of M^d nodes; d = {d} is out of reach "
                f"(use arch='mlp')")
        t0 = time.time()
        grids = self.CNN_GRIDS[d] if self.grid == "cv" else (int(self.grid),)
        is_val_w = self._split_walkers(traj.n_walkers)
        groups = is_val_w.astype(np.int64)
        # Selection: one run per candidate grid on the training walkers, each
        # stopped at its best held-out gain. The same held-out walkers score
        # every candidate, so their gains are directly comparable.
        runs = {}
        for M in grids:
            self._lattice(traj, M)
            G, Yc, S, cnt, offsets = self._lattice_stats(traj, groups, 2)
            self._ystd = self._cnn_scale(G.sum(0), Yc.sum(0))
            net, h, best = self._train_cnn(d, offsets, (G[0], Yc[0], cnt[0]),
                                           self.max_steps, (G[1], Yc[1], cnt[1]))
            h["best_step"] = best[1]
            h["best_gain"] = best[0]
            h["y2_train"] = float(S[0] / cnt[0])
            h["y2_val"] = float(S[1] / cnt[1])
            runs[M] = (h, G, Yc, S, cnt, offsets, self._ystd)
        M = max(runs, key=lambda m: runs[m][0]["best_gain"])
        h1, G, Yc, S, cnt, offsets, self._ystd = runs[M]
        self._lattice(traj, M)
        self.history = {"select": h1,
                        "grid_cv": {m: (r[0]["best_gain"], r[0]["best_step"])
                                    for m, r in runs.items()},
                        "select_by_grid": {m: r[0] for m, r in runs.items()}}
        steps = h1["best_step"]
        if self.refit:
            # Full-batch: every step already sees all of its data, so the
            # validated step count carries over unchanged.
            net, h2, _ = self._train_cnn(d, offsets, (G.sum(0), Yc.sum(0), cnt.sum()),
                                         max(steps, 1))
            self.history["refit"] = h2
            self.history["refit_steps"] = steps
        else:
            net, _, _ = self._train_cnn(d, offsets, (G[0], Yc[0], cnt[0]), max(steps, 1))
        torch = _torch()
        with torch.no_grad():
            self._grid = (net().double() * self._ystd).float().cpu().numpy()
        self.net = net.eval()
        self.params["grid"] = M
        self.history.update(
            arch="cnn", noise_floor=self.noise_floor(traj),
            mean_y2=float((S.sum() / cnt.sum()).item()),
            fit_seconds=time.time() - t0,
            n_params=sum(p.numel() for p in net.parameters()),
            n_val_walkers=int(is_val_w.sum()), grid=M)

    def _grid_predict(self, x):
        d, M = self.box.d, self._M
        i0, t = self._corners(x)
        i0 = i0.astype(np.int64)
        out = np.zeros((len(x), d))
        for e in itertools.product((0, 1), repeat=d):
            w = np.ones(len(x))
            idx = []
            for k in range(d):
                w *= t[:, k] if e[k] else 1.0 - t[:, k]
                idx.append((i0[:, k] + e[k]) % M)
            out += w[:, None] * self._grid[(slice(None),) + tuple(idx)].T
        return out

    # ------------------------------------------------------------------
    # the interface
    # ------------------------------------------------------------------
    def fit(self, traj):
        """Train on ``(x, dx/dt)``; see the module docstring for the procedure."""
        self.box = traj.box
        if traj.n_walkers < 3:
            raise ValueError("NeuralDrift holds out whole walkers; it needs >= 3")
        if self.arch == "mlp":
            self._fit_mlp(traj)
        else:
            self._fit_cnn(traj)
        self.params["best_step"] = self.history["select"]["best_step"]
        self._fitted = True
        return self

    def predict_unchecked(self, x):
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        if self.arch == "cnn":
            return self._grid_predict(flat).reshape(lead + (flat.shape[1],))
        torch = _torch()
        dev = self._lo.device
        out = np.empty((len(flat), flat.shape[1]))
        with torch.no_grad():
            for s in range(0, len(flat), 500_000):
                xt = torch.as_tensor(flat[s:s + 500_000], dtype=torch.float32,
                                     device=dev)
                out[s:s + 500_000] = self._mlp_forward(self.net, xt).double().cpu().numpy()
        return out.reshape(lead + (flat.shape[1],))

    def predict(self, x):
        self._check_fitted()
        return self.predict_unchecked(x)

    def diagnostics(self) -> dict:
        """Numbers worth a column in a results table."""
        self._check_fitted()
        h = self.history
        return {"best_step": h["select"]["best_step"],
                "stopped_at": h["select"]["step"][-1],
                "refit_steps": h.get("refit_steps", 0),
                "n_params": h["n_params"],
                "fit_seconds": h["fit_seconds"],
                **({"grid": h["grid"]} if "grid" in h else {})}
