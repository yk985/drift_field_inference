"""Rung 5: score-based estimation from snapshots -- and where it must fail.

For an equilibrium system the stationary density is ``rho_ss ~ exp(-U/D)``, so

.. math::  \\nabla \\log \\rho_{ss}(x) = -\\nabla U(x) / D = b(x) / D ,

**the score is the drift, up to a factor of D.** A score estimated from
snapshots -- unordered positions, no time information -- is therefore a drift
estimator. Denoising score matching is the computation that trains a diffusion
model; here it estimates exactly the physical quantity we want.

The identity holds **only** for gradient drift. Add a rotational part
``b_rot = omega A grad U`` and the stationary density does not change at all
(exactly, in this codebase -- see ``dfi.fields.base``), so the score does not
change, so *any* snapshot estimator returns the same answer for every ``omega``.
The rotational part is not hard to see in snapshots; it is absent from them.

Two estimators
--------------
:class:`KDEScore`
    Gaussian kernel density estimate, differentiated in closed form. Bandwidth
    chosen by cross-validated **score matching**, not by density likelihood --
    the two disagree, and the reason is the lesson of the classical rung.
:class:`DenoisingScoreMatching`
    An MLP ``s_theta(x, sigma)`` trained to denoise snapshots at a range of noise
    scales. The scale it is evaluated at is chosen by the same held-out score
    matching loss.

Both need ``D`` to turn a score into a drift. It is passed in: snapshots alone
cannot tell ``D`` apart from the depth of the potential (``U/D`` is all they
see), so from snapshots alone the drift is identifiable only up to that factor.

The held-out criterion (Hyvarinen score matching)
--------------------------------------------------
For any candidate score ``s``, up to a constant independent of ``s``,

.. math::  J(s) = E_{\\rho}\\Big[\\tfrac12 |s(x)|^2 + \\nabla\\cdot s(x)\\Big]
             = \\tfrac12 E_\\rho |s - \\nabla\\log\\rho|^2 + \\text{const},

so ``J`` evaluated on held-out samples ranks candidate scores by their squared
error against the true score -- without knowing the true score.
"""
from __future__ import annotations

import copy

import numpy as np

from .base import SnapshotEstimator, register

__all__ = ["KDEScore", "DenoisingScoreMatching"]


def _torch():
    import torch
    return torch


def _device(device):
    torch = _torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _offsets(q, xs, box):
    """``q[:, None] - xs[None]``, minimum image on a torus (torch)."""
    u = q[:, None, :] - xs[None, :, :]
    if box is not None and box.periodic:
        torch = _torch()
        L = torch.as_tensor(box.length, dtype=u.dtype, device=u.device)
        u = u - L * torch.round(u / L)
    return u


# --------------------------------------------------------------------------
# KDE
# --------------------------------------------------------------------------

@register
class KDEScore(SnapshotEstimator):
    """The classical snapshot estimator: KDE the density, differentiate the log.

    .. math::  \\hat\\rho(x) = \\frac1n\\sum_i K_h(x - x_i), \\qquad
               \\hat s(x) = \\nabla\\log\\hat\\rho(x)
               = -\\frac{1}{h^2}\\frac{\\sum_i w_i (x - x_i)}{\\sum_i w_i}

    with ``w_i = exp(-|x - x_i|^2 / 2h^2)`` (minimum image on a torus).

    Bandwidth
    ---------
    ``bandwidth='cv'`` (default) minimises the held-out score matching loss
    ``mean(Laplacian(rho)/rho - |s|^2/2)`` -- the Gaussian kernel gives the
    Laplacian in closed form too -- over a grid around Silverman's rule, on up
    to 20000 held-out samples against up to 40000 others, then rescales by ``(n_cv/n)^(1/(d+6))`` (the rate
    for estimating a *derivative* of a density). The bandwidth that maximises
    held-out *likelihood* is also recorded, as ``h_density_``: it is smaller.
    Differentiation amplifies high-frequency noise, so the score wants more
    smoothing than the density.

    Support
    -------
    At least ``min_count`` samples within ``2h`` -- the same presence rule as
    the kernel regression rung. In the tails the log-gradient is an average of
    offsets to a handful of samples, noisy exactly where the drift is largest.
    """

    key = "kde_score"
    label = "KDE score (snapshots)"
    color_slot = 5

    def __init__(self, bandwidth="cv", max_samples: int | None = 100_000,
                 min_count: float = 8.0, device=None, seed: int = 0, **params):
        super().__init__(bandwidth=bandwidth, **params)
        self.bandwidth = bandwidth
        self.max_samples = max_samples
        self.min_count = float(min_count)
        self.device = device
        self.seed = int(seed)
        self.samples = None
        self.box = None
        self.h_ = None
        self.h_density_ = None
        self.cv_ = None

    # -- sums ---------------------------------------------------------------
    def _sums(self, q, xs, hs, budget=3e7):
        """For each h: sum w, sum w u, sum w |u|^2, count within 2h, max log w.

        Weights are shifted by the nearest sample's distance before
        exponentiating, so a query far from every sample still gets finite
        ratios.
        """
        torch = _torch()
        n, d = xs.shape
        step = int(max(1, budget // max(n * d, 1)))
        out = {h: [[], [], [], [], []] for h in hs}
        for s in range(0, len(q), step):
            u = _offsets(q[s:s + step], xs, self.box)
            r2 = (u * u).sum(-1)
            rmin = r2.min(dim=1, keepdim=True).values
            for h in hs:
                w = torch.exp(-(r2 - rmin) / (2 * h * h))
                sw = w.sum(1)
                out[h][0].append(sw)
                out[h][1].append((w[..., None] * u).sum(1))
                out[h][2].append((w * r2).sum(1))
                out[h][3].append((r2 <= 4 * h * h).sum(1).double())
                out[h][4].append(-rmin[:, 0] / (2 * h * h))
        return {h: [torch.cat(v) for v in lists] for h, lists in out.items()}

    def _score_from(self, sums, h):
        sw, swu, _, _, _ = sums
        return -swu / (h * h * sw[:, None])

    def fit(self, samples, D, box=None):
        torch = _torch()
        dev = _device(self.device)
        x = np.asarray(samples, float)
        if x.ndim == 1:
            x = x[:, None]
        rng = np.random.default_rng(self.seed)
        if self.max_samples and len(x) > self.max_samples:
            x = x[rng.choice(len(x), self.max_samples, replace=False)]
        self.box, self.D = box, float(D)
        n, d = x.shape
        # Silverman's rule as the centre of the search.
        spread = float(np.mean(x.std(0)))
        if box is not None and box.periodic:
            spread = min(spread, float(np.min(box.length)) / np.sqrt(12))
        h0 = spread * (4.0 / (d + 2) / n) ** (1.0 / (d + 4))
        self.h_silverman_ = h0

        if self.bandwidth == "cv":
            perm = rng.permutation(n)
            # The held-out criterion is noisy -- its divergence term is large and
            # sign-changing -- so it needs many held-out points (with 5000 on 1D
            # OU it picked h = 0.038 where the truth's optimum is 0.08). The
            # training side of the CV can be subsampled instead; the rescaling
            # below accounts for it.
            n_val = min(20_000, n // 5)
            n_tr = min(40_000, n - n_val)
            val, tr = x[perm[:n_val]], x[perm[n_val:n_val + n_tr]]
            xt = torch.as_tensor(tr, dtype=torch.float64, device=dev)
            qv = torch.as_tensor(val, dtype=torch.float64, device=dev)
            hs = [float(h0 * f) for f in 2.0 ** np.linspace(-2, 3, 16)]
            if box is not None and box.periodic:
                hs = [h for h in hs if h < 0.35 * float(np.min(box.length))]
            sums = self._sums(qv, xt, hs)
            sm, ll = [], []
            for h in hs:
                sw, swu, swr2, _, lmax = sums[h]
                s = -swu / (h * h * sw[:, None])
                lap_over_rho = swr2 / (h ** 4 * sw) - d / h ** 2
                sm.append(float((lap_over_rho - 0.5 * (s * s).sum(1)).mean()))
                # held-out log-likelihood of the density estimate
                lr = (torch.log(sw) + lmax - np.log(len(tr))
                      - 0.5 * d * np.log(2 * np.pi * h * h))
                ll.append(float(lr.mean()))
            i_sm, i_ll = int(np.argmin(sm)), int(np.argmax(ll))
            scale = (len(tr) / n) ** (1.0 / (d + 6))
            self.h_ = hs[i_sm] * scale
            self.h_density_ = hs[i_ll] * (len(tr) / n) ** (1.0 / (d + 4))
            self.cv_ = {"h": np.array(hs), "score_matching": np.array(sm),
                        "log_likelihood": np.array(ll),
                        "h_score": hs[i_sm], "h_density": hs[i_ll]}
        elif self.bandwidth == "silverman":
            self.h_ = h0
        else:
            self.h_ = float(self.bandwidth)

        self.samples = torch.as_tensor(x, dtype=torch.float64, device=dev)
        self.params["bandwidth"] = self.h_
        if self.h_density_ is not None:
            self.params["bandwidth_density"] = self.h_density_
        self._cache = None
        self._fitted = True
        return self

    def _evaluate(self, x):
        torch = _torch()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        key = (flat.shape, hash(flat.tobytes()))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1], lead
        q = torch.as_tensor(flat, dtype=torch.float64, device=self.samples.device)
        sums = self._sums(q, self.samples, [self.h_])[self.h_]
        res = (self._score_from(sums, self.h_).cpu().numpy(), sums[3].cpu().numpy())
        self._cache = (key, res)
        return res, lead

    def predict_score(self, x):
        self._check_fitted()
        (score, _), lead = self._evaluate(x)
        return score.reshape(lead + (score.shape[-1],))

    def support(self, x):
        self._check_fitted()
        (_, count), lead = self._evaluate(x)
        return (count >= self.min_count).reshape(lead)

    def diagnostics(self) -> dict:
        self._check_fitted()
        return {"bandwidth": self.h_,
                "bandwidth_density": self.h_density_ or np.nan,
                "bandwidth_silverman": self.h_silverman_}


# --------------------------------------------------------------------------
# denoising score matching
# --------------------------------------------------------------------------

@register
class DenoisingScoreMatching(SnapshotEstimator):
    """Neural denoising score matching on snapshots (Vincent 2011; Song 2021).

    Train ``s_theta(x, sigma)`` so that, for a snapshot ``x`` and noise
    ``x_tilde = x + sigma eps``,

    .. math::  L = E\\,\\big|\\, \\sigma\\, s_\\theta(\\tilde x, \\sigma) + \\epsilon \\,\\big|^2 ,

    which is the usual loss ``|s + (x_tilde - x)/sigma^2|^2`` weighted by
    ``sigma^2`` so every scale contributes alike. Its minimiser is the score of
    the density smoothed by noise ``sigma``. ``sigma`` is drawn log-uniformly
    per sample in ``[sigma_min, sigma_max]``; the network outputs ``s / sigma``
    scaled back, so its raw output stays O(1) at every scale. ``sigma`` plays the
    role of diffusion time: this *is* diffusion-model training on the
    snapshots, stopping short of sampling.

    On a torus the noisy point is wrapped, and ``sigma_max`` is capped at a
    quarter of the box so the wrapped Gaussian is still close to a Gaussian.

    Which sigma to evaluate at
    --------------------------
    Small ``sigma`` is least smoothed but least trained -- the target
    ``-eps/sigma`` is noisiest there. ``sigma_eval='cv'`` evaluates the held-out
    score matching loss ``mean(|s|^2/2 + div s)`` (divergence by autograd) at a
    grid of ``sigma`` and picks the minimum: the same criterion that sets the
    KDE bandwidth. The per-epoch train and validation losses go into
    ``history``.
    """

    key = "dsm"
    label = "Denoising score matching"
    color_slot = 6

    def __init__(self, hidden=(256, 256, 256), epochs: int = 40,
                 batch_size: int = 4096, lr: float = 2e-3,
                 sigma_min: float = 0.01, sigma_max: float = 0.5,
                 sigma_eval="cv", periodic_features: bool = True,
                 max_samples: int | None = 500_000, ema: float = 0.999,
                 device=None, seed: int = 0, **params):
        super().__init__(hidden=hidden, epochs=epochs, lr=lr,
                         sigma_min=sigma_min, sigma_max=sigma_max, **params)
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_eval = sigma_eval
        self.periodic_features = bool(periodic_features)
        self.max_samples = max_samples
        self.ema = float(ema)
        self.device = device
        self.seed = int(seed)
        self.net = None
        self.box = None
        self.history: dict = {}

    def build_network(self, d_in: int, d_out: int):
        """MLP taking ``(encoded x, log sigma features)`` to ``sigma * score``."""
        nn = _torch().nn
        layers, w = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(w, h), nn.GELU()]
            w = h
        return nn.Sequential(*layers, nn.Linear(w, d_out))

    def _encode(self, x, sigma):
        torch = _torch()
        if self.box is not None and self.box.periodic and self.periodic_features:
            ph = 2 * np.pi * (x - self._lo) / self._L
            feats = [torch.sin(ph), torch.cos(ph)]
        else:
            feats = [(x - self._mu) / self._sd]
        ls = torch.log(sigma)[:, None]
        # a few Fourier features of log sigma, so the scale dependence is easy
        feats += [ls / 3.0, torch.sin(ls), torch.cos(ls)]
        return torch.cat(feats, dim=1)

    def _score(self, net, x, sigma):
        return net(self._encode(x, sigma)) / sigma[:, None]

    def _wrap(self, x):
        if self.box is not None and self.box.periodic:
            torch = _torch()
            return self._lo + torch.remainder(x - self._lo, self._L)
        return x

    def fit(self, samples, D, box=None):
        torch = _torch()
        dev = _device(self.device)
        x = np.asarray(samples, float)
        if x.ndim == 1:
            x = x[:, None]
        rng = np.random.default_rng(self.seed)
        if self.max_samples and len(x) > self.max_samples:
            x = x[rng.choice(len(x), self.max_samples, replace=False)]
        self.box, self.D = box, float(D)
        n, d = x.shape
        f32 = torch.float32
        if box is not None:
            self._lo = torch.as_tensor(box.lo, dtype=f32, device=dev)
            self._L = torch.as_tensor(box.length, dtype=f32, device=dev)
        self._mu = torch.as_tensor(x.mean(0), dtype=f32, device=dev)
        self._sd = torch.as_tensor(x.std(0) + 1e-12, dtype=f32, device=dev)
        smax = self.sigma_max
        if box is not None and box.periodic:
            smax = min(smax, 0.25 * float(np.min(box.length)))
        else:
            smax = min(smax, float(x.std(0).max()))
        self._smax = smax

        perm = rng.permutation(n)
        n_val = min(20_000, n // 10)
        X = torch.as_tensor(x, dtype=f32, device=dev)
        val, tr = X[perm[:n_val]], X[perm[n_val:]]
        torch.manual_seed(self.seed)
        d_in = (2 * d if (box is not None and box.periodic and self.periodic_features)
                else d) + 3
        net = self.build_network(d_in, d).to(dev)
        avg = copy.deepcopy(net)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        bs = min(self.batch_size, max(64, len(tr) // 4))
        steps_per = max(1, len(tr) // bs)
        total = self.epochs * steps_per
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=total)
        gen = torch.Generator(device=dev)
        gen.manual_seed(self.seed)
        lmin, lmax = np.log(self.sigma_min), np.log(smax)

        def dsm_loss(model, xb, g=None):
            sig = torch.exp(lmin + (lmax - lmin) * torch.rand(len(xb), device=dev,
                                                              generator=g))
            eps = torch.randn(xb.shape, device=dev, generator=g)
            xt = self._wrap(xb + sig[:, None] * eps)
            return ((sig[:, None] * self._score(model, xt, sig) + eps) ** 2).sum(1).mean()

        hist = {"epoch": [], "train": [], "val": []}
        vgen = torch.Generator(device=dev)
        for ep in range(self.epochs):
            order = torch.randperm(len(tr), device=dev, generator=gen)
            run = 0.0
            for s in range(steps_per):
                loss = dsm_loss(net, tr[order[s * bs:(s + 1) * bs]], gen)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    for pa, pn in zip(avg.parameters(), net.parameters()):
                        pa.lerp_(pn, 1 - self.ema)
                run += float(loss)
            vgen.manual_seed(12345)          # the same noise draws every epoch
            with torch.no_grad():
                vl = float(sum(dsm_loss(avg, val[i:i + 8192], vgen) * len(val[i:i + 8192])
                               for i in range(0, len(val), 8192)) / len(val))
            hist["epoch"].append(ep + 1)
            hist["train"].append(run / steps_per)
            hist["val"].append(vl)
        self.net = avg.eval()
        self.history = hist

        # choose the evaluation sigma by held-out implicit score matching
        grid = np.exp(np.linspace(lmin, np.log(min(smax, 0.3)), 12))
        if self.sigma_eval == "cv":
            losses = [self._implicit_sm(val[:10_000], float(s)) for s in grid]
            self.sigma_ = float(grid[int(np.argmin(losses))])
            self.history["sigma_grid"] = grid.tolist()
            self.history["sigma_sm_loss"] = losses
        else:
            self.sigma_ = float(self.sigma_min if self.sigma_eval is None
                                else self.sigma_eval)
        self.params["sigma_eval"] = self.sigma_
        self._fitted = True
        return self

    def _implicit_sm(self, xv, sigma):
        """Held-out ``mean(|s|^2/2 + div s)`` at noise level ``sigma``."""
        torch = _torch()
        tot, n = 0.0, 0
        for i in range(0, len(xv), 2048):
            xb = xv[i:i + 2048].clone().requires_grad_(True)
            sig = torch.full((len(xb),), sigma, device=xb.device)
            s = self._score(self.net, xb, sig)
            div = 0.0
            for j in range(xb.shape[1]):
                (g,) = torch.autograd.grad(s[:, j].sum(), xb, retain_graph=True)
                div = div + g[:, j]
            tot += float((0.5 * (s * s).sum(1) + div).detach().sum())
            n += len(xb)
        return tot / n

    def predict_score(self, x):
        """The score at ``sigma_``, the held-out-best noise level."""
        self._check_fitted()
        torch = _torch()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        dev = next(self.net.parameters()).device
        out = np.empty_like(flat)
        with torch.no_grad():
            for s in range(0, len(flat), 200_000):
                xb = torch.as_tensor(flat[s:s + 200_000], dtype=torch.float32, device=dev)
                sig = torch.full((len(xb),), self.sigma_, device=dev)
                out[s:s + 200_000] = self._score(self.net, xb, sig).double().cpu().numpy()
        return out.reshape(lead + (flat.shape[1],))

    def diagnostics(self) -> dict:
        self._check_fitted()
        return {"sigma_eval": self.sigma_,
                "final_val_loss": self.history["val"][-1]}
