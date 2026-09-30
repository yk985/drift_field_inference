"""The trajectory container -- the object every estimator consumes.

A ``Trajectories`` bundle is ``N`` walkers observed at ``T+1`` times spaced by
``dt``. The estimators never touch raw arrays: they ask for
:meth:`Trajectories.pairs`, which returns the ``(x, dx)`` pairs that a
Kramers-Moyal estimator needs, already corrected for periodic wrapping.

Two ``dt``-like quantities live here and confusing them is the classic way to
get a wrong bias curve:

``dt_integration``
    The Euler-Maruyama step actually used to *generate* the path. Simulation
    error, not a property of the data.
``dt`` (``= stride * dt_integration``)
    The interval at which positions were *recorded*. This is the one the
    estimator sees and the one the O(dt) discretisation bias scales with.

Keeping them separate is what lets you sweep the estimator's bias with the
underlying path held essentially exact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field

import numpy as np

from .fields.domain import Box


@dataclass
class Trajectories:
    """Recorded walker paths plus everything needed to interpret them.

    Attributes
    ----------
    x:
        Observed positions, shape ``(N, T+1, d)``. If ``obs_noise > 0`` this is
        the *noisy* observation -- the thing a real experiment would hand you.
    dt:
        Observation interval.
    D:
        Diffusion coefficient used to generate the paths.
    box:
        Domain (carries the periodicity that :meth:`increments` needs).
    obs_noise:
        Standard deviation of the per-coordinate measurement noise.
    x_true:
        The noiseless positions, kept only when ``obs_noise > 0``. This is a
        ground-truth luxury a real experiment does not have -- use it to
        diagnose, never as an estimator input.
    """

    x: np.ndarray
    dt: float
    D: float
    box: Box
    obs_noise: float = 0.0
    x_true: np.ndarray | None = None
    dt_integration: float | None = None
    burn_in: float = 0.0
    field_name: str = ""
    meta: dict = _dc_field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x)
        if self.x.ndim != 3:
            raise ValueError(
                f"x must have shape (n_walkers, n_frames, d), got {self.x.shape}")
        if self.x.shape[-1] != self.box.d:
            raise ValueError(
                f"x has last axis {self.x.shape[-1]} but the box is {self.box.d}D")

    # -- shape -------------------------------------------------------------
    @property
    def n_walkers(self) -> int:
        return self.x.shape[0]

    @property
    def n_frames(self) -> int:
        return self.x.shape[1]

    @property
    def n_transitions(self) -> int:
        """``N * T`` -- the sample size that actually governs estimator error.

        Neither ``N`` nor ``T`` alone predicts the error of a Kramers-Moyal
        estimator; their product does (at fixed ``dt``, and once each walker
        has decorrelated). This is the natural x-axis for the phase diagram.
        """
        return self.n_walkers * (self.n_frames - 1)

    @property
    def d(self) -> int:
        return self.x.shape[2]

    @property
    def duration(self) -> float:
        """Observed time span of one walker."""
        return (self.n_frames - 1) * self.dt

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.dt

    # -- what estimators consume ------------------------------------------
    def increments(self) -> np.ndarray:
        """Displacements ``x(t+dt) - x(t)``, shape ``(N, T, d)``.

        Minimum-image on a periodic box, so a walker wrapping around the edge
        contributes its true small step rather than a box-sized jump.
        """
        return self.box.displacement(self.x[:, :-1], self.x[:, 1:])

    def pairs(self, *, flatten: bool = True):
        """``(x_start, dx)`` -- the regression data for any KM-type estimator.

        Returns arrays of shape ``(N*T, d)`` when ``flatten`` (the default).
        The drift target is ``dx / dt``; the diffusion target is
        ``dx**2 / (2 dt)``.
        """
        x0 = self.x[:, :-1]
        dx = self.increments()
        if flatten:
            return x0.reshape(-1, self.d), dx.reshape(-1, self.d)
        return x0, dx

    def drift_target(self) -> tuple:
        """``(x, dx/dt)`` -- the empirical drift estimate at each observed point.

        Unbiased for ``b(x)`` only as ``dt -> 0``; at finite ``dt`` it carries
        the O(dt) bias the project is about. Enormously noisy pointwise (the
        noise is O(sqrt(2D/dt))), which is exactly why every estimator in the
        ladder is some form of local averaging.
        """
        x0, dx = self.pairs()
        return x0, dx / self.dt

    # -- reshaping ---------------------------------------------------------
    def subsample_time(self, stride: int) -> "Trajectories":
        """Keep every ``stride``-th frame, multiplying ``dt`` by ``stride``.

        The cheap way to run a ``dt`` sweep: simulate once at fine resolution,
        then subsample. Every returned bundle describes the *same* underlying
        paths, so differences across the sweep are purely the estimator's
        discretisation bias and not a different realisation.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")
        out = self.replace(x=self.x[:, ::stride], dt=self.dt * stride)
        if self.x_true is not None:
            out.x_true = self.x_true[:, ::stride]
        return out

    def subsample_walkers(self, n: int, rng=None) -> "Trajectories":
        """Keep ``n`` randomly chosen walkers -- the ``N`` axis of the sweep."""
        if n > self.n_walkers:
            raise ValueError(f"asked for {n} walkers, only have {self.n_walkers}")
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(self.n_walkers, size=n, replace=False)
        out = self.replace(x=self.x[idx])
        if self.x_true is not None:
            out.x_true = self.x_true[idx]
        return out

    def truncate(self, n_frames: int) -> "Trajectories":
        """Keep the first ``n_frames`` frames -- the ``T`` axis of the sweep."""
        out = self.replace(x=self.x[:, :n_frames])
        if self.x_true is not None:
            out.x_true = self.x_true[:, :n_frames]
        return out

    def with_observation_noise(self, sigma: float, rng=None) -> "Trajectories":
        """Add measurement noise to a clean bundle.

        Applied *after* simulation, which is the physically right order: the
        walker follows the true dynamics and the microscope is what is
        imprecise.

        The damage to a drift estimate is mostly **bias, not variance**. A
        noisy position ``x~ = x + eps`` sits, on average, downhill of the true
        one -- Tweedie's formula, ``E[x | x~] = x~ + sigma^2 grad log rho(x~)``
        -- so the increment measured from it points back uphill too little:

            E[ dx~/dt | x~ ]  =  b  +  sigma^2 grad log rho / dt .

        For a gradient field ``grad log rho_ss = b / D``, so every estimator
        that averages ``dx/dt`` converges to ``b (1 + sigma^2 / (D dt))``.
        Measured with local-linear kernel fits on the 2D random field (D = 0.3,
        dt = 1.05e-3), relative to the sigma = 0 fit: 1.080, 1.320, 2.265 at
        sigma = 0.005, 0.01, 0.02, against 1.079, 1.317, 2.268 predicted.
        It is quadratic in ``sigma``, grows as ``dt -> 0`` -- so finer sampling
        makes it worse -- and no choice of averaging removes it, because it is
        in the target. (The variance also rises, by ``2 sigma^2 / dt^2`` per
        component, but that is the smaller effect at these settings.)
        """
        rng = rng or np.random.default_rng(0)
        base = self.x_true if self.x_true is not None else self.x
        noisy = base + rng.normal(0.0, sigma, size=base.shape)
        if self.box.periodic:
            noisy = self.box.wrap(noisy)
        return self.replace(x=noisy, obs_noise=float(sigma), x_true=base)

    def replace(self, **kw) -> "Trajectories":
        d = dict(x=self.x, dt=self.dt, D=self.D, box=self.box,
                 obs_noise=self.obs_noise, x_true=self.x_true,
                 dt_integration=self.dt_integration, burn_in=self.burn_in,
                 field_name=self.field_name, meta=dict(self.meta))
        d.update(kw)
        return Trajectories(**d)

    # -- diagnostics -------------------------------------------------------
    def msd(self) -> np.ndarray:
        """Mean squared displacement from each walker's start, per frame.

        A free sanity check: at short times it must rise as ``2 d D t``
        regardless of the drift, and at long times it must saturate for a
        confined system (or, on a torus, plateau at the box scale). If it does
        not, ``D`` and ``dt`` disagree with what you think you simulated.
        """
        disp = self.box.displacement(self.x[:, :1], self.x)
        return np.mean(np.sum(disp ** 2, axis=-1), axis=0)

    def step_statistics(self) -> dict:
        """Summary numbers for a caption or a sanity check."""
        dx = self.increments()
        step = np.sqrt(np.sum(dx ** 2, axis=-1))
        expected = np.sqrt(2.0 * self.d * self.D * self.dt)
        return {
            "n_walkers": self.n_walkers,
            "n_frames": self.n_frames,
            "n_transitions": self.n_transitions,
            "dt": self.dt,
            "D": self.D,
            "rms_step": float(np.sqrt(np.mean(step ** 2))),
            "rms_step_pure_diffusion": float(expected),
            "max_step": float(step.max()),
            "obs_noise": self.obs_noise,
        }

    def caption(self) -> str:
        """One-line provenance string for a figure caption."""
        bits = [f"N={self.n_walkers}", f"T={self.n_frames - 1}",
                f"dt={self.dt:g}", f"D={self.D:g}"]
        if self.obs_noise:
            bits.append(f"sigma_obs={self.obs_noise:g}")
        if self.field_name:
            bits.insert(0, self.field_name)
        return "  ".join(bits)

    # -- persistence -------------------------------------------------------
    def save(self, path, *, compress: bool = True):
        """Write to ``.npz``. ``compress=False`` for caches: Brownian positions
        barely compress, and zlib on a few hundred MB costs more than it saves."""
        payload = dict(
            x=self.x, lo=self.box.lo, hi=self.box.hi,
            periodic=np.array(self.box.periodic),
            dt=np.array(self.dt), D=np.array(self.D),
            obs_noise=np.array(self.obs_noise),
            burn_in=np.array(self.burn_in),
            dt_integration=np.array(self.dt_integration
                                    if self.dt_integration is not None else -1.0),
            field_name=np.array(self.field_name),
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        if self.x_true is not None:
            payload["x_true"] = self.x_true
        (np.savez_compressed if compress else np.savez)(path, **payload)

    @classmethod
    def load(cls, path) -> "Trajectories":
        z = np.load(path, allow_pickle=False)
        dti = float(z["dt_integration"])
        return cls(
            x=z["x"], dt=float(z["dt"]), D=float(z["D"]),
            box=Box(z["lo"], z["hi"], bool(z["periodic"])),
            obs_noise=float(z["obs_noise"]),
            x_true=z["x_true"] if "x_true" in z.files else None,
            dt_integration=None if dti < 0 else dti,
            burn_in=float(z["burn_in"]), field_name=str(z["field_name"]),
            meta=json.loads(str(z["meta"])),
        )

    def __repr__(self):
        return (f"<Trajectories {self.n_walkers}x{self.n_frames}x{self.d} "
                f"dt={self.dt:g} D={self.D:g} "
                f"{self.n_transitions:,} transitions>")
