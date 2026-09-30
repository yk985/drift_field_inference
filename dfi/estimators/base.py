"""The estimator interface.

Everything downstream -- metrics, sweeps, phase diagrams, animations of an
estimate converging -- talks to estimators only through this interface. Write a
new method by subclassing, and the entire comparison machinery works on it
without modification.

Two families, deliberately kept apart, because the difference between them is
the scientific point of the project:

:class:`TrajectoryEstimator`
    Sees ``(x, dx)`` pairs -- positions *and* time ordering. Can in principle
    recover the whole drift, gradient and rotational parts alike.

:class:`SnapshotEstimator`
    Sees only unordered positions drawn from ``rho_ss``. Can recover
    ``-D grad log rho_ss``, which equals the drift **only** for a gradient
    field. On a system with a rotational component it is not that the method is
    weak -- the information is absent from the data. Keeping the two base
    classes separate makes that impossible to blur.

The contract
------------
``fit(data)`` then ``predict(x) -> (..., d)``. ``predict`` must accept any
batch shape and be safe to call on points the fit never saw; use
:meth:`DriftEstimator.support` to say where the answer is trustworthy, rather
than silently extrapolating.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..trajectories import Trajectories

__all__ = ["DriftEstimator", "TrajectoryEstimator", "SnapshotEstimator",
           "ZeroDrift", "OracleDrift", "REGISTRY", "register", "build"]

REGISTRY: dict = {}


def register(cls):
    """Class decorator: make an estimator constructible by name in sweeps."""
    REGISTRY[cls.key] = cls
    return cls


def build(key: str, **kwargs) -> "DriftEstimator":
    if key not in REGISTRY:
        raise KeyError(f"unknown estimator {key!r}; have {sorted(REGISTRY)}")
    return REGISTRY[key](**kwargs)


class DriftEstimator(ABC):
    """Base class for anything that estimates ``b(x)``."""

    key: str = "base"
    label: str = "Estimator"
    #: Categorical palette slot, so a method keeps its color across every
    #: figure in the project. Color follows the entity, never its rank.
    color_slot: int = 0
    #: Does this method use time ordering? Set False for snapshot methods.
    uses_dynamics: bool = True

    def __init__(self, **params):
        self.params = params
        self._fitted = False

    # -- required ----------------------------------------------------------
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Estimated drift at ``x``, shape ``(..., d)`` in and out."""

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before predicting")

    # -- optional ----------------------------------------------------------
    def predict_diffusion(self, x: np.ndarray):
        """Estimated ``D(x)`` from the second Kramers-Moyal moment.

        Optional. Implement it if your method estimates the noise amplitude as
        well as the drift -- it is a useful consistency check, since a badly
        chosen ``dt`` shows up as a biased ``D`` before it shows up as a biased
        ``b``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not estimate the diffusion coefficient")

    def support(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask: where is this estimate actually supported by data?

        Default is "everywhere", which is the honest answer only for global
        parametric fits. A binned or kernel method should return False where
        the local sample count is too low, so the error plots can separate
        "this method is wrong here" from "nothing ever visited here".
        """
        return np.ones(np.shape(x)[:-1], dtype=bool)

    def uncertainty(self, x: np.ndarray):
        """Per-point standard error of the estimate, if the method has one.

        Worth implementing for the classical estimators, where it is available
        in closed form -- an error bar you derived is a far stronger result
        than an error you measured against a ground truth you would not have in
        a real experiment.
        """
        raise NotImplementedError(
            f"{type(self).__name__} provides no uncertainty estimate")

    # -- presentation ------------------------------------------------------
    def describe(self) -> str:
        if not self.params:
            return self.label
        bits = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.label} ({bits})"

    def __repr__(self):
        return f"<{self.describe()}{'' if self._fitted else ' [unfitted]'}>"


class TrajectoryEstimator(DriftEstimator):
    """Estimators that consume ``(x, dx)`` pairs from trajectories."""

    uses_dynamics = True

    @abstractmethod
    def fit(self, traj: Trajectories) -> "TrajectoryEstimator":
        """Fit to a trajectory bundle. Return ``self``."""

    @staticmethod
    def regression_data(traj: Trajectories):
        """``(x, y)`` where ``y = dx/dt`` is the Kramers-Moyal drift target.

        Every method in the ladder regresses this same ``y`` on ``x``; they
        differ only in how they average it. ``y`` is a wildly noisy pointwise
        estimate -- its per-coordinate standard deviation is
        ``sqrt(2D/dt)``, which for typical settings is an order of magnitude
        larger than ``|b|`` itself. That is not a defect to be fixed; it is why
        local averaging is the whole game.
        """
        return traj.drift_target()

    @staticmethod
    def noise_level(traj: Trajectories) -> float:
        """``sqrt(2 D / dt)``: the per-coordinate noise on each raw target."""
        return float(np.sqrt(2.0 * traj.D / traj.dt))


class SnapshotEstimator(DriftEstimator):
    """Estimators that consume unordered samples from ``rho_ss``.

    The recoverable object is the score ``s(x) = grad log rho_ss(x)``, and the
    drift follows only under the equilibrium assumption ``b = D s``. Implement
    :meth:`predict_score` and the default :meth:`predict` multiplies by ``D``;
    that way the assumption is written down in one place instead of being
    smuggled in.
    """

    uses_dynamics = False

    def __init__(self, **params):
        super().__init__(**params)
        self.D: float | None = None

    @abstractmethod
    def fit(self, samples: np.ndarray, D: float, box=None) -> "SnapshotEstimator":
        """Fit to ``samples`` of shape ``(n, d)`` drawn from ``rho_ss``."""

    @abstractmethod
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Estimated ``grad log rho_ss(x)``."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.D is None:
            raise RuntimeError("fit() must record D before predict()")
        return self.D * self.predict_score(x)


# --------------------------------------------------------------------------
# Calibration baselines -- not methods, references
# --------------------------------------------------------------------------

@register
class ZeroDrift(TrajectoryEstimator):
    """Predicts ``b = 0`` everywhere. The null model.

    Its normalised error is 1.0 by construction, which is what makes it worth
    plotting: any method scoring near 1.0 has learned nothing, and a method
    scoring *above* 1.0 is actively worse than saying "no idea". Both happen at
    small sample sizes, and a reader cannot tell without this line on the plot.
    """

    key = "zero"
    label = "Zero drift (null model)"
    color_slot = 7

    def fit(self, traj):
        self._d = traj.d
        self._fitted = True
        return self

    def predict(self, x):
        self._check_fitted()
        return np.zeros_like(np.asarray(x, float))


@register
class OracleDrift(TrajectoryEstimator):
    """Returns the true field. The noise floor, not a method.

    Useful for exactly one thing: verifying that the evaluation harness itself
    reports ~0 error. If the oracle scores anything other than machine
    precision, the bug is in the metric or the evaluation grid, not in an
    estimator -- run this before believing any comparison.
    """

    key = "oracle"
    label = "Oracle (true field)"
    color_slot = 5

    def __init__(self, field=None, **params):
        super().__init__(**params)
        self.field = field

    def fit(self, traj, field=None):
        if field is not None:
            self.field = field
        if self.field is None:
            raise ValueError("OracleDrift needs the ground-truth field")
        self._fitted = True
        return self

    def predict(self, x):
        self._check_fitted()
        return self.field.drift(x)
