"""Closed-form systems -- the ones you unit-test the pipeline against.

Every estimator should be validated here first. If a method cannot recover
``b(x) = -kx`` from an Ornstein-Uhlenbeck process, whose stationary density,
autocorrelation and finite-``dt`` bias are all known analytically, then nothing
it says about a random field is trustworthy.

Systems, in the order the guide asks for them:

- :class:`LinearDrift` -- ``b = -Kx``. Covers 1D OU, the coupled-OU chain, and
  the linear non-equilibrium case in one class, with an exact Gaussian
  stationary density from the Lyapunov equation.
- :class:`DoubleWell1D` -- metastable, non-Gaussian, still a pure gradient.
- :class:`RadialRotational2D` -- gradient + rotational, the identifiability
  demonstrator with an analytic potential.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .base import DriftField
from .domain import Box

__all__ = ["LinearDrift", "ornstein_uhlenbeck", "coupled_ou_chain",
           "DoubleWell1D", "RadialRotational2D"]


class LinearDrift(DriftField):
    """``b(x) = -K x`` for a constant matrix ``K``.

    The Helmholtz split is the matrix split: with ``S = (K + K^T)/2`` and
    ``Q = (K - K^T)/2``, the potential is ``U = x^T S x / 2`` and the
    rotational part is ``-Q x``, which is divergence-free because ``tr Q = 0``.

    The stationary density is Gaussian for *any* ``K`` with eigenvalues in the
    right half-plane, with covariance solving the continuous Lyapunov equation

    .. math::  K \\Sigma + \\Sigma K^T = 2 D I.

    That matters: when ``K`` is asymmetric the density is **not**
    ``exp(-U/D)``, so this class is the honest counterexample to the naive
    Boltzmann reading -- and it still has an exact reference density to score
    a snapshot-based estimator against.
    """

    def __init__(self, K, box: Box | None = None, name: str | None = None):
        K = np.atleast_2d(np.asarray(K, float))
        if K.shape[0] != K.shape[1]:
            raise ValueError("K must be square")
        d = K.shape[0]
        super().__init__(box or Box.cube(d, 4.0, periodic=False), name)
        self.K = K
        self.S = 0.5 * (K + K.T)
        self.Q = 0.5 * (K - K.T)
        eig = np.linalg.eigvals(K)
        if np.any(eig.real <= 0):
            raise ValueError(
                "K has an eigenvalue with non-positive real part, so the "
                "process is not confining and has no stationary density")

    def drift(self, x):
        return -(np.asarray(x, float) @ self.K.T)

    def potential(self, x):
        x = np.asarray(x, float)
        return 0.5 * np.einsum("...i,ij,...j->...", x, self.S, x)

    def grad_potential(self, x):
        return np.asarray(x, float) @ self.S.T

    def rotational_part(self, x):
        if self.is_gradient:
            return np.zeros_like(np.asarray(x, float))
        return -(np.asarray(x, float) @ self.Q.T)

    def divergence(self, x, eps=1e-5):
        return np.full(np.shape(x)[:-1], -np.trace(self.K))

    @property
    def has_potential(self):
        return True

    @property
    def is_gradient(self):
        return bool(np.allclose(self.Q, 0.0))

    @property
    def boltzmann_exact(self):
        # Only when K is symmetric. For asymmetric K the density is still
        # Gaussian, but with a covariance that is not D * S^-1.
        return self.is_gradient

    # -- exact stationary state (valid for asymmetric K too) ---------------
    def stationary_covariance(self, D: float) -> np.ndarray:
        """``Sigma`` solving ``K Sigma + Sigma K^T = 2 D I``."""
        return linalg.solve_continuous_lyapunov(self.K, 2.0 * D * np.eye(self.d))

    def log_rho_ss(self, x, D):
        S = self.stationary_covariance(D)
        P = np.linalg.inv(S)
        x = np.asarray(x, float)
        return -0.5 * np.einsum("...i,ij,...j->...", x, P, x)

    def score_ss(self, x, D):
        P = np.linalg.inv(self.stationary_covariance(D))
        return -(np.asarray(x, float) @ P.T)

    def stationary_current(self, x, D):
        rho = np.exp(self.log_rho_ss(x, D))
        v = self.drift(x) - D * self.score_ss(x, D)
        return v * rho[..., None]

    def autocorrelation(self, t, D: float) -> np.ndarray:
        """``<x(0) x(t)^T>`` at stationarity: ``exp(-Kt) Sigma``.

        Use it to check that a simulation has actually equilibrated before
        estimating anything from it.
        """
        return linalg.expm(-self.K * float(t)) @ self.stationary_covariance(D)

    def relaxation_time(self) -> float:
        """Slowest timescale ``1 / min Re(eig K)`` -- the burn-in you owe."""
        return float(1.0 / np.min(np.linalg.eigvals(self.K).real))

    # -- torch -------------------------------------------------------------
    @property
    def supports_torch(self):
        return True

    def drift_torch(self, x):
        import torch
        K = torch.as_tensor(self.K, dtype=x.dtype, device=x.device)
        return -(x @ K.T)

    def summary(self):
        kind = "symmetric" if self.is_gradient else "asymmetric"
        return f"{super().summary()} | linear, K {kind}"


def ornstein_uhlenbeck(d: int = 1, k: float = 1.0, *, omega: float = 0.0,
                       box: Box | None = None) -> LinearDrift:
    """Isotropic OU, optionally with a rotational (asymmetric) component.

    ``b = -k x + omega * A x`` with ``A`` the canonical rotation in the first
    two coordinates. The workhorse validation system: with ``omega = 0`` the
    stationary density is ``N(0, (D/k) I)`` and the drift is exactly linear, so
    every estimator has a closed form to be checked against.
    """
    K = k * np.eye(d)
    if omega != 0.0:
        if d < 2:
            raise ValueError("a rotational component needs d >= 2")
        A = np.zeros((d, d))
        A[0, 1], A[1, 0] = -1.0, 1.0
        K = K - omega * k * A
    width = 4.0 * np.sqrt(1.0 / k)
    return LinearDrift(K, box or Box.cube(d, width, periodic=False),
                       name=f"OU-{d}D" + (f"-rot{omega:g}" if omega else ""))


def coupled_ou_chain(d: int = 6, k: float = 1.0, coupling: float = 2.0,
                     *, periodic_chain: bool = False,
                     box: Box | None = None) -> LinearDrift:
    """A chain of harmonically coupled OU coordinates.

    ``U = k/2 sum x_i^2 + c/2 sum (x_i - x_{i+1})^2``, so ``K = k I + c L`` with
    ``L`` the chain graph Laplacian. Symmetric, hence a pure gradient system,
    but with a stiff spectrum -- the fast and slow modes relax on very
    different timescales, which is what makes moderate-``d`` estimation hard
    for reasons that have nothing to do with the drift being nonlinear.
    """
    L = np.zeros((d, d))
    for i in range(d - 1):
        L[i, i] += 1.0
        L[i + 1, i + 1] += 1.0
        L[i, i + 1] -= 1.0
        L[i + 1, i] -= 1.0
    if periodic_chain and d > 2:
        L[0, 0] += 1.0
        L[-1, -1] += 1.0
        L[0, -1] -= 1.0
        L[-1, 0] -= 1.0
    K = k * np.eye(d) + coupling * L
    return LinearDrift(K, box or Box.cube(d, 4.0 / np.sqrt(k), periodic=False),
                       name=f"coupled-OU-{d}D")


class DoubleWell1D(DriftField):
    """``U(x) = a x^4 - b x^2``, so ``b(x) = -(4 a x^3 - 2 b x)``.

    Two wells at ``x = +-sqrt(b/(2a))`` separated by a barrier of height
    ``b^2/(4a)``. Metastable when the barrier is tall compared with ``D``:
    the walker spends long stretches in one well and rarely crosses, which
    starves the estimator of samples exactly where the drift is largest. That
    sampling asymmetry -- not the nonlinearity -- is what breaks naive binning
    here, and it is worth saying so in the write-up.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, box: Box | None = None):
        self.a, self.b = float(a), float(b)
        width = 2.5 * np.sqrt(b / (2.0 * a)) if b > 0 else 2.5
        super().__init__(box or Box(np.array([-width]), np.array([width]),
                                    periodic=False), name="double-well-1D")

    def potential(self, x):
        x = np.asarray(x, float)[..., 0]
        return self.a * x ** 4 - self.b * x ** 2

    def grad_potential(self, x):
        x = np.asarray(x, float)
        return 4.0 * self.a * x ** 3 - 2.0 * self.b * x

    def drift(self, x):
        return -self.grad_potential(x)

    def divergence(self, x, eps=1e-5):
        x = np.asarray(x, float)[..., 0]
        return -(12.0 * self.a * x ** 2 - 2.0 * self.b)

    @property
    def has_potential(self):
        return True

    @property
    def is_gradient(self):
        return True

    @property
    def well_positions(self):
        return np.array([-1.0, 1.0]) * np.sqrt(self.b / (2.0 * self.a))

    @property
    def barrier_height(self) -> float:
        return self.b ** 2 / (4.0 * self.a)

    def kramers_rate(self, D: float) -> float:
        """Kramers escape rate from one well, high-barrier asymptotics.

        ``r = (sqrt(|U''(0)| U''(x_min)) / 2pi) exp(-dU / D)``. Compare it with
        the observed number of well-to-well crossings to check that a
        trajectory is long enough to have explored both wells -- if it is not,
        no estimator can be expected to get the barrier region right.
        """
        curv_min = 8.0 * self.b
        curv_max = 2.0 * self.b
        return float(np.sqrt(curv_min * curv_max) / (2.0 * np.pi)
                     * np.exp(-self.barrier_height / D))

    @property
    def supports_torch(self):
        return True

    def drift_torch(self, x):
        return -(4.0 * self.a * x ** 3 - 2.0 * self.b * x)

    def summary(self):
        return (f"{super().summary()} | wells at +-"
                f"{self.well_positions[1]:.2f}, barrier {self.barrier_height:.2f}")


class RadialRotational2D(DriftField):
    """``b = -grad U(r) + omega (-y, x)`` with a radially symmetric potential.

    The textbook identifiability demonstrator. Because ``U`` depends only on
    ``r``, the gradient is radial while the rotational term is azimuthal, so
    the two are orthogonal everywhere and the rotational term is
    divergence-free. Hence ``rho_ss = exp(-U/D)/Z`` **exactly, for every**
    ``omega``: the snapshot distribution is literally identical at
    ``omega = 0`` and ``omega = 5``, while the trajectories look completely
    different. Sweeping ``omega`` and watching snapshot-based recovery flatline
    is the headline experiment of the project.

    Two potentials: ``'mexican'`` (a ring-shaped minimum at ``r0``, which puts
    the walkers where the rotation is fastest) and ``'quadratic'``.
    """

    def __init__(self, omega: float = 1.0, *, shape: str = "mexican",
                 a: float = 1.0, r0: float = 1.0, box: Box | None = None):
        self.omega = float(omega)
        self.shape = shape
        self.a, self.r0 = float(a), float(r0)
        width = 2.2 * r0 if shape == "mexican" else 4.0 / np.sqrt(a)
        super().__init__(box or Box.cube(2, width, periodic=False),
                         name=f"radial-rot-2D-omega{omega:g}")

    def _r2(self, x):
        return np.sum(np.asarray(x, float) ** 2, axis=-1)

    def potential(self, x):
        r2 = self._r2(x)
        if self.shape == "mexican":
            return self.a * (r2 - self.r0 ** 2) ** 2
        return 0.5 * self.a * r2

    def grad_potential(self, x):
        x = np.asarray(x, float)
        r2 = self._r2(x)
        if self.shape == "mexican":
            return (4.0 * self.a * (r2 - self.r0 ** 2))[..., None] * x
        return self.a * x

    def rotational_part(self, x):
        x = np.asarray(x, float)
        return self.omega * np.stack([-x[..., 1], x[..., 0]], axis=-1)

    def drift(self, x):
        return -self.grad_potential(x) + self.rotational_part(x)

    def divergence(self, x, eps=1e-5):
        r2 = self._r2(x)
        if self.shape == "mexican":
            return -(4.0 * self.a * (2.0 * r2 - 2.0 * self.r0 ** 2) + 8.0 * self.a * r2)
        return np.full(r2.shape, -2.0 * self.a)

    @property
    def has_potential(self):
        return True

    @property
    def is_gradient(self):
        return self.omega == 0.0

    @property
    def boltzmann_exact(self):
        return True  # radial U makes grad and rotation orthogonal everywhere

    @property
    def supports_torch(self):
        return True

    def drift_torch(self, x):
        import torch
        r2 = (x ** 2).sum(-1, keepdim=True)
        if self.shape == "mexican":
            g = 4.0 * self.a * (r2 - self.r0 ** 2) * x
        else:
            g = self.a * x
        rot = self.omega * torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        return -g + rot

    def summary(self):
        return f"{super().summary()} | {self.shape} U | omega={self.omega:g}"
