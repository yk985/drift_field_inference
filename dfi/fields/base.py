"""The drift-field interface every system in this project implements.

A ``DriftField`` is the *ground truth* of an experiment: it answers what ``b(x)``
is at any point, and -- crucially -- it knows its own Helmholtz decomposition,

.. math::  b(x) = -\\nabla U(x) + b_{rot}(x),\\qquad \\nabla\\cdot b_{rot} = 0.

Keeping that split exact rather than estimated is what lets us pose the
identifiability question sharply: a snapshot-only estimator can recover the
first term and provably cannot recover the second, and here we know both, so
"provably cannot" becomes a number on a plot.

The construction used by the random fields (see ``random_field.py``) goes one
step further. Take a constant antisymmetric matrix ``A`` and set

.. math::  b(x) = -\\nabla U(x) + \\omega\\, A \\nabla U(x).

Then two identities hold *pointwise and exactly*:

- ``div(A grad U) = A_ij d_i d_j U = 0`` -- antisymmetric contracted with the
  symmetric Hessian. So the second term is divergence-free: it is a genuine
  rotational component.
- ``grad U . A grad U = 0`` -- antisymmetric quadratic form. So the rotational
  term is everywhere orthogonal to ``grad rho_ss``.

Together these make the Boltzmann density *exactly* stationary for any
``omega``: substituting ``rho = exp(-U/D)`` into the Fokker-Planck equation
leaves the residual current ``J = omega (A grad U) rho``, which is divergence
free. So sweeping ``omega`` from 0 upward changes the dynamics drastically while
leaving the stationary density bit-for-bit unchanged. That is the cleanest
possible demonstration of the identifiability gap, and it works in any
dimension (in 2D, ``A = [[0,-1],[1,0]]`` is exactly the perpendicular gradient).

Note that in ``d = 1`` every antisymmetric matrix is zero: there is no
divergence-free field on a line, so every 1D drift is a gradient. That is
physics, not a limitation of the code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .domain import Box


class DriftField(ABC):
    """Base class for a drift field ``b(x)`` on a box.

    Subclasses must provide :meth:`drift`. Everything else has a working
    default (numerical where it has to be), and subclasses override whatever
    they can do exactly.
    """

    name: str = "field"

    def __init__(self, box: Box, name: str | None = None):
        self.box = box
        if name is not None:
            self.name = name

    # -- required ----------------------------------------------------------
    @abstractmethod
    def drift(self, x: np.ndarray) -> np.ndarray:
        """``b(x)`` for points ``x`` of shape ``(..., d)``; returns ``(..., d)``."""

    def __call__(self, x):
        return self.drift(x)

    # -- structure ---------------------------------------------------------
    @property
    def d(self) -> int:
        return self.box.d

    @property
    def has_potential(self) -> bool:
        """Whether ``U(x)`` is available in closed form."""
        return False

    @property
    def is_gradient(self) -> bool:
        """Whether ``b = -grad U`` exactly (no rotational component)."""
        return False

    @property
    def boltzmann_exact(self) -> bool:
        """Whether ``rho_ss ∝ exp(-U/D)`` holds exactly.

        True for gradient fields, and also for the ``omega A grad U``
        construction above -- which is the whole point of that construction.
        """
        return self.is_gradient

    def potential(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no closed-form potential")

    def grad_potential(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no closed-form grad U")

    def gradient_part(self, x: np.ndarray) -> np.ndarray:
        """The curl-free component ``-grad U``."""
        return -self.grad_potential(x)

    def rotational_part(self, x: np.ndarray) -> np.ndarray:
        """The divergence-free component ``b + grad U``."""
        if self.is_gradient:
            return np.zeros_like(np.asarray(x, float))
        return self.drift(x) - self.gradient_part(x)

    def helmholtz(self, x: np.ndarray) -> tuple:
        """``(gradient_part, rotational_part)`` evaluated at ``x``."""
        return self.gradient_part(x), self.rotational_part(x)

    # -- stationary state --------------------------------------------------
    def log_rho_ss(self, x: np.ndarray, D: float) -> np.ndarray:
        """``log rho_ss(x)`` up to an additive constant.

        Only meaningful when :attr:`boltzmann_exact`; raises otherwise, rather
        than silently handing back a wrong reference for a score-matching
        benchmark.
        """
        if not self.boltzmann_exact:
            raise NotImplementedError(
                f"{type(self).__name__} has no closed-form stationary density; "
                "estimate it from a long simulation instead"
            )
        return -self.potential(x) / D

    def score_ss(self, x: np.ndarray, D: float) -> np.ndarray:
        """``grad log rho_ss(x)``.

        For a gradient field this equals ``b(x)/D`` -- the identity that makes
        a score model a physical drift estimator. When a rotational component
        is present the identity breaks, and the difference is exactly what a
        snapshot-only method cannot see.
        """
        if not self.boltzmann_exact:
            raise NotImplementedError(
                f"{type(self).__name__} has no closed-form score")
        return -self.grad_potential(x) / D

    def stationary_current(self, x: np.ndarray, D: float) -> np.ndarray:
        """Probability current ``J = b rho - D grad rho`` up to the same
        normalisation as ``rho_ss``.

        Reduces to ``b_rot * rho`` when the Boltzmann form is exact, so it is
        zero exactly when the field is a gradient -- the microscopic signature
        of detailed balance.
        """
        if not self.boltzmann_exact:
            raise NotImplementedError(
                f"{type(self).__name__} has no closed-form current")
        rho = np.exp(self.log_rho_ss(x, D))
        return self.rotational_part(x) * rho[..., None]

    # -- differential diagnostics -----------------------------------------
    def divergence(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """``div b`` by central differences (subclasses override when exact)."""
        x = np.asarray(x, float)
        out = np.zeros(x.shape[:-1])
        for j in range(self.d):
            step = np.zeros(self.d)
            step[j] = eps
            out += (self.drift(x + step)[..., j]
                    - self.drift(x - step)[..., j]) / (2.0 * eps)
        return out

    def typical_scale(self, n: int = 4096, seed: int = 0) -> float:
        """RMS drift magnitude over the box -- the natural unit for errors.

        Reporting a drift error as a fraction of this makes numbers comparable
        across systems and dimensions.
        """
        rng = np.random.default_rng(seed)
        x = self.box.sample_uniform(n, rng)
        return float(np.sqrt(np.mean(np.sum(self.drift(x) ** 2, axis=-1))))

    def concentration(self, D: float, n: int = 20000, seed: int = 0) -> dict:
        """How tightly does ``rho_ss`` concentrate at this ``D``?

        Returns the spread of ``U/D`` and the resulting density dynamic range.
        This is the number to check before choosing ``D`` for an experiment:

        - ``range`` below ~10: the walkers cover the whole domain, every region
          is well sampled, and the estimator comparison is about the method
          rather than about sampling. Good for the phase diagram.
        - ``range`` above ~1000: the walkers sit in one basin and most of the
          field is never visited. Interesting physics (metastability), but any
          "error over the box" is then dominated by extrapolation.

        Choosing ``D`` by eye from a picture is how people end up comparing
        estimators on a dataset that never visited most of the field.
        """
        if not self.boltzmann_exact:
            raise NotImplementedError(
                f"{type(self).__name__} has no closed-form stationary density")
        rng = np.random.default_rng(seed)
        x = self.box.sample_uniform(n, rng)
        u = self.potential(x) / D
        u = u - u.min()
        return {"u_over_D_std": float(u.std()),
                "u_over_D_range": float(u.max()),
                "density_range": float(np.exp(min(u.max(), 700.0)))}

    # -- optional torch path ----------------------------------------------
    @property
    def supports_torch(self) -> bool:
        """Whether :meth:`drift_torch` is implemented (enables the GPU path)."""
        return False

    def drift_torch(self, x):
        """``b(x)`` for a torch tensor, for the GPU integrator."""
        raise NotImplementedError(
            f"{type(self).__name__} has no torch path; simulate with "
            "backend='numpy'"
        )

    # -- presentation ------------------------------------------------------
    def summary(self) -> str:
        bits = [f"{type(self).__name__}", self.box.describe()]
        if self.is_gradient:
            bits.append("gradient (equilibrium)")
        elif self.boltzmann_exact:
            bits.append("gradient + rotational, Boltzmann exact")
        else:
            bits.append("non-gradient")
        return " | ".join(bits)

    def __repr__(self):
        return f"<{self.summary()}>"


def random_antisymmetric(d: int, rng: np.random.Generator) -> np.ndarray:
    """A random constant antisymmetric matrix, normalised so that
    ``|A v| ~ |v|`` on average.

    Normalisation: for isotropic ``v``, ``E|Av|^2 / E|v|^2 = ||A||_F^2 / d``, so
    scaling by ``sqrt(d)/||A||_F`` puts the rotational term on the same footing
    as the gradient term. That makes ``omega`` directly readable as the
    rotational-to-gradient amplitude ratio rather than an arbitrary knob.

    In ``d = 1`` this is the zero matrix, correctly: a 1D drift is always a
    gradient.
    """
    if d == 1:
        return np.zeros((1, 1))
    if d == 2:
        # The canonical perpendicular-gradient rotation, so 2D pictures match
        # the textbook omega * (-y, x) intuition instead of a random rotation.
        return np.array([[0.0, -1.0], [1.0, 0.0]])
    m = rng.standard_normal((d, d))
    a = 0.5 * (m - m.T)
    frob = np.linalg.norm(a)
    if frob < 1e-12:  # pragma: no cover - measure zero
        return a
    return a * np.sqrt(d) / frob
