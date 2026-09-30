"""The spatial domain: an axis-aligned box, periodic or not.

The one subtle piece here is :meth:`Box.displacement`. On a torus a walker that
steps off the right edge reappears on the left, so the raw difference
``x[t+1] - x[t]`` is a huge jump across the whole box. Feeding that to a
Kramers-Moyal estimator produces enormous spurious drift at the boundary. The
minimum-image convention fixes it: take the representative of the displacement
that is shortest, which for a step much smaller than the box is the physical
one. *Every* increment in this project goes through that method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box ``[lo, hi)`` in ``d`` dimensions.

    Parameters
    ----------
    lo, hi:
        Corner coordinates, shape ``(d,)``.
    periodic:
        If True the box is a torus: walkers wrap, and displacements use the
        minimum-image convention.
    """

    lo: np.ndarray
    hi: np.ndarray
    periodic: bool = True

    def __post_init__(self):
        object.__setattr__(self, "lo", np.atleast_1d(np.asarray(self.lo, float)))
        object.__setattr__(self, "hi", np.atleast_1d(np.asarray(self.hi, float)))
        if self.lo.shape != self.hi.shape:
            raise ValueError("lo and hi must have the same shape")
        if np.any(self.hi <= self.lo):
            raise ValueError("every hi must exceed its lo")

    # -- basic geometry ----------------------------------------------------
    @property
    def d(self) -> int:
        return int(self.lo.size)

    @property
    def length(self) -> np.ndarray:
        """Side lengths, shape ``(d,)``."""
        return self.hi - self.lo

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def volume(self) -> float:
        return float(np.prod(self.length))

    @property
    def extent(self) -> tuple:
        """``(x0, x1, y0, y1)`` for ``imshow``. 2D only."""
        if self.d != 2:
            raise ValueError("extent is only defined for d == 2")
        return (self.lo[0], self.hi[0], self.lo[1], self.hi[1])

    @classmethod
    def cube(cls, d: int, half_width: float = 1.0, periodic: bool = True) -> "Box":
        """Centered cube ``[-half_width, half_width]^d``."""
        h = float(half_width)
        return cls(np.full(d, -h), np.full(d, h), periodic)

    @classmethod
    def unit(cls, d: int, length: float = 1.0, periodic: bool = True) -> "Box":
        """Box ``[0, length)^d``."""
        return cls(np.zeros(d), np.full(d, float(length)), periodic)

    # -- the two operations that matter ------------------------------------
    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Fold positions back into the box (identity if not periodic)."""
        if not self.periodic:
            return x
        return self.lo + np.mod(x - self.lo, self.length)

    def displacement(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``b - a``, minimum-image on a torus.

        This is the correct increment for a Kramers-Moyal estimator: it is the
        displacement the walker actually made, not the one implied by comparing
        wrapped coordinates. Ambiguous (and therefore wrong) only for steps
        longer than half a box side, which never happens for a stable
        Euler-Maruyama step.
        """
        delta = np.asarray(b) - np.asarray(a)
        if not self.periodic:
            return delta
        L = self.length
        return delta - L * np.round(delta / L)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo, self.hi)

    def reflect(self, x: np.ndarray) -> np.ndarray:
        """Fold positions into the box by mirroring at the walls.

        Implements a reflecting boundary via the triangle-wave map, which is
        the correct image construction for a specular wall and handles
        multiple reflections in one step.
        """
        L = self.length
        y = np.mod(x - self.lo, 2.0 * L)
        y = np.where(y > L, 2.0 * L - y, y)
        return self.lo + y

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the box."""
        return np.all((x >= self.lo) & (x <= self.hi), axis=-1)

    # -- sampling and grids ------------------------------------------------
    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` uniform points in the box, shape ``(n, d)``."""
        return rng.uniform(self.lo, self.hi, size=(n, self.d))

    def axes(self, shape) -> list:
        """Per-dimension coordinate vectors for a grid of the given shape.

        Periodic boxes use ``endpoint=False`` -- on a torus ``hi`` *is* ``lo``,
        so including both would duplicate a cell and break the FFT.
        """
        shape = self._as_shape(shape)
        return [np.linspace(self.lo[i], self.hi[i], shape[i],
                            endpoint=not self.periodic)
                for i in range(self.d)]

    def spacing(self, shape) -> np.ndarray:
        """Grid spacing per dimension for a grid of the given shape."""
        shape = self._as_shape(shape)
        n = np.asarray(shape, float)
        return self.length / (n if self.periodic else n - 1)

    def meshgrid(self, shape):
        """``d`` coordinate arrays of the grid shape, in ``'ij'`` indexing."""
        return np.meshgrid(*self.axes(shape), indexing="ij")

    def grid_points(self, shape) -> np.ndarray:
        """Flattened grid coordinates, shape ``(prod(shape), d)``."""
        mg = self.meshgrid(shape)
        return np.stack([m.ravel() for m in mg], axis=-1)

    def _as_shape(self, shape) -> tuple:
        if np.isscalar(shape):
            return (int(shape),) * self.d
        shape = tuple(int(s) for s in shape)
        if len(shape) != self.d:
            raise ValueError(f"shape {shape} does not match dimension {self.d}")
        return shape

    def to_index(self, x: np.ndarray, shape) -> np.ndarray:
        """Map coordinates to continuous grid-index space, shape ``(d, ...)``.

        This is the layout ``scipy.ndimage.map_coordinates`` wants.
        """
        shape = self._as_shape(shape)
        h = self.spacing(shape)
        idx = (np.asarray(x) - self.lo) / h
        return np.moveaxis(idx, -1, 0)

    def describe(self) -> str:
        kind = "periodic" if self.periodic else "open"
        return (f"{self.d}D {kind} box "
                + " x ".join(f"[{a:g}, {b:g}]" for a, b in zip(self.lo, self.hi)))
