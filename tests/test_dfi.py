"""Test suite.

Run with ``pytest -q`` from the project root.

Three groups:

``TestFields`` / ``TestSimulate``
    Guard the ground truth. If these break, nothing downstream means anything,
    because the "true" field or the "true" dynamics would be wrong.

``TestEstimators``
    **The harness for the code you are writing.** Every estimator in the
    registry is fitted to a 1D Ornstein-Uhlenbeck run, where ``b(x) = -kx`` is
    known exactly, and checked against it. Unimplemented methods report as
    skipped, so the suite is green today and turns into real coverage one rung
    at a time. Implement ``BinnedKramersMoyal.fit``, run ``pytest -q -k km``,
    and you get a verdict in about two seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfi import (Box, DoubleWell1D, ornstein_uhlenbeck, coupled_ou_chain,
                 random_grid_field, random_spectral_field, sample_stationary,
                 simulate, suggest_dt)
from dfi.estimators import REGISTRY, OracleDrift, ZeroDrift
from dfi.estimators.base import SnapshotEstimator
from dfi.metrics import decompose_error, drift_error, make_eval_set

RNG = np.random.default_rng(0)


# ==========================================================================
class TestBox:
    def test_minimum_image(self):
        """A wrap-around step must read as small, not box-sized."""
        box = Box.cube(2, 1.0, periodic=True)
        a = np.array([[0.95, 0.0]])
        b = np.array([[-0.95, 0.0]])
        d = box.displacement(a, b)
        assert np.allclose(d, [[0.1, 0.0]]), d

    def test_wrap_roundtrip(self):
        box = Box.cube(3, 2.0, periodic=True)
        x = RNG.uniform(-10, 10, (500, 3))
        w = box.wrap(x)
        assert box.contains(w).all()
        assert np.allclose(box.wrap(w), w)

    def test_reflect_stays_inside(self):
        box = Box.cube(2, 1.0, periodic=False)
        x = RNG.uniform(-6, 6, (2000, 2))
        assert box.contains(box.reflect(x)).all()

    def test_periodic_axes_exclude_endpoint(self):
        """On a torus hi *is* lo; including both would duplicate a cell."""
        box = Box.unit(1, 1.0, periodic=True)
        ax = box.axes(4)[0]
        assert np.allclose(ax, [0.0, 0.25, 0.5, 0.75])


# ==========================================================================
class TestFields:
    @pytest.fixture(scope="class")
    @classmethod
    def field(cls):
        return random_grid_field(d=2, resolution=128, omega=0.7, seed=1,
                                 correlation_length=0.5)

    def test_interpolated_gradient_is_consistent(self, field):
        """grad of the interpolant == interpolant of the grad, to ~1e-4."""
        x = field.box.sample_uniform(2000, RNG)
        eps = 1e-4
        num = np.zeros_like(x)
        for j in range(2):
            s = np.zeros(2); s[j] = eps
            num[:, j] = (field.potential(field.box.wrap(x + s))
                         - field.potential(field.box.wrap(x - s))) / (2 * eps)
        an = field.grad_potential(x)
        assert np.abs(num - an).max() / an.std() < 1e-3

    def test_rotational_part_is_orthogonal_to_gradient(self, field):
        x = field.box.sample_uniform(3000, RNG)
        g, r = field.grad_potential(x), field.rotational_part(x)
        scale = (np.linalg.norm(g, axis=1) * np.linalg.norm(r, axis=1)).mean()
        assert np.abs(np.sum(g * r, axis=1)).max() / scale < 1e-10

    def test_rotational_part_is_divergence_free(self, field):
        x = field.box.sample_uniform(2000, RNG)
        eps = 1e-4
        div = np.zeros(len(x))
        for j in range(2):
            s = np.zeros(2); s[j] = eps
            div += (field.rotational_part(field.box.wrap(x + s))[:, j]
                    - field.rotational_part(field.box.wrap(x - s))[:, j]) / (2 * eps)
        assert np.abs(div).max() / np.abs(field.divergence(x)).std() < 1e-3

    def test_omega_leaves_density_bit_identical(self, field):
        """The identifiability premise, at machine precision."""
        a = field.with_omega(0.0).rho_ss_on_grid(0.3)
        b = field.with_omega(3.0).rho_ss_on_grid(0.3)
        assert np.array_equal(a, b)

    def test_omega_does_change_the_drift(self, field):
        x = field.box.sample_uniform(1000, RNG)
        a = field.with_omega(0.0).drift(x)
        b = field.with_omega(3.0).drift(x)
        assert np.abs(a - b).max() > 0.5 * np.abs(a).std()

    def test_fokker_planck_stationarity(self, field):
        """div(b rho - D grad rho) == 0 for the exact Boltzmann density."""
        from dfi.fields.noise import spectral_gradient
        D = 0.3
        rho = field.rho_ss_on_grid(D)
        J = field.drift_on_grid() * rho[None] - D * spectral_gradient(rho, field.box)
        divJ = sum(spectral_gradient(J[j], field.box)[j] for j in range(2))
        ell = field.meta["correlation_length"]
        assert np.abs(divJ).max() * ell / np.abs(J).max() < 1e-4

    def test_1d_has_no_rotational_component(self):
        """No divergence-free field exists on a line -- physics, not a bug."""
        f = random_grid_field(d=1, resolution=512, omega=2.0, seed=0)
        x = f.box.sample_uniform(200, RNG)
        assert np.abs(f.rotational_part(x)).max() < 1e-12

    def test_spectral_field_gradient_is_analytic(self):
        f = random_spectral_field(d=5, n_features=256, omega=0.5, seed=2)
        x = f.box.sample_uniform(500, RNG)
        eps = 1e-5
        num = np.zeros_like(x)
        for j in range(5):
            s = np.zeros(5); s[j] = eps
            num[:, j] = (f.potential(x + s) - f.potential(x - s)) / (2 * eps)
        assert np.abs(num - f.grad_potential(x)).max() < 1e-5

    def test_spectral_field_is_periodic(self):
        f = random_spectral_field(d=4, n_features=128, seed=3)
        x = f.box.sample_uniform(200, RNG)
        assert np.abs(f.potential(x) - f.potential(x + f.box.length)).max() < 1e-9

    def test_ou_stationary_covariance(self):
        ou = ornstein_uhlenbeck(d=1, k=2.0)
        assert np.isclose(ou.stationary_covariance(0.5)[0, 0], 0.25)

    def test_rotational_ou_keeps_isotropic_covariance(self):
        """Circulation does not change the stationary state of a linear system."""
        ou = ornstein_uhlenbeck(d=2, k=1.0, omega=1.5)
        assert np.allclose(ou.stationary_covariance(0.3), 0.3 * np.eye(2))
        assert not ou.is_gradient

    def test_coupled_chain_is_stiff_and_gradient(self):
        ch = coupled_ou_chain(d=6, k=1.0, coupling=2.0)
        ev = np.linalg.eigvalsh(ch.K)
        assert ch.is_gradient and ev.max() / ev.min() > 5


# ==========================================================================
class TestSimulate:
    def test_ou_recovers_closed_form(self):
        """Stationary variance and the MSD curve, against the exact OU forms.

        Tolerances are *derived*, not guessed. The MSD at each time is a mean
        of ``N`` squared Gaussian displacements, so its relative standard error
        is ``sqrt(2/N)`` -- independent of ``t``, and 2.6% at ``N = 3000``. Any
        pointwise tolerance below about ``3 sqrt(2/N)`` fails on Monte Carlo
        noise alone no matter how correct the integrator is. The deviations are
        also strongly correlated across ``t`` (every point shares one ``x(0)``),
        so averaging along the curve does not help much either.

        The reference is the exact OU curve ``2(D/k)(1 - e^{-kt})``, not the
        free-diffusion line ``2dDt``. The two already differ by 6% at
        ``kt = 0.12``, so testing against the free line out there would measure
        the quality of an approximation rather than the correctness of the code.
        """
        k, D = 2.0, 0.5
        n = 3000
        ou = ornstein_uhlenbeck(d=1, k=k)
        tr = simulate(ou, n_walkers=n, n_steps=2000, dt=2e-3, D=D,
                      burn_in=5.0, x0="center", boundary="none", seed=0,
                      store_dtype=np.float64)
        var = tr.x.var()
        assert np.isclose(var, D / k, rtol=0.05)

        msd, t = tr.msd(), tr.times
        msd_true = 2 * (D / k) * (1 - np.exp(-k * t))
        rtol = 3.0 * np.sqrt(2.0 / n)          # 3 sigma of the MSD estimator

        early = slice(1, 40)
        assert np.allclose(msd[early], msd_true[early], rtol=rtol), (
            f"max relative deviation "
            f"{np.abs(msd[early]/msd_true[early] - 1).max():.4f} exceeds "
            f"{rtol:.4f}")

        # The free-diffusion limit, only where it is actually valid (k t << 1).
        assert np.allclose(msd[1:4], 2 * D * t[1:4], rtol=rtol)

        # Long time: MSD saturates at twice the measured variance.
        assert abs(msd[-500:].mean() - 2 * var) < 3 * (2 * var * np.sqrt(2.0 / n))

    def test_periodic_density_matches_boltzmann(self):
        f = random_grid_field(d=2, resolution=64, seed=3,
                              correlation_length=0.5)
        D = 0.3
        tr = simulate(f, n_walkers=1500, n_steps=1500, dt=suggest_dt(f, D),
                      D=D, burn_in=6.0, seed=1)
        nb = 32
        H, _, _ = np.histogram2d(tr.x[..., 0].ravel(), tr.x[..., 1].ravel(),
                                 bins=nb, range=[[-1, 1], [-1, 1]])
        H /= H.sum()
        rho = f.rho_ss_on_grid(D)
        ref = rho.reshape(nb, 2, nb, 2).mean(axis=(1, 3)); ref /= ref.sum()
        assert 0.5 * np.abs(H - ref).sum() < 0.10

    def test_walkers_stay_in_the_box(self):
        f = random_grid_field(d=2, resolution=64, seed=0)
        tr = simulate(f, n_walkers=200, n_steps=300, dt=1e-3, D=0.3, seed=0)
        assert f.box.contains(tr.x).all()

    def test_increments_are_small_despite_wrapping(self):
        """The minimum-image convention, end to end."""
        f = random_grid_field(d=2, resolution=64, seed=0)
        tr = simulate(f, n_walkers=400, n_steps=400, dt=1e-3, D=0.5, seed=0,
                      check=False)
        step = np.linalg.norm(tr.increments(), axis=-1)
        assert step.max() < 0.25 * f.box.length.min()

    def test_substeps_do_not_change_the_stationary_state(self):
        f = random_grid_field(d=2, resolution=64, seed=2)
        D, dt = 0.3, 4e-3
        v = [simulate(f, n_walkers=800, n_steps=800, dt=dt, D=D, substeps=s,
                      burn_in=6.0, seed=1, check=False).x.std()
             for s in (1, 4)]
        assert abs(v[0] - v[1]) / v[0] < 0.05

    def test_observation_noise_inflates_increment_variance(self):
        """The sigma/dt pathology: extra variance that does not vanish."""
        f = random_grid_field(d=2, resolution=64, seed=0)
        dt = 1e-3
        tr = simulate(f, n_walkers=300, n_steps=500, dt=dt, D=0.3, seed=0)
        sig = 0.01
        noisy = tr.with_observation_noise(sig, np.random.default_rng(0))
        got = noisy.increments().var()
        want = tr.increments().var() + 2 * sig ** 2
        assert np.isclose(got, want, rtol=0.1)

    def test_snapshot_sampler_matches_exact_density(self):
        f = random_grid_field(d=2, resolution=64, seed=3,
                              correlation_length=0.5)
        D = 0.3
        s = sample_stationary(f, 100_000, D, rng=np.random.default_rng(0))
        nb = 32
        H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=nb,
                                 range=[[-1, 1], [-1, 1]])
        H /= H.sum()
        rho = f.rho_ss_on_grid(D)
        ref = rho.reshape(nb, 2, nb, 2).mean(axis=(1, 3)); ref /= ref.sum()
        assert 0.5 * np.abs(H - ref).sum() < 0.09

    def test_roundtrip_save_load(self, tmp_path):
        f = random_grid_field(d=2, resolution=64, seed=0)
        tr = simulate(f, n_walkers=20, n_steps=30, dt=1e-3, D=0.3, seed=0)
        p = tmp_path / "t.npz"
        tr.save(p)
        back = __import__("dfi").Trajectories.load(p)
        assert np.array_equal(tr.x, back.x) and back.dt == tr.dt


# ==========================================================================
class TestMetrics:
    def test_oracle_scores_zero_and_null_scores_one(self):
        """If this fails the harness is broken, not the estimator."""
        f = random_grid_field(d=2, resolution=64, seed=0)
        D = 0.3
        tr = simulate(f, n_walkers=100, n_steps=100, dt=1e-3, D=D, seed=0)
        ev = make_eval_set(f, D, n=4000, seed=1)
        assert drift_error(OracleDrift(f).fit(tr), f, ev)["nrmse"] < 1e-12
        assert np.isclose(drift_error(ZeroDrift().fit(tr), f, ev)["nrmse"], 1.0)

    def test_error_decomposition_isolates_the_missing_circulation(self):
        """A perfect gradient-only estimate must show all its error as
        rotational, and none as gradient. This is the identifiability result
        expressed as two numbers."""
        f = random_grid_field(d=2, resolution=64, seed=0).with_omega(1.5)
        x = f.box.sample_uniform(4000, RNG)
        b_hat = f.gradient_part(x)              # the best a snapshot can do
        parts = decompose_error(f, x, b_hat)
        assert np.abs(parts["e_grad"]).max() < 1e-9
        assert np.sqrt(np.mean(parts["e_rot"] ** 2)) > 0.5 * np.sqrt(
            np.mean(parts["scale_rot"] ** 2))


# ==========================================================================
class TestEstimators:
    """Validation for the estimators you write. Skips what is not done yet."""

    K = 2.0
    D = 0.4

    @pytest.fixture(scope="class")
    @classmethod
    def ou_setup(cls):
        field = ornstein_uhlenbeck(d=1, k=cls.K)
        traj = simulate(field, n_walkers=400, n_steps=4000, dt=2e-3, D=cls.D,
                        burn_in=5.0, boundary="none", seed=0,
                        store_dtype=np.float64)
        ev = make_eval_set(field, cls.D, n=6000, measure="stationary", seed=1,
                           traj=traj)
        return field, traj, ev

    @pytest.mark.parametrize("key", sorted(REGISTRY))
    def test_recovers_linear_drift(self, key, ou_setup):
        """Every estimator must recover ``b(x) = -kx`` on 1D OU.

        This is the floor. A method that cannot do this has a bug, not a
        limitation -- there is nothing here but a straight line through the
        origin, sampled 1.6 million times.
        """
        field, traj, ev = ou_setup
        cls = REGISTRY[key]
        try:
            est = cls(field=field) if cls is OracleDrift else cls()
            if isinstance(est, SnapshotEstimator):
                snaps = sample_stationary(field, 200_000, self.D,
                                          rng=np.random.default_rng(2))
                est.fit(snaps, self.D, box=field.box)
            else:
                try:
                    est.fit(traj, field=field)
                except TypeError:
                    est.fit(traj)
            res = drift_error(est, field, ev)
        except NotImplementedError as exc:
            pytest.skip(f"{key} not implemented yet ({exc})")

        if key == "zero":
            assert np.isclose(res["nrmse"], 1.0)
            return
        assert res["nrmse"] < 0.15, (
            f"{key}: normalised error {res['nrmse']:.3f} on 1D OU. "
            f"cosine={res.get('cosine'):.3f}, scale_ratio="
            f"{res.get('scale_ratio'):.3f}. A low cosine means the shape is "
            f"wrong; a scale_ratio far from 1 usually means a missing 1/dt or "
            f"a factor of 2 in D.")

    @pytest.mark.parametrize("key", ["binned_km", "kernel", "sfi", "nn"])
    def test_beats_the_null_model_on_a_random_field(self, key):
        """A harder check: a 2D random field, where nothing is linear."""
        field = random_grid_field(d=2, resolution=96, seed=4,
                                  correlation_length=0.5)
        D = 0.3
        traj = simulate(field, n_walkers=600, n_steps=2500,
                        dt=suggest_dt(field, D), D=D, burn_in=6.0, seed=0)
        ev = make_eval_set(field, D, n=6000, seed=1, traj=traj)
        try:
            est = REGISTRY[key]()
            est.fit(traj)
            res = drift_error(est, field, ev)
        except NotImplementedError as exc:
            pytest.skip(f"{key} not implemented yet ({exc})")
        assert res["nrmse"] < 0.5, f"{key}: nrmse {res['nrmse']:.3f}"

    @pytest.fixture(scope="class")
    @classmethod
    def sfi_setup(cls):
        from dfi.estimators.basis import BasisProjection
        field = random_grid_field(d=2, resolution=96, seed=4,
                                  correlation_length=0.5)
        D = 0.3
        traj = simulate(field, n_walkers=400, n_steps=2500,
                        dt=suggest_dt(field, D), D=D, burn_in=6.0, seed=0)
        return BasisProjection, field, traj

    def test_sfi_error_prediction_matches_measured_error(self, sfi_setup):
        """The fit's own noise-only error prediction, d p sigma^2 / n.

        Measured on the sampled points -- the average the prediction is for --
        with a basis that spans the field, so no bias enters. With p d = 162
        degrees of freedom one draw of the ratio fluctuates by ~11%; the band
        is three of those.
        """
        BasisProjection, field, traj = sfi_setup
        est = BasisProjection(basis="fourier", degree=25).fit(traj)
        x = traj.x[:, :-1].reshape(-1, 2)[::5]
        mse = np.mean(np.sum((est.predict(x) - field.drift(x)) ** 2, -1))
        ratio = mse / est.error_bound()
        assert 0.66 < ratio < 1.34, f"measured/predicted = {ratio:.3f}"

    def test_sfi_sub_fit_equals_direct_fit(self, sfi_setup):
        """Nested bases: a leading block of the normal equations is the fit.

        On the float64 CPU path, where the claim is exact. The GPU path
        accumulates in float32 blocks whose size depends on the basis, so two
        fits of different sizes agree there only to ~1e-6 relative.
        """
        BasisProjection, field, traj = sfi_setup
        big = BasisProjection(basis="fourier", degree=25, device="cpu").fit(traj)
        small = BasisProjection(basis="fourier", degree=9, device="cpu").fit(traj)
        sub = big.sub_fit(small.n_basis)
        assert np.allclose(sub.coef, small.coef, rtol=1e-6, atol=1e-9)
        assert np.isclose(sub.error_bound(), small.error_bound(), rtol=1e-6)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_cnn_lattice_loss_equals_the_loss_over_transitions(self, periodic):
        """The CNN trains on lattice statistics instead of the transitions.

        For a multilinear grid that is exact, not an approximation: the
        quadratic form over neighbour offsets must reproduce the plain mean
        squared error of the interpolated grid at every transition, on a torus
        and on an open box alike.
        """
        pytest.importorskip("torch")
        import torch
        from dfi.estimators.neural import NeuralDrift
        if periodic:
            field = random_grid_field(d=2, resolution=64, seed=1,
                                      correlation_length=0.5)
        else:
            field = ornstein_uhlenbeck(k=2.0, d=2)
        D = 0.3
        traj = simulate(field, n_walkers=6, n_steps=400,
                        dt=suggest_dt(field, D), D=D, burn_in=1.0, seed=0)
        est = NeuralDrift(arch="cnn", device="cpu")
        est.box = traj.box
        est._lattice(traj, 16)
        G, Y, S, cnt, offsets = est._lattice_stats(
            traj, np.zeros(traj.n_walkers, np.int64), 1)
        grid = torch.as_tensor(np.random.default_rng(3).normal(size=(2, 16, 16)))
        lattice = float((est._quad(grid, G[0], Y[0], offsets) + S[0]) / cnt[0])
        est._grid, est._M = grid.numpy(), 16
        # In float64 throughout: trajectories are stored as float32, and
        # dx / dt taken in float32 rounds at ~1e-7, above this tolerance.
        x, dx = traj.pairs()
        x, y = x.astype(float), dx.astype(float) / traj.dt
        direct = float(np.mean(np.sum((est._grid_predict(x) - y) ** 2, axis=1)))
        assert np.isclose(lattice, direct, rtol=1e-10), (lattice, direct)

    def test_amortised_statistics_respect_the_lattice_symmetries(self):
        """Augmentation is only valid if it commutes with the data pipeline.

        Reflecting (or transposing) the walkers and then splatting must give
        exactly the reflected (transposed) statistics, with the matching vector
        component negated (swapped); and group statistics must add up to the
        whole dataset's. Checked to machine precision.
        """
        torch = pytest.importorskip("torch")
        from dfi.estimators.amortised import (_inverse_dihedral, dihedral,
                                              splat_groups)
        from dfi.trajectories import Trajectories
        field = random_grid_field(d=2, resolution=64, seed=3,
                                  correlation_length=0.5)
        traj = simulate(field, n_walkers=24, n_steps=300,
                        dt=suggest_dt(field, 0.3), D=0.3, burn_in=0.5, seed=0)
        M = 32
        N, Y, S, n = splat_groups(traj, M, groups=np.arange(24) % 4, n_groups=4)
        N1, Y1, S1, n1 = splat_groups(traj, M)
        assert np.allclose(N.sum(0), N1[0], rtol=1e-5)
        assert np.allclose(Y.sum(0), Y1[0], rtol=1e-4, atol=1e-3)
        assert np.isclose(S.sum(), S1[0]) and n.sum() == n1[0]
        stack = torch.as_tensor(np.concatenate([Y1, N1[:, None]], 1))
        for k, transform in ((1, lambda x: x[..., ::-1]),
                             (2, lambda x: x * np.array([-1.0, 1.0]))):
            moved = Trajectories(x=transform(traj.x).copy(), dt=traj.dt,
                                 D=traj.D, box=traj.box)
            Nm, Ym, _, _ = splat_groups(moved, M)
            expect = dihedral(stack, k, [(0, 1)])[0].numpy()
            assert np.allclose(expect[2], Nm[0], atol=1e-4)
            assert np.allclose(expect[:2], Ym[0], rtol=1e-4, atol=1e-2)
        x = torch.randn(2, 5, 8, 8)
        for k in range(8):
            back = _inverse_dihedral(dihedral(x, k, [(0, 1), (2, 3)]), k,
                                     [(0, 1), (2, 3)])
            assert torch.allclose(back, x)

    def test_cnn_beats_the_null_model_on_a_random_field(self):
        pytest.importorskip("torch")
        from dfi.estimators.neural import NeuralDrift
        field = random_grid_field(d=2, resolution=96, seed=4,
                                  correlation_length=0.5)
        D = 0.3
        traj = simulate(field, n_walkers=600, n_steps=2500,
                        dt=suggest_dt(field, D), D=D, burn_in=6.0, seed=0)
        ev = make_eval_set(field, D, n=6000, seed=1, traj=traj)
        est = NeuralDrift(arch="cnn").fit(traj)
        res = drift_error(est, field, ev)
        assert res["nrmse"] < 0.5, f"cnn: nrmse {res['nrmse']:.3f}"

    def test_snapshot_methods_cannot_see_circulation(self):
        """The project's central claim, as an executable assertion.

        A snapshot estimator's error against the *full* drift must grow with
        omega, while its error against the *gradient part* must not. If a
        snapshot method ever scores well against the full drift at large omega,
        something is leaking dynamical information into it.
        """
        base = random_grid_field(d=2, resolution=96, seed=4,
                                 correlation_length=0.5)
        D = 0.3
        errs = {}
        for omega in (0.0, 2.0):
            field = base.with_omega(omega)
            snaps = sample_stationary(field, 200_000, D,
                                      rng=np.random.default_rng(0))
            try:
                est = REGISTRY["kde_score"]()
                est.fit(snaps, D, box=field.box)
                ev = make_eval_set(field, D, n=4000, seed=1)
                errs[omega] = drift_error(est, field, ev)
            except NotImplementedError as exc:
                pytest.skip(f"kde_score not implemented yet ({exc})")
        assert errs[2.0]["nrmse"] > 2 * errs[0.0]["nrmse"]
        assert errs[2.0]["nrmse_grad"] < 1.5 * errs[0.0]["nrmse_grad"] + 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
