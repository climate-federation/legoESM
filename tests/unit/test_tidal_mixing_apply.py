"""Unit tests for the tidal-mixing tracer-apply helper."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.coupler.tidal_mixing_apply import (
    _build_tridiag,
    _thomas_solve_columns,
    apply_tidal_mixing_step,
)


class _FakeState:
    """Stub matching the lat-lon C-grid state surface."""

    def __init__(self, n_lat=4, n_lon=4, nlev=5, T_init=5.0, S_init=34.7,
                 mask_init=1.0):
        self.T = Field(
            jnp.full((n_lat, n_lon, nlev), T_init, dtype=jnp.float64),
            name="T", dims=("lat", "lon", "lev"), units="degC",
        )
        self.S = Field(
            jnp.full((n_lat, n_lon, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("lat", "lon", "lev"), units="PSU",
        )
        self.land_mask = Field(
            jnp.full((n_lat, n_lon), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("lat", "lon"), units="1",
        )

    def _replace(self, **kw):
        new = _FakeState.__new__(_FakeState)
        for attr in ("T", "S", "land_mask"):
            setattr(new, attr, getattr(self, attr))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


# ==============================================================================
# Thomas solver
# ==============================================================================

class TestThomasSolver:

    def test_identity_diagonal(self):
        # A = I → x = d
        a = np.zeros((2, 4))
        b = np.ones((2, 4))
        c = np.zeros((2, 4))
        d = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        x = _thomas_solve_columns(a, b, c, d)
        assert np.allclose(x, d)

    def test_match_dense_solver(self):
        rng = np.random.default_rng(0)
        n = 6
        # Build a diagonally dominant tridiagonal.
        a = rng.uniform(-0.2, 0.0, size=(3, n))
        c = rng.uniform(-0.2, 0.0, size=(3, n))
        b = 1.0 + np.abs(a) + np.abs(c) + rng.uniform(0.0, 0.1, size=(3, n))
        a[..., 0] = 0.0
        c[..., -1] = 0.0
        d = rng.standard_normal(size=(3, n))
        x = _thomas_solve_columns(a, b, c, d)
        # Dense reconstruction.
        for col in range(3):
            M = (
                np.diag(b[col])
                + np.diag(a[col, 1:], k=-1)
                + np.diag(c[col, :-1], k=1)
            )
            x_dense = np.linalg.solve(M, d[col])
            assert np.allclose(x[col], x_dense, atol=1e-10)


# ==============================================================================
# Tridiagonal builder
# ==============================================================================

class TestTridiag:

    def test_zero_K_gives_identity(self):
        h = np.full((2, 2, 4), 100.0)
        K = np.zeros_like(h)
        a, b, c = _build_tridiag(h, K, dt=3600.0)
        assert np.allclose(a, 0.0)
        assert np.allclose(c, 0.0)
        assert np.allclose(b, 1.0)

    def test_no_flux_boundaries(self):
        h = np.full((2, 2, 4), 100.0)
        K = np.full_like(h, 1.0e-4)
        a, _, c = _build_tridiag(h, K, dt=3600.0)
        # Top has no left flux; bottom has no right flux.
        assert np.all(a[..., 0] == 0.0)
        assert np.all(c[..., -1] == 0.0)


# ==============================================================================
# Apply step
# ==============================================================================

class TestApplyTidalMixingStep:

    def test_zero_K_noop(self):
        st = _FakeState(T_init=10.0, S_init=35.0)
        K = np.zeros((4, 4, 5))
        h = np.full((4, 4, 5), 100.0)
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=3600.0)
        assert np.allclose(np.asarray(new.T.data), 10.0)
        assert np.allclose(np.asarray(new.S.data), 35.0)

    def test_uniform_T_noop(self):
        # Spatially uniform tracer → diffusion is the identity.
        st = _FakeState(T_init=7.0)
        K = np.full((4, 4, 5), 1.0e-3)
        h = np.full((4, 4, 5), 50.0)
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=3600.0)
        assert np.allclose(np.asarray(new.T.data), 7.0, atol=1e-12)

    def test_gradient_smoothing(self):
        # Linear gradient + uniform K → strictly monotone smoothing.
        n_lat, n_lon, nlev = 1, 1, 8
        T_profile = np.linspace(0.0, 14.0, nlev)
        T_arr = np.broadcast_to(T_profile, (n_lat, n_lon, nlev)).copy()
        st = _FakeState(n_lat=n_lat, n_lon=n_lon, nlev=nlev)
        st.T = Field(jnp.asarray(T_arr), name="T",
                     dims=("lat", "lon", "lev"), units="degC")
        K = np.full_like(T_arr, 1.0e-3)
        h = np.full_like(T_arr, 50.0)
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=86400.0)
        T_new = np.asarray(new.T.data)[0, 0, :]
        # End-points moved toward the column mean (no-flux BC pushes them
        # toward the mean as diffusion proceeds).
        assert T_new[0] > T_profile[0]
        assert T_new[-1] < T_profile[-1]
        # Profile remains monotone increasing.
        assert np.all(np.diff(T_new) >= -1.0e-10)

    def test_volume_weighted_conservation(self):
        # h-weighted ∫T dV must be invariant under no-flux mixing.
        rng = np.random.default_rng(0)
        nlev = 6
        T0 = rng.standard_normal(size=(2, 2, nlev)) * 5.0 + 10.0
        h = np.full((2, 2, nlev), 80.0)
        K = np.full_like(h, 1.0e-3)
        st = _FakeState(n_lat=2, n_lon=2, nlev=nlev)
        st.T = Field(jnp.asarray(T0), name="T",
                     dims=("lat", "lon", "lev"), units="degC")
        S0 = rng.standard_normal(size=(2, 2, nlev)) * 0.5 + 34.7
        st.S = Field(jnp.asarray(S0), name="S",
                     dims=("lat", "lon", "lev"), units="PSU")
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=43200.0)
        # h-weighted sums equal pre-step values.
        T_sum_pre = np.sum(T0 * h, axis=-1)
        T_sum_post = np.sum(np.asarray(new.T.data) * h, axis=-1)
        assert np.allclose(T_sum_pre, T_sum_post, atol=1e-9, rtol=1e-9)
        S_sum_pre = np.sum(S0 * h, axis=-1)
        S_sum_post = np.sum(np.asarray(new.S.data) * h, axis=-1)
        assert np.allclose(S_sum_pre, S_sum_post, atol=1e-9, rtol=1e-9)

    def test_volume_weighted_conservation_nonuniform_h(self):
        # Same conservation invariant must hold under varying layer
        # thickness (partial-cell coordinate).
        rng = np.random.default_rng(1)
        nlev = 6
        h_profile = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
        h = np.broadcast_to(h_profile, (2, 2, nlev)).copy()
        T0 = rng.standard_normal(size=(2, 2, nlev)) * 5.0 + 10.0
        K = np.full_like(h, 1.0e-3)
        st = _FakeState(n_lat=2, n_lon=2, nlev=nlev)
        st.T = Field(jnp.asarray(T0), name="T",
                     dims=("lat", "lon", "lev"), units="degC")
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=43200.0)
        T_sum_pre = np.sum(T0 * h, axis=-1)
        T_sum_post = np.sum(np.asarray(new.T.data) * h, axis=-1)
        assert np.allclose(T_sum_pre, T_sum_post, atol=1e-9, rtol=1e-9)

    def test_land_cells_untouched(self):
        st = _FakeState(T_init=5.0)
        mask = np.ones((4, 4))
        mask[0, 0] = 0.0
        st.land_mask = Field(jnp.asarray(mask), name="land_mask",
                             dims=("lat", "lon"), units="1")
        # Big K, big dt, nontrivial gradient → land cell must still be
        # the pre-step value.
        n_lat, n_lon, nlev = 4, 4, 5
        T_profile = np.linspace(0.0, 8.0, nlev)
        T_arr = np.broadcast_to(T_profile, (n_lat, n_lon, nlev)).copy()
        st.T = Field(jnp.asarray(T_arr), name="T",
                     dims=("lat", "lon", "lev"), units="degC")
        K = np.full_like(T_arr, 1.0e-2)
        h = np.full_like(T_arr, 50.0)
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=86400.0)
        T_new = np.asarray(new.T.data)
        assert np.allclose(T_new[0, 0, :], T_profile)
        # Ocean cell: top + bottom moved toward the column mean.
        assert T_new[1, 1, 0] > T_profile[0]
        assert T_new[1, 1, -1] < T_profile[-1]

    def test_implicit_stable_large_dt(self):
        # CFL = K dt / dz² = 1e-3 · 86400 / 50² ≈ 35 → would blow up
        # explicit; implicit must remain monotone + bounded.
        nlev = 6
        T_profile = np.linspace(0.0, 10.0, nlev)
        T_arr = np.broadcast_to(T_profile, (1, 1, nlev)).copy()
        st = _FakeState(n_lat=1, n_lon=1, nlev=nlev)
        st.T = Field(jnp.asarray(T_arr), name="T",
                     dims=("lat", "lon", "lev"), units="degC")
        K = np.full_like(T_arr, 1.0e-3)
        h = np.full_like(T_arr, 50.0)
        new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=86400.0)
        T_new = np.asarray(new.T.data)[0, 0, :]
        assert np.all(np.isfinite(T_new))
        assert np.all(T_new >= T_profile.min() - 1.0e-12)
        assert np.all(T_new <= T_profile.max() + 1.0e-12)

    def test_shape_mismatch_raises(self):
        st = _FakeState()
        K = np.zeros((4, 4, 3))
        h = np.zeros((4, 4, 5))
        with pytest.raises(ValueError):
            apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=3600.0)


# ==============================================================================
# Grid-agnosticism: the same apply must run on an MPAS (nCells, nlev) state
# ==============================================================================
class _FakeMPASState:
    """Stub matching the MPAS ocean-state surface: (nCells, nlev) tracers,
    (nCells,) 1-D fields -- the shapes apply_tidal_mixing_step must accept."""

    def __init__(self, n_cells=6, nlev=5, T_init=5.0, S_init=34.7,
                 mask_init=1.0):
        self.T = Field(
            jnp.full((n_cells, nlev), T_init, dtype=jnp.float64),
            name="T", dims=("nCells", "lev"), units="degC",
        )
        self.S = Field(
            jnp.full((n_cells, nlev), S_init, dtype=jnp.float64),
            name="S", dims=("nCells", "lev"), units="PSU",
        )
        self.land_mask = Field(
            jnp.full((n_cells,), mask_init, dtype=jnp.float64),
            name="land_mask", dims=("nCells",), units="1",
        )

    def _replace(self, **kw):
        new = _FakeMPASState.__new__(_FakeMPASState)
        for attr in ("T", "S", "land_mask"):
            setattr(new, attr, getattr(self, attr))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


def test_apply_runs_unchanged_on_an_mpas_shaped_state():
    """apply_tidal_mixing_step is a per-column Thomas solve over the trailing
    axis, so it must run on (nCells, nlev) exactly as on (n_lat, n_lon, nlev).
    The driver previously warned '--tidal-mixing not supported on grid=mpas'
    although the apply, compute_tidal_diffusivity and compute_layer_thickness
    are all trailing-axis / broadcast operations with no grid assumption."""
    st = _FakeMPASState(n_cells=6, nlev=5)
    K = np.full((6, 5), 1.0e-3)
    h = np.full((6, 5), 100.0)
    new = apply_tidal_mixing_step(st, K_tidal=K, h_partial=h, dt=3600.0)
    assert new.T.data.shape == (6, 5) and new.S.data.shape == (6, 5)


def test_mpas_land_cells_are_untouched():
    """The land-mask contract holds on the 1-D (nCells,) mask too: masked cells
    keep their pre-step values."""
    st = _FakeMPASState(n_cells=4, nlev=5)
    mask = np.array([1.0, 0.0, 1.0, 0.0])
    st = st._replace(land_mask=Field(
        jnp.asarray(mask), name="land_mask", dims=("nCells",), units="1"))
    # non-uniform T so mixing would change interior columns
    T = np.tile(np.linspace(2.0, 10.0, 5), (4, 1))
    st = st._replace(T=Field(
        jnp.asarray(T), name="T", dims=("nCells", "lev"), units="degC"))
    K = np.full((4, 5), 5.0e-3)
    h = np.full((4, 5), 50.0)
    new = np.asarray(apply_tidal_mixing_step(
        st, K_tidal=K, h_partial=h, dt=3600.0).T.data)
    # land cells (1, 3) unchanged; ocean cells (0, 2) mixed
    np.testing.assert_allclose(new[1], T[1])
    np.testing.assert_allclose(new[3], T[3])
    assert not np.allclose(new[0], T[0])
    assert not np.allclose(new[2], T[2])
