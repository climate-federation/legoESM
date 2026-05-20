"""Unit tests for the SSS-restoring per-step applicator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step
from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig


class _FakeGrid:
    """Minimal LatLonGrid-like stub providing lat/lon for the kernel."""

    def __init__(self, n_lat=18, n_lon=36):
        self.n_lat = n_lat
        self.n_lon = n_lon
        dlat = np.pi / n_lat
        dlon = 2.0 * np.pi / n_lon
        self.dlat = float(dlat)
        self.dlon = float(dlon)
        self.lat = np.linspace(
            -np.pi / 2 + 0.5 * dlat, np.pi / 2 - 0.5 * dlat, n_lat,
        )
        self.lon = np.linspace(0.0, 2.0 * np.pi - dlon, n_lon)


class _FakeZCoord:
    """Stub z_coord exposing ``dz_ref[0]`` only (the applicator's
    actual surface-layer-thickness lookup happens inside
    ``compute_sss_restoring_flux`` via ``z1_m`` in the config)."""
    dz_ref = np.array([10.0, 50.0, 100.0])


class _FakeState:
    """Mirrors the bits of ``LatLonCGridOceanState`` the applicator uses.

    Only ``S`` and ``land_mask`` are touched.  ``_replace`` is the
    NamedTuple-style copy method.
    """

    def __init__(self, n_lat=18, n_lon=36, nlev=3,
                 S_init=34.0, mask_init=1.0):
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
        for k in ("S", "land_mask"):
            setattr(new, k, getattr(self, k))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


class TestApplySSSRestoring:

    def test_disabled_is_noop(self):
        grid = _FakeGrid()
        state = _FakeState()
        S_before = np.asarray(state.S.data).copy()
        new_state = apply_sss_restoring_step(
            state,
            S_target=np.full(S_before.shape[:2], 35.0),
            ice_concentration=None,
            config=SSSRestoringConfig(enabled=False),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=3600.0,
        )
        assert new_state is state  # identity returned

    def test_zero_bias_does_not_change_S(self):
        grid = _FakeGrid()
        state = _FakeState(S_init=34.7)
        S_target = np.full((grid.n_lat, grid.n_lon), 34.7)
        new_state = apply_sss_restoring_step(
            state,
            S_target=S_target,
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=365.0, regions=(),
            ),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=3600.0,
        )
        S_top_old = np.asarray(state.S.data)[..., 0]
        S_top_new = np.asarray(new_state.S.data)[..., 0]
        assert np.allclose(S_top_new, S_top_old)

    def test_salty_bias_freshens_surface(self):
        """Model is saltier than target → restoring REDUCES S_top."""
        grid = _FakeGrid()
        state = _FakeState(S_init=35.5)
        S_target = np.full((grid.n_lat, grid.n_lon), 34.7)
        new_state = apply_sss_restoring_step(
            state,
            S_target=S_target,
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=30.0, regions=(),
            ),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,  # 1 day
        )
        S_top_new = np.asarray(new_state.S.data)[..., 0]
        assert np.all(S_top_new < 35.5)
        # Approximate: with τ=30d and dt=1d, dS ≈ -(35.5-34.7)/30 = -0.0267
        expected_change = -(35.5 - 34.7) / 30.0
        observed_change = float(np.mean(S_top_new - 35.5))
        assert abs(observed_change - expected_change) < 1.0e-3

    def test_fresh_bias_salinifies_surface(self):
        grid = _FakeGrid()
        state = _FakeState(S_init=34.0)
        S_target = np.full((grid.n_lat, grid.n_lon), 34.7)
        new_state = apply_sss_restoring_step(
            state,
            S_target=S_target,
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=30.0, regions=(),
            ),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,
        )
        S_top_new = np.asarray(new_state.S.data)[..., 0]
        assert np.all(S_top_new > 34.0)

    def test_land_cells_untouched(self):
        """Restoring must not modify cells where ``land_mask=0``."""
        grid = _FakeGrid()
        state = _FakeState(S_init=35.5)
        # Mark half the cells as land.
        mask_arr = np.asarray(state.land_mask.data, dtype=np.float64).copy()
        mask_arr[: grid.n_lat // 2] = 0.0
        state.land_mask = Field(
            jnp.asarray(mask_arr), name=state.land_mask.name,
            dims=state.land_mask.dims, units=state.land_mask.units,
        )
        new_state = apply_sss_restoring_step(
            state,
            S_target=np.full((grid.n_lat, grid.n_lon), 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=10.0, regions=(),
            ),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,
        )
        S_top_new = np.asarray(new_state.S.data)[..., 0]
        # Land cells preserve initial value.
        assert np.allclose(S_top_new[: grid.n_lat // 2], 35.5)
        # Ocean cells changed.
        assert np.all(S_top_new[grid.n_lat // 2:] < 35.5)

    def test_only_surface_layer_modified(self):
        grid = _FakeGrid()
        state = _FakeState(S_init=35.5, nlev=3)
        new_state = apply_sss_restoring_step(
            state,
            S_target=np.full((grid.n_lat, grid.n_lon), 34.7),
            ice_concentration=None,
            config=SSSRestoringConfig(
                enabled=True, tau_restore_days_default=10.0, regions=(),
            ),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,
        )
        S_new = np.asarray(new_state.S.data)
        # Layers 1 + 2 untouched.
        assert np.allclose(S_new[..., 1], 35.5)
        assert np.allclose(S_new[..., 2], 35.5)
        # Layer 0 modified.
        assert np.all(S_new[..., 0] < 35.5)

    def test_ice_gating_reduces_restoring(self):
        grid = _FakeGrid()
        state_open = _FakeState(S_init=35.5)
        state_ice = _FakeState(S_init=35.5)
        config = SSSRestoringConfig(
            enabled=True, tau_restore_days_default=30.0, regions=(),
        )
        common_args = dict(
            S_target=np.full((grid.n_lat, grid.n_lon), 34.7),
            config=config,
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,
        )
        new_open = apply_sss_restoring_step(
            state_open, ice_concentration=None, **common_args,
        )
        new_ice = apply_sss_restoring_step(
            state_ice,
            ice_concentration=np.ones((grid.n_lat, grid.n_lon)),
            **common_args,
        )
        dS_open = np.mean(
            np.asarray(new_open.S.data)[..., 0] - 35.5
        )
        dS_ice = np.mean(
            np.asarray(new_ice.S.data)[..., 0] - 35.5
        )
        # Ice-covered restoring must be weaker (less freshening).
        assert abs(dS_ice) < abs(dS_open)
