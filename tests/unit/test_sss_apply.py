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


class TestLat2dOverride:
    """The optional lat2d_deg/lon2d_deg override (the tripole path) must drive
    the OMIP-2 region masks instead of the 1-D grid.lat/grid.lon broadcast."""

    def test_override_coords_select_region_tau(self):
        grid = _FakeGrid(n_lat=8, n_lon=8)
        common = dict(
            S_target=np.full((grid.n_lat, grid.n_lon), 34.0),
            ice_concentration=None,
            config=SSSRestoringConfig(enabled=True),   # default OMIP-2 regions
            grid=grid, z_coord=_FakeZCoord(), dt=86400.0,
        )
        # All cells forced to 80 N / 0 E => Arctic region (tau=30 d, strong).
        lat2d = np.full((grid.n_lat, grid.n_lon), 80.0)
        lon2d = np.full((grid.n_lat, grid.n_lon), 0.0)
        arc = apply_sss_restoring_step(
            _FakeState(grid.n_lat, grid.n_lon, S_init=35.5),
            lat2d_deg=lat2d, lon2d_deg=lon2d, **common)
        # All cells forced to 5 N / 200 E => open tropics (default tau=365 d, weak).
        lat2d_trop = np.full((grid.n_lat, grid.n_lon), 5.0)
        lon2d_trop = np.full((grid.n_lat, grid.n_lon), 200.0)
        trop = apply_sss_restoring_step(
            _FakeState(grid.n_lat, grid.n_lon, S_init=35.5),
            lat2d_deg=lat2d_trop, lon2d_deg=lon2d_trop, **common)
        dS_arc = np.mean(35.5 - np.asarray(arc.S.data)[..., 0])
        dS_trop = np.mean(35.5 - np.asarray(trop.S.data)[..., 0])
        assert dS_arc > 0.0 and dS_trop > 0.0
        assert dS_arc > 5.0 * dS_trop      # Arctic tau=30 d >> interior tau=365 d


class _DeviceLeafGuard:
    """Device-leaf stand-in for the persistent-sharded lane's S leaf.

    Encodes the host-transfer contract of ``apply_sss_restoring_step``
    mechanically (scaling-M2, codex batch4 HIGH): a FULL-leaf host conversion
    (``np.asarray`` on the whole 3-D array — the old
    ``np.asarray(state.S.data)[..., 0]`` + full-copy write-back pattern)
    raises, while device-side slicing (``[..., 0]``) and the device-side
    ``.at[...].set`` surface scatter delegate to the wrapped jax array.
    """

    def __init__(self, jarr):
        self._jarr = jnp.asarray(jarr)

    def __getitem__(self, idx):
        return self._jarr[idx]          # device-side slice

    @property
    def at(self):
        return self._jarr.at            # device-side functional update

    def __array__(self, dtype=None, copy=None):
        raise AssertionError(
            "full-3-D host conversion of the S leaf in the SSS-restoring "
            "per-step path (slice-before-convert contract violated)")

    @property
    def ndim(self):
        return self._jarr.ndim

    @property
    def shape(self):
        return self._jarr.shape

    @property
    def dtype(self):
        return self._jarr.dtype


class TestHostTransferContract:
    """Slice-before-convert + device-side surface write-back (scaling-M2)."""

    def _config(self):
        return SSSRestoringConfig(
            enabled=True, tau_restore_days_default=10.0, regions=(),
        )

    def test_guard_is_not_vacuous(self):
        with pytest.raises(AssertionError, match="slice-before-convert"):
            np.asarray(_DeviceLeafGuard(jnp.zeros((4, 5, 3))))

    def test_slice_before_convert_and_device_write_back(self):
        """With an S leaf whose FULL-array host conversion raises, the
        restoring step must still run (only the 2-D surface slice crosses to
        host; the write-back is a device-side surface scatter) and produce
        bit-identical values to the plain-leaf call."""
        grid = _FakeGrid()
        state = _FakeState(S_init=35.5, nlev=3)
        common = dict(
            S_target=np.full((grid.n_lat, grid.n_lon), 34.7),
            ice_concentration=None,
            config=self._config(),
            grid=grid,
            z_coord=_FakeZCoord(),
            dt=86400.0,
        )
        ref = apply_sss_restoring_step(state, **common)

        guarded = state._replace(S=Field(
            _DeviceLeafGuard(state.S.data),
            name=state.S.name, dims=state.S.dims, units=state.S.units,
        ))
        out = apply_sss_restoring_step(guarded, **common)

        S_out = np.asarray(out.S.data)      # real jax array: .at().set output
        S_ref = np.asarray(ref.S.data)
        np.testing.assert_array_equal(
            S_out, S_ref,
            err_msg="guarded-leaf SSS restoring diverged from plain leaf")
        # Deep layers untouched; surface actually restored (non-vacuous).
        assert np.allclose(S_out[..., 1:], 35.5)
        assert np.all(S_out[..., 0] < 35.5)
