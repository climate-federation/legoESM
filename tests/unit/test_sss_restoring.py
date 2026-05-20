"""Unit tests for OMIP-2 surface salinity restoring + WOA SSS loader."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.forcing.sss_restoring import (
    SSSRestoringConfig,
    RegionMaskSpec,
    DEFAULT_OMIP2_REGIONS,
    build_region_masks,
    compute_sss_restoring_flux,
    interp_woa_sss_to_grid,
    SECONDS_PER_DAY,
)
from legoesm.ocean.forcing.woa_sss import (
    synthetic_woa_sss,
    load_woa_sss,
    WOA_LON_NATIVE,
    WOA_LAT_NATIVE,
)


# ==============================================================================
# WOA SSS loader
# ==============================================================================

class TestWOASSSLoader:

    def test_synthetic_shape_and_range(self):
        sss, lat, lon = synthetic_woa_sss()
        assert sss.shape == (WOA_LAT_NATIVE, WOA_LON_NATIVE)
        assert lat.shape == (WOA_LAT_NATIVE,)
        assert lon.shape == (WOA_LON_NATIVE,)
        assert (sss >= 28.0).all() and (sss <= 37.0).all()

    def test_synthetic_subtropical_maxima_zonal(self):
        sss, lat, lon = synthetic_woa_sss()
        # Subtropical maxima around ±25° should be > ITCZ near 0°.
        sss_zonal = sss.mean(axis=1)
        idx_n_subtrop = np.argmin(np.abs(lat - 25.0))
        idx_s_subtrop = np.argmin(np.abs(lat + 25.0))
        idx_itcz = np.argmin(np.abs(lat))
        assert sss_zonal[idx_n_subtrop] > sss_zonal[idx_itcz]
        assert sss_zonal[idx_s_subtrop] > sss_zonal[idx_itcz]

    def test_synthetic_polar_freshening(self):
        sss, lat, lon = synthetic_woa_sss()
        sss_zonal = sss.mean(axis=1)
        idx_arctic = np.argmin(np.abs(lat - 80.0))
        idx_subtrop = np.argmin(np.abs(lat - 25.0))
        assert sss_zonal[idx_arctic] < sss_zonal[idx_subtrop]

    def test_load_falls_back_to_synthetic(self, tmp_path):
        sss, lat, lon = load_woa_sss(cache_dir=tmp_path)
        assert sss.shape[0] == WOA_LAT_NATIVE
        assert sss.shape[1] == WOA_LON_NATIVE


# ==============================================================================
# Region masks
# ==============================================================================

class TestRegionMasks:

    def _grid(self, n_lat=18, n_lon=36):
        lat = jnp.linspace(-85.0, 85.0, n_lat)
        lon = jnp.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon)
        lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")
        return lat2d, lon2d

    def test_arctic_region_active_at_high_latitude(self):
        lat2d, lon2d = self._grid()
        config = SSSRestoringConfig(enabled=True)
        inv_tau, region_active = build_region_masks(lat2d, lon2d, config)
        # Arctic cells (lat > 70) should have shorter tau than default.
        arctic_mask = lat2d > 75.0
        equator_mask = jnp.abs(lat2d) < 15.0
        assert float(jnp.mean(inv_tau[arctic_mask])) > float(jnp.mean(inv_tau[equator_mask]))

    def test_default_tau_in_interior(self):
        lat2d, lon2d = self._grid()
        config = SSSRestoringConfig(
            enabled=True, tau_restore_days_default=365.0,
        )
        inv_tau, _ = build_region_masks(lat2d, lon2d, config)
        # Tropical Pacific interior should sit at the default 1/(365·86400) s⁻¹.
        interior = (jnp.abs(lat2d) < 20.0) & (lon2d > 180.0) & (lon2d < 270.0)
        expected_inv_tau = 1.0 / (365.0 * SECONDS_PER_DAY)
        assert jnp.allclose(
            inv_tau[interior], expected_inv_tau, rtol=5e-2,
        )

    def test_empty_regions_uniform(self):
        lat2d, lon2d = self._grid()
        config = SSSRestoringConfig(
            enabled=True, regions=(),
            tau_restore_days_default=180.0,
        )
        inv_tau, region_active = build_region_masks(lat2d, lon2d, config)
        expected = 1.0 / (180.0 * SECONDS_PER_DAY)
        assert jnp.allclose(inv_tau, expected)
        assert jnp.all(region_active == 0.0)


# ==============================================================================
# Restoring flux
# ==============================================================================

class TestSSSRestoringFlux:

    def _grid_and_target(self, n_lat=18, n_lon=36):
        lat = jnp.linspace(-85.0, 85.0, n_lat)
        lon = jnp.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon)
        lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")
        S_target = jnp.full(lat2d.shape, 34.7)
        return lat2d, lon2d, S_target

    def test_zero_diff_zero_flux(self):
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target
        ice = jnp.zeros_like(lat2d)
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice,
            SSSRestoringConfig(enabled=True),
        )
        assert jnp.allclose(out["freshwater_flux"], 0.0, atol=1e-12)
        assert jnp.allclose(out["salt_flux"], 0.0, atol=1e-12)

    def test_fresh_bias_drives_positive_fw_into_ocean(self):
        """Model fresher than target → restoring adds FW (positive) is FALSE.

        Convention: dS/dt = -(S_model - S_target)/tau.  If S_model <
        S_target the model is too fresh → tendency POSITIVE (salinify).
        Equivalent FW flux is NEGATIVE (remove FW from ocean).
        """
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target - 0.5    # model 0.5 PSU too fresh
        ice = jnp.zeros_like(lat2d)
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice,
            SSSRestoringConfig(enabled=True),
        )
        # dS/dt should be positive (salinify), FW flux negative.
        assert jnp.all(out["dS_dt_top"] > 0.0)
        assert jnp.all(out["freshwater_flux"] < 0.0)

    def test_salty_bias_drives_positive_fw_into_ocean(self):
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target + 0.5
        ice = jnp.zeros_like(lat2d)
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice,
            SSSRestoringConfig(enabled=True),
        )
        assert jnp.all(out["dS_dt_top"] < 0.0)
        assert jnp.all(out["freshwater_flux"] > 0.0)

    def test_piston_velocity_relation(self):
        """``F_FW = − rho_0 · z1 · dS/dt / S_target`` (verifies derivation)."""
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target + 1.0
        ice = jnp.zeros_like(lat2d)
        config = SSSRestoringConfig(
            enabled=True, tau_restore_days_default=365.0,
            z1_m=10.0, regions=(),
        )
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice, config,
        )
        expected_F = (
            -constants.rho_ocean * 10.0 * out["dS_dt_top"] / S_target
        )
        assert jnp.allclose(out["freshwater_flux"], expected_F, rtol=1e-5)

    def test_ice_gate_suppresses_restoring_under_ice(self):
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target + 1.0
        # Cell A: open water; cell B: fully ice-covered.
        ice = jnp.full(lat2d.shape, 0.0)
        ice_full = jnp.full(lat2d.shape, 1.0)
        config = SSSRestoringConfig(enabled=True)
        out_open = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice, config,
        )
        out_ice = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice_full, config,
        )
        # Open-water restoring must be stronger.
        assert float(jnp.mean(jnp.abs(out_open["freshwater_flux"]))) > \
               float(jnp.mean(jnp.abs(out_ice["freshwater_flux"])))
        # Fully ice-covered should be ~zero.
        assert jnp.all(jnp.abs(out_ice["freshwater_flux"]) < 1e-6)

    def test_arctic_stronger_restoring_than_interior(self):
        lat2d, lon2d, S_target = self._grid_and_target()
        S_model = S_target + 1.0
        ice = jnp.zeros_like(lat2d)
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice,
            SSSRestoringConfig(enabled=True),
        )
        # Arctic cells (lat > 75) should have larger |FW flux| than
        # tropics (|lat| < 10).
        arctic = lat2d > 75.0
        tropics = jnp.abs(lat2d) < 10.0
        assert float(jnp.mean(jnp.abs(out["freshwater_flux"][arctic]))) > \
               float(jnp.mean(jnp.abs(out["freshwater_flux"][tropics])))

    def test_max_flux_cap(self):
        lat2d, lon2d, S_target = self._grid_and_target()
        # Huge bias: 100 PSU too salty (unphysical, just to test cap).
        S_model = S_target + 100.0
        ice = jnp.zeros_like(lat2d)
        config = SSSRestoringConfig(enabled=True)
        out = compute_sss_restoring_flux(
            S_model, S_target, lat2d, lon2d, ice, config,
        )
        assert jnp.all(jnp.abs(out["freshwater_flux"]) <= config.max_flux_kg_m2_s)


# ==============================================================================
# WOA SSS interp to model grid
# ==============================================================================

class TestInterpWOASSSToGrid:

    def test_interp_uniform_preserves_value(self):
        sss_woa = jnp.full((180, 360), 34.7)
        lat_woa = jnp.linspace(-89.5, 89.5, 180)
        lon_woa = jnp.linspace(0.0, 359.0, 360)
        lat_target = jnp.array([0.0, 30.0, -45.0, 75.0])
        lon_target = jnp.array([0.0, 90.0, 180.0, 270.0])
        out = interp_woa_sss_to_grid(
            sss_woa, lat_woa, lon_woa, lat_target, lon_target,
        )
        assert jnp.allclose(out, 34.7, atol=1e-6)

    def test_interp_linear_field(self):
        """Linear lat-dependent SSS interpolates correctly to mid-points."""
        lat_woa = jnp.linspace(-89.5, 89.5, 180)
        lon_woa = jnp.linspace(0.0, 359.0, 360)
        # SSS = lat (degrees) — perfectly linear test
        sss_woa = jnp.broadcast_to(lat_woa[:, None], (180, 360)).astype(jnp.float64)
        # Sample at exact WOA latitudes + arbitrary longitudes.
        lat_target = jnp.array([10.0, -25.0, 60.0])
        lon_target = jnp.array([100.0, 200.0, 50.0])
        out = interp_woa_sss_to_grid(
            sss_woa, lat_woa, lon_woa, lat_target, lon_target,
        )
        # At lat = 10 we expect SSS ≈ 10 (within 1° interp width).
        assert jnp.allclose(out, jnp.array([10.0, -25.0, 60.0]), atol=1.0)
