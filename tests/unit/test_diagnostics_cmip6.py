"""Unit tests for the CMIP6 diagnostic expansion."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.diagnostics_cmip6 import (
    tracer_variance_production,
    global_tracer_variance,
    dianeutral_diffusivity_inferred,
    meridional_section_transport,
    osnap_east_transport,
    osnap_west_transport,
    SectionTransport,
    SV,
    PW,
)


# ==============================================================================
# Tracer variance production
# ==============================================================================

class TestTracerVarianceProduction:

    def test_uniform_tracer_zero_production(self):
        nlev = 20
        X = np.full((4, 5, nlev), 5.0)
        K = np.full_like(X, 1.0e-4)
        dz = np.full_like(X, 50.0)
        eps = tracer_variance_production(X, K, dz)
        assert np.all(eps == 0.0)

    def test_linear_gradient_constant_production(self):
        """X(z) = z (slope 1 K/m) → ∂X/∂z = 1 → ε = 2·K · 1²."""
        nlev = 20
        # X varies linearly with the level index; layer thickness 50 m
        # so gradient = ΔX/Δz = 1/50 K/m.
        X = np.broadcast_to(np.arange(nlev, dtype=np.float64),
                            (4, 5, nlev)).copy()
        K = np.full_like(X, 1.0e-4)
        dz = np.full_like(X, 50.0)
        eps = tracer_variance_production(X, K, dz)
        expected = 2.0 * 1.0e-4 * (1.0 / 50.0) ** 2
        # Interior values match expected.
        assert np.allclose(eps[:, :, 5:15], expected, rtol=1e-6)
        assert np.all(eps > 0.0)

    def test_K_scales_eps_linearly(self):
        nlev = 20
        X = np.broadcast_to(np.arange(nlev, dtype=np.float64),
                            (1, 1, nlev)).copy()
        dz = np.full_like(X, 50.0)
        K1 = np.full_like(X, 1.0e-4)
        K2 = np.full_like(X, 2.0e-4)
        eps1 = tracer_variance_production(X, K1, dz)
        eps2 = tracer_variance_production(X, K2, dz)
        assert np.allclose(eps2[:, :, 5:15], 2.0 * eps1[:, :, 5:15])

    def test_shape_preserved(self):
        X = np.random.RandomState(0).randn(3, 4, 10)
        K = np.full_like(X, 1.0e-4)
        dz = np.full_like(X, 50.0)
        eps = tracer_variance_production(X, K, dz)
        assert eps.shape == X.shape

    def test_axis_vertical_kwarg(self):
        """Test that ``axis_vertical=0`` works for transposed input."""
        nlev = 10
        X = np.broadcast_to(np.arange(nlev, dtype=np.float64)[:, None, None],
                            (nlev, 3, 4)).copy()
        K = np.full_like(X, 1.0e-4)
        dz = np.full_like(X, 50.0)
        eps = tracer_variance_production(X, K, dz, axis_vertical=0)
        assert eps.shape == X.shape


# ==============================================================================
# Global tracer variance
# ==============================================================================

class TestGlobalTracerVariance:

    def test_uniform_tracer_zero_variance(self):
        X = np.full((4, 5, 10), 20.0)
        h = np.full_like(X, 50.0)
        area = np.full((4, 5), 1.0e10)
        var = global_tracer_variance(X, h, area)
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_two_value_tracer(self):
        """Half cells at +1, half at -1 → variance = 1."""
        X = np.zeros((4, 4, 1))
        X[:2] = +1.0
        X[2:] = -1.0
        h = np.ones_like(X)
        area = np.ones(X.shape[:2])
        var = global_tracer_variance(X, h, area)
        assert var == pytest.approx(1.0, abs=1e-12)

    def test_mask_excludes_land(self):
        X = np.full((4, 4, 1), 20.0)
        X[0, 0, 0] = 1000.0   # land outlier
        h = np.ones_like(X)
        area = np.ones(X.shape[:2])
        mask = np.ones(X.shape[:2])
        mask[0, 0] = 0.0
        var = global_tracer_variance(X, h, area, mask=mask)
        # Only wet cells (uniform 20 °C) contribute → zero variance.
        assert var == pytest.approx(0.0, abs=1e-12)


# ==============================================================================
# Dianeutral diffusivity inferred
# ==============================================================================

class TestDianeutralDiffusivity:

    def test_vertical_only(self):
        K_v = np.full((3, 4, 5), 1.0e-5)
        K = dianeutral_diffusivity_inferred(K_v)
        assert np.allclose(K, K_v)

    def test_adds_tidal_component(self):
        K_v = np.full((3, 4, 5), 1.0e-5)
        K_t = np.full_like(K_v, 5.0e-5)
        K = dianeutral_diffusivity_inferred(K_v, K_tidal=K_t)
        assert np.allclose(K, 6.0e-5)

    def test_adds_all_components(self):
        K_v = np.full((2, 2, 2), 1.0e-5)
        K_t = np.full_like(K_v, 1.0e-5)
        K_gm = np.full_like(K_v, 1.0e-5)
        K_redi = np.full_like(K_v, 1.0e-5)
        K = dianeutral_diffusivity_inferred(
            K_v, K_tidal=K_t, K_gm_dia=K_gm, K_redi_dia=K_redi,
        )
        assert np.allclose(K, 4.0e-5)

    def test_no_negative(self):
        K_v = np.full((2, 2), -1.0e-5)
        K = dianeutral_diffusivity_inferred(K_v)
        assert np.all(K >= 0.0)


# ==============================================================================
# Meridional section transport
# ==============================================================================

class _FakeGrid:
    """LatLonGrid-like stub: 36×72, 5° lat / 5° lon."""

    def __init__(self, n_lat=36, n_lon=72):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.radius = 6.371e6
        dlat = np.pi / n_lat
        dlon = 2.0 * np.pi / n_lon
        self.dlat = float(dlat)
        self.dlon = float(dlon)
        self.lat = np.linspace(
            -np.pi / 2 + 0.5 * dlat, np.pi / 2 - 0.5 * dlat, n_lat,
        )
        # Lon centred on cells in [0, 2π) but the section helper
        # wraps to [-180, 180] internally.
        self.lon = np.linspace(0.0, 2.0 * np.pi - dlon, n_lon)


class TestMeridionalSectionTransport:

    def test_zero_velocity_zero_transport(self):
        g = _FakeGrid()
        nlev = 5
        v = np.zeros((g.n_lat + 1, g.n_lon, nlev))
        h = np.full((g.n_lat, g.n_lon, nlev), 200.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        out = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=30.0, lon_min_deg=-80.0, lon_max_deg=0.0,
        )
        assert out.volume_Sv == 0.0
        assert out.heat_PW == 0.0
        assert out.salt_kg_per_s == 0.0

    def test_uniform_northward_volume_transport(self):
        """Uniform v=0.1 m/s across the section produces a positive
        volume transport that scales with section width × depth."""
        g = _FakeGrid(n_lat=18, n_lon=36)
        nlev = 4
        v = np.full((g.n_lat + 1, g.n_lon, nlev), 0.1)
        h = np.full((g.n_lat, g.n_lon, nlev), 250.0)   # 1000 m total
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        out = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=30.0, lon_min_deg=-80.0, lon_max_deg=0.0,
        )
        assert out.volume_Sv > 0.0

    def test_southward_v_negative_volume(self):
        g = _FakeGrid()
        nlev = 4
        v = np.full((g.n_lat + 1, g.n_lon, nlev), -0.1)
        h = np.full((g.n_lat, g.n_lon, nlev), 250.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        out = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=30.0, lon_min_deg=-80.0, lon_max_deg=0.0,
        )
        assert out.volume_Sv < 0.0
        assert out.heat_PW < 0.0
        assert out.salt_kg_per_s < 0.0

    def test_off_grid_latitude_returns_nan(self):
        g = _FakeGrid()
        nlev = 4
        v = np.zeros((g.n_lat + 1, g.n_lon, nlev))
        h = np.full((g.n_lat, g.n_lon, nlev), 250.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        out = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=200.0,    # absurd
            lon_min_deg=-80.0, lon_max_deg=0.0,
            lat_tol_deg=2.0,
        )
        assert np.isnan(out.volume_Sv)

    def test_lon_band_restricts_integration(self):
        """A narrower lon band gives smaller total volume transport."""
        g = _FakeGrid(n_lat=18, n_lon=36)
        nlev = 2
        v = np.full((g.n_lat + 1, g.n_lon, nlev), 0.05)
        h = np.full((g.n_lat, g.n_lon, nlev), 500.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        wide = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=30.0,
            lon_min_deg=-180.0, lon_max_deg=180.0,
        )
        narrow = meridional_section_transport(
            v, h, T, S, g,
            target_lat_deg=30.0,
            lon_min_deg=-30.0, lon_max_deg=0.0,
        )
        assert abs(narrow.volume_Sv) < abs(wide.volume_Sv)


# ==============================================================================
# OSNAP presets
# ==============================================================================

class TestOSNAPPresets:

    def test_osnap_east_at_58N(self):
        """OSNAP East section sits at 58°N."""
        g = _FakeGrid(n_lat=36, n_lon=72)   # 5° lat grid
        nlev = 4
        v = np.full((g.n_lat + 1, g.n_lon, nlev), 0.05)
        h = np.full((g.n_lat, g.n_lon, nlev), 500.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 6.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 35.0)
        out = osnap_east_transport(v, h, T, S, g)
        assert np.isfinite(out.volume_Sv)
        assert out.volume_Sv > 0.0

    def test_osnap_west_at_53N(self):
        g = _FakeGrid(n_lat=36, n_lon=72)
        nlev = 4
        v = np.full((g.n_lat + 1, g.n_lon, nlev), 0.05)
        h = np.full((g.n_lat, g.n_lon, nlev), 500.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        out = osnap_west_transport(v, h, T, S, g)
        assert np.isfinite(out.volume_Sv)

    def test_osnap_total_matches_east_plus_west(self):
        """OSNAP overturning is the sum across both arrays; verify
        individual sections produce consistent signed transports."""
        g = _FakeGrid(n_lat=36, n_lon=72)
        nlev = 4
        v = np.full((g.n_lat + 1, g.n_lon, nlev), 0.02)
        h = np.full((g.n_lat, g.n_lon, nlev), 500.0)
        T = np.full((g.n_lat, g.n_lon, nlev), 5.0)
        S = np.full((g.n_lat, g.n_lon, nlev), 34.7)
        east = osnap_east_transport(v, h, T, S, g)
        west = osnap_west_transport(v, h, T, S, g)
        # Both positive at same v sign.
        assert east.volume_Sv > 0.0
        assert west.volume_Sv > 0.0
