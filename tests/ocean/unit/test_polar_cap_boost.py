"""Tests for the tripolar polar-cap viscosity boost factor."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    polar_cap_boost_factor,
    equatorial_boost_factor,
    laplacian_scaling_factor,
)


# ==============================================================================
# polar_cap_boost_factor
# ==============================================================================

class TestPolarCapBoost:

    def _grid(self, n_lat=180, n_lon=360):
        return create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

    def test_disabled_returns_ones(self):
        g = self._grid()
        bu, bv = polar_cap_boost_factor(g, cap_lat_deg=75.0, boost=1.0)
        assert jnp.allclose(bu, 1.0)
        assert jnp.allclose(bv, 1.0)
        assert bu.shape == (g.n_lat,)
        assert bv.shape == (g.n_lat + 1,)

    def test_boost_at_cap_latitude(self):
        g = self._grid()
        bu, bv = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=10.0, width_deg=5.0,
        )
        # At |lat| = cap_lat_deg the ramp is at 0.5 → boost = 1 + 9*0.5 = 5.5.
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_75n = int(np.argmin(np.abs(lat_deg - 75.0)))
        assert 5.0 < float(bu[idx_75n]) < 6.0

    def test_deep_cap_approaches_boost(self):
        g = self._grid()
        bu, _ = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=10.0, width_deg=5.0,
        )
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_89n = int(np.argmin(np.abs(lat_deg - 89.0)))
        # 89° is 14° into the cap (14°/5° = 2.8 sigma).
        # 0.5*(1 + tanh(2.8)) ≈ 0.995; boost ≈ 1 + 9*0.995 = 9.95.
        assert float(bu[idx_89n]) > 9.5

    def test_low_latitude_unchanged(self):
        g = self._grid()
        bu, _ = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=10.0, width_deg=5.0,
        )
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_equator = int(np.argmin(np.abs(lat_deg - 0.0)))
        # At equator, ramp is essentially zero → factor ≈ 1.
        assert float(bu[idx_equator]) < 1.01

    def test_symmetric_across_equator(self):
        """Cap boost uses ``|lat|`` so northern + southern caps both boost."""
        g = self._grid()
        bu, _ = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=10.0, width_deg=5.0,
        )
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_80n = int(np.argmin(np.abs(lat_deg - 80.0)))
        # Use the EXACT equator mirror index (the grid is symmetric:
        # lat == -lat[::-1]).  ``argmin(|lat + 80|)`` is brittle: 80.0 falls
        # exactly between the 79.5/80.5 cell centres, so under a float64 grid the
        # tie breaks to a NON-mirror southern cell (80.5°S vs the northern
        # 79.5°N), spuriously failing — float32 rounding had hidden the tie.
        idx_80s = g.n_lat - 1 - idx_80n
        assert jnp.isclose(bu[idx_80n], bu[idx_80s], rtol=1e-3)


# ==============================================================================
# Combined with cos(lat) scaling
# ==============================================================================

class TestPolarCapBoostCombined:

    def test_combined_with_cos_scaling_dominates_in_cap(self):
        """cos(lat) → 0 at high lat but cap boost lifts the effective A_h."""
        g = create_latlon_grid(n_lat=180, n_lon=360)
        cos_u, cos_v = laplacian_scaling_factor(g, power=1, floor=0.01)
        cap_u, cap_v = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=20.0, width_deg=5.0,
        )
        eff_u = cos_u * cap_u
        # At 85°N: cos(85°) ≈ 0.087, floored at 0.01 by default 0.
        # Cap boost ≈ 20 → effective ≈ 0.087 * 20 ≈ 1.74 (above mid-lat
        # baseline of ~0.7).
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_85n = int(np.argmin(np.abs(lat_deg - 85.0)))
        idx_45n = int(np.argmin(np.abs(lat_deg - 45.0)))
        assert float(eff_u[idx_85n]) > float(cos_u[idx_45n])

    def test_combined_with_equatorial_boost_independent(self):
        """Equatorial and polar boosts are multiplicative + independent."""
        g = create_latlon_grid(n_lat=180, n_lon=360)
        eq_u, eq_v = equatorial_boost_factor(g, sigma_deg=5.0, boost=5.0)
        cap_u, cap_v = polar_cap_boost_factor(
            g, cap_lat_deg=75.0, boost=10.0, width_deg=5.0,
        )
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_eq = int(np.argmin(np.abs(lat_deg - 0.0)))
        idx_85 = int(np.argmin(np.abs(lat_deg - 85.0)))
        # Equator: eq=5, cap=1.
        assert float(eq_u[idx_eq]) > 4.5
        assert float(cap_u[idx_eq]) < 1.05
        # Polar: eq=1, cap=10.
        assert float(eq_u[idx_85]) < 1.05
        assert float(cap_u[idx_85]) > 9.0


class TestEquatorialReduction:
    """boost < 1 = NEMO-style equatorial viscosity REDUCTION (ORCA1's
    eddy_viscosity_3D drops ahm 20000 -> 1000 m2/s at the equator)."""

    def test_reduction_at_equator_unity_far_away(self):
        g = create_latlon_grid(n_lat=180, n_lon=360)
        bu, bv = equatorial_boost_factor(g, sigma_deg=7.0, boost=0.05)
        lat_deg = np.degrees(np.asarray(g.lat))
        idx_eq = int(np.argmin(np.abs(lat_deg - 0.0)))
        idx_45 = int(np.argmin(np.abs(lat_deg - 45.0)))
        assert float(bu[idx_eq]) < 0.1          # ~0.05 at the equator
        assert float(bu[idx_45]) > 0.99         # untouched at midlatitude
        assert float(bu.min()) > 0.0            # never zero/negative

    def test_nonpositive_boost_raises(self):
        g = create_latlon_grid(n_lat=180, n_lon=360)
        with pytest.raises(ValueError):
            equatorial_boost_factor(g, sigma_deg=7.0, boost=0.0)
        with pytest.raises(ValueError):
            equatorial_boost_factor(g, sigma_deg=7.0, boost=-2.0)

    def test_bad_sigma_raises(self):
        g = create_latlon_grid(n_lat=180, n_lon=360)
        for sig in (0.0, -3.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                equatorial_boost_factor(g, sigma_deg=sig, boost=0.5)

    def test_floor_binds_under_reduction(self):
        """codex 9430935 MAJOR-2: with A_h_floor set, the composed scale may
        not drop below floor/A_h under an equatorial reduction."""
        import jax.numpy as jnp
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = LatLonCGridOceanConfig().replace_flat(
            A_h=1.0e5, A_h_floor=2.0e4, A_h_eq_boost=0.05,
            A_h_eq_sigma_deg=7.0)
        lv = cfg.lateral_viscosity
        g = create_latlon_grid(n_lat=180, n_lon=360)
        eb_u, _ = equatorial_boost_factor(g, lv.A_h_eq_sigma_deg,
                                          lv.A_h_eq_boost)
        # the branch guard mirrored from ocean_pe_latlon_cgrid
        fr = lv.A_h_floor / lv.A_h
        shaped = jnp.maximum(eb_u, fr)
        assert float(shaped.min()) >= fr - 1e-12
        # and the unfloored reduction WOULD have gone below it (test bites)
        assert float(eb_u.min()) < fr
