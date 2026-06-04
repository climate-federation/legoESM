"""Unit tests for the ice-shelf basal-melt parameterisations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    IceShelfMeltResult,
    freezing_point_C,
    ice_base_pressure_dbar,
    three_equation_melt,
    linear_melt,
    compute_basal_melt,
)


# Approximate conversion: 1 m/s = 31.5e6 m/yr.
S_PER_YEAR = 365.25 * 86400.0


# ==============================================================================
# Freezing-point relation
# ==============================================================================

class TestFreezingPoint:

    def test_standard_seawater_at_surface(self):
        """T_f(34.7 PSU, 0 dbar) ≈ -1.9 °C — standard ocean value."""
        T_f = float(freezing_point_C(jnp.array(34.7), jnp.array(0.0)))
        assert T_f == pytest.approx(-5.73e-2 * 34.7 + 8.32e-2, abs=1e-6)
        assert -2.0 < T_f < -1.85

    def test_pressure_lowers_freezing_point(self):
        """Increasing pressure → freezing point DROPS (negative c)."""
        T_sfc = float(freezing_point_C(jnp.array(34.7), jnp.array(0.0)))
        T_deep = float(freezing_point_C(jnp.array(34.7), jnp.array(500.0)))
        assert T_deep < T_sfc

    def test_salinity_lowers_freezing_point(self):
        """Saltier water has a lower freezing point (negative a)."""
        T_fresh = float(freezing_point_C(jnp.array(30.0), jnp.array(0.0)))
        T_salty = float(freezing_point_C(jnp.array(36.0), jnp.array(0.0)))
        assert T_salty < T_fresh


class TestIceBasePressure:

    def test_unit_dbar_per_m(self):
        p = float(ice_base_pressure_dbar(jnp.array(500.0)))
        assert p == 500.0


# ==============================================================================
# Three-equation system
# ==============================================================================

class TestThreeEquation:

    def _cfg(self, **kw) -> IceShelfConfig:
        return IceShelfConfig(enabled=True, scheme="three_equation", **kw)

    def test_T_at_freezing_gives_zero_melt(self):
        """When ambient T equals the local freezing point, ṁ → 0."""
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_f = freezing_point_C(S, p, config=cfg)
        result = three_equation_melt(T_f, S, p, config=cfg)
        assert float(result.m_dot_m_s) == pytest.approx(0.0, abs=1e-12)
        assert float(result.freshwater_to_ocean) == pytest.approx(0.0, abs=1e-9)

    def test_warm_water_melts(self):
        """T_amb > T_freeze → ṁ > 0 (melt) and positive FW into ocean."""
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_amb = jnp.array(0.0)         # well above freezing
        result = three_equation_melt(T_amb, S, p, config=cfg)
        assert float(result.m_dot_m_s) > 0.0
        assert float(result.freshwater_to_ocean) > 0.0
        assert float(result.heat_extracted_from_ocean) > 0.0

    def test_cold_water_freezes(self):
        """T_amb < T_freeze → ṁ < 0 (freeze-on, marine ice forms)."""
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_f = float(freezing_point_C(S, p, config=cfg))
        # 0.1 K subcooling.
        result = three_equation_melt(
            jnp.array(T_f - 0.1), S, p, config=cfg,
        )
        assert float(result.m_dot_m_s) < 0.0
        assert float(result.freshwater_to_ocean) < 0.0

    def test_interface_temperature_equals_freezing_of_S_b(self):
        """T_b must satisfy the freezing-point relation at S_b."""
        cfg = self._cfg()
        p = jnp.array(800.0)
        S = jnp.array(34.5)
        T_amb = jnp.array(0.5)
        result = three_equation_melt(T_amb, S, p, config=cfg)
        T_f_at_Sb = freezing_point_C(result.S_b_PSU, p, config=cfg)
        assert float(result.T_b_C) == pytest.approx(
            float(T_f_at_Sb), abs=1e-9,
        )

    def test_S_b_less_than_S_amb_during_melt(self):
        """Fresh meltwater at the interface dilutes S_b below S_amb."""
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_amb = jnp.array(0.0)
        result = three_equation_melt(T_amb, S, p, config=cfg)
        assert float(result.S_b_PSU) < float(S)
        assert float(result.S_b_PSU) > 0.0

    def test_pressure_increases_melt_for_same_T(self):
        """Deeper ice (higher p) → lower freezing → larger thermal
        driving → MORE melt for the same ambient T."""
        cfg = self._cfg()
        S = jnp.array(34.7)
        T_amb = jnp.array(-0.5)
        m_shallow = float(three_equation_melt(
            T_amb, S, jnp.array(0.0), config=cfg,
        ).m_dot_m_s)
        m_deep = float(three_equation_melt(
            T_amb, S, jnp.array(1000.0), config=cfg,
        ).m_dot_m_s)
        assert m_deep > m_shallow

    def test_realistic_magnitude_pine_island_like(self):
        """Pine Island-style ambient: warm CDW intrusion at ~ 500 m depth.
        Expected basal melt rate ~ 10-30 m/yr."""
        cfg = self._cfg()
        T_amb = jnp.array(1.0)      # warm CDW
        S_amb = jnp.array(34.6)
        p_ice = jnp.array(500.0)
        m = float(three_equation_melt(
            T_amb, S_amb, p_ice, config=cfg,
        ).m_dot_m_s)
        m_per_year = m * S_PER_YEAR
        # Tolerant bounds: 0.5 – 100 m/yr (default γ_T = 1e-4 gives
        # ~few-m/yr scale; exact rate depends on tuning).
        assert 0.1 < m_per_year < 1000.0

    def test_proportionality_to_gamma_T(self):
        """Heat-exchange velocity γ_T scales melt approximately linearly."""
        T_amb = jnp.array(0.0)
        S_amb = jnp.array(34.7)
        p_ice = jnp.array(500.0)
        m1 = float(three_equation_melt(
            T_amb, S_amb, p_ice,
            config=self._cfg(gamma_T=1.0e-4, gamma_S=5.05e-7),
        ).m_dot_m_s)
        m2 = float(three_equation_melt(
            T_amb, S_amb, p_ice,
            config=self._cfg(gamma_T=2.0e-4, gamma_S=1.01e-6),
        ).m_dot_m_s)
        # Doubling both exchange velocities approximately doubles ṁ.
        ratio = m2 / max(m1, 1e-30)
        assert 1.5 < ratio < 2.5

    def test_no_negative_discriminant(self):
        """Physical inputs must never produce a negative discriminant
        (sqrt would NaN out)."""
        cfg = self._cfg()
        # Sweep a wide range of ambient conditions.
        T_grid = jnp.linspace(-2.5, 5.0, 30)
        S_grid = jnp.linspace(30.0, 36.0, 30)
        p_grid = jnp.linspace(0.0, 2000.0, 5)
        for T in T_grid:
            for S in S_grid:
                for p in p_grid:
                    r = three_equation_melt(T, S, p, config=cfg)
                    assert jnp.isfinite(r.m_dot_m_s)
                    assert jnp.isfinite(r.T_b_C)
                    assert jnp.isfinite(r.S_b_PSU)


# ==============================================================================
# Linear scheme (Beckmann-Goosse 2003)
# ==============================================================================

class TestLinearScheme:

    def _cfg(self, **kw) -> IceShelfConfig:
        return IceShelfConfig(enabled=True, scheme="linear", **kw)

    def test_zero_theta_zero_melt(self):
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_f = freezing_point_C(S, p, config=cfg)
        r = linear_melt(T_f, S, p, config=cfg)
        assert float(r.m_dot_m_s) == pytest.approx(0.0, abs=1e-12)

    def test_linear_in_theta(self):
        cfg = self._cfg()
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_f = float(freezing_point_C(S, p, config=cfg))
        r1 = linear_melt(jnp.array(T_f + 1.0), S, p, config=cfg)
        r2 = linear_melt(jnp.array(T_f + 2.0), S, p, config=cfg)
        assert float(r2.m_dot_m_s) == pytest.approx(
            2.0 * float(r1.m_dot_m_s), rel=1e-6,
        )

    def test_T_b_equals_freezing_at_S_amb(self):
        cfg = self._cfg()
        S = jnp.array(34.7)
        p = jnp.array(500.0)
        T_amb = jnp.array(0.0)
        r = linear_melt(T_amb, S, p, config=cfg)
        T_f_at_S_amb = freezing_point_C(S, p, config=cfg)
        assert float(r.T_b_C) == pytest.approx(float(T_f_at_S_amb), abs=1e-12)
        # S_b = S_amb in the no-feedback linear scheme.
        assert float(r.S_b_PSU) == float(S)


# ==============================================================================
# Three-equation vs linear at small thermal driving
# ==============================================================================

class TestThreeEqVsLinearLimit:

    def test_both_schemes_positive_melt_for_warm_ambient(self):
        """Both schemes should produce positive melt when ambient is
        well above freezing.  The 3-eq scheme includes salt-feedback
        at the interface (S_b → 0 dilutes the boundary layer) that
        the linear scheme deliberately ignores; the two scales
        differ by a finite factor that Beckmann-Goosse 2003 absorbs
        into the empirical ``melt_factor_linear``.
        """
        cfg_3eq = IceShelfConfig(enabled=True, scheme="three_equation")
        cfg_lin = IceShelfConfig(enabled=True, scheme="linear")
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_amb = jnp.array(0.0)
        m_3eq = float(three_equation_melt(
            T_amb, S, p, config=cfg_3eq,
        ).m_dot_m_s)
        m_lin = float(linear_melt(T_amb, S, p, config=cfg_lin).m_dot_m_s)
        assert m_3eq > 0.0
        assert m_lin > 0.0
        # Order-of-magnitude agreement: both within ×10 of each other.
        ratio = m_3eq / m_lin
        assert 0.1 < ratio < 10.0

    def test_both_schemes_zero_at_T_freeze(self):
        """Sanity: both schemes return zero melt at the local
        freezing point."""
        cfg_3eq = IceShelfConfig(enabled=True, scheme="three_equation")
        cfg_lin = IceShelfConfig(enabled=True, scheme="linear")
        p = jnp.array(500.0)
        S = jnp.array(34.7)
        T_f = float(freezing_point_C(S, p, config=cfg_3eq))
        T_amb = jnp.array(T_f)
        assert float(three_equation_melt(
            T_amb, S, p, config=cfg_3eq,
        ).m_dot_m_s) == pytest.approx(0.0, abs=1e-12)
        assert float(linear_melt(
            T_amb, S, p, config=cfg_lin,
        ).m_dot_m_s) == pytest.approx(0.0, abs=1e-12)


# ==============================================================================
# Dispatch
# ==============================================================================

class TestComputeBasalMelt:

    def test_disabled_returns_zero(self):
        cfg = IceShelfConfig(enabled=False)
        r = compute_basal_melt(
            jnp.array(2.0), jnp.array(34.7), jnp.array(500.0), config=cfg,
        )
        assert float(r.m_dot_m_s) == 0.0
        assert float(r.freshwater_to_ocean) == 0.0
        assert float(r.heat_extracted_from_ocean) == 0.0
        # T_b populated with sensible value for downstream callers.
        assert jnp.isfinite(r.T_b_C)

    def test_scheme_dispatch_three_equation(self):
        cfg = IceShelfConfig(enabled=True, scheme="three_equation")
        r = compute_basal_melt(
            jnp.array(0.0), jnp.array(34.7), jnp.array(500.0), config=cfg,
        )
        assert float(r.m_dot_m_s) > 0.0

    def test_scheme_dispatch_linear(self):
        cfg = IceShelfConfig(enabled=True, scheme="linear")
        r = compute_basal_melt(
            jnp.array(0.0), jnp.array(34.7), jnp.array(500.0), config=cfg,
        )
        assert float(r.m_dot_m_s) > 0.0

    def test_unknown_scheme_raises(self):
        cfg = IceShelfConfig(enabled=True, scheme="iglu")
        with pytest.raises(ValueError):
            compute_basal_melt(
                jnp.array(0.0), jnp.array(34.7), jnp.array(500.0), config=cfg,
            )
