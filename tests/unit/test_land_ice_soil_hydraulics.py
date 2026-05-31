"""Category 3: Soil Hydraulics -- Retention Curves & Conductivity.

Tests all soil water retention curve (SWRC) models for mathematical
correctness and physical consistency: Van Genuchten, Clapp-Hornberger,
Brooks-Corey, PDI, Lu.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.land.soil_hydraulics import (
    theta_from_psi, psi_from_theta, hydraulic_conductivity, moisture_capacity,
    van_genuchten_Se, van_genuchten_theta, van_genuchten_psi, van_genuchten_K, van_genuchten_C,
    clapp_hornberger_psi, clapp_hornberger_theta, clapp_hornberger_K, clapp_hornberger_C,
    brooks_corey_Se, brooks_corey_theta, brooks_corey_K,
    pdi_theta, pdi_psi, pdi_K, pdi_C,
    lu_theta, lu_psi, lu_K, lu_C,
    interblock_K,
    SoilHydraulicsConfig,
)


CONFIG = SoilHydraulicsConfig()  # VG loam defaults


# ===================================================================
# 3a  Van Genuchten -- roundtrip consistency
# ===================================================================


class Test3a_VGRoundtrip:
    def test_psi_theta_roundtrip(self):
        psi_vals = -jnp.logspace(-2, 3, 50)
        theta = van_genuchten_theta(psi_vals, CONFIG)
        psi_recovered = van_genuchten_psi(theta, CONFIG)
        rel_err = jnp.abs(psi_recovered - psi_vals) / jnp.abs(psi_vals)
        assert jnp.all(rel_err < 1e-4)

    def test_saturation_at_zero(self):
        theta_0 = van_genuchten_theta(jnp.array([0.0]), CONFIG)
        assert jnp.allclose(theta_0, CONFIG.theta_sat, atol=1e-10)

    def test_dry_limit(self):
        theta_dry = van_genuchten_theta(jnp.array([-1e6]), CONFIG)
        assert float(theta_dry[0]) < CONFIG.theta_r + 0.01


# ===================================================================
# 3b  Van Genuchten -- monotonicity
# ===================================================================


class Test3b_VGMonotonicity:
    def test_theta_monotone(self):
        psi_vals = jnp.linspace(-100.0, 0.0, 200)
        theta = van_genuchten_theta(psi_vals, CONFIG)
        dtheta = jnp.diff(theta)
        assert jnp.all(dtheta >= -1e-12)

    def test_K_monotone(self):
        psi_vals = jnp.linspace(-100.0, -0.01, 200)
        K = van_genuchten_K(psi_vals, CONFIG)
        dK = jnp.diff(K)
        assert jnp.all(dK >= -1e-15)


# ===================================================================
# 3c  Van Genuchten -- saturation bounds
# ===================================================================


class Test3c_VGBounds:
    def test_theta_bounded(self):
        psi_vals = jnp.linspace(-1000.0, 0.0, 500)
        theta = van_genuchten_theta(psi_vals, CONFIG)
        assert jnp.all(theta >= CONFIG.theta_r - 1e-10)
        assert jnp.all(theta <= CONFIG.theta_sat + 1e-10)

    def test_K_bounded(self):
        psi_vals = jnp.linspace(-1000.0, 0.0, 500)
        K = van_genuchten_K(psi_vals, CONFIG)
        assert jnp.all(K >= 0.0)
        assert jnp.all(K <= CONFIG.K_sat + 1e-15)

    def test_K_at_saturation(self):
        K_sat_calc = van_genuchten_K(jnp.array([0.0]), CONFIG)
        assert jnp.allclose(K_sat_calc, CONFIG.K_sat, rtol=1e-6)


# ===================================================================
# 3d  Van Genuchten -- moisture capacity
# ===================================================================


class Test3d_VGCapacity:
    def test_C_positive_unsaturated(self):
        psi_vals = jnp.linspace(-100.0, -0.01, 200)
        C = van_genuchten_C(psi_vals, CONFIG)
        assert jnp.all(C > 0.0)

    def test_C_zero_at_saturation(self):
        C_sat = van_genuchten_C(jnp.array([0.0]), CONFIG)
        assert jnp.allclose(C_sat, 0.0, atol=1e-10)


# ===================================================================
# 3e  Clapp-Hornberger -- consistency
# ===================================================================


class Test3e_ClappHornberger:
    def test_theta_bounded(self):
        psi_vals = jnp.linspace(-100.0, CONFIG.psi_sat, 200)
        theta = clapp_hornberger_theta(psi_vals, CONFIG)
        assert jnp.all(theta >= 0.0)
        assert jnp.all(theta <= CONFIG.theta_sat + 1e-10)

    def test_K_bounded(self):
        theta_vals = jnp.linspace(0.01, CONFIG.theta_sat, 200)
        K = clapp_hornberger_K(theta_vals, CONFIG)
        assert jnp.all(K >= 0.0)
        assert jnp.all(K <= CONFIG.K_sat + 1e-10)

    def test_K_at_saturation(self):
        K_sat_calc = clapp_hornberger_K(jnp.array([CONFIG.theta_sat]), CONFIG)
        assert jnp.allclose(K_sat_calc, CONFIG.K_sat, rtol=1e-4)

    def test_monotonicity(self):
        theta_vals = jnp.linspace(0.05, CONFIG.theta_sat, 200)
        K = clapp_hornberger_K(theta_vals, CONFIG)
        dK = jnp.diff(K)
        assert jnp.all(dK >= -1e-15)


# ===================================================================
# 3f  Brooks-Corey -- consistency
# ===================================================================


class Test3f_BrooksCorey:
    def test_saturation_above_air_entry(self):
        Se = brooks_corey_Se(jnp.array([CONFIG.psi_b, CONFIG.psi_b + 0.1]), CONFIG)
        assert jnp.allclose(Se, 1.0, atol=1e-10)

    def test_theta_bounded(self):
        psi_vals = jnp.linspace(-100.0, CONFIG.psi_b, 200)
        theta = brooks_corey_theta(psi_vals, CONFIG)
        assert jnp.all(theta >= CONFIG.theta_r - 1e-10)
        assert jnp.all(theta <= CONFIG.theta_sat + 1e-10)

    def test_K_at_saturation(self):
        K = brooks_corey_K(jnp.array([CONFIG.psi_b]), CONFIG)
        assert jnp.allclose(K, CONFIG.K_sat, rtol=1e-4)

    def test_monotonicity(self):
        psi_vals = jnp.linspace(-100.0, CONFIG.psi_b, 200)
        theta = brooks_corey_theta(psi_vals, CONFIG)
        dtheta = jnp.diff(theta)
        assert jnp.all(dtheta >= -1e-12)


# ===================================================================
# 3g  PDI -- consistency
# ===================================================================


class Test3g_PDI:
    def test_theta_bounded(self):
        psi_vals = jnp.linspace(-1000.0, 0.0, 200)
        theta = pdi_theta(psi_vals, CONFIG)
        assert jnp.all(theta >= 0.0)
        assert jnp.all(theta <= CONFIG.theta_sat + 1e-10)

    def test_K_positive(self):
        psi_vals = jnp.linspace(-1000.0, -0.001, 200)
        K = pdi_K(psi_vals, CONFIG)
        assert jnp.all(K > 0.0)

    def test_saturation_at_zero(self):
        theta_0 = pdi_theta(jnp.array([0.0]), CONFIG)
        assert jnp.allclose(theta_0, CONFIG.theta_sat, atol=1e-6)

    def test_K_approaches_Ksat_at_saturation(self):
        # Regression: the wet-end smoothstep used to be inverted, so K(h~0)
        # collapsed to K_sat * Kr_crit (~0.34 * K_sat) instead of K_sat.
        # Walk h from h_crit down through 1e-12 m and confirm K converges
        # monotonically to K_sat.
        psi_wet = -jnp.array([1e-12, 1e-9, 1e-6, 1e-3])
        K_wet = pdi_K(psi_wet, CONFIG)
        assert jnp.all(K_wet <= CONFIG.K_sat * (1.0 + 1e-6))
        assert jnp.all(K_wet >= K_wet[-1])  # monotone increase toward h=0
        assert jnp.isclose(K_wet[0], CONFIG.K_sat, rtol=1e-6, atol=0.0)


# ===================================================================
# 3h  Lu three-regime -- consistency
# ===================================================================


class Test3h_Lu:
    def test_theta_bounded(self):
        psi_vals = jnp.linspace(-1000.0, 0.0, 200)
        theta = lu_theta(psi_vals, CONFIG)
        assert jnp.all(theta >= 0.0)
        assert jnp.all(theta <= CONFIG.theta_sat + 1e-10)

    def test_saturation_at_zero(self):
        theta_0 = lu_theta(jnp.array([0.0]), CONFIG)
        assert jnp.allclose(theta_0, CONFIG.theta_sat, atol=1e-6)

    def test_overall_monotonicity(self):
        """theta should generally increase with psi (wetter at higher psi)."""
        psi_vals = jnp.linspace(-500.0, -0.01, 1000)
        theta = lu_theta(psi_vals, CONFIG)
        # Verify overall trend: last value should exceed first
        assert float(theta[-1]) > float(theta[0]), "Lu theta not increasing overall"
        # Allow steep but continuous transition near saturation:
        # check no isolated spikes (non-monotonic dips) > 0.01
        dtheta = jnp.diff(theta)
        n_negative = int(jnp.sum(dtheta < -0.001))
        assert n_negative == 0, f"{n_negative} negative jumps > 0.001 in Lu curve"


# ===================================================================
# 3i  Cross-model comparison at saturation
# ===================================================================


class Test3i_CrossModel:
    def test_all_agree_at_saturation(self):
        psi_0 = jnp.array([0.0])
        theta_vg = van_genuchten_theta(psi_0, CONFIG)
        theta_pdi = pdi_theta(psi_0, CONFIG)
        theta_lu = lu_theta(psi_0, CONFIG)
        theta_bc = brooks_corey_theta(jnp.array([CONFIG.psi_b]), CONFIG)
        for name, val in [("VG", theta_vg), ("PDI", theta_pdi), ("Lu", theta_lu), ("BC", theta_bc)]:
            assert jnp.allclose(val, CONFIG.theta_sat, atol=1e-4), (
                f"{name} at saturation: {float(val[0])} != {CONFIG.theta_sat}"
            )


# ===================================================================
# 3j  Interblock conductivity
# ===================================================================


class Test3j_InterblocK:
    def test_geometric_mean_bounds(self):
        K_above = jnp.array([1.0, 0.1, 1e-6])
        K_below = jnp.array([0.1, 1.0, 1e-3])
        K_half = interblock_K(K_above, K_below)
        assert jnp.all(K_half >= jnp.minimum(K_above, K_below) - 1e-15)
        assert jnp.all(K_half <= jnp.maximum(K_above, K_below) + 1e-15)

    def test_geometric_mean_formula(self):
        K_above = jnp.array([4.0])
        K_below = jnp.array([9.0])
        K_half = interblock_K(K_above, K_below)
        assert jnp.allclose(K_half, 6.0, rtol=1e-6)


# ===================================================================
# 3k  Parametrize across all SWRC models
# ===================================================================


@pytest.mark.parametrize("curve", ["van_genuchten", "clapp_hornberger", "brooks_corey", "pdi", "lu"])
class Test3k_SWRC_Parametrized:
    def test_theta_bounded(self, curve):
        config = SoilHydraulicsConfig(retention_curve=curve)
        psi_vals = jnp.linspace(-100.0, -0.01, 100)
        theta = theta_from_psi(psi_vals, config)
        assert jnp.all(jnp.isfinite(theta))
        assert jnp.all(theta >= 0.0)
        assert jnp.all(theta <= config.theta_sat + 0.01)

    def test_K_nonnegative(self, curve):
        config = SoilHydraulicsConfig(retention_curve=curve)
        psi_vals = jnp.linspace(-100.0, -0.01, 100)
        theta = theta_from_psi(psi_vals, config)
        K = hydraulic_conductivity(psi_vals, theta, config)
        assert jnp.all(jnp.isfinite(K))
        assert jnp.all(K >= 0.0)


# ===================================================================
# 3l  Van Genuchten -- AD-safety at the saturation boundary
# ===================================================================


class Test3l_VanGenuchtenADSafety:
    """Reverse-mode gradients must be finite at the saturation boundary psi=0.

    Regression: ``van_genuchten_K``'s Mualem term ``(1-(1-Se^{1/m})^m)^2`` had an
    infinite gradient as Se -> 1, and ``van_genuchten_C``'s ``|alpha psi|^{n-1}``
    a NaN gradient at psi=0 (n_vg < 2, and the ``where(psi>=0, 0, C)`` turns the
    inf into 0*inf).  Both saturation states are reached every infiltration /
    ponding event — richards' ``K_top = hydraulic_conductivity(psi[:,0], ...)``
    and ``moisture_capacity`` run on the raw psi each Picard iteration — so a NaN
    here poisons the soil-column adjoint.  Flooring the singular arguments fixes
    the gradient while keeping the forward exact (K(psi=0)==K_sat, C(psi=0)==0).
    """

    def test_vg_K_grad_finite_through_saturation(self):
        cfg = SoilHydraulicsConfig(retention_curve="van_genuchten")
        # dry -> exact saturation (psi=0) -> ponded (psi>0)
        psi = jnp.array([-10.0, -1.0, -1e-3, -1e-9, 0.0, 1e-3])
        g = jax.grad(lambda p: jnp.sum(van_genuchten_K(p, cfg)))(psi)
        assert jnp.all(jnp.isfinite(g)), f"van_genuchten_K grad not finite: {g}"
        # Public dispatch path (what richards' K_top exercises on raw psi).
        g2 = jax.grad(
            lambda p: jnp.sum(hydraulic_conductivity(p, theta_from_psi(p, cfg), cfg))
        )(psi)
        assert jnp.all(jnp.isfinite(g2)), f"hydraulic_conductivity grad not finite: {g2}"

    def test_vg_C_grad_finite_at_saturation(self):
        cfg = SoilHydraulicsConfig(retention_curve="van_genuchten")
        psi = jnp.array([-10.0, -1.0, -1e-3, -1e-9, 0.0])
        g = jax.grad(lambda p: jnp.sum(van_genuchten_C(p, cfg)))(psi)
        assert jnp.all(jnp.isfinite(g)), f"van_genuchten_C grad not finite: {g}"
        # Public dispatch path (moisture_capacity, richards Picard iteration).
        g2 = jax.grad(
            lambda p: jnp.sum(moisture_capacity(p, theta_from_psi(p, cfg), cfg))
        )(psi)
        assert jnp.all(jnp.isfinite(g2)), f"moisture_capacity grad not finite: {g2}"
