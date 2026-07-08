"""Category 7: Stomatal Conductance & Farquhar Photosynthesis."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.stomata import (
    arrhenius, peaked_arrhenius,
    farquhar_photosynthesis, ball_berry_gs, medlyn_gs, jarvis_gs,
    coupled_farquhar_stomata, compute_stomatal_beta,
    StomataConfig,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Tight tolerances require float64 precision."""
    import jax
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


CFG = StomataConfig(enabled=True)
SHAPE = (4,)


class Test7a_Arrhenius:
    def test_at_25C(self):
        T = jnp.full(SHAPE, 298.15)
        val = arrhenius(60.0, CFG.Ha_Vc, T)
        assert jnp.allclose(val, 60.0, rtol=1e-6)

    def test_increases_with_T(self):
        v_low = arrhenius(60.0, CFG.Ha_Vc, jnp.full(SHAPE, 290.0))
        v_high = arrhenius(60.0, CFG.Ha_Vc, jnp.full(SHAPE, 310.0))
        assert jnp.all(v_high > v_low)

    def test_positive(self):
        vals = arrhenius(60.0, CFG.Ha_Vc, jnp.linspace(250.0, 330.0, 50))
        assert jnp.all(vals > 0)
        assert jnp.all(jnp.isfinite(vals))


class Test7b_PeakedArrhenius:
    def test_at_25C(self):
        val = peaked_arrhenius(120.0, CFG.Ha_J, CFG.Hd_J, CFG.S_J, jnp.full(SHAPE, 298.15))
        assert jnp.allclose(val, 120.0, rtol=1e-4)

    def test_has_peak(self):
        T_range = jnp.linspace(280.0, 330.0, 100)
        vals = peaked_arrhenius(120.0, CFG.Ha_J, CFG.Hd_J, CFG.S_J, T_range)
        assert 280.0 < float(T_range[int(jnp.argmax(vals))]) < 330.0
        assert jnp.all(vals > 0)


class Test7c_FarquharBasic:
    def test_positive_A_normal(self):
        A_net, _ = farquhar_photosynthesis(
            jnp.full(SHAPE, 280.0), jnp.full(SHAPE, 500.0), jnp.full(SHAPE, 298.15), CFG)
        assert jnp.all(A_net > 0)
        assert jnp.all(A_net < 50.0)

    def test_dark_respiration_only(self):
        A_net, _ = farquhar_photosynthesis(
            jnp.full(SHAPE, 280.0), jnp.zeros(SHAPE), jnp.full(SHAPE, 298.15), CFG)
        assert jnp.all(A_net < 0)

    def test_A_increases_with_light(self):
        T = jnp.full(SHAPE, 298.15)
        Ci = jnp.full(SHAPE, 280.0)
        A_low, _ = farquhar_photosynthesis(Ci, jnp.full(SHAPE, 100.0), T, CFG)
        A_high, _ = farquhar_photosynthesis(Ci, jnp.full(SHAPE, 1000.0), T, CFG)
        assert jnp.all(A_high > A_low)

    def test_A_increases_with_CO2(self):
        T = jnp.full(SHAPE, 298.15)
        APAR = jnp.full(SHAPE, 500.0)
        A_low, _ = farquhar_photosynthesis(jnp.full(SHAPE, 200.0), APAR, T, CFG)
        A_high, _ = farquhar_photosynthesis(jnp.full(SHAPE, 600.0), APAR, T, CFG)
        assert jnp.all(A_high > A_low)


class Test7e_FarquharSoilStress:
    def test_drought_reduces_A(self):
        args = (jnp.full(SHAPE, 280.0), jnp.full(SHAPE, 500.0), jnp.full(SHAPE, 298.15), CFG)
        A_wet, _ = farquhar_photosynthesis(*args, beta_soil=jnp.full(SHAPE, 1.0))
        A_dry, _ = farquhar_photosynthesis(*args, beta_soil=jnp.full(SHAPE, 0.1))
        assert jnp.all(A_wet > A_dry)


class Test7f_BallBerry:
    def test_gs_at_zero_A(self):
        gs = ball_berry_gs(jnp.zeros(SHAPE), jnp.full(SHAPE, 0.7), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        assert jnp.allclose(gs, CFG.g0, atol=1e-10)

    def test_gs_increases_with_A(self):
        gs_low = ball_berry_gs(jnp.full(SHAPE, 5.0), jnp.full(SHAPE, 0.7), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        gs_high = ball_berry_gs(jnp.full(SHAPE, 20.0), jnp.full(SHAPE, 0.7), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        assert jnp.all(gs_high > gs_low)

    def test_gs_increases_with_RH(self):
        gs_dry = ball_berry_gs(jnp.full(SHAPE, 10.0), jnp.full(SHAPE, 0.3), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        gs_humid = ball_berry_gs(jnp.full(SHAPE, 10.0), jnp.full(SHAPE, 0.9), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        assert jnp.all(gs_humid > gs_dry)

    def test_gs_ge_g0(self):
        gs = ball_berry_gs(jnp.full(SHAPE, -5.0), jnp.full(SHAPE, 0.7), jnp.full(SHAPE, 400.0), CFG.g1_bb, CFG.g0)
        assert jnp.all(gs >= CFG.g0 - 1e-15)


class Test7g_Medlyn:
    def test_gs_at_zero_A(self):
        gs = medlyn_gs(jnp.zeros(SHAPE), jnp.full(SHAPE, 1.0), jnp.full(SHAPE, 400.0), CFG.g1_med, CFG.g0)
        assert jnp.allclose(gs, CFG.g0, atol=1e-10)

    def test_gs_decreases_with_VPD(self):
        gs_low = medlyn_gs(jnp.full(SHAPE, 10.0), jnp.full(SHAPE, 0.5), jnp.full(SHAPE, 400.0), CFG.g1_med, CFG.g0)
        gs_high = medlyn_gs(jnp.full(SHAPE, 10.0), jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 400.0), CFG.g1_med, CFG.g0)
        assert jnp.all(gs_low > gs_high)

    def test_gs_ge_g0(self):
        gs = medlyn_gs(jnp.full(SHAPE, -5.0), jnp.full(SHAPE, 1.0), jnp.full(SHAPE, 400.0), CFG.g1_med, CFG.g0)
        assert jnp.all(gs >= CFG.g0 - 1e-15)


class Test7h_Jarvis:
    def test_positive_with_light(self):
        """gs should be meaningfully positive under well-lit conditions."""
        gs = jarvis_gs(
            T=jnp.full(SHAPE, 298.15), sw_down=jnp.full(SHAPE, 500.0),
            q_air=jnp.full(SHAPE, 0.015), p_surface=jnp.full(SHAPE, 1e5),
            beta_soil=jnp.full(SHAPE, 1.0), config=CFG,
        )
        assert jnp.all(gs > 0.01)
        assert jnp.all(gs <= CFG.gs_max)

    def test_dark_near_zero(self):
        gs = jarvis_gs(
            T=jnp.full(SHAPE, 298.15), sw_down=jnp.zeros(SHAPE),
            q_air=jnp.full(SHAPE, 0.01), p_surface=jnp.full(SHAPE, 1e5),
            beta_soil=jnp.full(SHAPE, 1.0), config=CFG,
        )
        assert jnp.all(gs < 0.01 * CFG.gs_max)

    def test_gs_bounded(self):
        gs = jarvis_gs(
            T=jnp.full(SHAPE, 298.15), sw_down=jnp.full(SHAPE, 300.0),
            q_air=jnp.full(SHAPE, 0.01), p_surface=jnp.full(SHAPE, 1e5),
            beta_soil=jnp.full(SHAPE, 0.5), config=CFG,
        )
        assert jnp.all(gs >= 0.0)
        assert jnp.all(gs <= CFG.gs_max)


class Test7i_CoupledSolver:
    def test_converges(self):
        gs, gpp = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 400.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG,
        )
        assert jnp.all(gs > 0)
        assert jnp.all(gpp > 0)
        assert jnp.all(jnp.isfinite(gs))
        assert jnp.all(jnp.isfinite(gpp))


class Test7j_StomatalBeta:
    def test_beta_bounded(self):
        beta = compute_stomatal_beta(
            jnp.full(SHAPE, 0.15), jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG)
        assert jnp.all(beta >= 0.0)
        assert jnp.all(beta <= 1.0)


class Test7k_DroughtGPP:
    def test_drought_reduces_gpp(self):
        _, gpp_wet = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 400.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.9), CFG)
        _, gpp_dry = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 400.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.1), CFG)
        assert jnp.all(gpp_wet > gpp_dry)


class Test7l_CO2Fertilization:
    def test_higher_co2_higher_gpp(self):
        _, gpp_low = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 400.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG)
        _, gpp_high = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 800.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG)
        assert jnp.all(gpp_high > gpp_low)

    def test_higher_co2_lower_gs(self):
        gs_low, _ = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 400.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG)
        gs_high, _ = coupled_farquhar_stomata(
            jnp.full(SHAPE, 298.15), jnp.full(SHAPE, 300.0), 800.0,
            jnp.full(SHAPE, 0.01), jnp.full(SHAPE, 1e5),
            jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 0.8), CFG)
        assert jnp.all(gs_low > gs_high)
