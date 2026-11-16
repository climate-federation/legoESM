"""Sandoval et al. (2026) quantum yield in the P-model (step C).

Pinned:
* phi0_sandoval reproduces the user's JULES-form reference values (the
  ``phi0_S_siteAI`` column of ~/jules_src/kphio_ai_test/site_ai_comparison.csv,
  phi0 at the site growth temperature) for four flux sites; the only
  difference allowed is the gas constant (8.314462618 here vs 8.3145 there);
* f(Topt) = 1, i.e. phi0(Topt) = phi_peak(AI) = (1/9)/(1+AI^m)^n; with the
  T/Topt factor the maximum itself lies < 1 K above Topt;
* the peak falls monotonically with aridity and is 1/9 at AI -> 0;
* the optimum and the subdaily daily optimum use it; the canopy light
  response uses it under light_response='pmodel_phi0';
* refusals: missing inputs, inputs under the temperature method, a
  non-default kphio under Sandoval; mGDD0 clamped to [1, 50] degC at init.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.p_model import (
    PModelConfig, acclimated_capacities, capacities_at, init_pmodel_acclim,
    phi0_c3, phi0_sandoval,
)

_ENTRY_X64 = jax.config.read("jax_enable_x64")
SV = PModelConfig(kphio_method="sandoval")


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _ENTRY_X64)


# site, growth T [degC], AI, mGDD0 [degC], JULES-form phi0 (user's script,
# R = 8.3145; site_ai_comparison.csv written 2026-11-05)
_JULES_REF = [
    ("DE-Hai", 12.976, 0.94614, 10.446, 0.073052),
    ("DE-Obe", 14.078, 1.1872, 11.162, 0.075289),
    ("US-MMS", 20.019, 0.92881, 14.903, 0.10359),
    ("US-Ton", 21.612, 2.269, 15.593, 0.073555),
]


@pytest.mark.parametrize("site,tg,ai,mg,ref", _JULES_REF)
def test_matches_jules_form_reference(site, tg, ai, mg, ref):
    phi = float(phi0_sandoval(jnp.asarray(constants.T_freeze + tg), ai, mg))
    # reference printed to 5 significant figures; R differs by 4e-6 relative
    assert phi == pytest.approx(ref, rel=1e-4), site


def _topt(mg):
    s0, s1 = 3468.185, -0.6680158
    ds = s0 * mg ** s1
    hd = 295.0 * ds
    return hd / (ds - constants.R_universal * np.log(70885.39 / (hd - 70885.39)))


@pytest.mark.parametrize("ai,mg", [(0.3, 5.0), (1.0, 15.0), (3.0, 25.0)])
def test_peak_at_topt_equals_aridity_peak(ai, mg):
    topt = _topt(mg)
    peak = (1.0 / 9.0) / (1.0 + ai ** 4.090556) ** 0.121122
    assert float(phi0_sandoval(jnp.asarray(topt), ai, mg)) == pytest.approx(peak, rel=1e-10)
    # With the T/Topt factor (JULES form) the maximum lies slightly ABOVE
    # Topt (the factor adds d ln T/dT = 1/T to the slope), by < 1 K, and
    # exceeds the Topt value by a negligible amount.
    t = jnp.linspace(topt - 20.0, topt + 10.0, 3001)
    phi = np.asarray(phi0_sandoval(t, ai, mg))
    t_max = float(t[int(np.argmax(phi))])
    assert 0.0 < t_max - topt < 1.0
    assert phi.max() / peak - 1.0 < 1e-3


def test_peak_decreases_with_aridity():
    ai = jnp.asarray([1e-6, 0.5, 1.0, 2.0, 5.0, 20.0])
    topt = _topt(15.0)
    phi = np.asarray(phi0_sandoval(jnp.full(6, topt), ai, 15.0))
    assert phi[0] == pytest.approx(1.0 / 9.0, rel=1e-8)
    assert np.all(np.diff(phi) < 0.0)


def test_optimum_scales_with_phi0_ratio():
    """Vcmax at the growth temperature is proportional to phi0 (Wang 2017
    coordination), so Sandoval/temperature capacities scale by the phi0
    ratio at the same drivers."""
    T = jnp.asarray([constants.T_freeze + 18.0])
    args = (T, jnp.asarray([1000.0]), jnp.asarray([410.0]), jnp.asarray([1e5]),
            jnp.asarray([800.0]))
    c_t = capacities_at(*args, PModelConfig())
    c_s = capacities_at(*args, SV, kphio_inputs=(jnp.asarray([1.2]), jnp.asarray([12.0])))
    ratio = float(phi0_sandoval(T, 1.2, 12.0)[0] / phi0_c3(T, PModelConfig().kphio)[0])
    assert float(c_s.vcmax25_leaf[0] / c_t.vcmax25_leaf[0]) == pytest.approx(ratio, rel=1e-10)
    assert float(c_s.chi[0]) == float(c_t.chi[0])     # phi0 does not enter chi


def test_state_carries_inputs_and_capacities_use_them():
    st = init_pmodel_acclim(2, t_init_K=290.0, ps_init_pa=1e5, cfg=SV,
                            sandoval_inputs=(jnp.asarray([0.5, 3.0]), 0.2),
                            dtype=jnp.float64)
    np.testing.assert_array_equal(st.mgdd0_c, [1.0, 1.0])     # clamped
    caps = acclimated_capacities(st, SV)
    assert float(caps.vcmax25_leaf[1]) < float(caps.vcmax25_leaf[0])  # arid column
    sub = PModelConfig(kphio_method="sandoval", acclimation="subdaily_noon")
    s2 = init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=sub,
                            sandoval_inputs=(1.0, 60.0), dtype=jnp.float64)
    assert float(s2.mgdd0_c[0]) == 50.0 and float(s2.vcmax25_real[0]) > 0.0


def test_invalid_sandoval_inputs_are_refused():
    with pytest.raises(ValueError, match="aridity index >= 0"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=SV,
                           sandoval_inputs=(-0.5, 10.0))
    with pytest.raises(ValueError, match="finite"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=SV,
                           sandoval_inputs=(1.0, float("nan")))


def test_refusals():
    with pytest.raises(ValueError, match="needs sandoval_inputs"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=SV)
    with pytest.raises(ValueError, match="silently inert"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=PModelConfig(),
                           sandoval_inputs=(1.0, 10.0))
    with pytest.raises(ValueError, match="silently inert"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5,
                           cfg=SV._replace(kphio=0.125), sandoval_inputs=(1.0, 10.0))
    with pytest.raises(ValueError, match="unknown PModelConfig.kphio_method"):
        init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5,
                           cfg=PModelConfig(kphio_method="fixed"))
    ema = init_pmodel_acclim(1, t_init_K=290.0, ps_init_pa=1e5, cfg=PModelConfig())
    with pytest.raises(ValueError, match="carries no aridity_index"):
        acclimated_capacities(ema, SV)


def test_light_response_uses_sandoval_phi0():
    from legoesm.land.canopy.photosynthesis import c3_assimilation
    T = constants.T_freeze + 15.0
    a = lambda I, **kw: c3_assimilation(  # noqa: E731
        jnp.asarray(T), jnp.asarray(300.0), I, jnp.asarray(500.0), jnp.asarray(20.0),
        rjv25=jnp.asarray(2.0), **kw).a_gross
    s_clm = jax.grad(lambda I: a(I))(1e-3)
    s_sv = jax.grad(lambda I: a(I, phi0_sandoval=(jnp.asarray(2.0), jnp.asarray(12.0))))(1e-3)
    expected = float(phi0_sandoval(jnp.asarray(T), 2.0, 12.0)) / (0.5 * 0.85 / 4.0)
    assert float(s_sv / s_clm) == pytest.approx(expected, rel=2e-3)
