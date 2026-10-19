"""Kessler ``parameter_set="cloudbench"``: the CloudBench one-moment warm-rain rates.

* Each rate is pinned against an independent NumPy evaluation written from the
  source formulas with LITERAL coefficients (swirl-lm 98f96f80 microphysics.proto /
  microphysics_one_moment_constants.py, Azimi et al. 2024 Table 3 EKI, Chen et al.
  2022 Table B1), not from the module constants.
* RULE 0: per column, total water (vapour + cloud + rain + surface precipitation)
  and moist enthalpy (c_pd T + L_v q_v; precipitation leaves as liquid) close to
  round-off; the precipitation term is shown to be non-negligible, so dropping it
  fails the water budget.
* Gradients of the new rates pass ``jax.test_util.check_grads`` (order 2).
* The classic set stays the default and an unknown set raises.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.physics.microphysics import kessler as K
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_specific_humidity, saturation_vapor_pressure

from legoesm import constants

CB = KesslerConfig(parameter_set="cloudbench")


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# --- independent reference (literal source coefficients) ---------------------
def _lam(q_r, rho):
    """Radius-based Marshall-Palmer lambda: (Gamma(4) m0 n0 / (q rho r0^3))^(1/4)."""
    r0, n0, rho_w = 1e-3, 1.6e7, 1e3
    m0 = 4.0 / 3.0 * math.pi * rho_w * r0 ** 3
    return (6.0 * m0 * n0 / (q_r * rho * r0 ** 3)) ** 0.25


def _v0(rho):
    return math.sqrt(8.0 * 1e-3 * constants.g / (3.0 * 0.55) * (1e3 / rho - 1.0))


def _ref_accretion(q_c, q_r, rho):
    chi_a, del_a, chi_v, del_v, e_lr, r0, n0 = 16.61, 3.00, 0.205, 0.228, 0.8, 1e-3, 1.6e7
    sig = 2.0 + 0.5 + del_a + del_v
    lam = _lam(q_r, rho)
    return (n0 * math.pi * r0 ** 2 * chi_a * chi_v * _v0(rho) * q_c * e_lr
            * math.gamma(sig + 1.0) / lam * (1.0 / (r0 * lam)) ** sig)


def _ref_evaporation(T, q_v, q_r, q_sat, rho):
    k, dv, nu, a_v, b_v, chi_v, del_v, r0, n0 = (
        2.4e-2, 2.26e-5, 1.6e-5, 1.5, 0.98, 0.205, 0.228, 1e-3, 1.6e7)
    lv, rv = constants.L_v, constants.R_v
    p_sat = float(saturation_vapor_pressure(jnp.asarray(T)))
    g = 1.0 / (lv / (k * T) * (lv / (rv * T) - 1.0) + rv * T / (p_sat * dv))
    lam = _lam(q_r, rho)
    s = q_v / q_sat
    vent = a_v + b_v * (nu / dv) ** (1.0 / 3.0) * (1.0 / (r0 * lam)) ** (
        0.5 * (0.5 + del_v)) * math.sqrt(2.0 * chi_v * _v0(rho) / (nu * lam)) * math.gamma(
        0.5 * (0.5 + del_v + 5.0))
    return max(-4.0 * math.pi * n0 / rho * (s - 1.0) * g / lam ** 2 * vent, 0.0)


def _ref_chen(q_r, rho, diameter=False):
    q = math.exp(0.115231 * rho)
    a = [0.044612 * q, -0.263166 * q, 4.7178 * q * rho ** -0.47335]
    b = [2.2955 - 0.038465 * rho, 2.2955 - 0.038465 * rho, 1.1451 - 0.038465 * rho]
    c = [0.0, 0.184325e3, 0.184325e3]
    lam = _lam(q_r, rho) / (2.0 if diameter else 1.0)
    v = sum(ai * 1e3 ** bi * math.gamma(bi + 4.0) / math.gamma(4.0)
            * (1.0 / (1.0 + ci / lam)) ** 4 / (lam + ci) ** bi
            for ai, bi, ci in zip(a, b, c))
    return max(v, 0.0)


# --- rates vs the reference ---------------------------------------------------
@pytest.mark.parametrize("q_c,q_r,rho", [(1e-3, 1e-4, 1.1), (3e-4, 2e-3, 0.9),
                                         (8e-4, 5e-6, 1.2)])
def test_accretion_matches_source_formula(q_c, q_r, rho):
    got = float(K.cloudbench_accretion(jnp.asarray(q_c), jnp.asarray(q_r), jnp.asarray(rho)))
    np.testing.assert_allclose(got, _ref_accretion(q_c, q_r, rho), rtol=1e-12)


def test_autoconversion_is_the_swirl_lm_threshold_form():
    q_c = jnp.asarray([0.0, 4e-4, 5e-4, 1.5e-3])
    np.testing.assert_allclose(np.asarray(K.cloudbench_autoconversion(q_c)),
                               [0.0, 0.0, 0.0, 1e-3 / 1e3], rtol=0, atol=1e-18)


@pytest.mark.parametrize("T,q_v,q_r,rho", [(290.0, 8e-3, 1e-4, 1.1),
                                           (285.0, 2e-3, 1e-3, 1.0)])
def test_evaporation_matches_source_formula(T, q_v, q_r, rho):
    q_sat = 1.2e-2
    got = float(K.cloudbench_rain_evaporation_rate(
        jnp.asarray(T), jnp.asarray(q_v), jnp.asarray(q_r), jnp.asarray(q_sat),
        jnp.asarray(rho)))
    ref = _ref_evaporation(T, q_v, q_r, q_sat, rho)
    assert ref > 0.0
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_evaporation_is_zero_at_or_above_saturation():
    got = K.cloudbench_rain_evaporation_rate(
        jnp.asarray(290.0), jnp.asarray(1.3e-2), jnp.asarray(1e-4), jnp.asarray(1.2e-2),
        jnp.asarray(1.1))
    assert float(got) == 0.0


@pytest.mark.parametrize("q_r,rho", [(1e-5, 1.1), (1e-4, 1.1), (1e-3, 1.1), (1e-4, 0.8)])
def test_chen_fall_speed_matches_table_b1_with_radius_lambda(q_r, rho):
    got = float(K.chen2022_rain_fall_speed(jnp.asarray(q_r), jnp.asarray(rho)))
    np.testing.assert_allclose(got, _ref_chen(q_r, rho), rtol=1e-12)


def test_chen_fall_speed_reproduces_the_radius_for_diameter_substitution():
    """~2.1 m/s at 0.1 g/kg (swirl-lm form, reproduced on purpose) vs ~4.0 m/s with
    the diameter conversion CloudMicrophysics.jl applies."""
    v = float(K.chen2022_rain_fall_speed(jnp.asarray(1e-4), jnp.asarray(1.1)))
    assert 2.0 < v < 2.3
    assert 3.8 < _ref_chen(1e-4, 1.1, diameter=True) < 4.1
    assert float(K.chen2022_rain_fall_speed(jnp.asarray(0.0), jnp.asarray(1.1))) == 0.0


# --- gradients ---------------------------------------------------------------
_KG = 1.0e-3  # rates take kg/kg; check_grads' FD step (1e-4) needs O(1) inputs
_GRAD_CASES = {
    "accretion": (lambda qc, qr, r: K.cloudbench_accretion(qc * _KG, qr * _KG, r),
                  (0.7, 0.2, 1.05)),
    "chen_fall_speed": (lambda qr, r: K.chen2022_rain_fall_speed(qr * _KG, r), (0.2, 1.05)),
    "evaporation": (lambda T, qv, qr, r: K.cloudbench_rain_evaporation_rate(
        T, qv * _KG, qr * _KG, 1.2e-2, r), (289.0, 7.0, 0.2, 1.05)),
    "autoconversion": (lambda qc: K.cloudbench_autoconversion(qc * _KG), (0.9,)),
}


@pytest.mark.parametrize("name", sorted(_GRAD_CASES))
def test_new_rates_pass_check_grads(name):
    """Mixing ratios enter in g/kg so the finite-difference step is small against them."""
    fn, args = _GRAD_CASES[name]
    check_grads(fn, tuple(jnp.asarray(x) for x in args), order=2, modes=("fwd", "rev"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_scheme_gradient_is_finite_at_rain_and_cloud_free_start(dtype):
    """jit(grad) wrt q_v, q_c AND q_r at q_r = 0 everywhere and q_c = 0 in the lower
    column: the guarded fractional powers of q_r (accretion, evaporation, Chen speed)
    must not leak NaN/inf."""
    T, q_v, q_c, q_r, p, rho, dz = (a.astype(dtype) for a in _column())
    q_r = jnp.zeros_like(q_r)

    def loss(qv, qc, qr):
        out = _call(T, qv, qc, qr, p, rho, dz, 5.0, CB)
        return (jnp.sum(out.dq_r_dt) + jnp.sum(out.dq_c_dt) + jnp.sum(out.dT_dt)
                + jnp.sum(out.precipitation))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q_v, q_c, q_r)
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(grads[2]))) > 0.0      # rain sensitivity is not cut
    g_fall = jax.grad(K.chen2022_rain_fall_speed)(jnp.asarray(0.0, dtype),
                                                  jnp.asarray(1.1, dtype))
    assert bool(jnp.isfinite(g_fall))


# --- each rate reaches the scheme's tendencies (isolated single cells) -------
def _cell(T, q_v, q_c, q_r, p=8.0e4, rho=1.0, dz=5.0e3, dt=1.0):
    a = lambda v: jnp.array([[v]], dtype=jnp.float64)       # noqa: E731
    return _call(a(T), a(q_v), a(q_c), a(q_r), a(p), a(rho), a(dz), dt, CB)


def _f(x):
    return float(np.asarray(x).ravel()[0])


def _qs(T, p):
    return _f(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))


def test_scheme_autoconversion_is_the_only_cloud_sink_when_isolated():
    """Saturated (no adjustment), q_r = 0 (no accretion/evaporation/settling)."""
    qs = _qs(290.0, 8.0e4)
    o = _cell(290.0, qs, 9.0e-4, 0.0)
    auto = (9.0e-4 - 5.0e-4) / 1.0e3
    assert _f(o.dq_c_dt) == pytest.approx(-auto, rel=1e-12)
    assert _f(o.dq_r_dt) == pytest.approx(auto, rel=1e-12)


def test_scheme_accretion_is_the_only_cloud_sink_below_threshold():
    """Saturated, q_c below 5e-4 (no autoconversion)."""
    qs = _qs(290.0, 8.0e4)
    o = _cell(290.0, qs, 3.0e-4, 1.0e-4)
    assert _f(o.dq_c_dt) == pytest.approx(-_ref_accretion(3.0e-4, 1.0e-4, 1.0), rel=1e-12)


def test_scheme_rain_evaporation_feeds_vapour_and_cools():
    """Sub-saturated, cloud-free rainy cell: dq_v = +evap (unclamped), dT = -L_v evap/c_pd."""
    T, p = 290.0, 8.0e4
    qs = _qs(T, p)
    q_v, q_r = 0.8 * qs, 1.0e-4
    o = _cell(T, q_v, 0.0, q_r)
    evap = _ref_evaporation(T, q_v, q_r, qs, 1.0)
    assert 0.0 < evap < 1e-2 * q_r                           # far below both caps
    assert _f(o.dq_v_dt) == pytest.approx(evap, rel=1e-12)
    assert _f(o.dT_dt) == pytest.approx(-constants.L_v * evap / constants.c_pd, rel=1e-12)


def test_scheme_surface_precipitation_uses_the_chen_fall_speed():
    """Single saturated layer, q_c = 0: precipitation = rho q_r v_Chen (unclamped)."""
    qs = _qs(290.0, 8.0e4)
    q_r, rho = 1.0e-4, 1.0
    o = _cell(290.0, qs, 0.0, q_r, rho=rho)
    assert _f(o.precipitation) == pytest.approx(rho * q_r * _ref_chen(q_r, rho), rel=1e-12)


# --- RULE 0 budgets ------------------------------------------------------------
def _column():
    """Two top-down columns (surface last): cloudy aloft, sub-saturated rain below."""
    nlev = 6
    p = jnp.linspace(7.0e4, 1.0e5, nlev)[None, :].repeat(2, 0)
    T = jnp.linspace(280.0, 298.0, nlev)[None, :].repeat(2, 0)
    q_sat = saturation_specific_humidity(T, p)
    # sub-saturated mid levels (rain evaporates) over a near-saturated surface
    # layer, so rain also reaches the ground at every tested step
    rh = jnp.array([[1.002, 1.001, 1.0, 0.9, 0.97, 0.999],
                    [1.0, 1.003, 0.95, 0.85, 0.96, 0.998]])
    q_v = rh * q_sat
    q_c = jnp.array([[9e-4, 1.2e-3, 6e-4, 0.0, 0.0, 0.0],
                     [2e-4, 8e-4, 1e-4, 0.0, 0.0, 0.0]])
    q_r = jnp.array([[1e-5, 2e-4, 5e-4, 6e-4, 4e-4, 3e-3],
                     [0.0, 5e-5, 1e-4, 2e-3, 1e-3, 4e-3]])
    rho = p / (constants.R_d * T)
    dz = jnp.full_like(p, 300.0)
    return T, q_v, q_c, q_r, p, rho, dz


def _call(T, q_v, q_c, q_r, p, rho, dz, dt, cfg):
    z = jnp.zeros_like(q_c)
    hyd = HydrometeorState(q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)
    p_half = jnp.concatenate([p[:, :1] * 0.98, p], axis=1)
    return K.kessler_microphysics(T, q_v, hyd, p, p_half, rho, dz, dt, cfg)


@pytest.mark.parametrize("dt", [1.0, 30.0, 600.0])
def test_water_and_enthalpy_close_to_round_off(dt):
    T, q_v, q_c, q_r, p, rho, dz = _column()
    out = _call(T, q_v, q_c, q_r, p, rho, dz, dt, CB)
    w = rho * dz
    d_water = jnp.sum(w * (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt), axis=1) * dt
    precip = out.precipitation.reshape(-1) * dt
    scale = jnp.sum(w * (q_v + q_c + q_r), axis=1)
    # non-vacuous: rain does reach the surface, so omitting it breaks the budget
    assert bool(jnp.all(precip > 1e-9 * scale))
    np.testing.assert_allclose(np.asarray(d_water + precip), 0.0,
                               atol=float(1e-14 * jnp.max(scale)))
    d_h = jnp.sum(w * (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt),
                  axis=1) * dt
    h_scale = jnp.sum(w * constants.L_v * q_v, axis=1)
    np.testing.assert_allclose(np.asarray(d_h), 0.0, atol=float(1e-14 * jnp.max(h_scale)))
    # the processes the budget couples are active in this column
    q_sat = saturation_specific_humidity(T, p)
    assert float(jnp.min(out.dq_v_dt)) < 0.0                 # condensation
    assert float(jnp.max(K.cloudbench_rain_evaporation_rate(
        T, q_v, q_r, q_sat, rho))) > 0.0                      # rain evaporation
    assert float(jnp.max(K.cloudbench_autoconversion(q_c))) > 0.0
    assert float(jnp.max(K.cloudbench_accretion(q_c, q_r, rho))) > 0.0
    assert float(jnp.max(jnp.abs(out.dq_c_dt))) > 0.0
    new = {k: v + dt * getattr(out, f"d{k}_dt") for k, v in
           (("q_v", q_v), ("q_c", q_c), ("q_r", q_r))}
    for k, v in new.items():
        assert float(jnp.min(v)) >= -1e-15, k


def test_cloudbench_set_differs_from_classic():
    T, q_v, q_c, q_r, p, rho, dz = _column()
    a = _call(T, q_v, q_c, q_r, p, rho, dz, 5.0, KesslerConfig())
    b = _call(T, q_v, q_c, q_r, p, rho, dz, 5.0, CB)
    assert not np.allclose(np.asarray(a.precipitation), np.asarray(b.precipitation))


# --- selection -----------------------------------------------------------------
def test_classic_is_the_default_and_unknown_set_raises():
    assert KesslerConfig().parameter_set == "classic"
    T, q_v, q_c, q_r, p, rho, dz = _column()
    with pytest.raises(ValueError, match="unknown Kessler"):
        _call(T, q_v, q_c, q_r, p, rho, dz, 5.0, KesslerConfig(parameter_set="cloudbnech"))


def test_cloudbench_set_ignores_the_classic_rate_fields():
    """The classic float knobs (autoconversion, accretion, evaporation, fall speed)
    must not leak into the CloudBench rates."""
    T, q_v, q_c, q_r, p, rho, dz = _column()
    tweaked = KesslerConfig(parameter_set="cloudbench", autoconversion_threshold=2e-3,
                            autoconversion_rate=5e-3, accretion_coeff=6.0,
                            evaporation_coeff=3.0, rain_fall_speed=12.0)
    a = _call(T, q_v, q_c, q_r, p, rho, dz, 5.0, CB)
    b = _call(T, q_v, q_c, q_r, p, rho, dz, 5.0, tweaked)
    for f in ("dT_dt", "dq_v_dt", "dq_c_dt", "dq_r_dt", "precipitation"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))
