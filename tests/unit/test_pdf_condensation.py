"""Uniform-PDF (Sundqvist-compatible) saturation adjustment kernel.

Each test names the planted error it catches: (a) a wrong PDF integral,
(b) a spurious clear-fraction evaporation sink or non-idempotent equilibrium,
(c) a wrong sign of the thermodynamic response, (d) a cover relation that is
not Sundqvist's, (e) a broken water / moist-enthalpy identity (also shown to
FAIL with the latent-heat sign flipped), (f) a detached or wrong implicit
gradient, jit/eager divergence, float32 blow-ups.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere._future.pdf_condensation import (  # noqa: E402
    pdf_saturation_adjustment,
    uniform_pdf_cloud,
)
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402

RH_C = 0.85
DT = 112.5


def _sat(T, p):
    return float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p)))


def _equilibrium(c, T=280.0, p=7.0e4, rh_c=RH_C):
    s = _sat(T, p)
    D = (1.0 - rh_c) * s
    return jnp.asarray(s - D * (1.0 - c) ** 2), jnp.asarray(D * c ** 2), jnp.asarray(T), jnp.asarray(p)


def test_a_closed_form_matches_numerical_pdf_integration():
    rng = np.random.default_rng(0)
    for k in range(5):
        T, p, rh_c = 270.0 + 20.0 * rng.random(), 5.0e4 + 5.0e4 * rng.random(), 0.7 + 0.25 * rng.random()
        s = _sat(T, p)
        for q_t in (0.5 * s, 0.9 * s, s, 1.05 * s, 1.5 * s, 1.0e-9):
            c, Q, s_out = (float(v) for v in uniform_pdf_cloud(jnp.asarray(q_t), jnp.asarray(T), jnp.asarray(p), rh_c))
            D = max(min((1.0 - rh_c) * s, q_t), 1e-12)
            q = np.linspace(q_t - D, q_t + D, 4001)
            Q_num = np.trapezoid(np.maximum(q - s, 0.0), q) / (2.0 * D)
            c_num = np.trapezoid((q > s).astype(float), q) / (2.0 * D)
            assert abs(s_out - s) < 1e-12 * s
            np.testing.assert_allclose(Q, Q_num, rtol=1e-5, atol=1e-15)   # trapezoid error at the kink
            np.testing.assert_allclose(c, c_num, rtol=1e-6, atol=2e-3)


def test_b_partial_cloud_equilibrium_persists_and_is_restored():
    q_v, q_c, T, p = _equilibrium(0.4)
    for _ in range(20):
        x, c, T_new = pdf_saturation_adjustment(T, q_v, q_c, p, DT, RH_C)
        assert abs(float(x)) < 1e-12
        q_v, q_c, T = q_v - x, q_c + x, T_new
    assert float(q_v) / _sat(float(T), float(p)) < 1.0 and float(q_c) > 0.0
    assert abs(float(c) - 0.4) < 1e-6
    x, _, _ = pdf_saturation_adjustment(T, q_v, 0.7 * q_c, p, DT, RH_C)   # a fake clear-air sink
    assert float(x) > 0.0, "the kernel must condense back toward the equilibrium"


def test_c_thermodynamic_response_signs_and_positivity():
    q_v, q_c, T, p = _equilibrium(0.4)
    x_cool, _, _ = pdf_saturation_adjustment(T - 1.0, q_v, q_c, p, DT, RH_C)
    x_warm, _, _ = pdf_saturation_adjustment(T + 1.0, q_v, q_c, p, DT, RH_C)
    assert float(x_cool) > 0.0 and float(x_warm) < 0.0
    s = _sat(float(T), float(p)); D = (1.0 - RH_C) * s
    q_t = s + 2.0 * D                                   # overcast: q_c = q_t - s
    x_ov, c_ov, _ = pdf_saturation_adjustment(T + 1.0, jnp.asarray(s), jnp.asarray(q_t - s), p, DT, RH_C)
    assert float(x_ov) < 0.0 and float(c_ov) == 1.0
    x_dry, c_dry, _ = pdf_saturation_adjustment(T, jnp.asarray(0.5 * s), jnp.asarray(0.0), p, DT, RH_C)
    assert float(x_dry) == 0.0 and float(c_dry) == 0.0
    for x, qv, qc in ((x_cool, q_v, q_c), (x_warm, q_v, q_c), (x_ov, jnp.asarray(s), jnp.asarray(q_t - s))):
        assert float(qv - x) >= 0.0 and float(qc + x) >= 0.0


def test_d_cover_after_adjustment_is_the_sundqvist_relation():
    rng = np.random.default_rng(1)
    T = jnp.asarray(265.0 + 30.0 * rng.random(200)); p = jnp.asarray(5.0e4 + 5.0e4 * rng.random(200))
    s = saturation_mixing_ratio(T, p)
    q_t = s * (0.5 + 0.8 * rng.random(200)); q_c = q_t * 0.3 * rng.random(200); q_v = q_t - q_c
    x, c, T_new = pdf_saturation_adjustment(T, q_v, q_c, p, DT, RH_C)
    rh = np.asarray((q_v - x) / saturation_mixing_ratio(T_new, p)); c = np.asarray(c)
    m = (c > 0.02) & (c < 0.98)
    assert m.sum() > 20
    np.testing.assert_allclose(c[m], 1.0 - np.sqrt((1.0 - rh[m]) / (1.0 - RH_C)), rtol=1e-6)


def _random_columns(n, seed):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(250.0 + 50.0 * rng.random(n)); p = jnp.asarray(3.0e4 + 7.0e4 * rng.random(n))
    s = saturation_mixing_ratio(T, p)
    q_t = s * (0.4 + 1.0 * rng.random(n)); q_c = q_t * 0.4 * rng.random(n)
    return T, q_t - q_c, q_c, p


def test_e_water_and_moist_enthalpy_conserved_and_planted_sign_error_caught():
    T, q_v, q_c, p = _random_columns(1000, 2)
    x, _, T_new = pdf_saturation_adjustment(T, q_v, q_c, p, DT, RH_C)
    np.testing.assert_allclose(np.asarray((q_v - x) + (q_c + x)), np.asarray(q_v + q_c), rtol=0, atol=1e-15)
    h_res = np.asarray(constants.c_pd * (T_new - T) - constants.L_v * x)
    assert np.abs(h_res).max() < 1e-9 * max(1.0, float(constants.L_v * jnp.abs(x).max()))
    # planted error: latent heat with the wrong sign breaks the enthalpy identity
    x_bad, _, T_bad = pdf_saturation_adjustment(T, q_v, q_c, p, DT, RH_C, l_over_cp=-constants.L_v / constants.c_pd)
    h_bad = np.asarray(constants.c_pd * (T_bad - T) - constants.L_v * x_bad)
    assert np.abs(h_bad).max() > 1e-3 * float(constants.L_v * jnp.abs(x_bad).max())


def test_f_jit_parity_implicit_gradient_and_float32():
    f = lambda T, q_v, q_c, p: pdf_saturation_adjustment(T, q_v, q_c, p, DT, RH_C)[0]
    states = [_equilibrium(0.3), _equilibrium(0.6, T=290.0, p=9.0e4), _equilibrium(0.8, T=270.0, p=5.0e4)]
    for q_v, q_c, T, p in states:
        T2 = T - 0.3                                   # off the equilibrium: a genuine transfer
        x_e = f(T2, q_v, q_c, p); x_j = jax.jit(f)(T2, q_v, q_c, p)
        assert abs(float(x_e - x_j)) < 1e-14
        g = jax.grad(lambda a, b, c_: f(a, b, c_, p), argnums=(0, 1, 2))(T2, q_v, q_c)
        for i, (arg, name) in enumerate(((T2, "T"), (q_v, "q_v"), (q_c, "q_c"))):
            h = 1e-4 * max(abs(float(arg)), 1e-3)
            args = [T2, q_v, q_c]
            args_p = list(args); args_p[i] = arg + h
            args_m = list(args); args_m[i] = arg - h
            fd = (float(f(*args_p, p)) - float(f(*args_m, p))) / (2.0 * h)
            np.testing.assert_allclose(float(g[i]), fd, rtol=1e-4, atol=1e-12, err_msg=name)
    s = _sat(280.0, 7.0e4); D = (1.0 - RH_C) * s
    for q_v, q_c in ((jnp.asarray(0.5 * s), jnp.asarray(0.0)), (jnp.asarray(s), jnp.asarray(2.0 * D))):
        g = jax.grad(lambda a, b, c_: f(a, b, c_, jnp.asarray(7.0e4)), argnums=(0, 1, 2))(jnp.asarray(280.0), q_v, q_c)
        assert all(np.isfinite(float(v)) for v in g)
    T32, q_v32, q_c32, p32 = (jnp.asarray(v, dtype=jnp.float32) for v in _random_columns(64, 3))
    x32, c32, T32n = pdf_saturation_adjustment(T32, q_v32, q_c32, p32, DT, RH_C)
    assert x32.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(x32))) and bool(jnp.all(jnp.isfinite(T32n)))
    g32 = jax.grad(lambda a: jnp.sum(pdf_saturation_adjustment(a, q_v32, q_c32, p32, DT, RH_C)[0]))(T32)
    assert bool(jnp.all(jnp.isfinite(g32)))


# Parked module: see its docstring.
pytestmark = pytest.mark.skip(
    reason="parked in _future/: not wired into production (ponytail item pdf_condensation)")
