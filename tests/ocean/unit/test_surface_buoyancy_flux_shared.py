"""Direct unit test for the shared grid-agnostic surface buoyancy-flux kernel
(#518 item 1).

`surface_buoyancy_flux` replaces three byte-identical-in-logic copies: the
lat-lon `integration.py` inline block, `k_profiles._surface_buoyancy_flux`, and
`mpas_integration._mpas_surface_buoyancy_flux`.  This pins BOTH conventions to
the exact pre-#518 inline formulas:

* ``real_salt_in_qs=True`` (lat-lon): real salt feeds BOTH the buoyancy and the
  combined non-local Q_sfc_S.
* ``real_salt_in_qs=False`` (MPAS): real salt feeds the buoyancy ONLY; Q_sfc_S
  carries the freshwater term only (zero baseline).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.eos import (
    rho_0 as _RHO_0,
    c_sw as _C_SW,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)
from legoesm.ocean.physics.vertical_mixing._shared import surface_buoyancy_flux


def _ref_latlon(q_net, fw, salt, T_sfc, S_sfc, g, rho_0, c_sw):
    """Exact pre-#518 lat-lon inline formula (real salt in combined Q_sfc_S)."""
    p_sfc = jnp.zeros_like(T_sfc)
    Q_sfc_T = None
    B_f = None
    if q_net is not None:
        Q_sfc_T = q_net / (rho_0 * c_sw)
        alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
        B_f = -g * alpha * Q_sfc_T
    Q_sfc_S = None
    if fw is not None or salt is not None:
        beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
        Q_sfc_S = jnp.zeros_like(S_sfc)
        if fw is not None:
            Q_sfc_S = Q_sfc_S - S_sfc * fw / rho_0
        if salt is not None:
            Q_sfc_S = Q_sfc_S + salt * 1.0e3 / rho_0
        B_salt = g * beta * Q_sfc_S
        B_f = B_salt if B_f is None else (B_f + B_salt)
    return B_f, Q_sfc_T, Q_sfc_S


def _ref_mpas(q_net, fw, salt, T_sfc, S_sfc, g, rho_0, c_sw):
    """Exact pre-#518 MPAS inline formula (real salt in buoyancy only)."""
    p_sfc = jnp.zeros_like(T_sfc)
    Q_sfc_T = None
    B_f = None
    if q_net is not None:
        Q_sfc_T = q_net / (rho_0 * c_sw)
        alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
        B_f = -g * alpha * Q_sfc_T
    Q_sfc_S = None
    if fw is not None or salt is not None:
        beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
        B_salt = jnp.zeros_like(S_sfc)
        Q_sfc_S = jnp.zeros_like(S_sfc)
        if fw is not None:
            Q_sfc_S = -S_sfc * jnp.asarray(fw, S_sfc.dtype) / rho_0
            B_salt = B_salt + g * beta * Q_sfc_S
        if salt is not None:
            B_salt = B_salt + g * beta * (
                jnp.asarray(salt, S_sfc.dtype) * 1.0e3 / rho_0)
        B_f = B_salt if B_f is None else (B_f + B_salt)
    return B_f, Q_sfc_T, Q_sfc_S


def _surface(seed=0):
    rng = np.random.default_rng(seed)
    n = 16
    T = jnp.asarray(2.0 + 20.0 * rng.random(n), dtype=jnp.float64)
    S = jnp.asarray(33.0 + 3.0 * rng.random(n), dtype=jnp.float64)
    q = jnp.asarray(rng.standard_normal(n) * 50.0, dtype=jnp.float64)
    fw = jnp.asarray(rng.standard_normal(n) * 1e-6, dtype=jnp.float64)
    salt = jnp.asarray(rng.standard_normal(n) * 1e-5, dtype=jnp.float64)
    return q, fw, salt, T, S


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b))


_CASES = [
    "qfs", "q__", "_f_", "__s", "qf_", "_fs", "q_s", "___",
]


def _select(q, fw, salt, code):
    return (
        q if code[0] == "q" else None,
        fw if code[1] == "f" else None,
        salt if code[2] == "s" else None,
    )


def test_latlon_byte_identical():
    q, fw, salt, T, S = _surface()
    g, rho_0, c_sw = constants.g, _RHO_0, _C_SW
    for code in _CASES:
        qn, f, s = _select(q, fw, salt, code)
        got = surface_buoyancy_flux(
            qn, f, s, T, S, g=g, rho_0=rho_0, c_sw=c_sw, real_salt_in_qs=True)
        ref = _ref_latlon(qn, f, s, T, S, g, rho_0, c_sw)
        for g_, r_ in zip(got, ref):
            assert _eq(g_, r_), code


def test_mpas_byte_identical():
    q, fw, salt, T, S = _surface()
    g, rho_0, c_sw = constants.g, _RHO_0, _C_SW
    for code in _CASES:
        qn, f, s = _select(q, fw, salt, code)
        got = surface_buoyancy_flux(
            qn, f, s, T, S, g=g, rho_0=rho_0, c_sw=c_sw, real_salt_in_qs=False)
        ref = _ref_mpas(qn, f, s, T, S, g, rho_0, c_sw)
        for g_, r_ in zip(got, ref):
            assert _eq(g_, r_), code


def test_conventions_differ_on_real_salt():
    """The two conventions agree on B_f but differ on Q_sfc_S when real salt
    is present: lat-lon folds it into Q_sfc_S, MPAS keeps Q_sfc_S=0 (fw=None)."""
    _, _, salt, T, S = _surface()
    salt_pos = jnp.abs(salt) + 1e-6
    g, rho_0, c_sw = constants.g, _RHO_0, _C_SW
    B_ll, _, Qs_ll = surface_buoyancy_flux(
        None, None, salt_pos, T, S, g=g, rho_0=rho_0, c_sw=c_sw,
        real_salt_in_qs=True)
    B_mp, _, Qs_mp = surface_buoyancy_flux(
        None, None, salt_pos, T, S, g=g, rho_0=rho_0, c_sw=c_sw,
        real_salt_in_qs=False)
    # MPAS: salt-only -> Q_sfc_S stays at zero baseline.
    assert np.allclose(np.asarray(Qs_mp), 0.0)
    # Lat-lon: salt enters Q_sfc_S (>0 brine in).
    assert np.all(np.asarray(Qs_ll) > 0.0)
    # Both destabilise (B_f > 0) by the same total buoyancy.
    assert np.all(np.asarray(B_ll) > 0.0)
    assert np.all(np.asarray(B_mp) > 0.0)


def test_no_forcing_returns_all_none():
    _, _, _, T, S = _surface()
    g, rho_0, c_sw = constants.g, _RHO_0, _C_SW
    for flag in (True, False):
        out = surface_buoyancy_flux(
            None, None, None, T, S, g=g, rho_0=rho_0, c_sw=c_sw,
            real_salt_in_qs=flag)
        assert out == (None, None, None)
