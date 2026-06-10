"""Tests for ``eos="veros_gsw"`` — Veros's ``eq_of_state_type=5``.

TEOS-10 48-term computationally-efficient rational polynomial for in-situ
density rho(SA, CT, p) (IOC, SCOR and IAPSO 2010), ported
coefficient-for-coefficient from ``veros/core/density/gsw.py`` — the EOS the
Veros ``global_4deg`` setup uses (``settings.eq_of_state_type = 5``).

Parity provenance: the hardcoded anchors below were computed from the actual
Veros kernels (``veros/core/density/gsw.py`` @ /home/dbalwada/veros,
NumPy backend, 2026-06-10) via::

    gsw_rho(S, T, press), gsw_drhodT(S, T, press), gsw_drhodS(S, T, press),
    gsw_dyn_enthalpy(S, T, press),
    -(1024.0 / 9.81) * gsw_dHdT(S, T, press),   # = get_int_drhodT, type 5
    -(1024.0 / 9.81) * gsw_dHdS(S, T, press),   # = get_int_drhodS, type 5

with ``press = abs(zt)`` in METERS (Veros's meters-as-dbar convention). The
full-grid parity sweep (17 T x 11 S x 15 global_4deg levels) gave max-rel
3.8e-14 (rho), 6.3e-15 (drhodT), 3.8e-15 (drhodS), 2.2e-14 (int_drhodS) and
~5e-12 only at near-zero crossings of Hd / int_drhodT (ULP-level XLA-vs-NumPy
log/sqrt differences; range-scaled error <= 2e-13).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.eos import (
    VerosGswConfig,
    make_eos_fn,
    veros_gsw_drhodS,
    veros_gsw_drhodT,
    veros_gsw_dyn_enthalpy,
    veros_gsw_eos,
    veros_gsw_int_drhodTS_dynamic_enthalpy,
)

jax.config.update("jax_enable_x64", True)

# Veros conversion constants (veros/core/density/gsw.py ``rho0`` and the
# get_rho.py ``1024.0 / 9.81`` int_drhodT/S prefactor).
_RHO0 = 1024.0
_GRAV = 9.81

# Veros-computed regression anchors — see module docstring for provenance.
# (T_C, S, press_m_as_dbar,
#  rho_anom, drhodT, drhodS,
#  Hd, int_drhodT, int_drhodS)
_VEROS_ANCHORS = [
    (-2.0, 30.0, 25.0,
     0.13183190426252622, -0.010756501259498105, 0.8086656610997323,
     -0.016915102542526483, -0.257380553536643, 20.12527713663619),
    (2.0, 34.5, 455.0,
     5.573900359779827, -0.09131378298406036, 0.7926191596607082,
     -19.48159933826173, -38.11765046182414, 356.745085409133),
    (10.0, 35.0, 1250.0,
     8.379697331427678, -0.19739634817545842, 0.7644891013101258,
     -66.54536952451963, -226.5728390971812, 946.757923869318),
    (20.0, 36.0, 85.0,
     1.7572350197040123, -0.2667074832939526, 0.7501050088241684,
     -1.2740441646243426, -22.442833291189455, 63.30016033202508),
    (25.0, 38.0, 2495.0,
     11.713423090463039, -0.34080254429246637, 0.7259004993194179,
     -156.4279527588078, -797.4116647255568, 1798.0664143061304),
    (29.0, 33.0, 4855.0,
     15.88380485394714, -0.3832594606908437, 0.7053828310345975,
     -290.8852837108716, -1704.123355591727, 3430.713143939649),
]


def _p_pa(press_m):
    """Veros press (meters-as-dbar) -> legoESM Pa, inverse of the wrapper's
    ``press = p / (rho_0 * grav)`` conversion."""
    return press_m * _RHO0 * _GRAV


def _anchor_arrays():
    a = np.asarray(_VEROS_ANCHORS)
    T = jnp.asarray(a[:, 0])
    S = jnp.asarray(a[:, 1])
    press = jnp.asarray(a[:, 2])
    return T, S, press, a


def test_default_config_matches_veros_source():
    cfg = VerosGswConfig()
    assert cfg.rho_0 == _RHO0
    assert cfg.grav == _GRAV


def test_rho_matches_veros_anchors():
    T, S, press, a = _anchor_arrays()
    rho = veros_gsw_eos(T, S, _p_pa(press))
    np.testing.assert_allclose(
        np.asarray(rho) - _RHO0, a[:, 3], rtol=1e-10, atol=1e-12,
    )


def test_drhodT_drhodS_match_veros_anchors():
    T, S, press, a = _anchor_arrays()
    drt = veros_gsw_drhodT(T, S, _p_pa(press))
    drs = veros_gsw_drhodS(T, S, _p_pa(press))
    np.testing.assert_allclose(np.asarray(drt), a[:, 4], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(drs), a[:, 5], rtol=1e-10)


def test_dyn_enthalpy_matches_veros_anchors():
    T, S, press, a = _anchor_arrays()
    hd = veros_gsw_dyn_enthalpy(T, S, _p_pa(press))
    # atol floor covers the near-zero Hd anchor (ULP-level transcendental
    # XLA-vs-NumPy differences amplify relatively at zero crossings).
    np.testing.assert_allclose(np.asarray(hd), a[:, 6], rtol=1e-10, atol=1e-11)


def test_int_drhodTS_matches_veros_anchors():
    """Exact dynamic-enthalpy integrands = -(rho_0/grav)*gsw_dHdT/S, the
    type-5 branch of Veros get_int_drhodT/S. z_full is negative below the
    surface; Veros press = |z_full| in meters."""
    T, S, press, a = _anchor_arrays()
    int_T, int_S = veros_gsw_int_drhodTS_dynamic_enthalpy(T, S, -press)
    np.testing.assert_allclose(np.asarray(int_T), a[:, 7], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(np.asarray(int_S), a[:, 8], rtol=1e-10, atol=1e-11)


def test_autodiff_matches_analytic_derivatives():
    """jax.grad of the ported rho must agree with the ported analytic
    drhodT/drhodS (catches polynomial-coefficient transcription errors:
    the two code paths share v01..v48 but the derivatives add independent
    a01..a33 / b01..b24 tables). Required tol 1e-6; observed ~1e-13."""
    T, S, press, _ = _anchor_arrays()
    p = _p_pa(press)
    gT = jax.vmap(jax.grad(veros_gsw_eos, argnums=0))(T, S, p)
    gS = jax.vmap(jax.grad(veros_gsw_eos, argnums=1))(T, S, p)
    np.testing.assert_allclose(
        np.asarray(gT), np.asarray(veros_gsw_drhodT(T, S, p)), rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(gS), np.asarray(veros_gsw_drhodS(T, S, p)), rtol=1e-6,
    )


def test_autodiff_dyn_enthalpy_matches_int_drhodTS():
    """jax.grad of the ported Hd must agree with the analytic Maple-generated
    dHdT/dHdS (exposed via the int_drhodTS prefactor -(rho_0/grav))."""
    T, S, press, _ = _anchor_arrays()
    p = _p_pa(press)
    hT = jax.vmap(jax.grad(veros_gsw_dyn_enthalpy, argnums=0))(T, S, p)
    hS = jax.vmap(jax.grad(veros_gsw_dyn_enthalpy, argnums=1))(T, S, p)
    int_T, int_S = veros_gsw_int_drhodTS_dynamic_enthalpy(T, S, -press)
    pref = -(_RHO0 / _GRAV)
    np.testing.assert_allclose(np.asarray(hT) * pref, np.asarray(int_T), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(hS) * pref, np.asarray(int_S), rtol=1e-6)


def test_dispatch_via_make_eos_fn():
    fn = make_eos_fn("veros_gsw")
    T, S, press, a = _anchor_arrays()
    np.testing.assert_allclose(
        np.asarray(fn(T, S, _p_pa(press))) - _RHO0, a[:, 3],
        rtol=1e-10, atol=1e-12,
    )


def test_differentiability_smoke_finite_nonzero():
    """grads wrt T, S, p are finite, nonzero, and physically signed."""
    T0 = jnp.asarray(10.0)
    S0 = jnp.asarray(35.0)
    p0 = jnp.asarray(_p_pa(1250.0))
    gT = float(jax.grad(veros_gsw_eos, argnums=0)(T0, S0, p0))
    gS = float(jax.grad(veros_gsw_eos, argnums=1)(T0, S0, p0))
    gp = float(jax.grad(veros_gsw_eos, argnums=2)(T0, S0, p0))
    for g in (gT, gS, gp):
        assert np.isfinite(g) and g != 0.0
    assert gT < 0   # warming reduces density (T > T_max-density)
    assert gS > 0   # salting increases density
    assert gp > 0   # compression increases density
    # dyn-enthalpy path differentiable too (log/sqrt branch)
    gH = float(jax.grad(veros_gsw_dyn_enthalpy, argnums=0)(T0, S0, p0))
    assert np.isfinite(gH) and gH != 0.0


def test_pressure_dependence_active():
    """Unlike nonlin3, gsw has compressibility — deeper is denser."""
    T0 = jnp.asarray(10.0)
    S0 = jnp.asarray(35.0)
    rho_surf = float(veros_gsw_eos(T0, S0, jnp.asarray(0.0)))
    rho_deep = float(veros_gsw_eos(T0, S0, jnp.asarray(_p_pa(4000.0))))
    assert rho_deep > rho_surf
    # ~4.5 kg/m^3 per 1000 dbar for seawater
    assert 10.0 < rho_deep - rho_surf < 25.0
