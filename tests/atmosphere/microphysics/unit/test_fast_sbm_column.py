"""fast_sbm as a switchable scheme: dispatch + column-operator physics.

Dispatch is exercised through the PUBLIC MicrophysicsConfig (CLAUDE.md
config-dispatch rule). Physics: condensation closure on (ncol, nlev)
fields, emergent autoconversion (cloud→rain mass transfer through the
resolved spectrum — no parameterized rate), clear-cell fixed point.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics import (
    FastSBMConfig,
    MicrophysicsConfig,
    fast_sbm_microphysics,
    make_zero_hydrometeors,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    _get_microphysics_fn,
)
from legoesm.thermo import saturation_vapor_pressure

jax.config.update("jax_enable_x64", True)

NCOL, NLEV = 2, 3
P0, T0 = 9.0e4, 283.0
DT = 2.0


def _fields(rh, q_c=1.0e-3, q_r=0.0):
    T = jnp.full((NCOL, NLEV), T0)
    p = jnp.full((NCOL, NLEV), P0)
    e = rh * float(saturation_vapor_pressure(jnp.asarray(T0)))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)
    hyd = hyd._replace(q_c=jnp.full((NCOL, NLEV), q_c),
                       q_r=jnp.full((NCOL, NLEV), q_r))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)
    return T, q_v, hyd, p, p_half, rho, dz


def test_dispatch_via_public_config():
    cfg = MicrophysicsConfig(scheme="fast_sbm")
    name, fn, scheme_cfg = _get_microphysics_fn(cfg)
    assert name == "fast_sbm"
    assert fn is fast_sbm_microphysics
    assert isinstance(scheme_cfg, FastSBMConfig)
    with pytest.raises(ValueError, match="Unknown microphysics"):
        _get_microphysics_fn(MicrophysicsConfig(scheme="fast_sbmm"))


def test_unknown_collision_kernel_raises():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    with pytest.raises(ValueError, match="collision_kernel"):
        fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT,
                              FastSBMConfig(collision_kernel="hal"))


def test_condensation_closure_on_fields():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert out.dT_dt.shape == (NCOL, NLEV)
    dql = np.asarray(out.dq_c_dt + out.dq_r_dt)
    assert np.all(dql > 0.0)                       # supersaturated → grows
    # Vapor/heat closure per cell.
    np.testing.assert_allclose(np.asarray(out.dq_v_dt), -dql, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(out.dT_dt),
        (constants.L_v / constants.c_pd) * dql, rtol=1e-10)
    # Warm-only: ice tendencies identically zero.
    np.testing.assert_array_equal(np.asarray(out.dq_i_dt), 0.0)
    np.testing.assert_array_equal(np.asarray(out.precipitation), 0.0)


def test_emergent_autoconversion_dense_vs_thin():
    # Mass crossing KRDROP comes from resolved coalescence: a dense cloud
    # must convert far more cloud→rain than a thin one (no tuned
    # autoconversion threshold/rate anywhere in the scheme).
    T, q_v, hyd_thin, p, p_half, rho, dz = _fields(1.0, q_c=5.0e-5)
    _, _, hyd_dense, _, _, _, _ = _fields(1.0, q_c=2.0e-3)
    cfg = FastSBMConfig(collision_kernel="hall")
    out_thin = fast_sbm_microphysics(T, q_v, hyd_thin, p, p_half, rho, dz,
                                     DT, cfg)
    out_dense = fast_sbm_microphysics(T, q_v, hyd_dense, p, p_half, rho,
                                      dz, DT, cfg)
    rain_thin = float(out_thin.dq_r_dt[0, 0])
    rain_dense = float(out_dense.dq_r_dt[0, 0])
    assert rain_dense > 0.0
    assert rain_dense > 50.0 * max(rain_thin, 0.0) or rain_thin <= 0.0
    # Coalescence conserves liquid: rain gain ≈ cloud loss at S = 0 ...
    # (condensation at exactly S=0 contributes ~nothing).
    np.testing.assert_allclose(
        float(out_dense.dq_c_dt[0, 0] + out_dense.dq_r_dt[0, 0]),
        0.0, atol=5.0e-9)


def test_clear_cell_fixed_point():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.05, q_c=0.0, q_r=0.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    # No spectrum, no activation in the adapter → nothing happens.
    for fld in (out.dT_dt, out.dq_v_dt, out.dq_c_dt, out.dq_r_dt):
        np.testing.assert_allclose(np.asarray(fld), 0.0, atol=1e-15)


def test_column_jit_and_grad():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)

    @jax.jit
    def total_heating(qv):
        out = fast_sbm_microphysics(T, qv, hyd, p, p_half, rho, dz, DT)
        return jnp.sum(out.dT_dt)

    val = float(total_heating(q_v))
    g = jax.grad(total_heating)(q_v)
    assert np.isfinite(val) and val > 0.0
    assert np.all(np.isfinite(np.asarray(g)))
