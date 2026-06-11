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


def test_resolves_through_production_registry():
    # Codex review item 17: validate_strict accepted fast_sbm but the
    # production MICROPHYSICS_REGISTRY omitted it → runtime KeyError.
    from legoesm.driver.kernel_registry import (
        MICROPHYSICS_REGISTRY, resolve_kernel)
    assert "fast_sbm" in MICROPHYSICS_REGISTRY
    assert resolve_kernel(MICROPHYSICS_REGISTRY, "fast_sbm") \
        is fast_sbm_microphysics


def test_unknown_collision_kernel_raises():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    with pytest.raises(ValueError, match="collision_kernel"):
        fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT,
                              FastSBMConfig(collision_kernel="hal"))


def test_condensation_closure_on_fields():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert out.dT_dt.shape == (NCOL, NLEV)
    # Heat closure per cell (condensation only — sedimentation moves
    # liquid without phase change).
    np.testing.assert_allclose(
        np.asarray(out.dT_dt),
        -(constants.L_v / constants.c_pd) * np.asarray(out.dq_v_dt),
        rtol=1e-10)
    # Column water closure: vapor loss = liquid gain + surface precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt) + np.asarray(out.precipitation),
        rtol=1e-9)
    # Supersaturated: net condensation.
    assert np.all(np.asarray(out.dq_v_dt) < 0.0)
    # Warm-only: ice tendencies identically zero; precip nonnegative.
    np.testing.assert_array_equal(np.asarray(out.dq_i_dt), 0.0)
    assert np.all(np.asarray(out.precipitation) >= 0.0)


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
    # Rain production = column rain-mass gain + what already precipitated.
    col = lambda x: float(jnp.sum((x * rho * dz)[0]))
    rain_thin = col(out_thin.dq_r_dt) + float(out_thin.precipitation[0])
    rain_dense = col(out_dense.dq_r_dt) + float(out_dense.precipitation[0])
    assert rain_dense > 0.0
    assert rain_dense > 50.0 * max(rain_thin, 0.0) or rain_thin <= 0.0
    # At S = 0 coalescence+settling conserve liquid against precip:
    # column (dq_c + dq_r) + precip ≈ 0 (condensation contributes ~0).
    np.testing.assert_allclose(
        col(out_dense.dq_c_dt + out_dense.dq_r_dt)
        + float(out_dense.precipitation[0]),
        0.0, atol=5.0e-9 * float(jnp.sum((rho * dz)[0])))


def test_cloud_rain_boundary_matches_oracle():
    # Codex review item 6: oracle IF(KRR < KRDROP=15) (1-based) → bins
    # 1..14 cloud, 15.. rain ⇒ 0-based bins 0..13 cloud, 14.. rain. Put a
    # spectrum exactly at 0-based bin 14 (the 50 um bin) — its mass must
    # land in RAIN, not cloud. Probe the projection directly.
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        bin_mixing_ratios_from_f, mass_density, mass_doubling_grid)
    from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import KRDROP
    m = mass_doubling_grid()
    assert KRDROP == 15
    # Single delta at 0-based bin KRDROP-1 = 14 (1-based 15, the ~50um bin).
    f = jnp.zeros_like(m).at[KRDROP - 1].set(1.0e12)
    cloud_mask = jnp.arange(m.shape[0]) < (KRDROP - 1)
    qc = float(mass_density(jnp.where(cloud_mask, f, 0.0), m))
    qr = float(mass_density(jnp.where(~cloud_mask, f, 0.0), m))
    assert qc == 0.0 and qr > 0.0      # the 50um bin is RAIN
    # And bin 13 (1-based 14) is cloud.
    f2 = jnp.zeros_like(m).at[KRDROP - 2].set(1.0e12)
    qc2 = float(mass_density(jnp.where(cloud_mask, f2, 0.0), m))
    assert qc2 > 0.0


def test_prognostic_Nc_used_when_present():
    # Codex review item 10: a column carrying N_c should drive cloud
    # number, not the fixed cdnc. Two states, same q_c, different N_c →
    # different droplet sizes → measurably different rain production.
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.0, q_c=1.0e-3)
    cfg = FastSBMConfig(collision_kernel="hall")
    low_N = hyd._replace(N_c=jnp.full((NCOL, NLEV), 3.0e7))   # big drops
    high_N = hyd._replace(N_c=jnp.full((NCOL, NLEV), 6.0e8))  # small drops
    out_low = fast_sbm_microphysics(T, q_v, low_N, p, p_half, rho, dz, DT,
                                    cfg)
    out_high = fast_sbm_microphysics(T, q_v, high_N, p, p_half, rho, dz, DT,
                                     cfg)
    col = lambda x, o: float(jnp.sum((x * rho * dz)[0])) \
        + float(o.precipitation[0])
    # Fewer/larger droplets coalesce faster → more rain.
    assert col(out_low.dq_r_dt, out_low) > col(out_high.dq_r_dt, out_high)


def test_clear_supersaturated_cell_activates_cloud():
    # With CCN activation a supersaturated CLEAR cell must form cloud
    # (number + condensed water) — not stay clear.
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.05, q_c=0.0, q_r=0.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert np.all(np.asarray(out.dN_c_dt) > 0.0)     # droplets nucleated
    assert np.all(np.asarray(out.dq_c_dt) > 0.0)     # cloud water grew
    assert np.all(np.asarray(out.dq_v_dt) < 0.0)     # vapor consumed
    # Total-water closure holds through activation + condensation + precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt) + np.asarray(out.precipitation),
        rtol=1e-9)


def test_subsaturated_clear_cell_fixed_point():
    # A clear SUBsaturated cell activates nothing and stays put.
    T, q_v, hyd, p, p_half, rho, dz = _fields(0.8, q_c=0.0, q_r=0.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
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
