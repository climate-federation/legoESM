"""Unit tests for the switchable SDM column microphysics operator.

Covers the condensation/evaporation tendencies, total-water + positivity
guarantees, and the factory/registry wiring that makes ``scheme="sdm"``
selectable.

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_column.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import relative_humidity, saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    make_zero_hydrometeors,
)
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.sdm import SDMConfig, sdm_microphysics
from legoesm.atmosphere.physics.microphysics.integration import (
    _get_microphysics_fn,
    min_tracer_slots,
)


def _q_v_for_S(S, T, p):
    """Vapor mixing ratio giving saturation ratio S at (T, p)."""
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    e = S * e_s
    return constants.epsilon * e / (p - e)


def _columns(S_list, q_c_list, T=283.0, p=9.0e4):
    n = len(S_list)
    q_v = jnp.asarray([_q_v_for_S(S, T, p) for S in S_list]).reshape(n, 1)
    Tarr = jnp.full((n, 1), T)
    p_full = jnp.full((n, 1), p)
    p_half = jnp.full((n, 2), p)
    rho = jnp.full((n, 1), p / (constants.R_d * T))
    dz = jnp.full((n, 1), 100.0)
    hyd = make_zero_hydrometeors(n, 1)._replace(
        q_c=jnp.asarray(q_c_list).reshape(n, 1))
    return Tarr, q_v, hyd, p_full, p_half, rho, dz


def test_clear_and_supersaturated_and_evaporating_cells():
    # col0: cloudy + supersaturated -> condensation; col1: cloudy + subsaturated
    # -> evaporation; col2: clear + subsaturated -> no change.
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[1.02, 0.95, 0.90], q_c_list=[1.0e-4, 1.0e-4, 0.0])
    out = sdm_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, dt=1.0)
    assert isinstance(out, MicrophysicsOutput)
    dq_c = np.asarray(out.dq_c_dt).ravel()
    dq_v = np.asarray(out.dq_v_dt).ravel()
    dT = np.asarray(out.dT_dt).ravel()
    assert dq_c[0] > 0.0          # supersaturated cloudy -> grows
    assert dq_c[1] < 0.0          # subsaturated cloudy   -> evaporates
    assert dq_c[2] == 0.0         # clear                 -> nothing
    # total water conserved per cell, latent heating consistent
    assert np.allclose(dq_v, -dq_c, rtol=1e-12)
    assert np.allclose(dT, constants.L_v / constants.c_pd * dq_c, rtol=1e-12)
    assert dT[0] > 0.0 and dT[1] < 0.0
    # ice/rain/number tendencies are zero (condensation-only adapter)
    for fld in (out.dq_r_dt, out.dq_i_dt, out.dq_s_dt, out.dq_g_dt,
                out.dN_c_dt, out.dN_r_dt, out.dN_i_dt):
        assert np.all(np.asarray(fld) == 0.0)
    assert np.all(np.asarray(out.precipitation) == 0.0)


def test_no_growth_roundtrip_at_saturation():
    """At S=1 with no curvature the droplet does not grow, so the
    q_c -> R -> q_c reconstruction/inverse must round-trip exactly (dq_c=0).
    A broken reconstruction or inverse would leak a spurious source/sink that
    the sign/conservation tests cannot see. Includes ultra-thin cells exercising
    the r_min_reconstruct N_eff closure branch (no cloudy/clear gate — the
    closure is exactly continuous in q_c)."""
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[1.0, 1.0, 1.0],
        q_c_list=[1.0e-4, 2.0e-12, 5.0e-13])  # normal, just-cloudy, clear
    out = sdm_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, 1.0, cfg)
    dq_c = np.asarray(out.dq_c_dt).ravel()
    assert np.allclose(dq_c, 0.0, atol=1e-18)
    assert np.allclose(np.asarray(out.dq_v_dt).ravel(), -dq_c, atol=1e-18)
    assert np.allclose(np.asarray(out.dT_dt).ravel(), 0.0, atol=1e-15)


def test_donor_clamps_keep_water_nonnegative():
    """Extreme supersaturation cannot condense more than the available vapor;
    extreme subsaturation cannot evaporate more than the available cloud."""
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[5.0, 0.01], q_c_list=[1.0e-3, 1.0e-3])
    dt = 100.0
    out = sdm_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, dt=dt)
    dq_c = np.asarray(out.dq_c_dt).ravel()
    q_v_arr = np.asarray(q_v).ravel()
    q_c_arr = np.asarray(hyd.q_c).ravel()
    # condensation bounded by vapor; evaporation bounded by cloud
    assert dq_c[0] <= q_v_arr[0] / dt + 1e-20
    assert dq_c[1] >= -q_c_arr[1] / dt - 1e-20
    # q_v and q_c stay >= 0 after one explicit step
    assert q_v_arr[0] + np.asarray(out.dq_v_dt).ravel()[0] * dt >= -1e-15
    assert q_c_arr[1] + dq_c[1] * dt >= -1e-15


def test_reconstructed_box_coalescence_produces_rain_and_conserves_liquid():
    """Opt-in column_do_coalescence is a per-step reconstructed box-SDM path:
    a broad low-N cloud near the rain split should transfer cloud mass to rain
    while conserving total liquid in a saturated no-condensation cell."""
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[1.0], q_c_list=[2.0e-3])
    hyd = hyd._replace(N_c=jnp.full((1, 1), 1.0e7))
    cfg = SDMConfig(
        column_do_coalescence=True,
        column_n_sd=64,
        column_seed=3,
        include_curvature=False,
        include_solute=False,
        collision_kernel="golovin",
        golovin_b=1.0e5,
    )
    out = sdm_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, 10.0, cfg)
    dqc = float(out.dq_c_dt[0, 0])
    dqr = float(out.dq_r_dt[0, 0])
    dqv = float(out.dq_v_dt[0, 0])
    assert dqr > 0.0
    assert dqc < 0.0
    assert float(out.dN_r_dt[0, 0]) > 0.0
    assert abs(dqc + dqr) < 1.0e-14
    assert abs(dqv) < 1.0e-14
    assert abs(float(out.dT_dt[0, 0])) < 1.0e-10


def test_box_coalescence_activation_forms_cloud_and_clamps_positive():
    """codex-flagged fixes: (4) activation — a SUPERSATURATED CLEAR cell
    (q_c=0) must nucleate + grow cloud (dq_c>0) instead of staying clear;
    (1) condensation never drives q_v negative; (2) number tendencies never
    drive N_r negative. Water conserved."""
    cfg = SDMConfig(column_do_coalescence=True, column_n_sd=32, column_seed=0,
                    include_curvature=False, include_solute=False)
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[1.05], q_c_list=[0.0])              # clear + supersaturated
    out = sdm_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, 30.0, cfg)
    dqc = float(out.dq_c_dt[0, 0]); dqv = float(out.dq_v_dt[0, 0]); dt = 30.0
    assert dqc > 0.0                                   # (4) activation → cloud
    assert float(q_v[0, 0]) + dqv * dt >= -1e-15       # (1) q_v stays ≥ 0
    assert float(hyd.N_r[0, 0]) + float(out.dN_r_dt[0, 0]) * dt >= -1e-12  # (2)
    assert abs(dqv + dqc + float(out.dq_r_dt[0, 0])) < 1e-13   # water conserved


def test_sdm_slot_contract_is_config_aware():
    assert min_tracer_slots("sdm") == 2
    assert min_tracer_slots(
        "sdm", SDMConfig(column_do_coalescence=True)) == 8


def test_column_jit_and_grad():
    cfg = SDMConfig()
    T, q_v, hyd, p_full, p_half, rho, dz = _columns(
        S_list=[1.03], q_c_list=[2.0e-4])

    @jax.jit
    def total_heating(qv):
        out = sdm_microphysics(T, qv, hyd, p_full, p_half, rho, dz, 1.0, cfg)
        return jnp.sum(out.dT_dt)

    assert jnp.isfinite(total_heating(q_v))
    g = jax.grad(total_heating)(q_v)
    assert jnp.all(jnp.isfinite(g))


def test_factory_dispatch_selects_sdm():
    name, fn, scheme_cfg = _get_microphysics_fn(MicrophysicsConfig(scheme="sdm"))
    assert name == "sdm"
    assert fn is sdm_microphysics
    assert isinstance(scheme_cfg, SDMConfig)


def test_unknown_scheme_still_raises():
    with pytest.raises(ValueError, match="Unknown microphysics scheme"):
        _get_microphysics_fn(MicrophysicsConfig(scheme="not_a_scheme"))
