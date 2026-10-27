"""Canaries for the parts of the Sundqvist microphysics with no oracle test.

The precipitation stage is pinned against the running Zhao-Carr oracle in
``tests/atmosphere/microphysics/unit/test_sundqvist_zhao_carr_oracle.py``.  This
file pins the NOT-YET-FAITHFUL condensation surrogate and the shipped defaults,
so a silent change to either trips it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    diagnose_sundqvist_process_rates,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Numeric faithfulness checks in float64 (restored after)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _hydro(q_c, q_r=None):
    """HydrometeorState with only q_c (and optional q_r); everything else zero."""
    q_c = jnp.asarray(q_c, dtype=jnp.float64)
    z = jnp.zeros_like(q_c)
    q_r = z if q_r is None else jnp.asarray(q_r, dtype=jnp.float64)
    return HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z,
    )


def _column(T_K, q_c, rh, p_pa=8.0e4, rho_val=1.0, dz_val=500.0, dt=100.0):
    """Build (T, q_v, hydro, p_full, p_half, rho, dz, dt) for target RH.

    RH is set via q_v = rh * q_sat(T, p). Shapes (1, nlev).
    """
    T = jnp.asarray(T_K, dtype=jnp.float64)[None, :]
    q_c = jnp.asarray(q_c, dtype=jnp.float64)[None, :]
    p_full = jnp.full_like(T, p_pa)
    q_sat = saturation_specific_humidity(T, p_full)
    q_v = jnp.asarray(rh, dtype=jnp.float64)[None, :] * q_sat
    rho = jnp.full_like(T, rho_val)
    dz = jnp.full_like(T, dz_val)
    nlev = T.shape[1]
    p_half = jnp.linspace(p_pa - 1.0e4, p_pa + 1.0e4, nlev + 1)[None, :]
    return T, q_v, _hydro(q_c[0]), p_full, p_half, rho, dz, dt


def _rates(cfg, T_K, q_c, rh, **kw):
    T, q_v, hydro, p_full, p_half, rho, dz, dt = _column(T_K, q_c, rh, **kw)
    return diagnose_sundqvist_process_rates(
        T=T, q_v=q_v, hydrometeors=hydro, p_full=p_full, p_half=p_half,
        rho=rho, dz=dz, dt=dt, config=cfg,
    )


# ===========================================================================
# The precipitation stage (SBK89 release, ice route, evaporation, melting) is a
# port of the Zhao-Carr oracle and is pinned against the RUNNING oracle in
# tests/atmosphere/microphysics/unit/test_sundqvist_zhao_carr_oracle.py.
# What stays here are the NOT-YET-FAITHFUL condensation surrogate and the
# shipped defaults.
# ===========================================================================
def test_departure_condensation_removes_supersaturation_toward_qsat():
    """SURROGATE condensation FORM: sigmoid(k*(RH-RH_crit))*max(q_v-q_sat,0)/dt.

    This pins the scheme's SIMPLIFIED condensation form (a departure from SBK89's
    full condensation closure), NOT the saturation curve: q_sat is taken from the
    model's own ``saturation_specific_humidity`` in both the fixture and the expected
    value, so this deliberately does not certify the saturation routine (that is
    ``thermo``'s oracle, tested separately) — only that the scheme applies the
    sigmoid RH gate to the supersaturation ``q_v-q_sat``.

    At RH>1 (supersaturated) it removes q_v-q_sat (drives toward q_sat); at
    RH_crit<RH<1 (unsaturated) it is ZERO -- the documented fix (does NOT strip
    vapor down to RH_crit*q_sat).
    """
    cfg = SundqvistConfig()
    dt = 100.0
    T = np.array([290.0])
    p = 8.0e4
    q_sat = float(saturation_specific_humidity(jnp.array([[290.0]]), jnp.full((1, 1), p))[0, 0])

    # Supersaturated RH=1.2 -> remove q_v - q_sat with a near-1 sigmoid gate.
    r_sup = _rates(cfg, T, np.array([1e-4]), rh=np.array([1.2]), p_pa=p, dt=dt)
    rh = 1.2
    gate = 1.0 / (1.0 + np.exp(-cfg.sigmoid_sharpness * (rh - cfg.rh_crit)))
    expected = gate * (rh * q_sat - q_sat) / dt
    assert np.allclose(float(np.asarray(r_sup.condensation)[0, 0]), expected, rtol=1e-9, atol=1e-16)

    # Unsaturated (RH_crit<RH=0.9<1): no supersaturation -> condensation == 0.
    r_unsat = _rates(cfg, T, np.array([1e-4]), rh=np.array([0.9]), p_pa=p, dt=dt)
    assert float(np.asarray(r_unsat.condensation)[0, 0]) == 0.0


def test_departure_default_constants():
    """Shipped ``SundqvistConfig`` defaults (canary): a silent change trips this.

    The oracle hard-codes qc_crit 3e-4, c1 300 and c2 0.5; the shipped
    qc_crit / c2 / evap_coeff defaults differ (pending user decision), and the
    Gaussian peak/width fields have no oracle counterpart (non-default raises).
    """
    c = SundqvistConfig()
    assert c.rh_crit == 0.8
    assert c.auto_rate == 1e-3         # c_0 [1/s]
    assert c.qc_crit == 5e-4           # q_c,crit [kg/kg]
    assert c.coalescence_enh_coeff == 300.0   # c1 (SBK89 F1)
    assert c.bergeron_enh_coeff == 3.0        # c2 (F2 Gaussian amplitude)
    assert c.bergeron_T_peak_K == constants.T_freeze - 15.0
    assert c.bergeron_T_width_K == 7.0
    assert c.evap_coeff == 5e-4
    assert c.psautco == 0.0
    assert c.wminco_liquid == 0.0
    assert c.wminco_ice == 0.0
