"""Smoke tests for warm-rain microphysics helpers (`_warm_rain.py`).

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
        tests/unit/test_warm_rain.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
)


@pytest.fixture
def column_state():
    """Build a small (ncol, nlev) column with realistic warm-cloud values."""
    ncol, nlev = 4, 6
    T = jnp.full((ncol, nlev), 285.0)            # mid-troposphere temperature [K]
    p = jnp.full((ncol, nlev), 8.0e4)            # ~800 hPa
    rho = p / (constants.R_d * T)
    q_v = jnp.full((ncol, nlev), 1.5e-2)         # high vapor (likely supersat)
    q_c = jnp.full((ncol, nlev), 5.0e-4)         # cloud water
    q_r = jnp.full((ncol, nlev), 2.0e-4)         # rain
    N_c = 1.0e8 * jnp.ones((ncol, nlev))         # cloud droplet number [1/kg]
    N_r = 1.0e3 * jnp.ones((ncol, nlev))         # rain number [1/kg]
    return T, p, rho, q_v, q_c, q_r, N_c, N_r


def test_saturation_adjustment(column_state):
    T, p, _, q_v, _, _, _, _ = column_state
    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
    assert cond.shape == T.shape
    assert q_sat.shape == T.shape
    assert jnp.all(jnp.isfinite(cond))
    assert jnp.all(jnp.isfinite(q_sat))
    assert jnp.all(q_sat > 0.0)
    # Supersaturated air → positive condensation
    assert float(jnp.mean(cond)) > 0.0


def test_effective_Nc(column_state):
    *_, N_c, _ = column_state
    Nc_eff = effective_Nc(N_c, Nc_0=5.0e7)
    assert Nc_eff.shape == N_c.shape
    assert jnp.all(jnp.isfinite(Nc_eff))
    assert jnp.all(Nc_eff > 1.0)
    # Where N_c is set (>1) the effective value passes through.
    assert jnp.allclose(Nc_eff, N_c)
    # Where N_c is zero, fallback kicks in.
    fallback = effective_Nc(jnp.zeros_like(N_c), Nc_0=5.0e7)
    assert jnp.allclose(fallback, 5.0e7)


def test_autoconversion_sb(column_state):
    _, _, rho, _, q_c, _, N_c, _ = column_state
    dq_au, dN_au, x_c = autoconversion_sb(
        q_c, N_c, rho, k_au=9.44e9, x_star=2.6e-10,
    )
    assert dq_au.shape == q_c.shape
    assert dN_au.shape == q_c.shape
    assert x_c.shape == q_c.shape
    assert jnp.all(jnp.isfinite(dq_au))
    assert jnp.all(jnp.isfinite(dN_au))
    assert jnp.all(jnp.isfinite(x_c))
    assert jnp.all(dq_au >= 0.0)
    assert jnp.all(x_c > 0.0)


def test_accretion(column_state):
    _, _, rho, _, q_c, q_r, _, _ = column_state
    rate = accretion(q_c, q_r, rho, k_ac=5.25)
    assert rate.shape == q_c.shape
    assert jnp.all(jnp.isfinite(rate))
    assert jnp.all(rate >= 0.0)


def test_self_collection_breakup(column_state):
    _, _, rho, _, _, q_r, _, N_r = column_state
    dN_sc, dN_br = self_collection_breakup(
        N_r, q_r, rho, k_sc=7.12, breakup_sharpness=10.0, D_eq=9.0e-4,
    )
    assert dN_sc.shape == N_r.shape
    assert dN_br.shape == N_r.shape
    assert jnp.all(jnp.isfinite(dN_sc))
    assert jnp.all(jnp.isfinite(dN_br))
    # Self-collection always reduces number; breakup either zero or positive.
    assert jnp.all(dN_sc <= 0.0)
    assert jnp.all(dN_br >= 0.0)


def test_rain_evaporation(column_state):
    T, p, _, _, _, q_r, _, _ = column_state
    # Subsaturated vapor profile to drive evaporation.
    q_v_low = 1.0e-3 * jnp.ones_like(q_r)
    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
    evap = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3)
    assert evap.shape == q_r.shape
    assert jnp.all(jnp.isfinite(evap))
    assert jnp.all(evap >= 0.0)
    # With substantial subsaturation, average evaporation is strictly positive.
    assert float(jnp.mean(evap)) > 0.0


def test_jit_compiles(column_state):
    T, p, rho, q_v, q_c, q_r, N_c, N_r = column_state

    @jax.jit
    def chain(T_, q_v_, q_c_, q_r_, N_c_, N_r_, rho_, p_):
        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)
        Nc_eff = effective_Nc(N_c_, Nc_0=5.0e7)
        dq_au, dN_au, _ = autoconversion_sb(
            q_c_, Nc_eff, rho_, k_au=9.44e9, x_star=2.6e-10,
        )
        acc = accretion(q_c_, q_r_, rho_, k_ac=5.25)
        evap = rain_evaporation(q_v_, q_r_, q_sat, evap_coeff=1.0e-3)
        return cond + dq_au + dN_au + acc + evap

    out = chain(T, q_v, q_c, q_r, N_c, N_r, rho, p)
    assert out.shape == T.shape
    assert jnp.all(jnp.isfinite(out))
