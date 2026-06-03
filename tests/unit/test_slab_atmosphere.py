"""Direct unit tests for the 0-D single-layer gray slab atmosphere (#6 rung 1).

Validates the cheapest atmosphere brick against analytic limits, energy
conservation, equilibrium convergence, and differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.slab import (
    SlabAtmosphereConfig,
    SlabAtmosphereModel,
    SlabAtmosphereState,
    slab_equilibrium,
    slab_step,
    slab_tendency,
    toa_imbalance,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)  # f64 for the analytic + grad-check tolerances

_S = constants.S_0 / 4.0  # global-mean TOA insolation [W m⁻²]


def _T_eff(albedo: float, insolation: float = _S) -> float:
    return float((((1.0 - albedo) * insolation) / constants.sigma_sb) ** 0.25)


def test_single_layer_greenhouse_textbook_limit():
    # a = 0 (transparent SW), ε = 1 (opaque LW): the classic result is
    # T_atm = T_eff and T_sfc = 2**(1/4) · T_eff.
    cfg = SlabAtmosphereConfig(emissivity=1.0, sw_atm_absorption=0.0, albedo=0.30)
    eq = slab_equilibrium(_S, cfg)
    T_eff = _T_eff(0.30)
    assert eq.T_atm == pytest.approx(T_eff, rel=1e-12)
    assert eq.T_sfc == pytest.approx(2.0 ** 0.25 * T_eff, rel=1e-12)
    # surface is warmer than the air it radiates to (greenhouse)
    assert float(eq.T_sfc) > float(eq.T_atm)


def test_equilibrium_is_a_steady_state():
    cfg = SlabAtmosphereConfig(emissivity=0.8, sw_atm_absorption=0.1)
    eq = slab_equilibrium(_S, cfg)
    dT_atm_dt, dT_sfc_dt = slab_tendency(eq, _S, cfg)
    assert float(jnp.abs(dT_atm_dt)) < 1e-18
    assert float(jnp.abs(dT_sfc_dt)) < 1e-18
    # TOA balance: absorbed SW == OLR at equilibrium
    assert float(jnp.abs(toa_imbalance(eq, _S, cfg))) < 1e-9


def test_energy_conservation_identity():
    # The two net budgets must sum to the TOA imbalance for ANY state (H cancels),
    # i.e. C_atm·dT_atm/dt + C_sfc·dT_sfc/dt == (1-α)S - OLR.  Use a non-equilibrium
    # state and a non-zero sensible coupling to exercise the H-cancellation.
    cfg = SlabAtmosphereConfig(emissivity=0.7, sw_atm_absorption=0.05, sensible_coeff=5.0)
    state = SlabAtmosphereState(T_atm=jnp.asarray(250.0), T_sfc=jnp.asarray(295.0))
    dT_atm_dt, dT_sfc_dt = slab_tendency(state, _S, cfg)
    net_total = cfg.c_atm * dT_atm_dt + cfg.c_sfc * dT_sfc_dt
    assert float(net_total) == pytest.approx(float(toa_imbalance(state, _S, cfg)), rel=1e-10)


def test_run_converges_to_equilibrium():
    # Equilibrium is heat-capacity-independent (it depends only on ε, α, a, S), so
    # use small equal capacities for a fast relaxation timescale (~days) — the run
    # then reaches the SAME analytic radiative equilibrium within a few thousand
    # hourly steps.  (The default 50 m ocean mixed layer has a ~460-day timescale.)
    cfg = SlabAtmosphereConfig(emissivity=0.8, sw_atm_absorption=0.1,
                               c_atm=1e6, c_sfc=1e6)
    model = SlabAtmosphereModel(cfg, dt=3600.0)
    ic = SlabAtmosphereState(T_atm=jnp.asarray(200.0), T_sfc=jnp.asarray(320.0))
    final, hist = model.run(ic, _S, nsteps=3000, save_every=1000)
    eq = slab_equilibrium(_S, cfg)
    assert float(jnp.abs(final.T_atm - eq.T_atm)) < 1e-3
    assert float(jnp.abs(final.T_sfc - eq.T_sfc)) < 1e-3
    assert len(hist) == 3  # save_every bookkeeping
    # residual TOA imbalance has decayed toward zero
    assert float(jnp.abs(toa_imbalance(final, _S, cfg))) < 1e-2


def test_greenhouse_monotonic_in_emissivity():
    cold = slab_equilibrium(_S, SlabAtmosphereConfig(emissivity=0.2))
    warm = slab_equilibrium(_S, SlabAtmosphereConfig(emissivity=0.95))
    assert float(warm.T_sfc) > float(cold.T_sfc)  # stronger greenhouse → warmer surface


def test_analytic_equilibrium_differentiable_vs_fd():
    # d(T_sfc_eq)/dε via reverse-mode AD must match a central finite difference.
    def T_sfc_eq(eps):
        cfg = SlabAtmosphereConfig(emissivity=eps, sw_atm_absorption=0.1)
        return slab_equilibrium(_S, cfg).T_sfc

    eps0 = 0.8
    g_ad = float(jax.grad(T_sfc_eq)(eps0))
    h = 1e-6
    g_fd = (float(T_sfc_eq(eps0 + h)) - float(T_sfc_eq(eps0 - h))) / (2 * h)
    assert g_ad == pytest.approx(g_fd, rel=1e-6)
    assert g_ad > 0.0  # more emissivity → warmer surface
    # The validation must not break the JIT / grad-of-JIT path when emissivity is
    # itself a tracer (parameter-sensitivity under jit): the domain check skips
    # traced fields, so the analytic equilibrium stays traceable.
    assert float(jax.jit(T_sfc_eq)(eps0)) == pytest.approx(float(T_sfc_eq(eps0)), rel=1e-12)
    assert float(jax.grad(jax.jit(T_sfc_eq))(eps0)) == pytest.approx(g_ad, rel=1e-9)


def test_stepped_run_is_differentiable():
    # grad of the final surface temperature wrt insolation through the time loop
    cfg = SlabAtmosphereConfig(emissivity=0.8)

    def final_T_sfc(insolation):
        state = SlabAtmosphereState(T_atm=jnp.asarray(250.0), T_sfc=jnp.asarray(288.0))
        for _ in range(20):
            state = slab_step(state, 3600.0, insolation, cfg)
        return state.T_sfc

    g = float(jax.grad(final_T_sfc)(_S))
    assert jnp.isfinite(g) and g > 0.0  # more sun → warmer surface


def test_zero_emissivity_with_sw_absorption_is_rejected():
    # ε=0 (transparent LW) with a>0 (air absorbs SW) has no radiative equilibrium
    # — the a/ε term is singular; slab_equilibrium must reject it, not diverge.
    cfg = SlabAtmosphereConfig(emissivity=0.0, sw_atm_absorption=0.2)
    with pytest.raises(ValueError, match="emissivity must be in"):
        slab_equilibrium(_S, cfg)
    with pytest.raises(ValueError, match="emissivity must be in"):
        SlabAtmosphereModel(cfg)


def test_config_domain_validation():
    with pytest.raises(ValueError, match="sw_atm_absorption"):
        slab_equilibrium(_S, SlabAtmosphereConfig(sw_atm_absorption=1.5))
    with pytest.raises(ValueError, match="albedo"):
        slab_equilibrium(_S, SlabAtmosphereConfig(albedo=1.0))
    with pytest.raises(ValueError, match="c_sfc must be > 0"):
        slab_equilibrium(_S, SlabAtmosphereConfig(c_sfc=0.0))


def test_vectorized_over_columns():
    cfg = SlabAtmosphereConfig(emissivity=0.8)
    insol = jnp.asarray([_S, _S * 0.5, _S * 1.2])  # 3 columns
    eq = slab_equilibrium(insol, cfg)
    assert eq.T_sfc.shape == (3,)
    # warmer where more insolation
    assert float(eq.T_sfc[2]) > float(eq.T_sfc[0]) > float(eq.T_sfc[1])
