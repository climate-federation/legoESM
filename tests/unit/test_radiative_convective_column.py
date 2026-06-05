"""Direct unit tests for the multi-layer gray RCE column (#6 rung 2).

The column composes the gray radiation engine + dry convective adjustment + a
slab-surface energy balance.  The direct equilibrium condition is that the
per-level radiative heating rate and the net surface flux both vanish; the harness
drives the column there.  (The net TOA flux only approaches zero up to the gray
scheme's intrinsic O(1) W/m² flux-closure residual, so it is not the convergence
metric — see the module docstring.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.idealized import radiative_convective_column as rcc
from legoesm.atmosphere.idealized.radiative_convective_column import (
    RadiativeConvectiveColumn,
    RCEColumnConfig,
    rce_surface_net_flux,
)
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_S = constants.S_0 / 4.0                                  # global-mean insolation [W m⁻²]
_GRAY = GrayRadiationConfig(tau_equator=2.0, tau_pole=2.0)  # moderate, uniform optical depth


def _max_heating(col, state):
    rad, _, _ = rcc._radiation(state, _S, col.sigma, col.config)
    return float(jnp.max(jnp.abs(rad.heating_rate)))


def _net_sfc(col, state):
    rad, _, _ = rcc._radiation(state, _S, col.sigma, col.config)
    return float(rce_surface_net_flux(rad, col.config)[0])


def test_step_smoke_shapes_and_finite():
    col = RadiativeConvectiveColumn(RCEColumnConfig(nlev=20, gray=_GRAY), dt=1800.0)
    state = col.initial_state(T0=250.0, T_sfc0=288.0)
    nxt = col.step(state, _S)
    assert nxt.T.shape == (1, 20)
    assert nxt.T_sfc.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(nxt.T))) and bool(jnp.isfinite(nxt.T_sfc[0]))


def _relax(convective: bool, nsteps: int = 3000):
    # Small surface heat capacity → fast (~days) relaxation (equilibrium is
    # heat-capacity-independent), so the column reaches near-equilibrium in a few
    # thousand 30-min steps.  Returns the steadiness of the FIRST vs LAST step
    # (max |ΔT| per step) — the universal equilibrium metric (works for both pure
    # radiative and radiative-convective: at a steady state ΔT/step → 0, whereas
    # the radiative heating rate is non-zero in the convecting region).
    cfg = RCEColumnConfig(nlev=15, c_sfc=1e6, convective_adjustment=convective, gray=_GRAY)
    col = RadiativeConvectiveColumn(cfg, dt=1800.0)
    state = col.initial_state(T0=255.0, T_sfc0=290.0)
    step = jax.jit(col.step)
    s1 = step(state, _S)
    dT_start = float(jnp.max(jnp.abs(s1.T - state.T)))
    state = s1
    for _ in range(nsteps - 1):
        state = step(state, _S)
    dT_end = float(jnp.max(jnp.abs(step(state, _S).T - state.T)))
    return col, state, dT_start, dT_end


def test_radiative_equilibrium_relaxes():
    col, state, dT_start, dT_end = _relax(convective=False)
    assert dT_end < dT_start * 0.1            # column became ≥10× steadier
    assert abs(_net_sfc(col, state)) < 0.5    # surface energy balance closed
    assert _max_heating(col, state) < 1e-5    # near radiative equilibrium (<~0.9 K/day)
    assert bool(jnp.all(jnp.isfinite(state.T)))


def test_convective_adjustment_is_guarded_not_yet_supported():
    # The dry convective adjustment is deferred: the existing dca scheme targets
    # the saturated moist adiabat + moist-CAPE gate, so it is NOT a dry adjustment.
    # The column must REFUSE convective_adjustment=True (fail fast), not silently
    # produce wrong (moist-adiabat-in-a-dry-column) physics.
    with pytest.raises(NotImplementedError, match="dry adjustment"):
        RadiativeConvectiveColumn(RCEColumnConfig(nlev=10, convective_adjustment=True))


def test_column_is_differentiable():
    cfg = RCEColumnConfig(nlev=12, convective_adjustment=False, gray=_GRAY)
    col = RadiativeConvectiveColumn(cfg, dt=1800.0)
    state = col.initial_state(T0=255.0, T_sfc0=290.0)

    def one_step_T_sfc(insolation):
        return col.step(state, insolation).T_sfc[0]

    g = float(jax.grad(one_step_T_sfc)(_S))
    assert jnp.isfinite(g) and g > 0.0   # more sun → warmer surface tendency


def test_harness_input_validation():
    with pytest.raises(ValueError, match="dt must be > 0"):
        RadiativeConvectiveColumn(dt=0.0)
    with pytest.raises(ValueError, match="nlev must be >= 1"):
        RadiativeConvectiveColumn(RCEColumnConfig(nlev=0))
    col = RadiativeConvectiveColumn(RCEColumnConfig(nlev=4))
    with pytest.raises(ValueError, match="nsteps must be >= 1"):
        col.run(col.initial_state(), _S, nsteps=0)
