"""Wet/dry-leaf energy balance — a wet leaf evaporates more (interception loss).

The leaf energy balance blends the DRY (stomatal) latent conductance with the
WET one (boundary-layer only, no stomata): a wetted fraction fwet>0 raises LE
above the pure-transpiration value.  ``fwet=0`` must recover the dry balance
EXACTLY (so interception-off runs are unchanged), and fwet must never let LE
exceed the fully-wet (potential) bound.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.land.canopy.energy_balance import (
    leaf_energy_balance_bt,
    leaf_energy_balance_pm,
)

# A daytime broadleaf leaf: absorbed SW/LW, warm humid canopy air, open stomata.
_ARGS = dict(
    An=jnp.array([12.0]), ASW=jnp.array([300.0]), ALW=jnp.array([-40.0]),
    Tf=jnp.array([298.0]), Ps=jnp.array([1.0e5]), Ca=jnp.array([400.0]),
    Tc=jnp.array([296.0]), q_c=jnp.array([0.012]), RH_c=jnp.array([0.6]),
    VPD_c=jnp.array([1200.0]), lam=jnp.array([2.5e6]), Cp=jnp.array([1004.0]),
    rhoa=jnp.array([1.15]), Rb=jnp.array([40.0]), m=jnp.array([9.0]),
    b0=jnp.array([0.01]),
)


def _le_bt(fwet):
    q_f = jnp.array([0.020])
    _, LE, _, _, _, _ = leaf_energy_balance_bt(
        _ARGS["An"], _ARGS["ASW"], _ARGS["ALW"], _ARGS["Tf"], _ARGS["Ps"],
        _ARGS["Ca"], _ARGS["Tc"], q_f, _ARGS["q_c"], _ARGS["RH_c"],
        _ARGS["VPD_c"], _ARGS["lam"], _ARGS["Cp"], _ARGS["rhoa"], _ARGS["Rb"],
        _ARGS["m"], _ARGS["b0"], fwet=fwet)
    return float(LE[0])


def test_fwet_zero_recovers_dry_balance_bt():
    """Interception-off (fwet=0) must be bit-identical to the pure-stomatal LE."""
    _, LE0, _, _, _, _ = leaf_energy_balance_bt(
        _ARGS["An"], _ARGS["ASW"], _ARGS["ALW"], _ARGS["Tf"], _ARGS["Ps"],
        _ARGS["Ca"], _ARGS["Tc"], jnp.array([0.020]), _ARGS["q_c"],
        _ARGS["RH_c"], _ARGS["VPD_c"], _ARGS["lam"], _ARGS["Cp"], _ARGS["rhoa"],
        _ARGS["Rb"], _ARGS["m"], _ARGS["b0"])          # default fwet
    assert np.isclose(float(LE0[0]), _le_bt(jnp.array([0.0])), rtol=0, atol=0)


def test_wet_leaf_evaporates_more_bt():
    """A wetted leaf (fwet>0) evaporates MORE than a dry one (interception loss)."""
    assert _le_bt(jnp.array([0.05])) > _le_bt(jnp.array([0.0]))
    assert _le_bt(jnp.array([0.2])) > _le_bt(jnp.array([0.05]))


def test_fully_wet_is_the_upper_bound_bt():
    """fwet=1 gives the boundary-layer-limited potential LE; partial wetting
    stays between the dry and fully-wet values."""
    dry, half, wet = _le_bt(jnp.array([0.0])), _le_bt(jnp.array([0.5])), _le_bt(jnp.array([1.0]))
    assert dry <= half <= wet + 1e-9


def _le_pm(fwet):
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.land.canopy.energy_balance import canopy_met_variables
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        _ARGS["Ps"], _ARGS["Tc"], _ARGS["q_c"],
        CanopyConfig().rh_cap_smoothing_width)
    _, LE, _, _, _, _ = leaf_energy_balance_pm(
        _ARGS["An"], _ARGS["ASW"], _ARGS["ALW"], _ARGS["Tf"], _ARGS["Ps"],
        _ARGS["Ca"], _ARGS["Tc"], VPD_c, RH_c, desTc, ddesTc, gamma_c,
        _ARGS["Cp"], _ARGS["rhoa"], _ARGS["Rb"], _ARGS["m"], _ARGS["b0"],
        fwet=fwet)
    return float(LE[0])


def test_wet_leaf_evaporates_more_pm():
    """PM also boosts LE with wetness, and fwet=0 uses rc=rs exactly (the
    where-guard), so a dry PM leaf is the legacy balance."""
    dry = _le_pm(jnp.array([0.0]))
    assert _le_pm(jnp.array([0.05])) > dry
    assert _le_pm(jnp.array([0.3])) > _le_pm(jnp.array([0.05]))
