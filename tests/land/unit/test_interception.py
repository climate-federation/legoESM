"""Canonical canopy-water interception — conservation + bounds + differentiability.

Locks the shared formulation (``land/canopy/interception.py``) that both the
two-leaf and CLM-ML canopies use: water closes to machine precision, the store
is capped and non-negative, the wetted fraction is bounded, and the whole thing
is jax-differentiable (needed for per-site parameter training).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.land.canopy.interception import (
    InterceptionConfig,
    max_canopy_water,
    update_canopy_water,
    water_balance_residual,
    wetted_fraction,
)

_CFG = InterceptionConfig()
_DT = 1800.0


def test_water_closes_to_machine_precision():
    """precip = throughfall + dW/dt + wet_evap, over a range of states."""
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.uniform(0.0, 0.4, 20))          # kg m-2
    precip = jnp.asarray(rng.uniform(0.0, 5e-4, 20))    # kg m-2 s-1 (wet)
    le_pot = jnp.asarray(rng.uniform(-50.0, 300.0, 20)) # W m-2 (dew..evap)
    pai = jnp.asarray(rng.uniform(0.5, 6.0, 20))
    W_new, tf, ev, fwet = update_canopy_water(
        W, precip, le_pot, pai, _DT, _CFG, constants.L_v)
    res = water_balance_residual(W, W_new, precip, tf, ev, _DT)
    # keep only columns whose store did not clamp at the zero floor (those
    # legitimately destroy sub-floor round-off); the rest must close exactly.
    unclamped = np.asarray(W_new) > 1e-9
    assert np.max(np.abs(np.asarray(res)[unclamped])) < 1e-12


def test_store_capped_at_maximum_and_nonnegative():
    """Heavy rain saturates the store to h2ocanmx; it never goes negative."""
    pai = jnp.asarray([3.0, 3.0])
    W = jnp.asarray([0.0, 0.29])
    heavy = jnp.asarray([1e-2, 1e-2])         # kg m-2 s-1, floods the canopy
    W_new, tf, ev, fwet = update_canopy_water(
        W, heavy, jnp.zeros(2), pai, _DT, _CFG, constants.L_v)
    cap = np.asarray(max_canopy_water(pai, _CFG))
    assert np.all(np.asarray(W_new) <= cap + 1e-12)
    assert np.all(np.asarray(W_new) >= 0.0)
    assert np.all(np.asarray(tf) > 0.0)       # excess drips through


def test_drip_appears_only_when_saturated():
    """A dry canopy under light rain intercepts all of it (drip==0)."""
    pai = jnp.asarray([4.0])
    W = jnp.asarray([0.0])
    light = jnp.asarray([1e-6])
    W_new, tf, ev, _ = update_canopy_water(
        W, light, jnp.zeros(1), pai, _DT, _CFG, constants.L_v)
    # direct throughfall only (1-fpi); no drip because storage not exceeded
    fpi = _CFG.interception_fraction * np.tanh(4.0)
    assert np.isclose(float(tf[0]), float(light[0]) * (1.0 - fpi), rtol=1e-6)


def test_wetted_fraction_bounded():
    pai = jnp.asarray([0.0, 2.0, 5.0])
    W = jnp.asarray([0.0, 0.2, 10.0])     # last is over-saturated
    fwet = np.asarray(wetted_fraction(W, pai, _CFG))
    assert np.all(fwet >= 0.0)
    assert np.all(fwet <= _CFG.maximum_leaf_wetted_fraction + 1e-12)
    assert fwet[0] == 0.0                 # empty canopy is dry


def test_dew_deposition_grows_store():
    """A downward (negative) potential flux deposits dew, increasing W."""
    pai = jnp.asarray([3.0])
    W = jnp.asarray([0.05])
    dew = jnp.asarray([-100.0])           # W m-2 downward
    W_new, tf, ev, _ = update_canopy_water(
        W, jnp.zeros(1), dew, pai, _DT, _CFG, constants.L_v)
    assert float(W_new[0]) > float(W[0])   # dew added
    assert float(ev[0]) < 0.0             # flux is downward (deposition)


def test_evaporation_cannot_exceed_storage():
    """A large evaporative demand on a nearly-dry canopy is storage-limited."""
    pai = jnp.asarray([3.0])
    W = jnp.asarray([1e-4])
    W_new, tf, ev, _ = update_canopy_water(
        W, jnp.zeros(1), jnp.asarray([1e4]), pai, _DT, _CFG, constants.L_v)
    assert float(W_new[0]) >= 0.0
    assert float(ev[0]) * _DT <= float(W[0]) + 1e-12


def test_differentiable_through_parameters():
    """Gradients w.r.t. the tunable parameters are finite (per-site training)."""
    pai = jnp.asarray([3.0]); W = jnp.asarray([0.1])
    precip = jnp.asarray([1e-4]); le = jnp.asarray([150.0])

    def loss(dewmx):
        cfg = _CFG._replace(dewmx=dewmx)
        W_new, tf, ev, _ = update_canopy_water(
            W, precip, le, pai, _DT, cfg, constants.L_v)
        return jnp.sum(ev)

    g = jax.grad(loss)(0.1)
    assert np.isfinite(float(g))
