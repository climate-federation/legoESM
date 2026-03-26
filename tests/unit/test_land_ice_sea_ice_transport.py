"""Category 10: Sea Ice Transport & ITD (unit-level tests).

Tests the state conversion roundtrips and ITD-related functions
that do not require a real cubed-sphere grid.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ice.state import (
    SeaIceState, DynamicSeaIceState, init_dynamic_ice_state,
    dynamic_to_slab, slab_to_dynamic,
)
from legoesm.core.field import Field


SHAPE = (6, 4, 4)
DIMS = ("face", "x", "y")


def _field(val, name="", units=""):
    return Field(data=jnp.full(SHAPE, val, jnp.float64), name=name, dims=DIMS, units=units)


class Test10j_StateRoundtrip:
    def test_slab_to_dynamic_to_slab(self):
        slab = SeaIceState(
            h_ice=_field(1.5, "h_ice", "m"),
            T_ice=_field(262.0, "T_ice", "K"),
            concentration=_field(0.85, "concentration", "1"),
        )
        dyn = slab_to_dynamic(slab)
        assert isinstance(dyn, DynamicSeaIceState)
        slab2 = dynamic_to_slab(dyn)
        assert jnp.allclose(slab2.h_ice.data, slab.h_ice.data)
        assert jnp.allclose(slab2.T_ice.data, slab.T_ice.data)
        assert jnp.allclose(slab2.concentration.data, slab.concentration.data)

    def test_init_dynamic_state(self):
        state = init_dynamic_ice_state(SHAPE)
        assert state.h_ice.data.shape == SHAPE
        assert jnp.allclose(state.h_ice.data, 0.0)
        assert jnp.allclose(state.u_ice.data, 0.0)
        assert jnp.allclose(state.sigma_11.data, 0.0)
