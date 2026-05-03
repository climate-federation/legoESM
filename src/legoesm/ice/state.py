"""Sea ice model state containers.

Two state types:
- ``SeaIceState``: Slab thermodynamics only (3 fields, backward compatible).
- ``DynamicSeaIceState``: Full dynamic ice with velocity, stress tensor,
  and optional multi-category fields.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.ice.itd import aggregate_state


class SeaIceState(NamedTuple):
    """Thermodynamic slab sea ice state.

    All fields have shape (6, n, n).
    """
    h_ice: Field               # Ice thickness [m]
    T_ice: Field               # Ice surface temperature [K]
    concentration: Field       # Ice areal fraction [0-1]


class DynamicSeaIceState(NamedTuple):
    """Full dynamic sea ice state with velocity and stress tensor.

    Thickness/temperature/concentration may have shape (6, n, n) for
    single-category or (6, n, n, n_cat) for multi-category ice.
    Velocity and stress are always (6, n, n).
    """
    h_ice: Field               # Ice thickness [m]
    T_ice: Field               # Ice surface temperature [K]
    concentration: Field       # Ice areal fraction [0-1]
    u_ice: Field               # Ice velocity x-component [m/s]
    v_ice: Field               # Ice velocity y-component [m/s]
    sigma_11: Field            # Stress tensor component [N/m]
    sigma_22: Field            # Stress tensor component [N/m]
    sigma_12: Field            # Stress tensor component [N/m]


def init_dynamic_ice_state(shape: tuple[int, ...]) -> DynamicSeaIceState:
    """Create a zero-initialized DynamicSeaIceState.

    Parameters
    ----------
    shape : tuple
        Spatial shape, typically (6, n, n).
    """
    dims = ("face", "x", "y")
    return DynamicSeaIceState(
        h_ice=Field(data=jnp.zeros(shape), name="h_ice", dims=dims, units="m"),
        T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=dims, units="K"),
        concentration=Field(data=jnp.zeros(shape), name="ice_concentration", dims=dims, units="1"),
        u_ice=Field(data=jnp.zeros(shape), name="u_ice", dims=dims, units="m/s"),
        v_ice=Field(data=jnp.zeros(shape), name="v_ice", dims=dims, units="m/s"),
        sigma_11=Field(data=jnp.zeros(shape), name="sigma_11", dims=dims, units="N/m"),
        sigma_22=Field(data=jnp.zeros(shape), name="sigma_22", dims=dims, units="N/m"),
        sigma_12=Field(data=jnp.zeros(shape), name="sigma_12", dims=dims, units="N/m"),
    )


def dynamic_to_slab(state: DynamicSeaIceState) -> SeaIceState:
    """Convert DynamicSeaIceState to SeaIceState (drops dynamics fields).

    If multi-category, aggregates to single category first.
    """
    h = state.h_ice.data
    T = state.T_ice.data
    a = state.concentration.data

    # Multi-category: aggregate
    if h.ndim > 3:
        h_agg, T_agg, a_agg = aggregate_state(h, T, a)
        return SeaIceState(
            h_ice=state.h_ice.replace(data=h_agg),
            T_ice=state.T_ice.replace(data=T_agg),
            concentration=state.concentration.replace(data=a_agg),
        )

    return SeaIceState(
        h_ice=state.h_ice,
        T_ice=state.T_ice,
        concentration=state.concentration,
    )


def slab_to_dynamic(state: SeaIceState, shape: tuple[int, ...] | None = None) -> DynamicSeaIceState:
    """Convert SeaIceState to DynamicSeaIceState (adds zero dynamics fields)."""
    s = shape or state.h_ice.data.shape
    dims = state.h_ice.dims
    return DynamicSeaIceState(
        h_ice=state.h_ice,
        T_ice=state.T_ice,
        concentration=state.concentration,
        u_ice=Field(data=jnp.zeros(s), name="u_ice", dims=dims, units="m/s"),
        v_ice=Field(data=jnp.zeros(s), name="v_ice", dims=dims, units="m/s"),
        sigma_11=Field(data=jnp.zeros(s), name="sigma_11", dims=dims, units="N/m"),
        sigma_22=Field(data=jnp.zeros(s), name="sigma_22", dims=dims, units="N/m"),
        sigma_12=Field(data=jnp.zeros(s), name="sigma_12", dims=dims, units="N/m"),
    )
