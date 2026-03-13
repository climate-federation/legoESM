"""Conservation fixers for the lat-lon FV ocean model.

Volume (free surface), heat (T), and salt (S) conservation via
uniform additive corrections.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig


def _area_sum(field_2d, mask, grid):
    """Area-weighted sum over ocean cells."""
    return jnp.sum(field_2d * mask * grid.area)


def fix_volume_latlon(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    min_water_column_m: float | None = None,
) -> LatLonOceanState:
    """Fix volume conservation via uniform eta correction."""
    mask = state_old.land_mask.data
    weighted_area = mask * grid.area
    vol_old = jnp.sum(state_old.eta.data * weighted_area)
    vol_new = jnp.sum(state_new.eta.data * weighted_area)
    ocean_area = jnp.sum(weighted_area)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_candidate = state_new.eta.data + correction * mask

    if min_water_column_m is not None:
        eta_floor = (
            jnp.asarray(min_water_column_m, dtype=eta_candidate.dtype)
            - state_new.H_bathy.data
        )
        eta_candidate = jnp.maximum(eta_candidate, eta_floor) * mask

    return state_new._replace(eta=state_new.eta.replace(data=eta_candidate))


def fix_heat_latlon(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float | None = None,
) -> LatLonOceanState:
    """Fix heat conservation via uniform T correction."""
    mask = state_old.land_mask.data
    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )

    weighted_area = mask * grid.area
    heat_old = jnp.sum(jnp.sum(state_old.T.data * h_k_old, axis=-1) * weighted_area)
    heat_new = jnp.sum(jnp.sum(state_new.T.data * h_k_new, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area)

    correction = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)
    T_fixed = state_new.T.data + correction * mask[..., jnp.newaxis]
    return state_new._replace(T=state_new.T.replace(data=T_fixed))


def fix_salt_latlon(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float | None = None,
) -> LatLonOceanState:
    """Fix salt conservation via uniform S correction."""
    mask = state_old.land_mask.data
    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )

    weighted_area = mask * grid.area
    salt_old = jnp.sum(jnp.sum(state_old.S.data * h_k_old, axis=-1) * weighted_area)
    salt_new = jnp.sum(jnp.sum(state_new.S.data * h_k_new, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area)

    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)
    S_fixed = state_new.S.data + correction * mask[..., jnp.newaxis]
    return state_new._replace(S=state_new.S.replace(data=S_fixed))


def latlon_ocean_conservation_fixer(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonOceanConfig,
) -> LatLonOceanState:
    """Apply all conservation fixers in sequence: volume, heat, salt."""
    if config.fix_volume:
        state_new = fix_volume_latlon(
            state_new, state_old, grid,
            min_water_column_m=config.min_water_column_m,
        )
    if config.fix_heat:
        state_new = fix_heat_latlon(
            state_new, state_old, grid, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
    if config.fix_salt:
        state_new = fix_salt_latlon(
            state_new, state_old, grid, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
    return state_new
