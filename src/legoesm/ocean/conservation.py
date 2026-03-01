"""Conservation fixers for the ocean model.

Volume (free surface), heat (T), and salt (S) conservation via
uniform additive corrections, matching the atmosphere pattern in
core/conservation.py.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators import _is_distributed
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import OceanState, OceanConfig


def _ocean_global_sum(local_value):
    """MPI-aware global sum for scalar or vector reductions."""
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_value)
    return local_value


def _ocean_area_sum(field_2d, mask, grid):
    """Area-weighted sum over ocean cells only, MPI-aware."""
    local_sum = jnp.sum(field_2d * mask * grid.area)
    return _ocean_global_sum(local_sum)


def _ocean_volume_sum(field_3d, h_k, mask, grid):
    """Volume-weighted sum over ocean cells, MPI-aware."""
    integrand = jnp.sum(field_3d * h_k, axis=-1)  # depth-integrated (6,n,n)
    return _ocean_area_sum(integrand, mask, grid)


def fix_volume_ocean(
    state_new: OceanState,
    state_old: OceanState,
    grid: CubedSphereGrid,
) -> OceanState:
    """Fix volume conservation via uniform eta correction.

    Ensures global integral of eta * area is preserved.
    """
    mask = state_old.land_mask.data
    weighted_area = mask * grid.area
    local_terms = jnp.stack([
        jnp.sum(state_old.eta.data * weighted_area),
        jnp.sum(state_new.eta.data * weighted_area),
        jnp.sum(weighted_area),
    ])
    vol_old, vol_new, ocean_area = _ocean_global_sum(local_terms)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_fixed = state_new.eta.replace(
        data=state_new.eta.data + correction * mask,
    )
    return state_new._replace(eta=eta_fixed)


def fix_heat_ocean(
    state_new: OceanState,
    state_old: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
) -> OceanState:
    """Fix heat conservation via uniform T correction.

    Ensures global integral of T * h_k * area is preserved.
    """
    mask = state_old.land_mask.data

    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, state_new.H_bathy.data, z_coord,
    )

    weighted_area = mask * grid.area
    local_terms = jnp.stack([
        jnp.sum(jnp.sum(state_old.T.data * h_k_old, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(state_new.T.data * h_k_new, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area),
    ])
    heat_old, heat_new, ocean_volume = _ocean_global_sum(local_terms)
    correction = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)

    T_fixed = state_new.T.replace(
        data=state_new.T.data + correction * mask[..., jnp.newaxis],
    )
    return state_new._replace(T=T_fixed)


def fix_salt_ocean(
    state_new: OceanState,
    state_old: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
) -> OceanState:
    """Fix salt conservation via uniform S correction.

    Ensures global integral of S * h_k * area is preserved.
    """
    mask = state_old.land_mask.data

    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, state_new.H_bathy.data, z_coord,
    )

    weighted_area = mask * grid.area
    local_terms = jnp.stack([
        jnp.sum(jnp.sum(state_old.S.data * h_k_old, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(state_new.S.data * h_k_new, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area),
    ])
    salt_old, salt_new, ocean_volume = _ocean_global_sum(local_terms)
    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)

    S_fixed = state_new.S.replace(
        data=state_new.S.data + correction * mask[..., jnp.newaxis],
    )
    return state_new._replace(S=S_fixed)


def ocean_conservation_fixer(
    state_new: OceanState,
    state_old: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
) -> OceanState:
    """Apply all ocean conservation fixers in sequence.

    Order: volume first, then heat, then salt.
    """
    if config.fix_volume:
        state_new = fix_volume_ocean(state_new, state_old, grid)
    if config.fix_heat:
        state_new = fix_heat_ocean(state_new, state_old, grid, z_coord)
    if config.fix_salt:
        state_new = fix_salt_ocean(state_new, state_old, grid, z_coord)
    return state_new
