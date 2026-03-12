"""Conservation fixers for MPAS ocean on Voronoi meshes.

Fixes volume, heat, and salt conservation via uniform corrections,
using area-weighted integrals on the Voronoi mesh.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.vertical import compute_layer_thickness


def _ocean_area_sum_mpas(field_2d, mask, mesh):
    """Area-weighted sum over ocean cells.

    Parameters
    ----------
    field_2d : jax.Array, shape (nCells,)
    mask : jax.Array, shape (nCells,)
        Ocean mask (1=ocean).
    mesh : VoronoiMesh

    Returns
    -------
    float
    """
    return jnp.sum(field_2d * mask * mesh.areaCell)


def _ocean_volume_sum_mpas(field_3d, h_k, mask, mesh):
    """Volume-weighted sum over ocean cells and levels.

    Parameters
    ----------
    field_3d : jax.Array, shape (nCells, nlev)
    h_k : jax.Array, shape (nCells, nlev)
    mask : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    float
    """
    return jnp.sum(field_3d * h_k * mask[:, jnp.newaxis] * mesh.areaCell[:, jnp.newaxis])


def fix_volume_mpas(state_new, state_old, mesh, z_coord, min_water_column_m=None):
    """Fix volume conservation via uniform eta correction.

    Ensures global integral of eta * area is preserved.

    Parameters
    ----------
    state_new : MPASOceanState
    state_old : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    min_water_column_m : float or None

    Returns
    -------
    MPASOceanState
    """
    mask = state_old.land_mask.data
    eta_old = state_old.eta.data
    eta_new = state_new.eta.data

    vol_old = _ocean_area_sum_mpas(eta_old, mask, mesh)
    vol_new = _ocean_area_sum_mpas(eta_new, mask, mesh)
    ocean_area = _ocean_area_sum_mpas(jnp.ones_like(mask), mask, mesh)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1e-30)
    eta_fixed = eta_new + correction * mask

    # Ensure minimum water column (ocean cells only)
    if min_water_column_m is not None:
        H_bathy = state_new.H_bathy.data
        eta_floor = min_water_column_m - H_bathy
        eta_fixed = jnp.where(
            mask > 0.5, jnp.maximum(eta_fixed, eta_floor), eta_fixed,
        )

    return state_new._replace(
        eta=state_new.eta.replace(data=eta_fixed),
    )


def fix_heat_mpas(state_new, state_old, mesh, z_coord, min_water_column_m=None):
    """Fix heat conservation via uniform T correction.

    Parameters
    ----------
    state_new, state_old : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    min_water_column_m : float or None

    Returns
    -------
    MPASOceanState
    """
    mask = state_old.land_mask.data
    H_bathy = state_old.H_bathy.data

    h_k_old = compute_layer_thickness(
        state_old.eta.data, H_bathy, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, H_bathy, z_coord,
        min_water_column_m=min_water_column_m,
    )

    heat_old = _ocean_volume_sum_mpas(state_old.T.data, h_k_old, mask, mesh)
    heat_new = _ocean_volume_sum_mpas(state_new.T.data, h_k_new, mask, mesh)
    vol_new = _ocean_volume_sum_mpas(
        jnp.ones_like(state_new.T.data), h_k_new, mask, mesh,
    )

    correction = (heat_old - heat_new) / jnp.maximum(vol_new, 1e-30)
    T_fixed = state_new.T.data + correction * mask[:, jnp.newaxis]

    return state_new._replace(
        T=state_new.T.replace(data=T_fixed),
    )


def fix_salt_mpas(state_new, state_old, mesh, z_coord, min_water_column_m=None):
    """Fix salt conservation via uniform S correction.

    Parameters
    ----------
    state_new, state_old : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    min_water_column_m : float or None

    Returns
    -------
    MPASOceanState
    """
    mask = state_old.land_mask.data
    H_bathy = state_old.H_bathy.data

    h_k_old = compute_layer_thickness(
        state_old.eta.data, H_bathy, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, H_bathy, z_coord,
        min_water_column_m=min_water_column_m,
    )

    salt_old = _ocean_volume_sum_mpas(state_old.S.data, h_k_old, mask, mesh)
    salt_new = _ocean_volume_sum_mpas(state_new.S.data, h_k_new, mask, mesh)
    vol_new = _ocean_volume_sum_mpas(
        jnp.ones_like(state_new.S.data), h_k_new, mask, mesh,
    )

    correction = (salt_old - salt_new) / jnp.maximum(vol_new, 1e-30)
    S_fixed = state_new.S.data + correction * mask[:, jnp.newaxis]

    return state_new._replace(
        S=state_new.S.replace(data=S_fixed),
    )


def mpas_ocean_conservation_fixer(state_new, state_old, mesh, z_coord, config):
    """Apply all conservation fixers in sequence.

    Parameters
    ----------
    state_new, state_old : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig

    Returns
    -------
    MPASOceanState
    """
    min_col = config.min_water_column_m

    if config.fix_volume:
        state_new = fix_volume_mpas(state_new, state_old, mesh, z_coord, min_col)
    if config.fix_heat:
        state_new = fix_heat_mpas(state_new, state_old, mesh, z_coord, min_col)
    if config.fix_salt:
        state_new = fix_salt_mpas(state_new, state_old, mesh, z_coord, min_col)

    return state_new
