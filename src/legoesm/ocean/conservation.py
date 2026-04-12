"""Conservation fixers for the ocean model.

Volume (free surface), heat (T), and salt (S) conservation via
uniform additive corrections.  Works with cubed-sphere, lat-lon
C-grid, and any other grid that exposes a 2D ``area`` attribute.

Limitations
-----------
These are *uniform additive* fixers — a single scalar correction is
applied to every ocean cell.  This acts as spurious globally-uniform
diapycnal mixing and violates local conservation.  The proper fix is
flux-form tracer advection with barotropic-baroclinic flux
reconciliation (Hallberg 1997, Higdon 2005).  See issue #59.

The combined fixer (``ocean_conservation_fixer``) computes all
corrections simultaneously from the original pre-fix state, avoiding
order-dependent bias between volume, heat, and salt fixers.  Precision
is upcasted to accumulation dtype before global reductions to prevent
catastrophic cancellation.
"""

from __future__ import annotations

import jax.numpy as jnp

from typing import Union

from legoesm.core.operators import _is_distributed
from legoesm.core.precision import cast
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import (
    OceanState, OceanConfig, LatLonCGridOceanState, LatLonCGridOceanConfig,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.latlon import LatLonGrid

# Types accepted by the conservation fixer (cubed-sphere + lat-lon C-grid)
_OceanStateT = Union[OceanState, LatLonCGridOceanState]
_OceanConfigT = Union[OceanConfig, LatLonCGridOceanConfig]
_GridT = Union[CubedSphereGrid, LatLonGrid]


def _ocean_global_sum(local_value):
    """MPI-aware global sum for scalar or vector reductions."""
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_value)
    return local_value


def _ocean_area_sum(field_2d, mask, grid):
    """Area-weighted sum over ocean cells only, MPI-aware.

    Upcasts inputs to accumulation precision (fp64 in mixed mode) to
    prevent catastrophic cancellation in global reductions.
    """
    _M = "ocean_diagnostics"
    field_acc = cast(field_2d, _M, "accumulate")
    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    local_sum = jnp.sum(field_acc * mask_acc * area_acc)
    return _ocean_global_sum(local_sum)


def _ocean_volume_sum(field_3d, h_k, mask, grid):
    """Volume-weighted sum over ocean cells, MPI-aware.

    Upcasts to accumulation precision before the depth integration.
    """
    _M = "ocean_diagnostics"
    field_acc = cast(field_3d, _M, "accumulate")
    h_k_acc = cast(h_k, _M, "accumulate")
    integrand = jnp.sum(field_acc * h_k_acc, axis=-1)  # depth-integrated (6,n,n)
    return _ocean_area_sum(integrand, mask, grid)


def fix_volume_ocean(
    state_new: _OceanStateT,
    state_old: _OceanStateT,
    grid: _GridT,
    min_water_column_m: float | None = None,
) -> _OceanStateT:
    """Fix volume conservation via uniform eta correction.

    Ensures global integral of eta * area is preserved.
    """
    _M = "ocean_diagnostics"
    mask = state_old.land_mask.data
    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc
    local_terms = jnp.stack([
        jnp.sum(cast(state_old.eta.data, _M, "accumulate") * weighted_area),
        jnp.sum(cast(state_new.eta.data, _M, "accumulate") * weighted_area),
        jnp.sum(weighted_area),
    ])
    vol_old, vol_new, ocean_area = _ocean_global_sum(local_terms)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_candidate = state_new.eta.data + correction * mask
    if min_water_column_m is not None:
        if min_water_column_m <= 0.0:
            raise ValueError(
                f"min_water_column_m must be > 0, got {min_water_column_m!r}",
            )
        eta_floor = (
            jnp.asarray(min_water_column_m, dtype=eta_candidate.dtype)
            - state_new.H_bathy.data
        )
        eta_candidate = jnp.maximum(eta_candidate, eta_floor) * mask
    eta_fixed = state_new.eta.replace(
        data=eta_candidate,
    )
    return state_new._replace(eta=eta_fixed)


def fix_heat_ocean(
    state_new: _OceanStateT,
    state_old: _OceanStateT,
    grid: _GridT,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float | None = None,
) -> _OceanStateT:
    """Fix heat conservation via uniform T correction.

    Ensures global integral of T * h_k * area is preserved.
    """
    _M = "ocean_diagnostics"
    mask = state_old.land_mask.data

    if min_water_column_m is not None and min_water_column_m <= 0.0:
        raise ValueError(
            f"min_water_column_m must be > 0, got {min_water_column_m!r}",
        )

    h_k_old = compute_layer_thickness(
        state_old.eta.data,
        state_old.H_bathy.data,
        z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data,
        state_new.H_bathy.data,
        z_coord,
        min_water_column_m=min_water_column_m,
    )

    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_new_acc = cast(h_k_new, _M, "accumulate")
    local_terms = jnp.stack([
        jnp.sum(jnp.sum(cast(state_old.T.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(cast(state_new.T.data, _M, "accumulate") * h_k_new_acc, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(h_k_new_acc, axis=-1) * weighted_area),
    ])
    heat_old, heat_new, ocean_volume = _ocean_global_sum(local_terms)
    correction = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)

    T_fixed = state_new.T.replace(
        data=state_new.T.data + correction.astype(state_new.T.data.dtype) * mask[..., jnp.newaxis],
    )
    return state_new._replace(T=T_fixed)


def fix_salt_ocean(
    state_new: _OceanStateT,
    state_old: _OceanStateT,
    grid: _GridT,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float | None = None,
) -> _OceanStateT:
    """Fix salt conservation via uniform S correction.

    Ensures global integral of S * h_k * area is preserved.
    """
    _M = "ocean_diagnostics"
    mask = state_old.land_mask.data

    if min_water_column_m is not None and min_water_column_m <= 0.0:
        raise ValueError(
            f"min_water_column_m must be > 0, got {min_water_column_m!r}",
        )

    h_k_old = compute_layer_thickness(
        state_old.eta.data,
        state_old.H_bathy.data,
        z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data,
        state_new.H_bathy.data,
        z_coord,
        min_water_column_m=min_water_column_m,
    )

    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_new_acc = cast(h_k_new, _M, "accumulate")
    local_terms = jnp.stack([
        jnp.sum(jnp.sum(cast(state_old.S.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(cast(state_new.S.data, _M, "accumulate") * h_k_new_acc, axis=-1) * weighted_area),
        jnp.sum(jnp.sum(h_k_new_acc, axis=-1) * weighted_area),
    ])
    salt_old, salt_new, ocean_volume = _ocean_global_sum(local_terms)
    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)

    S_fixed = state_new.S.replace(
        data=state_new.S.data + correction.astype(state_new.S.data.dtype) * mask[..., jnp.newaxis],
    )
    return state_new._replace(S=S_fixed)


def ocean_conservation_fixer(
    state_new: _OceanStateT,
    state_old: _OceanStateT,
    grid: _GridT,
    z_coord: OceanZStarCoordinate,
    config: _OceanConfigT,
) -> _OceanStateT:
    """Apply all ocean conservation fixers simultaneously.

    All corrections are computed from the ORIGINAL state_old's layer
    thicknesses to avoid order-dependent bias between volume, heat,
    and salt fixers.
    """
    mask = state_old.land_mask.data
    min_wc = config.min_water_column_m
    _M = "ocean_diagnostics"  # precision module for accumulations

    # Precompute layer thicknesses from the OLD state (before any fixer).
    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_wc,
    )

    # Upcast shared arrays to accumulation precision for global sums.
    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area_acc = mask_acc * area_acc

    # --- Volume (eta) correction ---
    eta_corrected = state_new.eta.data
    if config.fix_volume:
        vol_terms = jnp.stack([
            jnp.sum(cast(state_old.eta.data, _M, "accumulate") * weighted_area_acc),
            jnp.sum(cast(state_new.eta.data, _M, "accumulate") * weighted_area_acc),
            jnp.sum(weighted_area_acc),
        ])
        vol_old, vol_new, ocean_area = _ocean_global_sum(vol_terms)
        eta_correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
        eta_corrected = state_new.eta.data + eta_correction.astype(eta_corrected.dtype) * mask
        if min_wc is not None:
            eta_floor = jnp.asarray(min_wc, dtype=eta_corrected.dtype) - state_new.H_bathy.data
            eta_corrected = jnp.maximum(eta_corrected, eta_floor) * mask

    # Recompute h_k from the CORRECTED eta so heat/salt integrals use
    # layer thicknesses consistent with the actual post-fix state.
    # Using h_k_old here would conserve against a stale geometry.
    h_k_corrected = compute_layer_thickness(
        eta_corrected, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_wc,
    )
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_fix_acc = cast(h_k_corrected, _M, "accumulate")

    # --- Heat (T) correction ---
    T_corrected = state_new.T.data
    if config.fix_heat:
        heat_terms = jnp.stack([
            jnp.sum(jnp.sum(cast(state_old.T.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area_acc),
            jnp.sum(jnp.sum(cast(state_new.T.data, _M, "accumulate") * h_k_fix_acc, axis=-1) * weighted_area_acc),
            jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc),
        ])
        heat_old, heat_new, ocean_vol = _ocean_global_sum(heat_terms)
        T_correction = (heat_old - heat_new) / jnp.maximum(ocean_vol, 1.0)
        T_corrected = state_new.T.data + T_correction.astype(T_corrected.dtype) * mask[..., jnp.newaxis]

    # --- Salt (S) correction ---
    S_corrected = state_new.S.data
    if config.fix_salt:
        salt_terms = jnp.stack([
            jnp.sum(jnp.sum(cast(state_old.S.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area_acc),
            jnp.sum(jnp.sum(cast(state_new.S.data, _M, "accumulate") * h_k_fix_acc, axis=-1) * weighted_area_acc),
            jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc),
        ])
        salt_old, salt_new, ocean_vol = _ocean_global_sum(salt_terms)
        S_correction = (salt_old - salt_new) / jnp.maximum(ocean_vol, 1.0)
        S_corrected = state_new.S.data + S_correction.astype(S_corrected.dtype) * mask[..., jnp.newaxis]

    # Apply all corrections simultaneously
    return state_new._replace(
        eta=state_new.eta.replace(data=eta_corrected),
        T=state_new.T.replace(data=T_corrected),
        S=state_new.S.replace(data=S_corrected),
    )
