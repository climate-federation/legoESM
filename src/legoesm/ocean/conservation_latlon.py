"""Conservation fixers for the lat-lon FV ocean model.

Volume (free surface), heat (T), and salt (S) conservation via
uniform additive corrections.

Limitations
-----------
These are *uniform additive* fixers — a single scalar correction is
applied to every ocean cell.  This acts as spurious globally-uniform
diapycnal mixing and violates local conservation.  The proper fix is
flux-form tracer advection with barotropic-baroclinic flux
reconciliation (Hallberg 1997, Higdon 2005).  See issue #59.

The combined fixer (``latlon_ocean_conservation_fixer``) computes all
corrections simultaneously from the original pre-fix state, avoiding
order-dependent bias between volume, heat, and salt fixers.  Precision
is upcasted to accumulation dtype before global reductions to prevent
catastrophic cancellation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.precision import cast
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig

_M = "ocean_diagnostics"  # precision module for accumulations


def fix_volume_latlon(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    min_water_column_m: float | None = None,
) -> LatLonOceanState:
    """Fix volume conservation via uniform eta correction."""
    mask = state_old.land_mask.data
    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc

    vol_old = jnp.sum(cast(state_old.eta.data, _M, "accumulate") * weighted_area)
    vol_new = jnp.sum(cast(state_new.eta.data, _M, "accumulate") * weighted_area)
    ocean_area = jnp.sum(weighted_area)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_candidate = state_new.eta.data + correction.astype(state_new.eta.data.dtype) * mask

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

    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_new_acc = cast(h_k_new, _M, "accumulate")

    heat_old = jnp.sum(jnp.sum(cast(state_old.T.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area)
    heat_new = jnp.sum(jnp.sum(cast(state_new.T.data, _M, "accumulate") * h_k_new_acc, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new_acc, axis=-1) * weighted_area)

    correction = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)
    T_fixed = state_new.T.data + correction.astype(state_new.T.data.dtype) * mask[..., jnp.newaxis]
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

    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area = mask_acc * area_acc
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_new_acc = cast(h_k_new, _M, "accumulate")

    salt_old = jnp.sum(jnp.sum(cast(state_old.S.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area)
    salt_new = jnp.sum(jnp.sum(cast(state_new.S.data, _M, "accumulate") * h_k_new_acc, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new_acc, axis=-1) * weighted_area)

    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)
    S_fixed = state_new.S.data + correction.astype(state_new.S.data.dtype) * mask[..., jnp.newaxis]
    return state_new._replace(S=state_new.S.replace(data=S_fixed))


def latlon_ocean_conservation_fixer(
    state_new: LatLonOceanState,
    state_old: LatLonOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonOceanConfig,
) -> LatLonOceanState:
    """Apply all conservation fixers simultaneously.

    All corrections are computed from the original state_old's layer
    thicknesses to avoid order-dependent bias between volume, heat,
    and salt fixers (matching cubed-sphere ``ocean_conservation_fixer``).
    """
    mask = state_old.land_mask.data
    min_wc = config.min_water_column_m

    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area_acc = mask_acc * area_acc

    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_wc,
    )

    # --- Volume (eta) correction ---
    eta_corrected = state_new.eta.data
    if config.fix_volume:
        vol_old = jnp.sum(cast(state_old.eta.data, _M, "accumulate") * weighted_area_acc)
        vol_new = jnp.sum(cast(state_new.eta.data, _M, "accumulate") * weighted_area_acc)
        ocean_area = jnp.sum(weighted_area_acc)
        eta_correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
        eta_corrected = state_new.eta.data + eta_correction.astype(eta_corrected.dtype) * mask
        if min_wc is not None:
            eta_floor = jnp.asarray(min_wc, dtype=eta_corrected.dtype) - state_new.H_bathy.data
            eta_corrected = jnp.maximum(eta_corrected, eta_floor) * mask

    # Recompute h_k from corrected eta for heat/salt integrals.
    h_k_corrected = compute_layer_thickness(
        eta_corrected, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_wc,
    )
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_fix_acc = cast(h_k_corrected, _M, "accumulate")

    # --- Heat (T) correction ---
    T_corrected = state_new.T.data
    if config.fix_heat:
        heat_old = jnp.sum(jnp.sum(cast(state_old.T.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area_acc)
        heat_new = jnp.sum(jnp.sum(cast(state_new.T.data, _M, "accumulate") * h_k_fix_acc, axis=-1) * weighted_area_acc)
        ocean_vol = jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc)
        T_correction = (heat_old - heat_new) / jnp.maximum(ocean_vol, 1.0)
        T_corrected = state_new.T.data + T_correction.astype(T_corrected.dtype) * mask[..., jnp.newaxis]

    # --- Salt (S) correction ---
    S_corrected = state_new.S.data
    if config.fix_salt:
        salt_old = jnp.sum(jnp.sum(cast(state_old.S.data, _M, "accumulate") * h_k_old_acc, axis=-1) * weighted_area_acc)
        salt_new = jnp.sum(jnp.sum(cast(state_new.S.data, _M, "accumulate") * h_k_fix_acc, axis=-1) * weighted_area_acc)
        ocean_vol = jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc)
        S_correction = (salt_old - salt_new) / jnp.maximum(ocean_vol, 1.0)
        S_corrected = state_new.S.data + S_correction.astype(S_corrected.dtype) * mask[..., jnp.newaxis]

    return state_new._replace(
        eta=state_new.eta.replace(data=eta_corrected),
        T=state_new.T.replace(data=T_corrected),
        S=state_new.S.replace(data=S_corrected),
    )
