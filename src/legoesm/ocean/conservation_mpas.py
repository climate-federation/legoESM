"""Conservation fixers for MPAS ocean on Voronoi meshes.

Fixes volume, heat, and salt conservation via uniform corrections,
using area-weighted integrals on the Voronoi mesh.

Limitations
-----------
These are *uniform additive* fixers — a single scalar correction is
applied to every ocean cell.  This acts as spurious globally-uniform
diapycnal mixing and violates local conservation.  The proper fix is
flux-form tracer advection with barotropic-baroclinic flux
reconciliation (Hallberg 1997, Higdon 2005).  See issue #59.

The combined fixer (``mpas_ocean_conservation_fixer``) computes all
corrections simultaneously from the original pre-fix state, avoiding
order-dependent bias.  Precision is upcasted to accumulation dtype
before global reductions to prevent catastrophic cancellation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.vertical import compute_layer_thickness


def _ocean_area_sum_mpas(field_2d, mask, mesh):
    """Area-weighted sum over ocean cells with float64 accumulation.

    Explicit float64 upcast prevents catastrophic cancellation in global
    reductions (the default precision policy may resolve "accumulate" to
    float32, which is insufficient for conservation fixers).
    """
    field_acc = field_2d.astype(jnp.float64)
    mask_acc = mask.astype(jnp.float64)
    area_acc = mesh.areaCell.astype(jnp.float64)
    return jnp.sum(field_acc * mask_acc * area_acc)


def _ocean_volume_sum_mpas(field_3d, h_k, mask, mesh):
    """Volume-weighted sum over ocean cells and levels with float64 accumulation."""
    field_acc = field_3d.astype(jnp.float64)
    h_k_acc = h_k.astype(jnp.float64)
    mask_acc = mask.astype(jnp.float64)
    area_acc = mesh.areaCell.astype(jnp.float64)
    return jnp.sum(
        field_acc * h_k_acc * mask_acc[:, jnp.newaxis] * area_acc[:, jnp.newaxis]
    )


def fix_volume_mpas(state_new, state_old, mesh, z_coord, min_water_column_m=None):
    """Fix volume conservation via uniform eta correction."""
    mask = state_old.land_mask.data

    vol_old = _ocean_area_sum_mpas(state_old.eta.data, mask, mesh)
    vol_new = _ocean_area_sum_mpas(state_new.eta.data, mask, mesh)
    ocean_area = _ocean_area_sum_mpas(jnp.ones_like(mask), mask, mesh)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    eta_fixed = state_new.eta.data + correction.astype(state_new.eta.data.dtype) * mask

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
    """Fix heat conservation via uniform T correction."""
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

    correction = (heat_old - heat_new) / jnp.maximum(vol_new, 1.0)
    T_fixed = state_new.T.data + correction.astype(state_new.T.data.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        T=state_new.T.replace(data=T_fixed),
    )


def fix_salt_mpas(state_new, state_old, mesh, z_coord, min_water_column_m=None):
    """Fix salt conservation via uniform S correction."""
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

    correction = (salt_old - salt_new) / jnp.maximum(vol_new, 1.0)
    S_fixed = state_new.S.data + correction.astype(state_new.S.data.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        S=state_new.S.replace(data=S_fixed),
    )


def mpas_ocean_conservation_fixer(state_new, state_old, mesh, z_coord, config):
    """Apply all conservation fixers simultaneously.

    All corrections are computed from the original state_old's layer
    thicknesses to avoid order-dependent bias between volume, heat,
    and salt fixers (matching cubed-sphere ``ocean_conservation_fixer``).
    """
    mask = state_old.land_mask.data
    min_col = config.min_water_column_m

    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_col,
    )

    # Explicit float64 upcast for all accumulations — the default precision
    # policy may resolve "accumulate" to float32, which loses ~7 digits and
    # makes conservation fixers ineffective.
    _f64 = jnp.float64
    mask_acc = mask.astype(_f64)
    area_acc = mesh.areaCell.astype(_f64)
    weighted_area_acc = mask_acc * area_acc

    # --- Volume (eta) correction ---
    eta_corrected = state_new.eta.data
    if config.fix_volume:
        vol_old = jnp.sum(state_old.eta.data.astype(_f64) * weighted_area_acc)
        vol_new = jnp.sum(state_new.eta.data.astype(_f64) * weighted_area_acc)
        ocean_area = jnp.sum(weighted_area_acc)
        eta_correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
        eta_corrected = state_new.eta.data + eta_correction.astype(eta_corrected.dtype) * mask
        if min_col is not None:
            eta_floor = min_col - state_new.H_bathy.data
            eta_corrected = jnp.where(
                mask > 0.5, jnp.maximum(eta_corrected, eta_floor), eta_corrected,
            )

    # Recompute h_k from corrected eta for heat/salt integrals.
    h_k_corrected = compute_layer_thickness(
        eta_corrected, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_col,
    )
    h_k_old_acc = h_k_old.astype(_f64)
    h_k_fix_acc = h_k_corrected.astype(_f64)

    # --- Heat (T) correction ---
    T_corrected = state_new.T.data
    if config.fix_heat:
        heat_old = jnp.sum(jnp.sum(state_old.T.data.astype(_f64) * h_k_old_acc, axis=-1) * weighted_area_acc)
        heat_new = jnp.sum(jnp.sum(state_new.T.data.astype(_f64) * h_k_fix_acc, axis=-1) * weighted_area_acc)
        ocean_vol = jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc)
        T_correction = (heat_old - heat_new) / jnp.maximum(ocean_vol, 1.0)
        T_corrected = state_new.T.data + T_correction.astype(T_corrected.dtype) * mask[:, jnp.newaxis]

    # --- Salt (S) correction ---
    S_corrected = state_new.S.data
    if config.fix_salt:
        salt_old = jnp.sum(jnp.sum(state_old.S.data.astype(_f64) * h_k_old_acc, axis=-1) * weighted_area_acc)
        salt_new = jnp.sum(jnp.sum(state_new.S.data.astype(_f64) * h_k_fix_acc, axis=-1) * weighted_area_acc)
        ocean_vol = jnp.sum(jnp.sum(h_k_fix_acc, axis=-1) * weighted_area_acc)
        S_correction = (salt_old - salt_new) / jnp.maximum(ocean_vol, 1.0)
        S_corrected = state_new.S.data + S_correction.astype(S_corrected.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        eta=state_new.eta.replace(data=eta_corrected),
        T=state_new.T.replace(data=T_corrected),
        S=state_new.S.replace(data=S_corrected),
    )
