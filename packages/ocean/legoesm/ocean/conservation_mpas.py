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
order-dependent bias.  Global reductions are routed through the
``ocean_diagnostics`` precision policy (mixed mode → float64, fp32
mode → float32, Metal → float32 via backend clamp) rather than a
hard-coded float64 upcast.  See issue #167.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.precision import cast
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.parallel.reductions import global_sum_if_distributed

_ACC_MODULE = "ocean_diagnostics"


def _ownership_weight(mask, owned_mask):
    """Combine land mask with optional owned-cell mask.

    Under MPI partitioning each rank holds owned + halo cells.  Halo
    cells are also stored on neighbouring ranks, so summing them on
    every rank double-counts.  ``owned_mask`` (1.0 for owned cells,
    0.0 for halo cells) zeros out halo contributions in the local sum
    *before* the global allreduce.

    For single-rank or shard-replicated runs ``owned_mask`` is None
    and we return ``mask`` unchanged.
    """
    if owned_mask is None:
        return mask
    return mask * owned_mask.astype(mask.dtype)


def _ocean_area_sum_mpas(field_2d, mask, mesh, owned_mask=None):
    """Area-weighted sum over ocean cells. MPI-aware (#177)."""
    field_acc = cast(field_2d, _ACC_MODULE, "accumulate")
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
    return global_sum_if_distributed(jnp.sum(field_acc * mask_acc * area_acc))


def _ocean_volume_sum_mpas(field_3d, h_k, mask, mesh, owned_mask=None):
    """Volume-weighted sum over ocean cells. MPI-aware (#177)."""
    field_acc = cast(field_3d, _ACC_MODULE, "accumulate")
    h_k_acc = cast(h_k, _ACC_MODULE, "accumulate")
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
    return global_sum_if_distributed(jnp.sum(
        field_acc * h_k_acc * mask_acc[:, jnp.newaxis] * area_acc[:, jnp.newaxis]
    ))


def _ocean_volume_sums_old_new(field_old, h_k_old, field_new, h_k_new,
                                mask, mesh, owned_mask=None):
    """Stack the 3 (old / new / volume) volume-weighted scalar sums into
    a single ``global_sum_if_distributed`` allreduce.

    The old leg uses ``h_k_old``; both new legs share ``h_k_new``, so
    the new pair is fused into one column reduction first, then the 3
    scalars are stacked for a single ``global_sum_if_distributed``.

    ``owned_mask`` (optional) zeros out halo cells in the local sum
    before the global allreduce — see :func:`_ownership_weight`.
    """
    fold_acc = cast(field_old, _ACC_MODULE, "accumulate")
    fnew_acc = cast(field_new, _ACC_MODULE, "accumulate")
    h_old_acc = cast(h_k_old, _ACC_MODULE, "accumulate")
    h_new_acc = cast(h_k_new, _ACC_MODULE, "accumulate")
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
    weighted = mask_acc[:, jnp.newaxis] * area_acc[:, jnp.newaxis]
    # ``new`` pair shares ``h_new_acc * weighted`` weight — fuse first
    # (collapse over all spatial axes so the result is a 2-vector).
    _new_pair = jnp.sum(
        jnp.stack([fnew_acc, jnp.ones_like(fnew_acc)], axis=-1)
        * (h_new_acc * weighted)[..., None],
        axis=tuple(range(fnew_acc.ndim)),
    )
    field_old_sum = jnp.sum(fold_acc * h_old_acc * weighted)
    return global_sum_if_distributed(jnp.stack([field_old_sum, _new_pair[..., 0], _new_pair[..., 1]]))


def fix_volume_mpas(state_new, state_old, mesh, z_coord,
                    min_water_column_m=None, owned_mask=None):
    """Fix volume conservation via uniform eta correction.

    ``owned_mask`` (optional, shape ``(n_local_cells,)``) restricts the
    local accumulators to owned cells before the global allreduce so
    halo cells are not double-counted under MPI partitioning.
    """
    mask = state_old.land_mask.data

    # Three area-weighted scalars (eta_old, eta_new, ocean_area) share
    # the ``mask * areaCell`` weight on the same horizontal axes — fuse
    # the local column reduction and the allreduce.
    eta_old_acc = cast(state_old.eta.data, _ACC_MODULE, "accumulate")
    eta_new_acc = cast(state_new.eta.data, _ACC_MODULE, "accumulate")
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
    weighted = mask_acc * area_acc
    _local = jnp.sum(
        jnp.stack([eta_old_acc, eta_new_acc, jnp.ones_like(eta_old_acc)], axis=-1)
        * weighted[..., None],
        axis=tuple(range(eta_old_acc.ndim)),
    )
    vol_old, vol_new, ocean_area = global_sum_if_distributed(_local)

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


def fix_heat_mpas(state_new, state_old, mesh, z_coord,
                  min_water_column_m=None, owned_mask=None):
    """Fix heat conservation via uniform T correction.

    ``owned_mask`` (optional) — see :func:`fix_volume_mpas`.
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

    heat_old, heat_new, vol_new = _ocean_volume_sums_old_new(
        state_old.T.data, h_k_old, state_new.T.data, h_k_new, mask, mesh,
        owned_mask=owned_mask,
    )

    correction = (heat_old - heat_new) / jnp.maximum(vol_new, 1.0)
    T_fixed = state_new.T.data + correction.astype(state_new.T.data.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        T=state_new.T.replace(data=T_fixed),
    )


def fix_salt_mpas(state_new, state_old, mesh, z_coord,
                  min_water_column_m=None, owned_mask=None):
    """Fix salt conservation via uniform S correction.

    ``owned_mask`` (optional) — see :func:`fix_volume_mpas`.
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

    salt_old, salt_new, vol_new = _ocean_volume_sums_old_new(
        state_old.S.data, h_k_old, state_new.S.data, h_k_new, mask, mesh,
        owned_mask=owned_mask,
    )

    correction = (salt_old - salt_new) / jnp.maximum(vol_new, 1.0)
    S_fixed = state_new.S.data + correction.astype(state_new.S.data.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        S=state_new.S.replace(data=S_fixed),
    )


def mpas_ocean_conservation_fixer(
    state_new, state_old, mesh, z_coord, config,
    expected_dHeat: float = 0.0,
    expected_dSalt: float = 0.0,
    owned_mask=None,
):
    """Apply all conservation fixers simultaneously (#166, #177).

    expected_dHeat/dSalt: expected forcing change [tracer*m³] **as a
    global quantity** (caller is responsible for the cross-rank sum).

    owned_mask : optional (n_local_cells,) array.  Under MPI, restricts
    the local accumulators to owned cells before the global allreduce
    so halo cells are not double-counted across neighbouring ranks.
    """
    mask = state_old.land_mask.data
    min_col = config.min_water_column_m

    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_col,
    )

    # All accumulations use the ``ocean_diagnostics`` accumulate dtype
    # (float64 in mixed mode, float32 in pure fp32 or on Metal). See
    # issue #167 — the previous code hard-coded float64 which crashed
    # on backends without x64 support.
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
    weighted_area_acc = mask_acc * area_acc

    # --- Volume (eta) correction (3 sums batched into 1 allreduce) ---
    eta_corrected = state_new.eta.data
    if config.fix_volume:
        eta_old_acc = cast(state_old.eta.data, _ACC_MODULE, "accumulate")
        eta_new_acc = cast(state_new.eta.data, _ACC_MODULE, "accumulate")
        # Three area-weighted scalar sums share the ``weighted_area_acc``
        # weight on the same horizontal axes — stack the integrands and
        # reduce once with ``axis=(0, 1, ...)``.
        _vol_stack = jnp.stack(
            [eta_old_acc, eta_new_acc, jnp.ones_like(eta_old_acc)], axis=-1,
        ) * weighted_area_acc[..., None]
        vol_terms = jnp.sum(_vol_stack, axis=tuple(range(eta_old_acc.ndim)))
        vol_old, vol_new, ocean_area = global_sum_if_distributed(vol_terms)
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
    h_k_old_acc = cast(h_k_old, _ACC_MODULE, "accumulate")
    h_k_fix_acc = cast(h_k_corrected, _ACC_MODULE, "accumulate")

    # --- Heat (T) correction (3 sums batched into 1 allreduce; #166) ---
    # The 3 inner column-axis reductions also share the level axis;
    # stack them into one ``jnp.sum(..., axis=-2)`` and then collapse
    # the area-weighted outer sum with ``axis=0`` so the whole 3-term
    # diagnostic costs one column reduction + one area reduction.
    T_corrected = state_new.T.data
    if config.fix_heat:
        T_old_acc = cast(state_old.T.data, _ACC_MODULE, "accumulate")
        T_new_acc = cast(state_new.T.data, _ACC_MODULE, "accumulate")
        _heat_inner = jnp.sum(
            jnp.stack(
                [T_old_acc * h_k_old_acc, T_new_acc * h_k_fix_acc, h_k_fix_acc],
                axis=-1,
            ),
            axis=-2,
        )
        heat_terms = jnp.sum(_heat_inner * weighted_area_acc[..., None], axis=0)
        heat_old, heat_new, ocean_vol = global_sum_if_distributed(heat_terms)
        heat_expected = heat_old + jnp.asarray(expected_dHeat, dtype=heat_old.dtype)
        T_correction = (heat_expected - heat_new) / jnp.maximum(ocean_vol, 1.0)
        T_corrected = state_new.T.data + T_correction.astype(T_corrected.dtype) * mask[:, jnp.newaxis]

    # --- Salt (S) correction (3 sums batched into 1 allreduce) ---
    S_corrected = state_new.S.data
    if config.fix_salt:
        S_old_acc = cast(state_old.S.data, _ACC_MODULE, "accumulate")
        S_new_acc = cast(state_new.S.data, _ACC_MODULE, "accumulate")
        _salt_inner = jnp.sum(
            jnp.stack(
                [S_old_acc * h_k_old_acc, S_new_acc * h_k_fix_acc, h_k_fix_acc],
                axis=-1,
            ),
            axis=-2,
        )
        salt_terms = jnp.sum(_salt_inner * weighted_area_acc[..., None], axis=0)
        salt_old, salt_new, ocean_vol = global_sum_if_distributed(salt_terms)
        salt_expected = salt_old + jnp.asarray(expected_dSalt, dtype=salt_old.dtype)
        S_correction = (salt_expected - salt_new) / jnp.maximum(ocean_vol, 1.0)
        S_corrected = state_new.S.data + S_correction.astype(S_corrected.dtype) * mask[:, jnp.newaxis]

    return state_new._replace(
        eta=state_new.eta.replace(data=eta_corrected),
        T=state_new.T.replace(data=T_corrected),
        S=state_new.S.replace(data=S_corrected),
    )
