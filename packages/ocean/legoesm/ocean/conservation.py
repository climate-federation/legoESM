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

from typing import Callable, Union

from legoesm.core.operators import _is_distributed
from legoesm.core.precision import cast
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import (
    OceanState, OceanConfig, LatLonCGridOceanState, LatLonCGridOceanConfig,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.latlon import LatLonGrid
from legoesm.parallel.reductions import global_sum_mpi

# Types accepted by the conservation fixer (cubed-sphere + lat-lon C-grid)
_OceanStateT = Union[OceanState, LatLonCGridOceanState]
_OceanConfigT = Union[OceanConfig, LatLonCGridOceanConfig]
_GridT = Union[CubedSphereGrid, LatLonGrid]


def ocean_global_sum(local_value):
    """MPI-aware global sum for scalar or vector reductions.

    NOTE: this gates only on ``_is_distributed()`` (the mpi4jax/sharded flag),
    whereas the MPAS twin :func:`legoesm.parallel.reductions.global_sum_if_distributed`
    also reduces when ``jax.process_count() > 1`` (JAX multi-host).  Under the
    ocean's actual MPI usage the two are identical (``_is_distributed`` is set,
    ``process_count`` stays 1).  They differ only for a JAX-multi-host ocean run
    without the MPI flag, where this form would under-reduce.  Unifying onto the
    canonical helper is deliberately deferred until that path can be validated
    (single-rank vs MPI vs sharded), to avoid a silent reduction-semantics change
    in the conservation fixers.
    """
    if _is_distributed():
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
    return ocean_global_sum(local_sum)


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
    vol_old, vol_new, ocean_area = ocean_global_sum(local_terms)

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
    T_old_acc = cast(state_old.T.data, _M, "accumulate")
    T_new_acc = cast(state_new.T.data, _M, "accumulate")
    # Three column reductions share the level axis; stack their
    # integrands and reduce once.  Then the area-weighted outer sum
    # collapses to one ``jnp.sum`` over the horizontal axes.
    _heat_inner = jnp.sum(
        jnp.stack(
            [T_old_acc * h_k_old_acc, T_new_acc * h_k_new_acc, h_k_new_acc],
            axis=-1,
        ),
        axis=-2,
    )
    local_terms = jnp.sum(
        _heat_inner * weighted_area[..., None],
        axis=tuple(range(weighted_area.ndim)),
    )
    heat_old, heat_new, ocean_volume = ocean_global_sum(local_terms)
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
    S_old_acc = cast(state_old.S.data, _M, "accumulate")
    S_new_acc = cast(state_new.S.data, _M, "accumulate")
    # Same 3-into-1 fusion as the heat fixer.
    _salt_inner = jnp.sum(
        jnp.stack(
            [S_old_acc * h_k_old_acc, S_new_acc * h_k_new_acc, h_k_new_acc],
            axis=-1,
        ),
        axis=-2,
    )
    local_terms = jnp.sum(
        _salt_inner * weighted_area[..., None],
        axis=tuple(range(weighted_area.ndim)),
    )
    salt_old, salt_new, ocean_volume = ocean_global_sum(local_terms)
    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)

    S_fixed = state_new.S.replace(
        data=state_new.S.data + correction.astype(state_new.S.data.dtype) * mask[..., jnp.newaxis],
    )
    return state_new._replace(S=S_fixed)


def conservation_fixer_core(
    state_new,
    state_old,
    z_coord: OceanZStarCoordinate,
    *,
    fix_volume: bool,
    fix_heat: bool,
    fix_salt: bool,
    min_water_column_m: float | None,
    weighted_area_acc,
    reduce_fn: Callable,
    apply_eta_floor: Callable,
    expected_dHeat: float = 0.0,
    expected_dSalt: float = 0.0,
):
    """Topology-agnostic core of the simultaneous ocean conservation fixer.

    Computes the uniform volume (eta), heat (T) and salt (S) additive
    corrections from the ORIGINAL ``state_old`` layer thicknesses (heat/salt
    use the post-volume-fix geometry) and applies them simultaneously.  This
    is the shared implementation behind :func:`ocean_conservation_fixer`
    (structured 2D ``grid.area``) and
    :func:`legoesm.ocean.conservation_mpas.mpas_ocean_conservation_fixer`
    (Voronoi/MPAS 1D ``mesh.areaCell`` + MPI ownership mask).

    The three topology differences are parameterised explicitly:

    weighted_area_acc
        Per-cell area weight ALREADY upcast to accumulation precision and
        multiplied by the (effective) land/ownership mask, broadcast to the
        horizontal field shape (``(6, n, n)`` / ``(nlat, nlon)`` structured,
        ``(n_local_cells,)`` MPAS-with-owned-mask-folded-in).
    reduce_fn
        Global-reduction callable applied to the stacked local scalar sums
        (``ocean_global_sum`` structured / ``global_sum_if_distributed`` MPAS).
    apply_eta_floor
        Callable ``(eta_corrected, H_bathy, mask, min_water_column_m) ->
        eta_floored`` encoding the topology-specific minimum-water-column
        floor (structured ``max(eta, floor) * mask`` vs MPAS
        ``where(mask > 0.5, max(eta, floor), eta)``).  Invoked only when
        ``fix_volume`` and ``min_water_column_m is not None``.

    The MPAS-only ``expected_dHeat`` / ``expected_dSalt`` global forcing
    increments default to ``0.0`` (the structured case), reproducing the
    original ``heat_old`` / ``salt_old`` targets exactly.
    """
    mask = state_old.land_mask.data
    _M = "ocean_diagnostics"  # precision module for accumulations

    # Precompute layer thicknesses from the OLD state (before any fixer).
    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )

    # --- Volume (eta) correction ---
    eta_corrected = state_new.eta.data
    if fix_volume:
        eta_old_acc = cast(state_old.eta.data, _M, "accumulate")
        eta_new_acc = cast(state_new.eta.data, _M, "accumulate")
        # Three area-weighted scalar sums share the same horizontal axes
        # and the ``weighted_area_acc`` weight — stack the integrands and
        # reduce once locally so XLA fires one sum kernel.
        _vol_stack = jnp.stack(
            [eta_old_acc, eta_new_acc, jnp.ones_like(eta_old_acc)], axis=-1,
        ) * weighted_area_acc[..., None]
        vol_terms = jnp.sum(
            _vol_stack, axis=tuple(range(weighted_area_acc.ndim)),
        )
        vol_old, vol_new, ocean_area = reduce_fn(vol_terms)
        eta_correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
        eta_corrected = state_new.eta.data + eta_correction.astype(eta_corrected.dtype) * mask
        if min_water_column_m is not None:
            eta_corrected = apply_eta_floor(
                eta_corrected, state_new.H_bathy.data, mask, min_water_column_m,
            )

    # Recompute h_k from the CORRECTED eta so heat/salt integrals use
    # layer thicknesses consistent with the actual post-fix state.
    # Using h_k_old here would conserve against a stale geometry.
    h_k_corrected = compute_layer_thickness(
        eta_corrected, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_fix_acc = cast(h_k_corrected, _M, "accumulate")

    # --- Heat (T) correction ---
    # Stack the 3 inner column reductions and the 3 outer area sums per
    # tracer so each correction pays one column reduction + one area
    # reduction instead of three of each.
    T_corrected = state_new.T.data
    if fix_heat:
        T_old_acc = cast(state_old.T.data, _M, "accumulate")
        T_new_acc = cast(state_new.T.data, _M, "accumulate")
        _heat_inner = jnp.sum(
            jnp.stack(
                [T_old_acc * h_k_old_acc, T_new_acc * h_k_fix_acc, h_k_fix_acc],
                axis=-1,
            ),
            axis=-2,
        )
        heat_terms = jnp.sum(
            _heat_inner * weighted_area_acc[..., None],
            axis=tuple(range(weighted_area_acc.ndim)),
        )
        heat_old, heat_new, ocean_vol = reduce_fn(heat_terms)
        heat_expected = heat_old + jnp.asarray(expected_dHeat, dtype=heat_old.dtype)
        T_correction = (heat_expected - heat_new) / jnp.maximum(ocean_vol, 1.0)
        T_corrected = state_new.T.data + T_correction.astype(T_corrected.dtype) * mask[..., jnp.newaxis]

    # --- Salt (S) correction ---
    S_corrected = state_new.S.data
    if fix_salt:
        S_old_acc = cast(state_old.S.data, _M, "accumulate")
        S_new_acc = cast(state_new.S.data, _M, "accumulate")
        _salt_inner = jnp.sum(
            jnp.stack(
                [S_old_acc * h_k_old_acc, S_new_acc * h_k_fix_acc, h_k_fix_acc],
                axis=-1,
            ),
            axis=-2,
        )
        salt_terms = jnp.sum(
            _salt_inner * weighted_area_acc[..., None],
            axis=tuple(range(weighted_area_acc.ndim)),
        )
        salt_old, salt_new, ocean_vol = reduce_fn(salt_terms)
        salt_expected = salt_old + jnp.asarray(expected_dSalt, dtype=salt_old.dtype)
        S_correction = (salt_expected - salt_new) / jnp.maximum(ocean_vol, 1.0)
        S_corrected = state_new.S.data + S_correction.astype(S_corrected.dtype) * mask[..., jnp.newaxis]

    # Apply all corrections simultaneously
    return state_new._replace(
        eta=state_new.eta.replace(data=eta_corrected),
        T=state_new.T.replace(data=T_corrected),
        S=state_new.S.replace(data=S_corrected),
    )


def _structured_eta_floor(eta_corrected, H_bathy, mask, min_water_column_m):
    """Structured-grid minimum-water-column floor: ``max(eta, floor) * mask``."""
    eta_floor = jnp.asarray(min_water_column_m, dtype=eta_corrected.dtype) - H_bathy
    return jnp.maximum(eta_corrected, eta_floor) * mask


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
    _M = "ocean_diagnostics"  # precision module for accumulations
    mask = state_old.land_mask.data
    # Upcast shared arrays to accumulation precision for global sums.
    mask_acc = cast(mask, _M, "accumulate")
    area_acc = cast(grid.area, _M, "accumulate")
    weighted_area_acc = mask_acc * area_acc

    return conservation_fixer_core(
        state_new,
        state_old,
        z_coord,
        fix_volume=config.fix_volume,
        fix_heat=config.fix_heat,
        fix_salt=config.fix_salt,
        min_water_column_m=config.min_water_column_m,
        weighted_area_acc=weighted_area_acc,
        reduce_fn=ocean_global_sum,
        apply_eta_floor=_structured_eta_floor,
    )
