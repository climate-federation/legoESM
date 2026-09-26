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
from legoesm.ocean.conservation import conservation_fixer_core
from legoesm.parallel.reductions import global_sum_if_distributed

_ACC_MODULE = "ocean_diagnostics"


def _mpas_eta_floor(eta_corrected, H_bathy, mask, min_water_column_m):
    """MPAS minimum-water-column floor: ``where(mask > 0.5, max(eta, floor), eta)``."""
    eta_floor = min_water_column_m - H_bathy
    return jnp.where(
        mask > 0.5, jnp.maximum(eta_corrected, eta_floor), eta_corrected,
    )


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


def mpas_ocean_conservation_fixer(
    state_new, state_old, mesh, z_coord, config,
    expected_dHeat: float = 0.0,
    expected_dSalt: float = 0.0,
    owned_mask=None,
    reduce_fn=None,
):
    """Apply all conservation fixers simultaneously (#166, #177).

    expected_dHeat/dSalt: expected forcing change [tracer*m³] **as a
    global quantity** (caller is responsible for the cross-rank sum).

    owned_mask : optional (n_local_cells,) array.  Under MPI, restricts
    the local accumulators to owned cells before the global allreduce
    so halo cells are not double-counted across neighbouring ranks.
    """
    mask = state_old.land_mask.data

    # All accumulations use the ``ocean_diagnostics`` accumulate dtype
    # (float64 in mixed mode, float32 in pure fp32 or on Metal). See
    # issue #167 — the previous code hard-coded float64 which crashed
    # on backends without x64 support.  ``owned_mask`` (when supplied)
    # zeros out halo cells in the local sum before the global allreduce.
    eff_mask = _ownership_weight(mask, owned_mask)
    mask_acc = cast(eff_mask, _ACC_MODULE, "accumulate")
    area_acc = cast(mesh.areaCell, _ACC_MODULE, "accumulate")
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
        # ``reduce_fn`` (list -> list cross-rank SUM) from the distributed
        # context when the caller carries one (SPMD psum / MPI allreduce);
        # the historical ``is_multi_process``-gated allreduce otherwise.
        reduce_fn=(global_sum_if_distributed if reduce_fn is None
                   else (lambda v: reduce_fn([v])[0])),
        apply_eta_floor=_mpas_eta_floor,
        expected_dHeat=expected_dHeat,
        expected_dSalt=expected_dSalt,
    )
