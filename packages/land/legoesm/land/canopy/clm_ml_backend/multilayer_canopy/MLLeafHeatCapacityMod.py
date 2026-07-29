"""
JAX translation of MLLeafHeatCapacityMod Fortran module.

Calculate leaf heat capacity for each canopy layer.

Original Fortran module: MLLeafHeatCapacityMod
Fortran lines 1-75

Differentiability notes
-----------------------
The inner layer loop is replaced by ``jax.vmap`` over the full layer
dimension (indices 1..nlevmlcan).  Layers with ``dpai == 0`` are masked
via ``jnp.where`` rather than a Python ``if``, so the function is fully
differentiable and ``jax.jit``-compatible.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import cpliq  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod import pftcon  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import cpbio, fcarbon, fwater  # noqa: F401

# ---------------------------------------------------------------------------
# Per-layer kernel — vmapped over the layer axis
# ---------------------------------------------------------------------------


def _cpleaf_layer(dpai_ic, slatop_pft):
    """Leaf heat capacity for one canopy layer (differentiable).

    Args:
        dpai_ic:    Plant-area index increment for this layer (m2/m2).
        slatop_pft: Specific leaf area at canopy top for this PFT (m2/gC).

    Returns:
        Heat capacity (J/K/m2 leaf); 0 when dpai_ic == 0.
    """
    # Leaf carbon mass per area: m2/gC → kg C/m2 — Fortran line 57
    lma = 1.0 / slatop_pft * 0.001
    # Leaf dry mass per area — Fortran line 58
    dry_weight = lma / fcarbon
    # Leaf fresh mass per area — Fortran line 59
    fresh_weight = dry_weight / (1.0 - fwater)
    # Leaf water content — Fortran line 60
    leaf_water = fwater * fresh_weight
    # Heat capacity — Fortran line 61
    cpleaf_val = cpbio * dry_weight + cpliq * leaf_water
    # Zero-out empty layers without Python if — Fortran lines 62-64
    return jnp.where(dpai_ic > 0.0, cpleaf_val, 0.0)


# vmap over layer axis; slatop_pft is a patch-level scalar (broadcast)
_cpleaf_layers = jax.vmap(_cpleaf_layer, in_axes=(0, None))


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def LeafHeatCapacity(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Calculate leaf heat capacity for each canopy layer.

    Mirrors Fortran subroutine ``LeafHeatCapacity`` (lines 22-75).

    Reference: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496,
    doi:10.5194/gmd-11-1467-2018, eq. (A29).

    The derivation converts specific leaf area (m2/gC) to leaf mass
    per area and then accounts for the water fraction of fresh biomass
    (Fortran lines 55-65):

    .. code-block:: none

        lma          = 1/slatop * 0.001          [kg C / m2 leaf]
        dry_weight   = lma / fcarbon             [kg DM / m2 leaf]
        fresh_weight = dry_weight / (1 - fwater) [kg FM / m2 leaf]
        leaf_water   = fwater * fresh_weight     [kg H2O / m2 leaf]
        cpleaf       = cpbio * dry_weight + cpliq * leaf_water  [J/K/m2 leaf]

    Layers with ``dpai == 0`` receive ``cpleaf = 0``.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (0-based values, length num_filter).
        mlcanopy_inst: Canopy container; ``cpleaf_profile`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    slatop = pftcon.slatop
    dpai = mlcanopy_inst.dpai_profile  # shape (num_patch, nlevmlcan+1)
    cpleaf = mlcanopy_inst.cpleaf_profile  # shape (num_patch, nlevmlcan+1)

    for fp in range(1 if grid is not None else num_filter):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else filter_patch[fp]  # grid.p may be TRACED (S3)
        pft = patch.itype[p]  # JAX int — dynamic index, differentiable (traced-p safe)

        # vmap over all layers (index 1..nlevmlcan); dpai==0 layers yield 0
        layers = _cpleaf_layers(dpai[p, 1:], slatop[pft])  # shape (nlevmlcan,)
        cpleaf = cpleaf.at[p, 1:].set(layers)

    return mlcanopy_inst._replace(cpleaf_profile=cpleaf)
