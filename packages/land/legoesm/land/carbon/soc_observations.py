"""Per-archetype OBSERVED soil-organic-carbon (SOC) target for Stage-B carbon
calibration.

The differentiable archetype forward map
(:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`) predicts a
per-archetype equilibrium SOC; the calibration loss compares it against an
OBSERVED per-archetype SOC derived from the surfdata organic-carbon field:

* column SOC [kgC/m2] of a grid cell = the vertical integral of the per-layer
  organic-carbon density over the soil column, ``sum_layer organic * dz`` (from
  ``global_surface_data.GlobalSurfaceData.organic`` [kg/m3] and the soil-layer
  thicknesses ``DZSOI`` [m]);
* the per-archetype target = the COVER-WEIGHTED mean of that column SOC over the
  grid cells assigned to the archetype (via ``cell_archetype_id`` /
  ``cell_archetype_weight`` from
  :func:`legoesm.land.carbon.global_init.build_archetypes`).

Pure NumPy: this is a FIXED target (built once, never inside a JAX-traced model
step), mirroring the Stage-A archetype build so it stays cheap and non-diff.

Units convention: :func:`column_soc` takes ``organic`` as an organic-*carbon*
density [kgC/m3], so the column integral is already [kgC/m2] with NO further
rescale.  A source that reports organic MATTER must be converted to carbon BEFORE
calling.  In particular the raw CLM5 surfdata ``ORGANIC`` [kg/m3] is organic
MATTER (its own units attribute states an assumed carbon content 0.58 gC/gOM = the
van Bemmelen 1/1.724 factor; ORGANIC saturates at 130 kg/m3 = pure-peat OM
density), so the Stage-B trainer's ``_load_surfdata_organic`` multiplies it by
``om_to_oc`` (0.58 for CLM5; 1.0 for a source already in carbon such as HWSD
``ORG_CARBON``) -- the archetype SOC target must be carbon-consistent with the
model's ``som_total`` [gC/m2 -> kgC/m2].
"""

from __future__ import annotations

import numpy as np


def column_soc(organic, dz):
    """Vertically integrate per-layer organic-carbon density to a column SOC.

    Parameters
    ----------
    organic : array (..., n_layer)
        Per-layer organic-carbon density [kgC/m3].
    dz : array (n_layer,) or (..., n_layer)
        Soil-layer thicknesses [m]; broadcast against ``organic`` (a single
        per-column profile, or per-cell thicknesses).

    Returns
    -------
    array (...,)
        Column soil organic carbon [kgC/m2] = ``sum_layer organic * dz``.
    """
    organic = np.asarray(organic, dtype=float)
    dz = np.asarray(dz, dtype=float)
    if organic.shape[-1] != dz.shape[-1]:
        raise ValueError(
            f"organic layer axis {organic.shape[-1]} != dz layer axis "
            f"{dz.shape[-1]}; both must be n_layer.")
    return np.sum(organic * dz, axis=-1)


def per_archetype_cover_weight(
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Total assigned land cover per archetype [dimensionless cover fraction sum].

    ``cover[a] = sum_{(c,p): id[c,p]==a} w[c,p]`` -- the denominator of the
    cover-weighted per-archetype mean in :func:`per_archetype_observed_soc`,
    factored out so the Stage-B calibration loss can weight each archetype's SOC
    residual by exactly the SAME total-cover measure the observed target averages
    over (no duplicated ``bincount`` numerics).  An archetype with no assigned
    (cell, PFT) pair gets ``0.0`` (never member cells -> zero cover).

    Parameters
    ----------
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent / below threshold /
        bare (see :func:`legoesm.land.carbon.global_init.build_archetypes`).
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype summed cover weight [-]; ``0.0`` for an unassigned archetype.
    """
    cid = np.asarray(cell_archetype_id)
    cw = np.asarray(cell_archetype_weight, dtype=float)
    if cid.shape != cw.shape:
        raise ValueError(
            f"cell_archetype_id {cid.shape} and cell_archetype_weight "
            f"{cw.shape} must have the same (ncell, npft) shape.")
    if cid.ndim != 2:
        raise ValueError(
            f"cell_archetype_id must be 2-D (ncell, npft); got shape {cid.shape}.")
    if n_arch is None:
        n_arch = int(cid.max()) + 1 if (cid.size and cid.max() >= 0) else 0
    n_arch = int(n_arch)
    valid = cid >= 0
    flat_id = cid[valid].astype(int)
    flat_w = cw[valid]
    return np.bincount(flat_id, weights=flat_w, minlength=n_arch)[:n_arch]


def per_archetype_observed_soc(
    organic,
    dz,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Cover-weighted per-archetype observed SOC [kgC/m2].

    For archetype ``a`` the target is the cover-weighted mean of the COLUMN SOC of
    every grid cell assigned to it::

        SOC_obs[a] = ( sum_{(c,p): id[c,p]==a} w[c,p] * col_soc[c] )
                     / ( sum_{(c,p): id[c,p]==a} w[c,p] )

    SOC is a per-cell column quantity (independent of PFT), so a cell's column SOC
    is shared by every (cell, PFT) pair that maps to an archetype there, each
    weighted by that PFT's cover ``cell_archetype_weight``.  Mirrors the
    cover-weighted archetype climate means in
    :func:`legoesm.land.carbon.global_init.build_archetypes`.

    Parameters
    ----------
    organic : array (ncell, n_layer)
        Per-cell per-layer organic-carbon density [kgC/m3].
    dz : array (n_layer,) or (ncell, n_layer)
        Soil-layer thicknesses [m].
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where the PFT is absent / below
        the occupancy threshold / bare (see ``build_archetypes``).
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes (rows in the ``ArchetypeTable``).  Defaults to
        ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed SOC [kgC/m2].  ``NaN`` for an archetype with zero
        total assigned cover (no member cells) -- surfaced, never silently 0.
    """
    col = column_soc(organic, dz)                       # (ncell,)
    cid = np.asarray(cell_archetype_id)
    cw = np.asarray(cell_archetype_weight, dtype=float)
    if cid.shape != cw.shape:
        raise ValueError(
            f"cell_archetype_id {cid.shape} and cell_archetype_weight "
            f"{cw.shape} must have the same (ncell, npft) shape.")
    if cid.ndim != 2:
        raise ValueError(
            f"cell_archetype_id must be 2-D (ncell, npft); got shape {cid.shape}.")
    if col.shape[0] != cid.shape[0]:
        raise ValueError(
            f"organic ncell {col.shape[0]} != membership ncell {cid.shape[0]}.")

    if n_arch is None:
        n_arch = int(cid.max()) + 1 if (cid.size and cid.max() >= 0) else 0
    n_arch = int(n_arch)

    # Broadcast the per-cell column SOC over the PFT axis, then accumulate the
    # cover-weighted numerator / denominator per archetype over the valid pairs.
    col_bcast = np.broadcast_to(col[:, None], cid.shape)
    valid = cid >= 0
    flat_id = cid[valid].astype(int)
    flat_w = cw[valid]
    flat_soc = col_bcast[valid]
    num = np.bincount(flat_id, weights=flat_w * flat_soc, minlength=n_arch)[:n_arch]
    # Denominator = total assigned cover per archetype (the SAME measure the
    # Stage-B loss weights each archetype's SOC residual by); factored into the
    # shared helper so the cover-weight bincount is defined once.
    den = per_archetype_cover_weight(cid, cw, n_arch=n_arch)

    out = np.full(n_arch, np.nan, dtype=float)
    nz = den > 0.0
    out[nz] = num[nz] / den[nz]
    return out
