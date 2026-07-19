"""Per-archetype OBSERVED leaf-area-index (LAI) target for Stage-B carbon calibration.

The MODELLED side (:func:`legoesm.land.carbon.live_pool_forward.compute_biomass_lai`)
predicts a per-archetype simulated LAI (``C_fol / LCMA``); the calibration ``lai`` loss
term compares it against an OBSERVED per-archetype LAI derived from the CLM5 surfdata
``MONTHLY_LAI`` climatology, cover-weighted per archetype (mirroring
:mod:`legoesm.land.carbon.soc_observations` and :mod:`legoesm.land.carbon.sif_observations`).

Unlike SOC / SIF (per-CELL quantities), the surfdata LAI is PER-(PFT, cell)
(``MONTHLY_LAI(month, pft, lat, lon)``): every natural/crop PFT carries its OWN monthly
LAI climatology.  An archetype is a ``(PFT, climate-cluster)`` group, so its observed LAI
is the cover-weighted mean over the (cell, PFT) pairs assigned to it of that PFT's LAI at
that cell -- a per-(cell, PFT) cover-weighted mean, via the SHARED
:func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` (which
accepts a ``(ncell, npft)`` value array; one cover-weighting definition, no duplicated
``bincount`` numerics).

GROWING-SEASON METRIC: the 12 monthly LAI values are aggregated to the ANNUAL MEAN
(:func:`annual_mean_lai`).  Rationale: the modelled forward is the closed-form ANNUAL-MEAN
foliar stock ``C_fol = (a_fol + a_lab) * leaf_lifespan * NPP``
(:func:`legoesm.land.carbon.live_pool_forward.compute_live_pools`, a throughput x residence
steady state = time-mean stock), so the OBSERVED target MUST share the SAME temporal
definition (an annual mean) for the loss to be an UNBIASED comparison -- a growing-season
maximum or growing-season mean is systematically LARGER than the time-mean and would bias
the fit (the closed form omits the seasonal foliage amplitude by construction).  Recovering
a growing-season-max / -mean target alongside a seasonal-amplitude forward is a documented
refinement (like the SIF per-archetype LAI / soil-moisture refinements).

Pure NumPy: a FIXED target (built once, never inside a JAX-traced model step), like
``soc_observations`` / ``sif_observations``.

DATA SOURCE: the raw CLM5 surfdata ``MONTHLY_LAI`` on the surfdata's NATIVE grid (the same
native ``clm5_surfdata`` cover path the observed SOC / SIF use -- no regrid); NO new
download.  Its PFT axis is the 17-PFT CLM5 ordering aligned with the reconstructed cover
(the assumption ``clm_surface_map`` makes at ``MONTHLY_LAI`` load).  Missing / degenerate
(cell, PFT) LAI is guarded to ``NaN`` so the cover-weighted mean excludes it and an
all-missing archetype surfaces ``NaN`` (masked out of the loss), never a fabricated 0.
For a self-contained test without the surfdata, :func:`synthetic_observed_lai` fabricates a
physically-oriented per-archetype target.
"""

from __future__ import annotations

import numpy as np
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean

from legoesm import constants

# --- synthetic (test-only) observed-LAI field [m2/m2]; NOT a real product ---
# Physically-oriented offsets so brighter/warmer archetypes carry more leaf area (the
# allocation/LCMA params then have signal to fit); an annual-mean LAI scale (O(1) m2/m2)
# deliberately offset from the default-parameter simulated LAI so the synthetic dry-run has
# a clear residual.
_LAI_SYNTH_BASE = 0.5             # baseline annual-mean LAI [m2/m2]
_LAI_SYNTH_PER_W = 0.01           # per growing-season mean shortwave [m2/m2 / (W m-2)]
_LAI_SYNTH_PER_K = 0.05           # per degree above freezing [m2/m2 / K]
# Smallest LAI treated as a valid observation [m2/m2]: at/below this a (cell, PFT) LAI is a
# bare/degenerate cell (no canopy), guarded to NaN so it is excluded from the archetype mean
# rather than dragging it toward 0.
_LAI_MIN_VALID = 1.0e-4


def annual_mean_lai(monthly_lai):
    """Annual-mean LAI over the 12-month climatology, collapsing the leading time axis.

    ``monthly_lai(month, ...)`` -> mean over ``month`` (axis 0), IGNORING NaN gaps
    (``nanmean``); an all-NaN (never-observed) (cell, PFT) stays NaN (the intended "no
    data").  The documented GROWING-SEASON metric (see the module docstring): an annual
    MEAN, consistent with the annual-mean foliar-stock forward.
    """
    arr = np.asarray(monthly_lai, dtype=float)
    if arr.shape[0] != 12:
        raise SystemExit(
            f"MONTHLY_LAI leading axis expected 12 months; got {arr.shape[0]}.")
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def per_archetype_observed_lai(
    lai_cell_pft,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Cover-weighted per-archetype observed LAI [m2/m2] from a per-(cell, PFT) LAI.

    For archetype ``a`` the target is the cover-weighted mean of the per-(cell, PFT) LAI of
    every (cell, PFT) pair assigned to it::

        LAI_obs[a] = ( sum_{(c,p): id[c,p]==a} w[c,p] * lai_cell_pft[c,p] )
                     / ( sum_{(c,p): id[c,p]==a} w[c,p] )

    LAI is a per-(cell, PFT) quantity (each PFT has its own canopy), so -- unlike the
    per-cell SOC / SIF -- the value used for pair ``(c, p)`` is that PFT's LAI at that cell.
    Delegates to the SHARED
    :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` with a
    ``(ncell, npft)`` value array (the same cover-weighting the observed SOC / SIF targets
    use -- no duplicated numerics).

    Parameters
    ----------
    lai_cell_pft : array (ncell, npft)
        Per-(cell, PFT) observed annual-mean LAI [m2/m2]; ``NaN`` where missing / bare
        (excluded from the mean).
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent / below threshold / bare.
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed LAI [m2/m2].  ``NaN`` for an archetype with no finite-LAI
        member cover -- surfaced, never silently 0.
    """
    return per_archetype_cover_weighted_mean(
        lai_cell_pft, cell_archetype_id, cell_archetype_weight, n_arch=n_arch)


def load_surfdata_lai(surf_path, *, ncell, n_pft, lai_var=None):
    """Per-(cell, PFT) observed annual-mean LAI ``(ncell, npft)`` [m2/m2] from CLM5 surfdata.

    ``MONTHLY_LAI(month, pft, nlat, nlon)`` is collapsed to the ANNUAL MEAN over the 12
    months (:func:`annual_mean_lai`), reshaped to the SAME row-major ``(ncell = nlat*nlon)``
    order the CLM5 cover loader uses, and transposed to ``(ncell, npft)`` so the (cell, PFT)
    indices align with the archetype membership.  The PFT axis MUST match the cover
    ``n_pft`` (the 17-PFT CLM5 ordering aligned with the reconstructed cover -- the
    assumption ``clm_surface_map`` makes at ``MONTHLY_LAI`` load); a mismatch is a hard
    error, never a silent misalignment.  Non-finite or ``<= _LAI_MIN_VALID`` (bare /
    degenerate) LAI is PRESERVED as ``NaN`` (never fabricated to 0) so the per-archetype
    cover-weighted mean averages only the observed (cell, PFT) pairs.

    Parameters
    ----------
    surf_path : str
        Raw CLM5 surfdata NetCDF path (``--surf-path``; carries ``MONTHLY_LAI``).
    ncell : int
        Expected number of cells (= the cover ``ncell``); a mismatch raises.
    n_pft : int
        Expected number of PFTs (= the cover ``npft``); a MONTHLY_LAI PFT-axis mismatch
        raises (an ``natpft``-only file needs the crop PFTs padded -- a follow-up).
    lai_var : str, optional
        LAI variable name; defaults to ``MONTHLY_LAI`` (auto-detected from a small
        candidate list if absent).
    """
    import xarray as xr

    ds = xr.open_dataset(surf_path, decode_times=False)
    try:
        candidates = ("MONTHLY_LAI", "monthly_lai")
        var = lai_var or next((v for v in candidates if v in ds), None)
        if var is None:
            raise SystemExit(
                f"surfdata {surf_path} has no MONTHLY_LAI variable (looked for "
                f"{candidates}); pass --lai-var. Available: {sorted(ds.data_vars)[:40]}")
        arr = np.asarray(ds[var].values, dtype=float)   # (month, pft, nlat, nlon)
    finally:
        ds.close()
    if arr.ndim != 4:
        raise SystemExit(
            f"MONTHLY_LAI expected 4-D (month, pft, nlat, nlon); got shape {arr.shape}.")
    lai_annual = annual_mean_lai(arr)                   # (pft, nlat, nlon)
    npft_lai, n_lat, n_lon = lai_annual.shape
    if npft_lai != int(n_pft):
        raise SystemExit(
            f"MONTHLY_LAI has {npft_lai} PFTs but the cover has {int(n_pft)} "
            f"(the surfdata LAI must share the 17-PFT CLM5 ordering aligned with the "
            f"reconstructed cover; an natpft-only file needs the crop PFTs padded -- a "
            f"follow-up). Use --dry-run-synthetic for a self-contained test.")
    if n_lat * n_lon != int(ncell):
        raise SystemExit(
            f"MONTHLY_LAI ncell {n_lat * n_lon} (= {n_lat}x{n_lon}) != cover ncell "
            f"{int(ncell)}; the surfdata LAI must be on the same native grid as the cover "
            f"(clm5_surfdata). Use --dry-run-synthetic for a self-contained test.")
    # (pft, nlat, nlon) -> (ncell, npft) row-major (i_lat, i_lon), matching the cover.
    lai_cell_pft = np.moveaxis(lai_annual, 0, -1).reshape(n_lat * n_lon, npft_lai)
    # Guard missing / bare / degenerate cells -> NaN (excluded from the archetype mean).
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(lai_cell_pft) | (lai_cell_pft <= _LAI_MIN_VALID)
    lai_cell_pft = np.where(bad, np.nan, lai_cell_pft)
    return lai_cell_pft


def synthetic_observed_lai(table):
    """Deterministic per-archetype observed LAI [m2/m2] for ``--dry-run-synthetic``
    (no data files).

    LAI rises with growing-season shortwave and warmth (brighter/warmer archetypes carry
    more leaf area), so the allocation / LCMA parameters have signal to fit.  NOT a real
    product -- see :func:`load_surfdata_lai` for the surfdata source.
    """
    sw = np.asarray(table.sw_mean_w, dtype=float)
    mat_c = np.asarray(table.mat_k, dtype=float) - constants.T_freeze
    return (_LAI_SYNTH_BASE
            + _LAI_SYNTH_PER_W * np.maximum(sw, 0.0)
            + _LAI_SYNTH_PER_K * np.maximum(mat_c, 0.0))
