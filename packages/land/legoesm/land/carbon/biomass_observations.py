"""Per-archetype OBSERVED live-biomass target for Stage-B carbon calibration.

The MODELLED side (:func:`legoesm.land.carbon.live_pool_forward.compute_biomass_lai`)
predicts a per-archetype simulated live biomass ``(C_fol + C_root + C_wood)/1000``
[kgC/m2]; the calibration ``biomass`` loss term compares it against an OBSERVED
per-archetype biomass derived from a GRIDDED biomass product, cover-weighted per archetype
(mirroring :mod:`legoesm.land.carbon.soc_observations` /
:mod:`legoesm.land.carbon.sif_observations`):

* per-cell observed biomass from a gridded NetCDF [kgC/m2], reshaped to the SAME row-major
  ``(ncell = nlat*nlon)`` order the cover loader uses so cell indices align with the
  archetype membership;
* the per-archetype target = the COVER-WEIGHTED mean of that per-cell biomass over the grid
  cells assigned to the archetype, via the SHARED
  :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` (one
  cover-weighted-mean definition, no duplicated numerics).

Pure NumPy: a FIXED target (built once, never inside a JAX-traced model step).

REAL SOURCE / FOLLOW-UP: a gridded above-ground-biomass product (ESA-CCI Biomass, or NASA
GEDI L4B AGB) reports ABOVE-GROUND biomass [Mg/ha of dry matter or carbon] and must be (a)
converted to CARBON [kgC/m2] (~0.47-0.5 gC/g dry matter) and (b) reconciled with the
model's TOTAL live biomass, which INCLUDES roots (``C_root``): either add a root-shoot
adjustment to the observation, or compare against the above-ground pools ``(C_fol +
C_wood)`` only.  Regridding onto the surfdata grid + that AGB->carbon / above-ground->total
conversion is a DATA-PREP FETCHER FOLLOW-UP -- exactly like the SIF radiance->photon-flux
follow-up and the ERA5 climatology builder that feeds the SOC path; this module provides
only the loader (a pre-regridded file the user supplies via ``--biomass-obs``) and, for a
self-contained test without a product, :func:`synthetic_observed_biomass`.
"""

from __future__ import annotations

import numpy as np
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean

from legoesm import constants

# --- synthetic (test-only) observed-biomass field [kgC/m2]; NOT a real product ---
# Physically-oriented offsets so brighter/warmer archetypes carry more standing biomass (the
# allocation/residence params then have signal to fit); a total-live-biomass scale (O(1-10)
# kgC/m2) deliberately offset from the default-parameter simulated biomass so the synthetic
# dry-run has a clear residual.
_BIO_SYNTH_BASE_KG = 1.0          # baseline live biomass [kgC/m2]
_BIO_SYNTH_PER_W = 0.02           # per growing-season mean shortwave [kgC/m2 / (W m-2)]
_BIO_SYNTH_PER_K = 0.10           # per degree above freezing [kgC/m2 / K]


def per_archetype_observed_biomass(
    biomass_cell,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Cover-weighted per-archetype observed biomass [kgC/m2].

    For archetype ``a`` the target is the cover-weighted mean of the per-cell observed
    biomass of every grid cell assigned to it::

        BIO_obs[a] = ( sum_{(c,p): id[c,p]==a} w[c,p] * biomass_cell[c] )
                     / ( sum_{(c,p): id[c,p]==a} w[c,p] )

    Biomass is a per-cell quantity (a gridded product does not resolve PFTs), so a cell's
    value is shared by every (cell, PFT) pair that maps to an archetype there.  Delegates to
    the SHARED :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean`
    (also used by the observed SOC / SIF / LAI targets -- no duplicated numerics).

    Parameters
    ----------
    biomass_cell : array (ncell,)
        Per-cell observed biomass [kgC/m2]; ``NaN`` where missing (excluded from the mean).
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent / below threshold / bare.
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed biomass [kgC/m2].  ``NaN`` for an archetype with zero
        finite-value member cover -- surfaced, never silently 0.
    """
    return per_archetype_cover_weighted_mean(
        biomass_cell, cell_archetype_id, cell_archetype_weight, n_arch=n_arch)


def load_gridded_biomass(biomass_path, *, ncell, biomass_var=None):
    """Per-cell observed biomass ``(ncell,)`` [kgC/m2] from a gridded biomass NetCDF.

    ``biomass(..., nlat, nlon)`` is time-averaged (any leading axes collapsed via
    ``nanmean``) then reshaped to the SAME row-major ``(ncell = nlat*nlon)`` order the CLM5
    cover loader uses, so cell indices align with the archetype membership.  Missing cells
    are PRESERVED as ``NaN`` (never fabricated to 0), so the per-archetype cover-weighted
    mean averages only the observed cells and an unobserved archetype gets ``NaN`` (a
    biomass product is gap-heavy over non-forest / high-latitude / ocean).  The product MUST
    already be on the surfdata grid (same shape AND lat/lon orientation) and in carbon units
    [kgC/m2] -- regridding + the AGB->carbon / above-ground->total conversion is the
    data-prep FETCHER FOLLOW-UP (see the module docstring); a shape mismatch is a hard
    error, never a silent misalignment.

    Parameters
    ----------
    biomass_path : str
        Gridded biomass NetCDF path (``--biomass-obs``).
    ncell : int
        Expected number of cells (= the cover ``ncell``); a mismatch raises.
    biomass_var : str, optional
        Biomass variable name; auto-detected from a small candidate list if omitted.
    """
    import xarray as xr

    ds = xr.open_dataset(biomass_path, decode_times=False)
    try:
        candidates = ("biomass", "BIOMASS", "agb", "AGB", "biomass_kgC_m2", "cveg")
        var = biomass_var or next((v for v in candidates if v in ds), None)
        if var is None:
            raise SystemExit(
                f"gridded biomass {biomass_path} has no recognised biomass variable "
                f"(looked for {candidates}); pass --biomass-var. Available: "
                f"{sorted(ds.data_vars)[:40]}")
        arr = np.asarray(ds[var].values, dtype=float)
    finally:
        ds.close()
    while arr.ndim > 2:
        with np.errstate(invalid="ignore"):
            arr = np.nanmean(arr, axis=0)
    if arr.ndim != 2:
        raise SystemExit(
            f"gridded biomass variable expected 2-D (nlat, nlon) after time-averaging; "
            f"got shape {arr.shape}.")
    biomass_cell = np.asarray(arr, dtype=float).reshape(-1)   # row-major (i_lat, i_lon)
    if biomass_cell.shape[0] != int(ncell):
        raise SystemExit(
            f"gridded biomass ncell {biomass_cell.shape[0]} (= {arr.shape[0]}x{arr.shape[1]}) "
            f"!= cover ncell {int(ncell)}; the biomass product must be pre-regridded onto "
            f"the surfdata grid (same shape AND lat/lon orientation) -- a data-prep "
            f"follow-up. Use --dry-run-synthetic for a self-contained test.")
    return biomass_cell


def synthetic_observed_biomass(table):
    """Deterministic per-archetype observed biomass [kgC/m2] for ``--dry-run-synthetic``
    (no data files).

    Biomass rises with growing-season shortwave and warmth (brighter/warmer archetypes
    accumulate more standing biomass), so the allocation / residence parameters have signal
    to fit.  NOT a real product -- see :func:`load_gridded_biomass` for the AGB source + the
    AGB->carbon / above-ground->total conversion follow-up.
    """
    sw = np.asarray(table.sw_mean_w, dtype=float)
    mat_c = np.asarray(table.mat_k, dtype=float) - constants.T_freeze
    return (_BIO_SYNTH_BASE_KG
            + _BIO_SYNTH_PER_W * np.maximum(sw, 0.0)
            + _BIO_SYNTH_PER_K * np.maximum(mat_c, 0.0))
