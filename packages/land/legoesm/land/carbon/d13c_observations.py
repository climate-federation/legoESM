"""Per-archetype OBSERVED leaf carbon-isotope (leaf delta13C) target for Stage-B calibration.

The MODELLED side (:func:`legoesm.land.carbon.d13c_forward.simulate_archetype_d13c`) predicts
a per-archetype simulated LEAF delta13C [permil] from the coupled Farquhar-stomata ``Ci/Ca``;
the calibration ``d13c`` loss term compares it against an OBSERVED per-archetype leaf delta13C
derived from a GRIDDED leaf/ecosystem delta13C product, cover-weighted per archetype (mirroring
:mod:`legoesm.land.carbon.soc_observations` / :mod:`legoesm.land.carbon.sif_observations` /
:mod:`legoesm.land.carbon.biomass_observations`):

* per-cell observed leaf delta13C from a gridded NetCDF [permil], reshaped to the SAME
  row-major ``(ncell = nlat*nlon)`` order the cover loader uses so cell indices align with the
  archetype membership;
* the per-archetype target = the COVER-WEIGHTED mean of that per-cell delta13C over the grid
  cells assigned to the archetype, via the SHARED
  :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean` (one
  cover-weighted-mean definition, no duplicated numerics).

Pure NumPy: a FIXED target (built once, never inside a JAX-traced model step).

C4 MASKING (correctness).  The model photosynthesis is C3-only and the simulated forward
applies the C3 discrimination form; C4 plants discriminate FAR less (leaf delta13C ~
-11..-14 permil vs C3 -25..-32 permil).  :func:`c4_archetype_mask` flags the C4 archetypes
so the trainer EXCLUDES them from the delta13C term rather than fitting the C3 forward to a
C4 observation (a magnitude error).  A faithful C4 discrimination (Farquhar-Cerling with a
Collatz C4 ``Ci``) is a documented FOLLOW-UP.

REAL SOURCE / FOLLOW-UP.  A gridded leaf/ecosystem delta13C product is SPARSE: options are a
leaf-economics delta13C compilation regridded to a global field (e.g. the Cornwell et al.
2018 global leaf delta13C dataset) or a modelled/observed global delta13C-of-soil/ecosystem
map.  Fetching such a product, converting it to a LEAF delta13C on the model's convention
(ecosystem/soil delta13C carries an additional respiration/decomposition offset from leaf
delta13C), and regridding it onto the surfdata grid is a DATA-PREP FETCHER FOLLOW-UP -- exactly
like the SIF radiance->photon-flux and the ESA-CCI/GEDI AGB->carbon follow-ups.  This module
provides only the loader (a pre-regridded file supplied via ``--d13c-obs``) and, for a
self-contained test without a product, :func:`synthetic_observed_d13c`.
"""

from __future__ import annotations

import numpy as np
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean

from legoesm import constants

# --- synthetic (test-only) observed leaf-delta13C field [permil]; NOT a real product ---
# Physically-oriented: warmer/drier archetypes run a lower Ci/Ca (more water-stressed
# stomata), discriminate LESS, and so have a LESS NEGATIVE leaf delta13C.  Values stay in the
# physical C3 leaf band (~ -22..-34 permil) and are deliberately offset from the
# default-parameter simulated delta13C so the synthetic dry-run has a residual for the
# water-use-efficiency params to fit.
_D13C_SYNTH_BASE_PERMIL = -28.0   # baseline leaf delta13C at the freezing reference [permil]
_D13C_SYNTH_PER_K = 0.12          # less negative per degC of growing-season warmth [permil/K]
_D13C_SYNTH_MAX_WARM_K = 30.0     # cap the warmth term so the target stays in the C3 band [K]


def c4_archetype_mask(pft_id):
    """Per-archetype C4 flag ``(n_arch,)`` bool -- ``True`` where the archetype's PFT is C4.

    The model's Farquhar is C3-only, so the C3 discrimination forward is NOT valid for these
    archetypes (their C3-kinetics ``Ci`` is not a faithful C4 leaf state); the trainer uses
    this mask to EXCLUDE them from the ``d13c`` calibration term.  Uses the shared PFT-name
    classifier :func:`legoesm.land.surface_params.is_c4` (single source of truth).
    """
    from legoesm.land.surface_params import CLM5_PFT_NAMES, is_c4

    n_names = len(CLM5_PFT_NAMES)
    pid = np.asarray(pft_id, dtype=int).ravel()
    return np.array(
        [bool(0 <= int(p) < n_names and is_c4(CLM5_PFT_NAMES[int(p)])) for p in pid],
        dtype=bool,
    )


def per_archetype_observed_d13c(
    d13c_cell,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
):
    """Cover-weighted per-archetype observed leaf delta13C [permil].

    For archetype ``a`` the target is the cover-weighted mean of the per-cell observed leaf
    delta13C of every grid cell assigned to it::

        D13C_obs[a] = ( sum_{(c,p): id[c,p]==a} w[c,p] * d13c_cell[c] )
                      / ( sum_{(c,p): id[c,p]==a} w[c,p] )

    delta13C is a per-cell quantity (a gridded product does not resolve PFTs), so a cell's
    value is shared by every (cell, PFT) pair that maps to an archetype there.  Delegates to
    the SHARED :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean`
    (also used by the observed SOC / SIF / biomass targets -- no duplicated numerics).

    Parameters
    ----------
    d13c_cell : array (ncell,)
        Per-cell observed leaf delta13C [permil]; ``NaN`` where missing (excluded from the mean).
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent / below threshold / bare.
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT); ``0`` where the id is ``-1``.
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed leaf delta13C [permil].  ``NaN`` for an archetype with zero
        finite-value member cover -- surfaced, never silently 0.
    """
    return per_archetype_cover_weighted_mean(
        d13c_cell, cell_archetype_id, cell_archetype_weight, n_arch=n_arch)


def load_gridded_d13c(d13c_path, *, ncell, d13c_var=None):
    """Per-cell observed leaf delta13C ``(ncell,)`` [permil] from a gridded delta13C NetCDF.

    ``delta13C(..., nlat, nlon)`` is time-averaged (any leading axes collapsed via
    ``nanmean``) then reshaped to the SAME row-major ``(ncell = nlat*nlon)`` order the CLM5
    cover loader uses, so cell indices align with the archetype membership.  Missing cells are
    PRESERVED as ``NaN`` (never fabricated to 0), so the per-archetype cover-weighted mean
    averages only the observed cells and an unobserved archetype gets ``NaN`` (a leaf-delta13C
    product is SPARSE -- it is a compiled/interpolated field, gap-heavy off the sampled
    biomes).  The product MUST already be on the surfdata grid (same shape AND lat/lon
    orientation) and be a LEAF delta13C [permil] on the model convention -- regridding + the
    ecosystem/soil->leaf delta13C reconciliation is the data-prep FETCHER FOLLOW-UP (see the
    module docstring); a shape mismatch is a hard error, never a silent misalignment.

    Parameters
    ----------
    d13c_path : str
        Gridded leaf-delta13C NetCDF path (``--d13c-obs``).
    ncell : int
        Expected number of cells (= the cover ``ncell``); a mismatch raises.
    d13c_var : str, optional
        delta13C variable name; auto-detected from a small candidate list if omitted.
    """
    import xarray as xr

    ds = xr.open_dataset(d13c_path, decode_times=False)
    try:
        candidates = ("d13c", "D13C", "delta13C", "leaf_d13c", "d13C_leaf", "delta13c", "dc13")
        var = d13c_var or next((v for v in candidates if v in ds), None)
        if var is None:
            raise SystemExit(
                f"gridded delta13C {d13c_path} has no recognised delta13C variable "
                f"(looked for {candidates}); pass --d13c-var. Available: "
                f"{sorted(ds.data_vars)[:40]}")
        arr = np.asarray(ds[var].values, dtype=float)
    finally:
        ds.close()
    while arr.ndim > 2:
        with np.errstate(invalid="ignore"):
            arr = np.nanmean(arr, axis=0)
    if arr.ndim != 2:
        raise SystemExit(
            f"gridded delta13C variable expected 2-D (nlat, nlon) after time-averaging; "
            f"got shape {arr.shape}.")
    d13c_cell = np.asarray(arr, dtype=float).reshape(-1)   # row-major (i_lat, i_lon)
    if d13c_cell.shape[0] != int(ncell):
        raise SystemExit(
            f"gridded delta13C ncell {d13c_cell.shape[0]} (= {arr.shape[0]}x{arr.shape[1]}) "
            f"!= cover ncell {int(ncell)}; the delta13C product must be pre-regridded onto "
            f"the surfdata grid (same shape AND lat/lon orientation) -- a data-prep "
            f"follow-up. Use --dry-run-synthetic for a self-contained test.")
    return d13c_cell


def synthetic_observed_d13c(table):
    """Deterministic per-archetype observed leaf delta13C [permil] for ``--dry-run-synthetic``
    (no data files).

    Warmer/drier archetypes run a lower Ci/Ca (more water-stressed stomata), discriminate
    less, and so have a LESS NEGATIVE leaf delta13C, so the water-use-efficiency parameters
    have signal to fit.  Values stay in the physical C3 leaf band; NOT a real product -- see
    :func:`load_gridded_d13c` for the sparse leaf-delta13C source + the ecosystem->leaf
    reconciliation follow-up.
    """
    mat_c = np.asarray(table.mat_k, dtype=float) - constants.T_freeze
    warmth = np.clip(mat_c, 0.0, _D13C_SYNTH_MAX_WARM_K)
    return _D13C_SYNTH_BASE_PERMIL + _D13C_SYNTH_PER_K * warmth
