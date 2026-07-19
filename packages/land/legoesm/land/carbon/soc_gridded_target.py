"""Per-archetype OBSERVED SOC target reduced from a GRIDDED SOC observation.

The Stage-B carbon calibration compares a per-archetype MODELLED equilibrium SOC
against a per-archetype OBSERVED SOC.  :mod:`legoesm.land.carbon.soc_observations`
builds that observed target from the raw CLM5 surfdata ``ORGANIC`` column (read on
the cover's native grid).  This module builds the SAME per-archetype target from an
already-GRIDDED per-cell SOC field ``soc(lat, lon)`` [kgC/m2] -- the exact NetCDF the
gridded validator :mod:`scripts.validate.carbon_soc_vs_gridded` scores the model map
against (built by ``scripts/data/build_soilgrids_soc.py``: the depth-matched CLM5
``surfdata_organic`` product, or the independent ISRIC ``soilgrids`` 2.0 product).

Why a second path.  Calibrating against the SAME gridded per-cell obs the validator
uses makes the tuning target and the validation target ONE field (no depth/unit/grid
drift between "what we fit" and "what we score"), AND it supports the SoilGrids
cross-check product, which -- unlike the CLM5 ``ORGANIC`` column -- has no per-layer
density to depth-integrate (it is delivered as a single 0-100 cm per-cell stock).

Reduction (identical cover-weighting to the organic-column path).  Each grid cell's
per-cell SOC is sampled onto the archetype build's cover cells by a nearest-centre
``(lat, lon)`` lookup (robust to the flatten order and to a longitude-frame offset),
then the shared cover-weighted-mean core
(:func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean`) averages
each archetype's member ``(cell, PFT)`` cover -- NO duplicated ``bincount`` numerics.  A
cell whose obs is NaN (ocean / missing) or whose nearest obs centre is farther than the
grid tolerance (off-grid) is EXCLUDED from BOTH the numerator and the denominator, so a
gap never contributes a fabricated 0.

Grid expectation.  The SUPPORTED path is the obs already on the cover grid (the gridded
validator builds its obs on the model IC grid -- the SAME grid the archetype cover comes
from), so the nearest ``(lat, lon)`` lookup is EXACT (distance ~0) and the tuning target
equals the per-cell values the validator scores against.  The off-grid tolerance is a GUARD
(an obs on a grossly different grid drops to NaN and the caller's "no finite obs" check
fires loudly), NOT a conservative regrid -- point the calibration's ``--soc-obs-nc`` at an
obs on the cover grid, as the validator's own products are.

Units convention: the input ``soc`` field is an organic-CARBON column stock [kgC/m2]
(the validator's schema; the ``surfdata_organic`` product already applies the van
Bemmelen organic-matter->carbon 0.58 factor and the depth integral, and SoilGrids is a
carbon stock).  The reduction is unit-preserving -- the per-archetype target is [kgC/m2],
directly comparable to the model ``som_total`` [gC/m2] / 1000 the calibration extracts.

Pure NumPy: a FIXED target built once (never inside a JAX-traced model step), mirroring
:mod:`legoesm.land.carbon.soc_observations`.
"""

from __future__ import annotations

import numpy as np

from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean

# Longitude period [deg] for the circular nearest-centre match (a grid-geometry
# constant, not a physical constant -- the full lon span, like the model grid's own
# 360-deg wrap).  Lets a cover cell at the 0/360 seam match the correct obs column
# even when the obs field and the cover use different lon frames (0..360 vs -180..180).
_LON_PERIOD_DEG = 360.0
# Default off-grid tolerance as a FRACTION of the median grid spacing: a cover cell
# whose nearest obs centre is farther than ``_TOL_SPACING_FRAC`` x (median centre
# spacing) is treated as OFF the obs grid -> NaN (excluded).  0.6 > 0.5 admits an
# exact cell-centre match (distance 0) and a half-cell staggered match, but rejects a
# cell that lands a full cell or more away (a genuine grid mismatch).
_TOL_SPACING_FRAC = 0.6


def load_gridded_soc(path, *, soc_var="soc"):
    """Load a gridded SOC obs ``(soc2d, lat, lon)`` from a NetCDF.

    ``soc2d`` is ``(nlat, nlon)`` [kgC/m2]; ``lat``/``lon`` are the 1-D centres [deg].
    Matches the schema written by ``scripts/data/build_soilgrids_soc.py``
    (dims ``(lat, lon)``, variable ``soc`` [kgC/m2]) and read by the gridded validator
    (:func:`scripts.validate.carbon_soc_vs_gridded.align_obs`).

    Deferred ``xarray`` import (as in the surfdata reader) so importing this module
    stays dependency-light.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        if soc_var not in ds:
            raise ValueError(
                f"gridded SOC obs {path} has no {soc_var!r} variable "
                f"(available: {sorted(ds.data_vars)}).")
        da = ds[soc_var]
        # Enforce (lat, lon) axis ORDER by name -- a shape check alone silently
        # transposes a square soc(lon, lat) field into the wrong geography.
        if "lat" in da.dims and "lon" in da.dims:
            da = da.transpose("lat", "lon")
        soc = np.asarray(da.values, dtype=float)
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
    finally:
        ds.close()
    if soc.ndim != 2:
        raise ValueError(
            f"gridded SOC {soc_var!r} must be 2-D (lat, lon); got shape {soc.shape}.")
    if soc.shape != (lat.size, lon.size):
        raise ValueError(
            f"gridded SOC {soc.shape} != (lat={lat.size}, lon={lon.size}).")
    return soc, lat, lon


def _nearest_index(centres, query, *, period=None):
    """Nearest index into ASCENDING ``centres`` for each ``query`` + its distance.

    Returns ``(idx, dist)``.  ``period`` (e.g. 360 deg for longitude) makes the match
    CIRCULAR: distances wrap, and the first/last centre are candidates for a query at
    the seam, so a query just past ``centres[-1]`` snaps back to ``centres[0]``.
    """
    c = np.asarray(centres, dtype=float)
    q = np.asarray(query, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError(f"centres must be a non-empty 1-D array; got shape {c.shape}.")
    pos = np.searchsorted(c, q)
    idx_hi = np.clip(pos, 0, c.size - 1)
    idx_lo = np.clip(pos - 1, 0, c.size - 1)
    d_hi = np.abs(q - c[idx_hi])
    d_lo = np.abs(q - c[idx_lo])
    if period is not None:
        d_hi = np.minimum(d_hi, period - d_hi)
        d_lo = np.minimum(d_lo, period - d_lo)
    take_lo = d_lo <= d_hi
    idx = np.where(take_lo, idx_lo, idx_hi)
    dist = np.where(take_lo, d_lo, d_hi)
    if period is not None:
        # Circular seam: the true nearest for a query near the wrap point can be the
        # first or last centre, which the interior searchsorted bracket misses.
        for cand_idx in (0, c.size - 1):
            d_cand = np.abs(q - c[cand_idx])
            d_cand = np.minimum(d_cand, period - d_cand)
            better = d_cand < dist
            idx = np.where(better, cand_idx, idx)
            dist = np.where(better, d_cand, dist)
    return idx.astype(int), dist


def _ascending(coord, axis_len, name):
    """Return ``(coord_sorted_ascending, order)`` so a field axis can be reordered.

    A gridded obs may store latitude north->south (descending); the nearest-centre
    lookup needs ascending centres.  ``order`` indexes the ORIGINAL field axis so the
    caller can reorder the field to match the ascending centres.
    """
    coord = np.asarray(coord, dtype=float)
    if coord.size != axis_len:
        raise ValueError(
            f"{name} size {coord.size} != field axis {axis_len}.")
    order = np.argsort(coord, kind="stable")
    return coord[order], order


def sample_grid_to_cells(field2d, obs_lat, obs_lon, cell_lat, cell_lon, *,
                         tol_deg=None):
    """Sample a gridded ``field2d(lat, lon)`` onto ``(ncell,)`` cover cells.

    For each cover cell ``c`` the value is ``field2d`` at the obs grid cell whose centre
    is nearest ``(cell_lat[c], cell_lon[c])`` -- a lookup by COORDINATE, so it is robust
    to the cover's flatten order and to a longitude-frame offset (0..360 vs -180..180;
    matched circularly with period :data:`_LON_PERIOD_DEG`).  A cell whose nearest obs
    centre is farther than ``tol_deg`` in latitude OR longitude is OFF the obs grid and
    returns ``NaN`` (excluded downstream); a cell landing on a ``NaN`` obs value (ocean /
    missing) stays ``NaN``.

    ``tol_deg`` defaults to :data:`_TOL_SPACING_FRAC` x the median centre spacing of each
    axis (so an exact same-grid match -- distance 0 -- always passes).

    Parameters
    ----------
    field2d : array (nlat, nlon)
        Gridded per-cell field [kgC/m2] (obs SOC).
    obs_lat, obs_lon : array (nlat,), (nlon,)
        Field centres [deg]; either ascending or descending (sorted internally).
    cell_lat, cell_lon : array (ncell,)
        Cover-cell centres [deg] (from the archetype-build inputs).
    tol_deg : float, optional
        Off-grid tolerance [deg]; default derived from the obs grid spacing.

    Returns
    -------
    array (ncell,)
        Per-cell sampled value [kgC/m2]; ``NaN`` where off-grid or missing.
    """
    field = np.asarray(field2d, dtype=float)
    if field.ndim != 2:
        raise ValueError(f"field2d must be 2-D (nlat, nlon); got shape {field.shape}.")
    nlat, nlon = field.shape
    lat_asc, lat_order = _ascending(obs_lat, nlat, "obs_lat")
    lon_asc, lon_order = _ascending(obs_lon, nlon, "obs_lon")
    field = field[np.ix_(lat_order, lon_order)]     # rows/cols now ascending-aligned

    clat = np.asarray(cell_lat, dtype=float).ravel()
    clon = np.asarray(cell_lon, dtype=float).ravel()
    if clat.shape != clon.shape:
        raise ValueError(
            f"cell_lat {clat.shape} and cell_lon {clon.shape} must have the same "
            f"(ncell,) shape.")
    # Bring cell longitudes into the obs longitude frame [lon0, lon0+360) so the
    # ascending searchsorted brackets them; the circular match then handles the seam.
    lon0 = float(lon_asc[0])
    clon_in_frame = lon0 + np.mod(clon - lon0, _LON_PERIOD_DEG)

    ilat, dlat = _nearest_index(lat_asc, clat)
    ilon, dlon = _nearest_index(lon_asc, clon_in_frame, period=_LON_PERIOD_DEG)

    if tol_deg is None:
        # A singleton axis has NO derivable spacing, so a default tolerance would be
        # meaningless (every query would snap to the sole row/column).  Require an
        # explicit tol_deg for a degenerate grid rather than silently mis-assign.
        if lat_asc.size < 2 or lon_asc.size < 2:
            raise ValueError(
                f"sample_grid_to_cells: obs grid ({lat_asc.size} lat x {lon_asc.size} "
                f"lon) has a singleton axis with no derivable off-grid tolerance; "
                f"pass an explicit tol_deg.")
        lat_tol = _TOL_SPACING_FRAC * _median_spacing(lat_asc)
        lon_tol = _TOL_SPACING_FRAC * _median_spacing(lon_asc)
    else:
        lat_tol = lon_tol = float(tol_deg)
    on_grid = (dlat <= lat_tol) & (dlon <= lon_tol)

    vals = field[ilat, ilon]
    return np.where(on_grid, vals, np.nan)


def _median_spacing(centres):
    """Median absolute spacing of 1-D ``centres`` [deg] (>= a tiny floor)."""
    c = np.asarray(centres, dtype=float)
    if c.size < 2:
        return _LON_PERIOD_DEG                     # a single centre spans the axis
    return float(np.median(np.abs(np.diff(c))))


def per_archetype_soc_from_grid(
    field2d,
    obs_lat,
    obs_lon,
    cell_lat,
    cell_lon,
    cell_archetype_id,
    cell_archetype_weight,
    *,
    n_arch=None,
    tol_deg=None,
):
    """Cover-weighted per-archetype observed SOC [kgC/m2] from a gridded obs field.

    Samples the gridded per-cell SOC onto the cover cells
    (:func:`sample_grid_to_cells`), then cover-weight-averages over each archetype's
    member ``(cell, PFT)`` pairs via the SHARED
    :func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weighted_mean`
    (the SAME reduction the organic-column target uses, so the two paths agree cell for
    cell on the same grid).  ``NaN`` obs cells (ocean / off-grid) are excluded from the
    per-archetype mean; an archetype with no finite-obs member cover returns ``NaN``.

    Parameters
    ----------
    field2d : array (nlat, nlon)
        Gridded per-cell SOC obs [kgC/m2].
    obs_lat, obs_lon : array (nlat,), (nlon,)
        Obs grid centres [deg].
    cell_lat, cell_lon : array (ncell,)
        Cover-cell centres [deg] (``GlobalCarbonInputs.cell_lat_deg`` /
        ``cell_lon_deg``), aligning the obs grid to the archetype membership.
    cell_archetype_id : array (ncell, npft) int
        Archetype index per (cell, PFT); ``-1`` where absent (from
        :func:`legoesm.land.carbon.global_init.build_archetypes`).
    cell_archetype_weight : array (ncell, npft) float
        PFT cover weight per (cell, PFT).
    n_arch : int, optional
        Number of archetypes.  Defaults to ``cell_archetype_id.max() + 1``.
    tol_deg : float, optional
        Off-grid tolerance [deg]; default from the obs grid spacing.

    Returns
    -------
    array (n_arch,)
        Per-archetype observed SOC [kgC/m2]; ``NaN`` for an archetype with no
        finite-obs member cover -- surfaced, never silently 0.
    """
    cid = np.asarray(cell_archetype_id)
    if cid.ndim != 2:
        raise ValueError(
            f"cell_archetype_id must be 2-D (ncell, npft); got shape {cid.shape}.")
    cell_soc = sample_grid_to_cells(
        field2d, obs_lat, obs_lon, cell_lat, cell_lon, tol_deg=tol_deg)
    if cell_soc.shape[0] != cid.shape[0]:
        raise ValueError(
            f"sampled ncell {cell_soc.shape[0]} != membership ncell {cid.shape[0]}.")
    return per_archetype_cover_weighted_mean(
        cell_soc, cell_archetype_id, cell_archetype_weight, n_arch=n_arch)
