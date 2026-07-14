"""LUH2 / LUH3 land-use source: transient cover -> CLM5 17-PFT (producer side).

The Land-Use Harmonization 2 dataset (LUH2, Hurtt et al. 2020, *Geosci. Model
Dev.*, doi:10.5194/gmd-13-5425-2020; the LUH3 successor is variable-compatible)
distributes each grid cell's ice/water-free land over **12 land-use states**:

    primf secdf primn secdn   -- natural: primary/secondary forest & non-forest
    pastr range               -- managed grass: pasture & rangeland
    c3ann c4ann c3per c4per c3nfx  -- croplands (annual / perennial / N-fixing)
    urban                     -- built-up

Each state is a **fraction of the grid cell** in [0, 1]; the states plus the
static ice+water fraction sum to 1.  A run's *transient cover* is the annual
time series of these 12 fields.

Crosswalk to the model's 17-PFT axis (v1)
-----------------------------------------
LUH2 states carry the crop / pasture / natural **split** but NOT which
climate-zoned natural PFT grows on the natural fraction (needleleaf-boreal vs
broadleaf-tropical ...).  That assignment comes from a **potential-natural-
vegetation (PNV) shape** -- here reused from the model's existing CLM5 surfdata
natural-PFT distribution (no new dataset), exactly as CLM's ``mksurfdata`` uses a
reference PFT map.  The mapping to :data:`legoesm.land.surface_params.CLM5_PFT_NAMES`
(indices in brackets) is:

  - natural (primf+secdf+primn+secdn) -> the 15 natural PFTs [0..14], distributed
    by the (per-cell, renormalised) PNV natural shape; all-bare PNV -> bare_soil [0];
  - C3 crops (c3ann+c3per+c3nfx) -> crop_c3 [15];   C4 crops (c4ann+c4per) -> crop_c4 [16];
  - managed grass (pastr+range)  -> c3_grass [13] / c4_grass [14] by a per-cell
    C4-grass fraction (reused from the PNV grass C3:C4 ratio);
  - urban -> bare_soil [0] (the model has no urban landunit; documented v1 proxy).

The crosswalk is a pure **redistribution**: for every cell and year the output
17-PFT vector sums to the input 12-state total (area conserved to machine
precision) -- the correctness anchor in ``tests/land/surface_data/test_luh2.py``.

Deliberate v1 simplifications (documented, revisited later):
  - primary vs secondary and forest vs non-forest are NOT distinguished inside
    ``pft_frac`` (biophysics only needs the PFT distribution); that distinction is
    carried separately by the LUH2 *transition* fields for E_LUC bookkeeping.
  - lake / glacier fractions are taken from the (static) base surfdata, not from
    LUH2's combined ice+water field.

Host-side only (NumPy / xarray); NOT traced.  Ocean / non-land cells stay NaN so
the downstream NaN-aware regrid drops them cleanly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# 17-PFT axis indices + the PNV shape / C4-grass math are shared with the
# anthropogenic overlay (HYDE/Pongratz/KK10) — single source, no re-derivation.
from legoesm.land.surface_data.sources.anthropogenic import (
    PFT_IDX_BARE as _IDX_BARE,
    PFT_IDX_C3_GRASS as _IDX_C3_GRASS,
    PFT_IDX_C4_GRASS as _IDX_C4_GRASS,
    PFT_IDX_CROP_C3 as _IDX_CROP_C3,
    PFT_IDX_CROP_C4 as _IDX_CROP_C4,
    PFT_IDX_NATURAL as _IDX_NATURAL,
    c4_grass_fraction_from_base,
    normalised_pnv_shape,
)

# LUH2 state variable names, grouped by how they map to the 17-PFT axis.
LUH2_NATURAL_STATES = ("primf", "secdf", "primn", "secdn")
LUH2_C3_CROP_STATES = ("c3ann", "c3per", "c3nfx")
LUH2_C4_CROP_STATES = ("c4ann", "c4per")
LUH2_GRASS_STATES = ("pastr", "range")
LUH2_URBAN_STATE = "urban"
LUH2_STATE_NAMES = (
    LUH2_NATURAL_STATES
    + LUH2_GRASS_STATES
    + LUH2_C3_CROP_STATES
    + LUH2_C4_CROP_STATES
    + (LUH2_URBAN_STATE,)
)


class LUH2Config(NamedTuple):
    """Variable / coordinate names for a LUH2 (or LUH3) states NetCDF."""

    lat_var: str = "lat"
    lon_var: str = "lon"
    time_var: str = "time"
    # LUH2 encodes the calendar year as an offset from this base (states files
    # use ``years since 0850-01-01``); the reader adds it to the time axis.
    year_base: int = 850


def read_luh2_states(
    path: str | None = None,
    config: LUH2Config = LUH2Config(),
    *,
    years: tuple[int, int] | None = None,
    dataset=None,
) -> dict:
    """Read the 12 LUH2 land-use state time series from a states NetCDF.

    Parameters
    ----------
    path : file to open (ignored if ``dataset`` is given).
    config : variable / coordinate names.
    years : optional inclusive ``(start, end)`` calendar-year window to subset.
    dataset : an already-open ``xarray.Dataset`` (for testing without a file).

    Returns
    -------
    dict with ``lat`` / ``lon`` (1-D, deg), ``years`` (``(nyear,)`` int), and
    ``states`` -- a dict ``{state_name: (nyear, lat, lon) float64}`` for every
    name in :data:`LUH2_STATE_NAMES`.  Fractions are left in native [0, 1];
    non-land cells stay NaN.
    """
    import xarray as xr  # noqa: F401

    ds = dataset if dataset is not None else xr.open_dataset(path, decode_times=False)
    try:
        lat = np.asarray(ds[config.lat_var].values, dtype=np.float64)
        lon = np.asarray(ds[config.lon_var].values, dtype=np.float64)
        yr = np.asarray(ds[config.time_var].values) + config.year_base
        yr = np.rint(yr).astype(int)

        sel = np.arange(yr.size)
        if years is not None:
            lo, hi = years
            # Integer indexer, NOT a first..last slice: robust to a non-monotonic
            # time axis (a contiguous slice between two matches could span
            # out-of-window years).
            sel = np.nonzero((yr >= lo) & (yr <= hi))[0]
            if sel.size == 0:
                raise ValueError(
                    f"no LUH2 years in requested window {years}; "
                    f"file covers {int(yr.min())}-{int(yr.max())}.")
        # Emit years in increasing order — the runtime interp_annual (jnp.interp)
        # assumes a monotonic year axis; a non-monotonic source would misblend.
        sel = sel[np.argsort(yr[sel], kind="stable")]

        missing = [s for s in LUH2_STATE_NAMES if s not in ds]
        if missing:
            raise ValueError(
                f"LUH2 states file is missing required state variables: {missing}. "
                f"Expected all of {LUH2_STATE_NAMES}.")

        def _state(name):
            # Transpose by NAME to the documented (time, lat, lon): a file stored
            # (time, lon, lat) would otherwise be mislabelled — a silent
            # geographic transpose that passes unnoticed on square grids.
            da = ds[name].isel({config.time_var: sel}).transpose(
                config.time_var, config.lat_var, config.lon_var)
            return np.asarray(da.values, dtype=np.float64)

        states = {s: _state(s) for s in LUH2_STATE_NAMES}
        return {"lat": lat, "lon": lon, "years": yr[sel], "states": states}
    finally:
        if dataset is None:
            ds.close()


def luh2_states_to_pft_frac(
    states: dict,
    pnv_natural: np.ndarray,
    c4_grass_frac: np.ndarray,
) -> np.ndarray:
    """Redistribute LUH2 12-state cover onto the CLM5 17-PFT axis.

    Parameters
    ----------
    states : ``{name: (nyear, lat, lon)}`` LUH2 state fractions (native [0, 1]).
    pnv_natural : ``(>=15, lat, lon)`` potential-natural-vegetation PFT shape
        (only the 15 natural rows are used; per-cell magnitude is irrelevant).
    c4_grass_frac : ``(lat, lon)`` in [0, 1] -- fraction of managed grass routed
        to C4 (the rest goes to C3).

    Returns
    -------
    ``(nyear, 17, lat, lon)`` PFT cover fraction of grid cell, ``CLM5_PFT_NAMES``
    order.  Area-conserving: ``sum_pft out == sum_state in`` per cell and year.
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    nyear = next(iter(states.values())).shape[0]
    ny, nx = pnv_natural.shape[1], pnv_natural.shape[2]
    shape = normalised_pnv_shape(pnv_natural)                # (15, lat, lon)
    c4f = np.asarray(c4_grass_frac, dtype=np.float64)        # (lat, lon)

    def _sum(names):
        return sum(np.asarray(states[n], dtype=np.float64) for n in names)

    natural = _sum(LUH2_NATURAL_STATES)                      # (nyear, lat, lon)
    grass = _sum(LUH2_GRASS_STATES)
    c3_crop = _sum(LUH2_C3_CROP_STATES)
    c4_crop = _sum(LUH2_C4_CROP_STATES)
    urban = np.asarray(states[LUH2_URBAN_STATE], dtype=np.float64)

    out = np.zeros((nyear, N_PFT_CLM5, ny, nx), dtype=np.float64)
    # Natural fraction spread over the 15 natural PFTs by the PNV shape.
    out[:, _IDX_NATURAL, :, :] = natural[:, None, :, :] * shape[None, :, :, :]
    # Managed grass -> C3 / C4 grass PFTs.
    out[:, _IDX_C4_GRASS, :, :] += grass * c4f[None, :, :]
    out[:, _IDX_C3_GRASS, :, :] += grass * (1.0 - c4f[None, :, :])
    # Croplands -> the two crop PFTs.
    out[:, _IDX_CROP_C3, :, :] += c3_crop
    out[:, _IDX_CROP_C4, :, :] += c4_crop
    # Urban -> bare soil (no urban landunit; documented v1 proxy).
    out[:, _IDX_BARE, :, :] += urban
    return out


def build_transient_pft_frac(luh2: dict, base_pft_frac: np.ndarray) -> np.ndarray:
    """Convenience: LUH2 states + a base 17-PFT map -> transient ``pft_frac``.

    Derives the PNV natural shape (base natural rows) and the C4-grass fraction
    from ``base_pft_frac`` (a single-slice ``(17, lat, lon)`` base surfdata map,
    e.g. from :func:`legoesm.land.surface_data.sources.clm5_surfdata.read_clm5_cover_veg`)
    and returns ``(nyear, 17, lat, lon)`` transient cover.  The single call a
    LUH2 producer script makes.
    """
    c4f = c4_grass_fraction_from_base(base_pft_frac)
    return luh2_states_to_pft_frac(luh2["states"], np.asarray(base_pft_frac), c4f)
