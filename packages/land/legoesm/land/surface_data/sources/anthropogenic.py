"""Shared anthropogenic-cover -> CLM5 17-PFT overlay (producer side).

Unlike LUH2 (which resolves the full state breakdown), the HYDE, Pongratz and
KK10 reconstructions provide only the **anthropogenic** land fractions (cropland,
pasture, and — for HYDE — built-up).  The natural-vegetation backdrop is supplied
by a potential-natural-vegetation (PNV) shape, reused from the model's existing
CLM5 base natural-PFT distribution (no new dataset), exactly as for LUH2.

This module is the crosswalk those three sources share: given per-cell cropland /
pasture / urban fractions and a PNV shape, it produces the 17-PFT cover
(``CLM5_PFT_NAMES`` order) by

  - natural (the residual, ``1 - crop - pasture - urban``) -> the 15 natural PFTs
    by the renormalised PNV shape;
  - cropland -> crop_c3 / crop_c4 by a per-cell C4-crop fraction;
  - pasture  -> c3_grass / c4_grass by a per-cell C4-grass fraction;
  - urban    -> bare_soil (the model has no urban landunit).

Anthropogenic fractions that (from noisy source data) exceed 1 are scaled down
proportionally so the per-cell 17-PFT total is exactly 1 (area conserved) — the
test anchor.  The PNV helpers here are also reused by
:mod:`legoesm.land.surface_data.sources.luh2` (single source of the shape math).

Host-side only (NumPy); NOT traced.  Fractions are of the land area (the producer
scales by the base land fraction, as for LUH2).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# CLM5 17-PFT axis indices this crosswalk writes into (subscripts, not coeffs).
PFT_IDX_BARE = 0
PFT_IDX_NATURAL = slice(0, 15)     # 15 natural PFTs (bare + trees/shrubs/grass)
PFT_IDX_C3_GRASS = 13
PFT_IDX_C4_GRASS = 14
PFT_IDX_CROP_C3 = 15
PFT_IDX_CROP_C4 = 16


def normalised_pnv_shape(pnv_natural: np.ndarray) -> np.ndarray:
    """Per-cell natural-PFT shape ``(15, lat, lon)`` summing to 1, bare where empty.

    ``pnv_natural`` is a base natural-PFT distribution (its first 15 PFT rows);
    its per-cell sum is arbitrary (a *shape*).  A NaN band at a valid cell
    contributes zero weight (never poisons the column).  Cells whose natural
    shape is empty (desert / no reference vegetation) fall back to pure bare soil
    so any natural fraction there lands on a valid PFT row instead of vanishing.
    """
    shape = np.asarray(pnv_natural, dtype=np.float64)[:15]
    shape = np.where(np.isfinite(shape), shape, 0.0)
    total = shape.sum(axis=0, keepdims=True)                 # (1, lat, lon)
    empty = total <= 0.0
    norm = np.where(empty, 0.0, np.divide(shape, np.where(empty, 1.0, total)))
    norm[PFT_IDX_BARE] = np.where(empty[0], 1.0, norm[PFT_IDX_BARE])
    return norm


def c4_grass_fraction_from_base(base_pft_frac: np.ndarray) -> np.ndarray:
    """Per-cell C4 fraction of grass from a base 17-PFT distribution.

    ``c4_grass / (c3_grass + c4_grass)``; cells with no grass default to all-C3.
    """
    base = np.asarray(base_pft_frac, dtype=np.float64)
    c3g, c4g = base[PFT_IDX_C3_GRASS], base[PFT_IDX_C4_GRASS]
    denom = c3g + c4g
    return np.where(denom > 0.0, np.divide(c4g, np.where(denom > 0.0, denom, 1.0)), 0.0)


def c4_crop_fraction_from_base(base_pft_frac: np.ndarray) -> np.ndarray:
    """Per-cell C4 fraction of cropland from a base 17-PFT distribution.

    ``crop_c4 / (crop_c3 + crop_c4)``; cells with no crop default to all-C3.
    """
    base = np.asarray(base_pft_frac, dtype=np.float64)
    c3, c4 = base[PFT_IDX_CROP_C3], base[PFT_IDX_CROP_C4]
    denom = c3 + c4
    return np.where(denom > 0.0, np.divide(c4, np.where(denom > 0.0, denom, 1.0)), 0.0)


def anthropogenic_to_pft_frac(
    crop: np.ndarray,
    pasture: np.ndarray,
    urban: np.ndarray,
    pnv_natural: np.ndarray,
    c4_grass_frac: np.ndarray,
    c4_crop_frac: np.ndarray,
    land_frac: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Overlay anthropogenic cover on a PNV backdrop -> ``(nyear, 17, lat, lon)``.

    Parameters
    ----------
    crop, pasture, urban : ``(nyear, lat, lon)`` fractions of the GRID CELL [0, 1]
        (the same convention as ``land_frac``).  Non-finite or negative values are
        treated as zero (missing / invalid source data).
    pnv_natural : ``(>=15, lat, lon)`` PNV natural-PFT shape.
    c4_grass_frac, c4_crop_frac : ``(lat, lon)`` in [0, 1].
    land_frac : scalar or ``(lat, lon)`` grid-cell land fraction [0, 1] -- the
        cover *budget*.  The natural PFTs fill ``land_frac - anthropogenic``; if the
        anthropogenic total exceeds ``land_frac`` it is scaled down to fit (nothing
        is created).  Default 1.0 = whole cell is land.

    Returns
    -------
    ``(nyear, 17, lat, lon)`` PFT cover fraction of the GRID CELL, ``CLM5_PFT_NAMES``
    order; sums to ``land_frac`` per cell and year (so ``f_land = sum_pft`` holds and
    the ocean / lake / glacier mask carried by ``land_frac`` is preserved).
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    def _clean(a):
        # Missing (NaN, from a coastal regrid) or negative source data -> 0 so it
        # can never poison the conserved column or make a PFT fraction negative.
        a = np.asarray(a, dtype=np.float64)
        return np.clip(np.where(np.isfinite(a), a, 0.0), 0.0, None)

    crop, pasture, urban = _clean(crop), _clean(pasture), _clean(urban)
    nyear, ny, nx = crop.shape
    shape = normalised_pnv_shape(pnv_natural)                # (15, lat, lon)
    c4g = np.asarray(c4_grass_frac, dtype=np.float64)
    c4c = np.asarray(c4_crop_frac, dtype=np.float64)
    lf = np.broadcast_to(np.asarray(land_frac, dtype=np.float64), crop.shape)

    # Scale the anthropogenic total down to the land budget (noisy sources can
    # exceed it), scaling the three fractions together so nothing is created;
    # natural is the residual.  Per-cell 17-PFT total is then exactly land_frac.
    anthro = crop + pasture + urban
    over = anthro > lf
    scale = np.where(over, np.divide(lf, np.where(over, anthro, 1.0)), 1.0)
    crop = crop * scale
    pasture = pasture * scale
    urban = urban * scale
    natural = lf - (crop + pasture + urban)

    out = np.zeros((nyear, N_PFT_CLM5, ny, nx), dtype=np.float64)
    out[:, PFT_IDX_NATURAL, :, :] = natural[:, None, :, :] * shape[None, :, :, :]
    out[:, PFT_IDX_CROP_C4, :, :] += crop * c4c[None, :, :]
    out[:, PFT_IDX_CROP_C3, :, :] += crop * (1.0 - c4c[None, :, :])
    out[:, PFT_IDX_C4_GRASS, :, :] += pasture * c4g[None, :, :]
    out[:, PFT_IDX_C3_GRASS, :, :] += pasture * (1.0 - c4g[None, :, :])
    out[:, PFT_IDX_BARE, :, :] += urban
    return out


def build_anthropogenic_pft_frac(
    anthro: dict,
    base_pft_frac: np.ndarray,
    land_frac: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Convenience: anthropogenic fractions + a base 17-PFT map -> ``pft_frac``.

    ``anthro`` carries ``crop`` / ``pasture`` / ``urban`` each ``(nyear, lat, lon)``
    grid-cell fractions (missing keys default to zero).  Derives the PNV natural
    shape and the C4 grass/crop fractions from ``base_pft_frac`` (a single-slice
    ``(17, lat, lon)`` base map) and returns ``(nyear, 17, lat, lon)`` transient
    cover summing to ``land_frac`` per cell — the single call a HYDE / Pongratz /
    KK10 producer makes.  ``land_frac`` (scalar or ``(lat, lon)``) is the grid-cell
    land budget the natural PFTs fill around the anthropogenic cover.
    """
    base = np.asarray(base_pft_frac, dtype=np.float64)
    crop = anthro["crop"]
    zero = np.zeros_like(np.asarray(crop, dtype=np.float64))
    pasture = anthro.get("pasture", zero)
    urban = anthro.get("urban", zero)
    return anthropogenic_to_pft_frac(
        crop, pasture, urban, base,
        c4_grass_fraction_from_base(base), c4_crop_fraction_from_base(base),
        land_frac=land_frac)


class AnthropogenicSourceConfig(NamedTuple):
    """Variable / coordinate names for an anthropogenic-cover NetCDF.

    HYDE, Pongratz and KK10 all store the same shape of data (annual cropland /
    pasture / built-up on a lat-lon grid) under different variable names, and
    some store absolute areas (km²) rather than fractions.  One reader
    (:func:`read_anthropogenic_states`) covers all three; the differences are
    just this config.  Each of ``*_vars`` is summed into one land-fraction field
    (e.g. HYDE pasture = ``pasture`` + ``rangeland``).
    """

    crop_vars: tuple[str, ...]
    pasture_vars: tuple[str, ...] = ()
    urban_vars: tuple[str, ...] = ()
    lat_var: str = "lat"
    lon_var: str = "lon"
    time_var: str = "time"
    # Added to the raw time axis to recover the calendar year (0 if the file
    # already stores calendar years).
    year_base: int = 0
    # If set, every cover field is divided by this per-cell area variable to
    # convert an absolute area (km²) into a fraction of the grid cell (HYDE).
    area_var: str | None = None


def read_anthropogenic_states(
    path: str | None = None,
    config: AnthropogenicSourceConfig = None,
    *,
    years: tuple[int, int] | None = None,
    dataset=None,
) -> dict:
    """Read cropland / pasture / built-up time series from a NetCDF.

    Returns a dict with ``lat`` / ``lon`` (1-D deg), ``years`` (``(nyear,)`` int)
    and ``crop`` / ``pasture`` / ``urban`` -- each ``(nyear, lat, lon)`` fraction
    of the grid cell in [0, 1] (absolute-area files are divided by
    ``config.area_var``).  Missing cover groups (empty ``*_vars``) come back as
    zeros.  Non-land cells stay NaN for the downstream NaN-aware regrid.
    """
    import xarray as xr  # noqa: F401

    if config is None:
        raise ValueError("read_anthropogenic_states requires an AnthropogenicSourceConfig")

    ds = dataset if dataset is not None else xr.open_dataset(path, decode_times=False)
    try:
        lat = np.asarray(ds[config.lat_var].values, dtype=np.float64)
        lon = np.asarray(ds[config.lon_var].values, dtype=np.float64)
        yr = np.rint(np.asarray(ds[config.time_var].values) + config.year_base).astype(int)

        sel = np.arange(yr.size)
        if years is not None:
            lo, hi = years
            # Integer indexer (not a first..last slice): robust to non-monotonic
            # or non-contiguous time axes, matching the LUH2 reader.
            sel = np.nonzero((yr >= lo) & (yr <= hi))[0]
            if sel.size == 0:
                raise ValueError(
                    f"no years in requested window {years}; "
                    f"file covers {int(yr.min())}-{int(yr.max())}.")
        # Emit years in increasing order — the runtime interp_annual (jnp.interp)
        # assumes a monotonic year axis; a non-monotonic source would misblend.
        sel = sel[np.argsort(yr[sel], kind="stable")]

        all_vars = config.crop_vars + config.pasture_vars + config.urban_vars
        missing = [v for v in all_vars if v not in ds]
        if missing:
            raise ValueError(
                f"anthropogenic states file is missing variables: {missing}. "
                f"Expected all of {all_vars}.")

        area = None
        if config.area_var is not None:
            if config.area_var not in ds:
                raise ValueError(
                    f"area_var {config.area_var!r} not in file; needed to convert "
                    f"absolute areas to fractions.")
            # (lat, lon) cell area, broadcast against (nyear, lat, lon) below.
            area = np.asarray(
                ds[config.area_var].transpose(config.lat_var, config.lon_var).values,
                dtype=np.float64)

        def _group(names):
            if not names:
                return np.zeros((sel.size, lat.size, lon.size), dtype=np.float64)
            total = None
            for name in names:
                da = ds[name].isel({config.time_var: sel}).transpose(
                    config.time_var, config.lat_var, config.lon_var)
                arr = np.asarray(da.values, dtype=np.float64)
                total = arr if total is None else total + arr
            if area is not None:
                # km² -> fraction; guard zero-area cells (poles/fill) to NaN.
                total = np.divide(total, area[None, :, :],
                                  out=np.full_like(total, np.nan),
                                  where=area[None, :, :] > 0.0)
            return total

        return {
            "lat": lat,
            "lon": lon,
            "years": yr[sel],
            "crop": _group(config.crop_vars),
            "pasture": _group(config.pasture_vars),
            "urban": _group(config.urban_vars),
        }
    finally:
        if dataset is None:
            ds.close()
