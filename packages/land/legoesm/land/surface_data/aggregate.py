"""Area-weighted fine→coarse block aggregation for regular lat-lon rasters.

Primary surface rasters (e.g. HWSD2 at 30 arc-second) are far finer than any
legoESM grid, so the producer first **upscales** them to a manageable regular
lat-lon target (default 0.25 deg) by ``cos(lat)`` area-weighted averaging over
the integer block of fine pixels inside each coarse cell, ignoring NODATA.  The
coarse file is then regridded to the actual (cubed-sphere / Gaussian / MPAS)
model grid by the runtime loader.

Two layers:
  - :func:`area_weighted_block_mean` — the pure reduction, unit-tested on small
    in-memory arrays.  Supports an optional trailing layer axis (soil depths).
  - :func:`aggregate_raster_streaming` — reads a (possibly 1.9 GB) BIL memmap in
    coarse-row chunks, applies a caller ``transform`` (raw pixels → physical
    values + validity), and assembles the coarse field without ever holding the
    whole fine raster in memory.

Host-side only; not differentiable, never traced.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from legoesm.land.surface_data.raster import EnviBilHeader, cell_centers

# Floor to avoid 0/0 in fully-NODATA blocks (result is masked to NaN anyway).
_TINY = 1.0e-30


def row_cos_weights(lat_deg: np.ndarray) -> np.ndarray:
    """``cos(lat)`` area weight per raster row, clamped non-negative.

    Longitude cells of a regular grid have equal area at fixed latitude, so the
    only latitude dependence of a pixel's area is ``cos(lat)``.
    """
    return np.maximum(np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64))), 0.0)


def area_weighted_block_mean(
    values: np.ndarray,        # (Hf, Wf) or (Hf, Wf, L)
    valid: np.ndarray,         # (Hf, Wf) or (Hf, Wf, L) bool
    row_weights: np.ndarray,   # (Hf,)
    by: int,
    bx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-reduce a fine field to ``(ny, nx[, L])`` by area-weighted mean.

    ``Hf = ny*by`` and ``Wf = nx*bx`` must hold exactly.  Each coarse cell's
    value is the ``cos(lat)``-weighted mean over its **valid** fine pixels;
    cells with no valid pixel are ``NaN``.

    ``valid`` may be ``(Hf, Wf)`` (a shared mask, e.g. land) or ``(Hf, Wf, L)``
    (per-layer validity, e.g. HWSD properties with ``-9`` data gaps that differ
    by depth) — the per-layer mean then only averages the pixels valid *for that
    layer*.  Returns ``(mean, valid_area_frac)`` where ``valid_area_frac`` is the
    ``(ny, nx)`` area-weighted fraction of the cell that was valid in *any* layer
    (the land fraction when ``valid`` is a land mask).
    """
    has_layers = values.ndim == 3
    v = values if has_layers else values[..., None]      # (Hf, Wf, L)
    Hf, Wf, L = v.shape
    if Hf % by or Wf % bx:
        raise ValueError(f"fine shape ({Hf},{Wf}) not divisible by block ({by},{bx}).")
    ny, nx = Hf // by, Wf // bx

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 2:
        valid = np.broadcast_to(valid[..., None], (Hf, Wf, L))
    rw = np.asarray(row_weights, dtype=np.float64)[:, None, None]       # (Hf, 1, 1)

    w = rw * valid                                                      # (Hf, Wf, L)
    sum_w = w.reshape(ny, by, nx, bx, L).sum(axis=(1, 3))              # (ny, nx, L)
    vw = np.where(valid, v.astype(np.float64), 0.0) * w
    sum_vw = vw.reshape(ny, by, nx, bx, L).sum(axis=(1, 3))           # (ny, nx, L)

    # Land/validity fraction: any-layer-valid pixels, area-weighted by all pixels.
    cell_valid = valid.any(axis=-1)                                    # (Hf, Wf)
    rw2 = np.asarray(row_weights, dtype=np.float64)[:, None]
    sum_wv = (rw2 * cell_valid).reshape(ny, by, nx, bx).sum(axis=(1, 3))
    tot_w = np.broadcast_to(rw2, (Hf, Wf)).reshape(ny, by, nx, bx).sum(axis=(1, 3))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(sum_w > 0.0, sum_vw / np.maximum(sum_w, _TINY), np.nan)
        valid_area_frac = sum_wv / np.maximum(tot_w, _TINY)

    return (mean if has_layers else mean[..., 0]), valid_area_frac


def block_factors(header: EnviBilHeader, res_deg: float) -> tuple[int, int]:
    """Integer (by, bx) fine-pixels-per-coarse-cell for a target resolution.

    Raises if ``res_deg`` is not an exact integer multiple of the raster's pixel
    size in both directions (we only support clean block aggregation).
    """
    by = res_deg / header.ydim
    bx = res_deg / header.xdim
    by_i, bx_i = int(round(by)), int(round(bx))
    if abs(by - by_i) > 1e-6 or abs(bx - bx_i) > 1e-6:
        raise ValueError(
            f"target {res_deg} deg is not an integer multiple of raster pixel "
            f"({header.ydim} x {header.xdim} deg)."
        )
    if header.nrows % by_i or header.ncols % bx_i:
        raise ValueError(
            f"raster {header.nrows}x{header.ncols} not divisible by block "
            f"({by_i},{bx_i}) for {res_deg} deg."
        )
    return by_i, bx_i


def coarse_grid_centers(
    res_deg: float, lon0: float = -180.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre ``(lat, lon)`` of a global regular grid at ``res_deg``.

    ``lat`` runs north→south (descending) to match BIL row order; ``lon`` is
    ascending from ``lon0`` (default -180, matching HWSD2).
    """
    nx = int(round(360.0 / res_deg))
    ny = int(round(180.0 / res_deg))  # coeff-ok: latitude span [deg] (global grid geometry)
    lon = lon0 + (np.arange(nx, dtype=np.float64) + 0.5) * res_deg
    lat = 90.0 - (np.arange(ny, dtype=np.float64) + 0.5) * res_deg  # coeff-ok: north pole latitude [deg]
    return lat, lon


# Transform: raw fine block (uint) + its row latitudes -> (values, valid mask).
# ``values`` may carry a trailing layer axis; ``valid`` is (h, w) bool.
BlockTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def aggregate_raster_streaming(
    fine: np.ndarray,
    header: EnviBilHeader,
    res_deg: float,
    transform: BlockTransform,
    *,
    coarse_rows_per_chunk: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream a regular fine raster to a coarse grid in row-chunks.

    ``fine`` is any 2-D array-like indexable by row-slice (typically a
    ``np.memmap`` over the BIL).  For each chunk of ``coarse_rows_per_chunk``
    output rows, the corresponding fine-row strip is read, passed through
    ``transform`` to get physical values + a validity mask, then block-reduced by
    :func:`area_weighted_block_mean`.  Returns
    ``(mean, valid_area_frac, coarse_lat, coarse_lon)`` with ``mean`` shaped
    ``(ny, nx[, L])`` per the transform's output.
    """
    by, bx = block_factors(header, res_deg)
    ny, nx = header.nrows // by, header.ncols // bx
    lat_fine, _ = cell_centers(header)

    out_mean: np.ndarray | None = None
    out_frac = np.zeros((ny, nx), dtype=np.float64)

    for c0 in range(0, ny, coarse_rows_per_chunk):
        c1 = min(c0 + coarse_rows_per_chunk, ny)
        r0, r1 = c0 * by, c1 * by
        block = np.asarray(fine[r0:r1, :])          # materialize this strip only
        lat_block = lat_fine[r0:r1]
        values, valid = transform(block, lat_block)
        mean, frac = area_weighted_block_mean(
            values, valid, row_cos_weights(lat_block), by, bx
        )
        if out_mean is None:
            out_mean = np.full((ny, nx) + mean.shape[2:], np.nan, dtype=np.float64)
        out_mean[c0:c1] = mean
        out_frac[c0:c1] = frac

    clat, clon = coarse_grid_centers(res_deg, lon0=header.ulxmap - 0.5 * header.xdim)
    return out_mean, out_frac, clat, clon
