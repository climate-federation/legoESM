"""Loader for the NEMO ISF 'spe' prescribed ice-shelf melt forcing.

Reads the Depoorter et al. (2013) ice-shelf freshwater distribution as
shipped with the NEMO ORCA1 reference configuration
(``runoff-icb_DaiTrenberth_Depoorter.nc``: eORCA1 curvilinear, monthly
``sornfisf`` [kg/m²/s, >= 0 into the ocean] plus the injection depth
band ``sodepmin_isf``/``sodepmax_isf`` [m]) and regrids it to the model
tracer grid — nearest-wet-neighbour with the monthly melt totals
preserved (the physically constrained quantity), exactly the zdfiwm
loader's convention.

ORCA1 namelist: ``ln_isf = ln_isfpar_mlt = .true.``, ``cn_isfpar_mlt =
'spe'`` (melt READ from this file; no cavity model).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from legoesm.ocean.forcing.curvilinear_regrid import (
    NearestWetRegridder,
    coords_match,
    estimate_curvilinear_cell_area,
)

_FWF_VAR = "sornfisf"
_ZMIN_VAR = "sodepmin_isf"
_ZMAX_VAR = "sodepmax_isf"
# Minimum injection-band thickness [m] so the overlap weights stay
# well-conditioned when zmax ≈ zmin in the source file.
_MIN_BAND_M = 1.0


class ISFSpeForcing(NamedTuple):
    """Monthly prescribed ISF melt on the model grid (leading axis = 12)."""
    fwf: np.ndarray      # (12, ...) melt freshwater INTO the ocean [kg/m²/s]
    zmin: np.ndarray     # (12, ...) band top [m, positive down, >= 0]
    zmax: np.ndarray     # (12, ...) band bottom [m, > zmin]


def load_isf_spe_forcing(
    path: str,
    lat_T,
    lon_T,
    *,
    land_mask=None,
    paired_cells: bool = False,
    target_area=None,
) -> ISFSpeForcing:
    """Load + regrid the 'spe' ISF melt fields onto the model tracer grid.

    Same-mesh targets (the eORCA1 tripole) pass through untouched; other
    grids get nearest-WET-neighbour + per-month global melt-total
    renormalisation (cos-lat weights, or ``target_area`` when given).
    ``zmin`` is clamped >= 0 (the source stores small negative band tops
    near the surface) and the band is kept at least ``1 m`` thick.

    ``paired_cells=True`` marks 1-D ``lat_T``/``lon_T`` as PAIRED unstructured
    cell centres (MPAS Voronoi ``(nCells,)``) rather than regular grid axes —
    no outer-product meshgrid, ``structured=False`` nearest-wet lookup.  Pass
    ``target_area`` (cell areas, same shape) for the melt-total weights on
    quasi-uniform meshes where cos-lat is wrong.
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        for v in (_FWF_VAR, _ZMIN_VAR, _ZMAX_VAR):
            if v not in ds.variables:
                raise ValueError(
                    f"ISF spe forcing file {path!r} is missing {v!r} "
                    "(expected the NEMO runoff-icb_DaiTrenberth_Depoorter "
                    "layout)")
        fwf = np.nan_to_num(
            np.asarray(ds.variables[_FWF_VAR][:], dtype=np.float64), nan=0.0)
        zmin = np.nan_to_num(
            np.asarray(ds.variables[_ZMIN_VAR][:], dtype=np.float64), nan=0.0)
        zmax = np.nan_to_num(
            np.asarray(ds.variables[_ZMAX_VAR][:], dtype=np.float64), nan=0.0)
        src_lat = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
        src_lon = np.asarray(ds.variables["nav_lon"][:], dtype=np.float64)
    if fwf.ndim != 3 or fwf.shape[0] != 12:
        raise ValueError(
            f"expected monthly (12, ny, nx) {_FWF_VAR}; got {fwf.shape}")

    lat_T = np.asarray(lat_T, dtype=np.float64)
    lon_T = np.asarray(lon_T, dtype=np.float64)
    if not paired_cells and lat_T.ndim == 1 and lon_T.ndim == 1:
        lat_T, lon_T = np.meshgrid(lat_T, lon_T, indexing="ij")

    if not paired_cells and coords_match(src_lat, src_lon, lat_T, lon_T):
        out_fwf, out_zmin, out_zmax = fwf.copy(), zmin.copy(), zmax.copy()
    else:
        src_wet = np.any(fwf > 0.0, axis=0)
        regrid = NearestWetRegridder(src_lon, src_lat, src_wet, lon_T, lat_T,
                                     structured=not paired_cells)
        months = range(12)
        out_fwf = np.stack([regrid(fwf[m]) for m in months])
        out_zmin = np.stack([regrid(zmin[m]) for m in months])
        out_zmax = np.stack([regrid(zmax[m]) for m in months])
        # Nearest-neighbour paints every target cell with SOME shelf value;
        # keep melt only where the source actually melts nearby is not
        # meaningful on a coarse grid — instead preserve each month's
        # global melt total and let the renormalised field carry the
        # Antarctic-margin pattern the nearest lookup produces.
        if target_area is not None:
            # Weights must share one unit or the renorm ratio is meaningless:
            # target passes true cell areas [m^2], so estimate the source's
            # curvilinear cell areas (seam/fold-aware) to match.
            src_w = estimate_curvilinear_cell_area(src_lat, src_lon)
            tgt_w = np.asarray(target_area, dtype=np.float64)
        else:
            src_w = np.cos(np.deg2rad(src_lat))
            tgt_w = np.cos(np.deg2rad(lat_T))
        tgt_m = (np.asarray(land_mask, dtype=np.float64)
                 if land_mask is not None else np.ones_like(lat_T))
        # Restrict the painted melt to cells whose nearest source cell is
        # within the shelf band's latitude range — nearest-neighbour would
        # otherwise smear Antarctic melt arbitrarily far equatorward.
        _lat_limit = float(src_lat[src_wet].max()) + 2.0
        _keep = (lat_T <= _lat_limit).astype(np.float64)
        for m in months:
            out_fwf[m] = out_fwf[m] * _keep * tgt_m
            src_total = float((fwf[m] * src_w).sum())
            tgt_total = float((out_fwf[m] * tgt_w).sum())
            if tgt_total > 0.0 and src_total > 0.0:
                out_fwf[m] *= src_total / tgt_total

    if land_mask is not None:
        out_fwf = out_fwf * np.asarray(land_mask, dtype=np.float64)[None]
    out_zmin = np.maximum(out_zmin, 0.0)
    out_zmax = np.maximum(out_zmax, out_zmin + _MIN_BAND_M)
    return ISFSpeForcing(fwf=out_fwf, zmin=out_zmin, zmax=out_zmax)
