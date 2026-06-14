"""Assemble the v1 harmonized ``legoesm_surfdata`` file from multiple sources.

v1 composition (see ``sources/clm5_surfdata.py`` for the v1-vs-v2 rationale):
  - **cover + PFT + LAI** from a CLM5 surfdata file (internally consistent, no
    ERA5 needed) on the CLM grid;
  - **soil** from the observational HWSD v2.0 product (``sources/hwsd2.py``),
    regridded onto the CLM grid (HWSD soil is better than CLM's; v1 caps it at the
    CLM/model resolution, which is fine for ~2 deg AMIP land).

The output is the single regular-lat-lon file the runtime loader
(:func:`legoesm.land.global_surface_data.load_global_surface_data`, preset
``"legoesm_surfdata"``) regrids to the model grid.

v2 swap-in: replace ``read_clm5_cover_veg`` with the ESA/Luo/GIMMS observational
sources (+ ``esa_to_clm5_pft`` / ``bioclimate`` or ESA-native PFTs).  Only the
cover/PFT/LAI inputs to :func:`build_v1_surfdata` change; soil + schema + loader
are unchanged.

Host-side only; not traced.
"""

from __future__ import annotations

import numpy as np

from legoesm.grids.regridding import compute_latlon_to_voronoi_weights, regrid_scalar
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_data.sources.clm5_surfdata import (
    read_clm5_cover_veg,
    CLM5SurfdataConfig,
)

_SOIL_VARS = ("sand_pct", "clay_pct", "organic", "bulk_density")


def _regrid_layered_latlon(field_lyx, weights, ny, nx):
    """Regrid a ``(nlayer, slat, slon)`` regular field to ``(nlayer, ny, nx)``."""
    a = np.moveaxis(np.asarray(field_lyx), 0, -1)             # (slat, slon, nlayer)
    out = np.asarray(regrid_scalar(a.astype(np.float32), weights))   # (ntarget, nlayer)
    nlayer = a.shape[-1]
    return np.moveaxis(out.reshape(ny, nx, nlayer), -1, 0)    # (nlayer, ny, nx)


def build_v1_surfdata(
    clm_path: str,
    hwsd_soil_nc: str,
    out_path: str,
    *,
    clm_config: CLM5SurfdataConfig = CLM5SurfdataConfig(),
    soil_k_neighbors: int = 4,
) -> str:
    """Build the v1 harmonized surfdata: CLM cover/PFT/LAI + HWSD soil.

    Reads cover/PFT/LAI from ``clm_path`` (CLM grid), regrids the HWSD soil file
    ``hwsd_soil_nc`` (e.g. ``legoesm_surfdata_soil_0p25.nc``) onto that grid, and
    writes the combined ``legoesm_surfdata`` NetCDF to ``out_path``.  Cover is a
    single (stationary) year.
    """
    import xarray as xr

    clm = read_clm5_cover_veg(clm_path, clm_config)
    tgt_lat, tgt_lon = clm["lat"], clm["lon"]
    ny, nx = tgt_lat.size, tgt_lon.size

    hs = xr.open_dataset(hwsd_soil_nc)
    try:
        src_lat = np.asarray(hs["lat"].values, dtype=np.float64)
        src_lon = np.asarray(hs["lon"].values, dtype=np.float64)
        lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)
        weights = compute_latlon_to_voronoi_weights(
            np.deg2rad(src_lat), np.deg2rad(src_lon),
            np.deg2rad(lat2d.ravel()), np.deg2rad(lon2d.ravel()),
            k_neighbors=soil_k_neighbors,
        )
        soil = {v: _regrid_layered_latlon(hs[v].values, weights, ny, nx) for v in _SOIL_VARS}
        soil_dz = np.asarray(hs["soil_dz"].values, dtype=np.float64)
    finally:
        hs.close()

    write_surfdata(
        out_path,
        lat=tgt_lat, lon=tgt_lon, soil_dz=soil_dz,
        sand_pct=soil["sand_pct"], clay_pct=soil["clay_pct"],
        organic=soil["organic"], bulk_density=soil["bulk_density"],
        soil_color=clm["soil_color"],
        year=np.array([clm["year"]], dtype=np.float64),
        f_land=clm["f_land"][None], f_lake=clm["f_lake"][None],
        f_glacier=clm["f_glacier"][None],
        pft_frac=clm["pft_frac"][None],                      # (1, 17, ny, nx)
        monthly_lai=clm["monthly_lai"], monthly_sai=clm["monthly_sai"],
        monthly_height_top=clm["monthly_height_top"],
        monthly_height_bot=clm["monthly_height_bot"],
        source="legoESM v1 surfdata: CLM5 cover/PFT/LAI + HWSD v2.0 soil",
    )
    return out_path
