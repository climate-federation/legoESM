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

v2 swap-in: replace ``read_clm5_cover_veg`` with an observational cover/PFT/LAI
source.  Only the cover/PFT/LAI inputs to :func:`build_v1_surfdata` change; soil
+ schema + loader are unchanged.

Host-side only; not traced.
"""

from __future__ import annotations

import numpy as np

from legoesm.grids.regridding import conservative_regrid_latlon
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_data.sources.clm5_surfdata import (
    read_clm5_cover_veg,
    CLM5SurfdataConfig,
)

_SOIL_VARS = ("sand_pct", "clay_pct", "organic", "bulk_density")


def build_v1_surfdata(
    clm_path: str,
    hwsd_soil_nc: str,
    out_path: str,
    *,
    clm_config: CLM5SurfdataConfig = CLM5SurfdataConfig(),
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

        def _regrid_soil(field_lyx):
            # (nlayer, slat, slon) -> conservative -> (nlayer, ny, nx)
            a = np.moveaxis(np.asarray(field_lyx, dtype=np.float64), 0, -1)  # (slat,slon,L)
            out = conservative_regrid_latlon(a, src_lat, src_lon, tgt_lat, tgt_lon)
            return np.moveaxis(out, -1, 0)

        soil = {v: _regrid_soil(hs[v].values) for v in _SOIL_VARS}
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


# Percent <-> fraction: the harmonized schema stores cover in percent [0-100]
# (the loader's pct_scale=0.01 converts back); LUH2 states are fractions [0-1].
_FRAC_TO_PCT = 100.0


def build_luh2_transient_surfdata(
    base_surfdata_nc: str,
    luh2_states_nc: str,
    out_path: str,
    *,
    years: tuple[int, int] | None = None,
    luh2_config=None,
) -> str:
    """Build a TRANSIENT ``legoesm_surfdata`` file: annual LUH2 cover on a base grid.

    Composition (v1 land-use):
      - **grid + everything but cover** comes from an existing static harmonized
        surfdata (:func:`build_v1_surfdata` output): its lat/lon target grid, its
        base 17-PFT map (reused as the potential-natural-vegetation shape and the
        C4-grass ratio), soil, monthly vegetation, lake and glacier — all copied
        through unchanged (they carry no year axis and are treated as static);
      - **transient cover** comes from the LUH2 states file: the 12 states are
        conservatively regridded 0.25 deg -> the base grid, then crosswalked to the
        17-PFT axis per year (:func:`legoesm.land.surface_data.sources.luh2.build_transient_pft_frac`),
        replacing ``pft_frac`` with a ``(nyear, 17, lat, lon)`` series and setting
        ``f_land`` to the per-year PFT sum.

    The LUH2-derived cover is written in **percent** to match the schema
    convention (the loader re-applies ``pct_scale``).  Only cover/PFT/LAI change
    versus the base — soil + schema + loader are unchanged (the v2 swap-in the
    module docstring anticipates).  Host-side; not traced.
    """
    import xarray as xr

    from legoesm.land.surface_data.sources.luh2 import (
        LUH2Config,
        build_transient_pft_frac,
        read_luh2_states,
    )

    base = xr.open_dataset(base_surfdata_nc, decode_times=False)
    try:
        tgt_lat = np.asarray(base["lat"].values, dtype=np.float64)
        tgt_lon = np.asarray(base["lon"].values, dtype=np.float64)
        base_pft = np.asarray(base["pft_frac"].values, dtype=np.float64)
        if base_pft.ndim == 4:                       # (year, npft, lat, lon) -> drop year
            base_pft = base_pft[0]

        def _get(name):
            return np.asarray(base[name].values, dtype=np.float64) if name in base else None

        def _static2d(name):
            # A base cover fraction stored (year, lat, lon); take its single slice.
            a = _get(name)
            return a[0] if (a is not None and a.ndim == 3) else a

        soil = {v: _get(v) for v in _SOIL_VARS}
        veg = {v: _get(v) for v in
               ("monthly_lai", "monthly_sai", "monthly_height_top", "monthly_height_bot")}
        soil_color = _get("soil_color")
        soil_dz = _get("soil_dz")
        cell_area = _get("cell_area")
        f_lake = _static2d("f_lake")
        f_glacier = _static2d("f_glacier")
    finally:
        base.close()

    luh2 = read_luh2_states(luh2_states_nc, config=luh2_config or LUH2Config(), years=years)
    src_lat = np.asarray(luh2["lat"], dtype=np.float64)
    src_lon = np.asarray(luh2["lon"], dtype=np.float64)

    def _regrid_state(arr):                          # (nyear, slat, slon) -> (nyear, ny, nx)
        a = np.moveaxis(np.asarray(arr, dtype=np.float64), 0, -1)   # (slat, slon, nyear)
        r = conservative_regrid_latlon(a, src_lat, src_lon, tgt_lat, tgt_lon)
        return np.moveaxis(r, -1, 0)

    luh2_g = {"states": {s: _regrid_state(a) for s, a in luh2["states"].items()}}
    pft_frac = build_transient_pft_frac(luh2_g, base_pft)          # (nyear, 17, ny, nx) [0-1]
    f_land = pft_frac.sum(axis=1)                                  # (nyear, ny, nx) [0-1]

    nyear = pft_frac.shape[0]

    def _broadcast_year(a):
        # Copy a static (lat, lon) base fraction across the LUH2 year axis so the
        # transient file has a consistent (year, lat, lon) cover group.
        return None if a is None else np.broadcast_to(a, (nyear,) + a.shape).copy()

    write_surfdata(
        out_path,
        lat=tgt_lat, lon=tgt_lon, soil_dz=soil_dz,
        sand_pct=soil["sand_pct"], clay_pct=soil["clay_pct"],
        organic=soil["organic"], bulk_density=soil["bulk_density"],
        soil_color=soil_color, cell_area=cell_area,
        year=np.asarray(luh2["years"], dtype=np.float64),
        f_land=f_land * _FRAC_TO_PCT,
        f_lake=_broadcast_year(f_lake), f_glacier=_broadcast_year(f_glacier),
        pft_frac=pft_frac * _FRAC_TO_PCT,
        monthly_lai=veg["monthly_lai"], monthly_sai=veg["monthly_sai"],
        monthly_height_top=veg["monthly_height_top"],
        monthly_height_bot=veg["monthly_height_bot"],
        source="legoESM transient surfdata: LUH2 land-use cover + base soil/veg",
    )
    return out_path
