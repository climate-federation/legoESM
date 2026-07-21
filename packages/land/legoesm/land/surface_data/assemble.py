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
    import os

    import xarray as xr

    clm = read_clm5_cover_veg(clm_path, clm_config)
    tgt_lat, tgt_lon = clm["lat"], clm["lon"]

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
        # Provenance: name the CLM source and record that cover is gated by
        # LANDFRAC_PFT (percent-of-land -> percent-of-gridcell).
        source=("legoESM v1 surfdata: CLM5 cover/PFT/LAI (LANDFRAC_PFT-gated) + "
                f"HWSD v2.0 soil; CLM source={os.path.basename(clm_path)}"),
    )
    return out_path


# Percent <-> fraction: the harmonized schema stores cover in percent [0-100]
# (the loader's pct_scale=0.01 converts back); LUH2 states are fractions [0-1].
_FRAC_TO_PCT = 100.0


def _regrid_time_field(arr, src_lat, src_lon, tgt_lat, tgt_lon):
    """Conservative regrid a ``(nyear, slat, slon)`` field -> ``(nyear, ny, nx)``."""
    a = np.moveaxis(np.asarray(arr, dtype=np.float64), 0, -1)   # (slat, slon, nyear)
    r = conservative_regrid_latlon(a, src_lat, src_lon, tgt_lat, tgt_lon)
    return np.moveaxis(r, -1, 0)


def _load_base_static(base_surfdata_nc: str) -> dict:
    """Read the static (non-cover) fields + base 17-PFT map from a base surfdata.

    Everything but the transient cover — grid, soil, monthly vegetation, lake /
    glacier and the base PFT map (reused as the PNV shape + C4 ratios) — is copied
    through unchanged; a producer only swaps in a new ``pft_frac`` time series.
    """
    import xarray as xr

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

        f_land_pct = _static2d("f_land")             # percent [0-100]; None if absent
        return {
            "tgt_lat": tgt_lat, "tgt_lon": tgt_lon, "base_pft": base_pft,
            "soil": {v: _get(v) for v in _SOIL_VARS},
            "veg": {v: _get(v) for v in
                    ("monthly_lai", "monthly_sai", "monthly_height_top", "monthly_height_bot")},
            "soil_color": _get("soil_color"), "soil_dz": _get("soil_dz"),
            "cell_area": _get("cell_area"),
            "f_lake": _static2d("f_lake"), "f_glacier": _static2d("f_glacier"),
            # Fraction [0-1] for use as the anthropogenic-overlay land budget.
            "f_land_frac": None if f_land_pct is None else f_land_pct / _FRAC_TO_PCT,
        }
    finally:
        base.close()


def _write_transient(out_path, base, years, pft_frac, source) -> str:
    """Write a transient surfdata from static base fields + a ``(nyear,17,ny,nx)`` cover.

    ``f_land`` is the per-year PFT sum; static lake/glacier are broadcast across the
    year axis; cover is scaled fraction -> percent to match the schema convention.
    """
    f_land = pft_frac.sum(axis=1)                                 # (nyear, ny, nx) [0-1]
    nyear = pft_frac.shape[0]

    def _broadcast_year(a):
        return None if a is None else np.broadcast_to(a, (nyear,) + a.shape).copy()

    soil, veg = base["soil"], base["veg"]
    write_surfdata(
        out_path,
        lat=base["tgt_lat"], lon=base["tgt_lon"], soil_dz=base["soil_dz"],
        sand_pct=soil["sand_pct"], clay_pct=soil["clay_pct"],
        organic=soil["organic"], bulk_density=soil["bulk_density"],
        soil_color=base["soil_color"], cell_area=base["cell_area"],
        year=np.asarray(years, dtype=np.float64),
        f_land=f_land * _FRAC_TO_PCT,
        f_lake=_broadcast_year(base["f_lake"]),
        f_glacier=_broadcast_year(base["f_glacier"]),
        pft_frac=pft_frac * _FRAC_TO_PCT,
        monthly_lai=veg["monthly_lai"], monthly_sai=veg["monthly_sai"],
        monthly_height_top=veg["monthly_height_top"],
        monthly_height_bot=veg["monthly_height_bot"],
        source=source,
    )
    return out_path


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
    from legoesm.land.surface_data.sources.luh2 import (
        LUH2Config,
        build_transient_pft_frac,
        read_luh2_states,
    )

    base = _load_base_static(base_surfdata_nc)
    luh2 = read_luh2_states(luh2_states_nc, config=luh2_config or LUH2Config(), years=years)
    src_lat = np.asarray(luh2["lat"], dtype=np.float64)
    src_lon = np.asarray(luh2["lon"], dtype=np.float64)

    luh2_g = {"states": {
        s: _regrid_time_field(a, src_lat, src_lon, base["tgt_lat"], base["tgt_lon"])
        for s, a in luh2["states"].items()}}
    pft_frac = build_transient_pft_frac(luh2_g, base["base_pft"])  # (nyear, 17, ny, nx) [0-1]

    return _write_transient(
        out_path, base, luh2["years"], pft_frac,
        source="legoESM transient surfdata: LUH2 land-use cover + base soil/veg")


def build_anthropogenic_transient_surfdata(
    base_surfdata_nc: str,
    anthro_nc: str,
    out_path: str,
    *,
    dataset: str,
    years: tuple[int, int] | None = None,
    crop_share: float = 0.5,
) -> str:
    """Build a TRANSIENT surfdata from a HYDE / Pongratz / KK10 reconstruction.

    Same composition as :func:`build_luh2_transient_surfdata` (static base + swapped
    cover), but these datasets carry only the anthropogenic fraction, so the natural
    backdrop is the base 17-PFT map (PNV shape) and the crop / pasture / urban
    fractions are overlaid via
    :func:`legoesm.land.surface_data.sources.anthropogenic.build_anthropogenic_pft_frac`.
    ``dataset`` selects the reader; ``crop_share`` splits KK10's single total.
    """
    from legoesm.land.surface_data.sources.anthropogenic import build_anthropogenic_pft_frac
    from legoesm.land.surface_data.sources.hyde import read_hyde
    from legoesm.land.surface_data.sources.kk10 import read_kk10
    from legoesm.land.surface_data.sources.pongratz import read_pongratz

    if dataset == "hyde":
        anthro = read_hyde(anthro_nc, years=years)
    elif dataset == "pongratz":
        anthro = read_pongratz(anthro_nc, years=years)
    elif dataset == "kk10":
        anthro = read_kk10(anthro_nc, crop_share=crop_share, years=years)
    else:
        raise ValueError(
            f"build_anthropogenic_transient_surfdata: unknown dataset {dataset!r}; "
            f"expected one of 'hyde', 'pongratz', 'kk10'.")

    base = _load_base_static(base_surfdata_nc)
    src_lat = np.asarray(anthro["lat"], dtype=np.float64)
    src_lon = np.asarray(anthro["lon"], dtype=np.float64)

    regridded = {
        k: _regrid_time_field(anthro[k], src_lat, src_lon, base["tgt_lat"], base["tgt_lon"])
        for k in ("crop", "pasture", "urban")}
    # The anthropogenic overlay fills natural PFTs around the crop/pasture/urban
    # cover WITHIN the base land fraction, so the ocean / lake / glacier mask is
    # preserved and pft_frac sums to f_land (not 1).  Fall back to whole-cell land
    # if the base file has no f_land.
    land_frac = base["f_land_frac"]
    land_budget = 1.0 if land_frac is None else land_frac
    pft_frac = build_anthropogenic_pft_frac(regridded, base["base_pft"], land_frac=land_budget)

    return _write_transient(
        out_path, base, anthro["years"], pft_frac,
        source=f"legoESM transient surfdata: {dataset} land-use cover + base soil/veg")
