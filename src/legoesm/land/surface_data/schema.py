"""Harmonized ``legoesm_surfdata`` NetCDF schema + writer (producer side).

The surface-data producer aggregates each primary source (HWSD2 soil, later
GIMMS LAI4g, ESA-CCI biomass, ...) onto a common regular lat-lon grid and writes
them into **one** harmonized NetCDF with stable variable names, units and dim
order.  That file is the single artifact consumed by the runtime loader
:func:`legoesm.land.global_surface_data.load_global_surface_data` (via the
``"legoesm_surfdata"`` preset), which regrids it once to the model grid.

The schema deliberately mirrors what the loader's reader already understands, so
the *same* reader code path serves both CLM5 surfdata and our native file — the
only differences are the variable names (the preset) and that we publish clean
1-D ``lat``/``lon`` coordinates (avoiding CLM's 2-D ``LATIXY``/``LONGXY`` trap).

Dim conventions (matching the loader's transpose logic):
  - soil fields: ``(soil_layer, lat, lon)`` with a 1-D ``soil_dz`` [m] of layer
    *thicknesses* (the loader integrates these to midpoint depths).
  - cover fractions: ``(year, lat, lon)``; PFT cover ``(year, npft, lat, lon)``.
  - monthly vegetation: ``(month, npft, lat, lon)``.
  - percent fields (sand/clay, cover, PFT) are stored in **percent [0-100]**; the
    loader's ``pct_scale=0.01`` converts to fraction.

Every data group is optional in :func:`write_surfdata` so a source can contribute
just its fields (e.g. HWSD writes soil only); a *complete* file ready for the
loader is assembled by writing all groups together (or merging per-source files).

Host-side only; not traced.
"""

from __future__ import annotations

import numpy as np

# FAO HWSD v2.0 fixed depth layers D1..D7 thicknesses [m] (0-20-40-60-80-100-150-200 cm).
HWSD2_LAYER_DZ = np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.50, 0.50], dtype=np.float64)

# Canonical variable names + (units, long_name) for the harmonized file.  The
# loader's "legoesm_surfdata" preset references exactly these names.
SURFDATA_VARS: dict[str, tuple[str, str]] = {
    # soil (static)
    "sand_pct": ("percent", "sand mass fraction per soil layer"),
    "clay_pct": ("percent", "clay mass fraction per soil layer"),
    "organic": ("kg m-3", "soil organic matter density per layer"),
    "bulk_density": ("kg m-3", "soil bulk density per layer"),
    "soil_color": ("1", "soil colour class (integer)"),
    "soil_dz": ("m", "soil layer thickness"),
    "cell_area": ("m2", "grid cell area"),
    # cover fractions (annual transient)
    "year": ("year", "calendar year of transient slice"),
    "f_land": ("percent", "land-surface (soil+vegetation) fraction of gridcell"),
    "f_lake": ("percent", "inland water fraction of gridcell"),
    "f_glacier": ("percent", "snow/ice (glacier) fraction of gridcell"),
    # pft_frac axis is the ESA-native land composition (bare + 10 veg PFTs;
    # LAND_PFT_NAMES), NOT CLM5's climate-zoned 17 PFTs.
    "pft_frac": ("percent", "within-land PFT cover fraction (ESA taxonomy)"),
    # C3/C4 is a continuous pathway field, not a cover split (Still et al.-type).
    "c4_fraction": ("1", "fraction of grass that is C4 (0-1)"),
    # vegetation state (monthly climatology)
    "monthly_lai": ("m2 m-2", "monthly leaf area index"),
    "monthly_sai": ("m2 m-2", "monthly stem area index"),
    "monthly_height_top": ("m", "monthly canopy top height"),
    "monthly_height_bot": ("m", "monthly canopy bottom height"),
}


def _da(dims, data, name):
    import xarray as xr
    units, long_name = SURFDATA_VARS[name]
    return xr.DataArray(
        np.asarray(data), dims=dims, name=name,
        attrs={"units": units, "long_name": long_name},
    )


def write_surfdata(
    path: str,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    soil_dz: np.ndarray | None = None,
    # soil group
    sand_pct: np.ndarray | None = None,        # (soil_layer, lat, lon)
    clay_pct: np.ndarray | None = None,
    organic: np.ndarray | None = None,
    bulk_density: np.ndarray | None = None,
    soil_color: np.ndarray | None = None,      # (lat, lon)
    cell_area: np.ndarray | None = None,       # (lat, lon)
    c4_fraction: np.ndarray | None = None,     # (lat, lon) [0-1]
    # cover group
    year: np.ndarray | None = None,            # (year,)
    f_land: np.ndarray | None = None,          # (year, lat, lon)
    f_lake: np.ndarray | None = None,
    f_glacier: np.ndarray | None = None,
    pft_frac: np.ndarray | None = None,        # (year, npft, lat, lon)
    # vegetation group
    monthly_lai: np.ndarray | None = None,     # (month, npft, lat, lon)
    monthly_sai: np.ndarray | None = None,
    monthly_height_top: np.ndarray | None = None,
    monthly_height_bot: np.ndarray | None = None,
    source: str = "legoesm surface-data producer",
) -> None:
    """Write a harmonized ``legoesm_surfdata`` NetCDF from the provided fields.

    ``lat``/``lon`` are 1-D cell centres [deg]; all data groups are optional so a
    single source can contribute only its variables.  Soil/cover/PFT percentages
    are written as-is (the loader applies ``pct_scale``).
    """
    import xarray as xr

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    data_vars: dict = {}

    SY = ("soil_layer", "lat", "lon")
    for nm, arr in (("sand_pct", sand_pct), ("clay_pct", clay_pct),
                    ("organic", organic), ("bulk_density", bulk_density)):
        if arr is not None:
            data_vars[nm] = _da(SY, arr, nm)
    if soil_color is not None:
        data_vars["soil_color"] = _da(("lat", "lon"), soil_color, "soil_color")
    if cell_area is not None:
        data_vars["cell_area"] = _da(("lat", "lon"), cell_area, "cell_area")
    if c4_fraction is not None:
        data_vars["c4_fraction"] = _da(("lat", "lon"), c4_fraction, "c4_fraction")
    if soil_dz is not None:
        data_vars["soil_dz"] = _da(("soil_layer",), soil_dz, "soil_dz")

    if year is not None:
        data_vars["year"] = _da(("year",), year, "year")
    for nm, arr in (("f_land", f_land), ("f_lake", f_lake), ("f_glacier", f_glacier)):
        if arr is not None:
            data_vars[nm] = _da(("year", "lat", "lon"), arr, nm)
    if pft_frac is not None:
        data_vars["pft_frac"] = _da(("year", "npft", "lat", "lon"), pft_frac, "pft_frac")

    for nm, arr in (("monthly_lai", monthly_lai), ("monthly_sai", monthly_sai),
                    ("monthly_height_top", monthly_height_top),
                    ("monthly_height_bot", monthly_height_bot)):
        if arr is not None:
            data_vars[nm] = _da(("month", "npft", "lat", "lon"), arr, nm)

    ds = xr.Dataset(
        data_vars,
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"title": "legoESM harmonized surface boundary data", "source": source},
    )

    # Store floats as float32 + zlib: surface boundary data needs nowhere near
    # float64 precision (sand is a 0-100 percent, LAI ~0-7), and the fields are
    # spatially smooth with large NaN ocean areas, so this roughly halves the
    # dtype and compresses well (~5-8x overall) with no meaningful accuracy loss.
    comp = {"zlib": True, "complevel": 4, "shuffle": True}
    encoding = {}
    for name, da in ds.data_vars.items():
        enc = dict(comp)
        if np.issubdtype(da.dtype, np.floating):
            enc["dtype"] = "float32"
        encoding[name] = enc
    for name in ds.coords:
        encoding[name] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, encoding=encoding)
