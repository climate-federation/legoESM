"""Extract single-point surface parameters from a harmonized surfdata file.

For offline single-column runs (e.g. ``scripts/run_lmip.py``): given a lat/lon,
read the nearest gridcell of a ``legoesm_surfdata`` file and return the surface
properties the land model needs — dominant PFT, its monthly LAI cycle, a
column-representative soil texture, and the soil-colour class.

Host-side; not traced.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PointSurfaceParams(NamedTuple):
    """Surface parameters at one gridcell of a harmonized surfdata file."""

    lat: float
    lon: float
    dominant_pft: str            # CLM5 PFT name of the dominant cover
    dominant_pft_index: int
    lai_monthly: np.ndarray      # (12,) LAI of the dominant PFT [m2/m2]
    sand_pct: float              # column-mean percent sand
    clay_pct: float              # column-mean percent clay
    organic: float               # column-mean organic
    soil_color: int              # CLM soil-colour class (1..20)


def surface_params_at_point(
    surfdata_path: str,
    lat_deg: float,
    lon_deg: float,
    *,
    dataset=None,
) -> PointSurfaceParams:
    """Read surface parameters at the gridcell nearest ``(lat_deg, lon_deg)``.

    Longitudes are matched modulo 360 so the query works whether the file uses
    [-180,180) or [0,360).  Soil texture is averaged over the soil-layer axis to
    a single column-representative value (the scalar hydraulics config the v1
    model uses).  ``dataset`` may be injected for testing.
    """
    import xarray as xr
    from legoesm.land.surface_params import CLM5_PFT_NAMES

    ds = dataset if dataset is not None else xr.open_dataset(surfdata_path)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        i = int(np.argmin(np.abs(lat - lat_deg)))
        dlon = np.abs(((lon - lon_deg + 180.0) % 360.0) - 180.0)
        j = int(np.argmin(dlon))

        pft = np.asarray(ds["pft_frac"].isel(year=0, lat=i, lon=j).values, dtype=np.float64)
        dom = int(np.argmax(pft))
        lai = np.asarray(ds["monthly_lai"].isel(npft=dom, lat=i, lon=j).values, dtype=np.float64)

        def soil_mean(var):
            return float(np.nanmean(np.asarray(ds[var].isel(lat=i, lon=j).values)))

        return PointSurfaceParams(
            lat=float(lat[i]), lon=float(lon[j]),
            dominant_pft=CLM5_PFT_NAMES[dom], dominant_pft_index=dom,
            lai_monthly=lai,
            sand_pct=soil_mean("sand_pct"), clay_pct=soil_mean("clay_pct"),
            organic=soil_mean("organic"),
            soil_color=int(round(float(ds["soil_color"].isel(lat=i, lon=j).values))),
        )
    finally:
        if dataset is None:
            ds.close()
