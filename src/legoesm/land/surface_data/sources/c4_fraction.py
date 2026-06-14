"""C4 vegetation source (Luo et al. 2024): observational C4 grass/crop area.

Supplies the observational C3/C4 split that the ESA->CLM5 crosswalk needs.
legoESM does **not** split grass cover into C3/C4 classes from a climate rule;
instead it reads C4 *area* from an observational map and forms the C4 fraction
*of grass* at assembly time (``C4 grass area / ESA grass cover``) — keeping the
data faithful and the pathway split observational.

The Luo et al. (2024) product (``C4_distribution_NUS_v2.2.nc``) is a 0.5 deg,
2001-2019 map with ``C4_grass_area`` and ``C4_crop_area`` (plus total
``C4_area`` and ``*_un`` uncertainty), in **percent of land-surface area**, dims
``(years, lon, lat)`` (note: lon before lat), NaN over ocean.  For v1 a single
year (2010) is taken and held stationary, consistent with the ESA PFT map.

This source is coarser (0.5 deg) than the 0.25 deg harmonized grid, so it is
**interpolated up** (KD-tree IDW, the same regridder the runtime loader uses) to
the common grid; values are returned as fractions [0-1] of gridcell area, ocean
filled with 0.

References
----------
- Luo, X., Zhou, H., Satriawan, T. W., Tian, J., Zhao, R., Keenan, T. F.,
  Griffith, D. M., Sitch, S., Smith, N. G., and Still, C. J. (2024): Mapping the
  global distribution of C4 vegetation using observations and optimality theory.
  Nature Communications, 15, 1219. https://doi.org/10.1038/s41467-024-45606-3
"""

from __future__ import annotations

import numpy as np

from legoesm.grids.regridding import compute_latlon_to_voronoi_weights, regrid_scalar

# Canonical output key -> variable name in the Luo NetCDF.
C4_NC_VARMAP: dict[str, str] = {
    "c4_grass_area": "C4_grass_area",
    "c4_crop_area": "C4_crop_area",
}


def load_c4_areas(
    nc_path: str | None,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    year: int = 2010,
    dataset=None,
    k_neighbors: int = 4,
) -> dict:
    """Read Luo C4 grass/crop area for ``year`` and regrid to a target lat-lon grid.

    ``target_lat``/``target_lon`` are 1-D cell centres [deg] of the harmonized
    grid.  Returns ``{"c4_grass_area", "c4_crop_area"}`` each shaped
    ``(len(target_lat), len(target_lon))`` as fractions [0-1] of gridcell area
    (ocean = 0).  ``dataset`` may be injected for testing.
    """
    import xarray as xr  # noqa: F401

    ds = dataset if dataset is not None else xr.open_dataset(nc_path, decode_times=False)
    try:
        years = np.asarray(ds["years"].values)
        iy = int(np.argmin(np.abs(years - year)))
        src_lat = np.asarray(ds["lat"].values, dtype=np.float64)
        src_lon = np.asarray(ds["lon"].values, dtype=np.float64)

        tgt_lat = np.asarray(target_lat, dtype=np.float64)
        tgt_lon = np.asarray(target_lon, dtype=np.float64)
        ny, nx = tgt_lat.size, tgt_lon.size
        lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)        # (ny, nx)
        weights = compute_latlon_to_voronoi_weights(
            np.deg2rad(src_lat), np.deg2rad(src_lon),
            np.deg2rad(lat2d.ravel()), np.deg2rad(lon2d.ravel()),
            k_neighbors=k_neighbors,
        )

        out = {}
        for key, var in C4_NC_VARMAP.items():
            # file dims (years, lon, lat) -> select year, orient (lat, lon).
            da = ds[var].isel(years=iy).transpose("lat", "lon")
            arr = np.nan_to_num(np.asarray(da.values, dtype=np.float64), nan=0.0) / 100.0
            regridded = np.asarray(regrid_scalar(arr.astype(np.float32), weights))
            out[key] = np.clip(regridded.reshape(ny, nx), 0.0, 1.0)
        return out
    finally:
        if dataset is None:
            ds.close()


def c4_fraction_of_grass(
    c4_area: np.ndarray, grass_cover: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
    """C4 fraction *of grass* = C4 area / grass cover, clipped to [0, 1].

    Both inputs are gridcell-area fractions on the same grid.  Where grass cover
    is negligible the fraction is 0 (no grass -> the C4 weighting is irrelevant).
    Clipping guards against minor disagreement between the two datasets.
    """
    c4_area = np.asarray(c4_area, dtype=np.float64)
    grass_cover = np.asarray(grass_cover, dtype=np.float64)
    frac = np.where(grass_cover > eps, c4_area / np.maximum(grass_cover, eps), 0.0)
    return np.clip(frac, 0.0, 1.0)
