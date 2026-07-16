"""Dai & Trenberth global river-runoff climatology loader.

Reference
---------
Dai, A., & Trenberth, K. E. (2002). Estimates of freshwater discharge
from continents: Latitudinal and seasonal variations.  *J.
Hydrometeorology*, 3(6), 660–687.  Updated: Dai et al. (2009).

What
----
The Dai-Trenberth product distributes the freshwater discharge of the
925 largest rivers as monthly climatologies at their mouth locations.
This loader returns a ``RiverRunoffData`` NamedTuple of N rivers with
their (lat, lon) coordinates and ``(12, N)`` monthly flux in kg/s.

When the on-disk NetCDF cache is missing the loader returns a built-in
synthetic climatology of the 16 largest rivers (Amazon, Congo,
Mississippi, ...) with seasonal cycles consistent with hemispheric
hydrology — so smoke tests don't depend on the ~10 MB DT download.

Once the per-river fluxes are available, :func:`project_runoff_to_grid`
projects them onto a model grid (lat-lon or arbitrary lat / lon
arrays) by binning each river into the nearest ocean cell and
dividing by the cell area to produce a [kg/m²/s] forcing field that
plugs into ``FreshwaterForcing.runoff``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np


SECONDS_PER_DAY = 86400.0


class RiverRunoffData(NamedTuple):
    """Per-river monthly climatology.

    Fields
    ------
    latitudes : ndarray ``(N,)`` — degrees N
    longitudes : ndarray ``(N,)`` — degrees E, ``[0, 360)`` wrap
    monthly_flux_kg_s : ndarray ``(12, N)`` — kg/s
    names : tuple[str, ...] — diagnostic river names
    """
    latitudes: np.ndarray
    longitudes: np.ndarray
    monthly_flux_kg_s: np.ndarray
    names: tuple[str, ...]


# 16 largest rivers — annual-mean discharge from Dai & Trenberth 2002
# Table 4 (rounded to 1e5 kg/s), mouth coordinates from CIA Factbook /
# Wikipedia.  Northern-hemisphere snowmelt-fed rivers peak May-Jul;
# tropical rivers follow the monsoon.  Magnitudes are intentionally
# approximate so the synthetic climatology stays self-contained.
_SYNTHETIC_RIVERS: tuple[dict, ...] = (
    {"name": "Amazon",      "lat":   -0.2,  "lon": -49.7,  "mean": 2.00e8, "phase": 3.0},
    {"name": "Congo",       "lat":   -6.1,  "lon":  12.4,  "mean": 4.00e7, "phase": 11.0},
    {"name": "Yangtze",     "lat":   31.5,  "lon": 121.5,  "mean": 3.00e7, "phase": 7.0},
    {"name": "Ganges",      "lat":   22.0,  "lon":  89.0,  "mean": 3.00e7, "phase": 8.0},
    {"name": "Plata",       "lat":  -34.0,  "lon": -58.5,  "mean": 2.20e7, "phase": 4.0},
    {"name": "Yenisei",     "lat":   71.3,  "lon":  83.2,  "mean": 1.90e7, "phase": 6.0},
    {"name": "Mississippi", "lat":   29.2,  "lon": -89.3,  "mean": 1.70e7, "phase": 5.0},
    {"name": "Lena",        "lat":   72.4,  "lon": 126.7,  "mean": 1.70e7, "phase": 6.0},
    {"name": "Mekong",      "lat":   10.5,  "lon": 105.6,  "mean": 1.50e7, "phase": 9.0},
    {"name": "Ob",          "lat":   66.7,  "lon":  69.0,  "mean": 1.30e7, "phase": 6.0},
    {"name": "Orinoco",     "lat":    8.6,  "lon": -62.2,  "mean": 3.50e7, "phase": 8.0},
    {"name": "Niger",       "lat":    4.3,  "lon":   6.0,  "mean": 6.50e6, "phase": 9.0},
    {"name": "Mackenzie",   "lat":   69.4,  "lon": -135.0, "mean": 1.00e7, "phase": 6.0},
    {"name": "Saint Lawrence", "lat": 49.0, "lon": -67.0,  "mean": 1.20e7, "phase": 5.0},
    {"name": "Zambezi",     "lat":  -18.3,  "lon":  36.4,  "mean": 3.40e6, "phase": 2.0},
    {"name": "Columbia",    "lat":   46.2,  "lon": -124.0, "mean": 7.00e6, "phase": 6.0},
)


def synthetic_dai_trenberth(
    *,
    seasonal_amplitude: float = 0.5,
) -> RiverRunoffData:
    """Built-in 16-river climatology.

    Each river's monthly flux is

        Q(month) = mean · (1 + amplitude · cos(2π · (month − phase) / 12))

    with northern snowmelt rivers peaking around June (phase ≈ 6) and
    tropical / monsoon rivers in late summer to early autumn.  Default
    ``seasonal_amplitude=0.5`` gives a 1.5× peak / 0.5× minimum cycle,
    consistent with observed amplitudes for the largest catchments.

    Returns
    -------
    :class:`RiverRunoffData`
    """
    len(_SYNTHETIC_RIVERS)
    lats = np.array([r["lat"] for r in _SYNTHETIC_RIVERS], dtype=np.float64)
    # Normalise to [0, 360) for downstream consistency.
    lons = np.array(
        [r["lon"] % 360.0 for r in _SYNTHETIC_RIVERS], dtype=np.float64,
    )
    means = np.array([r["mean"] for r in _SYNTHETIC_RIVERS], dtype=np.float64)
    phases = np.array([r["phase"] for r in _SYNTHETIC_RIVERS], dtype=np.float64)
    names = tuple(r["name"] for r in _SYNTHETIC_RIVERS)

    month = np.arange(1, 13, dtype=np.float64)[:, None]    # (12, 1)
    season = 1.0 + seasonal_amplitude * np.cos(
        2.0 * np.pi * (month - phases[None, :]) / 12.0,
    )
    monthly_flux = means[None, :] * season
    return RiverRunoffData(
        latitudes=lats,
        longitudes=lons,
        monthly_flux_kg_s=monthly_flux,
        names=names,
    )


def load_dai_trenberth(
    *,
    cache_dir: Optional[Path] = None,
    allow_synthetic: bool = True,
) -> RiverRunoffData:
    """Load the Dai-Trenberth river-runoff climatology.

    Reads ``<cache_dir>/dai_trenberth_rivers.nc`` (CF-compliant
    NetCDF expected variables: ``lat``, ``lon``, ``time``,
    ``runoff`` in kg/s, optional ``river_name``).  Falls back to the
    synthetic 16-river climatology when the file is missing.

    Parameters
    ----------
    cache_dir : Path or None
        Directory containing ``dai_trenberth_rivers.nc``.  Default
        ``~/.legoesm_cache/obs/dai_trenberth``.
    allow_synthetic : bool
        When the cache is absent, return the built-in synthetic data
        if True; raise ``FileNotFoundError`` if False.

    Returns
    -------
    :class:`RiverRunoffData`
    """
    if cache_dir is None:
        root = _default_cache_dir()
    else:
        root = Path(cache_dir)
    nc_path = root / "dai_trenberth_rivers.nc"
    if nc_path.exists():
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "Loading Dai-Trenberth NetCDF needs xarray; install "
                "with ``pip install xarray``"
            ) from exc
        # Context-manage the file so repeated loader calls do not
        # leak file descriptors.  Arrays are materialised with
        # ``np.asarray`` before the dataset closes.
        with xr.open_dataset(nc_path) as ds:
            lats = np.asarray(ds["lat"].values, dtype=np.float64)
            lons = np.asarray(ds["lon"].values, dtype=np.float64) % 360.0
            flux = np.asarray(ds["runoff"].values, dtype=np.float64)
            # Expect ``(time, river)`` axis order; transpose if
            # reversed.
            if flux.shape[0] == lats.shape[0] and flux.shape[-1] == 12:
                flux = flux.T
            if "river_name" in ds.variables:
                names = tuple(str(n) for n in ds["river_name"].values)
            else:
                names = tuple(f"river_{i}" for i in range(lats.shape[0]))
        return RiverRunoffData(
            latitudes=lats,
            longitudes=lons,
            monthly_flux_kg_s=flux,
            names=names,
        )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"Dai-Trenberth cache not found at {nc_path} and "
            "allow_synthetic=False"
        )
    return synthetic_dai_trenberth()


def _default_cache_dir() -> Path:
    from legoesm.ocean.fidelity import cache as _cache
    return _cache.sub("obs") / "dai_trenberth"


# ==============================================================================
# Point → grid projection
# ==============================================================================

def _nearest_cell_indices(
    river_lat_deg: np.ndarray,
    river_lon_deg: np.ndarray,
    grid_lat_deg: np.ndarray,
    grid_lon_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-cell (j_lat, i_lon) indices for each river mouth.

    Handles longitude wrap-around in [0, 360) by taking the minimum
    of two candidate distances (direct + ±360°-shifted).  Latitude
    uses straight nearest.
    """
    lat = np.asarray(grid_lat_deg, dtype=np.float64)
    lon = np.asarray(grid_lon_deg, dtype=np.float64) % 360.0
    rlat = np.asarray(river_lat_deg, dtype=np.float64)
    rlon = np.asarray(river_lon_deg, dtype=np.float64) % 360.0
    # Nearest in lat.
    j = np.argmin(np.abs(rlat[:, None] - lat[None, :]), axis=1)
    # Lon wrap: min of direct vs ±360°.
    dlon_direct = np.abs(rlon[:, None] - lon[None, :])
    dlon = np.minimum(dlon_direct, 360.0 - dlon_direct)
    i = np.argmin(dlon, axis=1)
    return j.astype(np.int64), i.astype(np.int64)


def project_runoff_to_grid(
    rivers: RiverRunoffData,
    *,
    grid_lat_deg: np.ndarray,
    grid_lon_deg: np.ndarray,
    cell_area_m2: np.ndarray,
    month: Optional[int] = None,
    ocean_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bin river mouths onto a lat-lon grid as a [kg/m²/s] field.

    For each river:
    1. Find the nearest grid cell.
    2. Add ``flux_kg_s / cell_area_m2`` to that cell.

    Multiple rivers landing in the same cell add together.  No
    coastal spreading is applied — the caller can convolve the result
    with a coastal Gaussian if needed.  When ``ocean_mask`` is
    provided, rivers landing on land cells are reassigned to the
    nearest ocean cell within a small neighbourhood; rivers with no
    ocean cell within the search radius are silently dropped (with a
    diagnostic warning if rare).

    Parameters
    ----------
    rivers : RiverRunoffData
    grid_lat_deg : ndarray ``(n_lat,)`` — degrees N, ascending.
    grid_lon_deg : ndarray ``(n_lon,)`` — degrees E, ``[0, 360)``.
    cell_area_m2 : ndarray ``(n_lat, n_lon)`` — cell area in m².
    month : int or None
        1–12 for monthly snapshot; None for annual mean.
    ocean_mask : ndarray ``(n_lat, n_lon)`` of {0, 1} or None
        1 = ocean.  When provided, rivers landing on land are
        relocated to the nearest ocean cell within a 5° box; if no
        ocean cell exists in that box the river is dropped.

    Returns
    -------
    runoff_kg_m2_s : ndarray ``(n_lat, n_lon)``
        Freshwater flux into ocean [kg/m²/s], non-negative.
    """
    n_lat = grid_lat_deg.shape[0]
    n_lon = grid_lon_deg.shape[0]
    out = np.zeros((n_lat, n_lon), dtype=np.float64)

    j, i = _nearest_cell_indices(
        rivers.latitudes, rivers.longitudes,
        grid_lat_deg, grid_lon_deg,
    )

    # Flux per river [kg/s].
    if month is None:
        flux_per_river = rivers.monthly_flux_kg_s.mean(axis=0)
    else:
        flux_per_river = rivers.monthly_flux_kg_s[(month - 1) % 12]

    for k in range(rivers.latitudes.shape[0]):
        jr = int(j[k])
        ir = int(i[k])
        if ocean_mask is not None and not bool(ocean_mask[jr, ir]):
            # Search a 5° box for the nearest ocean cell.
            jr2, ir2 = _nearest_ocean_within_box(
                jr, ir, ocean_mask,
                grid_lat_deg, grid_lon_deg, max_deg=5.0,
            )
            if jr2 is None:
                continue
            jr, ir = jr2, ir2
        cell = float(cell_area_m2[jr, ir])
        if cell <= 0.0:
            continue
        out[jr, ir] += float(flux_per_river[k]) / cell
    return out


def _nearest_ocean_within_box(
    j0: int,
    i0: int,
    ocean_mask: np.ndarray,
    grid_lat_deg: np.ndarray,
    grid_lon_deg: np.ndarray,
    *,
    max_deg: float = 5.0,
) -> Tuple[Optional[int], Optional[int]]:
    """Find nearest ocean cell to (j0, i0) within ``max_deg``."""
    n_lat = grid_lat_deg.shape[0]
    n_lon = grid_lon_deg.shape[0]
    lat0 = float(grid_lat_deg[j0])
    lon0 = float(grid_lon_deg[i0]) % 360.0
    # Approx degrees per row in lat (assumes ascending uniform-ish lat).
    dlat = (
        float(grid_lat_deg[-1] - grid_lat_deg[0]) / max(n_lat - 1, 1)
        if n_lat > 1 else 1.0
    )
    dlon = 360.0 / max(n_lon, 1)
    half_j = max(int(np.ceil(max_deg / max(abs(dlat), 1e-6))), 1)
    half_i = max(int(np.ceil(max_deg / max(abs(dlon), 1e-6))), 1)
    best = None
    best_dist = np.inf
    max_sq = float(max_deg) ** 2
    for dj in range(-half_j, half_j + 1):
        j = j0 + dj
        if j < 0 or j >= n_lat:
            continue
        dlat_abs = abs(float(grid_lat_deg[j]) - lat0)
        if dlat_abs > max_deg:
            continue
        for di in range(-half_i, half_i + 1):
            i = (i0 + di) % n_lon
            if not bool(ocean_mask[j, i]):
                continue
            lon_diff = abs(float(grid_lon_deg[i]) % 360.0 - lon0)
            dlon_abs = min(lon_diff, 360.0 - lon_diff)
            # Enforce the documented radial 5°-box cutoff so coarse
            # grids do not let ``ceil`` widen the search beyond the
            # API contract.
            if dlon_abs > max_deg:
                continue
            d = dlat_abs ** 2 + dlon_abs ** 2
            if d > max_sq:
                continue
            if d < best_dist:
                best_dist = d
                best = (j, i)
    if best is None:
        return None, None
    return best


def project_runoff_to_mpas_cells(
    rivers: RiverRunoffData,
    *,
    lat_cell_deg: np.ndarray,
    lon_cell_deg: np.ndarray,
    area_cell_m2: np.ndarray,
    month: Optional[int] = None,
    ocean_mask: Optional[np.ndarray] = None,
    max_search_deg: float = 5.0,
) -> np.ndarray:
    """Bin river mouths onto an MPAS (unstructured) mesh as a [kg/m²/s] field.

    Unstructured counterpart of :func:`project_runoff_to_grid`. Where the
    lat-lon version bins onto a tensor-product ``(n_lat, n_lon)`` mesh (nearest
    lat/lon cell, then relocate within a planar box if that cell is land), this
    bins onto a flat ``(nCells,)`` cell list and assigns each river to the
    globally nearest OCEAN cell by great-circle distance
    (:func:`legoesm.ocean.bathymetry.haversine_km`). The two are ANALOGOUS, not
    identical: on a coastline the structured "nearest-then-relocate" and the
    unstructured "nearest ocean" can pick different cells, and the cutoff is
    measured from the river (here) vs from the selected land cell (lat-lon).
    Both conserve the flux they accept and drop a river with no ocean cell
    within ``max_search_deg``.

    Conservation: total mass flux is preserved except for rivers dropped for
    lack of a nearby ocean cell. ``sum(out * area_cell_m2)`` over ocean cells
    equals the summed flux of the ASSIGNED rivers, exactly as the lat-lon
    version conserves ``sum(out * cell_area)``.

    Parameters
    ----------
    rivers : RiverRunoffData
    lat_cell_deg : ndarray ``(nCells,)`` — cell-centre latitude [deg N].
    lon_cell_deg : ndarray ``(nCells,)`` — cell-centre longitude [deg E].
    area_cell_m2 : ndarray ``(nCells,)`` — cell area [m²].
    month : int or None — 1–12 monthly snapshot; None for the annual mean.
    ocean_mask : ndarray ``(nCells,)`` of {0, 1} or None
        1 = ocean. Rivers are assigned only to ocean cells; with no mask every
        cell is a candidate.
    max_search_deg : float
        Radial cutoff [deg] converted to km at ~111 km/deg; a river with no
        ocean cell within this radius is dropped.

    Returns
    -------
    runoff_kg_m2_s : ndarray ``(nCells,)``
        Freshwater flux into the ocean [kg/m²/s], non-negative.
    """
    from legoesm import constants
    from legoesm.ocean.bathymetry import haversine_km

    lat_c = np.asarray(lat_cell_deg, dtype=np.float64)
    lon_c = np.asarray(lon_cell_deg, dtype=np.float64) % 360.0
    area = np.asarray(area_cell_m2, dtype=np.float64)
    n_cells = lat_c.shape[0]
    out = np.zeros(n_cells, dtype=np.float64)

    # Candidate cells = ocean cells (or all cells when no mask given).
    if ocean_mask is None:
        cand = np.arange(n_cells)
    else:
        cand = np.nonzero(np.asarray(ocean_mask).astype(bool).reshape(-1))[0]
    if cand.size == 0:
        return out  # no ocean cells -> nothing to receive runoff
    cand_lat = lat_c[cand]
    cand_lon = lon_c[cand]

    # Flux per river [kg/s].
    if month is None:
        flux_per_river = rivers.monthly_flux_kg_s.mean(axis=0)
    else:
        flux_per_river = rivers.monthly_flux_kg_s[(month - 1) % 12]

    # Cutoff in km via the SAME Earth radius haversine_km uses, so the radial
    # test is exact rather than a ~0.18%-short 111 km/deg approximation that
    # would open a narrow false-drop band at the boundary (codex).
    km_per_deg = float(constants.R_earth) * 1.0e-3 * (np.pi / 180.0)
    max_km = float(max_search_deg) * km_per_deg
    r_lat = np.asarray(rivers.latitudes, dtype=np.float64)
    r_lon = np.asarray(rivers.longitudes, dtype=np.float64) % 360.0
    for k in range(r_lat.shape[0]):
        d = haversine_km(r_lat[k], r_lon[k], cand_lat, cand_lon)
        m = int(np.argmin(d))
        if float(d[m]) > max_km:
            continue  # no ocean cell within the search radius -> drop
        cidx = int(cand[m])
        cell = float(area[cidx])
        if cell <= 0.0:
            continue
        out[cidx] += float(flux_per_river[k]) / cell
    return out
