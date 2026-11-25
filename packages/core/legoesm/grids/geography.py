"""One geography file for every component.

A regular lat-lon NetCDF (global attribute ``legoesm_geography_version = 1``):

- ``elevation(lat, lon)`` [m] -- land above 0, ocean depth below (required);
- ``land_fraction`` [0, 1] -- on ``(lat, lon)`` or on its own coordinates
  (e.g. ``(lat_lf, lon_lf)``, a present-day sftlf at native resolution);
  absent -> the area fraction with ``elevation > 0``;
- ``barrier_lat`` / ``barrier_lon`` ``(barrier, vertex)`` [deg, NaN padded]
  -- polylines (great-circle segments) closed on every model grid: every
  cell owning a densified point of the line (step <= min(0.05 deg, 1/4 of
  the finest cell)) becomes land at elevation 0 (kept out of the terrain
  smoothing), and a barrier whose consecutive points jump between
  non-neighbour cells is refused, so the closed cells form a face-connected
  wall.  An arc that clips a cell corner between two points need not close
  that cell (the wall stays watertight, displaced by under one step);
- CLM-surfdata variables (CLM names and dims) -- the land surface; absent ->
  uniform C3 grassland (LAI 2) on loam.

Lakes (``lake_fraction``) and ice sheets (``ice_sheet_thickness``) are
refused for now.

Regrid modes (``ExperimentConfig.geography_regrid``):
  ``legacy`` -- elevation binned + land fraction bilinear: today's
               ``--topography`` + ``--land-mask-file`` path, bitwise;
  ``binned`` -- land fraction binned area-weighted with the elevation.
"""
from __future__ import annotations

import numpy as np

GEOGRAPHY_VERSION = 1
REGRID_MODES = ("legacy", "binned")
_REFUSED_VARS = ("ice_sheet_thickness", "lake_fraction")
# a land map is PCT_NAT_PFT plus these; any of them without PCT_NAT_PFT would
# be silently replaced by the uniform default, so it is refused (VOTED VC)
_CLM_MAP_VARS = ("PCT_NATVEG", "PCT_CROP", "PCT_CFT", "PCT_GLACIER", "PCT_SAND",
                 "PCT_CLAY", "SOIL_COLOR", "MONTHLY_LAI")

# --- barrier densification (VOTED V4: min(0.05 deg, 1/4 finest cell); leak
# check = consecutive points in the same or neighbouring cells, VOTED V14) ---
_DENSIFY_MAX_DEG = 0.05
_DENSIFY_CELL_FRACTION = 0.25

# --- uniform grassland where the file has no land map (user 2026-11-20: slab
# land + two-leaf canopy, grass; VOTED G1 C3 grass PFT 13 a/a/?, G2 constant
# MONTHLY_LAI 2.0 b/b/a, G3 soil = V6 loam centroid + colour 15 a/a/a) ---
_GRASS_PFT = 13               # CLM5 c3_grass (natural-PFT index)
_GRASS_LAI = 2.0              # [m2 m-2], every month
_N_NAT_PFT, _N_CFT, _N_MONTHS = 15, 2, 12
_SOIL_SAND_PCT = 43.0
_SOIL_CLAY_PCT = 18.0
_SOIL_COLOR = 15
_SOIL_LEVELS = 10   # CLM5 nlevsoi; the loader averages the top root-zone layers
# --- the same grass as a harmonized legoesm_surfdata dataset, for the
# two-leaf canopy on the atmosphere lane (VOTED W4 a/a/a, W5 a/a/a).  Values
# not set above come from the production harmonized file
# data/legoesm_surfdata_c260716.nc: C3-grass canopy top/bottom (uniform there
# = CLM constants), median annual-mean SAI over C3-grass-dominated cells, and
# per-layer global land medians of organic matter and bulk density on its
# 7-layer soil grid. ---
_GRASS_HTOP_M, _GRASS_HBOT_M = 0.5, 0.01
_GRASS_SAI = 0.2412291020154953
_SURF_SOIL_DZ_M = (0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5)
_SURF_ORGANIC_KG_M3 = (1.8839999437332153, 1.0223841667175293, 0.8671855330467224,
                       0.7176077961921692, 0.6410700678825378, 0.594687283039093,
                       0.4879292845726013)
_SURF_BULK_DENSITY_KG_M3 = (1342.104248046875, 1408.5548095703125, 1438.8062744140625,
                            1459.920654296875, 1471.047607421875, 1493.966796875,
                            1539.11083984375)
_SURF_GRID_DEG = 10.0
_N_PFT = _N_NAT_PFT + _N_CFT


def open_geography(path: str):
    """Open and validate a geography file; returns the ``xarray.Dataset``."""
    import xarray as xr
    ds = xr.open_dataset(path)
    errs = []
    if ds.attrs.get("legoesm_geography_version") != GEOGRAPHY_VERSION:
        errs.append(f"global attribute legoesm_geography_version != {GEOGRAPHY_VERSION}")
    for v in _REFUSED_VARS:
        if v in ds.variables:
            errs.append(f"{v} is not supported yet")
    if "elevation" not in ds:
        errs.append("no 'elevation' variable")
    else:
        el = ds["elevation"]
        if el.dims != ("lat", "lon"):
            errs.append(f"elevation dims {el.dims} != ('lat', 'lon')")
        if el.attrs.get("units") != "m":
            errs.append(f"elevation units {el.attrs.get('units')!r} != 'm'")
        if not np.all(np.isfinite(el.values)):
            errs.append("elevation must be finite everywhere (no missing values)")
    if "land_fraction" in ds:
        lf = ds["land_fraction"]
        if lf.ndim != 2:
            errs.append(f"land_fraction dims {lf.dims} are not (lat, lon)-like")
        else:
            for d in lf.dims:
                if d not in ds.coords:
                    errs.append(f"land_fraction dimension {d!r} has no coordinate")
        x = np.asarray(lf.values, dtype=np.float64)
        if not (np.all(np.isfinite(x)) and x.min() >= 0.0 and x.max() <= 1.0):
            errs.append("land_fraction must be finite and in [0, 1] (a fraction, not %)")
    for c in {"lat", "lon"} | (set(ds["land_fraction"].dims) if "land_fraction" in ds else set()):
        if c in ds.coords:
            d = np.diff(np.asarray(ds[c].values, dtype=np.float64))
            if not (np.all(d > 0) or np.all(d < 0)):
                errs.append(f"coordinate {c!r} is not strictly monotone")
    partial = [v for v in _CLM_MAP_VARS if v in ds.variables]
    if partial and "PCT_NAT_PFT" not in ds.variables:
        errs.append(f"land-map variables {partial} without PCT_NAT_PFT (a partial "
                    "land map would be replaced by the uniform default)")
    has_bl, has_bo = "barrier_lat" in ds, "barrier_lon" in ds
    if has_bl != has_bo:
        errs.append("barrier_lat and barrier_lon come together")
    elif has_bl:
        bl, bo = np.asarray(ds["barrier_lat"].values), np.asarray(ds["barrier_lon"].values)
        if bl.shape != bo.shape or bl.ndim != 2:
            errs.append("barrier_lat/barrier_lon must share one (barrier, vertex) shape")
        elif not np.array_equal(np.isnan(bl), np.isnan(bo)):
            errs.append("barrier_lat/barrier_lon NaN padding differs")
        elif np.any(np.sum(~np.isnan(bl), axis=1) < 2):
            errs.append("every barrier needs at least 2 vertices")
        elif np.nanmax(np.abs(bl)) > 90.0:
            errs.append("barrier_lat outside [-90, 90]")
    if errs:
        ds.close()
        raise ValueError(f"geography file {path}: " + "; ".join(errs))
    return ds


def geography_has_land(path: str) -> bool:
    """Any land in the file: a barrier, else ``land_fraction > 0`` when the
    file carries one (it overrides the elevation), else ``elevation > 0``.
    The same answer on every MPI rank, unlike a rank-local land fraction."""
    ds = open_geography(path)
    try:
        lf = ds["land_fraction"] if "land_fraction" in ds else ds["elevation"]
        return bool("barrier_lat" in ds or np.any(lf.values > 0.0))
    finally:
        ds.close()


def barrier_polylines(ds) -> list:
    """``[(lat_deg, lon_deg), ...]`` one pair of vertex arrays per barrier."""
    if "barrier_lat" not in ds:
        return []
    bl = np.asarray(ds["barrier_lat"].values, dtype=np.float64)
    bo = np.asarray(ds["barrier_lon"].values, dtype=np.float64)
    return [(a[~np.isnan(a)], b[~np.isnan(b)]) for a, b in zip(bl, bo)]


def _unit(lat_deg, lon_deg):
    la, lo = np.deg2rad(np.asarray(lat_deg)), np.deg2rad(np.asarray(lon_deg))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _densify(lat_deg, lon_deg, step_rad):
    """Points along each great-circle segment at most ``step_rad`` apart
    (``(m, 3)`` unit vectors, segment end points included)."""
    v = _unit(lat_deg, lon_deg)
    pts = [v[:1]]
    for a, b in zip(v[:-1], v[1:]):
        ang = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
        n = max(1, int(np.ceil(ang / step_rad)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        if ang < 1e-12:
            pts.append(np.repeat(b[None], n, 0))
            continue
        pts.append((np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang))
    return np.concatenate(pts)


def _grid_cells(grid):
    """Flattened cell centres (rad), unit-sphere area weights, neighbours."""
    from legoesm.grids.topography import neighbour_table
    lat = np.asarray(grid.grid_lat, dtype=np.float64).ravel()
    lon = np.asarray(grid.grid_lon, dtype=np.float64).ravel()
    table = neighbour_table(grid)
    if table is None:
        raise ValueError("geography barriers need a grid with a neighbour table "
                         "(this mesh has none)")
    return lat, lon, table[0], np.asarray(table[1], dtype=np.float64).ravel()


def barrier_owners(grid, polylines) -> list:
    """Per barrier, the owner cell (the binning's ownership rule) of every
    densified point, in order along the barrier (step <= min(0.05 deg, 1/4
    of the finest cell))."""
    from legoesm.grids.topography import grid_ownership, point_owner
    lat, lon, _, area = _grid_cells(grid)
    finest = float(np.sqrt(4.0 * np.pi * area.min() / area.sum()))
    step = min(np.deg2rad(_DENSIFY_MAX_DEG), _DENSIFY_CELL_FRACTION * finest)
    own = grid_ownership(grid)
    out = []
    for bl, bo in polylines:
        p = _densify(bl, bo, step)
        out.append(point_owner(np.arcsin(np.clip(p[:, 2], -1, 1)),
                               np.arctan2(p[:, 1], p[:, 0]), lat, lon, **own))
    return out


def closed_barrier_cells(grid, owners, width_cells: int = 1) -> np.ndarray:
    """Bool ``(n_cells,)``: every cell a barrier touches (``owners``), widened
    to ``width_cells`` total cells (odd: ``(width_cells - 1) / 2`` neighbour
    rings each side)."""
    if width_cells < 1 or width_cells % 2 == 0:
        raise ValueError(f"barrier width {width_cells} must be an odd number of cells >= 1")
    lat, _, nb, _ = _grid_cells(grid)
    closed = np.zeros(lat.size, dtype=bool)
    for o in owners:
        closed[o] = True
    valid = nb >= 0
    for _ in range((width_cells - 1) // 2):
        closed = closed | np.any(np.where(valid, closed[np.where(valid, nb, 0)], False), axis=0)
    return closed


def barrier_gaps(grid, owners) -> int:
    """Consecutive barrier points whose owners are neither the same cell nor
    neighbours (0 = every face the barrier crosses has a closed cell on a
    mesh with 3 cells per vertex; a nonzero count is refused)."""
    _, _, nb, _ = _grid_cells(grid)
    gaps = 0
    for o in owners:
        a, b = o[:-1], o[1:]
        gaps += int(np.sum((a != b) & ~np.any(nb[:, a] == b[None, :], axis=0)))
    return gaps


def _require_canonical_axes(lat, lon, what: str):
    """Fields binned straight from the file (no reordering) need the terrain
    loader's own axis convention, else the two would bin differently."""
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
            and lon[0] >= 0.0 and lon[-1] < 360.0):
        raise ValueError(f"{what} needs ascending lat and lon ascending in [0, 360) "
                         "(no reordering or periodic de-duplication is applied)")


def geography_terrain(grid, path: str, mode: str, *, smoothing_passes: int,
                      barrier_width_cells: int = 1):
    """``(phis [m2 s-2], f_land)`` of ``grid`` from a geography file, shaped
    like the grid (as :func:`~legoesm.grids.topography.load_real_topography`)."""
    from legoesm.grids.topography import TopographyConfig, load_real_topography
    if mode not in REGRID_MODES:
        raise ValueError(f"geography regrid mode {mode!r} not in {REGRID_MODES}")
    ds = open_geography(path)
    try:
        lines = barrier_polylines(ds)
        has_lf = "land_fraction" in ds
        lf_dims = ds["land_fraction"].dims if has_lf else ()
        f_src = None
        if mode == "binned" and has_lf:
            lat, lon = (np.asarray(ds[c].values, dtype=np.float64) for c in ("lat", "lon"))
            if lf_dims != ("lat", "lon"):
                raise ValueError("geography_regrid='binned' needs land_fraction on "
                                 f"(lat, lon), got {lf_dims}")
            _require_canonical_axes(lat, lon, "geography_regrid='binned' land_fraction")
            f_src = np.asarray(ds["land_fraction"].values, dtype=np.float64)
    finally:
        ds.close()
    closed = None
    if lines:
        owners = barrier_owners(grid, lines)
        n_gap = barrier_gaps(grid, owners)
        if n_gap:
            raise ValueError(
                f"geography {path}: a barrier jumps between non-neighbour cells at "
                f"{n_gap} points on this grid (it could leak); refusing")
        closed = closed_barrier_cells(grid, owners, barrier_width_cells)
    cfg = TopographyConfig(
        source="file", path=path, elev_var="elevation", lat_var="lat", lon_var="lon",
        smoothing_passes=smoothing_passes,
        land_mask_path=path if (has_lf and mode == "legacy") else "",
        land_mask_var="land_fraction",
        land_mask_lat_var=lf_dims[0] if has_lf else "",
        land_mask_lon_var=lf_dims[1] if has_lf else "")
    phis, f_land = load_real_topography(grid, config=cfg, f_land_src=f_src,
                                        closed_cells=closed)
    return phis, f_land


def geography_ocean_depth(grid, path: str, *, H_min: float, H_max: float):
    """Ocean depth [m, positive down] on every ``grid`` cell: the area-binned
    mean of ``-elevation`` over the cell's ocean source points (VOTED D1
    a/a/a), clipped to ``[H_min, H_max]`` -- a cell with no ocean source point
    gets ``H_min`` (D2 a/a/a), deeper than the z-coordinate bottom is clamped
    (D3 b/a/b); no smoothing (D4 a/a/a).  Which cells are wet is the caller's
    (the co-located ocean: atmosphere ``f_land < 1``, user 2026-11-20)."""
    from legoesm.grids.topography import bin_latlon_to_cells, grid_ownership
    ds = open_geography(path)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        el = np.asarray(ds["elevation"].values, dtype=np.float64)
        lf = None
        if "land_fraction" in ds:
            if ds["land_fraction"].dims != ("lat", "lon"):
                raise ValueError(
                    f"geography {path}: ocean depth needs land_fraction on the "
                    f"elevation grid (lat, lon), got {ds['land_fraction'].dims}")
            lf = np.asarray(ds["land_fraction"].values, dtype=np.float64)
    finally:
        ds.close()
    _require_canonical_axes(lat, lon, "geography ocean depth")  # VOTED VA
    # below-sea-level LAND (land_fraction 1) is not ocean (VOTED a/a/a)
    sea = (el < 0.0) * (1.0 if lf is None else 1.0 - lf)
    c_lat, c_lon, _, _ = _grid_cells(grid)
    sea_c, depth_sum = bin_latlon_to_cells(lat, lon, [sea, sea * -el], c_lat, c_lon,
                                           **grid_ownership(grid))
    depth = np.divide(depth_sum, sea_c, out=np.zeros_like(sea_c), where=sea_c > 0.0)
    return np.clip(depth, H_min, H_max)


def geography_clm_surface(path: str):
    """The land-surface map of a geography file as a CLM-surfdata
    ``xarray.Dataset`` (for :func:`~legoesm.land.clm_surface_map.load_clm_surface`):
    the file's own CLM variables, else uniform C3 grassland."""
    import xarray as xr
    ds = open_geography(path)
    if "PCT_NAT_PFT" in ds:
        return ds
    ds.close()
    one = np.ones((1, 1))
    lev = np.ones((_SOIL_LEVELS, 1, 1))
    nat = np.zeros((_N_NAT_PFT, 1, 1))
    nat[_GRASS_PFT] = 100.0
    lai = np.zeros((_N_MONTHS, _N_NAT_PFT + _N_CFT, 1, 1))
    lai[:, _GRASS_PFT] = _GRASS_LAI
    return xr.Dataset(
        {"PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), nat),
         "PCT_NATVEG": (("lsmlat", "lsmlon"), 100.0 * one),
         "PCT_CROP": (("lsmlat", "lsmlon"), 0.0 * one),
         "PCT_CFT": (("cft", "lsmlat", "lsmlon"), np.full((_N_CFT, 1, 1), 50.0)),
         "PCT_GLACIER": (("lsmlat", "lsmlon"), 0.0 * one),
         "MONTHLY_LAI": (("time", "lsmpft", "lsmlat", "lsmlon"), lai),
         "PCT_SAND": (("nlevsoi", "lsmlat", "lsmlon"), _SOIL_SAND_PCT * lev),
         "PCT_CLAY": (("nlevsoi", "lsmlat", "lsmlon"), _SOIL_CLAY_PCT * lev),
         "SOIL_COLOR": (("lsmlat", "lsmlon"), np.full((1, 1), _SOIL_COLOR))},
        coords={"lsmlat": [0.0], "lsmlon": [0.0]})


def geography_land_surfdata(path: str):
    """The grass land surface of a geography file with no land map of its own,
    as a harmonized ``legoesm_surfdata`` dataset (the two-leaf canopy's per-PFT
    parameters); ``None`` when the file carries its own CLM land map."""
    from legoesm.land.surface_data.schema import surfdata_dataset
    ds = open_geography(path)
    own = "PCT_NAT_PFT" in ds
    ds.close()
    if own:
        return None
    h = 0.5 * _SURF_GRID_DEG
    lat = np.arange(-90.0 + h, 90.0, _SURF_GRID_DEG)
    lon = np.arange(h, 360.0, _SURF_GRID_DEG)
    plane = np.ones((lat.size, lon.size))
    nl = len(_SURF_SOIL_DZ_M)
    soil = lambda v: np.asarray(v, dtype=np.float64)[:, None, None] * plane

    def veg(v):
        a = np.zeros((_N_MONTHS, _N_PFT, lat.size, lon.size))
        a[:, _GRASS_PFT] = v
        return a

    pft = np.zeros((1, _N_PFT, lat.size, lon.size))
    pft[0, _GRASS_PFT] = 100.0
    return surfdata_dataset(
        lat=lat, lon=lon, soil_dz=np.asarray(_SURF_SOIL_DZ_M),
        sand_pct=soil([_SOIL_SAND_PCT] * nl), clay_pct=soil([_SOIL_CLAY_PCT] * nl),
        organic=soil(_SURF_ORGANIC_KG_M3), bulk_density=soil(_SURF_BULK_DENSITY_KG_M3),
        soil_color=np.full(plane.shape, _SOIL_COLOR),
        f_land=100.0 * plane[None], f_lake=0.0 * plane[None], f_glacier=0.0 * plane[None],
        pft_frac=pft, monthly_lai=veg(_GRASS_LAI), monthly_sai=veg(_GRASS_SAI),
        monthly_height_top=veg(_GRASS_HTOP_M), monthly_height_bot=veg(_GRASS_HBOT_M),
        source=f"legoesm geography uniform C3 grass ({path})")
