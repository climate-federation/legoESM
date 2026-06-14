"""Global surface-data loader for AMIP land (plan ``docs/land_canopy_amip_plan.md`` §10/M5).

Loads geophysical **boundary data** (NOT model parameters), regrids every field
to the model grid **once** (host-side, weights cached), and assembles a single
:class:`GlobalSurfaceData` container.  Three data categories:

1. **Static soil properties** (per model soil layer): sand/clay/organic/bulk
   density + an integer soil-colour class.
2. **Dynamic surface-type fractions** (annual transient land-use): land / lake /
   glacier gridcell fractions + within-land PFT cover fractions.  The PFT axis is
   preserved (tile-ready, §10a).
3. **Dynamic vegetation state** (monthly climatology): LAI / SAI / canopy height.

plus the static gridcell area.

Downstream consumers — pedotransfer (texture → hydraulic/thermal properties),
soil-colour → background albedo, the glacier coupler tile, and the active surface
scheme — are **separate sibling tasks**, deliberately not implemented here.  This
module's sole responsibility is *read → regrid → assemble*.

Reuse:
  - :func:`legoesm.grids.regridding.compute_latlon_to_voronoi_weights` /
    :func:`~legoesm.grids.regridding.regrid_scalar` — regular lat-lon source →
    any model grid (cubed-sphere / Gaussian / MPAS), via KD-tree + IDW.
  - :mod:`legoesm.forcing.amip` — the monthly-forcing read/regrid/time-interp
    pattern this mirrors.

Everything here is **host-side, load-once** (NumPy / xarray / scipy); the file
I/O is *not* differentiable.  The per-step time-interpolation helpers
(:func:`interp_annual`, :func:`interp_monthly`) are pure-JAX and traceable.

Call pattern — regrid ONCE, interpolate per step
------------------------------------------------
:func:`load_global_surface_data` is the **only** function that regrids, and it
runs **once at initialization**, outside the time loop / ``lax.scan``.  It must
NOT be called from inside a land step or any JIT/scan body — it uses xarray +
scipy KD-trees and cannot be traced.  No land subroutine should import it into
the hot loop.

After that one call, every field in :class:`GlobalSurfaceData` already lives on
the model grid:

  - **Soil properties** are static — extracted once, then frozen into the land
    params/state.  Never touched again, even for a transient run.
  - **LAI / SAI / canopy height** are stored as the 12 pre-regridded monthly
    maps ``(12, ncol, npft)``.  Per timestep you call
    ``interp_monthly(gsd.lai_monthly, day_of_year)`` — a cheap pure-JAX gather
    + linear blend of those stored slices (``O(ncol*npft)``).  **No regridding
    happens per step.**
  - **Transient cover fractions** ``(nyear, ncol, ...)`` are blended per year
    via ``interp_annual`` — again indexing pre-regridded slices, no regrid.

Recommended wiring (mirrors :class:`legoesm.forcing.amip.AMIPForcing` /
``SegmentForcing``): load once at setup, carry the monthly/annual arrays as an
explicit *traced argument* into the scan, and call ``interp_*`` inside the step.
Passing the arrays as args (not closure captures) avoids recompiles as time
advances.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import jax.numpy as jnp

from legoesm.grids.regridding import (
    RegridWeights,
    compute_latlon_to_voronoi_weights,
    conservative_regrid_latlon,
    regrid_scalar,
    regrid_scalar_nan_aware,
)
from legoesm.land.soil_grid import SoilGrid, SoilGridConfig, make_soil_grid

# Number of months in the monthly vegetation climatology.
_N_MONTH = 12
# Mid-month day-of-year centres (non-leap), used by ``interp_monthly``.
_MONTH_CENTRES = np.array(
    [15.5, 45.0, 74.5, 105.0, 135.5, 166.0,
     196.5, 227.5, 258.0, 288.5, 319.0, 349.5],
    dtype=np.float64,
)
# Gridcell area unit conversion (surfdata ``AREA`` is km²; model wants m²).
_KM2_TO_M2 = 1.0e6


# ===========================================================================
# Config + container
# ===========================================================================
class GlobalSurfaceDataConfig(NamedTuple):
    """Paths + source variable names for the surface-data loader.

    Mirrors :class:`legoesm.forcing.amip.AMIPForcingConfig`.  Variable-name
    fields let one config target CLM5 surfdata, MODIS, or a custom file without
    code changes.  All three ``*_path`` may point at the same file.
    """

    dataset: str = "clm5_surfdata"
    surf_path: str = ""        # static surfdata: soil + colour + area
    landuse_path: str = ""     # annual transient cover fractions
    veg_path: str = ""         # monthly LAI / SAI / height

    # --- soil (static) ---
    sand_var: str = "PCT_SAND"          # (nlevsoi, nlat, nlon) percent
    clay_var: str = "PCT_CLAY"          # (nlevsoi, nlat, nlon) percent
    organic_var: str = "ORGANIC"        # (nlevsoi, nlat, nlon) kg/m^3
    bulk_density_var: str = "BULK_DENSITY"
    color_var: str = "SOIL_COLOR"       # (nlat, nlon) int class
    soil_depth_var: str = "DZSOI"       # (nlevsoi,) source layer thicknesses [m]

    # --- cover fractions (annual transient) ---
    land_frac_var: str = "PCT_NATVEG"   # (nyear?, nlat, nlon) percent of gridcell
    lake_var: str = "PCT_LAKE"
    glacier_var: str = "PCT_GLACIER"
    pft_frac_var: str = "PCT_NAT_PFT"   # (nyear?, npft, nlat, nlon) percent of natveg
    year_var: str = "YEAR"

    # --- vegetation state (monthly climatology) ---
    lai_var: str = "MONTHLY_LAI"        # (12, npft, nlat, nlon)
    sai_var: str = "MONTHLY_SAI"
    htop_var: str = "MONTHLY_HEIGHT_TOP"
    hbot_var: str = "MONTHLY_HEIGHT_BOT"

    # --- gridcell area (static) ---
    area_var: str = "AREA"              # (nlat, nlon) km^2 (optional)

    # --- source grid coords ---
    lat_var: str = "lsmlat"
    lon_var: str = "lsmlon"

    # --- knobs ---
    pct_scale: float = 0.01             # percent -> fraction
    k_neighbors: int = 4                # IDW neighbours for regridding
    area_scale: float = _KM2_TO_M2      # AREA source-unit -> m^2 (CLM km^2); 1.0 if m^2


class GlobalSurfaceData(NamedTuple):
    """Geophysical surface boundary data on the model grid.

    Spatial dimension ``ncol`` is the flattened model grid (``6*n*n`` for
    cubed-sphere, ``n_lat*n_lon`` for Gaussian, ``nCells`` for MPAS).  ``npft``
    and the soil-layer axis are preserved for tile-readiness (§10a); a v1 driver
    calls :func:`dominant_pft_collapse` to reduce ``npft`` to 1.
    """

    # ---- static soil properties (model soil layers) ----
    sand_frac: jnp.ndarray       # (ncol, n_soil_layers) [0-1]
    clay_frac: jnp.ndarray       # (ncol, n_soil_layers) [0-1]
    organic: jnp.ndarray         # (ncol, n_soil_layers) [kg/m^3]
    bulk_density: jnp.ndarray    # (ncol, n_soil_layers) [kg/m^3]
    soil_color: jnp.ndarray      # (ncol,) int class

    # ---- static gridcell area ----
    cell_area: jnp.ndarray       # (ncol,) [m^2]

    # ---- dynamic surface-type fractions (annual transient) ----
    years: jnp.ndarray           # (nyear,)
    f_land: jnp.ndarray          # (nyear, ncol) [0-1]
    f_lake: jnp.ndarray          # (nyear, ncol) [0-1]
    f_glacier: jnp.ndarray       # (nyear, ncol) [0-1]
    pft_frac: jnp.ndarray        # (nyear, ncol, npft) [0-1], sum-to-1 over npft

    # ---- dynamic vegetation state (monthly climatology) ----
    months: jnp.ndarray          # (12,) day-of-year centres
    lai_monthly: jnp.ndarray     # (12, ncol, npft) [m^2/m^2]
    sai_monthly: jnp.ndarray     # (12, ncol, npft) [m^2/m^2]
    htop_monthly: jnp.ndarray    # (12, ncol, npft) [m]
    hbot_monthly: jnp.ndarray    # (12, ncol, npft) [m]

    config: GlobalSurfaceDataConfig


class _RawSurfaceFields(NamedTuple):
    """Source-grid arrays + coords, oriented spatial-leading (lat, lon, ...).

    The intermediate produced by the xarray reader and consumed by
    :func:`build_global_surface_data`.  Keeping the pure assembly separate from
    file I/O makes the regrid / remap / renormalize path unit-testable on
    synthetic NumPy data without real surfdata files.
    """

    src_lat: np.ndarray          # (nlat,) [deg]
    src_lon: np.ndarray          # (nlon,) [deg]
    src_soil_depth: np.ndarray   # (n_src_layer,) layer-midpoint depths [m]

    sand: np.ndarray             # (nlat, nlon, n_src_layer) [0-1]
    clay: np.ndarray             # (nlat, nlon, n_src_layer) [0-1]
    organic: np.ndarray          # (nlat, nlon, n_src_layer) [kg/m^3]
    bulk_density: np.ndarray     # (nlat, nlon, n_src_layer) [kg/m^3]
    soil_color: np.ndarray       # (nlat, nlon) int class

    years: np.ndarray            # (nyear,)
    f_land: np.ndarray           # (nyear, nlat, nlon) [0-1]
    f_lake: np.ndarray           # (nyear, nlat, nlon) [0-1]
    f_glacier: np.ndarray        # (nyear, nlat, nlon) [0-1]
    pft_frac: np.ndarray         # (nyear, nlat, nlon, npft) [0-1]

    lai: np.ndarray              # (12, nlat, nlon, npft) [m^2/m^2]
    sai: np.ndarray              # (12, nlat, nlon, npft)
    htop: np.ndarray             # (12, nlat, nlon, npft) [m]
    hbot: np.ndarray             # (12, nlat, nlon, npft) [m]

    cell_area: Optional[np.ndarray]  # (nlat, nlon) [m^2] or None -> grid metric


# ===========================================================================
# Presets
# ===========================================================================
def get_surfdata_preset(name: str) -> GlobalSurfaceDataConfig:
    """Return a config with preset variable names for a known dataset.

    Raises ``ValueError`` on an unknown name — never silently falls back to a
    default (CLAUDE.md dispatch discipline).
    """
    if name == "clm5_surfdata":
        return GlobalSurfaceDataConfig(dataset="clm5_surfdata")
    if name == "legoesm_surfdata":
        # Native harmonized file written by legoesm.land.surface_data.schema.
        # Clean 1-D lat/lon coords; cell_area already in m^2 (area_scale=1.0).
        return GlobalSurfaceDataConfig(
            dataset="legoesm_surfdata",
            sand_var="sand_pct", clay_var="clay_pct", organic_var="organic",
            bulk_density_var="bulk_density", color_var="soil_color",
            soil_depth_var="soil_dz",
            land_frac_var="f_land", lake_var="f_lake", glacier_var="f_glacier",
            pft_frac_var="pft_frac", year_var="year",
            lai_var="monthly_lai", sai_var="monthly_sai",
            htop_var="monthly_height_top", hbot_var="monthly_height_bot",
            area_var="cell_area", lat_var="lat", lon_var="lon",
            area_scale=1.0,
        )
    if name == "modis":
        # MODIS land-cover / LAI fallback (plan §12 open item #1).
        return GlobalSurfaceDataConfig(
            dataset="modis",
            pft_frac_var="PFT",
            lai_var="Lai",
            lat_var="lat",
            lon_var="lon",
        )
    raise ValueError(
        f"Unknown surfdata preset: {name!r}. "
        f"Use 'clm5_surfdata', 'modis', or build a custom GlobalSurfaceDataConfig."
    )


# ===========================================================================
# Target grid coordinates
# ===========================================================================
def _target_latlon_flat(grid) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Flattened (lat, lon) [radians] and the model grid's spatial shape.

    Handles Gaussian (``lat2d``/``lon2d``), cubed-sphere (``lat``/``lon``, both
    multi-D), and unstructured MPAS (``latCell``/``lonCell``).
    """
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat = np.asarray(grid.lat2d)
        lon = np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat = np.asarray(grid.latCell)
        lon = np.asarray(grid.lonCell)
    elif hasattr(grid, "lat") and hasattr(grid, "lon"):
        lat = np.asarray(grid.lat)
        lon = np.asarray(grid.lon)
        if lat.shape != lon.shape:
            raise ValueError(
                "Grid .lat/.lon must share shape for a point regrid target; "
                f"got {lat.shape} vs {lon.shape}. Provide .lat2d/.lon2d."
            )
    else:
        raise ValueError(
            "Grid exposes no recognised lat/lon attributes "
            "(.lat2d/.lon2d, .latCell/.lonCell, or matching-shape .lat/.lon)."
        )
    return lat.ravel(), lon.ravel(), tuple(int(s) for s in lat.shape)


def _regular_latlon_1d(grid):
    """Return ``(tgt_lat_deg, tgt_lon_deg)`` if ``grid`` is a regular lat-lon grid.

    Regular means ``lat2d`` is constant along each row and ``lon2d`` constant down
    each column.  ``grid.lat2d``/``lon2d`` are in **radians** (the regridder
    convention); returns degrees.  ``None`` for irregular grids (cubed-sphere,
    MPAS) which fall back to KD-tree IDW.
    """
    if not (hasattr(grid, "lat2d") and hasattr(grid, "lon2d")):
        return None
    lat2d = np.rad2deg(np.asarray(grid.lat2d, dtype=np.float64))
    lon2d = np.rad2deg(np.asarray(grid.lon2d, dtype=np.float64))
    if lat2d.ndim != 2:
        return None
    if np.allclose(lat2d, lat2d[:, :1]) and np.allclose(lon2d, lon2d[:1, :]):
        return lat2d[:, 0], lon2d[0, :]
    return None


def _build_weights(
    src_lat_deg: np.ndarray,
    src_lon_deg: np.ndarray,
    tgt_lat_rad: np.ndarray,
    tgt_lon_rad: np.ndarray,
    k_neighbors: int,
) -> RegridWeights:
    """KD-tree + IDW weights from a regular lat-lon source to the target points."""
    return compute_latlon_to_voronoi_weights(
        np.deg2rad(np.asarray(src_lat_deg, dtype=np.float64)),
        np.deg2rad(np.asarray(src_lon_deg, dtype=np.float64)),
        np.asarray(tgt_lat_rad, dtype=np.float64),
        np.asarray(tgt_lon_rad, dtype=np.float64),
        k_neighbors=k_neighbors,
    )


def _regrid(src_field: np.ndarray, weights: RegridWeights) -> jnp.ndarray:
    """Regrid a spatial-leading ``(nlat, nlon, *trailing)`` field → ``(ncol, *trailing)``.

    ``regrid_scalar`` flattens the leading spatial dims to ``src_flat_size`` and
    preserves trailing dims, so the source must be oriented with (lat, lon)
    first — matching the meshgrid order inside the weight builder.
    """
    return regrid_scalar(jnp.asarray(src_field, dtype=jnp.float32), weights)


def _regrid_nearest(src_field: np.ndarray, weights: RegridWeights) -> jnp.ndarray:
    """Nearest-neighbour regrid of a 2-D ``(nlat, nlon)`` field → ``(ncol,)``.

    Picks the single closest source cell (max IDW weight) rather than blending —
    correct for categorical fields like the integer soil-colour class, where an
    inverse-distance average of class indices is meaningless.
    """
    flat = jnp.asarray(src_field, dtype=jnp.float32).ravel()
    idx = weights.src_indices                                   # (ncol, k)
    nearest = idx[jnp.arange(idx.shape[0]), jnp.argmax(weights.weights, axis=-1)]
    return flat[nearest].reshape(weights.target_shape)


# ===========================================================================
# Soil vertical remap
# ===========================================================================
def _remap_soil_layers(
    field: jnp.ndarray,        # (ncol, n_src_layer)
    src_depth: np.ndarray,     # (n_src_layer,) midpoint depths [m]
    dst_depth: np.ndarray,     # (n_dst_layer,) midpoint depths [m]
) -> jnp.ndarray:
    """Linear interpolation in depth from source to model soil layers.

    Depths below/above the source range are clamped to the nearest source layer
    (``jnp.interp`` edge behaviour) — appropriate for texture, which is roughly
    constant outside the measured profile.
    """
    src_depth = np.asarray(src_depth, dtype=np.float64)
    dst_depth = jnp.asarray(dst_depth, dtype=field.dtype)
    order = np.argsort(src_depth)
    src_sorted = jnp.asarray(src_depth[order], dtype=field.dtype)
    field_sorted = field[:, order]
    # Interpolate each column independently along the depth axis.
    return jnp.stack(
        [jnp.interp(dst_depth, src_sorted, field_sorted[i]) for i in range(field.shape[0])],
        axis=0,
    )


# ===========================================================================
# Fraction renormalization (partition-of-unity)
# ===========================================================================
def _renorm_cover(
    f_land: jnp.ndarray, f_lake: jnp.ndarray, f_glacier: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Clip to [0,1] and scale down so land+lake+glacier ≤ 1 per cell.

    The remainder (1 - sum) is implicitly ocean/other.  Only cells whose loaded
    fractions sum to > 1 (interpolation overshoot) are rescaled; valid cells are
    untouched.  Mirrors the conservative guard in ``coupler.tile_fractions``.
    """
    f_land = jnp.clip(f_land, 0.0, 1.0)
    f_lake = jnp.clip(f_lake, 0.0, 1.0)
    f_glacier = jnp.clip(f_glacier, 0.0, 1.0)
    total = f_land + f_lake + f_glacier
    scale = jnp.where(total > 1.0, 1.0 / jnp.maximum(total, 1.0), 1.0)
    return f_land * scale, f_lake * scale, f_glacier * scale


def _renorm_pft(pft_frac: jnp.ndarray) -> jnp.ndarray:
    """Normalize PFT fractions to sum to 1 over the last axis where any cover exists.

    Bare-soil columns (all-zero) stay all-zero rather than being forced to a
    spurious uniform split.
    """
    pft_frac = jnp.clip(pft_frac, 0.0, 1.0)
    total = jnp.sum(pft_frac, axis=-1, keepdims=True)
    return jnp.where(total > 0.0, pft_frac / jnp.maximum(total, 1e-30), pft_frac)


# ===========================================================================
# Pure assembly (testable without files)
# ===========================================================================
def build_global_surface_data(
    raw: _RawSurfaceFields,
    grid,
    config: GlobalSurfaceDataConfig,
    soil_grid: SoilGrid | None = None,
) -> GlobalSurfaceData:
    """Regrid + remap + renormalize source fields into a :class:`GlobalSurfaceData`.

    Pure with respect to file I/O — operates on the NumPy arrays in ``raw`` — so
    it is exercised directly by unit tests on synthetic data.
    """
    if soil_grid is None:
        soil_grid = make_soil_grid(SoilGridConfig())

    tgt_lat, tgt_lon, target_shape = _target_latlon_flat(grid)
    ncol = int(np.prod(target_shape))
    # KD-tree weights are still used for the categorical soil-colour nearest pick.
    w = _build_weights(raw.src_lat, raw.src_lon, tgt_lat, tgt_lon, config.k_neighbors)

    # Continuous fields use CONSERVATIVE, NaN-aware regridding when the target is a
    # regular lat-lon grid (area-weighted; ocean NaNs never bleed into coastal
    # cells).  Irregular targets (cubed-sphere/MPAS) fall back to KD-tree IDW.
    _reg = _regular_latlon_1d(grid)
    if _reg is not None:
        _slat = np.asarray(raw.src_lat, dtype=np.float64)
        _slon = np.asarray(raw.src_lon, dtype=np.float64)
        _tlat, _tlon = _reg

        def _rg(field):
            a = np.asarray(field, dtype=np.float64)
            trailing = a.shape[2:]
            ll = int(np.prod(trailing)) if trailing else 1
            out = conservative_regrid_latlon(
                a.reshape(a.shape[0], a.shape[1], ll), _slat, _slon, _tlat, _tlon)
            flat = out.reshape(-1, ll)                      # (ncol, ll)
            return jnp.asarray(flat if trailing else flat[:, 0])
    else:
        # Irregular target (cubed-sphere / MPAS): NaN-aware KD-tree IDW to cell
        # centres (matches the model's point-based topography regrid, but won't
        # bleed ocean NaN into coastal cells).
        def _rg(field):
            return regrid_scalar_nan_aware(jnp.asarray(field, dtype=jnp.float32), w)

    # --- soil: regrid (ncol, n_src_layer) then remap to model layers ---
    dst_depth = np.asarray(soil_grid.z_node)

    def _soil(field):
        regridded = _rg(field)                              # (ncol, n_src_layer)
        return _remap_soil_layers(regridded, raw.src_soil_depth, dst_depth)

    sand = _soil(raw.sand)
    clay = _soil(raw.clay)
    organic = _soil(raw.organic)
    bulk_density = _soil(raw.bulk_density)
    # Soil colour is an integer class — nearest source cell, not IDW blend.
    soil_color = jnp.round(_regrid_nearest(raw.soil_color, w)).astype(jnp.int32)

    # --- cover fractions (nyear, ncol) ---
    def _frac_annual(field):  # source (nyear, nlat, nlon) -> (ncol, nyear) -> (nyear, ncol)
        src = np.moveaxis(np.asarray(field), 0, -1)         # (nlat, nlon, nyear)
        out = _rg(src)                                      # (ncol, nyear)
        return jnp.moveaxis(out, -1, 0)                     # (nyear, ncol)

    f_land = _frac_annual(raw.f_land)
    f_lake = _frac_annual(raw.f_lake)
    f_glacier = _frac_annual(raw.f_glacier)
    f_land, f_lake, f_glacier = _renorm_cover(f_land, f_lake, f_glacier)

    # pft_frac: source (nyear, nlat, nlon, npft) -> (nyear, ncol, npft).
    # regrid_scalar collapses multi-dim trailing to one axis, so reshape back.
    pft_src = np.moveaxis(np.asarray(raw.pft_frac), 0, 2)   # (nlat, nlon, nyear, npft)
    nyear, npft = pft_src.shape[2], pft_src.shape[3]
    pft_out = _rg(pft_src).reshape((ncol, nyear, npft))
    pft_frac = _renorm_pft(jnp.moveaxis(pft_out, 0, 1))     # (nyear, ncol, npft)

    # --- vegetation state (12, ncol, npft) ---
    def _veg_monthly(field):  # source (12, nlat, nlon, npft) -> (12, ncol, npft)
        src = np.moveaxis(np.asarray(field), 0, 2)          # (nlat, nlon, 12, npft)
        npft_v = src.shape[3]
        out = _rg(src).reshape((ncol, _N_MONTH, npft_v))
        return jnp.moveaxis(out, 0, 1)                      # (12, ncol, npft)

    lai = _veg_monthly(raw.lai)
    sai = _veg_monthly(raw.sai)
    htop = _veg_monthly(raw.htop)
    hbot = _veg_monthly(raw.hbot)

    # --- gridcell area ---
    if raw.cell_area is not None:
        cell_area = _rg(raw.cell_area)                      # (ncol,)
    else:
        cell_area = jnp.asarray(grid.grid_area, dtype=jnp.float32).ravel()
        if cell_area.shape[0] != ncol:
            raise ValueError(
                f"grid.grid_area has {cell_area.shape[0]} cells, expected {ncol}."
            )

    return GlobalSurfaceData(
        sand_frac=sand, clay_frac=clay, organic=organic,
        bulk_density=bulk_density, soil_color=soil_color,
        cell_area=cell_area,
        years=jnp.asarray(raw.years, dtype=jnp.float32),
        f_land=f_land, f_lake=f_lake, f_glacier=f_glacier, pft_frac=pft_frac,
        months=jnp.asarray(_MONTH_CENTRES, dtype=jnp.float32),
        lai_monthly=lai, sai_monthly=sai, htop_monthly=htop, hbot_monthly=hbot,
        config=config,
    )


# ===========================================================================
# xarray reader
# ===========================================================================
def _orient_latlon_last(da, lat_dim: str, lon_dim: str):
    """Transpose an xarray DataArray so (..., lat, lon) -> spatial-leading numpy.

    Returns a NumPy array with axes ``(lat, lon, *trailing)``.
    """
    other = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(lat_dim, lon_dim, *other).values
    return np.asarray(arr, dtype=np.float64)


def load_global_surface_data(
    config: GlobalSurfaceDataConfig,
    grid,
    *,
    soil_grid: SoilGrid | None = None,
    surf_ds=None,
    landuse_ds=None,
    veg_ds=None,
) -> GlobalSurfaceData:
    """Read surfdata NetCDF, regrid to ``grid``, return :class:`GlobalSurfaceData`.

    Datasets may be injected (``surf_ds`` / ``landuse_ds`` / ``veg_ds``) for
    testing; otherwise they are opened from the corresponding ``*_path`` in
    ``config`` (falling back to ``surf_path`` when a specialized path is empty).
    Host-side and load-once; not differentiable.
    """
    import xarray as xr  # lazy: keep module import clean and xarray optional

    opened: list = []

    def _resolve(ds, path):
        nonlocal opened
        if ds is not None:
            return ds
        if not path:
            path = config.surf_path
        if not path:
            raise ValueError("No dataset and no path supplied for a required file.")
        d = xr.open_dataset(path)
        opened.append(d)
        return d

    try:
        sds = _resolve(surf_ds, config.surf_path)
        lds = _resolve(landuse_ds, config.landuse_path)
        vds = _resolve(veg_ds, config.veg_path)

        lat_dim, lon_dim = config.lat_var, config.lon_var
        src_lat = np.asarray(sds[lat_dim].values, dtype=np.float64)
        src_lon = np.asarray(sds[lon_dim].values, dtype=np.float64)
        s = config.pct_scale

        # Source soil-layer depths: explicit DZSOI -> cumulative midpoints, else
        # fall back to the model soil grid's node depths (1:1 layer assumption).
        if config.soil_depth_var in sds:
            dz = np.asarray(sds[config.soil_depth_var].values, dtype=np.float64)
            edges = np.concatenate([[0.0], np.cumsum(dz)])
            src_soil_depth = 0.5 * (edges[:-1] + edges[1:])
        else:
            sg = soil_grid if soil_grid is not None else make_soil_grid(SoilGridConfig())
            src_soil_depth = np.asarray(sg.z_node)

        def _annual(ds, var):
            """(.., lat, lon) -> (nyear, lat, lon); add a unit year axis if static."""
            da = ds[var]
            o = _orient_latlon_last(da, lat_dim, lon_dim)   # (lat, lon, *other)
            if da.ndim == 2:
                return o[None, ...]                          # (1, lat, lon)
            return np.moveaxis(o, -1, 0)                      # (nyear, lat, lon)

        def _pft(ds, var):
            """PCT_NAT_PFT (.., npft, lat, lon) -> (nyear, lat, lon, npft)."""
            da = ds[var]
            o = _orient_latlon_last(da, lat_dim, lon_dim)   # (lat, lon, *other)
            if da.ndim == 3:                                 # (npft, lat, lon)
                return o[None, ...]                          # (1, lat, lon, npft)
            return np.moveaxis(o, 2, 0)                      # (year, lat, lon, npft)

        def _veg(ds, var):
            """MONTHLY_* (12, npft, lat, lon) -> (12, lat, lon, npft)."""
            o = _orient_latlon_last(ds[var], lat_dim, lon_dim)  # (lat, lon, 12, npft)
            return np.moveaxis(o, 2, 0)                          # (12, lat, lon, npft)

        years = (np.asarray(lds[config.year_var].values, dtype=np.float64)
                 if config.year_var in lds else np.array([0.0]))

        raw = _RawSurfaceFields(
            src_lat=src_lat, src_lon=src_lon, src_soil_depth=src_soil_depth,
            sand=_orient_latlon_last(sds[config.sand_var], lat_dim, lon_dim) * s,
            clay=_orient_latlon_last(sds[config.clay_var], lat_dim, lon_dim) * s,
            organic=_orient_latlon_last(sds[config.organic_var], lat_dim, lon_dim),
            bulk_density=_orient_latlon_last(sds[config.bulk_density_var], lat_dim, lon_dim),
            soil_color=_orient_latlon_last(sds[config.color_var], lat_dim, lon_dim),
            years=years,
            f_land=_annual(lds, config.land_frac_var) * s,
            f_lake=_annual(lds, config.lake_var) * s,
            f_glacier=_annual(lds, config.glacier_var) * s,
            pft_frac=_pft(lds, config.pft_frac_var) * s,
            lai=_veg(vds, config.lai_var),
            sai=_veg(vds, config.sai_var),
            htop=_veg(vds, config.htop_var),
            hbot=_veg(vds, config.hbot_var),
            cell_area=(_orient_latlon_last(sds[config.area_var], lat_dim, lon_dim) * config.area_scale
                       if config.area_var in sds else None),
        )
        return build_global_surface_data(raw, grid, config, soil_grid=soil_grid)
    finally:
        for d in opened:
            d.close()


# ===========================================================================
# Per-step time interpolation (pure JAX, traceable)
# ===========================================================================
def interp_annual(field: jnp.ndarray, years: jnp.ndarray, year: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation of an annual-transient field to ``year``.

    ``field`` has shape ``(nyear, ...)``; ``years`` is ``(nyear,)``.  Years
    outside the range clamp to the endpoints.  Single-year inputs return that
    year unchanged.
    """
    nyear = field.shape[0]
    if nyear == 1:
        return field[0]
    frac = jnp.interp(year, years, jnp.arange(nyear, dtype=field.dtype))
    lo = jnp.clip(jnp.floor(frac).astype(jnp.int32), 0, nyear - 1)
    hi = jnp.clip(lo + 1, 0, nyear - 1)
    w = (frac - lo).astype(field.dtype)
    return field[lo] * (1.0 - w) + field[hi] * w


def interp_monthly(field: jnp.ndarray, day_of_year: jnp.ndarray) -> jnp.ndarray:
    """Cyclic linear interpolation of a 12-month climatology to ``day_of_year``.

    ``field`` has shape ``(12, ...)`` with values at mid-month centres
    (``_MONTH_CENTRES``).  Wraps across the Dec→Jan boundary.
    """
    centres = jnp.asarray(_MONTH_CENTRES, dtype=field.dtype)
    year_len = 365.0
    # Build a wrapped 14-point ladder: Dec(prev) .. months .. Jan(next).
    ext_days = jnp.concatenate([centres[-1:] - year_len, centres, centres[:1] + year_len])
    idx = jnp.concatenate([jnp.array([_N_MONTH - 1]), jnp.arange(_N_MONTH), jnp.array([0])])
    doy = jnp.mod(day_of_year, year_len)
    frac = jnp.interp(doy, ext_days, jnp.arange(ext_days.shape[0], dtype=field.dtype))
    lo = jnp.clip(jnp.floor(frac).astype(jnp.int32), 0, ext_days.shape[0] - 1)
    hi = jnp.clip(lo + 1, 0, ext_days.shape[0] - 1)
    w = (frac - lo).astype(field.dtype)
    return field[idx[lo]] * (1.0 - w) + field[idx[hi]] * w


# ===========================================================================
# v1-only tile collapse (§10a)
# ===========================================================================
def dominant_pft_collapse(gsd: GlobalSurfaceData) -> GlobalSurfaceData:
    """Collapse the PFT axis to the single dominant PFT per column (v1 only).

    The dominant PFT is chosen from the time-mean cover fraction so the choice is
    stable across years.  The PFT axis is kept at size 1 (not dropped) so the
    restart format and downstream shapes are unchanged when v2 tiling restores a
    real PFT/tile axis.  This is the **only** place one-PFT-per-column is assumed.
    """
    mean_frac = jnp.mean(gsd.pft_frac, axis=0)          # (ncol, npft)
    dom = jnp.argmax(mean_frac, axis=-1)                # (ncol,)
    ncol = dom.shape[0]
    cols = jnp.arange(ncol)

    def _select_t(field):  # (nt, ncol, npft) -> (nt, ncol, 1)
        sel = field[:, cols, dom]                       # (nt, ncol)
        return sel[:, :, None]

    # pft_frac collapses to the total within-land cover of the dominant PFT,
    # i.e. the sum over PFTs (the column's vegetated fraction), placed in slot 0.
    veg_total = jnp.sum(gsd.pft_frac, axis=-1, keepdims=True)   # (nyear, ncol, 1)

    return gsd._replace(
        pft_frac=veg_total,
        lai_monthly=_select_t(gsd.lai_monthly),
        sai_monthly=_select_t(gsd.sai_monthly),
        htop_monthly=_select_t(gsd.htop_monthly),
        hbot_monthly=_select_t(gsd.hbot_monthly),
    )
