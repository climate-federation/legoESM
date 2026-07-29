"""JRA55-do v1.4+ "corrected" forcing loader for legoESM OMIP runs.

Pipeline:

    raw JRA55-do (Gregorian dates, mixed cadence, native lat-lon)
        → leap-day drop → noleap calendar
        → conservative regrid to model 1° grid (via grids.conservative_regrid)
        → resample to common 3-hourly time axis
        → consolidated Zarr cache
        → per-step `load_jra55_slice(cache, day)` with linear-in-time interp

Two entry points:

* :func:`build_jra55_cache` — one-time pre-processing, called by
  ``scripts/prepare_omip_forcing.py`` (Phase 5).  CPU-only, NumPy /
  xarray work; writes Zarr.  Caller passes either a path to the source
  Zarr (downloaded externally) or any xarray-openable URL.

* :func:`load_jra55_slice` — runtime loader, called by the driver.
  Reads two adjacent records from the cache and returns a
  :class:`JRA55Slice` linearly interpolated to ``day`` (fractional
  simulation day since 1958-01-01 on the noleap calendar).

Design choices
--------------

* **Cache stores conservatively-regridded fields on the *model* grid.**
  All atmospheric variables live on the same 1° regular lat-lon grid
  the ocean uses.  This makes the runtime loader a pure 2-D lookup; no
  regridding at integration time.
* **Leap days are dropped at cache-build time**, not at the driver.
  The cache time axis is contiguous noleap (365 days/year × N years
  × 8 records/day) so the driver's linear-in-time interp uses the
  index, not absolute dates.
* **Mixed cadences (3-hourly winds, 6-hourly T/q/P) are resampled to
  a common 3-hourly axis** at cache-build time.  The 6-hourly fields
  are linearly interpolated between adjacent samples.  Documented in
  the cache attrs.
* **Runoff (`friver`) is stored on the model grid as the conservative
  regrid output** — i.e., it lands on whatever grid cells the source
  land grid maps to, including land cells.  Coastal redistribution is
  the **driver's** responsibility (Item 4 glue), not this module's,
  because it requires the model land mask.

This module deliberately does not handle:

* GCS / S3 download — caller stages the source Zarr locally first.
* The model land mask — see ``ocean_model_latlon_cgrid`` callers.
* The construction of ``AtmToSurface`` / ``FreshwaterForcing`` —
  driver-level glue in Item 4.

Variable schema
---------------

CMOR names from the JRA55-do v1.4+ "corrected" distribution:

==========  ====================================  ================
CMOR name   Description                           Units
==========  ====================================  ================
``uas``     10 m eastward wind                    m/s
``vas``     10 m northward wind                   m/s
``tas``     2 m air temperature                   K
``huss``    2 m specific humidity                 kg/kg
``psl``     sea-level pressure                    Pa
``rsds``    surface downwelling shortwave         W/m²
``rlds``    surface downwelling longwave          W/m²
``prra``    rainfall flux                         kg/m²/s
``prsn``    snowfall flux                         kg/m²/s
``friver``  river runoff                          kg/m²/s
==========  ====================================  ================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.forcing.time_utils import (
    NOLEAP_DAYS_PER_YEAR,
    date_to_day,
    day_to_date,
    is_feb_29,
    noleap_day_of_year,
)
from legoesm.grids.conservative_regrid import (
    ConservativeRegridWeights,
    apply_conservative_regrid,
    check_axis_span,
    compute_overlap_weights,
)
from legoesm.ocean.freshwater import FreshwaterForcing


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Records per noleap day on the cache time axis (3-hourly).
RECORDS_PER_DAY: int = 8

#: Normalisation for the forcing remap, matching the OMIP2 applicator (kept as a
#: named constant in both places rather than a bare literal, since it is a physics
#: policy).  JRA55-do is Gaussian with outermost centre ~+-89.570 deg -> inferred
#: outer edge ~+-89.849 deg, a 0.151 deg polar gap (3.4x narrower than CORE-II's),
#: so the polar destination row is partly covered: 0.977 at 1 deg, 0.633 at 0.25
#: deg, and uncovered entirely below ~0.15 deg.  Every cached channel is INTENSIVE
#: (tas, huss, psl, winds, radiative and precip flux densities) and the shortfall is
#: a DATA GAP, so 'fracarea' returns the mean of the overlapping source rather than
#: coverage x field, and polar_fill covers rows beyond the source's band.  Not
#: strictly conservative by construction -- correct magnitude is what matters for an
#: intensive field.  See regrid_polar_coverage_2026-07-24.md.
_FORCING_NORMALIZATION: str = "fracarea"

#: Variables this module reads / regrids / caches.  Exactly the set
#: needed to populate AtmToSurface and the freshwater path for tropical
#: OMIP without sea ice.
JRA55_VARIABLES: tuple[str, ...] = (
    "uas", "vas",                 # winds (3-hourly)
    "tas", "huss", "psl",         # T, q, p_sl (6-hourly)
    "rsds", "rlds",               # SW down, LW down (3-hourly)
    "prra", "prsn",               # rainfall, snowfall (3-hourly)
    "friver",                     # river runoff (daily)
)

#: Fields whose source cadence is sub-3-hourly: stored as-is.
#: All others get linearly interpolated to the 3-hourly axis at
#: cache-build time.
_NATIVE_3HOURLY: frozenset[str] = frozenset({
    "uas", "vas", "rsds", "rlds", "prra", "prsn",
})

#: Plausibility ranges per variable (units as in the schema above).
#: Values outside these are logged as warnings at cache-build time —
#: a guard against silently bad source data, not a hard rejection.
_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "uas":    (-100.0, 100.0),
    "vas":    (-100.0, 100.0),
    "tas":    (180.0, 340.0),     # K
    "huss":   (0.0, 0.06),        # kg/kg
    "psl":    (87000.0, 110000.0),  # Pa
    "rsds":   (0.0, 1500.0),      # W/m²  (peak insolation under no clouds)
    "rlds":   (50.0, 600.0),      # W/m²
    "prra":   (0.0, 1e-2),        # kg/m²/s
    "prsn":   (0.0, 1e-3),        # kg/m²/s
    "friver": (0.0, 1.0),         # kg/m²/s — large only in major river mouths
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JRA55DoConfig:
    """Configuration for building / consuming a JRA55-do cache.

    Attributes
    ----------
    source_path : str
        Local path or remote URL to the source JRA55-do Zarr / NetCDF
        store.  Must contain all variables in :data:`JRA55_VARIABLES`
        on a regular lat-lon grid.
    years : tuple[int, int]
        Inclusive ``(year_start, year_end)`` window. OMIP-2 is
        ``(1958, 2018)``.
    target_lat_edges : np.ndarray
        Model-grid latitude cell edges, radians, ascending.
    target_lon_edges : np.ndarray
        Model-grid longitude cell edges, radians, ascending.
    cache_dir : Path
        Directory in which to write the consolidated Zarr cache.
    ref_year : int
        Reference year for noleap day numbering. Default 1958.
    cache_filename : str
        Output filename inside ``cache_dir``. Default
        ``"jra55_do_v14_omip2_1deg_noleap.zarr"``.
    """
    source_path: str
    years: tuple[int, int]
    target_lat_edges: np.ndarray
    target_lon_edges: np.ndarray
    cache_dir: Path
    ref_year: int = 1958
    cache_filename: str = "jra55_do_v14_omip2_1deg_noleap.zarr"

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir) / self.cache_filename

    @property
    def n_years(self) -> int:
        return self.years[1] - self.years[0] + 1

    @property
    def n_records(self) -> int:
        """Total number of 3-hourly records on the cache time axis."""
        return self.n_years * NOLEAP_DAYS_PER_YEAR * RECORDS_PER_DAY


class JRA55Slice(NamedTuple):
    """One 2-D snapshot of JRA55-do forcing fields on the model grid.

    All fields share shape ``(n_lat, n_lon)``. Returned by
    :func:`load_jra55_slice` and consumed by Item-4 driver glue.
    """
    uas: jnp.ndarray
    vas: jnp.ndarray
    tas: jnp.ndarray
    huss: jnp.ndarray
    psl: jnp.ndarray
    rsds: jnp.ndarray
    rlds: jnp.ndarray
    prra: jnp.ndarray
    prsn: jnp.ndarray
    friver: jnp.ndarray


# ---------------------------------------------------------------------------
# Source dataset opening (mirrors forcing/external.py:_open_forcing_dataset)
# ---------------------------------------------------------------------------

def _open_source_dataset(path: str):
    """Open a JRA55-do source store as xarray Dataset.

    Mirrors :func:`legoesm.forcing.external._open_forcing_dataset` but
    stays separate to avoid pulling in unrelated forcing imports.
    Always uses ``decode_times=False`` because JRA55-do uses CF-time
    encoded as ``"days since 1958-01-01"`` which we parse manually.
    """
    import os
    import xarray as xr

    if os.path.isdir(path) or path.endswith(".zarr"):
        return xr.open_zarr(path, decode_times=False)
    return xr.open_dataset(path, decode_times=False)


# ---------------------------------------------------------------------------
# Time-axis handling
# ---------------------------------------------------------------------------

def _parse_time_axis_to_gregorian(time_values, time_units: str):
    """Parse a CF time axis into ``(year, month, day, hour)`` arrays.

    Accepts ``"<unit> since YYYY-MM-DD[...]"`` for ``unit`` in
    ``{"days", "hours", "seconds", "minutes"}``. JRA55-do's
    interannual files use ``"days since 1958-01-01 00:00:00"``;
    xarray-written outputs (e.g. from ``make_ryf.py``) often
    auto-encode in ``"hours since ..."``.

    A small dependency-free Gregorian-date arithmetic layer is used so
    that callers don't need ``cftime`` for the offline cache build.
    """
    import re

    m = re.match(
        r"^\s*(days|hours|minutes|seconds)\s+since\s+"
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        time_units,
    )
    if not m:
        raise ValueError(
            f"Cannot parse time units {time_units!r}; expected "
            "'<days|hours|minutes|seconds> since YYYY-MM-DD ...'"
        )
    unit = m.group(1)
    epoch_year = int(m.group(2))
    epoch_month = int(m.group(3))
    epoch_day = int(m.group(4))

    # Convert the unit-of-time to fractional days so the rest of the
    # function is unit-agnostic.
    unit_to_days = {
        "days": 1.0,
        "hours": 1.0 / 24.0,
        "minutes": 1.0 / (24.0 * 60.0),
        "seconds": 1.0 / 86400.0,
    }
    time_in_days = np.asarray(time_values, dtype=np.float64) * unit_to_days[unit]
    days_int = np.floor(time_in_days).astype(np.int64)
    hours = (time_in_days - days_int) * 24.0

    # Convert (epoch_year, epoch_month, epoch_day) + days_int → Gregorian.
    # Use the Julian-day algorithm to keep this dependency-free and
    # exact for the 1958–2018 window.
    epoch_jdn = _gregorian_to_jdn(epoch_year, epoch_month, epoch_day)
    jdn = epoch_jdn + days_int
    year, month, day = _jdn_to_gregorian(jdn)
    return year, month, day, hours


def _gregorian_to_jdn(year, month, day):
    """Convert proleptic-Gregorian (Y, M, D) → Julian Day Number.

    Vectorised across (year, month, day).  Standard formula
    (Henry F. Fliegel & Thomas C. Van Flandern 1968).
    """
    year = np.asarray(year, dtype=np.int64)
    month = np.asarray(month, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (
        day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )


def _jdn_to_gregorian(jdn):
    """Inverse: JDN → proleptic-Gregorian (year, month, day)."""
    jdn = np.asarray(jdn, dtype=np.int64)
    a = jdn + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + (m // 10)
    return year, month, day


def _build_noleap_record_index(
    src_year: np.ndarray,
    src_month: np.ndarray,
    src_day: np.ndarray,
    src_hour: np.ndarray,
    years: tuple[int, int],
    ref_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute, for every source record, its target index on the
    consolidated noleap 3-hourly cache axis (or -1 if it should be
    dropped — leap day or out of window).

    Returns
    -------
    keep_mask : np.ndarray (bool, shape (n_src,))
        True for records kept on the cache axis.
    cache_index : np.ndarray (int64, shape (n_src,))
        Target index ``[0, n_records)`` on the cache time axis. Only
        meaningful where ``keep_mask`` is True.
    """
    n_src = src_year.size
    in_window = (src_year >= years[0]) & (src_year <= years[1])
    not_leap = np.array([
        not is_feb_29(int(y), int(m), int(d))
        for y, m, d in zip(src_year, src_month, src_day)
    ])
    keep_mask = in_window & not_leap

    # For kept records, compute noleap day index then the 3-hourly slot.
    cache_index = np.full(n_src, -1, dtype=np.int64)
    for i in np.flatnonzero(keep_mask):
        sim_day = date_to_day(
            int(src_year[i]), int(src_month[i]), int(src_day[i]),
            hour=0.0, ref_year=ref_year, calendar="noleap",
        )
        # Round hour to nearest 3-hour slot. JRA55-do delivers exact
        # 0/3/6/9/... or 0/6/12/18 hours; rounding handles tiny float
        # noise without bias.
        slot = int(round(float(src_hour[i]) / 3.0)) % RECORDS_PER_DAY
        cache_index[i] = int(sim_day) * RECORDS_PER_DAY + slot
    return keep_mask, cache_index


# ---------------------------------------------------------------------------
# Variable normalisation (CMOR names → consistent layout)
# ---------------------------------------------------------------------------

def _resolve_lat_lon_dims(da):
    """Identify the (lat, lon) dimension names on a JRA55-do DataArray.

    JRA55-do is distributed with either ``(lat, lon)`` or
    ``(latitude, longitude)`` axis names depending on tooling. Returns
    ``(lat_name, lon_name)`` after normalisation.
    """
    candidates = {"lat", "latitude", "y"}
    lon_candidates = {"lon", "longitude", "x"}
    lat_name = next((d for d in da.dims if d in candidates), None)
    lon_name = next((d for d in da.dims if d in lon_candidates), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"DataArray has no lat/lon dimensions matching {candidates} "
            f"× {lon_candidates}; got {da.dims}"
        )
    return lat_name, lon_name


def _ascending_lat(da, lat_name: str):
    """Return ``da`` with latitude axis ascending (south → north).

    JRA55-do ships latitude descending (90 → −90); the conservative
    regrid expects ascending.
    """
    lat_vals = da[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        return da.isel({lat_name: slice(None, None, -1)})
    return da


def _grid_edges_from_centers(
    centers_deg: np.ndarray,
    periodic: bool = False,
) -> np.ndarray:
    """Edges of a regular lat-lon grid given cell centres in degrees.

    Assumes uniform spacing.  Returns edges in **radians**.

    Parameters
    ----------
    centers_deg : 1-D array
        Cell centre coordinates in degrees.
    periodic : bool
        If True, force the last edge to be exactly ``first_edge + 360``
        so the grid spans the full longitude circle.  This prevents
        the conservative regridding from under-weighting the last cell
        when the source grid's last centre is slightly less than
        ``360 - dx/2`` (e.g. JRA55 TL319 at 640 points has its last
        centre at 359.4375° and an inferred last edge at 359.72° —
        0.28° short of 360°, causing a ~28% weight deficit on the
        target grid's last column).
    """
    if centers_deg.size < 2:
        raise ValueError("Need at least 2 centres to infer edges")
    dx = float(centers_deg[1] - centers_deg[0])
    edges_deg = np.empty(centers_deg.size + 1, dtype=np.float64)
    edges_deg[:-1] = centers_deg - dx / 2.0
    edges_deg[-1] = centers_deg[-1] + dx / 2.0
    if periodic:
        edges_deg[-1] = edges_deg[0] + 360.0
    return np.deg2rad(edges_deg)


def _lon_wrap_dataarray(da, lon_name: str):
    """Append the first longitude column at the end as a ghost wrap column.

    This ensures conservative regridding has source coverage across the
    periodic boundary.  The ghost column's coordinate is first_lon + 360.
    """
    import xarray as xr
    first_col = da.isel({lon_name: 0})
    ghost_lon = float(da[lon_name][0]) + 360.0
    ghost_col = first_col.assign_coords({lon_name: ghost_lon})
    ghost_col = ghost_col.expand_dims(lon_name)
    return xr.concat([da, ghost_col], dim=lon_name)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_jra55_cache(
    config: JRA55DoConfig,
    *,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Build the JRA55-do noleap cache on the model grid.

    Reads the source store, drops leap days, regrids each variable
    conservatively to the model 1° grid, resamples 6-hourly fields to
    the common 3-hourly axis by linear interpolation, and writes a
    single consolidated Zarr.

    Parameters
    ----------
    config : JRA55DoConfig
        Source path, year window, target grid, cache directory.
    overwrite : bool
        If False (default) and the cache already exists, return its
        path without re-building.  If True, rebuild from scratch.
    progress : bool
        Print one-line per-variable progress to stdout.

    Returns
    -------
    Path
        The cache Zarr path.
    """
    import xarray as xr

    out = Path(config.cache_path)
    if out.exists() and not overwrite:
        if progress:
            print(f"[jra55_do] cache exists, skipping build: {out}")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    ds = _open_source_dataset(config.source_path)

    # Time axis: parse and build the cache index.
    time_units = ds["time"].attrs.get("units", "days since 1958-01-01")
    src_year, src_month, src_day, src_hour = _parse_time_axis_to_gregorian(
        ds["time"].values, time_units,
    )
    keep_mask, cache_index = _build_noleap_record_index(
        src_year, src_month, src_day, src_hour,
        config.years, config.ref_year,
    )
    if not np.any(keep_mask):
        raise ValueError(
            f"No source records fall in {config.years} after leap-day drop"
        )

    # Source grid (assumes a single shared regular lat-lon for all
    # vars — true for JRA55-do v1.4+ corrected distribution).
    sample_var = next(v for v in JRA55_VARIABLES if v in ds.data_vars)
    sample_da = _ascending_lat(ds[sample_var], _resolve_lat_lon_dims(ds[sample_var])[0])
    lat_name, lon_name = _resolve_lat_lon_dims(sample_da)
    src_lat_edges = _grid_edges_from_centers(sample_da[lat_name].values)
    src_lon_edges = _grid_edges_from_centers(sample_da[lon_name].values)
    # Clamp src/target lat edges into [-pi/2, pi/2]: for a COARSE source the
    # uniform-spacing edge inference can overshoot the pole (the 8-row test
    # fixture's 86 deg outer centre lands at ~98 deg), where sin() caps below 1
    # and leaves a spurious polar gap -- the polar rows would then regrid a
    # constant to LESS than its value.  INERT on real TL319 JRA55-do, whose
    # inferred outer edge is ~+-89.85 deg; kept as a cheap guard for coarse or
    # irregular inputs.  _grid_edges_from_centers returns RADIANS.
    src_lat_edges = np.clip(src_lat_edges, -np.pi / 2, np.pi / 2)
    target_lat_edges = np.clip(config.target_lat_edges, -np.pi / 2, np.pi / 2)

    # Periodic longitude wrap: the source grid may not cover the full
    # [0°, 360°] range of the target (e.g. JRA55 TL319 at 640 points
    # has edges [-0.28°, 359.72°] which leaves a 0.28° gap at the
    # wrap point).  Fix by appending one ghost column at +360°.
    # The ghost column's data will be the first column's data (wrap).
    # PRECONDITION the wrap-pad relies on, checked BEFORE padding: the RAW source
    # must tile the full 360 deg.  The +360 ghost below would turn a partial
    # source into one cell spanning the whole missing sector, which then reports
    # COMPLETE longitude coverage to the weight builder.
    check_axis_span(src_lon_edges, 2.0 * np.pi,
                    name="JRA55-do source longitude")
    src_lon_edges_deg = np.degrees(src_lon_edges)
    target_lon_max = np.degrees(config.target_lon_edges[-1])
    if src_lon_edges_deg[-1] < target_lon_max - 1e-6:
        # Add a ghost cell that wraps the first source cell to the end.
        # Ghost edge = second_edge + 360 (so the ghost cell has the
        # same width as the first cell).
        ghost_edge_deg = src_lon_edges_deg[1] + 360.0
        src_lon_edges = np.append(src_lon_edges, np.deg2rad(ghost_edge_deg))
        _lon_wrap_pad = True
    else:
        _lon_wrap_pad = False

    # fracarea + polar_fill treat JRA55-do's physical polar gap (see the constant's
    # rationale above), after which every destination cell sums to 1 and
    # require_full_coverage is the STRICT invariant again -- so a longitude
    # seam/ghost deficit still raises.  The raw-longitude precondition the wrap-pad
    # above relies on is asserted before that pad.
    # See regrid_polar_coverage_2026-07-24.md
    weights = compute_overlap_weights(
        src_lat_edges, src_lon_edges,
        target_lat_edges, config.target_lon_edges,
        require_full_coverage=True,
        normalization=_FORCING_NORMALIZATION,
        polar_fill=True,
    )

    # Allocate output arrays on the cache axis.
    n_dst_lat = config.target_lat_edges.size - 1
    n_dst_lon = config.target_lon_edges.size - 1
    n_records = config.n_records
    dst_arrays: dict[str, np.ndarray] = {}

    for var in JRA55_VARIABLES:
        if var not in ds.data_vars:
            raise KeyError(
                f"Source dataset is missing required variable {var!r}"
            )
        if progress:
            print(f"[jra55_do] regridding {var} ...", flush=True)
        da_var = ds[var]
        # If we added a ghost longitude column, pad the DataArray so
        # the source shape matches the extended weights.
        if _lon_wrap_pad:
            da_var = _lon_wrap_dataarray(da_var, lon_name)
        out_arr = _regrid_and_align_variable(
            da_var, var, weights, keep_mask, cache_index,
            n_records, n_dst_lat, n_dst_lon,
        )
        _check_plausible(var, out_arr)
        dst_arrays[var] = out_arr

    # Assemble xarray Dataset and write Zarr.
    lat_centers = 0.5 * (config.target_lat_edges[1:] + config.target_lat_edges[:-1])
    lon_centers = 0.5 * (config.target_lon_edges[1:] + config.target_lon_edges[:-1])
    cache_ds = xr.Dataset(
        {var: (("time", "lat", "lon"), dst_arrays[var]) for var in JRA55_VARIABLES},
        coords={
            "time": np.arange(n_records, dtype=np.int64),
            "lat": np.rad2deg(lat_centers),
            "lon": np.rad2deg(lon_centers),
        },
        attrs={
            "source": config.source_path,
            "year_start": config.years[0],
            "year_end": config.years[1],
            "ref_year": config.ref_year,
            "calendar": "noleap",
            "records_per_day": RECORDS_PER_DAY,
            "n_records": n_records,
            "description": (
                "JRA55-do v1.4+ corrected forcing, leap-day-dropped, "
                "conservatively regridded to the legoESM model grid, "
                "linearly interpolated to a 3-hourly axis."
            ),
        },
    )

    chunks = {"time": RECORDS_PER_DAY,  # 1 day of 3-hourly = ~4 MB/var
              "lat": n_dst_lat, "lon": n_dst_lon}
    cache_ds.chunk(chunks).to_zarr(
        str(out), mode="w", consolidated=True,
    )
    if progress:
        print(f"[jra55_do] cache written: {out}")
    return out


def _regrid_and_align_variable(
    da,
    var: str,
    weights: ConservativeRegridWeights,
    keep_mask: np.ndarray,
    cache_index: np.ndarray,
    n_records: int,
    n_dst_lat: int,
    n_dst_lon: int,
) -> np.ndarray:
    """Regrid one variable and place it on the noleap 3-hourly axis.

    Steps:
      1. Drop records with ``keep_mask == False`` (leap days, out of window).
      2. Conservatively regrid each remaining record to the model grid.
      3. Scatter onto the cache time axis at ``cache_index`` slots.
      4. For 6-hourly fields: linearly interpolate the empty 3-hourly
         slots between adjacent populated records.
    """
    lat_name, lon_name = _resolve_lat_lon_dims(da)
    da = _ascending_lat(da, lat_name)
    src_data = np.asarray(da.values)  # shape (n_time_src, n_src_lat, n_src_lon)
    n_src_lat = src_data.shape[1]
    n_src_lon = src_data.shape[2]
    if (n_src_lat, n_src_lon) != weights.src_shape:
        raise ValueError(
            f"{var}: source shape {(n_src_lat, n_src_lon)} does not "
            f"match weights.src_shape {weights.src_shape}"
        )

    # Use the JAX apply for vectorised conservative regrid.
    src_jnp = jnp.asarray(src_data[keep_mask], dtype=jnp.float64)
    regridded = np.asarray(apply_conservative_regrid(src_jnp, weights))

    # Allocate the cache-axis array filled with NaN; populate from
    # mapped indices, then linearly fill the rest.
    out = np.full((n_records, n_dst_lat, n_dst_lon), np.nan, dtype=np.float64)
    out[cache_index[keep_mask]] = regridded

    if var not in _NATIVE_3HOURLY:
        out = _linfill_time_axis(out)
    else:
        if np.any(np.isnan(out)):
            # 3-hourly source should populate every slot; missing slots
            # mean the source has gaps. Linfill as a fallback but warn.
            n_missing = int(np.sum(np.isnan(out[..., 0, 0])))
            print(
                f"[jra55_do] WARNING: {var} has {n_missing} missing 3-hourly "
                "slot(s); filling by linear interpolation in time."
            )
            out = _linfill_time_axis(out)

    return out


def _linfill_time_axis(arr: np.ndarray) -> np.ndarray:
    """Linearly fill NaN slots along the leading (time) axis.

    Boundary NaNs are extended by nearest-neighbour. Per-spatial-cell
    independent — vectorised via NumPy.
    """
    n_t = arr.shape[0]
    flat = arr.reshape(n_t, -1).copy()
    n_cells = flat.shape[1]
    t_idx = np.arange(n_t, dtype=np.float64)
    for c in range(n_cells):
        col = flat[:, c]
        valid = ~np.isnan(col)
        if not np.any(valid):
            continue
        # np.interp does linear interp + edge extension automatically.
        flat[:, c] = np.interp(t_idx, t_idx[valid], col[valid])
    return flat.reshape(arr.shape)


def _check_plausible(var: str, arr: np.ndarray) -> None:
    """Warn if values lie outside the documented plausible range."""
    lo, hi = _PLAUSIBLE_RANGE[var]
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    if a_min < lo or a_max > hi:
        print(
            f"[jra55_do] WARNING: {var} range [{a_min:.3g}, {a_max:.3g}] "
            f"falls outside plausible [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Runtime slice loader
# ---------------------------------------------------------------------------

def _floor_indices_and_alpha(day: float, ref_year: int = 1958) -> tuple[int, int, float]:
    """Map fractional simulation ``day`` to two adjacent cache indices
    and a linear-interpolation weight ``alpha``.

    Returns
    -------
    i_lo, i_hi : int
        Indices into the cache time axis bracketing ``day``. Where
        ``day`` lands exactly on a cache slot, ``i_lo == i_hi``.
    alpha : float
        Linear-interp weight in [0, 1] such that
        ``f(day) = (1 - alpha) * f[i_lo] + alpha * f[i_hi]``.
    """
    if day < 0.0:
        raise ValueError(f"day must be non-negative; got {day}")
    # The cache axis is 3-hourly: index = day * RECORDS_PER_DAY.
    pos = day * RECORDS_PER_DAY
    i_lo = int(np.floor(pos))
    alpha = float(pos - i_lo)
    i_hi = i_lo if alpha == 0.0 else i_lo + 1
    return i_lo, i_hi, alpha


def load_jra55_slice(
    cache_path: str | Path,
    day: float,
    ref_year: int = 1958,
    *,
    cycle: bool = False,
) -> JRA55Slice:
    """Load a single JRA55Slice at fractional simulation ``day``.

    Linearly interpolates between the two bracketing 3-hourly cache
    records. Returns 2-D JAX arrays on the model grid.

    Parameters
    ----------
    cache_path : str or Path
        Path to the Zarr cache produced by :func:`build_jra55_cache`.
    day : float
        Fractional simulation day since ``ref_year-01-01``, noleap.
        ``date_to_day(...)`` produces this from a calendar date.
    ref_year : int
        Must match the cache's ``ref_year`` attribute (validated).
    cycle : bool
        If True, ``day`` is wrapped modulo the cache length so a
        single-year cache can drive a multi-year run (the Stewart 2020
        Repeat Year Forcing path). When False (default), ``day`` past
        the cache length raises ``IndexError``.
    """
    import xarray as xr

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    cache_ref = int(ds.attrs.get("ref_year", ref_year))
    if cache_ref != ref_year:
        raise ValueError(
            f"cache ref_year={cache_ref} does not match request {ref_year}"
        )

    n_records = int(ds.attrs["n_records"])
    if cycle:
        cache_length_days = n_records / RECORDS_PER_DAY
        day = day % cache_length_days
    i_lo, i_hi, alpha = _floor_indices_and_alpha(day, ref_year=ref_year)
    # In cycle mode, also wrap the upper bracket if it overflows so
    # the linear interp works at the wrap boundary.
    if cycle and i_hi >= n_records:
        i_hi = i_hi % n_records
    if i_hi >= n_records:
        raise IndexError(
            f"day={day} (cache slot {i_hi}) exceeds cache length "
            f"{n_records}"
        )

    fields: dict[str, jnp.ndarray] = {}
    for var in JRA55_VARIABLES:
        lo = ds[var].isel(time=i_lo).values
        if alpha == 0.0:
            arr = lo
        else:
            hi = ds[var].isel(time=i_hi).values
            arr = (1.0 - alpha) * lo + alpha * hi
        fields[var] = jnp.asarray(arr, dtype=jnp.float64)

    return JRA55Slice(**fields)


def load_jra55_block(
    cache_path: str | Path,
    start_day: float,
    n_steps: int,
    dt: float,
    ref_year: int = 1958,
    *,
    cycle: bool = False,
) -> list[JRA55Slice]:
    """Load *n_steps* consecutive JRA55Slices in one bulk read.

    Opens the Zarr cache **once**, reads a contiguous slab per variable
    covering all required time indices, then interpolates per-step on
    the host.  This replaces ``n_steps`` individual
    :func:`load_jra55_slice` calls (each of which reopens the dataset),
    eliminating the dominant I/O overhead in the scan-block path.

    Parameters
    ----------
    cache_path : str or Path
        Path to the Zarr cache.
    start_day : float
        Fractional simulation day of the first step.
    n_steps : int
        Number of consecutive steps to load.
    dt : float
        Timestep in seconds.
    ref_year : int
        Must match the cache's ``ref_year`` attribute.
    cycle : bool
        RYF modulo-cycling mode (see :func:`load_jra55_slice`).

    Returns
    -------
    list[JRA55Slice]
        One slice per step, linearly interpolated between the
        bracketing 3-hourly records.
    """
    import xarray as xr

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    cache_ref = int(ds.attrs.get("ref_year", ref_year))
    if cache_ref != ref_year:
        raise ValueError(
            f"cache ref_year={cache_ref} does not match request {ref_year}"
        )
    n_records = int(ds.attrs["n_records"])
    cache_length_days = n_records / RECORDS_PER_DAY

    # Compute all (i_lo, i_hi, alpha) pairs upfront.
    indices: list[tuple[int, int, float]] = []
    for k in range(n_steps):
        day = start_day + k * dt / 86400.0
        if cycle:
            day = day % cache_length_days
        i_lo, i_hi, alpha = _floor_indices_and_alpha(day, ref_year=ref_year)
        if cycle and i_hi >= n_records:
            i_hi = i_hi % n_records
        if i_hi >= n_records:
            raise IndexError(
                f"day={day} (cache slot {i_hi}) exceeds cache length "
                f"{n_records}"
            )
        indices.append((i_lo, i_hi, alpha))

    # Determine the contiguous range of time indices needed.
    all_idxs = set()
    for i_lo, i_hi, _ in indices:
        all_idxs.add(i_lo)
        all_idxs.add(i_hi)
    sorted_idxs = sorted(all_idxs)

    # If indices are contiguous (common case: consecutive steps within
    # the same day-block), use a single slice read.  Otherwise fall
    # back to fancy indexing.
    idx_min, idx_max = sorted_idxs[0], sorted_idxs[-1]
    contiguous = len(sorted_idxs) == (idx_max - idx_min + 1)

    # Bulk-read each variable once.
    var_data: dict[str, np.ndarray] = {}
    idx_to_pos: dict[int, int] = {}
    for var in JRA55_VARIABLES:
        if contiguous:
            slab = ds[var].isel(time=slice(idx_min, idx_max + 1)).values
            for j, idx in enumerate(range(idx_min, idx_max + 1)):
                idx_to_pos[idx] = j
        else:
            slab = ds[var].isel(time=sorted_idxs).values
            for j, idx in enumerate(sorted_idxs):
                idx_to_pos[idx] = j
        var_data[var] = slab

    # Build per-step slices with linear interpolation.
    slices: list[JRA55Slice] = []
    for i_lo, i_hi, alpha in indices:
        fields: dict[str, jnp.ndarray] = {}
        for var in JRA55_VARIABLES:
            lo = var_data[var][idx_to_pos[i_lo]]
            if alpha == 0.0:
                arr = lo
            else:
                hi = var_data[var][idx_to_pos[i_hi]]
                arr = (1.0 - alpha) * lo + alpha * hi
            fields[var] = jnp.asarray(arr, dtype=jnp.float64)
        slices.append(JRA55Slice(**fields))

    return slices


# ---------------------------------------------------------------------------
# Driver-side glue: JRA55Slice -> AtmToSurface, FreshwaterForcing
# ---------------------------------------------------------------------------
#
# These functions are the bridge between the JRA55-do cache and the
# legoESM coupler/ocean structs.  They do *not* belong in the cache
# builder (which is grid-/model-agnostic) and they do *not* belong in
# the coupler (which is forcing-agnostic).  The ocean driver
# (Item 4 of the tropical OMIP plan) calls them per step:
#
#     slice = load_jra55_slice(cache, day)
#     atm   = jra55_to_atm_surface(slice, lat_rad, lon_rad, day)
#     resp  = ocean_tile_response(atm, sst, u_o, v_o, coupler_cfg)
#     fw    = jra55_to_freshwater(slice, resp.lhflx)
#     state = model.step(state, dt, freshwater=fw, surface_forcing=...)
#
# Both functions are pure JAX and AD-clean.  Neither accesses disk —
# the only I/O is in ``load_jra55_slice``.


def _jra55_cos_zenith(
    lat_rad,
    lon_rad,
    day: float,
    ref_year: int = 1958,
):
    """Cosine of solar zenith angle at fractional simulation ``day``.

    Maps the noleap simulation day back to a Gregorian calendar
    (year, month, day_of_month, hour) and feeds the noleap day-of-year
    + hour into the canonical
    :func:`legoesm.atmosphere.physics.radiation.solar.cos_zenith_angle`.
    The day-of-year is **noleap** (1..365); the diurnal phase tracks
    the simulation hour.

    Parameters
    ----------
    lat_rad, lon_rad : jax.Array
        Latitude / longitude in radians.  Shape-broadcastable.
    day : float
        Fractional simulation day since ``ref_year-01-01`` on the
        noleap calendar.
    ref_year : int
        Reference year for ``day`` (default 1958).

    Returns
    -------
    jax.Array
        cos(θ_z), clipped to [-1, 1].  Negative values denote night;
        callers that need only daytime SW typically apply
        ``jnp.maximum(., 0)``.
    """
    year, month, day_of_month, hour = day_to_date(day, ref_year=ref_year)
    doy = noleap_day_of_year(month, day_of_month)
    return cos_zenith_angle(lat_rad, lon_rad, float(doy), float(hour))


def jra55_to_atm_surface(
    slice: JRA55Slice,
    lat_rad,
    lon_rad,
    day: float,
    *,
    ref_year: int = 1958,
    co2_ppmv: float = 400.0,
) -> AtmToSurface:
    """Bridge a JRA55Slice into the coupler's :class:`AtmToSurface`.

    Builds every field of :class:`AtmToSurface` from the slice plus
    the time/geometry needed for the solar zenith angle:

    - Winds, T, q, p_sl, SW/LW, precipitation: pass-through from slice
      (units / sign conventions match the coupler contract).
    - ``rho_lowest`` is reconstructed from the ideal-gas law using
      virtual temperature: ``ρ = p / (R_d · T_v)`` with
      ``T_v = T (1 + 0.61 q)``.
    - ``p_lowest = p_surface = psl`` because JRA55-do delivers all
      atmospheric state at the surface (10 m for winds, 2 m for T/q).
      The ~2 m offset for T/q is handled by the bulk-flux solver via
      :data:`CouplerConfig.z_t_atm` (see Item 1 audit), not here.
    - ``cos_zenith`` is computed from absolute date + lat/lon.

    Parameters
    ----------
    slice : JRA55Slice
        Output of :func:`load_jra55_slice`. All fields are 2-D on the
        model grid.
    lat_rad, lon_rad : jax.Array
        Latitude / longitude of the grid in radians.  Shape must
        broadcast against the slice fields' 2-D shape.
    day : float
        Fractional simulation day since ``ref_year-01-01`` on the
        noleap calendar.
    ref_year : int
        Reference year (default 1958, matches the JRA55-do cache).
    co2_ppmv : float
        Static CO₂ for the radiation scheme (default 400 ppmv —
        OMIP-2 protocol leaves CO₂ static at present-day for forced
        ocean runs).

    Returns
    -------
    AtmToSurface
        Ready for ``ocean_tile_response``.
    """
    # Virtual temperature: T_v = T (1 + (1/epsilon - 1) q); canonical coefficient
    # (~0.608), not the rounded 0.61 literal (~0.4% drift).
    T_v = slice.tas * (1.0 + (1.0 / constants.epsilon - 1.0) * slice.huss)
    rho_a = slice.psl / (constants.R_d * T_v)

    cos_z = _jra55_cos_zenith(lat_rad, lon_rad, day, ref_year=ref_year)
    # Broadcast cos_z to the field shape so all entries are 2-D arrays.
    cos_z_full = jnp.broadcast_to(cos_z, slice.tas.shape).astype(slice.tas.dtype)

    return AtmToSurface(
        sw_down=slice.rsds,
        lw_down=slice.rlds,
        precip_total=slice.prra + slice.prsn,
        precip_snow=slice.prsn,
        T_lowest=slice.tas,
        q_lowest=slice.huss,
        u_lowest=slice.uas,
        v_lowest=slice.vas,
        p_lowest=slice.psl,
        p_surface=slice.psl,
        rho_lowest=rho_a,
        cos_zenith=cos_z_full,
        co2_ppmv=jnp.asarray(co2_ppmv, dtype=slice.tas.dtype),
        has_radiation=jnp.asarray(1.0, dtype=slice.tas.dtype),
        has_precipitation=jnp.asarray(1.0, dtype=slice.tas.dtype),
    )


def jra55_to_freshwater(
    slice: JRA55Slice,
    lhflx,
    *,
    L_v: float | None = None,
) -> FreshwaterForcing:
    """Bridge a JRA55Slice + computed latent heat flux into
    :class:`FreshwaterForcing`.

    Constructs the four-field freshwater struct that
    ``LatLonCGridOceanModel.step(freshwater=...)`` consumes.  Sign
    convention follows ``ocean.freshwater``:

    - ``precip``: rainfall + snowfall, positive into ocean.
    - ``evap``: derived from latent heat flux via ``E = L_h / L_v``.
      Positive **upward** (out of ocean), matching the
      :class:`FreshwaterForcing` contract.
    - ``runoff``: pass-through ``slice.friver``.  This is the *raw*
      conservative regrid output, i.e., it lives on the source land
      grid.  The driver is responsible for redistributing it to ocean
      coastal cells using its land mask (uniform-coastal-by-band per
      ``tropical_omip_plan.md`` §4 Gap 6).  This module does not
      perform that redistribution because it does not know the model
      land mask.
    - ``ice_fw``: identically zero in tropical OMIP (no sea ice).
      Phase B will replace this with the sea-ice-derived flux.

    Parameters
    ----------
    slice : JRA55Slice
    lhflx : jax.Array, shape matching slice fields
        Latent heat flux from the bulk-flux solver, **W/m², positive
        upward** (as returned by :func:`ocean_tile_response`).
    L_v : float, optional
        Latent heat of vaporisation [J/kg]. Defaults to
        ``constants.L_v``.

    Returns
    -------
    FreshwaterForcing
    """
    L = constants.L_v if L_v is None else L_v
    evap = lhflx / L
    return FreshwaterForcing(
        precip=slice.prra + slice.prsn,
        evap=evap,
        runoff=slice.friver,
        ice_fw=jnp.zeros_like(slice.prra),
        restoring=jnp.zeros_like(slice.prra),
    )


def regrid_jra55_slice(
    slc: JRA55Slice,
    regrid_weights,
) -> JRA55Slice:
    """Regrid all fields of a :class:`JRA55Slice` to a new grid.

    Uses precomputed :class:`~legoesm.grids.regridding.RegridWeights`
    (e.g. from :func:`~legoesm.grids.regridding.compute_latlon_to_voronoi_weights`)
    to interpolate every 2-D field in *slc* from the cache lat-lon grid
    onto the target mesh (e.g. MPAS Voronoi cell centres).

    Parameters
    ----------
    slc : JRA55Slice
        Source slice on the cache grid.
    regrid_weights : RegridWeights
        Precomputed interpolation weights.

    Returns
    -------
    JRA55Slice
        Slice with all fields on the target grid.
    """
    from legoesm.grids.regridding import regrid_scalar

    return JRA55Slice(**{
        name: regrid_scalar(getattr(slc, name), regrid_weights)
        for name in JRA55Slice._fields
    })
