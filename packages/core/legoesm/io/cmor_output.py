"""CF/CMOR-compliant NetCDF output pipeline for legoESM.

Writes model output as CF-1.7 / CMIP-6.2 / CMOR 3.x compliant NetCDF4
files, suitable for submission to CMIP6-class model intercomparisons.
The ``Conventions`` value is READ from the vendored table Header, not
hard-coded -- the "CF-1.8" this module used to advertise is rejected by
the CMIP6 CV regex ``^CF-1.7 CMIP-6.[0-2]( UGRID-1.0){0,}$``.  Each
variable is stored in its own file following the CMIP6 DRS:

    <var>_<table>_<model>_<experiment>_<variant>_<grid>_<time-range>.nc

The module is self-contained: it handles CMOR variable metadata
(standard_name, long_name, units, cell_methods), CF coordinate
attributes, time axis with bounds, and global attributes.

Dependencies
------------
- xarray (lazy import)
- netCDF4 (lazy import, used only for compression tuning if needed)
- numpy (for JAX array conversion)

Usage
-----
    from legoesm.io.cmor_output import CFWriter

    writer = CFWriter(
        output_dir="output/cmor",
        experiment_id="amip",
        model_id="legoESM-1-0",
        freq="mon",
        calendar="noleap",
        ref_date="0001-01-01",
    )

    writer.write_field(
        var_name="tas",
        data=t2m_array,          # numpy or JAX array, shape (nlat, nlon)
        time=15.0,               # days since ref_date
        time_bounds=(0.0, 30.0),
        lat=lat_1d,
        lon=lon_1d,
    )

    writer.write_monthly(monthly_data, lat_1d, lon_1d, plev=plev_pa)
    writer.close()

Notes
-----
- JAX arrays are converted to numpy before writing.
- Cubed-sphere to lat-lon regridding is NOT performed here; the caller
  should regrid via ``legoesm.grids.regridding`` before passing data.

References
----------
- CF Conventions 1.8: https://cfconventions.org/
- CMOR 3 specification: https://cmor.llnl.gov/
- CMIP6 data reference syntax (DRS):
  https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6aA
"""

from __future__ import annotations

import datetime
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import numpy as np

from legoesm.io.cmor_table_loader import (
    build_all_tables,
    conventions_string,
    coordinate_axis,
    experiment_title,
    table_header,
)

logger = logging.getLogger(__name__)

# CMIP6 fill / missing sentinel.  Every official table Header declares
# ``"missing_value": "1e20"``; the writer used to leave xarray's NaN
# default in place, which PrePARE rejects.
CMIP6_MISSING_VALUE: float = 1.0e20

# CMIP6 ``type`` -> NumPy on-disk dtype.  Every atmosphere/ocean table
# entry legoESM writes declares ``real``, i.e. float32 on disk; the
# writer used to emit float64 because it followed the model's internal
# precision policy.  This affects OUTPUT ONLY -- computation precision is
# untouched.
_CMOR_TYPE_DTYPE: Dict[str, np.dtype] = {
    "real": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "integer": np.dtype(np.int32),
}


# Standard CMIP6 license text (Creative Commons Attribution 4.0 International)
CMIP6_LICENSE = (
    "CMIP6 model data produced by legoESM is licensed under a "
    "Creative Commons Attribution 4.0 International License "
    "(https://creativecommons.org/licenses/). Consult "
    "https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use "
    "governing CMIP6 output, including citation requirements and "
    "proper acknowledgment."
)

# Realm assignment for each CMOR table (CMIP6 CV: required_global_attributes)
_TABLE_REALM: Dict[str, str] = {
    "Amon": "atmos",
    "day": "atmos",
    "Aday": "atmos",   # legacy alias — "day" is the CMIP6 CV value
    "Lmon": "land",
    "Omon": "ocean",
    "Oyr": "ocean",
    "Ofx": "ocean",
    "SImon": "seaIce",
    "SIyr": "seaIce",
    "fx": "atmos",
}

# Cell-measure variables each realm's files refer to.  They are written as
# separate fx/Ofx files alongside the variable files in the DRS, so a data
# file names them in ``external_variables``.  A file that CONTAINS one of
# these must not also name it (CF-1.7 2.6.3); ``_global_attrs`` filters the
# variable being written out of this list.
_REALM_EXTERNAL_VARIABLES: Dict[str, Tuple[str, ...]] = {
    "atmos": ("areacella",),
    "land": ("areacella", "sftlf"),
    "ocean": ("areacello",),
    "seaIce": ("areacello",),
}

# CMIP6 surface-field reference heights [m].  These variables are defined
# at a fixed height above the surface rather than at a model level, so they
# carry a scalar ``height`` coordinate per CMIP6 spec.
_VAR_REFERENCE_HEIGHT: Dict[str, float] = {
    # 2 m air-temperature / humidity diagnostics
    "tas": 2.0,
    "tasmin": 2.0,
    "tasmax": 2.0,
    "huss": 2.0,
    "hurs": 2.0,
    # 10 m wind diagnostics
    "uas": 10.0,
    "vas": 10.0,
    "sfcWind": 10.0,
    "sfcWindmax": 10.0,
}


# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when actually writing files
# ---------------------------------------------------------------------------

def _import_xarray():
    """Lazily import xarray."""
    import xarray as xr
    return xr


def _import_netcdf4():
    """Lazily import netCDF4 (optional, for low-level compression)."""
    try:
        import netCDF4
        return netCDF4
    except ImportError:
        return None


def _to_numpy(arr) -> np.ndarray:
    """Convert a JAX array (or anything array-like) to a numpy ndarray."""
    return np.asarray(arr)


def _resolve_output_dtype(explicit_dtype: str | None) -> np.dtype:
    """Resolve the output dtype for CMOR variable data.

    Priority:
    1. Explicit dtype string (e.g. ``"float64"``) — always wins.
    2. Precision policy's storage dtype — used as a fallback.
       Float32 storage maps to ``np.float32``; anything wider maps
       to ``np.float64``.
    3. ``np.float32`` — safe default if the precision system is
       unavailable.

    Parameters
    ----------
    explicit_dtype : str or None
        If not None, a NumPy dtype string (e.g. ``"float32"``).

    Returns
    -------
    np.dtype
        Resolved NumPy dtype for on-disk variable data.
    """
    if explicit_dtype is not None:
        return np.dtype(explicit_dtype)

    try:
        from legoesm.core.precision import get_policy
        import jax.numpy as jnp

        policy_dtype = get_policy().storage
        if policy_dtype == jnp.float32:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    except Exception:
        return np.dtype(np.float32)


def _dtype_for_entry(entry: Dict[str, Any]) -> np.dtype:
    """Resolve the on-disk dtype from the CMOR table entry's ``type``.

    CMIP6 declares ``type: real`` for every variable legoESM writes, so
    files must be float32 on disk regardless of the model's internal
    precision (which stays float64 under ``JAX_ENABLE_X64``).  This is a
    STORAGE decision only -- no computation is downcast.

    An unknown ``type`` raises rather than defaulting: silently writing
    the wrong width is precisely the kind of drift this module now
    guards against.
    """
    cmor_type = entry.get("type", "real")
    try:
        return _CMOR_TYPE_DTYPE[cmor_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CMOR type {cmor_type!r}; expected one of "
            f"{sorted(_CMOR_TYPE_DTYPE)}"
        ) from exc


def _variable_attrs(entry: Dict[str, Any]) -> Dict[str, str]:
    """Build the CF/CMIP6 variable attributes from a CMOR table entry.

    Single source for every write path (``write_field``,
    ``write_monthly``, ``write_fixed``) so the three cannot drift.
    ``positive`` and ``cell_measures`` are emitted only when the table
    declares them -- an empty string in the table means "this variable
    has none", and writing ``positive=""`` is itself invalid.
    """
    attrs: Dict[str, str] = {
        "standard_name": entry["standard_name"],
        "long_name": entry["long_name"],
        "units": entry["units"],
        "cell_methods": entry["cell_methods"],
    }
    positive = entry.get("positive", "")
    if positive:
        # Sign convention comes from the TABLE, never from a guess:
        # CMIP6 defines tauu/tauv as ``surface_downward_eastward_stress``
        # (positive="down", i.e. with the wind), so a negative tropical
        # zonal-mean tauu is correct for the trades and must not be
        # "corrected".
        attrs["positive"] = positive
    measures = entry.get("cell_measures", "")
    if measures:
        attrs["cell_measures"] = measures
    # CMIP6 requires ``missing_value`` alongside ``_FillValue`` and
    # requires them equal and of the variable's own type.  ``_FillValue``
    # is set through the encoding (xarray owns that key); this is the
    # companion attribute, which was missing from every file we wrote.
    attrs["missing_value"] = _dtype_for_entry(entry).type(CMIP6_MISSING_VALUE)
    return attrs


def merged_variable_attrs(
    entry: Dict[str, Any],
    extra_attrs: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Table attributes with a producer's overrides merged in.

    Producers legitimately override ``cell_methods`` to be HONEST about
    sampling -- a monthly value built from once-daily instantaneous samples
    is not a continuous time mean, so the MPAS lean path relabels it (see
    ``DiagnosticsCollector.cmip_snapshot_vars``).  What a producer knows is
    the TIME clause; the AREA clause belongs to the variable and must keep
    matching the table.

    CMIP6 spells an area-aggregated field ``"area: mean time: <op>"``
    (with ``"area: time: mean"`` as the shorthand when both are means),
    while plev fields such as ``ua``/``va`` carry a bare ``"time: <op>"``
    and must NOT gain an ``area:`` clause.  So a producer cannot hard-code
    one string for a mixed set of variables, and every one that tried wrote
    a bare ``"time: point"`` onto ``tas``/``psl`` -- reintroducing, on
    exactly the snapshot-fed files, the missing-``area:`` defect the
    table-driven metadata otherwise fixed.

    This is the one place that knows both the override and the table entry,
    so the area clause is restored here rather than in each producer.
    """
    attrs = _variable_attrs(entry)
    if not extra_attrs:
        return attrs
    attrs.update(extra_attrs)
    override = extra_attrs.get("cell_methods")
    if override and str(entry["cell_methods"]).startswith("area:"):
        if not str(override).startswith("area:"):
            attrs["cell_methods"] = f"area: mean {override}"
    return attrs


# =========================================================================
# CMOR Variable Tables
# =========================================================================

# The CMOR variable metadata is READ FROM THE OFFICIAL CMIP6 TABLES
# vendored under ``cmor_tables/`` -- it is NOT retyped here.  The
# hand-maintained dicts this replaced had drifted from the tables in 88
# places (wrong ``cell_methods`` on 29 of 35 ``Amon`` variables, wrong
# ``cli``/``clw`` units, no ``positive``, no ``cell_measures``), which is
# exactly the failure mode a table-driven writer cannot have.
#
# Refresh the vendored copy with
# ``scripts/data/build_cmor_table_subset.py``; adding an output variable
# means adding it to that script's WANTED map, never editing JSON or
# adding a dict here.
CMOR_TABLES: Dict[str, Dict[str, Dict[str, Any]]] = build_all_tables()

# Standard CMIP6 pressure levels [Pa], read from the official
# ``plev19`` coordinate axis rather than retyped.
CMIP6_PLEV19 = np.array(
    [float(p) for p in coordinate_axis("plev19")["requested"]]
)


# =========================================================================
# Helper: look up CMOR metadata for a variable name
# =========================================================================

def lookup_cmor_entry(
    var_name: str,
    table: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """Return ``(table_id, entry_dict)`` for a CMOR variable name.

    Parameters
    ----------
    var_name : str
        Short CMOR variable name (e.g. ``"tas"``).
    table : str, optional
        Force a specific table (``"Amon"`` or ``"Lmon"``).  If *None*,
        searches all tables and returns the first match.

    Raises
    ------
    KeyError
        If *var_name* is not found in any table.
    """
    if table is not None:
        tbl = CMOR_TABLES.get(table)
        if tbl is None:
            raise KeyError(f"Unknown CMOR table: {table!r}")
        if var_name not in tbl:
            raise KeyError(
                f"Variable {var_name!r} not in table {table!r}"
            )
        return table, tbl[var_name]

    for tbl_id, tbl in CMOR_TABLES.items():
        if var_name in tbl:
            return tbl_id, tbl[var_name]
    raise KeyError(
        f"Variable {var_name!r} not found in any CMOR table"
    )


# =========================================================================
# Coordinate builders
# =========================================================================

def _make_lat_da(lat: np.ndarray):
    """Build a CF-compliant latitude DataArray."""
    xr = _import_xarray()
    return xr.DataArray(
        np.asarray(lat, dtype=np.float64),
        dims=("lat",),
        attrs={
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
            "bounds": "lat_bnds",
        },
    )


def _make_lon_da(lon: np.ndarray):
    """Build a CF-compliant longitude DataArray."""
    xr = _import_xarray()
    return xr.DataArray(
        np.asarray(lon, dtype=np.float64),
        dims=("lon",),
        attrs={
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
            "bounds": "lon_bnds",
        },
    )


def _cell_bounds_from_centers(
    centers: np.ndarray,
) -> np.ndarray:
    """Compute cell edges from cell centers.

    For a uniform-spacing grid the edges sit halfway between neighboring
    centers; the outermost edges are extrapolated by the same half-step.
    Returns a ``(n, 2)`` array of ``(lower_edge, upper_edge)`` pairs.

    Note: this does not wrap modulo 360 for circular axes (e.g. longitude);
    callers are responsible for providing centers on a canonical interval.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        # Degenerate: use a nominal 1-unit wide cell centered on the point.
        half = 0.5
        return np.array([[c[0] - half, c[0] + half]], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    lower_first = c[0] - (mids[0] - c[0])
    upper_last = c[-1] + (c[-1] - mids[-1])
    edges = np.concatenate([[lower_first], mids, [upper_last]])
    return np.stack([edges[:-1], edges[1:]], axis=-1)


def _make_lat_bnds_da(lat: np.ndarray):
    """Build a latitude cell-bounds DataArray, shape ``(nlat, 2)``.

    Edges are clipped to ``[-90, 90]`` so that polar cells do not extend
    off the sphere, which would otherwise fail CF/CMIP validation.
    """
    xr = _import_xarray()
    bnds = _cell_bounds_from_centers(np.asarray(lat, dtype=np.float64))
    bnds = np.clip(bnds, -90.0, 90.0)
    return xr.DataArray(
        bnds,
        dims=("lat", "bnds"),
        attrs={"units": "degrees_north"},
    )


def _make_lon_bnds_da(lon: np.ndarray):
    """Build a longitude cell-bounds DataArray, shape ``(nlon, 2)``."""
    xr = _import_xarray()
    bnds = _cell_bounds_from_centers(np.asarray(lon, dtype=np.float64))
    return xr.DataArray(
        bnds,
        dims=("lon", "bnds"),
        attrs={"units": "degrees_east"},
    )


def _make_height_da(height_m: float):
    """Build a scalar reference-height coordinate (CMIP6 tas/uas/…)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.float64(height_m),
        attrs={
            "standard_name": "height",
            "long_name": "height",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        },
    )


def _make_plev_da(plev: np.ndarray):
    """Build a CF-compliant pressure-level DataArray.

    Parameters
    ----------
    plev : array-like
        Pressure levels in Pa, ordered top-to-bottom (ascending pressure
        = descending altitude) or bottom-to-top.  The output is sorted
        in descending order (highest pressure first) per CMIP convention.
    """
    xr = _import_xarray()
    plev_np = np.sort(np.asarray(plev, dtype=np.float64))[::-1]
    return xr.DataArray(
        plev_np,
        dims=("plev",),
        attrs={
            "standard_name": "air_pressure",
            "long_name": "Pressure",
            "units": "Pa",
            "axis": "Z",
            "positive": "down",
        },
    )


def _make_depth_da(depth: np.ndarray, *, kind: str = "soil"):
    """Build a CF-compliant depth DataArray.

    ``kind="soil"`` (default): depth below land surface (used by ``Lmon``
    soil fields like ``tsl``).
    ``kind="ocean"``: depth below sea surface (used by ``Omon`` 3-D ocean
    fields like ``thetao``, ``so``, ``uo``, ``vo``, ``wo``,
    ``rhopoto``).  CMIP6 OMIP uses the same dim name ``depth`` for the
    ocean vertical axis with ``positive="down"``.
    """
    xr = _import_xarray()
    if kind == "ocean":
        long_name = "Ocean Depth"
    elif kind == "soil":
        long_name = "Depth Below Land Surface"
    else:
        raise ValueError(
            f"_make_depth_da: kind must be 'ocean' or 'soil'; got {kind!r}."
        )
    return xr.DataArray(
        np.asarray(depth, dtype=np.float64),
        dims=("depth",),
        attrs={
            "standard_name": "depth",
            "long_name": long_name,
            "units": "m",
            "axis": "Z",
            "positive": "down",
        },
    )


def _make_time_da(
    time_val: float,
    ref_date: str,
    calendar: str,
):
    """Build a scalar time DataArray (days since ref_date)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.array([time_val], dtype=np.float64),
        dims=("time",),
        attrs={
            "standard_name": "time",
            "long_name": "Time",
            "units": f"days since {ref_date}",
            "calendar": calendar,
            "axis": "T",
        },
    )


def _make_time_bounds_da(
    bounds: Tuple[float, float],
    ref_date: str,
    calendar: str,
):
    """Build a time_bnds DataArray, shape (1, 2)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.array([[bounds[0], bounds[1]]], dtype=np.float64),
        dims=("time", "bnds"),
        attrs={
            "units": f"days since {ref_date}",
            "calendar": calendar,
        },
    )


# =========================================================================
# CMIP6 metadata helpers
# =========================================================================

_VARIANT_RE = re.compile(
    r"^r(?P<r>\d+)i(?P<i>\d+)p(?P<p>\d+)f(?P<f>\d+)$"
)


def _parse_variant_label(variant_label: str) -> Tuple[int, int, int, int]:
    """Parse a CMIP6 variant label into its four indices.

    ``"r1i1p1f1"`` → ``(1, 1, 1, 1)`` → ``(realization, initialization,
    physics, forcing)``.  Raises ``ValueError`` on malformed input.
    """
    m = _VARIANT_RE.match(variant_label.strip())
    if m is None:
        raise ValueError(
            f"Malformed CMIP6 variant_label {variant_label!r}; "
            "expected 'r<i>i<i>p<i>f<i>' (e.g. 'r1i1p1f1')."
        )
    return (
        int(m.group("r")),
        int(m.group("i")),
        int(m.group("p")),
        int(m.group("f")),
    )


# Ordered, descending CMIP6 nominal_resolution CV thresholds [km].
# See https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_CV.json
# ``nominal_resolution`` — the smallest bucket whose upper bound is ≥ the
# actual grid spacing.  We follow the CMIP6 spec: pick the CV value that
# best brackets the mean great-circle cell dimension in km.
_NOMINAL_RES_BUCKETS_KM: Tuple[Tuple[float, str], ...] = (
    (0.5, "0.5 km"),
    (1.0, "1 km"),
    (2.5, "2.5 km"),
    (5.0, "5 km"),
    (10.0, "10 km"),
    (25.0, "25 km"),
    (50.0, "50 km"),
    (100.0, "100 km"),
    (250.0, "250 km"),
    (500.0, "500 km"),
    (1000.0, "1000 km"),
    (2500.0, "2500 km"),
    (5000.0, "5000 km"),
    (10000.0, "10000 km"),
)


def _compute_nominal_resolution(
    lat: np.ndarray, lon: np.ndarray,
) -> str:
    """Pick the CMIP6 CV ``nominal_resolution`` string for a lat-lon grid.

    Uses the mean grid spacing in degrees, converted to km at the equator
    (1° ≈ 111.19 km), and rounds up to the nearest CMIP6 CV bucket.
    """
    lat_np = np.asarray(lat, dtype=np.float64)
    lon_np = np.asarray(lon, dtype=np.float64)
    if lat_np.size < 2 or lon_np.size < 2:
        return "unknown"
    dlat = float(np.mean(np.abs(np.diff(lat_np))))
    dlon = float(np.mean(np.abs(np.diff(lon_np))))
    # Equatorial km for the longer side (coarser spacing dominates)
    spacing_deg = max(dlat, dlon)
    spacing_km = spacing_deg * 111.19
    for threshold, label in _NOMINAL_RES_BUCKETS_KM:
        if spacing_km <= threshold:
            return label
    return "10000 km"


def _generate_tracking_id() -> str:
    """Return a CMIP6-style tracking ID (``hdl:21.14100/<uuid>``)."""
    return f"hdl:21.14100/{uuid.uuid4()}"


class _DailyRemap(NamedTuple):
    """Where an accumulator's daily field really belongs in CMIP6."""

    out_name: str
    table: str
    plev_pa: Optional[float] = None


# The daily accumulator's internal names are not all CMIP6 variables.
# Verified against ALL 48 official CMIP6 tables at the pinned ref:
#   * ``rsut`` exists in ``Amon`` and ``CFday`` -- but NOT in ``day``.
#   * ``ua850`` / ``va850`` exist in NO CMIP6 table whatsoever; the
#     CMIP6 form is ``ua``/``va`` on the ``plev8`` coordinate in ``day``.
# The DATA is unchanged -- this only puts it under the name and table
# CMIP6 defines for it.
#
# CAVEAT (publication blocker, reported not hidden): ``day/ua`` and
# ``day/va`` are declared on ``plev8`` (8 levels: 1000, 850, 700, 500,
# 250, 100, 50, 10 hPa).  legoESM's daily accumulator carries only the
# 850 hPa level, so the emitted files hold a 1-element pressure axis.
# Filling the other seven with missing data would fabricate coverage;
# producing them for real needs a change to the daily accumulator, which
# is a DATA change and out of scope for this metadata fix.
_DAILY_VARIABLE_REMAP: Dict[str, _DailyRemap] = {
    "rsut": _DailyRemap("rsut", "CFday"),
    "ua850": _DailyRemap("ua", "day", plev_pa=85_000.0),
    "va850": _DailyRemap("va", "day", plev_pa=85_000.0),
}


def _format_drs_time_range(
    t_start: float,
    t_end: float,
    ref_date: str,
    calendar: str,
    freq: str,
) -> str:
    """Format the CMIP6 DRS filename time range for a data span.

    Parameters
    ----------
    t_start, t_end : float
        The span's time-bounds endpoints, in ``days since ref_date``.
        ``t_end`` is EXCLUSIVE (a January monthly mean has bounds
        ``[0, 31)``), so the end label is taken just inside it -- using
        ``t_end`` directly would label a January-only file "197901-197902".
    freq : str
        ``"mon"`` -> ``YYYYMM-YYYYMM``; anything sub-monthly (``day``,
        ``6hr``, ...) -> ``YYYYMMDD-YYYYMMDD``.

    Notes
    -----
    Uses the model's own calendar, so a ``noleap`` run is dated on the
    365-day calendar it actually integrated -- not a proleptic Gregorian
    approximation that would drift a day per leap year.
    """
    import cftime

    units = f"days since {ref_date}"
    # Step just inside the exclusive upper bound.  1e-3 day = 86.4 s,
    # far below the shortest CMIP6 output interval and far above float64
    # round-off at century-scale day counts.
    d0 = cftime.num2date(float(t_start), units, calendar=calendar)
    d1 = cftime.num2date(max(float(t_end) - 1.0e-3, float(t_start)), units,
                         calendar=calendar)
    if freq == "mon":
        return f"{d0.year:04d}{d0.month:02d}-{d1.year:04d}{d1.month:02d}"
    return (
        f"{d0.year:04d}{d0.month:02d}{d0.day:02d}-"
        f"{d1.year:04d}{d1.month:02d}{d1.day:02d}"
    )


def table_realm(table_id: str) -> str:
    """Return the CMIP6 ``realm`` CV value for a CMOR table.

    Public accessor for ``_TABLE_REALM`` so callers (and tests) do not
    import the private mapping across modules.
    """
    return _TABLE_REALM.get(table_id, "atmos")


# ``institution_id`` and ``source_id`` are CONTROLLED vocabularies: a
# value is only valid once PCMDI has merged a registration PR into
# WCRP-CMIP/CMIP6_CVs.  legoESM's defaults below are NOT registered --
# "CU" is not a CMIP6 institution_id and "legoESM-1-0" is not a CMIP6
# source_id -- so files carrying them will be rejected at publication no
# matter how correct the rest of the metadata is.  We cannot register
# them from here and we will not pretend they are valid: both are
# constructor parameters, this check warns once per process, and the
# blocker is stated in the module docstring.
_DEFAULT_UNREGISTERED_INSTITUTION_ID = "CU"
_DEFAULT_UNREGISTERED_SOURCE_ID = "legoESM-1-0"
_CV_REGISTRATION_WARNED: set = set()


def check_cv_registration(institution_id: str, source_id: str) -> List[str]:
    """Return (and warn about) identifiers not registered in the CMIP6 CV.

    Returns the list of unregistered attribute names, empty when both
    have been changed away from legoESM's placeholder defaults.  This is
    a REMINDER, not a validator: it cannot confirm that a non-default
    value *is* registered, only that the known-unregistered defaults are
    still in place.
    """
    unregistered: List[str] = []
    if institution_id == _DEFAULT_UNREGISTERED_INSTITUTION_ID:
        unregistered.append("institution_id")
    if source_id == _DEFAULT_UNREGISTERED_SOURCE_ID:
        unregistered.append("source_id")
    key = (institution_id, source_id)
    if unregistered and key not in _CV_REGISTRATION_WARNED:
        _CV_REGISTRATION_WARNED.add(key)
        logger.warning(
            "CMIP6 CV: %s (institution_id=%r, source_id=%r) are legoESM "
            "placeholders and are NOT in the CMIP6 controlled vocabulary. "
            "Output is CF-valid but WILL be rejected at ESGF publication "
            "until both are registered via a PR to WCRP-CMIP/CMIP6_CVs. "
            "Pass registered values to CFWriter(institution_id=..., "
            "model_id=...) once that lands.",
            ", ".join(unregistered), institution_id, source_id,
        )
    return unregistered


# =========================================================================
# Global attributes
# =========================================================================

def _global_attrs(
    experiment_id: str,
    model_id: str,
    variant_label: str = "r1i1p1f1",
    grid_label: str = "gn",
    institution: str = "Columbia University",
    institution_id: str = "CU",
    source: str = "legoESM: Differentiable Earth System Model in JAX",
    source_type: str = "AGCM",
    sub_experiment_id: str = "none",
    parent_experiment_id: str = "no parent",
    parent_source_id: str = "no parent",
    parent_variant_label: str = "no parent",
    parent_activity_id: str = "no parent",
    parent_time_units: str = "no parent",
    license_text: str = CMIP6_LICENSE,
    further_info_url: str = "",
    nominal_resolution: str = "unknown",
    tracking_id: str = "",
    table_id: str = "Amon",
    grid: str = "",
) -> Dict[str, str]:
    """Return standard CF/CMIP6 global attributes.

    Populates all attributes required by the CMIP6 controlled vocabulary
    (``required_global_attributes`` in the CMIP6_CV.json), so the output
    passes PrePARE / cmip6-cmor-tables metadata validation once the
    ``institution_id`` / ``source_id`` are registered with PCMDI.

    The *parent_** attributes default to ``"no parent"``, which is the
    CV-compliant sentinel for experiments branched from no parent run
    (e.g. ``amip``, ``piControl``).  For branched experiments pass the
    actual parent identifiers.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        r_idx, i_idx, p_idx, f_idx = _parse_variant_label(variant_label)
    except ValueError:
        r_idx = i_idx = p_idx = f_idx = 1
    return {
        # From the vendored tables, validated against the CV regex
        # ``^CF-1.7 CMIP-6.[0-2]( UGRID-1.0){0,}$``.  The writer used to
        # hard-code "CF-1.8", which that regex rejects outright.
        "Conventions": conventions_string(),
        # REQUIRED by the CV and previously missing entirely.  Taken from
        # the CMOR table Header so it always describes the tables the
        # metadata actually came from.
        "data_specs_version": table_header(table_id)["data_specs_version"],
        "mip_era": "CMIP6",
        "activity_id": "CMIP",
        "experiment_id": experiment_id,
        # REQUIRED by the CV and previously missing: the CV's long title
        # for ``experiment_id`` (e.g. amip -> "AMIP").
        "experiment": experiment_title(experiment_id),
        # REQUIRED by the CV and previously missing: free-text
        # description of the grid the data is ON.
        "grid": grid,
        "sub_experiment": "none",
        "sub_experiment_id": sub_experiment_id,
        "institution": institution,
        "institution_id": institution_id,
        "source_id": model_id,
        "source": source,
        "source_type": source_type,
        "product": "model-output",
        "realm": "atmos",
        "variant_label": variant_label,
        "realization_index": np.int32(r_idx),
        "initialization_index": np.int32(i_idx),
        "physics_index": np.int32(p_idx),
        "forcing_index": np.int32(f_idx),
        "grid_label": grid_label,
        "nominal_resolution": nominal_resolution,
        "creation_date": now,
        "tracking_id": tracking_id,
        "license": license_text,
        # REQUIRED by the CV, which fixes its form as
        #   https://furtherinfo.es-doc.org/
        #     <mip_era>.<institution_id>.<source_id>.<experiment_id>
        #     .<sub_experiment_id>.<variant_label>
        # The writer used to emit "" (empty), which fails validation.
        # Derived here rather than invented; the URL only RESOLVES once
        # the ES-DOC documentation is registered (see
        # ``check_cv_registration``).
        "further_info_url": further_info_url or (
            "https://furtherinfo.es-doc.org/CMIP6."
            f"{institution_id}.{model_id}.{experiment_id}"
            f".{sub_experiment_id}.{variant_label}"
        ),
        "parent_experiment_id": parent_experiment_id,
        "parent_source_id": parent_source_id,
        "parent_variant_label": parent_variant_label,
        "parent_activity_id": parent_activity_id,
        "parent_time_units": parent_time_units,
        "branch_method": "no parent",
        "branch_time_in_child": np.float64(0.0),
        "branch_time_in_parent": np.float64(0.0),
        "frequency": "",
        "table_id": "",
        "variable_id": "",
        "history": f"Created by legoESM CFWriter on {now}",
    }


# =========================================================================
# CFWriter class
# =========================================================================

class CFWriter:
    """Manage CF/CMOR-compliant NetCDF output for a legoESM experiment.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory.  Sub-directories are created per table.
    experiment_id : str
        Experiment identifier (e.g. ``"amip"``, ``"piControl"``).
    model_id : str
        Model source identifier (e.g. ``"legoESM-1-0"``).
    freq : str
        Output frequency label: ``"mon"`` or ``"day"``.
    calendar : str
        CF calendar type.  Default ``"noleap"`` (equivalent to
        ``"365_day"``).
    ref_date : str
        Reference date for the time axis, e.g. ``"0001-01-01"``.
    variant_label : str
        CMIP6 variant label.  Default ``"r1i1p1f1"``.
    grid_label : str
        Grid label.  Default ``"gn"`` (native grid).
    institution : str
        Institution string for global attributes.
    compress_level : int
        NetCDF4 deflate compression level (0-9).  Default 4.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_id: str,
        model_id: str,
        freq: str = "mon",
        calendar: str = "noleap",
        ref_date: str = "1850-01-01",
        variant_label: str = "r1i1p1f1",
        grid_label: str = "gn",
        institution: str = "Columbia University",
        institution_id: str = "CU",
        source_type: str = "AGCM",
        sub_experiment_id: str = "none",
        parent_experiment_id: str = "no parent",
        parent_source_id: str = "no parent",
        parent_variant_label: str = "no parent",
        parent_activity_id: str = "no parent",
        parent_time_units: str = "no parent",
        license_text: str = CMIP6_LICENSE,
        further_info_url: str = "",
        compress_level: int = 4,
        grid: str = "native regular latitude-longitude grid",
        drs_tree: bool = False,
    ) -> None:
        # Validate variant_label up front — malformed labels would
        # otherwise silently fall back to (1,1,1,1) for the index
        # attributes, which is a subtle CMIP6 validation failure.
        _parse_variant_label(variant_label)

        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.freq = freq
        self.calendar = calendar
        self.ref_date = ref_date
        self.variant_label = variant_label
        self.grid_label = grid_label
        self.institution = institution
        self.institution_id = institution_id
        self.source_type = source_type
        self.sub_experiment_id = sub_experiment_id
        self.parent_experiment_id = parent_experiment_id
        self.parent_source_id = parent_source_id
        self.parent_variant_label = parent_variant_label
        self.parent_activity_id = parent_activity_id
        self.parent_time_units = parent_time_units
        self.license_text = license_text
        self.further_info_url = further_info_url
        self.compress_level = compress_level
        self.grid = grid
        self.drs_tree = drs_tree
        # DRS dataset version directory (only used when drs_tree=True).
        self._drs_version = "v" + datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y%m%d")

        # Warn once if the CMIP6 CV identifiers are still legoESM's
        # unregistered placeholders (see check_cv_registration).
        check_cv_registration(institution_id, model_id)

        # (table_id, var_name) -> path currently on disk, and the
        # [min t_start, max t_end] the file spans.  ``write_field``
        # appends across a whole run, so the DRS time range is only known
        # incrementally: the file is renamed after every append rather
        # than at close(), so a crashed run still leaves a correctly
        # named file describing exactly the data it contains.
        self._series_path: Dict[Tuple[str, str], Path] = {}
        self._series_span: Dict[Tuple[str, str], List[float]] = {}

        # Track open datasets for appending
        self._open_datasets: Dict[str, Any] = {}  # var_name -> xr.Dataset
        self._file_paths: Dict[str, Path] = {}    # var_name -> file path

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _base_global_attrs(
        self,
        table_id: str,
        var_name: str,
        nominal_resolution: str = "unknown",
    ) -> Dict[str, Any]:
        """Build global attributes for a specific variable file.

        A fresh ``tracking_id`` (CMIP6-style persistent-handle UUID) is
        generated per file, and ``realm`` is derived from *table_id*.
        """
        attrs = _global_attrs(
            experiment_id=self.experiment_id,
            model_id=self.model_id,
            variant_label=self.variant_label,
            grid_label=self.grid_label,
            institution=self.institution,
            institution_id=self.institution_id,
            source_type=self.source_type,
            sub_experiment_id=self.sub_experiment_id,
            parent_experiment_id=self.parent_experiment_id,
            parent_source_id=self.parent_source_id,
            parent_variant_label=self.parent_variant_label,
            parent_activity_id=self.parent_activity_id,
            parent_time_units=self.parent_time_units,
            license_text=self.license_text,
            further_info_url=self.further_info_url,
            nominal_resolution=nominal_resolution,
            tracking_id=_generate_tracking_id(),
            table_id=table_id,
            grid=self.grid,
        )
        attrs["realm"] = table_realm(table_id)
        # ``frequency`` comes from the TABLE ENTRY, not from the writer's
        # own ``freq``.  One CFWriter serves several tables (the AMIP
        # driver constructs it with freq="mon" and then writes the ``day``
        # table through it), so using self.freq stamped frequency="mon"
        # on every daily file -- confirmed on all 9 shipped ``day`` files.
        attrs["frequency"] = self._frequency_for(table_id, var_name)
        attrs["table_id"] = table_id
        attrs["variable_id"] = var_name
        # external_variables: areacella for atmos, areacella/sftlf for land,
        # areacello for ocean.  These cell-area files live alongside the
        # variable files in the DRS and are referenced by name.
        #
        # CF-1.7 2.6.3 forbids naming a variable that IS in this file:
        # "the variables named by external_variables ... must not be present
        # in the file".  The measure files themselves (fx/areacella,
        # Ofx/areacello, Lmon-realm sftlf) are exactly that collision, so the
        # variable being written is filtered out -- confirmed by cfchecks
        # 4.1.0, which reported it as a hard ERROR on areacella_fx.
        realm = attrs["realm"]
        externals = _REALM_EXTERNAL_VARIABLES.get(realm, ())
        externals = tuple(name for name in externals if name != var_name)
        if externals:
            attrs["external_variables"] = " ".join(externals)
        return attrs

    def _output_path(
        self,
        var_name: str,
        table_id: str,
        time_range: str = "",
    ) -> Path:
        """Build the DRS-compliant output file path.

        Filename pattern (CMIP6 DRS):
            <var>_<table>_<source>_<expt>_<variant>_<grid>[_<trange>].nc

        Directory layout: ``<output_dir>/<table_id>/`` by default.  The
        full ESGF DRS tree
        ``<mip_era>/<activity>/<institution>/<source>/<experiment>/
        <variant>/<table>/<var>/<grid_label>/<version>/`` is available via
        ``CFWriter(drs_tree=True)`` but is NOT the default: every
        downstream legoESM consumer (``scripts/validate/
        run_amip_climateeval.py`` and the analysis notebooks) globs
        ``cmor/<table>/*.nc``, so flipping the default would break them
        for a layout that only matters at the moment of ESGF submission.
        The filenames -- which are what validators actually check -- are
        DRS-correct either way.
        """
        if self.drs_tree:
            table_dir = (
                self.output_dir / "CMIP6" / "CMIP" / self.institution_id
                / self.model_id / self.experiment_id / self.variant_label
                / table_id / var_name / self.grid_label / self._drs_version
            )
        else:
            table_dir = self.output_dir / table_id
        table_dir.mkdir(parents=True, exist_ok=True)

        parts = [
            var_name,
            table_id,
            self.model_id,
            self.experiment_id,
            self.variant_label,
            self.grid_label,
        ]
        if time_range:
            parts.append(time_range)
        filename = "_".join(parts) + ".nc"
        return table_dir / filename

    def _encoding_for(
        self,
        var_name: str,
        entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the NetCDF encoding dict for a data variable.

        The dtype comes from the CMOR table's ``type`` (CMIP6 declares
        ``real`` everywhere legoESM writes, i.e. float32 on disk), and
        the fill/missing sentinel is the tables' ``1e20`` rather than
        xarray's NaN default.  CMIP6 requires BOTH ``_FillValue`` and
        ``missing_value`` and requires them equal.
        """
        output_dtype = (
            _dtype_for_entry(entry) if entry is not None
            else _resolve_output_dtype(None)
        )
        enc: Dict[str, Any] = {
            "dtype": output_dtype.str,
            "_FillValue": output_dtype.type(CMIP6_MISSING_VALUE),
        }
        if self.compress_level > 0:
            enc["zlib"] = True
            enc["complevel"] = self.compress_level
        return enc

    def _frequency_for(self, table_id: str, var_name: str) -> str:
        """Return the CMIP6 ``frequency`` for one variable, from the table.

        Falls back to the writer's own ``freq`` only for variables that
        predate the vendored tables (the explicitly non-CMIP6 legacy
        entries), which carry no ``frequency`` field.
        """
        entry = CMOR_TABLES.get(table_id, {}).get(var_name, {})
        return entry.get("frequency") or self.freq

    def _resolve_series_path(
        self,
        series_key: Tuple[str, str],
        var_name: str,
        table_id: str,
    ) -> Path:
        """Return the path ``write_field`` should write/append to.

        Order of preference:
          1. the ranged file this process already wrote for this
             (table, variable) and is still extending;
          2. a legacy un-ranged file left by an earlier run (so a
             restart chained onto pre-fix output keeps appending to it
             instead of silently starting a second series);
          3. a fresh un-ranged path, renamed immediately after the write.
        """
        tracked = self._series_path.get(series_key)
        if tracked is not None and tracked.exists():
            return tracked

        base = self._output_path(var_name, table_id)
        if base.exists():
            return base

        # A RESTART builds a fresh CFWriter with an empty track, so the
        # in-memory map cannot find the file the previous leg wrote --
        # and that file now has a time-range suffix, so the un-ranged
        # base name misses it too.  Recover by globbing the stem; without
        # this a chained run starts a SECOND series and the days end up
        # split across two files.
        stem = base.stem
        matches = sorted(
            p for p in base.parent.glob(f"{stem}_*.nc")
            if re.fullmatch(rf"{re.escape(stem)}_\d{{6,8}}-\d{{6,8}}", p.stem)
        )
        if not matches:
            return base
        if len(matches) > 1:
            logger.warning(
                "CMOR %s/%s: %d existing time-ranged files match %s; appending "
                "to the earliest (%s). Concatenate or clean up the extras "
                "before publication.",
                table_id, var_name, len(matches), stem, matches[0].name,
            )
        found = matches[0]
        # Seed the span from the file's own bounds so the rename after
        # this append describes the FULL range, not just the new leg.
        self._series_span.setdefault(
            series_key, self._span_from_file(found),
        )
        self._series_path[series_key] = found
        return found

    @staticmethod
    def _span_from_file(path: Path) -> List[float]:
        """Return ``[t_start_min, t_end_max]`` from a file's time bounds."""
        nc4 = _import_netcdf4()
        if nc4 is None:  # pragma: no cover - netCDF4 is a hard dependency
            raise RuntimeError("netCDF4 is required to resume a CMOR series")
        with nc4.Dataset(str(path)) as ds:
            if "time_bnds" in ds.variables:
                bnds = np.asarray(ds.variables["time_bnds"][:], dtype=np.float64)
                return [float(bnds.min()), float(bnds.max())]
            t = np.asarray(ds.variables["time"][:], dtype=np.float64)
            return [float(t.min()), float(t.max())]

    def _finalize_series_path(
        self,
        series_key: Tuple[str, str],
        out_path: Path,
        var_name: str,
        table_id: str,
        time_bounds: Tuple[float, float],
    ) -> Path:
        """Rename *out_path* so its name carries the span it now covers.

        Renaming after EVERY append (rather than once at ``close()``)
        means a run killed mid-flight still leaves a correctly named file
        describing exactly the data inside it -- and node failures do
        happen.  The rename is intra-directory, so it is a metadata-only
        operation on any POSIX filesystem.
        """
        span = self._series_span.get(series_key)
        if span is None:
            span = [float(time_bounds[0]), float(time_bounds[1])]
        else:
            span = [min(span[0], float(time_bounds[0])),
                    max(span[1], float(time_bounds[1]))]
        self._series_span[series_key] = span

        time_range = _format_drs_time_range(
            span[0], span[1], self.ref_date, self.calendar,
            self._frequency_for(table_id, var_name),
        )
        target = self._output_path(var_name, table_id, time_range)
        if target != out_path:
            if target.exists() and target != out_path:
                # A prior run already produced this exact range; the
                # freshly written file supersedes it.
                target.unlink()
            out_path.rename(target)
        self._series_path[series_key] = target
        return target

    @staticmethod
    def _coord_encoding(ds: Any) -> Dict[str, Dict[str, Any]]:
        """Encoding that keeps ``_FillValue`` OFF coordinate variables.

        xarray adds ``_FillValue = NaN`` to every float variable it
        writes, including coordinates and bounds.  CF forbids it on
        coordinate variables, and every real file we shipped had it on
        ``time``/``lat``/``lon``.  Coordinates and bounds also stay
        float64 -- the CMOR ``plev``/``height`` axes declare
        ``type: double``, and a float32 time axis loses sub-second
        resolution at century-scale day counts.
        """
        enc: Dict[str, Dict[str, Any]] = {}
        for name in (
            "lat", "lon", "plev", "depth", "height",
            "lat_bnds", "lon_bnds",
        ):
            if name in ds.variables:
                enc[name] = {"_FillValue": None, "dtype": "float64"}
        # ``time``/``time_bnds`` dtype is governed by xarray's CF time
        # encoding; forcing it here would fight the units/calendar logic.
        for name in ("time", "time_bnds"):
            enc.pop(name, None)
            if name in ds.variables:
                enc[name] = {"_FillValue": None}
        return enc

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def write_field(
        self,
        var_name: str,
        data: Any,
        time: float,
        time_bounds: Tuple[float, float],
        lat: Any,
        lon: Any,
        plev: Optional[Any] = None,
        depth: Optional[Any] = None,
        table: Optional[str] = None,
        extra_attrs: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Write a single field snapshot as a CF-compliant NetCDF file.

        Parameters
        ----------
        var_name : str
            CMOR short variable name (e.g. ``"tas"``).
        data : array-like
            Field data.  Shape must match the variable's declared
            dimensions: ``(nlat, nlon)`` for 2-D surface fields,
            ``(nplev, nlat, nlon)`` for 3-D pressure-level fields, or
            ``(ndepth, nlat, nlon)`` for soil-depth fields.
            JAX arrays are converted to numpy automatically.
        time : float
            Time coordinate value (days since *ref_date*).
        time_bounds : tuple of float
            ``(t_start, t_end)`` for the averaging period.
        lat : array-like
            Latitude values in degrees north, shape ``(nlat,)``.
        lon : array-like
            Longitude values in degrees east, shape ``(nlon,)``.
        plev : array-like, optional
            Pressure levels in Pa.  Required for 3-D atmospheric fields.
        depth : array-like, optional
            Soil depth levels in metres.  Required for ``tsl`` and
            similar soil-depth fields.
        table : str, optional
            Force a CMOR table (``"Amon"`` or ``"Lmon"``).
        extra_attrs : dict, optional
            Additional variable-level attributes to include.

        Returns
        -------
        Path
            Path to the written NetCDF file.

        Notes
        -----
        Cubed-sphere data must be regridded to a regular lat-lon grid
        before calling this method.  Use
        ``legoesm.grids.regridding.apply_regrid_weights()`` or an
        equivalent offline regridding step.
        """
        xr = _import_xarray()
        table_id, entry = lookup_cmor_entry(var_name, table=table)
        output_dtype = _dtype_for_entry(entry)
        data_np = _to_numpy(data).astype(output_dtype)

        # --- Build coordinates ---
        coords: Dict[str, Any] = {}
        dims: List[str] = ["time"]

        time_da = _make_time_da(time, self.ref_date, self.calendar)
        time_bnds_da = _make_time_bounds_da(
            time_bounds, self.ref_date, self.calendar,
        )
        coords["time"] = time_da

        declared_dims = entry["dimensions"]

        if "plev" in declared_dims:
            if plev is None:
                raise ValueError(
                    f"Variable {var_name!r} requires pressure levels "
                    f"(plev), but none were provided."
                )
            coords["plev"] = _make_plev_da(_to_numpy(plev))
            dims.append("plev")

        if "depth" in declared_dims:
            if depth is None:
                raise ValueError(
                    f"Variable {var_name!r} requires depth levels, "
                    f"but none were provided."
                )
            # Explicit per-table depth-kind dispatch: every table that
            # uses a ``depth`` dimension MUST appear here so the
            # coordinate carries the right CF ``long_name``.  An
            # unknown table fails fast rather than silently writing
            # ``Depth Below Land Surface`` for ocean variables.
            _DEPTH_KIND_BY_TABLE = {
                "Omon": "ocean",
                "Lmon": "soil",
                # Add new tables here as variables with ``depth``
                # are introduced.
            }
            try:
                depth_kind = _DEPTH_KIND_BY_TABLE[table_id]
            except KeyError as exc:
                raise ValueError(
                    f"Variable {var_name!r} uses dim 'depth' but its "
                    f"table {table_id!r} is not registered in the "
                    f"depth-kind dispatch.  Add it to _DEPTH_KIND_BY_TABLE "
                    f"in cmor_output.py."
                ) from exc
            coords["depth"] = _make_depth_da(
                _to_numpy(depth), kind=depth_kind,
            )
            dims.append("depth")

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        coords["lat"] = _make_lat_da(lat_np)
        coords["lon"] = _make_lon_da(lon_np)
        dims.extend(["lat", "lon"])

        # CMIP6 scalar reference-height coordinate for surface diagnostics
        # (tas @ 2 m, uas/vas @ 10 m, etc.).  Attach it to the DataArray's
        # own ``coords`` (not the surrounding Dataset) so that xarray's
        # auto-``coordinates``-attribute logic only tags the data variable
        # — tagging ``time_bnds``/``lat_bnds``/``lon_bnds`` with
        # ``coordinates=height`` would be invalid CF.
        ref_height_m = _VAR_REFERENCE_HEIGHT.get(var_name)
        if ref_height_m is not None:
            coords["height"] = _make_height_da(ref_height_m)

        # Add leading time dimension to data
        data_np = np.expand_dims(data_np, axis=0)  # (1, ...)

        # --- Build DataArray ---
        var_attrs = merged_variable_attrs(entry, extra_attrs)

        da = xr.DataArray(
            data_np,
            dims=tuple(dims),
            coords=coords,
            attrs=var_attrs,
            name=var_name,
        )

        # --- Assemble Dataset ---
        ds = da.to_dataset()
        ds["time_bnds"] = time_bnds_da
        ds["time"].attrs["bounds"] = "time_bnds"
        ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
        ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
        # Suppress xarray's auto ``coordinates`` attribute on bnds vars:
        # scalar Dataset coords like ``height`` would otherwise be
        # written onto every variable, producing invalid CF output on
        # time_bnds/lat_bnds/lon_bnds.
        for _bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
            if _bnds in ds.variables:
                ds[_bnds].encoding["coordinates"] = None
        # NOTE on bnds units: xarray normalizes CF time encoding so that
        # ``time_bnds`` inherits units/calendar from its parent ``time``
        # variable (per CF 1.8) and strips explicit attrs on serialize.
        # Attempts to re-attach via attrs or encoding are no-ops. This
        # is valid CF: modern CDO/ESMValTool accept bnds without units.
        ds.attrs = self._base_global_attrs(
            table_id, var_name,
            nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
        )

        # --- Write to disk ---
        # ``write_field`` APPENDS across a whole run, so the DRS time
        # range is only known incrementally.  Resolve the file we are
        # already extending (if any), append, then rename to the range
        # the file now spans -- see ``_finalize_series_path``.  The
        # writer previously never passed a time_range at all, so every
        # file the MPAS lane produced was missing the DRS time suffix
        # that ``write_monthly_series`` already emitted correctly.
        series_key = (table_id, var_name)
        out_path = self._resolve_series_path(series_key, var_name, table_id)
        encoding = {
            var_name: self._encoding_for(var_name, entry),
        }
        encoding.update(self._coord_encoding(ds))
        if out_path.exists():
            # Append by extending the time dimension in place.
            # This avoids reading + concatenating + rewriting the
            # entire file, which is O(n²) over a multi-year run.
            # Guard BOTH write paths (netCDF4 append and the xarray-concat
            # fallback): a chained restart can re-flush a window it already
            # wrote, and neither path checked for it.
            _xr_guard = _import_xarray()
            _existing_t = None
            if _xr_guard is not None:
                _chk = _xr_guard.open_dataset(out_path, decode_times=False)
                _existing_t = np.asarray(_chk["time"].values, dtype=np.float64)
                _chk.close()
            if _existing_t is not None and _existing_t.size:
                _t = float(time)
                # Tolerance: far below any real output spacing (1 day for
                # `day`, ~30 for `mon`) and far above float64 round-trip error
                # at century times. Times are stored float64 (verified in a
                # real Amon file), so this is an equality test, not a bin.
                if np.any(np.abs(_existing_t - _t) < 1.0e-3):
                    logger.warning(
                        "CMOR %s/%s: time %.4f is ALREADY on disk — skipping "
                        "the duplicate write (a chained restart re-flushed a "
                        "window it had already written). The FIRST value is "
                        "kept: the re-flush is a PARTIAL re-accumulation of "
                        "the window, verified on a real century arm where the "
                        "repeat's rsdt fell between the true month and the "
                        "next one.",
                        table_id, var_name, _t,
                    )
                    return out_path
                if _t < float(_existing_t.max()):
                    raise ValueError(
                        f"CMOR {table_id}/{var_name}: refusing to append time "
                        f"{_t} before the last written time "
                        f"{float(_existing_t.max())} — the time axis must be "
                        "monotonically increasing"
                    )
            nc4 = _import_netcdf4()
            if nc4 is not None:
                with nc4.Dataset(str(out_path), "a") as ncf:
                    t_idx = len(ncf.dimensions["time"])
                    # A CHAINED run re-flushes a window it already wrote when a
                    # link restarts inside that window, and a blind append then
                    # writes a SECOND row for the same time.  Observed in a real
                    # century arm: two t=105 rows with different rsut, which
                    # made a matched-window comparison ambiguous (-6.8% vs
                    # -5.7% depending on which row was picked).  Skip the
                    # repeat and say so LOUDLY -- overwriting would risk
                    # replacing a COMPLETE month with a partial re-accumulation
                    # after a mid-window restart.
                    ncf.variables["time"][t_idx] = float(time)
                    ncf.variables["time_bnds"][t_idx, :] = [
                        time_bounds[0], time_bounds[1],
                    ]
                    ncf.variables[var_name][t_idx] = data_np[0]
            else:
                # Fallback: xarray concat (original O(n²) path)
                existing = xr.open_dataset(out_path, decode_times=False)
                ds = xr.concat([existing, ds], dim="time")
                existing.close()
                ds.to_netcdf(
                    out_path, format="NETCDF4",
                    encoding=encoding, unlimited_dims=["time"],
                )
        else:
            ds.to_netcdf(
                out_path, format="NETCDF4",
                encoding=encoding, unlimited_dims=["time"],
            )

        return self._finalize_series_path(
            series_key, out_path, var_name, table_id, time_bounds,
        )

    def write_monthly(
        self,
        monthly_data: Dict[str, Any],
        lat: Any,
        lon: Any,
        plev: Optional[Any] = None,
        depth: Optional[Any] = None,
        start_year: int = 1,
    ) -> List[Path]:
        """Write monthly-mean data from a MonthlyAccumulator to NetCDF.

        This convenience method takes the output of
        ``MonthlyAccumulator.finalize()`` (a dict with keys like
        ``"zonal_tas"``, ``"scalar_pr"``, ``"profile_ta"``, etc.) and
        writes each recognized CMOR variable to its own CF-compliant
        file.

        Parameters
        ----------
        monthly_data : dict
            Output of ``MonthlyAccumulator.finalize()``.  Expected keys:

            - ``"months"`` : list of ``(year, month)`` tuples
            - ``"zonal_<name>"`` : ``(n_months, n_lat)`` — 2-D fields
            - ``"profile_<name>"`` : ``(n_months, n_lat, nlev)`` — 3-D
            - ``"scalar_<name>"`` : ``(n_months,)`` — global means
            - ``"lat"`` : ``(n_lat,)`` — latitude bin centers

        lat : array-like
            1-D latitude for the output grid (degrees north).
        lon : array-like
            1-D longitude for the output grid (degrees east).
        plev : array-like, optional
            Pressure levels in Pa for 3-D fields.
        depth : array-like, optional
            Soil depth levels in metres for soil-profile fields.
        start_year : int
            Year offset for time axis computation (default 1).

        Returns
        -------
        list of Path
            Paths to all files written.

        Notes
        -----
        The MonthlyAccumulator stores zonal means (1-D in latitude).
        This method broadcasts them to ``(nlat, nlon)`` by repeating
        along the longitude axis, which is appropriate for zonal-mean
        diagnostics.  For full 2-D output, the caller should provide
        regridded data in the accumulator.
        """
        xr = _import_xarray()
        months: List[Tuple[int, int]] = monthly_data.get("months", [])
        if not months:
            return []

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        nlat = lat_np.shape[0]
        nlon = lon_np.shape[0]
        n_months = len(months)

        # Days in each month (noleap / 365_day calendar)
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Compute time axis: mid-month in days since ref_date
        time_vals = np.empty(n_months, dtype=np.float64)
        time_bnds = np.empty((n_months, 2), dtype=np.float64)
        for i, (yr, mo) in enumerate(months):
            year_offset = (yr - start_year) * 365.0
            day_start = year_offset + sum(month_days[:mo - 1])
            day_end = day_start + month_days[mo - 1]
            time_vals[i] = 0.5 * (day_start + day_end)
            time_bnds[i, 0] = day_start
            time_bnds[i, 1] = day_end

        written: List[Path] = []
        written_vars: set = set()  # track var names already written

        # Process keys in priority order: zonal > profile > scalar
        # so that spatially-resolved data takes precedence.
        priority = {"zonal_": 0, "profile_": 1, "scalar_": 2}
        sorted_keys = sorted(
            (k for k in monthly_data if k not in ("months", "lat")),
            key=lambda k: next(
                (v for p, v in priority.items() if k.startswith(p)), 9
            ),
        )

        # ---- Pre-fetch all device arrays in a single ``jax.device_get`` ----
        # so the JAX runtime can pipeline the per-variable transfers in
        # parallel.  The previous per-key ``_to_numpy(arr_raw)`` chain
        # forced ~50 sequential device→host blocking syncs at every
        # CMIP6 monthly write.
        _device_values = [monthly_data[k] for k in sorted_keys]
        try:
            _host_values = jax.device_get(_device_values)
        except Exception:
            _host_values = _device_values
        _host_lookup = {k: v for k, v in zip(sorted_keys, _host_values)}

        # ---- Process each key in the monthly_data dict ----
        for key in sorted_keys:
            arr_raw = _host_lookup[key]
            arr = _to_numpy(arr_raw)

            # Determine CMOR variable name from key prefix
            if key.startswith("zonal_"):
                var_name = key[len("zonal_"):]
            elif key.startswith("profile_"):
                var_name = key[len("profile_"):]
            elif key.startswith("scalar_"):
                var_name = key[len("scalar_"):]
            else:
                # Unknown key format — skip
                continue

            # Skip if we already wrote a higher-priority version
            if var_name in written_vars:
                continue

            # Check if this is a recognized CMOR variable
            try:
                table_id, entry = lookup_cmor_entry(var_name)
            except KeyError:
                # Not a standard CMOR variable — skip silently
                continue

            declared_dims = entry["dimensions"]
            output_dtype = _dtype_for_entry(entry)

            # Build the field array in the correct shape
            if key.startswith("zonal_"):
                # arr shape: (n_months, n_lat_bins)
                # Broadcast to (n_months, nlat, nlon)
                if arr.shape[1] != nlat:
                    # Interpolate from accumulator latitude bins to
                    # output latitude
                    acc_lat = _to_numpy(monthly_data.get(
                        "lat",
                        np.linspace(-90, 90, arr.shape[1]),
                    ))
                    from numpy import interp as np_interp
                    new_arr = np.empty((n_months, nlat), dtype=output_dtype)
                    for t in range(n_months):
                        new_arr[t] = np_interp(lat_np, acc_lat, arr[t])
                    arr = new_arr

                # Broadcast along longitude
                field = np.broadcast_to(
                    arr[:, :, np.newaxis],
                    (n_months, nlat, nlon),
                ).copy().astype(output_dtype)

            elif key.startswith("profile_"):
                # arr shape: (n_months, n_lat_bins, nlev)
                if "plev" not in declared_dims:
                    continue

                nplev = arr.shape[2] if arr.ndim == 3 else 0
                if nplev == 0:
                    continue

                # Interpolate latitude if needed
                if arr.shape[1] != nlat:
                    acc_lat = _to_numpy(monthly_data.get(
                        "lat",
                        np.linspace(-90, 90, arr.shape[1]),
                    ))
                    from numpy import interp as np_interp
                    new_arr = np.empty(
                        (n_months, nlat, nplev), dtype=output_dtype,
                    )
                    for t in range(n_months):
                        for k in range(nplev):
                            new_arr[t, :, k] = np_interp(
                                lat_np, acc_lat, arr[t, :, k],
                            )
                    arr = new_arr

                # Broadcast along longitude: (n_months, nplev, nlat, nlon)
                # Rearrange from (n_months, nlat, nplev) -> (n_months, nplev, nlat)
                arr_transposed = np.transpose(arr, (0, 2, 1))
                field = np.broadcast_to(
                    arr_transposed[:, :, :, np.newaxis],
                    (n_months, nplev, nlat, nlon),
                ).copy().astype(output_dtype)

            elif key.startswith("scalar_"):
                # arr shape: (n_months,)
                # Broadcast to (n_months, nlat, nlon) as uniform field
                field = np.broadcast_to(
                    arr[:, np.newaxis, np.newaxis],
                    (n_months, nlat, nlon),
                ).copy().astype(output_dtype)

            else:
                continue

            # ---- Build xarray Dataset ----
            coords: Dict[str, Any] = {}
            dims_list: List[str] = ["time"]

            time_da = xr.DataArray(
                time_vals,
                dims=("time",),
                attrs={
                    "standard_name": "time",
                    "long_name": "Time",
                    "units": f"days since {self.ref_date}",
                    "calendar": self.calendar,
                    "axis": "T",
                    "bounds": "time_bnds",
                },
            )
            coords["time"] = time_da

            if "plev" in declared_dims and key.startswith("profile_"):
                plev_vals = (
                    _to_numpy(plev) if plev is not None
                    else CMIP6_PLEV19[:nplev]
                )
                coords["plev"] = _make_plev_da(plev_vals)
                dims_list.append("plev")

            if "depth" in declared_dims:
                if depth is not None:
                    coords["depth"] = _make_depth_da(_to_numpy(depth))
                    dims_list.append("depth")
                else:
                    continue  # skip depth-requiring vars without depth

            coords["lat"] = _make_lat_da(lat_np)
            coords["lon"] = _make_lon_da(lon_np)
            dims_list.extend(["lat", "lon"])

            # Attach scalar reference height to the DataArray's own coords
            # (not the Dataset via assign_coords) so xarray does not
            # auto-tag time_bnds/lat_bnds/lon_bnds with coordinates="height".
            ref_height_m = _VAR_REFERENCE_HEIGHT.get(var_name)
            if ref_height_m is not None:
                coords["height"] = _make_height_da(ref_height_m)

            var_attrs = _variable_attrs(entry)

            da = xr.DataArray(
                field,
                dims=tuple(dims_list),
                coords=coords,
                attrs=var_attrs,
                name=var_name,
            )
            ds = da.to_dataset()

            # Time bounds
            ds["time_bnds"] = xr.DataArray(
                time_bnds,
                dims=("time", "bnds"),
                attrs={
                    "units": f"days since {self.ref_date}",
                    "calendar": self.calendar,
                },
            )
            # Spatial cell bounds (required by CF/CMIP6)
            ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
            ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
            # Suppress xarray's auto ``coordinates`` attribute on bnds
            # variables so scalar coords like ``height`` don't produce
            # invalid CF output (coordinates="height" on bnds is wrong).
            for _bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
                if _bnds in ds.variables:
                    ds[_bnds].encoding["coordinates"] = None
            # xarray strips explicit bnds units per CF normalization;
            # see note in ``write_field``.

            ds.attrs = self._base_global_attrs(
                table_id, var_name,
                nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
            )

            # Time range string for filename
            yr0, mo0 = months[0]
            yr1, mo1 = months[-1]
            time_range = (
                f"{yr0:04d}{mo0:02d}-{yr1:04d}{mo1:02d}"
            )

            out_path = self._output_path(var_name, table_id, time_range)
            encoding = {var_name: self._encoding_for(var_name, entry)}
            encoding.update(self._coord_encoding(ds))

            ds.to_netcdf(
                out_path,
                format="NETCDF4",
                encoding=encoding,
                unlimited_dims=["time"],
            )
            written.append(out_path)
            written_vars.add(var_name)

        return written

    def write_daily(
        self,
        daily_data: Dict[str, Any],
        lat: Any,
        lon: Any,
        extra_attrs_by_var: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Path]:
        """Write daily-mean data from a ``SpatialDailyAccumulator`` to NetCDF.

        Consumes the output of ``SpatialDailyAccumulator.finalize()``:

        - ``"days"`` : list of ``(year, doy)`` tuples
        - ``"field_2d_<name>"`` : ``(n_days, nlat, nlon)`` — daily mean
        - ``"field_2d_<name>_min"`` / ``"_max"`` : daily extremes

        Each recognized variable is written to the CMIP6 ``day`` table.
        ``tas`` extremes become ``tasmin`` / ``tasmax`` via the standard
        CMIP6 naming convention.

        Parameters
        ----------
        daily_data : dict
            Output of ``SpatialDailyAccumulator.finalize()``.
        lat, lon : array-like
            1-D latitude / longitude of the output grid.
        extra_attrs_by_var : dict, optional
            Per-variable attribute overrides, ``{var_name: {attr: value}}``
            (e.g. an honest ``cell_methods`` for snapshot-sampled fields —
            issue #1353).  Variables not in the dict keep table defaults.

        Returns
        -------
        list of Path
            Unique paths written (one per variable, with all days
            appended as the ``time`` dimension).
        """
        days: List[Tuple[int, int]] = daily_data.get("days", [])
        if not days:
            return []

        # Accumulator key → CMIP6 variable name
        key_to_var: Dict[str, str] = {}
        for k in daily_data:
            if not k.startswith("field_2d_"):
                continue
            inner = k[len("field_2d_"):]
            if inner.endswith("_min"):
                key_to_var[k] = inner[:-len("_min")] + "min"
            elif inner.endswith("_max"):
                key_to_var[k] = inner[:-len("_max")] + "max"
            else:
                key_to_var[k] = inner

        written: List[Path] = []
        seen: set = set()
        for key, var_name in key_to_var.items():
            # Map the accumulator's name onto a REAL CMIP6 daily
            # variable (see _DAILY_VARIABLE_REMAP) and skip anything the
            # target table does not define.
            remap = _DAILY_VARIABLE_REMAP.get(var_name)
            out_var = remap.out_name if remap else var_name
            out_table = remap.table if remap else "day"
            out_plev = remap.plev_pa if remap else None
            try:
                lookup_cmor_entry(out_var, table=out_table)
            except KeyError:
                continue

            field = _to_numpy(daily_data[key])
            if field.ndim != 3 or field.shape[0] != len(days):
                continue

            out_path: Optional[Path] = None
            for i, (yr, doy) in enumerate(days):
                # ref_date corresponds to year {self._cmip_start_year}.
                # Buckets store year relative to that, so simply:
                #   day-offset = yr * 365 + (doy - 1)   (noleap)
                d0 = float(yr * 365 + (doy - 1))
                # A remapped single-level field arrives as (nlat, nlon)
                # and must be written on a length-1 vertical axis.
                slab = field[i]
                if out_plev is not None:
                    slab = slab[np.newaxis, ...]
                out_path = self.write_field(
                    var_name=out_var,
                    data=slab,
                    time=d0 + 0.5,
                    time_bounds=(d0, d0 + 1.0),
                    lat=lat,
                    lon=lon,
                    plev=(
                        np.array([out_plev], dtype=np.float64)
                        if out_plev is not None else None
                    ),
                    table=out_table,
                    extra_attrs=(extra_attrs_by_var or {}).get(var_name),
                )
            if out_path is not None and out_path not in seen:
                written.append(out_path)
                seen.add(out_path)

        return written

    def write_fixed(
        self,
        var_name: str,
        data: Any,
        lat: Any,
        lon: Any,
        extra_attrs: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Write a time-invariant field (``fx`` table) to NetCDF.

        ``fx`` files are written once per experiment and contain no time
        dimension. Typical variables: ``orog`` (surface altitude),
        ``areacella`` (cell area), ``sftlf`` (land fraction).

        Parameters
        ----------
        var_name : str
            Short CMOR variable name; must be in the ``fx`` table.
        data : array-like
            2-D field, shape ``(nlat, nlon)``.
        lat, lon : array-like
            1-D latitude / longitude.
        extra_attrs : dict, optional
            Additional variable attributes.

        Returns
        -------
        Path
            Written file path.
        """
        xr = _import_xarray()
        table_id, entry = lookup_cmor_entry(var_name, table="fx")
        output_dtype = _dtype_for_entry(entry)

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        data_np = _to_numpy(data).astype(output_dtype)
        if data_np.shape != (lat_np.shape[0], lon_np.shape[0]):
            raise ValueError(
                f"write_fixed: expected data shape ({lat_np.shape[0]}, "
                f"{lon_np.shape[0]}), got {data_np.shape}"
            )

        var_attrs = merged_variable_attrs(entry, extra_attrs)

        da = xr.DataArray(
            data_np,
            dims=("lat", "lon"),
            coords={
                "lat": _make_lat_da(lat_np),
                "lon": _make_lon_da(lon_np),
            },
            attrs=var_attrs,
            name=var_name,
        )
        ds = da.to_dataset()
        ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
        ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
        # Same bnds-coord suppression as in write_field / write_monthly.
        for _bnds in ("lat_bnds", "lon_bnds"):
            if _bnds in ds.variables:
                ds[_bnds].encoding["coordinates"] = None

        ds.attrs = self._base_global_attrs(
            table_id, var_name,
            nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
        )
        # fx files have no time axis — override frequency.
        ds.attrs["frequency"] = "fx"

        out_path = self._output_path(var_name, table_id)
        encoding = {var_name: self._encoding_for(var_name, entry)}
        encoding.update(self._coord_encoding(ds))
        ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
        return out_path

    def close(self) -> None:
        """Finalize and close any open dataset handles.

        This is a no-op in the current implementation (each
        ``write_field`` / ``write_monthly`` call writes and closes
        immediately), but is provided for forward compatibility with
        buffered writing.
        """
        for ds in self._open_datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        self._open_datasets.clear()
        self._file_paths.clear()

    def __enter__(self) -> "CFWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CFWriter(output_dir={self.output_dir!r}, "
            f"experiment_id={self.experiment_id!r}, "
            f"model_id={self.model_id!r}, "
            f"freq={self.freq!r})"
        )
