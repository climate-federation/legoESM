"""xarray Dataset conversion and NetCDF/Zarr output for ocean test matrix.

Converts the internal snapshot and timeseries dicts into properly
coordinated xr.Dataset objects and saves them as NetCDF4 or Zarr.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Field metadata (units, long_name) for CF-like attributes
# ---------------------------------------------------------------------------

_FIELD_ATTRS = {
    "eta":       {"units": "m",     "long_name": "Sea surface height"},
    "SST":       {"units": "degC",  "long_name": "Sea surface temperature"},
    "SSS":       {"units": "PSU",   "long_name": "Sea surface salinity"},
    "u_sfc":     {"units": "m/s",   "long_name": "Surface zonal velocity"},
    "v_sfc":     {"units": "m/s",   "long_name": "Surface meridional velocity"},
    "speed_sfc": {"units": "m/s",   "long_name": "Surface speed"},
    "T_3d":      {"units": "degC",  "long_name": "Temperature"},
    "S_3d":      {"units": "PSU",   "long_name": "Salinity"},
    "u_3d":      {"units": "m/s",   "long_name": "Zonal velocity"},
    "v_3d":      {"units": "m/s",   "long_name": "Meridional velocity"},
    "speed_3d":  {"units": "m/s",   "long_name": "Speed"},
    "w_3d":      {"units": "m/s",   "long_name": "Vertical velocity"},
    "w_133m":    {"units": "m/s",   "long_name": "Vertical velocity at 134m depth"},
    "w_sfc":     {"units": "m/s",   "long_name": "Surface vertical velocity"},
    "land_mask": {"units": "1",     "long_name": "Land mask (1=ocean, 0=land)"},
}


def _is_3d_field(key: str, arr: np.ndarray, n_spatial_dims: int) -> bool:
    """Check if a field has a vertical dimension beyond the spatial dims."""
    # 3D fields have one extra trailing dimension (depth levels)
    return arr.ndim > n_spatial_dims


def snapshots_to_dataset(
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    z_coord=None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert snapshots dict to an xr.Dataset with proper coordinates.

    Parameters
    ----------
    snapshots : dict[int, dict[str, np.ndarray]]
        Mapping step_number → {field_name: array}.
    dt : float
        Time step in seconds.
    coord_kind : str
        Grid type: "latlon", "cube", "mpas", "gaussian".
    lon_deg, lat_deg : np.ndarray
        Cell-center coordinates in degrees.
    z_coord : OceanZCoordinate, optional
        Vertical coordinate for depth levels.
    attrs : dict, optional
        Global attributes (experiment name, grid type, etc.).

    Returns
    -------
    xr.Dataset
        Dataset with time, spatial, and optionally depth coordinates.
    """
    if not snapshots:
        return xr.Dataset()

    sorted_steps = sorted(snapshots.keys())
    times_days = np.array([s * dt / 86400.0 for s in sorted_steps],
                          dtype=np.float64)

    # Depth coordinate
    depth = None
    if z_coord is not None:
        depth = -np.asarray(z_coord.z_full_ref, dtype=np.float64)  # positive down

    # Collect fields that exist at all timesteps
    all_keys = set()
    for step_data in snapshots.values():
        all_keys.update(step_data.keys())

    data_vars = {}
    for key in sorted(all_keys):
        timestep_arrays = []
        for step in sorted_steps:
            if key not in snapshots[step]:
                break
            timestep_arrays.append(
                np.asarray(snapshots[step][key], dtype=np.float64))
        else:
            # All timesteps present — stack along time axis
            stacked = np.stack(timestep_arrays, axis=0)
            dims, coords = _dims_and_coords_for_field(
                key, stacked, coord_kind, lon_deg, lat_deg, depth, times_days)
            field_attrs = _FIELD_ATTRS.get(key, {})
            data_vars[key] = xr.DataArray(
                data=stacked, dims=dims, coords=coords, attrs=field_attrs)

    ds_attrs = {"coord_kind": coord_kind}
    if attrs:
        ds_attrs.update(attrs)

    return xr.Dataset(data_vars, attrs=ds_attrs)


def _dims_and_coords_for_field(
    key: str,
    stacked: np.ndarray,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    depth: np.ndarray | None,
    times_days: np.ndarray,
) -> tuple[list[str], dict]:
    """Determine dimension names and coordinates for a field array.

    Returns (dims, coords) appropriate for the grid type and field shape.
    """
    n_times = stacked.shape[0]
    field_shape = stacked.shape[1:]  # spatial shape of a single timestep

    if coord_kind in ("latlon", "gaussian"):
        return _latlon_dims_coords(
            key, field_shape, lon_deg, lat_deg, depth, times_days)
    elif coord_kind == "cube":
        return _cube_dims_coords(
            key, field_shape, lon_deg, lat_deg, depth, times_days)
    elif coord_kind in ("mpas", "mpas_regional"):
        return _mpas_dims_coords(
            key, field_shape, lon_deg, lat_deg, depth, times_days)
    else:
        # Fallback: generic numbered dimensions
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]
        coords = {"time": ("time", times_days, {"units": "days"})}
        return dims, coords


def _latlon_dims_coords(key, field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for lat-lon grid fields."""
    lon_1d = np.asarray(lon_deg, dtype=np.float64).ravel()
    lat_1d = np.asarray(lat_deg, dtype=np.float64).ravel()

    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lat": ("lat", lat_1d, {"units": "degrees_north"}),
        "lon": ("lon", lon_1d, {"units": "degrees_east"}),
    }

    if len(field_shape) == 2:
        # 2D field: (lat, lon)
        dims = ["time", "lat", "lon"]
    elif len(field_shape) == 3:
        # 3D field: (lat, lon, depth)
        n_depth = field_shape[2]
        if depth is not None and len(depth) == n_depth:
            coords["depth"] = ("depth", depth, {"units": "m", "positive": "down"})
        else:
            coords["depth"] = ("depth", np.arange(n_depth, dtype=np.float64),
                               {"units": "level_index"})
        dims = ["time", "lat", "lon", "depth"]
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]

    return dims, coords


def _cube_dims_coords(key, field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for cubed-sphere grid fields."""
    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lon": (["face", "y", "x"], np.asarray(lon_deg, dtype=np.float64),
                {"units": "degrees_east"}),
        "lat": (["face", "y", "x"], np.asarray(lat_deg, dtype=np.float64),
                {"units": "degrees_north"}),
    }

    if len(field_shape) == 3:
        # 2D field: (face, y, x) — e.g., (6, 24, 24)
        dims = ["time", "face", "y", "x"]
    elif len(field_shape) == 4:
        # 3D field: (face, y, x, depth)
        n_depth = field_shape[3]
        if depth is not None and len(depth) == n_depth:
            coords["depth"] = ("depth", depth, {"units": "m", "positive": "down"})
        else:
            coords["depth"] = ("depth", np.arange(n_depth, dtype=np.float64),
                               {"units": "level_index"})
        dims = ["time", "face", "y", "x", "depth"]
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]

    return dims, coords


def _mpas_dims_coords(key, field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for MPAS unstructured grid fields."""
    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lonCell": ("nCells", np.asarray(lon_deg, dtype=np.float64).ravel(),
                    {"units": "degrees_east"}),
        "latCell": ("nCells", np.asarray(lat_deg, dtype=np.float64).ravel(),
                    {"units": "degrees_north"}),
    }

    if len(field_shape) == 1:
        # 2D field: (nCells,)
        dims = ["time", "nCells"]
    elif len(field_shape) == 2:
        # 3D field: (nCells, depth)
        n_depth = field_shape[1]
        if depth is not None and len(depth) == n_depth:
            coords["depth"] = ("depth", depth, {"units": "m", "positive": "down"})
        else:
            coords["depth"] = ("depth", np.arange(n_depth, dtype=np.float64),
                               {"units": "level_index"})
        dims = ["time", "nCells", "depth"]
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]

    return dims, coords


def timeseries_to_dataset(diag: dict, dt: float,
                          attrs: dict | None = None) -> xr.Dataset:
    """Convert diagnostics dict to an xr.Dataset with time coordinate.

    Parameters
    ----------
    diag : dict
        Mapping scalar_name → list of values. Must contain "steps" and "times".
    dt : float
        Time step in seconds (used for time coordinate if "times" missing).
    attrs : dict, optional
        Global attributes.
    """
    if not diag:
        return xr.Dataset()

    steps = np.array(diag.get("steps", []), dtype=np.int64)
    times = np.array(diag.get("times", steps * dt / 86400.0), dtype=np.float64)

    data_vars = {}
    for key, values in diag.items():
        if key in ("steps", "times"):
            continue
        arr = np.array(values, dtype=np.float64)
        if arr.shape == times.shape:
            data_vars[key] = xr.DataArray(
                data=arr, dims=["time"],
                coords={"time": ("time", times, {"units": "days"})})

    ds_attrs = {}
    if attrs:
        ds_attrs.update(attrs)

    return xr.Dataset(data_vars, attrs=ds_attrs)


def _arrays_to_latlon_dataset(latlon_arrays: dict) -> xr.Dataset:
    """Convert the regridded latlon_arrays dict (from _save_snapshot_data) to xr.Dataset.

    The dict has keys: "steps", "times_days", "lat", "lon", and field arrays
    with shape (n_times, n_lat, n_lon) or (n_times, n_lat, n_lon, n_depth).
    """
    times = latlon_arrays.get("times_days", np.array([]))
    lat = latlon_arrays.get("lat", np.array([]))
    lon = latlon_arrays.get("lon", np.array([]))

    meta_keys = {"steps", "times_days", "lat", "lon",
                 "source_lon_range", "source_lat_range"}

    coords = {
        "time": ("time", times, {"units": "days"}),
        "lat": ("lat", lat, {"units": "degrees_north"}),
        "lon": ("lon", lon, {"units": "degrees_east"}),
    }

    data_vars = {}
    for key, arr in latlon_arrays.items():
        if key in meta_keys:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            # (time, lat, lon)
            data_vars[key] = xr.DataArray(
                data=arr, dims=["time", "lat", "lon"],
                coords=coords, attrs=_FIELD_ATTRS.get(key, {}))
        elif arr.ndim == 4:
            # (time, lat, lon, depth)
            depth_coords = dict(coords)
            n_depth = arr.shape[3]
            depth_coords["depth"] = ("depth", np.arange(n_depth, dtype=np.float64),
                                     {"units": "level_index"})
            data_vars[key] = xr.DataArray(
                data=arr, dims=["time", "lat", "lon", "depth"],
                coords=depth_coords, attrs=_FIELD_ATTRS.get(key, {}))

    ds_attrs = {"description": "Regridded to regular lat-lon grid"}
    if "source_lon_range" in latlon_arrays:
        ds_attrs["source_lon_range"] = str(latlon_arrays["source_lon_range"])
    if "source_lat_range" in latlon_arrays:
        ds_attrs["source_lat_range"] = str(latlon_arrays["source_lat_range"])

    return xr.Dataset(data_vars, attrs=ds_attrs)


def save_dataset(ds: xr.Dataset, path: Path, fmt: str = "netcdf"):
    """Save an xr.Dataset to disk.

    Parameters
    ----------
    ds : xr.Dataset
    path : Path
        Output path (without extension — extension is added based on fmt).
    fmt : str
        "netcdf" → .nc, "zarr" → .zarr directory.
    """
    path = Path(path)
    if fmt == "netcdf":
        out = path.with_suffix(".nc")
        ds.to_netcdf(out, engine="netcdf4")
        return out
    elif fmt == "zarr":
        out = path.with_suffix(".zarr")
        ds.to_zarr(out, mode="w")
        return out
    else:
        raise ValueError(f"Unknown format: {fmt!r} (expected 'netcdf' or 'zarr')")
