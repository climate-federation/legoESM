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


def _depth_coord(n_depth: int, depth: np.ndarray | None) -> tuple:
    """Build a depth coordinate tuple for xarray, reusing real depth values when available."""
    if depth is not None and len(depth) == n_depth:
        return ("depth", depth, {"units": "m", "positive": "down"})
    return ("depth", np.arange(n_depth, dtype=np.float64), {"units": "level_index"})


def stacked_arrays_to_dataset(
    arrays: dict[str, np.ndarray],
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    depth: np.ndarray | None = None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert pre-stacked arrays dict to an xr.Dataset with proper coordinates.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Mapping field_name → array with shape (n_times, ...).
        Must also contain "times_days" (1D float array).
    coord_kind : str
        Grid type: "latlon", "cube", "mpas", "gaussian".
    lon_deg, lat_deg : np.ndarray
        Cell-center coordinates in degrees.
    depth : np.ndarray, optional
        Depth values in meters (positive down). If None, uses level indices.
    attrs : dict, optional
        Global attributes.

    Returns
    -------
    xr.Dataset
    """
    times_days = arrays.get("times_days", np.array([]))
    meta_keys = {"steps", "times_days"}

    data_vars = {}
    for field_key, arr in arrays.items():
        if field_key in meta_keys:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim < 2:
            continue
        field_shape = arr.shape[1:]
        dims, coords = _dims_and_coords_for_field(
            field_shape, coord_kind, lon_deg, lat_deg, depth, times_days)
        data_vars[field_key] = xr.DataArray(
            data=arr, dims=dims, coords=coords,
            attrs=_FIELD_ATTRS.get(field_key, {}))

    ds_attrs = {"coord_kind": coord_kind}
    if attrs:
        ds_attrs.update(attrs)
    return xr.Dataset(data_vars, attrs=ds_attrs)


def _dims_and_coords_for_field(
    field_shape: tuple,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    depth: np.ndarray | None,
    times_days: np.ndarray,
) -> tuple[list[str], dict]:
    """Determine dimension names and coordinates for a field array."""
    if coord_kind in ("latlon", "gaussian"):
        return _latlon_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days)
    elif coord_kind == "cube":
        return _cube_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days)
    elif coord_kind in ("mpas", "mpas_regional"):
        return _mpas_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days)
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]
        coords = {"time": ("time", times_days, {"units": "days"})}
        return dims, coords


def _latlon_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for lat-lon grid fields."""
    lon_1d = np.asarray(lon_deg, dtype=np.float64).ravel()
    lat_1d = np.asarray(lat_deg, dtype=np.float64).ravel()

    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lat": ("lat", lat_1d, {"units": "degrees_north"}),
        "lon": ("lon", lon_1d, {"units": "degrees_east"}),
    }

    if len(field_shape) == 2:
        dims = ["time", "lat", "lon"]
    elif len(field_shape) == 3:
        coords["depth"] = _depth_coord(field_shape[2], depth)
        dims = ["time", "lat", "lon", "depth"]
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]
    return dims, coords


def _cube_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for cubed-sphere grid fields."""
    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lon": (["face", "y", "x"], np.asarray(lon_deg, dtype=np.float64),
                {"units": "degrees_east"}),
        "lat": (["face", "y", "x"], np.asarray(lat_deg, dtype=np.float64),
                {"units": "degrees_north"}),
    }

    if len(field_shape) == 3:
        dims = ["time", "face", "y", "x"]
    elif len(field_shape) == 4:
        coords["depth"] = _depth_coord(field_shape[3], depth)
        dims = ["time", "face", "y", "x", "depth"]
    else:
        dims = ["time"] + [f"dim_{i}" for i in range(len(field_shape))]
    return dims, coords


def _mpas_dims_coords(field_shape, lon_deg, lat_deg, depth, times_days):
    """Dims/coords for MPAS unstructured grid fields."""
    coords = {
        "time": ("time", times_days, {"units": "days"}),
        "lonCell": ("nCells", np.asarray(lon_deg, dtype=np.float64).ravel(),
                    {"units": "degrees_east"}),
        "latCell": ("nCells", np.asarray(lat_deg, dtype=np.float64).ravel(),
                    {"units": "degrees_north"}),
    }

    if len(field_shape) == 1:
        dims = ["time", "nCells"]
    elif len(field_shape) == 2:
        coords["depth"] = _depth_coord(field_shape[1], depth)
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


def _arrays_to_latlon_dataset(
    latlon_arrays: dict,
    depth: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert regridded latlon_arrays dict to xr.Dataset.

    Parameters
    ----------
    latlon_arrays : dict
        Keys: "steps", "times_days", "lat", "lon", and field arrays
        with shape (n_times, n_lat, n_lon) or (n_times, n_lat, n_lon, n_depth).
    depth : np.ndarray, optional
        Depth values in meters (positive down) for 3D fields.
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
            data_vars[key] = xr.DataArray(
                data=arr, dims=["time", "lat", "lon"],
                coords=coords, attrs=_FIELD_ATTRS.get(key, {}))
        elif arr.ndim == 4:
            depth_coords = dict(coords)
            depth_coords["depth"] = _depth_coord(arr.shape[3], depth)
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
