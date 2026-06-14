"""Machine-learning ozone parameterization (Ma et al., UKESM ridge regression).

Implements an alternative ozone source for RRTMGP based on the per-gridpoint
ridge-regression predictor of

    Ma, Y. et al. "Machine learning ozone parameterization"
    https://github.com/YYilingMa/machine-learning-ozone-parameterization
    (UKESM piControl / 4xCO2 training, CC-BY-4.0)

Predictor: temperature column T(c, lat, lon) -> ozone column O3(z, lat, lon).
Per-gridpoint StandardScaler on T (x_mean, x_scale) and on O3 (y_mean,
y_scale) plus a ridge coefficient cube `coefs(c, z, lat, lon)`.  All
weights ship as NetCDF (see `training/Ridge_train_create_ncfiles.py`
upstream).

Forward pass per column (j, i):

    x_norm[c]  = (T[c] - x_mean[c, j, i]) / x_scale[c, j, i]
    y_norm[z]  = sum_c coefs[c, z, j, i] * x_norm[c]
    O3[z]      = y_norm[z] * y_scale[z, j, i] + y_mean[z, j, i]

The prediction is fully differentiable wrt T (matmul + affine), so the
parameterization is compatible with `jax.grad` through radiation.

Coefficient files
-----------------
Three NetCDF files produced by the upstream script:

    Scaler_x_*.nc : x_mean, x_scale     dims (z, lat, lon)  (T scaler)
    Scaler_y_*.nc : y_mean, y_scale     dims (z, lat, lon)  (O3 scaler)
    coefs_*.nc    : coefs               dims (c, z, lat, lon)

Additionally we require UKESM full-level pressures (z,) which the upstream
repo does not currently bundle.  Until shipped, supply a sidecar
``plev.npy`` or ``plev.nc`` with the 76 UKESM training pressures [Pa] in
the same directory; the loader will pick it up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants


class MLOzoneCoefficients(NamedTuple):
    """Pre-loaded ridge weights and training-grid descriptors.

    All arrays are stored as JAX arrays so the predictor is JIT-friendly.

    Fields
    ------
    coefs : (c, z, lat, lon) ridge weights.
    x_mean, x_scale : (z, lat, lon) T scaler.
    y_mean, y_scale : (z, lat, lon) O3 scaler.
    lat_uk : (n_lat,) UKESM latitude centers [degrees, ascending].
    lon_uk : (n_lon,) UKESM longitude centers [degrees, in [0, 360)].
    p_uk : (n_lev,) UKESM full-level pressures [Pa, ascending top->bottom].
    """
    coefs: jnp.ndarray
    x_mean: jnp.ndarray
    x_scale: jnp.ndarray
    y_mean: jnp.ndarray
    y_scale: jnp.ndarray
    lat_uk: jnp.ndarray
    lon_uk: jnp.ndarray
    p_uk: jnp.ndarray


def _read_netcdf_vars(path: Path, var_names: list[str]) -> dict[str, np.ndarray]:
    """Read a small set of variables from a NetCDF4 file (host-side)."""
    import netCDF4  # type: ignore
    with netCDF4.Dataset(str(path), "r") as ds:
        return {name: np.asarray(ds.variables[name][:]) for name in var_names}


def _load_plev_sidecar(directory: Path, n_lev: int) -> np.ndarray:
    """Locate UKESM training pressures next to the coefficient files.

    Looks for ``plev.npy``, ``plev.nc``, or ``pressure_levels.nc`` in the
    same directory.  Raises if none found; the predictor requires the
    training-grid pressure coordinate to do vertical interpolation.
    """
    npy = directory / "plev.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
    else:
        nc_candidates = [directory / "plev.nc", directory / "pressure_levels.nc"]
        nc_path = next((p for p in nc_candidates if p.exists()), None)
        if nc_path is None:
            raise FileNotFoundError(
                f"ML ozone: need UKESM pressure coordinate next to coefficient "
                f"files (expected one of plev.npy / plev.nc / pressure_levels.nc "
                f"in {directory}).  Provide an ascending 1-D array of full-level "
                f"pressures [Pa] with length {n_lev}."
            )
        vars_ = _read_netcdf_vars(nc_path, ["plev"])
        arr = vars_["plev"].astype(np.float64)
    if arr.ndim != 1 or arr.shape[0] != n_lev:
        raise ValueError(
            f"ML ozone: pressure coordinate has shape {arr.shape}; "
            f"expected ({n_lev},)."
        )
    return arr


def load_ml_ozone_coefficients(path: str | Path) -> MLOzoneCoefficients:
    """Load ridge regression weights + scalers from a directory of NetCDFs.

    Parameters
    ----------
    path : str or Path
        Directory containing ``coefs_*.nc``, ``Scaler_x_*.nc``,
        ``Scaler_y_*.nc``, and a pressure-coordinate sidecar (see
        :func:`_load_plev_sidecar`).  The first match for each glob is
        used; for experiment-specific weights (piCTRL vs 4×CO2) point at
        a directory containing the desired pair.

    Returns
    -------
    MLOzoneCoefficients
        All arrays as ``jnp.float64``.  Loading is performed once outside
        JIT; the returned NamedTuple is JIT-static-friendly because each
        leaf is a JAX array.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"ML ozone weights directory not found: {directory}")

    def _first(pattern: str) -> Path:
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"ML ozone: no file matching {pattern!r} in {directory}"
            )
        return matches[0]

    coefs_path = _first("coefs*.nc")
    sx_path = _first("Scaler_x*.nc")
    sy_path = _first("Scaler_y*.nc")

    coefs_vars = _read_netcdf_vars(coefs_path, ["coefs"])
    sx_vars = _read_netcdf_vars(sx_path, ["x_mean", "x_scale"])
    sy_vars = _read_netcdf_vars(sy_path, ["y_mean", "y_scale"])

    coefs = coefs_vars["coefs"].astype(np.float64)  # (c, z, lat, lon)
    if coefs.ndim != 4:
        raise ValueError(f"coefs has shape {coefs.shape}; expected 4D (c, z, lat, lon).")
    n_c, n_z, n_lat, n_lon = coefs.shape
    if n_c != n_z:
        raise ValueError(
            f"ML ozone: predictor and target level counts differ "
            f"(c={n_c}, z={n_z}); expected square."
        )

    def _check_scaler_shape(name: str, arr: np.ndarray):
        if arr.shape != (n_z, n_lat, n_lon):
            raise ValueError(
                f"{name} has shape {arr.shape}; expected {(n_z, n_lat, n_lon)}."
            )

    x_mean = sx_vars["x_mean"].astype(np.float64); _check_scaler_shape("x_mean", x_mean)
    x_scale = sx_vars["x_scale"].astype(np.float64); _check_scaler_shape("x_scale", x_scale)
    y_mean = sy_vars["y_mean"].astype(np.float64); _check_scaler_shape("y_mean", y_mean)
    y_scale = sy_vars["y_scale"].astype(np.float64); _check_scaler_shape("y_scale", y_scale)

    # UKESM N96 grid: 144 latitudes (centered, 1.25° spacing) × 192 longitudes.
    # If a grid file exists alongside, prefer it; otherwise reconstruct.
    grid_path = directory / "grid.nc"
    if grid_path.exists():
        gvars = _read_netcdf_vars(grid_path, ["lat", "lon"])
        lat_uk = np.asarray(gvars["lat"], dtype=np.float64)
        lon_uk = np.asarray(gvars["lon"], dtype=np.float64)
    else:
        # UKESM N96 atmospheric grid uses cell-centered lats from -89.375
        # to 89.375 with step 180/n_lat, and lons from 0 to 360-step with
        # step 360/n_lon.  This matches CMIP6 UKESM1 NetCDF metadata.
        dlat = 180.0 / n_lat
        dlon = 360.0 / n_lon
        lat_uk = -90.0 + 0.5 * dlat + dlat * np.arange(n_lat)  # coeff-ok: UKESM latitude grid [deg]
        lon_uk = 0.5 * dlon + dlon * np.arange(n_lon)
    if lat_uk.shape != (n_lat,) or lon_uk.shape != (n_lon,):
        raise ValueError(
            f"ML ozone: lat/lon shapes {lat_uk.shape}/{lon_uk.shape} "
            f"do not match coefs {(n_lat, n_lon)}."
        )

    p_uk = _load_plev_sidecar(directory, n_lev=n_z)

    # Ensure ascending pressure (top → bottom) for log-p interp helpers below.
    if p_uk[0] > p_uk[-1]:
        order = np.argsort(p_uk)
        p_uk = p_uk[order]
        # Reorder z axis on every variable that carries it.
        x_mean = x_mean[order, :, :]
        x_scale = x_scale[order, :, :]
        y_mean = y_mean[order, :, :]
        y_scale = y_scale[order, :, :]
        coefs = coefs[order, :, :, :]  # reorder predictor axis c
        coefs = coefs[:, order, :, :]  # reorder output axis z
    # Ensure ascending lat.
    if lat_uk[0] > lat_uk[-1]:
        lat_uk = lat_uk[::-1]
        x_mean = x_mean[:, ::-1, :]
        x_scale = x_scale[:, ::-1, :]
        y_mean = y_mean[:, ::-1, :]
        y_scale = y_scale[:, ::-1, :]
        coefs = coefs[:, :, ::-1, :]
    # Wrap lon into [0, 360).
    lon_uk = np.mod(lon_uk, 360.0)
    sort_lon = np.argsort(lon_uk)
    lon_uk = lon_uk[sort_lon]
    x_mean = x_mean[:, :, sort_lon]
    x_scale = x_scale[:, :, sort_lon]
    y_mean = y_mean[:, :, sort_lon]
    y_scale = y_scale[:, :, sort_lon]
    coefs = coefs[:, :, :, sort_lon]

    return MLOzoneCoefficients(
        coefs=jnp.asarray(coefs),
        x_mean=jnp.asarray(x_mean),
        x_scale=jnp.asarray(x_scale),
        y_mean=jnp.asarray(y_mean),
        y_scale=jnp.asarray(y_scale),
        lat_uk=jnp.asarray(lat_uk),
        lon_uk=jnp.asarray(lon_uk),
        p_uk=jnp.asarray(p_uk),
    )


# ---------------------------------------------------------------------------
# Interpolation helpers (JAX, differentiable in field values)
# ---------------------------------------------------------------------------

def _bilinear_weights_1d(target: jnp.ndarray, src: jnp.ndarray,
                         *, periodic_period: float | None = None
                         ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(idx_lo, idx_hi, w_hi)`` for 1-D linear interpolation.

    Result: ``f_interp = (1 - w_hi) * f[idx_lo] + w_hi * f[idx_hi]``.

    For periodic axes (longitude), pass ``periodic_period=360.0`` so wraps
    are handled symmetrically near the seam.
    """
    n = src.shape[0]
    if periodic_period is None:
        # Clamp target into [src[0], src[-1]].
        t = jnp.clip(target, src[0], src[-1])
        idx_hi = jnp.clip(jnp.searchsorted(src, t, side="right"), 1, n - 1)
        idx_lo = idx_hi - 1
        x0 = src[idx_lo]; x1 = src[idx_hi]
        w_hi = jnp.where(x1 > x0, (t - x0) / (x1 - x0), 0.0)
        return idx_lo, idx_hi, jnp.clip(w_hi, 0.0, 1.0)

    # Periodic: wrap target into [src[0], src[0] + period).
    t = jnp.mod(target - src[0], periodic_period) + src[0]
    idx_hi = jnp.searchsorted(src, t, side="right")
    idx_lo = jnp.mod(idx_hi - 1, n)
    idx_hi_w = jnp.mod(idx_hi, n)
    x0 = src[idx_lo]
    x1 = src[idx_hi_w]
    # Handle the wrap segment: when idx_lo == n-1 and idx_hi == 0, x1 < x0 by
    # the period.  Shift x1 by the period so the lerp distance is positive.
    x1_shift = jnp.where(x1 >= x0, x1, x1 + periodic_period)
    w_hi = (t - x0) / (x1_shift - x0)
    return idx_lo, idx_hi_w, jnp.clip(w_hi, 0.0, 1.0)


def _vertical_log_p_interp(field_src: jnp.ndarray, log_p_src: jnp.ndarray,
                           log_p_tgt: jnp.ndarray) -> jnp.ndarray:
    """Linear interp in log-p along last axis, with clamp-to-bound extrapolation.

    Parameters
    ----------
    field_src : (..., n_src)
    log_p_src : (n_src,)  ascending
    log_p_tgt : (..., n_tgt)
    """
    n_src = log_p_src.shape[0]
    t = jnp.clip(log_p_tgt, log_p_src[0], log_p_src[-1])
    idx_hi = jnp.clip(jnp.searchsorted(log_p_src, t, side="right"), 1, n_src - 1)
    idx_lo = idx_hi - 1
    x0 = log_p_src[idx_lo]
    x1 = log_p_src[idx_hi]
    w = jnp.where(x1 > x0, (t - x0) / (x1 - x0), 0.0)
    w = jnp.clip(w, 0.0, 1.0)
    f0 = jnp.take(field_src, idx_lo, axis=-1)
    f1 = jnp.take(field_src, idx_hi, axis=-1)
    return (1.0 - w) * f0 + w * f1


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def predict_ozone_ml(
    T: jnp.ndarray,
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    p_full: jnp.ndarray,
    coefs: MLOzoneCoefficients,
    *,
    mmr_to_vmr: bool = True,
    o3_floor: float = 1.0e-10,
) -> jnp.ndarray:
    """Predict ozone VMR from the model temperature column via ridge regression.

    Parameters
    ----------
    T : (ncol, nlev)
        Model-grid temperature [K].
    lat : (ncol,) [rad]
        Latitude of each column.
    lon : (ncol,) [rad]
        Longitude of each column.
    p_full : (ncol, nlev) [Pa]
        Full-level pressures of each column.
    coefs : MLOzoneCoefficients
        Pre-loaded UKESM ridge weights and grid metadata.
    mmr_to_vmr : bool, default True
        If True, treat the ridge output as O3 *mass mixing ratio* (kg/kg) and
        convert to volume mixing ratio via ``M_dry / M_o3``.  Set False if the
        upstream weights are already in VMR.
    o3_floor : float, default 1e-10
        Minimum VMR returned (avoids negative absorber concentrations).

    Returns
    -------
    o3_vmr : (ncol, nlev)
        Ozone VMR on the model vertical grid.

    Notes
    -----
    All operations are pure JAX and differentiable wrt ``T``.  Lat/lon enter
    only as gather indices and are treated as static for AD purposes.
    """
    lat_deg = jnp.degrees(lat)
    lon_deg = jnp.degrees(lon)

    j_lo, j_hi, w_lat = _bilinear_weights_1d(lat_deg, coefs.lat_uk)
    i_lo, i_hi, w_lon = _bilinear_weights_1d(lon_deg, coefs.lon_uk,
                                             periodic_period=360.0)

    def _gather_corner(arr_zll, j_idx, i_idx):
        # arr_zll: (..., n_lat, n_lon).  Output: (..., ncol).
        return arr_zll[..., j_idx, i_idx]

    def _bilinear(arr_zll):
        f00 = _gather_corner(arr_zll, j_lo, i_lo)
        f01 = _gather_corner(arr_zll, j_lo, i_hi)
        f10 = _gather_corner(arr_zll, j_hi, i_lo)
        f11 = _gather_corner(arr_zll, j_hi, i_hi)
        w_lat_b = w_lat
        w_lon_b = w_lon
        # broadcast against leading axes
        return (
            (1 - w_lat_b) * ((1 - w_lon_b) * f00 + w_lon_b * f01)
            + w_lat_b * ((1 - w_lon_b) * f10 + w_lon_b * f11)
        )

    # Bilinear interpolate scalers and coefficients onto model columns.
    x_mean_col = _bilinear(coefs.x_mean)   # (n_z, ncol)
    x_scale_col = _bilinear(coefs.x_scale) # (n_z, ncol)
    y_mean_col = _bilinear(coefs.y_mean)   # (n_z, ncol)
    y_scale_col = _bilinear(coefs.y_scale) # (n_z, ncol)
    coefs_col = _bilinear(coefs.coefs)     # (n_c, n_z, ncol)

    # Vertical interp model T -> UKESM pressure levels per column.
    log_p_src_model = jnp.log(jnp.maximum(p_full, 1.0e-10))     # (ncol, nlev_m)
    log_p_uk = jnp.log(jnp.maximum(coefs.p_uk, 1.0e-10))        # (n_z,)
    # Re-target: for each column, interp T(model log-p) -> T(UKESM log-p).
    # Use per-column 1-D linear interp.
    # Shape protocol: log_p_src varies per column, so vmap over columns.
    import jax
    def _interp_one_col(T_col, log_p_col):
        # T_col: (nlev_m,), log_p_col: (nlev_m,)
        # _vertical_log_p_interp expects ascending log_p_src.  Sort here.
        order = jnp.argsort(log_p_col)
        return _vertical_log_p_interp(
            T_col[order], log_p_col[order], log_p_uk
        )
    T_uk = jax.vmap(_interp_one_col)(T, log_p_src_model)   # (ncol, n_z)
    T_uk = T_uk.T                                           # (n_z, ncol)

    # Standardize, matmul, denormalize per column at UKESM resolution.
    x_norm = (T_uk - x_mean_col) / jnp.where(
        x_scale_col == 0.0, 1.0, x_scale_col
    )                                                       # (n_z, ncol)
    # y_norm[z, col] = sum_c coefs_col[c, z, col] * x_norm[c, col]
    y_norm = jnp.einsum("czk,ck->zk", coefs_col, x_norm)    # (n_z, ncol)
    o3_uk = y_norm * y_scale_col + y_mean_col               # (n_z, ncol)

    # Vertical interp UKESM -> model levels per column.
    o3_uk_col = o3_uk.T                                     # (ncol, n_z)
    def _interp_o3_one_col(o3_col, log_p_tgt_col):
        return _vertical_log_p_interp(o3_col, log_p_uk, log_p_tgt_col)
    o3_model = jax.vmap(_interp_o3_one_col)(o3_uk_col, log_p_src_model)

    if mmr_to_vmr:
        o3_model = o3_model * (constants.M_dry / constants.M_o3)

    return jnp.maximum(o3_model, o3_floor)
