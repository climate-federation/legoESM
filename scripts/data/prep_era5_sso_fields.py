"""Build IFS subgrid-orography input fields from ERA5's own invariant fields.

The IFS Lott-Miller subgrid-orography drag needs, per model column, the subgrid
height standard deviation, anisotropy gamma, principal-axis angle theta and
mean slope sigma; the IFS turbulent orographic form drag (TOFD) needs the
standard deviation of the filtered subgrid orography.  ERA5 carries all of them
(sdor, isor, anor, slor, sdfor) on its 0.25 deg grid.  This script:

1. ``--fetch``: pulls the five fields plus surface geopotential from ARCO-ERA5 (anonymous GCS) at one time, asserts they are
   bitwise identical at a second, distant time (they are invariant), and writes
   them to a raw NetCDF and a provenance JSON (URL, time indices, sha256 per
   field).
2. ``--build``: aggregates the raw fields into ``block_deg`` blocks, so the
   file holds the variance the model does NOT resolve (#1712).  This is read
   by ``legoesm.grids.topography.load_subgrid_orography``, which samples it
   bilinearly onto every model grid.

Aggregation (decisions voted 2026-11-15; see the PR body):

- std:  std_tot^2 = area-mean(sdor^2) + area-variance of the ERA5 resolved
  height inside the block (height NOT clipped at 0: ERA5 z is surface height,
  not bathymetry).
- gamma/theta/sigma: via the Baines-Palmer tensor K = <hx^2+hy^2>/2,
  L = <hx^2-hy^2>/2, M = <hx hy>, which are averages of gradient products and
  so aggregate linearly.  The resolved ERA5 terrain that becomes subgrid at
  block scale adds its own gradient tensor (slopes inside each block only),
  so std and K/L/M describe the same terrain.  The file stores
  K, L, M, so a runtime bilinear sample never interpolates an angle;
  gamma/theta/sigma are stored as well.
- sdfor: rms of sdfor only (the TOFD band is fixed, not model-relative).

Usage::

    python scripts/data/prep_era5_sso_fields.py --fetch --raw era5_sso_raw.nc
    python scripts/data/prep_era5_sso_fields.py --build --raw era5_sso_raw.nc \\
        --out era5_sso_1deg.nc --block-deg 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_subgrid_orography import block_area_mean  # noqa: E402

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_POLE_COS = 1e-6     # coeff-ok: rows with cos(lat) below this are pole rows (ERA5: 6e-17 vs next 4e-3)
RAW_VARS = {
    "sdor": "standard_deviation_of_orography",
    "isor": "anisotropy_of_sub_gridscale_orography",
    "anor": "angle_of_sub_gridscale_orography",
    "slor": "slope_of_sub_gridscale_orography",
    "sdfor": "standard_deviation_of_filtered_subgrid_orography",
    "z": "geopotential_at_surface",
}


def sso_tensor(sigma, gamma, theta):
    """Baines-Palmer K, L, M from slope sigma, anisotropy gamma, angle theta [rad]."""
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    g2 = np.asarray(gamma, dtype=np.float64) ** 2
    t2 = 2.0 * np.asarray(theta, dtype=np.float64)
    s = 0.5 * s2 * (1.0 - g2)
    return 0.5 * s2 * (1.0 + g2), s * np.cos(t2), s * np.sin(t2)


def sso_from_tensor(K, L, M):
    """Inverse of :func:`sso_tensor`.  Zero-slope cells -> sigma 0, gamma 1, theta 0."""
    K = np.asarray(K, dtype=np.float64)
    s = np.hypot(L, M)
    major = K + s
    flat = major <= 0.0
    safe = np.where(flat, 1.0, major)
    sigma = np.where(flat, 0.0, np.sqrt(np.maximum(major, 0.0)))
    gamma = np.where(flat, 1.0, np.sqrt(np.clip((K - s) / safe, 0.0, 1.0)))
    theta = np.where(flat, 0.0, 0.5 * np.arctan2(M, L))
    return sigma, gamma, theta


def _native_points(raw):
    """ERA5 point grid as cells: lat ascending, the -90 row dropped, lon from 0.

    Each 0.25 deg point is taken as its own cell, so no field is interpolated
    (a half-cell bilinear shift would erase the resolved 2-point variance).  The
    dropped south-pole row has zero area weight.
    """
    r = raw.sortby("latitude")
    lat = np.asarray(r["latitude"].values, dtype=np.float64)
    lon = np.asarray(r["longitude"].values, dtype=np.float64) % 360.0
    if not (np.isclose(lat[0], -90.0) and np.isclose(lat[-1], 90.0)):
        raise ValueError("expected a pole-to-pole ERA5 point grid")
    order = np.argsort(lon, kind="stable")
    lon = lon[order]
    dlat, dlon = np.diff(lat), np.diff(lon)
    if not (np.allclose(dlat, dlat[0]) and np.allclose(dlon, dlat[0])
            and np.isclose(lon.size * dlat[0], 360.0)):
        raise ValueError("expected a regular, global, equal-spacing lat-lon point grid")
    r = r.isel(latitude=slice(1, None), longitude=order)
    return r, lat[1:], lon


def resolved_slope_tensor(h, lat, factor: int, res_deg: float):
    """Block-mean Baines-Palmer K, L, M of the resolved terrain inside each block.

    At block scale the model keeps only each block's mean height, so every slope
    inside a block is subgrid, and every jump between blocks is resolved.  Each
    2x2 square of points wholly inside a block contributes its slope moments
    from its four edges, squared before averaging (so two-point and checkerboard
    variation count): hx^2 = mean of the two east-west edges squared, hy^2 = mean
    of the two north-south edges squared, hx*hy = mean of the four edge products.
    Per square this matrix is positive semi-definite (Cauchy-Schwarz), and all
    three moments share the square's cos(lat) weight, so K >= hypot(L, M)
    holds for every block.  Each zonal edge uses its own row's metric; an edge
    lying on a pole row has zero length and is dropped.
    """
    from legoesm import constants

    n_lat, n_lon = h.shape
    hb = h.reshape(n_lat // factor, factor, n_lon // factor, factor)
    latb = np.deg2rad(lat.reshape(-1, factor))
    d = np.deg2rad(res_deg) * constants.R_earth
    cos_m = np.cos(0.5 * (latb[:, 1:] + latb[:, :-1]))[:, :, None, None]
    cos_r = np.cos(latb)[:, :, None, None]
    a, b = hb[:, :-1, :, :-1], hb[:, :-1, :, 1:]      # south-west, south-east
    c, e = hb[:, 1:, :, :-1], hb[:, 1:, :, 1:]        # north-west, north-east
    # Each zonal edge uses its own row's metric; an edge on a pole row (zero
    # length) is dropped and the square keeps its other zonal edge.
    on_s, on_n = cos_r[:, :-1] > _POLE_COS, cos_r[:, 1:] > _POLE_COS
    hxs = np.where(on_s, (b - a) / (d * np.where(on_s, cos_r[:, :-1], 1.0)), 0.0)
    hxn = np.where(on_n, (e - c) / (d * np.where(on_n, cos_r[:, 1:], 1.0)), 0.0)
    n_x = on_s.astype(np.float64) + on_n
    hyw, hye = (c - a) / d, (e - b) / d
    w = np.broadcast_to(cos_m, a.shape)
    ws = w.sum(axis=(1, 3))

    def wmean(x):
        return (w * x).sum(axis=(1, 3)) / ws

    hx2 = wmean((hxs ** 2 + hxn ** 2) / n_x)
    hy2 = wmean(0.5 * (hyw ** 2 + hye ** 2))
    hxhy = wmean(0.5 * (hxs + hxn) / n_x * (hyw + hye))
    return 0.5 * (hx2 + hy2), 0.5 * (hx2 - hy2), hxhy


def build_era5_sso(raw, *, block_deg: float):
    """PURE: raw ERA5 point-grid Dataset -> block-aggregated SSO Dataset."""
    import xarray as xr
    from legoesm import constants

    r, lat, lon = _native_points(raw)
    fine_res_deg = float(np.median(np.diff(lat)))
    factor_f = float(block_deg) / fine_res_deg
    factor = int(round(factor_f))
    if abs(factor_f - factor) > 1e-6 or factor < 2:
        raise ValueError(
            f"block_deg must be an integer multiple (>= 2x) of {fine_res_deg}; "
            f"got {block_deg!r}")
    n_lat, n_lon = lat.size, lon.size
    if n_lat % factor or n_lon % factor:
        raise ValueError(f"{n_lat}x{n_lon} grid does not tile into factor {factor}")

    for v in ("sdor", "sdfor", "slor", "isor", "anor", "z"):
        if not np.isfinite(r[v].values).all():
            raise ValueError(f"raw field {v} has non-finite values")
    for v, lo, hi in (("sdor", 0.0, np.inf), ("sdfor", 0.0, np.inf),
                      ("slor", 0.0, np.inf), ("isor", 0.0, 1.0)):
        x = r[v].values
        if x.min() < lo or x.max() > hi:
            raise ValueError(f"raw field {v} outside [{lo}, {hi}]")
    # Tensor on the native points, so no angle is ever averaged.
    K, L, M = sso_tensor(r["slor"].values, r["isor"].values, r["anor"].values)
    fine = {"K": K, "L": L, "M": M,
            "sdor2": np.asarray(r["sdor"].values, dtype=np.float64) ** 2,
            "sdfor2": np.asarray(r["sdfor"].values, dtype=np.float64) ** 2,
            "h": np.asarray(r["z"].values, dtype=np.float64) / constants.g_ifs}

    mean = {v: block_area_mean(fine[v], lat, factor) for v in fine}

    h_mu = np.repeat(np.repeat(mean["h"], factor, axis=0), factor, axis=1)
    var_resolved = block_area_mean((fine["h"] - h_mu) ** 2, lat, factor)
    dK, dL, dM = resolved_slope_tensor(fine["h"], lat, factor, fine_res_deg)
    mean["K"], mean["L"], mean["M"] = mean["K"] + dK, mean["L"] + dL, mean["M"] + dM
    std = np.sqrt(np.maximum(mean["sdor2"], 0.0) + var_resolved)
    std_filt = np.sqrt(np.maximum(mean["sdfor2"], 0.0))
    sigma, gamma, theta = sso_from_tensor(mean["K"], mean["L"], mean["M"])

    lat_b = lat.reshape(-1, factor).mean(axis=1)
    lon_b = lon.reshape(-1, factor).mean(axis=1)
    dims = ("lat", "lon")
    out = xr.Dataset(
        {"SSO_STDH": (dims, std), "SSO_STDH_FILT": (dims, std_filt),
         "SSO_K": (dims, mean["K"]), "SSO_L": (dims, mean["L"]),
         "SSO_M": (dims, mean["M"]), "SSO_SIGMA": (dims, sigma),
         "SSO_GAMMA": (dims, gamma), "SSO_THETA": (dims, theta)},
        coords={"lat": lat_b, "lon": lon_b},
        attrs={"block_deg": float(block_deg), "fine_res_deg": float(fine_res_deg),
               "source": f"ERA5 invariants via {ARCO_URL}",
               "history": f"prep_era5_sso_fields.py --block-deg {block_deg} "
                          f"--fine-res-deg {fine_res_deg}"})
    units = {"SSO_STDH": "m", "SSO_STDH_FILT": "m", "SSO_K": "1", "SSO_L": "1",
             "SSO_M": "1", "SSO_SIGMA": "1", "SSO_GAMMA": "1", "SSO_THETA": "rad"}
    for k, u in units.items():
        out[k].attrs["units"] = u
    out["lat"].attrs = {"units": "degrees_north"}
    out["lon"].attrs = {"units": "degrees_east"}
    return out


def fetch_era5_sso(raw_path: str, *, time_a: str = "1979-01-01T00",
                  time_b: str = "2020-01-01T00") -> dict:
    """Fetch the invariant fields at ``time_a``; refuse unless bitwise equal at ``time_b``."""
    import gcsfs
    import xarray as xr

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(ARCO_URL), chunks=None)
    names = list(RAW_VARS.values())
    # ARCO pads times outside the ERA5 record with NaN, and NaN == NaN under
    # equal_nan would pass an invariance check on two empty slices: refuse NaN.
    a = ds[names].sel(time=time_a).load()
    b = ds[names].sel(time=time_b).load()
    for n in names:
        if not (np.isfinite(a[n].values).all() and np.isfinite(b[n].values).all()):
            raise RuntimeError(f"{n} has non-finite values at {time_a} or {time_b}")
        if not np.array_equal(a[n].values, b[n].values):
            raise RuntimeError(f"{n} differs between {time_a} and {time_b}")
    raw = a.rename({long: short for short, long in RAW_VARS.items()}).drop_vars("time")
    raw.to_netcdf(raw_path)
    prov = {"url": ARCO_URL, "time_a": time_a, "time_b": time_b,
            "sha256": {k: hashlib.sha256(np.ascontiguousarray(raw[k].values).tobytes())
                       .hexdigest() for k in RAW_VARS}}
    Path(raw_path).with_suffix(".json").write_text(json.dumps(prov, indent=2))
    return prov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--raw", required=True)
    p.add_argument("--out")
    p.add_argument("--block-deg", type=float)
    a = p.parse_args(argv)
    if a.fetch:
        fetch_era5_sso(a.raw)
    if a.build:
        import xarray as xr
        if not a.out or a.block_deg is None:
            p.error("--build needs --out and --block-deg")
        raw = xr.open_dataset(a.raw)
        try:
            build_era5_sso(raw, block_deg=a.block_deg).to_netcdf(a.out)
        finally:
            raw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
