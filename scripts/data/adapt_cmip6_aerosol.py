#!/usr/bin/env python
"""IDEALIZED simple-plume anthropogenic aerosol -> a gridded AOD for testing.

*** NOT the reference MACv2-SP. ***  The official input4MIPs guidance is that the
gridded MACv2-SP forcing MUST be computed with the reference ``mo_simple_plumes``
Fortran routine — use that for production.  This script is a SIMPLIFIED,
STRUCTURAL approximation (a rotated asymmetric-Gaussian plume + annual/seasonal
scaling + Angstrom slope) for idealized runs, smoke tests, and the deck plumbing;
its geometry and seasonal handling are DELIBERATELY simpler than the paper's
(Stevens et al. 2017, GMD 10, 433-476), and its magnitudes are NOT validated
against the reference.  Do not present its output as the CMIP6 MACv2-SP forcing.

``download_cmip6_forcing.py --channels aerosol`` fetches the MACv2-SP
(``MPI-M-MACv2SP-1-0``) parameter file (plume centres/widths/rotations/AOD +
annual/feature weights).  The AMIP loader (``forcing/external.py``,
``AerosolConfig``) wants a gridded ``aod(time, lat, lon)`` [dimensionless,
550 nm] which it zonal-averages to a column AOD; this evaluates the simplified
plumes onto a lat-lon grid to produce that field.  Per-band SSA / asymmetry are
NOT produced here (they would feed ``RRTMGPConfig.aerosol_ssa_bands/
aerosol_g_bands`` from PR #1276); only the 550 nm column AOD is written.
The vertical profile (integrates to the column AOD anyway) and the Twomey
cloud-droplet effect are out of scope.  Structural behaviour (plume location,
E/W asymmetry, annual + seasonal scaling, Angstrom slope) is unit-tested;
faithful magnitudes require the reference routine.

Usage::

    python scripts/data/adapt_cmip6_aerosol.py --in MACv2.0-SP_v1.nc \\
        --out data/cmip6_forcing/aerosol_macv2sp.nc \\
        --resolution 96 --year 2000

Consume with ``run_amip_smoke_deck.py --aerosol-forcing external --aerosol-file
<out>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Monthly mid-point "days since 0000-01-01" for the output time axis + seasonal
# cycle.  ``15.5 + 30.4375*m`` (30.4375 = 365.25/12) — the SAME convention the
# synthetic ``generate_amip_forcing.make_aerosol_clim`` uses and the loader's
# cyclic monthly interpolation (``_cyclic_phase_anchor``) is validated against.
_MONTH_MID_DOY = 15.5 + 30.4375 * np.arange(12)
_DAYS_PER_YEAR = 365.0
_REF_WAVELENGTH_NM = 550.0


def _rotated_asymmetric_gaussian(dlon, dlat, theta, sig_lon_e, sig_lon_w,
                                 sig_lat_n, sig_lat_s):
    """One MACv2-SP plume-feature spatial shape, normalised to a peak of 1.

    Rotate the offset from the plume centre by ``theta`` into the plume frame,
    then a 2-D Gaussian whose longitudinal width differs east/west of the centre
    and whose latitudinal width differs north/south (Stevens et al. 2017: the
    plumes trail off asymmetrically downwind).  ``dlon``/``dlat`` broadcast over
    the grid.
    """
    ct, st = np.cos(theta), np.sin(theta)
    x = dlon * ct + dlat * st          # along-plume (rotated lon)
    y = -dlon * st + dlat * ct         # cross-plume (rotated lat)
    sig_x = np.where(x >= 0.0, sig_lon_e, sig_lon_w)
    sig_y = np.where(y >= 0.0, sig_lat_n, sig_lat_s)
    return np.exp(-0.5 * (x / sig_x) ** 2 - 0.5 * (y / sig_y) ** 2)


def _seasonal_weight(doy, phase_doy, amplitude):
    """Per-plume seasonal cycle: ``1 + amplitude*cos(2*pi*(doy-phase)/365)``,
    clipped at 0 (a plume cannot contribute a negative AOD)."""
    return np.clip(
        1.0 + amplitude * np.cos(2.0 * np.pi * (doy - phase_doy) / _DAYS_PER_YEAR),
        0.0, None)


def evaluate_plume_aod(lon_deg, lat_deg, plumes, year: int, doy: float) -> np.ndarray:
    """550 nm anthropogenic AOD on the (lon, lat) grid for one (year, day).

    ``plumes`` is a list of dicts, one per plume, each with: ``lon``, ``lat``,
    ``theta`` [rad], ``sig_lon_e/w``, ``sig_lat_n/s`` [deg], ``aod`` (peak 550 nm
    AOD), ``features`` (list of ``(weight, dtheta, seasonal_phase_doy,
    seasonal_amplitude)``), and ``year_weight`` (a dict ``year -> factor`` OR a
    ``(years, factors)`` table).  Returns ``aod(lat, lon)`` >= 0.
    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)               # (nlat, nlon)
    total = np.zeros_like(lon2d, dtype=np.float64)
    for p in plumes:
        # Positive-finite widths + finite peak: a zero/NaN width would divide by
        # zero into NaN AOD (which np.clip does NOT repair) and leak into the
        # written field. Fail loudly instead.
        for _k in ("sig_lon_e", "sig_lon_w", "sig_lat_n", "sig_lat_s"):
            _s = p[_k]
            if not (np.isfinite(_s) and _s > 0.0):
                raise ValueError(f"plume width {_k}={_s!r} must be positive and "
                                 f"finite (zero/NaN divides into NaN AOD).")
        if not np.isfinite(p["aod"]):
            raise ValueError(f"plume peak aod={p['aod']!r} must be finite.")
        yw = _year_factor(p["year_weight"], year)
        if yw == 0.0:
            continue
        # shortest signed longitude offset (wrap at +-180).
        dlon = ((lon2d - p["lon"] + 180.0) % 360.0) - 180.0
        dlat = lat2d - p["lat"]
        shape = np.zeros_like(total)
        for (fw, dtheta, phase, amp) in p["features"]:
            g = _rotated_asymmetric_gaussian(
                dlon, dlat, p["theta"] + dtheta,
                p["sig_lon_e"], p["sig_lon_w"], p["sig_lat_n"], p["sig_lat_s"])
            shape = shape + fw * g * _seasonal_weight(doy, phase, amp)
        total = total + p["aod"] * yw * shape
    result = np.clip(total, 0.0, None)
    # Backstop: np.clip preserves NaN and +Inf, so a non-finite anywhere upstream
    # (year factor, feature weight/phase/amp, plume position/rotation, grid coords,
    # or an overflow) would leak into the written AOD. One check catches them all.
    if not np.all(np.isfinite(result)):
        raise ValueError("evaluated AOD is non-finite — check year weights, "
                         "seasonal amplitude/phase, plume position/rotation, or "
                         "the lon/lat grid.")
    return result


def _year_factor(year_weight, year: int) -> float:
    """Annual scaling for a plume: a dict lookup, a ``(years, factors)`` table
    (linearly interpolated + end-clamped), or a bare scalar."""
    if isinstance(year_weight, dict):
        return float(year_weight.get(year, 0.0))
    if isinstance(year_weight, (tuple, list)) and len(year_weight) == 2:
        years, factors = np.asarray(year_weight[0]), np.asarray(year_weight[1])
        return float(np.interp(year, years, factors))
    return float(year_weight)


def spectral_aod_scale(wavelength_nm: float, angstrom: float) -> float:
    """Angstrom-law scale from 550 nm AOD to another wavelength:
    ``(lambda/550)^(-alpha)``."""
    return float((wavelength_nm / _REF_WAVELENGTH_NM) ** (-angstrom))


# ---------------------------------------------------------------------------
# Build the deck AOD field (monthly climatology for a year)
# ---------------------------------------------------------------------------

def build_aod_climatology(plumes, year: int, nlat: int, nlon: int) -> tuple:
    """12-month ``aod(time, lat, lon)`` climatology for ``year`` on a regular
    lat-lon grid.  Returns ``(mid_days, lat, lon, aod)``."""
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    aod = np.stack(
        [evaluate_plume_aod(lon, lat, plumes, year, float(doy))
         for doy in _MONTH_MID_DOY], axis=0)           # (12, nlat, nlon)
    return _MONTH_MID_DOY.copy(), lat, lon, aod


def to_dataset(mid_days, lat, lon, aod) -> object:
    """Wrap ``(mid_days, lat, lon, aod)`` in the loader's aerosol Dataset schema
    (``aod(time, lat, lon)``, monthly noleap climatology)."""
    import xarray as xr

    return xr.Dataset(
        data_vars={"aod": (("time", "lat", "lon"), aod,
                           {"units": "1",
                            "long_name": "MACv2-SP anthropogenic AOD @ 550 nm"})},
        coords={
            "time": ("time", np.asarray(mid_days, dtype=np.float64),
                     {"units": "days since 0000-01-01 00:00:00",
                      "calendar": "noleap"}),
            "lat": ("lat", np.asarray(lat, dtype=np.float64),
                    {"units": "degrees_north"}),
            "lon": ("lon", np.asarray(lon, dtype=np.float64),
                    {"units": "degrees_east"}),
        },
        attrs={"source": "scripts/data/adapt_cmip6_aerosol.py (MACv2-SP simple "
                         "plume, Stevens et al. 2017; from-spec, unvalidated vs "
                         "reference)"},
    )


# ---------------------------------------------------------------------------
# Read MACv2-SP parameters
# ---------------------------------------------------------------------------

def read_macv2sp_plumes(ds, *, plume_map: dict | None = None) -> list:
    """Extract the plume parameter list from an opened MACv2-SP Dataset.

    FAILS CLOSED: every parameter that DEFINES a plume (centre, rotation, the
    four widths, the peak AOD, the annual-weight table, the feature weights) MUST
    be present — a missing one raises, never a silently-invented default (an
    unnoticed default is garbage physics, codex review).

    Orientation is resolved by DIMENSION NAME, not array shape, so a transposed
    (or square) table can never be silently mis-axised: every per-plume variable
    must carry the plume dimension of ``plume_lat``, and ``year_weight`` must
    carry both that plume dimension and the ``years`` dimension.  A non-finite
    value in any required array also raises.

    The reference file's 12-month ``time_weight`` seasonal table is DELIBERATELY
    NOT consumed: this simplified evaluator's single-cosine seasonal surrogate
    (``_seasonal_weight``) does not faithfully map onto it, so reading it would be
    a fidelity bug.  File-derived plumes are therefore annual-mean shapes
    (seasonal amplitude 0); the cosine surrogate is reachable only via the
    ``evaluate_plume_aod`` dict API for idealized runs.

    ``plume_map`` overrides the variable-name mapping (MACv2-SP releases differ;
    inspect with ``--list-vars``) — keys are the logical names ``lat lon theta
    aod sig_lon_e sig_lon_w sig_lat_n sig_lat_s ftr_weight years year_weight``.
    """
    names = {
        "lat": "plume_lat", "lon": "plume_lon", "theta": "plume_theta",
        "aod": "aod_spmx", "sig_lon_e": "sig_lon_E", "sig_lon_w": "sig_lon_W",
        "sig_lat_n": "sig_lat_N", "sig_lat_s": "sig_lat_S",
        "ftr_weight": "ftr_weight", "years": "years", "year_weight": "year_weight",
    }
    if plume_map:
        names.update(plume_map)

    def _da(key, *, required):
        v = names.get(key)
        if not v or v not in ds.variables:
            if required:
                raise ValueError(
                    f"MACv2-SP file lacks required variable {v!r} (logical "
                    f"{key!r}); this is a simplified plume evaluator — check the "
                    f"file/version, pass --list-vars, or use the reference "
                    f"mo_simple_plumes routine for the production forcing.")
            return None
        return ds[v]

    lat_da = _da("lat", required=True)
    plume_dim = lat_da.dims[0]
    n = lat_da.sizes[plume_dim]

    def _plume_first(key):
        """Required per-plume array, transposed so the plume axis leads (by NAME,
        not shape) — refuses any array not carrying the plume dimension."""
        da = _da(key, required=True)
        if plume_dim not in da.dims:
            raise ValueError(
                f"{names[key]!r} has dims {da.dims}, which do not include the "
                f"plume dimension {plume_dim!r}; refusing to guess the axis order.")
        arr = np.asarray(da.transpose(plume_dim, ...).values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{names[key]!r} contains non-finite values.")
        return arr

    lat = np.asarray(lat_da.values, dtype=np.float64)
    if not np.all(np.isfinite(lat)):
        raise ValueError("plume_lat contains non-finite values.")
    lon, theta, aod = _plume_first("lon"), _plume_first("theta"), _plume_first("aod")
    sig_lon_e, sig_lon_w = _plume_first("sig_lon_e"), _plume_first("sig_lon_w")
    sig_lat_n, sig_lat_s = _plume_first("sig_lat_n"), _plume_first("sig_lat_s")
    ftr_w = _plume_first("ftr_weight")             # (n,) or (n, nfeatures)

    years_da = _da("years", required=True)
    year_dim = years_da.dims[0]
    years = np.asarray(years_da.values, dtype=np.float64)
    if not np.all(np.isfinite(years)):
        raise ValueError("years contains non-finite values.")
    if years.size > 1 and not np.all(np.diff(years) > 0.0):
        raise ValueError("years must be strictly increasing (the annual factor "
                         "is np.interp'd, which needs an ascending x axis).")
    yw_da = _da("year_weight", required=True)
    if plume_dim not in yw_da.dims or year_dim not in yw_da.dims:
        raise ValueError(
            f"year_weight has dims {yw_da.dims}; expected both the plume "
            f"dimension {plume_dim!r} and the year dimension {year_dim!r}.")
    year_w = np.asarray(yw_da.transpose(year_dim, plume_dim).values,
                        dtype=np.float64)          # (nyr, n) — oriented by name
    if not np.all(np.isfinite(year_w)):
        raise ValueError("year_weight contains non-finite values.")

    plumes = []
    for i in range(n):
        nf = ftr_w.shape[1] if ftr_w.ndim == 2 else 1
        feats = []
        for f in range(nf):
            fw = float(ftr_w[i, f]) if ftr_w.ndim == 2 else float(ftr_w[i])
            # dtheta/phase/amp = 0: the file carries no surrogate this simplified
            # evaluator faithfully consumes (see docstring).
            feats.append((fw, 0.0, 0.0, 0.0))
        plumes.append({
            "lon": float(lon[i]), "lat": float(lat[i]), "theta": float(theta[i]),
            "sig_lon_e": float(sig_lon_e[i]), "sig_lon_w": float(sig_lon_w[i]),
            "sig_lat_n": float(sig_lat_n[i]), "sig_lat_s": float(sig_lat_s[i]),
            "aod": float(aod[i]), "features": feats,
            "year_weight": (years, year_w[:, i]),
        })
    return plumes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_file", type=Path, required=True,
                   help="MACv2-SP parameter file (MACv2.0-SP_v1.nc).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output gridded AOD NetCDF (deck --aerosol-file).")
    p.add_argument("--year", type=int, default=2000,
                   help="Year for the annual scaling (default 2000).")
    p.add_argument("--resolution", type=int, default=96,
                   help="Number of latitudes (nlon = 2*nlat). Default 96.")
    p.add_argument("--list-vars", action="store_true",
                   help="Print the MACv2-SP file's variables and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import xarray as xr

    args = parse_args(argv)
    with xr.open_dataset(args.in_file, decode_times=False) as ds:
        if args.list_vars:
            print("MACv2-SP variables:", list(ds.variables))
            return 0
        plumes = read_macv2sp_plumes(ds)
    nlat = args.resolution
    mid_days, lat, lon, aod = build_aod_climatology(plumes, args.year, nlat,
                                                    2 * nlat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    to_dataset(mid_days, lat, lon, aod).to_netcdf(args.out)
    print(f"Wrote {args.out} (12-month AOD climatology for {args.year}, "
          f"{nlat}x{2 * nlat}; peak AOD {aod.max():.3f})")
    print("Consume: run_amip_smoke_deck.py --aerosol-forcing external "
          f"--aerosol-file {args.out}")
    print("NOTE: from-specification MACv2-SP (Stevens 2017); validate vs the "
          "reference sp_aop / real parameter file before production use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
