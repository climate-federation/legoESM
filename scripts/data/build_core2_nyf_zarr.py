#!/usr/bin/env python
"""Build the legoESM CORE-II Normal-Year-Forcing zarr cache from the raw
NEMO/ORCA1 COREv2 NetCDF files.

This makes legoESM's ``load_core2_nyf`` read the *exact same* CORE-II Normal
Year Forcing (Large & Yeager 2009) that the NEMO ORCA1 reference run is forced
with — the prerequisite for a faithful legoESM-vs-NEMO comparison
(see ``OMIP_faithful.md``).

Source files (NEMO ORCA1 ``INPUTS/`` after ``deploy.sh``), all on the CORE
T62 grid (LAT=94 Gaussian, LON=192 regular 1.875 deg):

    u_10.15JUNE2009.nc   U_10   [m/s]      6-hourly (1460)
    v_10.15JUNE2009.nc   V_10   [m/s]      6-hourly (1460)
    t_10.15JUNE2009.nc   T_10   [K]        6-hourly (1460)
    q_10.15JUNE2009.nc   Q_10   [kg/kg]    6-hourly (1460)
    ncar_rad...nc        SWDN, LWDN [W/m2] daily    (365)
    ncar_precip...nc     RAIN, SNOW [kg/m2/s] monthly (12)

Target: an xarray Dataset on the native CORE-II **6-hourly (1460-record)** time
axis -- preserving the wind frequency so legoESM recomputes the *nonlinear* bulk
fluxes (tau ~ Cd(|U|)|U|U; turbulent heat ~ |U|dT) from the same sub-daily winds
NEMO uses. Daily radiation (365) and monthly precip (12) are broadcast onto the
6-hourly axis. Variables are exactly those
``legoesm.ocean.forcing.load_core2_nyf`` reads back (lon, lat, time_s, u10, v10,
T_air, q_air, sw_down, lw_down, precip, runoff), written to
``<fidelity-cache>/forcing/core2_nyf/nyf.zarr``.

Runoff: set to zero here (NEMO applies the Dai-Trenberth-Depoorter runoff via a
separate file on the eORCA1 grid; legoESM carries it via
``ocean/forcing/dai_trenberth.py`` / SSS restoring instead). Documented in
``OMIP_faithful.md`` as a known difference for the first comparison pass.

Usage (sbatch / compute node):
    python scripts/build_core2_nyf_zarr.py \
        --inputs-dir /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Calendar constants. CORE-II Normal Year is a 365-day perpetual year; winds /
# T / q are stored 6-hourly (4 records/day = 1460 records/year). We KEEP that
# native frequency so legoESM evaluates the nonlinear bulk fluxes
# (tau ~ Cd(|U|)|U|U) at sub-daily wind speed, as NEMO does -- averaging the
# winds to daily first would systematically weaken the annual-mean stress
# (flux of mean != mean of flux).
_SEC_PER_6H = 21600.0
_REC_PER_DAY = 4
_N_REC = 1460
_DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def _default_out() -> Path:
    """Canonical legoESM CORE-II cache path (mirrors core2._cache_dir())."""
    from legoesm.ocean.fidelity import cache as _cache
    return _cache.sub("forcing") / "core2_nyf" / "nyf.zarr"


def build(inputs_dir: Path, out: Path, wind_variant: str = "mod") -> Path:
    import xarray as xr

    def _open(name: str) -> "xr.Dataset":
        p = inputs_dir / f"{name}.15JUNE2009.nc"
        if not p.exists():
            raise FileNotFoundError(f"missing COREv2 file: {p}")
        return xr.open_dataset(p, decode_times=False)

    du, dv = _open("u_10"), _open("v_10")
    dt, dq = _open("t_10"), _open("q_10")
    drad, dprec = _open("ncar_rad"), _open("ncar_precip")
    dslp = _open("slp")

    lat = np.asarray(du["LAT"].values, dtype=np.float64)   # (94,) Gaussian
    lon = np.asarray(du["LON"].values, dtype=np.float64)   # (192,) 0..360
    n_lat, n_lon = lat.size, lon.size

    def _f64(a):
        return np.asarray(a, dtype=np.float64)

    # u / v / T / q: KEEP the native 6-hourly axis (1460 records).
    #
    # WIND VARIANT.  The CORE-II files carry both the raw NCEP-derived winds
    # (U_10/V_10) and the Large & Yeager corrected ones (U_10_MOD/V_10_MOD),
    # and NEMO ORCA1's namelist_cfg reads the _MOD pair (sn_wndi=U_10_MOD,
    # sn_wndj=V_10_MOD).  We built from the base fields, which are the raw
    # data -- a true statement that justified the wrong choice, because the
    # oracle uses the correction.
    #
    # The _MOD fields are the scatterometer (QuikSCAT) tropical adjustment:
    # NCEP-1 10 m winds are ~20-30% too weak over the equatorial Pacific, and
    # measured on this file's own grid the paired speed sqrt(u^2+v^2) differs
    # by +32.6% over nino3 and +8.9% globally, with only a 3-10 degree turning
    # angle -- so it is a SPEED correction, not a rotation
    # (scripts/validate/ocean_fidelity/core2_base_vs_mod_winds.py).  Since
    # stress goes as the square, the base fields give roughly 1.8x too little
    # equatorial wind stress.
    _WIND_VARS = {"mod": ("U_10_MOD", "V_10_MOD"), "base": ("U_10", "V_10")}
    if wind_variant not in _WIND_VARS:
        raise SystemExit(f"--wind-variant must be one of {sorted(_WIND_VARS)}")
    _un, _vn = _WIND_VARS[wind_variant]
    for _nm, _ds in ((_un, du), (_vn, dv)):
        if _nm not in _ds:
            raise SystemExit(f"{_nm} not in the CORE-II file; refusing to "
                             "silently fall back to another variable")
    print(f"[wind] variant {wind_variant!r} -> {_un}/{_vn}")
    u10 = _f64(du[_un].values)
    v10 = _f64(dv[_vn].values)
    # T / q / radiation / precip: the SAME _MOD-vs-base choice as the winds.
    # GLM's review: the CORE bulk formulae were tuned with the corrected set,
    # so mixing raw winds with corrected humidity (or vice versa) is
    # inconsistent. Verified before switching -- every base/_MOD pair has
    # IDENTICAL shape, time axis and physical range (job 9450592); only the
    # `units` attribute is absent on the _MOD variants, which this builder
    # never reads. Deltas are modest next to the wind: T +1.2 K, q -3%,
    # SWDN -5.4 W/m2, LWDN -0.7 W/m2, precip +15%.
    _t_var = "T_10_MOD" if wind_variant == "mod" else "T_10"
    _q_var = "Q_10_MOD" if wind_variant == "mod" else "Q_10"
    T_air = _f64(dt[_t_var].values)                        # K
    # CORE-II Q_10 carries small (~-6e-3) negative specific humidities over
    # arid land (~3.8% of points, min over the Sahel) — a known artifact of the
    # source product. All such points are land (masked when regridded to the
    # ocean grid), so this clip to the physical floor is negligible vs what NEMO
    # ingests over open ocean, and prevents negative-humidity NaNs downstream.
    q_air = np.maximum(_f64(dq[_q_var].values), 0.0)       # kg/kg
    if u10.shape[0] != _N_REC:
        raise ValueError(
            f"expected {_N_REC} 6-hourly wind records, got {u10.shape[0]}"
        )

    # Radiation is daily (365): broadcast each day across its 4 6-hourly slots.
    _sw_var = "SWDN_MOD" if wind_variant == "mod" else "SWDN"
    _lw_var = "LWDN_MOD" if wind_variant == "mod" else "LWDN"
    sw_down = np.repeat(_f64(drad[_sw_var].values), _REC_PER_DAY, axis=0)
    lw_down = np.repeat(_f64(drad[_lw_var].values), _REC_PER_DAY, axis=0)
    # Precip is monthly (12): broadcast each month across its (days*4) slots.
    # KEEP snow as its own channel (NEMO reads SNOW separately for the
    # snow-fusion / snow-heat-content terms of q_ns); precip stays the TOTAL.
    snow = np.repeat(
        _f64(dprec["SNOW"].values),                          # kg/m^2/s
        _DAYS_PER_MONTH * _REC_PER_DAY, axis=0,
    )
    # PRC_MOD is the corrected TOTAL precipitation -- the direct analogue of
    # our RAIN+SNOW construction (measured means 2.656e-5 vs 2.657e-5), and
    # what NEMO's sn_prec reads. SNOW has no _MOD variant and NEMO reads it
    # plain, so the snow channel above is unchanged either way.
    _prc = (_f64(dprec["PRC_MOD"].values) if wind_variant == "mod"
            else _f64(dprec["RAIN"].values) + _f64(dprec["SNOW"].values))
    precip = np.repeat(_prc, _DAYS_PER_MONTH * _REC_PER_DAY, axis=0)
    runoff = np.zeros_like(precip)                          # see module docstring
    # Sea-level pressure: 6-hourly like the winds (NEMO sn_slp), used for
    # moist-air density + the Goff saturation humidity.
    slp = _f64(dslp["SLP"].values)
    if float(np.nanmedian(slp)) < 2000.0:                   # hPa -> Pa guard
        slp = slp * 100.0

    for nm, a in [("u10", u10), ("v10", v10), ("T_air", T_air),
                  ("q_air", q_air), ("sw_down", sw_down), ("lw_down", lw_down),
                  ("precip", precip), ("runoff", runoff), ("snow", snow),
                  ("slp", slp)]:
        if a.shape != (_N_REC, n_lat, n_lon):
            raise ValueError(f"{nm} shape {a.shape} != {(_N_REC, n_lat, n_lon)}")

    time_s = (np.arange(_N_REC, dtype=np.float64) + 0.5) * _SEC_PER_6H

    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
            "T_air": (("time", "lat", "lon"), T_air),
            "q_air": (("time", "lat", "lon"), q_air),
            "sw_down": (("time", "lat", "lon"), sw_down),
            "lw_down": (("time", "lat", "lon"), lw_down),
            "precip": (("time", "lat", "lon"), precip),
            "runoff": (("time", "lat", "lon"), runoff),
            "snow": (("time", "lat", "lon"), snow),
            "slp": (("time", "lat", "lon"), slp),
            "time_s": (("time",), time_s),
        },
        coords={"lon": ("lon", lon), "lat": ("lat", lat)},
        attrs={
            "source": "CORE-II Normal Year Forcing (Large & Yeager 2009), "
                      "from NEMO ORCA1 INPUTS",
            "note": "runoff=0 here; Dai-Trenberth applied separately by NEMO",
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    ds.to_zarr(out, mode="w")
    return out


def _validate(out: Path) -> None:
    """Load via the production reader + sanity-check physical ranges."""
    from legoesm.ocean.forcing.core2 import load_core2_nyf
    f = load_core2_nyf(cache_dir=out.parent, n_time=365)
    checks = {
        # Polar/continental 10 m air over Antarctica is genuinely ~183 K in the
        # CORE-II NYF; those points are land-masked in the ocean grid.
        "T_air[K]": (f.T_air, 180.0, 330.0),
        "q_air[kg/kg]": (f.q_air, 0.0, 0.06),
        "sw_down[W/m2]": (f.sw_down, 0.0, 500.0),
        "lw_down[W/m2]": (f.lw_down, 50.0, 500.0),
        "precip[kg/m2/s]": (f.precip, 0.0, 1e-3),
        "|u10|[m/s]": (np.abs(f.u10), 0.0, 60.0),
        "snow[kg/m2/s]": (f.snow, 0.0, 1e-3),
        # Upper bound 1200 hPa: CORE-II SLP over the Antarctic/Greenland ice
        # sheets is a below-ground sea-level reduction (fictitious, up to
        # ~1160 hPa) — those cells are land-masked when regridded to ocean.
        "slp[Pa]": (f.slp, 87000.0, 120000.0),
    }
    print(f"  loaded: lon{f.lon.shape} lat{f.lat.shape} time{f.time_s.shape}")
    ok = True
    for nm, (a, lo, hi) in checks.items():
        amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
        within = (amin >= lo - 1e-6) and (amax <= hi + 1e-6)
        ok = ok and within
        print(f"  {'OK ' if within else 'BAD'} {nm}: [{amin:.4g}, {amax:.4g}] "
              f"(expect [{lo}, {hi}])")
    if not ok:
        raise SystemExit("range validation FAILED — inspect units/variables")
    print("  range validation PASSED")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-dir", type=Path, required=True,
                   help="NEMO ORCA1 INPUTS dir containing *.15JUNE2009.nc")
    p.add_argument("--out", type=Path, default=None,
                   help="output nyf.zarr path (default: legoESM core2 cache)")
    p.add_argument("--wind-variant", choices=("mod", "base"), default="mod",
                   help="Which CORE-II variant to read for EVERY corrected "
                        "channel. 'mod' (default) = the Large & Yeager "
                        "bias-corrected fields NEMO's namelist reads: "
                        "U_10_MOD/V_10_MOD, T_10_MOD, Q_10_MOD, SWDN_MOD, "
                        "LWDN_MOD, PRC_MOD (SNOW has no _MOD and is read "
                        "plain, as NEMO does). 'base' = the raw fields this "
                        "builder used before 2026-08-21, kept only to "
                        "reproduce old caches. The name is historical -- it "
                        "selects all channels, not just the wind.")
    args = p.parse_args()
    out = args.out if args.out is not None else _default_out()
    print(f"building CORE-II NYF zarr  ->  {out}  (wind {args.wind_variant})")
    build(args.inputs_dir, out, wind_variant=args.wind_variant)
    print("validating ...")
    _validate(out)
    print(f"DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
