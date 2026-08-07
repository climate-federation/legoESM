#!/usr/bin/env python
"""Stage ERA5 monthly-mean reference variables for the AMIP pattern scorer.

Builds ``<ref_root>/reanalysis_ERA5/mon/<var>/<var>_native6_ERA5_Amon_mon_
<start>-<end>.nc`` from the Levante ERA5 pool so that
``scripts/plot/plot_amip_pattern_eval.py`` resolves them with no code change.

Currently supported variables (ECMWF parameter codes):

===== ==== ================ ======================================= =========
var   code pool tree        standard_name                           units
===== ==== ================ ======================================= =========
wap   135  pl/an/1M         lagrangian_tendency_of_air_pressure     Pa s-1
psl   151  sf/an/1M         air_pressure_at_sea_level               Pa
===== ==== ================ ======================================= =========

Why these two.  ``wap`` answers *where the model lifts air*, the direct
question behind a cloud-placement defect; ``psl`` replaces the ERA5 ``ps``
proxy the scorer currently substitutes for it.

Grid / level / time conventions are matched to the already-staged variables
(``ua``, ``ta``, ``ps``, ...) so nothing downstream has to special-case these
files:

* horizontal: 0.25 deg regular lat-lon, 1440 x 721, lat ASCENDING -90..90,
  lon 0..359.75.  The pool ``1M`` files are on a *reduced Gaussian N320*
  grid, so a bilinear remap is required; the source grid is recorded in the
  ``source_grid`` global attribute.
* vertical (``wap`` only): CMIP6 ``plev19``, DESCENDING 100000..100 Pa.  This
  is exactly the axis the model's own CMOR output uses
  (``legoesm.io.cmor_output.CMIP6_PLEV19``), so the scorer's
  ``ref.interp(plev=model.plev)`` becomes an identity rather than an
  interpolation.  The staged ``ua``/``ta`` carry the full 37 ERA5 levels;
  plev19 is a strict subset of those, and the extra 18 levels are pure
  volume with no model counterpart.
* time: monthly means stamped at the TRUE month midpoint in
  ``days since 1850-1-1 00:00:00`` (proleptic_gregorian), with ``time_bnds``
  spanning the calendar month -- byte-compatible with the staged files.

Sign convention (stated per the sign-check gate).  ``wap = omega = Dp/Dt``
is POSITIVE DOWNWARD: pressure increases downward, so descending air gives
omega > 0 and ascent gives omega < 0.  This is the CMIP6 ``wap`` convention
and matches ``legoesm.grids.vertical.compute_omega_total``.  ``--verify``
asserts it against known features (ITCZ/warm pool ascent, subtropical-high
descent) and FAILS the run if the sign is flipped, because a flipped omega
would silently reverse every downstream circulation comparison.

Usage::

    python scripts/data/prep_era5_reference_mon.py --var wap psl \\
        --start-year 1979 --end-year 2014 \\
        --ref-root /scratch/b/b309178/climateeval_data --verify

Run this on a COMPUTE node (``sbatch --account=bb1596 --partition=compute``);
the merged ``wap`` file is ~34 GB and the login node is shared.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ERA5_ROOT = Path("/pool/data/ERA5/E5")
CDO = "/sw/spack-levante/cdo-2.2.2-4z4icb/bin/cdo"

# Reference epoch of the already-staged files.
TIME_UNITS = "days since 1850-1-1 00:00:00"
CALENDAR = "proleptic_gregorian"

# Target horizontal grid, matching the staged ua/ta/ps files exactly.
TARGET_GRID = """\
gridtype  = lonlat
gridsize  = 1038240
xsize     = 1440
ysize     = 721
xname     = lon
xlongname = "Longitude"
xunits    = "degrees_east"
yname     = lat
ylongname = "Latitude"
yunits    = "degrees_north"
xfirst    = 0
xinc      = 0.25
yfirst    = -90
yinc      = 0.25
"""

# CMIP6 plev19 [Pa], descending.  Kept in sync with
# legoesm.io.cmor_output.CMIP6_PLEV19 by test_prep_era5_reference_mon.
PLEV19_PA = (
    100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0,
    30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, 5000.0,
    3000.0, 2000.0, 1000.0, 500.0, 100.0,
)


class VarSpec:
    """Everything that differs between one staged variable and another."""

    def __init__(self, name, code, tree, standard_name, long_name, units,
                 is_3d):
        self.name = name
        self.code = code
        self.tree = tree          # "pl" or "sf"
        self.standard_name = standard_name
        self.long_name = long_name
        self.units = units
        self.is_3d = is_3d

    def grib(self, year: int) -> Path:
        stem = "E5pl00" if self.tree == "pl" else "E5sf00"
        return (ERA5_ROOT / self.tree / "an" / "1M" / str(self.code)
                / f"{stem}_1M_{year}_{self.code}.grb")


VAR_SPECS = {
    "wap": VarSpec(
        "wap", 135, "pl",
        "lagrangian_tendency_of_air_pressure",
        "omega (=dp/dt)", "Pa s-1", is_3d=True,
    ),
    "psl": VarSpec(
        "psl", 151, "sf",
        "air_pressure_at_sea_level",
        "Sea Level Pressure", "Pa", is_3d=False,
    ),
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def month_midpoint_time_axis(start_year: int, end_year: int):
    """Monthly time centres + bounds, in ``days since 1850-1-1``.

    Returns ``(centres, bnds)`` with shapes ``(n,)`` and ``(n, 2)``.  Each
    centre is the exact midpoint of its calendar month, which is the
    convention the staged ERA5 files use (1979-01 -> 47131.5, bounds
    47116/47147), NOT a fixed day-of-month offset -- a fixed offset would
    drift against the staged files in short months.
    """
    import pandas as pd

    epoch = pd.Timestamp("1850-01-01")
    edges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            edges.append(pd.Timestamp(year=year, month=month, day=1))
    edges.append(pd.Timestamp(year=end_year + 1, month=1, day=1))
    days = np.array([(e - epoch).total_seconds() / 86400.0 for e in edges])
    bnds = np.stack([days[:-1], days[1:]], axis=1)
    centres = bnds.mean(axis=1)
    return centres, bnds


def cell_bounds(centres: np.ndarray) -> np.ndarray:
    """Contiguous cell bounds for a monotonic 1-D coordinate, shape (n, 2).

    Interior edges are midpoints between neighbours; the two outer edges are
    extrapolated by half the adjacent spacing.  Used for ``lat_bnds`` /
    ``lon_bnds`` so the emitted files carry the same CF metadata as the
    already-staged variables.
    """
    c = np.asarray(centres, dtype=float)
    if c.size < 2:
        raise ValueError("cell_bounds needs at least 2 points")
    mid = 0.5 * (c[:-1] + c[1:])
    lo = np.concatenate([[c[0] - (mid[0] - c[0])], mid])
    hi = np.concatenate([mid, [c[-1] + (c[-1] - mid[-1])]])
    return np.stack([lo, hi], axis=1)


def stamp(day_since_1850: float) -> str:
    """Format a time coordinate value as the staged filename timestamp.

    ``47131.5`` -> ``"19790116120000"`` (the ``ua``/``ta``/``ps`` pattern).
    """
    import pandas as pd

    t = pd.Timestamp("1850-01-01") + pd.Timedelta(days=float(day_since_1850))
    return t.strftime("%Y%m%d%H%M%S")


def build_cdo_command(spec: VarSpec, src: Path, dst: Path,
                      grid_file: Path) -> list[str]:
    """CDO pipeline: reduced-Gaussian GRIB -> staged-convention NetCDF.

    Operators apply RIGHT to LEFT, so the order below is:
    ``setgridtype,regular`` (reduced -> full Gaussian) -> ``sellevel``
    (plev19 subset, done BEFORE the remap so only 19 levels are
    interpolated) -> ``remapbil`` (-> 0.25 deg regular lat-lon) ->
    ``invertlev`` (-> descending pressure) -> ``chname``.
    """
    chain = [CDO, "-s", "-f", "nc4", f"-chname,var{spec.code},{spec.name}"]
    if spec.is_3d:
        levels = ",".join(f"{p:g}" for p in PLEV19_PA)
        chain += ["-invertlev", f"-remapbil,{grid_file}",
                  f"-sellevel,{levels}"]
    else:
        chain += [f"-remapbil,{grid_file}"]
    chain += ["-setgridtype,regular", str(src), str(dst)]
    return chain


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"{res.stdout}\n{res.stderr}"
        )


def _finalise_metadata(path: Path, spec: VarSpec, start_year: int,
                       end_year: int) -> None:
    """Rewrite the time axis and CF attributes of the merged file IN PLACE.

    Only coordinate variables are touched, so this costs nothing on a 34 GB
    payload.  CDO emits ``hours since <first month>``; the staged files use
    month-midpoint ``days since 1850-1-1`` with bounds.
    """
    import netCDF4 as nc

    centres, bnds = month_midpoint_time_axis(start_year, end_year)
    with nc.Dataset(path, "a") as ds:
        n = ds.dimensions["time"].size
        if n != centres.size:
            raise RuntimeError(
                f"{path.name}: {n} time records but {centres.size} months "
                f"expected for {start_year}-{end_year}"
            )
        if "bnds" not in ds.dimensions:
            ds.createDimension("bnds", 2)

        tv = ds.variables["time"]
        tv[:] = centres
        tv.units = TIME_UNITS
        tv.calendar = CALENDAR
        tv.standard_name = "time"
        tv.long_name = "time"
        tv.axis = "T"
        tv.bounds = "time_bnds"

        if "time_bnds" not in ds.variables:
            ds.createVariable("time_bnds", "f8", ("time", "bnds"))
        ds.variables["time_bnds"][:, :] = bnds

        for cname in ("lat", "lon"):
            cv = ds.variables[cname]
            bname = f"{cname}_bnds"
            if bname not in ds.variables:
                ds.createVariable(bname, "f8", (cname, "bnds"))
            ds.variables[bname][:, :] = cell_bounds(cv[:])
            cv.bounds = bname

        if spec.is_3d:
            pv = ds.variables["plev"]
            pv.units = "Pa"
            pv.standard_name = "air_pressure"
            pv.long_name = "pressure"
            pv.positive = "down"
            pv.axis = "Z"
            pv.stored_direction = "decreasing"

        dv = ds.variables[spec.name]
        dv.standard_name = spec.standard_name
        dv.long_name = spec.long_name
        dv.units = spec.units

        ds.source_grid = (
            "ERA5 pool /pool/data/ERA5/E5 monthly means (1M) on a reduced "
            "Gaussian N320 grid; converted with CDO setgridtype,regular and "
            "remapped bilinearly to 0.25 deg regular lat-lon"
        )
        ds.source_param = f"ECMWF parameter {spec.code}"
        ds.comment = (
            "Contains modified Copernicus Climate Change Service Information"
        )
        ds.staged_by = "scripts/data/prep_era5_reference_mon.py"


def stage_variable(spec: VarSpec, start_year: int, end_year: int,
                   ref_root: Path, tmp_root: Path) -> Path:
    """Extract, remap, merge and finalise one variable.  Returns the path."""
    out_dir = ref_root / "reanalysis_ERA5" / "mon" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [y for y in range(start_year, end_year + 1)
               if not spec.grib(y).exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec.name}: no ERA5 pool GRIB for years {missing} "
            f"(looked under {spec.grib(start_year).parent})"
        )

    centres, _ = month_midpoint_time_axis(start_year, end_year)
    fname = (f"{spec.name}_native6_ERA5_Amon_mon_"
             f"{stamp(centres[0])}-{stamp(centres[-1])}.nc")
    final = out_dir / fname

    with tempfile.TemporaryDirectory(prefix=f"era5_{spec.name}_",
                                     dir=str(tmp_root)) as td:
        tmp = Path(td)
        grid_file = tmp / "target_grid.txt"
        grid_file.write_text(TARGET_GRID)

        yearly = []
        for year in range(start_year, end_year + 1):
            dst = tmp / f"{spec.name}_{year}.nc"
            _run(build_cdo_command(spec, spec.grib(year), dst, grid_file))
            yearly.append(dst)
            log.info("  %s %d done", spec.name, year)

        log.info("  merging %d yearly files -> %s", len(yearly), final.name)
        _run([CDO, "-s", "-f", "nc4", "mergetime",
              *[str(p) for p in yearly], str(final)])

    _finalise_metadata(final, spec, start_year, end_year)
    log.info("  wrote %s (%.1f GB)", final,
             final.stat().st_size / 1024 ** 3)
    return final


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _area_weighted_mean(field: np.ndarray, lat_deg: np.ndarray) -> float:
    """cos(lat)-weighted global mean.  A raw mean on a lat-lon grid is wrong."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float)))
    w = np.broadcast_to(w[:, None], field.shape)
    good = np.isfinite(field)
    if not good.all():
        raise ValueError("non-finite values in field (NaN is fatal)")
    return float((field * w).sum() / w.sum())


def _box_mean(da, lat0, lat1, lon0, lon1) -> float:
    sub = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    return _area_weighted_mean(np.asarray(sub.values), sub.lat.values)


def verify_wap(path: Path) -> bool:
    """Assert the omega SIGN convention against known circulation features.

    Positive-downward omega means the ITCZ / warm pool / Amazon must be
    NEGATIVE (ascent) and the subtropical highs POSITIVE (descent).  A file
    with ascent positive is flipped relative to CMIP6 and would reverse
    every downstream comparison, so this returns False rather than warning.
    """
    import xarray as xr

    ok = True
    # chunk by SINGLE month: one chunk is 19*721*1440*4 = 79 MB, so the
    # verification never materialises the ~34 GB payload (the scorer's
    # own _open_ref chunks the same way after an 87 GB ta OOM).
    with xr.open_dataset(path, chunks={"time": 1}) as ds:
        da = ds["wap"]
        print(f"  units = {da.attrs.get('units')!r}  "
              f"plev = {da.plev.size} levels "
              f"{da.plev.values[0]:g}..{da.plev.values[-1]:g} Pa")
        w = da.sel(plev=50000.0).mean("time").load()
        if not np.isfinite(w.values).all():
            print("  FAIL: NaN/Inf in omega500")
            return False

        gm = _area_weighted_mean(np.asarray(w.values), w.lat.values)
        print(f"  area-weighted global mean omega500 = {gm:+.6f} Pa/s "
              f"(expect ~0)")
        if abs(gm) > 5e-3:
            print("  FAIL: global mean omega is not near zero")
            ok = False

        ascent = {
            "warm pool / Maritime Continent (5S-5N, 100-150E)": (-5, 5, 100, 150),
            "W Pacific ITCZ (2N-10N, 130-170E)": (2, 10, 130, 170),
            "Amazon (10S-2S, 290-305E)": (-10, -2, 290, 305),
        }
        descent = {
            "SE Pacific high (30S-15S, 240-280E)": (-30, -15, 240, 280),
            "N Atlantic high (20N-35N, 320-350E)": (20, 35, 320, 350),
            "Sahara / Arabia (18N-30N, 10E-50E)": (18, 30, 10, 50),
            "SE Atlantic high (30S-15S, 340-360E)": (-30, -15, 340, 360),
        }
        print("  ASCENT regions (omega must be NEGATIVE):")
        for label, box in ascent.items():
            v = _box_mean(w, *box)
            flag = "ok" if v < 0 else "FAIL"
            if v >= 0:
                ok = False
            print(f"    {flag:4s} {v:+.4f} Pa/s  {label}")
        print("  DESCENT regions (omega must be POSITIVE):")
        for label, box in descent.items():
            v = _box_mean(w, *box)
            flag = "ok" if v > 0 else "FAIL"
            if v <= 0:
                ok = False
            print(f"    {flag:4s} {v:+.4f} Pa/s  {label}")
    return ok


def verify_psl(path: Path) -> bool:
    """Global mean near 1013 hPa, plus the DJF centres of action."""
    import xarray as xr

    ok = True
    with xr.open_dataset(path, chunks={"time": 1}) as ds:
        da = ds["psl"]
        print(f"  units = {da.attrs.get('units')!r}")
        ann = (da.mean("time") / 100.0).load()
        if not np.isfinite(ann.values).all():
            print("  FAIL: NaN/Inf in psl")
            return False
        gm = _area_weighted_mean(np.asarray(ann.values), ann.lat.values)
        print(f"  area-weighted global mean psl = {gm:.2f} hPa "
              f"(expect ~1011-1013)")
        if not (1008.0 < gm < 1016.0):
            print("  FAIL: global mean psl out of range")
            ok = False

        djf = (da.sel(time=da["time"].dt.month.isin([12, 1, 2]))
                 .mean("time") / 100.0).load()
        checks = {
            # (label, box, comparison, threshold hPa)
            "Icelandic low (55N-70N, 320-350E)":
                ((55, 70, 320, 350), "lt", 1005.0),
            "Aleutian low (45N-60N, 170-210E)":
                ((45, 60, 170, 210), "lt", 1008.0),
            "Siberian high (40N-60N, 80E-120E)":
                ((40, 60, 80, 120), "gt", 1020.0),
        }
        print("  DJF centres of action:")
        for label, (box, cmp_, thr) in checks.items():
            v = _box_mean(djf, *box)
            good = v < thr if cmp_ == "lt" else v > thr
            if not good:
                ok = False
            sym = "<" if cmp_ == "lt" else ">"
            print(f"    {'ok' if good else 'FAIL':4s} {v:8.2f} hPa  "
                  f"(expect {sym} {thr:.0f})  {label}")
    return ok


VERIFIERS = {"wap": verify_wap, "psl": verify_psl}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--var", nargs="+", default=["wap", "psl"],
                    choices=sorted(VAR_SPECS),
                    help="variables to stage (default: both)")
    ap.add_argument("--start-year", type=int, default=1979)
    ap.add_argument("--end-year", type=int, default=2014)
    ap.add_argument("--ref-root", type=Path,
                    default=Path("/scratch/b/b309178/climateeval_data"),
                    help="scorer --ref-root; files land under "
                         "<ref-root>/reanalysis_ERA5/mon/<var>/")
    ap.add_argument("--tmp-root", type=Path,
                    default=Path("/scratch/b/b309178/era5_stage/work"))
    ap.add_argument("--verify", action="store_true",
                    help="run the sign-convention and sanity checks")
    ap.add_argument("--verify-only", type=Path, default=None, metavar="PATH",
                    help="skip extraction and verify an existing file")
    args = ap.parse_args(argv)

    if args.verify_only is not None:
        var = args.verify_only.name.split("_")[0]
        if var not in VERIFIERS:
            raise SystemExit(f"cannot infer variable from {args.verify_only}")
        print(f"=== verifying {var}: {args.verify_only} ===")
        return 0 if VERIFIERS[var](args.verify_only) else 1

    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be >= --start-year")
    args.tmp_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for name in args.var:
        spec = VAR_SPECS[name]
        log.info("staging %s (%d-%d)", name, args.start_year, args.end_year)
        path = stage_variable(spec, args.start_year, args.end_year,
                              args.ref_root, args.tmp_root)
        if args.verify:
            print(f"\n=== verifying {name}: {path} ===")
            if not VERIFIERS[name](path):
                failures.append(name)
            print()

    if failures:
        print(f"VERIFICATION FAILED for: {', '.join(failures)}")
        return 1
    print("all requested variables staged and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
