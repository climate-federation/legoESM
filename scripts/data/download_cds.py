#!/usr/bin/env python3
"""Resumable monthly downloads from the Copernicus CDS: ORAS5 and ERA5 forcing.

``--dataset oras5`` (consolidated product): one request per (variable, year)
for single-level fields and per (variable, year, month) for all-level fields,
unpacked into ``<out>/<variable>/``.

``--dataset era5`` (single levels, interpolated to 1 deg by the server): per
(year, month), the state variables (winds, 2 m temperature and dewpoint,
sea-level pressure) at 00, 03, ..., 21 UTC, and the hourly-accumulated
radiation and precipitation at EVERY hour (each ERA5 value is a one-hour
total, so 3-hourly means need all of them); into ``<out>/instant/`` and
``<out>/accum/``, plus the land-sea mask once into ``<out>/static/``.

A request whose result is already recorded in ``<out>/manifest.jsonl`` is
skipped, so re-running resumes. Every file must open with netCDF4 before it
counts; zip results are unpacked by basename only. Needs ``~/.cdsapirc`` and
the dataset licence accepted on the CDS website.

    scripts/data/download_cds.py --dataset oras5 --out /data/abyssal/dbalwada/reanalysis/oras5 \
        --years 1993 2014 --single sea_surface_height --all-levels potential_temperature
    scripts/data/download_cds.py --dataset era5 --out /data/abyssal/dbalwada/reanalysis/era5_1deg \
        --years 1993 2014
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ORAS5 = "reanalysis-oras5"
ERA5 = "reanalysis-era5-single-levels"
CONSOLIDATED_LAST_YEAR = 2014  # ORAS5 2015 onward is the separately forced operational product
MONTHS = [f"{m:02d}" for m in range(1, 13)]
ERA5_INSTANT = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
                "2m_dewpoint_temperature", "mean_sea_level_pressure"]
ERA5_ACCUM = ["surface_solar_radiation_downwards", "surface_thermal_radiation_downwards",
              "total_precipitation", "snowfall"]
ERA5_GRID = [1.0, 1.0]


def build_requests(years, single, all_levels):
    """ORAS5 jobs ``(dataset, request, subdir, tag)``."""
    if any(y > CONSOLIDATED_LAST_YEAR for y in years):
        raise ValueError(f"consolidated ORAS5 ends in {CONSOLIDATED_LAST_YEAR}")
    jobs = []
    for var in single:
        for y in years:
            jobs.append((ORAS5, {"product_type": ["consolidated"],
                                 "vertical_resolution": "single_level", "variable": [var],
                                 "year": [str(y)], "month": MONTHS}, var, f"{var}_{y}_all"))
    for var in all_levels:
        for y in years:
            for m in MONTHS:
                jobs.append((ORAS5, {"product_type": ["consolidated"],
                                     "vertical_resolution": "all_levels", "variable": [var],
                                     "year": [str(y)], "month": [m]}, var, f"{var}_{y}_{m}"))
    return jobs


def build_era5_requests(years):
    """ERA5 jobs: per month, 3-hourly state variables and hourly accumulations; static mask."""
    common = {"product_type": ["reanalysis"], "grid": ERA5_GRID, "data_format": "netcdf",
              "download_format": "unarchived"}
    jobs = [(ERA5, dict(common, variable=["land_sea_mask"], year=["1993"], month=["01"],
                        day=["01"], time=["00:00"]), "static", "era5_land_sea_mask")]
    for y in years:
        for m in range(1, 13):
            days = [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
            when = {"year": [str(y)], "month": [f"{m:02d}"], "day": days}
            jobs.append((ERA5, dict(common, variable=ERA5_INSTANT, **when,
                                    time=[f"{h:02d}:00" for h in range(0, 24, 3)]),
                         "instant", f"era5_instant_{y}{m:02d}"))
            jobs.append((ERA5, dict(common, variable=ERA5_ACCUM, **when,
                                    time=[f"{h:02d}:00" for h in range(24)]),
                         "accum", f"era5_accum_{y}{m:02d}"))
    return jobs


def request_key(req) -> str:
    return json.dumps(req, sort_keys=True)


def done_keys(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    return {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}


def _verified(tmp: Path, target: Path) -> Path:
    import netCDF4

    with netCDF4.Dataset(tmp):  # must open before it counts
        pass
    tmp.replace(target)
    return target


def unpack(zip_path: Path, dest: Path) -> list[Path]:
    """Extract only plain ``*.nc`` members, by basename (no paths from the archive)."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.endswith(".nc") or name != info.filename:
                raise ValueError(f"unexpected archive member {info.filename!r}")
            tmp = dest / f"{name}.part"
            with z.open(info) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            out.append(_verified(tmp, dest / name))
    if not out:
        raise ValueError(f"no netCDF files in {zip_path}")
    return out


def place(download: Path, dest: Path, tag: str) -> list[Path]:
    """A zip is unpacked; a bare netCDF result is verified and named ``<tag>.nc``."""
    with open(download, "rb") as fh:
        is_zip = fh.read(2) == b"PK"
    if is_zip:
        files = unpack(download, dest)
        download.unlink()
        return files
    dest.mkdir(parents=True, exist_ok=True)
    return [_verified(download, dest / f"{tag}.nc")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(make_client, job, out: Path, manifest: Path, lock: threading.Lock) -> str:
    dataset, req, subdir, tag = job
    download = out / "_incoming" / f"{tag}.download"
    download.parent.mkdir(parents=True, exist_ok=True)
    make_client().retrieve(dataset, req, str(download))  # one client per request: no shared session
    files = place(download, out / subdir, tag)
    row = {"key": request_key(req), "request": req,
           "files": {f.name: sha256(f) for f in files}}
    with lock:
        with open(manifest, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return tag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=("oras5", "era5"), default="oras5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    p.add_argument("--single", nargs="*", default=[], help="ORAS5 single-level variable names")
    p.add_argument("--all-levels", nargs="*", default=[], help="ORAS5 all-level variable names")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    years = range(args.years[0], args.years[1] + 1)
    jobs = (build_era5_requests(years) if args.dataset == "era5"
            else build_requests(years, args.single, args.all_levels))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    done = done_keys(manifest)
    todo = [j for j in jobs if request_key(j[1]) not in done]
    print(f"{len(jobs)} requests, {len(jobs) - len(todo)} already done, {len(todo)} to fetch",
          flush=True)
    if args.dry_run or not todo:
        return 0
    import cdsapi

    def make_client():
        return cdsapi.Client(quiet=True, progress=False)

    lock = threading.Lock()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, make_client, j, args.out, manifest, lock): j
                   for j in todo}
        for fut in as_completed(futures):
            try:
                print("DONE", fut.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - reported; a re-run retries it
                failed += 1
                print("FAILED", futures[fut][3], repr(error)[:500], flush=True)
    print(f"=== {len(todo) - failed} fetched, {failed} failed (re-run to retry)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
