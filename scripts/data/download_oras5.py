#!/usr/bin/env python3
"""Resumable ORAS5 monthly download from the Copernicus CDS (consolidated product).

One request per (variable, year) for single-level fields and per (variable,
year, month) for all-level fields; each returned zip is unpacked into
``<out>/<variable>/`` and deleted, and a request whose files are already on
disk is skipped, so the script can be re-run until everything is present.
Every unpacked file is opened with netCDF4 before it counts, and
``<out>/manifest.jsonl`` records each completed request (request, files, sha256).

Needs ``~/.cdsapirc`` and the ORAS5 licence accepted on the CDS website.

    scripts/data/download_oras5.py --out /data/abyssal/dbalwada/reanalysis/oras5 \
        --years 1993 2014 --single sea_surface_height sea_surface_temperature \
        --all-levels potential_temperature salinity --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "reanalysis-oras5"
CONSOLIDATED_LAST_YEAR = 2014  # 2015 onward is the separately forced operational product
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def build_requests(years, single, all_levels):
    """CDS request dicts, each tagged with the output sub-directory it fills."""
    if any(y > CONSOLIDATED_LAST_YEAR for y in years):
        raise ValueError(f"consolidated ORAS5 ends in {CONSOLIDATED_LAST_YEAR}")
    reqs = []
    for var in single:
        for y in years:
            reqs.append({"product_type": ["consolidated"], "vertical_resolution": "single_level",
                         "variable": [var], "year": [str(y)], "month": MONTHS})
    for var in all_levels:
        for y in years:
            for m in MONTHS:
                reqs.append({"product_type": ["consolidated"], "vertical_resolution": "all_levels",
                             "variable": [var], "year": [str(y)], "month": [m]})
    return reqs


def request_key(req) -> str:
    return json.dumps(req, sort_keys=True)


def done_keys(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    return {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}


def unpack(zip_path: Path, dest: Path) -> list[Path]:
    """Extract only plain ``*.nc`` members, by basename (no paths from the archive)."""
    import netCDF4

    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.endswith(".nc") or name != info.filename:
                raise ValueError(f"unexpected archive member {info.filename!r}")
            target = dest / name
            tmp = target.with_suffix(".nc.part")
            with z.open(info) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            with netCDF4.Dataset(tmp):  # must open before it counts
                pass
            tmp.replace(target)
            out.append(target)
    if not out:
        raise ValueError(f"no netCDF files in {zip_path}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(make_client, req, out: Path, manifest: Path, lock: threading.Lock) -> str:
    var = req["variable"][0]
    tag = f"{var}_{req['year'][0]}_{'-'.join(req['month']) if len(req['month']) == 1 else 'all'}"
    zip_path = out / "_incoming" / f"{tag}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    make_client().retrieve(DATASET, req, str(zip_path))  # one client per request: no shared session
    files = unpack(zip_path, out / var)
    zip_path.unlink()
    row = {"key": request_key(req), "request": req,
           "files": {f.name: sha256(f) for f in files}}
    with lock:
        with open(manifest, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return tag


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    p.add_argument("--single", nargs="*", default=[], help="single-level CDS variable names")
    p.add_argument("--all-levels", nargs="*", default=[], help="all-level CDS variable names")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    years = range(args.years[0], args.years[1] + 1)
    reqs = build_requests(years, args.single, args.all_levels)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    todo = [r for r in reqs if request_key(r) not in done_keys(manifest)]
    print(f"{len(reqs)} requests, {len(reqs) - len(todo)} already done, {len(todo)} to fetch",
          flush=True)
    if args.dry_run or not todo:
        return 0
    import cdsapi

    def make_client():
        return cdsapi.Client(quiet=True, progress=False)

    lock = threading.Lock()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, make_client, r, args.out, manifest, lock): r
                   for r in todo}
        for fut in as_completed(futures):
            try:
                print("DONE", fut.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - reported; a re-run retries it
                failed += 1
                print("FAILED", request_key(futures[fut]), repr(error)[:500], flush=True)
    print(f"=== {len(todo) - failed} fetched, {failed} failed (re-run to retry)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
