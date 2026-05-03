#!/usr/bin/env python
"""Download the JRA55-do v1.6.0 raw IAF window for one or more years.

The output is a directory of per-variable per-year NetCDF files
matching the input4MIPs layout, ready for ``make_ryf.py`` to apply
the Stewart 2020 smooth wraparound blending.

Source: MRI-JRA55-do-1-6-0 (final version) on input4MIPs/ESGF, served
publicly through the LLNL Globus HTTPS endpoint
(``g-eba899.6b7bd8.0ec8.data.globus.org``) — no authentication needed.

Resumable: ``curl --continue-at -`` picks up partial files. Each
download is verified against the size reported by the ESGF Solr
catalog at fetch time; mismatches are flagged but the file is kept
so the user can inspect it.

Layout written to disk (relative to ``--out-dir``):

    <out-dir>/
        atmos/3hrPt/uas/v20240531/uas_..._199001010000-199012312100.nc
        atmos/3hrPt/uas/v20240531/uas_..._199101010000-199112312100.nc
        ... (vas, tas, huss, psl)
        atmos/3hr/rsds/v20240531/rsds_..._199001010130-199012312230.nc
        ... (rlds, prra, prsn)
        land/day/friver/v20240531/friver_..._19900101-19901231.nc
        ... (friver 1991)

Usage::

    python scripts/download_jra55_iaf.py \\
        --years 1990 1991 \\
        --out-dir data/jra55_iaf

By default the 10 variables our tropical-OMIP pipeline uses are
fetched; restrict via ``--vars`` if you want to test with a small
subset first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Variable manifest. Three CMIP6 tables, each with its own timestamp scheme:
#   3hrPt  — instantaneous, 0000–2100 UTC
#   3hr    — time-mean, 0130–2230 UTC (centred on 3-hour windows)
#   day    — daily mean
# Path / table mapping discovered via the ESGF Solr catalog.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VarSpec:
    name: str
    realm: str        # "atmos" | "land"
    table: str        # "3hrPt" | "3hr" | "day"
    ts_scheme: str    # "instant" | "mean3h" | "daily"


VARS: list[VarSpec] = [
    VarSpec("uas",    "atmos", "3hrPt", "instant"),
    VarSpec("vas",    "atmos", "3hrPt", "instant"),
    VarSpec("tas",    "atmos", "3hrPt", "instant"),
    VarSpec("huss",   "atmos", "3hrPt", "instant"),
    VarSpec("psl",    "atmos", "3hrPt", "instant"),
    VarSpec("rsds",   "atmos", "3hr",   "mean3h"),
    VarSpec("rlds",   "atmos", "3hr",   "mean3h"),
    VarSpec("prra",   "atmos", "3hr",   "mean3h"),
    VarSpec("prsn",   "atmos", "3hr",   "mean3h"),
    VarSpec("friver", "land",  "day",   "daily"),
]

VERSION_TAG = "v20240531"   # MRI-JRA55-do-1-6-0 publication tag
GRID_TAG = "gr"


def _expected_filename(var: VarSpec, year: int) -> str:
    if var.ts_scheme == "instant":
        ts = f"{year}01010000-{year}12312100"
    elif var.ts_scheme == "mean3h":
        ts = f"{year}01010130-{year}12312230"
    else:  # daily
        ts = f"{year}0101-{year}1231"
    return (
        f"{var.name}_input4MIPs_atmosphericState_OMIP_"
        f"MRI-JRA55-do-1-6-0_{GRID_TAG}_{ts}.nc"
    )


def _local_path(out_dir: Path, var: VarSpec, year: int) -> Path:
    return (
        out_dir / var.realm / var.table / var.name / VERSION_TAG
        / _expected_filename(var, year)
    )


# ---------------------------------------------------------------------------
# ESGF Solr lookup — gets the public Globus HTTPS URL + reported size.
# ---------------------------------------------------------------------------

ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search/"


def _solr_query(var: VarSpec) -> list[dict]:
    """Return all File-type docs for a single variable in MRI-JRA55-do-1-6-0."""
    qs = urllib.parse.urlencode({
        "project": "input4MIPs",
        "source_id": "MRI-JRA55-do-1-6-0",
        "variable_id": var.name,
        "type": "File",
        "format": "application/solr+json",
        "limit": 5000,
    })
    url = f"{ESGF_SEARCH}?{qs}"
    last_err = None
    for attempt in range(4):
        out = subprocess.run(
            ["curl", "-sL", "--max-time", "60", url],
            capture_output=True, text=True,
        )
        if out.returncode == 0 and out.stdout:
            try:
                return json.loads(out.stdout)["response"]["docs"]
            except (json.JSONDecodeError, KeyError) as e:
                last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"ESGF Solr query for {var.name} failed: {last_err}")


def _pick_globus_https(doc: dict) -> str | None:
    """Extract the public Globus HTTPS replica URL from a Solr doc."""
    for u in doc.get("url", []):
        plain = u.split("|")[0]
        if plain.startswith("https://g-") and "data.globus.org" in plain \
                and "HTTPServer" in u:
            return plain
    return None


def _resolve_url(var: VarSpec, year: int) -> tuple[str, int] | None:
    """Find the Globus HTTPS URL + size for one variable-year file."""
    fname = _expected_filename(var, year)
    docs = _solr_query(var)
    for doc in docs:
        if doc.get("title") == fname:
            url = _pick_globus_https(doc)
            if url:
                return url, int(doc.get("size", 0))
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def _download_one(url: str, dest: Path, expected_size: int,
                  force: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        actual = dest.stat().st_size
        if actual == expected_size:
            print(f"  [skip] {dest.name} ({_human(actual)})")
            return True
        print(f"  [resume] {dest.name} "
              f"({_human(actual)} of {_human(expected_size)})")
    else:
        print(f"  [get] {dest.name} ({_human(expected_size)})")

    cmd = [
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "10",
        "--continue-at", "-", "--show-error", "--silent",
        "--output", str(dest), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  [error] curl exit {proc.returncode} for {dest.name}",
              file=sys.stderr)
        return False
    actual = dest.stat().st_size
    if actual != expected_size:
        print(f"  [warn] {dest.name}: got {_human(actual)}, "
              f"expected {_human(expected_size)}", file=sys.stderr)
        return False
    print(f"  [done] {dest.name}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--years", type=int, nargs="+", required=True,
                   help="Calendar years to fetch (e.g. ``--years 1990 1991`` "
                        "for the RYF9091 window).")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory (will be created).")
    p.add_argument("--vars", type=str, nargs="*", default=None,
                   help="Restrict to a subset of variables. Default: all 10.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if local file matches expected size.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print URLs and sizes without downloading.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    selected_vars = (
        [v for v in VARS if v.name in set(args.vars)] if args.vars else VARS
    )
    if not selected_vars:
        print(f"No matching variables for --vars {args.vars}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("JRA55-do v1.6.0 IAF download")
    print("=" * 70)
    print(f"  years    : {args.years}")
    print(f"  vars     : {[v.name for v in selected_vars]}")
    print(f"  out_dir  : {out_dir}")
    print()

    # Resolve all URLs and report total size up front so the user can
    # decide whether to commit before bytes start moving.
    plan: list[tuple[VarSpec, int, str, int, Path]] = []
    print("Resolving URLs from ESGF Solr ...")
    for var in selected_vars:
        for year in args.years:
            res = _resolve_url(var, year)
            if res is None:
                print(f"  [missing] {var.name} {year}", file=sys.stderr)
                continue
            url, size = res
            dest = _local_path(out_dir, var, year)
            plan.append((var, year, url, size, dest))
            print(f"  {var.name:7s} {year}  {_human(size):>10s}  -> {dest.name}")

    total = sum(s for _, _, _, s, _ in plan)
    print(f"\nPlan: {len(plan)} files, {_human(total)} total")
    if args.dry_run:
        return 0

    print("\nDownloading ...")
    failures = 0
    for var, year, url, size, dest in plan:
        ok = _download_one(url, dest, size, force=args.force)
        if not ok:
            failures += 1
    print(f"\nDone. {len(plan) - failures}/{len(plan)} files complete.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
