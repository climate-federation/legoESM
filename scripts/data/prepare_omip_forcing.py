#!/usr/bin/env python
"""Stage JRA55-do v1.4+ "corrected" forcing into a noleap Zarr cache.

Pre-processes a source JRA55-do store (local or downloaded) into the
single consolidated cache the tropical-OMIP driver consumes per step.

The heavy lifting lives in :mod:`legoesm.forcing.jra55_do`; this script
is a thin CLI wrapper that constructs a :class:`JRA55DoConfig` from
command-line arguments, builds the regular lat-lon target grid edges
from a single ``--target-resolution-deg`` value, and calls
:func:`build_jra55_cache`.

Usage
-----

    # Standard tropical-OMIP cache: 1958-2018, 1° regular lat-lon, noleap.
    python scripts/prepare_omip_forcing.py \\
        --source /scratch/jra55_do_v14/raw.zarr \\
        --years 1958 2018 \\
        --target-resolution-deg 1.0 \\
        --cache-dir /scratch/legoESM/jra55_do_omip2

    # Smoke test on a single year at coarser resolution:
    python scripts/prepare_omip_forcing.py \\
        --source /tmp/jra55_smoke.zarr \\
        --years 2000 2000 \\
        --target-resolution-deg 5.0 \\
        --cache-dir /tmp/jra55_cache_smoke

The cache is written once and re-used across runs.  Pass
``--overwrite`` to force a rebuild.

Source data
-----------

The source store must contain all of the variables in
:data:`JRA55_VARIABLES` on a regular lat-lon grid with cell **centres
at half-cell offsets** from the poles (and from the 0/360 meridian),
e.g. ±88.875° for ~1.25° spacing.  Centres coinciding with ±90° or
0° break the conservative regrid because cell edges then fall outside
the valid latitude range or wrap across the dateline.  This convention
matches the JRA55-do v1.4+ "corrected" distribution.

The ``time`` coordinate must be CF-encoded as
``"days since YYYY-MM-DD ..."`` on a Gregorian calendar (the default
JRA55-do convention).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# Repo root on path so this works without an editable install if needed.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Factored out so tests can drive the parser without subprocess
    invocation.
    """
    p = argparse.ArgumentParser(
        description=(
            "Stage JRA55-do v1.4+ corrected forcing into a noleap Zarr "
            "cache for tropical-OMIP runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", type=str, required=True,
        help="Path or URL to the source JRA55-do Zarr/NetCDF store.",
    )
    p.add_argument(
        "--years", type=int, nargs=2, metavar=("START", "END"),
        required=True,
        help=(
            "Inclusive year window (e.g. ``--years 1958 2018`` for "
            "OMIP-2). Both endpoints are inclusive."
        ),
    )
    p.add_argument(
        "--target-resolution-deg", type=float, default=1.0,
        help=(
            "Target regular lat-lon resolution in degrees (default 1.0). "
            "Must divide 180 evenly so latitude edges land on ±90°."
        ),
    )
    p.add_argument(
        "--cache-dir", type=str, required=True,
        help="Directory in which to write the Zarr cache.",
    )
    p.add_argument(
        "--cache-filename", type=str,
        default="jra55_do_v14_omip2_1deg_noleap.zarr",
        help="Output Zarr filename inside --cache-dir.",
    )
    p.add_argument(
        "--ref-year", type=int, default=1958,
        help=(
            "Reference year (day 0). Must match the driver's expectation. "
            "OMIP-2 default 1958."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Force rebuild even if the cache already exists.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-variable progress output.",
    )
    return p.parse_args(argv)


def _build_target_edges(
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_edges, lon_edges) in radians for a regular global
    lat-lon grid at the requested resolution.

    Latitude edges include ±90 exactly so a half-cell offset of cell
    centres lands at e.g. ±89.5 (for 1°). Longitude edges run from 0
    to 360 inclusive so a half-cell offset places the first centre at
    ``resolution/2`` (e.g. 0.5° for 1°).
    """
    if resolution_deg <= 0.0:
        raise ValueError(
            f"--target-resolution-deg must be > 0; got {resolution_deg}"
        )
    # Latitude must divide [−90, 90] evenly.
    if abs(180.0 / resolution_deg - round(180.0 / resolution_deg)) > 1e-9:
        raise ValueError(
            f"--target-resolution-deg ({resolution_deg}) must divide 180 "
            "evenly so latitude cell edges land exactly on ±90°."
        )
    if abs(360.0 / resolution_deg - round(360.0 / resolution_deg)) > 1e-9:
        raise ValueError(
            f"--target-resolution-deg ({resolution_deg}) must divide 360 "
            "evenly so longitude cell edges land exactly on 0/360°."
        )
    n_lat = int(round(180.0 / resolution_deg))
    n_lon = int(round(360.0 / resolution_deg))
    lat_edges = np.deg2rad(np.linspace(-90.0, 90.0, n_lat + 1))
    lon_edges = np.deg2rad(np.linspace(0.0, 360.0, n_lon + 1))
    return lat_edges, lon_edges


def _validate_args(args: argparse.Namespace) -> None:
    """Sanity checks beyond what argparse types enforce."""
    y0, y1 = args.years
    if y1 < y0:
        raise ValueError(
            f"--years {y0} {y1}: end year must be >= start year"
        )
    if not (1900 <= y0 <= 2200) or not (1900 <= y1 <= 2200):
        raise ValueError(
            f"--years {y0} {y1}: out of plausible range [1900, 2200]"
        )
    if not (1900 <= args.ref_year <= 2200):
        raise ValueError(
            f"--ref-year {args.ref_year}: out of plausible range [1900, 2200]"
        )
    src = Path(args.source)
    # Allow remote URLs (gs://, s3://, http) to pass through; only
    # validate that local paths exist, since xarray will give a better
    # error than this script for remote auth issues.
    if not (
        args.source.startswith(("gs://", "s3://", "http://", "https://"))
        or src.exists()
    ):
        raise FileNotFoundError(f"--source path does not exist: {args.source}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = parse_args(argv)
    _validate_args(args)

    # Lazy imports — keep --help fast and avoid xarray/zarr import
    # cost when the user just wants to read the help text.
    from legoesm.forcing.jra55_do import (
        JRA55DoConfig,
        JRA55_VARIABLES,
        RECORDS_PER_DAY,
        build_jra55_cache,
    )
    from legoesm.forcing.time_utils import NOLEAP_DAYS_PER_YEAR

    lat_edges, lon_edges = _build_target_edges(args.target_resolution_deg)
    n_lat = lat_edges.size - 1
    n_lon = lon_edges.size - 1
    n_years = args.years[1] - args.years[0] + 1
    n_records = n_years * NOLEAP_DAYS_PER_YEAR * RECORDS_PER_DAY

    cache_dir = Path(args.cache_dir)
    cache_path = cache_dir / args.cache_filename

    print("=" * 70)
    print("legoESM JRA55-do cache builder")
    print("=" * 70)
    print(f"  source         : {args.source}")
    print(f"  years          : {args.years[0]}–{args.years[1]} "
          f"({n_years} yr, noleap)")
    print(f"  target grid    : {n_lat}×{n_lon} regular lat-lon "
          f"@ {args.target_resolution_deg}°")
    print(f"  cadence        : 3-hourly  ({RECORDS_PER_DAY} records/day)")
    print(f"  total records  : {n_records:,}")
    print(f"  variables      : {', '.join(JRA55_VARIABLES)}")
    print(f"  ref_year       : {args.ref_year}")
    print(f"  cache path     : {cache_path}")
    print(f"  overwrite      : {args.overwrite}")
    print("=" * 70)

    cfg = JRA55DoConfig(
        source_path=args.source,
        years=tuple(args.years),
        target_lat_edges=lat_edges,
        target_lon_edges=lon_edges,
        cache_dir=cache_dir,
        ref_year=args.ref_year,
        cache_filename=args.cache_filename,
    )

    t0 = time.time()
    out_path = build_jra55_cache(
        cfg, overwrite=args.overwrite, progress=not args.quiet,
    )
    elapsed = time.time() - t0

    # Final summary — useful for capturing in run logs.
    if out_path.exists():
        size_bytes = sum(
            f.stat().st_size for f in Path(out_path).rglob("*") if f.is_file()
        )
        size_gb = size_bytes / (1024.0 ** 3)
    else:
        size_gb = float("nan")

    print("-" * 70)
    print(f"  cache size     : {size_gb:.2f} GB")
    print(f"  elapsed        : {elapsed:.1f} s")
    print(f"  status         : OK")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
