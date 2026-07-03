#!/usr/bin/env python
"""Build the harmonized ``legoesm_surfdata`` NetCDF from primary sources.

Requires the following datasets:
    https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/
*   **HWSD2 raster**: The Harmonized World Soil Database version 2.0 (HWSD2) raster data.
*   **HWSD2 attribute table**: The attribute database for the HWSD2 raster: HWSD2.mdb 
*   **CLM5 surfdata**: A CLM5 surfdata NetCDF file.

``builds data/legoesm_surfdata_cYYMMDD.nc`` from raw inputs:

  1. **HWSD2 raster + attribute table -> 0.25 deg soil intermediate.**  Streams
     the 30 arc-sec ``HWSD2.bil`` (~1.86 GB) through the SHARE-weighted SMU
     lookup and ``cos(lat)`` area-aggregates to a regular lat-lon target
     (default 0.25 deg, conservative).  Runs ``aggregate_hwsd2_soil`` ->
     ``write_surfdata`` via :func:`build_hwsd2_surfdata`.

  2. **0.25 deg soil + CLM5 cover/PFT/LAI -> CLM-grid harmonized v1 surfdata.**
     Regrids the 0.25 deg HWSD soil onto the CLM5 grid (conservative,
     NaN-aware), combines with the CLM5-reconstructed cover/PFT/LAI from
     :func:`read_clm5_cover_veg`, and writes one NetCDF via
     :func:`build_surfdata`.

The 0.25 deg intermediate is kept on disk by default. The output NetCDF is consumed at run
time by :func:`legoesm.land.global_surface_data.load_global_surface_data`
under the ``"legoesm_surfdata"`` preset.

Inputs (paths required)
-----------------------
* ``--hwsd-bil``   : HWSD2.bil raster (FAO HWSD v2.0).
* ``--hwsd-attr``  : HWSD2.mdb attribute database OR a previously-exported
                     ``HWSD2_LAYERS.csv`` (``.mdb`` requires ``mdbtools``).
* ``--clm-surfdata`` : a CLM5 surfdata NetCDF (e.g.
                     ``surfdata_1.9x2.5_*.nc``); supplies cover, PFT and
                     monthly LAI.

Outputs
-------
* ``--intermediate`` (default ``data/legoesm_surfdata_soil_0p25.nc``):
                     the 0.25 deg HWSD soil cache (skip step 1 if it
                     already exists with ``--skip-hwsd``).
* ``--out``          (default ``data/legoesm_surfdata_v1.nc``): the final
                     harmonized NetCDF the loader consumes.

Usage
-----
::

  python scripts/data/build_legoesm_surfdata.py \\
      --hwsd-bil ~/Downloads/HWSD2_RASTER/HWSD2.bil \\
      --hwsd-attr ~/Downloads/HWSD2_LAYERS.csv \\
      --clm-surfdata ~/Desktop/surfdata_1.9x2.5_SSP3-7.0_2015_16pfts_c250612.nc
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from legoesm.land.surface_data.assemble import build_v1_surfdata
from legoesm.land.surface_data.sources.hwsd2 import build_hwsd2_surfdata


_REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_INTERMEDIATE = _REPO_DATA_DIR / "legoesm_surfdata_soil_0p25.nc"
_DEFAULT_OUT = _REPO_DATA_DIR / "legoesm_surfdata_v1.nc"


def _existing(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"input file not found: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Inputs to step 1 only.  Not required (nor existence-checked) when
    # --skip-hwsd reuses an existing intermediate, since step 1 — the sole
    # consumer of the raster + attribute table — is skipped on that path.
    ap.add_argument("--hwsd-bil", default=None,
                    help="HWSD2.bil raster (30 arc-sec SMU ids); "
                         "required unless --skip-hwsd")
    ap.add_argument("--hwsd-attr", default=None,
                    help="HWSD2_LAYERS attribute table (.mdb or .csv); "
                         "required unless --skip-hwsd")
    ap.add_argument("--clm-surfdata", required=True, type=_existing,
                    help="CLM5 surfdata NetCDF (cover/PFT/LAI source for v1)")
    ap.add_argument("--intermediate", default=str(_DEFAULT_INTERMEDIATE),
                    help="path for the 0.25 deg HWSD soil intermediate "
                         "(reusable cache)")
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="path for the final harmonized legoesm_surfdata NetCDF")
    ap.add_argument("--res-deg", type=float, default=0.25,
                    help="intermediate resolution [deg] (step 1)")
    ap.add_argument("--rows-per-chunk", type=int, default=6,
                    help="coarse rows per chunk for the streaming aggregator "
                         "(memory vs throughput trade-off)")
    ap.add_argument("--skip-hwsd", action="store_true",
                    help="reuse an existing intermediate; skip step 1")
    args = ap.parse_args()

    # The step-1 inputs are required + must exist only when step 1 runs.  Under
    # --skip-hwsd they are unused, so a stale/absent path must not block reusing
    # the cached intermediate.
    if not args.skip_hwsd:
        for flag, val in (("--hwsd-bil", args.hwsd_bil), ("--hwsd-attr", args.hwsd_attr)):
            if val is None:
                ap.error(f"{flag} is required unless --skip-hwsd is set")
            _existing(val)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_hwsd:
        print(f"[1/2] HWSD2 raster + attrs -> {args.res_deg} deg soil "
              f"({Path(args.intermediate).name}) ...")
        t0 = time.time()
        soil = build_hwsd2_surfdata(
            args.hwsd_bil, args.hwsd_attr, args.intermediate,
            res_deg=args.res_deg, coarse_rows_per_chunk=args.rows_per_chunk,
        )
        nlat, nlon = soil["sand_pct"].shape[1:]
        print(f"     done in {time.time() - t0:.1f}s "
              f"(intermediate {nlat}x{nlon}, {soil['sand_pct'].shape[0]} soil layers)")
    else:
        if not os.path.exists(args.intermediate):
            sys.exit(f"--skip-hwsd set but intermediate {args.intermediate!r} not found")
        print(f"[1/2] reusing existing intermediate {args.intermediate}")

    print(f"[2/2] CLM5 cover/PFT/LAI + {Path(args.intermediate).name} -> "
          f"{Path(args.out).name} ...")
    t0 = time.time()
    build_v1_surfdata(args.clm_surfdata, args.intermediate, args.out)
    print(f"     done in {time.time() - t0:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
