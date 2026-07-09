#!/usr/bin/env python
"""Build a TRANSIENT ``legoesm_surfdata`` NetCDF from a HYDE / Pongratz / KK10 map.

Combines an existing static harmonized surfdata (soil + CLM5 base cover/PFT/LAI)
with an anthropogenic land-use reconstruction that carries only cropland /
pasture / built-up fractions.  The natural-vegetation backdrop reuses the base
17-PFT map (potential natural vegetation); the anthropogenic fractions are
conservatively regridded onto the base grid and overlaid to produce an annual
transient 17-PFT cover series.  Soil, monthly vegetation, lake and glacier are
copied from the base unchanged.

Unlike LUH2, these datasets have **no gross transitions**, so downstream E_LUC
bookkeeping uses net year-to-year cover change (understates shifting-cultivation
emissions — see ``legoesm.land.land_use_change``).

The output NetCDF is consumed at run time by the ``"legoesm_surfdata"`` preset.

Usage
-----
::

  python scripts/data/build_anthropogenic_surfdata.py \\
      --base-surfdata data/legoesm_surfdata_v1.nc \\
      --anthro        ~/data/hyde33/cover.nc \\
      --dataset       hyde \\
      --out           data/legoesm_surfdata_hyde_1850_2016.nc \\
      --year-start 1850 --year-end 2016
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from legoesm.land.surface_data.assemble import build_anthropogenic_transient_surfdata

# Reconstruction families (matches the runtime registry in
# legoesm.land.surface_data.datasets); "hyde" covers HYDE 3.2 and 3.3 (same vars).
_DATASETS = ("hyde", "pongratz", "kk10")


def _existing(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"input file not found: {path}")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-surfdata", required=True,
                    help="static harmonized surfdata NetCDF (grid + soil + base PFT/LAI)")
    ap.add_argument("--anthro", required=True,
                    help="anthropogenic-cover NetCDF (HYDE / Pongratz / KK10)")
    ap.add_argument("--dataset", required=True, choices=_DATASETS,
                    help="which reconstruction the --anthro file is")
    ap.add_argument("--out", required=True,
                    help="path for the transient legoesm_surfdata NetCDF")
    ap.add_argument("--year-start", type=int, default=None,
                    help="first calendar year to include (inclusive)")
    ap.add_argument("--year-end", type=int, default=None,
                    help="last calendar year to include (inclusive)")
    ap.add_argument("--crop-share", type=float, default=0.5,
                    help="KK10 only: fraction of the total anthropogenic area "
                         "assigned to cropland (rest -> pasture); default 0.5")
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    _existing(args.base_surfdata)
    _existing(args.anthro)
    if (args.year_start is None) != (args.year_end is None):
        raise SystemExit("--year-start and --year-end must be given together (or neither)")
    years = None if args.year_start is None else (args.year_start, args.year_end)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset} {args.anthro} + base {args.base_surfdata} -> "
          f"{Path(args.out).name} (years={years or 'all'}) ...")
    t0 = time.time()
    build_anthropogenic_transient_surfdata(
        args.base_surfdata, args.anthro, args.out,
        dataset=args.dataset, years=years, crop_share=args.crop_share)
    print(f"     done in {time.time() - t0:.1f}s\nwrote {args.out}")


if __name__ == "__main__":
    main()
