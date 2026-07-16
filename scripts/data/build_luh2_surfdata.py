#!/usr/bin/env python
"""Build a TRANSIENT ``legoesm_surfdata`` NetCDF from LUH2 land-use + a base file.

Combines an existing static harmonized surfdata (soil + CLM5 base cover/PFT/LAI,
e.g. the output of ``build_legoesm_surfdata.py``) with a LUH2 (or LUH3) states
file to produce an annual transient cover series on the base grid.  The 12 LUH2
land-use states are conservatively regridded onto the base grid and crosswalked
to the model's CLM5 17-PFT axis per year; soil, monthly vegetation, lake and
glacier are copied from the base unchanged.

The output NetCDF is consumed at run time by
:func:`legoesm.land.global_surface_data.load_global_surface_data` under the
``"legoesm_surfdata"`` preset (it already understands a multi-year ``pft_frac``).

Inputs
------
* ``--base-surfdata`` : static harmonized surfdata (grid, soil, base PFT, veg).
* ``--luh2-states``   : a LUH2/LUH3 states NetCDF (12 land-use state variables).

Usage
-----
::

  python scripts/data/build_luh2_surfdata.py \\
      --base-surfdata data/legoesm_surfdata_v1.nc \\
      --luh2-states   ~/data/luh2/states.nc \\
      --out           data/legoesm_surfdata_luh2_1850_2014.nc \\
      --year-start 1850 --year-end 2014
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from legoesm.land.surface_data.assemble import build_luh2_transient_surfdata
from legoesm.land.surface_data.sources.luh2 import LUH2Config


def _existing(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"input file not found: {path}")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-surfdata", required=True,
                    help="static harmonized surfdata NetCDF (grid + soil + base PFT/LAI)")
    ap.add_argument("--luh2-states", required=True,
                    help="LUH2/LUH3 states NetCDF (12 land-use state variables)")
    ap.add_argument("--out", required=True,
                    help="path for the transient legoesm_surfdata NetCDF")
    ap.add_argument("--year-start", type=int, default=None,
                    help="first calendar year to include (inclusive)")
    ap.add_argument("--year-end", type=int, default=None,
                    help="last calendar year to include (inclusive)")
    ap.add_argument("--year-base", type=int, default=LUH2Config().year_base,
                    help="calendar-year offset of the LUH2 time axis "
                         "(states files: 850)")
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    _existing(args.base_surfdata)
    _existing(args.luh2_states)
    if (args.year_start is None) != (args.year_end is None):
        raise SystemExit("--year-start and --year-end must be given together (or neither)")
    years = None if args.year_start is None else (args.year_start, args.year_end)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"LUH2 {args.luh2_states} + base {args.base_surfdata} -> {Path(args.out).name} "
          f"(years={years or 'all'}) ...")
    t0 = time.time()
    build_luh2_transient_surfdata(
        args.base_surfdata, args.luh2_states, args.out,
        years=years, luh2_config=LUH2Config(year_base=args.year_base))
    print(f"     done in {time.time() - t0:.1f}s\nwrote {args.out}")


if __name__ == "__main__":
    main()
