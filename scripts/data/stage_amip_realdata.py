#!/usr/bin/env python
"""Stage the real-data inputs for a faithful AMIP CMIP run.

Assembles the four pieces the production deck needs for realism:

  1. Correct GHG/ozone/solar/aerosol  — synthetic deck (already units-
     correct) OR real input4MIPs files you drop in --forcing-dir.
  2. ERA5 initial condition           — the PUBLIC ARCO ERA5 store on
     GCS (no credentials); this script only verifies reachability.
  3. Real prescribed SST/SIC          — PCMDI / input4MIPs AMIP II bcs
     (``tosbcs`` K, ``siconcbcs`` percent).  input4MIPs is OPEN data
     (no ESGF account) — fetch it with
     ``scripts/data/download_cmip6_forcing.py``; this script validates a
     staged file's variables + units against what the deck expects.
  4. Months of spin-up                — handled by the checkpointed
     ``run_amip30y_allgrids_local.sh`` launcher (not this script).

Usage::

    # Verify ERA5 reachability + check/stage a real SST file:
    python scripts/data/stage_amip_realdata.py \
        --sst-file /data/amip2/tosbcs_input4MIPs_*_gn_187001-202112.nc

    # Just print the ESGF instructions:
    python scripts/data/stage_amip_realdata.py --print-esgf

The deck command that consumes the staged data is printed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# CLAUDE.md "Constant and Parameter Discipline": no hardcoded C<->K offset.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from legoesm import constants  # noqa: E402

# Public ARCO ERA5 (mirrors run_amip_smoke_deck._DEFAULT_ARCO_ERA5).
_ARCO_ERA5 = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

_ESGF_INSTRUCTIONS = """\
Real AMIP II prescribed SST/SIC (PCMDI / input4MIPs):

  input4MIPs is OPEN data (served over the public LLNL Globus HTTPS endpoint,
  no ESGF account needed).  Fetch it automatically:

      python scripts/data/download_cmip6_forcing.py --channels sst_sic \\
          --out-dir data/cmip6_forcing

  That resolves source_id 'PCMDI-AMIP-1-1-9', variables 'tosbcs' (SST) and
  'siconcbcs' (sea-ice), and downloads the NetCDF (e.g.
  tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202112.nc)
  into data/cmip6_forcing/sst_sic/.  Then point the deck at it:

      --sst-file <path>   (the deck defaults --sst-var tosbcs,
                           --sic-var siconcbcs, --sst-offset 0,
                           --sic-scale 0.01 for this dataset)

  '--dry-run' shows the plan first; '--list' shows every forcing channel.

The loader's units-attribute guard cross-checks Kelvin/Celsius +
percent/fraction, so a wrong file/flag combination fails loudly.
"""


def _check_era5_reachable() -> bool:
    if not shutil.which("gsutil"):
        print("[stage] gsutil not found — install the Google Cloud SDK to "
              "read the public ARCO ERA5 store (or pass a local --ic-path "
              "to the deck).")
        return False
    print(f"[stage] checking public ARCO ERA5 reachability: {_ARCO_ERA5}")
    r = subprocess.run(["gsutil", "ls", _ARCO_ERA5 + "/"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[stage] ERA5 OK (public, no credentials needed).")
        return True
    print(f"[stage] ERA5 not reachable: {r.stderr.strip()[:200]}")
    return False


def _validate_sst_file(path: Path, sst_var: str, sic_var: str):
    """Return (ok, sst_offset, sic_scale) inferred from the file.

    ``sst_offset``/``sic_scale`` are the deck flags the heuristic
    recommends so the caller can print a SELF-CONSISTENT command (codex
    review: do not hardcode 0/0.01 regardless of what was detected).
    Returns (False, None, None) on any problem.
    """
    import xarray as xr
    import numpy as np
    if not path.exists():
        print(f"[stage] SST file not found: {path}")
        print(_ESGF_INSTRUCTIONS)
        return False, None, None
    try:
        with xr.open_dataset(path) as ds:
            ok = True
            for v in (sst_var, sic_var):
                if v not in ds.data_vars:
                    print(f"[stage] ERROR: variable {v!r} not in {path.name}; "
                          f"present: {list(ds.data_vars)}")
                    ok = False
                    continue
                units = ds[v].attrs.get("units", "<none>")
                vmin = float(np.nanmin(ds[v].values))
                vmax = float(np.nanmax(ds[v].values))
                print(f"[stage]   {v}: units={units!r} "
                      f"range=[{vmin:.3g}, {vmax:.3g}]")
            if not ok:
                return False, None, None
            tos_max = float(np.nanmax(np.asarray(ds[sst_var].values)))
            sic_max = float(np.nanmax(np.asarray(ds[sic_var].values)))
    except Exception as exc:  # noqa: BLE001 - surface any open/read error
        print(f"[stage] ERROR reading {path}: {exc}")
        return False, None, None

    # All-NaN guard: nanmax of an all-NaN array is NaN, which would
    # mis-route the Kelvin/Celsius + percent/fraction inference.
    if not (np.isfinite(tos_max) and np.isfinite(sic_max)):
        print("[stage] ERROR: SST/SIC field is all-NaN; cannot infer "
              "units. Check the variable names / file integrity.")
        return False, None, None

    if tos_max > 200.0:
        sst_offset = 0.0
        print("[stage]   -> SST looks like Kelvin: use --sst-offset 0")
    else:
        sst_offset = constants.T_freeze
        print(f"[stage]   -> SST looks like Celsius: use "
              f"--sst-offset {constants.T_freeze}")
    if sic_max > 1.5:
        sic_scale = 0.01
        print("[stage]   -> SIC looks like percent: use --sic-scale 0.01")
    else:
        sic_scale = 1.0
        print("[stage]   -> SIC looks like fraction: use --sic-scale 1.0")
    return True, sst_offset, sic_scale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sst-file", type=Path, default=None,
                    help="Real input4MIPs AMIP II bcs SST file to validate.")
    ap.add_argument("--sst-var", default="tosbcs")
    ap.add_argument("--sic-var", default="siconcbcs")
    ap.add_argument("--ic-path", default=_ARCO_ERA5,
                    help="ERA5 IC source (default: public ARCO ERA5).")
    ap.add_argument("--print-esgf", action="store_true",
                    help="Print the ESGF AMIP II SST download steps and exit.")
    args = ap.parse_args(argv)

    if args.print_esgf:
        print(_ESGF_INSTRUCTIONS)
        return 0

    era5_ok = _check_era5_reachable()
    sst_ok = True
    sst_offset = sic_scale = None
    if args.sst_file is not None:
        sst_ok, sst_offset, sic_scale = _validate_sst_file(
            args.sst_file, args.sst_var, args.sic_var)
    else:
        print("[stage] no --sst-file given; for a faithful AMIP CMIP run "
              "fetch the real input4MIPs SST with\n"
              "        python scripts/data/download_cmip6_forcing.py "
              "--channels sst_sic --out-dir data/cmip6_forcing\n"
              "        (see --print-esgf for the full walkthrough).")

    print("\n[stage] Deck command for the full real-data combination:")
    if args.sst_file and sst_ok:
        # Emit the SELF-CONSISTENT flags the heuristic inferred (codex
        # review: never hardcode 0/0.01 contradicting the detection).
        sst_args = (
            f" \\\n    --sst-file {args.sst_file}"
            f" \\\n    --sst-var {args.sst_var} --sic-var {args.sic_var}"
            f" \\\n    --sst-offset {sst_offset} --sic-scale {sic_scale}"
        )
    else:
        sst_args = "  # (add --sst-file <input4MIPs tosbcs file>)"
    print(
        "  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\\n"
        "  .venv/bin/python scripts/run/run_amip_smoke_deck.py \\\n"
        "    --forcing-dir data/forcing_amip --auto-generate \\\n"
        "    --start-year 1979 --end-year 2009 \\\n"
        "    --grid-type cubed_sphere --discretization finite_volume \\\n"
        "    --resolution 36 --days 10950 --dt-auto \\\n"
        "    --rad-update-steps 18 --diag-days 30 --checkpoint-days 365 \\\n"
        "    --radiation rrtmg \\\n"
        f"    --ic era5 --ic-path {args.ic_path}{sst_args} \\\n"
        "    --output results/amip30y_cube"
    )
    return 0 if (era5_ok and sst_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
