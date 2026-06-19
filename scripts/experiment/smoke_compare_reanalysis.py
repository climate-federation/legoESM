"""End-to-end SYNTHETIC smoke for the compare-reanalysis correction campaign.

OPERATOR PREFLIGHT.  Before committing a multi-day real-ERA5 HPC job, run this to
confirm the whole turnkey chain works in YOUR environment.  It chains the three
existing entry points on cheap synthetic data, in seconds:

  1. ``write_amip_clubb_lite_config`` -> a schema-valid AMIP ``clubb_lite`` base config
  2. ``make_synthetic_era5``          -> a synthetic ERA5 store in the exact
     ``load_era5_slice`` layout (WB2 levels, lat 90->-90)
  3. ``run_correction_campaign --dry-run`` -> the REAL campaign preamble: real config
     load, real ERA5 ingest (regrid + vertical interp), grid/sigma build, scheme +
     diagnosis-method validation, and the compute estimate -- WITHOUT the heavy run.

A green run means the config generator, ERA5 ingest, and campaign validation are
wired correctly in this environment; only then is the expensive real-ERA5 run worth
launching (swap the synthetic store for a real ERA5 zarr and drop ``--dry-run``).
This is NOT a science result -- the data is synthetic.

See ``docs/COMPARE_REANALYSIS.md`` for the full empirical-demonstration workflow.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Make the sibling ``scripts.*`` entry points importable when this file is run as a
# standalone CLI (``python scripts/experiment/smoke_compare_reanalysis.py``); under
# pytest the repo root is already on the path, so the guard makes this a no-op there.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_smoke(
    workdir: str,
    *,
    resolution: int = 8,
    nlev: int = 5,
    era5_nlat: int = 8,
    era5_nlon: int = 16,
) -> int:
    """Generate a config + synthetic ERA5, then dry-run the campaign in ``workdir``.

    Returns the campaign main's exit code (0 = the whole preamble validated).  The
    three sub-mains are invoked programmatically (not via a shell) so a failure at
    any stage propagates a non-zero code.  Raises ``RuntimeError`` if the config or
    ERA5 generation step fails (those must succeed before the campaign can be dry-run).
    """
    from scripts.data.make_synthetic_era5 import main as era5_main
    from scripts.experiment.write_amip_clubb_lite_config import main as config_main
    from scripts.run.run_correction_campaign import main as campaign_main

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "amip_clubb_lite.json"
    era5_path = work / "synthetic_era5.zarr"

    rc = config_main(
        [str(config_path), "--resolution", str(resolution), "--nlev", str(nlev)])
    if rc != 0:
        raise RuntimeError(f"config generation failed (exit {rc})")

    rc = era5_main(
        [str(era5_path), "--nlat", str(era5_nlat), "--nlon", str(era5_nlon),
         "--ntime", "1"])
    if rc != 0:
        raise RuntimeError(f"synthetic ERA5 generation failed (exit {rc})")

    # The REAL campaign main, --dry-run: validates config + ERA5 ingest + grid +
    # scheme/method WITHOUT the multi-day run (the iter-183 preflight).
    return campaign_main([
        "--config", str(config_path),
        "--era5-zarr", str(era5_path),
        "--mode", "amip",
        "--dry-run",
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workdir", default=None,
        help="directory for the generated config + ERA5 (default: a temp dir, removed on exit)")
    p.add_argument("--resolution", type=int, default=8,
                   help="lat-lon model resolution (n_lat; grid is n_lat x 2*n_lat)")
    p.add_argument("--nlev", type=int, default=5, help="vertical levels")
    args = p.parse_args(argv)

    def _go(wd: str) -> int:
        rc = run_smoke(wd, resolution=args.resolution, nlev=args.nlev)
        if rc == 0:
            print("[smoke] PASS: config -> synthetic ERA5 -> campaign dry-run all "
                  "validated. The turnkey chain is wired; swap in a real ERA5 zarr "
                  "and drop --dry-run for the empirical run.")
        else:
            print(f"[smoke] FAIL: campaign dry-run returned {rc}.")
        return rc

    if args.workdir is not None:
        return _go(args.workdir)
    with tempfile.TemporaryDirectory(prefix="compare_reanalysis_smoke_") as tmp:
        return _go(tmp)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
