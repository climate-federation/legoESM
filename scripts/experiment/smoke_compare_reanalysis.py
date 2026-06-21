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
launching (drop ``--dry-run``).  Pass ``--era5-zarr <path>`` to validate the
operator's OWN real ERA5 store instead of synthetic — the dry-run still loads +
regrids + vertically-interpolates it, so it catches a store-specific ingest problem
(variable names, levels, lat ordering) before the multi-day job.  With synthetic
data this is NOT a science result.

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
    era5_zarr: str | None = None,
    mode: str = "amip",
    resolution: int = 8,
    nlev: int = 5,
    era5_nlat: int = 8,
    era5_nlon: int = 16,
    ocean_only: bool = False,
    surface_flux: bool = False,
    align_insolation: bool = False,
    grid_type: str = "latlon",
    land_mask_path: str = "",
) -> int:
    """Generate a config (+ synthetic ERA5 unless ``era5_zarr`` is given), then dry-run.

    ``era5_zarr=None`` (default) generates a synthetic ERA5 store — a pure
    environment smoke.  Passing a REAL ERA5 zarr instead validates the operator's
    OWN ingest (variable-name resolution, levels, lat ordering, the regrid) against
    the actual store, since the campaign loads + regrids + vertically-interpolates
    the reference even in ``--dry-run`` (lines 64-67 of the campaign main).  ``mode``
    selects AMIP vs CMIP.

    ``ocean_only`` / ``surface_flux`` / ``align_insolation`` forward the realistic-run
    flags (iters 449/451/465) to the dry-run, so an operator can validate the EXACT
    flag combo they will launch with — catching a flag-wiring problem in THIS
    environment on cheap synthetic data, before the multi-day job (on a flat synthetic
    config ``--ocean-only`` is a safe no-op the dry-run still validates).

    Returns the campaign main's exit code (0 = the whole preamble validated).  The
    sub-mains are invoked programmatically (not via a shell) so a failure at any
    stage propagates a non-zero code.  Raises ``RuntimeError`` if the config or
    synthetic-ERA5 generation step fails (those must succeed before the dry-run).
    """
    from scripts.experiment.write_amip_clubb_lite_config import main as config_main
    from scripts.run.run_correction_campaign import main as campaign_main

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "amip_clubb_lite.json"

    config_argv = [str(config_path), "--resolution", str(resolution), "--nlev", str(nlev),
                   "--grid-type", grid_type]
    if land_mask_path:                                  # a real land mask => --ocean-only
        config_argv += ["--land-mask-path", land_mask_path]   # actually EXCLUDES land columns
    rc = config_main(config_argv)
    if rc != 0:
        raise RuntimeError(f"config generation failed (exit {rc})")

    if era5_zarr is None:
        from scripts.data.make_synthetic_era5 import main as era5_main
        era5_path = str(work / "synthetic_era5.zarr")
        rc = era5_main(
            [era5_path, "--nlat", str(era5_nlat), "--nlon", str(era5_nlon),
             "--ntime", "1"])
        if rc != 0:
            raise RuntimeError(f"synthetic ERA5 generation failed (exit {rc})")
    else:
        era5_path = era5_zarr   # the operator's REAL store — validate its actual ingest

    # The REAL campaign main, --dry-run: validates config + ERA5 ingest (load +
    # regrid + vertical interp) + grid + scheme/method WITHOUT the multi-day run.
    dry_argv = [
        "--config", str(config_path),
        "--era5-zarr", era5_path,
        "--mode", mode,
        "--dry-run",
    ]
    if ocean_only:
        dry_argv.append("--ocean-only")
    if surface_flux:
        dry_argv.append("--surface-flux")
    if align_insolation:
        dry_argv.append("--align-insolation")
    return campaign_main(dry_argv)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workdir", default=None,
        help="directory for the generated config + ERA5 (default: a temp dir, removed on exit)")
    p.add_argument("--resolution", type=int, default=8,
                   help="lat-lon model resolution (n_lat; grid is n_lat x 2*n_lat)")
    p.add_argument("--nlev", type=int, default=5, help="vertical levels")
    p.add_argument(
        "--era5-zarr", default=None,
        help="validate the operator's REAL ERA5 store's ingest instead of generating "
             "synthetic (the dry-run still loads + regrids it); default: synthetic")
    p.add_argument("--mode", choices=("amip", "cmip"), default="amip",
                   help="AMIP (prescribed SST) or CMIP (coupled); default amip")
    p.add_argument("--grid-type", default="latlon",
                   choices=("latlon", "cubed_sphere", "gaussian"),
                   help="model grid family for the generated config (default latlon)")
    p.add_argument("--ocean-only", action="store_true",
                   help="forward --ocean-only to the dry-run (validate the realistic "
                        "ocean-column ranking flag in this environment)")
    p.add_argument("--surface-flux", action="store_true",
                   help="forward --surface-flux to the dry-run (validate the realistic "
                        "SST-driven LES surface-flux flag)")
    p.add_argument("--align-insolation", action="store_true",
                   help="forward --align-insolation to the dry-run (validate the "
                        "seasonal-insolation alignment flag)")
    p.add_argument("--land-mask-path", default="",
                   help="a land-sea mask NetCDF (e.g. from make_synthetic_land_mask.py) so "
                        "--ocean-only ACTUALLY excludes land columns (else the flat config is "
                        "all-ocean and --ocean-only is a no-op)")
    args = p.parse_args(argv)

    def _go(wd: str) -> int:
        rc = run_smoke(wd, era5_zarr=args.era5_zarr, mode=args.mode,
                       resolution=args.resolution, nlev=args.nlev,
                       ocean_only=args.ocean_only, surface_flux=args.surface_flux,
                       align_insolation=args.align_insolation, grid_type=args.grid_type,
                       land_mask_path=args.land_mask_path)
        era5_kind = "REAL ERA5" if args.era5_zarr else "synthetic ERA5"
        if rc == 0:
            print(f"[smoke] PASS: config -> {era5_kind} -> campaign --dry-run "
                  f"({args.mode}) all validated. The turnkey chain (config load + "
                  "ERA5 ingest + grid + scheme/method) is wired; drop --dry-run for "
                  "the empirical run.")
        else:
            print(f"[smoke] FAIL: campaign dry-run returned {rc}.")
        return rc

    if args.workdir is not None:
        return _go(args.workdir)
    with tempfile.TemporaryDirectory(prefix="compare_reanalysis_smoke_") as tmp:
        return _go(tmp)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
