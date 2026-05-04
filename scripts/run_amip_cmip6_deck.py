#!/usr/bin/env python
"""End-to-end CMIP6 AMIP deck driver for legoESM.

This wraps :mod:`scripts.run_amip` and orchestrates the full forcing
suite (SST/SIC + transient GHG + ozone + solar + aerosol + volcanic).
Forcing files can either be pre-generated (e.g. by
:mod:`scripts.generate_amip_forcing`) or auto-generated on the fly.

Why this script
---------------
``run_amip.py`` is a low-level CLI that takes ~30 flags. The CMIP6 AMIP
protocol fixes most of those flags to specific values. This wrapper:

1. Generates synthetic CMIP6-shape forcing files when missing.
2. Sets the canonical RRTMG + Sundqvist clouds + Kessler + SBM stack.
3. Forces ``--ghg-forcing external`` and ``--ozone-forcing external``.
4. Sets ``--solar-source spectral_file`` with the canonical
   ``--solar-tsi-var TSI --solar-spectral-var SSI_frac`` flags so the
   per-band 14-fraction solar file from MPI-M is consumed correctly.
5. Wires aerosol + volcanic flags.
6. Defaults to a 12-month integration so wall-clock is bounded.

Usage::

    JAX_ENABLE_X64=1 python scripts/run_amip_cmip6_deck.py \\
        --resolution 16 --days 30 \\
        --output results/amip_deck_test

To run without forcing files (the script will generate them)::

    JAX_ENABLE_X64=1 python scripts/run_amip_cmip6_deck.py \\
        --auto-generate --resolution 16 --days 30

To target a specific grid type::

    --grid-type {cubed_sphere,gaussian,latlon,voronoi}

The script always passes ``--monthly-means`` so post-run validation
tools (``scripts/validate_amip.py``) can read zonal-mean diagnostics.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FORCING = _REPO_ROOT / "forcing_amip"

# CLAUDE.md "Constant and Parameter Discipline": never hardcode 273.15 etc.
sys.path.insert(0, str(_REPO_ROOT / "src"))
from legoesm import constants  # noqa: E402


def _check_forcing_files(forcing_dir: Path, start_year: int,
                         end_year: int) -> dict[str, Path]:
    """Verify the canonical 6-file deck exists in ``forcing_dir``.

    Ozone resolution (``ozone_amip_<sy>-<ey>.nc`` interannual *vs.*
    ``ozone_amip_clim.nc`` 12-month climatology): if both are present
    we prefer the **interannual** file because that's what real
    input4MIPs CMIP6 ozone is — the loader's non-cyclic dispatch
    (`_interp_monthly_noncyclic`, keyed on ``ntime > 12``) is the right
    code path to exercise for production AMIP.  The climatology is the
    fallback when the user has only generated the cyclic file.
    """
    interannual_o3 = forcing_dir / f"ozone_amip_{start_year}-{end_year}.nc"
    clim_o3 = forcing_dir / "ozone_amip_clim.nc"
    if interannual_o3.exists():
        ozone_path = interannual_o3
    elif clim_o3.exists():
        ozone_path = clim_o3
    else:
        # Default to climatology name so the missing-file message
        # mentions the file users see most often.
        ozone_path = clim_o3
    files = {
        "sst": forcing_dir / f"sst_sic_amip_{start_year}-{end_year}.nc",
        "ghg": forcing_dir / f"ghg_amip_{start_year}-{end_year}.nc",
        "ozone": ozone_path,
        "solar": forcing_dir / f"solar_amip_{start_year}-{end_year}.nc",
        "aerosol": forcing_dir / "aerosol_amip_clim.nc",
        "volcanic": forcing_dir / f"volcanic_amip_{start_year}-{end_year}.nc",
    }
    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        return {"_missing": ",".join(missing), **files}
    return files


def _auto_generate(forcing_dir: Path, start_year: int, end_year: int,
                   *, nlat_sst: int = 73, nlon_sst: int = 144) -> None:
    print(f"[deck] Auto-generating forcing files in {forcing_dir} …")
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "generate_amip_forcing.py"),
        "--out", str(forcing_dir),
        "--start-year", str(start_year),
        "--end-year", str(end_year),
        "--nlat-sst", str(nlat_sst),
        "--nlon-sst", str(nlon_sst),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Forcing dataset
    parser.add_argument("--forcing-dir", type=Path, default=_DEFAULT_FORCING,
                        help="Directory holding the AMIP forcing deck "
                             f"(default: {_DEFAULT_FORCING})")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--auto-generate", action="store_true", default=False,
                        help="Generate forcing files if any are missing")

    # Run-time configuration
    parser.add_argument("--resolution", type=int, default=16,
                        help="Cubed-sphere N or spectral truncation (default 16)")
    parser.add_argument("--nlev", type=int, default=30,
                        help="Number of vertical levels (default 30)")
    parser.add_argument("--days", type=int, default=30,
                        help="Simulation length in days (default 30)")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Time step (default 600 s)")
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--checkpoint-days", type=int, default=0)

    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "gaussian", "latlon", "voronoi"])
    parser.add_argument("--discretization", type=str, default="centered",
                        choices=["centered", "finite_volume", "cgrid",
                                  "latlon_cgrid", "cdgrid", "mpas", "spectral"])

    # Physics — defaults match a balanced CMIP6 AMIP stack
    parser.add_argument("--radiation", type=str, default="rrtmg")
    parser.add_argument("--rad-update-steps", type=int, default=6)
    parser.add_argument("--clouds", type=str, default="sundqvist")
    # Sundqvist microphysics (large-scale condensation) is the canonical
    # CMIP-physics default — Kessler in the integrated AMIP path produces
    # NaN winds at day ~2 with current SBM/clouds settings (tracked in
    # AMIP.md "Known issues"; surfaces in `tests/unit/test_amip_cmip6_deck.py`
    # follow-ups).
    parser.add_argument("--microphysics", type=str, default="sundqvist")
    parser.add_argument("--convection", type=str, default="sbm")
    parser.add_argument("--turbulence", type=str, default="louis")
    parser.add_argument("--gravity-wave-drag", type=str, default="none")
    parser.add_argument("--diurnal-cycle", action="store_true", default=True)
    parser.add_argument("--no-diurnal-cycle", dest="diurnal_cycle",
                        action="store_false")

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--monthly-means", action="store_true", default=True)

    # Auxiliary
    parser.add_argument("--volcanic-aerosol-scale", type=float, default=1.0,
                        help="Multiplier on volcanic AOD (0.0 disables)")
    parser.add_argument("--no-aerosol", action="store_true", default=False,
                        help="Skip aerosol forcing entirely")
    parser.add_argument("--no-volcanic", action="store_true", default=False,
                        help="Skip volcanic aerosol forcing entirely")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Build run_amip.py command line and print it (don't run)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded verbatim to run_amip.py")

    args = parser.parse_args(argv)

    forcing_dir = Path(args.forcing_dir).resolve()
    files = _check_forcing_files(forcing_dir, args.start_year, args.end_year)

    if "_missing" in files:
        if args.auto_generate:
            _auto_generate(forcing_dir, args.start_year, args.end_year)
            files = _check_forcing_files(forcing_dir, args.start_year, args.end_year)
            if "_missing" in files:
                raise RuntimeError(
                    f"Auto-generation completed but files still missing: "
                    f"{files['_missing']}"
                )
        else:
            print(f"[deck] ERROR — forcing dir {forcing_dir} is missing files: "
                  f"{files['_missing']}")
            print(f"[deck]   re-run with --auto-generate to create them, or "
                  "run scripts/generate_amip_forcing.py manually first.")
            return 2

    # Build the run_amip.py command
    run_amip = _REPO_ROOT / "scripts" / "run_amip.py"
    cmd = [
        sys.executable, str(run_amip),
        "--dataset", "custom",
        "--forcing-path", str(files["sst"]),
        "--sst-var", "sst",
        "--sic-var", "sic",
        "--sst-offset", str(constants.T_freeze),  # synthetic file is in Celsius
        "--sic-scale", "1.0",
        "--time-var", "time",
        "--lat-var", "lat",
        "--lon-var", "lon",
        # Grid
        "--resolution", str(args.resolution),
        "--nlev", str(args.nlev),
        "--grid-type", args.grid_type,
        "--discretization", args.discretization,
        # Integration
        "--days", str(args.days),
        "--dt", str(args.dt),
        "--diag-days", str(args.diag_days),
        "--checkpoint-days", str(args.checkpoint_days),
        # Radiation
        "--radiation", args.radiation,
        "--rad-update-steps", str(args.rad_update_steps),
        # Ozone
        "--ozone-source", "standard",
        "--ozone-forcing", "external",
        "--ozone-file", str(files["ozone"]),
        # GHG
        "--ghg-forcing", "external",
        "--ghg-file", str(files["ghg"]),
        # Solar
        "--solar-source", "spectral_file",
        "--solar-file", str(files["solar"]),
        "--solar-tsi-var", "TSI",
        "--solar-spectral-var", "SSI_frac",
        # Convection / clouds / microphysics
        "--convection", args.convection,
        "--turbulence", args.turbulence,
        "--gravity-wave-drag", args.gravity_wave_drag,
        "--clouds", args.clouds,
        "--microphysics", args.microphysics,
        # CMIP / time
        "--start-year", str(args.start_year),
        "--experiment", "amip",
        # Diagnostics
        "--clear-sky-diag",
    ]
    if args.diurnal_cycle:
        cmd.append("--diurnal-cycle")
    if args.monthly_means:
        cmd.append("--monthly-means")
    if not args.no_aerosol:
        cmd += [
            "--aerosol-forcing", "external",
            "--aerosol-file", str(files["aerosol"]),
            "--aerosol-reference-aod", "0.05",
        ]
    if not args.no_volcanic and args.volcanic_aerosol_scale > 0:
        cmd += [
            "--volcanic-aerosol-file", str(files["volcanic"]),
            "--volcanic-aerosol-scale", str(args.volcanic_aerosol_scale),
        ]
    if args.output:
        cmd += ["--output", args.output]
    # NOTE: --fix-moisture is *intentionally not passed*. The current
    # implementation (`fix_moisture_hydrostatic` in `core/conservation.py`)
    # rescales only `q_v`, not the prognostic condensate tracers
    # (`q_c`/`q_r`).  When microphysics precipitates water out of the
    # column, q_v decreases and the fixer multiplies it back up — a
    # spurious source of vapor that drives a runaway with Kessler-type
    # schemes (catalogued in AMIP.md "Known issues").  Sundqvist
    # microphysics is also prognostic-condensate, so the same caveat
    # applies; turning fix-moisture on for AMIP requires a `fix_total_water`
    # path that tracks cumulative precipitation, which is a follow-up.

    if args.extra:
        cmd += list(args.extra)

    # Tell the user which CMIP6 forcing channels will *actually* affect
    # the run.  The driver gates GHG/ozone/aerosol on
    # ``cfg.radiation in ("rrtmg", "rrtmgp")``: those channels are
    # configured but inert under ``--radiation gray``.  Surface SST/SIC
    # and solar TSI affect both gray and RRTMG paths.
    print("[deck] Command:")
    print("  " + " \\\n    ".join(cmd))
    print("[deck] Forcing-channel activity for this run:")
    rad = args.radiation
    rad_active = rad in ("rrtmg", "rrtmgp")
    flag = lambda b: "ACTIVE" if b else "inert (gray radiation)"
    print(f"  SST/SIC                              ACTIVE        (radiation-independent)")
    print(f"  Solar TSI + 14-band spectral         ACTIVE        (radiation-independent)")
    print(f"  Greenhouse gases (transient annual)  {flag(rad_active)}")
    print(f"  Ozone (cyclic clim or interannual)   {flag(rad_active)}")
    if not args.no_aerosol:
        print(f"  Tropospheric aerosol (Kinne)         {flag(rad_active)}")
    if not args.no_volcanic and args.volcanic_aerosol_scale > 0:
        print(f"  Volcanic stratospheric AOD           {flag(rad_active)}")
    if not rad_active:
        print(
            "[deck] NOTE: --radiation gray is FAST but disables all "
            "spectral-radiation-dependent forcings. Use --radiation rrtmg "
            "for the production CMIP6 AMIP physics; gray is for smoke "
            "testing the dycore + SST forcing path only."
        )
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    completed = subprocess.run(cmd, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
