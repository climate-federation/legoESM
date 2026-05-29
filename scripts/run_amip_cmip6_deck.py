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


def _check_forcing_files(
    forcing_dir: Path, start_year: int, end_year: int,
    *, require_aerosol: bool = True, require_volcanic: bool = True,
) -> dict[str, Path]:
    """Verify the canonical 6-file deck exists in ``forcing_dir``.

    Ozone resolution (``ozone_amip_<sy>-<ey>.nc`` interannual *vs.*
    ``ozone_amip_clim.nc`` 12-month climatology): if both are present
    we prefer the **interannual** file because that's what real
    input4MIPs CMIP6 ozone is — the loader's non-cyclic dispatch
    (`_interp_monthly_noncyclic`, keyed on ``ntime > 12``) is the right
    code path to exercise for production AMIP.  The climatology is the
    fallback when the user has only generated the cyclic file.

    Parameters
    ----------
    require_aerosol, require_volcanic : bool
        When False, the corresponding file is omitted from the
        missing-file check.  This lets users intentionally disable a
        channel (``--no-aerosol`` / ``--no-volcanic`` /
        ``--volcanic-aerosol-scale 0``) without supplying files for it.
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
    # Skip aerosol/volcanic from the *required* set when the caller
    # intends to disable them; the path is still returned in ``files``
    # so the caller can choose whether to forward it.
    skip = set()
    if not require_aerosol:
        skip.add("aerosol")
    if not require_volcanic:
        skip.add("volcanic")
    missing = [name for name, p in files.items()
               if name not in skip and not p.exists()]
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
                        choices=["cubed_sphere", "gaussian", "latlon",
                                 "mpas",
                                 # Legacy aliases for the SCVT mesh,
                                 # normalised by run_amip.py
                                 "voronoi", "icosahedral", "mpas_voronoi"])
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
    parser.add_argument("--ic", type=str, default="default",
                        choices=["default", "standard", "era5"],
                        help="Initial condition: 'default' (uniform T_init rest "
                             "state), 'standard' (realistic lapse-rate + "
                             "equator-pole gradient + thermal-wind jet; lat-lon "
                             "only — Earth-like CWV), or 'era5' (reanalysis from "
                             "--ic-path).")
    parser.add_argument("--ic-path", type=str, default="",
                        help="ERA5 Zarr path when --ic era5.")
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

    # ``ModelDriver`` only consumes the volcanic file when
    # ``aerosol_forcing == "external"``.  When the user disables
    # aerosol entirely, volcanic is silently dropped too — the
    # downstream activity report would otherwise lie.  Force the two
    # flags into a consistent state at the deck-driver boundary so
    # the rest of this function can reason about a single ``aerosol_active``
    # / ``volcanic_active`` state.
    aerosol_active = not args.no_aerosol
    volcanic_active = (not args.no_volcanic
                       and args.volcanic_aerosol_scale > 0
                       and aerosol_active)
    if (not aerosol_active) and (not args.no_volcanic) and (
        args.volcanic_aerosol_scale > 0
    ):
        # User asked to disable aerosol but left volcanic enabled.
        # ModelDriver gates volcanic on aerosol_forcing being external,
        # so volcanic would silently drop.  Inform the user up front
        # and treat the run as no-volcanic.
        print(
            "[deck] NOTE: --no-aerosol was passed without --no-volcanic; "
            "ModelDriver only loads volcanic forcing when aerosol "
            "forcing is external, so volcanic is being disabled too. "
            "Pass --no-volcanic explicitly to silence this notice, or "
            "drop --no-aerosol if you want both channels active."
        )

    forcing_dir = Path(args.forcing_dir).resolve()
    # Skip aerosol/volcanic from the missing-file check when those
    # channels are intentionally disabled — otherwise a user with only
    # SST/GHG/ozone/solar files (a perfectly valid no-aerosol AMIP run)
    # would hit a spurious missing-file error before the model starts.
    files = _check_forcing_files(
        forcing_dir, args.start_year, args.end_year,
        require_aerosol=aerosol_active,
        require_volcanic=volcanic_active,
    )

    if "_missing" in files:
        if args.auto_generate:
            _auto_generate(forcing_dir, args.start_year, args.end_year)
            files = _check_forcing_files(
                forcing_dir, args.start_year, args.end_year,
                require_aerosol=aerosol_active,
                require_volcanic=volcanic_active,
            )
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
    # Initial condition (default keeps the prior uniform-T_init behaviour; pass
    # --ic standard for a physically realistic lapse-rate + balanced-jet IC on
    # lat-lon — Earth-like column water vapour).
    cmd += ["--ic", args.ic]
    if args.ic == "era5":
        cmd += ["--ic-path", args.ic_path]
    if args.diurnal_cycle:
        cmd.append("--diurnal-cycle")
    if args.monthly_means:
        cmd.append("--monthly-means")
    if aerosol_active:
        cmd += [
            "--aerosol-forcing", "external",
            "--aerosol-file", str(files["aerosol"]),
            "--aerosol-reference-aod", "0.05",
        ]
    # ``volcanic_active`` requires ``aerosol_active`` (see top of main()
    # for the rationale): ModelDriver gates volcanic on
    # ``aerosol_forcing == "external"``, so passing
    # ``--volcanic-aerosol-file`` without ``--aerosol-forcing external``
    # would be silently ignored.
    if volcanic_active:
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
    #
    # Two grid paths bypass the external-forcing pipeline entirely
    # even when the user has requested rrtmg/rrtmgp:
    #
    # * gaussian/spectral routes through ``ModelDriver._run_spectral``,
    #   which hard-codes gray radiation + constant solar.
    # * voronoi/mpas routes through ``ModelDriver._run_mpas``, which
    #   builds the physics via ``make_physics(model_type="mpas", ...)``
    #   without calling ``_precompute_external_forcing`` and without
    #   passing ``SegmentForcing`` — the external o3/aerosol/ghg/solar
    #   configs are *set up* by ``_configure_external_forcing`` but
    #   never reach the radiation kernel on this grid.
    #
    # Without an explicit warning here the deck-driver activity report
    # would say "ACTIVE" while the run silently ignores those forcings
    # — the silent-bias case the iter-3/4 codex reviews flagged.
    print("[deck] Command:")
    print("  " + " \\\n    ".join(cmd))
    print("[deck] Forcing-channel activity for this run:")
    rad = args.radiation
    rad_active = rad in ("rrtmg", "rrtmgp")
    spectral_path = (args.grid_type == "gaussian"
                     and args.discretization == "spectral")
    mpas_path = (args.grid_type == "voronoi"
                 and args.discretization == "mpas")
    bypassed_path = spectral_path or mpas_path
    forcing_silently_dropped = rad_active and bypassed_path
    if forcing_silently_dropped:
        path_name = ("gaussian/spectral" if spectral_path
                     else "voronoi/mpas")
        print(
            f"[deck] WARNING: {path_name} routes through a code path "
            "that does NOT consume external CMIP6 forcings even when "
            "--radiation rrtmg is selected.  GHG / ozone / aerosol / "
            "volcanic configs are LOADED but the radiation kernel on "
            "this grid uses an internal default profile.  Use "
            "cubed_sphere/latlon for production CMIP6 AMIP runs, or "
            "run with --radiation gray here so the activity report "
            "below matches what the model actually does."
        )
    # Effective active state: a forcing is only ACTIVE if both
    # rrtmg/rrtmgp is selected AND the grid path actually consumes it.
    effective_active = rad_active and not bypassed_path

    def _flag(b: bool) -> str:
        if b:
            return "ACTIVE"
        if rad_active and spectral_path:
            return "inert (spectral path uses gray radiation)"
        if rad_active and mpas_path:
            return "inert (MPAS path bypasses external forcing)"
        return "inert (gray radiation)"

    if spectral_path:
        solar_label = "inert (spectral path uses constant S_0)"
    elif mpas_path:
        solar_label = "inert (MPAS path uses constant S_0)"
    else:
        solar_label = "ACTIVE"

    print(f"  SST/SIC                              ACTIVE        (radiation-independent)")
    print(f"  Solar TSI + 14-band spectral         {solar_label:<35}")
    print(f"  Greenhouse gases (transient annual)  {_flag(effective_active)}")
    print(f"  Ozone (cyclic clim or interannual)   {_flag(effective_active)}")
    if aerosol_active:
        print(f"  Tropospheric aerosol (Kinne)         {_flag(effective_active)}")
    if volcanic_active:
        print(f"  Volcanic stratospheric AOD           {_flag(effective_active)}")
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
