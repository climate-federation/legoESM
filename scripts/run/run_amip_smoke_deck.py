#!/usr/bin/env python
"""AMIP forcing-pipeline SMOKE deck (synthetic forcing) -- NOT production.

This is NOT the production AMIP configuration.  Production AMIP is
``scripts/run/run_amip.py --config config/amip/amip_production.yaml`` plus the
machine paths of ``config/amip/amip_production{,.ginsburg}.sh``.  This script
(formerly ``run_amip_cmip6_deck.py``; renamed by review 2026-10-10 F34) runs a
historical C16/L30 smoke stack -- RRTMG + Sundqvist clouds + Morrison + SBM +
Louis + McFarlane -- with FLAT topography (no land anywhere), no sub-grid
orography file and, by default, the SYNTHETIC ``data/forcing_amip`` deck.  It
exercises the forcing plumbing end to end; its climate means nothing.

It wraps :mod:`scripts.run_amip` and orchestrates the forcing suite (SST/SIC +
transient GHG + ozone + solar + aerosol + volcanic).  Forcing files can either
be pre-generated (e.g. by :mod:`scripts.generate_amip_forcing`) or
auto-generated on the fly:

1. Generates synthetic CMIP6-shape forcing files when missing.
2. Sets the smoke RRTMG + Sundqvist clouds + Morrison + SBM stack.
3. Forces ``--ghg-forcing external`` and ``--ozone-forcing external``.
4. Sets ``--solar-source spectral_file`` with the canonical
   ``--solar-tsi-var TSI --solar-spectral-var SSI_frac`` flags so the
   per-band 14-fraction solar file from MPI-M is consumed correctly.
5. Wires aerosol + volcanic flags.
6. Defaults to a 12-month integration so wall-clock is bounded.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py \\
        --resolution 16 --days 30 \\
        --output results/amip_deck_test

To run without forcing files (the script will generate them)::

    JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py \\
        --auto-generate --resolution 16 --days 30

To target a specific grid type::

    --grid-type {cubed_sphere,gaussian,latlon,voronoi}

The script always passes ``--monthly-means`` so post-run validation
tools (``scripts/validate/validate_amip_run.py``) can read zonal-mean diagnostics.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FORCING = _REPO_ROOT / "data" / "forcing_amip"
# Public Analysis-Ready Cloud-Optimized (ARCO) ERA5 on GCS — no
# credentials required.  Used as the default ERA5 IC source so the
# realistic --ic era5 path works out of the box (load_era5_ic handles
# the gs:// URI + auto-detects pressure levels and the nearest time).
_DEFAULT_ARCO_ERA5 = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)

# CLAUDE.md "Constant and Parameter Discipline": never hardcode 273.15 etc.
sys.path.insert(0, str(_REPO_ROOT / "src"))
from legoesm import constants  # noqa: E402


def _check_forcing_files(
    forcing_dir: Path, start_year: int, end_year: int,
    *, require_aerosol: bool = True, require_volcanic: bool = True,
    require_sst: bool = True,
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
    if not require_sst:
        # A real --sst-file is supplied separately; don't require the
        # synthetic sst_sic deck file.
        skip.add("sst")
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
        str(_REPO_ROOT / "scripts" / "data" / "generate_amip_forcing.py"),
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
    parser.add_argument("--dt-auto", action="store_true", default=False,
                        help="Use the cross-grid stability-ladder dt for "
                             "(grid, resolution) instead of --dt "
                             "(forwarded to run_amip; ladder-safe for "
                             "long production runs).")
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--checkpoint-days", type=int, default=0)

    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "gaussian", "latlon",
                                 "mpas",
                                 # Legacy aliases for the SCVT mesh,
                                 # normalised by run_amip.py
                                 "voronoi", "icosahedral", "mpas_voronoi"])
    # Default None → resolved per-grid in main() (codex r2): keep the deck's
    # historical 'finite_volume' for cube/latlon but pick 'mpas' for the
    # Voronoi mesh, whose only supported discretization is 'mpas'. A hard-coded
    # 'finite_volume' default made a bare --grid-type voronoi die at the dycore
    # factory on (hydrostatic, finite_volume, mpas).
    parser.add_argument("--discretization", type=str, default=None,
                        choices=["centered", "finite_volume", "cgrid",
                                  "latlon_cgrid", "cdgrid", "mpas", "spectral"])

    # Physics — defaults match a balanced CMIP6 AMIP stack
    parser.add_argument("--radiation", type=str, default="rrtmg")
    parser.add_argument("--rad-update-steps", type=int, default=6)
    # Clear-sky diagnostic: a SECOND clear-sky RRTMG pass per radiation call
    # (≈2× the radiation cost) needed for lwcre/swcre in ClimateEval Tier 2, but
    # pure overhead for a run that does not score CRE.  ON by default (the
    # production deck reports CRE); ``--no-clear-sky-diag`` drops the second pass
    # for cheaper spin-up / perf runs (issue #434 cheap lever).
    parser.add_argument("--clear-sky-diag", dest="clear_sky_diag",
                        action="store_true", default=True)
    parser.add_argument("--no-clear-sky-diag", dest="clear_sky_diag",
                        action="store_false")
    parser.add_argument("--clouds", type=str, default="sundqvist")
    # Morrison double-moment (M2005/MG) is the production default: the
    # most faithful + best-validated microphysics in the repo (SAM-oracle
    # validation in docs/dev-notes/CRM_faithful_SAM.md; RCEMIP-viable per
    # the microphysics-RCE campaign) with number-aware r_eff and the
    # aerosol-CCN coupling.  Sundqvist (the previous default) remains
    # the fast diagnostic fallback (--microphysics sundqvist).  Kessler
    # in the integrated AMIP path produces NaN winds at day ~2 with
    # current SBM/clouds settings (AMIP.md "Known issues").
    parser.add_argument("--microphysics", type=str, default="morrison")
    parser.add_argument("--convection", type=str, default="sbm")
    parser.add_argument("--turbulence", type=str, default="louis")
    # Match run_amip's production default: since the full-physics guard
    # (_require_full_physics_for_amip) rejects 'none', a deck default of
    # 'none' made every deck launch (and the all-grids smoke) die at parse
    # time (drift caught by the 2026-07-17 AMIP audit smoke run).
    parser.add_argument("--gravity-wave-drag", type=str, default="mcfarlane")
    parser.add_argument("--diurnal-cycle", action="store_true", default=True)
    parser.add_argument("--no-diurnal-cycle", dest="diurnal_cycle",
                        action="store_false")
    # Aerosol-CCN coupling (Andreae 2009 AOD->CCN -> specified Nc) and
    # zenith-dependent ocean albedo (Briegleb 1992): ON by default for
    # the production stack; both require/imply their upstream channel
    # (aerosol external forcing; rrtmg radiation).  Disable for
    # bit-compat with pre-2026-06 runs.
    parser.add_argument("--aerosol-ccn", dest="aerosol_ccn",
                        action="store_true", default=True)
    parser.add_argument("--no-aerosol-ccn", dest="aerosol_ccn",
                        action="store_false")
    parser.add_argument("--dynamic-albedo", dest="dynamic_albedo",
                        action="store_true", default=True)
    parser.add_argument("--no-dynamic-albedo", dest="dynamic_albedo",
                        action="store_false")

    # Output
    parser.add_argument("--ic", type=str, default="era5",
                        choices=["default", "standard", "era5"],
                        help="Initial condition (default 'era5' for "
                             "production AMIP CMIP realism): 'era5' "
                             "(reanalysis from --ic-path; supported on "
                             "cubed_sphere / latlon / gaussian — winds + "
                             "Earth-like moisture, no spin-up cold drift), "
                             "'standard' (lapse-rate + equator-pole "
                             "gradient + thermal-wind jet; lat-lon only), "
                             "or 'default' (uniform T_init rest state — "
                             "fast but unphysically weak winds + high CWV; "
                             "the only IC for voronoi/mpas until "
                             "era5_to_mpas_carry lands).")
    parser.add_argument("--ic-path", type=str, default="",
                        help="ERA5 Zarr path / GCS URI (required when "
                             "--ic era5).  Default: the public ARCO ERA5 "
                             "store (gs://gcp-public-data-arco-era5/...), "
                             "no credentials needed.")
    # Real (vs synthetic) prescribed SST/SIC.  ``--sst-file`` points the
    # deck at a real input4MIPs AMIP II boundary-condition file; the
    # var-name/unit flags default to that dataset's convention
    # (``tosbcs`` in K, ``siconcbcs`` in percent).  Omit ``--sst-file``
    # to use the synthetic deck SST (``sst`` in Celsius).
    parser.add_argument("--sst-file", type=str, default="",
                        help="Real prescribed-SST file (e.g. PCMDI/"
                             "input4MIPs AMIP II bcs) instead of the "
                             "synthetic deck SST.  Pair with --sst-var/"
                             "--sst-offset/--sic-var/--sic-scale or accept "
                             "the input4MIPs defaults.")
    parser.add_argument("--sst-var", type=str, default=None,
                        help="SST variable name (default: 'sst' synthetic, "
                             "'tosbcs' when --sst-file is given).")
    parser.add_argument("--sic-var", type=str, default=None,
                        help="SIC variable name (default: 'sic' synthetic, "
                             "'siconcbcs' when --sst-file is given).")
    parser.add_argument("--sst-offset", type=float, default=None,
                        help="Additive SST offset to Kelvin (default: "
                             "273.15 synthetic-Celsius, 0.0 for a Kelvin "
                             "input4MIPs file).")
    parser.add_argument("--sic-scale", type=float, default=None,
                        help="Multiplicative SIC scale to fraction "
                             "(default: 1.0 synthetic-fraction, 0.01 for a "
                             "percent input4MIPs file).")
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

    # Resolve the per-grid discretization default (codex r2): preserve the
    # deck's historical 'finite_volume' for cube/latlon; the Voronoi mesh only
    # supports 'mpas'.  Done here so the spectral_path / mpas_path activity
    # labels below and the forwarded flag all see a concrete value.
    if args.discretization is None:
        if args.grid_type in ("voronoi", "icosahedral", "mpas_voronoi",
                              "mpas"):
            args.discretization = "mpas"
        elif args.grid_type == "gaussian":
            args.discretization = "spectral"
        else:
            args.discretization = "finite_volume"

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
        require_sst=not args.sst_file,
    )

    if "_missing" in files:
        if args.auto_generate:
            _auto_generate(forcing_dir, args.start_year, args.end_year)
            files = _check_forcing_files(
                forcing_dir, args.start_year, args.end_year,
                require_aerosol=aerosol_active,
                require_volcanic=volcanic_active,
                require_sst=not args.sst_file,
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
                  "run scripts/data/generate_amip_forcing.py manually first.")
            return 2

    # Resolve the SST source + its var-name/unit convention.  A real
    # ``--sst-file`` (input4MIPs AMIP II bcs) defaults to that dataset's
    # convention (tosbcs in K, siconcbcs in percent); the synthetic deck
    # SST is ``sst`` in Celsius, fraction SIC.  Explicit --sst-var etc.
    # always win.  The amip.py units-attribute guard cross-checks the
    # chosen offset/scale against the file's ``units`` and the converted
    # values, so a wrong combination fails loudly rather than mis-forcing.
    # An SST var/unit override without --sst-file would point the
    # SYNTHETIC deck file at a real-data variable name -> late loader
    # KeyError.  Reject up front (codex review).
    if not args.sst_file and any(v is not None for v in (
            args.sst_var, args.sic_var, args.sst_offset, args.sic_scale)):
        print("[deck] ERROR: --sst-var/--sic-var/--sst-offset/--sic-scale "
              "only apply to a real --sst-file. The synthetic deck SST is "
              "'sst' in Celsius (offset 273.15), 'sic' as a fraction. "
              "Provide --sst-file <input4MIPs file> or drop the overrides.")
        return 2
    if args.sst_file:
        _sst_path = args.sst_file
        _sst_var = args.sst_var if args.sst_var is not None else "tosbcs"
        _sic_var = args.sic_var if args.sic_var is not None else "siconcbcs"
        _sst_off = args.sst_offset if args.sst_offset is not None else 0.0
        _sic_scl = args.sic_scale if args.sic_scale is not None else 0.01
    else:
        _sst_path = str(files["sst"])
        _sst_var = "sst"
        _sic_var = "sic"
        _sst_off = constants.T_freeze  # synthetic file is Celsius
        _sic_scl = 1.0

    # Build the run_amip.py command
    run_amip = _REPO_ROOT / "scripts" / "run" / "run_amip.py"
    cmd = [
        sys.executable, str(run_amip),
        "--dataset", "custom",
        "--forcing-path", _sst_path,
        "--sst-var", _sst_var,
        "--sic-var", _sic_var,
        "--sst-offset", str(_sst_off),
        "--sic-scale", str(_sic_scl),
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
        *(["--dt-auto"] if args.dt_auto else []),
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
        # The MPI-M CMIP6 SSI_frac file is in RRTMG-SW band order; rotate
        # it to RRTMGP order before g-point expansion (issue #322),
        # otherwise UV flux is dumped into the near-IR water-vapour band.
        "--solar-spectral-band-order", "rrtmg_sw",
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
        *(["--clear-sky-diag"] if args.clear_sky_diag else []),
    ]
    # Initial condition.  ``era5`` (the production default) gives
    # realistic winds + Earth-like moisture and avoids the uniform-IC
    # cold-start (weak winds, ~1 K/day drift, CWV ~80).  Resolve the IC
    # per grid so each grid uses the most realistic IC it supports:
    #   cubed_sphere / latlon / gaussian-spectral : era5 (needs --ic-path)
    #   voronoi / mpas                            : era5 not yet wired ->
    #                                               fall back to default
    # ``standard`` (the balanced-jet IC) is lat-lon only.
    _ic = args.ic
    # Did the user EXPLICITLY ask for this IC, or is it the era5 default?
    # An explicit request for an IC a grid cannot honour must FAIL, not
    # silently downgrade; the default may quietly fall back (codex review).
    _ic_explicit = "--ic" in sys.argv
    _mpas = (args.grid_type in ("voronoi", "mpas")
             or args.discretization == "mpas")
    if _ic == "era5":
        # Resolve the unsupported-grid fallback BEFORE the --ic-path
        # check: an MPAS run never uses ERA5, so it must not be forced to
        # supply an ERA5 path (codex review).
        if _mpas:
            if _ic_explicit:
                print("[deck] ERROR: --ic era5 explicitly requested but "
                      "ERA5 IC is not wired for voronoi/mpas (needs "
                      "era5_to_mpas_carry). Pass --ic default for this "
                      "grid, or run a supported grid (cubed_sphere/latlon/"
                      "gaussian).")
                return 2
            print("[deck] NOTE: ERA5 IC is not yet wired for voronoi/mpas; "
                  "the era5 DEFAULT falls back to --ic default here.")
            _ic = "default"
    elif _ic == "standard" and args.grid_type != "latlon":
        if _ic_explicit:
            print(f"[deck] ERROR: --ic standard explicitly requested but "
                  f"the balanced-jet standard IC is lat-lon only; "
                  f"{args.grid_type} is unsupported. Pass --ic era5 "
                  "(with --ic-path) or --ic default.")
            return 2
        print(f"[deck] NOTE: --ic standard is lat-lon only; "
              f"{args.grid_type} falls back to --ic default.")
        _ic = "default"
    cmd += ["--ic", _ic]
    if _ic == "era5":
        # An EXPLICIT --ic-path "" (e.g. from a shell-interpolated unset
        # var) is a user error, not a request for the ARCO default —
        # error rather than silently fall through (codex review).
        if "--ic-path" in sys.argv and not args.ic_path:
            print("[deck] ERROR: --ic-path was given but is empty (likely "
                  "an unset shell variable). Provide a real ERA5 zarr/GCS "
                  "URI, or omit --ic-path to use the public ARCO ERA5 "
                  "default.")
            return 2
        # Default to the public ARCO ERA5 store (no credentials) so the
        # realistic IC works out-of-box; an explicit --ic-path overrides.
        _ic_path = args.ic_path or _DEFAULT_ARCO_ERA5
        if not args.ic_path:
            print(f"[deck] NOTE: --ic era5 with no --ic-path; using the "
                  f"public ARCO ERA5 store {_ic_path} (override with "
                  "--ic-path).")
        cmd += ["--ic-path", _ic_path]
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
    # Aerosol-CCN + dynamic albedo run through the coupled physics
    # pipeline, which only the cubed-sphere / lat-lon paths use — the
    # MPAS and spectral standalone loops build physics via
    # ``make_physics`` and do not fill the specified-Nc field or the
    # pipeline albedo blend (run_amip hard-errors on --aerosol-ccn for
    # those grids).  Gate on grid path + the upstream requirements so
    # the default-on flags degrade gracefully.
    _pipeline_path = (args.grid_type in ("cubed_sphere", "latlon")
                      and args.discretization not in ("spectral", "mpas"))
    if (args.aerosol_ccn and aerosol_active
            and args.microphysics == "morrison" and _pipeline_path):
        cmd.append("--aerosol-ccn")
    elif args.aerosol_ccn:
        print("[deck] NOTE: aerosol-CCN coupling skipped "
              f"(aerosol_active={aerosol_active}, "
              f"microphysics={args.microphysics!r}, "
              f"grid={args.grid_type}/{args.discretization}; needs "
              "external aerosol + morrison + cubed_sphere/latlon).")
    # Zenith-dependent ocean albedo: meaningful for spectral radiation
    # only (gray has no surface SW dependence on albedo blending here).
    if (args.dynamic_albedo and args.radiation in ("rrtmg", "rrtmgp")
            and _pipeline_path):
        cmd.append("--dynamic-albedo")
    elif args.dynamic_albedo:
        print("[deck] NOTE: dynamic (zenith) ocean albedo skipped "
              f"(radiation={args.radiation!r}, "
              f"grid={args.grid_type}/{args.discretization}; needs "
              "rrtmg + cubed_sphere/latlon).")
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
    # affects every path.
    #
    # 2026-06-10: the gaussian/spectral and voronoi/mpas paths now run
    # the SAME unified physics pipeline (RRTMGP + convection +
    # microphysics + clouds) with external ozone / aerosol / transient
    # GHG threaded per step through the traced ``forcing`` dict, so
    # GHG/ozone/aerosol/volcanic are ACTIVE there under rrtmg.  The one
    # remaining gap on those two paths is the solar FILE (TSI +
    # 14-band spectral): they integrate with the configured constant
    # S_0 (annual TSI cycle ~0.1 W/m² is not threaded yet).
    print("[deck] Command:")
    print("  " + " \\\n    ".join(cmd))
    print("[deck] Forcing-channel activity for this run:")
    rad = args.radiation
    rad_active = rad in ("rrtmg", "rrtmgp")
    spectral_path = (args.grid_type == "gaussian"
                     and args.discretization == "spectral")
    mpas_path = (args.grid_type in ("voronoi", "mpas", "mpas_voronoi",
                                    "icosahedral")
                 and args.discretization == "mpas")
    effective_active = rad_active

    def _flag(b: bool) -> str:
        if b:
            return "ACTIVE"
        return "inert (gray radiation)"

    if spectral_path:
        solar_label = ("inert (spectral path uses constant S_0; "
                       "solar file not threaded)")
    elif mpas_path:
        solar_label = ("inert (MPAS path uses constant S_0; "
                       "solar file not threaded)")
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
