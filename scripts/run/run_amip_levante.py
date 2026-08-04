#!/usr/bin/env python
"""AMIP run using CMIP6 forcing files from the Levante pool.

Uses the real boundary-condition files from
  /pool/data/ICON/grids/public/mpim/
as described in docs/cmip6_forcings.md, and ERA5 initial conditions
prepared by scripts/data/prep_levante_era5_ic.py.

Forcing channels
----------------
SST/SIC         ICON R2B5 unstructured grid (auto-detected)
Solar           14-band spectral TSI from MPI-M CMIP6 file
GHG             Historical CO2/CH4/N2O/CFC from greenhouse_historical_plus.nc
Ozone           CMIP6 UReading interannual ozone (vmro3, multi-level)
Aerosol         Kinne fine-mode SW aerosol (yearly file)
Volcanic AOD    CMIP6 stratospheric volcanic aerosol (yearly file)

Usage::

    # 1-year AMIP, ERA5 IC
    python scripts/run_amip_levante.py \\
        --year 1979 --days 365 \\
        --ic-zarr /scratch/b/b309178/era5_ic_1979-01-01.zarr \\
        --output results/amip_levante_1979

    # Smoke test (5 days, no ERA5 IC)
    python scripts/run_amip_levante.py --year 1979 --days 5

    # Dry run — print run_amip.py command without executing
    python scripts/run_amip_levante.py --year 1979 --days 365 --dry-run

ERA5 IC preparation::

    python scripts/data/prep_levante_era5_ic.py \\
        --year 1979 --month 1 --day 1 \\
        --out /scratch/b/b309178/era5_ic_1979-01-01.zarr
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ROOT = Path("/pool/data/ICON/grids/public/mpim")

# Grid 0019 = R2B5 (default)
_GRID = "0019"

# Forcing file paths
_SST_FILE  = _ROOT / _GRID / "sst_and_seaice" / "r0001" / "bc_sst_1979_2016.nc"
_SIC_FILE  = _ROOT / _GRID / "sst_and_seaice" / "r0001" / "bc_sic_1979_2016.nc"
_SOLAR_FILE = _ROOT / "common" / "solar_radiation" / "swflux_14band_cmip6_1850-2299-v3.2.nc"
_GHG_FILE   = _ROOT / "independent" / "greenhouse_gases" / "greenhouse_historical_plus.nc"

# Ozone files by period
_O3_FILES = {
    range(1850, 1900): _ROOT / "common" / "ozone_cmip6_forcing" / "historical" /
        "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_185001-189912.nc",
    range(1900, 1950): _ROOT / "common" / "ozone_cmip6_forcing" / "historical" /
        "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_190001-194912.nc",
    range(1950, 2000): _ROOT / "common" / "ozone_cmip6_forcing" / "historical" /
        "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc",
    range(2000, 2015): _ROOT / "common" / "ozone_cmip6_forcing" / "historical" /
        "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_200001-201412.nc",
}

# Kinne aerosol: one file per year
def _kinne_file(year: int) -> Path:
    return _ROOT / "common" / "aerosol_kinne" / f"aeropt_kinne_sw_b14_fin_{year}_rast.nc"

# Volcanic aerosol: one file per year
def _volcanic_file(year: int) -> Path:
    return _ROOT / "common" / "aerosol_volcanic_cmip6" / \
        f"bc_aeropt_cmip6_volc_lw_b16_sw_b14_{year}.nc"


def _ozone_file(year: int) -> Path:
    for yr_range, path in _O3_FILES.items():
        if year in yr_range:
            return path
    raise ValueError(f"No ozone file for year {year}; available: 1850-2014")


def _check_files(files: list[Path]) -> list[str]:
    missing = [str(p) for p in files if not p.exists()]
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Simulation time
    parser.add_argument("--year", type=int, default=1979,
                        help="Start year (default 1979)")
    parser.add_argument("--days", type=int, default=365,
                        help="Simulation length in days (default 365)")
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--checkpoint-days", type=int, default=30)

    # Grid / numerics
    parser.add_argument("--resolution", type=int, default=48,
                        help="Cubed-sphere N (default 48 ≈ 2°)")
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "latlon"])
    parser.add_argument("--discretization", type=str, default="centered",
                        choices=["centered", "finite_volume", "latlon_cgrid"])
    parser.add_argument("--use-duogrid", action="store_true", default=False,
                        help="Enable FV3 Duo-Grid halo exchange (required for MPI multi-node)")

    # Physics stack
    parser.add_argument("--radiation", type=str, default="rrtmg",
                        choices=["gray", "rrtmg"])
    parser.add_argument("--rad-update-steps", type=int, default=6)
    parser.add_argument("--convection", type=str, default="sbm")
    parser.add_argument("--turbulence", type=str, default="louis")
    parser.add_argument("--clouds", type=str, default="sundqvist")
    parser.add_argument("--cloud-rh-crit-bl", type=float, default=0.55,
                        help="Critical RH for BL cloud (Sundqvist). Default 0.55 for AMIP.")
    parser.add_argument("--cloud-sigma-bl", type=float, default=0.85,
                        help="Sigma level above which BL rh_crit applies. Default 0.85.")
    parser.add_argument("--microphysics", type=str, default="sundqvist")
    parser.add_argument("--gravity-wave-drag", type=str, default="rayleigh")
    parser.add_argument("--diurnal-cycle", action="store_true", default=True)
    parser.add_argument("--no-diurnal-cycle", dest="diurnal_cycle", action="store_false")
    parser.add_argument("--land-mask-file", type=str, default="",
                        help="Land-sea-mask NetCDF (CMIP6 sftlf / ERA5 lsm). "
                             "Activates the slab-land surface tile.")
    parser.add_argument("--albedo-land-file", type=str, default="",
                        help="Static land-albedo NetCDF (e.g. ICON-extpar ALB); "
                             "overrides the latitude-vegetation default.")
    parser.add_argument("--albedo-land-month", type=int, default=0,
                        help="Month (1-12) from a monthly albedo climatology; "
                             "0 = annual mean.")

    # Initial conditions / restart
    parser.add_argument("--ic-zarr", type=str, default="",
                        help="Path to ERA5 IC Zarr store "
                             "(from scripts/data/prep_levante_era5_ic.py). "
                             "If empty, uses held_suarez default IC.")
    parser.add_argument("--restart-from", type=str, default="",
                        help="Checkpoint .npz from a previous segment. "
                             "When set, overrides --ic-zarr (state comes from checkpoint).")
    parser.add_argument("--restart-start-day", type=float, default=None,
                        help="Override start_day after loading checkpoint. "
                             "Pass 0.0 at year boundaries to reset the day counter to Jan 1.")

    # Forcing toggles
    parser.add_argument("--no-aerosol", action="store_true", default=False)
    parser.add_argument("--no-volcanic", action="store_true", default=False)
    parser.add_argument("--volcanic-scale", type=float, default=1.0)

    # Output
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--monthly-means", action="store_true", default=True)
    parser.add_argument("--no-monthly-means", dest="monthly_means", action="store_false")
    parser.add_argument("--cmip-output", action="store_true", default=False)
    # Clear-sky TOA diagnostic (rsutcs/rlutcs) + the cloud CMOR trio
    # (clt/clwvi/clivi).  ON by default — the deck reports CRE — but it is now
    # a REAL cost on the MPAS lane: it adds a second clouds-off radiation solve
    # per radiation step (~2x the radiation time).  It used to be silently
    # ignored there, so this flag was free; ``--no-clear-sky-diag`` is the
    # opt-out for a spin-up / throughput run that does not score CRE.
    parser.add_argument("--clear-sky-diag", dest="clear_sky_diag",
                        action="store_true", default=True)
    parser.add_argument("--no-clear-sky-diag", dest="clear_sky_diag",
                        action="store_false")

    # Multi-node MPI
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Launch run_amip.py via srun for multi-node MPI execution. "
                             "The SLURM job must allocate multiple nodes with "
                             "--ntasks-per-node=1. Number of ranks is read from "
                             "SLURM_NTASKS (override with --n-ranks).")
    parser.add_argument("--n-ranks", type=int, default=None,
                        help="Number of MPI ranks. Defaults to $SLURM_NTASKS when "
                             "--distributed is set.")

    parser.add_argument("--production-profile", action="store_true", default=False,
                        help="Pass --production-profile to run_amip.py (auto-sets rad-update-steps "
                             "to floor(3600/dt) and activates SPMD halo backend). "
                             "When set, --rad-update-steps is NOT forwarded explicitly so "
                             "run_amip.py can compute the production cadence from --dt.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print run_amip.py command and exit without running")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded verbatim to run_amip.py")

    args = parser.parse_args(argv)

    # --- Resolve forcing files ---
    year = args.year
    ozone_f  = _ozone_file(year)
    kinne_f  = _kinne_file(year)
    volc_f   = _volcanic_file(year)

    # Verify files exist
    required = [_SST_FILE, _SIC_FILE, _SOLAR_FILE, _GHG_FILE, ozone_f]
    if not args.no_aerosol:
        required.append(kinne_f)
    if not args.no_volcanic and not args.no_aerosol and args.volcanic_scale > 0:
        required.append(volc_f)
    if args.ic_zarr:
        required.append(Path(args.ic_zarr))
    if args.land_mask_file:
        required.append(Path(args.land_mask_file))
    if args.albedo_land_file:
        required.append(Path(args.albedo_land_file))

    missing = _check_files(required)
    if missing:
        print("[levante] ERROR — missing files:")
        for p in missing:
            print(f"  {p}")
        if str(kinne_f) in missing:
            available = [y for y in range(1845, 2101) if _kinne_file(y).exists()]
            if available:
                print(f"\n  Kinne aerosol: available years "
                      f"{min(available)}–{max(available)}")
        return 2

    # --- Build run_amip.py command ---
    run_amip = _REPO / "scripts" / "run" / "run_amip.py"
    cmd = [
        sys.executable, str(run_amip),
        # SST/SIC — ICON unstructured (auto-detected)
        "--dataset", "custom",
        "--forcing-path", str(_SST_FILE),
        "--sic-path", str(_SIC_FILE),
        "--sst-var", "tosbcs",      # Kelvin — no offset needed
        "--sic-var", "siconcbcs",   # percent → fraction
        "--sst-offset", "0.0",
        "--sic-scale", "0.01",
        "--time-var", "time",
        # Grid
        "--resolution", str(args.resolution),
        "--nlev", str(args.nlev),
        "--grid-type", args.grid_type,
        "--discretization", args.discretization,
        *(["--use-duogrid"] if args.use_duogrid else []),
        # Integration
        "--days", str(args.days),
        "--dt", str(args.dt),
        "--diag-days", str(args.diag_days),
        "--checkpoint-days", str(args.checkpoint_days),
        "--start-year", str(year),
        "--experiment", "amip",
        # Radiation
        "--radiation", args.radiation,
        # Only forward --rad-update-steps when --production-profile is NOT set;
        # production-profile auto-computes the 1-hour cadence from --dt.
        *(["--rad-update-steps", str(args.rad_update_steps)]
          if not args.production_profile else []),
        # Solar (14-band spectral MPI-M file)
        "--solar-source", "spectral_file",
        "--solar-file", str(_SOLAR_FILE),
        "--solar-tsi-var", "TSI",
        "--solar-spectral-var", "SSI_frac",
        # GHG
        "--ghg-forcing", "external",
        "--ghg-file", str(_GHG_FILE),
        # Ozone (interannual CMIP6)
        "--ozone-source", "standard",
        "--ozone-forcing", "external",
        "--ozone-file", str(ozone_f),
        # Physics
        "--convection", args.convection,
        "--turbulence", args.turbulence,
        "--gravity-wave-drag", args.gravity_wave_drag,
        "--clouds", args.clouds,
        "--cloud-rh-crit-bl", str(args.cloud_rh_crit_bl),
        "--cloud-sigma-bl", str(args.cloud_sigma_bl),
        "--microphysics", args.microphysics,
    ]

    # Diagnostics
    if args.clear_sky_diag:
        cmd.append("--clear-sky-diag")

    if args.diurnal_cycle:
        cmd.append("--diurnal-cycle")
    if args.monthly_means:
        cmd.append("--monthly-means")
    if args.cmip_output:
        cmd.append("--cmip-output")

    # Aerosol
    if not args.no_aerosol:
        cmd += [
            "--aerosol-forcing", "external",
            "--aerosol-file", str(kinne_f),
            "--aerosol-reference-aod", "0.05",
        ]
    if not args.no_volcanic and not args.no_aerosol and args.volcanic_scale > 0:
        cmd += [
            "--volcanic-aerosol-file", str(volc_f),
            "--volcanic-aerosol-scale", str(args.volcanic_scale),
        ]

    # ERA5 IC / restart — mutually exclusive; restart takes precedence
    if args.restart_from:
        cmd += ["--restart-from", args.restart_from]
        if args.restart_start_day is not None:
            cmd += ["--restart-start-day", str(args.restart_start_day)]
    elif args.ic_zarr:
        cmd += ["--ic", "era5", "--ic-path", args.ic_zarr]

    # Land-sea mask — activates the slab-land surface tile
    if args.land_mask_file:
        cmd += ["--land-mask-file", args.land_mask_file]
    if args.albedo_land_file:
        cmd += ["--albedo-land-file", args.albedo_land_file,
                "--albedo-land-month", str(args.albedo_land_month)]

    # Output
    if args.output:
        cmd += ["--output", args.output]

    if args.production_profile:
        cmd.append("--production-profile")

    if args.extra:
        cmd += list(args.extra)

    # --- Activity report ---
    print("[levante] Forcing-channel activity:")
    rad_active = args.radiation in ("rrtmg", "rrtmgp")
    print(f"  SST/SIC           ICON R2B5 unstructured  ACTIVE")
    print(f"  Solar TSI 14-band {_SOLAR_FILE.name}  ACTIVE")
    print(f"  GHG (transient)   {_GHG_FILE.name}  "
          f"{'ACTIVE' if rad_active else 'inert (gray radiation)'}")
    print(f"  Ozone (CMIP6)     {ozone_f.name}  "
          f"{'ACTIVE' if rad_active else 'inert (gray radiation)'}")
    if not args.no_aerosol:
        print(f"  Kinne aerosol     {kinne_f.name}  "
              f"{'ACTIVE' if rad_active else 'inert (gray radiation)'}")
    if not args.no_volcanic and not args.no_aerosol and args.volcanic_scale > 0:
        print(f"  Volcanic AOD      {volc_f.name}  "
              f"{'ACTIVE' if rad_active else 'inert (gray radiation)'}")
    if args.restart_from:
        day_info = f"  (start_day→{args.restart_start_day:.0f})" if args.restart_start_day is not None else ""
        print(f"  Restart           {args.restart_from}{day_info}")
    elif args.ic_zarr:
        print(f"  ERA5 IC           {args.ic_zarr}")
    else:
        print(f"  ERA5 IC           (none — using held_suarez default IC)")

    # Wrap with srun for multi-node MPI execution
    if args.distributed:
        n_ranks = args.n_ranks or int(os.environ.get("SLURM_NTASKS", 2))
        srun_prefix = [
            "srun",
            f"--ntasks={n_ranks}",
            "--ntasks-per-node=1",
            "--cpu-bind=none",
        ]
        cmd = srun_prefix + cmd
        print(f"[levante] Multi-node MPI: {n_ranks} ranks via srun")

    print("\n[levante] Command:")
    print("  " + " \\\n    ".join(cmd))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
