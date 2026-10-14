#!/usr/bin/env python
"""Multi-grid AMIP smoke test for the CMIP6 deck driver.

Runs a short (1-3 day) AMIP simulation on each supported grid type
(cubed_sphere / gaussian / latlon / voronoi) using the deck driver
and ``--radiation gray`` (so JIT compile times stay bounded).

Pass criteria:
  - Run completes (exit code 0)
  - Final state is finite (no NaN/Inf)
  - Mass conservation < 1e-3 (relative)
  - Energy budget residual < 500 W/m² (cold-start tolerance)

The script reports a one-line summary per grid and exits non-zero if
any grid fails. Used both as a CI smoke test and as documentation that
the AMIP CMIP6 deck driver works across all four grid topologies.

Usage::

    python scripts/smoke_test_amip_all_grids.py [--days 2] [--resolution 12]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Per-case configuration.  Keys are unique labels; values supply the
# grid_type / discretization / resolution / per-case ``extra`` overrides
# forwarded as ``--extra <args>`` to ``run_amip.py``.
#
# The matrix covers all (grid_type, discretization) combinations that
# the legoESM dispatch matrix supports for hydrostatic AMIP runs:
#
#   cubed_sphere/centered, cubed_sphere/finite_volume, cubed_sphere/cdgrid
#   latlon/centered, latlon/finite_volume, latlon/latlon_cgrid
#   gaussian/spectral
#   voronoi/mpas (currently fails — pre-existing dycore stability issue)
_GRID_RESOLUTIONS = {
    "cubed_sphere":     dict(grid_type="cubed_sphere",   resolution=12,
                              discretization="centered", extra=[]),
    "cubed_sphere_fv":  dict(grid_type="cubed_sphere",   resolution=12,
                              discretization="finite_volume", extra=[]),
    "cubed_sphere_cd":  dict(grid_type="cubed_sphere",   resolution=12,
                              discretization="cdgrid",   extra=[]),
    "latlon":           dict(grid_type="latlon",         resolution=24,
                              discretization="centered", extra=[]),
    "latlon_fv":        dict(grid_type="latlon",         resolution=24,
                              discretization="finite_volume", extra=[]),
    "latlon_cgrid":     dict(grid_type="latlon",         resolution=24,
                              discretization="latlon_cgrid", extra=[]),
    "gaussian":         dict(grid_type="gaussian",       resolution=21,
                              discretization="spectral", extra=[]),
    # Voronoi SCVT: ``resolution`` is the bisection level
    # (level=4 → 2562 cells, similar size to C24).
    #
    # MPAS-specific overrides:
    # - ``--turbulence none``: the TKE bridge isn't implemented (TKE
    #   expects cell-centered winds; MPAS stores edge-normal winds).
    #
    # The historical ``--dt 60`` workaround ("hidden CFL constraint",
    # AMIP.md Known issues #2) is RESOLVED 2026-06-10: the instability
    # was the run_amip global ``--time-integrator ssp_rk3`` default
    # silently overriding the MPAS dycore's ssp_rk54_scan default —
    # the biharmonic hyperdiffusion eigenvalues at dt=600 fall outside
    # ssp_rk3's stability region (primitive_eq_mpas.py stability
    # notes).  ``component_factory`` now maps the no-choice default to
    # the MPAS dycore default, so the standard dt=600 holds.
    "voronoi":          dict(grid_type="voronoi",        resolution=4,
                              discretization="mpas",
                              # --turbulence none trips the full-physics
                              # guard, so this smoke case must opt out
                              # explicitly (it is a dycore+forcing smoke,
                              # not a realism run).
                              extra=["--turbulence", "none",
                                     "--allow-disabled-physics"]),
}


_FORCING_START_YEAR = 1979
_FORCING_END_YEAR = 1980


def run_one(label: str, *, days: int, resolution: int | None,
            forcing_dir: Path,
            timeout: int = 360) -> tuple[bool, str]:
    """Execute one (grid, discretization) case and return (ok, summary).

    Parameters
    ----------
    forcing_dir : Path
        Where the AMIP CMIP6 forcing deck lives.  The orchestrator
        (``main()``) generates this once under a fresh temp dir and
        passes the same directory to every case so we don't pay the
        ~5 s generator cost per case.  Each case still gets its own
        ``--output`` temp dir for run artefacts.
    """
    info = _GRID_RESOLUTIONS[label]
    grid_type = info["grid_type"]
    res = resolution if resolution is not None else info["resolution"]
    disc = info["discretization"]
    case_extra = list(info.get("extra", []))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run" / "run_amip_smoke_deck.py"),
            "--forcing-dir", str(forcing_dir),
            "--start-year", str(_FORCING_START_YEAR),
            "--end-year", str(_FORCING_END_YEAR),
            "--grid-type", grid_type,
            "--discretization", disc,
            "--resolution", str(res),
            "--days", str(days),
            "--diag-days", "1",
            "--radiation", "gray",
            # Dycore + forcing smoke (not a realism run): use the cheap
            # uniform-IC so no ERA5 data is required.  The deck default
            # is now --ic era5 for production realism.
            "--ic", "default",
            "--no-aerosol", "--no-volcanic",
            "--output", str(out),
        ]
        if case_extra:
            cmd += ["--extra", *case_extra]
        env = os.environ.copy()
        env.setdefault("JAX_ENABLE_X64", "1")
        try:
            r = subprocess.run(cmd, env=env, timeout=timeout,
                                capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, f"{label} ({grid_type}/{disc}/n={res}): TIMEOUT"

        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-5:])
            return False, (f"{label} ({grid_type}/{disc}/n={res}): "
                            f"exit={r.returncode}\n  {tail}")

        # Validate output
        valid_cmd = [sys.executable,
                     str(_REPO_ROOT / "scripts" / "validate" / "validate_amip_run.py"),
                     str(out)]
        v = subprocess.run(valid_cmd, capture_output=True, text=True)
        if v.returncode != 0:
            return False, (f"{label} ({grid_type}/{disc}/n={res}): "
                            f"validation FAILED:\n"
                            f"{v.stdout.splitlines()[-3:]}")
        # Read results.txt for summary
        try:
            text = (out / "results.txt").read_text()
            t_line = next((l for l in text.splitlines()
                            if l.startswith("Final <T_atm>:")), "")
            p_line = next((l for l in text.splitlines()
                            if l.startswith("Final <Precip>:")), "")
        except FileNotFoundError:
            t_line, p_line = "", ""
        return True, f"{label} ({grid_type}/{disc}/n={res}): OK  {t_line}  {p_line}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Override resolution for all grids (else "
                             "per-grid defaults)")
    parser.add_argument("--cases", "--grids", nargs="*",
                        dest="cases",
                        choices=list(_GRID_RESOLUTIONS.keys()),
                        default=list(_GRID_RESOLUTIONS.keys()),
                        help="Case label(s); default = all")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--forcing-dir", type=Path, default=None,
                        help="Pre-existing forcing deck directory.  When "
                             "omitted the script auto-generates a 1-year "
                             "deck in a temp dir so the smoke test is "
                             "self-contained on a clean checkout where "
                             "``data/forcing_amip/`` (gitignored) is absent.")
    args = parser.parse_args(argv)

    print(f"=== AMIP CMIP6 deck smoke test ({args.days}-day runs) ===")
    print(f"Cases: {args.cases}")
    print()

    # Auto-generate the forcing deck once (under tmp_path) when no path
    # was supplied.  We share the same forcing dir across all cases so
    # the synthetic-deck generator only runs once per smoke-test
    # invocation.
    forcing_owns_tempdir = False
    if args.forcing_dir is None:
        forcing_tempdir = tempfile.mkdtemp(prefix="amip_smoke_forcing_")
        forcing_dir = Path(forcing_tempdir)
        forcing_owns_tempdir = True
        print(f"[smoke] Auto-generating forcing deck under {forcing_dir} …",
              flush=True)
        gen_cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "data" / "generate_amip_forcing.py"),
            "--out", str(forcing_dir),
            # Single-year deck — minimal coverage to keep the smoke
            # test fast.  Year window matches what we pass to the deck
            # driver via ``--start-year`` / ``--end-year`` so the
            # generated filenames line up with ``_check_forcing_files``.
            "--start-year", str(_FORCING_START_YEAR),
            "--end-year", str(_FORCING_END_YEAR),
            # Coarse synthetic SST grid — interpolation tests live in
            # the unit-test module, not here.
            "--nlat-sst", "37",
            "--nlon-sst", "72",
        ]
        gen_rc = subprocess.run(gen_cmd, capture_output=True, text=True)
        if gen_rc.returncode != 0:
            print(f"[smoke] FATAL: forcing generation failed: "
                  f"{gen_rc.stderr.strip()[-500:]}")
            return 2
    else:
        forcing_dir = Path(args.forcing_dir).resolve()

    try:
        results: list[tuple[str, bool, str]] = []
        for case in args.cases:
            print(f"Running {case} …", flush=True)
            ok, summary = run_one(case, days=args.days,
                                    resolution=args.resolution,
                                    forcing_dir=forcing_dir,
                                    timeout=args.timeout)
            print(f"  {summary}", flush=True)
            results.append((case, ok, summary))

        n_pass = sum(1 for _, ok, _ in results if ok)
        n_fail = len(results) - n_pass
        print()
        print(f"Summary: {n_pass}/{len(results)} cases passed")
        for case, ok, summary in results:
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {case}: {summary.splitlines()[0]}")

        return 0 if n_fail == 0 else 1
    finally:
        if forcing_owns_tempdir:
            import shutil
            shutil.rmtree(forcing_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
