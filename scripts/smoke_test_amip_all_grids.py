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

_REPO_ROOT = Path(__file__).resolve().parents[1]


# Resolution per grid type (chosen so each grid has a comparable cell count).
# ``extra`` lists per-grid overrides forwarded to ``run_amip.py``.
_GRID_RESOLUTIONS = {
    "cubed_sphere": dict(resolution=12, discretization="centered", extra=[]),
    "gaussian":     dict(resolution=21, discretization="spectral", extra=[]),  # T21
    "latlon":       dict(resolution=24, discretization="centered", extra=[]),
    # Voronoi SCVT: ``resolution`` is the bisection level
    # (level=4 → 2562 cells, similar size to C24).
    # The MPAS turbulence bridge isn't implemented (the integration
    # raises ``NotImplementedError`` because TKE expects cell-centered
    # winds while MPAS stores edge-normal winds), so disable it for
    # the smoke test.  See ``atmosphere/physics/turbulence/integration.py``.
    "voronoi":      dict(resolution=4, discretization="mpas",
                          extra=["--turbulence", "none"]),
}


def run_one(grid: str, *, days: int, resolution: int | None,
            timeout: int = 360) -> tuple[bool, str]:
    """Execute one grid case and return (ok, summary)."""
    info = _GRID_RESOLUTIONS[grid]
    res = resolution if resolution is not None else info["resolution"]
    disc = info["discretization"]
    grid_extra = list(info.get("extra", []))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_amip_cmip6_deck.py"),
            "--grid-type", grid,
            "--discretization", disc,
            "--resolution", str(res),
            "--days", str(days),
            "--diag-days", "1",
            "--radiation", "gray",
            "--no-aerosol", "--no-volcanic",
            "--output", str(out),
        ]
        if grid_extra:
            cmd += ["--extra", *grid_extra]
        env = os.environ.copy()
        env.setdefault("JAX_ENABLE_X64", "1")
        try:
            r = subprocess.run(cmd, env=env, timeout=timeout,
                                capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, f"{grid}/{disc}/n={res}: TIMEOUT"

        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-5:])
            return False, f"{grid}/{disc}/n={res}: exit={r.returncode}\n  {tail}"

        # Validate output
        valid_cmd = [sys.executable,
                     str(_REPO_ROOT / "scripts" / "validate_amip_run.py"),
                     str(out)]
        v = subprocess.run(valid_cmd, capture_output=True, text=True)
        if v.returncode != 0:
            return False, (f"{grid}/{disc}/n={res}: validation FAILED:\n"
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
        return True, f"{grid}/{disc}/n={res}: OK  {t_line}  {p_line}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Override resolution for all grids (else "
                             "per-grid defaults)")
    parser.add_argument("--grids", nargs="*",
                        choices=list(_GRID_RESOLUTIONS.keys()),
                        default=list(_GRID_RESOLUTIONS.keys()))
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args(argv)

    print(f"=== AMIP CMIP6 deck smoke test ({args.days}-day runs) ===")
    print(f"Grids: {args.grids}")
    print()

    results: list[tuple[str, bool, str]] = []
    for grid in args.grids:
        print(f"Running {grid} …", flush=True)
        ok, summary = run_one(grid, days=args.days,
                                resolution=args.resolution,
                                timeout=args.timeout)
        print(f"  {summary}", flush=True)
        results.append((grid, ok, summary))

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print()
    print(f"Summary: {n_pass}/{len(results)} grids passed")
    for grid, ok, summary in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {grid}: {summary.splitlines()[0]}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
