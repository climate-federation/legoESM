"""Regression for the coupled-voronoi (MPAS) surface radiative-flux export.

The lean ``_run_column`` loop advances the atmosphere with an operator-split
physics_fn that returns only a tendency, so the surface net radiative fluxes it
computes were discarded — ``_build_atm_forcing`` then read ``held_sw_net_sfc``
from ``_carry_aux`` where it defaulted to zeros, forcing the coupled ocean/land
tiles with ZERO shortwave (the #1202 coupled-voronoi gap).  The fix threads the
surface net fluxes out of the MPAS step (``HydrostaticTendencies.sw_net_sfc`` ->
``model._sfc_diag``) and stashes them into ``_carry_aux`` before the coupling
callback, exactly as the compiled cube/latlon path exports its ``PhysicsOutput``.

This runs a short coupled voronoi case and asserts the exported surface SW is
physically nonzero (a globe under gray radiation nets ~O(100) W/m^2), not the
pre-fix zero.  Subprocess (the driver's real ``_run_column`` path); ~30 s.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_coupled.py"


def test_coupled_voronoi_exports_nonzero_surface_sw(tmp_path):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid", "voronoi", "-n", "3",
        "--ocean", "slab", "--radiation", "gray",
        "--days", "2", "--diag-days", "1",
        "--output", str(tmp_path / "coupled_voronoi_swexport"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=600)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"coupled voronoi run failed: exit {result.returncode}\n"
        f"stdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-1500:]}")
    # The one-time export diagnostic must appear and report a nonzero mean SW.
    m = re.search(
        r"surface radiative forcing.*sw_net_sfc mean=([-\d.]+)", combined)
    assert m is not None, (
        "missing the coupled surface-flux export diagnostic — the MPAS loop "
        f"did not stash held_sw_net_sfc.\nstdout: {result.stdout[-1500:]}")
    mean_sw = float(m.group(1))
    assert mean_sw > 10.0, (
        f"exported surface net SW mean={mean_sw} W/m^2 is ~zero: the coupled "
        "ocean/land tiles are still radiatively unforced (the #1202 bug).")
    assert "Traceback" not in combined and "nan" not in result.stdout.lower(), \
        f"coupled voronoi run produced a traceback / NaN.\n{result.stdout[-1500:]}"
