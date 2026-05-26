"""Cross-grid RCE smoke at the iter-12 production-validated configs.

iter-12 (2026-05) measured 30-day RCE PASS on all 4 hydrostatic grid
families. These tests run a SHORTER (2-day) version of the same
configs as a CI-friendly regression so future commits don't quietly
break any of the grid×discretization paths.

Wall-time budget: 4 grids × ~10 s wall at 2 days = ~40 s total.

For the FULL 30-day production validation see
``scripts/run_rce_cross_grid.sh`` and CRM_implementation.md iter-12.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_RCE = REPO_ROOT / "scripts" / "run_rce.py"


def _run_rce(grid_type, discretization, resolution, days, output_dir):
    """Invoke scripts/run_rce.py with the iter-12-validated CLI."""
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(RUN_RCE),
        "--grid-type", grid_type,
        "--discretization", discretization,
        "--resolution", str(resolution),
        "--days", str(days),
        "--diag-days", "1",
        "--nlev", "20",
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600,
    )
    return result


def _parse_results(output_dir):
    """Read ``results.txt`` and return a dict of fields."""
    results_txt = output_dir / "results.txt"
    if not results_txt.exists():
        return None
    fields = {}
    for line in results_txt.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


@pytest.mark.parametrize("grid_type,discretization,resolution", [
    ("cubed_sphere", "cdgrid", 12),
    ("latlon", "latlon_cgrid", 16),
    ("voronoi", "mpas", 4),
    ("gaussian", "spectral", 21),
])
def test_rce_2day_smoke_passes(tmp_path, grid_type, discretization, resolution):
    """Per-grid 2-day RCE smoke must:
    1. exit cleanly (no crashes, no NaN).
    2. report status == PASS in results.txt.
    3. produce mean_T_sfc within 1 K of the IC (300 K).

    These bounds reflect the iter-12 30-day production measurements
    where all 4 grids land at mean_T_sfc within ±1 K of IC at equilibrium.
    """
    out_dir = tmp_path / f"{grid_type}_{resolution}"
    result = _run_rce(grid_type, discretization, resolution, days=2,
                      output_dir=out_dir)
    if result.returncode != 0:
        pytest.fail(
            f"{grid_type}/{discretization}@{resolution} exited nonzero "
            f"({result.returncode})\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    fields = _parse_results(out_dir)
    assert fields is not None, (
        f"{grid_type} did not produce results.txt — script failed silently?"
    )
    assert fields.get("status") == "PASS", (
        f"{grid_type}/{discretization}@{resolution}: "
        f"status={fields.get('status')} (want PASS). "
        f"notes={fields.get('notes')}"
    )
    notes = fields.get("notes", "")
    if "mean_T_sfc=" in notes:
        t_sfc_str = notes.split("mean_T_sfc=")[1].split(",")[0]
        t_sfc = float(t_sfc_str)
        assert abs(t_sfc - 300.0) < 1.0, (
            f"{grid_type}/{discretization}@{resolution}: "
            f"mean_T_sfc={t_sfc:.2f} drifted >1 K from IC 300 K — "
            f"slab-ocean coupling or radiation regression suspected. "
            f"See CRM_implementation.md iter-12 for expected values."
        )
