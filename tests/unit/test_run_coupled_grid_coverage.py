"""Grid coverage for the coupled driver (B1): all four atmosphere grids are
selectable; coupled slab/ocean runs on cubed_sphere/latlon/voronoi; gaussian
(spectral) coupled is gated (spectral state is coefficients — the atm->ocean
forcing extractor needs a full grid synthesis, a follow-up)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run.run_coupled import build_parser

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "run" / "run_coupled.py"


@pytest.mark.parametrize("grid", ["cubed_sphere", "latlon", "voronoi", "gaussian"])
def test_grid_choice_accepted(grid):
    """All four grids parse (the --grid choices were widened for coupled runs)."""
    args = build_parser().parse_args(["--grid", grid])
    assert args.grid == grid


def test_gaussian_coupled_is_gated():
    """Coupled --grid gaussian exits cleanly with a clear message (not a crash
    deep in the spectral forcing extractor)."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--preset", "full_coupled",
         "--grid", "gaussian", "--ocean", "slab", "--resolution", "8",
         "--days", "1"],
        env=env, capture_output=True, text=True, timeout=180)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "coupled gaussian should be gated, not run"
    assert "gaussian" in combined and "not wired" in combined, \
        f"missing the clear gate message.\n{combined[-600:]}"
    assert "Traceback" not in combined, \
        f"gate should be a clean SystemExit, not a traceback.\n{combined[-800:]}"
