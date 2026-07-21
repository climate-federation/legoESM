"""Grid coverage for the coupled driver (B1): all four atmosphere grids are
selectable; coupled slab/ocean runs on cubed_sphere/latlon/voronoi AND gaussian
(spectral) — the spectral state is synthesized to grid in _build_atm_forcing and
_run_spectral recomputes/stashes the surface radiation at the coupling boundary
(A2). A 3-D dynamic ocean on gaussian stays gated (spectral ocean is idealized)."""
from __future__ import annotations

import os
import re
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


def test_gaussian_dynamic_ocean_coupled_is_gated():
    """Coupled --grid gaussian with a 3-D DYNAMIC ocean is gated cleanly (a
    spectral dynamic ocean is idealized / not wired); the slab is supported."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--preset", "full_coupled",
         "--grid", "gaussian", "--ocean", "dynamic", "--resolution", "8",
         "--days", "1"],
        env=env, capture_output=True, text=True, timeout=180)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "coupled gaussian+dynamic should be gated"
    assert "gaussian" in combined and "slab ocean only" in combined, \
        f"missing the clear gate message.\n{combined[-600:]}"
    assert "Traceback" not in combined, \
        f"gate should be a clean SystemExit, not a traceback.\n{combined[-800:]}"


def test_gaussian_slab_coupled_runs(tmp_path):
    """Coupled --grid gaussian with a slab ocean RUNS end-to-end (A2): the
    spectral atm state is synthesized to grid in _build_atm_forcing, and
    _run_spectral recomputes + stashes the surface net radiation at the daily
    coupling boundary so the ocean is radiatively forced (not the zero-SW
    #1202-class bug). The equator-to-pole SST structure that develops is the
    tell-tale of nonzero insolation-driven forcing."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--grid", "gaussian", "--ocean", "slab",
         "--resolution", "16", "--days", "1", "--radiation", "gray",
         "--output", str(tmp_path / "coupled_gauss")],
        env=env, capture_output=True, text=True, timeout=600)
    combined = result.stdout + result.stderr
    assert result.returncode == 0 and "COMPLETED" in combined, \
        f"coupled gaussian slab run did not complete.\n{combined[-1200:]}"
    assert "Traceback" not in combined and "nan" not in result.stdout.lower(), \
        f"coupled gaussian produced a traceback / NaN.\n{combined[-1200:]}"
    # Nonzero radiative forcing => the ocean SST develops an equator-to-pole
    # spread (a zero-SW-forced slab would stay flat at its init).
    m = re.search(r"SST final \(ocean\): mean=[\d.]+K, range=\[([\d.]+), "
                  r"([\d.]+)\]K", combined)
    assert m, f"no SST range in output.\n{combined[-800:]}"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert hi - lo > 5.0, (
        f"SST spread {hi - lo:.1f}K too small — ocean under-forced (zero-SW "
        f"regression?)")
