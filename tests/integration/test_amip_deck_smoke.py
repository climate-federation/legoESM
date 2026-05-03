"""Integration smoke test for the AMIP CMIP6 deck driver.

This is a *very* short run (1-day, C12 cubed-sphere, gray radiation,
no aerosol/volcanic) that exercises the full
``run_amip_cmip6_deck.py`` → ``run_amip.py`` → ``ModelDriver`` chain.
It validates that:

- The forcing-file generator + deck driver compose correctly.
- The auto-generated forcing deck is consumed by the model loaders
  without raising.
- The resulting run produces a finite final state passing the
  ``validate_amip_run.py`` post-run checks.

The test is **skipped** in the default fast unit-test sweep (it takes
~30 s including JIT) — gate on the ``LEGOESM_RUN_AMIP_INTEGRATION``
environment variable so CI can flip it on for nightly runs without
slowing the per-PR loop.

To run::

    LEGOESM_RUN_AMIP_INTEGRATION=1 \\
        python -m pytest tests/integration/test_amip_deck_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECK_DRIVER = _REPO_ROOT / "scripts" / "run_amip_cmip6_deck.py"
_VALIDATOR = _REPO_ROOT / "scripts" / "validate_amip_run.py"


pytestmark = pytest.mark.skipif(
    os.environ.get("LEGOESM_RUN_AMIP_INTEGRATION") != "1",
    reason="Slow integration test; set LEGOESM_RUN_AMIP_INTEGRATION=1 to run.",
)


def _run_deck(out_dir: Path, *, grid_type: str = "cubed_sphere",
              discretization: str = "centered",
              resolution: int = 12,
              extra: list[str] | None = None,
              timeout: int = 240) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(_DECK_DRIVER),
        "--grid-type", grid_type,
        "--discretization", discretization,
        "--resolution", str(resolution),
        "--days", "1",
        "--diag-days", "1",
        "--radiation", "gray",
        "--no-aerosol", "--no-volcanic",
        "--output", str(out_dir),
    ]
    if extra:
        cmd += ["--extra", *extra]
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    return subprocess.run(cmd, env=env, timeout=timeout,
                            capture_output=True, text=True)


def _validate(out_dir: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_VALIDATOR), str(out_dir)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_deck_runs_and_validates_cubed_sphere(tmp_path):
    out = tmp_path / "amip_run"
    r = _run_deck(out)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stdout (tail) ---\n{r.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    assert (out / "results.txt").exists(), "results.txt was not written"
    assert (out / "timeseries.npz").exists(), "timeseries.npz was not written"

    v = _validate(out)
    assert v.returncode == 0, (
        f"validator FAILED on otherwise-OK deck run:\n"
        f"--- stdout ---\n{v.stdout}\n"
    )
    assert "ALL CHECKS PASSED" in v.stdout


def test_deck_runs_and_validates_latlon(tmp_path):
    out = tmp_path / "amip_run"
    r = _run_deck(out, grid_type="latlon", discretization="centered",
                   resolution=24)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_runs_and_validates_gaussian_spectral(tmp_path):
    out = tmp_path / "amip_run"
    r = _run_deck(out, grid_type="gaussian", discretization="spectral",
                   resolution=21)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_runs_and_validates_voronoi_mpas(tmp_path):
    """Voronoi/MPAS needs a 10× smaller dt than CFL would suggest
    (Known issue #2 — hidden stability constraint).  Disable
    turbulence (no edge→cell wind interpolation in the TKE bridge)
    and pass --dt 60 explicitly."""
    out = tmp_path / "amip_run"
    r = _run_deck(
        out, grid_type="voronoi", discretization="mpas", resolution=4,
        extra=["--turbulence", "none", "--dt", "60"],
        timeout=480,
    )
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_runs_and_validates_latlon_cgrid(tmp_path):
    """Lat-lon C-grid uses ``--discretization latlon_cgrid`` after
    the iter-7 cgrid alias fix."""
    out = tmp_path / "amip_run"
    r = _run_deck(out, grid_type="latlon", discretization="latlon_cgrid",
                   resolution=24)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout
