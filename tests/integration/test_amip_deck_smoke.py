"""Integration smoke test for the AMIP CMIP6 deck driver.

This is a *very* short run (1-day, C12 cubed-sphere, gray radiation,
no aerosol/volcanic) that exercises the full
``run_amip_smoke_deck.py`` → ``run_amip.py`` → ``ModelDriver`` chain.
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

The test is self-contained: it generates the forcing deck under
``tmp_path / "forcing"`` and points the driver at it via
``--forcing-dir`` and ``--auto-generate``.  This makes the test work
on a clean checkout where ``data/forcing_amip/`` (gitignored) does not
exist, and avoids contaminating any pre-existing forcing dataset on
the developer's machine.

To run::

    LEGOESM_RUN_AMIP_INTEGRATION=1 \\
        python -m pytest tests/integration/test_amip_deck_smoke.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECK_DRIVER = _REPO_ROOT / "scripts" / "run" / "run_amip_smoke_deck.py"
_VALIDATOR = _REPO_ROOT / "scripts" / "validate" / "validate_amip_run.py"


pytestmark = pytest.mark.skipif(
    os.environ.get("LEGOESM_RUN_AMIP_INTEGRATION") != "1",
    reason="Slow integration test; set LEGOESM_RUN_AMIP_INTEGRATION=1 to run.",
)


def _run_deck(out_dir: Path, *, forcing_dir: Path,
              grid_type: str = "cubed_sphere",
              discretization: str = "finite_volume",
              resolution: int = 12,
              radiation: str = "gray",
              days: int = 1,
              ic: str = "default",
              extra: list[str] | None = None,
              timeout: int = 240) -> subprocess.CompletedProcess:
    """Drive ``run_amip_smoke_deck.py`` with a self-contained
    auto-generated forcing deck under ``forcing_dir``.

    ``radiation`` selects the radiation backend; ``gray`` keeps the run
    fast (dycore + SST-forcing path only), while ``rrtmg`` exercises the
    full CMIP6 physics stack (correlated-k radiation consuming the
    transient GHG / ozone / spectral-solar forcing channels).
    """
    cmd = [
        sys.executable, str(_DECK_DRIVER),
        "--forcing-dir", str(forcing_dir),
        "--auto-generate",
        "--grid-type", grid_type,
        "--discretization", discretization,
        "--resolution", str(resolution),
        "--days", str(days),
        "--diag-days", "1",
        "--radiation", radiation,
        "--ic", ic,
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
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing)
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
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="latlon", discretization="centered",
                   resolution=24)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_runs_and_validates_gaussian_spectral(tmp_path):
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="gaussian", discretization="spectral",
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
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(
        out, forcing_dir=forcing,
        grid_type="voronoi", discretization="mpas", resolution=4,
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
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="latlon", discretization="latlon_cgrid",
                   resolution=24)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_runs_and_validates_latlon_finite_volume(tmp_path):
    """Lat-lon **finite-volume** AMIP path: ``--discretization
    finite_volume`` is the canonical CLI name for the lat-lon C-grid
    primitive-equation dycore (``latlon_cgrid_primitive_equations``).
    This pins the exact grid+discretization combination users select for
    a CMIP6 AMIP run on the lat-lon FV grid."""
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="latlon", discretization="finite_volume",
                   resolution=24)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout


def test_deck_latlon_fv_standard_ic_realistic_cwv(tmp_path):
    """The CMIP6 deck with ``--ic standard`` on lat-lon finite-volume must run,
    validate, and start from an Earth-like column water vapour (~10-30 kg/m^2),
    not the uniform-300 K default's ~84 kg/m^2."""
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="latlon", discretization="finite_volume",
                   resolution=24, ic="standard", days=2)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout
    # Earth-like CWV (the realism payoff of ic=standard).
    text = (out / "results.txt").read_text()
    cwv_line = next(l for l in text.splitlines() if "Final <CWV>" in l)
    cwv = float(cwv_line.split(":")[1].strip().split()[0])
    assert 5.0 < cwv < 40.0, f"ic=standard CWV {cwv} not Earth-like"


def test_deck_runs_and_validates_latlon_finite_volume_rrtmg(tmp_path):
    """End-to-end CMIP6 AMIP physics on the lat-lon finite-volume grid:
    correlated-k RRTMG radiation consuming the transient GHG / ozone /
    spectral-solar forcing channels (gray radiation leaves those inert).

    Heavier than the gray-radiation cases — the RRTMG correlated-k graph
    carries a multi-minute JIT compile that dominates wall time (the
    integration itself is ~1 s), so it lives behind the same nightly gate
    with an enlarged timeout.  Resolution is held at 12 (~280 s end to
    end here); larger grids inflate the dynamics-scan compile past the
    timeout without exercising any additional code path."""
    forcing = tmp_path / "forcing"
    out = tmp_path / "amip_run"
    r = _run_deck(out, forcing_dir=forcing,
                   grid_type="latlon", discretization="finite_volume",
                   resolution=12, radiation="rrtmg", timeout=900)
    assert r.returncode == 0, (
        f"deck driver failed (exit={r.returncode}):\n"
        f"--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )
    v = _validate(out)
    assert v.returncode == 0, v.stdout
