"""End-to-end smoke test for the plane CRM production stack.

Mirrors the hydrostatic ``test_rce_cross_grid_smoke.py`` regression
for the plane CRM (non-hydrostatic). Runs ``run_rce_mpi_long.py``
on a tiny 12x12x20 mesh for a short window with the F8/F10/iter-13
production-default config (clean Wing IC, no bubble, no qv noise,
hyperdiff=5e6, dt=5 s) and asserts:

1. The driver exits cleanly with finite diagnostics.
2. ``max|w|`` stays below 0.5 m/s (production envelope at F8 IC is
   ~6e-3 m/s; >0.5 m/s in a short smoke is a smoking gun for an
   in-flight numerical instability).
3. ``MSE`` drift stays below 1e-3 relative (production envelope is
   ~7e-5 over 28 min sim at 132x132; the same per-step drift on a
   12x12 mesh extrapolates well below 1e-3 over the smoke window).
4. ``CWV`` doesn't drift more than 0.1 mm from IC (55.55 mm) — the
   F8 IC is dynamically frozen until surface flux + radiation
   warm the column on hour-day timescales.

Wall budget: ~30 s on M5 Pro (rough estimate). Single-rank, no MPI,
no JAX JIT pre-compile.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rce_mpi_long.py"


def _run_driver(output_dir):
    """Invoke run_rce_mpi_long.py with the F8/F10 production defaults."""
    env = os.environ.copy()
    # FORCE JAX_PLATFORMS=cpu (override any exported value). iter-7
    # found that JAX_PLATFORMS=metal triggers MLIR legalisation
    # crashes on spectral / voronoi / latlon-cgrid paths; the same
    # backend can also break the plane CRM in subtle ways. Tests must
    # always run on CPU regardless of the developer's shell env.
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.005",        # ~7 min sim = 86 outer steps
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", "20",
        "--n-physics-substeps", "1",
        "--rad-call-interval-s", "1e9",  # disable radiation in smoke
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )
    return result


def _read_log(output_dir):
    """Read log.txt rows as a list of dicts."""
    log_path = Path(output_dir) / "log.txt"
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as fh:
        header = None
        for line in fh:
            line = line.strip()
            if line.startswith("# step,"):
                header = line.lstrip("# ").split(",")
            elif line.startswith("#") or not line:
                continue
            elif header is not None:
                parts = line.split(",")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
    return rows


def test_plane_crm_short_smoke_clean_ic(tmp_path):
    """Plane CRM with F8 clean IC + F10 dt=5 s + iter-13 hardened
    defaults must complete a 7-minute smoke window with all
    diagnostics in the production envelope.
    """
    out_dir = tmp_path / "rce_plane_smoke"
    result = _run_driver(out_dir)
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows"

    max_w_final = float(rows[-1]["max|w|"])
    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])

    # Anchor the IC CWV: the 12x12 Wing 2018 IC carries 55.001 mm at
    # this nlev/H. If a future commit silently shifts the Wing profile
    # coefficients, the drift assertion below would still pass against
    # the new IC and miss the regression — this gate makes the IC
    # itself part of the contract. Tolerance 0.01 mm (5e-4 relative)
    # is tight enough to detect any meaningful profile change but
    # loose enough for the area-weighted-integration roundoff.
    assert abs(cwv_first - 55.001) < 0.01, (
        f"plane CRM smoke: IC CWV={cwv_first:.4f} mm != 55.001 ± 0.01. "
        f"The Wing 2018 reference profile or its area weighting "
        f"changed — update this test's expected value if intentional, "
        f"otherwise diagnose the regression."
    )
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    assert max_w_final < 0.5, (
        f"plane CRM smoke: max|w|={max_w_final:.3e} m/s at end "
        f"exceeds 0.5 m/s production envelope (F8 measured ~6e-3 m/s "
        f"on 132x132). Check CRM_implementation.md F8/F10."
    )
    # CWV is measured against its OWN IC (not a hardcoded constant);
    # the small-grid IC carries a slightly different mean than the
    # 132x132 production grid (55.001 mm at 12x12 vs 55.550 mm at
    # 132x132 due to area-weighted integration of the Wing profile).
    cwv_drift = abs(cwv_final - cwv_first)
    assert cwv_drift < 0.1, (
        f"plane CRM smoke: CWV drifted {cwv_drift:.4f} mm in the "
        f"smoke window (IC={cwv_first:.4f}, final={cwv_final:.4f}). "
        f"F8 says CWV stays pinned at IC until convection spins up "
        f"on hour-day timescale; >0.1 mm drift in 86 outer steps "
        f"means radiation is unexpectedly active or the moist-mass "
        f"fixer regressed."
    )
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 1e-3, (
        f"plane CRM smoke: MSE drift {rel_mse_drift:.3e} relative "
        f"exceeds 1e-3 cap. F8/F10 production measurements: 7e-5 "
        f"relative over 28 min sim at 132x132."
    )


def _run_driver_with_radiation(output_dir):
    """Variant of _run_driver that ENABLES gray radiation. Catches a
    regression where radiation is broken in a way the dycore-only
    smoke would miss (e.g. NaN in the column-water-vapor reduction
    used by gray rad, or a crash in the first radiation tendency
    application)."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.002",       # ~3 min sim = ~35 outer steps
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", "10",
        "--n-physics-substeps", "1",
        "--rad-call-interval-s", "30.0",  # fire radiation every 30s
        "--output", str(output_dir),
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )


def test_plane_crm_short_smoke_with_radiation(tmp_path):
    """Plane CRM with gray radiation actively called every 30 s sim
    time. Codex iter-16 gap: the main dycore-only smoke disables
    radiation via --rad-call-interval-s=1e9, so a radiation
    regression (NaN in CWV reduction, broken first tendency
    application, etc.) would slip through CI.

    This shorter smoke exercises the radiation path with a tight
    call interval. Only asserts the driver exits cleanly + ``max|w|``
    stays bounded — radiation can drive larger drift than the
    dycore-only case over the same window, so the CWV/MSE caps
    above are not appropriate here."""
    out_dir = tmp_path / "rce_plane_smoke_rad"
    result = _run_driver_with_radiation(out_dir)
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero with radiation enabled "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows with radiation on"
    max_w_final = float(rows[-1]["max|w|"])
    assert max_w_final < 0.5, (
        f"plane CRM smoke with radiation: max|w|={max_w_final:.3e} "
        f"m/s at end exceeds 0.5 m/s production envelope. Possibly a "
        f"radiation tendency injecting too much energy too quickly — "
        f"check make_radiation_physics output."
    )
