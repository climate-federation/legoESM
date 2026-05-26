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


def _run_driver_production_scale(output_dir, *, n_outer_steps=60,
                                 log_every_steps=15):
    """Invoke run_rce_mpi_long.py at the iter-14 production scale
    (132x132x30 dx=2km dt=5s) for ``n_outer_steps`` outer steps.

    Defaults reproduce the iter-14/iter-15 measurement that locked
    in the plane CRM production envelope: clean Wing IC (no bubble,
    no qv noise), hyperdiff=5e6, Smag c_s=0.2 (passed explicitly so
    a future driver default change can't silently shift this
    regression), SI acoustic with off-centering=0.1 + 12 substeps,
    mass fixer on (driver default), radiation disabled to isolate
    dycore behaviour.

    ``--days`` carries +0.5*dt padding so ``int(total_t / dt)`` in
    the driver always lands at ``n_outer_steps`` exactly (without
    padding, float-roundoff truncates 60 → 59 silently, leaving
    ``rows[-1]`` on step 40 instead of step 60 when
    ``log_every_steps=20``).  ``log_every_steps`` defaults to 15 so
    steps 1, 15, 30, 45, 60 all log — transient spikes between
    log_every=20 rows would otherwise hide a CFL crash that
    self-recovers in 19 steps.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    days = (n_outer_steps + 0.5) * 5.0 / 86400.0
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "132", "--ny", "132", "--nlev", "30",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", f"{days:.8f}",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--smag-cs", "0.2",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", str(log_every_steps),
        "--n-physics-substeps", "1",
        "--rad-call-interval-s", "1e9",
        "--output", str(output_dir),
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=900,
    )


@pytest.mark.slow
def test_plane_crm_production_scale_132x132_envelope(tmp_path):
    """Nightly slow regression for the plane CRM 132x132 production
    config (5-min-sim sub-envelope of the iter-14 1-sim-hour smoke).

    iter-14 measured (725 steps = 1 sim-hour, 132x132x30 dx=2km dt=5s,
    clean Wing IC, mass fixer + Smag c_s=0.2 + SI acoustic + hyperdiff
    5e6, no radiation):

    | step | CWV [mm] | MSE [J/kg] | max|w| [m/s] |
    |------|----------|------------|--------------|
    |   1  |  55.550  |  4.2132e9  |  0.0e+00     |
    | 100  |  55.550  |  4.2131e9  |  3.9e-3      |
    | 300  |  55.550  |  4.2129e9  |  5.5e-3      |
    | 700  |  55.550  |  4.2125e9  |  6.1e-3      |

    Per-step MSE drift in iter-14 ≈ 2.3e-7 relative; this test runs
    60 outer steps so expected MSE drift ≈ 1.4e-5 (well below the 5e-5
    cap below). max|w| at step 60 is interpolated from iter-14 as
    ~2e-3 m/s (the actual logged value at first run was 2.08e-3).

    This is a strict SUB-envelope of the iter-14 measurement — the
    asserts below are tighter than the iter-14 ceiling because 60
    steps is only 8% of the iter-14 window. Catches any regression
    that silently destabilises the production-scale config (e.g. a
    future halo / Smag / hyperdiff / mass-fixer change that the 12x12
    smoke misses because the unstable mode is grid-scale).

    Wall budget ~2.5 min on M5 Pro (1.35 s/step × 60 + JIT compile).
    Marked ``slow``; runs via ``pytest -m slow``. Default skips it.
    """
    n_steps = 60
    log_every = 15
    out_dir = tmp_path / "rce_plane_prod"
    result = _run_driver_production_scale(
        out_dir, n_outer_steps=n_steps, log_every_steps=log_every,
    )
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero at production scale "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows at production scale"

    # Schema + step-count sanity: log_every=15 + n=60 yields exact rows
    # at steps 1, 15, 30, 45, 60. A short run (driver miscount) would
    # leave the final row at an earlier step and all envelope asserts
    # would silently evaluate the wrong state.
    expected_logged_steps = [1, 15, 30, 45, 60]
    logged_steps = [int(r["step"]) for r in rows]
    assert logged_steps == expected_logged_steps, (
        f"plane CRM production smoke: logged steps={logged_steps} "
        f"!= expected {expected_logged_steps}. Driver step counter or "
        f"log-every cadence regressed."
    )

    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    # iter-14 measured IC CWV = 55.550 mm at 132x132 Wing 2018.
    assert abs(cwv_first - 55.550) < 0.01, (
        f"plane CRM production smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 55.550 ± 0.01. iter-14 anchored this value at 132x132. "
        f"If intentional, update the test."
    )
    # max|w| envelope on EVERY logged row, not just the last — a
    # transient CFL spike that self-damps in <log_every steps would
    # otherwise pass silently. iter-14 saw monotone w growth up to
    # 6.1e-3 at step 700; at step 60 actual was 2.08e-3. Cap at
    # 0.05 m/s is 24x over that — catches any genuine in-flight CFL
    # crash without false positives from spinup noise.
    max_w_per_row = [float(r["max|w|"]) for r in rows]
    for step_idx, mw in zip(logged_steps, max_w_per_row):
        assert mw < 0.05, (
            f"plane CRM production smoke: max|w|={mw:.3e} m/s at step "
            f"{step_idx} exceeds 0.05 m/s safety cap. iter-14 measured "
            f"~6e-3 m/s peak across step 1..700; this is a smoking gun "
            f"for an in-flight CFL crash or stabilizer regression."
        )
    # Activity floor: by step 15 the SI acoustic substeps should have
    # produced non-zero max|w|. A dead/no-op simulation (e.g. step
    # function elided by a future refactor, frozen state pytree, JIT
    # eating the loop body) would pass every envelope above. Anchor
    # at step 15 because step 1 is logged BEFORE any time advance.
    step15_max_w = max_w_per_row[1]
    assert step15_max_w > 1e-6, (
        f"plane CRM production smoke: max|w|={step15_max_w:.3e} m/s "
        f"at step 15 — the dycore appears inactive. iter-14 saw "
        f"~2e-3 m/s by step 20 from baseline thermo gradient + SI "
        f"acoustic. <1e-6 means step() is a no-op."
    )
    # CWV pinned in iter-14 within rounding; 0.01 mm drift cap.
    cwv_drift = abs(cwv_final - cwv_first)
    assert cwv_drift < 0.01, (
        f"plane CRM production smoke: CWV drifted {cwv_drift:.4f} mm "
        f"(IC={cwv_first:.4f}, final={cwv_final:.4f}) in {n_steps} "
        f"steps. iter-14 saw zero drift at 132x132 over 700 steps. "
        f"Possibly the moist-mass fixer regressed."
    )
    # iter-14 saw 1.7e-4 relative drift over 725 steps (per-step rate
    # ≈ 2.3e-7). For 60 steps expected ≈ 1.4e-5. Cap at 5e-5: catches
    # ~3x per-step rate regression but stays well clear of the
    # one-sim-hour iter-14 ceiling (1.7e-4) so this is a STRICT
    # sub-envelope, not a re-statement.
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 5e-5, (
        f"plane CRM production smoke: MSE drift {rel_mse_drift:.3e} "
        f"relative exceeds 5e-5 sub-envelope cap. iter-14 measured "
        f"1.7e-4 over 725 steps (per-step ≈ 2.3e-7); 60 steps should "
        f"land at ~1.4e-5. >5e-5 means per-step rate has tripled — "
        f"investigate energy budget."
    )
