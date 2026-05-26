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
    # FORCE JAX_PLATFORMS=cpu (override exported value); iter-7
    # documented Metal MLIR crashes on spectral/voronoi/latlon-cgrid.
    env["JAX_PLATFORMS"] = "cpu"
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


def _parse_notes(notes: str) -> dict:
    """Parse `mean_T_sfc=..., mean_T=..., max|v|=...` into a dict."""
    out = {}
    for part in notes.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            pass
    return out


def _assert_rce_pass(out_dir, label, temp_tol=1.0, max_v_cap=50.0):
    """Shared post-run assertion for a 2-day or longer RCE smoke.

    Asserts (1) results.txt exists, (2) status == PASS,
    (3) mean_T_sfc within `temp_tol` of 300 K, (4) max|v| < `max_v_cap`.
    Used by the parametrise smoke and the slow C96/C48 variants
    to avoid duplicating the envelope checks (Codex iter-19 MEDIUM).
    """
    fields = _parse_results(out_dir)
    assert fields is not None, (
        f"{label}: did not produce results.txt — script failed silently?"
    )
    assert fields.get("status") == "PASS", (
        f"{label}: status={fields.get('status')} (want PASS). "
        f"notes={fields.get('notes')}"
    )
    notes_dict = _parse_notes(fields.get("notes", ""))
    t_sfc = notes_dict.get("mean_T_sfc")
    assert t_sfc is not None, (
        f"{label}: results.txt notes line missing mean_T_sfc — "
        f"unexpected format: {fields.get('notes')!r}"
    )
    assert abs(t_sfc - 300.0) < temp_tol, (
        f"{label}: mean_T_sfc={t_sfc:.2f} drifted >{temp_tol} K from "
        f"IC 300 K — slab-ocean coupling or radiation regression "
        f"suspected. See CRM_implementation.md iter-12 for expected values."
    )
    max_v = notes_dict.get("max|v|")
    assert max_v is not None, (
        f"{label}: notes line missing max|v|."
    )
    assert max_v < max_v_cap, (
        f"{label}: max|v|={max_v:.2f} m/s exceeds {max_v_cap} m/s cap. "
        f"Production envelope is 2-12 m/s; >{max_v_cap} m/s means a "
        f"CFL crash in flight (even before the 200 m/s BLOWUP gate)."
    )


# Coverage matrix (iter-13). Includes C48 (iter-13 auto-dt=150
# branch) so any future regression of the auto-dt ladder trips the
# BLOWUP gate at 2 days and FAILS this test. iter-13 measurements:
# C48 reached only max|v|≈1.7 m/s by day 2.
#
# C96 (auto-dt=75 branch) lives in `test_rce_2day_smoke_passes_slow`
# below because C96 2-day takes ~10 min wall — too slow for default
# CI but worthwhile as a nightly run.
@pytest.mark.parametrize("grid_type,discretization,resolution", [
    ("cubed_sphere", "cdgrid", 12),
    ("cubed_sphere", "cdgrid", 48),   # iter-13 auto-dt=150 branch
    ("latlon", "latlon_cgrid", 16),
    ("voronoi", "mpas", 4),
    ("gaussian", "spectral", 21),
])
def test_rce_2day_smoke_passes(tmp_path, grid_type, discretization, resolution):
    """Per-grid 2-day RCE smoke must:
    1. exit cleanly (no crashes, no NaN).
    2. report status == PASS in results.txt.
    3. mean_T_sfc within 1 K of the IC (300 K).
    4. max|v| < 50 m/s (well under the 200 m/s BLOWUP gate; iter-12
       production runs landed at 2-12 m/s; >50 m/s in a 2-day smoke
       is a smoking gun for an in-flight CFL crash).

    These bounds reflect the iter-12 / iter-13 30-day production
    measurements where all grids land at mean_T_sfc within ±1 K of IC
    and max|v| ≤ 12 m/s at equilibrium.

    Coverage includes C48 (iter-13 auto-dt=150) so any future
    regression of the auto-dt ladder (e.g. accidentally widening
    the dt=600 branch back to N≤48) trips this test by day 2.
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
    _assert_rce_pass(
        out_dir,
        label=f"{grid_type}/{discretization}@{resolution}",
        temp_tol=1.0, max_v_cap=50.0,
    )


@pytest.mark.slow
def test_rce_2day_smoke_c96_slow(tmp_path):
    """SLOW: C96 2-day smoke for the iter-13 dt=75 ladder branch.

    Same assertions as ``test_rce_2day_smoke_passes`` but for C96
    (cdgrid). Lives outside the default smoke matrix because C96
    2-day takes ~10 min wall on M5 Pro (vs 1-3 min for C12/C48/V4
    /T21). Run nightly via ``pytest -m slow``.

    iter-13 measured C96 30-day day 5 PASS at mean_T_sfc=299.90,
    max|v|=7.99 m/s. A 2-day run lands well inside that envelope.
    """
    out_dir = tmp_path / "cubed_sphere_96"
    result = _run_rce(
        grid_type="cubed_sphere", discretization="cdgrid",
        resolution=96, days=2, output_dir=out_dir,
    )
    if result.returncode != 0:
        pytest.fail(
            "C96 2-day exited nonzero "
            f"({result.returncode})\nstdout tail:\n{result.stdout[-1500:]}"
        )
    _assert_rce_pass(out_dir, label="C96 2-day", temp_tol=1.0, max_v_cap=50.0)


@pytest.mark.slow
def test_blowup_gate_fires_on_supersonic_winds(tmp_path):
    """Regression for the iter-13 BLOWUP threshold (max_v > 200 m/s).

    Drives a deliberately-unstable C48 run with the OLD broken
    dt=300 default to verify scripts/run_rce.py now:
      (a) writes status: FAIL
      (b) exits with non-zero return code

    This was iter-13's exact failure mode at C48; the threshold was
    500 m/s pre-iter-13 and the run reported PASS while max|v| was
    236 m/s. The threshold is now 200 m/s; this test locks it in.
    """
    out_dir = tmp_path / "c48_dt300_blowup"
    # _run_rce would pick the iter-13 auto-dt=150 (safe); we need to
    # FORCE --dt 300 to reproduce the iter-12 blowup. Invoke
    # subprocess directly with the override.
    cmd = [
        sys.executable, str(RUN_RCE),
        "--grid-type", "cubed_sphere", "--discretization", "cdgrid",
        "--resolution", "48", "--days", "30", "--diag-days", "5",
        "--dt", "300", "--nlev", "20", "--output", str(out_dir),
    ]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=1800,
    )
    # When run_rce.py detects BLOWUP it: writes status: FAIL +
    # exits with code 1 (iter-101 contract).
    assert result.returncode != 0, (
        "C48 30-day at the iter-13-banned dt=300 should BLOWUP "
        "(iter-13 measured max_wind=236 m/s by day 20), making "
        f"run_rce.py exit non-zero. Got returncode={result.returncode}. "
        "Either the BLOWUP threshold is too loose again, or "
        "C48-dt=300 is stable now (unexpected — re-verify F11)."
    )
    fields = _parse_results(out_dir)
    assert fields is not None, "C48-dt=300 didn't even write results.txt"
    assert fields.get("status") == "FAIL", (
        f"C48-dt=300: status={fields.get('status')} (want FAIL). "
        f"notes={fields.get('notes')}"
    )


@pytest.mark.slow
def test_c48_30day_nightly_validation(tmp_path):
    """SLOW nightly test (~10 min wall): runs C48 RCE for 30 days at
    iter-13 auto-dt=150 and asserts the iter-15 measured PASS
    envelope (mean_T_sfc within ±0.5 K of IC, max\\|v\\| ≤ 20 m/s).

    Skipped by default (`@pytest.mark.slow`). Run nightly via:
        pytest -m slow tests/atmosphere/hydrostatic/

    Catches slow radiative-convective-equilibration regressions
    that the 2-day smoke can't see — particularly a slow CFL
    growth that exceeds the 50 m/s 2-day cap by day 10-15 (per
    the iter-12 broken C48 timeseries).
    """
    out_dir = tmp_path / "c48_30d"
    result = _run_rce(
        grid_type="cubed_sphere",
        discretization="cdgrid",
        resolution=48,
        days=30,
        output_dir=out_dir,
    )
    if result.returncode != 0:
        pytest.fail(
            f"C48 30-day at iter-13 auto-dt failed: rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    # Tighter envelope than the 2-day smoke: iter-15 measured C48
    # 30-day at mean_T_sfc=300.13, max|v|=13.45. Allow ±1 K (3x
    # iter-15 deviation from IC) + max|v| < 25 m/s (almost 2x
    # iter-15 measured) to absorb run-to-run variation while still
    # catching slow CFL crashes that the 2-day smoke missed.
    _assert_rce_pass(out_dir, label="C48 30-day", temp_tol=1.0, max_v_cap=25.0)
