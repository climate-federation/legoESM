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

# iter-78: helpers extracted to sibling ``_rce_helpers.py`` module.
# Underscore prefix avoids pytest's ``test_*.py`` collection. Both
# this file and ``test_rce_helpers_unit.py`` import from the shared
# module — previously, the unit tests cross-imported from this test
# file which triggered pytest collection of THIS file as a side
# effect of the import.
from tests.atmosphere.hydrostatic._rce_helpers import (
    REPO_ROOT,
    RUN_RCE,
    _run_rce,
    _parse_results,
    _parse_notes,
    _assert_rce_pass,
    _assert_dt_used,
    _assert_max_wind_peak_below,
)


# Coverage matrix (iter-13). Includes C48 (iter-13 auto-dt=150
# branch) so any future regression of the auto-dt ladder trips the
# BLOWUP gate at 2 days and FAILS this test. iter-13 measurements:
# C48 reached only max|v|≈1.7 m/s by day 2.
#
# C96 (iter-20 auto-dt=37 branch — iter-13's dt=75 BLEW UP at
# day 20, iter-20 dropped to dt=37) lives in
# `test_rce_2day_smoke_c96_slow` below because C96 2-day takes
# ~10 min wall — too slow for default CI but worthwhile nightly.
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
    """SLOW: C96 2-day smoke for the iter-20 dt=37 ladder branch.

    Same assertions as ``test_rce_2day_smoke_passes`` but for C96
    (cdgrid). Lives outside the default smoke matrix because C96
    2-day takes ~10 min wall on M5 Pro (vs 1-3 min for C12/C48/V4
    /T21). Run nightly via ``pytest -m slow``.

    iter-22 measured C96 10-day at dt=37 PASS (mean_T_sfc=299.98,
    max\\|v\\|=9.07 m/s, wall=2506 s ≈ 250 s/day). A 2-day run
    lands well inside that envelope.

    iter-47: docstring corrected from "iter-13 dt=75" — that
    extrapolation BLEW UP at iter-20 (C96 30-day max\\|v\\|=527 m/s
    by day 20). The current ladder pins C96 → dt=37 (iter-20 fix).
    """
    out_dir = tmp_path / "cubed_sphere_96"
    # iter-45 Codex HIGH carryover: C96 2-day measured ~500-600 s on
    # M5 Pro (iter-22 timed C96 10-day at 2506 s ≈ 250 s/day). The
    # iter-44 default _run_rce timeout=600 cut too close — a 10%
    # slower box would silently hang. Bump to 1800 s (3x cushion).
    result = _run_rce(
        grid_type="cubed_sphere", discretization="cdgrid",
        resolution=96, days=2, output_dir=out_dir, timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            "C96 2-day exited nonzero "
            f"({result.returncode})\nstdout tail:\n{result.stdout[-1500:]}"
        )
    # iter-47: apply iter-46 shared hardening to C96. iter-20 BLOWUP
    # at C96/dt=75 is exactly the silent-pass class these helpers
    # exist to catch — a future ladder regression that re-routes
    # C96 onto the dt=75 branch would only fail at day 20 of a 30-day
    # run; the 2-day smoke (which lands well inside even the
    # dt=75 BLOWUP timeline at day 20) would still pass _assert_rce_pass.
    # The dt-assertion below catches that drift at the 2-day point.
    _assert_dt_used(out_dir, label="C96 2-day", expected_dt=37.0)
    _assert_max_wind_peak_below(out_dir, label="C96 2-day", cap=50.0)
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
    # iter-48: also pin that the SPIKE actually fired with a
    # supersonic-class peak. ``status: FAIL`` could in principle be
    # raised by a different failure mode (NaN earlier, runtime crash,
    # etc.) — the test name says "supersonic winds" so verify the
    # gating channel by scanning ``mean_timeseries.csv`` for a
    # peak ``max_wind`` >= 200 m/s (iter-13 measured 236 m/s by day
    # 20). Inverse of the iter-46 ``_assert_max_wind_peak_below``.
    mean_csv = out_dir / "mean_timeseries.csv"
    if mean_csv.exists():
        import csv
        peak_v = 0.0
        with open(mean_csv) as fh:
            reader = csv.DictReader(fh)
            if (reader.fieldnames is not None
                    and "max_wind" in reader.fieldnames):
                for row in reader:
                    try:
                        peak_v = max(peak_v, abs(float(row["max_wind"])))
                    except (TypeError, ValueError):
                        pass
        assert peak_v >= 200.0, (
            f"C48-dt=300: peak max|v|={peak_v:.2f} did NOT cross "
            f"the 200 m/s BLOWUP threshold (iter-13 measured "
            f"~236 m/s by day 20). status: FAIL fired but via a "
            f"different channel — check that the supersonic-wind "
            f"gate actually tripped. Source: {mean_csv}"
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
    # iter-45 Codex HIGH carryover: iter-15 measured C48 30-day at
    # wall=500.9 s. Default _run_rce timeout=600 left only 20%
    # margin — a slower box would silently hang. Bump to 1800 s
    # (3.6x cushion).
    result = _run_rce(
        grid_type="cubed_sphere",
        discretization="cdgrid",
        resolution=48,
        days=30,
        output_dir=out_dir,
        timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"C48 30-day at iter-13 auto-dt failed: rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    # iter-46: apply Codex iter-44 MEDIUM#1+#2 hardening (originally
    # landed for C72 only) to C48 too. Same silent-pass risks
    # apply: a ladder drift that silently routes C48 onto a
    # different dt branch (e.g. 75 instead of 150) would still
    # produce a 30-day run that passes the wider envelope checks
    # below — but it would NOT be testing the iter-13 dt=150
    # branch any longer.
    _assert_dt_used(out_dir, label="C48 30-day", expected_dt=150.0)
    # And catch a mid-run CFL spike that recovers by day 30
    # (iter-12 broken-C48 spike reached 240 m/s at days 20-25
    # then settled — the kind of trajectory the final-day notes
    # check misses).
    _assert_max_wind_peak_below(out_dir, label="C48 30-day", cap=25.0)
    # Tighter envelope than the 2-day smoke: iter-15 measured C48
    # 30-day at mean_T_sfc=300.13, max|v|=13.45. Allow ±1 K (3x
    # iter-15 deviation from IC) + max|v| < 25 m/s (almost 2x
    # iter-15 measured) to absorb run-to-run variation while still
    # catching slow CFL crashes that the 2-day smoke missed.
    _assert_rce_pass(out_dir, label="C48 30-day", temp_tol=1.0, max_v_cap=25.0)


@pytest.mark.slow
def test_c72_30day_nightly_validation(tmp_path):
    """SLOW nightly test (~40 min wall on M5 Pro): runs C72 RCE for
    30 days at iter-13 auto-dt=75 and asserts the iter-26 measured
    PASS envelope.

    iter-26 measured:
        dt                = 75 s
        final mean_T_sfc  = 299.81 K (−0.19 from IC = 300.0 K)
        max\\|v\\|        = 17.85 m/s
        wall              = 2373 s

    Codex iter-22 review flagged that the iter-13 dt=75 branch
    (N=49..72) was empirically anchored at N=49 only (the C48
    boundary) — iter-26 closed the upper-end gap by measuring C72
    directly. This test locks that measurement as a regression
    contract so a future ladder bump that re-tunes the dt=75
    branch must update this test in the same PR.

    Skipped by default (`@pytest.mark.slow`). Run nightly via:
        pytest -m slow tests/atmosphere/hydrostatic/

    Catches slow CFL growth past day 10 that the 2-day C96 slow
    smoke (test_rce_2day_smoke_c96_slow) and the C48 30-day nightly
    (test_c48_30day_nightly_validation) cannot expose at C72
    specifically.
    """
    out_dir = tmp_path / "c72_30d"
    # iter-44 Codex HIGH: iter-26 measured 2373 s wall on M5 Pro.
    # 4800 s gives a 2x cushion on slower runners + a hang guard.
    result = _run_rce(
        grid_type="cubed_sphere",
        discretization="cdgrid",
        resolution=72,
        days=30,
        output_dir=out_dir,
        timeout_s=4800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"C72 30-day at iter-13 auto-dt=75 failed: "
            f"rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    # iter-44 Codex MEDIUM#1 (refactored to shared helper in
    # iter-46): lock the dt actually used. iter-13/26 contract is
    # dt=75 for N in (48, 72]. Silent ladder drift would still
    # pass the wider envelope checks below.
    _assert_dt_used(out_dir, label="C72 30-day", expected_dt=75.0)
    # iter-44 Codex MEDIUM#2 (refactored to shared helper in
    # iter-46): check the PEAK max|v| across the 30-day timeseries,
    # not just the final day. iter-26 measured monotone rise to
    # ~18 m/s; a regression that spikes to 100+ mid-run and damps
    # by day 30 would slip through the last-day-only check.
    _assert_max_wind_peak_below(out_dir, label="C72 30-day", cap=25.0)
    # Tighter envelope than the 2-day C96 smoke: iter-26 measured
    # C72 30-day at mean_T_sfc=299.81 (Δ=-0.19), max|v|=17.85.
    # Allow ±1 K (5x iter-26 |Δ|) + max|v| < 25 m/s (1.4x iter-26
    # measured) — the same generous-but-meaningful margin used for
    # the C48 nightly. Catches slow CFL growth at C72 specifically.
    _assert_rce_pass(out_dir, label="C72 30-day", temp_tol=1.0, max_v_cap=25.0)


@pytest.mark.slow
def test_voronoi_v4_30day_nightly_validation(tmp_path):
    """SLOW nightly test (~2 min wall): runs voronoi V4 RCE for
    30 days at the iter-8/12 dt=300 pin and asserts the iter-12
    measured PASS envelope.

    iter-12 measured:
        dt                = 300 s
        final mean_T_sfc  = 300.85 K (+0.85 from IC = 300.0 K)
        max\\|v\\|        = 2.28 m/s
        wall              = 101 s

    iter-8 + iter-33 history:
        - iter-8: V4 dt>=450 BLOWUPs at day 1 — pinned to dt=300 in
          the auto_dt_rce ladder (``return 300.0`` for voronoi).
        - iter-33 Codex HIGH: V6 AMIP wrapper was running dt=600
          → BLOWUP risk; tightened to dt=60.
        - The MPAS dycore has a different stability profile than
          gravity-wave-CFL would predict (per ``cfl_max_dt`` ratio
          ~0.42 at V4 implies dt=712 should be safe — it ISN'T).

    Until iter-50 this dt=300 contract had no 30-day CI backing.
    Adds the same iter-46 hardening helpers used by C48/C72/C96:
    - ``_assert_dt_used(expected_dt=300.0)`` catches a ladder drift
      that re-routes voronoi onto a different dt branch.
    - ``_assert_max_wind_peak_below(cap=25.0)`` catches a mid-run
      CFL spike that recovered by day 30. iter-12 measured peak
      2.28 m/s; 25.0 cap is ~10x cushion.
    - ``_assert_rce_pass`` keeps the loose last-day envelope.

    Skipped by default (`@pytest.mark.slow`). Run nightly via:
        pytest -m slow tests/atmosphere/hydrostatic/

    Wall budget ~2 min on M5 Pro. timeout_s=1800 (15x cushion).
    """
    out_dir = tmp_path / "voronoi_v4_30d"
    result = _run_rce(
        grid_type="voronoi",
        discretization="mpas",
        resolution=4,
        days=30,
        output_dir=out_dir,
        timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"V4 30-day at dt=300 failed: rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    _assert_dt_used(out_dir, label="V4 30-day", expected_dt=300.0)
    # Codex iter-50 MEDIUM: cap=25.0 was 11x iter-12 peak (2.28
    # m/s). Tightened to 10.0 (still 4.4x cushion, but no longer
    # mirrors the cubed-sphere caps that came from a totally
    # different stability regime — MPAS dycore stays well below
    # synoptic-wave scales at V4 resolution).
    _assert_max_wind_peak_below(out_dir, label="V4 30-day", cap=10.0)
    # iter-12 measured V4 30-day at mean_T_sfc=300.85 (Δ=+0.85),
    # max|v|=2.28. Codex iter-50 HIGH: ``_assert_rce_pass`` uses
    # strict ``<`` against ``abs(t_sfc - 300.0) < temp_tol``, so
    # temp_tol=1.0 left only 0.15 K of positive-drift headroom
    # vs the measured 0.85 K warming. Bumped to 1.5 (0.65 K
    # cushion, robust to run-to-run variance on MPAS).
    _assert_rce_pass(
        out_dir, label="V4 30-day", temp_tol=1.5, max_v_cap=10.0,
    )


@pytest.mark.slow
def test_latlon_ll32_30day_nightly_validation(tmp_path):
    """SLOW nightly test (~3 min wall): runs LL32 (latlon C-grid)
    RCE for 30 days at the iter-12 measurement and asserts the
    PASS envelope.

    iter-12 measured:
        dt                = 81.844 s (auto-dt-ladder 150 post-clamped
                            via pole_cell_dx + cfl_max_dt; cfl=0.8)
        final mean_T_sfc  = 300.09 K (+0.09 from IC = 300.0 K)
        max\\|v\\|        = 11.19 m/s
        wall              = 179 s

    Until iter-51 the LL32 dt=82 contract had **no 30-day CI
    backing** — only the 2-day LL16 smoke gated latlon regressions
    (and LL16 dt=600 is the ladder branch, not the post-clamp).

    Uses ``_assert_dt_used(..., abs_tol=1e-2)`` because the
    pole-cell CFL clamp produces a non-integer value (81.844 s)
    that depends on the pole_cell_dx computation. abs_tol=1e-2
    is tight enough to catch a rounding/scaling regression yet
    loose enough to absorb cross-platform double-precision drift
    (Codex iter-51 LOW fix: original 0.5 was too loose — would
    accept an inadvertent ``round(dt)`` to 82.0 silently).

    Skipped by default. Run nightly via:
        pytest -m slow tests/atmosphere/hydrostatic/

    Wall budget ~3 min on M5 Pro. timeout_s=1800 (10x cushion).
    """
    out_dir = tmp_path / "ll32_30d"
    result = _run_rce(
        grid_type="latlon",
        discretization="latlon_cgrid",
        resolution=32,
        days=30,
        output_dir=out_dir,
        timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"LL32 30-day failed: rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    # Pin the post-clamped dt within 1e-2 of the iter-12 measurement
    # (Codex iter-51 LOW fix: 0.5 was too loose; pole_cell_dx +
    # cfl_max_dt is deterministic float math so 1e-2 absorbs only
    # cross-platform double-precision drift).
    _assert_dt_used(
        out_dir, label="LL32 30-day", expected_dt=81.844, abs_tol=1e-2,
    )
    # iter-12 peak max|v|=11.19 m/s. Cap 20.0 = 1.8x cushion;
    # latlon C-grid with pole clamp is steadier than cubed-sphere
    # at the same resolution but more lively than MPAS.
    _assert_max_wind_peak_below(out_dir, label="LL32 30-day", cap=20.0)
    # iter-12 mean_T_sfc=300.09 (Δ=+0.09). temp_tol=1.0 gives
    # ample headroom.
    _assert_rce_pass(
        out_dir, label="LL32 30-day", temp_tol=1.0, max_v_cap=20.0,
    )


@pytest.mark.slow
def test_gaussian_t21_30day_nightly_validation(tmp_path):
    """SLOW nightly test (~2 min wall): runs T21 (gaussian/spectral)
    RCE for 30 days at the iter-12 auto-dt=600 measurement.

    iter-12 measured:
        dt                = 600 s
        final mean_T_sfc  = 300.13 K (+0.13 from IC = 300.0 K)
        max\\|v\\|        = 8.43 m/s
        wall              = 113 s

    Until iter-51 the T21 dt=600 contract had no 30-day CI
    backing — only the 2-day T21 smoke gated gaussian/spectral
    regressions.

    Wall budget ~2 min on M5 Pro. timeout_s=1800 (15x cushion).
    """
    out_dir = tmp_path / "t21_30d"
    result = _run_rce(
        grid_type="gaussian",
        discretization="spectral",
        resolution=21,
        days=30,
        output_dir=out_dir,
        timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"T21 30-day failed: rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    _assert_dt_used(out_dir, label="T21 30-day", expected_dt=600.0)
    # iter-12 peak max|v|=8.43 m/s. Cap 20.0 = 2.4x cushion.
    _assert_max_wind_peak_below(out_dir, label="T21 30-day", cap=20.0)
    # iter-12 mean_T_sfc=300.13 (Δ=+0.13). temp_tol=1.0 gives ample
    # headroom for the spectral path's run-to-run variance.
    _assert_rce_pass(
        out_dir, label="T21 30-day", temp_tol=1.0, max_v_cap=20.0,
    )


@pytest.mark.slow
def test_c96_10day_nightly_validation(tmp_path):
    """SLOW nightly test (~42 min wall): runs C96 (cdgrid) RCE for
    10 days at the iter-20/22 measured dt=37 ladder branch.

    iter-22 measured:
        dt                = 37 s (iter-20 ladder)
        final mean_T_sfc  = 299.98 K (-0.02 from IC = 300.0 K)
        max\\|v\\|        = 9.07 m/s
        wall              = 2506 s

    Background: iter-20 found C96 dt=75 BLOWUP at day 20 (max wind
    reaches 175 by day 15, 527 by day 20). iter-22 dropped the
    ladder to dt=37 + measured a 10-day PASS. C96 30-day is
    wall-time-gated (~125 min on M5 Pro).

    iter-47 noted the C96 2-day smoke CANNOT distinguish dt=37
    (production) from dt=75 (BLOWUP-in-flight) because BOTH land
    at max wind ≈ 5-10 m/s by day 2. The iter-47 added
    ``_assert_dt_used`` catches a ladder-side regression at the
    2-day level, but ONLY exercises 2 days of integration. iter-73
    closes that gap with 10 days at dt=37 — long enough that the
    iter-20 BLOWUP trajectory would already be visible (max wind
    ≈ 26 m/s by day 10 in iter-20 broken run).

    Skipped by default. Run nightly via ``pytest -m slow``.
    timeout_s=6000 (2.4× iter-22's measured 2506 s wall — Codex
    iter-74 MEDIUM caught the iter-73 4800 was actually only 1.9×).
    """
    out_dir = tmp_path / "c96_10d"
    result = _run_rce(
        grid_type="cubed_sphere",
        discretization="cdgrid",
        resolution=96,
        days=10,
        output_dir=out_dir,
        timeout_s=6000,
    )
    if result.returncode != 0:
        pytest.fail(
            f"C96 10-day at iter-20/22 dt=37 failed: "
            f"rc={result.returncode}"
            f"\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    _assert_dt_used(out_dir, label="C96 10-day", expected_dt=37.0)
    # iter-22 peak max|v|=9.07 m/s. Cap 25.0 = 2.75x cushion;
    # iter-20 broken-dt=75 reached 175 m/s by day 15 — cap 25
    # is the C48/C72 nightly value, ample headroom for production
    # C96 + clean fail signal vs the iter-20 BLOWUP trajectory.
    _assert_max_wind_peak_below(out_dir, label="C96 10-day", cap=25.0)
    # iter-22 mean_T_sfc=299.98 (Δ=-0.02). temp_tol=1.0 generous.
    _assert_rce_pass(
        out_dir, label="C96 10-day", temp_tol=1.0, max_v_cap=25.0,
    )
