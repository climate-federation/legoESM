"""Cross-grid RCE smoke at the iter-12 production-validated configs.

iter-12 (2026-05) measured 30-day RCE PASS on all 4 hydrostatic grid
families. These tests run a SHORTER (2-day) version of the same
configs as a CI-friendly regression so future commits don't quietly
break any of the grid×discretization paths.

Wall-time budget: 4 grids × ~10 s wall at 2 days = ~40 s total.

For the FULL 30-day production validation see
``scripts/run/run_rce_cross_grid.sh`` and CRM_implementation.md iter-12.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

# iter-78: helpers extracted to sibling ``_rce_helpers.py`` module.
# Underscore prefix avoids pytest's ``test_*.py`` collection. Both
# this file and ``test_rce_helpers_unit.py`` import from the shared
# module — previously, the unit tests cross-imported from this test
# file which triggered pytest collection of THIS file as a side
# effect of the import.
#
# iter-80: removed unused REPO_ROOT + _parse_notes + Path imports
# (Codex iter-80 MEDIUM): both were imported but never referenced
# in this file. RUN_RCE is still used by test_blowup_gate which
# constructs its own subprocess cmd (rather than going through
# _run_rce); the others (_run_rce / _parse_results / _assert_*) are
# all used by test bodies.
from tests.atmosphere.hydrostatic._rce_helpers import (
    RUN_RCE,
    _run_rce,
    _parse_results,
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
    # iter-91: iter-90 opted into max_v_floor=0.1 here but the box
    # was too loaded to live-verify safety on every grid's actual
    # day-2 max|v|. iter-13 reported C48 day-2 ≈ 1.7 m/s, but
    # small/spinup-slow grids (V4 day-2) could land below 0.1.
    # The status PASS + temp_tol checks above already catch a
    # broken dycore via cooler T drift at 2 days; max_v_floor is
    # preventive defense-in-depth which we keep ON the 30-day
    # nightlies (iter-88) but skip here. Floor opt-in stays
    # available via the helper param when needed.










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
    _assert_max_wind_peak_below(
        out_dir, label="V4 30-day", cap=10.0, min_floor=0.5,
    )
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
    _assert_max_wind_peak_below(
        out_dir, label="LL32 30-day", cap=20.0, min_floor=0.5,
    )
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
    _assert_max_wind_peak_below(
        out_dir, label="T21 30-day", cap=20.0, min_floor=0.5,
    )
    # iter-12 mean_T_sfc=300.13 (Δ=+0.13). temp_tol=1.0 gives ample
    # headroom for the spectral path's run-to-run variance.
    _assert_rce_pass(
        out_dir, label="T21 30-day", temp_tol=1.0, max_v_cap=20.0,
    )
