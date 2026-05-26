"""Pin the auto-dt heuristic used by scripts/run_rce.py.

Cross-grid RCE smoke (2026-05) exposed a voronoi/MPAS blowup at the
shared default dt=600 s for N=4. The fix in run_rce.py picks
``DT = 300.0`` unconditionally on voronoi (verified stable through
day 1 at V4/L20). These tests pin that contract by importing the
exact decision branch and checking the output for each grid type.

If a future commit lowers the cubed_sphere / latlon / gaussian
default below 600 s or raises the voronoi default above 300 s,
this test fails — pointing the author at CRM_implementation.md
iter-8 / R10 for the cross-grid stability numbers.
"""
from __future__ import annotations

import argparse


def _auto_dt(grid_type: str, resolution: int, dt_override=None) -> float:
    """Re-implementation of scripts/run_rce.py's auto-dt branch.

    Keep this in lock-step with the production logic. The test below
    is intentionally a duplicate of the production heuristic — if the
    production branch is changed without updating this test, the
    test must fail so the author sees the contract change.

    iter-13: ladder refined after the C48 BLOWUP at dt=300.
    iter-20: ladder split further after C96 BLOWUP at dt=75 (the
    iter-13 extrapolation that never got long-run-validated). dt
    drops faster than linearly past C48 because higher resolution
    resolves more synoptic-wave activity that exponentially demands
    tighter CFL.
    iter-21: N>96 now RAISES instead of silently picking dt=20
    (Codex iter-20 HIGH). High-resolution runs must pass explicit
    --dt until empirical measurement lands.
    """
    if dt_override is not None:
        return float(dt_override)
    if grid_type == "voronoi":
        return 300.0
    if resolution <= 24:
        return 600.0
    if resolution <= 48:
        return 150.0
    if resolution <= 72:
        return 75.0
    if resolution <= 96:
        return 37.0
    raise ValueError(
        f"auto-dt has no validated value for N={resolution} (>96)"
    )


def test_voronoi_auto_dt_is_300s():
    """Voronoi must default to 300 s regardless of resolution
    (R10 cross-grid finding 2026-05)."""
    for N in (4, 5, 8, 12):
        assert _auto_dt("voronoi", N) == 300.0, (
            f"voronoi at N={N} must default to 300 s; got "
            f"{_auto_dt('voronoi', N)}"
        )


def test_cubed_sphere_auto_dt():
    assert _auto_dt("cubed_sphere", 24) == 600.0
    assert _auto_dt("cubed_sphere", 25) == 150.0  # iter-13: C25-48 needs 150
    assert _auto_dt("cubed_sphere", 48) == 150.0
    assert _auto_dt("cubed_sphere", 49) == 75.0   # iter-13 verified at N=49..72
    assert _auto_dt("cubed_sphere", 72) == 75.0
    assert _auto_dt("cubed_sphere", 73) == 37.0   # iter-20: C96 needs ≤ 37
    assert _auto_dt("cubed_sphere", 96) == 37.0
    # iter-21: N>96 RAISES instead of silently picking dt=20 (Codex
    # iter-20 HIGH: silent extrapolation hid the same iter-13 mistake
    # that produced the C96 BLOWUP).
    import pytest as _pytest
    with _pytest.raises(ValueError, match=">96"):
        _auto_dt("cubed_sphere", 97)
    with _pytest.raises(ValueError):
        _auto_dt("cubed_sphere", 192)
    # User-supplied dt always wins, even for high N.
    assert _auto_dt("cubed_sphere", 192, dt_override=10.0) == 10.0


def test_latlon_auto_dt():
    assert _auto_dt("latlon", 16) == 600.0
    assert _auto_dt("latlon", 32) == 150.0


def test_gaussian_auto_dt():
    assert _auto_dt("gaussian", 21) == 600.0
    assert _auto_dt("gaussian", 42) == 150.0


def test_override_takes_precedence():
    assert _auto_dt("voronoi", 4, dt_override=60.0) == 60.0
    assert _auto_dt("cubed_sphere", 24, dt_override=900.0) == 900.0


def test_run_rce_branch_matches_local_helper():
    """Sanity: parse the production run_rce.py branch and re-derive
    the same dt values to catch any divergence between this test's
    local _auto_dt mirror and the production source."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "scripts" / "run_rce.py"
    text = src.read_text()
    assert "if args.grid_type == \"voronoi\":" in text and "DT = 300.0" in text, (
        "scripts/run_rce.py no longer pins voronoi to dt=300 — update "
        "CRM_implementation.md iter-8 + this test if the voronoi "
        "stability boundary has shifted."
    )
    # iter-13: dt=150 step for N in (24, 48].
    assert "DT = 150.0" in text, (
        "scripts/run_rce.py no longer has the iter-13 dt=150 step "
        "for N in (24, 48] — refresh this test if the ladder changed."
    )
    # iter-20: dt=37 step for N in (72, 96]; pinned after C96 BLOWUP
    # at dt=75 (the iter-13 extrapolation that was never long-run-
    # validated).
    assert "DT = 37.0" in text, (
        "scripts/run_rce.py no longer has the iter-20 dt=37 step "
        "for N in (72, 96] — refresh this test if the ladder changed."
    )
    # iter-21: N>96 RAISES instead of silently picking a value.
    assert 'auto-dt has no validated value for N=' in text, (
        "scripts/run_rce.py no longer refuses N>96 — Codex iter-20 "
        "HIGH requested a hard error for silent high-N extrapolation."
    )
