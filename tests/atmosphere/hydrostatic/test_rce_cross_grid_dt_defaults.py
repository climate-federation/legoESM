"""Pin the auto-dt heuristic used by ``scripts/run_rce.py``.

iter-24: now imports ``legoesm.driver.rce_dt.auto_dt_rce`` directly
instead of hand-copying the production logic into a test mirror
(Codex iter-19 LOW finding). Any change to the production heuristic
that lands without updating this test will trip on the explicit
boundary assertions below.

Empirical lineage of each ladder boundary:
* voronoi V4 → dt=300 (iter-8 fix; dt>=450 blows up at day 1)
* C24 → 600 (iter-12 30-day PASS)
* C48 → 150 (iter-13/15 30-day PASS; dt=300 blows up at day 25)
* C72 → 75  (iter-13 small-N end; C72 30-day in flight at iter-22/23)
* C96 → 37  (iter-20 fix after dt=75 BLOWUP at day 20)
* N>96 → ValueError (iter-21 hard refusal; silent extrapolation
  produced the iter-13/20 cascade of BLOWUPs).
"""
from __future__ import annotations

import pytest

from legoesm.driver.rce_dt import auto_dt_rce


def _resolve_dt(grid_type, resolution, dt_override=None):
    """Reproduce the run_rce.py contract: explicit --dt wins, else
    auto-dt. Tests this contract here so we don't have to subprocess
    out to the script just to verify the precedence."""
    if dt_override is not None:
        return float(dt_override)
    return auto_dt_rce(grid_type, resolution)


def test_voronoi_auto_dt_is_300s():
    """Voronoi must default to 300 s regardless of resolution
    (R10 cross-grid finding 2026-05)."""
    for N in (4, 5, 8, 12):
        assert auto_dt_rce("voronoi", N) == 300.0, (
            f"voronoi at N={N} must default to 300 s; got "
            f"{auto_dt_rce('voronoi', N)}"
        )


def test_cubed_sphere_auto_dt():
    assert auto_dt_rce("cubed_sphere", 24) == 600.0
    assert auto_dt_rce("cubed_sphere", 25) == 150.0  # iter-13: C25-48 needs 150
    assert auto_dt_rce("cubed_sphere", 48) == 150.0
    assert auto_dt_rce("cubed_sphere", 49) == 75.0
    assert auto_dt_rce("cubed_sphere", 72) == 75.0
    assert auto_dt_rce("cubed_sphere", 73) == 37.0   # iter-20: C96 needs ≤37
    assert auto_dt_rce("cubed_sphere", 96) == 37.0
    # iter-21: N>96 RAISES instead of silently picking dt=20.
    with pytest.raises(ValueError, match=">96"):
        auto_dt_rce("cubed_sphere", 97)
    with pytest.raises(ValueError):
        auto_dt_rce("cubed_sphere", 192)


def test_latlon_auto_dt():
    assert auto_dt_rce("latlon", 16) == 600.0
    assert auto_dt_rce("latlon", 32) == 150.0


def test_gaussian_auto_dt():
    assert auto_dt_rce("gaussian", 21) == 600.0
    assert auto_dt_rce("gaussian", 42) == 150.0


def test_override_takes_precedence():
    """Explicit --dt always wins over auto-dt, even for high-N where
    the auto path would raise."""
    assert _resolve_dt("voronoi", 4, dt_override=60.0) == 60.0
    assert _resolve_dt("cubed_sphere", 24, dt_override=900.0) == 900.0
    # The override path bypasses the N>96 ValueError gate (this is
    # the documented escape hatch for high-resolution runs).
    assert _resolve_dt("cubed_sphere", 192, dt_override=10.0) == 10.0


def test_run_rce_uses_auto_dt_rce():
    """Sanity check that scripts/run_rce.py still imports + uses
    legoesm.driver.rce_dt.auto_dt_rce. If the production script is
    refactored to inline the ladder again, this test forces a
    refresh."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "scripts" / "run_rce.py"
    text = src.read_text()
    assert "from legoesm.driver.rce_dt import auto_dt_rce" in text, (
        "scripts/run_rce.py no longer imports auto_dt_rce from "
        "legoesm.driver.rce_dt — iter-24 refactor reverted? Update "
        "this test if the ladder source moved."
    )
    assert "auto_dt_rce(args.grid_type, N)" in text, (
        "scripts/run_rce.py no longer calls auto_dt_rce — refactor?"
    )
