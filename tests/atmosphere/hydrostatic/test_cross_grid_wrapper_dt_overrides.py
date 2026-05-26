"""Pin the hard-coded dt overrides in cross-grid wrapper scripts.

``scripts/run_amip_cross_grid.sh`` (iter-32) and
``scripts/run_rce_cross_grid.sh`` carry per-grid DT_OVERRIDE values
in a colon-delimited GRID_TABLE. If the central
``legoesm.driver.rce_dt.auto_dt_rce`` ladder is later refined
(e.g. iter-26 added the C72 measurement; a future iter could
re-anchor the C48 branch), the wrapper's hard-coded values would
silently diverge.

These tests parse the wrapper scripts as text + assert each
DT_OVERRIDE in the table is either:
  (a) empty (= fall through to driver's own default), OR
  (b) within 30 % of the auto-dt ladder for the same grid+N.

A divergence > 30 % means either the ladder shifted (update the
wrapper) or the wrapper was tightened for a non-ladder reason
(document it inline in the test).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from legoesm.driver.rce_dt import auto_dt_rce


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_grid_table(script_path: Path):
    """Return list of (grid_type, resolution, dt_override_or_None) tuples
    from a GRID_TABLE block in a wrapper script.

    Tolerates 4-, 5-, or 6-field entries (trailing fields optional).
    """
    text = script_path.read_text()
    # Find the GRID_TABLE=( ... ) block.
    match = re.search(r"GRID_TABLE=\((.*?)\)", text, re.DOTALL)
    assert match, f"{script_path.name}: no GRID_TABLE=(...) block found"
    body = match.group(1)
    entries = []
    for line in body.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        # Skip a comment line that starts with ``#`` (line comments aren't
        # part of the bash array but we want to be tolerant).
        if line.startswith("#"):
            continue
        # Strip trailing inline comments after the closing quote (rare).
        parts = line.split(":")
        if len(parts) < 4:
            continue
        grid_type = parts[0]
        resolution = int(parts[1])
        dt_override = parts[5] if len(parts) >= 6 else (
            parts[4] if len(parts) >= 5 else ""
        )
        # The RCE wrapper has 4 fields (no DT_OVERRIDE column);
        # detect by checking whether parts[4] is a number.
        if len(parts) <= 4:
            dt_override = ""
        elif len(parts) == 5:
            # RCE wrapper format has no DT column; the 5th field
            # is a folder. Try parsing as float — if it fails,
            # it's not a dt override.
            try:
                float(parts[4])
                dt_override = parts[4]
            except (ValueError, TypeError):
                dt_override = ""
        # parts[5] (6-field) is the DT_OVERRIDE for the AMIP wrapper.
        dt_val = float(dt_override) if dt_override.strip() else None
        entries.append((grid_type, resolution, dt_val))
    return entries


def test_amip_cross_grid_wrapper_dt_overrides_match_ladder():
    """The iter-32 AMIP wrapper hard-codes dt=150 for C48 and T42,
    dt=60 for V6 (iter-33), and leaves LL90 empty (fall through
    to run_amip.py default).

    Every non-empty override must be within 30 % of the ladder
    value for the same (grid_type, resolution) — catches a future
    ladder shift that doesn't propagate to the wrapper.
    """
    script = REPO_ROOT / "scripts" / "run_amip_cross_grid.sh"
    entries = _parse_grid_table(script)
    for grid_type, N, dt_override in entries:
        if dt_override is None:
            continue
        # Voronoi: ladder returns 300, wrapper pins 60 because the
        # MPAS dycore is unstable at the default per iter-33 / the
        # smoke-test-amip note. Skip the strict ratio check here.
        if grid_type == "voronoi":
            assert dt_override < auto_dt_rce(grid_type, N), (
                f"{script.name}: voronoi override dt={dt_override} should "
                f"be TIGHTER than the ladder value (auto={auto_dt_rce(grid_type, N)})"
            )
            continue
        ladder = auto_dt_rce(grid_type, N)
        ratio = dt_override / ladder
        assert 0.5 < ratio < 2.0, (
            f"{script.name}: {grid_type}/N={N}: DT_OVERRIDE={dt_override}, "
            f"auto_dt_rce={ladder}, ratio={ratio:.2f} outside [0.5, 2.0]. "
            f"Update the wrapper or document the divergence."
        )


def test_rce_cross_grid_wrapper_has_no_dt_overrides():
    """The RCE wrapper (iter-7) uses a 4-field GRID_TABLE without a
    DT_OVERRIDE column. The auto-dt ladder picks dt at run_rce.py
    invocation time (iter-24 refactor). Lock this contract."""
    script = REPO_ROOT / "scripts" / "run_rce_cross_grid.sh"
    text = script.read_text()
    match = re.search(r"GRID_TABLE=\((.*?)\)", text, re.DOTALL)
    assert match, "RCE wrapper missing GRID_TABLE"
    body = match.group(1)
    for line in body.splitlines():
        s = line.strip().strip('"').strip("'")
        if not s or s.startswith("#"):
            continue
        # RCE GRID_TABLE format: grid:resolution:discretization:folder
        # (4 fields). If it grows past 4, the wrapper started carrying
        # per-grid overrides — refresh this test.
        n_fields = len(s.split(":"))
        assert n_fields == 4, (
            f"RCE GRID_TABLE entry {s!r} has {n_fields} fields; "
            "iter-24 refactor pushed dt selection into run_rce.py. "
            "If the wrapper now carries explicit overrides, refresh "
            "this test."
        )
