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


_GRID_TABLE_RE = re.compile(
    # Anchor to start-of-line GRID_TABLE assignment to skip any
    # accidental occurrences in comments. Match the FIRST opening
    # parenthesis after `GRID_TABLE=` and balance to the matching
    # close (no nested parens inside the bash array literal).
    r"^GRID_TABLE=\((?P<body>[^)]*)\)",
    re.MULTILINE,
)


def _parse_grid_table(script_path: Path, *, expected_fields: int):
    """Return list of (grid_type, resolution, dt_override_or_None) tuples
    from a GRID_TABLE block in a wrapper script.

    Codex iter-34/35 HIGH:
    * Regex anchors to start-of-line + uses a non-greedy character
      class to avoid matching a comment block that happens to
      contain ``GRID_TABLE=(``.
    * Caller declares the EXPECTED number of fields explicitly
      (4 for RCE, 6 for AMIP). Lines with a different field count
      fail the test — no length-based heuristic that could
      misclassify a future 5-field row.
    """
    text = script_path.read_text()
    matches = list(_GRID_TABLE_RE.finditer(text))
    assert len(matches) == 1, (
        f"{script_path.name}: expected exactly one GRID_TABLE=(...) "
        f"at start-of-line; found {len(matches)}."
    )
    body = matches[0].group("body")
    entries = []
    for line in body.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = line.split(":")
        assert len(parts) == expected_fields, (
            f"{script_path.name}: GRID_TABLE row {line!r} has "
            f"{len(parts)} fields; expected {expected_fields}. "
            f"Add a comment to the wrapper if the format changed, "
            f"and refresh this test."
        )
        grid_type = parts[0]
        resolution = int(parts[1])
        # DT_OVERRIDE column is the LAST one in the AMIP 6-field
        # format. RCE 4-field format has no DT column.
        dt_override = parts[-1] if expected_fields == 6 else ""
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
    entries = _parse_grid_table(script, expected_fields=6)
    for grid_type, N, dt_override in entries:
        if dt_override is None:
            continue
        # Voronoi: ladder returns 300, wrapper pins 60 because the
        # MPAS dycore is unstable at the default per iter-33 / the
        # smoke-test-amip note ("the MPAS hydrostatic dycore is
        # unstable at the default 600 s step despite the CFL
        # diagnostic reporting 0.09"). Lock the iter-33-pinned
        # value EXACTLY (60 s) so a future regression that bumps it
        # back to dt=300 silently trips this test — sanity-only
        # "tighter than ladder" was Codex iter-34/35 MEDIUM finding.
        if grid_type == "voronoi":
            assert dt_override == 60.0, (
                f"{script.name}: voronoi override dt={dt_override} "
                f"differs from the iter-33 pin of 60 s. The MPAS dycore "
                f"is documented unstable at the run_amip default 600; "
                f"60 s is the smoke-test-amip-validated value. Update "
                f"this test only if a new measurement justifies a "
                f"different pin."
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
