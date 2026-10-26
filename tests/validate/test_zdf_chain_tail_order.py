"""Red tests for the ordered ZDF tail receipt."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

PROBE = (
    Path(__file__).parents[2]
    / "scripts/validate/ocean_fidelity/dino_1226/zdf_chain_tail_existing.py"
)


def _load_probe():
    os.environ.setdefault("DINO_1226_LANE", "d180")
    os.environ.setdefault("DINO_ORACLE_ROOT", "/tmp")
    spec = importlib.util.spec_from_file_location("zdf_chain_tail_existing", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> dict[str, dict[str, str]]:
    rows = {str(row): {"disposition": "VERIFIED"} for row in range(20, 33)}
    rows["24"]["disposition"] = "WAIVED"
    rows["28"]["disposition"] = "UNMEASURED-NEEDS-DUMP"
    for row in range(29, 33):
        rows[str(row)]["disposition"] = "UNMEASURED-BLOCKED-BY-ROW28"
    return rows


def test_ordered_stop_leaves_clean_prefix_unchanged():
    probe = _load_probe()
    rows = _rows()
    original = {key: value.copy() for key, value in rows.items()}
    assert probe._enforce_ordered_stop(rows) is None
    assert rows == original


def test_ordered_stop_blocks_every_later_row_at_first_divergence():
    probe = _load_probe()
    rows = _rows()
    rows["21"]["disposition"] = "DIVERGED"
    assert probe._enforce_ordered_stop(rows) == "21"
    assert rows["21"]["disposition"] == "DIVERGED"
    for row in range(22, 33):
        item = rows[str(row)]
        assert item["disposition"] == "UNMEASURED-BLOCKED-BY-ROW21"
        assert "targeting_preview_disposition" in item
