"""The committed producers of the D5 open-water fixture are the bytes that made it."""

from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_FIX = Path(__file__).parent / "fixtures" / "orca1_t4" / "rdg_openwater.json"
_SCRIPTS = {
    "extract.py": "scripts/validate/ice_fidelity/si3_rdg_openwater_extract.py",
    "budget.py": "scripts/validate/ice_fidelity/si3_rdg_openwater_budget.py",
}


def test_producers_compile_and_match_fixture_hashes() -> None:
    meta = json.loads(_FIX.read_text())
    for key, rel in _SCRIPTS.items():
        path = _ROOT / rel
        py_compile.compile(str(path), doraise=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["producer_sha256"][key], rel
        assert meta["producer_scripts"][key] == rel
