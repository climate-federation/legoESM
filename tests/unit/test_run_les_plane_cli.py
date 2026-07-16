"""Dispatch hardening for the plane-LES driver (scripts/run/run_les_plane.py).

The PBL surface-flux column ``_apply_pbl_column`` dispatches on the case NAME.
Per dispatch doctrine an unknown selection must raise ValueError — the old
bare ``else`` silently ran the Wangara prescribed-flux branch for any name.
The dispatch-hardening ratchet (tests/test_dispatch_hardening.py) skips
``scripts/``, so this direct test is the lock.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    path = _ROOT / "scripts" / "run" / "run_les_plane.py"
    spec = importlib.util.spec_from_file_location("run_les_plane", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unknown_case_raises_value_error():
    """The entry guard fires BEFORE any array work (case is a static Python
    string), so dummy state/hc arguments never get touched."""
    m = _load_driver()
    with pytest.raises(ValueError, match="Unknown LES case 'bomex'"):
        m._apply_pbl_column(None, 0.0, case="bomex", hc=None, z0=0.1,
                            dt=1.0, ug=8.0, vg=0.0)


def test_pbl_column_cases_in_sync_with_case_registry():
    """Every registered LES case must have a surface-flux branch (and vice
    versa): a new ``_CASES`` entry without a ``_PBL_COLUMN_CASES`` update —
    or a dispatch entry without a case definition — goes red here instead of
    silently running the wrong surface forcing."""
    m = _load_driver()
    assert set(m._PBL_COLUMN_CASES) == set(m._CASES)


def test_compressible_les_is_research_only_gated(monkeypatch):
    """This core does not sustain LES turbulence (relaminarises / can NaN), so the
    driver refuses to run without an explicit --research-only opt-in — it exits at
    the gate BEFORE any build/sim."""
    m = _load_driver()
    monkeypatch.setattr("sys.argv", ["run_les_plane.py", "--case", "neutral"])
    with pytest.raises(SystemExit):
        m.main()
