"""The cube Held-Suarez matrix case carries D-staggered winds by default (#1028)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "matrix"))
from run_atmosphere_test_matrix import _hs_persistent_d_enabled  # noqa: E402


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("LEGOESM_HS_PERSISTENT_D", raising=False)
    assert _hs_persistent_d_enabled() is True


@pytest.mark.parametrize("val,expect", [("0", False), ("1", True), ("off", False), ("", True)])
def test_env_override(monkeypatch, val, expect):
    monkeypatch.setenv("LEGOESM_HS_PERSISTENT_D", val)
    assert _hs_persistent_d_enabled() is expect
