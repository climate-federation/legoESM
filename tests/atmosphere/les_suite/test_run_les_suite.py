"""CPU dispatch tests for the LES-suite emission driver (no GPU LES integration).

The end-to-end emit (running the spectral core) is GPU-gated and exercised
separately; here we lock the registry dispatch + not-wired-regime hardening.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))


def _driver():
    import run_les_suite  # noqa: PLC0415
    return run_les_suite


def test_unknown_case_raises():
    m = _driver()
    with pytest.raises(Exception):
        m.main(["--case", "does_not_exist"])


def test_not_wired_regime_raises_systemexit():
    # a stratocumulus/moist case is registered but its emission is not wired yet;
    # the driver must refuse (SystemExit), not emit a wrong-regime artifact.
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "dycoms_rf01_sc"])


def test_wired_regimes_contains_dry_convective():
    m = _driver()
    assert "dry_convective" in m._WIRED_REGIMES
