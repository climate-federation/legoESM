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


# --- frame scheduling (codex round-2 regression) ------------------------------
def test_frame_schedule_production_evenly_spaced():
    m = _driver()
    # 7200 steps (dt=1, T=7200), 12 frames → 11 post-IC snapshots, last = n_steps
    rec = m.frame_step_schedule(7200, 12)
    assert len(rec) == 11
    assert rec == sorted(set(rec))           # strictly increasing + distinct
    assert rec[-1] == 7200                    # final frame at T
    assert all(1 <= k <= 7200 for k in rec)


def test_frame_schedule_distinct_and_capped_when_dt_coarse():
    m = _driver()
    # only 3 steps but 6 frames requested → at most 3 distinct post-IC snapshots
    rec = m.frame_step_schedule(3, 6)
    assert rec == [1, 2, 3]                    # distinct, no duplicates/drops
    assert rec == sorted(set(rec))


def test_frame_schedule_last_is_nsteps():
    m = _driver()
    for n_steps, frames in [(100, 5), (7, 4), (14400, 12), (5, 10)]:
        rec = m.frame_step_schedule(n_steps, frames)
        assert rec[-1] == n_steps
        assert rec == sorted(set(rec))
