"""Unit tests for the θ-consistent D7 significance ranking (pure logic).

The re-scoring wrapper (rescore_theta) runs a real SCM and is exercised end-to-end by the
campaign; here we test the pure rank_theta_significance grouping/gating with synthetic
records so it is fast + deterministic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite" / \
    "theta_significance.py"
_spec = importlib.util.spec_from_file_location("theta_significance", _MOD)
ts = importlib.util.module_from_spec(_spec)
sys.modules["theta_significance"] = ts  # dataclass() looks the module up by name
_spec.loader.exec_module(ts)


def _rec(scheme, theta_rmse, q0=0.06, regime="dry_convective"):
    return {"scheme": scheme, "theta_rmse": theta_rmse, "q0": q0,
            "regime": regime, "artifact": f"cbl__{q0}"}


def test_ranks_ascending_by_theta_and_flags_significant_margin():
    recs = [_rec("holtslag", 0.20), _rec("smag", 0.10), _rec("louis", 0.15)]
    out = ts.rank_theta_significance(recs, sigma_theta=0.02)
    assert len(out) == 1
    rk = out[0]
    assert [s for s, _ in rk.ranked] == ["smag", "louis", "holtslag"]  # ascending θ_rmse
    assert rk.top_margin == pytest.approx(0.05)  # 0.15 - 0.10
    assert rk.top_significant is True  # 0.05 > σ_LES(θ)=0.02


def test_margin_below_sigma_is_not_significant():
    recs = [_rec("a", 0.100), _rec("b", 0.108)]
    out = ts.rank_theta_significance(recs, sigma_theta=0.02)
    assert out[0].top_margin < 0.02
    assert out[0].top_significant is False


def test_no_sigma_theta_leaves_significance_unknown():
    recs = [_rec("a", 0.10), _rec("b", 0.20)]
    out = ts.rank_theta_significance(recs, sigma_theta=None)
    assert out[0].top_margin == pytest.approx(0.10)
    assert out[0].top_significant is None


def test_duplicate_scheme_in_slice_collapses_to_lowest():
    # a stale re-tune of the same closure must not double-count; keep the lower θ_rmse
    recs = [_rec("smag", 0.20), _rec("smag", 0.12), _rec("louis", 0.15)]
    out = ts.rank_theta_significance(recs, sigma_theta=0.01)
    ranked = dict(out[0].ranked)
    assert ranked["smag"] == 0.12
    assert [s for s, _ in out[0].ranked] == ["smag", "louis"]


def test_groups_by_flux_and_sorts_ascending():
    recs = [_rec("a", 0.3, q0=0.12), _rec("b", 0.1, q0=0.02),
            _rec("a", 0.2, q0=0.02)]
    out = ts.rank_theta_significance(recs, sigma_theta=0.05)
    assert [rk.q0 for rk in out] == [0.02, 0.12]  # ascending flux
    assert [s for s, _ in out[0].ranked] == ["b", "a"]  # 0.02 slice ranked


def test_records_without_theta_rmse_are_skipped():
    recs = [_rec("a", 0.1), {"scheme": "b", "q0": 0.06, "regime": "dry_convective"}]
    out = ts.rank_theta_significance(recs, sigma_theta=0.01)
    assert [s for s, _ in out[0].ranked] == ["a"]  # 'b' (no θ_rmse) dropped


def test_single_closure_slice_has_no_margin():
    out = ts.rank_theta_significance([_rec("solo", 0.1)], sigma_theta=0.01)
    assert out[0].top_margin is None
    assert out[0].top_significant is None


def test_split_trusted_gates_on_self_check():
    # precision-gate (codex finding): only self_check_ok is True is trusted; a mismatch
    # (False) or an unverifiable record (None, no stored best_loss) is excluded, not ranked.
    enriched = [
        {"scheme": "ok", "self_check_ok": True},
        {"scheme": "mismatch", "self_check_ok": False},
        {"scheme": "unverifiable", "self_check_ok": None},
        {"scheme": "absent"},  # no key at all -> untrusted
    ]
    trusted, untrusted = ts.split_trusted(enriched)
    assert [e["scheme"] for e in trusted] == ["ok"]
    assert {e["scheme"] for e in untrusted} == {"mismatch", "unverifiable", "absent"}
