from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate"))
import crm_mixed_precision_accuracy as acc  # noqa: E402


def _ckpt(scale=1.0, rho_shift=0.0, q_shift=0.0):
    rng = np.random.default_rng(0)
    shape = (4, 4, 3)
    tr = np.full(shape + (11,), 1e-3)
    tr[..., 0] += q_shift
    return {"u": scale * rng.normal(size=shape), "v": rng.normal(size=shape),
            "w": scale * rng.normal(size=shape), "theta_prime": scale * rng.normal(size=shape),
            "rho_prime": scale * rng.normal(size=shape) * 1e-3 + rho_shift, "tracers": tr}


def test_identical_runs_pass_with_zero_error():
    rho_ref, dz = np.array([1.0, 0.8, 0.6]), np.array([100.0, 100.0, 100.0])
    res = acc.compare(_ckpt(), _ckpt(), rho_ref, dz)
    assert res["pass"]
    assert res["theta_prime"] == 0.0 and res["water_rel_diff"] == 0.0


def test_field_error_above_threshold_fails():
    rho_ref, dz = np.array([1.0, 0.8, 0.6]), np.array([100.0, 100.0, 100.0])
    res = acc.compare(_ckpt(), _ckpt(scale=1.01), rho_ref, dz)
    assert not res["pass"]
    assert res["theta_prime"] == pytest.approx(0.01)


def test_budget_differences_are_relative():
    rho_ref, dz = np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])
    ref = _ckpt()
    run = dict(ref, rho_prime=ref["rho_prime"] + 0.01)
    res = acc.compare(ref, run, rho_ref, dz)
    m_ref = (rho_ref + ref["rho_prime"]).sum()
    assert res["dry_mass_rel_diff"] == pytest.approx(0.01 * ref["rho_prime"].size / m_ref)


def test_nonfinite_run_fails():
    rho_ref, dz = np.array([1.0, 0.8, 0.6]), np.array([100.0, 100.0, 100.0])
    run = _ckpt()
    run["tracers"] = run["tracers"].copy()
    run["tracers"][0, 0, 0, 0] = np.nan
    res = acc.compare(_ckpt(), run, rho_ref, dz)
    assert not res["finite"] and not res["pass"]


def test_zero_reference_field_raises():
    with pytest.raises(ValueError, match="identically zero"):
        acc.relative_rms(np.ones(3), np.zeros(3))


def test_budget_violations_fail_even_with_matching_fields():
    """Planted violations: fields within thresholds, but water removed or dry
    mass shifted beyond the budget bars must fail."""
    rho_ref, dz = np.array([1.0, 0.8, 0.6]), np.array([100.0, 100.0, 100.0])
    ref = _ckpt()
    dry = dict(ref, tracers=np.zeros_like(ref["tracers"]))
    res = acc.compare(ref, dry, rho_ref, dz)
    assert res["water_rel_diff"] == pytest.approx(-1.0) and not res["pass"]
    wet = dict(ref, tracers=ref["tracers"] * (1.0 + 1e-7))
    assert not acc.compare(ref, wet, rho_ref, dz)["pass"]
    shifted = dict(ref, rho_prime=ref["rho_prime"] + 1e-9)
    res = acc.compare(ref, shifted, rho_ref, dz)
    assert abs(res["dry_mass_rel_diff"]) > 1e-11 and not res["pass"]


def test_time_gate_needs_equal_times_of_at_least_one_hour():
    assert acc.time_gate(3600.0, 3600.0)
    assert acc.time_gate(7200.0, 7200.0 + 1e-9)
    assert not acc.time_gate(3600.0, 3594.0)       # different times
    assert not acc.time_gate(1800.0, 1800.0)       # short of 1 h
    assert not acc.time_gate(None, 3600.0)          # no time recorded
