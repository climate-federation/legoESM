"""Direct, non-vacuous tests for the #1455 channel seasonal probe."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts/validate/ocean_fidelity/dino_1226/channel_seasonal.py"
sys.path.insert(0, str(PROBE.parent))
SPEC = importlib.util.spec_from_file_location("channel_seasonal", PROBE)
S = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(S)


def _patterns(seed=1455):
    return np.random.default_rng(seed).normal(
        size=(len(S.HORIZONS), S.C.N_ROWS))


def test_pinned_git_object_rejects_wrong_hash():
    data = S._git_object(S.AUDIT_COMMIT, S.AUDIT_PATH, S.AUDIT_SHA256)
    assert len(data) > 1000
    with pytest.raises(SystemExit, match="exact committed reuse"):
        S._git_object(S.AUDIT_COMMIT, S.AUDIT_PATH, "0" * 64)


def test_wind_signature_guard_rejects_time_argument():
    from legoesm.ocean.experiments.dino import dino_wind_stress

    assert "lat_deg" in S._wind_signature_guard(dino_wind_stress)

    def time_dependent(lat_deg, t_seconds):
        return lat_deg + t_seconds

    with pytest.raises(ValueError, match="time-dependent"):
        S._wind_signature_guard(time_dependent)


def test_power_case_is_autocorrelated_and_has_three_horizon_dof():
    target, driver = S.autocorrelated_phase_case(1455, 0.95)
    assert target.shape == (len(S.HORIZONS), S.C.N_ROWS)
    assert np.allclose(target.sum(axis=0), 0.0, atol=1e-12)
    assert np.allclose(driver.sum(axis=0), 0.0, atol=1e-12)
    effective = [S.C.effective_rows(target[t], driver[t])[0]
                 for t in range(len(S.HORIZONS))]
    assert min(effective) < S.C.N_ROWS


def test_driver_refute_label_is_downgraded_but_registered_status_retained():
    entry = {"status": "REFUTES_PHASE_TRACKING"}
    phase = {
        "status": "REFUTES_PHASE_TRACKING",
        "ensemble_mean": entry.copy(),
        "members": [entry.copy() for _ in range(S.N_MEM)],
        "member_statuses": ["REFUTES_PHASE_TRACKING"] * S.N_MEM,
    }
    result = S.relabel_driver_phase(phase)
    assert result["status"] == S.NO_DETECTED_CODE
    assert result["registered_status"] == "REFUTES_PHASE_TRACKING"
    assert all(status == S.NO_DETECTED_CODE
               for status in result["member_statuses"])


def test_phase_decider_refuses_constant_profile():
    constant = np.ones((len(S.HORIZONS), S.C.N_ROWS))
    with pytest.raises(ValueError, match="UNMEASURABLE"):
        S.phase_decider(constant, _patterns(), n_boot=10)


def test_phase_collection_requires_every_member_to_confirm(monkeypatch):
    pattern = np.broadcast_to(_patterns(), (S.N_MEM,) + _patterns().shape).copy()
    calls = iter([
        {"status": "CONFIRMS_PHASE_TRACKING"},
        {"status": "CONFIRMS_PHASE_TRACKING"},
        {"status": "CONFIRMS_PHASE_TRACKING"},
        {"status": "UNRESOLVED"},
        {"status": "CONFIRMS_PHASE_TRACKING"},
    ])
    monkeypatch.setattr(S, "phase_decider", lambda *args, **kwargs: next(calls))
    result = S.phase_collection(pattern, pattern, n_boot=1)
    assert result["status"] == "UNRESOLVED"


def test_proportionality_distinguishes_fixed_fraction_from_orthogonal():
    predictor = _patterns()
    exact = S.proportionality(0.2 * predictor, predictor)
    assert exact["beta"] == pytest.approx(0.2)
    assert exact["nrmse"] == pytest.approx(0.0, abs=1e-14)

    target = np.roll(predictor, 1, axis=1)
    miss = S.proportionality(target, predictor)
    assert miss["nrmse"] > S.PERSIST_HI


def test_density_reducer_is_volume_weighted_not_layer_averaged():
    wet = S.R.A.tmask.copy()
    rho = np.where(wet, 1027.0, np.nan)
    rho[:, :, 0] += 0.1
    official, stratification = S.density_profiles(rho, wet)
    layer = np.mean(np.where(wet[S.CHANNEL_ROWS], rho[S.CHANNEL_ROWS], 0.0),
                    axis=(1, 2))
    assert official.dtype == np.float64
    assert stratification.dtype == np.float64
    assert not np.allclose(official, layer)


def test_thermal_wind_reducer_moves_on_meridional_density_plant():
    wet = S.R.A.tmask.copy()
    rho = np.where(wet, 1027.0, np.nan)
    baseline = S.thermal_wind_profile(rho, wet)
    rho[S.CHANNEL_ROWS.start + 1] += 0.01
    planted = S.thermal_wind_profile(rho, wet)
    assert np.max(np.abs(planted - baseline)) > 0.0


def test_upstream_artifact_hash_and_gap_plant():
    artifact = S.upstream_artifact()
    expected = np.asarray([
        artifact["gaps_paired_sv"][str(day)] for day in S.HORIZONS
    ]).transpose(1, 0, 2)
    nemo = np.zeros_like(expected)
    lego = expected.copy()
    assert S.verify_upstream_gaps(lego, nemo, artifact) == pytest.approx(0.0)
    lego[0, 0, 0] += 1e-6
    with pytest.raises(AssertionError):
        S.verify_upstream_gaps(lego, nemo, artifact)


def test_saturation_summary_calls_shared_engine(monkeypatch):
    values = np.arange(S.N_MEM * len(S.HORIZONS) * 2, dtype=np.float64).reshape(
        S.N_MEM, len(S.HORIZONS), 2)
    calls = []

    def fake(history):
        calls.append(history)
        return False, "plant unsaturated"

    monkeypatch.setattr(S.R, "saturated", fake)
    result = S.saturation_summary(values, values + 1.0)
    assert result["unsaturated"] == 2
    assert len(calls) == 2
    assert all(row["reason"] == "plant unsaturated" for row in result["rows"])
