"""Direct synthetic tests for the SWG predictability instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "scripts/validate/"
          "ocean_fidelity/testcases/nemo_testcase_l1_swg_online_predictability.py")
SPEC = importlib.util.spec_from_file_location("swg_online_predictability", SCRIPT)
predictability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictability)


def test_first_efold_crossing_recovers_exact_exponential_and_partial_curve():
    days = np.arange(6, dtype=np.float64)
    tau = 2.5
    rms = 1.0e-6 * np.exp(days / tau)
    assert predictability.first_efold_crossing(days, rms, rms[0]) == pytest.approx(tau)
    assert predictability.first_efold_crossing(
        days[:2], rms[:2], rms[0]) is None
    planted = rms.copy()
    planted[1] = np.e * rms[0]
    assert predictability.first_efold_crossing(days, planted, rms[0]) == 1.0


def test_first_efold_crossing_refuses_nonpositive_curve():
    with pytest.raises(ValueError, match="positive finite"):
        predictability.first_efold_crossing(
            np.array([0.0, 1.0]), np.array([1.0, 0.0]), 1.0)


def test_balanced_fit_selects_first_persistently_balanced_lead():
    days = np.arange(1.0, 51.0)
    zeta = np.exp(days / 10.0)
    ratio = np.full(days.shape, 0.05)
    ratio[:17] = 0.5
    fit = predictability.balanced_fit(days, zeta, ratio * zeta)
    assert fit["start_day"] == 18.0
    assert fit["stop_day"] == 50.0
    assert fit["efolding_days"] == pytest.approx(10.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    # A late fast-mode relapse must move/refuse the window, not be ignored.
    ratio[-1] = 0.5
    with pytest.raises(ValueError, match="no balanced fit window"):
        predictability.balanced_fit(days, zeta, ratio * zeta)


def test_rotational_velocity_inversion_is_linear_and_wall_bounded():
    wet = np.zeros((10, 10), dtype=bool)
    wet[1:-1, 1:-1] = True
    y, x = np.mgrid[:10, :10]
    zeta = np.where(wet, np.sin(x) * np.cos(y), 0.0)
    u, v = predictability.rotational_velocity_from_vorticity(
        zeta, wet, 50_000.0)
    u2, v2 = predictability.rotational_velocity_from_vorticity(
        2.0 * zeta, wet, 50_000.0)
    np.testing.assert_allclose(u2, 2.0 * u, rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(v2, 2.0 * v, rtol=1.0e-14, atol=0.0)
    assert np.all(u[~wet] == 0.0) and np.all(v[~wet] == 0.0)
    assert np.any(u[wet] != 0.0) and np.any(v[wet] != 0.0)


def test_tangent_decision_exits_nonzero_when_probe_is_inert(monkeypatch, tmp_path):
    monkeypatch.setattr(
        predictability, "forward_tangent_probe",
        lambda *args, **kwargs: {"steps": 1, "tangent_nonzero": False})
    assert predictability.main([
        "--tangent-restart", str(tmp_path / "restart.npz"),
        "--json", str(tmp_path / "out.json")]) == 1
