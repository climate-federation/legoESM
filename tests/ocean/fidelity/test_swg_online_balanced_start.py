"""Direct tests for the SWG online balanced-start instrument."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity"
         / "testcases/nemo_testcase_l1_swg_online_balanced_start.py")
_SPEC = importlib.util.spec_from_file_location("swg_balanced_start", _PATH)
balanced = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(balanced)


def test_divergent_projection_recovers_closed_wall_gradient():
    n = 8
    spacing = 2.0
    y, x = np.mgrid[:n, :n]
    phi = (np.cos(np.pi * (x + 0.5) / n)
           * np.cos(2.0 * np.pi * (y + 0.5) / n))
    u = np.zeros((n + 2, n + 2))
    v = np.zeros_like(u)
    mask = np.zeros_like(u, dtype=bool)
    mask[1:-1, 1:-1] = True
    u[1:-1, 1:-2] = np.diff(phi, axis=1) / spacing
    v[1:-2, 1:-1] = np.diff(phi, axis=0) / spacing
    u_div, v_div = balanced.divergent_velocity_projection(
        u, v, mask, spacing)
    np.testing.assert_allclose(u_div, u, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(v_div, v, rtol=0.0, atol=2.0e-15)
    assert balanced.fast_divergent_energy(u, v, mask, spacing) > 0.0
    # A non-conservative wall-normal plant must change the projection.
    planted = u.copy()
    planted[1:-1, -2] = 1.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            balanced.divergent_velocity_projection(
                planted, v, mask, spacing)[0], planted,
            rtol=0.0, atol=2.0e-15)


def test_initialization_ratio_uses_distinct_shock_and_closure_errors():
    wet = np.ones((2, 2), dtype=bool)
    conservative = np.zeros((48, 2, 2))
    target = np.full_like(conservative, 2.0)
    raw = np.full_like(conservative, 3.0)
    ratios = balanced.initialization_ratios(
        raw, conservative, target, wet)
    assert set(ratios) == {"6", "24", "48"}
    assert ratios["24"]["initialization_shock_vorticity_rms_s-1"] == 3.0
    assert ratios["24"]["genuine_closure_vorticity_rms_s-1"] == 2.0
    assert ratios["24"]["shock_to_closure_ratio"] == 1.5
    # Planting equality between raw and conservative must collapse shock to zero.
    planted = balanced.initialization_ratios(
        conservative, conservative, target, wet)
    assert planted["24"]["shock_to_closure_ratio"] == 0.0


def test_registered_admission_detects_jump_and_fast_energy_transient():
    record = {
        "conservative_relative_vorticity_error_rms_s-1": np.linspace(
            1.0, 48.0, 48).tolist(),
        "conservative_fast_divergent_ke_m2_s-2": np.ones(48).tolist(),
        "conservative_ssh_tendency_rms_m_s": np.ones(48).tolist(),
    }
    admitted = balanced._state_admission(record, "conservative")
    assert admitted["admitted"]
    jump = dict(record)
    jump["conservative_relative_vorticity_error_rms_s-1"] = (
        np.linspace(1.0, 48.0, 48) * 1.0e-8).tolist()
    jump["conservative_relative_vorticity_error_rms_s-1"][0] = 3.0e-8
    assert not balanced._state_admission(jump, "conservative")["no_jump"]
    fast = dict(record)
    fast["conservative_fast_divergent_ke_m2_s-2"] = np.ones(48).tolist()
    fast["conservative_fast_divergent_ke_m2_s-2"][0] = 2.1
    assert not balanced._state_admission(
        fast, "conservative")["fast_energy_bounded"]


def test_balanced_start_cli_exits_nonzero_on_failed_admission(
        monkeypatch, tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({"records": []}))
    monkeypatch.setattr(
        balanced, "apply_registered_admission",
        lambda report, root: report | {"admission_summary": {
            "conservative_admitted_for_training": False,
            "conservative_pass_count": 9,
            "raw_pass_count": 0,
        }})
    assert balanced.main([
        "--existing-json", str(existing), "--hourly-root", str(tmp_path),
        "--json", str(tmp_path / "out.json")]) == 1


@pytest.mark.parametrize("count,expected", [(8, 0), (7, 1)])
def test_dual_operator_cli_applies_eight_of_ten_rule(
        monkeypatch, tmp_path, count, expected):
    monkeypatch.setattr(
        balanced, "run_dual_operator_experiment",
        lambda *args, **kwargs: {"summary": {
            "filtered_pass_count": count, "box_pass_count": 10,
            "selected_objective": "template" if count >= 8 else "symmetric"}})
    assert balanced.main([
        "--dual-operator", "--hourly-root", str(tmp_path),
        "--json", str(tmp_path / f"out{count}.json")]) == expected
