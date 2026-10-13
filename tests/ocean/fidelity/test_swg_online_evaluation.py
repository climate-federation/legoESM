"""Direct tests for the online-closure held-out scorecard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l1_swg_online_evaluation.py")
SPEC = importlib.util.spec_from_file_location("swg_online_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation)


def _controls():
    return {
        "nu10": {
            "mean_transport_rmse_m2_s": 2.0,
            "ke_spectrum_relative_l1": 0.4,
            "wet_eke_map_nrmse": 0.5,
            "basin_eke_ratio": 0.7,
            "top_octave_ke_fraction": 0.18,
            "top_octave_enstrophy_fraction": 0.75,
            "native_checkerboard_index": 0.01,
        },
        "nu80": {
            "mean_transport_rmse_m2_s": 3.0,
            "ke_spectrum_relative_l1": 0.3,
            "wet_eke_map_nrmse": 0.6,
            "basin_eke_ratio": 0.38,
            "top_octave_ke_fraction": 0.15,
            "top_octave_enstrophy_fraction": 0.72,
            "native_checkerboard_index": 0.02,
        },
    }


def test_gate_requires_three_seeds_and_all_distinct_constraints():
    controls = _controls()
    good = {
        "mean_transport_rmse_m2_s": 1.0,
        "ke_spectrum_relative_l1": 0.2,
        "wet_eke_map_nrmse": 0.4,
        "basin_eke_ratio": 1.0,
        "top_octave_ke_fraction": 0.14,
        "top_octave_enstrophy_fraction": 0.70,
        "native_checkerboard_index": 0.01,
    }
    assert evaluation.held_out_gate([good] * 3, controls)["pass"]
    with pytest.raises(ValueError, match="three"):
        evaluation.held_out_gate([good] * 2, controls)
    for key, bad_value in (
            ("basin_eke_ratio", 0.79),
            ("top_octave_ke_fraction", 0.19),
            ("mean_transport_rmse_m2_s", 2.1)):
        planted = dict(good, **{key: bad_value})
        assert not evaluation.held_out_gate([planted] * 3, controls)["pass"]


def test_registered_adversarial_plants_both_fail():
    planted = evaluation.adversarial_plants(_controls())
    assert not planted["collapsed_variance"]["pass"]
    assert not planted["two_cell_grid_noise"]["pass"]
    assert planted["collapsed_variance"]["plant"]["collapsed_eke_ratio"] == 0.0
    assert planted["two_cell_grid_noise"]["plant"]["checkerboard_index"] == 1.0
