"""Direct tests for the online-closure held-out scorecard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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
            "mean_face_transport_rmse_m3_s": 2.0,
            "ke_spectrum_relative_l1": 0.4,
            "wet_eke_map_nrmse": 0.5,
            "basin_eke_ratio": 0.7,
            "top_octave_ke_fraction": 0.18,
            "top_octave_enstrophy_fraction": 0.75,
            "native_grid_noise_index": 0.01,
        },
        "nu80": {
            "mean_face_transport_rmse_m3_s": 3.0,
            "ke_spectrum_relative_l1": 0.3,
            "wet_eke_map_nrmse": 0.6,
            "basin_eke_ratio": 0.38,
            "top_octave_ke_fraction": 0.15,
            "top_octave_enstrophy_fraction": 0.72,
            "native_grid_noise_index": 0.02,
        },
    }


def test_gate_requires_three_seeds_and_all_distinct_constraints():
    controls = _controls()
    good = {
        "mean_face_transport_rmse_m3_s": 1.0,
        "ke_spectrum_relative_l1": 0.2,
        "wet_eke_map_nrmse": 0.3,
        "basin_eke_ratio": 1.0,
        "top_octave_ke_fraction": 0.004,
        "top_octave_enstrophy_fraction": 0.05,
        "native_grid_noise_index": 0.4,
        "physical_health": {"passes": True},
    }
    assert evaluation.held_out_gate([good] * 3, controls)["pass"]
    with pytest.raises(ValueError, match="three"):
        evaluation.held_out_gate([good] * 2, controls)
    for key, bad_value in (
            ("basin_eke_ratio", 0.79),
            ("top_octave_ke_fraction",
             1.01 * evaluation.HELD_OUT_THRESHOLDS[
                 "top_octave_ke_fraction"]),
            ("mean_face_transport_rmse_m3_s", 1.5e6)):
        planted = dict(good, **{key: bad_value})
        assert not evaluation.held_out_gate([planted] * 3, controls)["pass"]

    physically_failed = dict(good, physical_health={
        "passes": False, "all_finite": False,
        "minimum_total_layer_thickness_m": -1.0, "maximum_courant": 2.0})
    result = evaluation.held_out_gate([physically_failed] * 3, controls)
    assert not result["pass"]
    assert not result["physical_health_pass"]


def test_registered_adversarial_plants_both_fail():
    planted = evaluation.adversarial_plants(_controls())
    assert not planted["collapsed_variance"]["pass"]
    assert not planted["two_cell_grid_noise"]["pass"]
    assert not planted["removes_grid_noise_only"]["pass"]
    assert not planted["adds_eddy_energy_only"]["pass"]
    assert planted["collapsed_variance"]["plant"]["collapsed_eke_ratio"] == 0.0
    assert planted["two_cell_grid_noise"]["plant"]["grid_noise_index"] > 2.8


def test_arbitrary_native_arrays_use_same_score_and_health_path():
    card = evaluation.build_swg_card(evaluation.SWG_DG50_NAMELIST)
    shape = tuple(card.recipe.land_mask.shape)
    rng = np.random.default_rng(9)
    ssh = rng.normal(scale=1.0e-3, size=(3,) + shape)
    u = rng.normal(scale=1.0e-2, size=(3,) + shape)
    v = rng.normal(scale=1.0e-2, size=(3,) + shape)
    score = evaluation.score_native_trajectory((ssh, u, v), (ssh, u, v))
    assert score["mean_face_transport_rmse_m3_s"] == 0.0
    assert score["ke_spectrum_relative_l1"] == 0.0
    assert score["wet_eke_map_nrmse"] == 0.0
    assert score["basin_eke_ratio"] == 1.0
    assert score["physical_health"]["passes"]


def test_native_face_phase_flipped_checkerboard_fails_real_scorer():
    card = evaluation.build_swg_card(evaluation.SWG_DG50_NAMELIST)
    shape = tuple(card.recipe.land_mask.shape)
    rng = np.random.default_rng(2026)
    truth = (
        rng.normal(scale=1.0e-3, size=(4,) + shape),
        rng.normal(scale=1.0e-2, size=(4,) + shape),
        rng.normal(scale=1.0e-2, size=(4,) + shape),
    )
    jj, ii = np.indices(shape)
    checker = 0.1 * np.where((ii + jj) % 2, -1.0, 1.0)
    phases = np.asarray([1.0, -1.0, 1.0, -1.0])[:, None, None]
    candidate = (truth[0], truth[1] + phases * checker,
                 truth[2] - phases * checker)
    score = evaluation.score_native_trajectory(candidate, truth)
    assert score["native_grid_noise_index"] \
        > evaluation.HELD_OUT_THRESHOLDS["native_grid_noise_index"]
    assert not evaluation.held_out_gate(
        [score, score, score], _controls())["pass"]
