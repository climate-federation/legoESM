"""Direct controls for the T1 standalone runner and ensemble classifier."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"
sys.path.insert(0, str(TOOLS))


def _load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load("standalone_20y")
SCORER = _load("standalone_20y_score")


def test_sample_schedule_is_exact_registered_grid():
    days = RUNNER.sample_days()
    assert len(days) == 75
    assert days[:15] == tuple(range(360, 5401, 360))
    assert days[15:] == tuple(range(5430, 7201, 30))
    assert len(set(days)) == 75


def test_member_seed_contract_fails_closed():
    assert [RUNNER.seed_for_member(member) for member in range(6)] == [
        None, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="member must be"):
        RUNNER.seed_for_member(6)


def test_standalone_builder_has_no_bridge_level_and_resolves_card_default():
    _, grid, z_coord, state, model_cfg, _, _, _, receipt = (
        RUNNER.build_standalone(0))
    assert (grid.n_lat, grid.n_lon) == (195, 48)
    assert model_cfg.outer_integrator == "leapfrog"
    assert (model_cfg.barotropic.barotropic_cold_start_after_reconcile
            == "nemo_mlf_baro_corr")
    assert all(getattr(state, name) is None for name in (
        "T_before", "S_before", "eta_before", "u_before", "v_before"))
    assert receipt["changed_channels"] == []
    assert np.asarray(state.T.data).dtype == np.float64
    for name in (
        "nemo_gdept_0", "nemo_gdepw_0", "nemo_e3t_0", "nemo_e3w_0",
        "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t", "nemo_e1e2u",
        "nemo_e1e2v", "nemo_e2u", "nemo_e1v", "nemo_een_barotropic",
    ):
        assert getattr(z_coord, name) is not None, name


def test_temperature_plant_changes_only_now_level_temperature():
    control = RUNNER.build_standalone(0)[3]
    perturbed, receipt = RUNNER.perturb_temperature(control, 3)
    assert receipt["changed_channels"] == ["T"]
    assert receipt["seed"] == 3
    assert RUNNER.state_hashes(perturbed)["T"] != RUNNER.state_hashes(control)["T"]
    for name in RUNNER.state_hashes(control):
        if name != "T":
            assert RUNNER.state_hashes(perturbed)[name] == RUNNER.state_hashes(control)[name]


def test_runner_surface_has_no_restart_or_bridge_selector():
    source = (TOOLS / "standalone_20y.py").read_text()
    assert 'add_argument("--run-traj"' not in source
    assert 'add_argument("--run-stepdump"' not in source
    assert 'add_argument("--restart"' not in source
    assert "bridge_paths\": []" in source
    assert "restart_paths\": []" in source


def test_claim_length_refuses_the_measured_ic_geometry_mismatch():
    """The kt=2 peel makes the old claim-length admission a hard stop."""
    assert RUNNER.claim_admission_reasons() == (
        RUNNER.IC_GEOMETRY_BLOCKER,)
    assert "wet mask" in RUNNER.IC_GEOMETRY_BLOCKER


def _synthetic_reducer_inputs(native=(9, 10, 6)):
    expected = RUNNER._expected_reducer_input_shapes(native)
    fields = {
        name: np.arange(np.prod(expected[name]), dtype=np.float64).reshape(
            expected[name])
        for name in ("T", "S", "eta", "u", "v")
    }
    land = np.ones(expected["land_mask"], dtype=np.float64)
    return fields, land


def test_standalone_reducer_convention_expands_to_recorded_native_frame():
    native = (9, 10, 6)
    fields, land = _synthetic_reducer_inputs(native)
    expanded, receipt = RUNNER.expand_standalone_reducer_inputs(
        fields, land, native)
    assert expanded["T"].shape == native
    assert expanded["S"].shape == native
    assert expanded["land_mask"].shape == native[:2]
    assert expanded["u"].shape == (native[0], native[1] + 1, native[2])
    np.testing.assert_array_equal(expanded["T"][2:-2, 2:-2, :-1], fields["T"])
    np.testing.assert_array_equal(
        expanded["u"][:, 1:native[1] + 1, :][2:-2, 2:-2, :-1],
        fields["u"][:, 1:, :])
    np.testing.assert_array_equal(expanded["T"][2:-2, :2, :-1],
                                  fields["T"][:, -2:, :])
    assert not expanded["T"][:, :, -1].any()
    assert receipt["asserted_input_names"] == [
        "S", "T", "eta", "land_mask", "u", "v"]


@pytest.mark.parametrize("name", ["T", "S", "eta", "u", "v", "land_mask"])
def test_standalone_reducer_day0_shape_gate_catches_every_input(name):
    native = (9, 10, 6)
    fields, land = _synthetic_reducer_inputs(native)
    if name == "land_mask":
        land = land[:, :-1]
    else:
        fields[name] = fields[name][..., :-1]
    with pytest.raises(ValueError, match=rf"\b{name}\b.*actual=.*expected="):
        RUNNER.expand_standalone_reducer_inputs(fields, land, native)


def test_classifier_plants_cover_all_frozen_branches():
    assert SCORER.classifier_self_test() == {
        "confirm": "CONFIRM",
        "refute": "REFUTE",
        "unresolved": "UNRESOLVED",
        "quantized": "UNRESOLVED_QUANTIZED",
        "family_maximum": "FIRED",
    }


def test_classifier_rejects_member_count_drift():
    with pytest.raises(SCORER.AdmissionError, match="exactly six"):
        SCORER.classify_family(
            {"x": np.arange(5.0)}, {"x": np.arange(5.0)}, {"x": 0.0},
            rng=np.random.default_rng(1))
