"""SWG restart/stage reader and fail-closed ladder tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg_ladder.py")
_S = importlib.util.spec_from_file_location("swg_ladder", _P)
lad = importlib.util.module_from_spec(_S); _S.loader.exec_module(lad)
ROOT = lad.DEFAULT_ROOT


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_restart_reader_admits_exactly_ten_after_states():
    out = lad.read_restarts(ROOT)
    assert list(out) == list(range(1, 11))
    assert set(out[1]) == set(lad.RESTART_FIELDS)
    assert all(a.shape == (84, 84) for a in out[1].values())


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_reader_self_check_is_exact_and_score_plant_leaves_bar():
    clean = lad.self_check(ROOT, allow_dirty=True)
    plant = lad.self_check(ROOT, plant=True, allow_dirty=True)
    assert clean["all_at_bar"] and clean["all_bit_identical"]
    assert not plant["all_at_bar"] and not plant["all_bit_identical"]


@pytest.mark.skipif(not ROOT.is_dir(), reason="SWG probe not mounted")
def test_real_independent_ladder_executes_normal_production_step():
    out = lad.independent(ROOT, allow_dirty=True)
    assert len(out["per_kt"]) == 10
    assert all(len(step["rows"]) == 5 for step in out["per_kt"])
    assert out["first_kt_over_bar"] == 6
    assert all(row["bit_identical"]
               for step in out["per_kt"][:5] for row in step["rows"])
    first = next(row for row in out["per_kt"][5]["rows"]
                 if row["status"] == "DEBT")
    assert first["name"] == "independent.kt6.un"
    assert first["n_unequal"] == 1


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_carried_state_audit_covers_every_stage_and_state_field():
    out = lad.carried_state_audit(lad.DEFAULT_STAGE_ROOT, allow_dirty=True)
    rows = out["per_kt"][0]["rows"]
    stage = [row for row in rows if ".stage3_to_kt6." in row["name"]]
    rebuilt = [row for row in rows if ".rebuilt_kt6." in row["name"]]
    assert len(stage) == len(rebuilt) == 13
    assert all(row["bit_identical"] for row in stage)
    disposition = out["state_disposition"]
    assert len(disposition["scored_public"]) == 5
    assert all(row["bit_identical"]
               for row in disposition["unchanged"].values())
    assert len(disposition["optional_none"]) > 0
    assert disposition["redundant_faces"]["u_west"]["n_nonzero"] == 0
    assert disposition["redundant_faces"]["v_south"]["n_nonzero"] == 0


def test_stage_entry_reader_refuses_absent_acquisition(tmp_path):
    with pytest.raises(Exception, match="missing"):
        lad.read_stage_entries(tmp_path)


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_all_given_stages_execute_production_jit_and_are_bit_exact():
    clean = lad.ens_stage1(lad.DEFAULT_STAGE_ROOT, allow_dirty=True)
    assert clean["production_stage_executed"]
    assert clean["all_at_bar"] and clean["all_bit_identical"]
    assert clean["nondiscriminating"][0]["status"] == "NON-DISCRIMINATING"
    assert len(clean["per_kt"][0]["rows"]) == 13
    assert {row["status"] for row in clean["per_kt"][0]["rows"]} <= {
        "AT-BAR", "DEBT"}
    assert all("max_abs" in row for row in clean["per_kt"][0]["rows"])
    assert all(row["max_abs"] == 0.0
               for step in clean["per_kt"] for row in step["rows"])
    assert lad._exit_code(clean) == 0


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_chained_stages_stay_bit_exact_through_the_record():
    clean = lad.swe_rk3_stages(
        lad.DEFAULT_STAGE_ROOT, mode="chained", allow_dirty=True)
    assert clean["all_at_bar"] and clean["all_bit_identical"]
    assert len(clean["per_kt"]) == 26
    assert all(row["max_abs"] == 0.0
               for step in clean["per_kt"] for row in step["rows"])
    assert lad._exit_code(clean) == 0


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_recorded_stage3_wind_and_whole_step_inputs_are_discriminating():
    entries = lad.read_stage_entries(lad.DEFAULT_STAGE_ROOT)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    audit = lad._stage3_input_audit(entries, lad._card_masks(build_swg_card()))
    assert all(row["bit_identical"]
               for row in audit["wind_before_vs_now"].values())
    assert all(not row["bit_identical"] and row["n_unequal"] > 0
               for row in audit["whole_step_entry_vs_stage3_entry"].values())


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_recorded_nonzero_stage_entry_one_ulp_plant_leaves_bar():
    out = lad.ens_stage1(
        lad.DEFAULT_STAGE_ROOT, plant="ulp", allow_dirty=True)
    assert len(out["controls"]) == 1
    row = out["controls"][0]
    assert row["status"] == "DEBT"
    assert row["n_unequal"] > 0
    assert not row["bit_identical"]
    assert row["input_before"] != 0.0
    assert row["input_after"] == np.nextafter(
        row["input_before"], np.float64(np.inf))
    assert row["input_ulp_delta"] == 1
    assert lad._exit_code(out) == 1


def test_rows_use_card_t_u_v_masks_and_distinguish_wet_from_dry():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    masks = lad._card_masks(build_swg_card())
    assert {name: mask.shape for name, mask in masks.items()} == {
        "T": (84, 84), "U": (84, 84), "V": (84, 84), "F": (84, 84)}
    assert all(mask.any() and not mask.all() for mask in masks.values())
    for kind, mask in masks.items():
        ref = np.zeros(mask.shape, dtype=np.float64)
        dry = ref.copy()
        dry[tuple(np.argwhere(~mask)[0])] = 1.0
        assert lad._row(f"plant.{kind}.dry", ref, dry, mask)["status"] == "AT-BAR"
        wet = ref.copy()
        wet[tuple(np.argwhere(mask)[0])] = 1.0
        assert lad._row(f"plant.{kind}.wet", ref, wet, mask)["status"] == "DEBT"


@pytest.mark.parametrize("field,kind", [("utau_b", "U"), ("vtau_b", "V")])
def test_before_stress_uses_native_stagger_and_wrong_stagger_plant_fails(field, kind):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    masks = lad._card_masks(build_swg_card())
    selected = lad._field_mask(field, masks)
    assert np.array_equal(selected, masks[kind])
    # The reviewer's reproduced defect selected T; 80 cells distinguish it
    # from each correct stagger, so assigning the old mask must go red.
    assert np.count_nonzero(masks["T"] != masks[kind]) == 80
    with pytest.raises(AssertionError):
        assert np.array_equal(masks["T"], masks[kind])


def test_numerical_debt_makes_the_ladder_gate_nonzero():
    clean = {"per_kt": [{"kt": 1, "rows": [
        {"status": "AT-BAR", "bit_identical": True}]}]}
    debt = {"per_kt": [{"kt": 1, "rows": [
        {"status": "DEBT", "bit_identical": False}]}]}
    assert lad._exit_code(lad._gate_rows(clean)) == 0
    gated = lad._gate_rows(debt)
    assert gated["first_kt_over_bar"] == 1
    assert lad._exit_code(gated) == 1
