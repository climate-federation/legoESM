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
def test_real_independent_ladder_refuses_unmeasured_physics():
    with pytest.raises(ValueError, match="SWE three-stage"):
        lad.independent(ROOT, allow_dirty=True)


def test_stage_entry_reader_refuses_absent_acquisition(tmp_path):
    with pytest.raises(Exception, match="missing"):
        lad.read_stage_entries(tmp_path)


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_ens_stage1_is_unmeasured_until_production_stage_exists():
    clean = lad.ens_stage1(lad.DEFAULT_STAGE_ROOT, allow_dirty=True)
    assert not clean["production_stage_executed"]
    assert not clean["all_at_bar"] and not clean["all_bit_identical"]
    assert len(clean["per_kt"][0]["rows"]) == 13
    assert {row["status"] for row in clean["per_kt"][0]["rows"]} == {
        "UNMEASURED"}
    assert clean["first_kt_over_bar"] is None
    assert lad._exit_code(clean) == 1


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
@pytest.mark.parametrize("value", ("zero", "one"))
def test_stage_entry_one_ulp_input_plants_leave_bit_for_bit_bar(value):
    out = lad.ens_stage1(
        lad.DEFAULT_STAGE_ROOT, plant=value, allow_dirty=True)
    assert len(out["controls"]) == 1
    row = out["controls"][0]
    assert row["status"] == "DEBT"
    assert row["numerical_status"] == "AT-BAR"
    assert row["n_unequal"] == 1
    assert not row["bit_identical"]


def test_rows_use_card_t_u_v_masks_and_distinguish_wet_from_dry():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    masks = lad._card_masks(build_swg_card())
    assert {name: mask.shape for name, mask in masks.items()} == {
        "T": (84, 84), "U": (84, 84), "V": (84, 84)}
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
