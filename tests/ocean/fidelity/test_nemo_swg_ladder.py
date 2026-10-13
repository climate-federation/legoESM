"""SWG restart/stage reader and fail-closed ladder tests."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg_ladder.py")
_S = importlib.util.spec_from_file_location("swg_ladder", _P)
lad = importlib.util.module_from_spec(_S)
_S.loader.exec_module(lad)
ROOT = lad.DEFAULT_ROOT
MONTH_ROOT = lad.DEFAULT_MONTH_ROOT
DG50_STAGE_ROOT = lad.DG50_STAGE_ROOT
DG50_MONTH_ROOT = lad.DG50_MONTH_ROOT


def test_maddison_deck_selects_admitted_card_clocks_and_roots():
    stage = lad._deck_namelist("maddison")
    month = lad._deck_namelist("maddison", month=True)
    assert (stage.nn_SWG, stage.rn_Dt_s, stage.nn_itend,
            stage.nn_stock, stage.nn_write) == (16, 50.0, 10, 1, 1)
    assert (month.nn_SWG, month.rn_Dt_s, month.nn_itend,
            month.nn_stock, month.nn_write) == (4, 200.0, 12960, 432, 432)
    assert lad.DG_MONTH_RECORD_KTS == tuple(range(1, 12530, 432))
    assert lad.DG_MONTH_RESTART_KTS == tuple(range(432, 12961, 432))
    with pytest.raises(Exception, match="unknown SWG deck"):
        lad._deck_namelist("typo")


def test_dg50_deck_selects_preregistered_stage_and_month_clocks():
    stage = lad._deck_namelist("dg50")
    month = lad._deck_namelist("dg50", month=True)
    assert (stage.nn_SWG, stage.rn_Dt_s, stage.nn_itend,
            stage.nn_stock, stage.nn_write) == (2, 400.0, 10, 1, 1)
    assert (month.nn_SWG, month.rn_Dt_s, month.nn_itend,
            month.nn_stock, month.nn_write) == (2, 400.0, 6480, 216, 216)
    physical = set(stage._fields) - {"nn_itend", "nn_stock", "nn_write"}
    fine = lad._deck_namelist("maddison")
    differing = {name for name in physical if getattr(stage, name) != getattr(fine, name)}
    assert differing == {"nn_SWG", "rn_Dt_s"}


@pytest.mark.skipif(not DG50_STAGE_ROOT.is_dir(),
                    reason="admitted dg50 stage records not mounted")
def test_dg50_stage_admission_pins_records_restarts_and_outputs():
    pins = lad._pinned_stage_hashes(DG50_STAGE_ROOT)
    assert len(pins) == 30
    records = lad.read_restarts(
        DG50_STAGE_ROOT, namelist=lad._deck_namelist("dg50"))
    assert list(records) == list(range(1, 11))
    assert all(value.shape == (84, 84) for value in records[1].values())


@pytest.mark.skipif(not DG50_MONTH_ROOT.is_dir(),
                    reason="admitted dg50 month records not mounted")
def test_dg50_month_admission_reports_registered_numerical_debt():
    out = lad.month(DG50_MONTH_ROOT, deck="dg50", allow_dirty=True)
    assert out["record_counts"] == {"stage_entries": 30, "restarts": 30}
    assert len(out["per_kt"]) == 30
    rows = [row for day in out["per_kt"] for row in day["rows"]]
    assert len(rows) == 690
    assert sum(row["exact"] for row in rows) == 339
    assert sum(row["exact"] or row.get("numerical_status") == "AT-BAR"
               for row in rows) == 561
    first_numerical_debt = next(
        day["day"] for day in out["per_kt"]
        if any(not row["exact"]
               and row.get("numerical_status") != "AT-BAR"
               for row in day["rows"]))
    assert first_numerical_debt == 9
    assert max(row["max_abs"] for row in rows) == pytest.approx(
        9.478529072737274e-15)
    assert not out["all_at_bar"] and not out["all_bit_identical"]


def test_exact_row_gate_distinguishes_signed_zero_bits():
    reference = np.array([0.0, 1.0], dtype=np.float64)
    candidate = np.array([-0.0, 1.0], dtype=np.float64)
    row = lad._row("signed-zero-plant", reference, candidate,
                   np.ones(2, dtype=bool))
    assert not row["bit_identical"]
    assert row["n_unequal"] == 1
    assert row["first_unequal_cell"] == [0]
    assert row["max_abs"] == 0.0
    assert row["status"] == "DEBT"


def test_localized_comparison_separates_interior_wall_and_storage_debts():
    wet = np.zeros((5, 5), dtype=bool)
    wet[1:4, 1:4] = True
    reference = np.zeros((5, 5), dtype=np.float64)
    candidate = reference.copy()
    candidate[2, 2] = 1.0
    candidate[1, 1] = 2.0
    candidate[0, 0] = 3.0
    out = lad._localized_comparison(reference, candidate, wet)
    assert out["wet_interior"]["bit_unequal"] == 1
    assert out["wet_wall_adjacent"]["bit_unequal"] == 1
    assert out["dry_or_storage_halo"]["bit_unequal"] == 1
    signed = reference.copy()
    signed[0, 1] = -0.0
    zero = lad._localized_comparison(reference, signed, wet)
    assert zero["dry_or_storage_halo"]["signed_zero_only"] == 1


@pytest.mark.skipif(not lad.DG_STAGE_ROOT.is_dir(),
                    reason="admitted SWG_DG stage records not mounted")
def test_maddison_localization_runs_production_steps_and_keeps_bar_unchanged():
    out = lad.localize_maddison_debts(
        lad.DG_STAGE_ROOT, lad.DG_MONTH_ROOT, include_month=False,
        allow_dirty=True)
    assert out["bar_unchanged"] == "IEEE-word identity for every field"
    assert len(out["stage"]["from_rest_boundaries"]) == 2
    assert set(out["stage"]["forcing_representation"]) == {"utau", "vtau"}


@pytest.mark.skipif(not lad.DG_STAGE_ROOT.is_dir(),
                    reason="admitted SWG_DG stage records not mounted")
def test_maddison_admission_and_restart_reader_use_run_resolution():
    pins = lad._pinned_stage_hashes(lad.DG_STAGE_ROOT)
    assert len(pins) == 30
    records = lad.read_restarts(
        lad.DG_STAGE_ROOT, kts=tuple(range(1, 11)),
        namelist=lad._deck_namelist("maddison"))
    assert set(records[1]) == set(lad.RESTART_FIELDS)
    assert all(value.shape == (644, 644) for value in records[1].values())
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    state = lad._state_from_restart(
        build_swg_card(lad._deck_namelist("maddison")).recipe.initial_state,
        records[1])
    rebuilt = lad._restart_candidate(state)
    assert all(np.array_equal(rebuilt[name], records[1][name])
               for name in lad.RESTART_FIELDS)


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
    assert out["first_kt_over_bar"] is None
    assert out["all_at_bar"] and out["all_bit_identical"]
    assert all(row["bit_identical"]
               for step in out["per_kt"] for row in step["rows"])


@pytest.mark.skipif(not lad.DEFAULT_STAGE_ROOT.is_dir(),
                    reason="admitted SWG stage records not mounted")
def test_carried_state_audit_covers_every_stage_and_state_field():
    out = lad.carried_state_audit(lad.DEFAULT_STAGE_ROOT, allow_dirty=True)
    rows = out["per_kt"][0]["rows"]
    stage = [row for row in rows if ".stage3_to_kt6." in row["name"]]
    rebuilt = [row for row in rows if ".rebuilt_kt6." in row["name"]]
    whole_step6 = [row for row in rows if ".whole_step6." in row["name"]]
    assert len(stage) == len(rebuilt) == 13
    assert len(whole_step6) == 3
    assert all(row["bit_identical"] for row in stage)
    assert all(row["bit_identical"] for row in rebuilt + whole_step6)
    assert out["all_at_bar"] and out["all_bit_identical"]
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


def test_composition_summary_keeps_exact_and_numerical_counts_separate():
    out = lad._gate_rows({"per_kt": [
        {"day": 1, "kt": 216, "rows": [
            {"status": "AT-BAR", "bit_identical": True, "max_abs": 0.0},
            {"status": "DEBT", "numerical_status": "AT-BAR",
             "bit_identical": False, "max_abs": 5.0e-16},
        ]},
        {"day": 2, "kt": 432, "rows": [
            {"status": "DEBT", "bit_identical": False,
             "max_abs": 2.0e-15},
        ]},
    ]})
    summary = lad._composition_summary(out)
    assert summary == {
        "rows": 3,
        "exact_rows": 1,
        "rows_within_1e-15": 2,
        "maximum_absolute_difference": 2.0e-15,
        "first_day_not_bit_identical": 1,
        "first_day_over_1e-15": 2,
    }
    with pytest.raises(ValueError, match="unknown month composition"):
        lad.month_composition(Path("unused"), composition="nested",
                              allow_dirty=True)


def test_ulp_distance_orders_signed_zero_and_adjacent_floats():
    assert lad._ulp_distance(np.float64(-0.0), np.float64(0.0)) == 1
    one = np.float64(1.0)
    assert lad._ulp_distance(one, np.nextafter(one, np.inf)) == 1
    minus_one = np.float32(-1.0)
    assert lad._ulp_distance(
        minus_one, np.nextafter(minus_one, np.float32(-np.inf))) == 1


def test_month_reader_refuses_absent_admission(tmp_path):
    with pytest.raises(Exception, match="missing month admission"):
        lad._month_admission(tmp_path)


def test_content_pin_refuses_one_flipped_record_byte(tmp_path):
    record = tmp_path / "oracle_swgstage_kt00000001_s1.bin"
    original = b"admitted-stage-record"
    record.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    lad._require_sha256(record, expected)
    planted = bytearray(original)
    planted[-1] ^= 1
    record.write_bytes(planted)
    with pytest.raises(Exception, match="content SHA-256 mismatch"):
        lad._require_sha256(record, expected)


def test_shipped_admission_pins_manifest_and_every_oracle_file(
        tmp_path, monkeypatch):
    payloads = {
        "nemo": b"binary", "namelist_cfg": b"cfg", "namelist_ref": b"ref",
        "T.nc": b"T", "U.nc": b"U", "V.nc": b"V", "W.nc": b"W",
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    manifest_pins = {
        name: hashlib.sha256(payloads[name]).hexdigest()
        for name in ("nemo", "namelist_cfg", "namelist_ref")
    }
    manifest = "".join(
        f"{digest}  {name}\n" for name, digest in manifest_pins.items())
    (tmp_path / "manifest.sha256").write_text(manifest)
    file_pins = {"manifest.sha256": hashlib.sha256(
        manifest.encode()).hexdigest()}
    file_pins.update({
        f"grid_{kind}_{kind}.nc": hashlib.sha256(kind.encode()).hexdigest()
        for kind in "TUVW"
    })
    for kind in "TUVW":
        (tmp_path / f"grid_{kind}_{kind}.nc").write_bytes(kind.encode())
    monkeypatch.setattr(lad, "SHIPPED_MANIFEST", manifest_pins)
    monkeypatch.setattr(lad, "SHIPPED_SHA256", file_pins)
    admitted = lad._shipped_admission(tmp_path)
    assert set(admitted) == set("TUVW")
    (tmp_path / "grid_U_U.nc").write_bytes(b"u")
    with pytest.raises(Exception, match="content SHA-256 mismatch"):
        lad._shipped_admission(tmp_path)


def test_ioipsl_running_mean_preserves_source_association_at_storage():
    first = np.array([6100058474.907604], dtype=np.float64)
    third = np.array([6276331050.18479], dtype=np.float64)
    average = lad._ioipsl_moycum(None, first, 0)
    average = lad._ioipsl_moycum(average, first, 1)
    source_associated = lad._ioipsl_moycum(average, third, 2)
    reassociated = average + (third - average) / np.float64(3)
    assert source_associated[0] == np.float64(6158816000.0)
    assert source_associated.astype(np.float32)[0] != reassociated.astype(
        np.float32)[0]


def test_stored_row_checks_fp32_bits_but_uses_the_fp64_numerical_gate():
    reference = np.ones((2, 2), dtype=np.float32)
    exact = lad._stored_row("stored.exact", reference, reference.copy(),
                            np.ones((2, 2), dtype=bool))
    moved = reference.copy()
    moved[0, 0] = np.nextafter(moved[0, 0], np.float32(np.inf))
    debt = lad._stored_row("stored.ulp", reference, moved,
                           np.ones((2, 2), dtype=bool))
    assert exact["status"] == "AT-BAR" and exact["bit_identical"]
    assert debt["status"] == "DEBT" and not debt["bit_identical"]
    assert debt["stored_dtype"] == "float32"


def test_shared_shipped_statistics_are_deterministic_for_equal_series():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_swg_card()
    shape = (2, 84, 84)
    zeros = np.zeros(shape, dtype=np.float32)
    arrays1, summary1 = lad._shared_shipped_statistics(
        zeros, zeros, zeros, card)
    arrays2, summary2 = lad._shared_shipped_statistics(
        zeros.copy(), zeros.copy(), zeros.copy(), card)
    assert summary1 == summary2
    assert all(np.array_equal(arrays1[name], arrays2[name])
               for name in arrays1)
    assert summary1["transport_streamfunction_sv"]["min"] == 0.0
    assert summary1["kinetic_energy_m5_s-2"]["time_mean"] == 0.0


@pytest.mark.skipif(not MONTH_ROOT.is_dir(),
                    reason="admitted SWG month record not mounted")
def test_independent_thirty_day_rung_scores_every_daily_record():
    out = lad.month(MONTH_ROOT, allow_dirty=True)
    assert out["record_counts"] == {"stage_entries": 30, "restarts": 30}
    assert len(out["per_kt"]) == 30
    assert all(len(day["rows"]) == 23 for day in out["per_kt"])
    assert sum(len(day["rows"]) for day in out["per_kt"]) == 690
    assert out["first_day_over_bar"] is None
    assert out["first_kt_over_bar"] is None
    assert out["all_at_bar"] and out["all_bit_identical"]
    assert all(row["max_abs"] == 0.0
               for day in out["per_kt"] for row in day["rows"])


@pytest.mark.skipif(not MONTH_ROOT.is_dir(),
                    reason="admitted SWG month record not mounted")
def test_month_recorded_nonzero_input_one_ulp_plant_leaves_bar():
    out = lad.month_control(MONTH_ROOT, allow_dirty=True)
    assert all(row["bit_identical"] for row in out["per_kt"][0]["rows"])
    assert len(out["controls"]) == 1
    row = out["controls"][0]
    assert row["status"] == "DEBT"
    assert row["n_unequal"] > 0
    assert row["input_before"] != 0.0
    assert row["input_after"] == np.nextafter(
        row["input_before"], np.float64(np.inf))
    assert row["input_ulp_delta"] == 1
    assert row["input_selection"] == "largest_absolute_nonzero_wet_u"
    assert row["input_matches_admitted_boundary"] is True
    assert lad._exit_code(out) == 1
