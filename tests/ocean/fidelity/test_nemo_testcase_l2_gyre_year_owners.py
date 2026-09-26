"""Direct tests for the GYRE day-30 OWNER harness.

Every assertion is paired with a synthetic violation that must FAIL, so none
of them can pass vacuously.  The harness is loaded by path because it is a
script, not an installed module.

The forcing-gate and switch-trace modes need the NEMO record on /data, so the
tests that touch them are skipped when it is absent; the STATEMENT-level
properties they rest on -- that the literal transcription's new arguments
change nothing at their defaults, and that each plant moves a number -- are
tested unconditionally, because those are the parts that can rot.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HARNESS = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
           / "nemo_testcase_l2_gyre_year_owners.py")
RUN_SH = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
          / "nemo_testcase_l2_gyre_earlydays" / "run.sh")
NEMO_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                   "year_fromrest/nemo_seed0")
PROCESS_CARD = (ROOT / "scripts" / "validate" / "ocean_fidelity"
                / "testcases"
                / "nemo_testcase_l2_gyre_round123_process_budget")
VERTICAL_CARD = (ROOT / "scripts" / "validate" / "ocean_fidelity"
                 / "testcases"
                 / "nemo_testcase_l2_gyre_round125_vertical_decomposition")
VERTICAL_EARLY_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/"
    "oracle_process_budget/oracle_trazdf_matrix_kt00000001.bin")


@pytest.fixture(scope="module")
def harness():
    assert HARNESS.is_file(), HARNESS
    spec = importlib.util.spec_from_file_location("gyre_year_owners", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gyre_year_owners", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def literal(harness):
    return harness._round16()._literal_sbc


def test_self_check_passes_as_a_subprocess():
    """The harness's own self-check is the gate; run it the way CI would."""
    result = subprocess.run([sys.executable, str(HARNESS), "--self-check"],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "self-check: all checks passed" in result.stdout


def test_round125_vertical_layout_and_card_are_fail_closed(harness):
    assert harness.VERTICAL_RECORD_BYTES == 6_965_416
    assert len(harness.VERTICAL_RECORD_STEPS) == 362
    assert harness.VERTICAL_RECORD_STEPS[:2] == (1, 2)
    assert harness.VERTICAL_RECORD_STEPS[2:] == tuple(range(1081, 1441))
    assert 360 * harness.VERTICAL_RECORD_BYTES == 2_507_549_760
    assert 362 * harness.VERTICAL_RECORD_BYTES == 2_521_480_592

    patch = (VERTICAL_CARD / "trazdf_round125.patch").read_text()
    removed = [line for line in patch.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    assert "kt >= 1081 .AND. kt <= 1440" in patch
    run = (VERTICAL_CARD / "run.sh").read_text()
    for needle in (
            "GYRE_OMIP_L2_P3_SM_R125ZDFMAG", "EXPECTED_COUNT=",
            "EXPECTED_SIZE=", "EXPECTED_TOTAL=", "SYNTAX_PROOF_PASS",
            "vertical-stamp", "vertical-truncation",
            "vertical-matrix-ulp", "vertical-trajectory-ulp",
            "ROUND125_VERTICAL_RECORD_READY"):
        assert needle in run


def test_round125_vertical_calibration_and_plants_are_nonvacuous(harness):
    if not VERTICAL_EARLY_RECORD.is_file():
        pytest.skip("admitted Round-123 early tra_zdf record unavailable")
    record = harness._read_vertical_record(VERTICAL_EARLY_RECORD, 1)
    assert harness._vertical_calibration(record) == {
        "zwt_mix": 0, "zwi": 0, "zwd": 0, "zws": 0,
        "zwt_lu": 0, "rhs_T": 0, "fwd_T": 0, "sol_T": 0,
    }
    control = harness._vertical_matrix_ulp_control(record)
    assert control["field"] == "zwd"
    assert control["registered_coefficients_moved"] == 1
    trazdf = harness._trazdf()
    with pytest.raises(
            trazdf.RecordError,
            match="truncated|followed by neither|runs past EOF"):
        trazdf.read_trazdf_matrix(
            VERTICAL_EARLY_RECORD, expect_kt=1, truncate=True)


def test_round125_vertical_stamp_and_trajectory_controls_fail(tmp_path,
                                                              harness):
    manifest = tmp_path / "vertical_records.sha256"
    manifest.write_text("0" * 64 + "  frame.bin\n")
    digest = harness._sha256(manifest)
    stamp = tmp_path / "vertical_records.stamp"
    stamp.write_text(f"{digest} good vertical_records.sha256\n")
    harness._check_record_stamp(
        tmp_path, "good", manifest.name, stamp.name)
    with pytest.raises(harness.GateError, match="producer commit mismatch"):
        harness._check_record_stamp(
            tmp_path, "wrong", manifest.name, stamp.name)

    baseline = np.ones((2, 2, 2), dtype=np.float64)
    mask = np.ones_like(baseline, dtype=bool)
    control = harness._vertical_trajectory_ulp_control(
        baseline, baseline.copy(), mask)
    assert control["cells_unequal"] == 1
    broken = baseline.copy()
    broken[0, 0, 0] = np.nextafter(broken[0, 0, 0], np.inf)
    with pytest.raises(harness.GateError,
                       match="unplanted vertical/process Tbb fields differ"):
        harness._vertical_trajectory_ulp_control(broken, baseline, mask)


def _synthetic_process_record(path: Path, harness, *, kt: int = 1081):
    shape3 = (harness.PROCESS_JPI, harness.PROCESS_JPJ, harness.PROCESS_JPK)
    shape2 = (harness.PROCESS_JPI, harness.PROCESS_JPJ)

    def block(value, shape):
        return np.full(shape, value, dtype="=f8", order="F")

    with path.open("wb") as handle:
        handle.write(harness.PROCESS_MAGIC.encode("ascii"))
        handle.write(struct.pack(
            "=11i", 1, kt, 3, 1, 2, 3, 2,
            harness.PROCESS_JPI, harness.PROCESS_JPJ, harness.PROCESS_JPK,
            harness.PROCESS_STORAGE_BITS,
        ))
        handle.write(struct.pack("=d", harness.DT_S))
        arrays = (
            block(2.0, shape3),
            block(0.0, shape2), block(0.0, shape2), block(0.0, shape2),
            block(0.10, shape3), block(0.20, shape3),
            block(0.30, shape3), block(0.40, shape3),
            block(2.75, shape3),
        )
        for values in arrays:
            handle.write(values.tobytes(order="F"))


def test_round123_process_record_layout_and_reader(tmp_path, harness):
    record_path = tmp_path / "oracle_process_budget_kt00001081.bin"
    _synthetic_process_record(record_path, harness)
    assert record_path.stat().st_size == harness.PROCESS_RECORD_BYTES
    assert harness.PROCESS_RECORD_BYTES == 1_415_300
    record = harness.read_process_record(record_path)
    assert record["kstp"] == 1081
    assert record["Tbb"].shape == (22, 32, 31)
    assert record["r3t_Kbb"].shape == (22, 32)
    assert np.all(record["rhs_after_lateral_diffusion"] == 0.40)

    with pytest.raises(harness.GateError, match="1415292 bytes"):
        harness.read_process_record(record_path, truncate=True)


def test_round136_record_availability_refuses_endpoint_relabeling(harness):
    rows = harness.developed_record_availability()
    assert [row["day"] for row in rows] == [30, 90, 180, 240]
    assert [row["status"] for row in rows] == [
        "UNAVAILABLE_BY_RECORD", "UNAVAILABLE_BY_RECORD", "MEASURED",
        "UNAVAILABLE_BY_RECORD"]
    assert rows[2]["entry_step"] == 1080
    assert rows[2]["process_step"] == 1081
    assert rows[3]["entry_step"] == 1440
    assert rows[3]["process_step"] is None
    assert "step-1439" in rows[3]["reason"]


def test_round136_cumulative_boundaries_follow_recorded_write_order(
        tmp_path, harness):
    path = tmp_path / "oracle_process_budget_kt00001081.bin"
    _synthetic_process_record(path, harness)
    record = harness.read_process_record(path)
    rows = harness._process_cumulative_boundaries(record)
    assert tuple(rows) == harness.PROCESS_ROWS
    # qbb=qmm=qaa=1 in the synthetic record, so B = Tbb + dt*RHS.
    expected = {
        "geometry": 2.0,
        "advection": 2.0 + harness.DT_S * 0.10,
        "surface_boundary": 2.0 + harness.DT_S * 0.20,
        "shortwave": 2.0 + harness.DT_S * 0.30,
        "lateral_diffusion": 2.0 + harness.DT_S * 0.40,
        "vertical_diffusion": 2.75,
    }
    for name, value in expected.items():
        assert np.all(rows[name] == value), name


def test_round136_registry_and_signed_zero_controls_are_nonvacuous(harness):
    report = {
        "availability": harness.developed_record_availability(),
        "cumulative_boundaries": {
            name: {} for name in harness.PROCESS_ROWS},
        "geometry_operands": {
            name: {} for name in harness.DEVELOPED_GEOMETRY_OPERANDS},
        "increment_rows": {name: {} for name in harness.PROCESS_ROWS},
        "branches": {name: {} for name in harness.DEVELOPED_BRANCHES},
        "first_non_bit_process_call": "advection",
        "process_ranking": [
            {"rank": rank, "name": name}
            for rank, name in enumerate(harness.PROCESS_ROWS, 1)],
    }
    harness._validate_developed_registry(report)
    for plant, message in (
            ("missing-day", "requested-day registry"),
            ("missing-process-row", "process-row registry"),
            ("missing-branch", "branch registry"),
            ("missing-ranking-row", "process ranking")):
        with pytest.raises(harness.GateError, match=message):
            harness._validate_developed_registry(report, plant=plant)

    positive = np.asarray([[[0.0]]], dtype=np.float64)
    negative = np.asarray([[[-0.0]]], dtype=np.float64)
    row = harness._score_developed_row(
        positive, negative, np.ones_like(positive, dtype=bool))
    assert row["cells_unequal"] == 1
    assert row["max_abs"] == 0.0
    assert row["first_unequal_jik"] == [0, 0, 0]

    contiguous = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    noncontiguous = np.transpose(contiguous, (1, 0, 2))
    assert not noncontiguous.flags.c_contiguous
    assert harness._bit_mismatch_count(
        noncontiguous, np.array(noncontiguous, order="C")) == 0


def test_round152_process_ranking_is_complete_and_uses_one_step_rms(harness):
    rows = {
        name: {"cells_unequal": index + 1,
               "max_abs": float(index + 1),
               "rms": float(index + 1)}
        for index, name in enumerate(harness.PROCESS_ROWS)
    }
    ranking = harness._rank_developed_process_rows(rows)
    assert [row["name"] for row in ranking] == list(
        reversed(harness.PROCESS_ROWS))
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4, 5, 6]
    assert ranking[0]["rms_effective_tendency_K_s"] == (
        ranking[0]["rms_temperature_contribution_K"] / harness.DT_S)
    with pytest.raises(harness.GateError, match="ranking input is incomplete"):
        harness._rank_developed_process_rows(
            {name: rows[name] for name in harness.PROCESS_ROWS[:-1]})


def test_round154_fct_registry_and_first_statement_are_nonvacuous(harness):
    from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS

    scalar = np.ones((1,), dtype=np.float64)
    common_expected = {
        name: scalar.copy()
        for name in harness.DEVELOPED_FCT_COMMON_FIELDS
    }
    common_expected["e3t_3d"] = np.ones((1, 1, 1), dtype=np.float64)
    for name in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"):
        common_expected[name] = np.zeros((1, 1), dtype=np.float64)
    tracer_expected = {
        name: scalar.copy() for name in harness.DEVELOPED_FCT_TRACER_FIELDS
    }
    tracer_expected["rhs_entry"] = np.zeros_like(scalar)
    bundle = {
        "common": common_expected,
        "tracers": {name: {key: value.copy()
                            for key, value in tracer_expected.items()}
                    for name in ("T", "S")},
    }
    common_actual = {name: scalar.copy()
                     for name in harness.DEVELOPED_FCT_COMMON_FIELDS}
    common_actual["e3t_3d"] = np.ones((1, 1, 1), dtype=np.float64)
    for name in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"):
        common_actual[name] = np.ones((1, 1, 1), dtype=np.float64)
    observed_tracer = tuple(scalar.copy() for _ in range(
        13 + len(NEMO_FCT_TRACE_FIELDS)))
    observed = {"T": observed_tracer, "S": observed_tracer}
    exact = harness._developed_fct_mode_rows(
        observed, bundle, common_actual)
    assert exact["rows_scored"] == 61
    assert exact["bit_exact_rows"] == 61
    assert exact["first_non_bit_context"] == "NONE"
    assert exact["first_non_bit_statement"] == "NONE"

    first_u = 13 + NEMO_FCT_TRACE_FIELDS.index("first_u")
    planted_values = list(observed_tracer)
    planted_values[first_u] = np.nextafter(
        planted_values[first_u], np.inf)
    planted = harness._developed_fct_mode_rows(
        {"T": tuple(planted_values), "S": observed_tracer},
        bundle, common_actual)
    assert planted["first_non_bit_statement"] == "T.first_u"
    assert planted["tracers"]["T"]["rows"]["first_u"][
        "cells_unequal"] == 1


def test_round155_transport_registry_and_first_operand_are_nonvacuous(harness):
    fields_3d = {
        "e3u_0", "umask", "live_e3u_Kmm", "uu_Kmm", "corrected_u", "zFu",
    }
    expected = {
        name: np.ones((1, 1, 1) if name in fields_3d else (1, 1),
                      dtype=np.float64)
        for name in harness.DEVELOPED_TRANSPORT_U_ROWS
    }
    exact = harness._developed_transport_mode_rows(expected, expected)
    assert exact["rows_scored"] == len(harness.DEVELOPED_TRANSPORT_U_ROWS)
    assert exact["bit_exact_rows"] == exact["rows_scored"]
    assert exact["first_non_bit_row"] == "NONE"

    planted = {name: value.copy() for name, value in expected.items()}
    planted["un_adv"][0, 0] = np.nextafter(
        planted["un_adv"][0, 0], np.inf)
    moved = harness._developed_transport_mode_rows(planted, expected)
    assert moved["first_non_bit_row"] == "un_adv"
    assert moved["rows"]["un_adv"]["cells_unequal"] == 1

    incomplete = dict(planted)
    incomplete.pop("zFu")
    with pytest.raises(harness.GateError, match="registry is incomplete"):
        harness._developed_transport_mode_rows(incomplete, expected)


def _stage2_split_inputs(harness):
    """A small U column set whose depth mean is NEMO's own, by construction."""
    rng = np.random.default_rng(156)
    umask = np.ones((2, 3, 4), dtype=np.float64)
    e3u_0 = np.full((2, 3, 4), 10.0, dtype=np.float64)
    r1_hu_0 = np.full((2, 3), 1.0 / 40.0, dtype=np.float64)
    uu = rng.normal(size=(2, 3, 4))
    running = np.zeros((2, 3), dtype=np.float64)
    for level in range(4):
        running = running + uu[..., level] * e3u_0[..., level]
    uu_b = running * r1_hu_0
    oracle = {
        "umask": umask, "e3u_0": e3u_0, "r1_hu_0": r1_hu_0,
        "uu_Kmm": uu, "uu_b_Kmm": uu_b,
    }
    return oracle


def test_round156_stage2_split_separates_barotropic_from_baroclinic(harness):
    oracle = _stage2_split_inputs(harness)
    uu = oracle["uu_Kmm"]

    # A PURE depth-mean difference: installing NEMO's own depth mean must
    # remove it.  If the split were not measuring the external half this
    # residual would stay at the baseline.
    offset = np.array([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])
    barotropic = {
        "uu_Kmm": uu + offset[..., None],
        "uu_b_Kmm": oracle["uu_b_Kmm"] + offset,
    }
    report = harness._developed_stage2_velocity_split(barotropic, oracle)
    assert set(report["rows"]) == set(harness.DEVELOPED_STAGE2_SPLIT_ROWS)
    rows = report["rows"]
    assert rows["production_baseline"]["active_max_abs"] > 5e-7
    assert rows["nemo_depth_mean_substituted"]["active_max_abs"] < 1e-15
    assert rows["nemo_depth_mean_substituted"][
        "active_rms_removed_fraction"] > 0.99
    known = report["external_half_known_answer"]
    assert known["removed_minus_recorded_active_max_abs"] < 1e-15

    # A DEVIATION difference with zero depth mean: the same substitution must
    # remove none of it.  Without both arms "it was removed" proves nothing.
    deviation = np.zeros_like(uu)
    deviation[..., 0] = 1e-6
    deviation[..., 1] = -1e-6
    baroclinic = {
        "uu_Kmm": uu + deviation, "uu_b_Kmm": oracle["uu_b_Kmm"],
    }
    other = harness._developed_stage2_velocity_split(baroclinic, oracle)
    assert other["rows"]["production_baseline"]["active_max_abs"] > 5e-7
    assert other["rows"]["nemo_depth_mean_substituted"][
        "active_rms_removed_fraction"] < 1e-6

    # The calibration arm: NEMO's own field with NEMO's own depth mean.
    same = harness._developed_stage2_velocity_split(
        {"uu_Kmm": uu, "uu_b_Kmm": oracle["uu_b_Kmm"]}, oracle)
    assert same["rows"]["calibration_nemo_reprojection"][
        "active_max_abs"] < 1e-15

    # The ULP plant path perturbs NEMO's recorded target, not the model's.
    planted = harness._developed_stage2_velocity_split(
        barotropic, oracle, plant_index=(0, 0))
    assert planted["planted_uu_b_index"] == [0, 0]
    # A one-unit-in-the-last-place change to one column moves about thirty
    # cells by about 1e-16, which no maximum over the field can see, so the
    # control is the fingerprint of the whole reprojected field.
    assert (planted["reprojection_sha256"]
            != report["reprojection_sha256"])


def test_round123_process_budget_closes_and_ulp_control_moves(tmp_path,
                                                              harness):
    record_path = tmp_path / "oracle_process_budget_kt00001081.bin"
    _synthetic_process_record(record_path, harness)
    record = harness.read_process_record(record_path)
    rows = harness.process_temperature_rows(record)
    reconstructed = np.zeros_like(rows["geometry"])
    for name in (*harness.PROCESS_ROWS, "rounding_closure"):
        reconstructed += rows[name]
    endpoint = record["Taa"][..., :30] - record["Tbb"][..., :30]
    assert np.array_equal(reconstructed, endpoint)

    control = harness._process_sbc_ulp_control(
        record, np.ones(endpoint.shape, dtype=bool))
    assert control["raw_surface_rhs_increment_moved"]
    assert control["new_uint64"] != control["old_uint64"]
    # The frozen preregistration assumed one raw RHS ULP necessarily reaches
    # the temperature budget.  This calibrated counterexample proves why the
    # distinct effect-scale plant is required instead of pretending it did.
    low_dt_record = dict(record)
    low_dt_record["rDt"] = 1.0
    swallowed = harness._process_sbc_ulp_control(
        low_dt_record, np.ones(endpoint.shape, dtype=bool))
    assert sum(swallowed["decoded_temperature_rows_moved"].values()) == 0
    propagated = harness._process_sbc_effect_control(
        record, np.ones(endpoint.shape, dtype=bool))
    assert propagated["decoded_temperature_rows_moved"]["surface_boundary"]


def test_round124_lego_process_budget_closes_and_ulp_control_moves(harness):
    shape = (3, 4, 2)
    frame = {
        "Tbb": np.full(shape, 2.0),
        "B0": np.full(shape, 2.1),
        "Badv": np.full(shape, 2.2),
        "Bsbc": np.full(shape, 2.3),
        "Bqsr": np.full(shape, 2.4),
        "Bldf": np.full(shape, 2.5),
        "Bpre": np.full(shape, 2.55),
        "Taa": np.full(shape, 2.75),
    }
    rows = harness.lego_process_temperature_rows(frame)
    reconstructed = np.zeros(shape)
    for name in (*harness.PROCESS_ROWS, "rounding_closure"):
        reconstructed += rows[name]
    assert np.array_equal(reconstructed, frame["Taa"] - frame["Tbb"])

    planted = {name: np.array(value, copy=True)
               for name, value in frame.items()}
    planted["Bsbc"][0, 0, 0] = np.nextafter(
        planted["Bsbc"][0, 0, 0], np.inf)
    moved = harness.lego_process_temperature_rows(planted)
    mask = np.ones(shape, dtype=bool)
    assert harness._different_cells(
        rows["surface_boundary"], moved["surface_boundary"], mask) == 1
    # Removing the changed boundary makes the same assertion fail: the plant
    # is tied to the consumed row, not merely to a nonzero synthetic array.
    planted["Bsbc"] = np.array(frame["Bsbc"], copy=True)
    inert = harness.lego_process_temperature_rows(planted)
    assert harness._different_cells(
        rows["surface_boundary"], inert["surface_boundary"], mask) == 0


def test_round124_effect_control_reaches_downstream_not_carried_state(harness):
    from types import SimpleNamespace

    shape = (2, 2, 1)
    q = np.ones(shape[:2])
    base = np.full(shape, 2.0)
    delta = float(np.ldexp(1.0, -40))
    effect = harness.DT_S * delta
    control = SimpleNamespace(
        state_after=(np.array([1.0]),), Tbb=base, q_Kbb=q, q_Kmm=q,
        q_Kaa=q, boundaries=(base, base, base, base, base, base), Taa=base)
    downstream = base.copy()
    downstream[0, 0, 0] += effect
    planted = SimpleNamespace(
        state_after=(np.array([1.0]),), Tbb=base, q_Kbb=q, q_Kmm=q,
        q_Kaa=q,
        boundaries=(base, base, downstream, downstream, downstream,
                    downstream),
        Taa=downstream)
    report = harness._trace_effect_control(
        control, planted, np.ones(shape, dtype=bool), (0, 0, 0), delta)
    assert report["status"] == "PLANT-FIRED"
    assert report["moved_cells"]["Badv"] == 0
    assert report["moved_cells"]["Bsbc"] == 1
    assert report["carried_state_unequal_bytes"] == 0

    bad = SimpleNamespace(**{**planted.__dict__,
                             "state_after": (np.array([2.0]),)})
    with pytest.raises(harness.GateError, match="write-only effect plant"):
        harness._trace_effect_control(
            control, bad, np.ones(shape, dtype=bool), (0, 0, 0), delta)


def test_round124_process_hook_is_private_and_card_guarded():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks, LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card("GYRE-zco")
    assert "tracer_process_trace" not in card.recipe.model_config._fields
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(tracer_process_trace=()))
    assert model._nemo_ws_test_hooks.tracer_process_trace == ()
    with pytest.raises(ValueError, match=r"must be \(\) or"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_trace="public-selector"))

    # Round 126's larger vertical return graph must remain a distinct observer
    # so it cannot silently change Round 124's process-boundary fusion control.
    with pytest.raises(ValueError, match="requires tracer_process_trace"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                vertical_solve_trace=True))
    with pytest.raises(ValueError, match="requires tracer_process_trace"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_branch_activity=True))
    vertical = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True))
    assert vertical._nemo_ws_test_hooks.vertical_solve_trace is True
    branch = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), tracer_process_branch_activity=True))
    assert branch._nemo_ws_test_hooks.tracer_process_branch_activity is True


def test_round126_literal_matrix_and_solve_controls_are_nonvacuous(harness):
    shape = (2, 3, 4)
    wet = np.ones(shape, dtype=bool)
    K = np.full(shape[:-1] + (shape[-1] - 1,), 1.2e-5)
    e3t = np.broadcast_to(np.array([10.0, 20.0, 30.0, 40.0]), shape)
    e3w = np.broadcast_to(np.array([15.0, 25.0, 35.0]), K.shape)
    lower, diagonal, upper = harness._literal_vertical_matrix(
        K, e3t, e3w, wet)
    constant = np.full(shape, 7.0)
    content = e3t * constant
    solved = harness._literal_vertical_solve(
        content, lower, diagonal, upper, wet)
    assert np.max(np.abs(solved - constant)) < 4.0e-15

    planted = diagonal.copy()
    planted[0, 0, 0] = np.nextafter(planted[0, 0, 0], np.inf)
    assert harness._different_cells(diagonal, planted, wet) == 1
    moved = harness._literal_vertical_solve(
        content, lower, planted, upper, wet)
    assert harness._different_cells(solved, moved, wet) > 0


def test_round126_barotropic_checkpoint_histories_keep_their_2d_layout(
        harness):
    uu_b = np.arange(6.0).reshape(2, 3)
    vv_b = np.arange(6.0, 12.0).reshape(2, 3)
    u_face, v_face = harness._closed_barotropic_histories_to_faces(
        uu_b, vv_b)
    assert u_face.shape == (2, 4)
    assert v_face.shape == (3, 3)
    np.testing.assert_array_equal(u_face[:, 0], 0.0)
    np.testing.assert_array_equal(u_face[:, 1:], uu_b)
    np.testing.assert_array_equal(v_face[0, :], 0.0)
    np.testing.assert_array_equal(v_face[1:, :], vv_b)
    with pytest.raises(harness.GateError, match="two-dimensional"):
        harness._closed_barotropic_histories_to_faces(
            uu_b[..., None], vv_b)


def test_round127_shapley_cube_closes_and_exposes_cancellation(harness):
    control = harness._trigger_shapley_control()
    assert control == {
        "temperature": [1.0, 0.5, 0.5],
        "salinity": [0.0, 0.5, 0.0],
        "live_depth": [0.0, 0.0, -0.5],
    }
    incomplete = {subset: np.asarray(float(bool(subset & 1)))
                  for subset in range(7)}
    with pytest.raises(harness.GateError, match="all eight subsets"):
        harness._trigger_shapley(incomplete)


def test_round127_recorded_r3t_inverse_is_bit_strict(harness):
    H = np.asarray([[1000.0, 2500.0], [4000.0, 1.0]], dtype=np.float64)
    wet = np.asarray([[True, True], [True, False]])
    eta_source = np.asarray([[0.125, -0.75], [1.5, 0.0]], dtype=np.float64)
    reciprocal = np.float64(1.0) / np.where(wet, H, 1.0)
    recorded = np.where(wet, eta_source * reciprocal, 0.0)
    eta, control = harness._eta_for_recorded_r3t(recorded, H, wet)
    replay = np.where(wet, eta * reciprocal, 0.0)
    np.testing.assert_array_equal(
        replay[wet].view(np.uint64), recorded[wet].view(np.uint64))
    assert control["wet_cells_unequal"] == 0

    impossible = recorded.copy()
    impossible[0, 0] = np.nan
    with pytest.raises(harness.GateError, match="could not invert"):
        harness._eta_for_recorded_r3t(impossible, H, wet)


def test_round127_trigger_observer_is_private_and_default_return_unchanged():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card("GYRE-zco")
    assert "bn2_intermediate" not in card.recipe.model_config._fields
    assert _NEMOWSRK3TestHooks().bn2_intermediate == ""
    assert _NEMOWSRK3TestHooks(
        bn2_intermediate="masked_rn2").bn2_intermediate == "masked_rn2"


def test_round128_temperature_context_marginal_has_all_four_arms(harness):
    control = harness._temperature_context_marginal_control()
    assert control["observed"] == control["expected"]

    candidates = {index: np.asarray(float(index)) for index in range(4)}
    references = {index: np.asarray(0.0) for index in range(3)}
    with pytest.raises(harness.GateError, match="all four"):
        harness._temperature_context_marginal(candidates, references)


def test_round128_temperature_telescope_and_registry_plant(harness):
    shape = (2, 3, 4)
    wet = np.ones(shape, dtype=bool)
    nemo = np.full(shape, 2.0)
    incoming = np.full(shape, 0.125)
    cumulative = {
        name: np.full(shape, (index + 1) * 0.03125)
        for index, name in enumerate(harness.PROCESS_ROWS)
    }
    expected = np.array(nemo, copy=True)
    expected += incoming
    for name in harness.PROCESS_ROWS:
        expected += cumulative[name]
    lego = np.nextafter(expected, np.inf)
    boundaries, deltas, raw_residual = (
        harness._temperature_process_boundaries(
            nemo, lego, incoming, cumulative))
    assert raw_residual > 0.0
    np.testing.assert_array_equal(
        boundaries[-1][1].view(np.uint64), lego.view(np.uint64))

    combined = np.zeros(shape)
    for name in harness.TRIGGER_TEMPERATURE_PROCESS_ROWS:
        combined += deltas[name]
    np.testing.assert_allclose(combined, lego - nemo, atol=2.3e-16, rtol=0.0)
    planted = harness._temperature_process_registry_plant(
        nemo, lego, deltas, wet)
    assert planted["removed_row"] in harness.PROCESS_ROWS
    assert planted["endpoint_cells_moved"] > 0
    assert planted["endpoint_residual_max_abs_K"] > 0.0

    zero = {name: np.zeros(shape) for name in harness.PROCESS_ROWS}
    _, inert_deltas, _ = harness._temperature_process_boundaries(
        nemo, nemo + incoming, incoming, zero)
    with pytest.raises(harness.GateError, match="no nonzero physics row"):
        harness._temperature_process_registry_plant(
            nemo, nemo + incoming, inert_deltas, wet)


def test_round128_weighted_spatial_census_broadcasts_regions(harness):
    weights = np.ones((2, 3, 4), dtype=np.float64)
    depth = np.broadcast_to(
        np.asarray([25.0, 250.0, 1250.0, 2000.0]), weights.shape)
    west = np.asarray([[True, False, False], [True, False, False]])
    interior = np.asarray([[False, True, False], [False, True, False]])
    east = np.asarray([[False, False, True], [False, False, True]])
    south = np.asarray([[True, True, True], [False, False, False]])
    regions = {
        "west_third": west, "interior_third": interior,
        "east_third": east,
        f"emp_south_le_{harness.EMP_SPLIT_LAT_DEG}N": south,
        f"emp_north_gt_{harness.EMP_SPLIT_LAT_DEG}N": ~south,
    }
    census = harness._weighted_trigger_spatial_census(
        weights, depth, regions)
    assert census["absolute_cell_equivalents"] == 24.0
    assert census["depth"] == {
        "0_100m": 6.0, "100_1000m": 6.0, "below_1000m": 12.0}
    assert census["region"]["west_third"] == 8.0
    assert census["region"]["interior_third"] == 8.0
    assert census["region"]["east_third"] == 8.0

    with pytest.raises(harness.GateError, match="depth shape differs"):
        harness._weighted_trigger_spatial_census(
            weights, depth[..., :-1], regions)


def test_round123_acquisition_card_is_additive_and_fail_closed(harness):
    source_patch = (PROCESS_CARD / "stprk3_stg_round123.patch").read_text()
    removed = [line for line in source_patch.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    assert source_patch.count("WRITE(r123_unit)") == 9
    for statement in ("CALL tra_adv", "CALL tra_sbc_RK3", "CALL tra_ldf",
                      "CALL tra_zdf", "DEALLOCATE( l2_qsr_before )"):
        assert statement in source_patch

    run_sh = (PROCESS_CARD / "run.sh").read_text()
    assert "trap refuse_on_error ERR" in run_sh
    assert "TARGET_CFG=GYRE_OMIP_L2_P3_SM_R123PROC" in run_sh
    assert "EXPECTED_SIZE" in run_sh and "1415300" in run_sh
    assert "EXPECTED_TOTAL" in run_sh and "509508000" in run_sh
    assert 'cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"' in run_sh
    assert "'CALL tra_qsr'" in run_sh
    assert "process-sbc-effect" in run_sh
    assert "process-trajectory-ulp" in run_sh
    assert "ROUND123_PROCESS_RECORD_READY" in run_sh


def test_round177_record_header_tracks_the_compiled_writer():
    """The admission pins Kbb/Kmm/Krhs in the writer's literal order."""
    root = Path(
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_round177_tracer_ldf")
    patch = (root / "traldf_iso_round177.patch").read_text()
    run_sh = (root / "run.sh").read_text()
    assert "1, kt, Kbb, Kmm, Krhs" in patch
    assert (
        "expected_header = (1, 1081, 1, 2, 3, 36, 26, 31, 30, 1, 64, 38, 11)"
        in run_sh)
    process_patch = Path(
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_round123_process_budget/"
        "stprk3_stg_round123.patch").read_text()
    assert "kstg, Kbb, Kmm, Krhs, Kaa" in process_patch
    assert (
        "process_header != (1, 1081, 3, 1, 2, 3, 3, 36, 26, 31, 64)"
        in run_sh)


def _toy():
    lat = np.array([[18.0, 30.0, 42.0]])
    wet = np.ones_like(lat, dtype=bool)
    ct = np.array([[22.0, 18.0, 8.0]])
    pt = ct - 0.1
    return lat, wet, ct, pt


def test_literal_sbc_default_is_byte_unchanged(literal):
    """Round 16's own numbers must not move because the owner round needed
    an argument.  Byte equality, not a tolerance."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt)
    again, _ = literal(lat, wet, ct, pt, kt=1, nyear=1, qsr_pi=None)
    for field in ("qsr", "qns", "emp", "utau", "vtau"):
        assert np.array_equal(np.asarray(base[field]).view(np.uint64),
                              np.asarray(again[field]).view(np.uint64)), field


def test_every_forcing_field_moves_with_the_clock(literal):
    """Non-vacuity of the PHASE plant: if a field did not move between two
    values of ztime, a wrong phase could not be seen on it."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt, kt=1)
    later, _ = literal(lat, wet, ct, pt, kt=181)
    for field in ("qsr", "qns", "emp", "utau", "vtau"):
        assert not np.array_equal(np.asarray(base[field]).view(np.uint64),
                                  np.asarray(later[field]).view(np.uint64)), field


def test_the_nyear_term_is_a_no_op_inside_year_one_and_not_beyond(literal):
    """usrdef_sbc.f90:107-108 subtracts (nyear-1)*rjjhh*zyydd = one full
    8640 h period per year.  Inside year 1 it subtracts exactly 0.0, which is
    exact; at nyear=2 it moves the argument and the result."""
    lat, wet, ct, pt = _toy()
    inside, _ = literal(lat, wet, ct, pt, kt=181, nyear=1)
    beyond, _ = literal(lat, wet, ct, pt, kt=181, nyear=2)
    assert not np.array_equal(np.asarray(inside["qsr"]).view(np.uint64),
                              np.asarray(beyond["qsr"]).view(np.uint64))
    # and it is MATHEMATICALLY a no-op: one period, so the value barely moves
    assert float(np.max(np.abs(inside["qsr"] - beyond["qsr"]))) < 1.0e-11


def test_the_qsr_pi_plant_moves_qsr(literal):
    """usrdef_sbc.f90:136 writes the literal 3.1415, NOT rpi.  A transcription
    that reaches for pi is a real defect, so the plant must be visible."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt)
    swapped, _ = literal(lat, wet, ct, pt, qsr_pi=3.141592653589793)
    moved = float(np.max(np.abs(base["qsr"] - swapped["qsr"])))
    assert moved > 1.0e-4, moved


def test_regions_partition_the_wet_surface(harness):
    lat = np.tile(np.linspace(12.0, 50.0, 8)[:, None], (1, 8))
    wet = np.zeros((8, 8), dtype=bool)
    wet[1:-1, 1:-1] = True
    regions = harness._regions(lat, wet)
    thirds = sum(regions[name].astype(int)
                 for name in ("west_third", "interior_third", "east_third"))
    assert np.array_equal(thirds, wet.astype(int))
    # non-vacuity: every cut must actually select something
    for name, mask in regions.items():
        assert mask.any(), name
    # and a cut must never reach land
    for name, mask in regions.items():
        assert not (mask & ~wet).any(), name


def test_the_step_entry_registry_covers_the_card_writer_and_no_more():
    """The card writes sixty per-step entry dumps
    (cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90).  The registry must
    read all sixty and still FAIL CLOSED on the sixty-first."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for kt in (1, 2, 10, 11, 59, 60):
        assert time_level_for_dump(
            f"oracle_step_entry_kt{kt:08d}.bin") == "before"
    with pytest.raises(ValueError):
        time_level_for_dump("oracle_step_entry_kt00000061.bin")


def test_member_snapshot_cadence_refuses_a_partial_day():
    """run_member's new cadence names its files by DAY, so a cadence that is
    not a whole number of days would fold several steps onto one filename."""
    spec = importlib.util.spec_from_file_location(
        "gyre_year_fromrest_cadence",
        ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
        / "nemo_testcase_l2_gyre_year_fromrest.py")
    year = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(year)
    with pytest.raises(year.GateError):
        year.run_member(0, Path("/nonexistent"), days=1, snap_steps=1)
    # non-vacuity: the whole-day cadences pass this guard and fail LATER, on
    # something else, which proves the guard is what rejected snap_steps=1
    with pytest.raises(Exception) as info:
        year.run_member(0, Path("/nonexistent"), days=1, snap_steps=6)
    assert "snap_steps" not in str(info.value)


def test_run_sh_is_a_config_free_staging_script_and_never_builds():
    """The acquisition reuses the certified binary.  If it ever grows a
    makenemo call, the preregistration's ASKED item 1 is no longer describing
    the script and the receipt's admission argument is stale."""
    text = RUN_SH.read_text()
    # The test that FAILED when first written asserted the WORD was absent,
    # and the script's own comment explains that the agent must never run
    # makenemo.  What matters is an EXECUTED call, so strip comment lines.
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "makenemo" not in code
    assert "cp -r" not in code          # never copy a whole cfgs/ directory
    assert "mpirun" in code             # non-vacuity: the stripper kept code
    assert "BYTE-IDENTICAL" in text
    assert "GYRE_OMIP_L2_P3_00000180_restart.nc" in text
    # the refusal must be a refusal, not a warning
    assert "exit 71" in code
    # A word-grep passes with the wrong card, a deleted comparison and any
    # binary; an independent review said so.  These pin the three things that
    # make the record admissible.
    assert "SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP" in code
    assert 'cmp -s "$certified_binary" "$YEAR_RUN/nemo_pristine/nemo"' in code
    assert 'cmp -s "$RUN_DIR/GYRE_OMIP_L2_P3_00000180_restart.nc"' in code
    assert "NN_STOCK=6" in code and "NN_ITEND=180" in code


def _worktree_is_dirty() -> bool:
    """The harness stamps provenance and REFUSES on a dirty tree, so a plant
    run would die on the stamp rather than on the plant and the test would
    report a failure it did not cause."""
    # TRACKED modifications only: that is what provenance.git_sha refuses on
    # ("N tracked file(s) modified").  A first version used plain
    # `git status --porcelain`, so an untracked file belonging to ANOTHER
    # agent working the same branch skipped this test for no reason.
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         cwd=str(ROOT), capture_output=True, text=True)
    return bool(out.stdout.strip())


@pytest.mark.skipif(not NEMO_RECORD.is_dir(),
                    reason="the NEMO year record is not on this machine")
@pytest.mark.skipif(_worktree_is_dirty(),
                    reason="the harness refuses to stamp a dirty worktree")
def test_forcing_gate_plants_all_exit_non_zero():
    """Each plant must turn the BIT-EXACT forcing gate red.  Without this the
    gate's green is unfalsifiable."""
    for plant in ("forcing-phase", "forcing-qsr-pi", "forcing-nyear",
                  "forcing-stress-transpose"):
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--forcing-gate", "--days", "30",
             "--plant", plant],
            capture_output=True, text=True, cwd=str(ROOT))
        assert result.returncode == 1, (plant, result.stdout, result.stderr)
        assert "DEBT" in result.stdout, plant


def test_round157_stage2_face_score_is_bitwise_and_mask_aware(harness):
    """The stage scorer broadcasts a 2-D face mask through every level."""
    rng = np.random.default_rng(157)
    expected = rng.normal(size=(3, 4, 5))
    mask = np.ones(expected.shape[:2])
    mask[1, 2] = 0.0

    same = harness._score_stage2_face(expected.copy(), expected, mask)
    assert same["cells_unequal"] == 0
    assert same["active_cells_unequal"] == 0
    assert same["active_cells_scored"] == (3 * 4 - 1) * 5
    assert same["active_max_abs"] == 0.0

    # A one-unit-in-the-last-place move on an ACTIVE face is seen, and the
    # difference is far below anything a tolerance would catch.
    moved = expected.copy()
    moved[1, 1, 0] = np.nextafter(moved[1, 1, 0], np.inf)
    row = harness._score_stage2_face(moved, expected, mask)
    assert row["active_cells_unequal"] == 1
    assert 0.0 < row["active_max_abs"] < 1e-15

    # The same move on a DRY face must leave every active number alone, or
    # the walk would attribute land to a compiled statement.
    dry = expected.copy()
    dry[1, 2, -1] = dry[1, 2, -1] + 1.0
    dry_row = harness._score_stage2_face(dry, expected, mask)
    assert dry_row["cells_unequal"] == 1
    assert dry_row["active_cells_unequal"] == 0
    assert dry_row["active_max_abs"] == 0.0

    with pytest.raises(harness.GateError, match="stage-2 shapes differ"):
        harness._score_stage2_face(expected[:, :, :-1], expected, mask)


def test_round157_stage2_record_refuses_the_wrong_assignment_arm(harness,
                                                                 tmp_path,
                                                                 monkeypatch):
    """A record whose flags select the thickness-weighted arm is refused.

    The walk calibrates NEMO's VECTOR stage assignment.  If a future deck
    turned the vector form off, NEMO would take the thickness-weighted arm
    and every calibration in the walk would be proving the wrong statement,
    so the reader refuses instead of quietly scoring it.
    """
    groups = ("rhs_entry", "uu_vv_Kbb", "uu_vv_Kmm", "umask_vmask",
              "after_hpg", "after_vor", "after_adv", "uu_vv_Kaa_raw",
              "uu_vv_Kaa_final")
    fields = {name: (np.zeros((36, 26, 31)), np.zeros((36, 26, 31)))
              for name in groups}
    fields["rhd_ww"] = (
        np.zeros((36, 26, 31)), np.zeros((36, 26, 31)))
    fields["r3u_r3v_Kmm"] = (
        np.zeros((36, 26)), np.zeros((36, 26)))
    fields["ssh_Kmm_ssh_Kaa"] = (
        np.zeros((36, 26)), np.zeros((36, 26)))
    fields["rDt_r1_Dt"] = (np.float64(7200.0), np.float64(1.0 / 7200.0))

    class _Stub:
        RECORD = "oracle_developed_stage2_kt00001081.bin"

        @staticmethod
        def read_record(path):
            return {"meta": {"kt": 1081, "kstg": 2}, "sha256": "abc",
                    "fields": dict(fields, **{
                        "flags_vec_linssh": _Stub.flags})}

    (tmp_path / "round156_developed_stage2_admission.json").write_text(
        json.dumps({"status": "PASS", "record": {"sha256": "abc"}}))
    monkeypatch.setattr(harness, "_load", lambda *args, **kwargs: _Stub)

    _Stub.flags = (np.float64(1.0), np.float64(0.0))
    good = harness._developed_stage2_record(tmp_path)
    assert good["rows"]["uu_vv_Kbb_u"].shape == (22, 33, 30)
    assert good["rows"]["uu_vv_Kbb_v"].shape == (23, 32, 30)

    _Stub.flags = (np.float64(0.0), np.float64(0.0))
    with pytest.raises(harness.GateError,
                       match="vector stage-update arm"):
        harness._developed_stage2_record(tmp_path)


def test_round167_vertical_sensitivity_ranking_is_complete_and_signed(harness):
    """The magnitude ranking sees benefit, harm, and a missing real arm."""
    rows = {
        "free": {"T": {"rms": 2.0}},
        "heat_K": {"T": {"rms": 1.5}},
        "complete_K": {"T": {"rms": 2.25}},
    }
    ranked = harness._rank_vertical_sensitivity(rows)
    assert [row["arm"] for row in ranked] == ["heat_K", "complete_K"]
    assert ranked[0]["day240_T3D_rms_removed_K"] == 0.5
    assert ranked[0]["removed_fraction"] == 0.25
    assert ranked[1]["day240_T3D_rms_removed_K"] == -0.25

    broken = dict(rows)
    broken.pop("complete_K")
    with pytest.raises(harness.GateError, match="no complete_K arm"):
        harness._rank_vertical_sensitivity(broken)


def test_round174_solve_input_pair_ranks_both_arms_and_plants_fire(harness):
    """The paired scorer registers both arms and catches either scale plant."""
    shape = (20, 30, 30)
    wet = np.ones(shape, dtype=bool)
    baseline = np.full(shape, 10.0, dtype=np.float64)
    e3t = baseline + np.float64(1.0e-4)
    content = baseline.copy()
    content[:10] += np.float64(1.0e-5)
    fields = {
        "baseline": baseline,
        "source": baseline.copy(),
        "e3t_wet": e3t,
        "content_wet": content,
    }

    scored = harness._score_solve_input_temperatures(fields, wet)
    assert scored["baseline"]["cells_unequal"] == 0
    assert scored["baseline"]["day240_T3D_rms_K"] == 0.0
    assert [row["arm"] for row in scored["ranking"]] == [
        "e3t_wet", "content_wet"]
    assert scored["prediction_e3t_larger_than_content"] == "CONFIRMED"

    for plant in ("solve-input-e3t-scale",
                  "solve-input-content-scale"):
        with pytest.raises(harness.GateError, match="scale was caught"):
            harness._score_solve_input_temperatures(
                fields, wet, plant=plant)

    incomplete = dict(fields)
    incomplete.pop("content_wet")
    with pytest.raises(harness.GateError, match="registry is incomplete"):
        harness._score_solve_input_temperatures(incomplete, wet)

    moved_source = dict(fields)
    moved_source["source"] = baseline.copy()
    moved_source["source"][0, 0, 0] = np.nextafter(10.0, np.inf)
    with pytest.raises(harness.GateError, match="baseline moved 1 source"):
        harness._score_solve_input_temperatures(moved_source, wet)


def test_round174_solve_input_pair_refuses_dirty_worktree(
        harness, tmp_path, monkeypatch):
    """The record scorer must refuse before reading data from a dirty tree."""
    from legoesm.ocean.fidelity import provenance

    monkeypatch.setattr(
        provenance, "worktree_stamp",
        lambda: {"clean": False, "commit": "synthetic-dirty"})
    with pytest.raises(harness.GateError,
                       match="requires a clean committed tree"):
        harness.score_nemo_solve_input_pair(
            tmp_path / "pair", tmp_path / "baseline",
            tmp_path / "source", tmp_path / "mesh")


def test_round175_content_rows_find_the_first_moved_family(
        harness, monkeypatch):
    """The content registry distinguishes before state from accumulated RHS."""
    shape = (2, 3, 2)
    wet = np.ones(shape, dtype=bool)
    nemo = {
        "T_Kbb_in": np.full(shape, 3.0),
        "e3t_Kbb": np.full(shape, 4.0),
        "T_Krhs_in": np.full(shape, 5.0),
        "e3t_Kmm": np.full(shape, 6.0),
    }
    before = nemo["e3t_Kbb"] * nemo["T_Kbb_in"]
    accumulated = (np.float64(2.0) * nemo["e3t_Kmm"]) * nemo["T_Krhs_in"]
    nemo["rhs_T"] = before + accumulated
    monkeypatch.setattr(
        harness, "_vertical_field",
        lambda _record, name, _nlev=None: np.array(nemo[name], copy=True))
    observed = {
        "T_Kbb": nemo["T_Kbb_in"].copy(),
        "e3t_Kbb": nemo["e3t_Kbb"].copy(),
        "before_content": before.copy(),
        "T_Krhs": nemo["T_Krhs_in"].copy(),
        "e3t_Kmm": nemo["e3t_Kmm"].copy(),
        "accumulated_Krhs_content": accumulated.copy(),
        "content": nemo["rhs_T"].copy(),
        "rebuilt_content": nemo["rhs_T"].copy(),
        "advection_increment": np.ones(shape),
        "source_increment": accumulated - 1.0,
    }
    observed["accumulated_Krhs_content"][0, 0, 0] = np.nextafter(
        observed["accumulated_Krhs_content"][0, 0, 0], np.inf)
    observed["content"][0, 0, 0] = (
        observed["before_content"][0, 0, 0]
        + observed["accumulated_Krhs_content"][0, 0, 0])
    observed["rebuilt_content"] = (
        observed["before_content"]
        + observed["accumulated_Krhs_content"])
    scored = harness._content_walk_rows(
        observed, {"arrays": {"rDt": 2.0}}, wet)
    assert scored["first_non_bit"] == "accumulated_Krhs_content"
    assert scored["rows"]["before_content"]["bit_exact"] is True
    assert scored["rows"]["accumulated_Krhs_content"][
        "cells_unequal"] == 1


def test_round175_content_walk_refuses_dirty_worktree(
        harness, tmp_path, monkeypatch):
    """The developed walk refuses before opening either oracle record."""
    from legoesm.ocean.fidelity import provenance

    monkeypatch.setattr(
        provenance, "worktree_stamp",
        lambda: {"clean": False, "commit": "synthetic-dirty"})
    with pytest.raises(harness.GateError,
                       match="requires a clean committed tree"):
        harness.developed_content_producer_walk(
            tmp_path / "vertical", tmp_path / "daily",
            tmp_path / "audit", "0" * 40)


def test_round176_content_process_registry_and_projection(harness):
    """Every compiled process row contributes to the complete error."""
    shape = (2, 2, 2)
    wet = np.ones(shape, dtype=bool)
    expected = {
        "advection": np.full(shape, 1.0),
        "surface_boundary": np.full(shape, 3.0),
        "shortwave": np.full(shape, 6.0),
        "lateral_diffusion": np.full(shape, 10.0),
    }
    actual = {name: values.copy() for name, values in expected.items()}
    # One error in each isolated component; cumulative writes retain all
    # earlier changes, exactly as the compiled Krhs accumulator does.
    actual["advection"][0, 0, 0] += 0.25
    actual["surface_boundary"][0, 0, 0] += 0.50
    actual["shortwave"][0, 0, 0] += 1.00
    actual["lateral_diffusion"][0, 0, 0] += 2.00
    scored = harness._accumulated_content_process_rows(
        actual, expected, actual["lateral_diffusion"],
        expected["lateral_diffusion"], wet)
    assert scored["first_non_bit_cumulative_boundary"] == "advection"
    assert scored["registered_cumulative_order"] == [
        "advection", "surface_boundary", "shortwave",
        "lateral_diffusion", "complete_accumulated_content"]
    assert set(scored["isolated_components"]) == {
        "advection", "surface_boundary", "shortwave",
        "lateral_diffusion", "rounding_closure"}
    assert scored["reconstruction"]["bit_exact"] is True
    assert np.isclose(
        scored["signed_projection_sum_Km"],
        scored["complete_error_rms_Km"])

    incomplete = dict(actual)
    incomplete.pop("shortwave")
    with pytest.raises(harness.GateError, match="registry is incomplete"):
        harness._accumulated_content_process_rows(
            incomplete, expected, actual["lateral_diffusion"],
            expected["lateral_diffusion"], wet)


def test_round176_content_process_walk_refuses_dirty_worktree(
        harness, tmp_path, monkeypatch):
    """The process walk refuses before opening either oracle record."""
    from legoesm.ocean.fidelity import provenance

    monkeypatch.setattr(
        provenance, "worktree_stamp",
        lambda: {"clean": False, "commit": "synthetic-dirty"})
    with pytest.raises(harness.GateError,
                       match="requires a clean committed tree"):
        harness.developed_accumulated_content_process_walk(
            tmp_path / "process", tmp_path / "vertical",
            tmp_path / "daily", tmp_path / "audit", "0" * 40)


def test_round178_ldf_walk_pins_record_and_production_diagnostics(harness):
    """The developed walk consumes the admitted layout through the step."""
    assert harness.ROUND177_LDF_HEADER == (
        1, 1081, 1, 2, 3, 36, 26, 31, 30, 1, 64, 38, 11)
    assert len(harness.ROUND177_LDF_3D) == 38
    assert len(harness.ROUND177_LDF_2D) == 11
    model = Path(
        "packages/ocean/legoesm/ocean/dynamics/"
        "ocean_model_latlon_cgrid.py").read_text()
    operator = Path(
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/"
        "gm_redi_latlon_cgrid.py").read_text()
    assert "tracer_ldf_diagnostics" in model
    assert "return_redi_diagnostics=" in model
    assert '"e3u_flux": e3u_flux' in operator
    assert '"tendency": tend' in operator


def test_round178_ldf_walk_refuses_dirty_worktree(
        harness, tmp_path, monkeypatch):
    """The statement walk refuses before it opens the acquired record."""
    from legoesm.ocean.fidelity import provenance

    monkeypatch.setattr(
        provenance, "worktree_stamp",
        lambda: {"clean": False, "commit": "synthetic-dirty"})
    with pytest.raises(harness.GateError,
                       match="requires a clean committed tree"):
        harness.developed_tracer_ldf_statement_walk(
            tmp_path / "ldf", tmp_path / "daily",
            tmp_path / "audit", "0" * 40)
