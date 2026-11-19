import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_ladder_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_ladder_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _write_writer_identity_fixture(root: Path, *, halo_changes: bool) -> Path:
    root.mkdir()
    dims = (6, 6, 2)
    n3 = int(np.prod(dims))
    n2 = dims[0] * dims[1]
    zdf = root / "oracle_trazdf_matrix_kt00000001.bin"
    zdf.write_bytes(
        b"NEMO_L2_TRAZD_1 "
        + struct.pack(
            "=16i", 1, 1, 3, 1, 2, 3, 3, *dims, dims[2] - 1,
            3, 4, 3, 4, 64,
        )
    )
    for name in GATE.WRITER_STEP1_RECORDS:
        if name in GATE.WRITER_SCOPED_RECORDS:
            if "rktracer_operands" in name:
                stage = 1 if name.endswith("_s1.bin") else 2
                values = np.zeros(
                    len(GATE.TRACER_STAGE12_ARRAYS) * n3 + 3 * n2,
                    dtype=np.float64,
                )
                if halo_changes:
                    array = 2 if stage == 1 else 4  # zFu / zFw
                    full_index = 5 if stage == 1 else 0
                    values[array * n3 + full_index] = np.nextafter(0.0, 1.0)
                payload = (
                    b"NEMO_L2_RKTRA_1 "
                    + struct.pack(
                        "=11i", 1, 1, stage, 1, 1, 3, 3, *dims, 64)
                    + values.tobytes()
                )
            else:
                values = np.zeros(3 * n3, dtype=np.float64)
                payload = (
                    b"NEMO_L2_TRTRP_1 "
                    + struct.pack(
                        "=11i", 1, 1, 3, 1, 2, 3, 3, *dims, 64)
                    + values.tobytes()
                )
        else:
            payload = f"exact:{name}".encode("ascii")
        (root / name).write_bytes(payload)
    return zdf


def test_ladder_decks_have_distinct_preregistered_oracle_roots():
    assert set(GATE.DEFAULT_ORACLES) == {"r4", "r4_smag"}
    assert "round1/oracle_gyre_r4_stage" in str(GATE.DEFAULT_ORACLES["r4"])
    assert "round2/oracle_gyre_r4_smag_stage" in str(
        GATE.DEFAULT_ORACLES["r4_smag"])


def test_writer_identity_excludes_only_nemo_out_of_loop_halo_and_plant_fires(
    tmp_path,
):
    twin = tmp_path / "twin"
    candidate = tmp_path / "candidate"
    _write_writer_identity_fixture(twin, halo_changes=False)
    zdf = _write_writer_identity_fixture(candidate, halo_changes=True)
    report = GATE.compare_scalar_writer_identity(twin, candidate, zdf)
    assert report["verdict"] == "PASS"
    assert report["active_or_consumed_mismatches"] == 0
    assert {
        (row["array"], tuple(row["fortran_index_1based"]))
        for row in report["excluded_words"]
    } == {("zFu", (6, 1, 1)), ("zFw", (1, 1, 1))}

    planted = GATE.compare_scalar_writer_identity(
        twin, candidate, zdf, plant_active_word=True)
    assert planted["verdict"] == "DEBT"
    assert planted["active_or_consumed_mismatches"] == 1


def test_writer_identity_covers_shared_prognostic_and_rhs_records():
    assert "oracle_step_entry_kt00000002.bin" in GATE.WRITER_STEP1_RECORDS
    assert "oracle_rhs_kt00000001.bin" in GATE.WRITER_STEP1_RECORDS


def test_first_rk_difference_is_derived_in_stage_and_field_order():
    def row(name, unequal, maximum):
        return {"name": name, "n_unequal": unequal, "max_abs": maximum}

    walk = [
        {
            "stage": 1,
            "rows": [
                row("s1.T", 0, 0.0),
                row("s1.S", 1, 1.0e-13),
                row("s1.u", 0, 0.0),
                row("s1.v", 0, 0.0),
                row("s1.ssh", 0, 0.0),
            ],
        },
        {
            "stage": 2,
            "rows": [
                row("s2.T", 1, 2.0e-9),
                row("s2.S", 0, 0.0),
                row("s2.u", 0, 0.0),
                row("s2.v", 0, 0.0),
                row("s2.ssh", 0, 0.0),
            ],
        },
    ]
    assert GATE.first_rk_difference(walk)["field"] == "S"
    owner_scale = GATE.first_rk_difference(
        walk, minimum_max_abs=1.0e-10)
    assert (owner_scale["stage"], owner_scale["field"]) == (2, "T")


def test_step2_stage_records_are_registered_as_after_level():
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    assert [
        time_level_for_dump(f"oracle_stage_kt00000002_s{stage}.bin")
        for stage in (1, 2, 3)
    ] == ["after", "after", "after"]
    root = Path(GATE.admit_oracle("r4_step2", "scalar")["root"])
    observed = [
        GATE.read_stage(
            root / f"oracle_stage_kt00000002_s{stage}.bin",
            stage,
            dims=GATE.NEMO_DIMS,
            expected_kt=2,
            expected_kaa=GATE.STEP2_STAGE_KAA[stage],
        )["Kaa"]
        for stage in (1, 2, 3)
    ]
    assert observed == [1, 2, 1]


def test_step2_zdf_reader_requires_the_registered_step_number():
    from nemo_testcase_l2_gyre_phase3_gate import read_zdf_entry

    root = Path(GATE.admit_oracle("r4_step2", "scalar")["root"])
    path = root / "oracle_zdf_entry_kt00000002.bin"
    with pytest.raises(RuntimeError, match="bad header"):
        read_zdf_entry(path, dims=GATE.NEMO_DIMS)
    with pytest.raises(RuntimeError, match="bad header"):
        read_zdf_entry(path, dims=GATE.NEMO_DIMS, expected_kt=2)
    record = read_zdf_entry(
        path, dims=GATE.NEMO_DIMS, expected_kt=2, expected_level=3)
    assert record["avt"].shape == (82, 122, 31)


def test_tke_coefficient_split_distinguishes_prandtl_and_common_factor():
    nemo_avm = np.ones((1, 2, 2), dtype=np.float64)
    nemo_avt = np.full_like(nemo_avm, 0.5)
    lego_avm = np.array([[[1.0, 1.25], [1.0, 1.0]]])
    lego_avt = np.array([[[0.6, 0.625], [0.5, 0.5]]])
    mask = np.ones_like(nemo_avm, dtype=bool)

    split = GATE.split_tke_coefficient_residual(
        nemo_avt, nemo_avm, lego_avt, lego_avm, mask)

    assert split["summary"]["avt_avm_ratio_cells"] == 1
    assert split["summary"]["common_factor_cells"] == 1
    assert split["ratio_mask"].tolist() == [[[True, False], [False, False]]]
    assert split["pdlr_candidate_mask"].tolist() == [
        [[True, False], [False, False]]
    ]
    assert split["common_factor_mask"].tolist() == [
        [[False, True], [False, False]]
    ]

    # A ratio discrepancy on a namelist floor is empirical only: it cannot
    # be attributed to pdlr because NEMO's nested MAX branch is active.
    floor_avm = np.full((1, 1, 1), 1.2e-4)
    floor_avt = np.full((1, 1, 1), 1.2e-5)
    floor_split = GATE.split_tke_coefficient_residual(
        floor_avt, floor_avm, floor_avt * 1.01, floor_avm,
        np.ones_like(floor_avm, dtype=bool),
    )
    assert floor_split["ratio_mask"].item()
    assert not floor_split["pdlr_candidate_mask"].item()
    assert floor_split["summary"]["ratio_cells_on_a_floor_or_evd"] == 1

    evd_split = GATE.split_tke_coefficient_residual(
        np.array([[[100.0]]]), np.array([[[100.0]]]),
        np.array([[[99.0]]]), np.array([[[100.0]]]),
        np.ones((1, 1, 1), dtype=bool),
    )
    assert evd_split["ratio_mask"].item()
    assert not evd_split["pdlr_candidate_mask"].item()


def test_vertical_owner_classifier_refuses_same_max_passive_motion():
    row = lambda maximum, unequal=1: {
        "max_abs": maximum, "n_unequal": unequal,
    }
    confirmed = GATE.classify_vertical_owner(
        row(1.0e-9), row(0.0, 0), row(1.0e-14), row(1.0e-12),
        record_identity=row(0.0, 0))
    assert confirmed["verdict"] == "CONFIRMED_OWNER"

    merely_reduced = GATE.classify_vertical_owner(
        row(1.0e-9), row(1.0e-13), row(1.0e-14), row(1.0e-12),
        record_identity=row(0.0, 0))
    assert merely_reduced["verdict"] == "DOMINANT_BOUNDARY"

    # A passive field can move while retaining the same maximum residual.
    confounded = GATE.classify_vertical_owner(
        row(1.0e-9), row(1.0e-13), row(1.0e-9), row(1.0e-12),
        record_identity=row(0.0, 0))
    assert confounded["verdict"] == "CONFOUNDED"
    missed_plant = GATE.classify_vertical_owner(
        row(1.0e-9), row(1.0e-13), row(0.0, 0), row(0.0, 0),
        record_identity=row(0.0, 0))
    assert missed_plant["verdict"] == "CONFOUNDED"
    bad_record = GATE.classify_vertical_owner(
        row(1.0e-9), row(0.0, 0), row(0.0, 0), row(1.0e-12),
        record_identity=row(1.0e-15))
    assert bad_record["verdict"] == "CONFOUNDED"
    weak_response = GATE.classify_vertical_owner(
        row(1.0e-9), row(6.0e-10), row(0.0, 0), row(1.0e-12),
        record_identity=row(0.0, 0))
    assert weak_response["verdict"] == "NO_REGISTERED_DOMINANCE"


def test_all_math_flavour_oracles_are_admitted_by_content():
    expected_files = {
        ("r4", "vector"): 20,
        ("r4", "scalar"): 21,
        ("r4_smag", "vector"): 15,
        ("r4_smag", "scalar"): 15,
    }
    for deck in ("r4", "r4_smag"):
        for flavour in ("scalar", "vector"):
            admitted = GATE.admit_oracle(deck, flavour)
            assert admitted["deck"] == deck
            assert admitted["math_flavour"] == flavour
            assert len(admitted["checked_files"]) == expected_files[
                (deck, flavour)
            ]


def test_scalar_step2_oracle_requires_scoped_writer_identity_and_plant():
    admitted = GATE.admit_oracle("r4_step2", "scalar")
    assert len(admitted["checked_files"]) == 25
    assert admitted["producer_commit"] == (
        "63cebeb37ea426cd647688f20e0e0250ae194f3f")
    assert admitted["checked_proofs"]["writer_identity"]["verdict"] == "PASS"
    assert admitted["checked_proofs"]["active_word_plant"]["verdict"] == "DEBT"


def test_scalar_step2_tracer_and_qsr_readers_follow_even_rk_slots():
    root = Path(GATE.admit_oracle("r4_step2", "scalar")["root"])
    tracer = GATE.read_tracer_stage3_hires(
        root / "oracle_rktracer_stage3_kt00000002.bin", expected_kt=2)
    qsr = GATE.read_qsr_stage3_hires(
        root / "oracle_qsr_stage3_kt00000002.bin", expected_kt=2)
    assert (tracer["Kbb"], tracer["Kmm"], tracer["Krhs"], tracer["Kaa"]) == (
        3, 2, 1, 1)
    assert (qsr["Kmm"], qsr["Krhs"]) == (2, 1)
    np.testing.assert_array_equal(
        qsr["dT_dt"], tracer["after_qsr_T"] - tracer["after_sbc_T"])

    from nemo_testcase_l2_gyre_round35_trazdf_matrix import read_trazdf_matrix
    zdf = read_trazdf_matrix(
        root / "oracle_trazdf_matrix_kt00000002.bin",
        expect_kt=2, expected_domain=GATE.NEMO_DIMS)
    assert (zdf["header"]["jpi"], zdf["header"]["jpj"],
            zdf["header"]["jpk"]) == GATE.NEMO_DIMS
    assert np.asarray(zdf["arrays"]["rhs_T"]).shape == GATE.NEMO_DIMS


def test_oracle_admission_refuses_root_and_digest_mutations(tmp_path):
    with pytest.raises(ValueError, match="unregistered oracle root"):
        GATE.admit_oracle("r4", "scalar", tmp_path)

    registry = json.loads(GATE.ADMISSIONS_PATH.read_text(encoding="utf-8"))
    registry["admissions"]["r4:scalar"]["files"]["nemo"] = "0" * 64
    planted = tmp_path / "admissions.json"
    planted.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch for nemo"):
        GATE.admit_oracle("r4", "scalar", admissions_path=planted)

    registry = json.loads(GATE.ADMISSIONS_PATH.read_text(encoding="utf-8"))
    auxiliary = "oracle_rkstage2_terms_kt00000001.bin"
    registry["admissions"]["r4:scalar"]["files"][auxiliary] = "0" * 64
    planted.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match=f"digest mismatch for {auxiliary}"):
        GATE.admit_oracle("r4", "scalar", admissions_path=planted)

    registry = json.loads(GATE.ADMISSIONS_PATH.read_text(encoding="utf-8"))
    registry["admissions"]["r4_step2:scalar"]["proofs"][
        "writer_identity"
    ]["sha256"] = "0" * 64
    planted.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="proof digest mismatch for writer_identity"):
        GATE.admit_oracle("r4_step2", "scalar", admissions_path=planted)


def test_admitted_usrdef_binary_exposes_vector_and_scalar_sin_cos_no_exp():
    executable = GATE.DEFAULT_ORACLES["r4"] / "nemo"
    if not executable.is_file():
        pytest.skip("admitted r4 NEMO executable is unavailable")
    program = GATE.compiled_usrdef_math_program(executable)
    assert program["sha256"] == (
        "974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41")
    assert any(name.endswith("sin") for name in program["vector_sin_calls"])
    assert program["scalar_sin_calls"] == ["sin"]
    assert {name.rsplit("_", 1)[-1]
            for name in program["vector_transcendental_calls"]} == {"sin", "cos"}
    assert set(program["scalar_transcendental_calls"]) == {"sin", "cos"}


def test_admitted_scalar_usrdef_binary_refuses_vector_math():
    admitted = GATE.admit_oracle("r4", "scalar")
    program = GATE.compiled_usrdef_math_program(
        Path(admitted["root"]) / "nemo", expected_flavour="scalar")
    assert program["math_flavour"] == "scalar"
    assert program["vector_sin_calls"] == []
    assert program["scalar_sin_calls"] == ["sin"]
    assert program["vector_transcendental_calls"] == []
    assert set(program["scalar_transcendental_calls"]) == {"sin", "cos"}
    with pytest.raises(ValueError, match="vector oracle"):
        GATE.compiled_usrdef_math_program(
            Path(admitted["root"]) / "nemo", expected_flavour="vector")


def test_usrdef_math_inventory_refuses_missing_binary(tmp_path):
    with pytest.raises(FileNotFoundError, match="admitted NEMO executable"):
        GATE.compiled_usrdef_math_program(tmp_path / "nemo")


def test_exact_score_one_ulp_control_fires():
    values = np.array([[0.0, 1.0], [-2.0, 3.0]], dtype=np.float64)
    mask = np.ones_like(values, dtype=bool)
    assert GATE.exact_score("base", values, values, mask)["status"] == "AT-BAR"
    planted = GATE.exact_score(
        "plant", values, values, mask, plant_one_ulp=True)
    assert planted["status"] == "DEBT"
    assert planted["n_unequal"] == 1
    assert planted["max_ulp"] == 1
    assert planted["ulp_at_max_abs"] == 1
    assert planted["first_unequal_index"] == [0, 0]
    assert planted["max_abs_index"] == [0, 0]


def test_component_score_replays_accumulation_not_endpoint_subtraction():
    before = np.array([1.0e16], dtype=np.float64)
    component = np.array([1.0], dtype=np.float64)
    after = np.array([1.0e16], dtype=np.float64)
    mask = np.array([True])
    assert not np.array_equal(after - before, component)
    row = GATE.accumulated_component_score(
        "accumulated", before, after, component, mask)
    assert row["status"] == "AT-BAR"


def test_scalar_first_owner_discriminator_scores_increment_and_narrow_override():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'stage1_tracer_transport_override=oracle_transport' in source
    assert 'tracer_oracle["zFu"][..., :30]' in source
    assert 'tracer_oracle["zFv"][..., :30]' in source
    assert 'tracer_oracle["zFw"]' in source
    assert 'after_sbc_{field}' in source and 'after_advection_{field}' in source
    assert 'NEMO_stage1_transport_override' in source
    assert 'expose_momentum_operator=operator' in source
    assert 'stage2_momentum_rhs_override=' in source
    assert 'NEMO_completed_rhs_override_rows' in source
    assert 'NEMO_completed_rhs_raw_update_rows' in source
    assert '--stage2-composition-only' in source
    assert 'RAW_UPDATE_AT_BAR' in source
    assert 'control.one_ulp.raw_update.{field}' in source
    assert 'control.material_rhs_override' in source
    assert 'control.material_zFw_override' in source
    assert 'source_order_accumulation_rows' in source
    assert 'OPEN_INPUT_VERSUS_ARITHMETIC' in source
    assert '--stage3-geometry-only' in source
    assert 'UNMEASURED_EXECUTED_TRAZDF_OPERANDS' in source


def test_wind_expression_arms_preserve_compiled_association_and_reciprocal():
    rho = np.float64(1.0 / 1026.0)
    stress = np.array([[-0.2, 0.0]], dtype=np.float64)
    depth = np.array([[4194.0, 1.0]], dtype=np.float64)
    mask = np.array([[1.0, 0.0]], dtype=np.float64)
    current = np.array([[np.float64(1.0) / depth[0, 0], 0.0]])
    arms = GATE.wind_expression_arms(rho, stress, current, depth, mask)
    expected = (rho * stress[0, 0]) * (
        mask[0, 0] / ((depth[0, 0] + 1.0) - mask[0, 0]))
    assert arms["nemo_written"][0, 0].view(np.uint64) == expected.view(np.uint64)
    assert (
        arms["right_associated"][0, 0].view(np.uint64)
        != arms["nemo_written"][0, 0].view(np.uint64)
    )
    assert np.array_equal(arms["stored_reciprocal"][:, 1], np.zeros(1))


def test_smagorinsky_branch_census_is_exhaustive_and_upper_owns_final_min():
    diagnostics = {
        "t_lower_clamped": np.array([True, False, True, False]),
        "t_upper_clamped": np.array([False, False, True, True]),
        "f_lower_clamped": np.array([True, False, False]),
        "f_upper_clamped": np.array([False, False, True]),
    }
    census = GATE.coefficient_branch_census(
        diagnostics, np.ones(4, dtype=bool), np.ones(3, dtype=bool))
    assert census["T"]["counts"] == {
        "lower_limit": 1, "deformation": 1, "upper_limit": 2}
    assert census["F"]["counts"] == {
        "lower_limit": 1, "deformation": 1, "upper_limit": 1}
    assert sum(census["T"]["fractions"].values()) == 1.0


@pytest.mark.timeout(900)
def test_r4_wind_grouping_does_not_explain_oracle_slow_debt():
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
        nemo_qco_live_face_geometry_from_operands,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=1, nn_stock=1, nn_write=1))
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    trace = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)

    dtype = state.u.data.dtype
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(dtype)
    umask3, vmask3 = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    ops = nemo_qco_card_mesh_operands(
        h_ref, umask3.astype(dtype), vmask3.astype(dtype),
        card.recipe.grid, dtype)
    live = nemo_qco_live_face_geometry_from_operands(
        state.eta.data, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)

    got_u = np.asarray(trace.slow_forcing_operands["wind_r1_hu"])[:, 1:]
    got_v = np.asarray(trace.slow_forcing_operands["wind_r1_hv"])[1:, :]
    expected_u = np.asarray(live.r1_hu)
    expected_v = np.asarray(live.r1_hv)
    wet_u = np.asarray(state.u_mask.data[:, 1:], dtype=bool)
    wet_v = np.asarray(state.v_mask.data[1:, :], dtype=bool)
    assert np.array_equal(
        got_u[wet_u].view(np.uint64), expected_u[wet_u].view(np.uint64))
    assert np.array_equal(
        got_v[wet_v].view(np.uint64), expected_v[wet_v].view(np.uint64))

    # At the rest state the algebraic reciprocal happens to round identically;
    # this is measured explicitly rather than used as a non-vacuous plant.
    old_u = np.asarray(1.0 / trace.slow_forcing_operands["H_u"])[:, 1:]
    assert np.array_equal(
        old_u[wet_u].view(np.uint64), expected_u[wet_u].view(np.uint64))

    oracle = GATE.read_bt_substeps(
        GATE.DEFAULT_ORACLES["r4"] / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=GATE.NEMO_DIMS[:2])
    actual_slow = np.asarray(trace.slow_forcing[1])[:, 1:]
    arms = GATE.wind_expression_arms(
        trace.slow_forcing_operands["wind_r1_rho0"],
        np.asarray(trace.slow_forcing_operands["wind_tau_u"])[:, 1:],
        got_u,
        np.asarray(trace.slow_forcing_operands["H_u"])[:, 1:],
        wet_u,
    )
    scores = {
        name: GATE.exact_score(name, oracle["slow_u"][0], value, wet_u)
        for name, value in {
            "production_slow": actual_slow,
            "left_associated_wind": arms["nemo_written"],
            "right_associated_wind": arms["right_associated"],
            "reciprocal_first_wind": arms["reciprocal_first"],
        }.items()
    }
    assert np.array_equal(
        actual_slow[wet_u].view(np.uint64),
        arms["nemo_written"][wet_u].view(np.uint64))
    assert all(row["status"] == "DEBT" for row in scores.values())
    assert scores["production_slow"]["n_unequal"] == 4158
    assert scores["production_slow"]["ulp_at_max_abs"] == 2


def test_shared_entry_reader_accepts_runtime_high_resolution_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    payload = np.arange(4 * count + dims[0] * dims[1], dtype=np.float64)
    path = tmp_path / "oracle_step_entry_kt00000001.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L1_ENTRY_1 ")
        stream.write(struct.pack("=8i", 1, 1, 1, *dims, 2, 64))
        payload.tofile(stream)
    record = GATE.read_entry(path, dims=dims)
    assert record["kt"] == 1 and record["Nbb"] == 1
    assert record["T"].shape == (2, 2, 2)
    assert record["ssh"].shape == (2, 2)


def test_shared_stage_and_zdf_readers_accept_runtime_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    stage_path = tmp_path / "oracle_stage_kt00000001_s1.bin"
    with stage_path.open("wb") as stream:
        stream.write(b"NEMO_L1_STAGE_1 ")
        stream.write(struct.pack("=9i", 1, 1, 1, 3, *dims, 2, 64))
        np.arange(4 * count + dims[0] * dims[1], dtype=np.float64).tofile(stream)
    stage = GATE.read_stage(stage_path, 1, dims=dims)
    assert stage["T"].shape == (2, 2, 2)

    zdf_path = tmp_path / "oracle_zdf_entry_kt00000001.bin"
    with zdf_path.open("wb") as stream:
        stream.write(b"NEMO_L2_ZDF___1 ")
        stream.write(struct.pack("=7i", 1, 1, 1, *dims, 64))
        np.arange(count + 2 * 2 * 2, dtype=np.float64).tofile(stream)
    zdf = GATE.read_zdf_entry(zdf_path, dims=dims)
    assert zdf["avm"].shape == (2, 2, 2)
    assert zdf["avt"].shape == (2, 2, 2)


def test_shared_tracer_stage_reader_accepts_runtime_dimensions(tmp_path):
    dims = (6, 6, 2)
    count = np.prod(dims)
    path = tmp_path / "oracle_rktracer_operands_kt00000001_s1.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L2_RKTRA_1 ")
        stream.write(struct.pack("=11i", 1, 1, 1, 1, 1, 3, 3, *dims, 64))
        np.arange(15 * count + 3 * dims[0] * dims[1], dtype=np.float64).tofile(stream)
    record = GATE.read_tracer_stage12(path, 1, dims=dims)
    assert record["after_advection_T"].shape == (2, 2, 2)
    assert record["after_sbc_S"].shape == (2, 2, 2)


def test_ahm_and_dynldf_readers_validate_headers_and_staggering(tmp_path):
    dims = (6, 6, 3)
    count = int(np.prod(dims))
    ahm_path = tmp_path / "oracle_ahm_kt00000002.bin"
    with ahm_path.open("wb") as stream:
        stream.write(b"NEMO_L4_AHMTF__1")
        stream.write(struct.pack("=7i", 1, 2, 3, *dims, 64))
        np.arange(2 * count, dtype=np.float64).tofile(stream)
    ahm = GATE.read_ahm(ahm_path, expected_kt=2, dims=dims)
    assert ahm["ahmt"].shape == (2, 2, 2)
    assert ahm["ahmf"].shape == (3, 3, 2)
    assert np.all(ahm["ahmf"][0] == 0.0)

    dyn_path = tmp_path / "oracle_dynldf_kt00000002.bin"
    with dyn_path.open("wb") as stream:
        stream.write(b"NEMO_L4_DYNLDF_1")
        stream.write(struct.pack("=10i", 1, 2, 3, 3, 2, 1, *dims, 64))
        np.arange(4 * count, dtype=np.float64).tofile(stream)
    dyn = GATE.read_dynldf(dyn_path, expected_kt=2, dims=dims)
    assert dyn["u_before"].shape == (2, 2, 2)
    assert dyn["Kbb"] == 3 and dyn["Kmm"] == 2 and dyn["Krhs"] == 1

    bad = tmp_path / "bad.bin"
    bad.write_bytes(ahm_path.read_bytes().replace(b"NEMO_L4_AHMTF__1", b"NEMO_L4_AHMTF__X", 1))
    with np.testing.assert_raises_regex(ValueError, "magic"):
        GATE.read_ahm(bad, expected_kt=2, dims=dims)


def test_stage2_term_reader_validates_source_order_payload(tmp_path):
    dims = (6, 6, 3)
    count = int(np.prod(dims))
    path = tmp_path / "oracle_rkstage2_terms_kt00000001.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L2_RKTRM_1 ")
        stream.write(struct.pack("=9i", 1, 1, 2, 3, 2, *dims, 64))
        np.arange(8 * count, dtype=np.float64).tofile(stream)
    record = GATE.read_stage2_terms(path, dims=dims)
    assert record["before_u"].shape == (2, 2, 3)
    assert record["after_advection_v"].shape == (2, 2, 3)
    assert record["stage"] == 2 and record["Krhs"] == 2

    bad = tmp_path / "truncated.bin"
    bad.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="payload"):
        GATE.read_stage2_terms(bad, dims=dims)


def test_stage2_composition_reader_separates_raw_and_post_mean(tmp_path):
    dims = (6, 6, 3)
    n3 = int(np.prod(dims))
    n2 = dims[0] * dims[1]
    ni = (dims[0] - 4) * (dims[1] - 4)
    payload = np.arange(10 * n3 + 2 * n2 + 2 * ni, dtype=np.float64)
    path = tmp_path / "oracle_rkstage2_operands_kt00000001.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L2_RKSTG_1 ")
        stream.write(struct.pack(
            "=11i", 1, 1, 2, 1, 3, 2, 2, *dims, 64))
        payload.tofile(stream)
    record = GATE.read_stage2_composition(path, dims=dims)
    assert record["raw_Kaa_u"].shape == (2, 2, 3)
    assert record["post_mean_Kaa_v"].shape == (2, 2, 3)
    assert not np.array_equal(record["raw_Kaa_u"], record["post_mean_Kaa_u"])


def test_stage3_tracer_reader_accepts_selected_resolution(tmp_path):
    dims = (6, 6, 3)
    n3 = int(np.prod(dims))
    n2 = dims[0] * dims[1]
    payload = np.arange(16 * n3 + 3 * n2, dtype=np.float64)
    path = tmp_path / "oracle_rktracer_stage3_kt00000001.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L2_RKTR3_1 ")
        stream.write(struct.pack(
            "=11i", 1, 1, 3, 1, 2, 3, 3, *dims, 64))
        payload.tofile(stream)
    record = GATE.read_tracer_stage3_hires(path, dims=dims)
    assert record["after_qsr_T"].shape == (2, 2, 3)
    assert record["r3t_Kaa"].shape == (2, 2)

    bad = tmp_path / "bad.bin"
    bad.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="payload"):
        GATE.read_tracer_stage3_hires(bad, dims=dims)


def test_first_residual_removing_substitution_uses_zero_word_bar():
    rows = [
        {"variable": "avt", "temperature": {"n_unequal": 4}},
        {"variable": "content", "temperature": {"n_unequal": 0}},
        {"variable": "later", "temperature": {"n_unequal": 0}},
    ]
    assert GATE.first_residual_removing_substitution(rows) == "content"
    rows[1]["temperature"]["n_unequal"] = 1
    rows[2]["temperature"]["n_unequal"] = 2
    assert GATE.first_residual_removing_substitution(rows) is None


def test_minimal_exact_subsets_requires_every_requested_field():
    def row(members, t_words, s_words):
        return {
            "variable": "+".join(members),
            "members": list(members),
            "temperature": {"n_unequal": t_words},
            "salinity": {"n_unequal": s_words},
        }

    rows = [
        row(("formed_K",), 3, 4),
        row(("temperature_content",), 2, 4),
        row(("formed_K", "temperature_content"), 0, 4),
        row(("formed_K", "temperature_content", "salinity_content"), 0, 0),
        row(("formed_K", "temperature_content", "salinity_content", "e3t"), 0, 0),
    ]
    assert GATE.minimal_exact_subsets(
        rows, fields=("temperature",)) == [
            ["formed_K", "temperature_content"]]
    assert GATE.minimal_exact_subsets(rows) == [[
        "formed_K", "temperature_content", "salinity_content"]]


def test_content_materiality_requires_changed_words_and_lexicographic_gain():
    baseline = {"n_unequal": 8, "max_abs": 2.0e-14}
    assert GATE.content_substitution_is_material(
        baseline, {"n_unequal": 7, "max_abs": 3.0e-14}, 1)
    assert GATE.content_substitution_is_material(
        baseline, {"n_unequal": 8, "max_abs": 1.0e-14}, 1)
    assert not GATE.content_substitution_is_material(
        baseline, {"n_unequal": 8, "max_abs": 1.0e-14}, 0)
    assert not GATE.content_substitution_is_material(
        baseline, {"n_unequal": 9, "max_abs": 1.0e-14}, 1)


def test_vertical_solver_subset_search_is_exhaustive_and_preregistered_first():
    rows = GATE.ordered_vertical_solver_subsets()
    assert rows[:2] == [
        ("formed_K", "temperature_content"),
        ("formed_K", "temperature_content", "salinity_content"),
    ]
    assert len(rows) == 31
    assert len(set(rows)) == 31
    assert ("formed_K", "tracer_e3w", "tracer_e3t",
            "temperature_content", "salinity_content") in rows


def test_fixed_vertical_probe_exposes_independent_salinity_content():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOVerticalSolveTestInput,
        _NEMOWSTracerSolveTrace,
    )

    assert _NEMOVerticalSolveTestInput._fields[-2:] == (
        "salinity_content", "replace")
    assert _NEMOWSTracerSolveTrace._fields[6:8] == (
        "content_T", "content_S")


def test_content_walk_uses_existing_passive_trace_and_completed_boundary():
    source = (
        ROOT / "packages/ocean/legoesm/ocean/dynamics/"
        "ocean_model_latlon_cgrid.py"
    ).read_text(encoding="utf-8")
    assert "content_a = (advection_content_a" in source
    assert "return_final_content_terms" not in source
    gate_source = SCRIPT.read_text(encoding="utf-8")
    assert "control.step1_vertical_trace_passive" in gate_source
    assert "nemo_before_content - own_before_content" in gate_source
    assert "completed_content_after_exact_before" in gate_source
    assert '"term": "completed_content_boundary"' in gate_source
    assert "planted_content[wet] = np.nextafter" in gate_source
    assert "planted_term" not in gate_source
    assert 'parser.add_argument("--step1-content-walk-only"' in gate_source
    assert "if content_only:" in gate_source
    assert "step1-content-walk-v1" in gate_source
