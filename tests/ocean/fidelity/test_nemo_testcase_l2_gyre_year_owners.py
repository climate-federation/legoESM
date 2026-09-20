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
    vertical = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True))
    assert vertical._nemo_ws_test_hooks.vertical_solve_trace is True


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
