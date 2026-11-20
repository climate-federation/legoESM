"""Direct C4 runner/scorer controls, including restart and non-vacuity plants."""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean import bulk_flux_omip as bulk

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_decision188_gate as decision188,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    c4_year_runner as runner,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    c4_year_scorer as scorer,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    scorer as c_ladder_scorer,
)

REF_C4 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round31/"
    "oracle_coupled_c_ladder/REF_C4"
)


@pytest.fixture(scope="module")
def real_carry_template() -> bulk.NemoORCA2SI3Carry:
    """Use the production 179-leaf carry schema with scalar-sized test leaves."""

    carry, _forcing, _sponge, _card, _model = c_ladder_scorer._c2_cold_start(
        REF_C4, n_steps=2
    )

    def scalar_leaf(value):
        if isinstance(value, int):
            return value
        array = np.asarray(value)
        scalar = array.reshape(-1)[0] if array.size else 0
        return jnp.asarray(scalar, dtype=array.dtype)

    small = jax.tree_util.tree_map(scalar_leaf, carry)
    leaves, _ = jax.tree_util.tree_flatten(small)
    assert len(leaves) == 179
    return small._replace(kt=1)


def _carry(
    template: bulk.NemoORCA2SI3Carry, seed: float, kt: int
) -> bulk.NemoORCA2SI3Carry:
    def seeded(value):
        if isinstance(value, int):
            return value
        array = np.asarray(value)
        if array.dtype.kind in "fc":
            return jnp.asarray(seed, dtype=array.dtype)
        return jnp.asarray(array, dtype=array.dtype)

    return jax.tree_util.tree_map(seeded, template)._replace(kt=int(kt))


def _pair(carry: bulk.NemoORCA2SI3Carry) -> bulk.NemoORCA2SI3Carry:
    def update(value):
        array = jnp.asarray(value)
        if array.dtype.kind in "fc":
            return array + jnp.asarray(1.0, dtype=array.dtype)
        return array

    return jax.tree_util.tree_map(update, carry)._replace(kt=carry.kt + 2)


def _reference(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "C_DECK_RECEIPT.json").write_text(
        json.dumps({"status": "PASS_EXACT_R339_R5_DECK_ICE5", "rung": "C4"}) + "\n",
        encoding="utf-8",
    )
    return root


def _assert_tree_bits_equal(left, right) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for got, want in zip(left_leaves, right_leaves, strict=True):
        got_array = np.asarray(got)
        want_array = np.asarray(want)
        assert got_array.dtype == want_array.dtype
        assert got_array.shape == want_array.shape
        assert got_array.tobytes() == want_array.tobytes()


def _empty_month(month: int) -> dict:
    window = runner.year_metrics.calendar_month_day_ranges()[month - 1]
    return {
        **window,
        "format": runner.C4_METRIC_FORMAT,
        "status": "EVALUATED_COMPLETE_MONTH",
        "metrics": {},
    }


def _synthetic_reports():
    admission = decision188.minimal_envelope_admission()
    candidate = scorer._add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.0), 0.0
    )
    envelope = scorer._add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.001), 0.001
    )
    return candidate, envelope, admission


def test_restart_identity_compiled_pairs_save_restore_bitwise(
    tmp_path: Path, real_carry_template
) -> None:
    """N pairs straight equals k pairs + checkpoint + restore + N-k pairs."""

    cold = _carry(real_carry_template, 1.0, 1)
    executable = jax.jit(_pair).lower(cold).compile()
    straight = cold
    for _ in range(4):
        straight = executable(straight)
    jax.block_until_ready(straight)

    split = cold
    for _ in range(2):
        split = executable(split)
    jax.block_until_ready(split)
    checkpoint = tmp_path / runner.checkpoint_name(1, 4)
    runner.write_checkpoint(
        checkpoint,
        split,
        month=1,
        completed_kt=4,
        reference_root=_reference(tmp_path / "reference"),
        month_report=_empty_month(1),
    )
    restored, metadata, _report = runner.restore_checkpoint(
        checkpoint, cold, tmp_path / "reference", validate_calendar=False
    )
    assert metadata["next_kt"] == 5
    assert jax.typeof(restored.kt).weak_type
    assert jax.jit(_pair).lower(cold).as_text() == jax.jit(_pair).lower(restored).as_text()
    for _ in range(2):
        restored = executable(restored)
    jax.block_until_ready(restored)
    _assert_tree_bits_equal(restored, straight)


def test_checkpoint_carries_every_nemo_orca2_si3_leaf(
    tmp_path: Path, real_carry_template
) -> None:
    carry = _carry(real_carry_template, 2.0, 3)
    path = tmp_path / runner.checkpoint_name(1, 2)
    written = runner.write_checkpoint(
        path,
        carry,
        month=1,
        completed_kt=2,
        reference_root=_reference(tmp_path / "reference"),
        month_report=_empty_month(1),
    )
    leaves, _ = jax.tree_util.tree_flatten(carry)
    assert written["leaf_count"] == len(leaves) == 179
    restored, _, _ = runner.restore_checkpoint(
        path, _carry(real_carry_template, 2.0, 1), tmp_path / "reference",
        validate_calendar=False,
    )
    _assert_tree_bits_equal(restored, carry)


def test_scorer_refuses_checkpoint_set_with_missing_month(
    tmp_path: Path, real_carry_template
) -> None:
    reference = _reference(tmp_path / "reference")
    for month in range(1, 13):
        completed = _empty_month(month)["end_day"] * runner.year_metrics.STEPS_PER_DAY
        runner.write_checkpoint(
            tmp_path / runner.checkpoint_name(month, completed),
            _carry(real_carry_template, float(month), completed + 1),
            month=month,
            completed_kt=completed,
            reference_root=reference,
            month_report=_empty_month(month),
        )
    june_kt = _empty_month(6)["end_day"] * runner.year_metrics.STEPS_PER_DAY
    (tmp_path / runner.checkpoint_name(6, june_kt)).unlink()
    with pytest.raises(scorer.C4ScoreError, match="exactly 12 months, found 11"):
        scorer.load_checkpoint_report(tmp_path)


def test_month_checkpoint_validator_refuses_wrong_carry_clock(tmp_path: Path) -> None:
    report = _empty_month(1)
    expected = report["end_day"] * runner.year_metrics.STEPS_PER_DAY
    path = tmp_path / runner.checkpoint_name(1, expected)
    metadata = {"month": 1, "completed_kt": expected - 2, "next_kt": expected - 1}
    with pytest.raises(RuntimeError, match="carry clock is not the end"):
        runner.validate_month_checkpoint(path, metadata, report)


def test_scorer_assembles_all_twelve_checkpoint_months(
    tmp_path: Path, real_carry_template
) -> None:
    reference = _reference(tmp_path / "reference")
    for month in range(1, 13):
        completed = _empty_month(month)["end_day"] * runner.year_metrics.STEPS_PER_DAY
        runner.write_checkpoint(
            tmp_path / runner.checkpoint_name(month, completed),
            _carry(real_carry_template, float(month), completed + 1),
            month=month,
            completed_kt=completed,
            reference_root=reference,
            month_report=_empty_month(month),
        )
    report = scorer.load_checkpoint_report(tmp_path)
    assert report["nemo_root"] == str(reference.resolve())
    assert [month["month"] for month in report["months"]] == list(range(1, 13))


def test_c4_control_withholds_ice_bar_and_reports_ocean_separately() -> None:
    candidate, envelope, admission = _synthetic_reports()
    result = scorer.score_c4(candidate, envelope, admission)
    assert result["status"] == "PENDING_USER_ICE_BAR"
    assert result["ocean_d188"]["verdict"] == "PASS"
    assert result["ocean_d188"]["months_passed"] == 12
    assert all(
        row["pass"]
        for month in result["months"]
        for name, row in month["quantities"].items()
        if name not in runner.ICE_KEYS
    )
    assert set(runner.ICE_KEYS) <= set(result["months"][0]["quantities"])
    ice = result["months"][0]["quantities"]["ICE_AREA_NH"]
    assert ice["verdict"] == "PENDING_USER_ICE_BAR"
    assert set(ice["monthly_mean_relative_difference"]["candidate_bars"]) == {
        "fixed_0p01", "three_times_envelope",
    }
    assert set(ice["mean_daily_relative_difference"]["candidate_bars"]) == {
        "fixed_0p01", "three_times_envelope",
    }


def test_one_month_one_ocean_quantity_plant_flips_pass_to_fail() -> None:
    candidate, envelope, admission = _synthetic_reports()
    candidate["months"][2]["metrics"]["SST"]["rms"] = math.nextafter(0.01, math.inf)
    result = scorer.score_c4(candidate, envelope, admission)
    assert result["status"] == "PENDING_USER_ICE_BAR"
    assert result["ocean_d188"]["verdict"] == "FAIL"
    assert result["ocean_d188"]["months_passed"] == 11
    assert result["ocean_d188"]["first_failure"] == {
        "month": "M03_Mar", "quantity": "SST",
    }
    assert result["months"][2]["quantities"]["SST"]["pass"] is False


def test_one_month_ice_area_plant_fires_both_candidate_bars_without_verdict() -> None:
    candidate, envelope, admission = _synthetic_reports()
    planted = copy.deepcopy(candidate)
    metric = planted["months"][5]["metrics"]["ICE_AREA_SH"]
    metric["candidate_monthly_mean"] = 10.2
    metric["absolute_difference"] = 0.2
    metric["relative_abs_difference"] = 0.02
    metric["mean_daily_relative_abs_difference"] = 0.02
    result = scorer.score_c4(planted, envelope, admission)
    assert result["status"] == "PENDING_USER_ICE_BAR"
    row = result["months"][5]["quantities"]["ICE_AREA_SH"]
    assert row["verdict"] == "PENDING_USER_ICE_BAR"
    assert row["monthly_mean_relative_difference"]["would_pass_fixed_0p01"] is False
    assert row["monthly_mean_relative_difference"][
        "would_pass_three_times_envelope"
    ] is False


def test_zero_ice_envelope_is_refused_before_verdict() -> None:
    candidate, envelope, admission = _synthetic_reports()
    metric = envelope["months"][0]["metrics"]["ICE_VOLUME_NH"]
    metric["candidate_monthly_mean"] = metric["nemo_monthly_mean"]
    metric["absolute_difference"] = 0.0
    metric["relative_abs_difference"] = 0.0
    with pytest.raises(scorer.C4ScoreError, match="envelope errors must be positive"):
        scorer.score_c4(candidate, envelope, admission)


def test_builtin_plants_name_ocean_and_ice_failures() -> None:
    plants = scorer.run_plants()
    assert plants["control"] == "PENDING_USER_ICE_BAR"
    assert plants["ocean_control"] == "PASS"
    assert plants["one_month_ocean_quantity_plant"] == "FAIL_D188_M04_Apr_SST"
    assert plants["one_month_ice_area_plant"] == "BOTH_CANDIDATE_BARS_FIRE"


def test_ice_envelope_nemo_mean_must_equal_scored_nemo_mean() -> None:
    candidate, envelope, admission = _synthetic_reports()
    envelope["months"][4]["metrics"]["ICE_AREA_NH"]["nemo_monthly_mean"] = 10.1
    with pytest.raises(scorer.C4ScoreError, match="envelope and scored NEMO means differ"):
        scorer.score_c4(candidate, envelope, admission)


def test_negative_ice_candidate_mean_is_refused() -> None:
    candidate, envelope, admission = _synthetic_reports()
    candidate["months"][0]["metrics"]["ICE_VOLUME_SH"][
        "candidate_monthly_mean"
    ] = -1.0
    with pytest.raises(scorer.C4ScoreError, match="candidate monthly mean is negative"):
        scorer.score_c4(candidate, envelope, admission)


def test_legacy_ice_report_without_daily_error_cannot_receive_verdict() -> None:
    candidate, envelope, admission = _synthetic_reports()
    del candidate["months"][0]["metrics"]["ICE_AREA_SH"][
        "mean_daily_relative_abs_difference"
    ]
    with pytest.raises(scorer.C4ScoreError, match="mean daily relative error is missing"):
        scorer.score_c4(candidate, envelope, admission)


def test_checkpoint_archive_missing_a_schema_leaf_is_refused(
    tmp_path: Path, real_carry_template
) -> None:
    reference = _reference(tmp_path / "reference")
    carry = _carry(real_carry_template, 3.0, 3)
    path = tmp_path / runner.checkpoint_name(1, 2)
    runner.write_checkpoint(
        path,
        carry,
        month=1,
        completed_kt=2,
        reference_root=reference,
        month_report=_empty_month(1),
    )
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload.pop("carry_00001")
    path.unlink()
    np.savez(path, **payload)
    with pytest.raises(RuntimeError, match="archive leaves differ from its carry schema"):
        runner.inspect_checkpoint(path)


def test_scorer_refuses_input_identity_change_between_months(
    tmp_path: Path, real_carry_template
) -> None:
    reference = _reference(tmp_path / "reference")
    identity = runner.checkpoint_input_identity(reference)
    for month in range(1, 13):
        completed = _empty_month(month)["end_day"] * runner.year_metrics.STEPS_PER_DAY
        month_identity = dict(identity)
        if month == 12:
            month_identity["ice_package_source_sha256"] = "0" * 64
        runner.write_checkpoint(
            tmp_path / runner.checkpoint_name(month, completed),
            _carry(real_carry_template, float(month), completed + 1),
            month=month,
            completed_kt=completed,
            reference_root=reference,
            month_report=_empty_month(month),
            input_identity=month_identity,
        )
    with pytest.raises(scorer.C4ScoreError, match="input_identity changed"):
        scorer.load_checkpoint_report(tmp_path)


def test_checkpoint_identity_requires_receipt_and_hashes_ice_package(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path / "reference")
    identity = runner.checkpoint_input_identity(reference)
    assert len(identity["ice_package_source_sha256"]) == 64
    (reference / "C_DECK_RECEIPT.json").unlink()
    with pytest.raises(RuntimeError, match="missing required coupled deck receipt"):
        runner.checkpoint_input_identity(reference)


def test_resolved_c4_controls_come_from_coupled_deck() -> None:
    assert runner.resolved_c4_controls(REF_C4) == {
        "nn_itend": 2920,
        "rn_Dt": 10800.0,
        "nn_leapy": 0,
        "calendar": "noleap",
        "ice_dimensions": (5, 10, 5),
    }


def test_nemo_ice_reader_rejects_wrong_clock_and_fill_values(tmp_path: Path) -> None:
    import netCDF4

    expected_kt = runner.year_metrics.STEPS_PER_DAY
    paths = []
    for rank in (0, 1):
        path = tmp_path / f"ORCA2_{expected_kt:08d}_restart_ice_{rank:04d}.nc"
        paths.append(path)
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("time", 1)
            dataset.createDimension("cat", 2)
            dataset.createDimension("y", 148)
            dataset.createDimension("x", 90)
            kt_ice = dataset.createVariable("kt_ice", "i8")
            kt_ice.assignValue(expected_kt + 1)
            for name in ("a_i", "v_i"):
                field = dataset.createVariable(
                    name, "f8", ("time", "cat", "y", "x"), fill_value=1.0e20
                )
                field[:] = 0.25
    with pytest.raises(RuntimeError, match="kt_ice .* != restart kt"):
        runner.read_nemo_ice_day(tmp_path, 1)
    for path in paths:
        with netCDF4.Dataset(path, "r+") as dataset:
            dataset.variables["kt_ice"].assignValue(expected_kt)
    with netCDF4.Dataset(paths[0], "r+") as dataset:
        dataset.variables["a_i"][0, 0, 0, 0] = 1.0e20
    with pytest.raises(RuntimeError, match="a_i contains fill/masked values"):
        runner.read_nemo_ice_day(tmp_path, 1)


def test_ice_volume_units_and_equator_belongs_to_southern_hemisphere() -> None:
    fields = {
        "area": np.ones((1, 2), dtype=np.float64),
        "volume": np.ones((1, 2), dtype=np.float64),
    }
    geo = {
        "area_t": np.full((1, 2), 1.0e12, dtype=np.float64),
        "gphit": np.asarray([[1.0, 0.0]], dtype=np.float64),
    }
    rows = runner.ice_day_metrics(fields, fields, geo)
    assert rows["ICE_VOLUME_NH"] == {
        "nemo": 1.0, "candidate": 1.0, "units": "1e3 km3",
    }
    assert rows["ICE_VOLUME_SH"] == {
        "nemo": 1.0, "candidate": 1.0, "units": "1e3 km3",
    }


def test_gpu_is_refused_at_argument_parsing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        runner,
        "run_c4",
        lambda *_args, **_kwargs: pytest.fail("run_c4 must not be entered on GPU"),
    )
    with pytest.raises(SystemExit) as raised:
        runner.main([
            "--reference", str(tmp_path / "reference"),
            "--output-dir", str(tmp_path / "output"),
        ])
    assert raised.value.code == 2


def test_run_c4_resume_matches_straight_bitwise_on_real_carry_schema(
    monkeypatch, tmp_path: Path, real_carry_template, capsys
) -> None:
    windows = [
        {
            "month": month,
            "label": f"M{month:02d}",
            "start_day": month,
            "end_day": month,
            "available_days": [month],
            "saved_kts": [2 * month],
            "complete_month": True,
        }
        for month in (1, 2)
    ]
    reference = _reference(tmp_path / "reference")
    straight_dir = tmp_path / "straight"
    resumed_dir = tmp_path / "resumed"
    card = SimpleNamespace(
        ocean=SimpleNamespace(dt_s=10800.0),
        ice=SimpleNamespace(state=SimpleNamespace(nn_hls=0)),
    )
    compile_hlo = []

    class FakeInputs:
        def __init__(self, *_args, **_kwargs):
            pass

        def record(self, kt):
            scalar = jnp.asarray(float(kt), dtype=jnp.float64)
            return scalar, scalar

    def fake_compile(carry, *_args):
        program = jax.jit(lambda entry, *_unused: _pair(entry))
        lowered = program.lower(
            carry,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )
        compile_hlo.append(lowered.as_text())
        return lowered.compile(), 0.0, lambda: 1

    def fake_month_report(month, *_args, metric_format, ice_sst_sss_mask_source):
        return {
            **windows[month - 1],
            "format": metric_format,
            "status": "EVALUATED_COMPLETE_MONTH",
            "ice_sst_sss_mask_source": ice_sst_sss_mask_source,
            "metrics": {},
        }

    controls = {
        "nn_itend": 4,
        "rn_Dt": 10800.0,
        "nn_leapy": 0,
        "calendar": "noleap",
        "ice_dimensions": (5, 10, 5),
    }
    cold = _carry(real_carry_template, 1.0, 1)
    monkeypatch.setattr(runner, "TOTAL_PAIRS", 2)
    monkeypatch.setattr(runner.year_metrics, "STEPS_PER_DAY", 2)
    monkeypatch.setattr(
        runner.year_metrics, "calendar_month_day_ranges", lambda: copy.deepcopy(windows)
    )
    monkeypatch.setattr(runner, "resolved_c4_controls", lambda _root: controls)
    monkeypatch.setattr(
        runner.c_ladder_scorer,
        "_c2_cold_start",
        lambda *_args, **_kwargs: (cold, (jnp.asarray(0.0),), (), card, object()),
    )
    monkeypatch.setattr(runner, "C4InputReader", FakeInputs)
    monkeypatch.setattr(runner, "compile_pair", fake_compile)
    monkeypatch.setattr(runner.year_metrics.pilot_metrics, "static_geometry", lambda _r: {})
    monkeypatch.setattr(runner.year_metrics, "K0MLDEvaluator", lambda _r: object())
    monkeypatch.setattr(
        runner, "read_coupled_initial_ice_cover", lambda _r: np.zeros((148, 180))
    )
    monkeypatch.setattr(runner.ladder, "_candidate_fields", lambda _o: {})
    monkeypatch.setattr(runner.year_metrics.pilot_metrics, "read_nemo_day", lambda *_a: {})
    monkeypatch.setattr(
        runner.year_metrics.pilot_metrics, "day_metrics", lambda *_a: ({}, {})
    )
    monkeypatch.setattr(runner.year_metrics, "_augment_day_single_arm_stats", lambda *_a: None)
    monkeypatch.setattr(runner, "candidate_ice_fields", lambda *_a: {})
    monkeypatch.setattr(runner, "read_nemo_ice_day", lambda *_a: {})
    monkeypatch.setattr(runner, "ice_day_metrics", lambda *_a: {})
    monkeypatch.setattr(runner, "build_month_report", fake_month_report)

    straight = runner.run_c4(reference, straight_dir)
    assert straight["status"] == "PASS_C4_RUN_2_CHECKPOINTS"
    resumed_dir.mkdir()
    first_name = runner.checkpoint_name(1, 2)
    shutil.copy2(straight_dir / first_name, resumed_dir / first_name)
    assert runner.main([
        "--reference", str(reference),
        "--output-dir", str(resumed_dir),
        "--resume", str(resumed_dir / first_name),
    ]) == 0
    capsys.readouterr()

    straight_final, _, _ = runner.restore_checkpoint(
        straight_dir / runner.checkpoint_name(2, 4), cold, reference
    )
    resumed_final, _, _ = runner.restore_checkpoint(
        resumed_dir / runner.checkpoint_name(2, 4), cold, reference
    )
    _assert_tree_bits_equal(resumed_final, straight_final)
    assert len(compile_hlo) == 2
    assert compile_hlo[0] == compile_hlo[1]
