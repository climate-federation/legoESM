"""Direct unit tests and planted-violation controls for ORCA2 Round 343 speed benchmark."""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round343_speed_benchmark as r343,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_citation(citation: str) -> tuple[Path, int, int]:
    path_str, line_spec = citation.split(":", maxsplit=1)
    if "-" in line_spec:
        start_s, end_s = line_spec.split("-", maxsplit=1)
        return REPO_ROOT / path_str, int(start_s), int(end_s)
    line_no = int(line_spec)
    return REPO_ROOT / path_str, line_no, line_no


def test_switch_inventory_citations_and_class_defaults() -> None:
    inventory = r343.build_orca2_exact_vs_normal_switch_inventory()
    assert len(inventory) == 46
    cat_a = [s for s in inventory if s.category == "A_libm_guarded"]
    cat_b = [s for s in inventory if s.category == "B_source_rounding_or_order"]
    cat_c = [s for s in inventory if s.category == "C_unconditional_barrier"]
    cat_d = [s for s in inventory if s.category == "D_structural_scheme_selector"]
    assert len(cat_a) == 6
    assert len(cat_b) == 36
    assert len(cat_c) == 1
    assert len(cat_d) == 3
    assert sum(1 for s in inventory if s.switched_in_normal_mode) == 42

    for spec in inventory:
        for citation in (spec.exact_citation, spec.normal_citation):
            path, start_line, end_line = _parse_citation(citation)
            assert path.is_file(), f"missing cited file for {spec.name}: {path}"
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            assert 1 <= start_line <= end_line <= n_lines, (
                f"invalid line range {citation} (file has {n_lines} lines)"
            )


@pytest.mark.skipif(
    not r343.DEFAULT_DECK_ROOT.is_dir(),
    reason="ORCA2 K0 deck root not present on this machine",
)
def test_normal_mode_card_builder_and_planted_violations() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    base_card = bulk.build_nemo_orca2_k0_ocean_card(r343.DEFAULT_DECK_ROOT)
    normal_card = r343.build_normal_mode_orca2_k0_card(base_card, variant="cat_a_plus_b")

    summary = r343.verify_switch_inventory_against_card(base_card, normal_card)
    assert summary["switched_in_normal_mode"] == 42
    assert summary["category_a_count"] == 6
    assert summary["category_b_count"] == 36
    # Verify base_card was not mutated.
    assert base_card.ocean.transcendentals == "libm"
    assert normal_card.ocean.transcendentals == "native"

    # Control 1: planted exact value on a switched Category A setting must fail.
    bad_normal_a = normal_card._replace(
        ocean=normal_card.ocean._replace(transcendentals="libm")
    )
    with pytest.raises(r343.GateError, match="normal_card switch"):
        r343.verify_switch_inventory_against_card(base_card, bad_normal_a)

    # Control 2: planted exact value on a switched Category B setting must fail.
    mc_bad_b = normal_card.ocean.recipe.model_config._replace(
        een_vorticity_written_order=True
    )
    bad_normal_b = normal_card._replace(
        ocean=normal_card.ocean._replace(
            recipe=normal_card.ocean.recipe._replace(model_config=mc_bad_b)
        )
    )
    with pytest.raises(r343.GateError, match="een_vorticity_written_order"):
        r343.verify_switch_inventory_against_card(base_card, bad_normal_b)

    # Control 3: planted mutation on a retained Category D setting must fail.
    mc_bad_d = normal_card.ocean.recipe.model_config._replace(
        zdf_implicit_solver_evaluation="shared_thomas"
    )
    bad_normal_d = normal_card._replace(
        ocean=normal_card.ocean._replace(
            recipe=normal_card.ocean.recipe._replace(model_config=mc_bad_d)
        )
    )
    with pytest.raises(r343.GateError, match="retained switch"):
        r343.verify_switch_inventory_against_card(base_card, bad_normal_d)

    # Control 4: planted wrong normal_value in inventory must fail against class defaults.
    tampered_inventory = list(r343.build_orca2_exact_vs_normal_switch_inventory())
    tampered_inventory[2] = tampered_inventory[2]._replace(normal_value="nemo_scalar_libm")
    with pytest.raises(r343.GateError, match="class default"):
        r343.verify_switch_inventory_against_card(
            base_card, normal_card, inventory=tampered_inventory
        )


def test_step_timing_summary_and_planted_controls() -> None:
    # 2 warmup steps (100s, 90s) + 20 steady-state steps alternating 0.8s (odd kt) and 0.2s (even kt).
    step_times = [100.0, 90.0] + [0.8 if i % 2 == 0 else 0.2 for i in range(20)]
    summary = r343.compute_step_timing_summary(step_times)
    assert summary["total_steps"] == 22
    assert summary["warmup_steps"] == 2
    assert summary["steady_state_steps"] == 20
    assert np.isclose(summary["compile_warmup_total_s"], 190.0)
    assert np.isclose(summary["steady_odd_kt_median_s"], 0.8)
    assert np.isclose(summary["steady_even_kt_median_s"], 0.2)
    assert np.isclose(summary["steady_median_s"], 0.5)
    assert np.isclose(summary["steady_mean_s"], 0.5)
    assert np.isclose(summary["projected_year_steady_s"], 2920.0 * 0.5)
    assert np.isclose(summary["projected_year_with_compile_s"], 190.0 + 2918.0 * 0.5)

    # Control 1: fewer than 20 steady-state steps must fail.
    with pytest.raises(r343.GateError, match="steady"):
        r343.compute_step_timing_summary(step_times[:-1])

    # Control 2: non-positive step time must fail.
    bad_zero = list(step_times)
    bad_zero[5] = 0.0
    with pytest.raises(r343.GateError, match="finite and positive"):
        r343.compute_step_timing_summary(bad_zero)

    # Control 3: NaN step time must fail.
    bad_nan = list(step_times)
    bad_nan[3] = float("nan")
    with pytest.raises(r343.GateError, match="finite and positive"):
        r343.compute_step_timing_summary(bad_nan)


def test_daily_temperature_scorer_and_report_validator() -> None:
    ref = np.zeros((148, 180, 30), dtype=np.float64)
    wet = np.zeros((148, 180, 30), dtype=bool)
    wet[10:20, 10:20, :5] = True

    zero_score = r343.score_daily_temperature_against_reference(ref, ref, wet, kt=8)
    assert zero_score["day"] == 1
    assert zero_score["sst_rms_all_c"] == 0.0
    assert zero_score["sst_rms_wet_c"] == 0.0
    assert zero_score["sst_max_abs_c"] == 0.0
    assert zero_score["t3d_rms_all_c"] == 0.0
    assert zero_score["t3d_rms_wet_c"] == 0.0
    assert zero_score["t3d_max_abs_c"] == 0.0

    cand = ref.copy()
    cand[10:20, 10:20, 0] = 2.0  # 100 wet surface cells = +2.0 C
    score = r343.score_daily_temperature_against_reference(cand, ref, wet, kt=16)
    assert score["day"] == 2
    assert np.isclose(score["sst_rms_wet_c"], 2.0)
    assert np.isclose(score["sst_max_abs_c"], 2.0)
    assert np.isclose(score["sst_rms_all_c"], np.sqrt(100.0 * 4.0 / (148.0 * 180.0)))

    # Planted NaN must fail.
    cand_nan = cand.copy()
    cand_nan[0, 0, 0] = np.nan
    with pytest.raises(r343.GateError, match="non-finite"):
        r343.score_daily_temperature_against_reference(cand_nan, ref, wet, kt=8)

    # Build a synthetic valid report and verify planted report violations fail.
    timing = r343.compute_step_timing_summary([10.0, 10.0] + [0.5] * 20)
    mem = {"host_peak_rss_mib": 1024.0}
    exact_daily = [
        r343.score_daily_temperature_against_reference(ref, ref, wet, kt=kt)
        for kt in r343.DAILY_CHECK_STEPS_EXACT
    ]
    normal_daily = [
        r343.score_daily_temperature_against_reference(cand, ref, wet, kt=kt)
        for kt in r343.DAILY_CHECK_STEPS_8D
    ]
    valid_report = {
        "arms": {
            "exact_cpu_fp64": {"timing": timing, "memory": mem, "daily_accuracy": exact_daily},
            "normal_cpu_fp64": {"timing": timing, "memory": mem, "daily_accuracy": normal_daily},
            "normal_gpu_fp64": {"timing": timing, "memory": mem, "daily_accuracy": normal_daily},
            "normal_gpu_fp32": {"timing": timing, "memory": mem, "daily_accuracy": normal_daily},
        }
    }
    r343.validate_benchmark_report(valid_report)

    # Planted non-zero drift on exact_cpu_fp64 must fail.
    bad_exact = copy.deepcopy(valid_report)
    bad_exact["arms"]["exact_cpu_fp64"]["daily_accuracy"][0]["sst_max_abs_c"] = 1.0e-12
    with pytest.raises(r343.GateError, match="drifted from exact reference"):
        r343.validate_benchmark_report(bad_exact)

    # Planted missing day in normal_gpu_fp64 must fail.
    bad_days = copy.deepcopy(valid_report)
    bad_days["arms"]["normal_gpu_fp64"]["daily_accuracy"] = normal_daily[:-1]
    with pytest.raises(r343.GateError, match="expected 8 daily_accuracy kts"):
        r343.validate_benchmark_report(bad_days)


def test_round349_cpu_pinning_and_thread_tick_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = (
        r343.CpuTopologyRow(0, 0, 0, (0, 4)),
        r343.CpuTopologyRow(1, 1, 0, (1, 5)),
        r343.CpuTopologyRow(2, 0, 1, (2, 6)),
        r343.CpuTopologyRow(3, 1, 1, (3, 7)),
        r343.CpuTopologyRow(4, 0, 0, (0, 4)),
        r343.CpuTopologyRow(5, 1, 0, (1, 5)),
        r343.CpuTopologyRow(6, 0, 1, (2, 6)),
        r343.CpuTopologyRow(7, 1, 1, (3, 7)),
    )
    assert r343.select_physical_cpu_ids(4, topology) == (0, 1, 2, 3)
    assert r343.parse_cpu_id_list("0,2-4,7") == (0, 2, 3, 4, 7)
    assert r343.format_cpu_id_list((0, 2, 3)) == "0,2,3"
    with pytest.raises(r343.GateError, match="requested 5 physical cores"):
        r343.select_physical_cpu_ids(5, topology)
    with pytest.raises(r343.GateError, match="duplicate CPU id"):
        r343.parse_cpu_id_list("0,1,1")

    before = {100: 10, 101: 20}
    after = {
        100: 10 + r343.CLK_TCK,
        101: 20 + r343.CLK_TCK // 2,
    }
    usage = r343.summarize_thread_cpu_ticks(
        before,
        after,
        wall_s=1.0,
        process_ticks_before=30,
        process_ticks_after=30 + r343.CLK_TCK + r343.CLK_TCK // 2,
    )
    expected = 1.0 + (r343.CLK_TCK // 2) / r343.CLK_TCK
    assert np.isclose(usage["cores_busy"], expected)
    assert usage["os_threads_after"] == 2
    assert usage["thread_cpu_seconds"] == [
        {"tid": 100, "cpu_seconds": 1.0},
        {"tid": 101, "cpu_seconds": (r343.CLK_TCK // 2) / r343.CLK_TCK},
    ]

    affinity = (0, 1)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _tid: set(affinity))
    thread_record = r343.read_thread_affinity_record(affinity)
    assert thread_record["affinity_exact"] is True
    assert thread_record["expected_cpu_ids"] == list(affinity)
    assert all(row["affinity"] == list(affinity) for row in thread_record["threads"])
    with pytest.raises(r343.GateError, match="thread affinity differs"):
        r343.read_thread_affinity_record((*affinity, 2))


def test_round349_nemo_process_detection() -> None:
    sample = r343.ProcessCpuSample
    processes = {
        10: sample(0, "mpirun", 1),
        11: sample(0, "nemo", 10),
        20: sample(0, "orterun", 1),
        21: sample(0, "nemo.exe", 20),
        30: sample(0, "prterun", 1),
        31: sample(0, "nemo", 30),
        40: sample(0, "prterun", 1),
        41: sample(0, "python", 40),
        50: sample(0, "nemo-helper", 1),
    }
    assert r343.find_nemo_processes(processes) == [10, 11, 20, 21, 30, 31]


def test_round349_steady_window_uses_actual_steps_and_balances_parity() -> None:
    assert r343.round349_steady_timing_window(4) == {
        "first_step": 3,
        "last_step": 4,
    }
    assert r343.round349_steady_timing_window(18) == {
        "first_step": 3,
        "last_step": 18,
    }
    with pytest.raises(r343.GateError, match="empty"):
        r343.round349_steady_timing_window(2)
    with pytest.raises(r343.GateError, match="equal nn_fsbc parities"):
        r343.round349_steady_timing_window(5)


def test_round349_sypd_nemo_timing_and_time_parsers(tmp_path: Path) -> None:
    assert np.isclose(r343.compute_sypd(1.0), 86400.0 / 2920.0)
    with pytest.raises(r343.GateError, match="finite and positive"):
        r343.compute_sypd(0.0)

    timing_text = """
   |         Performance statistics between time step 7 and 92        |
 Timing : AVG values over all MPI processes:
 Section      | Net elapsed Time         | Full elapsed Time        | Frequency
 step         |   43.000000s (100.000 %) |   43.000000s (100.000 %) |        86
 """
    timing = r343.parse_nemo_timing_output(timing_text)
    assert timing["steady_steps"] == 86
    assert np.isclose(timing["seconds_per_step"], 0.5)

    gnu = r343.parse_gnu_time_metrics(
        "elapsed_s=50.0\nuser_s=150.0\nsystem_s=5.0\nmax_rss_kib=1024\n"
    )
    assert gnu == {
        "elapsed_s": 50.0,
        "user_s": 150.0,
        "system_s": 5.0,
        "max_rss_kib": 1024.0,
    }

    bad_frequency = timing_text.replace("86", "85")
    with pytest.raises(r343.GateError, match="does not match window"):
        r343.parse_nemo_timing_output(bad_frequency)

    run_dir = tmp_path / "nemo"
    run_dir.mkdir()
    (run_dir / "timing.output").write_text(timing_text)
    (run_dir / "gnu_time.txt").write_text(
        "elapsed_s=50\nuser_s=1\nsystem_s=1\nmax_rss_kib=1024\n"
    )
    (run_dir / "rank_time0.txt").write_text(
        "elapsed_s=49\nuser_s=44\nsystem_s=1\nmax_rss_kib=1024\n"
    )
    (run_dir / "binding_rank0.txt").write_text("Cpus_allowed_list:\t0\n")
    (run_dir / "nemo").write_bytes(b"certified")
    (run_dir / "namelist_cfg").write_text("&namrun\n/")
    binary_sha = r343.sha256_file(run_dir / "nemo")
    (run_dir / "producer_manifest.json").write_text(
        json.dumps({"binary_sha256": binary_sha})
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"loadavg": [0.0], "available_memory_gib": 200.0}))
    artifact = r343.build_round349_nemo_artifact(
        run_dir, cores=1, cpu_ids=(0,), preflight_path=preflight
    )
    assert artifact["rank_bindings"] == {"binding_rank0": [0]}
    assert np.isclose(artifact["row"]["busy_cores"], 0.9)

    (run_dir / "binding_rank0.txt").write_text("Cpus_allowed_list:\t1\n")
    with pytest.raises(r343.GateError, match="differs from request"):
        r343.build_round349_nemo_artifact(
            run_dir, cores=1, cpu_ids=(0,), preflight_path=preflight
        )


def _round349_complete_report() -> dict[str, object]:
    rows = []
    for cores in r343.ROUND349_CORE_COUNTS:
        cpu_ids = list(range(cores))
        rows.append(
            r343.round349_performance_row(
                model="NEMO",
                mode="nemo_5.0.2",
                cores=cores,
                seconds_per_step=1.0 / cores,
                busy_cores=float(cores),
                steps=96,
                steady_steps=86,
                cpu_ids=cpu_ids,
                background_load={"max": 0.0},
                artifact=f"nemo_n{cores}.json",
                measurement_provenance={"backend": "NEMO", "dtype": "float64", "identity": {"sha": "x"}},
                timing_window={"first_step": 7, "last_step": 92},
                seconds_per_step_statistic="MPI AVG elapsed / frequency",
            )
        )
        for mode in r343.ROUND349_MODES:
            seconds = 1.0 if cores == 1 else 0.9
            rows.append(
                r343.round349_performance_row(
                    model="legoESM",
                    mode=mode,
                    cores=cores,
                    seconds_per_step=seconds,
                    busy_cores=1.5 if cores == 40 else 1.0,
                    steps=18,
                    steady_steps=16,
                    cpu_ids=cpu_ids,
                    background_load={"max": 0.0},
                    artifact=f"legoesm_{mode}_n{cores}.json",
                    measurement_provenance={"backend": "cpu", "dtype": "float64", "identity": {"git": "x"}},
                    timing_window={"first_step": 3, "last_step": 18},
                    seconds_per_step_statistic="steady median",
                )
            )
            rows[-1]["thread_affinity_record"] = {
                "expected_cpu_ids": cpu_ids,
                "affinity_exact": True,
                "tf_XLAEigen_thread_count": cores,
                "other_worker_thread_count": 0,
                "threads": [
                    {"tid": 100 + index, "name": "tf_XLAEigen", "affinity": cpu_ids}
                    for index in range(cores)
                ],
            }
            rows[-1]["per_thread_cpu_seconds_by_step"] = [
                {"kt": kt, "threads": []} for kt in range(3, 19)
            ]
    profiles = [
        {
            "mode": "exact",
            "cores": cores,
            "compiled_kernel_wall_fraction": 0.75,
            "eager_or_dispatch_fraction": 0.25,
            "eager_surface_wall_fraction": 0.10,
            "cost_centres": [{"cost_centre": "EOS / density"}],
            "profile_step": r343.ROUND349_PROFILE_STEP,
            "trace_path": "trace.json.gz",
            "provenance": {"commit": "abc", "clean": True},
            "background_load": {"loadavg": [0.0]},
            "measurement_provenance": {"backend": "cpu", "dtype": "float64"},
        }
        for cores in (1, 40)
    ]
    return {"round": "orca2_r349", "rows": rows, "profiles": profiles}


def test_round349_report_schema_and_arithmetic_controls() -> None:
    report = _round349_complete_report()
    r343.validate_round349_report(report)

    bad_sypd = copy.deepcopy(report)
    bad_sypd["rows"][0]["sypd"] *= 2.0
    with pytest.raises(r343.GateError, match="invalid sypd"):
        r343.validate_round349_report(bad_sypd)

    missing_row = copy.deepcopy(report)
    missing_row["rows"].pop()
    with pytest.raises(r343.GateError, match="row set mismatch"):
        r343.validate_round349_report(missing_row)

    bad_profile = copy.deepcopy(report)
    bad_profile["profiles"][0]["eager_or_dispatch_fraction"] = 1.5
    with pytest.raises(r343.GateError, match="invalid profile"):
        r343.validate_round349_report(bad_profile)

    inconsistent_profile = copy.deepcopy(report)
    inconsistent_profile["profiles"][0]["eager_or_dispatch_fraction"] = 0.20
    with pytest.raises(r343.GateError, match="must sum to one"):
        r343.validate_round349_report(inconsistent_profile)

    zero_steps = copy.deepcopy(report)
    zero_steps["rows"][0]["steps"] = 0
    with pytest.raises(r343.GateError, match="steps must be positive"):
        r343.validate_round349_report(zero_steps)

    escaped_affinity = copy.deepcopy(report)
    lego_row = next(row for row in escaped_affinity["rows"] if row["model"] == "legoESM")
    lego_row["thread_affinity_record"]["affinity_exact"] = False
    with pytest.raises(r343.GateError, match="exact thread-affinity"):
        r343.validate_round349_report(escaped_affinity)


def test_round349_perfetto_profile_aggregation(tmp_path: Path) -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": 10,
                "args": {"name": "python-main"},
            },
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": 20,
                "args": {"name": "tf_XLAEigen"},
            },
            {
                "ph": "X",
                "name": r343.ROUND349_PROFILE_ANNOTATION,
                "pid": 1,
                "tid": 10,
                "ts": 0.0,
                "dur": 1000.0,
            },
            {
                "ph": "X",
                "name": "eos_density_kernel",
                "pid": 1,
                "tid": 20,
                "ts": 100.0,
                "dur": 600.0,
            },
            {
                "ph": "X",
                "name": "nemo_surface_card",
                "pid": 1,
                "tid": 10,
                "ts": 0.0,
                "dur": 100.0,
            },
            {
                "ph": "X",
                "name": r343.ROUND349_SURFACE_ANNOTATION,
                "pid": 1,
                "tid": 10,
                "ts": 0.0,
                "dur": 100.0,
            },
        ]
    }
    path = tmp_path / "perfetto_trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)
    profile = r343.summarize_round349_perfetto(path, cores=1)
    assert np.isclose(profile["step_wall_s"], 0.001)
    assert np.isclose(profile["compiled_kernel_wall_fraction"], 0.6)
    assert np.isclose(profile["eager_or_dispatch_fraction"], 0.4)
    assert np.isclose(profile["eager_surface_wall_fraction"], 0.1)
    centres = {row["cost_centre"] for row in profile["cost_centres"]}
    assert "EOS / density" in centres
    assert "surface bulk flux" in centres


def test_round349_dry_run_command_schema(tmp_path: Path) -> None:
    commands = r343.build_round349_legoesm_commands(
        output_dir=r343.ROUND349_SCRATCH_ROOT / "dry_run_test",
        core_counts=(1, 2),
        steps=r343.ROUND349_SMOKE_STEPS,
        smoke=True,
        include_profiles=False,
    )
    assert len(commands) == 2
    for (command, env), cores in zip(commands, (1, 2), strict=True):
        assert "taskset" in Path(command[0]).name
        assert command[1:3] == ["-c", r343.format_cpu_id_list(r343.select_physical_cpu_ids(cores))]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_ENABLE_X64"] == "1"
        assert env["OMP_NUM_THREADS"] == str(cores)
        assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=true"
        assert "intra_op_parallelism_threads" not in env["XLA_FLAGS"]
