"""Direct unit tests and planted-violation controls for ORCA2 Round 343 speed benchmark."""

from __future__ import annotations

import copy
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
