#!/usr/bin/env python3
"""Split rung-0 substep-3 AB3 midpoint V across its recorded inputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round146_boundary_association_gate as r146,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round195_transport_operands as r195,
)

SUBSTEP = 3
INPUT_ORDER = ("za1", "za2", "za3", "vn_e", "vb_e", "vbb_e")
RECORD_FIELDS = (
    "j003_ext_coef", "j002_va_new", "j001_va_new", "i000_vn_e",
    "j003_va_ext",
)
PLANTS = (
    "none", "registry-order", "coefficient-bit", "rotation-map",
    "missing-stream",
)


class GateError(RuntimeError):
    """The record, source order, or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _row(candidate, reference) -> dict[str, object]:
    return r146.exact_row(np.asarray(candidate), np.asarray(reference))


def _literal_midpoint(coefficients, now, before, before_before):
    import jax

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_midpoint_extrapolation,
    )

    return np.asarray(jax.device_get(nemo_literal_midpoint_extrapolation(
        coefficients, now, before, before_before)))


def _literal_terms(coefficients, now, before, before_before):
    """Expose the shared helper's written multiply/add boundaries offline."""

    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    now = jnp.asarray(now, dtype=jnp.float64)
    before = jnp.asarray(before, dtype=jnp.float64)
    before_before = jnp.asarray(before_before, dtype=jnp.float64)
    first = nemo_source_round(coefficients[0] * nemo_source_round(now))
    second = nemo_source_round(coefficients[1] * nemo_source_round(before))
    third = nemo_source_round(
        coefficients[2] * nemo_source_round(before_before))
    first_two = nemo_source_round(first + second)
    completed = nemo_source_round(first_two + third)
    values = jax.device_get((first, second, third, first_two, completed))
    return tuple(np.asarray(value) for value in values)


def split_inputs(trace: dict[str, object], oracle: dict[str, object]) -> dict[str, object]:
    """Replay the substep-3 midpoint statement from passive recorded inputs."""

    index = SUBSTEP - 1
    candidate_coefficients = np.asarray([
        trace["mid_weight_1"][index],
        trace["mid_weight_2"][index],
        trace["mid_weight_3"][index],
    ], dtype=np.float64)
    reference_coefficients = np.asarray(
        oracle["j003_ext_coef"], dtype=np.float64)
    require(candidate_coefficients.shape == reference_coefficients.shape == (3,),
            "AB3 coefficient shape moved")

    candidate_inputs = {
        "za1": candidate_coefficients[0],
        "za2": candidate_coefficients[1],
        "za3": candidate_coefficients[2],
        "vn_e": r178._native_v(trace["v_entry"][index]),
        "vb_e": r178._native_v(trace["v_history_b"][index]),
        "vbb_e": r178._native_v(trace["v_history_bb"][index]),
    }
    reference_inputs = {
        "za1": reference_coefficients[0],
        "za2": reference_coefficients[1],
        "za3": reference_coefficients[2],
        "vn_e": np.asarray(oracle["j002_va_new"], dtype=np.float64),
        "vb_e": np.asarray(oracle["j001_va_new"], dtype=np.float64),
        "vbb_e": np.asarray(oracle["i000_vn_e"], dtype=np.float64),
    }
    target = np.asarray(oracle["j003_va_ext"], dtype=np.float64)
    candidate_target = r178._native_v(trace["v_mid"][index])
    require(target.shape == candidate_target.shape, "midpoint V target shape moved")
    require(all(np.asarray(reference_inputs[name]).shape == target.shape
                for name in ("vn_e", "vb_e", "vbb_e")),
            "recorded AB3 history shape moved")

    candidate_replay = _literal_midpoint(
        candidate_coefficients, candidate_inputs["vn_e"],
        candidate_inputs["vb_e"], candidate_inputs["vbb_e"])
    record_replay = _literal_midpoint(
        reference_coefficients, reference_inputs["vn_e"],
        reference_inputs["vb_e"], reference_inputs["vbb_e"])
    rotation_control = _literal_midpoint(
        reference_coefficients, reference_inputs["vb_e"],
        reference_inputs["vn_e"], reference_inputs["vbb_e"])
    candidate_terms = _literal_terms(
        candidate_coefficients, candidate_inputs["vn_e"],
        candidate_inputs["vb_e"], candidate_inputs["vbb_e"])
    reference_terms = _literal_terms(
        reference_coefficients, reference_inputs["vn_e"],
        reference_inputs["vb_e"], reference_inputs["vbb_e"])
    require(_row(candidate_replay, candidate_target)["bit_exact"],
            "offline candidate replay does not reproduce passive midpoint V")

    operand_rows = {
        name: _row(candidate_inputs[name], reference_inputs[name])
        for name in INPUT_ORDER
    }
    term_names = (
        "za1_vn_e", "za2_vb_e", "za3_vbb_e", "first_two", "va_e",
    )
    term_rows = {
        name: _row(candidate, reference)
        for name, candidate, reference in zip(
            term_names, candidate_terms, reference_terms, strict=True)
    }
    term_rows["passive_va_e"] = _row(candidate_target, target)

    single = {}
    for name in INPUT_ORDER:
        substituted = dict(candidate_inputs)
        substituted[name] = reference_inputs[name]
        coefficients = np.asarray([
            substituted["za1"], substituted["za2"], substituted["za3"],
        ], dtype=np.float64)
        single[name] = _row(_literal_midpoint(
            coefficients, substituted["vn_e"], substituted["vb_e"],
            substituted["vbb_e"]), target)

    cumulative = {}
    substituted = dict(candidate_inputs)
    for name in INPUT_ORDER:
        substituted[name] = reference_inputs[name]
        coefficients = np.asarray([
            substituted["za1"], substituted["za2"], substituted["za3"],
        ], dtype=np.float64)
        cumulative[name] = _row(_literal_midpoint(
            coefficients, substituted["vn_e"], substituted["vb_e"],
            substituted["vbb_e"]), target)

    return {
        "input_order": list(INPUT_ORDER),
        "history_mapping": {
            "vn_e": "j002_va_new",
            "vb_e": "j001_va_new",
            "vbb_e": "i000_vn_e",
        },
        "operand_rows": operand_rows,
        "term_rows": term_rows,
        "record_replay_target": _row(record_replay, target),
        "rotation_control": _row(rotation_control, target),
        "single_substitution_va_e": single,
        "cumulative_substitution_va_e": cumulative,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate the frozen split and exercise its fail-closed controls."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "registry-order":
        report["split"]["input_order"][3:5] = reversed(
            report["split"]["input_order"][3:5])
    elif plant == "coefficient-bit":
        report["split"]["operand_rows"]["za1"]["bit_exact"] = False
        report["split"]["operand_rows"]["za1"]["differing_cells"] = 1
    elif plant == "rotation-map":
        report["split"]["record_replay_target"] = copy.deepcopy(
            report["split"]["rotation_control"])
    elif plant == "missing-stream":
        report["record_fields"]["j003_ext_coef"] = False

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("record_census") == {
        "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
    }, "record census moved")
    require(report.get("record_fields") == {name: True for name in RECORD_FIELDS},
            "required AB3 input stream is absent")
    require(all(report["passivity"].values()), "substep trace is not passive")
    split = report["split"]
    require(split.get("input_order") == list(INPUT_ORDER),
            "AB3 input registry reordered")
    require(split.get("history_mapping") == {
        "vn_e": "j002_va_new",
        "vb_e": "j001_va_new",
        "vbb_e": "i000_vn_e",
    }, "AB3 history rotation mapping moved")
    require(split["record_replay_target"]["bit_exact"],
            "recorded history rotation does not replay midpoint V")
    require(not split["rotation_control"]["bit_exact"],
            "history rotation control is vacuous")
    coefficient_rows = split["operand_rows"]
    require(all(coefficient_rows[name]["bit_exact"]
                for name in ("za1", "za2", "za3")),
            "AB3 coefficient is the first non-bit input")
    require(
        split["term_rows"]["passive_va_e"]["differing_cells"] == 15943,
        "round-195 midpoint V debt moved")

    first_nonbit = next(
        (name for name in INPUT_ORDER
         if not split["operand_rows"][name]["bit_exact"]), None)
    require(first_nonbit is not None,
            "AB3 input split unexpectedly has no non-bit input")
    closing_single = next(
        (name for name in INPUT_ORDER
         if split["single_substitution_va_e"][name]["bit_exact"]), None)
    closing_cumulative = next(
        (name for name in INPUT_ORDER
         if split["cumulative_substitution_va_e"][name]["bit_exact"]), None)
    report["first_nonbit_input"] = first_nonbit
    report["closing_single_substitution"] = closing_single
    report["closing_cumulative_input"] = closing_cumulative
    report["prediction_ledger"] = {
        "R196-P1": "CONFIRMED_RECORD_SUFFICIENT",
        "R196-P2": "CONFIRMED_COEFFICIENTS_EXACT",
        "R196-P3": (
            "CONFIRMED_VN_E_FIRST" if first_nonbit == "vn_e"
            else f"REFUTED_FIRST_{first_nonbit.upper()}"),
        "R196-P4": (
            f"CONFIRMED_SINGLE_{closing_single.upper()}" if closing_single
            else "REFUTED_NO_SINGLE_SUBSTITUTION_CLOSES"),
        "R196-P5": "CONFIRMED_CONTROLS_BIND",
    }
    report["status"] = "PASS_R196_SUBSTEP3_AB3_INPUT_SPLIT"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    """Run the shared passive trace and split the substep-3 AB3 inputs."""

    context = r195.measurement_context(
        deck_root, frame_root, spg_root, expect_commit)
    oracle = context["oracle"]
    record_fields = {name: name in oracle for name in RECORD_FIELDS}
    require(all(record_fields.values()),
            "round-96 record lacks a required AB3 input stream")
    result = {
        "format": "nemo-testcase-l4-orca2-round196-midpoint-v-inputs-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": context["stamp"],
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": record_fields,
        "passivity": context["passivity"],
        "split": split_inputs(context["trace"], oracle),
        "compiled_sources": [
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:486-511",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:817-819",
        ],
    }
    return classify(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit)), "runtime inputs are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r195.GateError, r146.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R196_SUBSTEP3_AB3_INPUT_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
