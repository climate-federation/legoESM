#!/usr/bin/env python3
"""Split rung-0 substep-2 vector-form V update across recorded inputs."""

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

SUBSTEP = 2
INPUT_ORDER = ("vn_e", "rDt_e", "zv_spg", "zv_trd", "zv_frc", "ssvmask")
RECORD_FIELDS = (
    "j001_va_new", "j002_zv_spg", "j002_trd_v", "i000_zv_frc",
    "i000_entry_sc", "j002_va_new",
)
PLANTS = (
    "none", "registry-order", "timestep-bit", "association",
    "missing-stream",
)


class GateError(RuntimeError):
    """The record, source order, replay, or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _row(candidate, reference) -> dict[str, object]:
    return r146.exact_row(np.asarray(candidate), np.asarray(reference))


def _literal_terms(inputs: dict[str, object]) -> dict[str, np.ndarray]:
    """Materialise dynspg_ts.f90:723-727 in its written association."""

    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    vn_e = jnp.asarray(inputs["vn_e"], dtype=jnp.float64)
    rdt = jnp.asarray(inputs["rDt_e"], dtype=jnp.float64)
    spg = jnp.asarray(inputs["zv_spg"], dtype=jnp.float64)
    trd = jnp.asarray(inputs["zv_trd"], dtype=jnp.float64)
    frc = jnp.asarray(inputs["zv_frc"], dtype=jnp.float64)
    mask = jnp.asarray(inputs["ssvmask"], dtype=jnp.float64)
    spg_trd = b(b(spg) + b(trd))
    rhs = b(spg_trd + b(frc))
    increment = b(rdt * rhs)
    before_mask = b(b(vn_e) + increment)
    raw = b(before_mask * mask)
    values = jax.device_get((spg_trd, rhs, increment, before_mask, raw))
    return {
        name: np.asarray(value) for name, value in zip(
            ("spg_plus_trd", "rhs", "increment", "before_mask", "raw_va_e"),
            values, strict=True)
    }


def _associate_v(native_v, card) -> np.ndarray:
    """Apply the shared seven-array association helper to one native V field."""

    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    state = card.recipe.initial_state
    v = jnp.asarray(r178._to_model_v(native_v), dtype=jnp.float64)
    u = jnp.zeros_like(jnp.asarray(state.u_mask.data, dtype=jnp.float64))
    eta = jnp.zeros_like(jnp.asarray(state.eta.data, dtype=jnp.float64))
    post = _nemo_external_mode_boundary_association(
        u, v, u, v, u, v, eta, card.recipe.grid,
    )[1]
    return r178._native_v(np.asarray(jax.device_get(post)))


def split_update(context: dict[str, object]) -> dict[str, object]:
    """Replay substep 2 from passive operands and split one variable at a time."""

    card = context["card"]
    state = context["state"]
    trace = context["trace"]
    oracle = context["oracle"]
    index = SUBSTEP - 1
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no raw NEMO mask bundle")

    candidate_inputs = {
        "vn_e": r178._native_v(trace["v_entry"][index]),
        "rDt_e": np.float64(card.dt_s / int(oracle["i000_entry_sc"][2])),
        "zv_spg": r178._native_v(trace["pgf_v"][index]),
        "zv_trd": r178._native_v(trace["trd_v"][index]),
        "zv_frc": r178._native_v(trace["slow_v"][index]),
        "ssvmask": r178._native_v(np.asarray(state.v_mask.data)),
    }
    reference_inputs = {
        "vn_e": np.asarray(oracle["j001_va_new"], dtype=np.float64),
        "rDt_e": np.float64(oracle["i000_entry_sc"][0]),
        "zv_spg": np.asarray(oracle["j002_zv_spg"], dtype=np.float64),
        "zv_trd": np.asarray(oracle["j002_trd_v"], dtype=np.float64),
        "zv_frc": np.asarray(oracle["i000_zv_frc"], dtype=np.float64),
        "ssvmask": np.max(np.asarray(raw.vmask, dtype=np.float64), axis=-1),
    }
    target = np.asarray(oracle["j002_va_new"], dtype=np.float64)
    require(all(np.asarray(candidate_inputs[name]).shape == target.shape
                for name in INPUT_ORDER if name != "rDt_e"),
            "candidate vector-update input shape moved")
    require(all(np.asarray(reference_inputs[name]).shape == target.shape
                for name in INPUT_ORDER if name != "rDt_e"),
            "recorded vector-update input shape moved")

    candidate_terms = _literal_terms(candidate_inputs)
    reference_terms = _literal_terms(reference_inputs)
    candidate_post = _associate_v(candidate_terms["raw_va_e"], card)
    reference_post = _associate_v(reference_terms["raw_va_e"], card)
    passive_post = r178._native_v(trace["v_exit"][index])

    operand_rows = {
        name: _row(candidate_inputs[name], reference_inputs[name])
        for name in INPUT_ORDER
    }
    term_rows = {
        name: _row(candidate_terms[name], reference_terms[name])
        for name in candidate_terms
    }
    term_rows.update({
        "candidate_replay_vs_passive": _row(candidate_post, passive_post),
        "record_replay_vs_target": _row(reference_post, target),
        "raw_record_vs_post_target": _row(reference_terms["raw_va_e"], target),
        "passive_post_vs_target": _row(passive_post, target),
    })

    single = {}
    cumulative = {}
    accumulated = dict(candidate_inputs)
    for name in INPUT_ORDER:
        substituted = dict(candidate_inputs)
        substituted[name] = reference_inputs[name]
        single_terms = _literal_terms(substituted)
        single[name] = _row(_associate_v(single_terms["raw_va_e"], card), target)
        accumulated[name] = reference_inputs[name]
        cumulative_terms = _literal_terms(accumulated)
        cumulative[name] = _row(
            _associate_v(cumulative_terms["raw_va_e"], card), target)

    return {
        "input_order": list(INPUT_ORDER),
        "record_mapping": {
            "vn_e": "j001_va_new", "rDt_e": "i000_entry_sc[0]",
            "zv_spg": "j002_zv_spg", "zv_trd": "j002_trd_v",
            "zv_frc": "i000_zv_frc", "ssvmask": "raw vmask maximum",
            "target": "j002_va_new",
        },
        "operand_rows": operand_rows,
        "term_rows": term_rows,
        "single_substitution_post_v": single,
        "cumulative_substitution_post_v": cumulative,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate the frozen split and exercise fail-closed controls."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "registry-order":
        report["split"]["input_order"][2:4] = reversed(
            report["split"]["input_order"][2:4])
    elif plant == "timestep-bit":
        report["split"]["operand_rows"]["rDt_e"]["bit_exact"] = False
        report["split"]["operand_rows"]["rDt_e"]["differing_cells"] = 1
    elif plant == "association":
        report["split"]["term_rows"]["record_replay_vs_target"] = copy.deepcopy(
            report["split"]["term_rows"]["raw_record_vs_post_target"])
    elif plant == "missing-stream":
        report["record_fields"]["j002_zv_spg"] = False

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("record_census") == {
        "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
    }, "record census moved")
    require(report.get("record_fields") == {name: True for name in RECORD_FIELDS},
            "required vector-update stream is absent")
    require(all(report["passivity"].values()), "substep trace is not passive")
    split = report["split"]
    require(split.get("input_order") == list(INPUT_ORDER),
            "vector-update input registry reordered")
    require(split["record_mapping"] == {
        "vn_e": "j001_va_new", "rDt_e": "i000_entry_sc[0]",
        "zv_spg": "j002_zv_spg", "zv_trd": "j002_trd_v",
        "zv_frc": "i000_zv_frc", "ssvmask": "raw vmask maximum",
        "target": "j002_va_new",
    }, "record mapping moved")
    rows = split["term_rows"]
    require(rows["candidate_replay_vs_passive"]["bit_exact"],
            "offline candidate replay does not reproduce passive V")
    require(rows["record_replay_vs_target"]["bit_exact"],
            "recorded update plus association does not reproduce target")
    require(not rows["raw_record_vs_post_target"]["bit_exact"],
            "boundary-association control is vacuous")
    require(rows["passive_post_vs_target"]["differing_cells"] == 16506,
            "round-196 current-V debt moved")
    operands = split["operand_rows"]
    require(operands["rDt_e"]["bit_exact"],
            "barotropic timestep is the first non-bit operand")

    first_nonbit = next(
        (name for name in INPUT_ORDER if not operands[name]["bit_exact"]), None)
    require(first_nonbit is not None,
            "vector-update split unexpectedly has no non-bit operand")
    closing_single = next(
        (name for name in INPUT_ORDER
         if split["single_substitution_post_v"][name]["bit_exact"]), None)
    closing_cumulative = next(
        (name for name in INPUT_ORDER
         if split["cumulative_substitution_post_v"][name]["bit_exact"]), None)
    report["first_nonbit_operand"] = first_nonbit
    report["closing_single_substitution"] = closing_single
    report["closing_cumulative_operand"] = closing_cumulative
    report["prediction_ledger"] = {
        "R197-P1": "CONFIRMED_RECORD_SUFFICIENT",
        "R197-P2": (
            "CONFIRMED_VN_E_EXACT" if operands["vn_e"]["bit_exact"]
            else "REFUTED_VN_E_NONBIT"),
        "R197-P3": (
            "CONFIRMED_ZV_SPG_FIRST" if first_nonbit == "zv_spg"
            else f"REFUTED_FIRST_{first_nonbit.upper()}"),
        "R197-P4": (
            f"CONFIRMED_SINGLE_{closing_single.upper()}" if closing_single
            else "REFUTED_NO_SINGLE_SUBSTITUTION_CLOSES"),
        "R197-P5": "CONFIRMED_ASSOCIATION_ADDS_NO_DEBT",
        "R197-P6": "CONFIRMED_CONTROLS_BIND",
    }
    report["status"] = "PASS_R197_SUBSTEP2_VECTOR_V_UPDATE_SPLIT"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    """Run the shared passive context and split only the substep-2 V update."""

    context = r195.measurement_context(
        deck_root, frame_root, spg_root, expect_commit)
    oracle = context["oracle"]
    record_fields = {name: name in oracle for name in RECORD_FIELDS}
    require(all(record_fields.values()),
            "round-96 record lacks a required vector-update stream")
    result = {
        "format": "nemo-testcase-l4-orca2-round197-vector-v-update-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": context["stamp"],
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": record_fields,
        "passivity": context["passivity"],
        "split": split_update(context),
        "compiled_sources": [
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779",
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
    print("STATUS PASS_R197_SUBSTEP2_VECTOR_V_UPDATE_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
