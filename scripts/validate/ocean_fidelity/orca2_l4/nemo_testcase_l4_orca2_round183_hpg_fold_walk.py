#!/usr/bin/env python3
"""Walk the admitted independent rung-0 HPG fold operands offline."""

# ruff: noqa: E402

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as r42,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as r93,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as r179,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round181_hpg_component_walk as r181,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round180_hpg1_acquisition import (
    check_record as r180,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round181_hpg_fold_acquisition import (
    check_record as fold_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)

FLOOR = np.float64(2.0e-10)
INPUT_ORDER = (
    "north_e3w", "north_rhd", "current_e3w", "current_rhd", "r1_e2v",
)
STATEMENT_ORDER = (
    "surface_north_product",
    "surface_current_product",
    "surface_difference",
    "surface_scaled",
    "interior_north_density_sum",
    "interior_current_density_sum",
    "interior_north_product",
    "interior_current_product",
    "interior_difference",
    "interior_increment",
    "topdown_accumulator",
)
ATOMIC_ORDER = (
    "candidate_all",
    "oracle_raw_hpg",
    "oracle_raw_hpg_e3v",
    "oracle_raw_hpg_e3v_vmask",
    "oracle_all",
)
PLANTS = (
    "none",
    "admission",
    "rank-placement",
    "record-bit",
    "source-order",
    "target-mask",
    "self-replay",
    "first-input",
    "first-statement",
    "north-arm",
    "atomic-endpoint",
    "endpoint-ulp",
)


class GateError(RuntimeError):
    """The admitted record or offline replay no longer closes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first(rows: dict[str, dict[str, object]], order) -> dict[str, object] | None:
    for name in order:
        if not rows[name]["at_floor"]:
            return {"boundary": name, **rows[name]}
    return None


def _replay_zhpj(
    north_e3w: np.ndarray,
    north_rhd: np.ndarray,
    current_e3w: np.ndarray,
    current_rhd: np.ndarray,
    r1_e2v: np.ndarray,
    g: float,
) -> dict[str, np.ndarray]:
    """Replay dynhpg's evaluated V operands in its scalar source order."""

    north_e3w = np.asarray(north_e3w, dtype=np.float64)
    north_rhd = np.asarray(north_rhd, dtype=np.float64)
    current_e3w = np.asarray(current_e3w, dtype=np.float64)
    current_rhd = np.asarray(current_rhd, dtype=np.float64)
    r1_e2v = np.asarray(r1_e2v, dtype=np.float64)
    require(
        north_e3w.shape
        == north_rhd.shape
        == current_e3w.shape
        == current_rhd.shape,
        "HPG operand shapes moved",
    )
    require(r1_e2v.shape == north_e3w.shape[:2], "HPG metric shape moved")

    north_density = np.empty_like(north_rhd)
    current_density = np.empty_like(current_rhd)
    north_density[..., 0] = north_rhd[..., 0]
    current_density[..., 0] = current_rhd[..., 0]
    north_density[..., 1:] = north_rhd[..., 1:] + north_rhd[..., :-1]
    current_density[..., 1:] = current_rhd[..., 1:] + current_rhd[..., :-1]
    north_product = north_e3w * north_density
    current_product = current_e3w * current_density
    difference = north_product - current_product
    zcoef0 = np.float64(-g) * np.float64(0.5)
    scale = zcoef0 * r1_e2v
    increment = scale[..., None] * difference
    accumulator = np.empty_like(increment)
    accumulator[..., 0] = increment[..., 0]
    for level in range(1, increment.shape[-1]):
        accumulator[..., level] = (
            accumulator[..., level - 1] + increment[..., level]
        )
    return {
        "north_density_sum": north_density,
        "current_density_sum": current_density,
        "north_product": north_product,
        "current_product": current_product,
        "difference": difference,
        "increment": increment,
        "accumulator": accumulator,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["status"] = "FAIL"
    elif plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "record-bit":
        report["record_control"]["differing_cells"] = 0
    elif plant == "source-order":
        report["statement_order"][0], report["statement_order"][1] = (
            report["statement_order"][1],
            report["statement_order"][0],
        )
    elif plant == "target-mask":
        report["target"]["cells"] -= 1
    elif plant == "self-replay":
        report["replay_closure"]["oracle_to_recorded_zhpj"]["at_floor"] = False
    elif plant == "first-input":
        first = report["first_input"]["boundary"]
        report["input_rows"][first]["at_floor"] = True
    elif plant == "first-statement":
        first = report["first_statement"]["boundary"]
        report["statement_rows"][first]["at_floor"] = True
    elif plant == "north-arm":
        report["north_only_replay"]["at_floor"] = False
    elif plant == "atomic-endpoint":
        report["atomic_rows"]["oracle_all"]["bit_exact"] = False
    elif plant == "endpoint-ulp":
        report["endpoint_ulp_control"]["bit_exact"] = True

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(report["admission"]["status"] == "PASS_R181_HPG_FOLD_ADMISSION",
            "fold record is not admitted")
    require(report["admission"]["rank_coverage"] == "exactly-once",
            "rank placement moved")
    require(report["admission"]["restart_identities"] == 20,
            "restart identity census moved")
    require(report["record_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "record one-ULP control did not fire")
    require(tuple(report["input_order"]) == INPUT_ORDER,
            "primitive input source order moved")
    require(tuple(report["statement_order"]) == STATEMENT_ORDER,
            "compiled statement source order moved")
    require(tuple(report["atomic_order"]) == ATOMIC_ORDER,
            "four-operand atomic order moved")
    require(report["target"] == {
        "cells": 68, "row": 147, "wet_levels": 1319},
        f"registered target moved: {report['target']}")
    require(all(row["at_floor"] for row in report["replay_closure"].values()),
            "exact-operand replay does not calibrate")
    first_input = _first(report["input_rows"], INPUT_ORDER)
    require(first_input == report["first_input"], "first input selector moved")
    first_statement = _first(report["statement_rows"], STATEMENT_ORDER)
    require(first_statement == report["first_statement"],
            "first statement selector moved")
    require(first_input is not None and first_statement is not None,
            "fold HPG debt disappeared")
    require(report["north_only_replay"]["at_floor"],
            "recorded north operands do not close zhpj")
    require(not report["north_only_arm_vacuous"],
            "north-only arm is vacuous")
    require(report["atomic_rows"]["oracle_all"]["bit_exact"],
            "oracle four-operand endpoint is not its own identity")
    require(report["atomic_rows"]["oracle_all"]["at_floor"],
            "oracle four-operand endpoint left the floor")
    require(report["endpoint_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "endpoint one-ULP control did not fire")

    report["prediction_ledger"] = {
        "R183-P1": "CONFIRMED",
        "R183-P2": "CONFIRMED",
        "R183-P3": (
            "CONFIRMED"
            if first_input["boundary"] == "north_e3w"
            else "REFUTED"
        ),
        "R183-P4": (
            "CONFIRMED"
            if first_statement["boundary"] == "surface_north_product"
            and report["north_only_replay"]["at_floor"]
            else "REFUTED"
        ),
        "R183-P5": "CONFIRMED",
        "R183-P6": "CONFIRMED",
    }
    report["status"] = "HELD_FIRST_HPG_FOLD_STATEMENT"
    return report


def measure(
    deck_root: Path,
    fold_root: Path,
    component_root: Path,
    baseline_root: Path,
    rhs_root: Path,
    static_root: Path,
    prior_json: Path,
    expect_commit: str,
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-183 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-183 replay requires production JIT on CPU")

    admission = fold_record.run(fold_root, baseline_root, "none")
    fold_records = [
        fold_record.read_record(
            fold_root / f"oracle_r181_hpgfold_rank{rank:04d}_kt00000001.bin")
        for rank in (0, 1)
    ]
    component_admission = r180.run(component_root, baseline_root)
    require(component_admission["status"] == "PASS_R180_HPG1_ADMISSION",
            "round-180 component record is not admitted")
    component_records = [
        r180.read_record(
            component_root / f"oracle_r180_hpg1_rank{rank:04d}_kt00000001.bin")
        for rank in (0, 1)
    ]
    oracle_rhs, rhs_census = r93.assemble_rhs(rhs_root)
    oracle_static, static_census = r170.assemble_record(static_root)

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    model.prime_step_caches(state)
    candidate_inputs = r42._candidate_inputs(model, state)
    candidate_literal = r42._literal_from_inputs(
        candidate_inputs, card.recipe.model_config.g, grid=card.recipe.grid)

    fold = {
        name: r181._assemble(fold_records, name)
        for name in fold_record.NAMES
    }
    oracle_metric = r181._assemble(component_records, "r1_e2v")[..., 0]
    oracle_zvap = r181._assemble(component_records, "zuap_v")
    oracle_component_zhpj = r181._assemble(component_records, "zhpi_v")
    candidate_metric = np.asarray(candidate_inputs["r1_e2v"], dtype=np.float64)
    candidate_operands = {
        "north_e3w": np.roll(np.asarray(candidate_inputs["e3w"]), -1, axis=0),
        "north_rhd": np.roll(np.asarray(candidate_inputs["rhd"]), -1, axis=0),
        "current_e3w": np.asarray(candidate_inputs["e3w"], dtype=np.float64),
        "current_rhd": np.asarray(candidate_inputs["rhd"], dtype=np.float64),
    }
    oracle_operands = {
        name: fold[name]
        for name in ("north_e3w", "north_rhd", "current_e3w", "current_rhd")
    }

    candidate_static = r170._candidate_reference_operands(card, state)
    oracle_vmask = np.asarray(oracle_static["vmask"])[..., :30]
    candidate_vmask = np.asarray(candidate_static["mask_v"])[..., :30]
    target = np.any(
        (oracle_vmask != 0.0) & (candidate_vmask == 0.0), axis=-1)
    target_locations = np.argwhere(target)
    target_row = int(target_locations[0, 0]) if target_locations.size else -1
    contributing = np.broadcast_to(target[..., None], oracle_vmask.shape) & (
        oracle_vmask != 0.0)
    target_summary = {
        "cells": int(np.count_nonzero(target)),
        "row": target_row,
        "wet_levels": int(np.count_nonzero(contributing)),
    }
    full_contributing = np.concatenate(
        (contributing, np.zeros_like(contributing[..., :1])), axis=-1)

    oracle_replay = _replay_zhpj(
        **oracle_operands, r1_e2v=oracle_metric,
        g=card.recipe.model_config.g)
    candidate_replay = _replay_zhpj(
        **candidate_operands, r1_e2v=candidate_metric,
        g=card.recipe.model_config.g)
    candidate_zhpj = r83.native_v(candidate_literal["zhpi_v"])
    replay_closure = {
        "fold_to_component_zhpj": r181._score(
            fold["zhpj"], oracle_component_zhpj, full_contributing),
        "oracle_to_recorded_zhpj": r181._score(
            oracle_replay["accumulator"], fold["zhpj"], full_contributing),
        "candidate_to_literal_zhpj": r181._score(
            candidate_replay["accumulator"], candidate_zhpj,
            contributing),
    }

    input_rows = {
        name: r181._score(
            candidate_operands[name], oracle_operands[name],
            full_contributing)
        for name in INPUT_ORDER[:-1]
    }
    input_rows["r1_e2v"] = r181._score(
        candidate_metric, oracle_metric, target)
    first_input = _first(input_rows, INPUT_ORDER)

    surface = contributing[..., 0]
    interior = np.array(contributing, copy=True)
    interior[..., 0] = False
    statement_rows = {
        "surface_north_product": r181._score(
            candidate_replay["north_product"][..., 0],
            oracle_replay["north_product"][..., 0], surface),
        "surface_current_product": r181._score(
            candidate_replay["current_product"][..., 0],
            oracle_replay["current_product"][..., 0], surface),
        "surface_difference": r181._score(
            candidate_replay["difference"][..., 0],
            oracle_replay["difference"][..., 0], surface),
        "surface_scaled": r181._score(
            candidate_replay["increment"][..., 0],
            oracle_replay["increment"][..., 0], surface),
        "interior_north_density_sum": r181._score(
            candidate_replay["north_density_sum"],
            oracle_replay["north_density_sum"], interior),
        "interior_current_density_sum": r181._score(
            candidate_replay["current_density_sum"],
            oracle_replay["current_density_sum"], interior),
        "interior_north_product": r181._score(
            candidate_replay["north_product"],
            oracle_replay["north_product"], interior),
        "interior_current_product": r181._score(
            candidate_replay["current_product"],
            oracle_replay["current_product"], interior),
        "interior_difference": r181._score(
            candidate_replay["difference"],
            oracle_replay["difference"], interior),
        "interior_increment": r181._score(
            candidate_replay["increment"],
            oracle_replay["increment"], interior),
        "topdown_accumulator": r181._score(
            candidate_replay["accumulator"],
            oracle_replay["accumulator"], contributing),
    }
    first_statement = _first(statement_rows, STATEMENT_ORDER)

    north_mixed = {
        name: np.array(value, copy=True)
        for name, value in candidate_operands.items()
    }
    for name in ("north_e3w", "north_rhd"):
        north_mixed[name][target, :] = oracle_operands[name][target, :]
    north_replay = _replay_zhpj(
        **north_mixed, r1_e2v=candidate_metric,
        g=card.recipe.model_config.g)
    north_only_replay = r181._score(
        north_replay["accumulator"], fold["zhpj"][..., :30], contributing)
    north_arm_delta = r181._score(
        north_replay["accumulator"],
        candidate_replay["accumulator"], contributing)

    oracle_raw = fold["zhpj"] + oracle_zvap
    oracle_rhs_hpg = r179._with_jpk_zero(oracle_rhs["after_hpg_v"])
    candidate_raw = r179._with_jpk_zero(
        r83.native_v(candidate_literal["sum_v"]))
    north_raw = np.array(candidate_raw, copy=True)
    north_raw[target, :30] = (
        north_replay["accumulator"][target, :]
        + oracle_zvap[target, :30]
    )
    oracle_e3v = np.asarray(oracle_static["e3v"])
    candidate_e3v = np.asarray(candidate_static["e3_v"])
    oracle_mask = np.asarray(oracle_static["vmask"])
    candidate_mask = np.asarray(candidate_static["mask_v"])
    oracle_r1 = np.asarray(oracle_static["r1_hv0"])
    candidate_r1 = np.asarray(candidate_static["r1_h0_v"])
    oracle_depth = r179._depth(
        oracle_rhs_hpg, oracle_e3v, oracle_mask, oracle_r1)
    atomic_values = {
        "candidate_all": r179._depth(
            candidate_raw, candidate_e3v, candidate_mask, candidate_r1),
        "oracle_raw_hpg": r179._depth(
            oracle_rhs_hpg, candidate_e3v, candidate_mask, candidate_r1),
        "oracle_raw_hpg_e3v": r179._depth(
            oracle_rhs_hpg, oracle_e3v, candidate_mask, candidate_r1),
        "oracle_raw_hpg_e3v_vmask": r179._depth(
            oracle_rhs_hpg, oracle_e3v, oracle_mask, candidate_r1),
        "oracle_all": oracle_depth,
    }
    atomic_rows = {
        name: r181._score(value, oracle_depth, target)
        for name, value in atomic_values.items()
    }
    north_fixed_depth = r179._depth(
        north_raw, candidate_e3v, candidate_mask, candidate_r1)

    prior = json.loads(prior_json.read_text())
    require(prior["candidate_calibration"]["reproduces_round179"],
            "round-181 candidate calibration moved")
    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    report = {
        "format": "nemo-testcase-l4-orca2-round183-hpg-fold-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR),
        "worktree": stamp,
        "admission": admission,
        "component_admission_status": component_admission["status"],
        "rhs_census": rhs_census,
        "static_census": static_census,
        "record_control": {
            "bit_exact": r181._bits_equal(one, next_one),
            "differing_cells": 1,
        },
        "input_order": list(INPUT_ORDER),
        "statement_order": list(STATEMENT_ORDER),
        "atomic_order": list(ATOMIC_ORDER),
        "target": target_summary,
        "replay_closure": replay_closure,
        "input_rows": input_rows,
        "first_input": first_input,
        "statement_rows": statement_rows,
        "first_statement": first_statement,
        "north_only_replay": north_only_replay,
        "north_only_arm_delta": north_arm_delta,
        "north_only_arm_vacuous": bool(north_arm_delta["bit_exact"]),
        "fold_component_to_rhs": r181._score(
            oracle_raw, oracle_rhs_hpg, full_contributing),
        "atomic_operand_rows": {
            "raw_hpg": r181._score(
                candidate_raw, oracle_rhs_hpg, full_contributing),
            "e3v": r181._score(
                candidate_e3v, oracle_e3v, full_contributing),
            "vmask": r181._score(
                candidate_mask, oracle_mask,
                np.broadcast_to(target[..., None], oracle_mask.shape)),
            "r1_hv0": r181._score(candidate_r1, oracle_r1, target),
        },
        "atomic_rows": atomic_rows,
        "north_fixed_atomic_row": r181._score(
            north_fixed_depth, oracle_depth, target),
        "prior_round181_component_rows": prior["component_rows"],
        "endpoint_ulp_control": {
            "bit_exact": r181._bits_equal(one, next_one),
            "differing_cells": 1,
        },
        "compiled_source": {
            "surface": (
                "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/"
                "dynhpg.f90:409-416"
            ),
            "interior": (
                "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/"
                "dynhpg.f90:445-453"
            ),
            "depth_average": (
                "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215"
            ),
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--fold-root", type=Path)
    parser.add_argument("--component-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--rhs-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--prior-json", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((
                args.deck_root,
                args.fold_root,
                args.component_root,
                args.baseline_root,
                args.rhs_root,
                args.static_root,
                args.prior_json,
                args.expect_commit,
            )), "measurement arguments missing")
            result = measure(
                args.deck_root,
                args.fold_root,
                args.component_root,
                args.baseline_root,
                args.rhs_root,
                args.static_root,
                args.prior_json,
                args.expect_commit,
            )
        else:
            require(args.report_in is not None,
                    "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (
        OSError,
        ValueError,
        KeyError,
        GateError,
        fold_record.Refusal,
        r180.Refusal,
        r93.GateError,
        r170.GateError,
        rung0.GateError,
    ) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
