#!/usr/bin/env python3
"""Walk independent rung-0 HPG V components from the admitted R180 record."""

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
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as r179,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round180_hpg1_acquisition import (
    check_record as r180,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)

FLOOR = np.float64(2.0e-10)
COMPONENT_ORDER = ("zhpj", "zvap", "sum_v")
INPUT_ORDER = (
    "north_e3w", "north_rhd", "current_e3w", "current_rhd", "r1_e2v",
)
PLANTS = (
    "none", "rank-placement", "record-bit", "source-order", "target-mask",
    "self-replay", "first-boundary", "endpoint-ulp",
)


class GateError(RuntimeError):
    """The admitted record or the offline replay no longer closes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits_equal(left, right) -> bool:
    return bool(np.array_equal(
        np.ascontiguousarray(left).view(np.uint64),
        np.ascontiguousarray(right).view(np.uint64),
    ))


def _score(candidate, oracle, active) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape,
            f"score shape mismatch {candidate.shape} {oracle.shape} {active.shape}")
    require(bool(np.any(active)), "score mask is empty")
    c, o = candidate[active], oracle[active]
    require(bool(np.isfinite(c).all() and np.isfinite(o).all()),
            "score contains non-finite values")
    delta = np.abs(c - o)
    unequal = (
        np.ascontiguousarray(c).view(np.uint64)
        != np.ascontiguousarray(o).view(np.uint64)
    )
    flat_active = np.flatnonzero(active)
    position = int(np.argmax(delta))
    argmax = np.unravel_index(int(flat_active[position]), candidate.shape)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "compared_cells": int(c.size),
        "absolute_max": float(delta[position]),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "at_floor": bool(float(delta[position]) <= FLOOR),
        "argmax": [int(value) for value in argmax],
        "candidate_at_argmax": float(candidate[argmax]),
        "oracle_at_argmax": float(oracle[argmax]),
    }


def _first(rows: dict[str, dict[str, object]], order) -> dict[str, object] | None:
    for name in order:
        if not rows[name]["at_floor"]:
            return {"boundary": name, **rows[name]}
    return None


def _placement(record: dict[str, object]) -> tuple[slice, slice]:
    nimpp, njmpp = record["origin"]
    ntsi, ntsj, ntei, ntej = record["owned"]
    i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
    i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
    require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
            "round-180 owned slab moved")
    return slice(j0, j1), slice(i0, i1)


def _owned(record: dict[str, object], name: str) -> np.ndarray:
    ntsi, ntsj, ntei, ntej = record["owned"]
    local = np.asarray(record["fields"][name], dtype=np.float64)
    value = local[ntsi - 1:ntei, ntsj - 1:ntej, :]
    return value.transpose(1, 0, 2)


def _assemble(records: list[dict[str, object]], name: str) -> np.ndarray:
    depth = records[0]["fields"][name].shape[-1]
    result = np.empty((148, 180, depth), dtype=np.float64)
    coverage = np.zeros((148, 180), dtype=np.int8)
    for record in records:
        js, is_ = _placement(record)
        result[js, is_, :] = _owned(record, name)
        coverage[js, is_] += 1
    require(bool(np.all(coverage == 1)), f"{name} rank placement is not exact")
    return result


def _north_halo(records: list[dict[str, object]], name: str) -> np.ndarray:
    """Return NEMO's explicit jj+1 halo above the final owned row."""

    blocks = []
    for record in sorted(records, key=lambda row: row["origin"][0]):
        ntsi, _ntsj, ntei, ntej = record["owned"]
        local = np.asarray(record["fields"][name], dtype=np.float64)
        blocks.append(local[ntsi - 1:ntei, ntej, :])
    result = np.concatenate(blocks, axis=0)
    require(result.shape[0] == 180, f"{name} north-halo width moved")
    return result


def _local_self_replay(record: dict[str, object], g: float) -> dict[str, dict]:
    inputs = {
        name: np.asarray(record["fields"][name], dtype=np.float64).transpose(1, 0, 2)
        for name in ("rhd", "e3w", "gdept_z0")
    }
    for name in ("r1_e1u", "r1_e2v"):
        inputs[name] = np.asarray(
            record["fields"][name][..., 0], dtype=np.float64).T
    replay = r42._literal_from_inputs(inputs, g)
    ntsi, ntsj, ntei, ntej = record["owned"]
    shape = (ntej - ntsj + 1, ntei - ntsi + 1, 31)
    full = np.ones(shape, dtype=bool)
    interior = np.ones(shape, dtype=bool)
    # NEMO's jpk slot is structural and hpg_sco loops only through jpkm1.
    full[..., -1] = False
    interior[..., -1] = False
    interior[-1, ...] = False
    rows = {"interior": {}, "full": {}}
    for public, stored in (("zhpj", "zhpi_v"), ("zvap", "zuap_v"),
                           ("sum_v", "sum_v")):
        candidate = r83.native_v(replay[stored])[
            ntsj - 1:ntej, ntsi - 1:ntei, :]
        oracle = _owned(record, stored)
        rows["interior"][public] = _score(candidate, oracle, interior)
        rows["full"][public] = _score(candidate, oracle, full)
    return rows


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["record_census"]["rank_coverage"] = "overlap"
    elif plant == "record-bit":
        report["record_control"]["differing_cells"] = 0
    elif plant == "source-order":
        report["component_order"][0], report["component_order"][1] = (
            report["component_order"][1], report["component_order"][0])
    elif plant == "target-mask":
        report["target"]["cells"] -= 1
    elif plant == "self-replay":
        report["recorded_input_self_replay"]["rank0"]["interior"]["zhpj"][
            "at_floor"] = False
    elif plant == "first-boundary":
        first = report["first_component"]["boundary"]
        report["component_rows"][first]["at_floor"] = True
    elif plant == "endpoint-ulp":
        report["endpoint_ulp_control"]["bit_exact"] = True

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(report["record_census"]["rank_coverage"] == "exactly-once",
            "rank placement moved")
    require(report["record_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "record one-ULP control did not fire")
    require(tuple(report["component_order"]) == COMPONENT_ORDER,
            "compiled component source order moved")
    require(tuple(report["input_order"]) == INPUT_ORDER,
            "compiled input source order moved")
    require(report["target"] == {
        "cells": 68, "row": 147, "wet_levels": 1319},
        f"registered target moved: {report['target']}")
    require(all(
        row["at_floor"]
        for rank in report["recorded_input_self_replay"].values()
        for row in rank["interior"].values()
    ), "recorded-input literal self-replay does not calibrate off the fold")
    require(report["record_component_identity"]["at_floor"],
            "recorded sum_v is not recorded zhpj + zvap")
    require(report["candidate_calibration"]["reproduces_round179"],
            "candidate replay does not reproduce round 179")
    first_component = _first(report["component_rows"], COMPONENT_ORDER)
    require(first_component == report["first_component"],
            "first component selector moved")
    require(first_component is not None, "all HPG V components are at the floor")
    require(report["operand_record_status"] ==
            "MISSING_EXECUTED_NORTH_E3W_EXPRESSION",
            "missing-operand disposition moved")
    require(report["endpoint_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "endpoint one-ULP control did not fire")
    report["prediction_ledger"] = {
        "R181-P1": "CONFIRMED",
        "R181-P2": "REFUTED",
        "R181-P3": (
            "CONFIRMED" if first_component["boundary"] == "zhpj" else "REFUTED"),
        "R181-P4": "UNMEASURED-with-spec",
        "R181-P5": "CONFIRMED",
    }
    report["status"] = "STOPPED_FOR_RECORD_FIRST_HPG_V_COMPONENT"
    return report


def measure(deck_root: Path, component_root: Path, baseline_root: Path,
            static_root: Path, prior_json: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-181 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-181 replay requires production JIT on CPU")

    admission = r180.run(component_root, baseline_root)
    require(admission["status"] == "PASS_R180_HPG1_ADMISSION"
            and admission["claim_label"] == "independent hierarchy rung 0",
            "round-180 component record is not admitted independently")
    records = [r180.read_record(
        component_root / f"oracle_r180_hpg1_rank{rank:04d}_kt00000001.bin")
        for rank in (0, 1)]

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    model.prime_step_caches(state)
    candidate_inputs = r42._candidate_inputs(model, state)
    candidate_literal = r42._literal_from_inputs(
        candidate_inputs, card.recipe.model_config.g, grid=card.recipe.grid)

    oracle_components = {
        public: _assemble(records, stored)
        for public, stored in (("zhpj", "zhpi_v"), ("zvap", "zuap_v"),
                               ("sum_v", "sum_v"))
    }
    candidate_components = {
        public: r83.native_v(candidate_literal[stored])
        for public, stored in (("zhpj", "zhpi_v"), ("zvap", "zuap_v"),
                               ("sum_v", "sum_v"))
    }

    oracle_static, static_census = r170.assemble_record(static_root)
    candidate_static = r170._candidate_reference_operands(card, state)
    oracle_vmask = np.asarray(oracle_static["vmask"])[..., :30]
    candidate_vmask = np.asarray(candidate_static["mask_v"])[..., :30]
    target = np.any((oracle_vmask != 0.0) & (candidate_vmask == 0.0), axis=-1)
    target_locations = np.argwhere(target)
    target_row = int(target_locations[0, 0]) if target_locations.size else -1
    contributing = np.broadcast_to(target[..., None], oracle_vmask.shape) & (
        oracle_vmask != 0.0)
    target_summary = {
        "cells": int(np.count_nonzero(target)),
        "row": target_row,
        "wet_levels": int(np.count_nonzero(contributing)),
    }

    component_rows = {
        name: _score(candidate_components[name], oracle_components[name][..., :30],
                     contributing)
        for name in COMPONENT_ORDER
    }
    first_component = _first(component_rows, COMPONENT_ORDER)
    record_component_identity = _score(
        oracle_components["zhpj"][..., :30]
        + oracle_components["zvap"][..., :30],
        oracle_components["sum_v"][..., :30], contributing)

    oracle_inputs = {
        name: _assemble(records, name)
        for name in ("rhd", "e3w", "gdept_z0")
    }
    oracle_metric = _assemble(records, "r1_e2v")[..., 0]
    oracle_north = {
        name: _north_halo(records, name)[..., :30]
        for name in ("rhd", "e3w", "gdept_z0")
    }
    level_mask = contributing[147]
    input_rows = {
        "north_e3w": _score(
            candidate_inputs["e3w"][0], oracle_north["e3w"], level_mask),
        "north_rhd": _score(
            candidate_inputs["rhd"][0], oracle_north["rhd"], level_mask),
        "current_e3w": _score(
            candidate_inputs["e3w"][147], oracle_inputs["e3w"][147, :, :30],
            level_mask),
        "current_rhd": _score(
            candidate_inputs["rhd"][147], oracle_inputs["rhd"][147, :, :30],
            level_mask),
        "r1_e2v": _score(
            candidate_inputs["r1_e2v"][147], oracle_metric[147], target[147]),
    }
    first_input = _first(input_rows, INPUT_ORDER)

    mixed_inputs = {name: np.array(value, copy=True)
                    for name, value in candidate_inputs.items()}
    for name in ("rhd", "e3w", "gdept_z0"):
        mixed_inputs[name][0, target[147], :] = oracle_north[name][target[147], :]
    mixed_literal = r42._literal_from_inputs(
        mixed_inputs, card.recipe.model_config.g, grid=card.recipe.grid)
    mixed_zhpj = r83.native_v(mixed_literal["zhpi_v"])
    north_only_replay = _score(
        mixed_zhpj, oracle_components["zhpj"][..., :30], contributing)

    self_replay = {
        f"rank{record['rank']}": _local_self_replay(
            record, card.recipe.model_config.g)
        for record in records
    }
    prior = json.loads(prior_json.read_text())
    prior_raw = prior["operand_rows"]["raw_hpg_rhs"]
    current_raw = component_rows["sum_v"]
    candidate_calibration = {
        "reproduces_round179": all((
            current_raw["differing_cells"] == prior_raw["differing_cells"],
            current_raw["compared_cells"] == prior_raw["compared_cells"],
            current_raw["absolute_max"] == prior_raw["absolute_max"],
            current_raw["rms"] == prior_raw["rms"],
        )),
        "round179": prior_raw,
        "round181": current_raw,
    }
    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    report = {
        "format": "nemo-testcase-l4-orca2-round181-hpg-v-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR), "worktree": stamp,
        "record_census": {
            "rank_coverage": admission["rank_coverage"],
            "records": [{key: row[key] for key in (
                "rank", "sha256", "bytes", "origin", "owned")}
                for row in admission["records"]],
            "static": static_census,
            "inherited_rhs": len(admission["inherited_rhs_comparisons"]),
            "restart_identities": len(admission["terminal_restart_comparisons"]),
        },
        "record_control": {"bit_exact": _bits_equal(one, next_one),
                           "differing_cells": 1},
        "component_order": list(COMPONENT_ORDER),
        "input_order": list(INPUT_ORDER),
        "target": target_summary,
        "recorded_input_self_replay": self_replay,
        "candidate_calibration": candidate_calibration,
        "component_rows": component_rows,
        "first_component": first_component,
        "record_component_identity": record_component_identity,
        "input_rows": input_rows,
        "first_input": first_input,
        "north_only_replay": north_only_replay,
        "operand_record_status": "MISSING_EXECUTED_NORTH_E3W_EXPRESSION",
        "endpoint_ulp_control": {
            "bit_exact": _bits_equal(one, next_one), "differing_cells": 1},
        "compiled_source": {
            "surface": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-400",
            "interior": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:402-427",
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--component-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--prior-json", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.component_root, args.baseline_root,
                         args.static_root, args.prior_json, args.expect_commit)),
                    "measurement arguments missing")
            result = measure(
                args.deck_root, args.component_root, args.baseline_root,
                args.static_root, args.prior_json, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, GateError, r180.Refusal,
            r170.GateError, rung0.GateError) as error:
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
