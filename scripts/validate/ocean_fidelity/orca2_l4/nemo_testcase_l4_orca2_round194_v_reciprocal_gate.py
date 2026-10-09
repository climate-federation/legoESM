#!/usr/bin/env python3
"""Gate NEMO's unmasked V reciprocal over all 65 external substeps."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

SUBSTEPS = 65
PLANTS = (
    "none", "record-bit", "passivity", "source-order",
    "masked-control", "arm-identity", "ulp",
)


class GateError(RuntimeError):
    """The record, arm, or exact recurrence moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bit_row(left, right) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "reciprocal row shape moved")
    unequal = int(np.count_nonzero(
        np.ascontiguousarray(left).view(np.uint64)
        != np.ascontiguousarray(right).view(np.uint64)))
    return {
        "bit_exact": unequal == 0,
        "differing_cells": unequal,
        "cells": int(left.size),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def _active_bit_row(left, right, active) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(left.shape == right.shape == active.shape,
            "active entry row shape moved")
    return _bit_row(left[active], right[active])


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "record-bit":
        report["rows"][0]["exit"]["bit_exact"] = False
    elif plant == "passivity":
        report["passivity"]["ssh"] = False
    elif plant == "source-order":
        report["rows"][0]["substep"] = 2
    elif plant == "masked-control":
        report["masked_control"]["differing_substeps"] = 0
    elif plant == "arm-identity":
        report["private_arm"]["unmasked_v_reciprocal"] = False
    elif plant == "ulp":
        report["one_ulp_control"]["differing_cells"] = 0

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("record_census") == {
        "coverage": "exactly-once", "rank_records": 2, "substeps": SUBSTEPS,
    }, "record census moved")
    require(report.get("private_arm") == {
        "external_mode_association": True,
        "raw_reference_depth": True,
        "unmasked_v_transport": True,
        "materialize_v_transport": True,
        "unmasked_v_reciprocal": True,
    }, "complete private arm moved")
    require(all(report["passivity"].values()), "substep trace is not passive")
    require(all(row["bit_exact"] for row in report["independent_entry"].values()),
            "independent entry moved")
    rows = report.get("rows", [])
    require(len(rows) == SUBSTEPS
            and [row["substep"] for row in rows] == list(range(1, SUBSTEPS + 1)),
            "65-substep source order moved")
    for row in rows:
        for name in ("incoming", "transport", "weight", "exit"):
            require(row[name]["bit_exact"],
                    f"substep {row['substep']} {name} is not bit-exact: "
                    f"{row[name]['differing_cells']} cells, "
                    f"max {row[name]['max_abs']}")
    require(report["masked_control"]["differing_substeps"] > 0
            and report["masked_control"]["first_differing_substep"] == 2,
            "masked reciprocal control did not reproduce the first debt")
    require(report["one_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1,
    }, "one-ULP accumulator control did not fire")
    report["prediction_ledger"] = {
        "R194-P1": "CONFIRMED_65_OF_65_BIT_EXACT",
        "R194-P6": "CONFIRMED_CONTROLS_BIND",
    }
    report["status"] = "PASS_R194_V_RECIPROCAL_65_SUBSTEPS"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-194 reciprocal gate requires its clean candidate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "reciprocal gate requires production JIT on CPU")

    oracle, census = r97.assemble_record(spg_root)
    require(len(census["records"]) == 2
            and all(row["icycle"] == SUBSTEPS for row in census["records"]),
            "round-96 record is not rank-complete at 65 substeps")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    frame = rung0.assemble_frame(frame_root, 1, 0)
    candidate = rung0.candidate_fields(state)
    masks = phase3_gate.expected_masks(card)
    entry_masks = {
        "T": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "S": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "u": np.asarray(masks["u"], dtype=bool),
        "v": np.asarray(masks["v"], dtype=bool),
        "ssh": np.asarray(masks["ssh"], dtype=bool),
    }
    entry = {
        name: _active_bit_row(candidate[name], frame[name], entry_masks[name])
        for name in ("T", "S", "u", "v", "ssh")
    }
    require(all(row["bit_exact"] for row in entry.values()),
            "independent entry is not exact")

    forcing = (
        np.asarray(oracle["i000_ssh_frc"]),
        r178._to_model_u(oracle["i000_zu_frc"]),
        r178._to_model_v(oracle["i000_zv_frc"]),
    )
    dt = float(oracle["i000_entry_sc"][0])
    count = int(oracle["i000_entry_sc"][2])
    reference_depth = rung0.ladder.build_reference_depth_override(card)

    def solve(trace: bool, reciprocal: bool):
        return jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, dt, count, card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
                _nemo_reference_face_depth_test_override=reference_depth,
                _nemo_unmasked_v_transport_test_override=True,
                _nemo_materialize_v_transport_test_override=True,
                _nemo_unmasked_v_reciprocal_test_override=reciprocal,
                _nemo_external_mode_association_test_override=True,
            )
        ))(state, *forcing)

    live_state, live_transport = jax.device_get(solve(False, True))
    traced_state, traced_transport, trace = jax.device_get(solve(True, True))
    _, _, masked_trace = jax.device_get(solve(True, False))
    passivity = {
        "ssh": bool(np.array_equal(live_state.eta.data, traced_state.eta.data)),
        "u": bool(np.array_equal(live_state.u.data, traced_state.u.data)),
        "v": bool(np.array_equal(live_state.v.data, traced_state.v.data)),
        "u_barotropic": bool(np.array_equal(
            live_state.uu_b.data, traced_state.uu_b.data)),
        "v_barotropic": bool(np.array_equal(
            live_state.vv_b.data, traced_state.vv_b.data)),
        "transport_u": bool(np.array_equal(live_transport[0], traced_transport[0])),
        "transport_v": bool(np.array_equal(live_transport[1], traced_transport[1])),
    }

    rows = []
    masked_differing = []
    for index in range(SUBSTEPS):
        jn = index + 1
        target_key = f"j{jn:03d}_vn_adv"
        # dynspg_ts.f90:382-383 initializes the accumulator to zero.  The
        # record's o000_vn_adv is the post-loop averaged output (:890-906),
        # not a substep-1 entry field; later entries are the preceding exits.
        incoming_oracle = (
            np.zeros_like(oracle[target_key]) if jn == 1
            else oracle[f"j{jn - 1:03d}_vn_adv"]
        )
        weight = np.asarray(trace["transport_weight"][index]).reshape(1)
        oracle_weight = np.asarray(oracle[f"j{jn:03d}_sum_coef"])[1:2]
        exit_value = r178._native_v(trace["transport_sum_v_exit"][index])
        rows.append({
            "substep": jn,
            "incoming": _bit_row(
                r178._native_v(trace["transport_sum_v_entry"][index]),
                incoming_oracle),
            "transport": _bit_row(
                r178._native_v(trace["transport_metric_v"][index]),
                oracle[f"j{jn:03d}_zhV"]),
            "weight": _bit_row(weight, oracle_weight),
            "exit": _bit_row(exit_value, oracle[target_key]),
        })
        masked_exit = r178._native_v(
            masked_trace["transport_sum_v_exit"][index])
        if not np.array_equal(masked_exit, oracle[target_key]):
            masked_differing.append(jn)

    zero = np.zeros((2,), dtype=np.float64)
    planted = zero.copy()
    planted[0] = np.nextafter(0.0, np.inf)
    return classify({
        "format": "nemo-testcase-l4-orca2-round194-v-reciprocal-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2,
            "substeps": SUBSTEPS,
        },
        "private_arm": {
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
            "unmasked_v_reciprocal": True,
        },
        "independent_entry": entry,
        "passivity": passivity,
        "rows": rows,
        "masked_control": {
            "differing_substeps": len(masked_differing),
            "first_differing_substep": (
                masked_differing[0] if masked_differing else None),
        },
        "one_ulp_control": _bit_row(planted, zero),
        "compiled_sources": [
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90:152",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608",
        ],
    })


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
            result = measure(args.deck_root, args.frame_root, args.spg_root,
                             args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, r97.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R194_V_RECIPROCAL_65_SUBSTEPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
