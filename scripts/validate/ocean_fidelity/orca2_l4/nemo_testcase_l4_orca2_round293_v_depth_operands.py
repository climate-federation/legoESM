#!/usr/bin/env python3
"""Split round 292's fold-image V-depth boundary on admitted operands."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round292_external_v_target as r292,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

FLOOR = np.float64(2.0e-10)
OPERAND_ORDER = (
    "hv_0", "r1_e1e2v", "e1e2t", "eta_mid_owned",
    "north_e1e2t_halo", "north_eta_halo", "ssvmask",
)
ARMS = (
    "baseline", "raw_hv_only", "raw_v_surface_mask",
    "mask_t_neighbor_pair",
)
PLANTS = (
    "none", "passivity", "source-order", "replay-calibration",
    "first-operand", "single-arm", "pair-closure", "locality",
    "consumer-closure", "one-bit",
)


class GateError(RuntimeError):
    """An admitted stream, operand split, or falsifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _complete(result: dict[str, object]) -> dict[str, object]:
    return result.get("complete", result)


def _score(left, right, support) -> dict[str, object]:
    return r228._difference(
        np.asarray(left, np.float64)[np.asarray(support, bool)],
        np.asarray(right, np.float64)[np.asarray(support, bool)],
    )


def _regional(left, right, support) -> dict[str, object]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    support = np.asarray(support, bool)
    require(left.shape == right.shape == support.shape == (148, 90),
            "rank-slab score shape moved")
    return {
        "complete": _score(left, right, support),
        "interior": _score(left[:-3], right[:-3], support[:-3]),
        "fold_band": _score(left[-3:], right[-3:], support[-3:]),
    }


def measure_offline(deck_root: Path, twin_a: Path, twin_b: Path,
                    round292_report: Path, expect_commit: str) -> dict[str, object]:
    """Split the geometric boundary without recompiling the full model step."""

    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_reference_face_depths,
        _nemo_ssh_avg_apply,
        _nemo_ssh_avg_prep,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-293 offline split requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu", "round 293 requires CPU")

    prior = json.loads(round292_report.read_text(encoding="utf-8"))
    require(prior["status"] == "PASS_R292_MEASUREMENT",
            "round-292 parent measurement moved")
    require(prior["first_valued"]["name"] == "face_depth_v_mid",
            "round-292 first boundary moved")
    require(prior["first_valued"]["regions"]["complete"]["unequal"] == 13,
            "round-292 depth census moved")
    require(prior["first_valued"]["regions"]["complete"]["max_abs"] == 719.0,
            "round-292 depth maximum moved")
    eta_parent = next(row for row in prior["rows"]
                      if row["substep"] == 1 and row["name"] == "eta_mid")
    v_parent = next(row for row in prior["rows"]
                    if row["substep"] == 1 and row["name"] == "v_mid")
    require(eta_parent["regions"]["complete"]["bit_exact"],
            "round-292 midpoint SSH is no longer exact")
    require(v_parent["regions"]["complete"]["bit_exact"],
            "round-292 midpoint V is no longer exact")

    admission = r235._admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    oracle_eta = np.asarray(substeps["eta_mid"][0], np.float64)
    oracle_v = np.asarray(advmean["velocity_v"][0], np.float64)
    oracle_depth = np.asarray(advmean["face_depth_v"][0], np.float64)
    oracle_transport = np.asarray(advmean["metric_v"][0], np.float64)
    support = np.asarray(advmean["r1_e1v"] != 0.0, bool)

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    cfg = card.recipe.model_config
    require(cfg.momentum_advection == "vector_invariant",
            "OMT-4 momentum branch moved")
    require(not cfg.barotropic.barotropic_literal_fold_unit_vector_form,
            "held literal fold unit unexpectedly selected")
    state = card.recipe.initial_state
    dtype = jnp.asarray(state.eta.data).dtype
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    eta = jnp.asarray(state.eta.data, dtype=dtype)
    area = jnp.asarray(grid.area, dtype=dtype)
    land = jnp.asarray(state.land_mask.data, dtype=dtype)
    u_mask = jnp.asarray(state.u_mask.data, dtype=dtype)
    v_mask = jnp.asarray(state.v_mask.data, dtype=dtype)
    require(np.array_equal(np.asarray(eta)[:, :90], oracle_eta),
            "initial/substep-1 midpoint SSH calibration moved")

    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data, dtype=dtype), land, grid, dtype)
    production_depth = _nemo_ssh_avg_apply(
        eta, u_mask, v_mask, grid, area, prep)[1]
    replay_depth = r245._v_depth_replay(
        eta, prep[1], prep[3], area, v_mask, grid)
    production_native = r235._native(production_depth, "v")
    replay_native = r235._native(replay_depth, "v")
    replay_calibration = _regional(
        replay_native, production_native, support)
    baseline_vs_nemo = _regional(
        production_native, oracle_depth, support)
    require(replay_calibration["complete"]["unequal"] == 0,
            "independent source replay missed production")
    require(baseline_vs_nemo["complete"]["unequal"] == 13
            and baseline_vs_nemo["complete"]["max_abs"] == 719.0,
            "offline replay does not reproduce round-292 boundary")

    raw_hu, raw_hv_compact = _nemo_literal_reference_face_depths(z_coord, dtype)
    del raw_hu
    raw_hv = np.asarray(z_coord.nemo_hv_0, np.float64)
    raw_area_t = np.asarray(z_coord.nemo_e1e2t, np.float64)
    raw_area_v = np.asarray(z_coord.nemo_e1e2v, np.float64)
    raw_r1_area_v = np.divide(
        1.0, raw_area_v, out=np.zeros_like(raw_area_v), where=raw_area_v > 0.0)
    raw_vmask = r245._raw_surface_vmask(card)
    raw_r1_compact = r245._compact_v(raw_r1_area_v)
    raw_vmask_compact = r245._compact_v(raw_vmask)

    operand_rows = {
        "hv_0": r228._difference(r235._native(prep[1], "v"), raw_hv[:, :90]),
        "r1_e1e2v": r228._difference(
            r235._native(prep[3], "v"), raw_r1_area_v[:, :90]),
        "e1e2t": r228._difference(np.asarray(area)[:, :90], raw_area_t[:, :90]),
        "ssvmask": r228._difference(
            r235._native(v_mask, "v"), raw_vmask[:, :90]),
    }

    def depth(*, raw_mask=False, t_neighbor=False):
        return r245._v_depth_replay(
            eta, raw_hv_compact, raw_r1_compact, raw_area_t,
            raw_vmask_compact if raw_mask else v_mask, grid,
            north_area_row=-2 if t_neighbor else -1,
            north_eta_row=-2 if t_neighbor else -1)

    arm_depths = {
        "baseline": production_depth,
        "raw_hv_only": depth(),
        "raw_v_surface_mask": depth(raw_mask=True),
        "mask_t_neighbor_pair": depth(raw_mask=True, t_neighbor=True),
    }
    direct_arms = {}
    e1v = np.reciprocal(
        np.asarray(advmean["r1_e1v"], np.float64),
        where=np.asarray(advmean["r1_e1v"]) != 0.0,
        out=np.zeros_like(np.asarray(advmean["r1_e1v"], np.float64)))
    for name, candidate_depth in arm_depths.items():
        native_depth = r235._native(candidate_depth, "v")
        product = np.asarray(e1v * oracle_v * native_depth, np.float64)
        direct_arms[name] = {
            "v_mid": _regional(oracle_v, oracle_v, support),
            "face_depth_v_mid": _regional(native_depth, oracle_depth, support),
            "metric_transport_v": _regional(
                product, oracle_transport, support),
        }

    return {
        "format": "nemo-testcase-l4-orca2-round293-v-depth-operands-v2",
        "status": "PASS_R293_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1_substep1",
        "execution": "offline-dual-replay-cpu-fp64-libm",
        "source": {
            "compiled_file": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90"),
            "lines": "509-520",
        },
        "worktree": stamp,
        "record_admission": admission,
        "parent_round292": {
            "worktree": prior["worktree"],
            "first_valued": prior["first_valued"],
            "eta_mid": eta_parent,
            "v_mid": v_parent,
        },
        "resolved_config": {
            "momentum_advection": cfg.momentum_advection,
            "literal_fold_unit_vector_form": False,
        },
        "operand_order": list(OPERAND_ORDER),
        "arms": list(ARMS),
        "passivity": {name: {"offline": True} for name in ARMS},
        "operand_rows": operand_rows,
        "baseline_replay_vs_production": replay_calibration["complete"],
        "baseline_vs_nemo": baseline_vs_nemo["complete"],
        "cumulative_substitution": {},
        "direct_arms": direct_arms,
        "one_bit_control": r228._difference(
            np.asarray([1.0]), np.nextafter(np.asarray([1.0]), np.inf)),
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        first = next(iter(report["passivity"]["baseline"]))
        report["passivity"]["baseline"][first] = False
    elif plant == "source-order":
        report["operand_order"][0], report["operand_order"][1] = (
            report["operand_order"][1], report["operand_order"][0])
    elif plant == "replay-calibration":
        report["baseline_replay_vs_production"]["unequal"] = 1
    elif plant == "first-operand":
        report["operand_rows"]["hv_0"]["unequal"] = 0
    elif plant == "single-arm":
        _complete(report["direct_arms"]["raw_hv_only"][
            "face_depth_v_mid"])["unequal"] = 1
    elif plant == "pair-closure":
        _complete(report["direct_arms"]["mask_t_neighbor_pair"][
            "face_depth_v_mid"])["unequal"] = 1
    elif plant == "locality":
        report["direct_arms"]["mask_t_neighbor_pair"][
            "face_depth_v_mid"]["interior"]["max_abs"] = 1.0
    elif plant == "consumer-closure":
        _complete(report["direct_arms"]["mask_t_neighbor_pair"][
            "metric_transport_v"])["unequal"] = 1
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R293_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] ==
            "independent_omt4_kt1_stage1_substep1", "claim label moved")
    require(tuple(report["operand_order"]) == OPERAND_ORDER,
            "compiled operand order moved")
    require(tuple(report["arms"]) == ARMS, "causal-arm registry moved")
    require(all(all(values.values())
                for values in report["passivity"].values()),
            "a diagnostic arm moved the completed state")
    require(report["baseline_replay_vs_production"]["unequal"] == 0,
            "source replay missed the executable face depth")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    operand_rows = report["operand_rows"]
    first = next(
        (name for name in ("hv_0", "r1_e1e2v", "e1e2t", "ssvmask")
         if operand_rows[name]["unequal"] > 0), None)
    require(first == "hv_0", f"first direct operand moved: {first}")

    raw_depth = report["direct_arms"]["raw_v_surface_mask"][
        "face_depth_v_mid"]
    raw_hv_depth = report["direct_arms"]["raw_hv_only"][
        "face_depth_v_mid"]
    pair_depth = report["direct_arms"]["mask_t_neighbor_pair"][
        "face_depth_v_mid"]
    pair_transport = report["direct_arms"]["mask_t_neighbor_pair"][
        "metric_transport_v"]
    baseline_transport = report["direct_arms"]["baseline"][
        "metric_transport_v"]
    require(_complete(raw_hv_depth)["unequal"] == 0,
            "raw hv_0 no longer closes the substep-1 depth boundary")
    require(_complete(raw_depth)["unequal"] == 0,
            "raw V surface mask arm moved from its registered null")
    require(_complete(pair_depth)["unequal"] == 0,
            "surface-mask/T-neighbour pair does not close the depth")
    require(_complete(pair_transport)["unequal"] == 0,
            "depth pair does not close the immediate V transport")
    require(_complete(baseline_transport)["unequal"] == 0,
            "substep-1 V transport is no longer the registered zero-V null")
    require(pair_depth["interior"]["max_abs"] <= FLOOR,
            "paired depth movement is not fold-local")

    report["first_direct_operand"] = first
    report["cancelling_partner"] = {
        "held_statement": "raw hv_0 fold-image face depth",
        "partner": None,
        "disposition": (
            "no partner acts at substep 1: raw hv_0 alone closes; the mask "
            "and T-neighbour arms are null at this boundary"),
        "raw_hv_only_depth_unequal": _complete(raw_hv_depth)["unequal"],
        "raw_mask_alone_depth_unequal": _complete(raw_depth)["unequal"],
        "pair_depth_unequal": _complete(pair_depth)["unequal"],
        "pair_transport_unequal": _complete(pair_transport)["unequal"],
    }
    report["prediction_ledger"] = {
        "R293-P1": "CONFIRMED",
        "R293-P2": "CONFIRMED",
        "R293-P3": "REFUTED_HV_0_IS_FIRST_NONBIT_OPERAND",
        "R293-P4": "REFUTED_RAW_HV_0_ALONE_CLOSES_AT_SUBSTEP1",
        "R293-P5": "REFUTED_CONSUMER_WAS_ALREADY_EXACT_ZERO_V",
        "R293-P6": "CONFIRMED",
    }
    report["status"] = "HELD_R293_HV0_SINGLE_STATEMENT_CLOSES_BOUNDARY"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-offline", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--round292-report", type=Path)
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure_offline:
            require(all((args.deck_root, args.twin_a, args.twin_b,
                         args.round292_report, args.expect_commit)),
                    "offline measurement arguments incomplete")
            result = measure_offline(
                args.deck_root, args.twin_a, args.twin_b,
                args.round292_report, args.expect_commit)
        else:
            require(args.report_in is not None, "classification report missing")
            result = classify(json.loads(args.report_in.read_text()),
                              plant=args.plant)
    except (GateError, r245.GateError, OSError, KeyError, TypeError,
            ValueError, IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
