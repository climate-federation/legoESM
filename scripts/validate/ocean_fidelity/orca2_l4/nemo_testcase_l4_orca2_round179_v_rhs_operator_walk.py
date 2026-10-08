#!/usr/bin/env python3
"""Replay rung-0 kt=1 V-RHS operators without observing the executable."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as r93,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round94_slow_walk as r94,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round172_passive_rhs_replay as r172,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round84_rhs_walk as r84,
)

FLOOR = np.float64(2.0e-10)
BOUNDARIES = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")
OPERANDS = ("raw_hpg_rhs", "e3v", "vmask", "r1_hv0")
ARMS = ("oracle", "candidate_e3v", "candidate_vmask", "candidate_r1_hv0")
PLANTS = (
    "none", "rank-placement", "record-bit", "source-order",
    "cross-record", "target-mask", "mask-arm", "endpoint-ulp",
)


class GateError(RuntimeError):
    """The admitted records or offline attribution no longer close."""


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
    max_position = int(np.argmax(delta))
    argmax = np.unravel_index(int(flat_active[max_position]), candidate.shape)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "compared_cells": int(c.size),
        "absolute_max": float(delta[max_position]),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "at_floor": bool(float(delta[max_position]) <= FLOOR),
        "argmax": [int(value) for value in argmax],
        "candidate_at_argmax": float(candidate[argmax]),
        "oracle_at_argmax": float(oracle[argmax]),
    }


def _with_jpk_zero(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 3 and value.shape[-1] == 30,
            "physical momentum accumulator must have 30 levels")
    return np.concatenate((value, np.zeros_like(value[..., :1])), axis=-1)


def _depth(rhs, e3v, vmask, reciprocal) -> np.ndarray:
    return r170._source_sum(e3v, rhs, vmask, reciprocal)


def _first(rows: dict[str, dict[str, object]]) -> dict[str, object] | None:
    for boundary in BOUNDARIES:
        if not rows[boundary]["at_floor"]:
            return {"boundary": boundary, **rows[boundary]}
    return None


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["record_census"]["rhs"]["coverage"] = "overlap"
    elif plant == "record-bit":
        report["record_control"]["differing_cells"] = 0
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "cross-record":
        report["cross_record_closure"]["after_zad_to_depth_v"]["bit_exact"] = False
    elif plant == "target-mask":
        report["target"]["cells"] -= 1
    elif plant == "mask-arm":
        report["mask_only_reproduces_candidate_hpg"] = False
    elif plant == "endpoint-ulp":
        report["endpoint_ulp_control"]["bit_exact"] = True

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(all(row["coverage"] == "exactly-once" for row in
                report["record_census"].values()), "rank placement moved")
    require(report["record_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "record one-ULP control did not fire")
    require(tuple(report["source_order"]) == BOUNDARIES,
            "compiled operator source order moved")
    require(tuple(report["operand_order"]) == OPERANDS,
            "operand source order moved")
    require(tuple(report["arm_order"]) == ARMS, "operand arm order moved")
    require(all(row["bit_exact"] for row in
                report["cross_record_closure"].values()),
            "admitted records no longer close")
    require(report["candidate_calibration"]["same_over_floor_set"],
            "offline candidate replay does not reproduce production debt set")
    require(report["candidate_calibration"]["all_faces_at_floor"],
            "offline candidate replay is not calibrated to production")
    require(report["target"] == {
        "cells": 68, "row": 147, "all_on_one_row": True},
        "registered 68-face target moved")
    first = _first(report["operator_rows"])
    require(first == report["first_operator"], "first operator selector moved")
    first_operand = next(
        (name for name in OPERANDS
         if not report["operand_rows"][name]["bit_exact"]), None)
    require(first_operand == report["first_operand"],
            "first operand selector moved")
    require(report["operand_arms"]["oracle"]["bit_exact"],
            "oracle HPG arm is not its own identity")
    require(report["operand_arms"]["candidate_e3v"]["bit_exact"]
            and report["operand_arms"]["candidate_r1_hv0"]["bit_exact"],
            "geometry-only HPG arm moved")
    require(not report["operand_arms"]["candidate_vmask"]["bit_exact"],
            "candidate-mask arm is vacuous")
    require(report["endpoint_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "endpoint one-ULP control did not fire")
    require(report["mask_only_reproduces_candidate_hpg"],
            "candidate-mask one-variable arm does not reproduce HPG boundary")
    report["prediction_ledger"] = {
        "R179-P1": "CONFIRMED",
        "R179-P2": "CONFIRMED",
        "R179-P3": (
            "CONFIRMED" if first is not None
            and first["boundary"] == "after_hpg" else "REFUTED"),
        "R179-P4": (
            "CONFIRMED" if report["first_operand"] == "vmask" else "REFUTED"),
        "R179-P5": "CONFIRMED",
    }
    report["status"] = "HELD_FIRST_V_RHS_OPERAND"
    return report


def measure(deck_root: Path, rhs_root: Path, slow_root: Path,
            static_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_hpg_sco_literal_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _nemo_qco_gdept_z0,
        compute_frozen_geom_density,
    )
    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w0_reference

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-179 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-179 replay requires production JIT on CPU")

    oracle_rhs, rhs_census = r93.assemble_rhs(rhs_root)
    oracle_slow, slow_census = r94.assemble_slow(slow_root)
    oracle_static, static_census = r170.assemble_record(static_root)
    oracle_spg, spg_census = r97.assemble_record(spg_root)

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    freshwater, surface = r93._forcing(state.eta.data.shape)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    traced = jax.device_get(traced_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    passivity = r172._state_rows(traced.state_after, ordinary)
    require(all(passivity.values()), "existing barotropic trace moved the step")

    offline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    offline.prime_step_caches(state)
    _total, _diagnostics, parts = jax.device_get(jax.jit(
        lambda source_state, forcing: r172._stage1_tendency(
            offline, source_state, forcing, card.dt_s, components=True)
    )(state, surface))
    accumulated = jax.device_get(jax.jit(r84.source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    candidate_ops = r170._candidate_reference_operands(card, state)
    candidate_e3v = np.asarray(candidate_ops["e3_v"])
    candidate_vmask = np.asarray(candidate_ops["mask_v"])
    candidate_r1 = np.asarray(candidate_ops["r1_h0_v"])
    oracle_e3v = np.asarray(oracle_static["e3v"])
    oracle_vmask = np.asarray(oracle_static["vmask"])
    oracle_r1 = np.asarray(oracle_static["r1_hv0"])

    oracle_depths = {}
    candidate_depths = {}
    for boundary in BOUNDARIES:
        oracle_depths[boundary] = _depth(
            _with_jpk_zero(oracle_rhs[f"{boundary}_v"]),
            oracle_e3v, oracle_vmask, oracle_r1)
        candidate_depths[boundary] = _depth(
            _with_jpk_zero(r83.native_v(accumulated[f"{boundary}_v"])),
            candidate_e3v, candidate_vmask, candidate_r1)

    # Read only the completed 2-D product.  The live-state wrapper was
    # certified passive in round 172; its operator intermediates remain
    # deliberately unread because materialising them was non-passive in
    # rounds 171/173.
    candidate_final = r83.native_v(traced.slow_forcing_producer["depth_v"])
    oracle_final = np.asarray(oracle_spg["i000_zv_frc"])
    target = np.abs(candidate_final - oracle_final) > FLOOR
    target_locations = np.argwhere(target)
    target_row = int(target_locations[0, 0]) if target_locations.size else -1
    target_summary = {
        "cells": int(np.count_nonzero(target)),
        "row": target_row,
        "all_on_one_row": bool(target_locations.size and
                                np.all(target_locations[:, 0] == target_row)),
    }

    cross_record = {
        "after_zad_to_depth_v": _score(
            oracle_depths["after_zad"], oracle_slow["depth_v"],
            np.ones_like(target)),
        "depth_v_to_completed_v": _score(
            oracle_slow["depth_v"], oracle_final, np.ones_like(target)),
    }
    candidate_delta = np.abs(candidate_depths["after_zad"] - candidate_final)
    candidate_replay_target = (
        np.abs(candidate_depths["after_zad"] - oracle_final) > FLOOR)
    candidate_calibration = {
        "all_faces_at_floor": bool(np.max(candidate_delta) <= FLOOR),
        "absolute_max": float(np.max(candidate_delta)),
        "same_over_floor_set": bool(np.array_equal(candidate_replay_target, target)),
    }
    operator_rows = {
        boundary: _score(candidate_depths[boundary], oracle_depths[boundary], target)
        for boundary in BOUNDARIES
    }
    first = _first(operator_rows)

    geom = compute_frozen_geom_density(
        state, card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    stretch = nemo_r3t_stretch(
        card.recipe.z_coord, state.eta.data, state.H_bathy.data,
        evaluation="nemo_reciprocal")[..., None]
    e3w_live = jnp.asarray(nemo_e3w0_reference(card.recipe.z_coord)) * stretch
    t_depth = jnp.asarray(card.recipe.z_coord.t_depth_ref)[None, None, :]
    gdept_z0 = _nemo_qco_gdept_z0(t_depth, stretch, state.eta.data)
    _raw_u, raw_v = jax.device_get(nemo_hpg_sco_literal_cgrid(
        geom[4], e3w_live, gdept_z0, card.recipe.grid,
        jnp.asarray(card.recipe.model_config.g, dtype=state.eta.data.dtype)))
    raw_v = r83.native_v(raw_v)
    oracle_hpg = _with_jpk_zero(oracle_rhs["after_hpg_v"])
    raw_hpg = _with_jpk_zero(raw_v)
    contributing = np.broadcast_to(target[..., None], oracle_hpg.shape) & (
        oracle_vmask != 0.0)
    operand_rows = {
        "raw_hpg_rhs": _score(raw_hpg, oracle_hpg, contributing),
        "e3v": _score(candidate_e3v, oracle_e3v, contributing),
        "vmask": _score(candidate_vmask, oracle_vmask,
                         np.broadcast_to(target[..., None], oracle_vmask.shape)),
        "r1_hv0": _score(candidate_r1, oracle_r1, target),
    }
    first_operand = next(
        (name for name in OPERANDS if not operand_rows[name]["bit_exact"]), None)

    arms = {
        "oracle": oracle_depths["after_hpg"],
        "candidate_e3v": _depth(oracle_hpg, candidate_e3v, oracle_vmask, oracle_r1),
        "candidate_vmask": _depth(oracle_hpg, oracle_e3v, candidate_vmask, oracle_r1),
        "candidate_r1_hv0": _depth(oracle_hpg, oracle_e3v, oracle_vmask, candidate_r1),
    }
    arm_rows = {
        name: _score(value, oracle_depths["after_hpg"], target)
        for name, value in arms.items()
    }
    mask_replay = _score(
        arms["candidate_vmask"], candidate_depths["after_hpg"], target)
    mask_reproduces = bool(mask_replay["bit_exact"])

    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    report = {
        "format": "nemo-testcase-l4-orca2-round179-v-rhs-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR), "worktree": stamp,
        "record_census": {
            "rhs": rhs_census,
            "slow": slow_census,
            "static": {"coverage": static_census["rank_coverage"],
                       **static_census},
            "spg": spg_census,
        },
        "record_control": {"bit_exact": _bits_equal(one, next_one),
                           "differing_cells": 1},
        "trace_passivity": passivity,
        "source_order": list(BOUNDARIES),
        "operand_order": list(OPERANDS),
        "arm_order": list(ARMS),
        "cross_record_closure": cross_record,
        "candidate_calibration": candidate_calibration,
        "target": target_summary,
        "operator_rows": operator_rows,
        "first_operator": first,
        "operand_rows": operand_rows,
        "first_operand": first_operand,
        "operand_arms": arm_rows,
        "mask_arm_to_candidate_hpg": mask_replay,
        "mask_only_reproduces_candidate_hpg": mask_reproduces,
        "endpoint_ulp_control": {
            "bit_exact": _bits_equal(one, next_one), "differing_cells": 1},
        "compiled_source": {
            "operator_order": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175",
            "vertical_average": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215",
            "hpg_store": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-434",
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--rhs-root", type=Path)
    parser.add_argument("--slow-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.rhs_root, args.slow_root,
                         args.static_root, args.spg_root, args.expect_commit)),
                    "measurement arguments missing")
            result = measure(
                args.deck_root, args.rhs_root, args.slow_root,
                args.static_root, args.spg_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, GateError, r93.GateError,
            r94.GateError, r97.GateError, r170.GateError, rung0.GateError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'} "
              f"{args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
