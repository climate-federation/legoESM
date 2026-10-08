#!/usr/bin/env python3
"""Walk the independent rung-0 kt=8 U exit depth and reciprocal."""

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
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

MAGIC = "NEMO_L4_R166SPG"
PREFIX = "oracle_r166_spg"
EXPECTED_KT = 8
EXPECTED_SUBSTEPS = 65
EXPECTED_GROUPS = 2106
EXPECTED_NONFINITE = 42
SOURCE_ORDER = (
    "exit_depth_u_pre_association",
    "exit_inverse_u_pre_association",
    "exit_depth_u_post_association",
    "exit_inverse_u_post_association",
)
PLANTS = ("none", "registered-count", "source-order", "depth-replay")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)


def _score(candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray) -> dict:
    """Bit score that reports, rather than rejects, non-finite candidates."""

    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape,
            "round-167 score shape mismatch")
    unequal = _bits(candidate) != _bits(oracle)
    wet_unequal = unequal & active
    finite_pair = np.isfinite(candidate) & np.isfinite(oracle) & active
    delta = np.abs(candidate - oracle)
    finite_max = float(np.max(delta[finite_pair], initial=0.0))
    return {
        "bit_exact": not bool(np.any(wet_unequal)),
        "differing_cells": int(np.count_nonzero(wet_unequal)),
        "full_domain_differing_cells": int(np.count_nonzero(unequal)),
        "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(candidate) & active)),
        "oracle_nonfinite": int(np.count_nonzero(~np.isfinite(oracle) & active)),
        "finite_absolute_max": finite_max,
    }


def _registered_summary(
    candidate_depth: np.ndarray, oracle_depth: np.ndarray,
    candidate_inverse: np.ndarray, oracle_inverse: np.ndarray,
    u_mask: np.ndarray,
) -> dict:
    registered = ~np.isfinite(candidate_inverse)
    flats = np.flatnonzero(registered)
    require(flats.size > 0, "kt=8 substep 2 has no non-finite U reciprocal")
    first = tuple(map(int, np.unravel_index(int(flats[0]), registered.shape)))
    depth_unequal = _bits(candidate_depth) != _bits(oracle_depth)
    inverse_replay = u_mask / ((oracle_depth + np.float64(1.0)) - u_mask)
    replay_unequal = _bits(inverse_replay) != _bits(oracle_inverse)
    return {
        "count": int(flats.size),
        "first_j_i": list(first),
        "candidate_depth_finite": bool(np.isfinite(candidate_depth[registered]).all()),
        "oracle_depth_finite": bool(np.isfinite(oracle_depth[registered]).all()),
        "oracle_inverse_finite": bool(np.isfinite(oracle_inverse[registered]).all()),
        "depth_differing_cells": int(np.count_nonzero(depth_unequal & registered)),
        "candidate_depth_min": float(np.min(candidate_depth[registered])),
        "candidate_depth_max": float(np.max(candidate_depth[registered])),
        "oracle_depth_min": float(np.min(oracle_depth[registered])),
        "oracle_depth_max": float(np.max(oracle_depth[registered])),
        "depth_replay_nonfinite": int(np.count_nonzero(
            ~np.isfinite(inverse_replay[registered]))),
        "depth_replay_differing_cells": int(np.count_nonzero(
            replay_unequal & registered)),
        "reciprocal_only_depth_differing_cells": int(np.count_nonzero(
            depth_unequal & registered)),
    }


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "registered-count":
        report["registered_cells"]["count"] -= 1
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "depth-replay":
        report["registered_cells"]["depth_replay_nonfinite"] = 1

    admission = report["admission"]
    require(admission["rank_coverage"] == "exactly-once",
            "rank coverage moved")
    require(len(admission["records"]) == 2 and all(
        row["icycle"] == EXPECTED_SUBSTEPS
        and row["groups"] == EXPECTED_GROUPS
        for row in admission["records"]), "record census moved")
    require(len(admission["terminal_restart_comparisons"]) == 20,
            "restart census moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "compiled source order moved")
    registered = report["registered_cells"]
    require(registered["count"] == EXPECTED_NONFINITE,
            "registered non-finite count moved")
    require(registered["oracle_inverse_finite"],
            "NEMO reciprocal is non-finite at a registered cell")
    require(registered["depth_replay_nonfinite"] == 0,
            "NEMO depth replay leaves a non-finite reciprocal")
    require(report["completed_kt"] == 7,
            "complete private arm no longer reaches kt=7")

    p2 = (registered["candidate_depth_finite"]
          and registered["oracle_depth_finite"]
          and registered["depth_differing_cells"] == EXPECTED_NONFINITE)
    p3 = (registered["depth_replay_nonfinite"] == 0
          and registered["reciprocal_only_depth_differing_cells"] > 0)
    report["prediction_dispositions"] = {
        "R167-P1": "CONFIRMED",
        "R167-P2": "CONFIRMED" if p2 else "REFUTED",
        "R167-P3": "CONFIRMED" if p3 else "REFUTED",
        "R167-P4": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND167_EXIT_DEPTH_WALK"
    return report


def measure(
    deck_root: Path, frame_root: Path, spg_root: Path, baseline_root: Path,
    expect_commit: str,
) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-167 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-167 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-167 walk requires production JIT on CPU")

    admission = check_record.run(
        spg_root, baseline_root, "none", expected_kt=EXPECTED_KT,
        prefix=PREFIX, expected_magic=MAGIC)
    oracle, census = r97.assemble_record(
        spg_root, prefix=PREFIX, expected_kt=EXPECTED_KT,
        expected_magic=MAGIC)
    require(census["coverage"] == "exactly-once",
            "assembled record coverage moved")

    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    ordinary_hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=ordinary_hooks)
    for kt in range(1, 8):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS round167 complete kt={kt}", file=sys.stderr, flush=True)

    # Keep the exact round-166 trace graph whose passivity and 42-cell boundary
    # are already admitted.  Returning seven additional association arrays
    # changes JIT fusion; apply the production association helper afterward to
    # the frozen substep-2 arrays instead of changing the observed graph.
    trace_hooks = ordinary_hooks._replace(expose_barotropic_substeps=True)
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=trace_hooks)
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = observed.substeps
    require(trace["eta_entry"].shape[0] == EXPECTED_SUBSTEPS,
            "production trace substep count moved")

    index = 1
    prefix = "j002"
    association_post = jax.device_get(
        _nemo_external_mode_boundary_association(
            trace["u_exit"][index], trace["v_exit"][index],
            trace["face_depth_u_exit"][index],
            trace["face_depth_v_exit"][index],
            trace["r1_face_depth_u_exit"][index],
            trace["r1_face_depth_v_exit"][index],
            trace["eta_continuity"][index], card.recipe.grid))
    candidate = {
        "exit_depth_u_pre_association": r97._native_u(
            trace["face_depth_u_exit"][index]),
        "exit_inverse_u_pre_association": r97._native_u(
            trace["r1_face_depth_u_exit"][index]),
        "exit_depth_u_post_association": r97._native_u(
            association_post[2]),
        "exit_inverse_u_post_association": r97._native_u(
            association_post[4]),
    }
    reference = {
        "exit_depth_u_pre_association": oracle[f"{prefix}_hu_e"],
        "exit_inverse_u_pre_association": oracle[f"{prefix}_hur_e"],
        "exit_depth_u_post_association": oracle[f"{prefix}_hu_e"],
        "exit_inverse_u_post_association": oracle[f"{prefix}_hur_e"],
    }
    masks = phase3_gate.expected_masks(card)
    active = np.asarray(masks["u"][..., 0], dtype=bool)
    rows = {
        name: _score(candidate[name], reference[name], active)
        for name in SOURCE_ORDER
    }
    registered = _registered_summary(
        candidate["exit_depth_u_pre_association"],
        reference["exit_depth_u_pre_association"],
        candidate["exit_inverse_u_pre_association"],
        reference["exit_inverse_u_pre_association"],
        active.astype(np.float64))
    raw = {
        "format": "nemo-testcase-l4-orca2-round167-exit-depth-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": EXPECTED_KT,
        "substep": index + 1,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "registered_cells": registered,
        "worktree": stamp,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)),
                "measurement requires deck/frame/SPG/baseline roots and commit")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None,
                    "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, check_record.Refusal,
            r97.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND167_EXIT_DEPTH_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
