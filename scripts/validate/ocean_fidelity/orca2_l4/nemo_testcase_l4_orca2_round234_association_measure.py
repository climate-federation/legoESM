#!/usr/bin/env python3
"""Measure the seven-field ORCA2 association with two passive executions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
    nemo_testcase_l4_orca2_round146_boundary_association_gate as r146,
)


PLANTS = ("none", "observer-bit", "post-bit", "registry")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str, *, plant: str = "none") -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-234 association worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-234 association commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-234 association requires production JIT on CPU")

    registry = [row[0] for row in r146.POST_FIELDS]
    if plant == "registry":
        registry[0], registry[1] = registry[1], registry[0]
    r146.validate_post_registry(registry)

    oracle, census = r97.assemble_record(spg_root)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)
    slow = (r97._to_model_u(oracle["i000_zu_frc"]),
            r97._to_model_v(oracle["i000_zv_frc"]))
    raw_history = (
        r97._to_model_u(oracle["i000_ub_e"]),
        r97._to_model_u(oracle["i000_ubb_e"]),
        r97._to_model_v(oracle["i000_vb_e"]),
        r97._to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )
    ordinary = r146._run(
        card, state, freshwater, surface, slow, raw_history,
        expose=False, arm=False)
    observed = r146._run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=False)
    ordinary_arrays = r146._state_arrays(ordinary)
    observed_arrays = r146._state_arrays(observed.state_after)
    if plant == "observer-bit":
        observed_arrays["eta"] = np.array(observed_arrays["eta"], copy=True)
        observed_arrays["eta"].flat[0] = np.nextafter(
            observed_arrays["eta"].flat[0], np.inf)
    passivity = {
        name: r146.exact_row(observed_arrays[name], ordinary_arrays[name])
        for name in r146.STATE_FIELDS
    }
    require(all(row["bit_exact"] for row in passivity.values()),
            "association observer moved ordinary state")

    trace = observed.substeps
    rows = {}
    for name, trace_name, oracle_name, face in r146.POST_FIELDS:
        candidate_model = np.asarray(trace[trace_name][0])
        candidate = (r97._native_u(candidate_model) if face == "u"
                     else r97._native_v(candidate_model) if face == "v"
                     else candidate_model)
        if plant == "post-bit" and name == "eta":
            candidate = np.array(candidate, copy=True)
            candidate.flat[0] = np.nextafter(candidate.flat[0], np.inf)
        rows[name] = r146.exact_row(candidate, oracle[oracle_name])
    require(all(row["bit_exact"] for row in rows.values()),
            "complete association is not bit-exact against NEMO")
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round234-association-measure-v1",
        "status": "PASS_R234_LITERAL_ASSOCIATION",
        "worktree": stamp,
        "passivity": passivity,
        "post_association_rows": rows,
        "record_census": census,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit, plant=args.plant)
    except (OSError, ValueError, KeyError, TypeError, GateError,
            r146.GateError, rhs_walk.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R234_LITERAL_ASSOCIATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
