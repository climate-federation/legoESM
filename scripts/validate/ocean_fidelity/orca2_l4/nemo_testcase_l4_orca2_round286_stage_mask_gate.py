#!/usr/bin/env python3
"""Round 286: gate and D96 census for the resolved WS-RK3 face masks.

The control changes only the Python function that supplies masks to the
already-selected WS-RK3 update.  It restores the pre-round compact masks;
the candidate is the committed production route.  Card configuration and
all other statements are identical within each same-tree comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
)

GateError = r276.GateError
require = r276.require
CARDS = r278.CARDS
ARM = "resolved_ws_stage_masks"


def replay_gate(before: dict, after: dict, *, plant: str = "none") -> dict:
    """Require the measured consumer to close while the stale probe stays put."""

    old = before["candidate_raw_vs_nemo"]
    new = after["candidate_raw_vs_nemo"]
    if plant == "consumer-bit":
        new = {**new, "unequal": 1, "bit_exact": False,
               "max_abs": float.fromhex("0x1p-52")}
    require(old["unequal"] == 1319 and not old["bit_exact"],
            "historical consumer control moved")
    require(new["unequal"] == 0 and new["bit_exact"]
            and new["max_abs"] == 0.0, "stage-update consumer did not close")
    require(before["operand_rows"]["vmask"]["unequal"] == 1319
            and after["operand_rows"]["vmask"]["unequal"] == 1319,
            "standalone compact-mask diagnostic unexpectedly moved")
    require(all(after["passivity"].values()), "passive state changed")
    require(after["one_bit_control"]["unequal"] == 1,
            "one-bit known-answer control did not fire")
    return {
        "format": "nemo-testcase-l4-orca2-round286-stage-mask-replay-v1",
        "status": "PASS_R286_STAGE_MASK_REPLAY",
        "consumer_unequal_before": old["unequal"],
        "consumer_unequal_after": new["unequal"],
        "stale_diagnostic_unequal": 1319,
        "plant": plant,
    }


@contextmanager
def _stage_mask_route(candidate: bool):
    """Select production masks or the pre-round compact-only control."""

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model

    original = model._nemo_ws_stage_face_masks
    if not candidate:
        def compact_only(z_coord, grid, dtype):
            wet_u, wet_v = model.compute_face_masks_3d(z_coord, grid)
            return wet_u.astype(dtype), wet_v.astype(dtype)

        model._nemo_ws_stage_face_masks = compact_only
    try:
        yield
    finally:
        model._nemo_ws_stage_face_masks = original


def _ladder(rung: str, label: str, deck_root: Path, twin_a: Path,
            candidate: bool) -> dict:
    with _stage_mask_route(candidate):
        if rung == "rung10":
            _, card = rung10.card_fields(deck_root)
            trajectory = rung10.candidate_trajectory(
                deck_root, twin_a, card, max_step=10,
                bridge_ssh=(label == "given_nemo_entry"))
            rows, first = [], None
            for checkpoint in trajectory["checkpoints"]:
                if first is None and checkpoint["first_non_bit_field"] is not None:
                    first = {"kt": checkpoint["kt"],
                             "checkpoint": checkpoint["checkpoint"],
                             "field": checkpoint["first_non_bit_field"]}
                for field, score in checkpoint["rows"].items():
                    rows.append({"kt": checkpoint["kt"],
                                 "checkpoint": checkpoint["checkpoint"],
                                 "field": field, **score})
            require(len(rows) == 200 and first is not None,
                    "shipped-card stage-mask ladder is incomplete")
            return {"rows": rows, "row_count": 200,
                    "first_non_bit_checkpoint": first}

        require(label == "independent", "OMT/rung-0 label must be independent")
        card = prior._selected_card(rung, deck_root)
        entry = rung0.assemble_frame(twin_a, 1, 0)
        state = (rung0.bridge_entry(card, entry)
                 if rung == "rung0" else card.recipe.initial_state)
        steps = tuple(range(1, 9 if rung == "omt1" else 11))
        ladder = omt1._run_ladder(card, twin_a, state, label, steps=steps)
        require(ladder["row_count"] == (160 if rung == "omt1" else 200),
                "stage-mask source ladder is incomplete")
        return ladder


def measure(rung: str, label: str, deck_root: Path, twin_a: Path,
            candidate: bool, expect_commit: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require((rung, label) in CARDS, f"unsupported card/label {(rung, label)}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-286 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-286 measurement requires production JIT on CPU")
    return {
        "format": "nemo-testcase-l4-orca2-round286-stage-mask-ladder-v1",
        "status": "PASS_R276_SOURCE_LADDER",
        "rung": rung, "label": label,
        "arm": ARM if candidate else "control",
        "selected_statements": [ARM] if candidate else [],
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "ladder": _ladder(rung, label, deck_root, twin_a, candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("control", "candidate"))
    parser.add_argument("--rung", choices=tuple(sorted({r for r, _ in CARDS})))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--replay-gate", nargs=2, type=Path)
    parser.add_argument("--plant", choices=("none", "exact-loss", "consumer-bit"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.rung, args.label, args.deck_root, args.twin_a,
                             args.measure == "candidate", args.expect_commit)
        elif args.classify:
            require(args.plant != "consumer-bit",
                    "consumer-bit plant belongs to --replay-gate")
            result = r276.classify(*args.classify, plant=args.plant)
        else:
            require(args.replay_gate is not None, "no gate operation selected")
            before = json.loads(args.replay_gate[0].read_text(encoding="utf-8"))
            after = json.loads(args.replay_gate[1].read_text(encoding="utf-8"))
            result = replay_gate(before, after, plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
