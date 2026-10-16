#!/usr/bin/env python3
"""Round 285: D122 census for the G4-certified ORCA2 statements.

The same-tree control disables only this round's six selectors.  The shipped
card enables all six.  Rung 0 and OMT-1..4 have no G4 heat/runoff/shortwave
forcing path, so B90 leaves the two stage-3 forcing-unit selectors off there;
the three stress statements and generic stage-1/2 tracer order remain live.
"""

from __future__ import annotations

import argparse
import json
import sys
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

STRESS_AND_STAGE12 = (
    "barotropic_wind_nemo_stored_depth",
    "surface_stress_nemo_fold_vector_sign",
    "zdf_surface_stress_nemo_face_thickness",
    "rk3_stage_tracer_written_order",
)
FORCED_STAGE3 = (
    "rk3_stage3_heat_nemo_written",
    "rk3_stage3_tracer_nemo_stored_thickness",
)
ALL = STRESS_AND_STAGE12 + FORCED_STAGE3
ARM = "g4_certified_set"
CARDS = r278.CARDS


def selected_card(card, *, candidate: bool, rung: str, plant: str = "none"):
    values = {name: False for name in ALL}
    if candidate:
        values.update({name: True for name in STRESS_AND_STAGE12})
        if rung == "rung10":
            values.update({name: True for name in FORCED_STAGE3})
    if plant == "retained-selector" and not candidate:
        values[STRESS_AND_STAGE12[0]] = True
    cfg = card.recipe.model_config._replace(**values)
    actual = {name for name in ALL if getattr(cfg, name)}
    expected = (set(STRESS_AND_STAGE12)
                | (set(FORCED_STAGE3) if rung == "rung10" else set())
                if candidate else set())
    require(actual == expected,
            f"G4 selector scope moved: {sorted(actual)} != {sorted(expected)}")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def _ladder(rung: str, label: str, deck_root: Path, twin_a: Path,
            candidate: bool):
    if rung == "rung10":
        _, base = rung10.card_fields(deck_root)
        card = selected_card(base, candidate=candidate, rung=rung)
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
                "shipped-card G4 ladder is incomplete")
        return {"rows": rows, "row_count": 200,
                "first_non_bit_checkpoint": first}

    require(label == "independent", "OMT/rung-0 label must be independent")
    base = prior._selected_card(rung, deck_root)
    card = selected_card(base, candidate=candidate, rung=rung)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    state = rung0.bridge_entry(card, entry) if rung == "rung0" \
        else card.recipe.initial_state
    steps = tuple(range(1, 9 if rung == "omt1" else 11))
    ladder = omt1._run_ladder(card, twin_a, state, label, steps=steps)
    require(ladder["row_count"] == (160 if rung == "omt1" else 200),
            "G4 source ladder is incomplete")
    return ladder


def measure(rung: str, label: str, deck_root: Path, twin_a: Path,
            candidate: bool, expect_commit: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require((rung, label) in CARDS, f"unsupported card/label {(rung, label)}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-285 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-285 measurement requires production JIT on CPU")
    return {
        "format": "nemo-testcase-l4-orca2-round285-g4-ladder-v1",
        "status": "PASS_R276_SOURCE_LADDER",
        "rung": rung, "label": label,
        "arm": ARM if candidate else "control",
        "selected_statements": list(
            ALL if candidate and rung == "rung10"
            else STRESS_AND_STAGE12 if candidate else ()),
        "inapplicable_forcing_statements": list(
            FORCED_STAGE3 if candidate and rung != "rung10" else ()),
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
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--plant", choices=("none", "exact-loss",
                                             "retained-selector"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")
            if args.plant == "retained-selector":
                _, base = rung10.card_fields(args.deck_root)
                selected_card(base, candidate=False, rung="rung10",
                              plant=args.plant)
                raise GateError("retained-selector plant stayed green")
            result = measure(args.rung, args.label, args.deck_root, args.twin_a,
                             args.measure == "candidate", args.expect_commit)
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        else:
            require(args.decide is not None, "no gate operation selected")
            result = r278.decide(args.decide)
            if args.plant == "exact-loss":
                raise GateError("exact-loss plant requires --classify")
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r276.decision96.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0 if result.get("d122_c_and_d", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
