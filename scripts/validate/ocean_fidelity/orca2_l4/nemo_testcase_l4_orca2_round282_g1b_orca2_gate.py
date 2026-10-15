#!/usr/bin/env python3
"""Round 282: D96 census of the two G1b statements on ORCA2.

The control and all three candidate arms run from one committed tree.  An arm
may change only the two fields imported from G1b; every Decision-122 selector
already active on the production ORCA2 card remains active.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round277_rung_census_gate as r277,
)

GateError = r276.GateError
require = r276.require

A = "een_nemo_relative_vorticity"
B = "barotropic_literal_fold_unit_vector_form"
ARMS = {
    "control": (),
    "npcrv_f_area": (A,),
    "vector_fold_unit": (B,),
    "pair": (A, B),
}
CARDS = (("rung0", "independent"), ("omt1", "independent"),
         ("omt2", "independent"), ("omt3", "independent"),
         ("omt4", "independent"), ("rung10", "given_nemo_entry"),
         ("rung10", "independent"))


def arm_card(card, arm: str, *, plant: str = "none"):
    """Change exactly A/B, preserving the production card's other fields."""

    require(arm in ARMS, f"unknown arm {arm}")
    cfg = card.recipe.model_config
    selected = set(ARMS[arm])
    if plant == "extra-selector":
        selected.add("rk3_stage1_ssh_nemo_literal")
    unknown = selected - {A, B}
    require(not unknown, f"arm changes fields outside A/B: {sorted(unknown)}")
    changed = cfg._replace(
        een_nemo_relative_vorticity=A in selected,
        barotropic=cfg.barotropic._replace(
            barotropic_literal_fold_unit_vector_form=B in selected))
    require(changed.momentum_advection == "vector_invariant"
            and changed.vorticity_scheme == "een_total",
            "G1b statements require ORCA2's vector/np_CRV program")
    require(card.recipe.z_coord.nemo_een_barotropic is not None,
            "card lacks raw NEMO F-area/face-depth operands")
    got = {name for name, value in (
        (A, changed.een_nemo_relative_vorticity),
        (B, changed.barotropic.barotropic_literal_fold_unit_vector_form))
        if value}
    require(got == set(ARMS[arm]),
            f"selected fields differ from arm: {sorted(got)}")
    return card._replace(recipe=card.recipe._replace(model_config=changed))


def _rung10_ladder(deck_root: Path, twin_a: Path, arm: str, label: str):
    _, base = rung10.card_fields(deck_root)
    card = arm_card(base, arm)
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
            "shipped-card G1b ladder is incomplete")
    return {"rows": rows, "row_count": len(rows),
            "first_non_bit_checkpoint": first}


def measure(rung: str, deck_root: Path, twin_a: Path, arm: str,
            label: str, expect_commit: str, plant: str) -> dict:
    require((rung, label) in CARDS, f"unsupported card/label {(rung, label)}")
    require(arm in ARMS, f"unknown arm {arm}")
    require(plant in ("none", "extra-selector"), "bad measurement plant")

    # The established rung-0/OMT runner calls r276.arm_card dynamically.
    r276.MODEL.add(A)
    r276.BAROTROPIC.add(B)
    r276.ARMS.update(ARMS)
    old = r276.arm_card
    r276.arm_card = arm_card
    try:
        if rung != "rung10":
            result = r276.measure(
                rung, deck_root, twin_a, arm, expect_commit, plant)
        else:
            import jax
            from legoesm.core.precision import (
                PrecisionPolicy, get_policy, set_policy)
            from legoesm.ocean.fidelity.provenance import worktree_stamp

            stamp = worktree_stamp()
            require(stamp["clean"]
                    and stamp["commit"].lower() == expect_commit.lower(),
                    "round-282 measurement requires its clean committed tree")
            set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
            require(get_policy() == PrecisionPolicy.fp64(
                transcendentals="libm") and bool(jax.config.jax_enable_x64),
                "fp64/libm is not active")
            require(jax.default_backend() == "cpu"
                    and not jax.config.jax_disable_jit,
                    "round-282 measurement requires production JIT on CPU")
            # Exercise the same arm guard/plant before running the trajectory.
            if plant != "none":
                _, base = rung10.card_fields(deck_root)
                arm_card(base, arm, plant=plant)
            result = {
                "rung": rung, "label": label, "arm": arm,
                "selected_statements": list(ARMS[arm]),
                "worktree": stamp,
                "execution": "production-jit-cpu-fp64-x64-libm",
                "ladder": _rung10_ladder(deck_root, twin_a, arm, label),
            }
    finally:
        r276.arm_card = old
    result.update({
        "format": "nemo-testcase-l4-orca2-round282-g1b-ladder-v1",
        "status": "PASS_R276_SOURCE_LADDER", "round": 282,
        "selected_statements": list(ARMS[arm]),
    })
    return result


def decide(census_paths: list[Path]) -> dict:
    """Apply D96 to one arm over every frozen card/label."""

    docs = [json.loads(path.read_text(encoding="utf-8"))
            for path in census_paths]
    require(all(d.get("status") == "PASS_R276_EARLY_LATE_CENSUS"
                for d in docs), "a census is not admitted")
    arms = {d["arm"] for d in docs}
    require(len(arms) == 1 and "control" not in arms,
            f"decision mixes arms: {sorted(arms)}")
    commits = {d["worktree_commit"] for d in docs}
    require(len(commits) == 1, "censuses come from different trees")
    got = sorted((d["rung"], d["label"]) for d in docs)
    require(got == sorted(CARDS), f"card/label set moved: {got}")
    cards = []
    for document in sorted(
            docs, key=lambda d: CARDS.index((d["rung"], d["label"]))):
        cards.append({
            "rung": document["rung"], "label": document["label"],
            "d96_eligible": document["decision96_eligible"],
            "exact_rows_lost": len(document["exact_row_losses"]),
            "first_over_bar": document["first_over_bar"],
            "rms_direction_census": document["rms_direction_census"],
            "final_stage3_ssh_max": document["final_stage3_ssh_max"],
        })
    passed = all(card["d96_eligible"] for card in cards)
    return {
        "format": "nemo-testcase-l4-orca2-round282-d96-decision-v1",
        "status": "PASS_R282_D96_DECISION" if passed
        else "FAIL_R282_D96_DECISION",
        "arm": arms.pop(), "worktree_commit": commits.pop(),
        "cards": cards, "d96_pass": passed,
    }


def _row(document: dict, target: tuple[int, str, str]) -> dict:
    for row in document["rows"]:
        if (row["kt"], row["checkpoint"], row["field"]) == target:
            return row
    raise GateError(f"missing ladder row {target}")


def _census(rows: list[dict]) -> dict[str, object]:
    counts = Counter(row["metrics"]["rms"]["direction"] for row in rows)
    rendered = {name: int(counts[name])
                for name in ("toward", "away", "equal")}
    votes = rendered["toward"] + rendered["away"]
    return {
        **rendered,
        "rms_moving_votes": votes,
        "strict_majority_toward": rendered["toward"] * 2 > votes,
        "strict_majority_away": rendered["away"] * 2 > votes,
    }


def classify(control_path: Path, candidate_path: Path, plant: str) -> dict:
    """D96 census, allowing a candidate to move the first-non-bit marker.

    D96 scores the control's first-over-bar row.  A qualifying fix may make
    that row bit-exact and therefore move the descriptive marker later; the
    older round-276 helper incorrectly refused that improvement.
    """

    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(control.get("status") == "PASS_R276_SOURCE_LADDER"
            and control.get("arm") == "control", "control is not admitted")
    require(candidate.get("status") == "PASS_R276_SOURCE_LADDER"
            and candidate.get("arm") != "control", "candidate is not admitted")
    require(control["rung"] == candidate["rung"]
            and control["label"] == candidate["label"],
            "card or claim label moved")
    require(control["worktree"]["commit"] == candidate["worktree"]["commit"],
            "control and candidate are not from one committed tree")
    before = copy.deepcopy(control["ladder"])
    after = copy.deepcopy(candidate["ladder"])
    if plant == "exact-loss":
        before["rows"][0]["bit_identical"] = True
        before["rows"][0]["unequal"] = 0
        after["rows"][0]["bit_identical"] = False
        after["rows"][0]["unequal"] = 1
    key = lambda row: (int(row["kt"]), row["checkpoint"], row["field"])
    old = {key(row): row for row in before["rows"]}
    new = {key(row): row for row in after["rows"]}
    require(len(old) in (160, 200) and old.keys() == new.keys(),
            "candidate ladder row set moved")
    moved, losses = [], []
    for target in sorted(old):
        left, right = old[target], new[target]
        if left == right:
            continue
        if left["bit_identical"] and not right["bit_identical"]:
            losses.append(target)
        metrics = {}
        for name in ("rms", "max_abs", "mean_abs_over_unequal"):
            delta = float(right[name]) - float(left[name])
            metrics[name] = {
                "before": left[name], "after": right[name], "delta": delta,
                "direction": "toward" if delta < 0 else "away" if delta > 0
                else "equal",
            }
        moved.append({
            "kt": target[0], "checkpoint": target[1], "field": target[2],
            "bit_identical_before": left["bit_identical"],
            "bit_identical_after": right["bit_identical"],
            "unequal_before": left["unequal"],
            "unequal_after": right["unequal"], "metrics": metrics,
        })
    early = [row for row in moved if row["kt"] <= 3]
    late = [row for row in moved if row["kt"] >= 4]
    if plant == "exact-loss":
        require(not losses,
                f"previously exact rows left the bar: {losses}")
    all_census = _census(moved)
    first = before["first_non_bit_checkpoint"]
    require(first is not None, "control has no first non-bit row")
    first_target = (int(first["kt"]), first["checkpoint"], first["field"])
    first_direction = r276.decision96._direction(
        _row(before, first_target), _row(after, first_target), "rms")
    last_kt = max(old)[0]
    final_target = (last_kt, "stage3", "ssh")
    eligible = (bool(all_census["strict_majority_toward"])
                and first_direction in ("toward", "equal") and not losses)
    away = [row for row in moved
            if row["metrics"]["rms"]["direction"] == "away"]
    return {
        "format": "nemo-testcase-l4-orca2-round282-census-v1",
        "status": "PASS_R276_EARLY_LATE_CENSUS",
        "rung": candidate["rung"], "label": candidate["label"],
        "arm": candidate["arm"],
        "selected_statements": candidate["selected_statements"],
        "worktree_commit": candidate["worktree"]["commit"],
        "row_count": len(old), "bit_touched_row_count": len(moved),
        "exact_row_losses": losses,
        "rms_direction_census": {
            "all": all_census, "early_kt1_3": _census(early),
            "late_kt4_plus": _census(late)},
        "first_away_kt": min((row["kt"] for row in away), default=None),
        "first_over_bar": {"target": list(first_target),
                           "rms_direction": first_direction},
        "first_non_bit_before": before["first_non_bit_checkpoint"],
        "first_non_bit_after": after["first_non_bit_checkpoint"],
        "final_stage3_ssh_max": {
            "kt": last_kt, "before": _row(before, final_target)["max_abs"],
            "after": _row(after, final_target)["max_abs"]},
        "decision96_eligible": eligible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=tuple(dict(CARDS)))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--plant", choices=(
        "none", "extra-selector", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.arm, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.rung, args.deck_root, args.twin_a, args.arm,
                             args.label, args.expect_commit, args.plant)
        elif args.classify:
            require(args.plant in ("none", "exact-loss"),
                    "bad classification plant")
            result = classify(*args.classify, plant=args.plant)
        else:
            require(args.decide is not None and args.plant == "none",
                    "decision inputs are required")
            result = decide(args.decide)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung10.GateError, r276.decision96.GateError,
            OSError, KeyError,
            TypeError, ValueError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0 if not result.get("status", "").startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
