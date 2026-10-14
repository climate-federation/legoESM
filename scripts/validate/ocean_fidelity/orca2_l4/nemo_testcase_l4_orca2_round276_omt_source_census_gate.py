#!/usr/bin/env python3
"""Measure round-276 source arms and classify early versus late OMT rows."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    nemo_testcase_l4_orca2_round274_g1_full_import_gate as round274,
    nemo_testcase_l4_orca2_round275_g1_source_bisect_gate as round275,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_decision96_gate as decision96,
)


class GateError(RuntimeError):
    """The source arm or its row census moved outside the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


MODEL = set(round275.MODEL) | {
    "zdf_tracer_nemo_rounded_thickness",
    "barotropic_rhs_ascending_level_sum",
}
BAROTROPIC = set(round275.BAROTROPIC)
COORDINATE = set(round275.COORDINATE)

ARMS: dict[str, tuple[str, ...]] = {
    "control": (),
    "barotropic_een_fold_pair": (
        "barotropic_een_ratio_fold_exchange",
        "barotropic_een_fe3mask_pivot_row",
    ),
    "stored_t_ratio": ("nemo_r3t_stored_product",),
    "stage_t_ratio": ("rk3_stage_r3t_nemo_literal",),
    "stage_velocity": ("rk3_stage_velocity_written_order",),
    "g2_momentum_divisors": ("zdf_momentum_nemo_face_thickness",),
    "g2_tracer_diagonal": ("zdf_tracer_nemo_rounded_thickness",),
    "g2_rhs_ascending_sum": ("barotropic_rhs_ascending_level_sum",),
}


def arm_card(card, arm: str, *, plant: str = "none"):
    """Select exactly one preregistered unit on the production rung card."""

    require(arm in ARMS, f"unknown arm {arm}")
    expected = set(ARMS[arm])
    names = list(expected)
    if plant == "extra-selector":
        extra = "een_vorticity_written_order"
        if extra in names:
            extra = "zdf_barotropic_removal_nemo_literal"
        names.append(extra)
    unknown = set(names) - MODEL - BAROTROPIC - COORDINATE
    require(not unknown, f"arm has unknown selectors: {sorted(unknown)}")

    # Start from the current card, then remove the whole imported statement
    # universe.  A private arm can therefore never inherit a prior candidate.
    cfg = card.recipe.model_config._replace(
        **{name: False for name in MODEL},
        barotropic=card.recipe.model_config.barotropic._replace(
            **{name: False for name in BAROTROPIC}),
    )
    cfg = cfg._replace(
        **{name: True for name in names if name in MODEL},
        barotropic=cfg.barotropic._replace(
            **{name: True for name in names if name in BAROTROPIC}),
    )
    z_coord = card.recipe.z_coord._replace(
        nemo_r3t_stored_product="nemo_r3t_stored_product" in names)
    selected = {name for name in MODEL if getattr(cfg, name)}
    selected |= {name for name in BAROTROPIC if getattr(cfg.barotropic, name)}
    if z_coord.nemo_r3t_stored_product:
        selected.add("nemo_r3t_stored_product")
    require(selected == expected,
            f"selected fields differ from arm: {sorted(selected)} != "
            f"{sorted(expected)}")
    return card._replace(recipe=card.recipe._replace(
        model_config=cfg, z_coord=z_coord))


def measure(rung: str, deck_root: Path, twin_a: Path, arm: str,
            expect_commit: str, plant: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(rung in ("rung0", "omt1", "omt2", "omt3", "omt4"),
            f"unsupported rung {rung}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-276 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-276 measurement requires production JIT on CPU")

    card = arm_card(prior._selected_card(rung, deck_root), arm, plant=plant)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    state = (rung0.bridge_entry(card, entry)
             if rung == "rung0" else card.recipe.initial_state)
    steps = tuple(range(1, 9 if rung == "omt1" else 11))
    ladder = omt1._run_ladder(
        card, twin_a, state, "independent", steps=steps)
    require(ladder["row_count"] == (160 if rung == "omt1" else 200),
            "source arm ladder is incomplete")
    return {
        "format": "nemo-testcase-l4-orca2-round276-source-ladder-v1",
        "status": "PASS_R276_SOURCE_LADDER",
        "rung": rung,
        "label": "independent",
        "arm": arm,
        "selected_statements": list(ARMS[arm]),
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "ladder": ladder,
    }


def _census(rows: list[dict]) -> dict[str, object]:
    counts = Counter(row["metrics"]["rms"]["direction"] for row in rows)
    rendered = {
        name: int(counts[name]) for name in ("toward", "away", "equal")
    }
    votes = rendered["toward"] + rendered["away"]
    return {
        **rendered,
        "rms_moving_votes": votes,
        "strict_majority_toward": rendered["toward"] * 2 > votes,
        "strict_majority_away": rendered["away"] * 2 > votes,
    }


def _row(document: dict, target: tuple[int, str, str]) -> dict:
    for row in document["rows"]:
        if (row["kt"], row["checkpoint"], row["field"]) == target:
            return row
    raise GateError(f"missing ladder row {target}")


def classify(control_path: Path, candidate_path: Path, plant: str) -> dict:
    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(control.get("status") == "PASS_R276_SOURCE_LADDER"
            and control.get("arm") == "control", "control is not admitted")
    require(candidate.get("status") == "PASS_R276_SOURCE_LADDER"
            and candidate.get("arm") != "control", "candidate is not admitted")
    require(control["rung"] == candidate["rung"], "rung moved")
    require(control["worktree"]["commit"] == candidate["worktree"]["commit"],
            "control and candidate are not from one committed tree")

    before = copy.deepcopy(control["ladder"])
    after = copy.deepcopy(candidate["ladder"])
    if plant == "exact-loss":
        before["rows"][0]["bit_identical"] = True
        before["rows"][0]["unequal"] = 0
        after["rows"][0]["bit_identical"] = False
        after["rows"][0]["unequal"] = 1
    comparison = decision96._comparison(
        {"independent": before}, {"independent": after})
    moved = comparison["moved_rows"]
    early = [row for row in moved if int(row["kt"]) <= 3]
    late = [row for row in moved if int(row["kt"]) >= 4]
    away = [row for row in moved
            if row["metrics"]["rms"]["direction"] == "away"]
    away_hist = Counter(int(row["kt"]) for row in away)
    away_first: dict[str, int] = {}
    for row in away:
        key = f"{row['checkpoint']}:{row['field']}"
        away_first[key] = min(int(row["kt"]), away_first.get(key, 10**9))

    first_key = after["first_non_bit_checkpoint"]
    require(first_key is not None, "candidate has no first non-bit row")
    first_target = (
        int(first_key["kt"]), first_key["checkpoint"], first_key["field"])
    first_direction = decision96._direction(
        _row(before, first_target), _row(after, first_target), "rms")
    last_kt = max(int(row["kt"]) for row in after["rows"])
    final_target = (last_kt, "stage3", "ssh")
    final_before = _row(before, final_target)
    final_after = _row(after, final_target)
    total_census = _census(moved)
    eligible = (
        bool(total_census["strict_majority_toward"])
        and first_direction in ("toward", "equal")
        and not comparison["bit_identical_losses"]
    )
    return {
        "format": "nemo-testcase-l4-orca2-round276-early-late-census-v1",
        "status": "PASS_R276_EARLY_LATE_CENSUS",
        "rung": candidate["rung"],
        "label": "independent",
        "arm": candidate["arm"],
        "selected_statements": candidate["selected_statements"],
        "worktree_commit": candidate["worktree"]["commit"],
        "row_count": comparison["row_count"],
        "bit_touched_row_count": comparison["moved_row_count"],
        "exact_row_losses": comparison["bit_identical_losses"],
        "rms_direction_census": {
            "all": total_census,
            "early_kt1_3": _census(early),
            "late_kt4_plus": _census(late),
        },
        "away_rows_by_kt": {
            str(kt): int(away_hist[kt]) for kt in sorted(away_hist)
        },
        "first_away_kt_by_checkpoint_field": dict(sorted(away_first.items())),
        "first_away_kt": min(away_hist) if away_hist else None,
        "first_over_bar": {
            "target": list(first_target), "rms_direction": first_direction,
        },
        "final_stage3_ssh_max": {
            "kt": last_kt,
            "before": final_before["max_abs"],
            "after": final_after["max_abs"],
        },
        "decision96_eligible": eligible,
        "late_only_away_majority": (
            not bool(_census(early)["strict_majority_away"])
            and bool(_census(late)["strict_majority_away"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument(
        "--rung", choices=("rung0", "omt1", "omt2", "omt3", "omt4"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=(
        "none", "extra-selector", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list_arms:
        print(json.dumps(ARMS, indent=2, sort_keys=True))
        return 0
    try:
        if args.measure:
            require(all((args.rung, args.deck_root, args.twin_a,
                         args.arm, args.expect_commit)),
                    "measurement arguments are incomplete")
            require(args.plant in ("none", "extra-selector"),
                    "bad measurement plant")
            result = measure(args.rung, args.deck_root, args.twin_a,
                             args.arm, args.expect_commit, args.plant)
        else:
            require(args.classify is not None,
                    "classification inputs are required")
            require(args.plant in ("none", "exact-loss"),
                    "bad classification plant")
            result = classify(*args.classify, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, round274.GateError, round275.GateError,
            decision96.GateError, OSError, KeyError, TypeError, ValueError,
            StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
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
