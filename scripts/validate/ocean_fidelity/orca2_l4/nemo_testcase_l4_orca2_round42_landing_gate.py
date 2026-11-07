#!/usr/bin/env python3
"""Round 42: run and compare the ten-step ORCA2 ladder after EOS80 HPG repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def run(deck_root: Path, record_root: Path, baseline_path: Path,
        *, plant_at_bar: bool = False,
        saved_arm_path: Path | None = None) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round25_outcome_gate as outcome,
    )

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round-42 ladder is CPU-only")

    baseline = json.loads(baseline_path.read_text())
    require(baseline.get("status") == "LADDER_MEASURED",
            "baseline is not a measured ladder")
    require(len(baseline["candidate_trajectory"]["checkpoints"]) == 40,
            "baseline is not the ten-step/40-checkpoint ladder")

    if saved_arm_path is None:
        _, card = ladder.card_fields(deck_root)
        trajectory = ladder.candidate_trajectory(
            deck_root, record_root, card, max_step=10)
        arm = {
            "status": "LADDER_MEASURED",
            "label": "given NEMO's entry",
            "worktree": worktree_stamp(),
            "card": card.case,
            "unmeasured_features": list(card.unmeasured_features),
            "candidate_trajectory": trajectory,
        }
    else:
        require(plant_at_bar, "--saved-arm is only valid for the plant")
        arm = json.loads(saved_arm_path.read_text())
        trajectory = arm["candidate_trajectory"]
    require(len(trajectory["checkpoints"]) == 40,
            "tip did not complete ten steps/40 checkpoints")
    if plant_at_bar:
        arm = copy.deepcopy(arm)
        planted = arm["candidate_trajectory"]["checkpoints"][0]["rows"]["T"]
        planted.update(bit_identical=False, unequal=1,
                       max_abs=float.fromhex("0x1p-1074"),
                       mean_abs_over_unequal=float.fromhex("0x1p-1074"),
                       first_unequal_index=[0, 0, 0])

    comparison = outcome._compare(baseline, arm, "round42_eos80_rhd")
    plant_fires = bool(comparison["at_bar_rows_left"]) if plant_at_bar else None
    if plant_at_bar:
        require(plant_fires, "AT-BAR plant did not fire")
    else:
        require(not comparison["at_bar_rows_left"],
                f"AT-BAR rows left: {comparison['at_bar_rows_left']}")
        require(comparison["first_non_bit_statement_unchanged"],
                "first non-bit statement moved")

    return {
        **arm,
        "status": "PLANT_FIRED" if plant_at_bar else "LANDED",
        "baseline_commit": baseline["worktree"]["commit"],
        "ladder_comparison": comparison,
        "plant_at_bar": plant_at_bar,
        "plant_fires": plant_fires,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant-at-bar", action="store_true")
    parser.add_argument("--saved-arm", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.baseline,
                     plant_at_bar=args.plant_at_bar,
                     saved_arm_path=args.saved_arm)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 1 if args.plant_at_bar else 0


if __name__ == "__main__":
    raise SystemExit(main())
