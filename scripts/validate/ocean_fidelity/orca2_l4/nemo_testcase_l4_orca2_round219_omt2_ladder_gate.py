#!/usr/bin/env python3
"""Build and score Decision-109 OMT-2 against its admitted frames."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_vortex_smt_zps_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round218_omt2_frame_record_gate as record_gate,
)

PLANTS = ("none", "card-module", "entry-bit")


class GateError(RuntimeError):
    """OMT-2 is not the admitted linear-drag one-module edge."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _same_value(left, right) -> bool:
    """Structural equality that is unambiguous for array-valued config leaves."""

    if type(left) is not type(right):
        return False
    if hasattr(left, "shape") and hasattr(left, "dtype"):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if dataclasses.is_dataclass(left):
        return all(_same_value(getattr(left, field.name),
                               getattr(right, field.name))
                   for field in dataclasses.fields(left))
    if hasattr(left, "_fields"):
        return all(_same_value(getattr(left, name), getattr(right, name))
                   for name in left._fields)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right))
    return bool(left == right)


def build_omt2_card(deck_root: Path, *, plant: str = "none"):
    """Restore the already-shared SMT-2 linear implicit drag composition."""

    card = omt1.build_omt1_card(deck_root)
    cfg = card.recipe.model_config
    shared = build_vortex_smt_zps_card(
        momentum="vector", rung="smt2").recipe.model_config
    drag = shared.bottom_drag
    if plant == "card-module":
        drag = drag._replace(bottom_drag_scheme="legacy")
    cfg = cfg._replace(
        bottom_drag=drag,
        zdf_drag_in_matrix=shared.zdf_drag_in_matrix,
        zdf_baroclinic_only=shared.zdf_baroclinic_only,
        barotropic_drag_substep=shared.barotropic_drag_substep,
    )
    return card._replace(
        case="ORCA2-OMT2-vector-linear-drag-zps",
        recipe=card.recipe._replace(model_config=cfg),
        unmeasured_features=(),
    )


def validate_omt2_card(deck_root: Path, card) -> dict[str, object]:
    base = omt1.build_omt1_card(deck_root)
    omt1.validate_omt1_card(base)
    before = base.recipe.model_config
    after = card.recipe.model_config

    # Compare every untouched field structurally, including array leaves.
    allowed = {
        "bottom_drag", "zdf_drag_in_matrix", "zdf_baroclinic_only",
        "barotropic_drag_substep",
    }
    for name in before._fields:
        if name not in allowed:
            require(_same_value(getattr(after, name), getattr(before, name)),
                    f"unexpected OMT-1 -> OMT-2 field replacement: {name}")

    bottom_changes = {
        name: {"before": getattr(before.bottom_drag, name),
               "after": getattr(after.bottom_drag, name)}
        for name in before.bottom_drag._fields
        if getattr(before.bottom_drag, name) != getattr(after.bottom_drag, name)
    }
    top_changes = {
        name: {"before": getattr(before, name), "after": getattr(after, name)}
        for name in (
            "zdf_drag_in_matrix", "zdf_baroclinic_only",
            "barotropic_drag_substep",
        )
        if getattr(before, name) != getattr(after, name)
    }
    require(bottom_changes == {
        "bottom_drag_scheme": {"before": "legacy", "after": "nemo_linear"},
    }, f"OMT-2 bottom-drag edge moved: {bottom_changes}")
    require(top_changes == {
        "zdf_drag_in_matrix": {"before": False, "after": True},
        "barotropic_drag_substep": {"before": False, "after": True},
    }, f"OMT-2 composition edge moved: {top_changes}")
    require(after.zdf_baroclinic_only is True,
            "OMT-2 lost the inherited NEMO baroclinic-only split")
    require(after.bottom_drag.bottom_drag_cd0 == 1.0e-3,
            "OMT-2 rn_Cd0 moved")
    require(after.bottom_drag.bottom_drag_uc0 == 0.4,
            "OMT-2 rn_Uc0 moved")
    require(card.case == "ORCA2-OMT2-vector-linear-drag-zps",
            "OMT-2 case identity moved")
    require(tuple(card.unmeasured_features) == (),
            "OMT-2 retains an unmeasured feature")
    return {
        "changed_bottom_drag_fields": bottom_changes,
        "changed_composition_fields": top_changes,
        "resolved_linear_drag": {
            "bottom_drag_scheme": after.bottom_drag.bottom_drag_scheme,
            "rn_Cd0": after.bottom_drag.bottom_drag_cd0,
            "rn_Uc0": after.bottom_drag.bottom_drag_uc0,
            "zdf_drag_in_matrix": after.zdf_drag_in_matrix,
            "zdf_baroclinic_only": after.zdf_baroclinic_only,
            "barotropic_drag_substep": after.barotropic_drag_substep,
        },
    }


def run(deck_root: Path, canonical: Path, calibration: Path, twin_a: Path,
        twin_b: Path, month: Path, *, plant: str = "none",
        atomic_fold_unit=None, claim_label: str = "both") -> dict[str, object]:
    import jax
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant}")
    require(claim_label in ("both", "independent", "given_nemo_entry"),
            f"unknown claim label {claim_label}")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "OMT-2 ladder requires production JIT on CPU")

    admission = record_gate.admit(
        canonical, calibration, twin_a, twin_b, month, "none")
    card = build_omt2_card(deck_root, plant=plant)
    selectors = validate_omt2_card(deck_root, card)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    independent_state = card.recipe.initial_state
    entry_actual = rung0.candidate_fields(independent_state)
    if plant == "entry-bit":
        entry = {name: np.array(value, copy=True) for name, value in entry.items()}
        entry["T"].flat[0] = np.nextafter(entry["T"].flat[0], np.inf)
    entry_identity = rung0.ladder.compare_card_entry_to_masked_record(
        entry_actual, entry)
    require(entry_identity["first_non_bit_field"] is None,
            "OMT-2 independent card entry is not bit-exact after the admitted "
            "dry-temperature signed-zero classification")
    given_state = rung0.bridge_entry(
        card, rung0.assemble_frame(twin_a, 1, 0))

    ladders = {}
    if claim_label in ("both", "independent"):
        ladders["independent"] = omt1._run_ladder(
            card, twin_a, independent_state, "independent",
            atomic_fold_unit=atomic_fold_unit, steps=record_gate.STEPS)
    if claim_label in ("both", "given_nemo_entry"):
        ladders["given_nemo_entry"] = omt1._run_ladder(
            card, twin_a, given_state, "given_nemo_entry",
            atomic_fold_unit=atomic_fold_unit, steps=record_gate.STEPS)
    return {
        "status": "PASS_R219_OMT2_CARD_AND_LADDERS",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_status": admission["status"],
        "card_case": card.case,
        "selectors": selectors,
        "entry_identity": entry_identity,
        **ladders,
        "stability_boundary": admission["month_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--twin-a", type=Path, required=True)
    parser.add_argument("--twin-b", type=Path, required=True)
    parser.add_argument("--month", type=Path, required=True)
    parser.add_argument("--atomic-fold-unit", action="store_true")
    parser.add_argument(
        "--claim-label",
        choices=("both", "independent", "given_nemo_entry"),
        default="both",
    )
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.candidate, args.calibration, args.twin_a,
            args.twin_b, args.month, plant=args.plant,
            atomic_fold_unit=(True if args.atomic_fold_unit else None),
            claim_label=args.claim_label,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt1.GateError, rung0.GateError, record_gate.GateError,
            OSError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
