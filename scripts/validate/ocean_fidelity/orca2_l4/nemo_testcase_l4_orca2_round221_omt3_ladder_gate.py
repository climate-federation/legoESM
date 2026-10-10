#!/usr/bin/env python3
"""Build and score Decision-109 OMT-3 against its admitted frames."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round220_omt3_frame_record_gate as record_gate,
)

PLANTS = ("none", "card-module", "entry-bit")


class GateError(RuntimeError):
    """OMT-3 is not the admitted lateral-momentum-diffusion edge."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def build_omt3_card(deck_root: Path, *, plant: str = "none"):
    """Restore rung 0's selected NEMO momentum-LDF module on OMT-2."""

    card = omt2.build_omt2_card(deck_root)
    cfg = card.recipe.model_config
    rung_cfg = rung0.build_rung0_card(deck_root).recipe.model_config
    viscosity = rung_cfg.lateral_viscosity
    if plant == "card-module":
        viscosity = viscosity._replace(A_h=0.0)
    cfg = cfg._replace(
        lateral_viscosity=viscosity,
        lateral_viscosity_operator=rung_cfg.lateral_viscosity_operator,
        lateral_viscosity_e3_weighting=rung_cfg.lateral_viscosity_e3_weighting,
        lateral_viscosity_coefficient_source=(
            rung_cfg.lateral_viscosity_coefficient_source),
        lateral_side_bc=rung_cfg.lateral_side_bc,
    )
    return card._replace(
        case="ORCA2-OMT3-vector-linear-drag-momentum-ldf-zps",
        recipe=card.recipe._replace(model_config=cfg),
        unmeasured_features=(),
    )


def validate_omt3_card(deck_root: Path, card) -> dict[str, object]:
    base = omt2.build_omt2_card(deck_root)
    omt2.validate_omt2_card(deck_root, base)
    before = base.recipe.model_config
    after = card.recipe.model_config

    allowed = {
        "lateral_viscosity", "lateral_viscosity_operator",
        "lateral_viscosity_e3_weighting",
        "lateral_viscosity_coefficient_source", "lateral_side_bc",
    }
    for name in before._fields:
        if name not in allowed:
            require(omt2._same_value(getattr(after, name), getattr(before, name)),
                    f"unexpected OMT-2 -> OMT-3 field replacement: {name}")

    viscosity_changes = {
        name: {
            "before": getattr(before.lateral_viscosity, name),
            "after": getattr(after.lateral_viscosity, name),
        }
        for name in before.lateral_viscosity._fields
        if getattr(before.lateral_viscosity, name)
        != getattr(after.lateral_viscosity, name)
    }
    require(viscosity_changes == {
        "A_h": {"before": 0.0, "after": 1.0e5},
    }, f"OMT-3 viscosity edge moved: {viscosity_changes}")
    resolved = {
        "A_h": after.lateral_viscosity.A_h,
        "operator": after.lateral_viscosity_operator,
        "e3_weighting": after.lateral_viscosity_e3_weighting,
        "coefficient_source": after.lateral_viscosity_coefficient_source,
        "side_bc": after.lateral_side_bc,
    }
    require(resolved == {
        "A_h": 1.0e5,
        "operator": "nemo_div_curl",
        "e3_weighting": "nemo_e3",
        "coefficient_source": "nemo_ahm_3d_file",
        "side_bc": "free_slip",
    }, f"OMT-3 resolved momentum-LDF selection moved: {resolved}")
    require(card.case == "ORCA2-OMT3-vector-linear-drag-momentum-ldf-zps",
            "OMT-3 case identity moved")
    require(tuple(card.unmeasured_features) == (),
            "OMT-3 retains an unmeasured feature")
    return {
        "changed_lateral_viscosity_fields": viscosity_changes,
        "resolved_momentum_ldf": resolved,
    }


def _score_card(card, selectors: dict[str, object], admission: dict,
                twin_a: Path, *, plant: str, atomic_fold_unit,
                claim_label: str, status: str, card_label: str) -> dict[str, object]:
    """Run the shared independent/given-entry OMT ladder protocol."""
    entry = rung0.assemble_frame(twin_a, 1, 0)
    independent_state = card.recipe.initial_state
    entry_actual = rung0.candidate_fields(independent_state)
    if plant == "entry-bit":
        entry = {name: np.array(value, copy=True) for name, value in entry.items()}
        entry["T"].flat[0] = np.nextafter(entry["T"].flat[0], np.inf)
    entry_identity = rung0.ladder.compare_card_entry_to_masked_record(
        entry_actual, entry)
    require(entry_identity["first_non_bit_field"] is None,
            f"{card_label} independent card entry is not bit-exact after the "
            "admitted dry-temperature signed-zero classification")
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
        "status": status,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_status": admission["status"],
        "card_case": card.case,
        "selectors": selectors,
        "entry_identity": entry_identity,
        **ladders,
        "stability_boundary": admission["month_boundary"],
    }


def run(deck_root: Path, canonical: Path, calibration: Path, twin_a: Path,
        twin_b: Path, month: Path, *, plant: str = "none",
        atomic_fold_unit=None, claim_label: str = "both") -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant}")
    require(claim_label in ("both", "independent", "given_nemo_entry"),
            f"unknown claim label {claim_label}")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "OMT-3 ladder requires production JIT on CPU")

    admission = record_gate.admit(
        canonical, calibration, twin_a, twin_b, month, "none")
    card = build_omt3_card(deck_root, plant=plant)
    selectors = validate_omt3_card(deck_root, card)
    return _score_card(
        card, selectors, admission, twin_a, plant=plant,
        atomic_fold_unit=atomic_fold_unit, claim_label=claim_label,
        status="PASS_R221_OMT3_CARD_AND_LADDERS", card_label="OMT-3")


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
    except (GateError, omt2.GateError, omt1.GateError, rung0.GateError,
            record_gate.GateError, OSError, ValueError) as error:
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
