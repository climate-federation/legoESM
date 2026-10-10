#!/usr/bin/env python3
"""Build and score Decision-109 OMT-4 against its admitted frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round222_omt4_frame_record_gate as record_gate,
)

PLANTS = omt3.PLANTS


class GateError(RuntimeError):
    """OMT-4 is not the admitted tracer-advection one-module edge."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def build_omt4_card(deck_root: Path, *, plant: str = "none"):
    """Restore rung 0's selected NEMO FCT tracer-advection module."""
    card = omt3.build_omt3_card(deck_root)
    cfg = card.recipe.model_config
    rung_cfg = rung0.build_rung0_card(deck_root).recipe.model_config
    tracer_advection = rung_cfg.tracer_advection
    if plant == "card-module":
        tracer_advection = "none"
    cfg = cfg._replace(tracer_advection=tracer_advection)
    return card._replace(
        case="ORCA2-OMT4-vector-linear-drag-momentum-ldf-fct-zps",
        recipe=card.recipe._replace(model_config=cfg),
        unmeasured_features=(),
    )


def validate_omt4_card(deck_root: Path, card) -> dict[str, object]:
    base = omt3.build_omt3_card(deck_root)
    omt3.validate_omt3_card(deck_root, base)
    before = base.recipe.model_config
    after = card.recipe.model_config
    for name in before._fields:
        if name != "tracer_advection":
            require(omt3.omt2._same_value(getattr(after, name), getattr(before, name)),
                    f"unexpected OMT-3 -> OMT-4 field replacement: {name}")
    require(before.tracer_advection == "none",
            "OMT-3 tracer-advection selector moved")
    require(after.tracer_advection == "fct2",
            f"OMT-4 tracer-advection edge moved: {after.tracer_advection}")
    require(card.case == "ORCA2-OMT4-vector-linear-drag-momentum-ldf-fct-zps",
            "OMT-4 case identity moved")
    require(tuple(card.unmeasured_features) == (),
            "OMT-4 retains an unmeasured feature")
    return {
        "changed_model_config_fields": {
            "tracer_advection": {"before": "none", "after": "fct2"},
        },
        "resolved_tracer_advection": {
            "scheme": after.tracer_advection,
            "nn_fct_h": 2,
            "nn_fct_v": 2,
        },
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
            "OMT-4 ladder requires production JIT on CPU")

    admission = record_gate.admit(
        canonical, calibration, twin_a, twin_b, month, "none")
    card = build_omt4_card(deck_root, plant=plant)
    selectors = validate_omt4_card(deck_root, card)
    return omt3._score_card(
        card, selectors, admission, twin_a, plant=plant,
        atomic_fold_unit=atomic_fold_unit, claim_label=claim_label,
        status="PASS_R223_OMT4_CARD_AND_LADDERS", card_label="OMT-4")


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
        choices=("both", "independent", "given_nemo_entry"), default="both")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.candidate, args.calibration, args.twin_a,
            args.twin_b, args.month, plant=args.plant,
            atomic_fold_unit=(True if args.atomic_fold_unit else None),
            claim_label=args.claim_label)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt3.GateError, omt3.omt2.GateError,
            omt3.omt1.GateError, rung0.GateError, record_gate.GateError,
            OSError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
