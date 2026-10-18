#!/usr/bin/env python3
"""Round 313: V4's literal internal-wave mixing and V3's double diffusion under D122.

Arms on rung 10 (the only ORCA2 card running internal-wave mixing), relative
to the round-312 shipped card (``control``):
  IWM_LIT  internal-wave form generic -> nemo_literal, tsdiff off.
  IWM      IWM_LIT + ln_tsdiff = .true. (namelist_cfg:424).
  DDM      literal double diffusion (namelist_cfg:393-395) on the shipped card.
  BOTH     IWM + DDM.
  D122     the card as shipped after this round's selection.
Rung 0 and OMT-1..4 run constant mixing: ``control`` and ``D122`` only.
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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
    nemo_testcase_l4_orca2_round285_g4_d122_gate as r285,
)

GateError = r276.GateError
require = r276.require
# Fields: (iwm.form, iwm.tsdiff, ddm.enabled, ddm.form).
SHIPPED = ("generic", False, False, "hybrid")
OFF = ("generic", False, False, "hybrid")
IWM_LIT = ("nemo_literal", False, False, "hybrid")
IWM = ("nemo_literal", True, False, "hybrid")
DDM = ("generic", False, True, "nemo_literal")
BOTH = ("nemo_literal", True, True, "nemo_literal")
ARM_VALUES = {"IWM_LIT": IWM_LIT, "IWM": IWM, "DDM": DDM, "BOTH": BOTH}
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = {"rung10": ("control", "D122", *ARM_VALUES)}
ARMS.update({rung: ("control", "D122") for rung in RUNGS[:-1]})
# This round's selection; None = not selected (constant-mixing cards).
D122_SELECTED: dict[str, tuple] = {"rung10": BOTH}


def _carried(card) -> tuple:
    vm = card.recipe.model_config.physics.vertical_mixing
    return (vm.iwm.form, bool(vm.iwm.tsdiff), bool(vm.ddm.enabled), vm.ddm.form)


def _base(rung: str) -> tuple:
    return SHIPPED if rung == "rung10" else OFF


def _with(card, values: tuple):
    form, tsdiff, ddm_on, ddm_form = values
    cfg = card.recipe.model_config
    vm = cfg.physics.vertical_mixing
    vm = vm._replace(iwm=vm.iwm._replace(form=form, tsdiff=tsdiff),
                     ddm=vm.ddm._replace(enabled=ddm_on, form=ddm_form))
    cfg = cfg._replace(physics=cfg.physics._replace(vertical_mixing=vm))
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS[rung], f"{rung}: arm {arm} is not defined")
    got = _carried(card)
    want = D122_SELECTED.get(rung)
    if arm == "D122":
        require(got == (want or _base(rung)),
                f"{rung}: shipped card carries {got}, not round 313's selection")
        return card
    if arm == "control" and want is not None and got == want:
        # Control on the selection tree: this round's selection stripped.
        card = _with(card, _base(rung))
        got = _carried(card)
    require(got == _base(rung),
            f"{rung}: control base carries {got}, not round 312's")
    if arm == "control":
        new = _with(card, IWM) if plant == "retained-selector" else card
        require(_carried(new) == _base(rung), f"{rung}: control arm moved the card")
        return new
    new = _with(card, ARM_VALUES[arm])
    require(_carried(new) == ARM_VALUES[arm], f"{rung}: arm {arm} carries {_carried(new)}")
    return new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure")
    parser.add_argument("--rung", choices=RUNGS)
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

            def select(base, *, candidate, rung):
                return selected_card(base, arm=args.measure, rung=rung,
                                     plant=args.plant)

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a, args.measure != "control",
                                      args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(
                format="nemo-testcase-l4-orca2-round313-iwm-ddm-v1",
                arm=args.measure,
                selected_statements=dict(zip(
                    ("iwm_form", "iwm_tsdiff", "ddm_enabled", "ddm_form"),
                    (D122_SELECTED.get(args.rung) or _base(args.rung))
                    if args.measure == "D122"
                    else ARM_VALUES.get(args.measure, _base(args.rung)))))
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        else:
            require(args.decide is not None, "no gate operation selected")
            result = r278.decide(args.decide)
    except (GateError, r276.decision96.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    if args.plant != "none":
        print(f"STATUS PLANT-GREEN: {args.plant} plant did not fire")
        return 3
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0 if result.get("d122_c_and_d", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
