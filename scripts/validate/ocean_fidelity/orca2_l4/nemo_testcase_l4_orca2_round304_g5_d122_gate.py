#!/usr/bin/env python3
"""Round 304: Decision-122 census for the G5-certified tracer-surface statements.

Candidate = the landed card: stage-1/2 stored Kmm divisor + EMP dilution
(``trasbc.f90:284-286``, r3t(Kaa) ``stprk3_stg.f90:177``) on every card, and the
stage-3 recorded-sfx salt Krhs (``trasbc.f90:308``) on rung 10 only (rung 0 and
OMT-1..4 have no surface forcing).  Control = both selectors off.  G5's
ice-ocean top drag is not imported: ORCA2's ln_lin bottom drag shares pCdU.
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
STAGE12 = "rk3_stage2_tracer_nemo_stored_thickness"
STAGE3_SALT = "rk3_stage3_salt_nemo_written"
ALL = (STAGE12, STAGE3_SALT)
ARM = "g5_tracer_surface_set"
CARDS = r278.CARDS


def expected(rung: str) -> set[str]:
    return {STAGE12, STAGE3_SALT} if rung == "rung10" else {STAGE12}


def selected_card(card, *, candidate: bool, rung: str, plant: str = "none"):
    cfg = card.recipe.model_config
    landed = {name for name in ALL if getattr(cfg, name)}
    require(landed == expected(rung),
            f"{rung}: landed G5 scope moved: {sorted(landed)}")
    if candidate:
        return card
    values = {name: False for name in ALL}
    if plant == "retained-selector":
        values[STAGE12] = True
    cfg = cfg._replace(**values)
    require(not any(getattr(cfg, name) for name in ALL),
            f"{rung}: control retains a G5 selector")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


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
            candidate = args.measure == "candidate"

            def select(base, *, candidate, rung):
                return selected_card(base, candidate=candidate, rung=rung,
                                     plant=args.plant)

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a, candidate, args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(
                format="nemo-testcase-l4-orca2-round304-g5-ladder-v1",
                arm=ARM if candidate else "control",
                selected_statements=sorted(expected(args.rung))
                if candidate else [],
                inapplicable_forcing_statements=[]
                if args.rung == "rung10" else [STAGE3_SALT])
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        else:
            require(args.decide is not None, "no gate operation selected")
            result = r278.decide(args.decide)
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
