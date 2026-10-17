#!/usr/bin/env python3
"""Round 302: Decision-96 census for ORCA2's tracer-only EVD (nn_evdm = 0).

NEMO ``zdf_evd`` replaces the viscosity only under ``IF( nn_evdm == 1 )``
(``ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121-135``); every ORCA2 deck runs
``nn_evdm = 0``.  The landed card carries ``nu_conv = 0``.  The same-tree
control restores the inherited GYRE value ``nu_conv = 100``; nothing else moves.
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
ARM = "orca2_evd_tracer_only"
CARDS = r278.CARDS
CONTROL_NU_CONV = 100.0


def selected_card(card, *, candidate: bool, rung: str, plant: str = "none"):
    """Candidate = the landed card (nu_conv 0); control = nu_conv 100."""
    cfg = card.recipe.model_config
    evd = cfg.physics.convection.enhanced_diffusion
    require((cfg.physics.convection.scheme, evd.K_conv, evd.nu_conv)
            == ("enhanced_diffusion", 100.0, 0.0),
            f"{rung}: landed card is not tracer-only EVD: {evd.nu_conv}")
    if candidate and plant == "none":
        return card
    evd = evd._replace(nu_conv=CONTROL_NU_CONV)
    physics = cfg.physics._replace(convection=cfg.physics.convection._replace(
        enhanced_diffusion=evd))
    return card._replace(recipe=card.recipe._replace(
        model_config=cfg._replace(physics=physics)))


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
    parser.add_argument("--plant", choices=("none", "exact-loss", "control-arm"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")
            candidate = args.measure == "candidate"

            def select(base, *, candidate, rung):
                card = selected_card(base, candidate=candidate, rung=rung,
                                     plant=args.plant)
                if args.plant == "control-arm":
                    require(card is base, "candidate is not the landed card")
                return card

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a, candidate, args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(format="nemo-testcase-l4-orca2-round302-evd-ladder-v1",
                          arm=ARM if candidate else "control",
                          selected_statements=["zdfevd.f90:121 nn_evdm=0"]
                          if candidate else [],
                          inapplicable_forcing_statements=[])
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
