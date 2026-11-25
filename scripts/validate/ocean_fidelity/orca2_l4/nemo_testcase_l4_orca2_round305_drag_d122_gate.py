#!/usr/bin/env python3
"""Round 305: Decision-133 drag selection measured under Decision 122.

Arms (same tree, one card per arm):
  control  the card with its pre-Decision-133 drag rows rebuilt.
  D133     the card as shipped: rung 0 and rung 10 on NEMO's ln_lin implicit
           bundle + G7's bottom statements; OMT-2..4 the bottom statements;
           OMT-1 (ln_drg_OFF) unchanged.  G5's top drag stays off.
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
BOTTOM = "nemo_bottom_drag_written_statements"
TOP = "implicit_ice_ocean_top_drag"
CARDS = r278.CARDS
# Decision 133 (user, 2026-10-17): the shipped cards select the drag set.
ARMS = {rung: ("control", "D133") for rung in
        ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")}
D133_BOTTOM = {"rung0", "omt2", "omt3", "omt4", "rung10"}
# The pre-Decision-133 drag rows each card carried (resolved config at
# a6c925444); control rebuilds them so the pair differs in drag only.
PRE_D133_DRAG = {
    "rung10": dict(bottom_drag_scheme="nemo_quadratic", bottom_drag_r=2.5e-4,
                   bottom_drag_bbl_thickness=100.0, bottom_drag_bg_velocity=0.1),
    "rung0": dict(bottom_drag_scheme="legacy", bottom_drag_r=4.0e-4,
                  bottom_drag_bbl_thickness=100.0, bottom_drag_bg_velocity=0.0),
}


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS[rung], f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    require(bool(getattr(cfg, BOTTOM)) == (rung in D133_BOTTOM)
            and not getattr(cfg, TOP),
            f"{rung}: shipped card does not carry Decision 133's selection")
    if arm == "control":
        cfg = cfg._replace(**{BOTTOM: False})
        if rung in PRE_D133_DRAG:
            cfg = cfg._replace(bottom_drag=cfg.bottom_drag._replace(
                **PRE_D133_DRAG[rung]))
        if rung == "rung0":
            cfg = cfg._replace(zdf_drag_in_matrix=False,
                               barotropic_drag_substep=False)
        if plant == "retained-selector":
            cfg = cfg._replace(**{BOTTOM: True})
        require(not getattr(cfg, BOTTOM) and not getattr(cfg, TOP),
                f"{rung}: control retains a round-305 selector")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def _leaves(value, prefix=""):
    """Flatten a card config to {dotted name: repr}; arrays by shape+sha256."""
    import hashlib

    import numpy as np
    if hasattr(value, "_fields"):
        out = {}
        for name in value._fields:
            out.update(_leaves(getattr(value, name), f"{prefix}{name}."))
        return out
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.ascontiguousarray(np.asarray(value))
        return {prefix[:-1]: f"array{arr.shape}:{hashlib.sha256(arr.tobytes()).hexdigest()}"}
    return {prefix[:-1]: repr(value)}


def resolved_configs(roots: dict[str, Path], arm: str = "D133") -> dict:
    """Every ORCA2 card's resolved model config + unmeasured tuple."""
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    )
    cards = {"rung10": rung10.card_fields(roots["rung10"])[1],
             "omt0": omt0.build_omt0_card(roots["rung0"])}
    for rung in ("rung0", "omt1", "omt2", "omt3", "omt4"):
        cards[rung] = prior._selected_card(rung, roots[rung])
    cards = {rung: (card if rung == "omt0" else
                    selected_card(card, arm=arm, rung=rung))
             for rung, card in cards.items()}
    return {rung: dict(_leaves(card.recipe.model_config),
                       unmeasured_features=repr(tuple(card.unmeasured_features)))
            for rung, card in cards.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("control", "D133"))
    parser.add_argument("--rung", choices=tuple(sorted(ARMS)))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    # arm-vs-arm (one variable): the first arm stands in as the control.
    parser.add_argument("--classify-arms", nargs=2, type=Path)
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--plant", choices=("none", "exact-loss",
                                             "retained-selector"),
                        default="none")
    parser.add_argument("--output", type=Path)
    # --resolved-config RUNG=ROOT ...: dump every card's resolved config.
    parser.add_argument("--resolved-config", nargs="+")
    args = parser.parse_args()
    try:
        if args.resolved_config:
            result = dict(resolved_configs(
                {k: Path(v) for k, v in
                 (item.split("=", 1) for item in args.resolved_config)},
                arm=args.measure or "D133"),
                status="RESOLVED_CONFIG")
        elif args.measure:
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
            result.update(format="nemo-testcase-l4-orca2-round305-drag-v1",
                          arm=args.measure, selected_statements=[
                              n for n, on in (
                                  (BOTTOM, args.rung in D133_BOTTOM),
                                  ("ln_lin bundle", args.rung in PRE_D133_DRAG))
                              if on and args.measure == "D133"])
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        elif args.classify_arms:
            base, cand = (json.loads(p.read_text(encoding="utf-8"))
                          for p in args.classify_arms)
            require(cand["arm"] != base["arm"],
                    "--classify-arms compares two different arms")
            result = r276.classify_documents(
                dict(base, arm="control"), cand, plant=args.plant)
            result["baseline_arm"] = base["arm"]
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
