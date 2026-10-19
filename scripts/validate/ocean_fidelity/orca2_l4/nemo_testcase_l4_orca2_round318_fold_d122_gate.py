#!/usr/bin/env python3
"""Round 318: the north-fold set under Decision 122 on the ORCA2 cards.

Every arm applies to all seven card/label pairs (same tree, all relative to
the shipped card):
  control   the card as shipped.
  G9 statements (rung G9, certified lane A round 52):
    VMASK   stage transport V mask after its fold exchange (stprk3_stg :283).
    FOLDX   vn_adv and r3v fold exchanges (dynspg_ts :844, stprk3_stg :169).
    FCTF    FCT fold images incl. the :311 half-row overwrite.
    PIVOT   end-of-stage pivot fold of uu/T/S (stprk3_stg :795-796).
    CLOCKP  stage-1/2 transport wzv dt/ratio pair (stprk3_stg :177, :221).
    KRHS3   stage-3 tracer Krhs order + trazdf content, literal FCT and the
            stored stage-3 thickness divisor (one switch, rounds 49-51).
    TWZV    tracer wzv stored Kmm r3t (stprk3_stg :177-179, :221-223).
    R3UV    stage transport stored r3u/r3v(Kmm) (stprk3_stg :282-283).
    G9      all eight.
  held fold statements:
    FOLDUNIT, RAWV, TOPV  (G6, held in round 308 / Decision 125)
    FOLDR                 (G8 fold rows, Decision 137)
  combinations: G9TOPV, G9FOLD (G9 + the three G6), G9FOLDR, ALL.
  D122      the card as shipped after this round's selection.
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
BT = "barotropic."
STATEMENTS = {
    "VMASK": {"rk3_stage_transport_raw_vmask": True},
    "FOLDX": {"rk3_stage_transport_fold_exchanges": True},
    "FCTF": {"fct_fold_exchange": True},
    "PIVOT": {"rk3_end_of_stage_pivot_fold": True},
    "CLOCKP": {"rk3_stage_wzv_nemo_clock_pair": True},
    "KRHS3": {"rk3_stage3_tracer_krhs_nemo_order": True},
    "TWZV": {"rk3_stage_tracer_wzv_stored_kmm_ratios": True},
    "R3UV": {"rk3_stage_transport_stored_r3uv": True},
}
G9_ARMS = tuple(STATEMENTS)
STATEMENTS["G9"] = {k: v for arm in G9_ARMS for k, v in STATEMENTS[arm].items()}
STATEMENTS.update({
    "FOLDUNIT": {BT + "barotropic_literal_fold_unit_vector_form": True},
    "RAWV": {"rk3_vector_raw_vmask": True},
    "TOPV": {"rk3_fold_exchange_top_v_row": True},
    "FOLDR": {"nemo_ldf_fold_rows": True},
})
STATEMENTS["G9TOPV"] = {**STATEMENTS["G9"], **STATEMENTS["TOPV"]}
STATEMENTS["G9FOLD"] = {**STATEMENTS["G9"], **STATEMENTS["FOLDUNIT"],
                        **STATEMENTS["RAWV"], **STATEMENTS["TOPV"]}
STATEMENTS["G9FOLDR"] = {**STATEMENTS["G9"], **STATEMENTS["FOLDR"]}
STATEMENTS["ALL"] = {**STATEMENTS["G9FOLD"], **STATEMENTS["FOLDR"]}
# Addendum (frozen before arms2): KRHS3 refuses on every card (rung 0/OMT
# keep the forced stage-3 heat/salt forms off; rung 10 runs the diffusive
# BBL), so G9N = G9 without it; and the pairs the singles point to.
STATEMENTS["G9N"] = {k: v for k, v in STATEMENTS["G9"].items()
                     if k not in STATEMENTS["KRHS3"]}
STATEMENTS["VT"] = {**STATEMENTS["VMASK"], **STATEMENTS["TOPV"]}
STATEMENTS["VTX"] = {**STATEMENTS["VT"], **STATEMENTS["FOLDX"]}
STATEMENTS["FP"] = {**STATEMENTS["FCTF"], **STATEMENTS["PIVOT"]}
STATEMENTS["G9NTOPV"] = {**STATEMENTS["G9N"], **STATEMENTS["TOPV"]}
STATEMENTS["G9NFOLD"] = {**STATEMENTS["G9N"], **STATEMENTS["FOLDUNIT"],
                         **STATEMENTS["RAWV"], **STATEMENTS["TOPV"]}
STATEMENTS["G9NFOLDR"] = {**STATEMENTS["G9N"], **STATEMENTS["FOLDR"]}
STATEMENTS["ALLN"] = {**STATEMENTS["G9NFOLD"], **STATEMENTS["FOLDR"]}
# Addendum 2 (frozen before arms3): which members G9NFOLD needs.
FOLD_MEMBERS = ("VMASK", "FOLDX", "FCTF", "PIVOT", "FOLDUNIT", "RAWV", "TOPV")
for _m in FOLD_MEMBERS:
    STATEMENTS["NO" + _m] = {k: v for k, v in STATEMENTS["G9NFOLD"].items()
                             if k not in STATEMENTS[_m]}
STATEMENTS["FOLDSET"] = {k: v for arm in FOLD_MEMBERS
                         for k, v in STATEMENTS[arm].items()}
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = ("control", "D122", *STATEMENTS)
# This round's selection: filled in after the arms (addendum).
D122_SELECTED: dict[str, dict] = {rung: {} for rung in RUNGS}


def _get(cfg, path):
    return (getattr(cfg.barotropic, path[len(BT):]) if path.startswith(BT)
            else getattr(cfg, path))


def _set(cfg, values):
    top = {k: v for k, v in values.items() if not k.startswith(BT)}
    bt = {k[len(BT):]: v for k, v in values.items() if k.startswith(BT)}
    if bt:
        top["barotropic"] = cfg.barotropic._replace(**bt)
    return cfg._replace(**top)


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    if arm == "D122":
        moved = {k: _get(cfg, k) for k in STATEMENTS["ALL"] if _get(cfg, k)}
        require(moved == D122_SELECTED[rung],
                f"{rung}: shipped card carries {moved}, not round 318's")
        return card
    want = {} if arm == "control" else STATEMENTS[arm]
    if plant == "retained-selector" and arm == "control":
        want = STATEMENTS["VMASK"]
    new = _set(cfg, want)
    for key in STATEMENTS["ALL"]:
        expect = want.get(key, _get(cfg, key))
        require(_get(new, key) == expect,
                f"{rung}: arm {arm} carries {key}={_get(new, key)!r}")
    if arm == "control" or plant != "none":
        require(new == cfg, f"{rung}: control arm moved the card")
    return card._replace(recipe=card.recipe._replace(model_config=new))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=ARMS)
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
            result.update(format="nemo-testcase-l4-orca2-round318-fold-v1",
                          arm=args.measure,
                          selected_statements=(
                              D122_SELECTED[args.rung]
                              if args.measure == "D122"
                              else STATEMENTS.get(args.measure, {})))
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
