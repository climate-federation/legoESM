#!/usr/bin/env python3
"""Round 312: D126 surface-only runoff and D135's V1 TKE forms under D122.

Arms on rung 10 (the only ORCA2 card running TKE and the only one spreading
runoff), relative to the round-311 shipped card (``control``):
  RNF        runoff_depth_spread_m 150 -> 0 (sbc_rnf_init nk_rnf=1).
  ETAU_CORE  nn_etau=1 penetration with nn_htau=1 (zdftke.f90:498-502, :864).
  ETAU_WMASK / ETAU_HTAU / ETAU_EXP  core + one V1 statement.
  ETAU       the five together (V1's values).
  MXLRAW     literal raw length MAX(rmxl_min,SQRT(2*en/zrn2)) (:643-645).
  MXL        MXLRAW + the dummy-bottom e3t(jpk) step (:690-692).
  MXLTERM    the step alone (the code refuses it without MXLRAW).
  LC         literal Langmuir source (:311-378, imlc=mbkt+1 at :354).
  TKE        ETAU + MXL + LC;  ALL  TKE + RNF.
  D122       the card as shipped after this round's selection.
Rung 0 and OMT-1..4 run constant mixing and already resolve runoff 0: they
carry ``control`` and ``D122`` only.
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
# V1's card values (gridladder-vmix 0b9443cef2, _orca2v1_tke_vertical).
ETAU_CORE = {"etau_mode": "below_ml", "etau_htau_mode": "latitude"}
ETAU_ONE = {"ETAU_WMASK": {"nemo_etau_wmask": True},
            "ETAU_HTAU": {"tke_htau_evaluation": "nemo_scalar_libm"},
            "ETAU_EXP": {"tke_etau_exponential_evaluation": "nemo_scalar_libm"}}
ETAU = {**ETAU_CORE, **{k: v for d in ETAU_ONE.values() for k, v in d.items()}}
MXLRAW = {"tke_mxl_raw_evaluation": "nemo_literal"}
MXL = {**MXLRAW, "nemo_mxl_terminal_e3t": True}
LC = {"tke_langmuir_evaluation": "nemo_literal"}
TKE_ARMS = {"ETAU_CORE": ETAU_CORE,
            **{k: {**ETAU_CORE, **v} for k, v in ETAU_ONE.items()},
            "ETAU": ETAU, "MXLRAW": MXLRAW, "MXL": MXL,
            "MXLTERM": {"nemo_mxl_terminal_e3t": True}, "LC": LC,
            "TKE": {**ETAU, **MXL, **LC}}
TKE_ARMS["ALL"] = TKE_ARMS["TKE"]
RNF_ARMS = ("RNF", "ALL")
TKE_FIELDS = tuple(sorted(TKE_ARMS["TKE"]))
# Round 311's shipped rung-10 values of every field an arm moves.
SHIPPED = {"etau_mode": "none", "etau_htau_mode": "constant10m",
           "nemo_etau_wmask": False, "tke_htau_evaluation": "jax_expression",
           "tke_etau_exponential_evaluation": "jax_expression",
           "tke_mxl_raw_evaluation": "factored",
           "nemo_mxl_terminal_e3t": False, "tke_langmuir_evaluation": "vectorized"}
SHIPPED_RUNOFF = 150.0
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = {"rung10": ("control", "D122", "RNF", *TKE_ARMS)}
ARMS.update({rung: ("control", "D122") for rung in RUNGS[:-1]})
# This round's selection (rung 10): D126 + every D135 group that passed.
D122_SELECTED = {"rung10": ({**ETAU, **MXL, **LC}, 0.0)}


def _with(card, tke_values: dict, runoff: float):
    cfg = card.recipe.model_config
    vm = cfg.physics.vertical_mixing
    cfg = cfg._replace(
        runoff_depth_spread_m=runoff,
        physics=cfg.physics._replace(vertical_mixing=vm._replace(
            tke=vm.tke._replace(**tke_values))))
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def _carried(card) -> tuple[dict, float]:
    cfg = card.recipe.model_config
    tke = cfg.physics.vertical_mixing.tke
    require(cfg.runoff_depth_spread_map is None, "runoff map is set")
    return ({f: getattr(tke, f) for f in TKE_FIELDS},
            float(cfg.runoff_depth_spread_m))


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS[rung], f"{rung}: arm {arm} is not defined")
    tke, runoff = _carried(card)
    if arm == "D122":
        want = D122_SELECTED.get(rung)
        if want is None:
            require(tke == SHIPPED and runoff == 0.0,
                    f"{rung}: shipped card carries {tke}, runoff {runoff}")
        else:
            require(tke == {**SHIPPED, **want[0]} and runoff == want[1],
                    f"{rung}: shipped card carries {tke}, runoff {runoff}, "
                    "not round 312's selection")
        return card
    base_runoff = SHIPPED_RUNOFF if rung == "rung10" else 0.0
    selected = D122_SELECTED.get(rung)
    if arm == "control" and selected is not None and (tke, runoff) == (
            {**SHIPPED, **selected[0]}, selected[1]):
        # Control on the selection tree: this round's selection stripped.
        card = _with(card, SHIPPED, base_runoff)
        tke, runoff = _carried(card)
    require(tke == SHIPPED and runoff == base_runoff,
            f"{rung}: control base carries {tke}, runoff {runoff}, not round 311's")
    if arm == "control":
        new = (_with(card, LC, base_runoff) if plant == "retained-selector"
               else card)
        require(_carried(new) == (SHIPPED, base_runoff),
                f"{rung}: control arm moved the card")
        return new
    new = _with(card, TKE_ARMS.get(arm, {}),
                0.0 if arm in RNF_ARMS else base_runoff)
    got = _carried(new)
    require(got == ({**SHIPPED, **TKE_ARMS.get(arm, {})},
                    0.0 if arm in RNF_ARMS else base_runoff),
            f"{rung}: arm {arm} carries {got}")
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
            result.update(format="nemo-testcase-l4-orca2-round312-runoff-tke-v1",
                          arm=args.measure,
                          selected_statements=(
                              D122_SELECTED.get(args.rung, ({}, 0.0))[0]
                              if args.measure == "D122"
                              else TKE_ARMS.get(args.measure, {})),
                          runoff_surface_only=(
                              args.measure in (*RNF_ARMS, "D122")))
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
