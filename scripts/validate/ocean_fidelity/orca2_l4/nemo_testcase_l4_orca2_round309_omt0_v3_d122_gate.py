#!/usr/bin/env python3
"""Round 309: OMT-0 under round 308's set, and rung V3's statements under D122.

OMT-0 (independent label, its admitted round-203 kt 1..10 frame record):
  control  round 308's eight selections off (after-SSH 'rk3_extrapolated').
  D122     the card as shipped (round 308's eight on).
  NOADV    D122 minus the two statements NEMO's np_LIN_dyn never reaches
           (vector Krhs/KEG order, dyn_zad association).
  LINDYN   only the statements whose cited compiled line np_LIN_dyn runs:
           wzv's stored r3 ratios (sshwzv.f90:297-298, reached from
           stp2d.f90:156 and the flux-form stage call stprk3_stg.f90:334)
           and the carried after-SSH slot (stprk3.f90:237).
Seven D122 pairs (rung 0, OMT-1..4, rung 10 both labels):
  control  V3's selectors off (the shipped card before this round).
  EVDSE    zdf_evd's trigger on the step-entry rn2/rn2b.
  DDM      NEMO-literal zdf_ddm (rung 10 only: the only deck with ln_zdfddm).
  V3       EVDSE + DDM (rung 10 only).
  D122     the card as shipped after this round's selection.
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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
    nemo_testcase_l4_orca2_round285_g4_d122_gate as r285,
)

GateError = r276.GateError
require = r276.require

# Round 308's landed set (its gate's SEL) and the pre-308 values.
R308_ON = {
    "rk3_vector_rhs_nemo_order": True, "een_vorticity_written_order": True,
    "een_nemo_top_f_u_halo": True, "nemo_wzv_stored_r3_ratios": True,
    "nemo_zad_written_association": True,
    "nemo_stage2_momentum_wzv_half_step": True,
    "nemo_stage_momentum_wzv_stored_kmm_ratios": True,
    "nemo_first_wzv_after_ssh": "rk3_extrapolated_carried",
}
R308_OFF = {k: (False if v is True else "rk3_extrapolated")
            for k, v in R308_ON.items()}
NOT_ON_LIN_DYN = ("rk3_vector_rhs_nemo_order", "nemo_zad_written_association")
LIN_DYN_RUNS = ("nemo_wzv_stored_r3_ratios", "nemo_first_wzv_after_ssh")
OMT0_ARMS = {"control": R308_OFF, "D122": R308_ON,
             "NOADV": {**R308_ON, **{k: R308_OFF[k] for k in NOT_ON_LIN_DYN}},
             "LINDYN": {**R308_OFF, **{k: R308_ON[k] for k in LIN_DYN_RUNS}}}

# ORCA2_OMIP_L4/EXP00 namelist_cfg &namzdf (the rung-10 deck): ln_zdfddm,
# rn_avts, rn_hsbfr.  Rung 0 and OMT-0..4 set ln_zdfddm = .false.
DDM_DECK = {"rn_avts": 1.0e-4, "rn_hsbfr": 1.6}
V3_STATEMENTS = {"EVDSE": ("evd",), "DDM": ("ddm",), "V3": ("evd", "ddm")}
RUNGS = ("omt0", "rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = {rung: ("control", "D122", "EVDSE") for rung in RUNGS[1:]}
ARMS["rung10"] = ("control", "D122", *V3_STATEMENTS)
ARMS["omt0"] = tuple(OMT0_ARMS)
# This round's selection, per card (filled in after the measurements).
D122_SELECTED: dict[str, tuple[str, ...]] = {rung: () for rung in RUNGS[1:]}


def _v3_on(cfg) -> set[str]:
    ph = cfg.physics
    on = set()
    if ph.convection.enhanced_diffusion.trigger_step_entry_bn2:
        on.add("evd")
    ddm = ph.vertical_mixing.ddm
    if ddm.enabled or ddm.form != "hybrid":
        on.add("ddm")
    return on


def _with_v3(cfg, want: set[str]):
    ph = cfg.physics
    ed = ph.convection.enhanced_diffusion._replace(
        trigger_step_entry_bn2="evd" in want)
    ddm = ph.vertical_mixing.ddm._replace(
        enabled="ddm" in want, form="nemo_literal" if "ddm" in want else "hybrid")
    if "ddm" in want:
        require({k: getattr(ddm, k) for k in DDM_DECK} == DDM_DECK,
                "card's rn_avts/rn_hsbfr are not the deck's")
    return cfg._replace(physics=ph._replace(
        convection=ph.convection._replace(enhanced_diffusion=ed),
        vertical_mixing=ph.vertical_mixing._replace(ddm=ddm)))


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(rung in ARMS and arm in ARMS[rung],
            f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    if rung == "omt0":
        shipped = {k: getattr(cfg, k) for k in R308_ON}
        require(shipped == R308_ON, f"omt0: shipped card carries {shipped}")
        want = dict(OMT0_ARMS[arm])
        if plant == "retained-selector" and arm == "control":
            want["een_vorticity_written_order"] = True
        new = cfg._replace(**want)
        got = {k: getattr(new, k) for k in R308_ON}
        require(got == OMT0_ARMS[arm], f"omt0: arm {arm} carries {got}")
    else:
        if arm == "D122":
            on = _v3_on(cfg)
            require(on == {s for n in D122_SELECTED[rung]
                           for s in V3_STATEMENTS[n]},
                    f"{rung}: shipped card carries {sorted(on)}, "
                    "not round 309's selection")
            return card
        want = set(V3_STATEMENTS.get(arm, ()))
        if plant == "retained-selector" and arm == "control":
            want = {"evd"}
        new = _with_v3(cfg, want)
        require(_v3_on(new) == set(V3_STATEMENTS.get(arm, ())),
                f"{rung}: arm {arm} carries {sorted(_v3_on(new))}")
        if arm == "control":
            require(new == cfg, f"{rung}: control arm moved the card")
    return card._replace(recipe=card.recipe._replace(model_config=new))


def measure_omt0(arm: str, deck_root: Path, twin_a: Path,
                 expect_commit: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-309 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-309 measurement requires production JIT on CPU")
    base = omt0.build_omt0_card(deck_root)
    omt0.validate_omt0_card(base)
    card = selected_card(base, arm=arm, rung="omt0")
    ladder = omt1._run_ladder(card, twin_a, card.recipe.initial_state,
                              "independent", steps=tuple(range(1, 11)))
    require(ladder["row_count"] == 200, "OMT-0 ladder is incomplete")
    return {
        "format": "nemo-testcase-l4-orca2-round309-omt0-v1",
        "status": "PASS_R276_SOURCE_LADDER",
        "rung": "omt0", "label": "independent", "arm": arm,
        "selected_statements": OMT0_ARMS[arm],
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "ladder": ladder,
    }


def decide_one(census_path: Path) -> dict:
    """D122 (c)+(d) on one census (OMT-0 is outside D122's seven pairs)."""

    d = json.loads(census_path.read_text(encoding="utf-8"))
    require(d.get("status") == "PASS_R276_EARLY_LATE_CENSUS",
            "census is not admitted")
    c = d["rms_direction_census"]["all"]
    first = d["first_over_bar"]["rms_direction"]
    ok = not d["exact_row_losses"] and first in ("toward", "equal")
    return {"status": "PASS_R309_ONE_PAIR" if ok else "FAIL_R309_ONE_PAIR",
            "rung": d["rung"], "label": d["label"], "arm": d["arm"],
            "exact_rows_lost": len(d["exact_row_losses"]),
            "first_over_bar": d["first_over_bar"],
            "census_all": [c["toward"], c["away"], c["equal"]],
            "final_stage3_ssh_max": d["final_stage3_ssh_max"],
            "d122_c_and_d": ok}


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
    parser.add_argument("--decide-one", type=Path)
    parser.add_argument("--plant", choices=("none", "exact-loss",
                                             "retained-selector"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")
            if args.rung == "omt0":
                require(args.label == "independent",
                        "OMT-0 is measured on the independent label")
                if args.plant == "retained-selector":
                    selected_card(omt0.build_omt0_card(args.deck_root),
                                  arm=args.measure, rung="omt0",
                                  plant=args.plant)
                result = measure_omt0(args.measure, args.deck_root,
                                      args.twin_a, args.expect_commit)
            else:
                def select(base, *, candidate, rung):
                    return selected_card(base, arm=args.measure, rung=rung,
                                         plant=args.plant)

                original, r285.selected_card = r285.selected_card, select
                try:
                    result = r285.measure(args.rung, args.label,
                                          args.deck_root, args.twin_a,
                                          args.measure != "control",
                                          args.expect_commit)
                finally:
                    r285.selected_card = original
                result.update(
                    format="nemo-testcase-l4-orca2-round309-v3-v1",
                    arm=args.measure,
                    selected_statements=list(
                        D122_SELECTED[args.rung] if args.measure == "D122"
                        else V3_STATEMENTS.get(args.measure, ())))
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        elif args.decide_one:
            result = decide_one(args.decide_one)
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
