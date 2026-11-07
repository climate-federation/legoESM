#!/usr/bin/env python3
"""ORCA2 round-13 gate: the river runoff's WATER, paired with its heat.

NEMO applies the runoff volume in TWO places and its tracer content in a
THIRD, and the three are different statements:

* the horizontal divergence -- ``phdivn(:,:,1)`` is decreased by
  ``rnf * r1_rho0 /`` the live top-cell thickness, the surface arm taken when
  neither depth option is selected (``sbcrnf.f90:279-283``, called at
  ``divhor.f90:142``);
* the barotropic sea-surface forcing -- ``r1_rho0 * ( emp - rnf )``, the
  subtraction formed BEFORE the density reciprocal multiplies
  (``stp2d.f90:278-281``);
* the tracer content ``rnf_tsc``, deposited at all three Runge-Kutta stages
  (``trasbc.f90:318-328``), which round 12 landed.

The statement this gate binds is the one legoESM had WRONG: the stage-1/2
concentration/dilution term reads ``emp`` ALONE (``trasbc.f90:282-288``).
legoESM's operand was its whole net freshwater, which includes the runoff, so
supplying the runoff's water there deposited a SECOND, nearly identical copy
of the runoff's heat (``rnf*T_top/rho0/h`` against ``MAX(sst,0)*rnf/rho0/h``)
instead of the compensating volume.  That is precisely why round 12's pairing
arm was confounded.

Binding row: the stage-1 tracer source rate the PRODUCTION step's stage helper
consumes must not move when the runoff's water is supplied.  At stage 1 the
live top-cell thickness is the step-entry one in both arms, so the comparison
is controlled; the gate REQUIRES that and prints it.  Stages 2 and 3 legally
move, because the water changes the sea surface, and they are reported as
moved rather than scored.

The divergence half is NOT re-implemented here: legoESM already carries it
(``nemo_transport_wzv_divergence_level``'s ``runoff_mass_flux`` branch) and
already has a gate that scores it against NEMO's own recorded ``ww`` and
``pFw`` (``nemo_testcase_l4_orca2_wzv_gate.py``).  This gate reuses round 12's
production harness for the arms rather than building a second one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm import constants  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

NX_G, NY_G = 180, 148
_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "divergence_call": f"{_PP}/divhor.f90:142",
    "divergence_surface_arm": f"{_PP}/sbcrnf.f90:279-283",
    "sea_surface_forcing": f"{_PP}/stp2d.f90:278-281",
    "dilution_reads_emp_alone": f"{_PP}/trasbc.f90:282-288",
    "tracer_content_deposit": f"{_PP}/trasbc.f90:318-328",
}


class GateError(RuntimeError):
    """A mechanically binding round-13 condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bitwise(candidate, oracle) -> dict[str, object]:
    a = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)).reshape(-1)
    b = np.ascontiguousarray(np.asarray(oracle, dtype=np.float64)).reshape(-1)
    require(a.size > 0 and a.shape == b.shape, "empty or mismatched score")
    unequal = int((a.view(np.uint64) != b.view(np.uint64)).sum())
    return {
        "scored_cells": int(a.size),
        "unequal": unequal,
        "bit_identical": unequal == 0,
        "max_abs": float(np.abs(a - b).max()),
    }


def run_gate(deck_root: Path, root: Path, json_out: Path | None,
             plant: bool = False, kt: int = 1) -> dict[str, object]:
    stamp = worktree_stamp()
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round12_runoff_gate as r12,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    # Three production arms.  "heat" is the runoff's TRACER content (round
    # 12's landing); "water" is its MASS, which reaches the divergence and the
    # sea surface through FreshwaterForcing.runoff.
    rates_hw, tops_hw, card, fields, _entry, s1_hw = r12._stage_sources(
        deck_root, root, kt, with_runoff=True, with_runoff_mass=True)
    rates_h, tops_h, _, _, _, s1_h = r12._stage_sources(
        deck_root, root, kt, with_runoff=True, with_runoff_mass=False)
    _, tops_none, _, _, _, s1_none = r12._stage_sources(
        deck_root, root, kt, with_runoff=False, with_runoff_mass=False)

    rows: dict[str, object] = {}

    # --- CONTROL 0: are BOTH of the water's owners reachable on this card?
    # The claim "legoESM already carries both owners" is only true where the
    # predicates those owners sit behind hold, so read them from the CARD'S
    # OWN resolved config rather than re-deriving them.  The divergence owner
    # (sbcrnf.f90:279-282 via divhor.f90:142) is reached only through the
    # literal WZV arm; the sea-surface owner (stp2d.f90:278-281) only when the
    # freshwater closure is not "none".
    _cfg = card.recipe.model_config
    _wzv = getattr(_cfg, "wzv_call2_evaluation", "generic")
    _closure = getattr(_cfg, "freshwater_closure", "virtual_salt_flux")
    rows["control_both_water_owners_are_reachable_on_this_card"] = {
        "wzv_call2_evaluation": _wzv,
        "freshwater_closure": _closure,
        "divergence_owner_reachable": _wzv == "nemo_literal",
        "sea_surface_owner_reachable": _closure != "none",
        "note": ("read from the production config, which is the same value "
                 "the model branches on; a card resolving 'generic' would "
                 "never reach sbc_rnf_div's transcription at all"),
    }
    print(f"CONTROL wzv_call2_evaluation={_wzv!r} "
          f"freshwater_closure={_closure!r}", file=sys.stderr)
    require(_wzv == "nemo_literal",
            "this card does not select the literal WZV arm, so the runoff "
            "never reaches the horizontal divergence and half the statement "
            "is unreachable")
    require(_closure != "none",
            "this card switches the freshwater closure off, so the runoff "
            "never reaches the sea-surface forcing")

    # --- CONTROL 1: is the stage-1 comparison controlled? -----------------
    # The two arms must divide by the SAME stage-1 thickness, or a moved
    # dilution rate would not be attributable to the operand.
    stage1_tops_equal = bool(np.array_equal(tops_hw[0], tops_h[0]))
    rows["control_stage1_thickness_is_the_same_in_both_arms"] = {
        "equal": stage1_tops_equal,
        "max_abs_difference": float(np.abs(tops_hw[0] - tops_h[0]).max()),
        "note": ("stage 1 divides by the step-entry thickness in both arms, "
                 "so any move in the stage-1 rate is the OPERAND"),
    }
    print("CONTROL stage-1 thickness identical in both arms: "
          f"{stage1_tops_equal}", file=sys.stderr)
    require(stage1_tops_equal,
            "the stage-1 live top thickness moved when the runoff water was "
            "withheld; the stage-1 comparison is not controlled")

    # --- CONTROL 2: the water must not be inert ---------------------------
    # If supplying it changed nothing anywhere, the bitwise row below would
    # pass for the wrong reason.
    stage2_moved = int(np.count_nonzero(tops_hw[1] != tops_h[1]))
    rows["control_the_water_is_not_inert"] = {
        "stage2_top_thickness_cells_moved": stage2_moved,
        "stage2_top_thickness_max_abs_move": float(
            np.abs(tops_hw[1] - tops_h[1]).max()),
        "note": ("the runoff water reaches the sea surface through "
                 "stp2d.f90:278-281 and the divergence through "
                 "sbcrnf.f90:279-283, so the stage-2 live thickness MUST "
                 "move; a zero here would make the binding row vacuous"),
    }
    print(f"CONTROL the water moves the stage-2 thickness on {stage2_moved} "
          "cells", file=sys.stderr)
    require(stage2_moved > 0,
            "supplying the runoff water moved nothing: the divergence and "
            "sea-surface owners are not reached, so the binding row below "
            "would be vacuous")

    # --- BINDING: trasbc.f90:282-288 dilutes with emp ALONE ---------------
    for tracer, name in ((0, "temperature"), (1, "salinity")):
        candidate = rates_hw[0][tracer]
        if plant and tracer == 0:
            candidate = r12._plant_one_value(candidate)
        rows[f"stage1_rate_is_unmoved_by_the_water_{name}"] = _bitwise(
            candidate, rates_h[0][tracer])

    # Stages 2 and 3 legally move: the water changed the sea surface, so they
    # divide by a different thickness.  Reported, never scored as a bar.
    for stage in (1, 2):
        rows[f"stage{stage + 1}_rate_legally_moves"] = {
            "temperature_cells_moved": int(np.count_nonzero(
                rates_hw[stage][0] != rates_h[stage][0])),
            "salinity_cells_moved": int(np.count_nonzero(
                rates_hw[stage][1] != rates_h[stage][1])),
            "why": ("the water changed the sea surface, so this stage's live "
                    "thickness is a different number -- not a bar"),
        }

    # --- MEASUREMENT: the two spellings of the density reciprocal ---------
    # stp2d.f90:281 multiplies by r1_rho0; legoESM's freshwater_eta_tendency
    # divides by rho_0.  Whether that is a real difference on this card is a
    # measurement, not an opinion.
    rho0 = float(card.recipe.model_config.rho_0)
    emp = np.asarray(fields["emp"], dtype=np.float64)
    rnf = np.asarray(fields["rnf"], dtype=np.float64)
    nemo_spelling = (1.0 / rho0) * (emp - rnf)
    lego_spelling = (-emp + rnf) / rho0
    rows["density_reciprocal_spelling"] = {
        "cells_where_the_two_spellings_differ": int(np.count_nonzero(
            nemo_spelling != -lego_spelling)),
        "of_cells": int(emp.size),
        "max_abs_difference": float(np.abs(nemo_spelling + lego_spelling).max()),
        "note": ("NEMO multiplies by r1_rho0 (stp2d.f90:281); legoESM's "
                 "freshwater_eta_tendency divides by rho_0.  REPORTED, not "
                 "landed -- changing it would move every card that carries "
                 "a surface freshwater flux"),
        "rho_0": rho0,
    }

    # --- THE DISCRIMINATOR ------------------------------------------------
    # Round 12: the heat alone moved the river-mouth cells ten times further
    # from NEMO.  Its explanation was that NEMO pairs the heat with the water.
    # With the water now paired through the RIGHT channel, this row decides it.
    oracle_s1 = ladder.read_state_frame(
        root / f"oracle_stage_kt{kt:08d}_s1.bin", kt=kt, stage=1)["T"]
    ny1, nx1, nz1 = oracle_s1.shape
    mouth = rnf[:ny1, :nx1] != 0.0
    for label, block in (("neither_heat_nor_water", s1_none),
                         ("heat_only_round12", s1_h),
                         ("heat_and_water_round13", s1_hw)):
        delta = np.abs(block[:ny1, :nx1, :nz1] - oracle_s1)
        rows[f"stage1_temperature_{label}"] = {
            "whole_field_max_abs": float(delta.max()),
            "river_mouth_top_cell_max_abs": float(delta[..., 0][mouth].max()),
            "river_mouth_top_cells": int(mouth.sum()),
        }

    improved = (rows["stage1_temperature_heat_and_water_round13"]
                ["river_mouth_top_cell_max_abs"])
    heat_only = (rows["stage1_temperature_heat_only_round12"]
                 ["river_mouth_top_cell_max_abs"])
    rows["discriminator"] = {
        "round12_hypothesis": ("the heat-only landing is ten times worse at "
                               "the river mouths because NEMO pairs that heat "
                               "with the runoff's water volume"),
        "preregistered_confirm_threshold_degC": 3.0e-04,
        "heat_only_river_mouth_max_degC": heat_only,
        "heat_and_water_river_mouth_max_degC": improved,
        "verdict": ("CONFIRMED" if improved < 3.0e-04
                    else "REFUTED" if improved >= 1.0e-03
                    else "PARTIAL"),
    }

    bar = [name for name, row in rows.items()
           if name.startswith("stage1_rate_is_unmoved")
           and not row["bit_identical"]]
    if bar:
        raise GateError(
            "the stage-1 tracer dilution rate MOVED when the runoff's water "
            "was supplied, so legoESM's operand is not NEMO's emp "
            f"(trasbc.f90:282-288): {bar} "
            + json.dumps({k: rows[k] for k in bar}, sort_keys=True))

    result = {
        "gate": "nemo_testcase_l4_orca2_round13_runoff_water_gate",
        "kt": kt,
        "status": "AT_BAR",
        "label": f"given NEMO's entry (kt={kt} recorded state and frames)",
        "citations": CITATIONS,
        "resolved": {
            "ln_rnf": "T (run ocean.output:534)",
            "ln_closea": "F (run ocean.output:118) -- no closed-sea runoff "
                         "redistribution runs",
            "nn_fwb": "2 (run ocean.output:532,1652) -- a scalar correction "
                      "to emp; it never touches rnf",
            "ln_rnf_depth": "F -- the surface arm of sbc_rnf_div runs",
        },
        "constants": {"rho_ocean_nemo": float(constants.rho_ocean_nemo)},
        "provenance": stamp,
        "rows": rows,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--kt", type=int, default=1)
    parser.add_argument("--plant", action="store_true",
                        help="move the production rate by one representable "
                             "value; the gate MUST refuse")
    args = parser.parse_args()
    try:
        result = run_gate(args.deck_root, args.record_root, args.json_out,
                          plant=args.plant, kt=args.kt)
    except GateError as exc:
        if args.plant:
            print("PLANT FIRED: the gate refuses a one-representable-value "
                  f"move -- {exc}")
            return 1
        print(f"FAIL: {exc}")
        return 2
    if args.plant:
        print("PLANT DID NOT FIRE: the gate accepted a moved rate, so it is "
              "vacuous")
        return 3
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
