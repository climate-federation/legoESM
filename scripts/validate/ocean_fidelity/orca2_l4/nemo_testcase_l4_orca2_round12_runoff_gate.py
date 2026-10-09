#!/usr/bin/env python3
"""Bit gate for ORCA2's RIVER-RUNOFF TRACER SOURCE.

Round 11's ladder named this statement as the first disagreement between the
ORCA2 card and NEMO.  NEMO's compiled owner is the river-runoff block of
``tra_sbc_RK3``::

    IF( ln_rnf ) THEN
       DO_2D
          IF( rnf(ji,jj) /= 0._wp ) THEN
             zdep = 1._wp / h_rnf(ji,jj)
             DO jk = 1, nk_rnf(ji,jj)
                pts(ji,jj,jk,jp_tem,Krhs) = pts(...) + rnf_tsc(ji,jj,jp_tem) * zdep
                pts(ji,jj,jk,jp_sal,Krhs) = pts(...) + rnf_tsc(ji,jj,jp_sal) * zdep

(``trasbc.f90:318-326``).  The block sits OUTSIDE ``SELECT CASE( kstg )``
(``trasbc.f90:278``), so it runs at ALL THREE Runge-Kutta stages -- unlike the
EMP/QNS block above it.  The record resolves neither ``ln_rnf_depth`` nor
``ln_rnf_depth_ini`` (neither prints its banner in ``ocean.output``), so
NEMO's surface arm sets ``nk_rnf = 1`` and ``h_rnf`` = the LIVE top-cell
thickness (``sbcrnf.f90:487-488``): the whole content lands in the top cell.

WHAT IS SCORED, and why it is not circular.  The gate runs the PRODUCTION step
twice on the record's own kt state -- once with the recorded runoff content
supplied and once with it withheld -- and reads the per-stage tracer source
rates the production stage helper consumes out of the live-operand trace.  The
compiled statement says the first must equal the second PLUS
``rnf_tsc * (1/h_rnf)``, with ``rnf_tsc`` read from the record and ``h_rnf``
the live top thickness that same stage divided by.  So:

  * row ARITHMETIC -- the production rate with the channel against
    (production rate without) + (the transcribed term).  Bitwise, all three
    stages, temperature and salinity.  This certifies the term's value AND its
    accumulation order.
  * row OPERAND -- legoESM's live top-cell thickness against NEMO's own
    ``e3t_0(1)*(1+r3t*tmask(1))`` on the record's step-entry sea surface.  A
    disagreement here is a DIFFERENT statement and is reported as one rather
    than folded into the row above.
  * row ATTRIBUTION -- how many cells carry a non-zero runoff at all, and what
    the stage-1 temperature disagreement looks like off them.  The ladder's
    hard-coded attribution of its whole stage-1 temperature row to this
    statement is a claim, and this row tests it.

Controls: a one-representable-value plant on the production rate must refuse;
the salinity content must be identically zero in the record (``zrnf_sal = 0``,
``sbcrnf.f90:175,227``) or the reading of the compiled source is wrong; and
feeding the stage-2 thickness to the stage-1 row must refuse.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

NX_G, NY_G, NZ = 180, 148, 30
_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "runoff_on": f"{_PP}/trasbc.f90:318",
    "content_tem": f"{_PP}/sbcrnf.f90:221",
    "content_sal": f"{_PP}/sbcrnf.f90:227",
    "zero_salinity": f"{_PP}/sbcrnf.f90:175",
    "surface_depth_arm": f"{_PP}/sbcrnf.f90:487-488",
    "stage_switch": f"{_PP}/trasbc.f90:278",
    "deposit": f"{_PP}/trasbc.f90:318-326",
    "reciprocal_first": f"{_PP}/trasbc.f90:321",
}


class GateError(RuntimeError):
    """A mechanically binding round-12 condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bitwise_score(candidate, oracle, where):
    a = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)[where])
    b = np.ascontiguousarray(np.asarray(oracle, dtype=np.float64)[where])
    require(a.size > 0, "no scoreable cells")
    unequal = int((a.view(np.uint64) != b.view(np.uint64)).sum())
    delta = np.abs(a - b)
    return {
        "scored_cells": int(a.size),
        "unequal": unequal,
        "bit_identical": unequal == 0,
        "max_abs": float(delta.max()),
    }


def _plant_one_value(a):
    out = np.array(a, dtype=np.float64, copy=True)
    flat = out.reshape(-1)
    idx = int(np.argmax(np.abs(np.nan_to_num(flat))))
    flat[idx] = np.nextafter(flat[idx], np.inf)
    return out


def _stage_sources(deck_root: Path, root: Path, kt: int, *, with_runoff: bool,
                   with_runoff_mass: bool = False):
    """Run the PRODUCTION step and return its per-stage tracer source rates."""
    import jax.numpy as jnp

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )

    _, card = ladder.card_fields(deck_root)
    # Decision 52's entry bridge, exactly as the ladder builds it: the card's
    # own initial state with NEMO's recorded sea surface, and nothing else.
    entry = ladder.assemble_state_fields(root, kt, stage=None)
    state = card.recipe.initial_state
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        eta=state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )
    surface_fields = ladder.assemble_surface_fields(root, kt)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, kt)
    require(getattr(surface, "runoff_tracer_content", None) is not None,
            "the ladder's surface forcing carries no runoff tracer content; "
            "this gate cannot bind on a channel that does not exist")
    if not with_runoff:
        surface = surface._replace(runoff_tracer_content=None)
    # NEMO's sea-surface forcing is r1_rho0*(emp - rnf) (stp2d.f90:278-281)
    # and the same runoff enters the horizontal divergence
    # (sbcrnf.f90:279-283 at divhor.f90:142); legoESM reaches both from
    # FreshwaterForcing.runoff.  Round 13 made the ladder supply it, so this
    # switch must be able to WITHHOLD it as well as add it -- otherwise both
    # arms would be the same run and the paired row would be vacuous.
    if with_runoff_mass:
        freshwater = freshwater._replace(
            runoff=jnp.asarray(surface_fields["rnf"]))
    else:
        freshwater = freshwater._replace(
            runoff=jnp.zeros_like(jnp.asarray(surface_fields["rnf"])))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    trace = model.step(state, dt=card.dt_s,
                       freshwater=freshwater, surface_forcing=surface)
    rates, thicknesses = trace.stage_tracer_sources
    rates = tuple(tuple(np.asarray(r, dtype=np.float64) for r in pair)
                  for pair in rates)
    tops = tuple(np.asarray(h, dtype=np.float64)[..., 0] for h in thicknesses)
    stage1_T = np.asarray(trace.stage_outputs[0][2], dtype=np.float64)
    return rates, tops, card, surface_fields, entry, stage1_T


def run_gate(deck_root: Path, root: Path, json_out: Path | None,
             plant: bool = False, kt: int = 1) -> dict[str, object]:
    stamp = worktree_stamp()
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    rates_on, tops, card, fields, entry, stage1_on = _stage_sources(
        deck_root, root, kt, with_runoff=True)
    rates_off, tops_off, _, _, _, stage1_off = _stage_sources(
        deck_root, root, kt, with_runoff=False)
    _, _, _, _, _, stage1_paired = _stage_sources(
        deck_root, root, kt, with_runoff=True, with_runoff_mass=True)
    for stage in range(3):
        require(np.array_equal(tops[stage], tops_off[stage]),
                f"stage {stage + 1} live top thickness moved when the runoff "
                "channel was withheld; the comparison is not controlled")

    rnf = np.asarray(fields["rnf"], dtype=np.float64)
    rnf_tsc = np.asarray(fields["rnf_tsc"], dtype=np.float64)
    require(rnf.shape == (NY_G, NX_G), f"recorded rnf shape {rnf.shape}")
    require(rnf_tsc.shape == (NY_G, NX_G, 2),
            f"recorded rnf_tsc shape {rnf_tsc.shape}")

    active = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
    wet_top = active[..., 0] > 0.0

    rows: dict[str, object] = {}
    # --- ARITHMETIC: the compiled statement, all three stages -------------
    for stage in range(3):
        zdep = 1.0 / np.maximum(tops[stage], 1.0e-10)
        for tracer, name in ((0, "temperature"), (1, "salinity")):
            deposit = rnf_tsc[..., tracer] * zdep * active[..., 0]
            oracle = np.array(rates_off[stage][tracer], copy=True)
            oracle[..., 0] = oracle[..., 0] + deposit
            candidate = rates_on[stage][tracer]
            if plant and stage == 0 and tracer == 0:
                candidate = _plant_one_value(candidate)
            rows[f"stage{stage + 1}_{name}"] = _bitwise_score(
                candidate, oracle, np.ones_like(oracle, dtype=bool))

    # Can this card's thicknesses tell the two spellings apart at all?  The
    # ARITHMETIC rows above are a reciprocal-first control only if
    # ``content * (1/h)`` and ``content / h`` are different fp64 values
    # somewhere; if they were not, a transcription that divided would pass
    # them vacuously.
    _zdep0 = 1.0 / np.maximum(tops[0], 1.0e-10)
    _mul = rnf_tsc[..., 0] * _zdep0
    _div = rnf_tsc[..., 0] / np.maximum(tops[0], 1.0e-10)
    rows["reciprocal_first_is_discriminable"] = {
        "note": ("cells where multiplying by the reciprocal and dividing give "
                 "different fp64 values; the bitwise rows above can only be a "
                 "reciprocal-first control where this is non-zero"),
        "cells": int(np.count_nonzero(_mul != _div)),
        "max_abs_difference": float(np.abs(_mul - _div).max()),
    }

    # --- CONTROL: the wrong stage's thickness must refuse ------------------
    zdep_wrong = 1.0 / np.maximum(tops[1], 1.0e-10)
    oracle_wrong = np.array(rates_off[0][0], copy=True)
    oracle_wrong[..., 0] = (oracle_wrong[..., 0]
                            + rnf_tsc[..., 0] * zdep_wrong * active[..., 0])
    rows["control_stage2_thickness_in_the_stage1_row"] = _bitwise_score(
        rates_on[0][0], oracle_wrong, np.ones_like(oracle_wrong, dtype=bool))

    # --- OPERAND: legoESM's live top thickness against NEMO's --------------
    mesh = _mesh(root)
    tmask = mesh["tmask"]
    ssmask = np.max(tmask, axis=-1)
    ht0 = np.sum(mesh["e3t_0"] * tmask, axis=-1)
    r1_ht0 = ssmask / (ht0 + 1.0 - ssmask)
    r3t = entry["ssh"] * r1_ht0
    h_rnf_nemo = mesh["e3t_0"][..., 0] * (1.0 + r3t * tmask[..., 0])
    diff = np.abs(tops[0] - h_rnf_nemo)
    rows["runoff_depth_operand"] = {
        "note": ("legoESM's live top-cell thickness at the stage-1 Kmm "
                 "against NEMO's own e3t_0(1)*(1+r3t*tmask(1)) on the "
                 "record's step-entry sea surface"),
        "wet_cells": int(wet_top.sum()),
        "unequal_on_wet": int(np.count_nonzero(
            tops[0][wet_top] != h_rnf_nemo[wet_top])),
        "max_abs_on_wet": float(diff[wet_top].max()),
        "unequal_where_the_runoff_is_nonzero": int(np.count_nonzero(
            tops[0][rnf != 0.0] != h_rnf_nemo[rnf != 0.0])),
    }

    # --- ATTRIBUTION: what this statement can and cannot own ---------------
    nonzero = rnf != 0.0
    rows["attribution"] = {
        "cells_with_nonzero_runoff": int(nonzero.sum()),
        "of_total_surface_cells": int(rnf.size),
        "cells_with_nonzero_runoff_content": int(
            np.count_nonzero(rnf_tsc[..., 0] != 0.0)),
        "max_abs_salinity_content": float(np.abs(rnf_tsc[..., 1]).max()),
        "salinity_content_is_identically_zero": bool(
            np.all(rnf_tsc[..., 1] == 0.0)),
        "max_abs_temperature_content": float(np.abs(rnf_tsc[..., 0]).max()),
    }

    # --- what the statement can and cannot own, measured -------------------
    # The ladder attributes its WHOLE stage-1 temperature row to this
    # statement.  Split that row by whether the cell carries runoff at all.
    oracle_s1 = ladder.read_state_frame(
        root / f"oracle_stage_kt{kt:08d}_s1.bin", kt=kt, stage=1)["T"]
    ny1, nx1, nz1 = oracle_s1.shape
    for name, block in (("with", stage1_on), ("without", stage1_off),
                        ("with_the_mass_paired", stage1_paired)):
        sub = block[:ny1, :nx1, :nz1]
        delta = np.abs(sub - oracle_s1)
        carries = (rnf[:ny1, :nx1] != 0.0)[..., None] & np.ones(
            (1, 1, nz1), dtype=bool)
        top_only = np.zeros((1, 1, nz1), dtype=bool)
        top_only[..., 0] = True
        scored = np.isfinite(delta) & (delta > 0.0)
        rows[f"stage1_temperature_row_{name}_the_channel"] = {
            "unequal_total": int(scored.sum()),
            "unequal_on_a_runoff_column": int((scored & carries).sum()),
            "unequal_off_every_runoff_column": int(
                (scored & ~carries).sum()),
            "unequal_in_the_top_cell_of_a_runoff_column": int(
                (scored & carries & top_only).sum()),
            "max_abs": float(delta[np.isfinite(delta)].max()),
            # Does the landing improve the cells it actually touches?  The
            # whole-field max is set by cells the runoff never reaches, so it
            # cannot answer that; this restriction can.
            "max_abs_in_the_top_cell_of_a_runoff_column": float(
                delta[..., 0][(rnf[:ny1, :nx1] != 0.0)].max()),
        }

    # Does the landing move the TRAJECTORY, not only the rate?  A statement
    # that is bit-exact in the rate and inert in the state would be a landing
    # with no consequence, and that is worth knowing either way.
    _moved = np.abs(stage1_on - stage1_off)
    rows["stage1_state_moved_by_the_channel"] = {
        "cells_moved": int(np.count_nonzero(stage1_on != stage1_off)),
        "max_abs": float(_moved.max()),
        "max_abs_in_the_top_cell": float(_moved[..., 0].max()),
    }

    result = {
        "gate": "nemo_testcase_l4_orca2_round12_runoff_gate",
        "kt": kt,
        "label": f"given NEMO's entry (kt={kt} recorded state)",
        "citations": CITATIONS,
        "resolved": {
            "ln_rnf": "T (run ocean.output:534)",
            "ln_rnf_depth": ("F -- the 'runoffs depth read in a file' banner "
                             "is absent from the run's ocean.output"),
            "ln_rnf_depth_ini": ("F -- the 'depth of runoff computed once' "
                                 "banner is absent"),
            "ln_rnf_tem": "F -- the 'runoffs temperatures' banner is absent",
            "ln_rnf_sal": "F -- the 'runoffs salinities' banner is absent",
            "rn_rfact": "1.0 (run ocean.output:651)",
        },
        "provenance": stamp,
        "rows": rows,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _mesh(root: Path) -> dict[str, np.ndarray]:
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round11_dynldf_operator_gate as ldf,
    )
    return ldf._stitch(root, "mesh_mask_{rank:04d}.nc",
                       ("e3t_0", "tmask"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--kt", type=int, default=1)
    parser.add_argument("--plant", action="store_true",
                        help="move the production rate by one representable "
                             "value; the gate must refuse")
    args = parser.parse_args()
    try:
        result = run_gate(args.deck_root, args.record_root, args.json_out,
                          plant=args.plant, kt=args.kt)
    except (GateError, AttributeError, TypeError, OSError, ValueError,
            struct.error) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    rows = result["rows"]
    at_bar = all(
        rows[f"stage{s}_{n}"]["bit_identical"]
        for s in (1, 2, 3) for n in ("temperature", "salinity"))
    control_fired = not rows[
        "control_stage2_thickness_in_the_stage1_row"]["bit_identical"]
    if args.plant:
        if at_bar:
            print("REFUSE: the plant did NOT fire -- the gate cannot bind",
                  file=sys.stderr)
            return 3
        print("PLANT FIRED: the gate refuses a one-representable-value move",
              file=sys.stderr)
        return 1
    if not control_fired:
        print("REFUSE: the wrong-stage-thickness control did NOT fire, so the "
              "gate does not resolve which thickness the deposit divides by",
              file=sys.stderr)
        return 3
    if not at_bar:
        print("REFUSE: the production runoff source is not the compiled "
              "trasbc river-runoff statement", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
