#!/usr/bin/env python3
"""Rule-12 discharge of the round-32 stage-3 correction on the two TANK cards.

WHY THIS EXISTS.  Round 32 moved legoESM's RK3 stage-3 barotropic correction
from before the implicit solve to after it, which is what
``stprk3_stg.F90:430`` followed by the correction block at ``:439-446`` says.
Rule 12 lets such a change land only if the changed operator is bit-exact
GIVEN NEMO'S OWN INPUTS on every card that EXECUTES it.  Both tanks execute it
-- ``ln_drgimp = T`` and ``ln_dynspg_ts = T`` on each
(``lock_kt1_10/ocean.output:560``, ``:752``;
``overflow_kt1_10/ocean.output:672``, ``:869``) -- and neither has ever been
discharged.  Round 32 ran the oracle-relative TRAJECTORY MOVE gate on them
instead, which is a different claim, and it FAILED on OVERFLOW.

WHAT IT DOES.  Exactly what the GYRE discharge does
(``nemo_testcase_l2_gyre_round32_ordering.py --mode rule12_correction``), on
each tank's own record: drive legoESM's own
``rk3_stage_barotropic_correction`` -- imported from the production module the
step function calls, not a copy -- with

* NEMO's ``puu(:,:,:,Kaa)`` as ``dyn_zdf`` LEAVES it (``uu_Kaa_out``),
* NEMO's ``uu_b(:,:,Kaa)``,
* NEMO's ``e3u_0``, ``umask``, ``hu_0`` and ``r1_hu_0`` from the record
  itself, dumped by the round-33 add-on patch,

and score the result against NEMO's own stage-3 velocity from
``oracle_stage_kt00000001_s3.bin``.

FAIL CLOSED.  Until the acquisition in ``nemo_testcase_l1_tanks_round33_zdf/``
has been run by the user, the record does not exist and this exits non-zero
with the command that creates it.  It never reports a status it could not
measure.

WHAT IT CANNOT SEE, per Rule 2:

* ROUND-34 RETRACTION: the sentence that stood here said the tanks' stage
  record carries ``T``, ``S``, ``u`` and ``ssh`` and NO ``v``, so only the u
  face could be discharged.  That was FALSE.  ``stprk3.F90:326-327`` writes
  ``ts``, ``uu``, ``vv`` and ``ssh``, and the reader's own payload check is
  ``4*count + nx*ny`` for exactly that reason -- it simply skipped the third
  block.  The record carried ``v`` all along, and so does the zdf-matrix
  record's whole v side (``vv_Kaa_out``, ``vv_b_Kaa``, ``vmask``, ``e3v_0``,
  ``hv_0``, ``r1_hv_0``).  BOTH faces are discharged here now.
* the operator DIVIDES by ``hu_0`` where ``stprk3_stg.F90:440`` MULTIPLIES by
  the precomputed ``r1_hu_0``, and ``domain.F90:159`` builds that reciprocal as
  ``ssumask/(hu_0 + 1 - ssumask)`` -- which is not ``1/hu_0`` bit for bit.  The
  record now carries BOTH, so the size of that difference is MEASURED and
  reported next to the number instead of argued about.  It stays an open row
  until the shared operator is changed, which is not this round's business.
* the divisor is NEMO's own ``hu_0``, not a rebuild.  LOCK's ``mesh_mask.nc``
  carries no ``e3*_0`` at all -- under ``key_vco_1d`` the vertical coordinate
  IS the 1-D ladder (``domzgr_substitute.h90:89`` defines ``e3u_0(i,j,k)`` as
  ``e3t_1d(k)``) -- which is why the geometry is dumped from inside dyn_zdf
  rather than read from the mesh file.
* a transposed mask would have the right shape on a square tile and pass; the
  tanks are not square, so a transposed mask raises here.
* the divisor is measured, not argued: the report carries how far NEMO's own
  ``hu_0``/``hv_0`` sits from the column sum of ``e3u_0*umask``, how far
  ``1/h`` sits from NEMO's precomputed reciprocal, and -- the row that
  matters -- how far the operator's DIVIDE sits from NEMO's MULTIPLY on
  NEMO's own column transport.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_round29_zdf_matrix import read_zdf_matrix
from nemo_testcase_phase3_stage_sweep_gate import (
    DIMS,
    GateError,
    _xy,
    _xyz,
    expected_masks,
    read_stage,
    require,
    score,
    sha256,
)

ZDF_MATRIX_RECORD = "oracle_zdf_matrix_kt00000001.bin"
STAGE3_RECORD = "oracle_stage_kt00000001_s3.bin"

CARDS = {
    "LOCK_EXCHANGE": {
        "case": "LOCK_EXCHANGE-zco",
        "root": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
                     "/round33_lock_zdf_matrix"),
        "twin": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3"
                     "/lock_kt1_10"),
        "drgimp_citation": "lock_kt1_10/ocean.output:560",
        "dynspg_ts_citation": "lock_kt1_10/ocean.output:752",
    },
    "OVERFLOW": {
        "case": "OVERFLOW-zps",
        "root": Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
                     "/round33_overflow_zdf_matrix"),
        "twin": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3"
                     "/overflow_kt1_10"),
        "drgimp_citation": "overflow_kt1_10/ocean.output:672",
        "dynspg_ts_citation": "overflow_kt1_10/ocean.output:869",
    },
}


def _absmax(rec: dict, name: str):
    array = rec["arrays"].get(name)
    return (None if array is None
            else float(np.max(np.abs(np.asarray(array, dtype=np.float64)))))


def record_geometry(rec: dict, case: str, face: str = "u") -> dict:
    """NEMO's own reference geometry, from the record, in the scored layout.

    Read rather than rebuilt.  ``e3u_0`` is a PREPROCESSOR MACRO -- ``e3t_1d(k)``
    under ``key_vco_1d`` (``domzgr_substitute.h90:89``) and ``e3u_3d(i,j,k)``
    under ``key_vco_3d`` (``:98``) -- so the instrument materialises it
    elementwise; ``hu_0`` and ``r1_hu_0`` are real arrays
    (``dom_oce.F90:167``).

    The two association rows the round-32 receipt correction registered are
    MEASURED here rather than named: how far the rebuilt column sum sits from
    NEMO's own ``hu_0``, and how far ``1/hu_0`` sits from NEMO's ``r1_hu_0``.
    """
    nx, ny, nz = DIMS[case]
    thickness = f"e3{face}_0"
    mask_name = f"{face}mask"
    depth_name = f"h{face}_0"
    reciprocal = f"r1_h{face}_0"
    velocity = f"{face}{face}_Kaa_out"
    missing = [n for n in (thickness, mask_name, depth_name, reciprocal,
                           velocity)
               if n not in rec["arrays"]]
    require(not missing,
            f"the record is short of {missing}: it was written WITHOUT the "
            "round-33 reference-geometry patch "
            "(nemo_testcase_l1_tanks_round33_zdf/dynzdf_round33_refgeom.patch), "
            "so the divisor could only be rebuilt and this gate refuses to "
            "quote a number it had to invent")

    def _3d(name):
        return _xyz(np.asarray(rec["arrays"][name], dtype=np.float64)
                    .ravel(order="F"), nx, ny, nz)

    def _2d(name):
        return _xy(np.asarray(rec["arrays"][name], dtype=np.float64)
                   .ravel(order="F"), nx, ny)

    e3u_0, umask = _3d(thickness), _3d(mask_name)
    hu_0, r1_hu_0 = _2d(depth_name), _2d(reciprocal)
    h_face = e3u_0 * umask
    wet2d = (umask > 0).any(axis=-1)
    rebuilt = h_face.sum(axis=-1)
    wet = wet2d & (hu_0 > 0.0)
    # THE DIVIDE-VERSUS-MULTIPLY ROW, measured on the quantity that actually
    # rides it: NEMO forms SUM(e3u_0*uu(Kaa)) * r1_hu_0 (stprk3_stg.F90:440)
    # and the operator forms the same sum / hu_0.  Comparing 1/hu_0 with
    # r1_hu_0 would be the WRONG test -- x/h and x*(1/h) differ even when
    # 1/h is the correctly rounded reciprocal -- so the column transport is
    # the operand here.
    transport = np.sum(_3d(velocity) * h_face, axis=-1)
    if wet.any():
        divided = transport[wet] / hu_0[wet]
        multiplied = transport[wet] * r1_hu_0[wet]
        divide_vs_multiply = float(np.max(np.abs(divided - multiplied)))
        reciprocal_gap = float(np.max(np.abs(1.0 / hu_0[wet] - r1_hu_0[wet])))
        rebuilt_gap = float(np.max(np.abs(rebuilt[wet] - hu_0[wet])))
    else:
        divide_vs_multiply = reciprocal_gap = rebuilt_gap = 0.0
    return {
        "h_face": h_face,
        "umask": umask,
        "depth": hu_0,
        "r1_depth": r1_hu_0,
        "wet2d": wet2d.astype(np.float64),
        "face": face,
        "open_association_rows": {
            "face": face,
            "rebuilt_sum_minus_nemo_depth_max_abs": rebuilt_gap,
            "one_over_depth_minus_nemo_reciprocal_max_abs": reciprocal_gap,
            "column_mean_divided_minus_multiplied_max_abs": divide_vs_multiply,
            "note": ("ROUND 36 CLOSED THIS ROW.  The operator now MULTIPLIES "
                     "by NEMO's stored r1_hu_0 (stprk3_stg.f90:522, "
                     "domain.f90:213) instead of dividing by hu_0, so the "
                     "third number below is the difference the discharge used "
                     "to inherit and no longer does.  The first two stay "
                     "measured: they say how far a REBUILT divisor sits from "
                     "NEMO's, which is what a card WITHOUT the "
                     "reference-geometry instrument would have to use."),
        },
    }


def run(card: str, *, plant: bool = False) -> dict:
    from legoesm.ocean.dynamics.barotropic_common import (
        rk3_stage_barotropic_correction,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    spec = CARDS[card]
    case = spec["case"]
    root = spec["root"]
    record = root / ZDF_MATRIX_RECORD
    if not record.is_file():
        raise SystemExit(
            f"UNMEASURED: {record} does not exist.  Create it with\n"
            f"    scripts/validate/ocean_fidelity/testcases/"
            f"nemo_testcase_l1_tanks_round33_zdf/run.sh {card}\n"
            "which builds a config COPY carrying the round-29 WRITE-only "
            "dyn_zdf instrument and runs it.  This gate reports nothing until "
            "then.")
    stage = spec["twin"] / STAGE3_RECORD
    require(stage.is_file(), f"{stage}: the twin's stage-3 record is missing")

    rec = read_zdf_matrix(record)
    nx, ny, nz = DIMS[case]
    require(tuple(rec["header"][k] for k in ("jpi", "jpj", "jpk")) == (nx, ny, nz),
            f"{record}: record dimensions {rec['header']} are not {case}'s")
    require(rec["header"]["kt"] == 1 and rec["header"]["kstg"] == 3,
            f"{record}: not the kt=1 stage-3 record")
    jpkm1 = int(rec["header"]["jpkm1"])

    oracle_stage = read_stage(stage, case, 3)
    masks = expected_masks(build_nemo_testcase_card(case))

    rows, association, inputs, not_applicable = [], {}, [], []
    for face in ("u", "v"):
        # NEMO'S OWN MASK decides whether this face exists on this card -- not
        # our reader, not our recipe -- and it is checked BEFORE the geometry
        # is demanded, because a face NEMO never wets needs no geometry.  Both
        # tanks are 2-D x-z boxes: NEMO's vmask is identically zero and its
        # vv/vv_b are identically zero with it, so there is no v face to
        # discharge.  That is NOT the claim "the record has no v" -- it does
        # -- and it is not UNMEASURED either.  It is NOT APPLICABLE, and a
        # measurement decides it.
        nemo_mask = rec["arrays"].get(f"{face}mask")
        require(nemo_mask is not None,
                f"the record carries no {face}mask, so NEMO's own answer to "
                "whether this face exists cannot be read")
        nemo_wet = int(np.count_nonzero(np.asarray(nemo_mask)))
        card_wet = int(masks[face].sum())
        # NEMO'S MASK ALONE decides.  An `or` here would let OUR recipe's mask
        # declare a face inapplicable on a card NEMO actually wets, and the
        # report would then print a nonzero NEMO wet count next to the words
        # "no wet face at all".  A DISAGREEMENT between the two is a hard
        # error, not a quiet skip -- it is a real finding about the card.
        require(bool(nemo_wet == 0) == bool(card_wet == 0),
                f"NEMO wets {nemo_wet} {face} cells and this card's own wet "
                f"{face}-face mask has {card_wet}: the two disagree about "
                "whether the face exists, which is a card-identity defect and "
                "not something this gate may skip")
        if nemo_wet == 0:
            velocity = _absmax(rec, f"{face}{face}_Kaa_out")
            target = _absmax(rec, f"{face}{face}_b_Kaa")
            # A dry face must also carry a dry velocity.  Requiring it, rather
            # than merely reporting it, is what makes NOT APPLICABLE a
            # measurement instead of an assertion.
            require(velocity == 0.0 and target == 0.0,
                    f"NEMO's {face}mask is empty but its {face}{face}_Kaa_out "
                    f"peaks at {velocity} and {face}{face}_b_Kaa at {target}; "
                    "a face with no wet cell cannot carry a velocity")
            not_applicable.append({
                "face": face,
                "reason": (f"NEMO's own {face}mask in this record has "
                           f"{nemo_wet} wet cells and the card's own wet "
                           f"{face}-face mask has {card_wet}; these tanks are "
                           f"2-D x-z boxes with no wet {face} face at all"),
                "nemo_velocity_max_abs": velocity,
                "nemo_barotropic_target_max_abs": target,
            })
            continue
        geom = record_geometry(rec, case, face)
        nlev = masks[face].shape[-1]
        mask3 = geom["umask"]
        require(not np.any(mask3[..., jpkm1:] != 0.0),
                f"the reference {face} face mask is wet above jpkm1, where "
                "stprk3_stg.F90:443 does not write")
        require(not np.any(mask3[..., nlev:] != 0.0),
                f"the reference {face} face mask is wet above the model's "
                "deepest level, so trimming would drop a scored cell")
        field = _xyz(np.asarray(rec["arrays"][f"{face}{face}_Kaa_out"],
                                dtype=np.float64).ravel(order="F"), nx, ny, nz)
        target = _xy(np.asarray(rec["arrays"][f"{face}{face}_b_Kaa"],
                                dtype=np.float64).ravel(order="F"), nx, ny)
        # ROUND 36: NEMO's OWN r1_hu_0/r1_hv_0 out of the record -- not
        # 1/hu_0, not a floored depth, nothing rebuilt.  The operator now
        # multiplies by it, as stprk3_stg.f90:522-523 writes.
        candidate = np.asarray(rk3_stage_barotropic_correction(
            field, target, geom["h_face"], geom["r1_depth"], mask3))
        if plant:
            candidate = candidate + 1.0
        row = score(f"{case}.kt1.stage3.rule12_correction.{face}",
                    oracle_stage[face][..., :nlev], candidate[..., :nlev],
                    masks[face])
        row["nemo_statement"] = ("stprk3_stg.F90:440,444" if face == "u"
                                 else "stprk3_stg.F90:441,445")
        rows.append(row)
        association[face] = geom["open_association_rows"]
        inputs += [
            f"{ZDF_MATRIX_RECORD}:{face}{face}_Kaa_out",
            f"{ZDF_MATRIX_RECORD}:{face}{face}_b_Kaa",
            f"{ZDF_MATRIX_RECORD}:e3{face}_0 / {face}mask / h{face}_0 / "
            f"r1_h{face}_0",
            f"{STAGE3_RECORD}:{face} (the oracle side)",
        ]
    # ROUND-36 REGISTER.  The multiply form made LOCK_EXCHANGE bit-exact and
    # left OVERFLOW's row BYTE-IDENTICAL to what it was, which refutes the
    # prediction that it would close both.  The owner is named, not guessed:
    # OVERFLOW's column has 101 levels and XLA's reduction of that sum differs
    # from NumPy's -- and from NEMO's ascending-k SUM -- on 5 of 606 columns
    # by 4.44e-16, which is 8.88e-19 through the reciprocal and 3.85e-34 in
    # the corrected velocity.  With the same arithmetic done in NumPy the
    # candidate is bit-identical to NEMO on all 61206 cells, which is why an
    # earlier review measured 0/16900 here: it did not go through XLA.
    # LOCK_EXCHANGE's 21-level column shows 0 of 390.
    # ...and it is stamped on the card it was MEASURED on.  LOCK_EXCHANGE's
    # 21-level column shows 0 of 390 columns split by the reduction, so
    # printing an OPEN reduction row on it would report a defect that card
    # does not have.
    open_rows = [] if card != "OVERFLOW" else [{
        "row": f"{case}.kt1.stage3.rule12_correction.u.column_sum_reduction",
        "owner": ("jnp.sum's reduction order over the level axis, against "
                  "NEMO's SUM at stprk3_stg.f90:522"),
        "boundary": ("only columns deep enough for XLA to split the "
                     "reduction: 5/606 at 101 levels, 0/390 at 21"),
        "status": "OPEN",
        "closed_by": ("an ordered accumulation in the operator, which is a "
                      "separate change from round 36's reciprocal and is not "
                      "made here"),
    }]
    open_rows.append({
        "row": f"{case}.kt1.stage3.rule12_correction.u.mask_placement",
        "owner": ("NEMO writes uu(jk) + zub*umask(jk) "
                  "(stprk3_stg.f90:541-542); the operator writes "
                  "(uu + zub)*stage_mask"),
        "boundary": ("MEASURED on both tanks: 0 of 8190 and 0 of 61206 bits "
                     "differ, because every dry face already carries +0.0.  "
                     "On GYRE it is 349 (u) and 212 (v) cells, ALL of them "
                     "signed zero, max|diff| exactly 0.0"),
        "status": "OPEN",
        "closed_by": "transcribing NEMO's per-level add; not round 36's ask",
    })
    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT", "a planted unit offset still read AT-BAR")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-tanks-round33-zdf-rule12-v2",
        "card": card,
        "case": case,
        "mode": "rule12_correction",
        "claim": ("legoESM's own rk3_stage_barotropic_correction, given "
                  "NEMO's dyn_zdf output, NEMO's uu_b/vv_b(Kaa) and NEMO's own "
                  "e3u_0/e3v_0, hu_0/hv_0 and r1_hu_0/r1_hv_0 read from the "
                  "record rather than rebuilt, reproduces NEMO's own stage-3 "
                  "velocity on BOTH faces of this tank"),
        "operator": ("legoesm.ocean.dynamics.barotropic_common."
                     "rk3_stage_barotropic_correction, imported from the "
                     "production module the step function calls"),
        "why_this_card_is_rule12_relevant": (
            f"NEMO runs dynzdf.F90:150-151 and the correction here: "
            f"ln_drgimp = T at {spec['drgimp_citation']} and ln_dynspg_ts = T "
            f"at {spec['dynspg_ts_citation']}"),
        "inputs_are_nemo": inputs,
        "inputs_reconstructed_not_nemo": [],
        "open_association_rows": association,
        "round36_open_rows": open_rows,
        "unmeasured": [],
        "not_applicable_faces": not_applicable,
        "retracted": [
            "ROUND 34: the v face was reported UNMEASURED here on the claim "
            "that this card's stage record carries no v.  It does -- "
            "stprk3.F90:326-327 writes ts, uu, vv and ssh, and the reader's "
            "own payload check is 4*count + nx*ny.  The reader skipped the "
            "third block; the oracle had provided it.  The reason the v "
            "face is not scored is different and it is MEASURED: NEMO's own "
            "vmask in this record is identically zero and its vv_Kaa_out and "
            "vv_b_Kaa are identically zero with it, because both tanks are "
            "2-D x-z boxes.  NOT APPLICABLE, not UNMEASURED."],
        "record": str(record),
        "record_sha256": sha256(record),
        "stage_record": str(stage),
        "stage_record_sha256": sha256(stage),
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True, choices=tuple(CARDS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.card, plant=args.plant)
    except GateError as exc:
        print(json.dumps({"status": "GATE-ERROR", "detail": str(exc)}, indent=2))
        return 1
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        print(f"{row['status']:<8} {row['name']:<52} "
              f"exact {str(row['exact']):<5} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g}")
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())
