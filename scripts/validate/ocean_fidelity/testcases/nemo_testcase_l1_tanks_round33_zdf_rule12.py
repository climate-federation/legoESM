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

* the tanks' stage record carries ``T``, ``S``, ``u`` and ``ssh`` and NO ``v``,
  so only the u face is discharged here.  The v face stays UNMEASURED on the
  tanks and is named as such in the report rather than omitted.
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


def record_geometry(rec: dict, case: str) -> dict:
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
    missing = [n for n in ("e3u_0", "umask", "hu_0", "r1_hu_0")
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

    e3u_0, umask = _3d("e3u_0"), _3d("umask")
    hu_0, r1_hu_0 = _2d("hu_0"), _2d("r1_hu_0")
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
    transport = np.sum(_3d("uu_Kaa_out") * h_face, axis=-1)
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
        "wet2d": wet2d.astype(np.float64),
        "open_association_rows": {
            "rebuilt_sum_minus_nemo_hu_0_max_abs": rebuilt_gap,
            "one_over_hu_0_minus_nemo_r1_hu_0_max_abs": reciprocal_gap,
            "column_mean_divided_minus_multiplied_max_abs": divide_vs_multiply,
            "note": ("the operator DIVIDES by depth_ref; NEMO MULTIPLIES by "
                     "r1_hu_0 (stprk3_stg.F90:440, domain.F90:159).  The third "
                     "number is the one that matters -- it is the difference "
                     "the discharge inherits -- and it is measured on NEMO's "
                     "own column transport.  Changing the shared operator to "
                     "multiply is not this round's business, so this stays "
                     "OPEN with its size measured."),
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

    geom = record_geometry(rec, case)
    oracle_stage = read_stage(stage, case, 3)
    masks = expected_masks(build_nemo_testcase_card(case))
    nlev = masks["u"].shape[-1]

    mask3 = geom["umask"]
    require(not np.any(mask3[..., jpkm1:] != 0.0),
            "the reference face mask is wet above jpkm1, where "
            "stprk3_stg.F90:443 does not write")
    require(not np.any(mask3[..., nlev:] != 0.0),
            "the reference face mask is wet above the model's deepest level, "
            "so trimming would drop a scored cell")

    field = _xyz(np.asarray(rec["arrays"]["uu_Kaa_out"],
                            dtype=np.float64).ravel(order="F"), nx, ny, nz)
    target = _xy(np.asarray(rec["arrays"]["uu_b_Kaa"],
                            dtype=np.float64).ravel(order="F"), nx, ny)
    candidate = np.asarray(rk3_stage_barotropic_correction(
        field, target, geom["h_face"], np.maximum(geom["depth"], 1e-10),
        geom["wet2d"], mask3))
    if plant:
        candidate = candidate + 1.0
    row = score(f"{case}.kt1.stage3.rule12_correction.u",
                oracle_stage["u"][..., :nlev], candidate[..., :nlev],
                masks["u"])
    row["nemo_statement"] = "stprk3_stg.F90:440,444-445"
    rows = [row]
    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT", "a planted unit offset still read AT-BAR")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-tanks-round33-zdf-rule12-v1",
        "card": card,
        "case": case,
        "mode": "rule12_correction",
        "claim": ("legoESM's own rk3_stage_barotropic_correction, given "
                  "NEMO's dyn_zdf output, NEMO's uu_b(Kaa) and NEMO's e3u_0 "
                  "with the column depth REBUILT from it, reproduces NEMO's "
                  "own stage-3 velocity on this tank"),
        "operator": ("legoesm.ocean.dynamics.barotropic_common."
                     "rk3_stage_barotropic_correction, imported from the "
                     "production module the step function calls"),
        "why_this_card_is_rule12_relevant": (
            f"NEMO runs dynzdf.F90:150-151 and the correction here: "
            f"ln_drgimp = T at {spec['drgimp_citation']} and ln_dynspg_ts = T "
            f"at {spec['dynspg_ts_citation']}"),
        "inputs_are_nemo": [
            f"{ZDF_MATRIX_RECORD}:uu_Kaa_out",
            f"{ZDF_MATRIX_RECORD}:uu_b_Kaa",
            f"{ZDF_MATRIX_RECORD}:e3u_0 / umask / hu_0 / r1_hu_0",
            f"{STAGE3_RECORD}:u (the oracle side)",
        ],
        "inputs_reconstructed_not_nemo": [],
        "open_association_rows": geom["open_association_rows"],
        "unmeasured": [
            "the v face: this card's stage record carries T, S, u and ssh and "
            "no v, so no v discharge is possible from it"],
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
        print(f"{row['status']:<8} {row['name']:<48} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g}")
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())
