#!/usr/bin/env python3
"""Round-103 record-consistency probe for the compiled TKE matrix/RHS block.

This checks the RECORDS, not legoESM.  It rebuilds NEMO's own
``zd_up``/``zd_lw``/``zdiag`` and right-hand side from NEMO's own recorded
operands, in the compiled source association, and compares them against
NEMO's own recorded outputs.  If that replay is not bit-exact the round's
production rows cannot be interpreted at all, because the reference side of
the comparison would already be wrong.

Compiled source.  Two builds are involved and their compiled ``zdftke.f90``
is IDENTICAL outside the recorder calls (verified: the two files differ in
zero lines once the ``r54_``/``r101_`` call lines are removed), so the same
statements carry different line numbers in each.  Citations below are to the
RECORD PRODUCER of the array being read.

  Round-59 build, which wrote ``matrix_*``/``rhs_pre_sweep``:
  ``GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:250-252`` the three
  zfact scalars; ``:417`` ``zcof``; ``:420-421`` ``zzd_up``; ``:422-423``
  ``zzd_lw``; ``:425-427`` the three matrix writes; ``:430-433`` the right
  hand side; ``:463`` the write call itself.

  Round-101 build, which wrote ``en_after_langmuir``: the same statements at
  ``GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436`` and
  ``:439-442``, with its post-Langmuir callback at ``:395``.

Operands come from the Round-59 record (``matrix_*``, ``sh2``, ``rn2``,
``avm_entry``, ``avt_entry``, ``dissl_entry``, ``e3t_Kmm``, ``e3w_Kmm``,
``tmask``, ``wmask``, ``rn_Dt``, ``rn_ediss``) and the post-Langmuir ``en``
image from the Round-101 record, which the consolidated gate has already
admitted as byte-duplicated against Round 59 on every consumed cell.

The plant arm advances one bit of a single operand and must make the replay
fail, so a green run is not green by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_testcase_l2_gyre_round54_tke_operands import (  # noqa: E402
    read_record as read_operands,
)

R59 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/"
           "oracle_tke_operands/oracle_tke_operands_kt00000002.bin")
R101 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101/"
            "oracle_tke_statement_walk")


def _unequal(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    ne = (np.ascontiguousarray(a).view(np.uint64)
          != np.ascontiguousarray(b).view(np.uint64))
    return int(np.count_nonzero(ne)), float(np.max(np.abs(a - b)))


def rebuild_block(a: dict, en_after_langmuir: np.ndarray,
                  p_sh2_override: np.ndarray | None = None,
                  avm_override: np.ndarray | None = None) -> dict:
    """Rebuild NEMO's four block outputs over jk = 2:jpkm1, source-associated.

    Every operand defaults to the recorded one.  ``p_sh2_override`` is the
    one-variable swap the attribution needs: substituting ONLY the shear
    production and leaving every other operand at NEMO's recorded value.
    """
    dt = np.float64(a["rn_Dt"])
    ediss = np.float64(a["rn_ediss"])
    zfact1 = np.float64(-0.5) * dt                 # R59 zdftke.f90:250
    zfact2 = np.float64(1.5) * dt * ediss          # R59 zdftke.f90:251
    zfact3 = np.float64(0.5) * ediss               # R59 zdftke.f90:252

    avm = np.asarray(a["avm_entry"], dtype=np.float64)
    if avm_override is not None:
        avm = np.asarray(avm_override, dtype=np.float64)
    avt = np.asarray(a["avt_entry"], dtype=np.float64)
    dissl = np.asarray(a["dissl_entry"], dtype=np.float64)
    rn2 = np.asarray(a["rn2"], dtype=np.float64)
    sh2 = np.asarray(a["sh2"], dtype=np.float64)[:, :, 1:30]
    if p_sh2_override is not None:
        sh2 = np.asarray(p_sh2_override, dtype=np.float64)
    e3t = np.asarray(a["e3t_Kmm"], dtype=np.float64)
    e3w = np.asarray(a["e3w_Kmm"], dtype=np.float64)
    tmask = np.asarray(a["tmask"], dtype=np.float64)
    wmask = np.asarray(a["wmask"], dtype=np.float64)

    # NEMO jk = 2..jpkm1 is zero-based index 1..29.
    k, km1, kp1 = slice(1, 30), slice(0, 29), slice(2, 31)
    amin = np.float64(2.0e-5)

    zcof = zfact1 * tmask[:, :, k]                              # :417
    zzd_up = zcof * np.maximum(avm[:, :, kp1] + avm[:, :, k], amin) / (
        e3t[:, :, k] * e3w[:, :, k])                            # :420-421
    zzd_lw = zcof * np.maximum(avm[:, :, k] + avm[:, :, km1], amin) / (
        e3t[:, :, km1] * e3w[:, :, k])                          # :422-423
    zdiag = 1.0 - zzd_lw - zzd_up + zfact2 * dissl[:, :, k] * wmask[:, :, k]

    # en at the right-hand side statement is the post-Langmuir image.
    en = np.asarray(en_after_langmuir, dtype=np.float64)[:, :, k]
    p_avt_rn2 = avt[:, :, k] * rn2[:, :, k]
    zfact3_dissl = zfact3 * dissl[:, :, k]
    dissipation = zfact3_dissl * en
    stratified = sh2 - p_avt_rn2
    parenthesized = stratified + dissipation
    scaled = dt * parenthesized
    increment = scaled * wmask[:, :, k]
    rhs = en + increment                                          # :430-433
    return {
        "zd_up": zzd_up,
        "zd_lw": zzd_lw,
        "zdiag": zdiag,
        "p_avt_rn2": p_avt_rn2,
        "zfact3_dissl": zfact3_dissl,
        "dissipation_product": dissipation,
        "after_stratification": stratified,
        "parenthesized_sum": parenthesized,
        "dt_product": scaled,
        "masked_increment": increment,
        "final_accumulation": rhs,
        "en_rhs": rhs,
    }


def replay(r59: Path, r101: Path, plant: str | None) -> dict:
    from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
        read_admitted_tke_statement_walk,
    )

    a = read_operands(r59)["arrays"]
    langmuir = np.asarray(
        read_admitted_tke_statement_walk(r101)["arrays"]["en_after_langmuir"])

    planted_at = None
    planted_baseline = None
    avm_override = None
    if plant == "operand-ulp":
        # A control that perturbs a zero is not a control: pick a cell the
        # statement is actually sensitive to (wet at jk and jk-1, so neither
        # zcof nor wmask annihilates it) and report the baseline it multiplies.
        avm = np.asarray(a["avm_entry"], dtype=np.float64)
        live = np.argwhere(
            (np.asarray(a["tmask"])[:, :, 1:30] != 0.0)
            & (np.asarray(a["wmask"])[:, :, 1:30] != 0.0)
            & (np.asarray(a["matrix_upper"])[:, :, 1:30] != 0.0))
        if live.size == 0:
            raise SystemExit(
                "REFUSE: no live cell to plant; the record is all land")
        i, j, kk = (int(v) for v in live[0])
        planted_at = (i, j, kk + 1)
        planted_baseline = float(avm[i, j, kk + 1])
        avm_override = avm.copy()
        avm_override[i, j, kk + 1] = np.nextafter(
            avm_override[i, j, kk + 1], np.float64(np.inf))

    rebuilt = rebuild_block(a, langmuir, avm_override=avm_override)
    k = slice(1, 30)
    rows = []
    for name, recorded_key, citation in (
        ("zd_up", "matrix_upper", "R59TKE zdftke.f90:425"),
        ("zd_lw", "matrix_lower", "R59TKE zdftke.f90:426"),
        ("zdiag", "matrix_diag", "R59TKE zdftke.f90:427"),
        ("en_rhs", "rhs_pre_sweep", "R59TKE zdftke.f90:430-433"),
    ):
        recorded = np.asarray(a[recorded_key])[:, :, k]
        n, mx = _unequal(rebuilt[name], recorded)
        rows.append({
            "field": name, "nemo_statement": citation,
            "cells": int(recorded.size), "n_unequal": n, "absolute_max": mx,
            "classification": "BIT" if n == 0 else "NOT-BIT",
        })
    ok = all(row["n_unequal"] == 0 for row in rows)
    return {
        "format": "nemo-testcase-l2-gyre-round103-record-replay-v1",
        "worktree": worktree_stamp(),
        "record_round59": str(r59),
        "record_round101": str(r101),
        "record_write_call": (
            "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:463"),
        "builds_identical_outside_recorder_calls": True,
        "domain": "NEMO levels 2:jpkm1, every owned cell, no mask exception",
        "plant": plant,
        "plant_index": planted_at,
        "plant_baseline_operand": planted_baseline,
        "rows": rows,
        "status": "PASS" if ok else "FAIL",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round59", type=Path, default=R59)
    p.add_argument("--round101", type=Path, default=R101)
    p.add_argument("--plant", choices=("operand-ulp",))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = replay(args.round59, args.round101, args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    if args.plant:
        if report["status"] == "PASS":
            print("REFUSE: the operand plant left the replay green")
            return 2
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
