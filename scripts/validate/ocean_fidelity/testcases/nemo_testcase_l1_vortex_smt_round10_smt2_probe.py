#!/usr/bin/env python
"""Round 222 / VORTEX_SMT round 10 -- the SMT-2 rung's NEMO-side instrument.

Three questions, each answered from NEMO's own admitted records and from the
card's own resolved geometry, never from a hand prediction:

1. **Sanity.**  NaN count, max |ssh| and max |u| per recorded step.  Linear
   bottom drag is a momentum SINK, so the ten-step maxima must not grow
   against the SMT-1 rung.

2. **Where does the drag act?**  ``zdfdrg.f90`` np_lin stores
   ``rCd0_bot = rn_Cd0 * ssmask`` and ``rCdU_bot = - rCd0_bot * rn_Uc0``
   at every T point, and ``dynzdf.f90:306`` / ``:473`` put it on the
   tridiagonal diagonal at ``iku = mbku(ji,jj)`` / ``ikv = mbkv(ji,jj)`` --
   the deepest wet U/V level, which over partial cells is
   ``MIN`` of the two adjacent columns' ``mbkt``.  The probe counts the wet
   columns, the wet U and V faces, and prints the bottom-level histogram of
   each, from the card's own resolved partial-cell geometry.

3. **What did the module move on NEMO's side?**  the record-to-record
   difference between the SMT-2 run and the SMT-1 (round 9) run, per field
   per step.  That difference IS namdrg's signature, independent of legoESM.

Usage:
  nemo_testcase_l1_vortex_smt_round10_smt2_probe.py --smt2 <dir> --smt1 <dir>
      [--output probe.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import read_entry  # noqa: E402

CASE = "VORTEX_SMT2_VEC-zps"
# namelist_ref &namdrg_bot:834 rn_Cd0, :835 rn_Uc0.
RN_CD0 = 1.0e-3
RN_UC0 = 0.4


def _bottom_level_map(card):
    """The card's resolved T/U/V bottom levels, in NEMO's own association.

    ``mbku(ji,jj) = MIN( mbkt(ji,jj), mbkt(ji+1,jj) )`` (and likewise in j)
    is the deepest level at which the FACE is wet; a face whose neighbour is
    land has no wet bottom level at all.
    """
    bl = np.asarray(card.recipe.z_coord.bottom_level)
    bl_u = np.minimum(np.roll(bl, 1, axis=1), bl)
    bl_v = np.minimum(bl[:-1, :], bl[1:, :])
    return bl, bl_u, bl_v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt2", type=Path, required=True)
    ap.add_argument("--smt1", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    card = build_nemo_testcase_card(CASE)
    bl, bl_u, bl_v = _bottom_level_map(card)

    out: dict = {
        "case": CASE,
        "rn_Cd0": RN_CD0, "rn_Uc0": RN_UC0,
        "linear_rCdU_bot_m_s": -(RN_CD0 * RN_UC0),
        "smt2_dir": str(args.smt2), "smt1_dir": str(args.smt1),
        "geometry": {
            "wet_columns": int(np.count_nonzero(bl >= 0)),
            "wet_u_faces": int(np.count_nonzero(bl_u >= 0)),
            "wet_v_faces": int(np.count_nonzero(bl_v >= 0)),
            "t_bottom_level_histogram": {
                str(k): int(v) for k, v in sorted(
                    collections.Counter(bl[bl >= 0].ravel().tolist()).items())},
            "u_bottom_level_histogram": {
                str(k): int(v) for k, v in sorted(
                    collections.Counter(
                        bl_u[bl_u >= 0].ravel().tolist()).items())},
            "v_bottom_level_histogram": {
                str(k): int(v) for k, v in sorted(
                    collections.Counter(
                        bl_v[bl_v >= 0].ravel().tolist()).items())},
        },
        "steps": {},
    }
    for kt in range(1, args.steps + 1):
        name = f"oracle_step_entry_kt{kt:08d}.bin"
        a = read_entry(args.smt2 / name, CASE)
        b = read_entry(args.smt1 / name, CASE)
        out["steps"][str(kt)] = {
            "nan": int(sum(int(np.isnan(a[f]).sum())
                           for f in ("T", "S", "u", "v", "ssh"))),
            "max_abs_ssh_m": float(np.max(np.abs(a["ssh"]))),
            "max_abs_u_m_s": float(np.max(np.abs(a["u"]))),
            "max_abs_v_m_s": float(np.max(np.abs(a["v"]))),
            "max_abs_T_K": float(np.max(np.abs(a["T"]))),
            "smt1_max_abs_ssh_m": float(np.max(np.abs(b["ssh"]))),
            "smt1_max_abs_u_m_s": float(np.max(np.abs(b["u"]))),
            "vs_smt1": {f: float(np.max(np.abs(a[f] - b[f])))
                        for f in ("T", "S", "u", "v", "ssh")},
        }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
