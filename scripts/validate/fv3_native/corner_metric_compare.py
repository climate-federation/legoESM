#!/usr/bin/env python
"""Are the port's and the oracle's SENTINEL-REGION metrics the same values?

WHY A COMPARISON AND NOT A TRANSPLANT. The widened transplant arm
(``nh_metric_transplant.py --widen-corners``, job 9454999) found ZERO
cells where the port is sentinel and the oracle is real, and refused
rather than report "unchanged" -- so the transplant instrument CANNOT
reach these cells at all. The reason is that the sentinel population
lives on BOTH sides: that run left 13980 oracle-sentinel cells alone
across the 2-D families. A value that is sentinel-magnitude on both
sides is invisible to a transplant, but it is NOT necessarily the same
number on both sides -- and these cells are read on this lane, because
``del6_vt_flux`` at nord=2 fills its work array over the whole padded
box including the corner diagonals (sw_core.F90:2051-2089) and
``copy_corners`` is skipped on a bounded domain on both sides
(tp_core.F90:139-141/160-162).

So this asks the only question left about them: at the cells one or both
sides mark sentinel, do the two sides hold the SAME value? Split by
corner region, because the NH error this is chasing sits one cell in
from the panel corners.

NO VERDICT IS PRINTED. Counts, magnitudes and locations only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_extchain_oracle import (   # noqa: E402
    derive_face_map,
    load_oracle,
    op_scalar,
)
from compare_gs_metrics import (        # noqa: E402
    FAMILIES,
    SWAP_PARTNER,
    region_masks,
    sent_lo,
    sent_mag,
)

N, NG = 48, 3
EXTCHAIN = "/burg-archive/glab/users/pg2328/fv3_duo_gaps/retro_gsmetrics/run_c48"


def _sent(a, fam):
    a = np.asarray(a, dtype=np.float64)
    m = (np.abs(a) >= sent_mag(fam)) | ~np.isfinite(a)
    lo = sent_lo(fam)
    if lo > 0.0:
        m |= np.abs(a) <= lo
    return m


def main() -> int:
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    gs6 = ctx["gs6"]
    orcm = load_oracle(EXTCHAIN)
    # Signature READ, not remembered: derive_face_map(orc, gs6, n, ng)
    # returns {port_face: (dist, oracle_tile, op)} and does NOT enforce
    # the bijection or the floor -- the caller does, as compare_gs_metrics
    # does at its own :259-268.
    fmap = derive_face_map(orcm, gs6, N, NG)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"face map is not a bijection: {tiles}")
    floor = max(fmap[t][0] for t in range(6))
    if floor > 1.0e-12:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: face-map coordinate floor "
            f"{floor:.3e} > 1e-12; nothing below means anything")
    print(f"face map bijection OK, coordinate floor {floor:.3e}")

    print(f"corner-region metric comparison, C48 ng={NG}")
    print("family      face  sentinel cells  identical  differ  "
          "max|d| among differing   in-corner differ")
    any_differ = False
    for fam in sorted(FAMILIES):
        tag, key, ish, jsh = FAMILIES[fam]
        if key not in gs6[0]:
            continue
        rm = region_masks(N, NG, ish, jsh)
        corner = np.zeros((N + 2 * NG + ish, N + 2 * NG + jsh), dtype=bool)
        for rk, rv in rm.items():
            if rk.endswith("_corner"):
                corner |= rv
        for pf in range(6):
            _d, ot, op = fmap[pf]
            fam_t = SWAP_PARTNER.get(fam, fam) if op[0] else fam
            o2 = np.asarray(orcm[ot]["arrays"][FAMILIES[fam_t][0]],
                            np.float64)
            p2o = op_scalar(np.asarray(gs6[pf][key], np.float64), op)
            if o2.shape != p2o.shape:
                continue
            sent = _sent(o2, fam) | _sent(p2o, fam)
            if not sent.any():
                continue
            same = (o2 == p2o) & sent
            diff = sent & ~same
            corner_o = op_scalar(corner.astype(np.float64), op) > 0.5
            n_corner_diff = int((diff & corner_o).sum())
            if diff.any():
                any_differ = True
                dmax = float(np.abs(o2 - p2o)[diff].max())
            else:
                dmax = 0.0
            print(f"  {fam:10s} {pf + 1}   {int(sent.sum()):6d}  "
                  f"{int(same.sum()):9d}  {int(diff.sum()):6d}  "
                  f"{dmax:22.6e}  {n_corner_diff:6d}")
    print("\nA cell counted 'identical' is bitwise equal on both sides, so "
          "whatever it is, both programs compute with the same number "
          "there. A cell counted 'differ' is a place the two programs "
          "disagree and no transplant can reach, because both sides look "
          "like a sentinel to the mask.")
    if not any_differ:
        print("\nNo sentinel-region cell differs between the two sides in "
              "any family or face.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
