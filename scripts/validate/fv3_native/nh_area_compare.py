#!/usr/bin/env python
"""Does the area the height update actually receives match the reference?

WHY A COMPARISON, AGAIN. The transplant arm built for this refused at its
own first control: the port's ``area`` at the corner-diagonal cell holds
2.55e+10, against an interior median of 3.56e+10 -- a real value, not the
BIG_NUMBER sentinel its premise required. ``nh_exchanged_area6``'s
docstring (fv3_native_dsw_tail_3d.py:312-328) asserts that "the
single-tile gridstruct leaves BIG_NUMBER sentinels in the corner-diagonal
halo cells of ``area``"; that sentence is what the arm was built on and
the measurement says it does not hold. Prose is a pointer, not a fact.

So this asks the question directly and with no premise: at every cell,
does ``ctx["nh_area6"]`` -- the array ``update_dz_d`` is actually handed,
after the port's exchange and its own corner fill -- equal the reference's
``gridstruct%area``? Split by region, corner wedges separate, since the
height error localises one cell in from the panel corners.

NO VERDICT IS PRINTED.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_extchain_oracle import (   # noqa: E402
    derive_face_map,
    load_oracle,
    op_scalar,
)
from compare_dyncore_stages import region_masks   # noqa: E402

N, NG = 48, 3
EXTCHAIN = Path("/burg-archive/glab/users/pg2328/fv3_duo_gaps/"
                "retro_gsmetrics/run_c48")


def main() -> int:
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_dsw_tail_3d import nh_exchanged_area6

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    m_a = N + 2 * NG
    ctx["hs6"] = [np.zeros((m_a, m_a), dtype=np.float64) for _ in range(6)]
    gs6 = ctx["gs6"]
    orcm = load_oracle(EXTCHAIN)
    fmap = derive_face_map(orcm, gs6, N, NG)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"face map is not a bijection: {tiles}")
    floor = max(fmap[t][0] for t in range(6))
    if floor > 1.0e-12:
        raise SystemExit(f"INSTRUMENT CONTROL FAILED: face-map floor "
                         f"{floor:.3e} > 1e-12")
    print(f"face map bijection OK, coordinate floor {floor:.3e}")

    area6 = nh_exchanged_area6(ctx)
    masks = region_masks(m_a, m_a, width=NG)

    # BOTH arrays, so the reading is not ambiguous between "the builder
    # is wrong" and "the port's own extra exchange changes it". The
    # gridstruct's area is what every earlier metric comparison scored;
    # the exchanged copy is what update_dz_d is handed.
    for tag, src in (("gridstruct area (what earlier arms scored)",
                      [gs6[t]["area"] for t in range(6)]),
                     ("area handed to update_dz_d (after the port's "
                      "own exchange + corner fill)", area6)):
        print(f"\n{tag}, vs the reference's own, by region")
        _report(fmap, orcm, src, masks)
    return 0


def _report(fmap, orcm, src, masks):
    print(" face->tile   interior       edge         corner      "
          "n differing (corner)")
    for pf in range(6):
        _d, ot, op = fmap[pf]
        o2 = np.asarray(orcm[ot]["arrays"]["M_AREA"], np.float64)
        p2o = op_scalar(np.asarray(src[pf], np.float64), op)
        if o2.shape != p2o.shape:
            raise SystemExit(f"face {pf + 1}: {p2o.shape} vs {o2.shape}")
        d = np.abs(o2 - p2o)
        # masks are built in port orientation; map them the same way the
        # data is mapped so a transpose cannot move a cell between
        # regions unnoticed.
        mo = {k: op_scalar(v.astype(np.float64), op) > 0.5
              for k, v in masks.items()}
        ncorner = int(((d > 0.0) & mo["corner"]).sum())
        print(f"  {pf + 1}->{ot + 1}      {d[mo['interior']].max():.4e}  "
              f"{d[mo['edge']].max():.4e}  {d[mo['corner']].max():.4e}  "
              f"{ncorner:6d}")
    print("\nA zero column is bitwise agreement on that region. Read the "
          "two tables against each other: a disagreement present in BOTH "
          "is the builder's; one that appears only in the second is "
          "created by the port's own exchange and corner fill.")


if __name__ == "__main__":
    sys.exit(main())
