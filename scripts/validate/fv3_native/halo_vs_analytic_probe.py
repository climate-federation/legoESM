#!/usr/bin/env python
"""Do the duo entry exchanges reproduce the analytic field in the halo?

WHY THIS EXISTS.  At C48/npz=5 on the J&W (test_case = -13) IC, one 240 s
acoustic sub-step puts a spurious 3.34 m/s increment on the D winds whose
every cell above 10% of peak sits within 3 cells of a panel boundary
(measured: 90 of 90, 74 of 74, ... 298 of 298 across the six faces),
growing linearly to 26 m/s over one dt_atmos against an oracle tendency
of 0.03-0.2 m/s.  "Edge-localised" narrows the cause to the halo
treatment but does not name it.

THE MEASUREMENT.  The oracle initialises only the compute window
(``test_cases.F90`` case(-13) loops ``i=is,ie`` / ``j=js,je``) and fills
the halos with ``ext_scalar`` / ``ext_vector`` / ``fill_corners``, so the
port's window-only IC plus exchanges is the FAITHFUL construction and
"fill the halos analytically instead" is not a candidate fix.  But the
J&W base state is smooth, so a CORRECT exchange must land close to the
analytic value evaluated at the halo lattice point -- close in the sense
of the k2e Lagrange remap's own truncation error, not to machine
precision.  A gross departure means the exchange is wrong.

WHAT THIS CAN AND CANNOT CONCLUDE.  Agreement does NOT prove the exchange
correct (both could be wrong the same way, and the analytic value at a
kinked halo node is not what the oracle's k2e remap targets either).
Disagreement at a level far above the remap's truncation error DOES
localise the defect to the exchange.  So this is a one-way test and it is
reported as one.

The scalars (delp, pt) and the D winds (u, v) are reported separately
because they go through different machinery -- ``ext_scalar`` on the A
lattice versus ``ext_vector`` on the D lattice -- and a defect in one
says nothing about the other.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

N, NG, KM = 48, 3, 5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--ng", type=int, default=NG)
    args = ap.parse_args(argv)

    from legoesm.core.fv3_native_acoustic_3d import exchange_state_halos_3d
    from legoesm.core.fv3_native_dcmip16_bc import GFS_CONSTANTS
    from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_face
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.core.fv3_native_state_3d import build_state_3d
    from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

    def gcdr(p1, p2, r):
        return _gcd(np.asarray(p1, float), np.asarray(p2, float)) * r

    ctx = build_six_face_duo_context(args.n, args.ng, use_ext_bundle=True,
                                     oracle_conventions=True)
    n, ng = ctx["n"], ctx["ng"]
    ak, bk, ptop, _ = set_eta_analytic(KM)
    m_a = n + 2 * ng

    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    st = build_state_3d(n, ng, KM, remap_follows=True)
    full = []
    for t in range(6):
        gs = ctx["gs6"][t]
        # window IC, exactly as the oracle initialises
        co = np.stack([np.asarray(gs["grid_lon"])[cc, cc],
                       np.asarray(gs["grid_lat"])[cc, cc]], -1)
        ce = np.stack([np.asarray(gs["agrid_lon"])[cs, cs],
                       np.asarray(gs["agrid_lat"])[cs, cs]], -1)
        o = dcmip16_bc_face(co, ce, ak, bk, KM, do_pert=True,
                            constants=GFS_CONSTANTS, great_circle_dist=gcdr)
        st[t]["delp"][cs, cs, :] = o["delp"]
        st[t]["pt"][cs, cs, :] = o["pt"]
        st[t]["u"][cs, cc, :] = o["u"]
        st[t]["v"][cc, cs, :] = o["v"]

        # the SAME analytic IC evaluated on the FULL padded lattice --
        # the reference the exchange is scored against. Not a proposed
        # replacement for the exchange: see the module docstring.
        cof = np.stack([np.asarray(gs["grid_lon"])[:m_a + 1, :m_a + 1],
                        np.asarray(gs["grid_lat"])[:m_a + 1, :m_a + 1]], -1)
        cef = np.stack([np.asarray(gs["agrid_lon"])[:m_a, :m_a],
                        np.asarray(gs["agrid_lat"])[:m_a, :m_a]], -1)
        full.append(dcmip16_bc_face(cof, cef, ak, bk, KM, do_pert=True,
                                    constants=GFS_CONSTANTS,
                                    great_circle_dist=gcdr))

    exchange_state_halos_3d(ctx, st, KM, scalars=True, winds=True)

    print(f"grid n={n} ng={ng} km={KM}\n")
    print("HALO ROWS ONLY (the compute window is identical by "
          "construction and is excluded, or the statistic would be "
          "diluted by 2300 exact cells):")
    for name in ("delp", "pt", "u", "v"):
        print(f"\n  {name}:")
        for t in range(6):
            got = st[t][name]
            want = full[t][name]
            if got.shape != want.shape:
                print(f"    face {t+1}: SHAPE {got.shape} vs {want.shape} "
                      f"-- cannot compare")
                continue
            # THE CORNER DIAGONALS MUST BE SCORED SEPARATELY, and the
            # first version of this probe did not do it. Those blocks
            # have no own-face lattice point, so the gridstruct leaves
            # agrid_lat at a SENTINEL rather than NaN -- and a sentinel
            # is a finite number, so sin(lat) of it lands somewhere in
            # [-1, 1] and the analytic reference returns a perfectly
            # plausible temperature. That is what produced a "37 K halo
            # error on every face" on the first run: an artefact of the
            # reference, not of the exchange. Split the four EDGE bands
            # (where the reference is meaningful) from the four
            # corner-diagonal blocks (where it is not) and report both.
            ma0, ma1 = got.shape[0], got.shape[1]
            if name in ("delp", "pt"):
                wi, wj = cs, cs
            elif name == "u":
                wi, wj = cs, cc
            else:
                wi, wj = cc, cs
            win = np.zeros((ma0, ma1), dtype=bool)
            win[wi, wj] = True
            in_i = np.zeros(ma0, dtype=bool); in_i[wi] = True
            in_j = np.zeros(ma1, dtype=bool); in_j[wj] = True
            edge = (~win) & (in_i[:, None] | in_j[None, :])
            diag = (~win) & ~edge
            for label, mask in (("edge", edge), ("cornerdiag", diag)):
                g = got[mask, :]
                w = want[mask, :]
                fin = np.isfinite(g) & np.isfinite(w)
                dropped = int(np.count_nonzero(~fin))
                if not fin.any():
                    print(f"    face {t+1} {label:10s}: no comparable slot")
                    continue
                scale = max(float(np.abs(w[fin]).max()),
                            float(np.abs(g[fin]).max()))
                d = np.abs(g[fin] - w[fin])
                rel = float(d.max() / scale) if scale else 0.0
                print(f"    face {t+1} {label:10s}: max|d|={d.max():11.5g}  "
                      f"rel={rel:9.3e}  scale={scale:10.5g}  "
                      f"dropped={dropped}  cells={int(mask.sum())}")
    print("\nONE-WAY TEST: a large rel localises the defect to the "
          "exchange; a small rel does NOT certify it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
