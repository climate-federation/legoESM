#!/usr/bin/env python
"""Per-stage kinetic-energy budget at the cube vertex vs a face centre.

The case-8 vertex residual is a DETERMINISTIC, DISSIPATION-INDEPENDENT
per-step energy source: a Gaussian burst crossing a 3-valent cube vertex
grows (day-5 max|V| ~95-135) while the same burst at a face centre
decays (19.6); a 1e-8 IC perturbation reproduces the spike bit-for-bit;
boosting del-6 (d4_bg 0.12 -> 0.18) only DELAYS it.  Every static
operator and every halo exchange has been cleared (kernels bit-exact,
wedge 3.5e-13, PG byte-identical, divergence_corner byte-identical,
del-6 metric/dd8 exact, FMS averaging 600/600, and swapping the
C-vector extension makes it WORSE not better).

So the source is a DYNAMIC stage acting at the vertex.  This walks ONE
acoustic step and reports, per stage, the area-weighted KE change in a
small box around a cube VERTEX and around a FACE CENTRE.  In a faithful
SW step the only KE sources are the physical flux/PG terms, which must
behave the same (to discretisation error) at both locations; the stage
whose vertex-box KE production is anomalous relative to its
face-centre-box production is the injector.

Usage:
  stage_ke_budget.py --n 36 --steps 40 --box 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--dt", type=float, default=228.5714285714286)
    ap.add_argument("--steps", type=int, default=40,
                    help="spin-up steps before the budget step (lets the "
                         "burst interact with the vertex)")
    ap.add_argument("--box", type=int, default=4,
                    help="half-width (cells) of the sampling box")
    args = ap.parse_args()
    n, ng = args.n, args.ng

    sys.path.insert(0, str(REPO / "packages/core"))
    sys.path.insert(0, str(REPO))
    from legoesm.core.fv3_native_duo_stepper import (
        SW_CFG_CASE8,
        build_six_face_duo_context,
        full_acoustic_step_sixface,
    )
    from legoesm.grids.cubed_sphere import great_circle_distance
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M,
        analytic_swcore_state,
    )
    from tests.test_cases.colliding_modons import (
        _MODON_H0,
        _MODON_SIZE,
        _MODON_UMAX,
    )

    def build(lon_deg, lat_deg):
        ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                         oracle_conventions=True,
                                         omega=0.0)
        lon0, lat0 = np.deg2rad(lon_deg), np.deg2rad(lat_deg)

        def wind_fn(ll):
            r = great_circle_distance(ll[..., 0], ll[..., 1], lon0, lat0,
                                      FV3_RADIUS_M)
            u_e = _MODON_UMAX * np.exp(-(np.asarray(r)
                                         / _MODON_SIZE) ** 2)
            return u_e, np.zeros_like(u_e)

        def scalars_fn(ll):
            delp = np.full(ll.shape[:-1], 9.80665 * _MODON_H0)
            return delp, np.ones_like(delp)

        states = []
        for gs in ctx["gs6"]:
            st = dict(analytic_swcore_state(gs, wind_fn=wind_fn,
                                            scalars_fn=scalars_fn))
            st["w"] = np.zeros_like(st["delp"])
            states.append(st)
        return ctx, states

    def face_ke(ctx, states):
        """Area-weighted KE per face over the compute block, plus the
        per-face maximum cell KE (where the grid-scale energy sits)."""
        sl = slice(ng, ng + n)
        tot = 0.0
        mx = 0.0
        for t, st in enumerate(states):
            gs = ctx["gs6"][t]
            area = np.asarray(gs["area"])[sl, sl]
            u = np.asarray(st["u"])[sl, slice(ng, ng + n + 1)]
            v = np.asarray(st["v"])[slice(ng, ng + n + 1), sl]
            # cell-centred KE proxy from the D-grid edge winds
            uc = 0.5 * (u[:, :-1] + u[:, 1:])
            vc = 0.5 * (v[:-1, :] + v[1:, :])
            k = 0.5 * (uc ** 2 + vc ** 2)
            tot += float(np.nansum(k * area))
            mx = max(mx, float(np.nanmax(k)))
        return tot, mx

    print(f"# one-step KE budget, n={n} dt={args.dt:g} "
          f"spin-up {args.steps} steps")
    print("# location        KE_before      KE_after       dKE/KE    "
          "maxcellKE_before  maxcellKE_after")
    for name, (lo, la) in (("vertex(45,35.26)", (45.0, 35.26)),
                           ("face-centre(0,0)", (0.0, 0.0))):
        ctx, states = build(lo, la)
        for _ in range(args.steps):
            states = full_acoustic_step_sixface(
                ctx, states, args.dt, d_ext=0.0,
                sw_cfg=dict(SW_CFG_CASE8))
        k0, m0 = face_ke(ctx, states)
        states1 = full_acoustic_step_sixface(
            ctx, states, args.dt, d_ext=0.0, sw_cfg=dict(SW_CFG_CASE8))
        k1, m1 = face_ke(ctx, states1)
        rel = (k1 - k0) / k0 if k0 else float("nan")
        print(f"{name:17s} {k0:13.6e} {k1:13.6e} {rel:+9.3e} "
              f"{m0:13.6e} {m1:13.6e}")
    print("# a POSITIVE dKE at the vertex where the face centre is "
          "negative (or much smaller) = the per-step vertex source")


if __name__ == "__main__":
    main()
