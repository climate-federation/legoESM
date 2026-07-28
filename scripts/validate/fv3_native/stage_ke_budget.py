#!/usr/bin/env python
"""Total-energy conservation budget: cube vertex vs face centre.

The case-8 vertex residual is a DETERMINISTIC, DISSIPATION-INDEPENDENT
per-step energy source: a Gaussian burst crossing a 3-valent cube vertex
grows (day-5 max|V| ~95-135) while the same burst at a face centre
decays (19.6); a 1e-8 IC perturbation reproduces the spike bit-for-bit;
boosting del-6 (d4_bg 0.12 -> 0.18) only DELAYS it.  Every static
operator and every halo exchange has been cleared (kernels bit-exact,
wedge 3.5e-13, PG byte-identical, divergence_corner byte-identical,
del-6 metric/dd8 exact, FMS averaging 600/600, and swapping the
C-vector extension makes it WORSE not better).

So the source is DYNAMIC.  This integrates whole acoustic steps and
reports the change in TOTAL shallow-water energy
    E = sum area * (0.5*h*|V|^2 + 0.5*g*h^2),   h = delp/g
for a burst seeded at a cube VERTEX vs at a FACE CENTRE.  Case-8 is
non-rotating with del-6 as the only sink, so a faithful integration
must give dE <= 0 at BOTH locations; dE > 0 at the vertex is
non-conservative injection.

KE alone is NOT a conservation test (it exchanges with PE), and the
mid-step state cannot be budgeted at all (between the stage block and
one_grad_p the D winds are in circulation form), so this measures whole
steps of total energy only.

Usage:
  stage_ke_budget.py --n 36 --steps 40 --window 20
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
    ap.add_argument("--window", type=int, default=20,
                    help="steps to budget over (whole steps only: the "
                         "mid-step state has circulation-form winds)")
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

    GRAV = 9.80665      # FV3 constants_mod (case-8 delp = grav*h)

    def face_ke(ctx, states):
        """TOTAL shallow-water energy over the compute block.

        KE alone is NOT conserved in SW -- it exchanges with potential
        energy -- so the conservation check must use
            E = sum area * ( 0.5*h*|V|^2 + 0.5*g*h^2 ),  h = delp/g.
        Returns (E_total, KE_total, PE_total, max cell KE).
        """
        sl = slice(ng, ng + n)
        e_tot = ke_tot = pe_tot = 0.0
        mx = 0.0
        for t, st in enumerate(states):
            gs = ctx["gs6"][t]
            area = np.asarray(gs["area"])[sl, sl]
            u = np.asarray(st["u"])[sl, slice(ng, ng + n + 1)]
            v = np.asarray(st["v"])[slice(ng, ng + n + 1), sl]
            h = np.asarray(st["delp"])[sl, sl] / GRAV
            # cell-centred wind proxy from the D-grid edge winds
            uc = 0.5 * (u[:, :-1] + u[:, 1:])
            vc = 0.5 * (v[:-1, :] + v[1:, :])
            k = 0.5 * h * (uc ** 2 + vc ** 2)
            p = 0.5 * GRAV * h ** 2
            ke_tot += float(np.nansum(k * area))
            pe_tot += float(np.nansum(p * area))
            mx = max(mx, float(np.nanmax(0.5 * (uc ** 2 + vc ** 2))))
        e_tot = ke_tot + pe_tot
        return e_tot, ke_tot, pe_tot, mx

    # NOTE: a stage/PG-tail split is NOT possible on this quantity --
    # between the stage block and one_grad_p the D winds are in
    # circulation form (one_grad_p multiplies by rdx/rdy), so mid-step
    # energy is not comparable.  Budget whole steps only.
    print(f"# TOTAL-ENERGY budget over {args.window} steps, n={n} "
          f"dt={args.dt:g}, spin-up {args.steps} steps.  Case-8 is "
          f"non-rotating with del-6 the only sink, so a faithful "
          f"integration must give dE <= 0 (E = KE + PE).")
    print("# location          dE/E           dKE/KE         dPE/PE"
          "         maxcellKE_after")
    for name, (lo, la) in (("vertex(45,35.26)", (45.0, 35.26)),
                           ("face-centre(0,0)", (0.0, 0.0))):
        ctx, states = build(lo, la)
        for _ in range(args.steps):
            states = full_acoustic_step_sixface(
                ctx, states, args.dt, d_ext=0.0,
                sw_cfg=dict(SW_CFG_CASE8))
        e0, k0, p0, _ = face_ke(ctx, states)
        for _ in range(args.window):
            states = full_acoustic_step_sixface(
                ctx, states, args.dt, d_ext=0.0,
                sw_cfg=dict(SW_CFG_CASE8))
        e1, k1, p1, m1 = face_ke(ctx, states)
        print(f"{name:17s} {(e1-e0)/e0:+14.3e} {(k1-k0)/k0:+14.3e} "
              f"{(p1-p0)/p0:+14.3e} {m1:16.6e}")
    print("# dE > 0 at the vertex while the face centre is <= 0 = the "
          "per-step non-conservative vertex source.  dKE alone is NOT "
          "a conservation test (KE exchanges with PE).")


if __name__ == "__main__":
    main()
