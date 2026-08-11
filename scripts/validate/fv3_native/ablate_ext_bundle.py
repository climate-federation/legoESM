"""A/B ablation: faithful ext bundle vs interim index-copy exchanges.

Controlled comparison — identical protocol to the SB5a characterization
gate (balanced W2, C12 default, dt=600 s, 12+36 steps; same metrics:
edge/interior wind departure, delp departure, mass), the ONLY variable
is ``use_ext_bundle``.  Prior measurements at C12/dt600:

- interim (bundle OFF): edge du12 7.8, saturating ~14; interior 0.62
- rejected position-only vector remap (72d506f51): edge du 22 (basis
  error, superseded by the faithful fv3_native_ext_vector flow)
"""

import argparse

import numpy as np


def run(n: int, ng: int, dt: float, ext_bundle: bool,
        vector_corner: str = "lagrange") -> dict:
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        run_duo_sw,
        w2_six_face_state,
    )

    # oracle_conventions=True (km=1 corpus migration, 2026-08-11):
    # c_sw refuses duogrid on unbounded metrics (fv_arrays.F90:1512).
    # The docstring's prior numbers (edge du12 7.8 / interior 0.62)
    # were measured on the pre-guard plain-conventions lane and are NOT
    # comparable to runs of this script from here on.
    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=ext_bundle,
                                     vector_corner=vector_corner,
                                     oracle_conventions=True)
    states0 = w2_six_face_state(ctx)
    slu = (slice(ng, ng + n), slice(ng, ng + n + 1))
    sld = (slice(ng, ng + n), slice(ng, ng + n))

    def mass(ss):
        return sum(float((ss[t]["delp"][sld]
                          * ctx["gs6"][t]["area"][sld]).sum())
                   for t in range(6))

    def du(ss):
        return max(float(np.abs(ss[t]["u"][slu]
                                - states0[t]["u"][slu]).max())
                   for t in range(6))

    def dui(ss):
        return max(float(np.abs((ss[t]["u"] - states0[t]["u"])
                                [ng + 2:ng + n - 2,
                                 ng + 2:ng + n - 1]).max())
                   for t in range(6))

    def ddelp(ss):
        return max(float(np.abs((ss[t]["delp"][sld]
                                 - states0[t]["delp"][sld])
                                / states0[t]["delp"][sld]).max())
                   for t in range(6))

    m0 = mass(states0)
    s12 = run_duo_sw(ctx, states0, dt=dt, nsteps=12)
    r = {"mass_rel_12": abs(mass(s12) - m0) / abs(m0),
         "du12": du(s12), "dui12": dui(s12), "ddelp12": ddelp(s12)}
    s48 = run_duo_sw(ctx, s12, dt=dt, nsteps=36)
    r.update({"mass_rel_48": abs(mass(s48) - m0) / abs(m0),
              "du48": du(s48), "dui48": dui(s48), "ddelp48": ddelp(s48)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--dt", type=float, default=600.0)
    args = ap.parse_args()

    for label, flag, vcorn in (("interim", False, "lagrange"),
                               ("ext-bundle", True, "lagrange"),
                               ("ext-a2d-corners", True, "a2d")):
        r = run(args.n, args.ng, args.dt, flag, vcorn)
        print(f"[{label}] " + "  ".join(
            f"{k}={v:.4g}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()
