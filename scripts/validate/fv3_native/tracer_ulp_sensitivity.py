"""One-ulp sensitivity of the tracer transport on a flat plateau.

The terminator cl2 one-step residual against the Fortran oracle (8e-4 of
qcly, 2026-09-14) lives only on the exactly-flat 2e-6 night-side plateau
next to the front, and the port's eager and compiled arms disagree there
by the same amount while sphum and cl are bit-identical between them.
The reading "a limiter branch is decided by rounding noise on the
plateau" is INFERRED from that.  This measures it: perturb the plateau by
ONE ulp in the IC and step once.  A smooth scheme moves the output by
~1 ulp; a branch flipping on the perturbation moves it by the difference
between its two branches.  Same for cl's zero plateau, nudged to +1 ulp
of qcly, to show whether cl matched the oracle only by virtue of exact
zeros.

Usage::

    python tracer_ulp_sensitivity.py [--n 48] [--km 5] [--n-split 8] [--dt 1920]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def plateau_mask(q, value, rtol=1e-12):
    return np.abs(np.asarray(q) - value) <= rtol * value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1920.0)
    ap.add_argument("--assert-envelope", action="store_true",
                    help="fail unless cl2's one-ulp response lies within "
                         "[0.5, 1.2] x the envelope the parity harness pins "
                         "(CL2_ULP_ENVELOPE_ABS) and cl's stays <= 10 ulp -- "
                         "drift in either direction, or a scheme that "
                         "became smooth, re-opens the ceiling")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
    from legoesm.grids.factory import create_fv3_duo_grid

    grid = create_fv3_duo_grid(args.n)
    model = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=args.km, hydrostatic=True,
                                                   n_split=args.n_split))
    ic = model.dcmip16_initial_state(do_pert=True, terminator=True)
    # q = [sphum, cl, cl2]
    ng, n = grid.ng, grid.n
    cs = slice(ng, ng + n)
    base = model.step(ic, args.dt)
    ulp = np.spacing(TERM_QCLY)             # one ulp at the qcly scale
    results = {}
    print(f"C{args.n} km={args.km} n_split={args.n_split} dt={args.dt}; "
          f"1 ulp at qcly = {ulp:.3e}")
    for iq, name, value in ((2, "cl2", TERM_QCLY / 2), (1, "cl", 0.0)):
        q0 = np.asarray(ic["q"][iq])
        mask = np.zeros_like(q0, dtype=bool)
        mask[:, cs, cs, :] = (plateau_mask(q0[:, cs, cs, :], value)
                              if value else q0[:, cs, cs, :] == 0.0)
        # ONE ulp OF THE PLATEAU VALUE (codex 2026-09-14: spacing(qcly)
        # is two ulp at the 2e-6 plateau). cl's plateau is exactly 0,
        # where an ulp is 5e-324 and meaningless; it is nudged by one ulp
        # of qcly instead, as a "smallest positive value" control.
        nudge = np.spacing(value) if value else ulp
        q1 = q0.copy()
        q1[mask] += nudge
        pert = dict(ic)
        pert["q"] = list(ic["q"])
        pert["q"][iq] = jnp.asarray(q1)
        out = model.step(pert, args.dt)
        a = np.asarray(base["q"][iq])[:, cs, cs, :]
        b = np.asarray(out["q"][iq])[:, cs, cs, :]
        d = np.abs(a - b)
        moved = d > 10 * nudge
        print(f"{name}: plateau cells perturbed {int(mask.sum())} by "
              f"{nudge:.3e}; one-step output max |change| {d.max():.3e} = "
              f"{d.max() / nudge:.3g} nudges; cells moved > 10x: "
              f"{int(moved.sum())}; per-level max "
              f"{[f'{d[..., k].max():.1e}' for k in range(d.shape[-1])]}")
        # the OTHER tracers must not move at all: passengers are independent
        for jq, other in ((0, "sphum"), (1, "cl"), (2, "cl2")):
            if jq == iq:
                continue
            dd = float(np.abs(np.asarray(base["q"][jq]) - np.asarray(out["q"][jq])).max())
            print(f"   {other} changed by {dd:.3e} (must be 0)")
            if dd != 0.0:
                print("PASSENGERS ARE NOT INDEPENDENT -- refusing")
                return 2
        results[name] = float(d.max())
    if args.assert_envelope:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "full_step_oracle_parity",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "full_step_oracle_parity.py"))
        parity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parity)
        env = parity.CL2_ULP_ENVELOPE_ABS
        ok_cl2 = 0.5 * env <= results["cl2"] <= 1.2 * env
        ok_cl = results["cl"] <= 10 * ulp        # cl was nudged by ulp(qcly)
        print(f"envelope check: cl2 {results['cl2']:.3e} vs pinned {env:.3e} "
              f"-> {'OK' if ok_cl2 else 'DRIFTED'}; cl {results['cl'] / ulp:.1f} ulp "
              f"-> {'OK' if ok_cl else 'NOT SMOOTH'}")
        return 0 if (ok_cl2 and ok_cl) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
