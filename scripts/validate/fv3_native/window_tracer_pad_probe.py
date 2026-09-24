"""Where does the window step's TRACER first go non-finite, and how does the
front move?  Steps the single-process window model (kt x kt windows on
6*kt*kt fake CPU devices) and, after every step, reports for the tracer --
and, as the control, for pt -- the NaN count per LOCAL row/column of each
window of face 0, split into body and pad rows.  Records only; interprets
nothing (2026-09-21, the step-3 tracer NaN of jobs 9910440/1).

Usage: XLA_FLAGS=--xla_force_host_platform_device_count=24 python
       window_tracer_pad_probe.py --n 24 --km 5 --kt 2 --pad 5 --n-split 8 --steps 4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _row_report(a, lay, w, label):
    """NaN counts by local row (axis 1) and column (axis 2) of window w."""
    face, oi, oj = lay.origins[w]
    ti = (w % (lay.kt * lay.kt)) // lay.kt
    tj = w % lay.kt
    bi0 = lay.block_start(ti) - oi
    bj0 = lay.block_start(tj) - oj
    bad = ~np.isfinite(a[w])
    if not bad.any():
        return f"    {label} window {w} (face {face}, tile {ti},{tj}): clean"
    rows = bad.reshape(bad.shape[0], -1).sum(axis=1)
    cols = bad.transpose(1, 0, *range(2, bad.ndim)).reshape(bad.shape[1], -1).sum(axis=1)

    def fmt(cnt, b0):
        nl = lay.nl
        out = []
        for i, c in enumerate(cnt):
            if c:
                where = "body" if b0 <= i < b0 + nl else "PAD"
                out.append(f"{i}({where}):{int(c)}")
        return " ".join(out)
    return (f"    {label} window {w} (face {face}, tile {ti},{tj}) body rows "
            f"[{bi0},{bi0 + lay.nl}) cols [{bj0},{bj0 + lay.nl}):\n"
            f"      rows {fmt(rows, bi0)}\n      cols {fmt(cols, bj0)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--km", type=int, default=5)
    ap.add_argument("--kt", type=int, default=2)
    ap.add_argument("--pad", type=int, default=5)
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--refresh", choices=("none", "before", "after", "both"),
                    default="none",
                    help="DISCRIMINATOR: force a full window pad refresh of "
                         "the tracer stack before/after the tracer step")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    ndev = 6 * args.kt * args.kt
    if jax.device_count() < ndev:
        print(f"REFUSED: need {ndev} devices, have {jax.device_count()}")
        return 2
    if args.refresh != "none":
        import jax.numpy as jnp
        import legoesm.core.fv3_dynamics as fdyn
        _orig = fdyn.tracer_2d_1l_sixface

        def _refresh_q(ctx, q5):
            wc = getattr(ctx.tab, "window_comm", None)
            if wc is None:
                return q5
            nb, nq, m0, m1, kmk = q5.shape
            flat = jnp.moveaxis(q5, 1, -1).reshape(nb, m0, m1, kmk * nq)
            flat = wc.refresh({"q": flat})["q"]
            return jnp.moveaxis(flat.reshape(nb, m0, m1, kmk, nq), -1, 1)

        def _wrapped(ctx, q_in, dp1, flux_cap, **kw):
            if args.refresh in ("before", "both"):
                q_in = _refresh_q(ctx, q_in)
            out = _orig(ctx, q_in, dp1, flux_cap, **kw)
            if args.refresh in ("after", "both"):
                out = dict(out); out["q"] = _refresh_q(ctx, out["q"])
            return out
        fdyn.tracer_2d_1l_sixface = _wrapped
        print(f"[probe] DISCRIMINATOR: tracer pad refresh forced {args.refresh}")
    grid = create_fv3_duo_grid(args.n)
    cfg = FV3DuoConfig(km=args.km, hydrostatic=True, n_split=args.n_split)
    mesh = Mesh(np.array(jax.devices()[:ndev]).reshape(6, args.kt, args.kt),
                ("face", "tile_i", "tile_j"))
    model = FV3DuoDynamicsModel(grid, cfg, step_spmd_mesh=mesh,
                                step_windows=(args.kt, args.pad))
    lay = model.window_layout
    print(f"[probe] C{args.n} km={args.km} kt={args.kt} pad={args.pad} "
          f"n_split={args.n_split} dt={args.dt} W={lay.W} nl={lay.nl} "
          f"origins[0:2]={lay.origins[:2]}")
    win = model.dcmip16_initial_state(do_pert=True)
    per_face = args.kt * args.kt
    for step in range(1, args.steps + 1):
        win = model.step(win, args.dt)
        q = np.asarray(win["q"][0])
        pt = np.asarray(win["state"]["pt"])
        print(f"[probe] step {step}: q nan={int((~np.isfinite(q)).sum())}/{q.size} "
              f"pt nan={int((~np.isfinite(pt)).sum())}/{pt.size}")
        for w in range(per_face):            # every window of face 0
            print(_row_report(q, lay, w, "q "))
        for w in range(per_face):
            r = _row_report(pt, lay, w, "pt")
            if "clean" not in r:
                print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
