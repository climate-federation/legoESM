#!/usr/bin/env python
"""WHERE does the NH moist k_split=2 step stop compiling?

``test_nh_moist_pkz_is_recomputed_not_trusted[2]`` dies with
``LLVM compilation error: Cannot allocate memory`` after a ~3.5 min
``jit_scan`` compile inside ``sim1_solver``. Two explanations were
pre-registered and BOTH REFUTED:

* job 9443826 -- the case ALONE at ``--mem=600G``: same failure, so it
  is not the job's memory limit;
* job 9443895 -- ``[2]`` alone in a FRESH process at 400G: same
  failure, so it is not code memory accumulated across the module's
  other graphs, nor the ``[1]`` parametrization running first.

What is left, and what this probe measures: the TEST ITSELF drives four
eager ``fv_dynamics_step`` calls (two arms x two pkz bundles), and at
k_split=2 each NH call compiles roughly twice the scan work of
k_split=1. ``test_moist_parity_against_the_spec[False-2]`` does ONE
such call and passes. So the question is a COUNT, and the answer is a
number this prints rather than a hypothesis: how many consecutive NH
moist k_split=2 steps can one process compile?

It also runs a second arm with ``jax.clear_caches()`` between calls, so
"does releasing JAX's caches help" is measured rather than assumed --
clear_caches frees JAX's own tables, and whether LLVM's already-JITted
code goes with them is exactly what is in doubt.

    python nh_ksplit2_compile_probe.py [--n 4] [--clear] [--k-split 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                    # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

N, NG, KM, NQ = 12, 3, 5, 2
AKAP, CP_AIR, BDT = 2.0 / 7.0, 1004.6, 60.0
KORD_MT, KORD_TM, KORD_TR = 9, -9, 9
ZVIR = 0.6077338443
Q_SCALE = 0.01


def build(hydrostatic, seed=21):
    from legoesm.core.fv3_native_state_3d import build_state_3d
    rng = np.random.default_rng(seed)
    ma = N + 2 * NG
    st = build_state_3d(N, NG, KM, hydrostatic=hydrostatic,
                        remap_follows=True)
    for t, face in enumerate(st):
        for k in range(KM):
            face["delp"][:, :, k] = 1.0e4 * (1.0 + 0.05 * t + 0.02 * k) \
                + 50.0 * rng.standard_normal((ma, ma))
            face["pt"][:, :, k] = 280.0 + 2.0 * t + 3.0 * k \
                + 0.5 * rng.standard_normal((ma, ma))
            face["u"][:, :, k] = 5.0 + 0.3 * t + 0.1 * k \
                + rng.standard_normal((ma, ma + 1))
            face["v"][:, :, k] = -4.0 + 0.2 * t - 0.1 * k \
                + rng.standard_normal((ma + 1, ma))
            if not hydrostatic:
                face["w"][:, :, k] = 0.2 * rng.standard_normal((ma, ma))
                face["delz"][:, :, k] = -(500.0 + 20.0 * k)
    return st


def tracers(seed=22, scale=Q_SCALE):
    rng = np.random.default_rng(seed)
    ma = N + 2 * NG
    return [[scale * (1.0 + 0.3 * t + 0.7 * iq
                      + 0.2 * rng.standard_normal((ma, ma, KM)))
             for iq in range(NQ)] for t in range(6)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4,
                    help="how many consecutive steps to attempt")
    ap.add_argument("--k-split", type=int, default=2)
    ap.add_argument("--clear", action="store_true",
                    help="jax.clear_caches() between steps -- the "
                         "MEASUREMENT of whether that helps, not an "
                         "assumption that it does")
    ap.add_argument("--hydrostatic", action="store_true")
    args = ap.parse_args(argv)

    from legoesm.core import fv3_dynamics as jdyn
    from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_eta import set_eta_analytic

    hydro = args.hydrostatic
    print(f"n={args.n} k_split={args.k_split} clear={args.clear} "
          f"hydrostatic={hydro}", flush=True)
    # use_ext_bundle/oracle_conventions and hs6 are the GATE's own
    # construction (tests/grids/test_fv3_dynamics.py::ctx); the JAX
    # stepper refuses anything else, and it refused this probe's first
    # version rather than letting it produce a number.
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ctx["hs6"] = [np.zeros((N + 2 * NG, N + 2 * NG), dtype=np.float64)
                  for _ in range(6)]
    jctx = build_jax_duo_stepper_context(ctx)
    ak, bk, ptop, _ks = set_eta_analytic(KM)

    for i in range(args.n):
        t0 = time.time()
        jst = state_3d_to_jax(build(hydro))
        _t = tracers()
        q = [jnp.asarray(np.stack([_t[f][iq] for f in range(6)]))
             for iq in range(NQ)]
        press = jdyn.p_var_hydrostatic(jst["delp"], ptop=ptop, akap=AKAP,
                                       n=N, ng=NG, km=KM)
        try:
            jdyn.fv_dynamics_step(
                jctx, jst, press, q=q, bdt=BDT, km=KM,
                k_split=args.k_split, n_split=2, ptop=ptop, ak=ak, bk=bk,
                akap=AKAP, cp_air=CP_AIR, kord_mt=KORD_MT,
                kord_tm=KORD_TM, kord_tr=KORD_TR,
                hydrostatic=hydro, w_limiter=not hydro,
                zvir=ZVIR, sphum_index=0)
        except Exception as exc:                    # noqa: BLE001
            print(f"step {i + 1}: FAILED after {time.time() - t0:.1f}s "
                  f"-- {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            print(f"VERDICT: {i} step(s) compiled, step {i + 1} did not.",
                  flush=True)
            return 1
        print(f"step {i + 1}: ok  ({time.time() - t0:.1f}s)", flush=True)
        if args.clear:
            jax.clear_caches()
    print(f"VERDICT: all {args.n} step(s) compiled.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
