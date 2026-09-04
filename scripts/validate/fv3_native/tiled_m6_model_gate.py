"""M6 model gate: FV3DuoDynamicsModel.step on sharded WINDOWS vs faces.

The full ``fv_dynamics`` step (acoustic loop + tracer transport + vertical
remap) through the model API: ``step_windows=(kt, pad)`` on a (6, kt, kt)
mesh (one window per device) against the same model on six faces, GSPMD-
sharded P('face') on 6 devices (the reference the SPMD substep gate
established: both partitioned programs, bitwise).  ``--steps`` outer steps
of ``--dt`` seconds; every field of the state bundle compared on OWNED
cells, bitwise; the window arm's output sharding is printed (a replicated
output means the step gathered).  Exit 0 bitwise, 1 differs, 2 refused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--km", type=int, default=10, choices=(5, 10))
    ap.add_argument("--nh", action="store_true")
    ap.add_argument("--kt", type=int, default=2)
    ap.add_argument("--pad", type=int, default=11)
    ap.add_argument("--steps", type=int, default=2,
                    help="outer steps to score (>= 1; 0 is refused)")
    ap.add_argument("--dt", type=float, default=900.0)
    ap.add_argument("--n-split", type=int, default=3)
    ap.add_argument("--distributed", action="store_true",
                    help="MULTI-PROCESS run (jax.distributed.initialize via "
                         "SLURM auto-detect): one window per process; the "
                         "in-process face-sharded reference is NOT run -- "
                         "compare against --ref-npz instead")
    ap.add_argument("--save-npz", type=str, default=None,
                    help="write the window arm's flat owned-cell outputs "
                         "after every step (process 0 only)")
    ap.add_argument("--ref-npz", type=str, default=None,
                    help="compare the window arm against these saved "
                         "outputs (from a single-process run) instead of / "
                         "in addition to the in-process reference")
    args = ap.parse_args(argv)

    import jax
    if args.distributed:
        jax.distributed.initialize(initialization_timeout=900)
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    rank0 = jax.process_index() == 0

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    if args.steps < 1:
        print("[m6] REFUSED: --steps must be >= 1 (nothing would be scored)")
        return 2
    ndev = 6 * args.kt * args.kt
    print(f"[m6] repo {sha} C{args.n} km={args.km} "
          f"{'NH' if args.nh else 'hydro'} kt={args.kt} pad={args.pad} "
          f"steps={args.steps} dt={args.dt} n_split={args.n_split} "
          f"devices={jax.device_count()} processes={jax.process_count()} "
          f"rank={jax.process_index()} local_devices="
          f"{jax.local_device_count()}")
    if args.distributed and jax.process_count() != ndev:
        print(f"[m6] REFUSED: {jax.process_count()} processes != {ndev} "
              f"(one window per process)")
        return 2
    if jax.device_count() < ndev:
        print(f"[m6] REFUSED: need {ndev} devices")
        return 2

    grid = create_fv3_duo_grid(args.n)
    cfg = FV3DuoConfig(km=args.km, hydrostatic=not args.nh,
                       n_split=args.n_split)

    # reference: faces, GSPMD-sharded P('face') on 6 devices (in-process
    # only: a multi-process job runs ONE program on all its processes)
    ref_model = ref = None
    if not args.distributed:
        mesh6 = Mesh(np.array(jax.devices()[:6]), ("face",))
        sh6 = NamedSharding(mesh6, P("face"))
        ref_model = FV3DuoDynamicsModel(grid, cfg, step_out_shardings=sh6,
                                        step_face_batched=True)
        ref = ref_model.dcmip16_initial_state(do_pert=True)
        ref = jax.tree_util.tree_map(
            lambda a: jax.device_put(a, sh6) if hasattr(a, "ndim")
            and a.ndim >= 3 and a.shape[0] == 6 else a, ref)
    saved_ref = np.load(args.ref_npz) if args.ref_npz else None

    # the window arm
    mesh = Mesh(np.array(jax.devices()[:ndev]).reshape(6, args.kt, args.kt),
                ("face", "tile_i", "tile_j"))
    try:
        win_model = FV3DuoDynamicsModel(grid, cfg, step_spmd_mesh=mesh,
                                        step_windows=(args.kt, args.pad))
    except ValueError as e:
        print(f"[m6] REFUSED: {e}")
        return 2
    lay = win_model.window_layout
    print(f"[m6] {lay}; barriers {win_model._window_comm.barrier_mode}")
    win = win_model.dcmip16_initial_state(do_pert=True)

    def leaves(bundle):
        """(path, array) for EVERY array leaf of the bundle: state, press,
        the tracer list, omga, the NH carry -- nothing skipped silently."""
        out = []
        for path, a in jax.tree_util.tree_leaves_with_path(bundle):
            if hasattr(a, "ndim"):
                out.append((jax.tree_util.keystr(path), a))
        return out

    def flat_leaves(bundle_w):
        """(path, host array) of the window bundle scattered to faces --
        under jax.distributed the scatter needs EVERY window's shard, so
        the arrays are gathered to the host first (process_allgather)."""
        if args.distributed:
            from jax.experimental import multihost_utils as mhu
            bundle_w = jax.tree_util.tree_map(
                lambda a: np.asarray(mhu.process_allgather(a, tiled=True))
                if hasattr(a, "sharding") else a, bundle_w)
        return leaves(win_model.to_flat(bundle_w))

    # IC round trip must be exact on EVERY leaf before any step is scored
    ic_flat = dict(flat_leaves(win))
    if ref is not None:
        ref_ic = leaves(ref)
    else:
        ref_ic = [(k[3:], saved_ref[k]) for k in saved_ref.files
                  if k.startswith("ic:")]
    bad = 0
    n_ic = 0
    for path, a in ref_ic:
        b = ic_flat.get(path)
        n_ic += 1
        if b is None or not np.array_equal(np.asarray(a), np.asarray(b)):
            bad += 1
            print(f"  IC {path}: window round trip differs or missing")
    if bad or n_ic < 10:
        print(f"[m6] REFUSED: IC round trip not exact ({bad} bad of {n_ic})")
        return 2
    print(f"[m6] IC round trip exact on {n_ic} leaves")
    n_expected = n_ic
    to_save = {f"ic:{p}": np.asarray(a) for p, a in ic_flat.items()}

    rc = 0
    for it in range(args.steps):
        win = win_model.step(win, args.dt)
        if ref is not None:
            ref = ref_model.step(ref, args.dt)
            ref_leaves = leaves(ref)
        else:
            ref_leaves = [(k[len(f"step{it + 1}:"):], saved_ref[k])
                          for k in saved_ref.files
                          if k.startswith(f"step{it + 1}:")]
        # every window output must still be window-sharded: a replicated
        # leaf means the step gathered the state somewhere
        replicated = [p for p, v in leaves(win)
                      if getattr(v, "sharding", None) is not None
                      and v.sharding.is_fully_replicated]
        wf = dict(flat_leaves(win))
        to_save.update({f"step{it + 1}:{p}": np.asarray(a)
                        for p, a in wf.items()})
        n_ok = n_bad = 0
        worst = 0.0
        for path, r in ref_leaves:
            w = wf.get(path)
            r = np.asarray(r)
            if w is None:
                print(f"  step {it + 1} {path}: MISSING on the window arm")
                n_bad += 1
                continue
            w = np.asarray(w)
            if r.shape != w.shape:
                print(f"  step {it + 1} {path}: SHAPE {w.shape} vs {r.shape}")
                n_bad += 1
                continue
            d = ((r != w) & ~(np.isnan(r) & np.isnan(w))).sum()
            if d:
                n_bad += 1
                fin = np.isfinite(r - w)
                rel = np.nanmax(np.abs(r - w)[fin] /
                                (np.abs(r[fin]) + 1e-300))
                worst = max(worst, float(rel))
                print(f"  step {it + 1} {path:24s} DIFF {int(d)}/{r.size} "
                      f"max rel {rel:.3e}")
            else:
                n_ok += 1
        print(f"[m6] step {it + 1}: {n_ok}/{n_expected} leaves BITWISE, "
              f"{n_bad} differ (worst rel {worst:.3e}); replicated window "
              f"outputs: {replicated or 'none'}")
        if n_bad or n_ok != n_expected:
            rc = 1
        if replicated:
            print("  REFUSED: a window output came back replicated")
            rc = 1
    if args.save_npz and rank0:
        np.savez(args.save_npz, **to_save)
        print(f"[m6] saved {len(to_save)} flat outputs -> {args.save_npz}")
    print(f"[m6] VERDICT kt={args.kt} pad={args.pad} steps={args.steps}: "
          f"{'BITWISE' if rc == 0 else 'DIFFERS'} vs "
          f"{'the saved reference ' + args.ref_npz if ref is None else 'the face-sharded model'}")
    return rc


if __name__ == "__main__":
    _rc = main()
    import jax
    if jax.process_count() > 1:
        # every rank leaves through the coordination service's shutdown
        # barrier, then skips interpreter finalization: with the mpi
        # collectives backend, MPI_Finalize aborts inside
        # mca_base_var_group_finalize AFTER the results are printed (job
        # 9635203: all 24 ranks BITWISE, then "Aborted (core dumped)"),
        # and an aborted exit is a refusal no matter what was printed.
        # Same fix as bench_fv3_duo_spmd_scaling.py's bare mode.
        import os
        from jax.experimental import multihost_utils as mhu
        mhu.sync_global_devices("m6_done")
        jax.distributed.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if _rc in (0, None) else int(_rc))
    sys.exit(_rc)
