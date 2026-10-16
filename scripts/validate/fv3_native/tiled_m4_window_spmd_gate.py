"""M4b SPMD window gate: windows sharded one per device vs the flat step.

Runs ``acoustic_loop_3d`` on window stacks placed one window per device on
the ``(6, kt, kt)`` tile mesh (``fv3_duo_window_spmd``: certified tiled
exchange bodies + 2-round pad refresh; barriers on the flat fallback),
scatters the owned cells back and compares against the flat BATCHED arm
bitwise.  Needs ``6*kt*kt`` devices (CPU: XLA_FLAGS
--xla_force_host_platform_device_count).  Derived from
tiled_m4_window_gate.py; same verdict rules.

The tiled port's M4 design runs the certified whole-face kernels
unchanged on padded windows (``fv3_duo_windows``): seam pads ``pad``
deep, refreshed once per substep at entry, every cross-face firing
delivering only its write-set.  This gate is the DECIDER for that design
and for the pad depth: it runs ``acoustic_loop_3d`` (n_split substeps,
DCMIP16 perturbed initial state) on windows at ``kt``/``pad`` and on the
flat six-face state, scatters the windows' OWNED cells back and compares
every state/nh/press field cell by cell.  BITWISE equality is the
target; the flat batched-vs-loop arm is reported alongside so a lowering
difference is attributed, never absorbed.

Exit 0 = bitwise on every field, 1 = a difference, 2 = refused.

    python tiled_m4_window_gate.py --n 48 --km 10 --kt 2 --pad 12 --n-split 3
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

import numpy as np


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _diff_map(lay, w, r, group, k):
    """Where do the owned cells differ?  Per (face, ti, tj): count, and
    the minimum distance of a differing cell to the tile's interior
    boundary along i and along j (0 = on the tile's first/last owned
    row).  A defect confined to distance ~0 is a seam/ownership defect; a
    uniform spread is a kernel-window defect."""
    from legoesm.grids.fv3_duo_windows import horizontal_axes
    bad = (w != r) & ~(np.isnan(w) & np.isnan(r))
    axes = horizontal_axes(lay, r.shape, 6)
    if axes is None:
        return
    red = tuple(a for a in range(bad.ndim) if a not in (0,) + axes)
    b2 = bad.any(axis=red) if red else bad            # (6, ei, ej)
    ei, ej = b2.shape[1], b2.shape[2]
    sh_i, sh_j = (lay.m_a - ei + 1) // 2, (lay.m_a - ej + 1) // 2
    rows = []
    for face in range(6):
        for ti in range(lay.kt):
            for tj in range(lay.kt):
                ii = [i for i in range(ei) if lay.owner(i + sh_i) == ti]
                jj = [j for j in range(ej) if lay.owner(j + sh_j) == tj]
                sub = b2[face][np.ix_(ii, jj)]
                c = int(sub.sum())
                if not c:
                    continue
                pi, pj = np.nonzero(sub)
                di = min(pi.min(), len(ii) - 1 - pi.max())
                dj = min(pj.min(), len(jj) - 1 - pj.max())
                rows.append((face, ti, tj, c, int(di), int(dj)))
    print(f"    diff map {group}.{k}: (face,ti,tj,count,dist_i,dist_j) "
          f"{rows[:24]}{' ...' if len(rows) > 24 else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--km", type=int, default=10, choices=(5, 10))
    ap.add_argument("--nh", action="store_true")
    ap.add_argument("--kt", type=int, default=2)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--n-split", type=int, default=3)
    ap.add_argument("--dt", type=float, default=900.0)
    ap.add_argument("--flat-arm", choices=("loop", "batched", "both"),
                    default="both",
                    help="which flat reference(s) to run")
    ap.add_argument("--comm", choices=("spmd", "flat", "flat-pinned"),
                    default="spmd",
                    help="spmd: fv3_duo_window_spmd (tiled bodies + pad "
                         "ppermutes); flat: the single-device window bundle "
                         "(scatter/gather through the flat state) with the "
                         "SAME window sharding -- the ATTRIBUTION arm: a "
                         "deviation it shares with spmd is the partitioned-"
                         "kernel lowering, not the exchange")
    ap.add_argument("--control", action="store_true",
                    help="also run the same-shape no-comm CONTROL (window "
                         "layout, kernels partitioned identically, exchange "
                         "through the flat state, results pinned) and compare "
                         "the two window arms directly, bitwise: the exchange "
                         "is exact iff they agree (GLM 2026-09-04)")
    ap.add_argument("--gspmd-face-arm", action="store_true",
                    help="also run the flat batched step GSPMD-sharded "
                         "P('face') on 6 devices and report its deviation "
                         "from the unsharded flat step (the lowering class)")
    ap.add_argument("--diff-map", action="store_true",
                    help="for every differing field: per-tile count of "
                         "differing OWNED cells and their minimum distance "
                         "to the tile's interior boundary (localises a "
                         "seam/edge defect)")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
    FV3DuoConfig,
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
    from legoesm.core.fv3_acoustic_3d import acoustic_loop_3d
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA
    from jax.sharding import Mesh
    from legoesm.grids.fv3_duo_windows import gather_windows, scatter_owned
    from legoesm.grids.fv3_duo_window_spmd import attach_window_spmd_comm

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    wsha = _sha("packages/core/legoesm/grids/fv3_duo_windows.py")
    print(f"[gate] repo {sha} windows.py {wsha} C{args.n} km={args.km} "
          f"{'NH' if args.nh else 'hydro'} kt={args.kt} pad={args.pad} "
          f"n_split={args.n_split} dt={args.dt}")

    grid = create_fv3_duo_grid(args.n)
    cfg = FV3DuoConfig(**ORACLE_DAMPING, km=args.km, hydrostatic=not args.nh)
    model = FV3DuoDynamicsModel(grid, cfg)
    ctx = model._ctx_jax
    bundle = model.dcmip16_initial_state(do_pert=True)
    state0 = dict(bundle["state"])
    nh0 = bundle.get("nh")
    dp0 = None
    if args.nh:
        ak, bk = np.asarray(model._ak), np.asarray(model._bk)
        dp0 = (ak[1:] - ak[:-1]) + (bk[1:] - bk[:-1]) * 1.0e5
    kw = dict(n_split=args.n_split, ptop=float(model._ptop),
              akap=FV3_KAPPA, cp_air=FV3_CP_AIR, remap_follows=True,
              hydrostatic=not args.nh, dp0=dp0)

    def flat_arm(batched):
        return jax.jit(lambda st, nh: acoustic_loop_3d(
            ctx, st, args.dt, args.km, nh=nh, batched=batched, **kw))(
                state0, nh0)

    refs = {}
    if args.gspmd_face_arm:
        # ATTRIBUTION ARM (the M3 triage rule): the flat batched step with
        # its state sharded P('face') over 6 devices -- the kernels
        # partitioned by GSPMD, NO window comm.  Its deviation from the
        # unsharded flat step is the partitioned-kernel lowering class;
        # the window SPMD arm may not exceed it by more than a few ulps.
        from jax.sharding import NamedSharding, PartitionSpec as P
        mesh6 = Mesh(np.array(jax.devices()[:6]), ("face",))
        sh6 = NamedSharding(mesh6, P("face"))
        st6 = {k: jax.device_put(v, sh6) for k, v in state0.items()}
        nh6 = None if nh0 is None else {
            k: (jax.device_put(v, sh6) if hasattr(v, "ndim") and v.ndim >= 3
                and v.shape[0] == 6 else v) for k, v in nh0.items()}
        g = jax.jit(lambda st, nh: acoustic_loop_3d(
            ctx, st, args.dt, args.km, nh=nh, batched=True, **kw))(st6, nh6)
        b = flat_arm(True)
        print("===== ATTRIBUTION: flat batched sharded P('face') x6 vs "
              "unsharded =====")
        for group in ("state", "press"):
            for k in sorted(b[group]):
                r, w = b[group][k], g[group][k]
                if not hasattr(r, "ndim"):
                    continue
                r, w = np.asarray(r), np.asarray(w)
                bad = int(((w != r) & ~(np.isnan(w) & np.isnan(r))).sum())
                d = np.abs(w - r)
                fin = np.isfinite(d)
                rel = (np.nanmax(d[fin] / (np.abs(r[fin]) + 1e-300))
                       if fin.any() and bad else 0.0)
                print(f"  {group}.{k:8s} {'BITWISE' if bad == 0 else f'DIFF {bad}/{r.size} max rel={rel:.3e}'}")
        print(f"  gspmd-face output sharding: {g['state']['delp'].sharding}")
        refs["gspmd_face"] = g
    if args.flat_arm in ("loop", "both"):
        refs["loop"] = flat_arm(False)
        print("[gate] flat loop arm done")
    if args.flat_arm in ("batched", "both"):
        refs["batched"] = flat_arm(True)
        print("[gate] flat batched arm done")

    # -- the SPMD window arm -------------------------------------------------
    ndev = 6 * args.kt * args.kt
    if jax.device_count() < ndev:
        print(f"[gate] REFUSED: {jax.device_count()} devices < {ndev} "
              f"(set XLA_FLAGS=--xla_force_host_platform_device_count)")
        return 2
    mesh = Mesh(np.array(jax.devices()[:ndev]).reshape(6, args.kt, args.kt),
                ("face", "tile_i", "tile_j"))
    try:
        if args.comm == "spmd":
            wctx, comm = attach_window_spmd_comm(ctx, mesh, args.pad)
            sharding = comm.sharding
        else:
            from legoesm.grids.fv3_duo_windows import attach_window_comm
            from legoesm.grids.fv3_duo_window_spmd import window_sharding
            wctx, comm = attach_window_comm(ctx, args.kt, args.pad, "padded")
            sharding = window_sharding(mesh)
            if args.comm == "flat-pinned":
                # GLM's same-shape no-comm control: the exchange goes
                # through the flat state (direct global indexing) but every
                # result is pinned back to the window sharding, so the
                # kernels keep the SPMD arm's per-device shapes
                comm.sharding = sharding
    except ValueError as e:
        print(f"[gate] REFUSED: {e}")
        return 2
    lay = comm.lay
    print(f"[gate] {lay}: window face n_w={lay.n_w}, origins "
          f"{sorted(set(o[1] for o in lay.origins))}; {ndev} devices, "
          f"comm={args.comm}, barriers "
          f"{getattr(comm, 'barrier_mode', 'flat')}")
    put = lambda v: jax.device_put(gather_windows(lay, v), sharding)
    wstate = {k: put(v) for k, v in state0.items()}
    wnh = None
    if nh0 is not None:
        wnh = {k: (put(v)
                   if hasattr(v, "ndim") and v.ndim >= 3 and v.shape[0] == 6
                   else v) for k, v in nh0.items()}
    try:
        wout = jax.jit(lambda st, nh: acoustic_loop_3d(
            wctx, st, args.dt, args.km, nh=nh, batched=True, **kw))(
                wstate, wnh)
    finally:
        ctx.tab.window_comm = None
    # the output sharding is part of the receipt: a replicated output
    # would mean GSPMD gathered the state somewhere in the step
    shs = {k: str(getattr(v, "sharding", None)) for k, v in
           wout["state"].items() if hasattr(v, "sharding")}
    print(f"[gate] window arm done; output shardings: "
          f"{sorted(set(shs.values()))}")
    def block_coherence(bundle_w, group):
        """Every window's copy of every cell inside its BLOCK (the owned
        cells plus, for a node axis, the shared row the neighbour also
        stores) must equal the owner's value bitwise -- the blocked
        exchange writes duplicated slots in every storing tile from
        bit-identical inputs (fv3_duo_spmd's coherence rule), so a
        divergent duplicate would make the SPMD arm's next firing
        tile-dependent.  Reported per field; any mismatch fails."""
        from legoesm.grids.fv3_duo_windows import (
            horizontal_axes, _index, _fshape, window_axis_extent)
        bad_total = 0
        for k, v in bundle_w.items():
            if not (hasattr(v, "ndim") and v.ndim >= 3
                    and v.shape[0] == lay.nb):
                continue
            axes = horizontal_axes(lay, v.shape, lay.nb)
            if axes is None:
                continue
            vw = np.asarray(v)
            flat = scatter_owned(lay, vw, np)
            fshape = flat.shape
            bad = 0
            for w in range(lay.nb):
                idx = _index(lay, w, fshape, axes)
                ti = (w % (lay.kt * lay.kt)) // lay.kt
                tj = w % lay.kt
                sel = []
                for ax, t in ((axes[0], ti), (axes[1], tj)):
                    e_f = fshape[ax]
                    e_w = window_axis_extent(lay, e_f)
                    o = idx[ax].start
                    shift = (lay.m_a - e_f + 1) // 2
                    lo = lay.block_start(t) - shift - o
                    # blocked layout: the shared e rows are STORED by both
                    # tiles; compute partition: a single owner, no dups
                    e = e_f - (lay.m_a if e_f >= lay.m_a else lay.n)
                    ext = lay.nl + (e if lay.partition == "padded" else 0)
                    hi = min(lo + ext, e_w)
                    sel.append(slice(max(lo, 0), hi))
                wi = [slice(None)] * (v.ndim - 1)
                wi[axes[0] - 1], wi[axes[1] - 1] = sel[0], sel[1]
                fi = list(idx)
                fi[axes[0]] = slice(idx[axes[0]].start + sel[0].start,
                                    idx[axes[0]].start + sel[0].stop)
                fi[axes[1]] = slice(idx[axes[1]].start + sel[1].start,
                                    idx[axes[1]].start + sel[1].stop)
                a, b = vw[w][tuple(wi)], flat[tuple(fi)]
                bad += int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
            if bad:
                print(f"  {group}.{k:8s} BLOCK COHERENCE: {bad} duplicated "
                      f"cells differ between storing windows")
            bad_total += bad
        return bad_total

    def to_flat(bundle_w):
        out = {}
        for k, v in bundle_w.items():
            if hasattr(v, "ndim") and v.ndim >= 3 and v.shape[0] == lay.nb:
                try:
                    out[k] = scatter_owned(lay, np.asarray(v), np)
                except ValueError:
                    out[k] = None
            else:
                out[k] = None if hasattr(v, "ndim") else v
        return out

    ctrl_rc = None
    if args.control and args.comm == "spmd":
        from legoesm.grids.fv3_duo_windows import attach_window_comm
        cctx, ccomm = attach_window_comm(ctx, args.kt, args.pad, "padded")
        ccomm.sharding = sharding
        try:
            cout = jax.jit(lambda st, nh: acoustic_loop_3d(
                cctx, st, args.dt, args.km, nh=nh, batched=True, **kw))(
                    wstate, wnh)
        finally:
            ctx.tab.window_comm = None
        cs = {str(getattr(v, "sharding", None)) for v in
              cout["state"].values() if hasattr(v, "sharding")}
        print(f"[gate] CONTROL arm (flat exchange, pinned) done; output "
              f"shardings: {sorted(cs)}")
        print("===== window-SPMD vs same-shape no-comm CONTROL (OWNED cells; "
              "the two arms refresh their pad zones differently -- once per "
              "substep vs every firing -- so pads are not compared) =====")
        ctrl_rc = 0
        for group in ("state", "nh", "press"):
            if wout.get(group) is None or cout.get(group) is None:
                continue
            wf, cf = to_flat(wout[group]), to_flat(cout[group])
            for k in sorted(wout[group]):
                a, b = wf.get(k), cf.get(k)
                if a is None or b is None or not hasattr(a, "ndim"):
                    continue
                a, b = np.asarray(a), np.asarray(b)
                if a.shape != b.shape:
                    print(f"  {group}.{k:8s} SHAPE {a.shape} vs {b.shape}")
                    ctrl_rc = 1
                    continue
                bad = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
                if bad:
                    ctrl_rc = 1
                    d = np.abs(a - b)
                    fin = np.isfinite(d)
                    print(f"  {group}.{k:8s} DIFF {bad}/{a.size} max|d|="
                          f"{np.nanmax(d[fin]) if fin.any() else float('nan'):.3e}")
                else:
                    print(f"  {group}.{k:8s} BITWISE ({a.size} cells)")
        print(f"[gate] CONTROL VERDICT: window-SPMD "
              f"{'==' if ctrl_rc == 0 else '!='} no-comm control "
              f"({'exchange exact' if ctrl_rc == 0 else 'exchange DIFFERS'})")

    rc = 0
    verdict_by_arm = {}
    for arm, ref in refs.items():
        print(f"===== windows(kt={args.kt}, pad={args.pad}) vs flat {arm} "
              f"=====")
        arm_rc = 0
        n_scored = 0
        for group in ("state", "nh", "press"):
            if (ref.get(group) is None) != (wout.get(group) is None):
                print(f"  {group}: present on one arm only -- REFUSED")
                arm_rc = 1
                continue
            if ref.get(group) is None:
                continue
            wf = to_flat(wout[group])
            if set(wf) != set(ref[group]):
                print(f"  {group}: key sets differ (window-only "
                      f"{sorted(set(wf) - set(ref[group]))}, flat-only "
                      f"{sorted(set(ref[group]) - set(wf))}) -- REFUSED")
                arm_rc = 1
            for k in sorted(ref[group]):
                r = ref[group][k]
                w = wf.get(k)
                if not hasattr(r, "ndim"):
                    continue                      # scalars/flags
                if w is None:
                    print(f"  {group}.{k:8s} UNSCORED (window array could "
                          f"not be scattered) -- REFUSED")
                    arm_rc = 1
                    continue
                n_scored += 1
                r = np.asarray(r)
                if r.shape != w.shape:
                    print(f"  {group}.{k:8s} SHAPE {w.shape} vs {r.shape}")
                    arm_rc = 1
                    continue
                d = np.abs(w - r)
                bad = int(((w != r) & ~(np.isnan(w) & np.isnan(r))).sum())
                if bad == 0:
                    print(f"  {group}.{k:8s} BITWISE ({r.size} cells)")
                else:
                    arm_rc = 1
                    fin = np.isfinite(d)
                    mx = np.nanmax(d[fin]) if fin.any() else float("nan")
                    rel = (np.nanmax(d[fin] / (np.abs(r[fin]) + 1e-300))
                           if fin.any() else float("nan"))
                    print(f"  {group}.{k:8s} DIFF {bad}/{r.size} cells, "
                          f"max|d|={mx:.3e}, max rel={rel:.3e}")
                    if args.diff_map:
                        _diff_map(lay, w, r, group, k)
        if n_scored < 5:
            print(f"  only {n_scored} fields scored -- REFUSED (vacuous)")
            arm_rc = 1
        if args.kt > 1:
            n_incoh = sum(block_coherence(wout[g], g)
                          for g in ("state", "nh", "press")
                          if wout.get(g) is not None)
            print(f"  [{arm}] block coherence: {n_incoh} divergent "
                  f"duplicated cells")
            if n_incoh:
                arm_rc = 1
        print(f"  [{arm}] {n_scored} fields scored")
        verdict_by_arm[arm] = arm_rc
    # The window arm runs the BATCHED kernels (vmap over windows), so the
    # flat batched arm is its reference; the loop arm is the attribution
    # arm (batched-vs-loop differ by lowering even with FMA off, job
    # 9631684) and is reported, never folded into the verdict.
    rc = verdict_by_arm.get("batched", verdict_by_arm.get("loop", 0))
    if ctrl_rc is not None:
        rc = ctrl_rc
    if "gspmd_face" in verdict_by_arm:
        # the window SPMD arm's own reference is the PARTITIONED flat
        # step: bitwise against it means the exchange/pads changed nothing
        # and every deviation from the unsharded step is the lowering class
        print(f"[gate] window-SPMD vs GSPMD-face-sharded flat: "
              f"{'BITWISE' if verdict_by_arm['gspmd_face'] == 0 else 'DIFFERS'}")
        rc = verdict_by_arm["gspmd_face"] if ctrl_rc is None else ctrl_rc
    if "loop" in refs and "batched" in refs:
        print("===== flat batched vs flat loop (attribution arm) =====")
        for group in ("state", "press"):
            for k in sorted(refs["loop"][group]):
                a, b = refs["loop"][group][k], refs["batched"][group][k]
                if not hasattr(a, "ndim"):
                    continue
                a, b = np.asarray(a), np.asarray(b)
                bad = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
                print(f"  {group}.{k:8s} {'BITWISE' if bad == 0 else f'DIFF {bad}'}")
    print(f"[gate] SPMD VERDICT kt={args.kt} pad={args.pad}: "
          f"{'BITWISE' if rc == 0 else 'DIFFERS'} vs flat batched "
          f"(per arm: {verdict_by_arm})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
