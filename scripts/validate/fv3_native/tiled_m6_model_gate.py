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
import ctypes
import os
import subprocess
import sys
import time

import numpy as np


def _phase_event(rank, phase, event, started=None):
    """Print one UTC, per-rank boundary around a compilation/dispatch phase."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = ("" if started is None
               else f" elapsed_s={time.perf_counter() - started:.6f}")
    print(f"[m6] PHASE {stamp} rank={rank} {phase} {event}{elapsed}",
          flush=True)


def _runtime_banner(jax):
    """Print runtime-resolved compile, device, library, and affinity state."""
    import jaxlib

    local_devices = jax.local_devices()
    gpu = any(device.platform in ("cuda", "gpu") for device in local_devices)
    nccl = "n/a"
    if gpu:
        try:
            library = ctypes.CDLL("libnccl.so.2")
            version = ctypes.c_int()
            library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
            library.ncclGetVersion.restype = ctypes.c_int
            rc = library.ncclGetVersion(ctypes.byref(version))
            if rc != 0:
                raise RuntimeError(f"ncclGetVersion returned {rc}")
            value = version.value
            nccl = f"{value // 10000}.{value // 100 % 100}.{value % 100}"
        except (OSError, AttributeError, RuntimeError) as exc:
            nccl = f"unavailable:{type(exc).__name__}"
    affinity = (sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else "unavailable")
    cache_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
    print(
        f"[m6] RUNTIME rank={jax.process_index()} "
        f"compile_cache={cache_dir or 'none'} "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r} "
        f"jax={jax.__version__} jaxlib={jaxlib.__version__} nccl={nccl} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')!r} "
        f"JAX_LOCAL_DEVICE_IDS={os.environ.get('JAX_LOCAL_DEVICE_IDS', 'all')!r} "
        f"visible_devices={[str(device) for device in local_devices]} "
        f"cpu_affinity={affinity}",
        flush=True,
    )


def leaves(bundle):
    """(path, array) for EVERY array leaf of the bundle: state, press, the
    tracer list, omga, the NH carry -- nothing skipped silently."""
    import jax
    out = []
    for path, a in jax.tree_util.tree_leaves_with_path(bundle):
        if hasattr(a, "ndim"):
            out.append((jax.tree_util.keystr(path), a))
    return out


def tracer_window_change(flat, prev, n: int, ng: int):
    """[(path, max |flat - prev| on the compute window)] for every tracer
    leaf (paths under ['q']), sorted by path.  Zero means the passenger
    did not move, which a sharded-vs-in-process comparison cannot see on
    its own (GLM 2026-09-13)."""
    cs = slice(ng, ng + n)
    out = []
    for path in sorted(p for p in flat if p.startswith("['q']")):
        a = np.asarray(flat[path])[:, cs, cs, :]
        b = np.asarray(prev[path])[:, cs, cs, :]
        out.append((path, float(np.abs(a - b).max())))
    return out


def _report_timing(args, times, jax, label):
    """Per-step wall time: median over the timed steps of the CROSS-RANK
    MAX (every rank times its own step; the slowest rank owns the step),
    printed with the rank count and node count -- a ladder row."""
    import numpy as np
    timed = np.asarray(times[args.steps:])
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils as mhu
        import jax.numpy as jnp
        allr = np.asarray(mhu.process_allgather(jnp.asarray(timed)))
        per_step_max = allr.max(axis=0)
        slowest = int(np.argmax(allr.mean(axis=1)))
    else:
        per_step_max, slowest = timed, 0
    print(f"[m6] TIMING {label}: ranks={jax.process_count()} nodes="
          f"{os.environ.get('SLURM_JOB_NUM_NODES', '?')} C{args.n} km={args.km} "
          f"steps={len(timed)}: p50 {np.median(per_step_max):.3f} s/step "
          f"(cross-rank max per step; p90 {np.percentile(per_step_max, 90):.3f}"
          f", max {per_step_max.max():.3f}, min {per_step_max.min():.3f}); "
          f"slowest rank {slowest} host "
          f"{os.uname().nodename if slowest == jax.process_index() else '?'}; "
          f"per-step host readback (nsplt guard) is inside every step")


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
    ap.add_argument("--nq", type=int, default=1,
                    help="passenger tracers in the IC: sphum plus nq-1 "
                         "longitude-modulated copies (nonzero, distinct); "
                         "a reference npz must carry the same nq")
    ap.add_argument("--terminator", action="store_true",
                    help="append the oracle's DCMIP16 terminator pair cl, cl2 "
                         "after the --nq passengers (the tracer set the "
                         "run_hydro_*_term_gfs oracle decks advect)")
    ap.add_argument("--gpu-max-rel-peak", type=float, default=None,
                    help="GPU rows (user call 2026-09-15): a leaf that is not "
                         "bitwise still passes if its max |window - ref| is "
                         "within this fraction of the leaf's peak. Two XLA:GPU "
                         "programs differ at rounding (measured 1.5e-14 of "
                         "peak on u/v, C24 kt=1 vs the 6-GPU flat reference, "
                         "job 9789931) and the gap grows with steps (2.6e-13 on "
                         "C48 u/v at step 2, ladder 9922513); the launchers "
                         "pass 1e-12 (user 2026-09-23). CPU rows leave it "
                         "unset and stay bitwise")
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
    ap.add_argument("--flat-ref", action="store_true",
                    help="no window arm: run the FLAT model face-sharded "
                         "P('face') on 6 devices (6 processes under "
                         "--distributed) and --save-npz its outputs -- the "
                         "certified reference every ladder row compares to")
    ap.add_argument("--diag-no-pad-refresh", action="store_true",
                    help="DIAGNOSTIC: skip the per-firing pad refresh (pads "
                         "refreshed at substep entry only); results are "
                         "WRONG, the bitwise gate is expected to fail and "
                         "the timing is printed as a decomposition arm, "
                         "never as a ladder row")
    ap.add_argument("--profile", type=str, default=None,
                    help="directory: jax.profiler trace (perfetto JSON) of "
                         "2 extra steps run AFTER the timed ones on process "
                         "0 (never inside the timing rows), summarised "
                         "per op by scripts/validate/fv3_native/"
                         "summarize_perfetto_trace.py")
    ap.add_argument("--pack-pad-refresh", action="store_true",
                    help="M8-A arm: one ppermute per direction per refresh "
                         "round for all the arrays of a firing (a pure "
                         "re-packing; bitwise-gated like any row)")
    ap.add_argument("--refresh-band", type=int, default=None,
                    help="M8-C arm: per-firing pad refresh restricted to "
                         "the face-edge bands of this depth (ng+1 is the "
                         "certified value; ng-1 must FAIL the gate)")
    ap.add_argument("--unfused-entry-refresh", action="store_true",
                    help="restore the three separate substep-entry "
                         "refreshes (bitwise, attribution arm for the "
                         "fused one)")
    ap.add_argument("--diag-no-entry-refresh", action="store_true",
                    help="DIAGNOSTIC: skip the substep-ENTRY full refresh "
                         "(results are WRONG; splits the residual between "
                         "that refresh and the per-firing shard_map "
                         "boundary) -- never a ladder row")
    ap.add_argument("--timing", type=int, default=0,
                    help="after the gated steps, run this many more steps "
                         "and report wall time per step (max over ranks, "
                         "exclusive nodes only) -- REFUSED unless the gated "
                         "steps were bitwise")
    args = ap.parse_args(argv)

    if args.distributed and not args.flat_ref and not args.ref_npz:
        print("[m6] REFUSED: --distributed needs a reference -- either "
              "--flat-ref (6 processes build it here) or --ref-npz "
              "<saved>; without one there is nothing to gate against")
        raise SystemExit(2)
    if args.gpu_max_rel_peak is not None and not (0.0 < args.gpu_max_rel_peak < 1e-6):
        print(f"[m6] REFUSED: --gpu-max-rel-peak {args.gpu_max_rel_peak} is not a "
              f"rounding-level ceiling (0, 1e-6)")
        raise SystemExit(2)
    if args.ref_npz and not args.distributed:
        # codex 2026-09-13: an in-process run builds its own reference and
        # the file was silently ignored -- a no-op that looked like a gate
        print("[m6] REFUSED: --ref-npz is only read by --distributed runs; "
              "an in-process run scores against the reference it builds "
              "itself and would ignore the file")
        raise SystemExit(2)
    import jax
    import jax.numpy as jnp
    if args.distributed:
        import os as _os
        ids = _os.environ.get("JAX_LOCAL_DEVICE_IDS")   # GPU: "0" = one GPU per task
        jax.distributed.initialize(
            initialization_timeout=900,
            local_device_ids=[int(i) for i in ids.split(",")] if ids else None)
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    rank0 = jax.process_index() == 0
    _runtime_banner(jax)

    def rss(stage):
        """peak resident set of THIS process so far (MB) -- the 216-rank
        C192 row was OOM-killed at 24.6 GB/rank (job 9648703); this says
        which stage's cost scales with the GLOBAL grid on every rank"""
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        import time as _t
        stage = f"{stage} @{_t.strftime('%H:%M:%S')}"
        if args.distributed:
            # every rank's peak: job 9648882 (216 ranks) showed ONE rank at
            # 27.9 GB while the mean was 5 GB -- the growth is not uniform
            from jax.experimental import multihost_utils as mhu
            allr = np.asarray(mhu.process_allgather(jnp.asarray(mb)))
            if rank0:
                print(f"[m6] rss {stage}: rank0 {mb:.0f} MB; over ranks "
                      f"mean {allr.mean():.0f} max {allr.max():.0f} "
                      f"(rank {int(allr.argmax())}) min {allr.min():.0f} MB; "
                      f"ranks >2x mean: "
                      f"{np.flatnonzero(allr > 2 * allr.mean()).tolist()}")
        elif rank0:
            print(f"[m6] rss {stage}: {mb:.0f} MB (rank 0 peak so far)")

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
          f"steps={args.steps} dt={args.dt} n_split={args.n_split} nq={args.nq} "
          f"terminator={args.terminator} "
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

    rss("start")
    grid = create_fv3_duo_grid(args.n)
    rss("grid")
    cfg = FV3DuoConfig(km=args.km, hydrostatic=not args.nh,
                       n_split=args.n_split)

    # reference: faces, GSPMD-sharded P('face') on 6 devices (in-process
    # only: a multi-process job runs ONE program on all its processes),
    # or -- --flat-ref -- the SAME flat model as the distributed reference
    # job on 6 processes, whose outputs are saved for the ladder rows
    ref_model = ref = None
    if args.flat_ref:
        if jax.device_count() != 6:
            print(f"[m6] REFUSED: --flat-ref needs exactly 6 devices, got "
                  f"{jax.device_count()}")
            return 2
        mesh6 = Mesh(np.array(jax.devices()), ("face",))
        sh6 = NamedSharding(mesh6, P("face"))
        ref_model = FV3DuoDynamicsModel(grid, cfg, step_out_shardings=sh6,
                                        step_face_batched=True)
        ref = ref_model.dcmip16_initial_state(do_pert=True, n_tracers=args.nq,
                                   terminator=args.terminator)
        ref = jax.tree_util.tree_map(
            lambda a: jax.device_put(a, sh6) if hasattr(a, "ndim")
            and a.ndim >= 3 and a.shape[0] == 6 else a, ref)
        to_save = {}

        def host(bundle):
            if args.distributed:
                from jax.experimental import multihost_utils as mhu
                bundle = jax.tree_util.tree_map(
                    lambda a: np.asarray(mhu.process_allgather(a, tiled=True))
                    if hasattr(a, "sharding") else a, bundle)
            return {p: np.asarray(a) for p, a in leaves(bundle)}
        to_save.update({f"ic:{p}": a for p, a in host(ref).items()})
        times = []
        for it in range(args.steps + args.timing):
            t0 = time.perf_counter()
            ref = ref_model.step(ref, args.dt)
            jax.block_until_ready(ref)
            times.append(time.perf_counter() - t0)
            if it < args.steps:
                to_save.update({f"step{it + 1}:{p}": a
                                for p, a in host(ref).items()})
            print(f"[m6] flat-ref step {it + 1}: nsplt="
                  f"{ref_model.last_nsplt.tolist()} wall={times[-1]:.3f}s")
            if (np.asarray(ref_model.last_nsplt) < 1).any():
                print(f"[m6] REFUSED: flat reference nsplt < 1 at step "
                      f"{it + 1} = NaN Courant; the deck is not stable at "
                      f"dt={args.dt} on C{args.n} (no reference saved)")
                return 2
        if args.save_npz and rank0:
            np.savez(args.save_npz, **to_save)
            print(f"[m6] saved {len(to_save)} flat outputs -> {args.save_npz}")
        if args.timing:
            _report_timing(args, times, jax, "flat faces x6")
        return 0
    if not args.distributed:
        mesh6 = Mesh(np.array(jax.devices()[:6]), ("face",))
        sh6 = NamedSharding(mesh6, P("face"))
        ref_model = FV3DuoDynamicsModel(grid, cfg, step_out_shardings=sh6,
                                        step_face_batched=True)
        ref = ref_model.dcmip16_initial_state(do_pert=True, n_tracers=args.nq,
                                   terminator=args.terminator)
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
    if args.refresh_band is not None and args.pack_pad_refresh:
        # 2026-09-09: this pair used to abort (MPI_ERR_TRUNCATE on 54 ranks,
        # an XLA collective-permute rendezvous check in one process) and was
        # refused here.  It no longer reproduces in either place -- the
        # program changed since (the flux barrier now fires once per step,
        # not once per level), so the failing executable is gone.  Allowed
        # again, with the arms named in the log.
        print("[m6] arms: band refresh for the firings, packed full refresh "
              "at substep entry")
    if args.refresh_band is not None:
        win_model._window_comm.refresh_band = args.refresh_band
        print(f"[m6] M8-C arm: band-restricted refresh, depth "
              f"{args.refresh_band}")
    if args.pack_pad_refresh:
        win_model._window_comm.pack_pad_refresh = True
        print("[m6] M8-A arm: packed pad refresh")
    if args.unfused_entry_refresh:
        win_model._window_comm.fuse_entry_refresh = False
        print("[m6] ARM: three separate substep-entry refreshes")
    if args.diag_no_entry_refresh:
        win_model._window_comm.entry_refresh = False
        print("[m6] DIAGNOSTIC ARM: substep-entry refresh OFF -- timing "
              "only, results not a ladder row")
    if args.diag_no_pad_refresh:
        win_model._window_comm.pad_refresh_per_firing = False
        print("[m6] DIAGNOSTIC ARM: per-firing pad refresh OFF -- timing "
              "only, results not a ladder row")
    print(f"[m6] {lay}; barriers {win_model._window_comm.barrier_mode}")
    rss("window model built")
    win = win_model.dcmip16_initial_state(do_pert=True, n_tracers=args.nq,
                                   terminator=args.terminator)
    rss("window IC")

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
    # completeness BOTH ways (codex 2026-09-05): a reference missing a leaf
    # the window arm has (a truncated npz) must refuse, not pass
    extra = sorted(set(ic_flat) - {p for p, _ in ref_ic})
    if extra:
        print(f"  IC: reference lacks window leaves {extra} -- REFUSED")
        bad += 1
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

    # Name the three boundaries that were indistinguishable in the old log.
    # lower().compile() warms the same jit cache used by model.step, leaving
    # the gate's step and scoring logic unchanged.
    inner = next((cell.cell_contents
                  for cell in (win_model._step_fn.__closure__ or ())
                  if hasattr(cell.cell_contents, "lower")), None)
    if inner is None:
        raise RuntimeError("M6 step wrapper contains no jitted callable")
    lower_args = (win["state"], win["press"], win["q"], args.dt,
                  win["omga"], win["nh"])
    phase_start = time.perf_counter()
    _phase_event(jax.process_index(), "lower", "begin")
    lowered = inner.lower(*lower_args)
    _phase_event(jax.process_index(), "lower", "complete", phase_start)
    phase_start = time.perf_counter()
    _phase_event(jax.process_index(), "compile", "begin")
    _compiled_step = lowered.compile()
    _phase_event(jax.process_index(), "compile", "complete", phase_start)

    rc = 0
    times = []
    prev_flat = ic_flat
    for it in range(args.steps):
        t0 = time.perf_counter()
        if it == 0:
            _phase_event(jax.process_index(), "first_dispatch", "begin")
        win = win_model.step(win, args.dt)
        if it == 0:
            _phase_event(jax.process_index(), "first_dispatch", "submitted")
        jax.block_until_ready(win)
        if it == 0:
            _phase_event(jax.process_index(), "first_dispatch", "complete", t0)
        times.append(time.perf_counter() - t0)
        rss(f"window step {it + 1}")
        # GLM 2026-09-05: the per-rank sub-cycle schedule must be IDENTICAL
        # on every process (a local max would silently under-advect) --
        # asserted with an allgather, not just printed
        ns = np.asarray(win_model.last_nsplt)
        if (ns < 1).any():
            print(f"  step {it + 1}: nsplt {ns.tolist()} < 1 = NaN Courant "
                  f"(floor(1 + cmax) is >= 1 for any finite cmax) -- REFUSED")
            rc = 1
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils as mhu
            import jax.numpy as jnp
            alln = np.asarray(mhu.process_allgather(jnp.asarray(ns)))
            if not np.all(alln == alln[0]):
                print(f"  step {it + 1}: nsplt DIFFERS across ranks: "
                      f"{alln.tolist()} -- REFUSED")
                rc = 1
        print(f"[m6] rank {jax.process_index()} step {it + 1}: nsplt="
              f"{ns.tolist()} wall={times[-1]:.3f}s")
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
        n_ok = n_bad = n_ok_tol = 0
        worst = worst_peak = 0.0
        # GLM 2026-09-13: "sharded == in-process" is vacuous for a tracer
        # BOTH arms leave untouched, so every passenger must have moved on
        # the compute window since the previous scored state
        for path, moved in tracer_window_change(wf, prev_flat, grid.n,
                                                grid.ng):
            print(f"  step {it + 1} {path}: max window change {moved:.3e}")
            if not moved > 0.0:
                print(f"  step {it + 1} {path}: tracer did not move on the "
                      f"compute window -- REFUSED")
                n_bad += 1
        prev_flat = wf
        extra = sorted(set(wf) - {p for p, _ in ref_leaves})
        if extra:
            print(f"  step {it + 1}: reference lacks window leaves {extra} "
                  f"-- REFUSED")
            n_bad += 1
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
            # a NaN state compares "bitwise" to another NaN state, so a
            # blown-up deck would certify vacuously (C192 at the C48 dt of
            # 900 s: reference nsplt 0 at step 2, i.e. NaN Courant; 2026-09-06)
            nonfinite = int((~np.isfinite(r)).sum() + (~np.isfinite(w)).sum())
            if nonfinite and r.dtype.kind == "f":
                print(f"  step {it + 1} {path}: {nonfinite} non-finite values "
                      f"(reference {int((~np.isfinite(r)).sum())}, window "
                      f"{int((~np.isfinite(w)).sum())}) -- REFUSED, the deck "
                      f"is not stable at this dt")
                n_bad += 1
                continue
            d = (r != w).sum()
            if d:
                fin = np.isfinite(r - w)
                rel = np.nanmax(np.abs(r - w)[fin] /
                                (np.abs(r[fin]) + 1e-300))
                # per-cell rel explodes on cells where the field is ~0
                # (v is exactly 0 in the IC: a 1e-16 rounding difference
                # read as "12x"); the certificate quantity on GPU is the
                # difference over the LEAF'S PEAK
                peak = float(np.abs(r[fin]).max()) if fin.any() else 0.0
                rel_peak = (float(np.abs(r - w)[fin].max()) / peak
                            if peak > 0 else float("inf"))
                worst = max(worst, float(rel))
                worst_peak = max(worst_peak, rel_peak)
                if args.gpu_max_rel_peak is not None \
                        and rel_peak <= args.gpu_max_rel_peak:
                    n_ok_tol += 1
                    tag = "within GPU ceiling"
                else:
                    n_bad += 1
                    tag = "DIFF"
                print(f"  step {it + 1} {path:24s} {tag} {int(d)}/{r.size} "
                      f"max rel {rel:.3e}  rel-to-peak {rel_peak:.3e}")
            else:
                n_ok += 1
        print(f"[m6] step {it + 1}: {n_ok}/{n_expected} leaves BITWISE"
              + (f", {n_ok_tol} within the GPU ceiling {args.gpu_max_rel_peak:g} "
                 f"of peak" if args.gpu_max_rel_peak is not None else "")
              + f", {n_bad} differ (worst rel {worst:.3e}, worst rel-to-peak "
              f"{worst_peak:.3e}); replicated window outputs: "
              f"{replicated or 'none'}")
        if n_bad or n_ok + n_ok_tol != n_expected:
            rc = 1
        if replicated:
            print("  REFUSED: a window output came back replicated")
            rc = 1
    if args.save_npz and rank0:
        np.savez(args.save_npz, **to_save)
        print(f"[m6] saved {len(to_save)} flat outputs -> {args.save_npz}")
    if args.timing:
        if jax.process_count() > 1:
            # every rank must take the SAME branch (codex 2026-09-05): a
            # rank whose reference file differs would otherwise skip the
            # timed collectives while the others enter them
            from jax.experimental import multihost_utils as mhu
            import jax.numpy as jnp
            rc = int(np.asarray(mhu.process_allgather(
                jnp.asarray(rc, dtype=jnp.int32))).max())
        diag_arm = args.diag_no_pad_refresh or args.diag_no_entry_refresh
        if rc != 0 and not diag_arm:
            print("[m6] TIMING REFUSED: the gated steps were not bitwise on "
                  "every rank -- a rank-count row on a wrong exchange is not "
                  "reported")
        else:
            timing_ok = True
            for it in range(args.timing):
                t0 = time.perf_counter()
                win = win_model.step(win, args.dt)
                jax.block_until_ready(win)
                times.append(time.perf_counter() - t0)
                # A TIMED STEP IS STILL A STEP (codex 2026-09-16): the
                # scored steps refuse a non-finite state, the timed ones did
                # not, so a deck that blew up AFTER the last scored step
                # still printed a p50 -- timing a diverged model. Every rank
                # must reach the SAME verdict before any of them leaves the
                # loop, or the ranks desynchronise on the next collective.
                # OWNED cells only, the same view the scored steps use: a
                # window's outer pad rows are non-finite BY DESIGN (the
                # stencil-reach cells no neighbour fills), so counting raw
                # shards refused every kt>=2 row while kt=1 (no seam pads)
                # passed -- ladders 9867744/5, 715-1485 "values" per rank
                # on a state whose owned cells scored within the ceiling.
                # flat_leaves allgathers, so every rank sees the same count.
                bad = int(sum(
                    int((~np.isfinite(np.asarray(v))).sum())
                    for _p, v in flat_leaves(win)
                    if getattr(v, "dtype", None) is not None
                    and v.dtype.kind == "f"))
                ns_t = np.asarray(win_model.last_nsplt)
                local_bad = int(bad > 0 or (ns_t < 1).any())
                if jax.process_count() > 1:
                    import jax.numpy as jnp
                    from jax.experimental import multihost_utils as mhu
                    local_bad = int(np.asarray(mhu.process_allgather(
                        jnp.asarray(local_bad, dtype=jnp.int32))).max())
                if local_bad:
                    print(f"[m6] TIMING REFUSED: timed step {it + 1} left a "
                          f"non-finite OWNED state on some rank ({bad} "
                          f"values, nsplt {ns_t.tolist()}) -- the deck is not "
                          f"stable at dt={args.dt}, so the timing would be of "
                          f"a diverged model")
                    rc = 1
                    timing_ok = False
                    break
            if args.profile and timing_ok:
                # 2 EXTRA steps after the timed ones (codex 2026-09-05:
                # profiled steps must not enter the timing distribution)
                if rank0:
                    jax.profiler.start_trace(args.profile,
                                             create_perfetto_trace=True)
                for _ in range(2):
                    win = win_model.step(win, args.dt)
                    jax.block_until_ready(win)
                if rank0:
                    jax.profiler.stop_trace()
                    print(f"[m6] profile trace (2 extra steps after the "
                          f"timed ones, rank 0) -> {args.profile}")
            if timing_ok:
                _report_timing(args, times, jax,
                               f"windows kt={args.kt} pad={args.pad}"
                               + (" DIAGNOSTIC no-per-firing-refresh (NOT a "
                                  "ladder row)" if diag_arm
                                  else ""))
    if rc == 0 and args.gpu_max_rel_peak is not None:
        verdict = f"WITHIN {args.gpu_max_rel_peak:g} OF PEAK (GPU ceiling)"
    else:
        verdict = "BITWISE" if rc == 0 else "DIFFERS"
    print(f"[m6] VERDICT kt={args.kt} pad={args.pad} steps={args.steps}: "
          f"{verdict} vs "
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
