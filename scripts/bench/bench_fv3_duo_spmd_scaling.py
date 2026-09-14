"""FV3 duo SPMD scaling: sharded (6 GPU) vs single-device wall time.

The one regime the 2-GPU verdict left open: multi-NODE aggregate
bandwidth (3 nodes x 2 GPU = 6 devices, 1 face/device).  Answers whether
spreading the six faces over three nodes' inter-node links beats a single
card, at a fixed problem size.

TWO ISOLATED MODES, run as SEPARATE launches (codex BLOCKER: timing both
arms in one process on one card lets the second arm inherit the first's
warm/fragmented allocator + clock state, biasing the ratio):

  --mode sharded : srun 6 tasks = 6 global devices, face-sharded + ring +
                   face-batched step.  Per sample: global barrier ->
                   perf_counter -> step -> block the WHOLE pytree ->
                   gather every rank's elapsed -> take the MAX (a
                   collective step's wall is the SLOWEST rank).  Reports
                   the median of per-sample maxima.
  --mode single  : ONE process, ONE GPU, unsharded model, no collectives.
                   Per sample: perf_counter -> step -> block the WHOLE
                   pytree.  Median.

The sbatch runs both and the [SCALING] lines are combined by hand /
downstream: speedup = single_ms / sharded_ms (>1 = the 6-GPU run is
FASTER = crossover).  Every run prints its resolved topology + card so a
mislaunch cannot masquerade as the intended 6-device measurement, and
refuses to report if any timed output leaf decayed from the face
sharding (non-vacuity).

Each mode reports TWO numbers (GLM mechanism review): a THROUGHPUT
number (K back-to-back steps, one final block, /K -- how a production
scan runs, and the one the scaling VERDICT uses) and a per-step
LATENCY median+mean (block every step -- eager latency, which
pessimizes the sharded arm's many small collective kernels the scan
would hide).  Use throughput for the FASTER/SLOWER call.

CAVEAT (GLM, not corrected here): the single-device arm runs 1 GPU on
a node while the sharded arm runs 2 GPUs/node -- two busy GPUs downclock
(DVFS), which biases AGAINST the sharded arm (understates its speedup).
That is the CONSERVATIVE direction for a FASTER verdict: a sharded win
measured here is real; a sharded loss could be partly DVFS.  Pinning
clocks (nvidia-smi -lgc) needs privileges we do not assume.

Correctness is owned by spmd_multiprocess_parity + the restart
round-trip; this is wall time only, but it asserts the SAME sharded
executable those gates certified is what gets timed (topology + output
sharding checks), and stamps the git SHA + config so the number is
traceable.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time


def _log(msg):
    print(f"[pid={os.getpid()} PROCID={os.environ.get('SLURM_PROCID')}] {msg}",
          flush=True, file=sys.stderr)


def _block_all(bundle):
    """Block until EVERY leaf of the bundle is ready (codex BLOCKER:
    blocking one leaf lets the rest stay in flight and reads the step
    artificially low -- differently for the two executables)."""
    import jax
    for leaf in jax.tree_util.tree_leaves(bundle):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _build(resolution, km):
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(resolution)
    cfg = FV3DuoConfig(km=km)
    return grid, cfg, FV3DuoDynamicsModel, np


def _assert_face_sharded(bundle, shard, when):
    """Refuse to report if any output leaf silently replicated (codex
    MAJOR: a scaling verdict on a replicated fallback is meaningless).
    Mirrors spmd_face_shard_parity's non-vacuity check -- is_fully_
    replicated is False AND the device->index map equals the requested
    face sharding's, per RAW leaf."""
    want = shard.devices_indices_map
    leaves = (list(bundle["state"].values())
              + list(bundle["press"].values()) + list(bundle["q"])
              + [bundle["omga"]]
              + (list(bundle["nh"].values()) if bundle.get("nh") else []))
    for leaf in leaves:
        sh = leaf.sharding
        if sh.is_fully_replicated or (
                sh.devices_indices_map(leaf.shape) != want(leaf.shape)):
            raise SystemExit(
                f"VACUOUS ({when}): a timed output leaf decayed from the "
                f"face sharding -- the scaling number would not describe "
                f"the sharded executable. shape={leaf.shape} sharding={sh}")


def _sha():
    try:
        from legoesm.io.git_provenance import git_provenance
        from pathlib import Path
        return git_provenance(Path(__file__)).commit[:12]
    except Exception:
        return "unknown"


def _run_sharded(args):
    import jax
    jax.distributed.initialize(initialization_timeout=180)
    jax.config.update("jax_enable_x64", True)
    from jax.experimental import multihost_utils as mhu
    rank = jax.process_index()
    nproc = jax.process_count()
    _log(f"init OK: process {rank}/{nproc}, local {jax.local_devices()}")

    # Enforce the advertised topology (codex MAJOR): exactly 6 global
    # devices, 6 processes, 1 device/process -- otherwise a mislaunch
    # (1/2/3 devices) would print a valid-looking but wrong-topology line.
    if jax.device_count() != 6 or nproc != 6 or jax.local_device_count() != 1:
        raise SystemExit(
            f"sharded mode needs 6 global devices / 6 processes / 1 "
            f"local device each (3 nodes x 2 GPU); got "
            f"device_count={jax.device_count()} process_count={nproc} "
            f"local_device_count={jax.local_device_count()}")

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    grid, cfg, Model, np = _build(args.resolution, args.km)
    mesh = Mesh(np.array(jax.devices()), ("face",))
    shard = NamedSharding(mesh, P("face"))
    model = Model(grid, cfg, step_out_shardings=shard, step_spmd_mesh=mesh,
                  step_face_batched=True)
    ic = model.dcmip16_initial_state(do_pert=True)

    def _put(x):
        h = np.asarray(x)
        return jax.make_array_from_callback(h.shape, shard, lambda idx: h[idx])

    def _global(bundle):
        out = {"state": {k: _put(v) for k, v in bundle["state"].items()},
               "press": {k: _put(v) for k, v in bundle["press"].items()},
               "q": [_put(q) for q in bundle["q"]],
               "omga": _put(bundle["omga"])}
        out["nh"] = (None if bundle.get("nh") is None else
                     {k: _put(v) for k, v in bundle["nh"].items()})
        return out

    b = _global(ic)
    for _ in range(args.n_warmup):
        b = model.step(b, args.dt)
    _block_all(b)
    _assert_face_sharded(b, shard, "after warmup")

    def _cross_rank_max(local_sec):
        allr = np.asarray(mhu.process_allgather(jax.numpy.asarray(local_sec)))
        return float(allr.max())               # slowest rank == wall

    # (1) PER-STEP LATENCY: block every step. Answers eager per-step
    # latency (NOT how a production scan runs -- see (2)).
    maxima = []
    for _ in range(args.n_timed):
        mhu.sync_global_devices("bench_lat")   # barrier BEFORE t0
        t0 = time.perf_counter()
        b = model.step(b, args.dt)
        _block_all(b)
        maxima.append(_cross_rank_max(time.perf_counter() - t0))

    # (2) THROUGHPUT: K back-to-back steps, ONE final block, /K -- the
    # number a scaling VERDICT should use (GLM: per-step blocking drains
    # the pipeline and pessimizes the sharded arm's many small collective
    # kernels, which a real loop hides step-over-step; the b->b data
    # dependency keeps K-step timing honest).
    K = args.n_timed
    mhu.sync_global_devices("bench_thr")
    t0 = time.perf_counter()
    for _ in range(K):
        b = model.step(b, args.dt)
    _block_all(b)
    thr = _cross_rank_max(time.perf_counter() - t0) / K
    _assert_face_sharded(b, shard, f"after timing ({args.n_timed}+{K} steps)")

    if rank == 0:
        lat_med = statistics.median(maxima)
        lat_mean = statistics.mean(maxima)
        card = str(jax.local_devices()[0])
        print(f"[SCALING] mode=sharded C{args.resolution} km={args.km} "
              f"6dev(6proc) {card} sha={_sha()}: "
              f"throughput {thr * 1e3:.2f} ms/step (K={K} back-to-back) | "
              f"latency median {lat_med * 1e3:.2f} mean {lat_mean * 1e3:.2f} "
              f"ms/step (per-step block, n={args.n_timed}, cross-rank max)",
              flush=True)
    return 0


def _run_single(args):
    # ONE process, ONE GPU, no jax.distributed -> no collectives, fully
    # isolated from the sharded launch (separate job).
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.local_device_count() < 1:
        raise SystemExit("single mode needs a visible local device")
    _log(f"single mode: local {jax.local_devices()}")
    grid, cfg, Model, np = _build(args.resolution, args.km)
    model = Model(grid, cfg)                    # unsharded
    ic = model.dcmip16_initial_state(do_pert=True)

    try:
        b = ic
        for _ in range(args.n_warmup):
            b = model.step(b, args.dt)
        _block_all(b)
        # (1) per-step latency (block each step)
        per = []
        for _ in range(args.n_timed):
            t0 = time.perf_counter()
            b = model.step(b, args.dt)
            _block_all(b)
            per.append(time.perf_counter() - t0)
        # (2) throughput: K back-to-back steps, one final block, /K --
        # the matched-protocol number the sharded arm also reports.
        K = args.n_timed
        t0 = time.perf_counter()
        for _ in range(K):
            b = model.step(b, args.dt)
        _block_all(b)
        thr = (time.perf_counter() - t0) / K
    except Exception as exc:
        # NARROW report (codex MAJOR: a broad "OOM" swallows real bugs).
        # Name the exception type + message so a shape/config error is not
        # mislabeled as capacity; still non-zero exit.
        _log(f"single-device arm FAILED: {type(exc).__name__}: {exc}")
        print(f"[SCALING] mode=single C{args.resolution} km={args.km} "
              f"FAILED {type(exc).__name__}", flush=True)
        raise SystemExit(2)

    if args.dump_state:
        # the FINAL state after n_warmup + 2*n_timed steps, every array
        # leaf, so two single-mode runs under different XLA flags can be
        # diffed bitwise (the autotuner-off question, 2026-09-14)
        leaves = {jax.tree_util.keystr(k): np.asarray(a)
                  for k, a in jax.tree_util.tree_leaves_with_path(b)
                  if hasattr(a, "ndim")}
        np.savez(args.dump_state, **leaves)
        _log(f"dumped {len(leaves)} state leaves -> {args.dump_state}")
    lat_med = statistics.median(per)
    lat_mean = statistics.mean(per)
    card = str(jax.local_devices()[0])
    print(f"[SCALING] mode=single C{args.resolution} km={args.km} "
          f"1dev {card} sha={_sha()}: "
          f"throughput {thr * 1e3:.2f} ms/step (K={K} back-to-back) | "
          f"latency median {lat_med * 1e3:.2f} mean {lat_mean * 1e3:.2f} "
          f"ms/step (per-step block, n={args.n_timed})", flush=True)
    return 0


def _run_bare(args):
    """BARE COLLECTIVE microbenchmark -- the model removed entirely.

    The ranks ladder measures a real exchange; when its cost grew with rank
    count the cause could be the transport, the collectives layer, or the
    model's own work.  This times a SINGLE ``ppermute`` of one small array
    on the same mesh, so whatever it shows is the runtime's collective and
    nothing else.  Reported as microseconds per round-trip next to the
    payload, the rank count and the node spread.
    """
    import jax

    if not args.single_process:
        # one GPU per task: the cgroup hides the node's other GPU but the
        # driver still counts it (tiled_m6_model_gate_gpu.sbatch, job
        # 9650417) -- honour the same pin the gate uses
        ids = os.environ.get("JAX_LOCAL_DEVICE_IDS")
        jax.distributed.initialize(
            initialization_timeout=600,
            local_device_ids=[int(i) for i in ids.split(",")] if ids else None)
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.experimental import multihost_utils as mhu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    world = jax.device_count()
    rank = jax.process_index()
    mesh = Mesh(np.array(jax.devices()), ("r",))
    perm = [(i, (i + 1) % world) for i in range(world)]   # neighbour ring
    nel = max(1, args.payload_kb * 1000 // 8)

    def body(x):
        return jax.lax.ppermute(x, "r", perm)

    fn = jax.jit(shard_map(body, mesh=mesh, in_specs=P("r"),
                           out_specs=P("r"), check_rep=False))
    shard = NamedSharding(mesh, P("r"))
    # Row i carries the value i, so the ring's receiver can CHECK that it got
    # its left neighbour's row: a no-op or mis-mapped permute would otherwise
    # time as a plausible fast number.
    host = np.arange(world, dtype=np.float64)[:, None] * np.ones((1, nel))
    xs = jax.make_array_from_callback(host.shape, shard, lambda idx: host[idx])
    for _ in range(5):
        y = fn(xs)
    jax.block_until_ready(y)
    ok = all(np.all(np.asarray(sh.data) == (sh.index[0].start - 1) % world)
             for sh in y.addressable_shards)
    # COLLECTIVE verdict: every rank learns whether ANY rank failed before
    # anyone returns, so a failing rank cannot leave the others hanging in
    # the next collective (codex 2026-09-04).
    if args.single_process:
        all_ok = ok
    else:
        all_ok = bool(np.asarray(mhu.process_allgather(
            jnp.asarray(ok))).all())
    if not all_ok:
        _log(f"[BARE] REFUSED: rank {rank} ok={ok}; some rank did not "
             f"receive its left neighbour's row -- the permute is not "
             f"delivering the data it is being timed on")
        if not args.single_process:
            mhu.sync_global_devices("bare_refused")
            jax.distributed.shutdown()
        return 1

    def _cross_max(sec):
        if args.single_process:
            return float(sec)
        return float(np.asarray(mhu.process_allgather(
            jnp.asarray(sec))).max())

    samples, local = [], []
    for _ in range(args.n_timed):
        if not args.single_process:
            mhu.sync_global_devices("bare")
        t = time.perf_counter()
        y = fn(xs)
        jax.block_until_ready(y)
        local.append(time.perf_counter() - t)
        samples.append(_cross_max(local[-1]))
    us = float(np.median(samples)) * 1e6
    # Per-rank receipt: the headline is a cross-rank MAX, so one slow node
    # owns the whole number.  Name it (job 9621213: 216 ranks on 36 nodes read
    # 30 ms while 18 nodes read 0.17 ms -- unattributable without this).
    my_med = float(np.median(local)) * 1e6
    if args.single_process:
        meds = np.array([my_med])
    else:
        meds = np.asarray(mhu.process_allgather(jnp.asarray(my_med)))
    worst = int(np.argmax(meds))
    if rank == worst:
        _log(f"[BARE-SLOWEST] rank={rank} host={os.uname().nodename} "
             f"per-rank median={my_med:.1f} us")
    if rank == 0:
        print(f"[BARE] backend={jax.default_backend()} ranks={world} "
              f"nodes={os.environ.get('SLURM_JOB_NUM_NODES', '?')} "
              f"payload={nel * 8 / 1000:.1f} KB/rank: ONE neighbour "
              f"ppermute = {us:.1f} us (median of {args.n_timed} cross-rank "
              f"maxima)")
        print(f"[BARE-RANKS] per-rank medians: min={meds.min():.1f} "
              f"median={np.median(meds):.1f} max={meds.max():.1f} us "
              f"(slowest rank {worst})")
    if not args.single_process:
        # Every rank leaves through the coordination service's shutdown
        # barrier.  Without it rank 0 exits first, its service dies, and any
        # rank still polling aborts with "JAX distributed service detected
        # fatal errors" (job 9621213: 10 stragglers on two nodes, exit 1,
        # arm refused after the number was already printed).
        mhu.sync_global_devices("bare_done")
        jax.distributed.shutdown()
    return 0


def _run_ranks(args):
    """VENUE receipt: does the tiled neighbour exchange hold up across many
    RANKS (one device per process), the regime the Fortran reference scales
    in?  Launched with ``srun`` (one python per task), CPU or GPU backend.

    Times the dominant exchange arm (``ext_scalar_allk``, 14 of the 23 hydro
    firings) on the ``(6, kt, kt)`` mesh over the WHOLE world, barrier-
    synchronised, reporting the median of per-sample cross-rank MAXIMA (a
    collective's wall is its slowest rank).  Near-linear compute scaling
    requires this number to stay ~FLAT as ranks grow (neighbour-only
    traffic); a rising curve is the fabric, and it is the venue verdict.

    CORRECTNESS TRACKS SCALE (non-negotiable, and cheap here): the flat
    input is replicated on every rank, so every rank independently
    recomputes the CERTIFIED single-device exchange and compares its own
    result BITWISE.  A bench that cannot show its answer is right is not a
    receipt; a rank-count row is REFUSED if the check fails.
    """
    import jax

    # --single-process: the INSTRUMENT DISCRIMINATOR. The identical check
    # on virtual devices in ONE process, so a refusal in the multi-process
    # run can be attributed to the multi-process path rather than to this
    # probe's own reference/slicing (validate the instrument first).
    if not args.single_process:
        jax.distributed.initialize(initialization_timeout=600)
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from jax.experimental import multihost_utils as mhu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    rank, nproc = jax.process_index(), jax.process_count()
    world = jax.device_count()
    _log(f"init OK: process {rank}/{nproc}, world {world} devices, "
         f"backend {jax.default_backend()}, local {jax.local_devices()}")
    kt = args.tiled
    need = 6 * kt * kt
    if world != need or (not args.single_process
                         and jax.local_device_count() != 1):
        raise SystemExit(
            f"ranks mode kt={kt} needs {need} global devices, 1 per "
            f"process; got world={world} nproc={nproc} "
            f"local={jax.local_device_count()}")

    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids import fv3_duo_halos as H

    grid = create_fv3_duo_grid(args.resolution)
    mesh = Mesh(np.array(jax.devices()).reshape(6, kt, kt),
                ("face", "tile_i", "tile_j"))
    t0 = time.perf_counter()
    ctx = build_jax_duo_stepper_context(grid.ctx_np, spmd_mesh=mesh)
    tab = ctx.tab
    comm = tab.tile_comm
    _log(f"tile_comm built in {time.perf_counter() - t0:.1f}s "
         f"(kt={comm.kt}, nl={comm.nl}, census depth={comm.depth})")

    m = grid.n + 2 * grid.ng
    K = args.km
    x = np.asarray(np.random.default_rng(7).standard_normal((6, m, m, K)))
    shard = NamedSharding(mesh, P("face"))
    xs = jax.make_array_from_callback(x.shape, shard, lambda idx: x[idx])

    fn = jax.jit(lambda a: comm.ext_scalar_allk(a, "A"),
                 out_shardings=shard)
    out = fn(xs)
    jax.block_until_ready(out)
    # every rank is an independent referee: the certified single-device
    # exchange of the same input, computed locally, must match BITWISE.
    # A global array spans non-addressable devices under multi-process,
    # so each rank compares its OWN addressable shard against the same
    # slice of the reference (the repo's multiprocess-parity pattern).
    tab.tile_comm = None
    # the certified path scatters with .at[] -- it needs a jax array, not
    # the host numpy input (jobs 9610142-44 died here, not in the fabric)
    ref = np.asarray(H.ext_scalar_sixface_allk(jax.numpy.asarray(x), tab,
                                               "A"))
    tab.tile_comm = comm
    sh0 = out.addressable_shards[0]
    got = np.asarray(sh0.data)
    ref_slice = ref[sh0.index]
    n_bad = int((got != ref_slice).sum())
    dmax = float(np.nanmax(np.abs(got - ref_slice))) if n_bad else 0.0
    scale = float(np.nanmax(np.abs(ref_slice))) or 1.0
    rel = dmax / scale
    # The reference is EAGER; the timed arm is COMPILED. On CPU those two
    # differ by FMA contraction in the exchange's weighted-stencil cells --
    # measured, not assumed: the M3 purity gate showed <=2.6e-14 relative,
    # confined to weighted-stencil targets, vanishing under
    # XLA_FLAGS=--xla_cpu_max_isa=AVX (jobs 9600012/9600021/9600062). So
    # the venue gate is bitwise ONLY under --bitwise (the pinned runs) and
    # otherwise requires that same envelope; either way a row whose
    # exchange is WRONG (O(field) off) is refused, which is the point.
    ok = (bool(np.array_equal(got, ref_slice, equal_nan=True))
          if args.bitwise else bool(rel <= 1e-13))
    all_ok = ok if args.single_process else bool(np.asarray(
        mhu.process_allgather(
            jax.numpy.asarray(1.0 if ok else 0.0))).min() > 0.5)
    if not all_ok:
        raise SystemExit(
            f"RANKS ROW REFUSED at kt={kt} ({need} ranks): the tiled "
            f"exchange does not match the certified single-device "
            f"exchange on this input (rank {rank}: nbad={n_bad}, "
            f"max rel {rel:.2e}, gate="
            f"{'bitwise' if args.bitwise else '<=1e-13 rel'}) -- a "
            f"timing on a wrong exchange is not a receipt")

    for _ in range(args.n_warmup):
        y = fn(xs)
    jax.block_until_ready(y)

    def _cross_max(sec):
        if args.single_process:
            return float(sec)
        return float(np.asarray(mhu.process_allgather(
            jax.numpy.asarray(sec))).max())

    samples = []
    for _ in range(args.n_timed):
        if not args.single_process:
            mhu.sync_global_devices("ranks_bench")
        t = time.perf_counter()
        y = fn(xs)
        jax.block_until_ready(y)
        samples.append(_cross_max(time.perf_counter() - t))
    med = float(np.median(samples)) * 1e3

    # payload bound per rank per firing: the schedule's own tables
    split = comm.splits["scalar_A"]
    rounds = sum(len(ph.perms) for ph in split.phases)
    payload = sum(int(ph.send_idx.shape[2]) * len(ph.perms)
                  for ph in split.phases)
    mb = payload * K * 8 / 1e6
    if rank == 0:
        nodes = os.environ.get("SLURM_JOB_NUM_NODES", "?")
        print(f"[RANKS] backend={jax.default_backend()} ranks={need} "
              f"kt={kt} nodes~{nodes} C{args.resolution} km={K} "
              f"nl={comm.nl}: exchange {med:.3f} ms/firing (median of "
              f"{args.n_timed} cross-rank maxima), {rounds} ppermute "
              f"rounds, <={mb:.3f} MB/rank/firing, certified on ALL "
              f"ranks vs the certified exchange "
              f"({'BITWISE' if args.bitwise else f'max rel {rel:.1e}'})")
    return 0


def _run_census(args):
    """STRUCTURAL scaling census of the M3 tile lane -- the pre-M4 ceiling,
    measured instead of asserted, and CPU-runnable (no GPU allocation).

    The M3 bridge tiles the EXCHANGE only: the step's state stays FLAT
    six-face and is pinned ``P('face')``, so on a ``(6, kt, kt)`` mesh it
    REPLICATES over the kt^2 tile devices of each face.  This mode reports,
    per state leaf, the PER-DEVICE shard shape at the requested kt and at
    the 6-device baseline.  If per-device elements do not fall while the
    device count grows kt^2, then per-device work does not fall either and
    strong-scaling efficiency is bounded by 1/kt^2 BY CONSTRUCTION -- the
    number M4 (kernel retiling) exists to move.  Reported as a bound, never
    as a timing.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    kt = args.tiled
    need = 6 * kt * kt
    if len(jax.devices()) < need:
        raise SystemExit(
            f"census kt={kt} needs {need} devices; got {len(jax.devices())} "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={need})")
    grid, cfg, Model, np = _build(args.resolution, args.km)
    base_mesh = Mesh(np.array(jax.devices()[:6]), ("face",))
    base_sh = NamedSharding(base_mesh, P("face"))
    tile_mesh = Mesh(np.array(jax.devices()[:need]).reshape(6, kt, kt),
                     ("face", "tile_i", "tile_j"))
    tile_sh = NamedSharding(tile_mesh, P("face"))
    model = Model(grid, cfg, step_out_shardings=tile_sh,
                  step_spmd_mesh=tile_mesh)
    ic = model.dcmip16_initial_state(do_pert=True)

    leaves = []
    for k, v in ic["state"].items():
        leaves.append((f"state.{k}", np.asarray(v).shape))
    for k, v in ic["press"].items():
        leaves.append((f"press.{k}", np.asarray(v).shape))
    for i, q in enumerate(ic["q"]):
        leaves.append((f"q[{i}]", np.asarray(q).shape))
    leaves.append(("omga", np.asarray(ic["omga"]).shape))
    if ic.get("nh") is not None:
        for k, v in ic["nh"].items():
            leaves.append((f"nh.{k}", np.asarray(v).shape))

    def _elems(sh, shape):
        s = sh.shard_shape(shape)
        n = 1
        for d in s:
            n *= d
        return s, n

    tot_base = tot_tile = 0
    print(f"[census] C{args.resolution} km={args.km} kt={kt}: "
          f"{need} devices (mesh {tuple(tile_mesh.devices.shape)}) vs "
          f"6-device baseline; step out_shardings = P('face')")
    print(f"  {'leaf':<16} {'global':<22} {'per-dev @6':<18} "
          f"{'per-dev @' + str(need):<18}")
    for name, shape in leaves:
        sb, nb = _elems(base_sh, shape)
        st, nt = _elems(tile_sh, shape)
        tot_base += nb
        tot_tile += nt
        print(f"  {name:<16} {str(shape):<22} {str(sb):<18} {str(st):<18}")
    ratio = (tot_base / tot_tile) if tot_tile else float("nan")
    ideal = need / 6.0
    print(f"[census] per-device state elements: {tot_base:,} at 6 devices, "
          f"{tot_tile:,} at {need} devices -> shrink factor {ratio:.3f}x "
          f"(ideal for {need} devices would be {ideal:.1f}x)")
    print(f"[census] flat state fully replicated across the tile axes: "
          f"{tile_sh.shard_shape(leaves[0][1]) == base_sh.shard_shape(leaves[0][1])}")
    if ratio < 1.01:
        bound = 1.0 / (kt * kt)
        print(f"[CEILING] per-device state and work do NOT fall with device "
              f"count on the M3 bridge: {need} devices each carry a whole "
              f"face and run the same kernels, so strong-scaling parallel "
              f"efficiency vs the 6-device arm is bounded by 1/kt^2 = "
              f"{bound:.3f} ({bound * 100:.0f}% of ideal) BY CONSTRUCTION. "
              f"This is the M3 scaling receipt: the exchange is tiled, the "
              f"COMPUTE is not. Removing this bound is exactly M4 (kernel "
              f"retiling); no timing run can beat it beforehand.")
    else:
        print(f"[census] per-device work DOES fall ({ratio:.2f}x) -- the "
              f"kernels are tiled; a timing receipt is now meaningful.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=("sharded", "single", "census", "ranks", "bare"),
                    required=True)
    ap.add_argument("--dump-state", default=None,
                    help="single mode: .npz of every state leaf after the "
                         "timed steps, for a bitwise diff between runs")
    ap.add_argument("--payload-kb", type=int, default=16,
                    help="bare mode: per-rank payload of the one ppermute")
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--km", type=int, default=10, choices=(5, 10))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-timed", type=int, default=20)
    ap.add_argument("--bitwise", action="store_true",
                    help="ranks mode: require BITWISE agreement with the "
                         "certified exchange (valid under the AVX pin; "
                         "otherwise the measured <=1e-13 rel envelope)")
    ap.add_argument("--single-process", action="store_true",
                    help="ranks mode: instrument discriminator -- run the "
                         "identical check on virtual devices in ONE "
                         "process (no jax.distributed)")
    ap.add_argument("--tiled", type=int, default=2, metavar="KT",
                    help="census mode: tile factor kt (mesh 6*kt^2 devices)")
    args = ap.parse_args(argv)
    _log(f"importing jax (mode={args.mode})")
    if args.mode == "bare":
        return _run_bare(args)
    if args.mode == "ranks":
        return _run_ranks(args)
    if args.mode == "census":
        return _run_census(args)
    return _run_sharded(args) if args.mode == "sharded" else _run_single(args)


if __name__ == "__main__":
    _rc = main()
    # Skip interpreter finalization.  Under
    # JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi jax registers collectives.Finalize
    # via atexit; on MPItrampoline -> MPIwrapper 2.11.1 -> OpenMPI 4.1.7a1 that
    # MPI_Finalize aborts inside mca_base_var_group_finalize (job 9621176, exit
    # 134) AFTER the measurement is printed, and the harness rightly refuses a
    # number from a nonzero exit.  The timed region is untouched; gloo and mpi
    # arms both take this path.  jax.distributed.shutdown() has already run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if _rc in (0, None) else 1)
