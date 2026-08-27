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

    lat_med = statistics.median(per)
    lat_mean = statistics.mean(per)
    card = str(jax.local_devices()[0])
    print(f"[SCALING] mode=single C{args.resolution} km={args.km} "
          f"1dev {card} sha={_sha()}: "
          f"throughput {thr * 1e3:.2f} ms/step (K={K} back-to-back) | "
          f"latency median {lat_med * 1e3:.2f} mean {lat_mean * 1e3:.2f} "
          f"ms/step (per-step block, n={args.n_timed})", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("sharded", "single"), required=True)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--km", type=int, default=10, choices=(5, 10))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-timed", type=int, default=20)
    args = ap.parse_args(argv)
    _log(f"importing jax (mode={args.mode})")
    return _run_sharded(args) if args.mode == "sharded" else _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
