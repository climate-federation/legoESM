"""A/B/C the halo-fill collective: coloured ppermute rounds vs ragged_all_to_all vs uniform all_to_all.

The MPAS SPMD lane fills its halo with an edge-coloured schedule of
SEQUENTIAL ``jax.lax.ppermute`` rounds (12-16 at 64-128 devices; the
colouring itself is provably optimal — Vizing gap 0 on every measured
config, PR #1512). Reference models (MPAS-A Fortran, ICON-GPU) instead
post ALL neighbour messages concurrently (grouped isend/irecv), paying
~one latency instead of ``rounds x latency``. The JAX-reachable
equivalent is ``jax.lax.ragged_all_to_all`` (one grouped-P2P collective,
zero-size slices for non-neighbours; JVP+transpose rules registered in
jax 0.10.0).

This microbenchmark de-risks that swap BEFORE any schedule-builder
refactor, on a SYNTHETIC ring-offset graph matching the production
degree (8-13 of 64-128):

* ``ppermute``  — one rotation ppermute per directed offset, sequential
  (same collective count as the production coloured schedule at gap 0).
* ``ragged``    — ONE ``ragged_all_to_all`` carrying every neighbour
  block; non-neighbours get zero-size slices.
* ``uniform``   — ``jax.lax.all_to_all`` (tiled): the ICON "pad to
  uniform" fallback; sends the pair payload to ALL n-1 peers. Its
  timing loop chains the raw all_to_all WITHOUT the staging gather
  (shape mismatch would break the carry), which biases IN FAVOUR of
  uniform — safe when it loses, revisit if it narrowly wins. Skipped
  at ``n_dev > 32`` (host operand grows O(n_dev^2)).

XLA:CPU has no ragged-all-to-all thunk (verified 2026-08-09: "HLO
opcode `ragged-all-to-all` is not supported by XLA:CPU ThunkEmitter"),
so semantics CANNOT be pre-checked on CPU. The instrument therefore
verifies itself in the SAME process before timing: every strategy's
received staging buffer must equal the analytic expectation exactly
(pattern payload, ``np.array_equal``) or the run aborts — no timing row
is emitted from an unverified exchange.

Usage
-----
    # intra-node, 4 local GPUs
    python scripts/bench/bench_halo_collectives.py --n-devices 4

    # inter-node (one process per GPU)
    srun --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
        python scripts/bench/bench_halo_collectives.py \
        --multicontroller --n-devices 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import numpy as np

AXIS = "dev"
# Host operand for the uniform arm is O(n_dev^2 * rows * width); above
# this device count the arm is skipped (and it is byte-inflated by
# (n_dev-1)/degree there anyway — its trend is receipted at small n_dev).
_UNIFORM_MAX_NDEV = 32


def build_offset_graph_metadata(n_dev: int, offsets: list[int], rows: int):
    """Ragged-a2a metadata for the ring-offset graph, host-side numpy.

    Graph: device ``d`` exchanges with ``(d + o) % n_dev`` for each
    ``o`` in ``offsets`` (directed; pass both signs for a symmetric
    halo). Send-buffer layout on ``d``: block ``j`` (j-th offset) holds
    the ``rows`` rows for destination ``(d + offsets[j])``. Receive
    staging layout: blocks ordered by SOURCE RANK ascending —
    ``ragged_all_to_all`` delivers at sender-chosen ``output_offsets``,
    and every strategy in this bench is mapped to this same layout so
    the verify stage can compare them byte-for-byte.

    Returns dict of (n_dev, n_dev) int32 arrays ``send_sizes``,
    ``input_offsets``, ``output_offsets``, ``recv_sizes`` plus the
    per-device ``recv_src_order`` (n_dev, degree) int32 (ascending
    sources), consistent with the ragged_all_to_all contract
    ``send_sizes == all_to_all(recv_sizes)`` (asserted here).
    """
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"duplicate offsets: {offsets}")
    if any(o % n_dev == 0 for o in offsets):
        raise ValueError(f"offset 0 mod n_dev={n_dev} in {offsets} "
                         f"(self-send)")
    dsts = [(o % n_dev) for o in offsets]
    if len(set(dsts)) != len(dsts):
        raise ValueError(
            f"offsets {offsets} alias mod n_dev={n_dev}; degree would "
            f"shrink and blocks would overlap")

    deg = len(offsets)
    send_sizes = np.zeros((n_dev, n_dev), np.int32)
    input_offsets = np.zeros((n_dev, n_dev), np.int32)
    output_offsets = np.zeros((n_dev, n_dev), np.int32)
    recv_sizes = np.zeros((n_dev, n_dev), np.int32)
    recv_src_order = np.zeros((n_dev, deg), np.int32)

    for d in range(n_dev):
        for j, o in enumerate(offsets):
            dst = (d + o) % n_dev
            send_sizes[d, dst] = rows
            input_offsets[d, dst] = j * rows
        srcs = sorted((d - o) % n_dev for o in offsets)
        recv_src_order[d] = srcs
        for k, s in enumerate(srcs):
            recv_sizes[d, s] = rows

    for d in range(n_dev):
        for dst in range(n_dev):
            if send_sizes[d, dst]:
                srcs = sorted((dst - o) % n_dev for o in offsets)
                output_offsets[d, dst] = srcs.index(d) * rows

    assert np.array_equal(send_sizes, recv_sizes.T), "size invariant broken"
    return {
        "send_sizes": send_sizes,
        "input_offsets": input_offsets,
        "output_offsets": output_offsets,
        "recv_sizes": recv_sizes,
        "recv_src_order": recv_src_order,
    }


def expected_staging(x: np.ndarray, meta: dict, rows: int) -> np.ndarray:
    """Analytic post-exchange staging buffer (the verify oracle)."""
    n_dev = x.shape[0]
    exp = np.empty_like(x)
    for d in range(n_dev):
        for k, s in enumerate(meta["recv_src_order"][d]):
            off = meta["input_offsets"][s, d]
            exp[d, k * rows:(k + 1) * rows] = x[s, off:off + rows]
    return exp


def pattern_payload(n_dev: int, deg: int, rows: int, width: int) -> np.ndarray:
    """Distinct value per (device, row) so misrouted blocks cannot match."""
    x = np.zeros((n_dev, deg * rows, width), np.float32)
    for d in range(n_dev):
        x[d] = (d * 100_000 + np.arange(deg * rows))[:, None]
    return x


def uniform_operand(x_np: np.ndarray, n_dev: int, offsets: list[int],
                    rows: int, width: int) -> np.ndarray:
    """Map the canonical per-offset payload into the uniform-a2a layout:
    per device, block i of n_dev = payload for peer i (zeros for
    non-neighbours)."""
    xu = np.zeros((n_dev, n_dev * rows, width), np.float32)
    for d in range(n_dev):
        for j, o in enumerate(offsets):
            dst = (d + o) % n_dev
            xu[d, dst * rows:(dst + 1) * rows] = \
                x_np[d, j * rows:(j + 1) * rows]
    return xu


# ---------------------------------------------------------------------------
# Strategy bodies (inside shard_map). Each verify body maps its receives
# into the SAME staging layout (blocks by ascending source rank).
# ---------------------------------------------------------------------------

def make_strategies(mesh, n_dev, offsets, rows, width, meta, shard_map, P):
    import jax
    import jax.numpy as jnp

    deg = len(offsets)
    # ppermute arm: per directed offset o, ONE rotation ppermute
    # (everyone sends its o-block to d+o, receives from d-o), placed at
    # the staging slot of source d-o. deg sequential collectives == the
    # production coloured schedule's collective count at Vizing gap 0.
    perms = [
        [(d, (d + o) % n_dev) for d in range(n_dev)] for o in offsets
    ]
    # staging slot (block index) of source (d - o) on device d, per
    # offset j — device-dependent, precomputed (n_dev, deg).
    slot_of_offset = np.zeros((n_dev, deg), np.int32)
    for d in range(n_dev):
        order = list(meta["recv_src_order"][d])
        for j, o in enumerate(offsets):
            slot_of_offset[d, j] = order.index((d - o) % n_dev)

    def ppermute_body(xl, slots):
        buf = jnp.zeros_like(xl[0])
        for j in range(deg):
            block = jax.lax.dynamic_slice_in_dim(xl[0], j * rows, rows)
            recv = jax.lax.ppermute(block, AXIS, perm=perms[j])
            buf = jax.lax.dynamic_update_slice_in_dim(
                buf, recv, slots[0, j] * rows, axis=0)
        return buf[None]

    def ragged_body(xl, ssl, iol, ool, rsl):
        out = jnp.zeros_like(xl[0])
        r = jax.lax.ragged_all_to_all(
            xl[0], out, iol[0], ssl[0], ool[0], rsl[0], axis_name=AXIS)
        return r[None]

    def uniform_verify_body(xl, gather_idx):
        r = jax.lax.all_to_all(xl[0], AXIS, split_axis=0, concat_axis=0,
                               tiled=True)
        stag = r.reshape(n_dev, rows, width)[gather_idx[0]]
        return stag.reshape(deg * rows, width)[None]

    def uniform_time_body(xl, gather_idx):
        # raw exchange only; carry keeps the (n_dev*rows, width) shape
        # so reps can chain. Staging gather omitted (favours uniform).
        del gather_idx
        r = jax.lax.all_to_all(xl[0], AXIS, split_axis=0, concat_axis=0,
                               tiled=True)
        return r[None]

    def wrap(body, n_args):
        specs = tuple([P(AXIS)] * n_args)
        # check_vma is the current spelling of the old check_rep
        # (matches bench_ppermute_microbench / sharded_dynamics.py).
        try:
            return shard_map(body, mesh=mesh, in_specs=specs,
                             out_specs=P(AXIS), check_vma=False)
        except TypeError:  # pragma: no cover - JAX < 0.9 spelling
            return shard_map(body, mesh=mesh, in_specs=specs,
                             out_specs=P(AXIS), check_rep=False)

    return {
        "ppermute": {
            "verify_fn": wrap(ppermute_body, 2),
            "time_fn": wrap(ppermute_body, 2),
            "meta_np": (slot_of_offset,),
            "n_collectives": deg,
        },
        "ragged": {
            "verify_fn": wrap(ragged_body, 5),
            "time_fn": wrap(ragged_body, 5),
            "meta_np": (meta["send_sizes"], meta["input_offsets"],
                        meta["output_offsets"], meta["recv_sizes"]),
            "n_collectives": 1,
        },
        "uniform": {
            "verify_fn": wrap(uniform_verify_body, 2),
            "time_fn": wrap(uniform_time_body, 2),
            "meta_np": (meta["recv_src_order"],),
            "n_collectives": 1,
            "operand_layout": "uniform",
        },
    }


def median_us(fn, args, n_warmup, n_iters):
    import jax
    out = None
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    return statistics.median(ts)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-devices", type=int, default=0)
    p.add_argument("--multicontroller", action="store_true")
    p.add_argument("--coordinator", default=None)
    p.add_argument("--offsets", default="1,-1,2,-2,5,-5,9,-9,13,-13",
                   help="Directed ring offsets (both signs for a "
                        "symmetric halo); degree = count. Default "
                        "degree 10 ~ the measured MPAS partition "
                        "neighbour count at 64-128 devices.")
    p.add_argument("--rows", type=int, default=700,
                   help="Rows per neighbour block (~the s9 per-pair "
                        "halo cell count).")
    p.add_argument("--width", type=int, default=130,
                   help="Trailing width (f32): ~nlev*(1+n_tracers)+2; "
                        "700x130 f32 = 364 KB per pair message.")
    p.add_argument("--strategies", default="ppermute,ragged,uniform")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-reps", type=int, default=32,
                   help="Chained halo fills in ONE jit; the 1-rep vs "
                        "n-rep difference cancels host dispatch (same "
                        "method as bench_ppermute_microbench).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.multicontroller:
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(args.coordinator)

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.parallel.geometry_consistency import addressable_shard_put
    from legoesm.parallel.shard_map_compat import shard_map

    devices = jax.devices()
    n_dev = args.n_devices or len(devices)
    if n_dev < 2:
        raise SystemExit("need >=2 devices")
    if n_dev > len(devices):
        raise SystemExit(f"--n-devices {n_dev} > {len(devices)} visible")

    offsets = [int(t) for t in args.offsets.split(",") if t.strip()]
    rows, width = args.rows, args.width
    deg = len(offsets)
    meta = build_offset_graph_metadata(n_dev, offsets, rows)
    mesh = Mesh(np.array(devices[:n_dev]), (AXIS,))
    shard = NamedSharding(mesh, P(AXIS))

    strategies = make_strategies(
        mesh, n_dev, offsets, rows, width, meta, shard_map, P)
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = set(wanted) - set(strategies)
    if unknown:
        raise SystemExit(f"unknown strategies {sorted(unknown)}; "
                         f"choose from {sorted(strategies)}")
    is_root = jax.process_index() == 0
    if "uniform" in wanted and n_dev > _UNIFORM_MAX_NDEV:
        if is_root:
            print(f"SKIPPING uniform: n_dev {n_dev} > {_UNIFORM_MAX_NDEV} "
                  f"(operand O(n_dev^2); arm receipted at small n_dev)")
        wanted = [w for w in wanted if w != "uniform"]

    # Sharded device arrays for every input (multicontroller-safe put —
    # never a whole-array device_put of a to-be-sharded arg).
    x_np = pattern_payload(n_dev, deg, rows, width)
    exp = expected_staging(x_np, meta, rows)
    x_dev = addressable_shard_put(x_np, shard)
    xu_dev = None
    if "uniform" in wanted:
        xu_np = uniform_operand(x_np, n_dev, offsets, rows, width)
        xu_dev = addressable_shard_put(xu_np, shard)
    for s in strategies.values():
        s["meta_dev"] = tuple(
            addressable_shard_put(m, shard) for m in s["meta_np"])

    # ------------------------------------------------------------------
    # VERIFY stage — every strategy against the analytic oracle, exact.
    # No timing row without this passing (instrument-validation rule).
    # ------------------------------------------------------------------
    for name in wanted:
        s = strategies[name]
        operand = xu_dev if s.get("operand_layout") == "uniform" else x_dev
        got_arr = jax.jit(s["verify_fn"])(operand, *s["meta_dev"])
        got = np.array(jax.device_get(got_arr))
        if not np.array_equal(got, exp):
            bad = int((got != exp).sum())
            raise SystemExit(
                f"VERIFY FAILED for {name}: {bad}/{exp.size} elements "
                f"differ — refusing to time an unverified exchange")
        if is_root:
            print(f"verify {name}: OK ({exp.size} elements exact)",
                  flush=True)

    # ------------------------------------------------------------------
    # TIME stage — chained reps, dispatch-subtracted
    # ------------------------------------------------------------------
    def chained(s, n_reps):
        body = s["time_fn"]

        @jax.jit
        def run(x0, *sargs):
            def one(_, v):
                return body(v, *sargs)
            return jax.lax.fori_loop(0, n_reps, one, x0)
        return run

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"

    pair_bytes = rows * width * 4
    rows_out = []
    for name in wanted:
        s = strategies[name]
        operand = xu_dev if s.get("operand_layout") == "uniform" else x_dev
        t1 = median_us(chained(s, 1), (operand, *s["meta_dev"]),
                       args.n_warmup, args.n_iters)
        tn = median_us(chained(s, args.n_reps), (operand, *s["meta_dev"]),
                       args.n_warmup, args.n_iters)
        per = (tn - t1) / (args.n_reps - 1)
        rec = {
            "component": "halo_collectives_microbench",
            "strategy": name,
            "n_devices": n_dev,
            "n_processes": jax.process_count(),
            "multicontroller": bool(args.multicontroller),
            "degree": deg,
            "offsets": offsets,
            "rows": rows,
            "width": width,
            "pair_bytes": pair_bytes,
            "n_collectives_per_fill": s["n_collectives"],
            "per_fill_us": round(per, 2),
            "single_call_us": round(t1, 2),
            "n_reps": args.n_reps,
            "n_iters": args.n_iters,
            "backend": jax.default_backend(),
            "git_sha": sha,
            "verified_exact": True,
        }
        rows_out.append(rec)
        if is_root:
            print(f"{name:9s} per-fill {per:9.2f} us  "
                  f"({s['n_collectives']} collectives, "
                  f"{pair_bytes/1024:.0f} KiB/pair)  single-call "
                  f"{t1:.0f} us", flush=True)

    if is_root:
        for rec in rows_out:
            print(json.dumps(rec))
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                for rec in rows_out:
                    f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
