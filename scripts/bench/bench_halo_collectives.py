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
refactor, on a SYNTHETIC ring-offset graph at production degree
(8-13 of 64-128 devices).

Production-faithfulness of the control (codex round-1 findings):

* the ``ppermute`` arm colours the SAME graph with the PRODUCTION
  colourers (``greedy_edge_coloring`` / ``multi_ordering_edge_coloring``
  from ``legoesm.parallel``, selected exactly like
  ``_build_ppermute_schedule``) into bidirectional pair-matching rounds,
  gathers each round's rows from the operand through an index array
  (mirroring ``cell_pack[sc]``), and scatters receives through a
  garbage-slot staging buffer (mirroring ``cell_local.at[rc].set`` with
  the ``max_lc`` pad target). Devices without a partner in a round
  still execute the collective, like production.
* the ``ragged`` arm performs the SAME index-gather (one fused gather
  for all rounds) and the same garbage-slot scatter, so the two arms
  differ in collective STRUCTURE only (k sequential rounds vs one
  grouped call), which is the variable under test.
* round payloads here are uniform across pairs, so production's
  pad-to-round-max is a no-op in this instrument; the real schedule's
  padding waste is NOT modelled (stated limitation).
* like production (``_ppermute_halo_fill``: every round's send gathers
  from ``cell_pack``, not from prior rounds' output), rounds carry NO
  data dependence on each other — serialization, if observed, comes
  from the runtime/stream schedule, which is precisely what is being
  measured. ``LEGOESM_XLA_OVERLAP`` must be pinned by the launcher;
  the effective value is recorded in every JSONL row.

* ``uniform`` — ``jax.lax.all_to_all`` (tiled): the ICON "pad to
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
(pattern payload, per-addressable-shard comparison) or the run aborts —
no timing row is emitted from an unverified exchange.

Usage
-----
    # intra-node, 4 local GPUs (offsets must not alias mod n_dev)
    python scripts/bench/bench_halo_collectives.py \
        --n-devices 4 --offsets 1,-1,2

    # inter-node (one process per GPU)
    srun --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
        python scripts/bench/bench_halo_collectives.py \
        --multicontroller --n-devices 8 --offsets 1,-1,2,-2,3,-3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time

import numpy as np

AXIS = "dev"
# Host operand for the uniform arm is O(n_dev^2 * rows * width); above
# this device count the arm is skipped (and it is byte-inflated by
# (n_dev-1)/degree there anyway — its trend is receipted at small n_dev).
_UNIFORM_MAX_NDEV = 32


def build_offset_graph_metadata(n_dev: int, offsets: list[int], rows: int,
                                seed: int | None = 0):
    """Ragged-a2a + coloured-round metadata for the ring-offset graph.

    Graph: device ``d`` exchanges with ``(d + o) % n_dev`` for each
    ``o`` in ``offsets`` (directed; pass both signs for a symmetric
    halo). Operand layout on ``d``: ``deg*rows`` owned rows; the rows
    SENT to destination block ``j`` (j-th offset) are
    ``send_row_idx[d, j*rows:(j+1)*rows]`` — a seeded shuffled index so
    both arms pay a real (non-iota) gather, mirroring production's
    ``cell_pack[sc]``. Receive staging layout: blocks ordered by SOURCE
    RANK ascending; strategy arms scatter through a garbage-slot buffer
    (production's ``max_lc`` pad-target pattern).

    Also colours the undirected pair graph with the PRODUCTION
    colourers, selected exactly like ``_build_ppermute_schedule``
    (multi-ordering only on a strict round win).

    Returns a dict with the ragged_all_to_all arrays (``send_sizes``,
    ``input_offsets``, ``output_offsets``, ``recv_sizes`` — all
    (n_dev, n_dev) int32, contract ``send_sizes == recv_sizes.T``
    asserted), ``recv_src_order`` (n_dev, deg), ``send_row_idx``
    (n_dev, deg*rows), and the coloured ``rounds`` (list of pair
    lists) + ``n_rounds`` + ``coloring_method``.
    """
    from legoesm.parallel import (greedy_edge_coloring,
                                  multi_ordering_edge_coloring)

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
    # The coloured control pairs each round bidirectionally; that needs
    # the neighbour set to be symmetric (o and -o both present mod n).
    if set(dsts) != {(-o) % n_dev for o in offsets}:
        raise ValueError(f"offsets {offsets} not symmetric mod {n_dev}; "
                         f"the halo exchange is bidirectional")

    deg = len(offsets)
    send_sizes = np.zeros((n_dev, n_dev), np.int32)
    input_offsets = np.zeros((n_dev, n_dev), np.int32)
    output_offsets = np.zeros((n_dev, n_dev), np.int32)
    recv_sizes = np.zeros((n_dev, n_dev), np.int32)
    recv_src_order = np.zeros((n_dev, deg), np.int32)

    if seed is None:      # identity gather — hand-checkable test cases
        send_row_idx = np.tile(np.arange(deg * rows, dtype=np.int32),
                               (n_dev, 1))
    else:
        rng = np.random.default_rng(seed)
        send_row_idx = np.stack([rng.permutation(deg * rows)
                                 for _ in range(n_dev)]).astype(np.int32)

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

    # Production colouring + production selection rule (strict win only).
    comm_pairs = set()
    for d in range(n_dev):
        for o in offsets:
            dst = (d + o) % n_dev
            comm_pairs.add((min(d, dst), max(d, dst)))
    greedy = greedy_edge_coloring(comm_pairs)
    n_greedy = max(greedy.values()) + 1
    multi, _max_deg = multi_ordering_edge_coloring(comm_pairs)
    n_multi = max(multi.values()) + 1
    if n_multi < n_greedy:
        colors, n_rounds, method = multi, n_multi, "multi_greedy"
    else:
        colors, n_rounds, method = greedy, n_greedy, "greedy"
    rounds: list[list[tuple[int, int]]] = [[] for _ in range(n_rounds)]
    for (u, v), c in colors.items():
        rounds[c].append((u, v))

    return {
        "send_sizes": send_sizes,
        "input_offsets": input_offsets,
        "output_offsets": output_offsets,
        "recv_sizes": recv_sizes,
        "recv_src_order": recv_src_order,
        "send_row_idx": send_row_idx,
        "rounds": rounds,
        "n_rounds": n_rounds,
        "coloring_method": method,
    }


def expected_staging(x: np.ndarray, meta: dict, rows: int) -> np.ndarray:
    """Analytic post-exchange staging buffer (the verify oracle).

    Device ``d``, staging block ``k`` (source ``s = recv_src_order[d,k]``)
    holds the rows source ``s`` gathered for destination ``d``:
    ``x[s, send_row_idx[s, j*rows:(j+1)*rows]]`` where ``j`` is the
    offset-block index of ``d`` on ``s`` (``input_offsets[s, d]/rows``).
    """
    n_dev = x.shape[0]
    exp = np.empty_like(x)
    for d in range(n_dev):
        for k, s in enumerate(meta["recv_src_order"][d]):
            off = meta["input_offsets"][s, d]
            idx = meta["send_row_idx"][s, off:off + rows]
            exp[d, k * rows:(k + 1) * rows] = x[s, idx]
    return exp


def pattern_payload(n_dev: int, deg: int, rows: int, width: int) -> np.ndarray:
    """Globally unique value per (device, row): d*deg*rows + j — no two
    rows anywhere share a value, so a misrouted block cannot match."""
    x = np.zeros((n_dev, deg * rows, width), np.float32)
    for d in range(n_dev):
        x[d] = (np.float64(d) * deg * rows + np.arange(deg * rows))[:, None]
    return x


def uniform_operand(x_np: np.ndarray, meta: dict, n_dev: int,
                    offsets: list[int], rows: int, width: int) -> np.ndarray:
    """Map the (gathered) per-offset payload into the uniform-a2a
    layout: per device, block i of n_dev = payload for peer i (zeros
    for non-neighbours). Gather applied host-side so the uniform arm
    exchanges the same block contents."""
    xu = np.zeros((n_dev, n_dev * rows, width), np.float32)
    for d in range(n_dev):
        for j, o in enumerate(offsets):
            dst = (d + o) % n_dev
            idx = meta["send_row_idx"][d, j * rows:(j + 1) * rows]
            xu[d, dst * rows:(dst + 1) * rows] = x_np[d, idx]
    return xu


# ---------------------------------------------------------------------------
# Strategy bodies (inside shard_map). Each verify body maps its receives
# into the SAME staging layout (blocks by ascending source rank).
# ---------------------------------------------------------------------------

def make_strategies(mesh, n_dev, offsets, rows, width, meta, shard_map, P):
    import jax
    import jax.numpy as jnp

    deg = len(offsets)
    n_stage = deg * rows

    # --- coloured bidirectional pair rounds (production structure) ----
    # Per round r and device d: partner (or -1), the offset-block j
    # sent to that partner, and the staging slot receiving from it
    # (garbage slot n_stage when no partner — production's max_lc pad).
    n_rounds = meta["n_rounds"]
    perms = []
    r_send_block = np.zeros((n_dev, n_rounds), np.int32)
    r_recv_slot = np.full((n_dev, n_rounds), deg, np.int32)  # deg=garbage blk
    dst_block = {}
    for d in range(n_dev):
        for j, o in enumerate(offsets):
            dst_block[(d, (d + o) % n_dev)] = j
    for r, pairs in enumerate(meta["rounds"]):
        perm = []
        for u, v in pairs:
            perm.append((u, v))
            perm.append((v, u))
            for a, b in ((u, v), (v, u)):
                r_send_block[a, r] = dst_block[(a, b)]
                order = list(meta["recv_src_order"][a])
                r_recv_slot[a, r] = order.index(b)
        perms.append(perm)

    def ppermute_body(xl, sri, sblk, rslot):
        # staging with one garbage BLOCK at the end (production's
        # garbage-slot scatter target), trimmed on return.
        stag = jnp.zeros((n_stage + rows, width), xl.dtype)
        for r in range(n_rounds):
            off = sblk[0, r] * rows
            idx = jax.lax.dynamic_slice_in_dim(sri[0], off, rows)
            block = xl[0][idx]                      # real gather
            recv = jax.lax.ppermute(block, AXIS, perm=perms[r])
            stag = jax.lax.dynamic_update_slice_in_dim(
                stag, recv, rslot[0, r] * rows, axis=0)
        return stag[:n_stage][None]

    def ragged_body(xl, sri, ssl, iol, ool, rsl):
        sendbuf = xl[0][sri[0]]                     # ONE fused gather
        out = jnp.zeros((n_stage, width), xl.dtype)
        r = jax.lax.ragged_all_to_all(
            sendbuf, out, iol[0], ssl[0], ool[0], rsl[0], axis_name=AXIS)
        return r[None]

    def uniform_verify_body(xl, gather_idx):
        r = jax.lax.all_to_all(xl[0], AXIS, split_axis=0, concat_axis=0,
                               tiled=True)
        stag = r.reshape(n_dev, rows, width)[gather_idx[0]]
        return stag.reshape(n_stage, width)[None]

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
            "verify_fn": wrap(ppermute_body, 4),
            "time_fn": wrap(ppermute_body, 4),
            "meta_np": (meta["send_row_idx"], r_send_block, r_recv_slot),
            "n_collectives": n_rounds,
        },
        "ragged": {
            "verify_fn": wrap(ragged_body, 6),
            "time_fn": wrap(ragged_body, 6),
            "meta_np": (meta["send_row_idx"], meta["send_sizes"],
                        meta["input_offsets"], meta["output_offsets"],
                        meta["recv_sizes"]),
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


def verify_addressable(got_arr, exp: np.ndarray, name: str) -> None:
    """Compare only THIS process's addressable shards (a global fetch of
    a P(dev)-sharded array raises in multicontroller — codex round 1).
    Any mismatch exits non-zero; with --kill-on-bad-exit the job dies."""
    n_checked = 0
    for shard in got_arr.addressable_shards:
        local = np.asarray(shard.data)
        want = exp[shard.index]
        if not np.array_equal(local, want):
            bad = int((local != want).sum())
            raise SystemExit(
                f"VERIFY FAILED for {name} on shard {shard.index}: "
                f"{bad}/{want.size} elements differ — refusing to time "
                f"an unverified exchange")
        n_checked += local.size
    if n_checked == 0:
        raise SystemExit(f"VERIFY FAILED for {name}: no addressable "
                         f"shards on this process")


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
    p.add_argument("--offsets", default="1,-1,2,-2,5,-5,7,-7,6,-6",
                   help="Directed ring offsets; must be symmetric and "
                        "alias-free mod n_devices; degree = count. "
                        "Default degree 10 ~ the measured MPAS "
                        "partition neighbour count at 64-128 devices "
                        "(valid for n_devices >= 15).")
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

    if is_root:
        print(f"coloured control: {meta['n_rounds']} rounds "
              f"({meta['coloring_method']}), degree {deg}", flush=True)

    # Sharded device arrays for every input (multicontroller-safe put —
    # never a whole-array device_put of a to-be-sharded arg).
    x_np = pattern_payload(n_dev, deg, rows, width)
    exp = expected_staging(x_np, meta, rows)
    x_dev = addressable_shard_put(x_np, shard)
    xu_dev = None
    if "uniform" in wanted:
        xu_np = uniform_operand(x_np, meta, n_dev, offsets, rows, width)
        xu_dev = addressable_shard_put(xu_np, shard)
    for s in strategies.values():
        s["meta_dev"] = tuple(
            addressable_shard_put(m, shard) for m in s["meta_np"])

    # ------------------------------------------------------------------
    # VERIFY stage — every strategy against the analytic oracle, exact,
    # per-addressable-shard. No timing row without this passing
    # (instrument-validation rule).
    # ------------------------------------------------------------------
    for name in wanted:
        s = strategies[name]
        operand = xu_dev if s.get("operand_layout") == "uniform" else x_dev
        got_arr = jax.jit(s["verify_fn"])(operand, *s["meta_dev"])
        verify_addressable(got_arr, exp, name)
        if is_root:
            print(f"verify {name}: OK (exact, per-shard)", flush=True)

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
            "coloring_method": meta["coloring_method"],
            "per_fill_us": round(per, 2),
            "single_call_us": round(t1, 2),
            "n_reps": args.n_reps,
            "n_iters": args.n_iters,
            "backend": jax.default_backend(),
            "git_sha": sha,
            "env_overlap": os.environ.get("LEGOESM_XLA_OVERLAP", ""),
            "env_xla_flags": os.environ.get("XLA_FLAGS", ""),
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
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                for rec in rows_out:
                    f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
