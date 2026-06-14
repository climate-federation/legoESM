#!/usr/bin/env python
"""Multi-node correctness validation of the tiled FULL fv3_sw_tendencies stage.

Runs ``make_tiled_fv3_sw_tendencies_stage_2d`` under REAL multi-controller
``jax.distributed`` across N nodes — so the in-stage halo ``lax.ppermute``
exchanges (B scalar, the shared cc-wind vector, the du/dv tendency vector) and
the per-tile reassembly become genuine CROSS-PROCESS (and cross-NODE)
collective-permute, not the single-process ``XLA_FLAGS=--xla_force_host_platform_
device_count=N`` virtual-device path the pytest gate uses.  Each process then
self-validates: its LOCAL tile output must be bit-identical (rel < 1e-9) to the
serial global ``fv3_sw_tendencies`` computed on the (replicated) inputs.  No
cross-process gather is needed — every process owns one (face, tile_i, tile_j)
tile and checks it against the matching global slice; ALL processes passing is
the verdict.

This is the user-requested multi-node Ginsburg validation: it proves the np>6
sub-face PRODUCTION shallow-water dycore tendency is CORRECT on real distributed
hardware.  It is a correctness check, NOT a benchmark (np>6 anti-scales on
Ginsburg's Gloo-TCP CPU fabric; wall-clock is expected to be slow).

np = 6*kt*kt processes (kt=2 -> 24, kt=3 -> 54).  Run e.g.:
  TILED_KT=2 srun --mpi=pmix -n 24 \
    $PY scripts/validate/validate_tiled_fv3_sw_multinode.py
Exit 0 = all tiles bit-identical on all processes; non-zero = failure (see log).
"""
from __future__ import annotations

import os
import sys


def log(msg: str) -> None:
    print(f"[mn pid={os.getpid()}] {msg}", flush=True)


def main() -> int:
    import jax

    jax.config.update("jax_enable_x64", True)
    try:
        jax.distributed.initialize()  # SLURM-aware multi-controller init
    except Exception as exc:  # noqa: BLE001
        log(f"jax.distributed.initialize FAILED: {type(exc).__name__}: {exc}")
        return 2

    pidx, pcount = jax.process_index(), jax.process_count()
    gdev, ldev = jax.devices(), jax.local_devices()
    if pidx == 0:
        log(f"process {pidx}/{pcount}  global_devices={len(gdev)} "
            f"local_devices={len(ldev)}")
    if len(gdev) <= len(ldev) and pcount > 1:
        log("WARNING: global<=local — jax.distributed did NOT federate "
            "processes (multi-controller not active).")

    import numpy as np
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.halo import set_halo_backend
    from legoesm.core.operators_cdgrid import fv3_sw_tendencies
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_fv3_sw_tendencies_stage_2d,
    )

    kt = int(os.environ.get("TILED_KT", "2"))
    N = int(os.environ.get("TILED_N", "24"))
    ndev = 6 * kt * kt
    if N % kt:
        log(f"N={N} not divisible by kt={kt}")
        return 3
    if len(gdev) != ndev:
        log(f"need {ndev} global devices for kt={kt}, got {len(gdev)}")
        return 3
    NL = N // kt

    set_halo_backend("local")  # serial reference uses the local cross-face pad
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))

    # Deterministic inputs — the SAME seed on every process, so the replicated
    # global field is bit-identical across processes (and == the serial ref).
    rng = np.random.default_rng(2024)
    h = 1.0e3 + rng.standard_normal((6, N, N))
    u_d = rng.standard_normal((6, N, N + 1))
    v_d = rng.standard_normal((6, N + 1, N))
    h_s = 10.0 * rng.standard_normal((6, N, N))

    # Serial reference (replicated; base case).
    dh_g, du_g, dv_g = (
        np.asarray(x) for x in fv3_sw_tendencies(
            jnp.asarray(h), jnp.asarray(u_d), jnp.asarray(v_d),
            jnp.asarray(h_s), cdg))

    # Global mesh from ALL processes' devices (the multi-controller fix:
    # jax.devices(), not jax.local_devices()).
    mesh = Mesh(np.array(gdev).reshape(6, kt, kt),
                axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdg, N, kt)
    sh = NamedSharding(mesh, P("face", None, None))

    def to_global(arr):
        # Build the face-sharded/tile-replicated global array from the
        # process-local (full, replicated) data — each device fills its shard.
        return jax.make_array_from_callback(
            arr.shape, sh, lambda idx: arr[idx])

    try:
        dh_t, du_t, dv_t = stage(
            to_global(h), to_global(u_d), to_global(v_d), to_global(h_s))
        jax.block_until_ready([dh_t, du_t, dv_t])
    except Exception as exc:  # noqa: BLE001
        log(f"tiled stage RUN FAILED: {type(exc).__name__}: {exc}")
        return 5

    # HLO cross-process collective check (process 0).
    if pidx == 0:
        try:
            import re
            hlo = jax.jit(
                lambda a, b, c, d: stage(a, b, c, d)).lower(
                to_global(h), to_global(u_d), to_global(v_d),
                to_global(h_s)).compile().as_text()
            cp = len(re.findall(r"\bcollective-permute\b", hlo))
            log(f"HLO: collective-permute={cp} (cross-process halo exchange)")
        except Exception as exc:  # noqa: BLE001
            log(f"HLO inspect skipped: {type(exc).__name__}: {exc}")

    # Each process self-validates its LOCAL tile(s) vs the serial global slice.
    def _worst(t_arr, g_full, stag_i, stag_j):
        """stag_i/stag_j: +1 if that axis is staggered (owns the shared face)."""
        w = 0.0
        for shard in t_arr.addressable_shards:
            blk = np.asarray(shard.data)            # (1, a, b)
            f = shard.index[0].start or 0
            ti = (shard.index[1].start or 0) // (NL + stag_i)
            tj = (shard.index[2].start or 0) // (NL + stag_j)
            g = g_full[f,
                       ti * NL: ti * NL + NL + stag_i,
                       tj * NL: tj * NL + NL + stag_j]
            w = max(w, float(np.max(np.abs(blk[0] - g))))
        return w

    w_h = _worst(dh_t, dh_g, 0, 0)        # dh_dt (nl, nl) cc exact
    w_du = _worst(du_t, du_g, 0, 1)       # du_d_dt (nl, nl+1) j-staggered
    w_dv = _worst(dv_t, dv_g, 1, 0)       # dv_d_dt (nl+1, nl) i-staggered
    sc = lambda g: float(np.max(np.abs(g))) + 1e-300
    rels = (w_h / sc(dh_g), w_du / sc(du_g), w_dv / sc(dv_g))
    ok = all(r < 1e-9 for r in rels)
    log(f"proc {pidx}: rel dh={rels[0]:.2e} du={rels[1]:.2e} dv={rels[2]:.2e} "
        f"-> {'OK' if ok else 'FAIL'}")

    # Reduce the verdict across all processes (allreduce of the worst rel).
    from jax.experimental import multihost_utils
    worst_all = multihost_utils.process_allgather(
        jnp.asarray([rels[0], rels[1], rels[2]]))
    worst_rel = float(np.max(np.asarray(worst_all)))
    if pidx == 0:
        verdict = "PASS" if worst_rel < 1e-9 else "FAIL"
        log(f"VERDICT (all {pcount} procs): worst rel={worst_rel:.3e} "
            f"-> MULTINODE_TILED_FV3_SW_{verdict}")
    return 0 if worst_rel < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
