"""Face-axis sharding parity for the FV3 duo jitted step (SPMD milestone 1).

The duo lane's jitted ``fv_dynamics`` step is one functional program over
face-STACKED ``(6, ...)`` arrays; every cross-face halo transfer is a static
index-table gather over that leading axis.  Sharding that axis over a device
mesh therefore needs NO rewrite of the halo machinery -- XLA inserts the
collectives -- and THIS probe measures whether the resulting numbers survive:

  arm A: the bundle on ONE device (the certified configuration);
  arm B: the SAME jitted program with every bundle leaf device_put onto a
         ``NamedSharding`` over the face axis (``n_shards`` devices, 6
         divisible by it: 1, 2, 3, or 6).

Both arms run in one process on the CPU host-device mesh
(``XLA_FLAGS=--xla_force_host_platform_device_count=<N>`` must be set BEFORE
jax imports -- the sbatch wrapper owns that), so this is also the
PORTABILITY gate: the identical script drives GPU meshes later by changing
only the platform.

Report-only by default (first output of a new instrument is untrusted);
``--max-abs`` gates once a measured bound exists.  Prints |d|max per field
plus per-arm wall time (the timing is DESCRIPTIVE at C48-CPU scale -- the
scaling claim belongs to the GPU ladder, not this probe).

WHAT THIS PROBE CANNOT SEE (codex MAJOR, by design): the step's jit is
unconstrained, so XLA may gather inputs, compute REPLICATED internally, and
reshard only the outputs -- numerical parity plus face-sharded outputs would
both pass.  The INTERIOR question is answered by the HLO collective census
the sbatch wrapper runs beside this probe (``--xla_dump_to`` + grep): full-
state all-gathers vs halo-strip-sized permutes is the discriminator that
picks GSPMD-as-is vs a shard_map exchange.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _flatten_bundle(bundle) -> dict:
    """Bundle -> {name: np.ndarray}, walking the structure directly
    (the restart test's independent-enumeration pattern)."""
    out = {}
    for k, v in bundle["state"].items():
        out[f"state.{k}"] = np.asarray(v)
    for k, v in bundle["press"].items():
        out[f"press.{k}"] = np.asarray(v)
    for i, qt in enumerate(bundle["q"]):
        out[f"q[{i}]"] = np.asarray(qt)
    out["omga"] = np.asarray(bundle["omga"])
    if bundle.get("nh") is not None:
        for k, v in bundle["nh"].items():
            out[f"nh.{k}"] = np.asarray(v)
    return out


def _shard_bundle(bundle, sharding):
    """device_put every FACE-STACKED leaf onto ``sharding``.  Every bundle
    leaf is face-leading today (codex-verified: state stack, press, q,
    omga, and the NH carry are all ``(6, ...)``); a leaf that is not is a
    broken premise, so the assert REFUSES rather than silently
    replicating."""
    import jax

    def _put(x):
        assert x.shape[0] == 6, (
            f"bundle leaf with leading axis {x.shape[0]} != 6 faces; the "
            f"face-shard probe's premise broke -- extend it deliberately")
        return jax.device_put(x, sharding)

    out = {"state": {k: _put(v) for k, v in bundle["state"].items()},
           "press": {k: _put(v) for k, v in bundle["press"].items()},
           "q": [_put(qt) for qt in bundle["q"]],
           "omga": _put(bundle["omga"])}
    out["nh"] = (None if bundle.get("nh") is None
                 else {k: _put(v) for k, v in bundle["nh"].items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--n-shards", type=int, default=6, choices=(2, 3, 6),
                    help="devices the face axis is split over (6 % n == 0)")
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--face-batched", action="store_true",
                    help="arm B additionally routes the 3-D phases "
                         "through their vmapped face-batched arms "
                         "(step_face_batched=True) -- the fix for the "
                         "per-face x[t] all-reduce storm.")
    ap.add_argument("--ring", action="store_true",
                    help="arm B additionally routes the step's halo "
                         "exchanges through the M3 shard_map ring "
                         "(step_spmd_mesh=mesh). Per-level ring parity "
                         "is bitwise; the K-BATCHED ring (v2a) "
                         "reassociates vmapped stencil dots, so expect "
                         "the few-ulp floor (measured 6e-15 rel), NOT "
                         "0.0. Gate with --gate-rtol/--gate-atol.")
    ap.add_argument("--max-abs", type=float, default=None,
                    help="gate: worst |sharded - single| over all fields. "
                         "Omit for report-only (first runs MEASURE the "
                         "bound; pass it once pinned). A single absolute "
                         "bound is scale-unfair across fields (codex); "
                         "prefer --gate-rtol/--gate-atol.")
    ap.add_argument("--gate-rtol", type=float, default=None,
                    help="per-field gate: |d|max <= gate-atol + "
                         "gate-rtol * |field|max. The fair form across "
                         "delp (1e4) and tracer (1e-2) scales.")
    ap.add_argument("--gate-atol", type=float, default=1e-12,
                    help="absolute floor for --gate-rtol (zero-scale "
                         "fields like w/omga on the hydrostatic arm).")
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    devs = jax.devices()
    if len(devs) < args.n_shards:
        raise SystemExit(
            f"need {args.n_shards} devices, have {len(devs)}. Set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count="
            f"{args.n_shards} (before jax imports) or run on a "
            f"{args.n_shards}-GPU allocation.")

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid

    bundle_grid = create_fv3_duo_grid(args.resolution)
    model = FV3DuoDynamicsModel(bundle_grid, FV3DuoConfig(km=args.km))
    ic = model.dcmip16_initial_state(do_pert=True)

    # ---- arm A: single device (the certified configuration) ----
    b = model.step(ic, args.dt)         # warmup: pays compile
    b = ic
    t0 = time.time()
    for _ in range(args.n_steps):
        b = model.step(b, args.dt)
    jax.block_until_ready(b["state"]["pt"])
    t_single = time.time() - t0
    single = _flatten_bundle(b)

    # ---- arm B: face axis sharded over n_shards devices ----
    # A SEPARATE model instance pins the step's out_shardings: with the
    # unconstrained jit, GSPMD keeps the interior distributed but resolves
    # the OUTPUTS replicated (measured, job 9483159), so the loop decays
    # after one step. Pinning the boundary is the fix; arm A keeps the
    # certified unconstrained jit.
    mesh = Mesh(np.array(devs[:args.n_shards]), ("face",))
    shard = NamedSharding(mesh, P("face"))
    model_sh = FV3DuoDynamicsModel(bundle_grid, FV3DuoConfig(km=args.km),
                                   step_out_shardings=shard,
                                   step_spmd_mesh=(mesh if args.ring
                                                   else None),
                                   step_face_batched=args.face_batched)
    model = model_sh                    # arm B steps below use the pinned jit
    b = _shard_bundle(ic, shard)
    b = model.step(b, args.dt)          # warmup: pays compile for this layout

    def _assert_face_sharded(bundle, when):
        """NON-VACUITY (codex BLOCKER): a parity pass on a silently
        replicated arm B proves nothing about SPMD.  ``device_set`` is NOT
        the test -- a fully-replicated NamedSharding still lists every
        device -- so assert per RAW jax leaf (no gather): not fully
        replicated, and the device->index map equals the requested face
        sharding's."""
        want = shard.devices_indices_map  # bound method; per-shape below
        leaves = list(bundle["state"].values()) \
            + list(bundle["press"].values()) + list(bundle["q"]) \
            + [bundle["omga"]] \
            + (list(bundle["nh"].values()) if bundle.get("nh") else [])
        for leaf in leaves:
            sh = leaf.sharding
            if sh.is_fully_replicated or (
                    sh.devices_indices_map(leaf.shape)
                    != want(leaf.shape)):
                raise SystemExit(
                    f"VACUOUS ({when}): a step-output leaf decayed from "
                    f"the face sharding -- parity on this arm would "
                    f"certify nothing. shape={leaf.shape} "
                    f"sharding={sh}")
        print(f"non-vacuity ({when}): all {len(leaves)} output leaves "
              f"carry the face sharding across {args.n_shards} devices")

    _assert_face_sharded(b, "after warmup step")
    b = _shard_bundle(ic, shard)        # fresh IC: timing excludes warmup
    t0 = time.time()
    for _ in range(args.n_steps):
        b = model.step(b, args.dt)
    jax.block_until_ready(b["state"]["pt"])
    t_shard = time.time() - t0
    _assert_face_sharded(b, f"after {args.n_steps} timed steps")
    sharded = _flatten_bundle(b)

    # ---- compare: every field, worst |d| and its scale ----
    assert set(single) == set(sharded)
    worst, worst_name = 0.0, ""
    rel_fail = []
    print(f"face-shard parity, C{args.resolution} km={args.km} "
          f"{args.n_steps} steps, {args.n_shards} shards "
          f"({devs[0].platform})"
          f"{', RING exchanges' if args.ring else ''}:")
    for name in sorted(single):
        a, s2 = single[name], sharded[name]
        # codex MAJOR: max() over a NaN diff never updates `worst`, so a
        # NaN field could PASS the gate silently. Non-finite = hard fail.
        if not (np.isfinite(a).all() and np.isfinite(s2).all()):
            raise SystemExit(
                f"NON-FINITE field {name}: single finite="
                f"{bool(np.isfinite(a).all())}, sharded="
                f"{bool(np.isfinite(s2).all())} -- parity is "
                f"meaningless, refusing to gate")
        d = float(np.abs(a - s2).max())
        sc = float(np.abs(a).max())
        print(f"  {name:12s} |d|max={d:.3e}  scale={sc:.3e}")
        if args.gate_rtol is not None and \
                d > args.gate_atol + args.gate_rtol * sc:
            rel_fail.append((name, d, sc))
        if d > worst:
            worst, worst_name = d, name
    print(f"WORST |d|max: {worst:.6e} at {worst_name}")
    if args.gate_rtol is not None:
        if rel_fail:
            for name, d, sc in rel_fail:
                print(f"GATE FAILED (rel): {name} |d|={d:.3e} > "
                      f"{args.gate_atol:.1e} + {args.gate_rtol:.1e}*{sc:.3e}")
            return 1
        print(f"GATE PASSED (per-field): |d| <= {args.gate_atol:.1e} + "
              f"{args.gate_rtol:.1e}*scale on all fields")
    print(f"wall: single {t_single:.2f}s, sharded {t_shard:.2f}s "
          f"(descriptive at this scale, NOT the scaling claim)")
    if args.max_abs is None:
        print("=== REPORT ONLY -- pass --max-abs to gate ===")
        return 0
    if worst > args.max_abs:
        print(f"GATE FAILED: {worst:.6e} > {args.max_abs:.1e}")
        return 1
    print(f"GATE PASSED: {worst:.6e} <= {args.max_abs:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
