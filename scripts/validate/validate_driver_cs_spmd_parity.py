"""2-process CPU parity smoke for the production cs_spmd driver mode.

Gate for the pinned production-CS-SPMD design (ginsburg plan addendum 6,
step 6): the SAME tiny cubed-sphere AMIP config is run

* ``--mode serial`` — single process, no sharding (the reference), final
  state saved to ``--out``;
* ``--mode spmd``   — under a multi-process launcher (``mpirun -np 2 …`` /
  ``srun``), ``distributed=True, distributed_mode='spmd'``: every process
  holds a shard of the face-sharded global state; process 0 gathers the
  final state (``multihost_utils.process_allgather``) and compares it to
  the ``--ref`` file.

Exit 0 on parity within ``--rtol/--atol``; non-zero, with a per-field
max-abs-diff table (argmax location + differing-point count), otherwise.

PARITY CONTRACT (matches the bench --cs-spmd receipts): the ppermute SPMD
halo backend deviates from the serial local corner fill by ~1e-9 at
isolated face-corner points (job 8462928: 6.7e-10 @ 5 steps; this gate,
job 8684964: one u point + one v point at 1.86e-9 after 1 step, ALL other
fields bit-exact).  That deterministic seed is chaos-amplified over long
horizons (~1e-4 after a 144-step day at C8), so the GATE is short-horizon
at the receipt tolerance — a 5-step lane with atol/rtol 5e-9.  Long lanes
are informational.

Run (compute node):
    JAX_PLATFORMS=cpu python scripts/validate/validate_driver_cs_spmd_parity.py \
        --mode serial --out /tmp/ref.npz
    JAX_PLATFORMS=cpu mpirun -np 2 python \
        scripts/validate/validate_driver_cs_spmd_parity.py \
        --mode spmd --ref /tmp/ref.npz
"""

from __future__ import annotations

import argparse
import sys


def _build_config(workdir: str):
    """Tiny C8/L4 gray-radiation dry config — identical for both modes."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere", resolution=8, nlev=4,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(discretization="centered", dt=600.0),
        # diag/checkpoint OFF by default: keeps the serial reference
        # byte-comparable (no host cadence).  The 5a/5b/5c IO gates opt in
        # via --checkpoint-days / --diag-days / --full-collect.
        output=OutputConfig(output_dir=workdir, diag_days=0,
                            checkpoint_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", microphysics="none",
        cloud_scheme="none", gravity_wave_drag="none",
    )


def _final_state_arrays(driver) -> dict:
    """Pull the driver's final prognostic state to host numpy, gathering
    sharded leaves to the GLOBAL array first (multi-controller: every
    process must dispatch the same gather)."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    out = {}
    fields = {}
    st = driver.state
    for name in st._fields:
        obj = getattr(st, name)
        data = getattr(obj, "data", obj)
        if data is None or not isinstance(data, jax.Array):
            continue
        fields[name] = data
    for name in ("q_v", "q_c", "q_r"):
        data = getattr(driver, name, None)
        if data is not None and isinstance(data, jax.Array):
            fields[f"tracer_{name}"] = data
    for name, data in fields.items():
        if jax.process_count() > 1 and not data.is_fully_addressable:
            data = multihost_utils.process_allgather(data, tiled=True)
        out[name] = np.asarray(data)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("serial", "spmd"), required=True)
    p.add_argument("--out", default=None,
                   help="npz path for the final state (serial mode).")
    p.add_argument("--ref", default=None,
                   help="serial-reference npz to compare against (spmd mode).")
    p.add_argument("--workdir", default="/tmp/cs_spmd_parity")
    p.add_argument("--days", type=float, default=1.0,
                   help="Run length [days]; fractional values bisect the "
                        "trajectory (0.00695 ~= one 600 s step).")
    p.add_argument("--radiation", default="gray",
                   help="Radiation scheme (bisect lever: 'none' isolates "
                        "the dry dycore from the physics path).")
    p.add_argument("--checkpoint-days", type=int, default=0,
                   help="Enable checkpointing at this cadence [days] "
                        "(cs_spmd step 5a gate: under --mode spmd the "
                        "gathered root-only write must produce ONE "
                        "loadable checkpoint_day_*.npz).")
    p.add_argument("--diag-days", type=int, default=0,
                   help="Enable perf-mode diagnostics at this cadence "
                        "[days] (cs_spmd step 5b gate: the scalar "
                        "SPMD-global collect must run and process 0 must "
                        "write the diagnostics file at finalize).")
    p.add_argument("--full-collect", action="store_true",
                   help="Force the FULL diagnostics collect "
                        "(diagnostics_perf_mode='never'; cs_spmd step 5c "
                        "gate: the sharded fields are gathered to host "
                        "replicas on every process before collect()).")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-9)
    args = p.parse_args(argv)

    # Argument contract FIRST — a bad invocation must fail before the
    # (expensive) driver setup/run, and before jax.distributed init.
    if args.mode == "serial" and not args.out:
        print("ERROR: --mode serial requires --out")
        return 2
    if args.mode == "spmd" and not args.ref:
        print("ERROR: --mode spmd requires --ref")
        return 2

    # Multi-controller bootstrap MUST run before any jax device use.  A
    # plain single-process launch short-circuits to (0, 1).
    from legoesm.parallel.distributed import (
        initialize_jax_distributed_multiprocess,
    )
    rank, n_procs = initialize_jax_distributed_multiprocess()

    import jax
    import numpy as np

    workdir = f"{args.workdir}/{args.mode}"
    cfg = _build_config(workdir)
    cfg = cfg._replace(days=args.days, radiation=args.radiation)
    if args.checkpoint_days > 0:
        cfg = cfg._replace(output=cfg.output._replace(
            checkpoint_days=args.checkpoint_days))
    if args.diag_days > 0:
        cfg = cfg._replace(output=cfg.output._replace(
            diag_days=args.diag_days))
    if args.full_collect:
        cfg = cfg._replace(output=cfg.output._replace(
            diagnostics_perf_mode="never"))
    if args.mode == "serial" and n_procs > 1:
        # Every rank would race/clobber the same --out (codex Medium).
        print("ERROR: --mode serial must run single-process "
              f"(got {n_procs}); use --mode spmd under a launcher.")
        return 2
    if args.mode == "spmd":
        if n_procs < 2:
            print("ERROR: --mode spmd needs a multi-process launch "
                  f"(mpirun -np 2 …); got {n_procs} process(es).")
            return 2
        cfg = cfg._replace(distributed=True, distributed_mode="spmd")

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(cfg, output_dir=workdir)
    driver.setup()
    driver.run()

    state = _final_state_arrays(driver)
    io_rank = rank == 0

    if args.mode == "serial":
        np.savez(args.out, **state)
        print(f"[serial] wrote {len(state)} fields -> {args.out}")
        return 0

    # spmd: compare on process 0 (every rank ran the same gathers above).
    # The compare is fully wrapped: an exception on rank 0 (missing /
    # corrupt --ref, ...) must still reach the exit-code broadcast below,
    # or the other ranks hang in broadcast_one_to_all (codex HIGH).
    rc = 0
    if io_rank:
        try:
            # cs_spmd step 5a gate: with checkpointing on, the gathered
            # root-only write must have produced loadable single-file
            # checkpoints (exactly one writer — process 0).
            if args.checkpoint_days > 0:
                from pathlib import Path as _Path
                _ckpts = sorted(_Path(workdir).glob("checkpoint_day_*.npz"))
                if not _ckpts:
                    print("  checkpoint: NO checkpoint_day_*.npz written "
                          f"under {workdir} FAIL")
                    rc = 1
                for _c in _ckpts:
                    _d = np.load(_c, allow_pickle=True)
                    print(f"  checkpoint: {_c.name} loadable "
                          f"({len(_d.files)} keys) OK")
            if args.diag_days > 0:
                from pathlib import Path as _Path
                # DiagnosticCollector.save writes timeseries.npz under the
                # run dir at finalize (root-gated).  Zero artifacts means
                # the perf-mode path silently wrote nothing.
                _diags = [p_ for pat in ("timeseries*.npz", "diagnostics*")
                          for p_ in _Path(workdir).glob(pat) if p_.is_file()]
                if not _diags:
                    print("  diagnostics: NO diagnostics* file written "
                          f"under {workdir} FAIL")
                    rc = 1
                else:
                    print(f"  diagnostics: {[p_.name for p_ in _diags]} OK")
            ref = np.load(args.ref)
            missing = sorted(set(ref.files) ^ set(state))
            if missing:
                print(f"FIELD-SET MISMATCH: {missing}")
                rc = 1
            for name in sorted(set(ref.files) & set(state)):
                a, b = ref[name], state[name]
                if a.shape != b.shape:
                    print(f"  {name}: SHAPE {a.shape} vs {b.shape}")
                    rc = 1
                    continue
                if a.size == 0:
                    continue
                close = np.allclose(a, b, rtol=args.rtol, atol=args.atol)
                diff = np.abs(a - b)
                max_abs = float(np.max(diff)) if a.size else 0.0
                loc = np.unravel_index(int(np.argmax(diff)), a.shape)
                n_bad = int(np.sum(~np.isclose(a, b, rtol=args.rtol,
                                               atol=args.atol)))
                print(f"  {name}: max|Δ|={max_abs:.3e} at {loc} "
                      f"({n_bad}/{a.size} pts differ) "
                      f"{'OK' if close else 'FAIL'}")
                if not close:
                    rc = 1
            print(f"[spmd] parity {'PASS' if rc == 0 else 'FAIL'} "
                  f"(np={n_procs}, {len(state)} fields)")
        except Exception as e:
            print(f"[spmd] compare ERROR on rank 0: {e!r}")
            rc = 2
    # Every process exits with the same code (rank 0 decides).
    rc_arr = jax.numpy.asarray(float(rc))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        rc_arr = multihost_utils.broadcast_one_to_all(rc_arr)
    return int(rc_arr)


if __name__ == "__main__":
    sys.exit(main())
