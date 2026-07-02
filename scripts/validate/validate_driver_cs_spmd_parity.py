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

Exit 0 on parity within ``--rtol/--atol`` (defaults match the proven bench
receipts: shard-local parity ~6.7e-10 @ 5 steps, job 8462928); non-zero,
with a per-field max-abs-diff table, otherwise.

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
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere", resolution=8, nlev=4,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(discretization="centered", dt=600.0),
        # diag/checkpoint OFF: required by the spmd milestone-1 refusal and
        # keeps the serial reference byte-comparable (no host cadence).
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
    import numpy as np
    import jax
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

    import numpy as np
    import jax

    workdir = f"{args.workdir}/{args.mode}"
    cfg = _build_config(workdir)
    cfg = cfg._replace(days=args.days, radiation=args.radiation)
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
    rc = 0
    if io_rank:
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
    # Every process exits with the same code (rank 0 decides).
    rc_arr = jax.numpy.asarray(float(rc))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        rc_arr = multihost_utils.broadcast_one_to_all(rc_arr)
    return int(rc_arr)


if __name__ == "__main__":
    sys.exit(main())
