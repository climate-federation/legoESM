"""Multi-PROCESS jax.distributed equivalence VALIDATION for the lat-band SPMD ocean
step — the multi-node correctness gate for ``make_sharded_ocean_step_global``.

Runs STANDALONE (not pytest): the root ``tests/conftest.py`` initializes the XLA
backend (``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection,
before a test module could call ``jax.distributed.initialize()`` (which must
precede backend init).  A standalone script bootstraps jax.distributed FIRST.

TWO-PHASE design (the SERIAL reference cannot run under jax.distributed):
``model.step`` (the single-device reference) runs OUTSIDE any ``shard_map``, so its
eta-floor mass-redistribution reduction (``eta_floor._global_sum_pair``) sees the
"local" halo backend and falls to the ``is_multi_process()`` branch ->
``batch_allreduce_mpi`` -> mpi4jax, which is NOT installed (the SPMD path is
pure-JAX psum, never mpi4jax).  So the reference is computed in PHASE 1 (single
process, no jax.distributed -> ``is_multi_process()`` is False -> the reduction is
the trivial local sum, no MPI) and PHASE 2 (mpirun) only runs the SPMD step (whose
in-``shard_map`` reductions route through the armed "spmd" psum backend) and
compares to the saved reference.  This mirrors the PRODUCTION ``--distributed``
path, where ``model.step`` is never called (the host loop calls the SPMD step).

* ``--phase ref --out PATH`` (single process, 4 CPU devices): compute the serial
  single-device reference + the single-process 4-device SPMD result; assert they
  agree to the FP-reassociation floor (this is the existing single-process gate);
  save the serial reference (u/v/eta/T/S) to ``PATH`` (.npz).
* ``--phase mp --ref PATH`` (mpirun -np 2, 4 global devices): bootstrap
  jax.distributed, run the MULTI-PROCESS SPMD step, and assert it matches the saved
  reference to the FP floor AND the gathered state is bitwise-replicated across the
  two processes (LAND-reduced verdict).  ``--tripole`` builds an active-north-fold
  synthetic tripole (eORCA025 is a tripole; the fold lands on the north = a REMOTE
  band).

Run (the sbatch ``run_multiprocess_cpu_equiv.sbatch`` wraps both phases)::

    # phase 1 (single process):
    env JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        JAX_ENABLE_X64=1 python <this> --phase ref --out /tmp/ref.npz
    # phase 2 (multi-process):
    mpirun -np 2 env JAX_PLATFORMS=cpu \
        XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
        python <this> --phase mp --ref /tmp/ref.npz

Exit 0 = PASS; non-zero = a phase failed (the launcher aborts).
"""
from __future__ import annotations

import argparse
import os
import sys

# JAX backend knobs — set before jax initializes.  Phase 1 forces 4 CPU devices in
# ONE process; phase 2 forces 2 per process (4 global under mpirun -np 2).  The
# launcher sets XLA_FLAGS explicitly; these are only fallback defaults.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

# Grid / run sizing (module-level so the pytest wrapper can import without MPI).
N_LAT, N_LON, NLEV = 48, 96, 10
DT, N_STEPS = 600.0, 3
ATOL, RTOL = 2.0e-4, 1.0e-3
_FIELDS = ("u", "v", "eta", "T", "S")


def _build_perturbed_state(grid, z_coord):
    """Rest state + small deterministic (seed-0) u/v/T/eta perturbations so the
    step exercises advection / Coriolis / PGF — identical to the single-process
    gate's fixture (so the serial reference matches) and identical on every
    process (the SPMD body runs the same program everywhere)."""
    import jax.numpy as jnp
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def _build_model(tripole: bool, implicit_cn: bool = False):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    if tripole:
        grid = create_synthetic_tripole(N_LAT, N_LON)
        assert grid.fold.is_active, "synthetic tripole must carry an active fold"
    else:
        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=4000.0)
    # implicit_cn exercises barotropic_implicit_latlon_cgrid's PCG + mass-
    # conservation reduction (the eORCA025 smoke uses it); its in-step reductions
    # must route through the "spmd" psum backend, not batch_allreduce_mpi.  Default
    # explicit_substep exercises the substep-loop eta-floor reduction.
    # from_flat: barotropic_solver is nested post-#501 (config.barotropic.*).
    cfg = (LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn")
           if implicit_cn else LatLonCGridOceanConfig())
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    return model, grid, z_coord


def _spmd_step_result(model, state0, n_dev):
    """Run N SPMD steps over an ``n_dev``-band lat mesh (global devices), gather."""
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step_global,
    )
    dev = create_latlon_mesh(n_devices=n_dev)
    assert dev.mesh.devices.size == n_dev, dev.mesh.devices.size
    model._ensure_vertex_mask(state0)
    step = make_sharded_ocean_step_global(model, dev.mesh)
    ss = state0
    for _ in range(N_STEPS):
        ss = step(ss, DT)
    return ss


def _phase_ref(tripole: bool, out_path: str, implicit_cn: bool = False) -> int:
    """PHASE 1 (single process, 4 CPU devices, NO jax.distributed): compute the
    serial single-device reference + the single-process 4-device SPMD result,
    assert they agree to the FP floor (the established single-process gate), and
    save the serial reference for phase 2."""
    import jax

    if jax.device_count() < 4:
        print(f"[SKIP] phase ref needs 4 local devices, have "
              f"{jax.device_count()} (set "
              "XLA_FLAGS=--xla_force_host_platform_device_count=4).")
        return 0

    model, _grid, _z = _build_model(tripole, implicit_cn=implicit_cn)
    state0 = _build_perturbed_state(model.grid, _z)

    # serial single-device reference (is_multi_process() is False here -> the
    # eta-floor reduction is the trivial local sum, NO mpi4jax).
    s = state0
    for _ in range(N_STEPS):
        s = model.step(s, DT)

    # single-process 4-device SPMD result (psum inside the shard_map; one process
    # -> no cross-process collective).
    ss = _spmd_step_result(model, state0, 4)

    ok = True
    for nm in _FIELDS:
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        try:
            np.testing.assert_allclose(b, a, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
            ok = False
            print(f"[FAIL ref] single-process SPMD {nm} != serial: "
                  f"{str(e).splitlines()[0]}")

    # Save the SERIAL reference (the source of truth phase 2 compares to).
    ref = {nm: np.asarray(getattr(s, nm).data) for nm in _FIELDS}
    np.savez(out_path, **ref)
    g = ("tripole" if tripole else "latlon") + ("/implicit_cn" if implicit_cn
                                                 else "")
    print(f"[{'PASS' if ok else 'FAIL'} ref] single-process SPMD ({g}) == serial "
          f"to FP floor; saved serial reference -> {out_path}", flush=True)
    return 0 if ok else 1


def _phase_mp(tripole: bool, ref_path: str, implicit_cn: bool = False) -> int:
    """PHASE 2 (mpirun, 4 global devices): bootstrap jax.distributed, run the
    MULTI-PROCESS SPMD step, compare to the phase-1 serial reference + assert the
    gathered state is bitwise-replicated across processes."""
    try:
        from mpi4py import MPI
    except ImportError:
        print("[SKIP] mpi4py not installed — cannot run the multi-process phase.")
        return 0

    from legoesm.parallel.distributed import (
        initialize_jax_distributed_multiprocess,
    )
    rank, nproc = initialize_jax_distributed_multiprocess()
    if nproc < 2:
        print("[SKIP] phase mp launched single-process (MPI size 1); needs "
              "mpirun -np 2 (the cross-process path is the point).")
        return 0

    import jax
    comm = MPI.COMM_WORLD

    if jax.device_count() < 4:
        if rank == 0:
            print(f"[SKIP] only {jax.device_count()} global devices; need 4 "
                  "(2 per process x np 2).")
        return 0

    ref = np.load(ref_path)             # serial reference from phase 1
    model, _grid, _z = _build_model(tripole, implicit_cn=implicit_cn)
    state0 = _build_perturbed_state(model.grid, _z)

    if rank == 0:
        _gl = ("tripole" if tripole else "latlon") + ("/implicit_cn"
                                                      if implicit_cn else "")
        print(f"[setup] {nproc} processes, {jax.device_count()} global devices "
              f"({jax.local_device_count()} local/proc), grid={_gl}", flush=True)

    # MULTI-PROCESS SPMD step: the lat mesh spans the GLOBAL device set (all 4
    # across both processes); the band halo + barotropic reductions cross the
    # process boundary via ppermute/psum (the "spmd" backend armed per-call inside
    # make_sharded_ocean_step).  Exactly the run_omip_core2 --distributed step.
    ss = _spmd_step_result(model, state0, 4)

    local_ok = True
    msgs = []
    # (1) gathered SPMD step matches the phase-1 serial reference to the FP floor.
    for nm in _FIELDS:
        a = ref[nm]
        b = np.asarray(getattr(ss, nm).data)
        try:
            np.testing.assert_allclose(b, a, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
            local_ok = False
            msgs.append(f"{nm}: {str(e).splitlines()[0]}")

    # (2) gathered shapes are GLOBAL (full domain, not a band).
    T_g = np.asarray(ss.T.data)
    if T_g.shape != (N_LAT, N_LON, NLEV):
        local_ok = False
        msgs.append(f"gathered T shape {T_g.shape} != ({N_LAT},{N_LON},{NLEV})")
    if np.asarray(ss.v.data).shape != (N_LAT + 1, N_LON, NLEV):
        local_ok = False
        msgs.append(f"gathered v shape {np.asarray(ss.v.data).shape} not n_lat+1")

    # (3) gathered state bitwise-identical across processes (replicated all-gather
    #     -> every process' host loop sees the same state).
    T_rank0 = comm.bcast(T_g if rank == 0 else None, root=0)
    if not np.array_equal(T_g, T_rank0):
        local_ok = False
        msgs.append(f"gathered T differs across processes "
                    f"(max|d|={float(np.max(np.abs(T_g - T_rank0))):.3e})")

    if not local_ok:
        print(f"[FAIL rank {rank}/{nproc}] " + " | ".join(msgs), flush=True)

    # All ranks must pass — LAND-reduce so the exit code is the WHOLE-job verdict.
    all_ok = comm.allreduce(local_ok, op=MPI.LAND)
    g = ("tripole" if tripole else "latlon") + ("/implicit_cn" if implicit_cn
                                                 else "")
    if rank == 0:
        if all_ok:
            print(f"[PASS] multi-process SPMD ({g}) == serial reference to FP "
                  f"floor (atol={ATOL}, rtol={RTOL}); gather replicated bitwise "
                  f"across {nproc} processes.", flush=True)
        else:
            print(f"[FAIL] multi-process SPMD ({g}) equivalence gate FAILED.",
                  flush=True)
    return 0 if all_ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["ref", "mp"], required=True,
                   help="ref = single-process serial+SPMD reference (writes --out); "
                        "mp = multi-process SPMD vs the saved reference (reads --ref)")
    p.add_argument("--out", default="/tmp/_spmd_mp_ref.npz",
                   help="(phase ref) where to write the serial reference .npz")
    p.add_argument("--ref", default="/tmp/_spmd_mp_ref.npz",
                   help="(phase mp) the serial reference .npz from phase ref")
    p.add_argument("--tripole", action="store_true",
                   help="active-north-fold synthetic tripole (eORCA025 is tripole; "
                        "the fold lands on the north band = a remote process)")
    p.add_argument("--implicit-cn", action="store_true",
                   help="barotropic_solver='implicit_cn' (the eORCA025 smoke "
                        "solver) -- exercises the PCG + mass-conservation reduction "
                        "SPMD psum path in barotropic_implicit_latlon_cgrid")
    args = p.parse_args(argv)
    if args.phase == "ref":
        return _phase_ref(args.tripole, args.out, implicit_cn=args.implicit_cn)
    return _phase_mp(args.tripole, args.ref, implicit_cn=args.implicit_cn)


if __name__ == "__main__":
    sys.exit(main())
