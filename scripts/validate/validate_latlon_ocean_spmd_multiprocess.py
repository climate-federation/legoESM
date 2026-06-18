"""Multi-PROCESS jax.distributed equivalence VALIDATION for the lat-band SPMD ocean
step — the multi-node correctness gate for ``make_sharded_ocean_step_global``.

This runs as a STANDALONE script under an MPI launcher (NOT pytest): the root
``tests/conftest.py`` calls ``ensure_metal_or_fallback()`` -> ``jax.default_backend()``
at collection, which initializes the XLA backend BEFORE a test module could call
``jax.distributed.initialize()`` (which must precede backend init).  A standalone
script bootstraps jax.distributed FIRST, with no conftest in the way (codex HIGH).

What it proves (CLAUDE.md "single-rank -> smallest distributed"): launch
``mpirun -np 2``; each process forces 2 CPU devices
(``--xla_force_host_platform_device_count=2``) so there are 4 CPU devices GLOBALLY
(2 local per process).  The bootstrap initializes jax.distributed (rank-0 hostname
coordinator); ``jax.devices()`` then spans all 4.  The gathered lat-band SPMD step
must match the SERIAL single-device step to the FP-reassociation floor (the same
tolerance the single-process gate uses — the residual is the split-explicit
barotropic ppermute/psum re-association, not a bug), AND the gathered state must be
bitwise-identical across both processes (so the OMIP host loop stays consistent).

Run (the sbatch ``_run_multiprocess_cpu_equiv.sbatch`` wraps this)::

    mpirun -np 2 env JAX_PLATFORMS=cpu \
      XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
      python scripts/validate/validate_latlon_ocean_spmd_multiprocess.py

Exit code 0 = PASS on every rank; non-zero = a rank failed (the launcher aborts).
A direct pytest wrapper (``tests/parallel/test_latlon_ocean_spmd_multiprocess.py``)
subprocess-launches this under mpirun for CI coverage.
"""
from __future__ import annotations

import os
import sys

# Two CPU devices PER PROCESS -> 4 global under ``mpirun -np 2``.  Must be set
# before jax initializes its backend.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

# n_lat divisible by the global device / band count (4).  Module-level so the
# pytest wrapper can import them without launching MPI.
N_LAT, N_LON, NLEV = 48, 96, 10
DT, N_STEPS = 600.0, 3
ATOL, RTOL = 2.0e-4, 1.0e-3


def _build_perturbed_state(grid, z_coord):
    """Rest state + small deterministic (seed-0) u/v/T/eta perturbations so the
    step exercises advection / Coriolis / PGF — identical to the single-process
    gate's fixture (so the serial reference matches) and identical on every
    process (the host loop runs the same Python everywhere)."""
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


def main(tripole: bool = False) -> int:
    # mpi4py FIRST (rank/size), then the jax.distributed bootstrap, then any jax
    # device query / array op.  NOTE: importing ``legoesm.parallel.distributed``
    # imports the ``jax`` MODULE, but that is not a backend/device query — the
    # backend only initializes on the first ``jax.devices()`` / array op, which is
    # AFTER ``initialize_jax_distributed_multiprocess`` here.  So the requirement
    # "jax.distributed.initialize before the first JAX device query/array op" holds.
    try:
        from mpi4py import MPI
    except ImportError:
        print("[SKIP] mpi4py not installed — cannot run the multi-process gate.")
        return 0

    from legoesm.parallel.distributed import (
        initialize_jax_distributed_multiprocess,
    )
    rank, nproc = initialize_jax_distributed_multiprocess()
    if nproc < 2:
        print("[SKIP] launched single-process (MPI size 1); the multi-process "
              "gate needs mpirun -np 2 (the cross-process path is the point).")
        return 0

    import jax
    comm = MPI.COMM_WORLD

    if jax.device_count() < 4:
        if rank == 0:
            print(f"[SKIP] only {jax.device_count()} global devices; need 4 "
                  "(2 per process x np 2 via "
                  "--xla_force_host_platform_device_count=2).")
        return 0

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step_global,
    )

    if rank == 0:
        print(f"[setup] {nproc} processes, {jax.device_count()} global devices "
              f"({jax.local_device_count()} local/proc), "
              f"grid={'tripole' if tripole else 'latlon'}", flush=True)

    # TRIPOLE variant (--tripole): the bipolar north fold is selected on the NORTH
    # band (axis_index==N-1), which under multi-process is a REMOTE process.  This
    # is the only case that exercises the north-fold band placed on another process
    # IN the same shard_map program as the cross-process ppermute/psum (codex MED:
    # the regular-grid multi-process case + the single-process tripole case don't
    # individually cover it).  eORCA025 is a tripole, so this is the faithful path.
    if tripole:
        grid = create_synthetic_tripole(N_LAT, N_LON)
        assert grid.fold.is_active, "synthetic tripole must carry an active fold"
    else:
        grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord, LatLonCGridOceanConfig())
    state0 = _build_perturbed_state(grid, z_coord)

    # The lat-band mesh spans the GLOBAL device set (all 4 across both processes
    # under jax.distributed) — the exact call run_omip_core2 --distributed makes.
    dev = create_latlon_mesh(n_devices=4)
    assert dev.mesh.devices.size == 4, dev.mesh.devices.size

    # serial single-device reference (host-side; identical on every process).
    s = state0
    for _ in range(N_STEPS):
        s = model.step(s, DT)

    # global-in/global-out SPMD step (scatter -> cross-process band step -> gather).
    model._ensure_vertex_mask(state0)
    step = make_sharded_ocean_step_global(model, dev.mesh)
    ss = state0
    for _ in range(N_STEPS):
        ss = step(ss, DT)            # cross-process ppermute/psum + all-gather

    # (1) gathered SPMD step matches the serial reference to the FP floor.
    local_ok = True
    msgs = []
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        try:
            np.testing.assert_allclose(b, a, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
            local_ok = False
            msgs.append(f"{nm}: {str(e).splitlines()[0]}")

    # (2) gathered shapes are GLOBAL (full domain, not a band) — the host loop
    #     must see the whole grid.
    T_g = np.asarray(ss.T.data)
    if T_g.shape != (N_LAT, N_LON, NLEV):
        local_ok = False
        msgs.append(f"gathered T shape {T_g.shape} != global ({N_LAT},{N_LON},{NLEV})")
    if np.asarray(ss.v.data).shape != (N_LAT + 1, N_LON, NLEV):
        local_ok = False
        msgs.append(f"gathered v shape {np.asarray(ss.v.data).shape} not n_lat+1 rows")

    # (3) gathered state is bitwise-identical across processes (the replicated
    #     all-gather gives every process the same array -> consistent host BCs).
    T_rank0 = comm.bcast(T_g if rank == 0 else None, root=0)
    if not np.array_equal(T_g, T_rank0):
        local_ok = False
        d = float(np.max(np.abs(T_g - T_rank0)))
        msgs.append(f"gathered T differs across processes (max|d|={d:.3e})")

    if not local_ok:
        print(f"[FAIL rank {rank}/{nproc}] " + " | ".join(msgs), flush=True)

    # All ranks must pass — reduce the boolean so the launcher's exit code reflects
    # the WHOLE job (a single-rank failure fails the gate).
    all_ok = comm.allreduce(local_ok, op=MPI.LAND)
    _g = "tripole" if tripole else "latlon"
    if rank == 0:
        if all_ok:
            print(f"[PASS] multi-process SPMD ({_g}) == serial to FP floor "
                  f"(atol={ATOL}, rtol={RTOL}); gather replicated bitwise across "
                  f"{nproc} processes.", flush=True)
        else:
            print(f"[FAIL] multi-process SPMD ({_g}) equivalence gate FAILED (see "
                  "per-rank messages above).", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    # --tripole runs the bipolar-north-fold variant (the fold lands on the north
    # band = a remote process).  Default = regular lat-lon.  Both are launched by
    # the CPU-equiv sbatch so multi-node coverage includes the eORCA025 fold path.
    _tripole = "--tripole" in sys.argv[1:]
    sys.exit(main(tripole=_tripole))
