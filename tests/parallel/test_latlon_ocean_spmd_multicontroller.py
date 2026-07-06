"""Route-B multi-controller gate for the lat-band SPMD ocean step.

The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
federate via ``jax.distributed`` into ONE multi-controller program, the
("lat",) mesh spans the GLOBAL device set, and ``make_sharded_ocean_step`` +
the band ppermute/psum halo (incl. the barotropic ``_global_sum_pair`` psum)
run UNCHANGED — collectives cross processes via the distributed runtime
(NCCL/gloo). NO mpi4jax anywhere in this file (mixed-stack deadlock hazard),
which is also why it lives in ``tests/parallel/``, NOT ``tests/distributed/``
(whose session conftest auto-arms the mpi4jax layout).

GATED: skips unless ``LEGOESM_JAX_DISTRIBUTED_TEST=1`` — ``jax.distributed
.initialize`` must run BEFORE the first backend touch, so this file must be
launched ALONE (its own pytest process per rank), never inside a shared pytest
session. Env-fragile by nature (gloo hostname on CPU, NCCL topology on GPU) —
the gate keeps default CI green.

Run (2 processes x 2 host CPU devices = 4 bands):
  LEGOESM_JAX_DISTRIBUTED_TEST=1 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
  mpiexec -n 2 python -m pytest \
      tests/parallel/test_latlon_ocean_spmd_multicontroller.py -x -q
(SLURM: srun -n 2 ... — coordinator auto-detected, OMPI env vars unneeded.)
"""
from __future__ import annotations

import os

import pytest

if os.environ.get("LEGOESM_JAX_DISTRIBUTED_TEST") != "1":
    pytest.skip(
        "multi-controller jax.distributed test: set "
        "LEGOESM_JAX_DISTRIBUTED_TEST=1 and launch this file alone under "
        "mpiexec/srun (see module docstring)",
        allow_module_level=True,
    )

import jax  # noqa: E402  (import gated so the skip never touches the backend)

# initialize() BEFORE any backend touch — the process count/id decision comes
# from the LAUNCHER env ONLY (a jax.process_count() query here would
# instantiate the local backend client pre-federation; codex atm round-1 HIGH).
_n = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                        os.environ.get("SLURM_NTASKS", "1")))
_r = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                        os.environ.get("SLURM_PROCID", "0")))
if _n > 1:
    if os.environ.get("SLURM_STEP_NODELIST"):
        jax.distributed.initialize()
    else:
        _coord = os.environ.get("LEGOESM_JAX_COORDINATOR", "127.0.0.1:29778")
        jax.distributed.initialize(
            coordinator_address=_coord, num_processes=_n, process_id=_r)
    # Verify federation AFTER init (safe to touch the backend now).
    if jax.process_count() != _n:
        pytest.skip(
            f"jax.distributed federated {jax.process_count()} processes, "
            f"launcher started {_n} — environment did not federate",
            allow_module_level=True,
        )

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: E402
    gather_state_latlon,
    make_sharded_ocean_step,
    shard_state_latlon,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402

# Reuse the single-process gate's perturbed-IC fixture — same conventions.
from tests.parallel.test_latlon_ocean_spmd_step import (  # noqa: E402
    _perturbed_state,
)


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _global_mesh():
    if jax.process_count() < 2:
        pytest.skip("needs >= 2 processes (mpiexec/srun -n 2)")
    return jax.sharding.Mesh(np.array(jax.devices()), axis_names=("lat",))


def test_multicontroller_ocean_step_matches_serial():
    """The single-process SPMD equivalence gate, multi-process: N global bands
    across >= 2 processes must reproduce the per-process serial reference
    (identical host ICs on every process) at the split-explicit re-association
    floor (same atol/rtol as test_latlon_ocean_spmd_step — NOT a bug margin,
    see its tolerance note), and the gather (the multiprocess
    ``replicate_leaf`` branch, exercised FOR REAL here) must return the full
    global state on every process."""
    mesh = _global_mesh()
    n_dev = mesh.devices.size
    n_lat, n_lon, nlev = 48, 96, 10
    if n_lat % n_dev != 0:
        pytest.skip(f"n_lat {n_lat} % global devices {n_dev} != 0")
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat())
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # Serial reference: every process computes it from the identical host IC.
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    # Prime the build-once vertex-mask cache from the CONCRETE state (the
    # serial run above already did; belt-and-braces for wrapper band masks).
    model._ensure_vertex_mask(state0)

    step = make_sharded_ocean_step(model, mesh)
    ss = shard_state_latlon(state0, mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    out = gather_state_latlon(ss, mesh)

    # Tolerance = the split-explicit FP re-association floor of the sharded
    # barotropic (see test_latlon_ocean_spmd_step's tolerance note verbatim).
    atol, rtol = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(out, nm).data)
        assert a.shape == b.shape, f"{nm} shape {b.shape} vs {a.shape}"
        np.testing.assert_allclose(
            b, a, atol=atol, rtol=rtol,
            err_msg=(
                f"multi-controller lat-band SPMD ocean ({n_dev} bands, "
                f"{jax.process_count()} processes) diverged from serial in "
                f"'{nm}' — cross-process ppermute/psum or the multi-controller "
                f"gather is wrong."))

    # Non-vacuity: the sharded split-explicit re-association must be present
    # (bands really ran) — a bit-identical result would mean the sharding
    # silently collapsed to single-device.
    u_diff = float(np.max(np.abs(
        np.asarray(out.u.data) - np.asarray(s.u.data))))
    assert u_diff > 1e-13, (
        "multi-controller ocean u is bit-identical to serial — the band "
        "decomposition did not actually run; this gate would be vacuous.")
