"""Route-B multi-controller gate for the lat-band SPMD atm step.

True multi-process ``jax.distributed`` parity: N processes (one per GPU on a
cluster; CPU host devices for the fabric-free check) federate into ONE
multi-controller program, the ("lat",) mesh spans the GLOBAL device set, and
``make_sharded_atm_latlon_step`` + the band ppermute/psum halo run UNCHANGED —
the collectives cross processes via the distributed runtime (NCCL/gloo). NO
mpi4jax anywhere in this file (mixing the mpi4jax halo machinery with
jax.distributed collectives in one program is the documented mixed-stack
deadlock hazard) — which is also why this file lives in ``tests/parallel/``,
NOT ``tests/distributed/`` (whose session conftest auto-arms the mpi4jax
layout).

GATED: skips unless ``LEGOESM_JAX_DISTRIBUTED_TEST=1`` — ``jax.distributed
.initialize`` must run BEFORE the first backend touch, so this file must be
launched ALONE (its own pytest process per rank), never inside a shared pytest
session. It is also env-fragile by nature (gloo needs a resolvable hostname on
CPU — broken on stock macOS; NCCL needs a working local topology) — the gate
keeps default CI green.

Run (2 processes x 2 host CPU devices = 4 bands):
  LEGOESM_JAX_DISTRIBUTED_TEST=1 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  mpiexec -n 2 python -m pytest \
      tests/parallel/test_atm_latlon_spmd_multicontroller.py -x -q
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
# from the LAUNCHER env ONLY (querying jax.process_count() here would
# instantiate the local backend client pre-federation, the exact order bug
# this file warns about; codex round-1 HIGH). SLURM auto-detects; OpenMPI
# needs the explicit coordinator (fixed localhost port, single-node check).
_n = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                        os.environ.get("SLURM_NTASKS", "1")))
_r = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                        os.environ.get("SLURM_PROCID", "0")))
if _n > 1:
    if os.environ.get("SLURM_STEP_NODELIST"):
        jax.distributed.initialize()
    else:
        _coord = os.environ.get("LEGOESM_JAX_COORDINATOR", "127.0.0.1:29777")
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
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (  # noqa: E402
    gather_state_atm_latlon,
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
)

# Reuse the Stage-5 gate's fixtures — same model/state/mesh conventions.
from tests.parallel.test_atm_latlon_spmd_step import (  # noqa: E402
    _model_and_state,
)


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _global_mesh():
    gdev = jax.devices()
    if jax.process_count() < 2:
        pytest.skip("needs >= 2 processes (mpiexec/srun -n 2)")
    return jax.sharding.Mesh(np.array(gdev), axis_names=("lat",))


def test_multicontroller_step_matches_serial():
    """The integrated Stage-5 gate, multi-process: N global bands across >= 2
    processes must reproduce the per-process serial reference (identical host
    ICs on every process) at the FV-PPM cut-truncation bound, and the gather
    (the multiprocess ``legoesm.parallel.latlon_spmd.replicate_leaf`` branch,
    exercised FOR REAL here) must return the full global state on every
    process."""
    mesh = _global_mesh()
    n_dev = mesh.devices.size
    model, state = _model_and_state(use_polar_filter=False)
    n_lat = int(model.grid.n_lat)
    if n_lat % n_dev != 0:
        pytest.skip(f"n_lat {n_lat} % global devices {n_dev} != 0")
    dt, n_steps = 100.0, 3

    # Serial reference: every process computes it from the identical host IC.
    s = state
    for _ in range(n_steps):
        s, _ = model._step_cgrid(s, dt, target_mass=None, physics_fn=None)

    sharded_step = make_sharded_atm_latlon_step(model, mesh)
    sc = shard_state_atm_latlon(state, mesh)
    for _ in range(n_steps):
        sc = sharded_step(sc, dt)
    out = gather_state_atm_latlon(sc, mesh)

    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(out, field))
        b = np.asarray(getattr(s, field))
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9,
            err_msg=(
                f"multi-controller lat-band SPMD ({n_dev} bands, "
                f"{jax.process_count()} processes) diverged from serial in "
                f"'{field}' — cross-process ppermute/psum or the "
                f"multi-controller gather is wrong."))

    # Non-vacuity: the FV-PPM cut truncation must be present (bands really ran).
    u_diff = float(np.max(np.abs(np.asarray(out.u) - np.asarray(s.u))))
    assert u_diff > 1e-13, (
        "multi-controller u is bit-identical to serial — the band "
        "decomposition did not actually run; this gate would be vacuous.")
