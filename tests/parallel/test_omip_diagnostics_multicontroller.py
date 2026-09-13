"""OMIP diagnostic scalars under real multi-controller sharding (2 processes).

The single-process gate (``tests/unit/test_run_omip_sharded_diagnostics.py``)
pins the numbers; this one pins the thing that actually broke the 1/12 degree
runs: with one process per device, a process holds only its own band, so any
host conversion of a whole state field is not merely expensive, it is illegal
(the remote shards are not addressable).  A run that gathered the global state
each diagnostic sample exhausted device memory on a 7.4 GiB allocation and hung
another arm; these scalars must come from reductions instead.

Launch (this file alone, backend untouched before federation):

    LEGOESM_JAX_DISTRIBUTED_TEST=1 JAX_PLATFORMS=cpu srun -n 2 \\
        python -m pytest tests/parallel/test_omip_diagnostics_multicontroller.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

if os.environ.get("LEGOESM_JAX_DISTRIBUTED_TEST") != "1":
    pytest.skip(
        "multi-controller jax.distributed test: set "
        "LEGOESM_JAX_DISTRIBUTED_TEST=1 and launch this file alone under "
        "mpiexec/srun (see module docstring)",
        allow_module_level=True,
    )

import jax  # noqa: E402  (import gated so the skip never touches the backend)
import jax.experimental.multihost_utils  # noqa: E402

# initialize() BEFORE any backend touch — the process count/id decision comes
# from the LAUNCHER env ONLY (same order contract as the atmosphere lane's
# multi-controller gate).
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
    if jax.process_count() != _n:
        pytest.skip(
            f"jax.distributed federated {jax.process_count()} processes, "
            f"launcher started {_n} — environment did not federate",
            allow_module_level=True,
        )

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: E402
    shard_state_latlon,
)
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    rest_state_latlon_cgrid_ocean,
)
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_N_LAT, _N_LON, _NLEV = 16, 32, 4


def _run_omip():
    spec = importlib.util.spec_from_file_location(
        "run_omip_mc_diag", _ROOT / "scripts" / "run" / "run_omip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["run_omip.py"]
    spec.loader.exec_module(mod)
    return mod


def _state():
    """Identical on every process (fixed seed), so the serial reference each
    process computes locally is the same state the sharded one is built from.
    """
    grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
    z_coord = create_ocean_z_star(n_levels=_NLEV, H_max=4000.0)
    land = np.ones((_N_LAT, _N_LON))
    land[:2, :4] = 0.0
    st = rest_state_latlon_cgrid_ocean(grid, z_coord,
                                       land_mask_override=jnp.asarray(land))
    rng = np.random.default_rng(0)
    return st._replace(
        T=st.T.replace(data=jnp.asarray(rng.normal(10.0, 2.0, st.T.data.shape))),
        S=st.S.replace(data=jnp.asarray(rng.normal(35.0, 0.5, st.S.data.shape))),
        eta=st.eta.replace(data=jnp.asarray(
            rng.normal(0.0, 0.01, st.eta.data.shape))),
        u=st.u.replace(data=jnp.asarray(rng.normal(0.0, 0.2, st.u.data.shape))),
        # NON-zero data on the dead top v-face row: the serial branch has to
        # mask it (the shard-time contract only checks the MASK), or the two
        # layouts disagree there
        v=st.v.replace(data=jnp.asarray(
            rng.normal(0.0, 0.2, st.v.data.shape))),
    ), grid, z_coord


def test_scalars_agree_with_serial_across_processes():
    if jax.process_count() < 2:
        pytest.skip("needs >= 2 processes (mpiexec/srun -n 2)")
    run_omip = _run_omip()
    state, grid, z_coord = _state()
    ref = run_omip._extract_scalars(state, "tripole", grid, z_coord)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("lat",))
    got = run_omip._extract_scalars(
        shard_state_latlon(state, mesh), "tripole", grid, z_coord)

    for k in ref:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, atol=1e-10,
                                   err_msg=f"scalar {k} (rank {_r})")
    # every process must agree on the reduced scalars, or the diagnostic is
    # reporting a band, not the ocean
    allv = jax.experimental.multihost_utils.process_allgather(
        jnp.asarray([got["SST"], got["SSS"], got["SSH"], got["max_speed"],
                     got["P_bt"]]))
    allv = np.asarray(allv)
    for r in range(1, allv.shape[0]):
        np.testing.assert_allclose(allv[r], allv[0], rtol=1e-12, atol=1e-12,
                                   err_msg=f"rank {r} disagrees with rank 0")
