"""Cube-face-scatter of the multilayer (Richards) land columns under MPI.

The multilayer land tile is embarrassingly parallel — each column solves an
independent soil/snow/canopy vertical problem, with NO lateral coupling — so
partitioning the columns onto cube faces and advancing each rank's owned faces
must reproduce the single-process (all-columns) advance BIT-for-BIT. This gates
the driver's ``_scatter_flat_columns`` / ``_gather_flat_columns`` helpers (the
flattened-column <-> face reshape that ``ModelDriver._setup_parallel`` uses to
scatter the soil state under cube-face MPI) and the face-partition-invariance
claim behind lifting the single-rank-only guard.

Run:
    mpirun -np 2 .venv/bin/python -m pytest \
        tests/distributed/test_multilayer_land_scatter_mpi.py
    mpirun -np 3 ...   mpirun -np 6 ...
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.core.coupling_fields import AtmToSurface  # noqa: E402
from legoesm.driver.model_driver import (  # noqa: E402
    _gather_flat_columns,
    _map_flat_column_leaves,
    _scatter_flat_columns,
)
from legoesm.land.config import MultiLayerLandConfig  # noqa: E402
from legoesm.land.multilayer_land import (  # noqa: E402
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.parallel.layout import make_layout  # noqa: E402

_N = 4                       # per-face resolution -> 6*4*4 = 96 columns
_NCOL = 6 * _N * _N


def _spatially_varying_forcing(ncol, seed):
    """Per-column DISTINCT forcing so a face-mixing scatter bug shows up
    (identical columns would hide a wrong permutation)."""
    rng = np.random.default_rng(seed)
    def f(lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, size=ncol))
    return AtmToSurface(
        sw_down=f(100.0, 400.0), lw_down=f(250.0, 400.0),
        precip_total=f(0.0, 2e-4), precip_snow=jnp.zeros(ncol),
        T_lowest=f(270.0, 300.0), q_lowest=f(0.002, 0.015),
        u_lowest=f(-8.0, 8.0), v_lowest=f(-8.0, 8.0),
        p_lowest=f(94000.0, 96000.0), p_surface=f(99000.0, 101000.0),
        rho_lowest=f(1.0, 1.3), cos_zenith=f(0.1, 0.9),
        co2_ppmv=jnp.full(ncol, 400.0),
        has_radiation=jnp.ones(ncol), has_precipitation=jnp.ones(ncol),
    )


def _scatter_forcing(forcing, layout, n):
    return AtmToSurface(**{
        k: _scatter_flat_columns(jnp.asarray(v), layout, n)
        for k, v in forcing._asdict().items()
    })


def _rank():
    c = MPI.COMM_WORLD
    return c.Get_rank(), c.Get_size()


def _advance(state, forcing, config, n_steps):
    for _ in range(n_steps):
        state, _resp, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0)
    return state


def test_scattered_land_advance_matches_serial():
    rank, nproc = _rank()
    if nproc not in (2, 3, 6):
        pytest.skip("run under mpirun -np {2,3,6} (6 must divide n_ranks)")
    if 6 % nproc:
        pytest.skip("n_ranks must divide the 6 cube faces")

    config = MultiLayerLandConfig()
    n_steps = 4
    # Global state + forcing — IDENTICAL on every rank (deterministic seeds).
    global_state = init_multilayer_land_state(_NCOL, config, T_init=285.0)
    # Perturb the soil per column so columns are distinguishable.
    rng = np.random.default_rng(0)
    global_state = global_state._replace(
        theta_soil=jnp.asarray(
            np.asarray(global_state.theta_soil)
            * rng.uniform(0.8, 1.0, size=global_state.theta_soil.shape)),
        T_soil=jnp.asarray(
            np.asarray(global_state.T_soil)
            + rng.uniform(-5.0, 5.0, size=global_state.T_soil.shape)),
    )
    global_forcing = _spatially_varying_forcing(_NCOL, seed=1)

    # --- Serial reference: advance ALL columns on one process ---
    ref = _advance(global_state, global_forcing, config, n_steps)

    # --- Distributed: scatter to owned faces, advance locally, gather ---
    layout = make_layout(rank, nproc, _N)
    local_state = _map_flat_column_leaves(
        global_state, _N, _NCOL,
        lambda x, n: _scatter_flat_columns(x, layout, n))
    local_forcing = _scatter_forcing(global_forcing, layout, _N)
    local_out = _advance(local_state, local_forcing, config, n_steps)

    # Gather every per-column leaf back to global and compare on rank 0. The
    # gather matches leaves by the LOCAL column count (owned faces * n * n).
    n_owned = len(layout.ownership.face_ids)
    local_ncol = n_owned * _N * _N
    gathered = _map_flat_column_leaves(
        local_out, _N, local_ncol,
        lambda x, n: _gather_flat_columns(x, layout, n, root_only=True))

    if rank == 0:
        for name in ("T_soil", "theta_soil", "psi_soil",
                     "runoff_surface", "snow_depth"):
            g = np.asarray(getattr(gathered, name))
            r = np.asarray(getattr(ref, name))
            np.testing.assert_allclose(
                g, r, rtol=0.0, atol=1e-12,
                err_msg=f"scattered land '{name}' != serial on {nproc} ranks")


def test_scatter_gather_roundtrip():
    """Gather ∘ scatter is identity (per-column, all ranks reconstruct the
    same global array — a rank-agreement / ordering check)."""
    rank, nproc = _rank()
    if 6 % nproc:
        pytest.skip("n_ranks must divide the 6 cube faces")
    layout = make_layout(rank, nproc, _N)
    # (ncol,) and (ncol, nlayers) both round-trip.
    for trailing in ((), (5,)):
        arr = jnp.asarray(
            np.arange(_NCOL * int(np.prod(trailing) or 1), dtype=np.float64)
            .reshape((_NCOL,) + trailing))
        local = _scatter_flat_columns(arr, layout, _N)
        back = _gather_flat_columns(local, layout, _N, root_only=False)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(arr))
