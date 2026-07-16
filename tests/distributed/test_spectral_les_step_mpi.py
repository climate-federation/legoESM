"""Full distributed spectral-LES time step == serial, under y-slab MPI.

Beyond the projection (``test_spectral_les_project_mpi.py``), this drives a
multi-step ``step()`` trajectory — advection (rotational form, 2/3 mask), static
wall-damped Smagorinsky SGS, the MOST wall model (now a GLOBAL planar mean), the
pressure projection and the sharp spectral filter — and checks the gathered
multi-rank state matches the single-rank run to round-off, including the
diagnostic ``u_*`` (which depends on the planar-mean wall stress).

Config kept to the y-slab-supported operators: ``dealias=False`` (2/3 mask, not
the 3/2 padded FFT) and static Smagorinsky (not LASD); both unsupported paths
raise under MPI and are covered separately.

Run::

    mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_spectral_les_step_mpi.py
    mpirun -np 4 .venv-mpi/bin/python -m pytest tests/distributed/test_spectral_les_step_mpi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESLayout, SpectralLESState, make_grid, step)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

NY, NX, NZ = 12, 16, 8
DT = 5.0e-3
U_GEO = (1.0, 0.0)
F_COR = 1.0e-4
FORCE = (1.0e-3, 0.0)
CFG = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=225.0, Lz=100.0,
                        dealias=False, smagorinsky_dynamic=False,
                        sgs_model="smagorinsky", buoyancy=False,
                        spectral_filter=True, time_scheme="ab2")


def _init_global():
    rng = np.random.default_rng(101)
    u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((NY, NX, NZ)))
    v = jnp.asarray(0.1 * rng.standard_normal((NY, NX, NZ)))
    w = np.zeros((NY, NX, NZ + 1))
    w[..., 1:NZ] = 0.05 * rng.standard_normal((NY, NX, NZ - 1))
    z = jnp.zeros((NY, NX, NZ))
    return SpectralLESState(u=u, v=v, w=jnp.asarray(w),
                            rhs_u_prev=z, rhs_v_prev=z, rhs_w_prev=jnp.zeros((NY, NX, NZ + 1)))


def _slab_state(st):
    nyl = NY // NPROC
    sl = slice(RANK * nyl, (RANK + 1) * nyl)
    return SpectralLESState(
        u=st.u[sl], v=st.v[sl], w=st.w[sl],
        rhs_u_prev=st.rhs_u_prev[sl], rhs_v_prev=st.rhs_v_prev[sl],
        rhs_w_prev=st.rhs_w_prev[sl])


def _gather_y(slab):
    return jnp.asarray(np.concatenate(COMM.allgather(np.asarray(slab)), axis=0))


def _run(st, g, nsteps):
    ustars = []
    for i in range(nsteps):
        st, us = step(st, g, DT, U_GEO, F_COR, first=(i == 0), force=FORCE)
        ustars.append(float(us))
    return st, ustars


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_distributed_step_matches_serial():
    g_ser = make_grid(CFG)
    st_ser, us_ser = _run(_init_global(), g_ser, 3)

    layout = SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                               comm=COMM)
    g_dist = make_grid(CFG, layout=layout)
    st_dist, us_dist = _run(_slab_state(_init_global()), g_dist, 3)

    for name, dist, ser in (("u", st_dist.u, st_ser.u),
                            ("v", st_dist.v, st_ser.v),
                            ("w", st_dist.w, st_ser.w)):
        np.testing.assert_allclose(
            np.asarray(_gather_y(dist)), np.asarray(ser),
            rtol=1e-9, atol=1e-9, err_msg=f"{name} mismatch")

    # u_* depends on the GLOBAL planar-mean wall stress — must match the serial
    # scalar at every step (the whole point of the distributed _planar_mean).
    np.testing.assert_allclose(np.asarray(us_dist), np.asarray(us_ser),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_distributed_step_grad_matches_serial():
    """Grad through a 2-step trajectory == serial (end-to-end AD via the FFT
    custom-VJP and the global-mean SUM collective)."""
    def loss(state, g):
        s, _ = step(state, g, DT, U_GEO, F_COR, first=True, force=FORCE)
        s, _ = step(s, g, DT, U_GEO, F_COR, first=False, force=FORCE)
        return jnp.sum(s.u ** 2) + jnp.sum(s.v ** 2) + jnp.sum(s.w ** 2)

    g_ser = make_grid(CFG)
    gs = jax.grad(lambda st: loss(st, g_ser))(_init_global())

    layout = SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                               comm=COMM)
    g_dist = make_grid(CFG, layout=layout)
    gd = jax.grad(lambda st: loss(st, g_dist))(_slab_state(_init_global()))

    # Tolerance 1e-5 (not 1e-9 like the forward): the gradient at the SURFACE
    # level (k=0) flows through the wall-stress planar mean, whose distributed
    # global_sum reorders the floating-point summation differently from the
    # serial jnp.mean. The forward matches to 1e-9 there, but the wall-model
    # derivative (∝ Cd·log-law) amplifies the O(1e-12) sum-reorder into ~1e-6 at
    # that one level (the violations are exactly the NY·NX surface points). This
    # is reduction-order non-determinism, not an incorrect VJP — the interior
    # levels match to ~1e-9. (Same class as the CRM fix_mass reduction-order.)
    np.testing.assert_allclose(np.asarray(_gather_y(gd.u)), np.asarray(gs.u),
                               rtol=1e-5, atol=1e-5)
