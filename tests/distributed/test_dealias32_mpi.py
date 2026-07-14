"""Distributed 3/2-rule de-aliasing == serial, under y-slab MPI.

The oracle-faithful de-aliasing pads each advection factor to the fine 3/2 grid,
multiplies alias-free in physical space, and truncates back. The 2-D spectral
zero-pad is separable: x is local (undecomposed), y uses one all-to-all transpose
(``parallel/distributed_fft.{distributed_pad_to_fine,distributed_truncate_from_fine}``).
Two 1-D inverse FFTs compose to the serial 2-D irfft2 normalisation, so the
distributed pad/truncate must reproduce the serial result to round-off, and a full
``dealias=True`` step must match single-rank.

Requires ny_local even (3·ny_local/2 integer): NY=16 ⇒ np 1/2/4 give ny_local
16/8/4 (all even). Run::

    mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_dealias32_mpi.py
    mpirun -np 4 .venv-mpi/bin/python -m pytest tests/distributed/test_dealias32_mpi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESLayout, SpectralLESState, make_grid, step,
    _pad_to_fine, _truncate_from_fine)
from legoesm.parallel.distributed_fft import (  # noqa: E402
    distributed_pad_to_fine, distributed_truncate_from_fine)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

NY, NX, NZ = 16, 16, 8


def _rand(seed, shape):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _y_slab(f, ny=NY):
    nyl = ny // NPROC
    return f[RANK * nyl:(RANK + 1) * nyl]


def _gather_y(slab):
    return jnp.asarray(np.concatenate(COMM.allgather(np.asarray(slab)), axis=0))


def _layout():
    return SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                             comm=COMM)


@pytest.mark.skipif(NY % NPROC != 0 or (NY // NPROC) % 2 != 0,
                    reason="needs ny_local even")
def test_pad_to_fine_matches_serial():
    f = _rand(1, (NY, NX, NZ))
    ser = _pad_to_fine(f)                                  # (3NY/2, 3NX/2, NZ)
    dist = distributed_pad_to_fine(_y_slab(f), ny_global=NY, nx=NX,
                                   n_ranks=NPROC, comm=COMM)
    np.testing.assert_allclose(np.asarray(_gather_y(dist)), np.asarray(ser),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(NY % NPROC != 0 or (NY // NPROC) % 2 != 0,
                    reason="needs ny_local even")
def test_truncate_from_fine_matches_serial():
    nyf, nxf = 3 * NY // 2, 3 * NX // 2
    ff = _rand(2, (nyf, nxf, NZ))
    ser = _truncate_from_fine(ff, NY, NX)
    dist = distributed_truncate_from_fine(_y_slab(ff, ny=nyf), ny_global=NY,
                                          nx=NX, n_ranks=NPROC, comm=COMM)
    np.testing.assert_allclose(np.asarray(_gather_y(dist)), np.asarray(ser),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(NY % NPROC != 0 or (NY // NPROC) % 2 != 0,
                    reason="needs ny_local even")
def test_pad_truncate_grad_matches_serial():
    f = _rand(3, (NY, NX, NZ))

    def loss_ser(x):
        return jnp.sum(_truncate_from_fine(_pad_to_fine(x), NY, NX) ** 2)

    def loss_dist(x):
        fine = distributed_pad_to_fine(x, ny_global=NY, nx=NX, n_ranks=NPROC,
                                       comm=COMM)
        back = distributed_truncate_from_fine(fine, ny_global=NY, nx=NX,
                                              n_ranks=NPROC, comm=COMM)
        return jnp.sum(back ** 2)

    gs = jax.grad(loss_ser)(f)
    gd = jax.grad(loss_dist)(_y_slab(f))
    np.testing.assert_allclose(np.asarray(_gather_y(gd)), np.asarray(gs),
                               rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(NY % NPROC != 0 or (NY // NPROC) % 2 != 0,
                    reason="needs ny_local even")
def test_dealiased_step_matches_serial():
    """Full dealias=True (3/2-rule) step() trajectory == single-rank."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=300.0, Lz=100.0,
                            dealias=True, smagorinsky_dynamic=False,
                            sgs_model="smagorinsky", spectral_filter=True,
                            time_scheme="ab2")
    dt, u_geo, f_cor, force = 2.0e-3, (1.0, 0.0), 1.0e-4, (1.0e-3, 0.0)

    def init():
        rng = np.random.default_rng(101)
        u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((NY, NX, NZ)))
        v = jnp.asarray(0.1 * rng.standard_normal((NY, NX, NZ)))
        w = np.zeros((NY, NX, NZ + 1))
        w[..., 1:NZ] = 0.05 * rng.standard_normal((NY, NX, NZ - 1))
        z = jnp.zeros((NY, NX, NZ))
        return SpectralLESState(u=u, v=v, w=jnp.asarray(w), rhs_u_prev=z,
                                rhs_v_prev=z,
                                rhs_w_prev=jnp.zeros((NY, NX, NZ + 1)))

    def run(st, g):
        for i in range(3):
            st, _ = step(st, g, dt, u_geo, f_cor, first=(i == 0), force=force)
        return st

    st_ser = run(init(), make_grid(cfg))
    gi = init()
    st0 = SpectralLESState(
        u=_y_slab(gi.u), v=_y_slab(gi.v), w=_y_slab(gi.w),
        rhs_u_prev=_y_slab(gi.rhs_u_prev), rhs_v_prev=_y_slab(gi.rhs_v_prev),
        rhs_w_prev=_y_slab(gi.rhs_w_prev))
    st_dist = run(st0, make_grid(cfg, layout=_layout()))

    for name, dist, ser in (("u", st_dist.u, st_ser.u),
                            ("v", st_dist.v, st_ser.v),
                            ("w", st_dist.w, st_ser.w)):
        np.testing.assert_allclose(
            np.asarray(_gather_y(dist)), np.asarray(ser),
            rtol=1e-9, atol=1e-9, err_msg=f"{name} mismatch")


@pytest.mark.skipif(NY % NPROC != 0 or (NY // NPROC) % 2 != 0,
                    reason="needs ny_local even")
def test_dealiased_scalar_step_matches_serial():
    """dealias=True step WITH an active θ scalar — exercises scalar_rhs's 3/2
    pad/truncate distributed path (a separate de-aliasing call site from
    advection; missed by the no-θ step test, codex-flagged 2026-06-09)."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=300.0, Lz=100.0,
                            dealias=True, smagorinsky_dynamic=False,
                            sgs_model="smagorinsky", spectral_filter=True,
                            buoyancy=True, time_scheme="ab2")
    dt, u_geo, f_cor, force = 2.0e-3, (1.0, 0.0), 1.0e-4, (1.0e-3, 0.0)

    def init():
        rng = np.random.default_rng(77)
        u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((NY, NX, NZ)))
        v = jnp.asarray(0.1 * rng.standard_normal((NY, NX, NZ)))
        w = np.zeros((NY, NX, NZ + 1))
        w[..., 1:NZ] = 0.05 * rng.standard_normal((NY, NX, NZ - 1))
        th = jnp.asarray(290.0 + rng.standard_normal((NY, NX, NZ)))
        z = jnp.zeros((NY, NX, NZ))
        return SpectralLESState(u=u, v=v, w=jnp.asarray(w), theta=th,
                                rhs_theta_prev=z, rhs_u_prev=z, rhs_v_prev=z,
                                rhs_w_prev=jnp.zeros((NY, NX, NZ + 1)))

    def run(st, g):
        for i in range(3):
            st, _ = step(st, g, dt, u_geo, f_cor, first=(i == 0), force=force,
                         sfc_theta_flux=0.05)
        return st

    st_ser = run(init(), make_grid(cfg))
    gi = init()
    st0 = SpectralLESState(
        u=_y_slab(gi.u), v=_y_slab(gi.v), w=_y_slab(gi.w), theta=_y_slab(gi.theta),
        rhs_theta_prev=_y_slab(gi.rhs_theta_prev), rhs_u_prev=_y_slab(gi.rhs_u_prev),
        rhs_v_prev=_y_slab(gi.rhs_v_prev), rhs_w_prev=_y_slab(gi.rhs_w_prev))
    st_dist = run(st0, make_grid(cfg, layout=_layout()))

    for name, dist, ser in (("u", st_dist.u, st_ser.u),
                            ("theta", st_dist.theta, st_ser.theta)):
        np.testing.assert_allclose(
            np.asarray(_gather_y(dist)), np.asarray(ser),
            rtol=1e-9, atol=1e-9, err_msg=f"{name} mismatch")
