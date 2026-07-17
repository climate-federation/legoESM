"""Distributed spectral-LES pressure projection == serial, under y-slab MPI.

The pressure PROJECTION (``project``) is the core incompressibility solve of the
spectral LES and the first operator wired onto the distributed 2-D FFT
(``parallel/distributed_fft.py``). It exercises the forward/inverse distributed
FFT, the per-kx tridiagonal vertical Poisson solve, the singular-mode gauge pin,
and the spectral gradients — entirely pointwise in wavenumber, no planar means —
so a y-slab decomposition must reproduce the single-rank result to round-off.

Run::

    mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_spectral_les_project_mpi.py
    mpirun -np 4 .venv-mpi/bin/python -m pytest tests/distributed/test_spectral_les_project_mpi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESLayout, make_grid, project)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

NY, NX, NZ = 12, 16, 6
DT = 0.1
CFG = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=2.0, Ly=1.5, Lz=1.0)


def _fields():
    """Deterministic global (u*, v*, w*) — identical on every rank.

    ``w`` is on faces (nz+1) with zero walls, ``u``/``v`` on centres (nz).
    """
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.standard_normal((NY, NX, NZ)))
    v = jnp.asarray(rng.standard_normal((NY, NX, NZ)))
    w = np.zeros((NY, NX, NZ + 1))
    w[..., 1:NZ] = rng.standard_normal((NY, NX, NZ - 1))   # zero at walls
    return u, v, jnp.asarray(w)


def _y_slab(f):
    nyl = NY // NPROC
    return f[RANK * nyl:(RANK + 1) * nyl]


def _gather_y(slab):
    chunks = COMM.allgather(np.asarray(slab))
    return jnp.asarray(np.concatenate(chunks, axis=0))


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_distributed_project_matches_serial():
    u, v, w = _fields()
    g_ser = make_grid(CFG)
    un_s, vn_s, wn_s = project(u, v, w, DT, g_ser)

    layout = SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                               comm=COMM)
    g_dist = make_grid(CFG, layout=layout)
    un_d, vn_d, wn_d = project(_y_slab(u), _y_slab(v), _y_slab(w), DT, g_dist)

    for name, dist, ser in (("u", un_d, un_s), ("v", vn_d, vn_s),
                            ("w", wn_d, wn_s)):
        np.testing.assert_allclose(
            np.asarray(_gather_y(dist)), np.asarray(ser),
            rtol=1e-9, atol=1e-9, err_msg=f"{name} mismatch")


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_distributed_project_divergence_free():
    """Projected velocity is discretely divergence-free (the projection's job)."""
    from legoesm.atmosphere.dynamics.les.spectral_les_plane import ddx, ddy, ddz_f2c
    u, v, w = _fields()
    layout = SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                               comm=COMM)
    g = make_grid(CFG, layout=layout)
    un, vn, wn = project(_y_slab(u), _y_slab(v), _y_slab(w), DT, g)
    div = ddx(un, g) + ddy(vn, g) + ddz_f2c(wn, g.dz)
    assert float(jnp.max(jnp.abs(div))) < 1e-9


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_distributed_project_grad_matches_serial():
    """Grad of a scalar of the projected field == serial (custom-VJP path)."""
    u, v, w = _fields()

    def loss(uu, gg, slab):
        un, vn, wn = project(uu, slab[0], slab[1], DT, gg)
        return jnp.sum(un ** 2) + jnp.sum(vn ** 2) + jnp.sum(wn ** 2)

    g_ser = make_grid(CFG)
    gs = jax.grad(loss)(u, g_ser, (v, w))

    layout = SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                               comm=COMM)
    g_dist = make_grid(CFG, layout=layout)
    gd = jax.grad(loss)(_y_slab(u), g_dist, (_y_slab(v), _y_slab(w)))

    np.testing.assert_allclose(np.asarray(_gather_y(gd)), np.asarray(gs),
                               rtol=1e-8, atol=1e-8)
