"""Serial-equivalence + AD tests for the slab distributed 2-D real FFT.

Run with::

    mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_distributed_fft_mpi.py
    mpirun -np 4 .venv-mpi/bin/python -m pytest tests/distributed/test_distributed_fft_mpi.py

The distributed FFT (``parallel/distributed_fft.py``) must reproduce the serial
``jnp.fft.rfft2`` path the spectral LES uses, to round-off, and pass gradients
(the spectral LES is differentiable). All comparisons are against the single-rank
``jnp.fft`` reference computed redundantly on every rank from a shared seed.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.parallel.distributed_fft import (  # noqa: E402
    distributed_rfft2,
    distributed_irfft2,
    kx_local_size,
    local_kx_slice,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

# Global problem; ny must divide by NPROC. 12 = 1·2·3·4·6·12 covers np 1/2/3/4/6.
NY, NX, NZ = 12, 16, 5


def _global_field():
    """Deterministic global real field, identical on every rank."""
    rng = np.random.default_rng(20260609)
    return jnp.asarray(rng.standard_normal((NY, NX, NZ)))


def _y_slab(f_global):
    ny_local = NY // NPROC
    return f_global[RANK * ny_local:(RANK + 1) * ny_local]


def _allgather_y(slab):
    """Assemble a y-slab field back to the global (ny, ...) array on all ranks."""
    local = np.asarray(slab)
    chunks = COMM.allgather(local)
    return jnp.asarray(np.concatenate(chunks, axis=0))


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_roundtrip_identity():
    """irfft2 ∘ rfft2 == identity on the distributed slab."""
    f = _global_field()
    slab = _y_slab(f)
    fh = distributed_rfft2(slab, ny_global=NY, nx=NX, n_ranks=NPROC, comm=COMM)
    back = distributed_irfft2(fh, ny_global=NY, nx=NX, n_ranks=NPROC, comm=COMM)
    np.testing.assert_allclose(np.asarray(back), np.asarray(slab),
                               rtol=0, atol=1e-12)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_spectral_derivative_matches_serial():
    """End-to-end ∂/∂x via the distributed FFT == serial rfft2 derivative.

    Exercises forward FFT, a kx-dependent spectral multiply on the local kx
    slab, and the inverse FFT — the exact chain the LES pressure/filter use.
    """
    f = _global_field()

    # Serial reference (full rfft2 path), computed on every rank.
    kx_full = 2.0 * jnp.pi * jnp.fft.rfftfreq(NX, d=1.0 / NX)   # (nx//2+1,)
    fh_serial = jnp.fft.rfft2(f, axes=(0, 1))
    ddx_serial = jnp.fft.irfft2(fh_serial * (1j * kx_full[None, :, None]),
                                axes=(0, 1), s=(NY, NX))

    # Distributed: padded kx, sliced to this rank's spectral columns.
    nkx = NX // 2 + 1
    nkx_pad = NPROC * kx_local_size(NX, NPROC)
    kx_pad = jnp.pad(kx_full, (0, nkx_pad - nkx))
    lo, hi = local_kx_slice(NX, NPROC, RANK)
    kx_loc = kx_pad[lo:hi]                                      # (nkx_local,)

    slab = _y_slab(f)
    fh = distributed_rfft2(slab, ny_global=NY, nx=NX, n_ranks=NPROC, comm=COMM)
    ddx_loc = distributed_irfft2(fh * (1j * kx_loc[None, :, None]),
                                 ny_global=NY, nx=NX, n_ranks=NPROC, comm=COMM)

    ddx_dist = _allgather_y(ddx_loc)
    np.testing.assert_allclose(np.asarray(ddx_dist), np.asarray(ddx_serial),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_gradient_matches_serial():
    """Grad through the distributed round-trip == grad through the serial one.

    Confirms the custom-VJP all-to-all carries cotangents correctly, so the
    distributed FFT is drop-in for the differentiable spectral LES.
    """
    f = _global_field()

    def loss_serial(g):
        fh = jnp.fft.rfft2(g, axes=(0, 1))
        back = jnp.fft.irfft2(fh * 2.0, axes=(0, 1), s=(NY, NX))
        return jnp.sum(back ** 2)

    def loss_dist(slab):
        fh = distributed_rfft2(slab, ny_global=NY, nx=NX, n_ranks=NPROC, comm=COMM)
        back = distributed_irfft2(fh * 2.0, ny_global=NY, nx=NX,
                                  n_ranks=NPROC, comm=COMM)
        return jnp.sum(back ** 2)

    g_serial_full = jax.grad(loss_serial)(f)
    g_serial = _y_slab(g_serial_full)
    g_dist = jax.grad(loss_dist)(_y_slab(f))

    np.testing.assert_allclose(np.asarray(g_dist), np.asarray(g_serial),
                               rtol=1e-9, atol=1e-9)
