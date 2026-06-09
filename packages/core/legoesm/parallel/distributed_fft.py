"""Transpose-based distributed 2-D real FFT for the plane spectral LES.

The serial spectral LES (``spectral_les_plane.py``) does every horizontal
operator — the pressure Poisson solve, the sharp spectral filter and the LASD
dynamic test filter — through ``jnp.fft.rfft2`` over a global ``(ny, nx, nz)``
field. That global FFT is the single reason the spectral LES is confined to one
rank: an MPI run needs a *distributed* 2-D FFT.

This module provides one under a **slab decomposition along y** (axis 0): rank
``r`` owns ``(ny_local, nx, nz)`` with ``ny = P * ny_local`` (``P`` ranks,
``n_ranks_x == 1``). A 2-D FFT is computed transpose-style:

    physical(y-slab)  --rfft_x-->  (ny_local, nkx, nz)
                      --transpose-->  (ny, nkx_local, nz)   [y now whole]
                      --fft_y-->     spectral(kx-slab)

and the inverse reverses it. The transpose is a single ``alltoall``.

Decomposition choice. Slab (1-D) is the natural and minimal decomposition for a
2-D FFT: exactly one transpose, one ``alltoall``. A 2-D pencil decomposition
would need two transposes; it is only worth it past ``P > min(ny, nkx)``, well
beyond a single socket. The plane LES MPI layout therefore uses ``n_ranks_x = 1``
for spectral runs; the compressible CRM/LES keeps its 2-D pencil ``step_halo``.

AD. ``mpi4jax.alltoall`` is a permutation (orthogonal data movement); its adjoint
is the same all-to-all applied to the cotangent. We wrap it in a ``custom_vjp``
so the spectral LES stays end-to-end differentiable — the project mandates that
only AD-safe collectives appear in differentiable paths, and a raw ``alltoall``
is not one. Complex arrays are carried as two real ``alltoall`` calls (real and
imaginary parts) so the wrapper never relies on complex collective support.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# AD-safe all-to-all (the transpose primitive)                                 #
# --------------------------------------------------------------------------- #
def _raw_alltoall(x, comm):
    """``mpi4jax.alltoall`` on a real array whose leading axis == n_ranks.

    Block ``i`` of the result on this rank is the block this rank's index
    selected out of rank ``i``'s input — the standard MPI all-to-all.
    """
    import mpi4jax

    return mpi4jax.alltoall(x, comm=comm)


def ad_alltoall(x, comm):
    """All-to-all on a real array ``(P, *rest)`` (``P == n_ranks``), AD-safe.

    The all-to-all is an orthogonal permutation of data across ranks, so its
    vector-Jacobian product is the *same* all-to-all applied to the incoming
    cotangent. That makes it usable inside ``jax.grad`` / ``eqx.filter_*``,
    unlike a bare ``mpi4jax.alltoall`` (allgather/alltoall are otherwise
    diagnostic-only in this codebase). ``comm`` is a non-differentiable static
    argument (``nondiff_argnums``); it is not a JAX type.
    """
    return _raw_alltoall(x, comm)


def _ad_alltoall_fwd(x, comm):
    # custom_vjp passes args in original order to fwd; no residuals needed (the
    # adjoint of a permutation does not depend on the primal value).
    return _raw_alltoall(x, comm), None


def _ad_alltoall_bwd(comm, _res, g):
    # nondiff args first, then residuals, then cotangent. Adjoint of an
    # all-to-all is the all-to-all of the cotangent (P×P equal blocks ⇒ the
    # operator is its own transpose). Return grad for the single diff arg x.
    return (_raw_alltoall(g, comm),)


ad_alltoall = jax.custom_vjp(ad_alltoall, nondiff_argnums=(1,))
ad_alltoall.defvjp(_ad_alltoall_fwd, _ad_alltoall_bwd)


def _alltoall_complex(z, comm):
    """All-to-all on a complex array ``(P, *rest)`` via two real transposes."""
    re = ad_alltoall(jnp.real(z), comm)
    im = ad_alltoall(jnp.imag(z), comm)
    return jax.lax.complex(re, im)


# --------------------------------------------------------------------------- #
# Padding helpers (the reduced kx axis nx//2+1 need not divide evenly by P)    #
# --------------------------------------------------------------------------- #
def kx_local_size(nx, n_ranks):
    """Per-rank length of the (zero-padded) reduced kx axis."""
    nkx = nx // 2 + 1
    return -(-nkx // n_ranks)  # ceil division


# --------------------------------------------------------------------------- #
# Forward / inverse distributed 2-D real FFT                                   #
# --------------------------------------------------------------------------- #
def distributed_rfft2(u_local, *, ny_global, nx, n_ranks, comm):
    """Distributed ``rfft2`` over (y, x) of a y-slab ``(ny_local, nx, nz)``.

    Parameters
    ----------
    u_local : real array ``(ny_local, nx, nz)`` — this rank's y-slab.
    ny_global : int — total ny (must equal ``n_ranks * ny_local``).
    nx : int — global (and local, x is undecomposed) x extent.
    n_ranks, comm : MPI mesh size and communicator (slab along y).

    Returns
    -------
    complex array ``(ny_global, nkx_local, nz)`` — full ky, this rank's kx slab.
    ``nkx_local = ceil((nx//2+1) / n_ranks)``; the padded high-kx columns on the
    last rank are exact zeros and carry no energy.
    """
    P = n_ranks
    ny_local, nx_in, nz = u_local.shape
    if nx_in != nx:
        raise ValueError(f"u_local x-extent {nx_in} != nx {nx}")
    if P * ny_local != ny_global:
        raise ValueError(
            f"n_ranks*ny_local = {P*ny_local} != ny_global {ny_global}")

    nkx = nx // 2 + 1
    nkx_loc = kx_local_size(nx, P)
    nkx_pad = P * nkx_loc

    # 1) local real FFT along x.
    fx = jnp.fft.rfft(u_local, axis=1)                    # (ny_local, nkx, nz)
    if nkx_pad != nkx:
        fx = jnp.pad(fx, ((0, 0), (0, nkx_pad - nkx), (0, 0)))

    # 2) transpose y-slab → kx-slab via one all-to-all.
    #    (ny_local, P*nkx_loc, nz) → (P, ny_local, nkx_loc, nz)
    blocks = fx.reshape(ny_local, P, nkx_loc, nz).transpose(1, 0, 2, 3)
    recv = _alltoall_complex(blocks, comm)                # (P, ny_local, nkx_loc, nz)
    #    block r holds rank r's y-slab for this rank's kx chunk ⇒ stack along y.
    fy_in = recv.reshape(ny_global, nkx_loc, nz)

    # 3) local complex FFT along y (now whole).
    return jnp.fft.fft(fy_in, axis=0)                     # (ny, nkx_loc, nz)


def distributed_irfft2(fh_local, *, ny_global, nx, n_ranks, comm):
    """Inverse of :func:`distributed_rfft2`.

    ``fh_local`` is ``(ny_global, nkx_local, nz)`` (full ky, this rank's kx
    slab); returns the real y-slab ``(ny_local, nx, nz)``.
    """
    P = n_ranks
    ny_in, nkx_loc, nz = fh_local.shape
    if ny_in != ny_global:
        raise ValueError(f"fh_local y-extent {ny_in} != ny_global {ny_global}")
    ny_local = ny_global // P
    nkx = nx // 2 + 1

    # 1) inverse FFT along y.
    gy = jnp.fft.ifft(fh_local, axis=0)                   # (ny, nkx_loc, nz)

    # 2) transpose kx-slab → y-slab (inverse all-to-all).
    #    split y into P blocks → (P, ny_local, nkx_loc, nz)
    blocks = gy.reshape(P, ny_local, nkx_loc, nz)
    recv = _alltoall_complex(blocks, comm)                # (P, ny_local, nkx_loc, nz)
    #    block r is kx-chunk r for this rank's y-slab ⇒ concat along kx.
    fx = recv.transpose(1, 0, 2, 3).reshape(ny_local, P * nkx_loc, nz)
    fx = fx[:, :nkx, :]                                   # drop kx padding

    # 3) local inverse real FFT along x.
    return jnp.fft.irfft(fx, n=nx, axis=1)                # (ny_local, nx, nz)


def local_kx_slice(nx, n_ranks, rank):
    """``(start, stop)`` into the padded reduced-kx axis owned by ``rank``.

    Lets callers slice the precomputed ``kx`` / ``k2`` wavenumber arrays to the
    columns this rank holds in spectral space, consistent with the padding in
    :func:`distributed_rfft2` (columns ``>= nx//2+1`` are zero-padding).
    """
    nkx_loc = kx_local_size(nx, n_ranks)
    return rank * nkx_loc, (rank + 1) * nkx_loc
