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

Decomposition choice. Slab (1-D) is the minimal decomposition for a 2-D FFT:
exactly one transpose, one ``alltoall``, but it caps the rank count at ``ny``
(and idles ranks in spectral space past ``nx//2+1``). The x-y PENCIL functions
at the end of this module lift that cap at two all-to-alls per transform (one
on a row sub-communicator, one on a column sub-communicator).

AD. ``mpi4jax.alltoall`` is a permutation (orthogonal data movement); its adjoint
is the same all-to-all applied to the cotangent. We wrap it in a ``custom_vjp``
so the spectral LES stays end-to-end differentiable — the project mandates that
only AD-safe collectives appear in differentiable paths, and a raw ``alltoall``
is not one. Complex arrays are carried as ONE packed real ``alltoall`` (real and
imaginary parts stacked inside each per-rank block) so the wrapper never relies
on complex collective support and pays a single collective latency per transpose.
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
    from legoesm.parallel.reductions import require_mpi_stack

    # Checked accessor (not a bare ``import mpi4jax``): runs the GPU-transport
    # preflight so this device-array all-to-all fails closed on a GPU-direct
    # misconfiguration instead of segfaulting.
    mpi4jax, _ = require_mpi_stack()
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
    """All-to-all on a complex array ``(P, *rest)`` via ONE packed real transpose.

    Real and imaginary parts are stacked on a new axis 1 — INSIDE each
    per-rank block, so the leading block axis stays ``P`` and each rank's
    block ``(2, *rest)`` travels intact through the collective. One
    all-to-all of 2x the bytes replaces two collectives: for a
    latency-bound transpose (the distributed-FFT scaling regime) halving
    the collective count is a straight win, and the AD story is unchanged
    (``ad_alltoall`` wraps any real ``(P, ...)`` array)."""
    packed = jnp.stack([jnp.real(z), jnp.imag(z)], axis=1)  # (P, 2, *rest)
    out = ad_alltoall(packed, comm)
    return jax.lax.complex(out[:, 0], out[:, 1])


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


# --------------------------------------------------------------------------- #
# 3/2-rule zero-padding / truncation between coarse and fine grids            #
# --------------------------------------------------------------------------- #
# The 3/2 de-aliasing pads a coarse (ny,nx) field to the fine (3ny/2, 3nx/2)
# grid (and truncates back). The 2-D spectral zero-pad is SEPARABLE, so it is
# done as two independent 1-D zero-pads: the x-direction is LOCAL (x is the
# undecomposed slab axis) and the y-direction uses one all-to-all transpose
# (y is the decomposed axis). Two 1-D inverse FFTs (norm 1/nxf then 1/nyf)
# compose to exactly the 2-D irfft2 normalisation 1/(nyf·nxf) of the serial path.
def _y_transpose_fft(real_y_slab, ny_global, n_ranks, comm):
    """Real ``(ny_local, B, nz)`` (y decomposed, B undecomposed) → complex
    ``(ny_global, B_loc, nz)`` spectrum (full ky, B-slab) via one all-to-all."""
    P = n_ranks
    ny_local, B, nz = real_y_slab.shape
    B_loc = -(-B // P)
    B_pad = P * B_loc
    f = real_y_slab
    if B_pad != B:
        f = jnp.pad(f, ((0, 0), (0, B_pad - B), (0, 0)))
    blk = f.reshape(ny_local, P, B_loc, nz).transpose(1, 0, 2, 3)
    recv = ad_alltoall(blk, comm)                     # (P, ny_local, B_loc, nz)
    yfull = recv.reshape(ny_global, B_loc, nz)        # full y, B-slab (real)
    return jnp.fft.fft(yfull, axis=0)                 # complex spectrum


def _y_itranspose(spec_full_ky, ny_out_global, n_ranks, comm, B):
    """Inverse of :func:`_y_transpose_fft` at an arbitrary output y-length:
    complex ``(ny_out_global, B_loc, nz)`` spectrum → real ``(ny_out_local, B,
    nz)`` (y decomposed). ``ny_out_global`` must be divisible by ``n_ranks``."""
    P = n_ranks
    ny_out_local = ny_out_global // P
    B_loc, nz = spec_full_ky.shape[1], spec_full_ky.shape[2]
    ycoarse = jnp.fft.ifft(spec_full_ky, axis=0).real    # (ny_out_global, B_loc, nz)
    blk = ycoarse.reshape(P, ny_out_local, B_loc, nz)
    recv = ad_alltoall(blk, comm)                        # (P, ny_out_local, B_loc, nz)
    out = recv.transpose(1, 0, 2, 3).reshape(ny_out_local, P * B_loc, nz)
    return out[:, :B, :]


def _pad_ky(spec, ny, nyf):
    """Zero-pad the full-ky axis (axis 0): coarse ``ny`` → fine ``nyf``,
    dropping the y-Nyquist row (matches the serial 3/2 pad)."""
    nyh = ny // 2
    B_loc, nz = spec.shape[1], spec.shape[2]
    out = jnp.zeros((nyf, B_loc, nz), spec.dtype)
    out = out.at[:nyh].set(spec[:nyh])                       # +ky
    out = out.at[nyf - nyh + 1:].set(spec[nyh + 1:ny])       # −ky
    return out


def _truncate_ky(spec, nyf, ny):
    """Inverse of :func:`_pad_ky`: fine ``nyf`` → coarse ``ny`` (keep low ky)."""
    nyh = ny // 2
    B_loc, nz = spec.shape[1], spec.shape[2]
    out = jnp.zeros((ny, B_loc, nz), spec.dtype)
    out = out.at[:nyh].set(spec[:nyh])
    out = out.at[ny - nyh + 1:].set(spec[nyf - nyh + 1:])
    return out


def distributed_pad_to_fine(f_local, *, ny_global, nx, n_ranks, comm):
    """Distributed 3/2 zero-pad: coarse y-slab ``(ny_local, nx, nz)`` → fine
    y-slab ``(3ny_local/2, 3nx/2, nz)``. Requires ``ny_local`` even (so
    ``nyf_local = 3ny_local/2`` is an integer)."""
    P = n_ranks
    ny_local, nx_in, nz = f_local.shape
    if nx_in != nx:
        raise ValueError(f"x-extent {nx_in} != nx {nx}")
    if P * ny_local != ny_global:
        raise ValueError(f"n_ranks*ny_local {P*ny_local} != ny_global {ny_global}")
    if ny_local % 2:
        raise ValueError(
            f"3/2-rule distributed needs ny_local even, got {ny_local} "
            f"(ny_global={ny_global}, n_ranks={P}); 3ny_local/2 must be integer")
    if nx % 2 or ny_global % 2:
        raise ValueError(
            f"3/2-rule needs even nx, ny_global (Nyquist-drop indexing); "
            f"got nx={nx}, ny_global={ny_global}")
    nyf, nxf = 3 * ny_global // 2, 3 * nx // 2
    nxr = nx // 2
    # 1) local x zero-pad (x undecomposed): rfft_x → place low kx → irfft_x.
    fxh = jnp.fft.rfft(f_local, axis=1)                    # (ny_local, nx//2+1, nz)
    padx = jnp.zeros((ny_local, nxf // 2 + 1, nz), dtype=fxh.dtype)
    padx = padx.at[:, :nxr, :].set(fxh[:, :nxr, :])        # drop x-Nyquist
    fx = jnp.fft.irfft(padx, n=nxf, axis=1)                # (ny_local, nxf, nz) real
    # 2) distributed y zero-pad: transpose → fft_y → pad ky → ifft_y → transpose.
    spec = _y_transpose_fft(fx, ny_global, P, comm)        # (ny_global, B_loc, nz)
    spec_f = _pad_ky(spec, ny_global, nyf)                 # (nyf, B_loc, nz)
    return _y_itranspose(spec_f, nyf, P, comm, nxf)        # (nyf_local, nxf, nz)


def distributed_truncate_from_fine(f_fine, *, ny_global, nx, n_ranks, comm):
    """Distributed 3/2 truncation: fine y-slab ``(3ny_local/2, 3nx/2, nz)`` →
    coarse y-slab ``(ny_local, nx, nz)`` with the oracle's 9/4 scaling. Inverse
    grid of :func:`distributed_pad_to_fine`."""
    P = n_ranks
    nyf_local, nxf, nz = f_fine.shape
    nyf = 3 * ny_global // 2
    if P * nyf_local != nyf:
        raise ValueError(f"n_ranks*nyf_local {P*nyf_local} != nyf {nyf}")
    nxr = nx // 2
    # 1) local x truncate: rfft_x → keep low kx → irfft_x(n=nx).
    fxh = jnp.fft.rfft(f_fine, axis=1)                     # (nyf_local, nxf//2+1, nz)
    outx = jnp.zeros((nyf_local, nx // 2 + 1, nz), dtype=fxh.dtype)
    outx = outx.at[:, :nxr, :].set(fxh[:, :nxr, :])
    fx = jnp.fft.irfft(outx, n=nx, axis=1)                 # (nyf_local, nx, nz) real
    # 2) distributed y truncate.
    spec = _y_transpose_fft(fx, nyf, P, comm)             # (nyf, B_loc, nz)
    spec_c = _truncate_ky(spec, nyf, ny_global)           # (ny_global, B_loc, nz)
    coarse = _y_itranspose(spec_c, ny_global, P, comm, nx)  # (ny_local, nx, nz)
    return (9.0 / 4.0) * coarse


# --------------------------------------------------------------------------- #
# x-y PENCIL decomposition (Py x Px rank grid)                                 #
# --------------------------------------------------------------------------- #
# Rank (ry, rx) = divmod(rank, Px) owns the physical block (ny/Py, nx/Px, nzf).
# Every pencil transform is the slab transform above run on the COLUMN
# communicator (size Py, the ranks sharing rx) over data made x-whole by one
# ROW all-to-all (size Px) that splits z instead (zero-padded to Px chunks; z
# is never transformed). Spectral layout per rank: (ny, kx chunk ry, z chunk
# rx) — every pointwise spectral multiply is z-independent, and the pressure
# projection gathers z with one more row transpose (``pencil_spec_to_zwhole``).
# The z-padding makes the padded length ambiguous (nz-1, nz and nz+1 fields
# coexist), so every inverse takes the caller's static ``nz_out``.
# Batching note: the z axis is a free axis of every row transpose, so several
# fields can be concatenated along z and moved through ONE transform pair.
def _alltoall_any(blocks, comm):
    if jnp.iscomplexobj(blocks):
        return _alltoall_complex(blocks, comm)
    return ad_alltoall(blocks, comm)


def chunk_slice(n, n_parts, idx):
    """``(start, stop)`` of chunk ``idx`` when length ``n`` is zero-padded to
    ``n_parts * ceil(n / n_parts)`` and split into equal chunks."""
    loc = -(-n // n_parts)
    return idx * loc, (idx + 1) * loc


def split_concat_transpose(x, split_axis, concat_axis, n_ranks, comm):
    """All-to-all transpose over ``comm`` (size ``n_ranks``), real or complex.

    ``split_axis`` is zero-padded to ``n_ranks * ceil(n / n_ranks)`` and cut
    into equal chunks; chunk ``j`` goes to rank ``j``. The blocks received are
    concatenated along ``concat_axis`` in source-rank order, so that axis grows
    by ``n_ranks`` (every rank holds the same local shape). AD via
    :func:`ad_alltoall` (a permutation; its adjoint is the same collective)."""
    P = n_ranks
    n = x.shape[split_axis]
    loc = -(-n // P)
    if P * loc != n:
        pad = [(0, 0)] * x.ndim
        pad[split_axis] = (0, P * loc - n)
        x = jnp.pad(x, pad)
    shp = list(x.shape)
    shp[split_axis:split_axis + 1] = [P, loc]
    blocks = jnp.moveaxis(x.reshape(shp), split_axis, 0)   # (P, ..., loc, ...)
    recv = _alltoall_any(blocks, comm)                     # block i from rank i
    out = jnp.moveaxis(recv, 0, concat_axis)               # (..., P, c, ...)
    oshp = list(out.shape)
    oshp[concat_axis:concat_axis + 2] = [oshp[concat_axis] * oshp[concat_axis + 1]]
    return out.reshape(oshp)


def _check_pencil(shape, ny_global, nx, n_ranks_y, n_ranks_x):
    nyl, nxl = shape[0], shape[1]
    if nyl * n_ranks_y != ny_global or nxl * n_ranks_x != nx:
        raise ValueError(
            f"pencil block {nyl}x{nxl} x mesh {n_ranks_y}x{n_ranks_x} != global "
            f"{ny_global}x{nx}")


def _trim_z(x, nz_out, n_ranks_x):
    if -(-nz_out // n_ranks_x) * n_ranks_x != x.shape[-1]:
        raise ValueError(
            f"nz_out={nz_out} inconsistent with padded z {x.shape[-1]} at Px={n_ranks_x}")
    return x[..., :nz_out]


def pencil_rfft2(u_local, *, ny_global, nx, n_ranks_y, n_ranks_x, comm_row,
                 comm_col):
    """Pencil ``rfft2`` of ``(ny/Py, nx/Px, nzf)`` → ``(ny, nkx_loc, nzl)``:
    full ky, kx chunk ``ry`` of ``Py`` (zero-padded, as the slab), z chunk ``rx``
    of ``Px`` (``nzl = ceil(nzf/Px)``, zero-padded)."""
    _check_pencil(u_local.shape, ny_global, nx, n_ranks_y, n_ranks_x)
    a = split_concat_transpose(u_local, 2, 1, n_ranks_x, comm_row)  # (nyl, nx, nzl)
    return distributed_rfft2(a, ny_global=ny_global, nx=nx, n_ranks=n_ranks_y,
                             comm=comm_col)


def pencil_irfft2(fh_local, *, nz_out, ny_global, nx, n_ranks_y, n_ranks_x,
                  comm_row, comm_col):
    """Inverse of :func:`pencil_rfft2`; returns ``(ny/Py, nx/Px, nz_out)``."""
    a = distributed_irfft2(fh_local, ny_global=ny_global, nx=nx,
                           n_ranks=n_ranks_y, comm=comm_col)       # (nyl, nx, nzl)
    b = split_concat_transpose(a, 1, 2, n_ranks_x, comm_row)        # (nyl, nxl, Px*nzl)
    return _trim_z(b, nz_out, n_ranks_x)


def pencil_pad_to_fine(f_local, *, ny_global, nx, n_ranks_y, n_ranks_x,
                       comm_row, comm_col):
    """Pencil 3/2 zero-pad: ``(ny/Py, nx/Px, nz)`` → ``(3ny/2Py, 3nx/2Px, nz)``.
    Needs ``ny/Py`` and ``nx/Px`` even."""
    _check_pencil(f_local.shape, ny_global, nx, n_ranks_y, n_ranks_x)
    if f_local.shape[1] % 2:
        raise ValueError(
            f"3/2-rule pencil needs nx/Px even, got {f_local.shape[1]}")
    nzf = f_local.shape[2]
    a = split_concat_transpose(f_local, 2, 1, n_ranks_x, comm_row)  # (nyl, nx, nzl)
    fine = distributed_pad_to_fine(a, ny_global=ny_global, nx=nx,
                                   n_ranks=n_ranks_y, comm=comm_col)  # (nyfl, nxf, nzl)
    b = split_concat_transpose(fine, 1, 2, n_ranks_x, comm_row)
    return _trim_z(b, nzf, n_ranks_x)


def pencil_truncate_from_fine(f_fine, *, ny_global, nx, n_ranks_y, n_ranks_x,
                              comm_row, comm_col):
    """Inverse grid of :func:`pencil_pad_to_fine` (with the 9/4 scaling)."""
    nzf = f_fine.shape[2]
    a = split_concat_transpose(f_fine, 2, 1, n_ranks_x, comm_row)   # (nyfl, nxf, nzl)
    c = distributed_truncate_from_fine(a, ny_global=ny_global, nx=nx,
                                       n_ranks=n_ranks_y, comm=comm_col)
    b = split_concat_transpose(c, 1, 2, n_ranks_x, comm_row)        # (nyl, nxl, Px*nzl)
    return _trim_z(b, nzf, n_ranks_x)


def pencil_spec_to_zwhole(fh_local, *, nz_out, n_ranks_x, comm_row):
    """Spectral ``(ny, nkx_loc, nzl)`` → ``(nky_loc, nkx_loc, nz_out)``: ky chunk
    ``rx`` of ``Px`` (zero-padded rows), whole z (padding levels trimmed)."""
    b = split_concat_transpose(fh_local, 0, 2, n_ranks_x, comm_row)
    return _trim_z(b, nz_out, n_ranks_x)


def pencil_zwhole_to_spec(fz_local, *, ny_global, n_ranks_x, comm_row):
    """Inverse of :func:`pencil_spec_to_zwhole` (z re-padded with zeros, padded
    ky rows dropped)."""
    b = split_concat_transpose(fz_local, 2, 0, n_ranks_x, comm_row)
    return b[:ny_global]


def layout_is_pencil(layout):
    return getattr(layout, "n_ranks_x", 1) > 1


def layout_rfft2(f, layout):
    """Slab or pencil forward transform selected by ``layout``. The slab call
    is exactly the pre-pencil one (bit-identical)."""
    if not layout_is_pencil(layout):
        return distributed_rfft2(f, ny_global=layout.ny_global, nx=layout.nx,
                                 n_ranks=layout.n_ranks, comm=layout.comm)
    return pencil_rfft2(f, ny_global=layout.ny_global, nx=layout.nx,
                        n_ranks_y=layout.n_ranks_y, n_ranks_x=layout.n_ranks_x,
                        comm_row=layout.comm_row, comm_col=layout.comm_col)


def layout_irfft2(fh, layout, nz_out):
    """Inverse of :func:`layout_rfft2` (``nz_out`` used only by the pencil)."""
    if not layout_is_pencil(layout):
        return distributed_irfft2(fh, ny_global=layout.ny_global, nx=layout.nx,
                                  n_ranks=layout.n_ranks, comm=layout.comm)
    if nz_out is None:
        raise ValueError("pencil inverse FFT needs the field's nz_out")
    return pencil_irfft2(fh, nz_out=nz_out, ny_global=layout.ny_global,
                         nx=layout.nx, n_ranks_y=layout.n_ranks_y,
                         n_ranks_x=layout.n_ranks_x, comm_row=layout.comm_row,
                         comm_col=layout.comm_col)
