"""2D-horizontal Fourier operators for the doubly-periodic plane CRM.

Companion to :mod:`legoesm.atmosphere.dynamics.les.plane_operators` (the
finite-difference Arakawa-C operator set). This module provides the
pseudo-spectral counterpart: exact differential operators on the
horizontal axes via 2D FFT, with vertical structure handled by the
existing column kernel in :mod:`compressible_euler`.

Layout convention
-----------------
Physical-space horizontal layout matches the FD plane: ``(ny, nx)``
or ``(ny, nx, nlev)`` with ``y`` on axis 0 and ``x`` on axis 1.
Spectral-space arrays are the real-input FFT of the horizontal
plane: ``rfft2`` shrinks the ``x`` axis from ``nx`` to ``nx//2 + 1``
because the negative-frequency half is the complex conjugate of the
positive half. Shape::

    physical: (ny, nx [, nlev])
    spectral: (ny, nx_r [, nlev])  with nx_r = nx // 2 + 1   (complex)

Wavenumbers
-----------
``kx_2d`` shape ``(ny, nx_r)`` holds ``2π·m/Lx`` for ``m = 0..nx_r-1``.
``ky_2d`` shape ``(ny, nx_r)`` holds ``2π·n/Ly`` for ``n = 0..ny-1``
with the standard FFT layout (negative-y bin appears at index ``ny -
|n|`` for the second half).

Differential operators
----------------------
All exact in spectral space::

    grad_x_spec(phi_hat) = 1j * kx * phi_hat
    grad_y_spec(phi_hat) = 1j * ky * phi_hat
    divergence_spec(u_hat, v_hat) = 1j*(kx*u_hat + ky*v_hat)
    laplacian_spec(phi_hat) = -(kx² + ky²) * phi_hat
    biharmonic_spec(phi_hat) = (kx² + ky²)² * phi_hat

PG / divergence adjoint pair is exact: the sum identity
``sum(phi · div(u, v)) = -sum(u · grad_x(phi)) - sum(v · grad_y(phi))``
holds to round-off without any discretisation correction (Parseval's
theorem applied to the analytical identity).

Dealiasing
----------
``two_thirds_dealias_mask(grid)`` returns a real ``(ny, nx_r)`` mask
that zeros every spectral coefficient outside the inner two-thirds
of the wavenumber range (Orszag 1971, de Boer 1971). Standard cure
for the quadratic aliasing produced when nonlinear products are
formed in physical space and the result is transformed back: a
product of two fields band-limited to ``|k| <= K`` contains
wavenumbers up to ``|k| <= 2K``, and any energy above ``|k| > K``
folds back into resolved scales unless filtered.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SpectralPlaneAxis(NamedTuple):
    """Pre-computed wavenumber arrays for the spectral plane operators.

    Cache once per (grid, dtype) and reuse across the entire integration
    so every operator call multiplies against the same precomputed
    array (XLA folds it into the kernel; no per-step allocation).
    """
    kx: jax.Array       # (ny, nx_r) real
    ky: jax.Array       # (ny, nx_r) real
    k2: jax.Array       # (ny, nx_r) real, kx**2 + ky**2
    nx: int
    ny: int
    nx_r: int           # rfft last-axis size
    Lx: float
    Ly: float
    dealias_mask: jax.Array | None   # (ny, nx_r) real or None


def make_spectral_axis(grid, dealias: bool = True, dtype=jnp.float64) -> SpectralPlaneAxis:
    """Build the wavenumber arrays for a ``PlaneGrid``.

    Parameters
    ----------
    grid : PlaneGrid
        Provides ``nx, ny, dx, dy, Lx, Ly``.
    dealias : bool
        If True, attach the 2/3-rule dealias mask. Set False to test
        aliasing behaviour or to drop the mask cost when the caller
        knows nonlinear products are bandwidth-limited.
    dtype : jnp dtype
        Real dtype for the wavenumbers (the spectral arrays are
        complex with twice this width: float64 -> complex128).

    Returns
    -------
    SpectralPlaneAxis
    """
    nx = int(grid.nx)
    ny = int(grid.ny)
    nx_r = nx // 2 + 1
    Lx = float(grid.Lx)
    Ly = float(grid.Ly)
    # rfft frequencies along x: 0, 1, ..., nx//2
    kx_1d = jnp.asarray(
        jnp.fft.rfftfreq(nx, d=Lx / nx) * 2.0 * jnp.pi, dtype=dtype,
    )
    # full fft frequencies along y: 0, 1, ..., ny//2, -ny//2+1, ..., -1
    ky_1d = jnp.asarray(
        jnp.fft.fftfreq(ny, d=Ly / ny) * 2.0 * jnp.pi, dtype=dtype,
    )
    kx_2d = jnp.broadcast_to(kx_1d[None, :], (ny, nx_r))
    ky_2d = jnp.broadcast_to(ky_1d[:, None], (ny, nx_r))
    k2 = kx_2d * kx_2d + ky_2d * ky_2d
    mask = two_thirds_dealias_mask(nx, ny, dtype=dtype) if dealias else None
    return SpectralPlaneAxis(
        kx=kx_2d, ky=ky_2d, k2=k2,
        nx=nx, ny=ny, nx_r=nx_r,
        Lx=Lx, Ly=Ly,
        dealias_mask=mask,
    )


def two_thirds_dealias_mask(nx: int, ny: int, dtype=jnp.float64) -> jax.Array:
    """2/3-rule dealiasing mask for the ``(ny, nx_r)`` rfft layout.

    Zeros spectral coefficients above ``|kx_index| > nx/3`` and
    ``|ky_index| > ny/3``. Standard remedy for the aliasing produced
    by quadratic nonlinearities (Orszag 1971).
    """
    nx_r = nx // 2 + 1
    kx_idx = jnp.arange(nx_r)
    # Cutoff at floor(nx/3): keep modes with |kx_idx| <= floor(nx/3).
    kx_cut = nx // 3
    mask_x = (kx_idx <= kx_cut).astype(dtype)
    # ky bins under fftfreq: 0..ny//2, then -ny//2+1..-1. Wrap into
    # signed-index space then threshold.
    ky_idx = jnp.where(
        jnp.arange(ny) <= ny // 2,
        jnp.arange(ny),
        jnp.arange(ny) - ny,
    )
    ky_cut = ny // 3
    mask_y = (jnp.abs(ky_idx) <= ky_cut).astype(dtype)
    return mask_y[:, None] * mask_x[None, :]


# --------------------------------------------------------------------- #
# Transform helpers                                                     #
# --------------------------------------------------------------------- #


def to_spec(phi: jax.Array) -> jax.Array:
    """Real horizontal field → spectral (rfft over the last 2 horizontal
    axes for a vertical-last ``(ny, nx [, nlev])`` array).

    Operates on axes ``(0, 1)`` (= ``(y, x)``). Output has the same
    trailing-axis structure as the input but the ``x`` axis is
    shrunk to ``nx_r = nx // 2 + 1`` complex coefficients.
    """
    return jnp.fft.rfftn(phi, axes=(0, 1))


def to_phys(phi_hat: jax.Array, nx: int, ny: int) -> jax.Array:
    """Spectral field → physical via ``irfftn`` on axes ``(0, 1)``.

    The ``s=(ny, nx)`` argument pins the inverse-transform output
    shape so the operator is unambiguous when the input ``nx_r`` axis
    could correspond to multiple ``nx`` values.
    """
    return jnp.fft.irfftn(phi_hat, s=(ny, nx), axes=(0, 1))


# --------------------------------------------------------------------- #
# Exact spectral differential operators                                 #
# --------------------------------------------------------------------- #


def grad_x_spec(phi_hat: jax.Array, axis: SpectralPlaneAxis) -> jax.Array:
    """``∂φ/∂x`` in spectral space. Exact: multiplication by ``i·kx``.

    Broadcasts ``axis.kx`` of shape ``(ny, nx_r)`` against any
    trailing-axis structure in ``phi_hat``.
    """
    return 1j * _bcast(axis.kx, phi_hat) * phi_hat


def grad_y_spec(phi_hat: jax.Array, axis: SpectralPlaneAxis) -> jax.Array:
    """``∂φ/∂y`` in spectral space. Exact: multiplication by ``i·ky``."""
    return 1j * _bcast(axis.ky, phi_hat) * phi_hat


def divergence_spec(
    u_hat: jax.Array, v_hat: jax.Array, axis: SpectralPlaneAxis,
) -> jax.Array:
    """``∂u/∂x + ∂v/∂y`` in spectral space. Adjoint of the gradient.

    The discrete identity ``sum(phi · div(u, v)) = -sum(u · grad_x(phi))
    - sum(v · grad_y(phi))`` is exact (Parseval); the pseudo-spectral
    PG/divergence pair is therefore energy-consistent by construction.
    """
    return 1j * (
        _bcast(axis.kx, u_hat) * u_hat + _bcast(axis.ky, v_hat) * v_hat
    )


def laplacian_spec(
    phi_hat: jax.Array, axis: SpectralPlaneAxis,
) -> jax.Array:
    """``∇²φ`` in spectral space. Exact: multiplication by ``-k²``."""
    return -_bcast(axis.k2, phi_hat) * phi_hat


def biharmonic_spec(
    phi_hat: jax.Array, axis: SpectralPlaneAxis,
) -> jax.Array:
    """``∇⁴φ`` in spectral space. Exact: multiplication by ``k⁴``.

    Pseudo-spectral biharmonic hyperdiffusion is dissipative for any
    non-negative coefficient AND has the exact ``sin⁴(πk/N)``
    eigenvalue spectrum (the discrete approximation in the FD path
    deviates from this at high ``k``); the spectral path is therefore
    immune to the discrete-stencil null-space modes that plague
    composed-Laplacian implementations.
    """
    k4 = _bcast(axis.k2, phi_hat) ** 2
    return k4 * phi_hat


def apply_dealias(phi_hat: jax.Array, axis: SpectralPlaneAxis) -> jax.Array:
    """Apply the 2/3 dealias mask. No-op if ``axis.dealias_mask is None``.

    Used after every nonlinear product (transform-back, multiply,
    transform-forward) to suppress the aliased high-``k`` modes.
    """
    if axis.dealias_mask is None:
        return phi_hat
    return phi_hat * _bcast(axis.dealias_mask, phi_hat)


# --------------------------------------------------------------------- #
# Internal: broadcast helper                                            #
# --------------------------------------------------------------------- #


def _bcast(arr2d: jax.Array, like: jax.Array) -> jax.Array:
    """Broadcast a ``(ny, nx_r)`` array to match the trailing-axis
    structure of ``like``.
    """
    if like.ndim == arr2d.ndim:
        return arr2d
    extra = like.ndim - arr2d.ndim
    return arr2d.reshape(arr2d.shape + (1,) * extra)
