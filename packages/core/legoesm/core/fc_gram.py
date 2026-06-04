"""FC-Gram (Fourier Continuation) spectral derivative core.

Grid-agnostic 1D mathematics for extending non-periodic data to periodic
via Gram polynomial matching, then computing spectral derivatives via FFT.

This module has ZERO grid imports. It is reusable for any mesh topology.

References
----------
- Lyon & Bruno (2010): High-order unconditionally stable FC-Gram methods
- Bruno & Lyon (2010): High-order FC-based PDE solvers

Key ideas
---------
1. Match `d` boundary values on each side to a Gram polynomial basis.
2. Extend the data by `C` continuation points that smoothly connect
   the two ends, making the extended signal periodic.
3. Apply FFT-based spectral differentiation on the periodic extension.
4. Restrict back to the original N points.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class FCGramMatrices(NamedTuple):
    """Precomputed matrices for FC-Gram continuation.

    Attributes
    ----------
    self_matrix : jax.Array, shape (C, d)
        Continuation values from same-side boundary data.
    boundary_matrix : jax.Array, shape (C, 2*d)
        Continuation values from both sides: [left_d | right_d].
    d : int
        Number of boundary points matched on each side.
    C : int
        Number of continuation points.
    """
    self_matrix: jax.Array
    boundary_matrix: jax.Array
    d: int
    C: int


def build_fc_gram_matrices(
    d: int = 2,
    C: int = 4,
    degree: int = 5,
    dtype: np.dtype = np.float32,
) -> FCGramMatrices:
    """Build FC-Gram continuation matrices.

    Constructs the least-squares Gram polynomial matching matrices
    that map `d` boundary values on each side to `C` continuation
    points creating a smooth periodic extension.

    Parameters
    ----------
    d : int
        Number of boundary points matched on each side.
    C : int
        Number of continuation points appended.
    degree : int
        Degree of Gram polynomial basis (should be >= 2*d - 1).
    dtype : numpy dtype
        Output dtype (float32 or float64).

    Returns
    -------
    FCGramMatrices
    """
    # Geometry: continuation bridges from f[-1] back to f[0].
    #   ..., f[-d], ..., f[-1] | cont[0], ..., cont[C-1] | f[0], ..., f[d-1], ...
    #
    # Strategy: cubic Hermite interpolation matching values AND slopes
    # at both boundaries.  With d=2:
    #   bnd_both = [f[-2], f[-1], f[0], f[1]]
    #   Right: value = f[-1],  slope = f[-1] - f[-2]
    #   Left:  value = f[0],   slope = f[1] - f[0]

    # Gap length in grid units (from position of f[-1] to position of f[0])
    L = float(C + 1)

    # Hermite evaluation at continuation points (normalized to [0,1])
    t = (np.arange(C, dtype=np.float64) + 1.0) / L  # in (0, 1)

    # Hermite basis polynomials
    h00 = 2*t**3 - 3*t**2 + 1   # value=1@0, 0@1
    h10 = t**3 - 2*t**2 + t     # slope=1@0 (per unit interval)
    h01 = -2*t**3 + 3*t**2      # value=0@0, 1@1
    h11 = t**3 - t**2            # slope=0@0, 1@1

    # boundary_matrix: maps bnd_both = [f[-d:], f[:d]] → C continuation
    # For d=2: bnd_both = [f[-2], f[-1], f[0], f[1]]
    # p0=f[-1], dp0=(f[-1]-f[-2]) in grid units, scale by L for unit interval
    # p1=f[0],  dp1=(f[1]-f[0])   in grid units, scale by L
    boundary_matrix_np = np.zeros((C, 2 * d), dtype=np.float64)
    boundary_matrix_np[:, d-2] = -h10 * L             # coeff of f[-2]
    boundary_matrix_np[:, d-1] = h00 + h10 * L        # coeff of f[-1]
    boundary_matrix_np[:, d]   = h01 - h11 * L        # coeff of f[0]
    boundary_matrix_np[:, d+1] = h11 * L              # coeff of f[1]

    # self_matrix: right-boundary only extrapolation (one-sided)
    self_matrix_np = np.zeros((C, d), dtype=np.float64)
    self_matrix_np[:, d-2] = -h10 * L
    self_matrix_np[:, d-1] = h00 + h10 * L

    return FCGramMatrices(
        self_matrix=jnp.array(self_matrix_np, dtype=dtype),
        boundary_matrix=jnp.array(boundary_matrix_np, dtype=dtype),
        d=d,
        C=C,
    )


def apply_continuation_1d(
    f: jax.Array,
    matrices: FCGramMatrices,
    axis: int = -1,
) -> jax.Array:
    """Extend a 1D signal to periodic via FC-Gram continuation.

    Appends C continuation points that smoothly bridge from the end
    of the signal back to the start, making [f | continuation] periodic.

    Uses d points from each end of the signal to compute the bridge.
    When the signal already includes halo data (e.g., from pad_halo),
    the continuation naturally benefits from the extra boundary info.

    Parameters
    ----------
    f : jax.Array, shape (..., N, ...)
        Signal along the specified axis.
    matrices : FCGramMatrices
        Precomputed continuation matrices.
    axis : int
        Axis along which to apply continuation.

    Returns
    -------
    jax.Array, shape (..., N + C, ...)
        Extended (periodic) data.
    """
    d = matrices.d
    C = matrices.C

    # Move target axis to last position for convenience
    f_moved = jnp.moveaxis(f, axis, -1)
    N = f_moved.shape[-1]

    # d points from each end of the signal
    bnd_right = f_moved[..., -d:]    # (..., d)
    bnd_left = f_moved[..., :d]      # (..., d)

    # boundary_matrix maps [right_d | left_d] → C continuation points
    bnd_both = jnp.concatenate([bnd_right, bnd_left], axis=-1)  # (..., 2*d)
    cont = jnp.einsum("ci,...i->...c", matrices.boundary_matrix, bnd_both)  # (..., C)

    # Extended signal: [f | continuation]
    f_ext = jnp.concatenate([f_moved, cont], axis=-1)  # (..., N+C)

    return jnp.moveaxis(f_ext, -1, axis)


def spectral_derivative_1d(
    f_extended: jax.Array,
    dx: float,
    N_original: int,
    C: int,
    axis: int = -1,
    order: int = 1,
) -> jax.Array:
    """Compute spectral derivative of FC-extended signal.

    Parameters
    ----------
    f_extended : jax.Array, shape (..., N+C, ...)
        FC-extended (periodic) data.
    dx : float
        Grid spacing.
    N_original : int
        Original (non-extended) number of points.
    C : int
        Number of continuation points.
    axis : int
        Axis along which to differentiate.
    order : int
        Derivative order (1, 2, ...).

    Returns
    -------
    jax.Array, shape (..., N_original, ...)
        Derivative restricted to original grid points.
    """
    f = jnp.moveaxis(f_extended, axis, -1)
    N_ext = N_original + C

    # FFT
    f_hat = jnp.fft.rfft(f, axis=-1)

    # Wavenumbers for the extended domain
    # Physical domain length for extended signal
    L = N_ext * dx
    k = 2.0 * jnp.pi * jnp.fft.rfftfreq(N_ext, d=dx)

    # Spectral derivative: multiply by (ik)^order
    multiplier = (1j * k) ** order
    df_hat = f_hat * multiplier

    # IFFT and restrict to original N points
    df = jnp.fft.irfft(df_hat, n=N_ext, axis=-1)
    df = df[..., :N_original]

    return jnp.moveaxis(df, -1, axis)


def spectral_filter_1d(
    f_extended: jax.Array,
    dx: float,
    N_original: int,
    C: int,
    axis: int = -1,
    cutoff_fraction: float = 2.0 / 3.0,
) -> jax.Array:
    """Apply spectral low-pass filter to FC-extended signal.

    Parameters
    ----------
    f_extended : jax.Array, shape (..., N+C, ...)
        FC-extended (periodic) data.
    dx : float
        Grid spacing.
    N_original : int
        Original number of points.
    C : int
        Number of continuation points.
    axis : int
        Axis along which to filter.
    cutoff_fraction : float
        Fraction of modes to keep (2/3 = standard dealiasing).

    Returns
    -------
    jax.Array, shape (..., N_original, ...)
        Filtered data restricted to original grid.
    """
    f = jnp.moveaxis(f_extended, axis, -1)
    N_ext = N_original + C

    f_hat = jnp.fft.rfft(f, axis=-1)

    # Zero out modes above cutoff
    n_modes = f_hat.shape[-1]
    k_cut = int(n_modes * cutoff_fraction)
    mask = jnp.where(jnp.arange(n_modes) < k_cut, 1.0, 0.0)
    f_hat_filtered = f_hat * mask

    f_filtered = jnp.fft.irfft(f_hat_filtered, n=N_ext, axis=-1)
    f_filtered = f_filtered[..., :N_original]

    return jnp.moveaxis(f_filtered, -1, axis)
