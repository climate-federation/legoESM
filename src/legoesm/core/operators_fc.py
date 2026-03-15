"""FC-Gram spectral operators on the cubed-sphere A-grid.

Drop-in replacements for operators.py (centered) and operators_fv.py (PPM),
using Fourier Continuation for spectral accuracy on each face.

Shared by all equation sets: atmosphere (SW, PE, CE) and ocean PE.

The key advantage: FC computes derivatives spectrally within each face,
giving exponential convergence for smooth fields. At face boundaries,
the FC continuation bridges the gap using halo data, avoiding the
Gibbs phenomenon that would arise from a naive FFT on non-periodic data.

Divergence damping operators are also provided, replacing the finite-
difference chained-halo approach in shallow_water_cgrid.py with a
single-exchange spectral computation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.fc_gram import (
    FCGramMatrices,
    build_fc_gram_matrices,
    apply_continuation_1d,
    spectral_derivative_1d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector


class FCOperatorConfig(NamedTuple):
    """Configuration for FC-Gram spectral operators.

    Attributes
    ----------
    matrices : FCGramMatrices
        Precomputed continuation matrices.
    div_damp_2 : float
        2nd-order divergence damping coefficient.
    div_damp_4 : float
        4th-order divergence damping coefficient.
    """
    matrices: FCGramMatrices
    div_damp_2: float = 0.0
    div_damp_4: float = 0.0


def build_fc_config(
    d: int = 2,
    C: int = 4,
    degree: int = 5,
    div_damp_2: float = 0.0,
    div_damp_4: float = 0.0,
    dtype=jnp.float32,
) -> FCOperatorConfig:
    """Convenience builder for FCOperatorConfig."""
    matrices = build_fc_gram_matrices(d=d, C=C, degree=degree, dtype=dtype)
    return FCOperatorConfig(
        matrices=matrices,
        div_damp_2=div_damp_2,
        div_damp_4=div_damp_4,
    )


# ==============================================================================
# Internal helpers
# ==============================================================================

def _fc_derivative_x_index(q_padded: jax.Array,
                           fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral x-derivative in INDEX space (dx=1.0) on padded data.

    Returns df/di (index-space derivative), shape (6, n, n).
    For physical derivative: df/dx = (1/hx) * df/di where hx = grid.dx/2.
    """
    C = fc_config.matrices.C
    n = q_padded.shape[1] - 2
    n_pad = n + 2

    q_row = q_padded[:, :, 1:-1]  # (6, n+2, n)

    f_ext = apply_continuation_1d(
        q_row, fc_config.matrices, axis=1,
    )  # (6, n+2+C, n)

    def _deriv_face(f_ext_face):
        return spectral_derivative_1d(
            f_ext_face, 1.0, n_pad, C, axis=0, order=1,
        )

    result_padded = jax.vmap(_deriv_face)(f_ext)  # (6, n+2, n)
    return result_padded[:, 1:-1, :]


def _fc_derivative_y_index(q_padded: jax.Array,
                           fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral y-derivative in INDEX space (dy=1.0) on padded data.

    Returns df/dj (index-space derivative), shape (6, n, n).
    For physical derivative: df/dy = (1/hy) * df/dj where hy = grid.dy/2.
    """
    C = fc_config.matrices.C
    n = q_padded.shape[2] - 2
    n_pad = n + 2

    q_col = q_padded[:, 1:-1, :]  # (6, n, n+2)

    f_ext = apply_continuation_1d(
        q_col, fc_config.matrices, axis=2,
    )  # (6, n, n+2+C)

    def _deriv_face(f_ext_face):
        return spectral_derivative_1d(
            f_ext_face, 1.0, n_pad, C, axis=1, order=1,
        )

    result_padded = jax.vmap(_deriv_face)(f_ext)  # (6, n, n+2)
    return result_padded[:, :, 1:-1]


def _fc_derivative_x(q_padded: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral x-derivative in PHYSICAL space on padded data.

    Computes df/di in index space via FC, then divides by the local
    metric hx = grid.dx/2 to get the physical derivative df/dx.
    Uses position-varying metrics for accuracy near face edges.

    Returns shape (6, n, n).
    """
    hx = grid.dx / 2.0  # (6, n, n) — position-varying
    return _fc_derivative_x_index(q_padded, fc_config) / hx


def _fc_derivative_y(q_padded: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral y-derivative in PHYSICAL space on padded data.

    Computes df/dj in index space via FC, then divides by the local
    metric hy = grid.dy/2 to get the physical derivative df/dy.
    Uses position-varying metrics for accuracy near face edges.

    Returns shape (6, n, n).
    """
    hy = grid.dy / 2.0  # (6, n, n) — position-varying
    return _fc_derivative_y_index(q_padded, fc_config) / hy


# ==============================================================================
# Standard A-grid operators
# ==============================================================================

def fc_gradient_x(q: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral x-gradient of scalar field.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    q_padded = pad_halo(q, interp_offsets=grid.halo_interp_offsets)
    return _fc_derivative_x(q_padded, grid, fc_config)


def fc_gradient_y(q: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral y-gradient of scalar field.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    q_padded = pad_halo(q, interp_offsets=grid.halo_interp_offsets)
    return _fc_derivative_y(q_padded, grid, fc_config)


def fc_divergence(u: jax.Array, v: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral divergence of vector field (u, v).

    Uses the discrete Gauss theorem form on orthogonal curvilinear coords:
        div = (1/A) * [d(u*hy)/di + d(v*hx)/dj]

    where d/di, d/dj are INDEX-SPACE FC spectral derivatives (dx=1.0),
    hx = grid.dx/2 and hy = grid.dy/2 are single-cell edge lengths,
    and A is the exact spherical cell area.

    Parameters
    ----------
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # Metric-weighted fluxes on padded grid
    flux_x = u_pad * grid.hy_ext  # (6, n+2, n+2)
    flux_y = v_pad * grid.hx_ext

    # Index-space spectral derivatives of metric-weighted fluxes
    d_flux_x_di = _fc_derivative_x_index(flux_x, fc_config)  # d(u*hy)/di
    d_flux_y_dj = _fc_derivative_y_index(flux_y, fc_config)  # d(v*hx)/dj

    return (d_flux_x_di + d_flux_y_dj) / grid.area


def fc_curl_z(u: jax.Array, v: jax.Array, grid: CubedSphereGrid,
              fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral vorticity (vertical component of curl).

    Uses the discrete Stokes theorem form:
        curl_z = (1/A) * [d(v*hy)/di - d(u*hx)/dj]

    where d/di, d/dj are index-space FC spectral derivatives.

    Parameters
    ----------
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    vort_x = v_pad * grid.hy_ext
    vort_y = u_pad * grid.hx_ext

    d_vort_x_di = _fc_derivative_x_index(vort_x, fc_config)  # d(v*hy)/di
    d_vort_y_dj = _fc_derivative_y_index(vort_y, fc_config)  # d(u*hx)/dj

    return (d_vort_x_di - d_vort_y_dj) / grid.area


def fc_laplacian(q: jax.Array, grid: CubedSphereGrid,
                 fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral Laplacian: div(grad(q)).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    gx = fc_gradient_x(q, grid, fc_config)
    gy = fc_gradient_y(q, grid, fc_config)
    return fc_divergence(gx, gy, grid, fc_config)


def fc_hyperdiffusion(q: jax.Array, grid: CubedSphereGrid,
                      fc_config: FCOperatorConfig,
                      coeff: float) -> jax.Array:
    """FC spectral hyperdiffusion: -coeff * nabla^4(q).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig
    coeff : float

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    lap1 = fc_laplacian(q, grid, fc_config)
    lap2 = fc_laplacian(lap1, grid, fc_config)
    return -coeff * lap2


def fc_flux_divergence(q: jax.Array, u: jax.Array, v: jax.Array,
                       grid: CubedSphereGrid,
                       fc_config: FCOperatorConfig) -> jax.Array:
    """Conservative flux-form transport: dq/dt = -div(q * v).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    return fc_divergence(q * u, q * v, grid, fc_config)


def fc_scalar_advection(q: jax.Array, u: jax.Array, v: jax.Array,
                        grid: CubedSphereGrid,
                        fc_config: FCOperatorConfig) -> jax.Array:
    """Advective transport: -v . grad(q).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    dq_dx = fc_gradient_x(q, grid, fc_config)
    dq_dy = fc_gradient_y(q, grid, fc_config)
    return -(u * dq_dx + v * dq_dy)


# ==============================================================================
# Divergence damping operators
# ==============================================================================

def fc_divergence_damping(u: jax.Array, v: jax.Array,
                          grid: CubedSphereGrid,
                          fc_config: FCOperatorConfig,
                          ) -> tuple[jax.Array, jax.Array]:
    """FC spectral divergence damping for momentum.

    Computes:
        2nd order: +nu2 * grad(div)
        4th order: -nu4 * grad(lap(div))

    One halo exchange → spectral derivatives for all orders.
    Replaces the 3-exchange finite-difference approach.

    Parameters
    ----------
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    du_damp, dv_damp : jax.Array, shape (6, n, n)
    """
    du_damp = jnp.zeros_like(u)
    dv_damp = jnp.zeros_like(v)

    nu2 = fc_config.div_damp_2
    nu4 = fc_config.div_damp_4

    if nu2 <= 0 and nu4 <= 0:
        return du_damp, dv_damp

    # Compute divergence spectrally
    div = fc_divergence(u, v, grid, fc_config)

    if nu2 > 0:
        grad_div_x = fc_gradient_x(div, grid, fc_config)
        grad_div_y = fc_gradient_y(div, grid, fc_config)
        du_damp = du_damp + nu2 * grad_div_x
        dv_damp = dv_damp + nu2 * grad_div_y

    if nu4 > 0:
        lap_div = fc_laplacian(div, grid, fc_config)
        grad_lap_x = fc_gradient_x(lap_div, grid, fc_config)
        grad_lap_y = fc_gradient_y(lap_div, grid, fc_config)
        du_damp = du_damp - nu4 * grad_lap_x
        dv_damp = dv_damp - nu4 * grad_lap_y

    return du_damp, dv_damp
