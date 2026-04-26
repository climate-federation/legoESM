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
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_vector,
    pad_halo_4d,
    pad_halo_vector_4d,
)


def _fc_pad_halo(field: jax.Array, grid) -> jax.Array:
    """Halo pad for scalar FC operators, dispatched on rank.

    For 3D ``(6, n, n)`` input uses the existing 2D ``pad_halo``; for
    4D ``(6, n, n, nlev)`` uses ``pad_halo_4d`` so all levels share a
    single MPI exchange.  Matches the original FC operator's
    ``pad_halo(q, interp_offsets=grid.halo_interp_offsets)`` call
    (no duogrid path).
    """
    if field.ndim == 3:
        return pad_halo(field, interp_offsets=grid.halo_interp_offsets)
    return pad_halo_4d(field, halo=1, interp_offsets=grid.halo_interp_offsets)


def _fc_pad_halo_vector(u: jax.Array, v: jax.Array, grid) -> tuple[jax.Array, jax.Array]:
    """Vector halo pad for FC operators, dispatched on rank."""
    if u.ndim == 3:
        return pad_halo_vector(
            u, v,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets,
        )
    return pad_halo_vector_4d(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
        halo=1,
    )


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

    Works for ``q_padded`` shape ``(6, n+2, n+2[, ...])`` — trailing
    axes (e.g. ``nlev``) are passively carried through.  ``axis=1``
    is the i (x) direction.  Returns df/di with shape
    ``(6, n, n[, ...])``.  ``spectral_derivative_1d`` is axis-aware,
    so we pass the i-axis index directly instead of wrapping the
    derivative in a per-face ``vmap``.
    """
    C = fc_config.matrices.C
    n = q_padded.shape[1] - 2
    n_pad = n + 2

    # Strip y halo on axis 2 — broadcasts over any trailing axes.
    q_row = q_padded[:, :, 1:-1]  # (6, n+2, n[, ...])

    f_ext = apply_continuation_1d(
        q_row, fc_config.matrices, axis=1,
    )  # (6, n+2+C, n[, ...])

    result_padded = spectral_derivative_1d(
        f_ext, 1.0, n_pad, C, axis=1, order=1,
    )  # (6, n+2, n[, ...])
    # Strip x halo from axis 1.
    return result_padded[:, 1:-1]  # (6, n, n[, ...])


def _fc_derivative_y_index(q_padded: jax.Array,
                           fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral y-derivative in INDEX space (dy=1.0) on padded data.

    Works for ``q_padded`` shape ``(6, n+2, n+2[, ...])``.  ``axis=2``
    is the j (y) direction.
    """
    C = fc_config.matrices.C
    n = q_padded.shape[2] - 2
    n_pad = n + 2

    # Strip x halo on axis 1.
    q_col = q_padded[:, 1:-1, :]  # (6, n, n+2[, ...])

    f_ext = apply_continuation_1d(
        q_col, fc_config.matrices, axis=2,
    )  # (6, n, n+2+C[, ...])

    result_padded = spectral_derivative_1d(
        f_ext, 1.0, n_pad, C, axis=2, order=1,
    )  # (6, n, n+2[, ...])
    # Strip y halo from axis 2.
    return result_padded[:, :, 1:-1]  # (6, n, n[, ...])


def _broadcast_metric_to_field(metric: jax.Array, field: jax.Array) -> jax.Array:
    """Add trailing singleton axes to ``metric`` so it broadcasts against ``field``."""
    extra = field.ndim - metric.ndim
    if extra <= 0:
        return metric
    return metric[(...,) + (None,) * extra]


def _fc_derivative_x(q_padded: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral x-derivative in PHYSICAL space on padded data.

    Computes df/di in index space via FC, then divides by the local
    metric hx = grid.dx/2 to get the physical derivative df/dx.
    Uses position-varying metrics for accuracy near face edges.

    Works for ``q_padded`` shape ``(6, n+2, n+2[, ...])``; returns
    shape ``(6, n, n[, ...])``.  ``hx`` is broadcast across any
    trailing (e.g. ``nlev``) axes.
    """
    hx = grid.dx / 2.0  # (6, n, n) — position-varying
    deriv = _fc_derivative_x_index(q_padded, fc_config)
    return deriv / _broadcast_metric_to_field(hx, deriv)


def _fc_derivative_y(q_padded: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """Spectral y-derivative in PHYSICAL space on padded data.

    Computes df/dj in index space via FC, then divides by the local
    metric hy = grid.dy/2 to get the physical derivative df/dy.

    Works for ``q_padded`` shape ``(6, n+2, n+2[, ...])``.
    """
    hy = grid.dy / 2.0  # (6, n, n) — position-varying
    deriv = _fc_derivative_y_index(q_padded, fc_config)
    return deriv / _broadcast_metric_to_field(hy, deriv)


# ==============================================================================
# Standard A-grid operators
# ==============================================================================

def fc_gradient_x(q: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig,
                  padded: jax.Array | None = None) -> jax.Array:
    """FC spectral x-gradient of scalar field.

    Accepts ``q`` shape ``(6, n, n)`` or ``(6, n, n, nlev)``.  The 4D
    path uses ``pad_halo_4d`` so all vertical levels share a single MPI
    halo exchange.

    Parameters
    ----------
    padded : jax.Array, optional
        Pre-padded field with halo=1.  When provided, the internal
        halo exchange is skipped — useful for paired (∂/∂x, ∂/∂y) calls
        on the same input where the halo can be shared.
    """
    q_padded = padded if padded is not None else _fc_pad_halo(q, grid)
    return _fc_derivative_x(q_padded, grid, fc_config)


def fc_gradient_y(q: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig,
                  padded: jax.Array | None = None) -> jax.Array:
    """FC spectral y-gradient of scalar field.

    Accepts 3D or 4D input as for :func:`fc_gradient_x`.  Optional
    ``padded=`` skips the internal halo exchange — see
    :func:`fc_gradient_x` for usage.
    """
    q_padded = padded if padded is not None else _fc_pad_halo(q, grid)
    return _fc_derivative_y(q_padded, grid, fc_config)


def fc_divergence(u: jax.Array, v: jax.Array, grid: CubedSphereGrid,
                  fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral divergence of vector field (u, v).

    Uses the discrete Gauss theorem form on orthogonal curvilinear coords:
        div = (1/A) * [d(u*hy)/di + d(v*hx)/dj]

    where d/di, d/dj are INDEX-SPACE FC spectral derivatives (dx=1.0),
    hx = grid.dx/2 and hy = grid.dy/2 are single-cell edge lengths,
    and A is the exact spherical cell area.

    Accepts ``u, v`` shape ``(6, n, n)`` or ``(6, n, n, nlev)``.  The
    4D path uses ``pad_halo_vector_4d`` so all levels share one MPI
    vector halo exchange.
    """
    u_pad, v_pad = _fc_pad_halo_vector(u, v, grid)

    # Metric-weighted fluxes on padded grid.  ``hy_ext`` is (6, n+2, n+2);
    # broadcast to match a possible trailing nlev axis.
    hy_ext = _broadcast_metric_to_field(grid.hy_ext, u_pad)
    hx_ext = _broadcast_metric_to_field(grid.hx_ext, v_pad)
    flux_x = u_pad * hy_ext
    flux_y = v_pad * hx_ext

    # Index-space spectral derivatives of metric-weighted fluxes
    d_flux_x_di = _fc_derivative_x_index(flux_x, fc_config)  # d(u*hy)/di
    d_flux_y_dj = _fc_derivative_y_index(flux_y, fc_config)  # d(v*hx)/dj

    sum_flux = d_flux_x_di + d_flux_y_dj
    return sum_flux / _broadcast_metric_to_field(grid.area, sum_flux)


def fc_curl_z(u: jax.Array, v: jax.Array, grid: CubedSphereGrid,
              fc_config: FCOperatorConfig) -> jax.Array:
    """FC spectral vorticity (vertical component of curl).

    Uses the discrete Stokes theorem form:
        curl_z = (1/A) * [d(v*hy)/di - d(u*hx)/dj]

    where d/di, d/dj are index-space FC spectral derivatives.

    Accepts 3D or 4D input as for :func:`fc_divergence`.
    """
    u_pad, v_pad = _fc_pad_halo_vector(u, v, grid)

    hy_ext = _broadcast_metric_to_field(grid.hy_ext, v_pad)
    hx_ext = _broadcast_metric_to_field(grid.hx_ext, u_pad)
    vort_x = v_pad * hy_ext
    vort_y = u_pad * hx_ext

    d_vort_x_di = _fc_derivative_x_index(vort_x, fc_config)  # d(v*hy)/di
    d_vort_y_dj = _fc_derivative_y_index(vort_y, fc_config)  # d(u*hx)/dj

    diff_vort = d_vort_x_di - d_vort_y_dj
    return diff_vort / _broadcast_metric_to_field(grid.area, diff_vort)


def fc_laplacian(q: jax.Array, grid: CubedSphereGrid,
                 fc_config: FCOperatorConfig,
                 padded: jax.Array | None = None) -> jax.Array:
    """FC spectral Laplacian: div(grad(q)).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n) or (6, n, n, nlev)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig
    padded : jax.Array, optional
        Pre-padded scalar field with halo=1.  When provided, the
        internal halo exchange is skipped *and* shared between the
        ``∂q/∂x`` and ``∂q/∂y`` calls below — saves one full halo
        exchange relative to the previous implementation that padded
        ``q`` inside each gradient call.

    Returns
    -------
    jax.Array, shape (6, n, n) or (6, n, n, nlev)
    """
    # Pad ``q`` once and share between the two gradient operators —
    # previously each ``fc_gradient_x/_y`` issued its own
    # ``_fc_pad_halo(q)`` collective on the same input, doubling the
    # halo cost.  Hot path: every fc_laplacian / fc_hyperdiffusion
    # call (the latter is two laplacians).
    q_pad = padded if padded is not None else _fc_pad_halo(q, grid)
    gx = fc_gradient_x(q, grid, fc_config, padded=q_pad)
    gy = fc_gradient_y(q, grid, fc_config, padded=q_pad)
    return fc_divergence(gx, gy, grid, fc_config)


def fc_hyperdiffusion(q: jax.Array, grid: CubedSphereGrid,
                      fc_config: FCOperatorConfig,
                      coeff: float,
                      padded: jax.Array | None = None) -> jax.Array:
    """FC spectral hyperdiffusion: -coeff * nabla^4(q).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n) or (6, n, n, nlev)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig
    coeff : float
    padded : jax.Array, optional
        Pre-padded scalar field with halo=1 forwarded to the inner
        Laplacian — share with a co-located explicit Laplacian on the
        same input to halve the halo cost when both are active.

    Returns
    -------
    jax.Array, shape (6, n, n) or (6, n, n, nlev)
    """
    lap1 = fc_laplacian(q, grid, fc_config, padded=padded)
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
