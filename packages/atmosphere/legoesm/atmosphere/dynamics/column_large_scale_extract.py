"""Extract a column's large-scale forcing from a GCM lat-lon state.

Stage 4 of ``docs/COMPARE_REANALYSIS.md`` (gap #3, **grid-side** half): derive
the large-scale environment of a flagged GCM column — large-scale subsidence
(``ω`` from the continuity equation), and horizontal advective tendencies of
potential temperature and water vapour (``−V·∇θ``, ``−V·∇q``) — and pack them
into the :class:`~legoesm.atmosphere.column_forcing.ColumnLargeScaleState` that
:func:`~legoesm.atmosphere.column_forcing.build_column_scm_forcing` (iter 5)
turns into the LES forcing.  Forcing the LES from the *GCM* column's large-scale
state is what makes the LES↔GCM discrepancy attribute to the physics/closure.

Reuse only (CLAUDE.md — no re-derived numerics):

* Horizontal derivatives via the metric-aware lat-lon operators
  :func:`legoesm.core.operators_latlon.gradient` / ``divergence`` (they handle
  the spherical metric + halos/poles correctly), applied per level.
* ``ω`` via the canonical continuity chain
  :func:`legoesm.grids.vertical.compute_sigma_dot_and_total` +
  :func:`~legoesm.grids.vertical.compute_pressure_velocity`.
* Potential temperature via :func:`legoesm.atmosphere.physics._shared.exner_function`.
* Coriolis handled in :func:`~legoesm.atmosphere.column_forcing.build_column_scm_forcing`.

Geostrophic wind (``∇Φ``) is a documented follow-up — the SCM forcing simply
disables geostrophic relaxation when ``u_geo`` is ``None``.  Cubed-sphere /
Gaussian extraction is also follow-up; this module is lat-lon (ERA5-native).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from legoesm.atmosphere.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.core.field import Field
from legoesm.core.operators_latlon import divergence, gradient
from legoesm.grids.vertical import (
    compute_pressure_velocity,
    compute_sigma_dot_and_total,
)


def advective_tendency(
    u: jax.Array, v: jax.Array, dphi_dx: jax.Array, dphi_dy: jax.Array
) -> jax.Array:
    """Large-scale horizontal advective tendency ``−(u ∂φ/∂x + v ∂φ/∂y)``.

    All inputs are co-located (same shape); returns the same shape.  The sign is
    the tendency *imposed on the column* by horizontal advection (warm-air
    advection ``∂φ/∂x<0`` with ``u>0`` ⇒ positive tendency).
    """
    u = jnp.asarray(u)
    return -(u * jnp.asarray(dphi_dx) + jnp.asarray(v) * jnp.asarray(dphi_dy))


def omega_from_divergence(
    div_3d: jax.Array, p_s: jax.Array, sigma_coord: Any
) -> jax.Array:
    """Pressure velocity ``ω`` from the horizontal-divergence profile.

    Closes the hydrostatic continuity equation: ``∂p_s/∂t = −p_s·D_total/(1−σ_top)``
    with ``D_total = Σ_k (∇·v)_k Δσ_k`` (the column-integrated divergence the
    dycore uses for ``dp_s/dt``; see :func:`compute_sigma_dot_and_total`), then
    ``ω = σ·∂p_s/∂t + p_s·σ̇`` via the canonical
    :func:`compute_pressure_velocity`.  ``div_3d`` is ``(..., nlev)``; returns
    ``ω`` ``(..., nlev)`` [Pa/s] (positive = sinking).
    """
    div_3d = jnp.asarray(div_3d)
    p_s = jnp.asarray(p_s, dtype=div_3d.dtype)
    sigma_dot, d_total = compute_sigma_dot_and_total(div_3d, sigma_coord)
    sigma_top = jnp.asarray(sigma_coord.sigma_half[0], dtype=div_3d.dtype)
    # D_total carries a trailing singleton (..., 1); squeeze to the surface
    # shape.  dp_s/dt = -p_s · D_total / (1 - σ_top) (continuity, the dycore's
    # surface-pressure-tendency closure).
    dp_s_dt = -p_s * d_total[..., 0] / (1.0 - sigma_top)
    return compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma_coord)


def _gradient_latlon_3d(field_3d: jax.Array, grid: Any) -> tuple[jax.Array, jax.Array]:
    """Per-level ``(∂/∂x, ∂/∂y)`` via the lat-lon operator, vmapped over levels."""
    def _one(scalar_2d):
        gx, gy = gradient(Field(data=scalar_2d), grid)
        return gx.data, gy.data

    return jax.vmap(_one, in_axes=-1, out_axes=-1)(jnp.asarray(field_3d))


def _divergence_latlon_3d(u3d: jax.Array, v3d: jax.Array, grid: Any) -> jax.Array:
    """Per-level horizontal divergence via the lat-lon operator (vmapped)."""
    def _one(us, vs):
        return divergence(Field(data=us), Field(data=vs), grid).data

    return jax.vmap(_one, in_axes=(-1, -1), out_axes=-1)(
        jnp.asarray(u3d), jnp.asarray(v3d)
    )


def extract_column_forcing_latlon(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma_coord: Any,
    lat_rad: float,
    col_index: tuple[int, int],
) -> ColumnLargeScaleState:
    """Build a column's :class:`ColumnLargeScaleState` from a lat-lon GCM state.

    ``T``/``q_v``/``u``/``v`` are ``(n_lat, n_lon, nlev)`` (surface-last), ``p_s``
    ``(n_lat, n_lon)``; ``grid`` is the :class:`LatLonGrid`; ``sigma_coord`` the
    vertical coordinate.  ``col_index = (i_lat, i_lon)`` selects the flagged
    column and MUST be static Python ints (the manifest provides them) — the
    gather is a static index, so under ``jax.jit`` pass ``col_index`` as a
    closed-over constant, not a traced argument.  ``lat_rad`` is the column's
    latitude.  Returns the column's
    profiles plus ``ω`` (large-scale subsidence) and ``θ``/``q_v`` advective
    tendencies — ready for
    :func:`~legoesm.atmosphere.column_forcing.build_column_scm_forcing`.

    Pure pressures use a sigma coordinate (``p = σ·p_s``); ``ω`` and the
    advective tendencies are diagnosed on the FULL grid (so the metric-correct
    stencils see the neighbourhood) and then the single column is gathered.
    """
    T = jnp.asarray(T)
    q_v = jnp.asarray(q_v, dtype=T.dtype)
    u = jnp.asarray(u, dtype=T.dtype)
    v = jnp.asarray(v, dtype=T.dtype)
    p_s = jnp.asarray(p_s, dtype=T.dtype)
    sigma_full = jnp.asarray(sigma_coord.sigma_full, dtype=T.dtype)
    p_full = p_s[..., None] * sigma_full

    theta = T / exner_function(p_full)
    th_x, th_y = _gradient_latlon_3d(theta, grid)
    q_x, q_y = _gradient_latlon_3d(q_v, grid)
    theta_adv_3d = advective_tendency(u, v, th_x, th_y)
    qv_adv_3d = advective_tendency(u, v, q_x, q_y)

    div_3d = _divergence_latlon_3d(u, v, grid)
    omega_3d = omega_from_divergence(div_3d, p_s, sigma_coord)

    i, j = int(col_index[0]), int(col_index[1])
    return ColumnLargeScaleState(
        lat_rad=lat_rad,
        T=T[i, j, :],
        p_full=p_full[i, j, :],
        q_v=q_v[i, j, :],
        omega=omega_3d[i, j, :],
        theta_adv=theta_adv_3d[i, j, :],
        qv_adv=qv_adv_3d[i, j, :],
    )
