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

Geostrophic wind (``∇Φ``, iter 29): :func:`geostrophic_wind_from_gradients`
diagnoses ``u_geo``/``v_geo`` from the sigma-surface geopotential + surface-
pressure gradients (two-term pressure-gradient conversion to the pressure
surface), populated by the **lat-lon** extractor and wired into the column LES as
the plane Coriolis reference wind (``f×(V − V_geo)``).  ``u_geo = None`` supplies
no geostrophic reference (within :data:`_MIN_GEOSTROPHIC_LAT_DEG` of the equator,
and on the cubed sphere pending a metric-correct east/north↔grid rotation — see
the cubed extractor); the plane Coriolis then falls back to ``f×V``.

Grids: ERA5 is lat-lon (``extract_column_forcing_latlon``), but the model's
flagship dycore is cubed-sphere, so a worst column flagged on the *native* model
grid needs cubed-sphere extraction (``extract_column_forcing_cubed_sphere``).
Both share the grid-agnostic continuity chain (:func:`omega_from_divergence`) and
advection (:func:`advective_tendency`); only the horizontal operators differ.
:func:`extract_column_forcing` dispatches on the grid type (raises on an
unsupported grid — Gaussian/Voronoi remain follow-ups).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from legoesm.atmosphere.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    exner_function,
    virtual_temperature,
)
from legoesm.core.field import Field
from legoesm.core.operators_latlon import divergence, gradient
from legoesm.grids.vertical import (
    compute_pressure_velocity,
    compute_sigma_dot_and_total,
)

from legoesm import constants

# --- geostrophic-balance validity (diagnostic convention, not tunable) -------
# Geostrophic balance is ill-posed near the equator (f → 0); below this latitude
# the column supplies NO geostrophic reference wind (u_geo/v_geo = None), so the
# column LES's plane Coriolis falls back to f×V (no geostrophic target).
_MIN_GEOSTROPHIC_LAT_DEG = 5.0  # ~|f| ≥ 1.3e-5 1/s; standard extratropical cutoff


def geostrophic_wind_from_gradients(
    dphi_dx_sigma: jax.Array,
    dphi_dy_sigma: jax.Array,
    dlnps_dx: jax.Array,
    dlnps_dy: jax.Array,
    T_v: jax.Array,
    f_c: float,
) -> tuple[jax.Array, jax.Array]:
    """Geostrophic wind from the sigma-surface geopotential + surface-pressure gradients.

    Uses the standard two-term pressure-gradient decomposition to convert the
    **sigma-surface** geopotential gradient ``∂Φ/∂·|_σ`` to the **pressure-surface**
    gradient ``∂Φ/∂·|_p`` that enters geostrophic balance::

        ∂Φ/∂x|_p = ∂Φ/∂x|_σ + R_d·T_v·∂ln p_s/∂x          (derivation: at constant p,
        dσ/dx|_p = −σ ∂ln p_s/∂x and ∂Φ/∂σ = −R_d T_v/σ ⇒ the +R_d T_v ∂ln p_s term)

    then geostrophic balance ``f v_g = ∂Φ/∂x|_p``, ``f u_g = −∂Φ/∂y|_p``::

        v_g = +(∂Φ/∂x|_σ + R_d·T_v·∂ln p_s/∂x) / f
        u_g = −(∂Φ/∂y|_σ + R_d·T_v·∂ln p_s/∂y) / f

    All inputs are the gathered single column (``(nlev,)`` for the Φ gradients +
    ``T_v``; the ``ln p_s`` gradient is column-scalar and broadcasts).  Components
    are in the SAME grid-aligned frame as the gradient operator (so they match the
    model's ``u``/``v``).  ``f_c`` is the column's scalar Coriolis; the caller
    guards ``|f_c|`` away from zero (see :data:`_MIN_GEOSTROPHIC_LAT_DEG`), so no
    masked division is needed here.  Pure-JAX, differentiable.
    """
    dphi_dx_p = dphi_dx_sigma + constants.R_d * T_v * dlnps_dx
    dphi_dy_p = dphi_dy_sigma + constants.R_d * T_v * dlnps_dy
    u_g = -dphi_dy_p / f_c
    v_g = dphi_dx_p / f_c
    return u_g, v_g


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


def _geopotential_full_grid(
    T: jax.Array, q_v: jax.Array, p_s: jax.Array, sigma_coord: Any
) -> jax.Array:
    """Geopotential ``Φ = g·z`` [m²/s²] at full levels on the full grid (any shape).

    Reuses the shared hydrostatic height integral
    (:func:`compute_heights_from_sigma`) over a flattened column axis, then
    reshapes back.  ``z`` is height ABOVE the surface, so ``Φ`` is the
    above-surface geopotential — exact for a flat/ocean surface (``z_s = 0``, the
    LES worst-column use case); the orographic surface-geopotential gradient
    ``g·∇z_s`` is a documented follow-up for terrain.
    """
    spatial = T.shape[:-1]
    nlev = T.shape[-1]
    ncol = 1
    for d in spatial:
        ncol *= int(d)
    sigma_half = jnp.asarray(sigma_coord.sigma_half, dtype=T.dtype)
    p_half = p_s.reshape(ncol, 1) * sigma_half[None, :]
    z_full, _ = compute_heights_from_sigma(
        T.reshape(ncol, nlev), p_half, q_v.reshape(ncol, nlev)
    )
    return constants.g * z_full.reshape(spatial + (nlev,))


def _geostrophic_wind_column(
    *,
    T: jax.Array,
    q_v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma_coord: Any,
    grad_fn,
    lat_rad: float,
    col_index: tuple[int, ...],
) -> tuple[jax.Array | None, jax.Array | None]:
    """Diagnose one column's geostrophic wind (``u_geo``/``v_geo``) or ``(None, None)``.

    Computes ``Φ = g·z`` + ``ln p_s`` on the FULL grid, takes their horizontal
    gradients with the grid's operator ``grad_fn(field_3d, grid) -> (gx, gy)``
    (so the metric-correct stencil sees the neighbourhood), gathers the flagged
    column, and applies :func:`geostrophic_wind_from_gradients`.  Returns
    ``(None, None)`` within :data:`_MIN_GEOSTROPHIC_LAT_DEG` of the equator (where
    geostrophic balance is ill-posed) — no geostrophic reference wind is supplied
    there, so the column LES's plane Coriolis falls back to ``f×V``.

    **Precondition** — ``grad_fn``'s ``(x, y)`` axes must be **geographic**
    (east/north), because geostrophic balance is applied directly in that frame.
    This holds for the lat-lon operator; the cubed-sphere operators return
    grid-axis derivatives on a non-orthogonal grid, so the cubed-sphere extractor
    does NOT call this (it supplies no geostrophic reference wind pending a
    metric-correct rotation; the plane Coriolis then falls back to f×V).
    """
    if abs(math.degrees(float(lat_rad))) < _MIN_GEOSTROPHIC_LAT_DEG:
        return None, None
    phi = _geopotential_full_grid(T, q_v, p_s, sigma_coord)
    dphi_dx, dphi_dy = grad_fn(phi, grid)
    ln_ps = jnp.log(jnp.clip(jnp.asarray(p_s, dtype=T.dtype), 1.0, None))[..., None]
    dlnps_dx, dlnps_dy = grad_fn(ln_ps, grid)
    T_v = virtual_temperature(T, q_v)
    idx = tuple(int(c) for c in col_index)
    # f_c host-side (jit-safe): lat_rad is static per the extractor contract, so
    # use the pure-Python Coriolis f = 2Ω sinφ.  The shared jnp-based
    # coriolis_f_c() calls float() on a jnp result, which is a tracer (and thus
    # errors) when the extractor itself is traced under jax.jit.
    f_c = 2.0 * constants.Omega * math.sin(float(lat_rad))
    return geostrophic_wind_from_gradients(
        dphi_dx[idx], dphi_dy[idx], dlnps_dx[idx], dlnps_dy[idx], T_v[idx], f_c
    )


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

    u_geo, v_geo = _geostrophic_wind_column(
        T=T, q_v=q_v, p_s=p_s, grid=grid, sigma_coord=sigma_coord,
        grad_fn=_gradient_latlon_3d, lat_rad=lat_rad, col_index=col_index,
    )

    i, j = int(col_index[0]), int(col_index[1])
    return ColumnLargeScaleState(
        lat_rad=lat_rad,
        T=T[i, j, :],
        p_full=p_full[i, j, :],
        q_v=q_v[i, j, :],
        omega=omega_3d[i, j, :],
        theta_adv=theta_adv_3d[i, j, :],
        qv_adv=qv_adv_3d[i, j, :],
        u_geo=u_geo,
        v_geo=v_geo,
    )


def _gradient_cubed_3d(field_3d: jax.Array, grid: Any) -> tuple[jax.Array, jax.Array]:
    """Per-level ``(∂/∂x, ∂/∂y)`` via the **4D-native** cubed-sphere operators.

    Uses :func:`legoesm.core.operators_3d.gradient_x_3d` / ``gradient_y_3d``,
    which take ``(6, n, n, nlev)`` directly and do ONE 4D halo exchange for all
    levels (``pad_halo_4d``).  This is mandatory (CLAUDE.md): never
    ``vmap(pad_halo)`` — a per-level ``vmap`` around the 2D operators would issue
    ``nlev`` halo messages and run the custom-VJP ``sendrecv`` *inside* ``vmap``,
    which is AD-/MPI-incorrect (mpi4jax batch-axis).  The edges/corners see the
    correct cross-face neighbourhood via the 4D halo.
    """
    from legoesm.core.operators_3d import gradient_x_3d, gradient_y_3d

    field_3d = jnp.asarray(field_3d)
    return gradient_x_3d(field_3d, grid), gradient_y_3d(field_3d, grid)


def _divergence_cubed_3d(u3d: jax.Array, v3d: jax.Array, grid: Any) -> jax.Array:
    """Horizontal divergence via the **4D-native** cubed-sphere operator.

    Uses :func:`legoesm.core.operators_3d.divergence_3d` (one 4D vector halo
    exchange with component rotation at face boundaries — not a per-level
    ``vmap``).  ``u3d``/``v3d`` are the grid-aligned velocity components the
    dycore stores.
    """
    from legoesm.core.operators_3d import divergence_3d

    return divergence_3d(jnp.asarray(u3d), jnp.asarray(v3d), grid)


def extract_column_forcing_cubed_sphere(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma_coord: Any,
    lat_rad: float,
    col_index: tuple[int, int, int],
) -> ColumnLargeScaleState:
    """Build a column's :class:`ColumnLargeScaleState` from a cubed-sphere state.

    Mirror of :func:`extract_column_forcing_latlon` for the model's native
    cubed-sphere grid: ``T``/``q_v``/``u``/``v`` are ``(6, n, n, nlev)``
    (surface-last), ``p_s`` ``(6, n, n)``; ``grid`` is the
    :class:`~legoesm.grids.cubed_sphere.CubedSphereGrid`.  ``col_index =
    (face, i, j)`` (static Python ints from the manifest) selects the flagged
    column; ``lat_rad`` is its latitude (``grid.grid_lat[face, i, j]``).  ``ω`` and
    the advective tendencies are diagnosed on the FULL grid (so the metric-correct
    cubed-sphere stencils see the cross-face neighbourhood) and the single column
    is then gathered.

    Reuses the grid-agnostic continuity chain (:func:`omega_from_divergence`) and
    advection (:func:`advective_tendency`) unchanged — only the horizontal
    operators are cubed-sphere.

    Caveat (metric, honest): the advection ``u ∂φ/∂x + v ∂φ/∂y`` neglects the
    small grid non-orthogonality cross-term on the equiangular cube (the grid
    axes are not exactly orthogonal away from face centres).  This is the same
    leading-order A-grid advection the collocated dycore uses and is adequate for
    a large-scale *forcing* tendency; it is NOT a conservation-critical flux.
    """
    T = jnp.asarray(T)
    q_v = jnp.asarray(q_v, dtype=T.dtype)
    u = jnp.asarray(u, dtype=T.dtype)
    v = jnp.asarray(v, dtype=T.dtype)
    p_s = jnp.asarray(p_s, dtype=T.dtype)
    sigma_full = jnp.asarray(sigma_coord.sigma_full, dtype=T.dtype)
    p_full = p_s[..., None] * sigma_full

    theta = T / exner_function(p_full)
    th_x, th_y = _gradient_cubed_3d(theta, grid)
    q_x, q_y = _gradient_cubed_3d(q_v, grid)
    theta_adv_3d = advective_tendency(u, v, th_x, th_y)
    qv_adv_3d = advective_tendency(u, v, q_x, q_y)

    div_3d = _divergence_cubed_3d(u, v, grid)
    omega_3d = omega_from_divergence(div_3d, p_s, sigma_coord)

    # Geostrophic forcing is NOT applied on the cubed sphere (u_geo/v_geo=None, so
    # no geostrophic reference wind is supplied — the column LES's plane Coriolis
    # falls back to f×V, exactly as for equatorial columns).  Geostrophic balance
    # is geographic
    # (east/north), but the cubed-sphere gradient operators return GRID-AXIS
    # derivatives on a NON-ORTHOGONAL grid; the metric-correct east/north↔grid
    # rotation (grid.cos_angle/sin_angle) is a follow-up that must be visually
    # verified for cube-edge/corner artifacts (CLAUDE.md) before it ships.  The
    # lat-lon path (ERA5-native, the primary comparison grid) is fully geostrophic.
    u_geo, v_geo = None, None

    f, i, j = int(col_index[0]), int(col_index[1]), int(col_index[2])
    return ColumnLargeScaleState(
        lat_rad=lat_rad,
        T=T[f, i, j, :],
        p_full=p_full[f, i, j, :],
        q_v=q_v[f, i, j, :],
        omega=omega_3d[f, i, j, :],
        theta_adv=theta_adv_3d[f, i, j, :],
        qv_adv=qv_adv_3d[f, i, j, :],
        u_geo=u_geo,
        v_geo=v_geo,
    )


def extract_column_forcing(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma_coord: Any,
    lat_rad: float,
    col_index: tuple[int, ...],
) -> ColumnLargeScaleState:
    """Dispatch column-forcing extraction on the grid type (raises on unknown).

    Routes a flagged column to the lat-lon (``col_index = (i_lat, i_lon)``) or
    cubed-sphere (``col_index = (face, i, j)``) extractor, validating the
    ``col_index`` arity for the grid so a mismatched index fails LOUDLY instead of
    mis-gathering.  An unsupported grid (Gaussian/Voronoi — follow-ups) raises
    :class:`ValueError` (dispatch hardening, CLAUDE.md) rather than silently
    falling through to a wrong path.
    """
    from legoesm.grids.cubed_sphere import CubedSphereGrid
    from legoesm.grids.latlon import LatLonGrid

    kwargs = dict(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid,
        sigma_coord=sigma_coord, lat_rad=lat_rad,
    )
    if isinstance(grid, LatLonGrid):
        if len(col_index) != 2:
            raise ValueError(
                f"lat-lon col_index must be (i_lat, i_lon); got {col_index!r}."
            )
        return extract_column_forcing_latlon(col_index=col_index, **kwargs)
    if isinstance(grid, CubedSphereGrid):
        if len(col_index) != 3:
            raise ValueError(
                f"cubed-sphere col_index must be (face, i, j); got {col_index!r}."
            )
        return extract_column_forcing_cubed_sphere(col_index=col_index, **kwargs)
    raise ValueError(
        f"extract_column_forcing: unsupported grid type {type(grid).__name__}; "
        f"only LatLonGrid and CubedSphereGrid are supported "
        f"(Gaussian/Voronoi are follow-ups)."
    )
