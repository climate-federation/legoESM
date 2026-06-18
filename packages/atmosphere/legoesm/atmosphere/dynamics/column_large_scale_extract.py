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
grid needs cubed-sphere extraction (``extract_column_forcing_cubed_sphere``); the
unstructured MPAS/Voronoi family has ``extract_column_forcing_voronoi`` (edge-
normal velocity + Perot reconstruction), and the Gaussian/spectral grid has
``extract_column_forcing_gaussian`` (pseudospectral divergence + flux-form
advection, reusing the dycore's SH operators).  All share the grid-agnostic
continuity chain (:func:`omega_from_divergence`); only the horizontal operators
differ (FD for lat-lon/cubed, TRiSK for Voronoi, spectral for Gaussian).
:func:`extract_column_forcing` dispatches on the grid type (raises on a truly
unsupported grid).
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


def _gradient_voronoi_3d(field_cell_3d: jax.Array, mesh: Any) -> tuple[jax.Array, jax.Array]:
    """Per-level cell-centered ``(∂/∂x_east, ∂/∂y_north)`` on an MPAS/Voronoi mesh.

    Two steps, both 4D-/3D-native (NEVER ``vmap(pad_halo)``; these local TRiSK
    operators carry no halo exchange):

    1. :func:`legoesm.core.operators_voronoi.gradient_edge_3d` gives the
       EDGE-NORMAL gradient ``∂φ/∂n = (φ[c2] − φ[c1])/dcEdge`` at every edge.
    2. :func:`legoesm.grids.voronoi.reconstruct_cell_velocity` (Perot 2000) maps
       those edge-normal components to the cell-centered VECTOR in geographic
       (east, north) — exactly the reconstruction MPAS uses for cell winds, here
       applied to ``∇φ`` (a vector field whose edge-normal component IS the
       gradient_edge output, so the reconstruction yields the cell ``∇φ``).

    The result is in the geographic frame (so it co-locates with the Perot-
    reconstructed cell wind, frame-consistent for the advective dot product).
    """
    from legoesm.core.operators_voronoi import gradient_edge_3d
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    grad_edge = gradient_edge_3d(jnp.asarray(field_cell_3d), mesh)  # (nEdges, nlev)
    gx, gy = reconstruct_cell_velocity(grad_edge, mesh)             # (nCells, nlev)
    return gx, gy


def _divergence_voronoi_3d(u_edge_3d: jax.Array, mesh: Any) -> jax.Array:
    """Cell-centered horizontal divergence of the EDGE-NORMAL velocity on a Voronoi
    mesh (:func:`legoesm.core.operators_voronoi.divergence_cell_3d`): the
    sign-weighted edge-flux sum over ``edgesOnCell`` / ``areaCell``.  Same units +
    sign convention (positive = outflow) as the lat-lon / cubed divergence the
    continuity chain (:func:`omega_from_divergence`) expects."""
    from legoesm.core.operators_voronoi import divergence_cell_3d

    return divergence_cell_3d(jnp.asarray(u_edge_3d), mesh)


def extract_column_forcing_voronoi(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u_edge: jax.Array,
    p_s: jax.Array,
    mesh: Any,
    sigma_coord: Any,
    lat_rad: float,
    col_index: tuple[int],
) -> ColumnLargeScaleState:
    """Build a column's :class:`ColumnLargeScaleState` from an MPAS/Voronoi state.

    The Voronoi sibling of :func:`extract_column_forcing_latlon` /
    ``_cubed_sphere`` for the model's unstructured grid family.  ``T``/``q_v`` are
    CELL-centered ``(nCells, nlev)`` (surface-last), ``p_s`` ``(nCells,)``;
    ``u_edge`` is the EDGE-NORMAL velocity ``(nEdges, nlev)`` (MPAS stores wind on
    edges, not cells — there is no cell ``v``); ``mesh`` is the
    :class:`~legoesm.grids.voronoi.VoronoiMesh`.  ``col_index = (cell,)`` (a static
    Python int from the manifest) selects the flagged cell; ``lat_rad`` is its
    latitude (``mesh.latCell[cell]``).

    The horizontal gradient + divergence use the local TRiSK operators on the FULL
    mesh, with the Perot reconstruction giving the cell-centered velocity AND
    gradient in the SAME geographic (east, north) frame; the grid-AGNOSTIC
    continuity (:func:`omega_from_divergence`) and advection
    (:func:`advective_tendency`) chains are reused unchanged.  The single cell
    column is then gathered.

    Caveat (metric, honest — same as the cubed extractor): the advection
    ``u ∂φ/∂x + v ∂φ/∂y`` with BOTH velocity and gradient Perot-reconstructed at
    the cell is leading-order A-grid; it is adequate for a large-scale *forcing*
    tendency and is NOT a conservation-critical flux (do not reuse it as the MPAS
    tracer transport scheme).

    Geostrophic reference wind is ENABLED here (``u_geo``/``v_geo`` from
    :func:`_geostrophic_wind_column`) — UNLIKE the cubed sphere, the Perot
    cell-gradient is already in the geographic (east, north) frame geostrophic
    balance requires, so the helper's documented precondition holds.  Within
    :data:`_MIN_GEOSTROPHIC_LAT_DEG` of the equator (f → 0 ill-posed) ``(None,
    None)`` is returned → the column LES falls back to ``f×V`` there, identical to
    the previous unconditional ``None`` (so equatorial columns are never changed).
    OUTSIDE the cutoff the geostrophic wind REPLACES the LES's ``f×V`` default
    (which was itself an approximation — the model wind as a geostrophic proxy):
    the pressure-gradient geostrophic reference is the physically-correct
    large-scale forcing, but it is NOT a guaranteed improvement for every column —
    on a coarse mesh the Perot ``Φ = g·z`` gradient is leading-order, so a specific
    column's reference may be noisier than ``f×V`` (a known trade-off the loop's
    iter-43 acceptance gate / iter-44 line search / bias monitor safeguard, the
    same safeguards as the C_K-vs-GCM-wp2 offset).  Accuracy is leading-order on a
    coarse mesh (the Perot gradient of ``Φ = g·z`` reconstructs the geographic
    geopotential gradient to ~the mesh resolution); it is a *forcing* reference,
    validated for a uniform state (→ zero geostrophic wind, exact) + a meridional
    gradient (→ sign-correct zonal jet) + finiteness in
    ``test_column_large_scale_extract_voronoi.py``.  Assumptions (per Codex): a
    valid mesh (positive ``areaCell`` / edge lengths — the Perot reconstruction
    divides by them, no degeneracy guard); the ``ln p_s`` ``clip(p_s, 1.0)`` never
    activates for realistic surface pressures (~1e5 Pa) so it cannot flatten the
    forcing; the geostrophic gradient path uses ``cellsOnEdge`` / ``edgesOnCell`` /
    ``angleEdge`` (NOT ``edgeSignOnCell``), so the edge-sign guard below — for the
    divergence/ω — already covers the only edge-sign dependency.

    **Scope (single-rank / full mesh):** the TRiSK operators are local gathers
    over the mesh connectivity with NO halo exchange — correct for a complete
    serial/global mesh (the worst-column spin-off use case), exactly like the
    lat-lon / cubed extractors.  Distributed-MPAS (rank-local mesh + ghost cells)
    handoff is a follow-up.  ``mesh.edgeSignOnCell`` MUST be populated (it is by
    :func:`~legoesm.grids.voronoi.create_voronoi_mesh`); an all-zero sign array
    (a mesh loaded without it) would silently ZERO the divergence + ``ω``, so it
    is rejected LOUDLY here.
    """
    import numpy as np

    # Guard the silent-zero-divergence failure mode (Codex #d): a mesh whose
    # edgeSignOnCell was never populated yields div ≡ 0 ⇒ ω ≡ 0.  Mesh connectivity
    # is static, so this host-side check is jit-safe (mesh arrays are concrete).
    if not bool(np.any(np.asarray(mesh.edgeSignOnCell) != 0)):
        raise ValueError(
            "extract_column_forcing_voronoi: mesh.edgeSignOnCell is all zero — the "
            "divergence (hence ω) would be silently zero; this mesh lacks edge-sign "
            "connectivity (e.g. a load_mpas_mesh mesh that did not populate it)."
        )

    T = jnp.asarray(T)
    q_v = jnp.asarray(q_v, dtype=T.dtype)
    u_edge = jnp.asarray(u_edge, dtype=T.dtype)
    p_s = jnp.asarray(p_s, dtype=T.dtype)
    sigma_full = jnp.asarray(sigma_coord.sigma_full, dtype=T.dtype)
    p_full = p_s[..., None] * sigma_full

    theta = T / exner_function(p_full)
    th_x, th_y = _gradient_voronoi_3d(theta, mesh)
    q_x, q_y = _gradient_voronoi_3d(q_v, mesh)

    from legoesm.grids.voronoi import reconstruct_cell_velocity
    u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)  # cell (east, north)
    theta_adv_3d = advective_tendency(u_cell, v_cell, th_x, th_y)
    qv_adv_3d = advective_tendency(u_cell, v_cell, q_x, q_y)

    div_3d = _divergence_voronoi_3d(u_edge, mesh)
    omega_3d = omega_from_divergence(div_3d, p_s, sigma_coord)

    # Geostrophic reference wind — ENABLED on Voronoi (unlike cubed-sphere): the
    # Perot cell-gradient (_gradient_voronoi_3d) is already in the GEOGRAPHIC
    # (east, north) frame geostrophic_wind_from_gradients requires (iter-73
    # orientation regression). The geostrophic gradient path uses gradient_edge_3d
    # (cellsOnEdge) + the Perot reconstruction (edgesOnCell/angleEdge), NOT
    # edgeSignOnCell, so the edge-sign guard above (for ω) already covers it.
    # Within _MIN_GEOSTROPHIC_LAT_DEG of the equator → (None, None) (f×V fallback,
    # identical to the old behavior there); OUTSIDE, it REPLACES the f×V default
    # with the pressure-gradient reference (a known trade-off, not a guarantee —
    # see the docstring; the loop's gate/line-search/bias-monitor are the safeguards).
    u_geo, v_geo = _geostrophic_wind_column(
        T=T, q_v=q_v, p_s=p_s, grid=mesh, sigma_coord=sigma_coord,
        grad_fn=_gradient_voronoi_3d, lat_rad=lat_rad, col_index=col_index,
    )

    c = int(col_index[0])
    return ColumnLargeScaleState(
        lat_rad=lat_rad,
        T=T[c, :],
        p_full=p_full[c, :],
        q_v=q_v[c, :],
        omega=omega_3d[c, :],
        theta_adv=theta_adv_3d[c, :],
        qv_adv=qv_adv_3d[c, :],
        u_geo=u_geo,
        v_geo=v_geo,
    )


def _spectral_divergence_3d(u3d: jax.Array, v3d: jax.Array, grid: Any) -> jax.Array:
    """Horizontal divergence ``∇·v`` on the Gaussian/spectral grid via the tested
    spherical-harmonic forward transform + synthesis.

    Reuses :func:`legoesm.grids.gaussian.vordiv_from_uv_3d` (the Hack & Jakob /
    Bourke pole-safe ``oc2``/``dmu`` operators — the SAME spectral divergence the
    dycore uses) on the PHYSICAL grid winds (NOT pre-cos-φ-weighted — the helper
    multiplies by ``cos φ`` internally; passing ``u·cos φ`` would double-apply it),
    then synthesizes the ``div_hat`` back to the ``(n_lat, n_lon, nlev)`` grid.
    Also valid for the divergence of an ARBITRARY vector field ``φ·V`` (the helper
    documents a generic ``F = (F_x, F_y)``), which the flux-form advection uses.
    Float64 / spectral-precision required (the Gaussian path is x64-only).
    """
    from legoesm.grids.gaussian import sh_synthesis_3d, vordiv_from_uv_3d

    _vor_hat, div_hat = vordiv_from_uv_3d(grid, jnp.asarray(u3d), jnp.asarray(v3d))
    return sh_synthesis_3d(grid, div_hat)


def _gradient_gaussian_3d(field_3d: jax.Array, grid: Any) -> tuple[jax.Array, jax.Array]:
    """Geographic east/north gradient ``(∂f/∂x, ∂f/∂y)`` of a grid-space scalar on
    the Gaussian grid — analyze to the SH spectrum, then the SHARED spectral gradient
    :func:`legoesm.grids.gaussian.spectral_gradient_3d` (the SAME operator the
    spectral NH dycore uses; promoted iter 90 — no parallel gradient).  Returns the
    geographic axes (``∂f/∂x = (1/(a cos φ))∂f/∂λ`` east, ``∂f/∂y = (1/a)∂f/∂φ``
    north) that :func:`_geostrophic_wind_column` requires.  Float64 / spectral
    precision."""
    from legoesm.grids.gaussian import sh_analysis_3d, spectral_gradient_3d

    return spectral_gradient_3d(grid, sh_analysis_3d(grid, jnp.asarray(field_3d)))


def extract_column_forcing_gaussian(
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
    """Build a column's :class:`ColumnLargeScaleState` from a Gaussian/spectral state.

    The 4th grid family's extractor (lat-lon / cubed-sphere / MPAS being the
    others).  ``T``/``q_v``/``u``/``v`` are ``(n_lat, n_lon, nlev)`` (surface-last,
    PHYSICAL winds — NOT pre-cos-φ-weighted), ``p_s`` ``(n_lat, n_lon)``; ``grid``
    is the :class:`~legoesm.grids.gaussian.GaussianGrid`; ``col_index = (i_lat,
    i_lon)`` (static Python ints from the manifest).  ``lat_rad`` is the column's
    latitude.

    Unlike the FD lat-lon / cubed-sphere extractors, the divergence + advection
    are computed PSEUDOSPECTRALLY: ``div = ∇·v`` via the spectral
    :func:`_spectral_divergence_3d`, and the advective tendency in FLUX form
    ``−V·∇φ = −∇·(φV) + φ·∇·V`` (algebraically exact; the dycore's own temperature
    tendency ``−∇·(Tv) + T·∇·v``).  This is therefore the DYCORE-CONSISTENT
    tendency — what the spectral model actually integrated — numerically DISTINCT
    from (not equivalent to) the FD ``advective_tendency`` the other extractors
    estimate.  The grid-product ``φ·u`` is formed before the transform, so the
    tendency carries the dycore's pseudospectral aliasing (no separate dealiasing
    mask is applied — this is forcing-diagnostic consistency, not alias-free
    exactness).  The grid-AGNOSTIC continuity (:func:`omega_from_divergence`) is
    reused unchanged.

    **Float64 required:** the spectral transforms need ``jax_enable_x64`` (the
    Gaussian grid's machinery is float64/complex128) — the same constraint as the
    spectral dycore.

    Geostrophic forcing IS supplied (iter 90): :func:`_geostrophic_wind_column` with
    the spectral scalar gradient :func:`_gradient_gaussian_3d` (geographic east/north
    — the precondition the shared helper requires, UNLIKE the cubed-sphere grid-axis
    operators).  Equatorial columns (within :data:`_MIN_GEOSTROPHIC_LAT_DEG`) still
    return ``(None, None)`` ⇒ the column LES falls back to ``f×V`` there.
    """
    # x64 GATE (Codex): the SH transforms need float64; fail LOUDLY for a float32
    # input (whether x64 is off, or float32 was passed under x64) rather than
    # silently promoting parts of the calculation.
    if jnp.asarray(T).dtype != jnp.float64:
        raise TypeError(
            "extract_column_forcing_gaussian requires float64 inputs (set "
            f"JAX_ENABLE_X64=1); got T dtype {jnp.asarray(T).dtype} — the spectral "
            "transforms are float64/complex128."
        )
    T = jnp.asarray(T)
    q_v = jnp.asarray(q_v, dtype=T.dtype)
    u = jnp.asarray(u, dtype=T.dtype)
    v = jnp.asarray(v, dtype=T.dtype)
    p_s = jnp.asarray(p_s, dtype=T.dtype)
    sigma_full = jnp.asarray(sigma_coord.sigma_full, dtype=T.dtype)
    p_full = p_s[..., None] * sigma_full

    theta = T / exner_function(p_full)
    div_3d = _spectral_divergence_3d(u, v, grid)               # ∇·v
    # Advective tendency −V·∇φ = −∇·(φV) + φ·∇·V (flux form; exact cancellation for
    # a uniform φ ⇒ zero advection to spectral precision).
    theta_adv_3d = -_spectral_divergence_3d(theta * u, theta * v, grid) + theta * div_3d
    qv_adv_3d = -_spectral_divergence_3d(q_v * u, q_v * v, grid) + q_v * div_3d

    omega_3d = omega_from_divergence(div_3d, p_s, sigma_coord)

    # Geostrophic reference wind from the spectral Φ gradient (geographic east/north),
    # or (None, None) within the equatorial cutoff → the LES plane Coriolis uses f×V.
    u_geo, v_geo = _geostrophic_wind_column(
        T=T, q_v=q_v, p_s=p_s, grid=grid, sigma_coord=sigma_coord,
        grad_fn=_gradient_gaussian_3d, lat_rad=lat_rad, col_index=col_index,
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

    Routes a flagged column to the lat-lon (``col_index = (i_lat, i_lon)``),
    cubed-sphere (``col_index = (face, i, j)``), MPAS/Voronoi
    (``col_index = (cell,)``), or Gaussian/spectral (``col_index = (i_lat,
    i_lon)``) extractor, validating the ``col_index`` arity for the grid so a
    mismatched index fails LOUDLY instead of mis-gathering.  A truly unsupported
    grid raises :class:`ValueError` (dispatch hardening, CLAUDE.md) rather than
    silently falling through.

    **Voronoi velocity convention:** MPAS stores the EDGE-NORMAL velocity (there is
    no cell ``v``), so for a :class:`~legoesm.grids.voronoi.VoronoiMesh` the ``u``
    argument carries the edge-normal velocity ``(nEdges, nlev)`` and ``v`` MUST be
    ``None`` (a non-``None`` ``v`` is REJECTED rather than silently ignored — a
    caller passing cell winds for MPAS would otherwise mis-extract).
    """
    from legoesm.grids.cubed_sphere import CubedSphereGrid
    from legoesm.grids.gaussian import GaussianGrid
    from legoesm.grids.latlon import LatLonGrid
    from legoesm.grids.voronoi import VoronoiMesh

    if isinstance(grid, VoronoiMesh):
        if len(col_index) != 1:
            raise ValueError(
                f"Voronoi col_index must be (cell,); got {col_index!r}."
            )
        if v is not None:
            raise ValueError(
                "Voronoi extraction takes the EDGE-NORMAL velocity in `u` "
                "(nEdges, nlev) and requires `v=None`; got a non-None v "
                "(MPAS has no cell-centred v — pass the edge velocity as u)."
            )
        n_edges = int(grid.nEdges)
        if jnp.asarray(u).shape[0] != n_edges:
            raise ValueError(
                f"Voronoi `u` must be edge-normal velocity with leading dim "
                f"nEdges={n_edges}; got shape {tuple(jnp.asarray(u).shape)}."
            )
        n_cells = int(grid.nCells)
        for name, arr in (("T", T), ("q_v", q_v), ("p_s", p_s)):
            if jnp.asarray(arr).shape[0] != n_cells:
                raise ValueError(
                    f"Voronoi `{name}` must have leading dim nCells={n_cells}; "
                    f"got shape {tuple(jnp.asarray(arr).shape)}."
                )
        return extract_column_forcing_voronoi(
            T=T, q_v=q_v, u_edge=u, p_s=p_s, mesh=grid,
            sigma_coord=sigma_coord, lat_rad=lat_rad, col_index=col_index,
        )

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
    if isinstance(grid, GaussianGrid):
        if len(col_index) != 2:
            raise ValueError(
                f"Gaussian col_index must be (i_lat, i_lon); got {col_index!r}."
            )
        n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
        t_shape = tuple(jnp.asarray(T).shape)
        if t_shape[:2] != (n_lat, n_lon):
            raise ValueError(
                f"Gaussian fields must be (n_lat={n_lat}, n_lon={n_lon}, nlev); got "
                f"T leading dims {t_shape[:2]}."
            )
        for name, arr in (("q_v", q_v), ("u", u), ("v", v)):
            if tuple(jnp.asarray(arr).shape) != t_shape:
                raise ValueError(
                    f"Gaussian `{name}` must match T's shape {t_shape}; got "
                    f"{tuple(jnp.asarray(arr).shape)}."
                )
        if tuple(jnp.asarray(p_s).shape) != (n_lat, n_lon):
            raise ValueError(
                f"Gaussian `p_s` must be (n_lat={n_lat}, n_lon={n_lon}); got "
                f"{tuple(jnp.asarray(p_s).shape)}."
            )
        return extract_column_forcing_gaussian(col_index=col_index, **kwargs)
    raise ValueError(
        f"extract_column_forcing: unsupported grid type {type(grid).__name__}; "
        f"only LatLonGrid, CubedSphereGrid, VoronoiMesh (MPAS) and GaussianGrid "
        f"(spectral) are supported."
    )
