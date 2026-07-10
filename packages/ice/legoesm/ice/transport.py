"""Sea ice tracer advection.

Advects ice tracers (thickness, concentration, temperature) by the ice
velocity field using **conservative monotone PPM** flux-form transport
on the cubed sphere (Colella–Woodward piecewise-parabolic with the
Colella–Woodward monotonicity limiter).  Volume ``h*a`` and area ``a``
remain in [0, ∞) and [0, 1] respectively without post-step clipping,
and enthalpy ``T*h*a`` is advected in flux form (conserving total
enthalpy to machine precision); the recovered ``T = enth/vol`` is clamped
only to a wide non-binding safety range so the energy-conserving transport
is never undone by a hard physical clip (see finding #2).

The previous centered scheme was non-monotone — it produced negative
thickness, concentration > 1, and out-of-range temperatures, which the
post-step clamps masked at the cost of conservation.  Codex
adversarial review finding #10.

All functions are JAX-compatible (differentiable, JIT-friendly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.core.operators_3d import fv_flux_divergence_3d
from legoesm.core.operators_fv import fv_flux_divergence
from legoesm.core.operators_fv_latlon import fv_flux_divergence_latlon
from legoesm.core.operators_voronoi import (
    divergence_cell,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.voronoi import VoronoiMesh


# Numerical-stability bounds for the temperature/enthalpy channel.  These
# MIRROR ``SeaIceConfig.T_ice_min`` / ``SeaIceConfig.T_melt_surface`` (both
# documented in config.py as numerics-only bounds, NOT physical conservation
# limits) so the signature defaults reference a Name, not a bare literal.
_T_ICE_MIN_DEFAULT: float = 180.0   # == SeaIceConfig.T_ice_min [K]
# Concentration floor used when recovering an intensive tracer ratio
# (h = vol/conc).  Floors the DENOMINATOR so the h = vol/conc spike is bounded.
_CONC_FLOOR: float = 1e-12          # [dimensionless area fraction]
# Volume floor [m of ice per m^2 of grid cell] deciding whether a cell still
# holds ice for the recovery branch.  Distinct from ``_CONC_FLOOR`` (a
# concentration) — they share the 1e-12 numeric value (the same threshold the
# thermo/lead-freeze recovery in sea_ice.py uses, ``V_after > 1e-12``) but
# carry DIFFERENT units, so name them separately.  A cell with 0 < vol <= this
# carries <1e-9 kg/m^2 of ice and is treated as ice-free.
_VOL_FLOOR: float = 1e-12           # [m of ice per m^2 grid cell]
# Wide, NON-BINDING numerical-safety clamp for the recovered temperature.
# Rationale (identical to the salinity channel in sea_ice.py): a monotone-PPM
# ratio can overshoot the donor min/max by a hair at CFL<=1, and under CFL>1
# (limiter monotonicity not guaranteed) it can overshoot grossly.  A HARD clamp
# at the physical melt point deletes/creates enthalpy with NO accounting (the
# flux-form enthalpy transport already conserves total energy to machine
# precision), so we clamp only to a wide range that catches NaN/absurd values
# for downstream stability while leaving the physical melt point to the
# conservative thermodynamics.  At the supported CFL<=1 (enforce with
# SeaIceConfig.transport_subcycles>1) this clamp is non-binding and energy is
# conserved exactly.
_T_SAFETY_MARGIN_K: float = 50.0    # widen [T_ice_min, T_max] by this on each side


def _is_latlon_grid(grid) -> bool:
    return isinstance(grid, LatLonGrid)


def _is_voronoi_mesh(grid) -> bool:
    return isinstance(grid, VoronoiMesh)


def _cell_velocity_to_edge_normal(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Project cell-centered (u_east, v_north) onto edge-normal direction.

    Cell-to-edge interpolation: average the two cells adjacent to
    each edge (boundary edges get the single valid cell repeated).
    Then dot with the edge-normal (cos(angleEdge), sin(angleEdge)).

    Parameters
    ----------
    u_cell, v_cell : array ``(nCells,)``
        Cell-centered east-north velocity components [m/s].
    mesh : VoronoiMesh

    Returns
    -------
    u_edge_normal : array ``(nEdges,)``
        Normal component of velocity at each edge [m/s].
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    # Boundary edges have c2 = -1; clamp + use c1 only via mask.
    c1_safe = jnp.maximum(c1, 0)
    c2_safe = jnp.maximum(c2, 0)
    interior = c2 >= 0
    u_e_x = jnp.where(
        interior,
        0.5 * (u_cell[c1_safe] + u_cell[c2_safe]),
        u_cell[c1_safe],
    )
    v_e_y = jnp.where(
        interior,
        0.5 * (v_cell[c1_safe] + v_cell[c2_safe]),
        v_cell[c1_safe],
    )
    cos_a = jnp.cos(mesh.angleEdge)
    sin_a = jnp.sin(mesh.angleEdge)
    return u_e_x * cos_a + v_e_y * sin_a


def fv_flux_divergence_voronoi(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Cell-centered flux divergence on a Voronoi mesh.

    Returns ``dq/dt = -div(q · v)`` so the caller integrates as
    ``q_new = q + dt · tendency`` — matching the cubed-sphere and
    lat-lon dispatchers.

    Edge scalar reconstruction is computed locally here (not via
    the shared ``thickness_flux``) so boundary edges where
    ``cellsOnEdge[1] == −1`` correctly fall back to the one valid
    cell.  No-flux closure is also applied at boundary edges
    (``u_edge_normal = 0``) to prevent spurious mass leaks on
    regional meshes.

    Parameters
    ----------
    q : array ``(nCells,)``
        Conservative scalar at cell centers (e.g. ice volume per cell).
    u, v : array ``(nCells,)``
        Cell-centered east-north velocity components [m/s].
    mesh : VoronoiMesh

    Returns
    -------
    tendency : array ``(nCells,)``
        Time tendency of ``q`` from flux-form transport.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    interior_edge = c2 >= 0
    c1_safe = jnp.maximum(c1, 0)
    c2_safe = jnp.maximum(c2, 0)

    u_edge_normal = _cell_velocity_to_edge_normal(u, v, mesh)
    # No-flux closure at boundary edges.
    u_edge_normal = jnp.where(interior_edge, u_edge_normal, 0.0)

    # Boundary-safe FIRST-ORDER UPWIND edge reconstruction of the transported
    # scalar.  The MPAS edge normal points c1 -> c2, so a positive normal
    # velocity makes c1 the upstream donor.  Centered 0.5*(q1+q2) is dispersive
    # on unstructured meshes and drives q negative / conc>1, which the
    # downstream jnp.maximum(vol,0)/clip(conc,0,1) silently turn into a mass
    # SOURCE (audit finding #1); upwind is monotone and positivity-preserving.
    # ponytail: first-order upwind; add a slope-limited (van Leer) edge value if
    # the numerical diffusion is too strong for MPAS sea-ice production.
    q_upwind = jnp.where(u_edge_normal >= 0.0, q[c1_safe], q[c2_safe])
    q_edge = jnp.where(interior_edge, q_upwind, q[c1_safe])
    flux_edge = q_edge * u_edge_normal

    div = divergence_cell(flux_edge, mesh)
    return -div


def _ppm_tendency_2d(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """PPM flux-divergence tendency for a single-category scalar.

    Dispatches on grid type:
      * ``CubedSphereGrid``: ``fv_flux_divergence`` (shape ``(6, n, n)``).
      * ``LatLonGrid``: ``fv_flux_divergence_latlon`` (shape
        ``(n_lat, n_lon)``).

    Sign convention: ``q_new = q + dt · tendency``.
    """
    if _is_latlon_grid(grid):
        return fv_flux_divergence_latlon(q, u, v, grid, limiter=True)
    if _is_voronoi_mesh(grid):
        return fv_flux_divergence_voronoi(q, u, v, grid)
    if isinstance(grid, CubedSphereGrid):
        return fv_flux_divergence(q, u, v, grid, limiter=True)
    raise TypeError(
        f"unsupported grid {type(grid).__name__} for _ppm_tendency_2d"
    )


def _ppm_tendency_per_category(
    q_cat: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """PPM tendency for a multi-category scalar (trailing category axis).

    Cubed-sphere path reuses ``fv_flux_divergence_3d`` (vmaps over
    trailing axis).  Lat-lon path vmaps ``fv_flux_divergence_latlon``
    over the trailing category axis.  MPAS Voronoi path vmaps
    ``fv_flux_divergence_voronoi`` similarly.
    """
    if _is_latlon_grid(grid):
        def _kernel(qk):
            return fv_flux_divergence_latlon(qk, u, v, grid, limiter=True)
        return jax.vmap(_kernel, in_axes=-1, out_axes=-1)(q_cat)
    if _is_voronoi_mesh(grid):
        def _kernel_v(qk):
            return fv_flux_divergence_voronoi(qk, u, v, grid)
        return jax.vmap(_kernel_v, in_axes=-1, out_axes=-1)(q_cat)
    if not isinstance(grid, CubedSphereGrid):
        raise TypeError(
            f"unsupported grid {type(grid).__name__} for "
            f"_ppm_tendency_per_category"
        )
    u_3d = jnp.broadcast_to(u[..., None], q_cat.shape)
    v_3d = jnp.broadcast_to(v[..., None], q_cat.shape)
    return fv_flux_divergence_3d(q_cat, u_3d, v_3d, grid, limiter=True)


def _ppm_one_substep(
    vol: jnp.ndarray,
    conc: jnp.ndarray,
    enth: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid,
    dt_sub: float,
    is_multicat: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One PPM substep of the conservation-form transport."""
    tendency = _ppm_tendency_per_category if is_multicat else _ppm_tendency_2d
    vol_new = vol + dt_sub * tendency(vol, u_ice, v_ice, grid)
    conc_new = conc + dt_sub * tendency(conc, u_ice, v_ice, grid)
    enth_new = enth + dt_sub * tendency(enth, u_ice, v_ice, grid)
    # PPM-with-limiter is monotone within its own CFL bound; defensive
    # clips guard against floating-point round-off only.
    return (
        jnp.maximum(vol_new, 0.0),
        jnp.clip(conc_new, 0.0, 1.0),
        enth_new,
    )


def advect_ice_tracers(
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    T_ice: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid,
    dt: float,
    T_ice_min: float = _T_ICE_MIN_DEFAULT,
    T_freeze_ocean: float = constants.T_freeze_ocean,
    T_max: float = constants.T_freeze,
    n_subcycles: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advect ice tracers by the ice velocity field.

    Uses conservative monotone PPM (Colella–Woodward with limiter)
    flux-form transport from ``core.operators_fv``.  Volume ``h*a``,
    concentration ``a``, and enthalpy ``T*h*a`` are each advected as
    independent conserved scalars.  The PPM scheme is monotone *within*
    its CFL-1 bound — if ``|u·dt/dx| > 1`` the limiter cannot guarantee
    monotonicity.  ``n_subcycles`` partitions the step into
    ``n_subcycles`` substeps of length ``dt/n_subcycles`` (executed
    with ``lax.scan``) so callers in storm conditions can force CFL
    safety by setting ``SeaIceConfig.transport_subcycles > 1``.

    Accepts both single-category 2D inputs ``(6, n, n)`` and
    multi-category 3D inputs ``(6, n, n, n_cat)``.

    Parameters
    ----------
    h_ice, concentration, T_ice : arrays (6, n, n) or (6, n, n, n_cat)
        Ice tracer fields.
    u_ice, v_ice : arrays (6, n, n)
        Ice velocity [m/s] — same wind across all categories.
    grid : CubedSphereGrid
    dt : float
        Timestep [s].
    T_ice_min, T_max : float
        Numerical-stability bounds [K] for the recovered intensive ratio
        (temperature, or salinity when this channel transports salt).  They
        are widened by ``_T_SAFETY_MARGIN_K`` into a NON-BINDING safety clamp
        (see finding #2) — the physical melt point is enforced conservatively
        by the downstream thermodynamics, not deleted here.
    T_freeze_ocean : float
        Ice-free fill value [K] for cells with no ice volume.
    n_subcycles : int, default 1
        Number of PPM substeps inside the step.  Static (treated as a
        Python int, not a traced value), so changing it triggers a
        recompile.

    Returns
    -------
    h_new, conc_new, T_new : arrays
        Updated tracer fields with the same shape as the inputs.
    """
    n_subcycles = int(max(n_subcycles, 1))

    # Conservation-form auxiliary fields.
    vol = h_ice * concentration                  # m of ice per m² of grid
    enth = T_ice * vol                           # K · m
    conc = concentration

    # Multi-category detection: base ndim depends on grid layout.
    # MPAS Voronoi: ``(nCells,)`` (1D); lat-lon: ``(n_lat, n_lon)`` (2D);
    # cubed-sphere: ``(6, n, n)`` (3D).  Trailing category axis adds one
    # rank.
    if _is_voronoi_mesh(grid):
        base_ndim = 1
    elif _is_latlon_grid(grid):
        base_ndim = 2
    else:
        base_ndim = 3
    is_multicat = h_ice.ndim == (base_ndim + 1)
    dt_sub = dt / n_subcycles

    if n_subcycles == 1:
        vol_new, conc_new, enth_new = _ppm_one_substep(
            vol, conc, enth, u_ice, v_ice, grid, dt_sub, is_multicat,
        )
    else:
        def _body(carry, _):
            v, c, e = carry
            return _ppm_one_substep(
                v, c, e, u_ice, v_ice, grid, dt_sub, is_multicat,
            ), None
        (vol_new, conc_new, enth_new), _ = lax.scan(
            _body, (vol, conc, enth), xs=None, length=n_subcycles,
        )

    # --- Recover thickness from the CONSERVED volume (finding #3) ----------
    # Mass-conservation convention: the advected, conserved quantity is the
    # ice VOLUME ``vol = h*conc`` (and enthalpy ``enth = T*vol``); ``h`` and
    # ``T`` are intensive ratios reconstructed AFTER transport.  Volume
    # retention MUST therefore be keyed off ``vol_new``, not ``conc_new``:
    # ``vol`` and ``conc`` are advected as INDEPENDENT PPM scalars (their
    # limiters can disagree), so a margin cell can end the step with
    # ``vol_new > 0`` while ``conc_new`` has limited to ~0.  Returning
    # ``h_new = 0`` there (the old ``has_ice = conc_new > 0`` test) would make
    # ``h_new*conc_new = 0 != vol_new`` and VANISH the volume — a mass sink.
    # Instead: where volume exists, floor the concentration at ``_CONC_FLOOR``
    # and set ``h = vol/conc_floored`` so the returned (h, conc) reconstruct
    # the conserved volume exactly (``h_new*conc_out == vol_new``); the floor
    # bounds the ``h = vol/conc`` spike at ``vol/_CONC_FLOOR``.  We do NOT
    # re-derive conc from a capped h here: this SAME helper transports the snow
    # and pond inventories (in the h-slot) and the salinity (in the T-slot),
    # and those callers pair the returned THICKNESS with the ICE concentration,
    # so altering ``conc_out`` (or capping h) would break their inventory
    # (codex R4-1).  The bare floor matches the model's existing recovery
    # convention (sea_ice.py ``vol/max(conc, 1e-12)``); the rare absurd-h margin
    # cell is bounded by ``vol/_CONC_FLOOR`` and handled by the same downstream
    # thermo that already consumes that pattern.
    # AD-safe: substitute a benign placeholder into the denominator BEFORE
    # dividing so the true-branch arithmetic is finite at every cotangent.
    has_vol = vol_new > _VOL_FLOOR
    conc_out = jnp.where(has_vol, jnp.maximum(conc_new, _CONC_FLOOR), 0.0)
    conc_safe = jnp.where(has_vol, conc_out, 1.0)
    h_new = jnp.where(has_vol, vol_new / conc_safe, 0.0)
    vol_safe = jnp.where(has_vol, vol_new, 1.0)
    # Empty (ice-free) cells are filled at the basal/ocean freezing point
    # T_freeze_ocean (271.35 K) — there is no ice surface there.
    T_new = jnp.where(has_vol, enth_new / vol_safe, T_freeze_ocean)

    # --- Energy-conserving temperature treatment (finding #2) -------------
    # ``enth`` is advected in flux form, which conserves total enthalpy to
    # machine precision; the conserved energy is then carried by ``T*vol``.
    # The PREVIOUS hard clip ``clip(T_new, T_ice_min, T_max)`` deleted/created
    # energy whenever it fired: under CFL>1 the independent vol/enth limiters
    # let ``T = enth/vol`` overshoot (verified up to ~700 K), and clamping T
    # to the melt point silently removed that enthalpy with no flux to credit
    # it -> energy non-conservation.  We instead clamp ONLY to a wide,
    # non-binding numerical-safety range (cf. the salinity channel in
    # sea_ice.py, which makes the identical choice for the same reason): this
    # catches NaN/absurd values for downstream stability without touching the
    # physical melt point.  At the supported CFL<=1 (force with
    # transport_subcycles>1) PPM is monotone, this clamp is non-binding, and
    # energy is conserved exactly across the step.
    T_lo = T_ice_min - _T_SAFETY_MARGIN_K
    T_hi = T_max + _T_SAFETY_MARGIN_K
    T_new = jnp.clip(T_new, T_lo, T_hi)

    return h_new, conc_out, T_new
