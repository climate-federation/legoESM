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

from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from legoesm.core.operators_3d import fv_flux_divergence_3d
from legoesm.core.operators_fv import fv_flux_divergence
from legoesm.core.operators_fv_latlon import fv_flux_divergence_latlon
from legoesm.core.source_rounding import nemo_source_round
from legoesm.core.operators_voronoi import (
    divergence_cell,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.voronoi import VoronoiMesh

from legoesm import constants

# Numerical-stability bounds for the temperature/enthalpy channel.  These
# MIRROR ``SeaIceConfig.T_ice_min`` / ``SeaIceConfig.T_melt_surface`` (both
# documented in config.py as numerics-only bounds, NOT physical conservation
# limits) so the signature defaults reference a Name, not a bare literal.
_T_ICE_MIN_DEFAULT: float = 180.0  # == SeaIceConfig.T_ice_min [K]
# Concentration floor used when recovering an intensive tracer ratio
# (h = vol/conc).  Floors the DENOMINATOR so the h = vol/conc spike is bounded.
_CONC_FLOOR: float = 1e-12  # [dimensionless area fraction]
# Volume floor [m of ice per m^2 of grid cell] deciding whether a cell still
# holds ice for the recovery branch.  Distinct from ``_CONC_FLOOR`` (a
# concentration) — they share the 1e-12 numeric value (the same threshold the
# thermo/lead-freeze recovery in sea_ice.py uses, ``V_after > 1e-12``) but
# carry DIFFERENT units, so name them separately.  A cell with 0 < vol <= this
# carries <1e-9 kg/m^2 of ice and is treated as ice-free.
_VOL_FLOOR: float = 1e-12  # [m of ice per m^2 grid cell]
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
_T_SAFETY_MARGIN_K: float = 50.0  # widen [T_ice_min, T_max] by this on each side

# SI3's Prather program carries these five 2-D polynomial moments for every
# transported extensive tracer.  The order is the restart order used by
# icedyn_adv_pra.F90:1397-1497.  Keeping one packed array per moment makes the
# carry a five-array pytree regardless of the number of transported tracers.
SI3PratherMoments = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
SI3_PRATHER_MOMENT_NAMES = ("sx", "sy", "sxx", "syy", "sxy")

# Fixed coefficients in the shipped SI3 Prather program.  These are named
# here, rather than hidden at arithmetic sites, so the source contract is
# reviewable by the repository physics-coefficient ratchet.
_SI3_PRA_SLOPE_CAP_FACTOR = 1.5  # icedyn_adv_pra.F90:553
_SI3_PRA_SECOND_MOMENT_MERGE_FACTOR = 5.0  # icedyn_adv_pra.F90:681,702
_SI3_PRA_SECOND_MOMENT_FACTOR = 2.0  # icedyn_adv_pra.F90:555,778
_SI3_PRA_FLUX_MOMENT_FACTOR = 3.0  # icedyn_adv_pra.F90:584,679,807,903
_SI3_PRA_ONE = 1.0  # icedyn_adv_pra.F90:547,575,770,798
_SI3_PRA_ZERO = 0.0  # icedyn_adv_pra.F90:549,563-567,772,786-790
_SI3_PRA_THREE = 3.0  # icedyn_adv_pra.F90:534
_SI3_PRA_AREA_FLOOR = 1.0e-20  # icedyn_adv_pra.F90:547,770; epsi20
_SI3_PRA_ICE_PRESENCE = 1.0e-10  # icedyn_adv_pra.F90:170,175; epsi10
_SI3_PRA_CFL_TWO_CYCLE_THRESHOLD = 0.5  # icedyn_adv_pra.F90:124-126


def si3_prather_pack_intensives(
    intensives: jnp.ndarray,
    cell_area: jnp.ndarray,
) -> jnp.ndarray:
    """Form SI3's extensive Prather work arrays in NEMO statement order.

    NEMO holds category fields as intensives between outer steps and evaluates
    each ``z0* = p* * e1e2t`` assignment at
    ``icedyn_adv_pra.F90:218-245`` only on entry to Prather advection.
    """

    if intensives.ndim != 3 or cell_area.shape != intensives.shape[:2]:
        raise ValueError("SI3 Prather intensive/area shapes disagree")
    return nemo_source_round(intensives * cell_area[..., None])


def si3_prather_unpack_intensives(
    contents: jnp.ndarray,
    cell_area: jnp.ndarray,
    wet: jnp.ndarray,
) -> jnp.ndarray:
    """Recover SI3 intensives as ``z0 * r1_e1e2t * tmask``.

    The reciprocal and the two left-to-right products reproduce
    ``icedyn_adv_pra.F90:355-381``.  Source rounding prevents XLA from
    contracting the written operations into a different bridge.
    """

    if contents.ndim != 3 or cell_area.shape != contents.shape[:2]:
        raise ValueError("SI3 Prather content/area shapes disagree")
    if wet.shape != contents.shape[:2]:
        raise ValueError("SI3 Prather content/mask shapes disagree")
    reciprocal_area = nemo_source_round(jnp.reciprocal(cell_area))
    intensives = nemo_source_round(contents * reciprocal_area[..., None])
    return nemo_source_round(intensives * wet[..., None])
_SI3_PRA_CFL_THREE_CYCLE_THRESHOLD = 1.5  # icedyn_adv_pra.F90:124-126
_SI3_PRA_HBIG_CONCENTRATION_THRESHOLD = 0.15  # icedyn_adv_pra.F90:1000
_SI3_PRA_HBIG_MAX_THICKNESS_M = 99.0  # namelist_ice_ref:46; ORCA1 does not override


def _is_latlon_grid(grid) -> bool:
    return isinstance(grid, LatLonGrid)


def _is_voronoi_mesh(grid) -> bool:
    return isinstance(grid, VoronoiMesh)


def _is_latlon_cgrid(grid) -> bool:
    from legoesm.grids.latlon import LatLonCGridGeometry

    return isinstance(grid, LatLonCGridGeometry)


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


def fv_flux_divergence_latlon_cgrid(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """Donor-cell (first-order upwind) flux-divergence tendency on the
    curvilinear lat-lon C-grid (tripole eORCA geometry).

    ``u, v`` are the ice model's CELL-CENTRED **geographic east/north**
    velocity components (A-grid convention, same as the other dispatch
    branches).  They are interpolated to the C-grid faces and rotated into
    grid-relative face-normal components with the LOCAL face angles — the
    same E-N -> face transform the OMIP wind-stress applicator uses.
    Geographic components are frame-independent scalars across the tripolar
    north fold (the partner cell holds the SAME physical vector), so the
    fold row folds with scalar (+1) parity and the i/j orientation flip
    across the seam is carried entirely by the local rotation angles; the
    transported scalar is reconstructed at faces by the fold-aware
    donor-cell upwind, and :func:`divergence_cgrid` closes the budget —
    global ``sum(q * area_T)`` is conserved to round-off.

    First-order upwind (monotone, diffusive) rather than PPM: the ice-edge
    is a moving front where monotonicity matters more than order, and the
    donor-cell scheme needs no curvilinear PPM machinery.  LAND CAVEAT
    (parity with the existing LatLonGrid branch, which also carries no
    mask): free-drift can push ice onto land cells; the OMIP runner masks
    every ice->ocean flux by the ocean mask, so land-ice never reaches the
    ocean budget, but it does sit in the ice inventory — same disclosed
    behavior as the A-grid lat-lon path.

    Sign convention: returns ``dq/dt`` (``q_new = q + dt * tendency``).
    """
    from legoesm.grids.operators_latlon_cgrid import (
        divergence_cgrid,
        fold_is_local,
        interp_cell_to_uface,
        interp_cell_to_vface,
        north_fold_mask,
        upwind_cell_to_uface,
        upwind_cell_to_vface,
    )

    u_face = interp_cell_to_uface(u) * grid.cos_alpha_u + interp_cell_to_uface(v) * grid.sin_alpha_u
    v_face = (
        -interp_cell_to_vface(u, grid) * grid.sin_alpha_v
        + interp_cell_to_vface(v, grid) * grid.cos_alpha_v
    )
    q_u = upwind_cell_to_uface(q, u_face)
    q_v = upwind_cell_to_vface(q, v_face, grid)
    flux_u = u_face * q_u
    flux_v = v_face * q_v
    # SEAM FLUX: one SHARED upwind flux per fold pair (codex r1+r2).  On the
    # real eORCA1.2 mesh the fold-paired cells are distinct geographic
    # locations, so the E-N interp/rotation gives each side of the seam an
    # independent flux estimate that does not pair-cancel (a global
    # conservation leak of O(1e-4)); and a bare flux antisymmetrization
    # F <- (F(i)-F(perm(i)))/2 restores the global sum but breaks the upwind
    # donor logic (it can extract ice from an EMPTY partner cell — a local
    # positivity/inventory violation, codex r2 #1).  Instead build the seam
    # from first principles: antisymmetrize the face VELOCITY (the pair share
    # ONE physical face, so v_pair(i) = (v(i) - v(perm(i)))/2 with
    # v(perm(i)) == -v(i) up to the two estimates), then select ONE shared
    # donor cell for the pair — top-row cell i when the flow leaves cell i
    # northward, else the partner cell perm(i).  At the partner index the
    # velocity is exactly negated and the where() selects the SAME donor, so
    # F(perm(i)) == -F(i) identically: exact pair cancellation AND true
    # donor-cell upwind (no flux out of an empty cell).  Gating mirrors
    # upwind_cell_to_vface: serial/MPI seam owner via fold_is_local, lat-band
    # SPMD via the traced north_fold_mask (fold.is_active alone is True on
    # EVERY band and would corrupt interior partition-top faces, codex r2 #2).
    fold = getattr(grid, "fold", None)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        # Build the shared velocity from FIRST PRINCIPLES rather than from
        # v_face's fold row: pad_ns_scalar writes the PARTNER-side estimate
        # into the seam row (side-swapped), so differencing v_face rows flips
        # the sign of the shared velocity.  Instead project the LAST CELL
        # ROW's geographic (u, v) onto the seam-face normal locally,
        #     L(j) = -uE[-1,j]*sin(a_v[-1,j]) + vN[-1,j]*cos(a_v[-1,j]),
        # and fold in the partner's projection with the orientation flip
        # (vector parity: the same physical face seen from the other side),
        #     v_pair(j) = (L(j) - L(perm_v(j))) / 2.
        # Under a uniform geographic flow across a real fold L(perm(j)) is
        # ~ -L(j), so v_pair recovers the FULL local projection.
        sin_av = jnp.asarray(grid.sin_alpha_v)[-1]
        cos_av = jnp.asarray(grid.cos_alpha_v)[-1]
        L = -u[-1] * sin_av + v[-1] * cos_av
        v_pair = 0.5 * (L - L[fold.perm_v])
        q_top = q[-1]
        q_donor = jnp.where(v_pair > 0, q_top, q_top[fold.perm_T])
        seam_flux = v_pair * q_donor
        if nmask is not None and not fold_is_local(grid):
            seam_flux = jnp.where(nmask, seam_flux, flux_v[-1])
        flux_v = flux_v.at[-1].set(seam_flux)
    return -divergence_cgrid(flux_u, flux_v, grid)


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
      * ``LatLonCGridGeometry`` (tripole): donor-cell upwind
        ``fv_flux_divergence_latlon_cgrid``.

    Sign convention: ``q_new = q + dt · tendency``.
    """
    if _is_latlon_grid(grid):
        return fv_flux_divergence_latlon(q, u, v, grid, limiter=True)
    if _is_voronoi_mesh(grid):
        return fv_flux_divergence_voronoi(q, u, v, grid)
    if _is_latlon_cgrid(grid):
        return fv_flux_divergence_latlon_cgrid(q, u, v, grid)
    if isinstance(grid, CubedSphereGrid):
        return fv_flux_divergence(q, u, v, grid, limiter=True)
    raise TypeError(f"unsupported grid {type(grid).__name__} for _ppm_tendency_2d")


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
    if _is_latlon_cgrid(grid):

        def _kernel_c(qk):
            return fv_flux_divergence_latlon_cgrid(qk, u, v, grid)

        return jax.vmap(_kernel_c, in_axes=-1, out_axes=-1)(q_cat)
    if not isinstance(grid, CubedSphereGrid):
        raise TypeError(f"unsupported grid {type(grid).__name__} for _ppm_tendency_per_category")
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
    vol = h_ice * concentration  # m of ice per m² of grid
    enth = T_ice * vol  # K · m
    conc = concentration

    # Multi-category detection: base ndim depends on grid layout.
    # MPAS Voronoi: ``(nCells,)`` (1D); lat-lon / tripole C-grid:
    # ``(n_lat, n_lon)`` (2D); cubed-sphere: ``(6, n, n)`` (3D).  Trailing
    # category axis adds one rank.
    if _is_voronoi_mesh(grid):
        base_ndim = 1
    elif _is_latlon_grid(grid) or _is_latlon_cgrid(grid):
        base_ndim = 2
    else:
        base_ndim = 3
    is_multicat = h_ice.ndim == (base_ndim + 1)
    dt_sub = dt / n_subcycles

    if n_subcycles == 1:
        vol_new, conc_new, enth_new = _ppm_one_substep(
            vol,
            conc,
            enth,
            u_ice,
            v_ice,
            grid,
            dt_sub,
            is_multicat,
        )
    else:

        def _body(carry, _):
            v, c, e = carry
            return _ppm_one_substep(
                v,
                c,
                e,
                u_ice,
                v_ice,
                grid,
                dt_sub,
                is_multicat,
            ), None

        (vol_new, conc_new, enth_new), _ = lax.scan(
            _body,
            (vol, conc, enth),
            xs=None,
            length=n_subcycles,
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


def zero_si3_prather_moments(contents: jnp.ndarray) -> SI3PratherMoments:
    """Cold-start SI3's five prognostic moments as dtype-identical zero arrays.

    This is the ``ln_rstart = .FALSE.`` arm at
    ``icedyn_adv_pra.F90:1361-1370``.  ``contents`` is a packed collection of
    extensive tracers with shape ``(x_with_halo, y_with_halo, n_tracer)``.
    """

    zero = jnp.zeros_like(contents)
    return zero, zero, zero, zero, zero


def _si3_prather_limit_x(
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    wet: jnp.ndarray,
) -> tuple[jnp.ndarray, SI3PratherMoments]:
    """SI3 x-sweep limiter, ``icedyn_adv_pra.F90:534-568``."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    one_third = sr(_SI3_PRA_ONE / _SI3_PRA_THREE)
    content = sr(jnp.maximum(_SI3_PRA_ZERO, content))
    slope_cap = sr(_SI3_PRA_SLOPE_CAP_FACTOR * content)
    negative_slope_cap = sr(-slope_cap)
    sx = sr(jnp.minimum(slope_cap, sr(jnp.maximum(negative_slope_cap, sx))))
    abs_sx = sr(jnp.abs(sx))
    second_upper = sr(
        sr(_SI3_PRA_SECOND_MOMENT_FACTOR * content) - sr(one_third * abs_sx)
    )
    second_lower = sr(jnp.maximum(sr(abs_sx - content), sxx))
    sxx = sr(jnp.minimum(second_upper, second_lower))
    negative_content = sr(-content)
    sxy = sr(jnp.minimum(content, sr(jnp.maximum(negative_content, sxy))))
    wet3 = wet[..., None].astype(content.dtype)
    active3 = content > _SI3_PRA_ZERO
    return content, (
        sr(jnp.where(active3, sr(sx * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sy * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sxx * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(syy * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sxy * wet3), _SI3_PRA_ZERO)),
    )


def _si3_prather_limit_y(
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    wet: jnp.ndarray,
) -> tuple[jnp.ndarray, SI3PratherMoments]:
    """SI3 y-sweep limiter, ``icedyn_adv_pra.F90:757-791``."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    one_third = sr(_SI3_PRA_ONE / _SI3_PRA_THREE)
    content = sr(jnp.maximum(_SI3_PRA_ZERO, content))
    slope_cap = sr(_SI3_PRA_SLOPE_CAP_FACTOR * content)
    negative_slope_cap = sr(-slope_cap)
    sy = sr(jnp.minimum(slope_cap, sr(jnp.maximum(negative_slope_cap, sy))))
    abs_sy = sr(jnp.abs(sy))
    second_upper = sr(
        sr(_SI3_PRA_SECOND_MOMENT_FACTOR * content) - sr(one_third * abs_sy)
    )
    second_lower = sr(jnp.maximum(sr(abs_sy - content), syy))
    syy = sr(jnp.minimum(second_upper, second_lower))
    negative_content = sr(-content)
    sxy = sr(jnp.minimum(content, sr(jnp.maximum(negative_content, sxy))))
    wet3 = wet[..., None].astype(content.dtype)
    active3 = content > _SI3_PRA_ZERO
    return content, (
        sr(jnp.where(active3, sr(sx * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sy * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sxx * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(syy * wet3), _SI3_PRA_ZERO)),
        sr(jnp.where(active3, sr(sxy * wet3), _SI3_PRA_ZERO)),
    )


def _si3_prather_merge_from_left(
    area: jnp.ndarray,
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    flux_area: jnp.ndarray,
    flux_content: jnp.ndarray,
    flux_moments: SI3PratherMoments,
) -> tuple[jnp.ndarray, jnp.ndarray, SI3PratherMoments]:
    """Apply SI3's positive-U receiver program (Fortran lines 667-686)."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    fx, fy, fxx, fyy, fxy = flux_moments
    area = sr(area + flux_area)
    alpha = sr(flux_area / area)
    one = sr(_SI3_PRA_ONE - alpha)
    alpha2 = sr(alpha * alpha)
    one2 = sr(one * one)
    content = sr(content + flux_content)
    displacement = sr(
        sr(alpha[..., None] * content) - sr(one[..., None] * flux_content)
    )
    sx = sr(
        sr(sr(alpha[..., None] * fx) + sr(one[..., None] * sx))
        + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * displacement)
    )
    first = sr(sr(alpha2[..., None] * fxx) + sr(one2[..., None] * sxx))
    alpha_one = sr(alpha * one)
    first_correction = sr(alpha_one[..., None] * sr(sx - fx))
    second_correction = sr(sr(one - alpha)[..., None] * displacement)
    correction = sr(
        _SI3_PRA_SECOND_MOMENT_MERGE_FACTOR
        * sr(first_correction - second_correction)
    )
    sxx = sr(first + correction)
    cross_base = sr(sr(alpha[..., None] * fxy) + sr(one[..., None] * sxy))
    negative_one = sr(-one)
    cross_pair = sr(
        sr(negative_one[..., None] * fy) + sr(alpha[..., None] * sy)
    )
    sxy = sr(cross_base + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * cross_pair))
    sy = sr(sy + fy)
    syy = sr(syy + fyy)
    return area, content, (sx, sy, sxx, syy, sxy)


def _si3_prather_merge_from_right(
    area: jnp.ndarray,
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    flux_area: jnp.ndarray,
    flux_content: jnp.ndarray,
    flux_moments: SI3PratherMoments,
) -> tuple[jnp.ndarray, jnp.ndarray, SI3PratherMoments]:
    """Apply SI3's negative-U receiver program (Fortran lines 688-707)."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    fx, fy, fxx, fyy, fxy = flux_moments
    area = sr(area + flux_area)
    alpha = sr(flux_area / area)
    one = sr(_SI3_PRA_ONE - alpha)
    alpha2 = sr(alpha * alpha)
    one2 = sr(one * one)
    content = sr(content + flux_content)
    negative_alpha = sr(-alpha)
    displacement = sr(
        sr(negative_alpha[..., None] * content)
        + sr(one[..., None] * flux_content)
    )
    sx = sr(
        sr(sr(alpha[..., None] * fx) + sr(one[..., None] * sx))
        + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * displacement)
    )
    first = sr(sr(alpha2[..., None] * fxx) + sr(one2[..., None] * sxx))
    alpha_one = sr(alpha * one)
    negative_sx = sr(-sx)
    first_correction = sr(alpha_one[..., None] * sr(negative_sx + fx))
    second_correction = sr(sr(one - alpha)[..., None] * displacement)
    correction = sr(
        _SI3_PRA_SECOND_MOMENT_MERGE_FACTOR
        * sr(first_correction + second_correction)
    )
    sxx = sr(first + correction)
    cross_base = sr(sr(alpha[..., None] * fxy) + sr(one[..., None] * sxy))
    cross_pair = sr(sr(one[..., None] * fy) - sr(alpha[..., None] * sy))
    sxy = sr(cross_base + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * cross_pair))
    sy = sr(sy + fy)
    syy = sr(syy + fyy)
    return area, content, (sx, sy, sxx, syy, sxy)


def _si3_prather_merge_from_below(
    area: jnp.ndarray,
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    flux_area: jnp.ndarray,
    flux_content: jnp.ndarray,
    flux_moments: SI3PratherMoments,
) -> tuple[jnp.ndarray, jnp.ndarray, SI3PratherMoments]:
    """Apply SI3's positive-V receiver program (Fortran lines 891-910)."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    fx, fy, fxx, fyy, fxy = flux_moments
    area = sr(area + flux_area)
    alpha = sr(flux_area / area)
    one = sr(_SI3_PRA_ONE - alpha)
    alpha2 = sr(alpha * alpha)
    one2 = sr(one * one)
    content = sr(content + flux_content)
    displacement = sr(
        sr(alpha[..., None] * content) - sr(one[..., None] * flux_content)
    )
    sy = sr(
        sr(sr(alpha[..., None] * fy) + sr(one[..., None] * sy))
        + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * displacement)
    )
    first = sr(sr(alpha2[..., None] * fyy) + sr(one2[..., None] * syy))
    alpha_one = sr(alpha * one)
    first_correction = sr(alpha_one[..., None] * sr(sy - fy))
    second_correction = sr(sr(one - alpha)[..., None] * displacement)
    correction = sr(
        _SI3_PRA_SECOND_MOMENT_MERGE_FACTOR
        * sr(first_correction - second_correction)
    )
    syy = sr(first + correction)
    cross_base = sr(sr(alpha[..., None] * fxy) + sr(one[..., None] * sxy))
    negative_one = sr(-one)
    cross_pair = sr(
        sr(negative_one[..., None] * fx) + sr(alpha[..., None] * sx)
    )
    sxy = sr(cross_base + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * cross_pair))
    sx = sr(sx + fx)
    sxx = sr(sxx + fxx)
    return area, content, (sx, sy, sxx, syy, sxy)


def _si3_prather_merge_from_above(
    area: jnp.ndarray,
    content: jnp.ndarray,
    moments: SI3PratherMoments,
    flux_area: jnp.ndarray,
    flux_content: jnp.ndarray,
    flux_moments: SI3PratherMoments,
) -> tuple[jnp.ndarray, jnp.ndarray, SI3PratherMoments]:
    """Apply SI3's negative-V receiver program (Fortran lines 912-930)."""

    sr = nemo_source_round
    sx, sy, sxx, syy, sxy = moments
    fx, fy, fxx, fyy, fxy = flux_moments
    area = sr(area + flux_area)
    alpha = sr(flux_area / area)
    one = sr(_SI3_PRA_ONE - alpha)
    alpha2 = sr(alpha * alpha)
    one2 = sr(one * one)
    content = sr(content + flux_content)
    negative_alpha = sr(-alpha)
    displacement = sr(
        sr(negative_alpha[..., None] * content)
        + sr(one[..., None] * flux_content)
    )
    sy = sr(
        sr(sr(alpha[..., None] * fy) + sr(one[..., None] * sy))
        + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * displacement)
    )
    first = sr(sr(alpha2[..., None] * fyy) + sr(one2[..., None] * syy))
    alpha_one = sr(alpha * one)
    negative_sy = sr(-sy)
    first_correction = sr(alpha_one[..., None] * sr(negative_sy + fy))
    second_correction = sr(sr(one - alpha)[..., None] * displacement)
    correction = sr(
        _SI3_PRA_SECOND_MOMENT_MERGE_FACTOR
        * sr(first_correction + second_correction)
    )
    syy = sr(first + correction)
    cross_base = sr(sr(alpha[..., None] * fxy) + sr(one[..., None] * sxy))
    cross_pair = sr(sr(one[..., None] * fx) - sr(alpha[..., None] * sx))
    sxy = sr(cross_base + sr(_SI3_PRA_FLUX_MOMENT_FACTOR * cross_pair))
    sx = sr(sx + fx)
    sxx = sr(sxx + fxx)
    return area, content, (sx, sy, sxx, syy, sxy)


def _si3_prather_x_substep(
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
    u_transport: jnp.ndarray,
    cell_area: jnp.ndarray,
    wet: jnp.ndarray,
    dt: float,
    *,
    initial_area: jnp.ndarray,
    first_sweep: bool,
    halo_width: int,
    subcycle_index: int,
    subcycles: int,
) -> tuple[jnp.ndarray, SI3PratherMoments, jnp.ndarray]:
    """One closed-box x sweep of SI3 ``adv_x`` (Fortran lines 499-719)."""

    n_x = contents.shape[0]
    ihls = 0 if subcycles == 1 else max(0, halo_width - subcycle_index)
    ji0 = 1 + ihls
    jj0 = (1 if first_sweep else 0) + ihls
    index = jnp.arange(n_x)
    index_y = jnp.arange(contents.shape[1])
    limit_mask = (index >= halo_width - ji0) & (index < n_x - halo_width + ji0)
    cross_mask = (index_y >= halo_width - jj0) & (index_y < contents.shape[1] - halo_width + jj0)
    receiver_width = ji0 - 1
    receiver_mask = (index >= halo_width - receiver_width) & (
        index < n_x - halo_width + receiver_width
    )
    face_mask = (index >= halo_width - ji0) & (index < n_x - halo_width + receiver_width)
    limit_mask_xy = limit_mask[:, None] & cross_mask[None, :]
    receiver_mask_xy = receiver_mask[:, None] & cross_mask[None, :]
    face_mask_xy = face_mask[:, None] & cross_mask[None, :]
    sr = nemo_source_round
    pcrh = _SI3_PRA_ONE if first_sweep else _SI3_PRA_ZERO
    one_minus_pcrh = sr(_SI3_PRA_ONE - pcrh)
    sweep_area = sr(
        jnp.maximum(
            sr(sr(pcrh * cell_area) + sr(one_minus_pcrh * initial_area)),
            _SI3_PRA_AREA_FLOOR,
        )
    )

    limited_content, limited_moments = _si3_prather_limit_x(contents, moments, wet)
    content = sr(jnp.where(limit_mask_xy[..., None], limited_content, contents))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(limit_mask_xy[..., None], new, old))
            for new, old in zip(limited_moments, moments, strict=True)
        ),
    )
    sx, sy, sxx, syy, sxy = moments

    positive = (u_transport >= 0.0) & face_mask_xy
    negative = (u_transport < 0.0) & face_mask_xy

    # NEMO first computes and removes every right-going slab
    # (icedyn_adv_pra.F90:570-616).  The negative-face flux loop then reads
    # those already-updated donor boxes at i+1 (:618-637).  Preserving that
    # ordering matters when one cell exports through both faces.
    positive_alpha = sr(
        jnp.where(
            positive,
            sr(sr(u_transport * dt) / sweep_area),
            _SI3_PRA_ZERO,
        )
    )
    positive_one = sr(_SI3_PRA_ONE - positive_alpha)
    positive_alpha2 = sr(positive_alpha * positive_alpha)
    positive_alpha3 = sr(positive_alpha2 * positive_alpha)
    # Keep SI3's written operation order: zalf is formed first and zfm is
    # reconstructed as zalf*zpsm (icedyn_adv_pra.F90:570-582).  Cancelling
    # this product algebraically changes receiver fractions at fp64.
    positive_flux_area = sr(
        jnp.where(
            positive, sr(positive_alpha * sweep_area), _SI3_PRA_ZERO
        )
    )
    positive_delta = sr(positive_one - positive_alpha)
    positive_flux_content = sr(
        positive_alpha[..., None]
        * sr(
            content
            + sr(
                positive_one[..., None]
                * sr(sx + sr(positive_delta[..., None] * sxx))
            )
        )
    )
    positive_flux_content = sr(
        jnp.where(positive[..., None], positive_flux_content, _SI3_PRA_ZERO)
    )
    positive_flux_x = sr(
        jnp.where(
            positive[..., None],
            sr(
                positive_alpha2[..., None]
                * sr(
                    sx
                    + sr(
                        sr(_SI3_PRA_FLUX_MOMENT_FACTOR * positive_one)[..., None]
                        * sxx
                    )
                )
            ),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_xx = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha3[..., None] * sxx),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_y = sr(
        jnp.where(
            positive[..., None],
            sr(
                positive_alpha[..., None]
                * sr(sy + sr(positive_one[..., None] * sxy))
            ),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_yy = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha[..., None] * syy),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_xy = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha2[..., None] * sxy),
            _SI3_PRA_ZERO,
        )
    )

    positive_one2 = sr(positive_one * positive_one)
    positive_one3 = sr(positive_one2 * positive_one)
    area_after_right = sr(sweep_area - positive_flux_area)
    content_after_right = sr(content - positive_flux_content)
    sx_after_right = sr(
        positive_one2[..., None]
        * sr(
            sx
            - sr(
                sr(_SI3_PRA_FLUX_MOMENT_FACTOR * positive_alpha)[..., None]
                * sxx
            )
        )
    )
    sxx_after_right = sr(positive_one3[..., None] * sxx)
    sy_after_right = sr(sy - positive_flux_y)
    syy_after_right = sr(syy - positive_flux_yy)
    sxy_after_right = sr(positive_one2[..., None] * sxy)

    # Negative fluxes use the post-positive residual of donor i+1 exactly as
    # NEMO does at icedyn_adv_pra.F90:618-664, rather than independently
    # extracting both slabs from the step-entry polynomial.
    negative_donor_area = jnp.roll(area_after_right, -1, axis=0)
    negative_donor_content = jnp.roll(content_after_right, -1, axis=0)
    negative_donor_sx = jnp.roll(sx_after_right, -1, axis=0)
    negative_donor_sy = jnp.roll(sy_after_right, -1, axis=0)
    negative_donor_sxx = jnp.roll(sxx_after_right, -1, axis=0)
    negative_donor_syy = jnp.roll(syy_after_right, -1, axis=0)
    negative_donor_sxy = jnp.roll(sxy_after_right, -1, axis=0)
    negative_velocity = sr(-u_transport)
    negative_alpha = sr(
        jnp.where(
            negative,
            sr(sr(negative_velocity * dt) / negative_donor_area),
            _SI3_PRA_ZERO,
        )
    )
    negative_one = sr(_SI3_PRA_ONE - negative_alpha)
    negative_alpha2 = sr(negative_alpha * negative_alpha)
    negative_alpha3 = sr(negative_alpha2 * negative_alpha)
    negative_flux_area = sr(
        jnp.where(
            negative,
            sr(negative_alpha * negative_donor_area),
            _SI3_PRA_ZERO,
        )
    )
    negative_delta = sr(negative_one - negative_alpha)
    negative_flux_content = sr(
        negative_alpha[..., None]
        * sr(
            negative_donor_content
            - sr(
                negative_one[..., None]
                * sr(
                    negative_donor_sx
                    - sr(negative_delta[..., None] * negative_donor_sxx)
                )
            )
        )
    )
    negative_flux_content = sr(
        jnp.where(negative[..., None], negative_flux_content, _SI3_PRA_ZERO)
    )
    negative_flux_x = jnp.zeros_like(negative_flux_content)
    negative_flux_xx = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha3[..., None] * negative_donor_sxx),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_y = sr(
        jnp.where(
            negative[..., None],
            sr(
                negative_alpha[..., None]
                * sr(
                    negative_donor_sy
                    - sr(negative_one[..., None] * negative_donor_sxy)
                )
            ),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_yy = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha[..., None] * negative_donor_syy),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_xy = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha2[..., None] * negative_donor_sxy),
            _SI3_PRA_ZERO,
        )
    )

    out_left = jnp.roll(negative, 1, axis=0)
    left_velocity = jnp.roll(u_transport, 1, axis=0)
    left_alpha = sr(
        jnp.where(
            out_left,
            sr(sr(sr(-left_velocity) * dt) / area_after_right),
            _SI3_PRA_ZERO,
        )
    )
    left_one = sr(_SI3_PRA_ONE - left_alpha)
    left_one2 = sr(left_one * left_one)
    left_one3 = sr(left_one2 * left_one)
    left_flux_area = sr(
        jnp.where(
            out_left,
            jnp.roll(negative_flux_area, 1, axis=0),
            _SI3_PRA_ZERO,
        )
    )
    left_flux_content = sr(
        jnp.where(
            out_left[..., None],
            jnp.roll(negative_flux_content, 1, axis=0),
            _SI3_PRA_ZERO,
        )
    )
    area = sr(area_after_right - left_flux_area)
    content = sr(content_after_right - left_flux_content)
    sx = sr(
        left_one2[..., None]
        * sr(
            sx_after_right
            + sr(
                sr(_SI3_PRA_FLUX_MOMENT_FACTOR * left_alpha)[..., None]
                * sxx_after_right
            )
        )
    )
    sxx = sr(left_one3[..., None] * sxx_after_right)
    left_flux_y = sr(
        jnp.where(
            out_left[..., None],
            jnp.roll(negative_flux_y, 1, axis=0),
            _SI3_PRA_ZERO,
        )
    )
    left_flux_yy = sr(
        jnp.where(
            out_left[..., None],
            jnp.roll(negative_flux_yy, 1, axis=0),
            _SI3_PRA_ZERO,
        )
    )
    sy = sr(sy_after_right - left_flux_y)
    syy = sr(syy_after_right - left_flux_yy)
    sxy = sr(left_one2[..., None] * sxy_after_right)
    moments = sx, sy, sxx, syy, sxy

    flux_area = sr(positive_flux_area + negative_flux_area)
    flux_content = sr(positive_flux_content + negative_flux_content)
    flux_moments = (
        sr(positive_flux_x + negative_flux_x),
        sr(positive_flux_y + negative_flux_y),
        sr(positive_flux_xx + negative_flux_xx),
        sr(positive_flux_yy + negative_flux_yy),
        sr(positive_flux_xy + negative_flux_xy),
    )

    incoming_left = receiver_mask_xy & jnp.roll(positive, 1, axis=0)
    left_area = sr(
        jnp.where(
            incoming_left, jnp.roll(flux_area, 1, axis=0), _SI3_PRA_ZERO
        )
    )
    left_content = sr(
        jnp.where(
            incoming_left[..., None],
            jnp.roll(flux_content, 1, axis=0),
            _SI3_PRA_ZERO,
        )
    )
    left_moments = cast(
        SI3PratherMoments,
        tuple(
            sr(
                jnp.where(
                    incoming_left[..., None],
                    jnp.roll(value, 1, axis=0),
                    _SI3_PRA_ZERO,
                )
            )
            for value in flux_moments
        ),
    )
    merged_area, merged_content, merged_moments = _si3_prather_merge_from_left(
        area, content, moments, left_area, left_content, left_moments
    )
    area = sr(jnp.where(incoming_left, merged_area, area))
    content = sr(jnp.where(incoming_left[..., None], merged_content, content))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_left[..., None], new, old))
            for new, old in zip(merged_moments, moments, strict=True)
        ),
    )

    incoming_right = receiver_mask_xy & negative
    right_area = sr(jnp.where(incoming_right, flux_area, _SI3_PRA_ZERO))
    right_content = sr(
        jnp.where(incoming_right[..., None], flux_content, _SI3_PRA_ZERO)
    )
    right_moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_right[..., None], value, _SI3_PRA_ZERO))
            for value in flux_moments
        ),
    )
    merged_area, merged_content, merged_moments = _si3_prather_merge_from_right(
        area, content, moments, right_area, right_content, right_moments
    )
    area = sr(jnp.where(incoming_right, merged_area, area))
    content = sr(jnp.where(incoming_right[..., None], merged_content, content))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_right[..., None], new, old))
            for new, old in zip(merged_moments, moments, strict=True)
        ),
    )
    return content, moments, area


def _si3_prather_y_substep(
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
    v_transport: jnp.ndarray,
    cell_area: jnp.ndarray,
    wet: jnp.ndarray,
    dt: float,
    *,
    initial_area: jnp.ndarray,
    first_sweep: bool,
    halo_width: int,
    subcycle_index: int,
    subcycles: int,
) -> tuple[jnp.ndarray, SI3PratherMoments, jnp.ndarray]:
    """One y sweep of SI3 ``adv_y`` (``icedyn_adv_pra.F90:722-943``)."""

    ihls = 0 if subcycles == 1 else max(0, halo_width - subcycle_index)
    ji0 = (1 if first_sweep else 0) + ihls
    jj0 = 1 + ihls
    index_x = jnp.arange(contents.shape[0])
    index_y = jnp.arange(contents.shape[1])
    cross_mask = (index_x >= halo_width - ji0) & (index_x < contents.shape[0] - halo_width + ji0)
    limit_mask = (index_y >= halo_width - jj0) & (index_y < contents.shape[1] - halo_width + jj0)
    receiver_width = jj0 - 1
    receiver_mask = (index_y >= halo_width - receiver_width) & (
        index_y < contents.shape[1] - halo_width + receiver_width
    )
    face_mask = (index_y >= halo_width - jj0) & (
        index_y < contents.shape[1] - halo_width + receiver_width
    )
    limit_mask_xy = cross_mask[:, None] & limit_mask[None, :]
    receiver_mask_xy = cross_mask[:, None] & receiver_mask[None, :]
    face_mask_xy = cross_mask[:, None] & face_mask[None, :]
    sr = nemo_source_round
    pcrh = _SI3_PRA_ONE if first_sweep else _SI3_PRA_ZERO
    one_minus_pcrh = sr(_SI3_PRA_ONE - pcrh)
    sweep_area = sr(
        jnp.maximum(
            sr(sr(pcrh * cell_area) + sr(one_minus_pcrh * initial_area)),
            _SI3_PRA_AREA_FLOOR,
        )
    )

    limited_content, limited_moments = _si3_prather_limit_y(contents, moments, wet)
    content = sr(jnp.where(limit_mask_xy[..., None], limited_content, contents))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(limit_mask_xy[..., None], new, old))
            for new, old in zip(limited_moments, moments, strict=True)
        ),
    )
    sx, sy, sxx, syy, sxy = moments
    positive = (v_transport >= 0.0) & face_mask_xy
    negative = (v_transport < 0.0) & face_mask_xy

    positive_alpha = sr(
        jnp.where(
            positive,
            sr(sr(v_transport * dt) / sweep_area),
            _SI3_PRA_ZERO,
        )
    )
    positive_one = sr(_SI3_PRA_ONE - positive_alpha)
    positive_alpha2 = sr(positive_alpha * positive_alpha)
    positive_alpha3 = sr(positive_alpha2 * positive_alpha)
    # Preserve zalf*zpsm exactly as written at icedyn_adv_pra.F90:793-805.
    positive_flux_area = sr(
        jnp.where(
            positive, sr(positive_alpha * sweep_area), _SI3_PRA_ZERO
        )
    )
    positive_delta = sr(positive_one - positive_alpha)
    positive_flux_content = sr(
        positive_alpha[..., None]
        * sr(
            content
            + sr(
                positive_one[..., None]
                * sr(sy + sr(positive_delta[..., None] * syy))
            )
        )
    )
    positive_flux_content = sr(
        jnp.where(positive[..., None], positive_flux_content, _SI3_PRA_ZERO)
    )
    positive_flux_y = sr(
        jnp.where(
            positive[..., None],
            sr(
                positive_alpha2[..., None]
                * sr(
                    sy
                    + sr(
                        sr(_SI3_PRA_FLUX_MOMENT_FACTOR * positive_one)[..., None]
                        * syy
                    )
                )
            ),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_yy = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha3[..., None] * syy),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_x = sr(
        jnp.where(
            positive[..., None],
            sr(
                positive_alpha[..., None]
                * sr(sx + sr(positive_one[..., None] * sxy))
            ),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_xx = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha[..., None] * sxx),
            _SI3_PRA_ZERO,
        )
    )
    positive_flux_xy = sr(
        jnp.where(
            positive[..., None],
            sr(positive_alpha2[..., None] * sxy),
            _SI3_PRA_ZERO,
        )
    )

    positive_one2 = sr(positive_one * positive_one)
    positive_one3 = sr(positive_one2 * positive_one)
    area_after_up = sr(sweep_area - positive_flux_area)
    content_after_up = sr(content - positive_flux_content)
    sy_after_up = sr(
        positive_one2[..., None]
        * sr(
            sy
            - sr(
                sr(_SI3_PRA_FLUX_MOMENT_FACTOR * positive_alpha)[..., None]
                * syy
            )
        )
    )
    syy_after_up = sr(positive_one3[..., None] * syy)
    sx_after_up = sr(sx - positive_flux_x)
    sxx_after_up = sr(sxx - positive_flux_xx)
    sxy_after_up = sr(positive_one2[..., None] * sxy)

    negative_donor_area = jnp.roll(area_after_up, -1, axis=1)
    negative_donor_content = jnp.roll(content_after_up, -1, axis=1)
    negative_donor_sx = jnp.roll(sx_after_up, -1, axis=1)
    negative_donor_sy = jnp.roll(sy_after_up, -1, axis=1)
    negative_donor_sxx = jnp.roll(sxx_after_up, -1, axis=1)
    negative_donor_syy = jnp.roll(syy_after_up, -1, axis=1)
    negative_donor_sxy = jnp.roll(sxy_after_up, -1, axis=1)
    negative_velocity = sr(-v_transport)
    negative_alpha = sr(
        jnp.where(
            negative,
            sr(sr(negative_velocity * dt) / negative_donor_area),
            _SI3_PRA_ZERO,
        )
    )
    negative_one = sr(_SI3_PRA_ONE - negative_alpha)
    negative_alpha2 = sr(negative_alpha * negative_alpha)
    negative_alpha3 = sr(negative_alpha2 * negative_alpha)
    negative_flux_area = sr(
        jnp.where(
            negative,
            sr(negative_alpha * negative_donor_area),
            _SI3_PRA_ZERO,
        )
    )
    negative_delta = sr(negative_one - negative_alpha)
    negative_flux_content = sr(
        negative_alpha[..., None]
        * sr(
            negative_donor_content
            - sr(
                negative_one[..., None]
                * sr(
                    negative_donor_sy
                    - sr(negative_delta[..., None] * negative_donor_syy)
                )
            )
        )
    )
    negative_flux_content = sr(
        jnp.where(negative[..., None], negative_flux_content, _SI3_PRA_ZERO)
    )
    negative_flux_y = sr(
        jnp.where(
            negative[..., None],
            sr(
                negative_alpha2[..., None]
                * sr(
                    negative_donor_sy
                    - sr(
                        sr(_SI3_PRA_FLUX_MOMENT_FACTOR * negative_one)[..., None]
                        * negative_donor_syy
                    )
                )
            ),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_yy = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha3[..., None] * negative_donor_syy),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_x = sr(
        jnp.where(
            negative[..., None],
            sr(
                negative_alpha[..., None]
                * sr(
                    negative_donor_sx
                    - sr(negative_one[..., None] * negative_donor_sxy)
                )
            ),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_xx = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha[..., None] * negative_donor_sxx),
            _SI3_PRA_ZERO,
        )
    )
    negative_flux_xy = sr(
        jnp.where(
            negative[..., None],
            sr(negative_alpha2[..., None] * negative_donor_sxy),
            _SI3_PRA_ZERO,
        )
    )

    out_down = jnp.roll(negative, 1, axis=1)
    down_velocity = jnp.roll(v_transport, 1, axis=1)
    down_alpha = sr(
        jnp.where(
            out_down,
            sr(sr(sr(-down_velocity) * dt) / area_after_up),
            _SI3_PRA_ZERO,
        )
    )
    down_one = sr(_SI3_PRA_ONE - down_alpha)
    down_one2 = sr(down_one * down_one)
    down_one3 = sr(down_one2 * down_one)
    down_flux_area = sr(
        jnp.where(
            out_down,
            jnp.roll(negative_flux_area, 1, axis=1),
            _SI3_PRA_ZERO,
        )
    )
    down_flux_content = sr(
        jnp.where(
            out_down[..., None],
            jnp.roll(negative_flux_content, 1, axis=1),
            _SI3_PRA_ZERO,
        )
    )
    area = sr(area_after_up - down_flux_area)
    content = sr(content_after_up - down_flux_content)
    sy = sr(
        down_one2[..., None]
        * sr(
            sy_after_up
            + sr(
                sr(_SI3_PRA_FLUX_MOMENT_FACTOR * down_alpha)[..., None]
                * syy_after_up
            )
        )
    )
    syy = sr(down_one3[..., None] * syy_after_up)
    down_flux_x = sr(
        jnp.where(
            out_down[..., None],
            jnp.roll(negative_flux_x, 1, axis=1),
            _SI3_PRA_ZERO,
        )
    )
    down_flux_xx = sr(
        jnp.where(
            out_down[..., None],
            jnp.roll(negative_flux_xx, 1, axis=1),
            _SI3_PRA_ZERO,
        )
    )
    sx = sr(sx_after_up - down_flux_x)
    sxx = sr(sxx_after_up - down_flux_xx)
    sxy = sr(down_one2[..., None] * sxy_after_up)
    moments = sx, sy, sxx, syy, sxy

    flux_area = sr(positive_flux_area + negative_flux_area)
    flux_content = sr(positive_flux_content + negative_flux_content)
    flux_moments = (
        sr(positive_flux_x + negative_flux_x),
        sr(positive_flux_y + negative_flux_y),
        sr(positive_flux_xx + negative_flux_xx),
        sr(positive_flux_yy + negative_flux_yy),
        sr(positive_flux_xy + negative_flux_xy),
    )
    incoming_below = receiver_mask_xy & jnp.roll(positive, 1, axis=1)
    below_area = sr(
        jnp.where(
            incoming_below, jnp.roll(flux_area, 1, axis=1), _SI3_PRA_ZERO
        )
    )
    below_content = sr(
        jnp.where(
            incoming_below[..., None],
            jnp.roll(flux_content, 1, axis=1),
            _SI3_PRA_ZERO,
        )
    )
    below_moments = cast(
        SI3PratherMoments,
        tuple(
            sr(
                jnp.where(
                    incoming_below[..., None],
                    jnp.roll(value, 1, axis=1),
                    _SI3_PRA_ZERO,
                )
            )
            for value in flux_moments
        ),
    )
    merged_area, merged_content, merged_moments = _si3_prather_merge_from_below(
        area, content, moments, below_area, below_content, below_moments
    )
    area = sr(jnp.where(incoming_below, merged_area, area))
    content = sr(jnp.where(incoming_below[..., None], merged_content, content))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_below[..., None], new, old))
            for new, old in zip(merged_moments, moments, strict=True)
        ),
    )
    incoming_above = receiver_mask_xy & negative
    above_area = sr(jnp.where(incoming_above, flux_area, _SI3_PRA_ZERO))
    above_content = sr(
        jnp.where(incoming_above[..., None], flux_content, _SI3_PRA_ZERO)
    )
    above_moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_above[..., None], value, _SI3_PRA_ZERO))
            for value in flux_moments
        ),
    )
    merged_area, merged_content, merged_moments = _si3_prather_merge_from_above(
        area, content, moments, above_area, above_content, above_moments
    )
    area = sr(jnp.where(incoming_above, merged_area, area))
    content = sr(jnp.where(incoming_above[..., None], merged_content, content))
    moments = cast(
        SI3PratherMoments,
        tuple(
            sr(jnp.where(incoming_above[..., None], new, old))
            for new, old in zip(merged_moments, moments, strict=True)
        ),
    )
    return content, moments, area


def _si3_periodic_halo_xy(value: jnp.ndarray, halo_width: int) -> jnp.ndarray:
    """Apply ICE_ADV2D's bi-periodic T-point halo (`usrdef_nam.F90:99`)."""

    interior = value[halo_width:-halo_width, halo_width:-halo_width, ...]
    padding = ((halo_width, halo_width), (halo_width, halo_width)) + ((0, 0),) * (value.ndim - 2)
    return nemo_source_round(jnp.pad(interior, padding, mode="wrap"))


def advect_si3_prather_1d(
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    cell_area: jnp.ndarray,
    wet: jnp.ndarray,
    dt: float,
    *,
    dx: float,
    dy: float,
    halo_width: int = 2,
    ice_volume_index: int | None = None,
    concentration_index: int | None = None,
    subcycles: int | None = None,
) -> tuple[jnp.ndarray, SI3PratherMoments, int]:
    """Selectable SI3 Prather transport for the certified 1-D ice rung.

    ``contents`` packs extensive quantities (SI3 `z0*` arrays) on a Cartesian
    C-grid including halos.  A caller must validate that V is identically zero:
    the 2-D alternating-sweep program is rung 3.2, not an implicit extension of
    this rung.  The returned integer is SI3's resolved CFL cycle count from
    ``icedyn_adv_pra.F90:116-132``.  Pass that integer as the static
    ``subcycles`` argument when enclosing the kernel in ``jax.jit``.
    """

    if contents.ndim != 3:
        raise ValueError("si3_prather requires (x, y, tracer) packed contents")
    if any(moment.shape != contents.shape for moment in moments):
        raise ValueError("si3_prather moment shapes must match packed contents")
    if u_ice.shape != contents.shape[:2] or v_ice.shape != u_ice.shape:
        raise ValueError("si3_prather velocity/contents shapes disagree")
    if cell_area.shape != u_ice.shape or wet.shape != u_ice.shape:
        raise ValueError("si3_prather metric/mask shapes disagree")
    if ice_volume_index is None and concentration_index is None:
        raise ValueError("si3_prather requires volume and concentration indices for Hbig")
    if ice_volume_index is None or concentration_index is None:
        raise ValueError("si3_prather Hbig requires both volume and concentration indices")
    if subcycles is None:
        cfl = float(jnp.max(jnp.abs(u_ice)) * dt / dx)
        subcycles = (
            3
            if cfl > _SI3_PRA_CFL_THREE_CYCLE_THRESHOLD
            else 2
            if cfl > _SI3_PRA_CFL_TWO_CYCLE_THRESHOLD
            else 1
        )
    if subcycles not in (1, 2, 3):
        raise ValueError(f"si3_prather subcycles must be 1, 2, or 3; got {subcycles}")
    out = contents
    out_moments = moments
    sr = nemo_source_round
    u_transport = sr(u_ice * dy)
    # SI3 records `ph_i` at the start of every subcycle, but `ph_i` is an
    # intensive work field not refreshed until ice_var_glo2eqv after the
    # entire dynamics call.  Therefore both subcycles see this same
    # step-entry thickness (icedyn_adv_pra.F90:142-160,355-367).
    volume = sr(out[..., ice_volume_index] / cell_area)
    concentration = sr(out[..., concentration_index] / cell_area)
    has_ice = concentration > _SI3_PRA_ICE_PRESENCE
    safe_concentration = sr(
        jnp.where(has_ice, concentration, _SI3_PRA_ONE)
    )
    thickness = sr(
        jnp.where(
            has_ice,
            sr(volume / safe_concentration),
            _SI3_PRA_ZERO,
        )
    )
    neighbors = [
        jnp.roll(jnp.roll(thickness, di, axis=0), dj, axis=1)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    ]
    h_max_all = sr(
        jnp.maximum(
            _SI3_PRA_AREA_FLOOR, jnp.max(jnp.stack(neighbors), axis=0)
        )
    )
    subcycle_dt = sr(dt / float(subcycles))
    for subcycle_index in range(1, subcycles + 1):
        ihls = 0 if subcycles == 1 else max(0, halo_width - subcycle_index)
        out, out_moments, _ = _si3_prather_x_substep(
            out,
            out_moments,
            u_transport,
            cell_area,
            wet,
            subcycle_dt,
            initial_area=cell_area,
            first_sweep=True,
            halo_width=halo_width,
            subcycle_index=subcycle_index,
            subcycles=subcycles,
        )
        # Hbig_pra.F90:946-1002: if transport creates ice thicker than the
        # pre-subcycle 9-point maximum at low concentration, increase only
        # concentration.  SI3 intentionally does not modify its moments here.
        volume = sr(out[..., ice_volume_index] / cell_area)
        concentration = sr(out[..., concentration_index] / cell_area)
        has_ice = concentration > _SI3_PRA_ZERO
        safe_concentration = sr(
            jnp.where(has_ice, concentration, _SI3_PRA_ONE)
        )
        thickness = sr(
            jnp.where(
                has_ice,
                sr(volume / safe_concentration),
                _SI3_PRA_ZERO,
            )
        )
        active = jnp.arange(out.shape[0])
        active = (active >= halo_width - ihls) & (active < out.shape[0] - halo_width + ihls)
        correct = (
            active[:, None]
            & (volume > _SI3_PRA_ZERO)
            & (concentration > _SI3_PRA_ZERO)
            & (thickness > h_max_all)
            & (concentration < _SI3_PRA_HBIG_CONCENTRATION_THRESHOLD)
        )
        corrected_concentration = sr(
            volume
            / sr(jnp.minimum(h_max_all, _SI3_PRA_HBIG_MAX_THICKNESS_M))
        )
        corrected_content = sr(corrected_concentration * cell_area)
        out = out.at[..., concentration_index].set(
            sr(
                jnp.where(
                    correct,
                    corrected_content,
                    out[..., concentration_index],
                )
            )
        )
    return out, out_moments, subcycles


def advect_si3_prather_2d(
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    cell_area: jnp.ndarray,
    wet: jnp.ndarray,
    dt: float,
    *,
    dx: float,
    dy: float,
    ice_step_index: int,
    nn_fsbc: int = 1,
    halo_width: int = 2,
    ice_volume_index: int | None = None,
    concentration_index: int | None = None,
    subcycles: int | None = None,
) -> tuple[jnp.ndarray, SI3PratherMoments, int]:
    """SI3 Prather x/y split for the shipped single-category ICE_ADV2D card.

    Odd ice steps start with x and even ice steps start with y; when CFL
    subcycling is active, the first direction alternates again with ``jt``.
    This is the exact selector at ``icedyn_adv_pra.F90:253-351``.
    """

    if contents.ndim != 3:
        raise ValueError("si3_prather requires (x, y, tracer) packed contents")
    if any(moment.shape != contents.shape for moment in moments):
        raise ValueError("si3_prather moment shapes must match packed contents")
    if u_ice.shape != contents.shape[:2] or v_ice.shape != u_ice.shape:
        raise ValueError("si3_prather velocity/contents shapes disagree")
    if cell_area.shape != u_ice.shape or wet.shape != u_ice.shape:
        raise ValueError("si3_prather metric/mask shapes disagree")
    if ice_volume_index is None or concentration_index is None:
        raise ValueError("si3_prather requires volume and concentration indices for Hbig")
    if ice_step_index < 1 or nn_fsbc < 1:
        raise ValueError("si3_prather requires positive ice_step_index and nn_fsbc")
    if subcycles is None:
        cfl = max(
            float(jnp.max(jnp.abs(u_ice)) * dt / dx),
            float(jnp.max(jnp.abs(v_ice)) * dt / dy),
        )
        subcycles = (
            3
            if cfl > _SI3_PRA_CFL_THREE_CYCLE_THRESHOLD
            else 2
            if cfl > _SI3_PRA_CFL_TWO_CYCLE_THRESHOLD
            else 1
        )
    if subcycles not in (1, 2, 3):
        raise ValueError(f"si3_prather subcycles must be 1, 2, or 3; got {subcycles}")

    out = contents
    out_moments = moments
    sr = nemo_source_round
    u_transport = sr(u_ice * dy)
    v_transport = sr(v_ice * dx)
    volume = sr(out[..., ice_volume_index] / cell_area)
    concentration = sr(out[..., concentration_index] / cell_area)
    has_ice = concentration > _SI3_PRA_ICE_PRESENCE
    safe_concentration = sr(
        jnp.where(has_ice, concentration, _SI3_PRA_ONE)
    )
    thickness = sr(
        jnp.where(
            has_ice,
            sr(volume / safe_concentration),
            _SI3_PRA_ZERO,
        )
    )
    neighbors = [
        jnp.roll(jnp.roll(thickness, di, axis=0), dj, axis=1)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    ]
    h_max_all = sr(
        jnp.maximum(
            _SI3_PRA_AREA_FLOOR, jnp.max(jnp.stack(neighbors), axis=0)
        )
    )
    subcycle_dt = sr(dt / float(subcycles))

    for subcycle_index in range(1, subcycles + 1):
        area = cell_area
        x_first = ((ice_step_index - 1) // nn_fsbc) % 2 == ((subcycle_index - 1) % 2)
        if x_first:
            out, out_moments, area = _si3_prather_x_substep(
                out,
                out_moments,
                u_transport,
                cell_area,
                wet,
                subcycle_dt,
                initial_area=area,
                first_sweep=True,
                halo_width=halo_width,
                subcycle_index=subcycle_index,
                subcycles=subcycles,
            )
            out, out_moments, area = _si3_prather_y_substep(
                out,
                out_moments,
                v_transport,
                cell_area,
                wet,
                subcycle_dt,
                initial_area=area,
                first_sweep=False,
                halo_width=halo_width,
                subcycle_index=subcycle_index,
                subcycles=subcycles,
            )
        else:
            out, out_moments, area = _si3_prather_y_substep(
                out,
                out_moments,
                v_transport,
                cell_area,
                wet,
                subcycle_dt,
                initial_area=area,
                first_sweep=True,
                halo_width=halo_width,
                subcycle_index=subcycle_index,
                subcycles=subcycles,
            )
            out, out_moments, area = _si3_prather_x_substep(
                out,
                out_moments,
                u_transport,
                cell_area,
                wet,
                subcycle_dt,
                initial_area=area,
                first_sweep=False,
                halo_width=halo_width,
                subcycle_index=subcycle_index,
                subcycles=subcycles,
            )

        volume = sr(out[..., ice_volume_index] / cell_area)
        concentration = sr(out[..., concentration_index] / cell_area)
        has_ice = concentration > _SI3_PRA_ZERO
        safe_concentration = sr(
            jnp.where(has_ice, concentration, _SI3_PRA_ONE)
        )
        thickness = sr(
            jnp.where(
                has_ice,
                sr(volume / safe_concentration),
                _SI3_PRA_ZERO,
            )
        )
        ihls = 0 if subcycles == 1 else max(0, halo_width - subcycle_index)
        active_x = jnp.arange(out.shape[0])
        active_x = (active_x >= halo_width - ihls) & (active_x < out.shape[0] - halo_width + ihls)
        active_y = jnp.arange(out.shape[1])
        active_y = (active_y >= halo_width - ihls) & (active_y < out.shape[1] - halo_width + ihls)
        correct = (
            active_x[:, None]
            & active_y[None, :]
            & (volume > _SI3_PRA_ZERO)
            & (concentration > _SI3_PRA_ZERO)
            & (thickness > h_max_all)
            & (concentration < _SI3_PRA_HBIG_CONCENTRATION_THRESHOLD)
        )
        corrected_concentration = sr(
            volume
            / sr(jnp.minimum(h_max_all, _SI3_PRA_HBIG_MAX_THICKNESS_M))
        )
        corrected_content = sr(corrected_concentration * cell_area)
        out = out.at[..., concentration_index].set(
            sr(
                jnp.where(
                    correct,
                    corrected_content,
                    out[..., concentration_index],
                )
            )
        )
        # The shipped root ICE_ADV2D domain is bi-periodic.  NEMO refreshes
        # contents and all five moments after every `jt` at
        # icedyn_adv_pra.F90:432-479.
        out = _si3_periodic_halo_xy(out, halo_width)
        out_moments = cast(
            SI3PratherMoments,
            tuple(_si3_periodic_halo_xy(value, halo_width) for value in out_moments),
        )
    return out, out_moments, subcycles
