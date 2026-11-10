"""Ocean baroclinic tendencies written once over the grid operator contract.

Boussinesq z* ocean, split-explicit (test-only pilot: the models still run
their own tendencies).  One form per unified term, shared by the lat-lon
C-grid and the MPAS / TRiSK mesh::

    div_k   = div(h_face u mask_face)                      continuity per level
    w       = diagnose_w_from_flux_div(div_k)               vertical velocity
    deta/dt = -sum_k div_k                                  free surface
    u'      = u - sum_k(h_face u) / sum_k(h_face)           baroclinic split
    du/dt   = -grad(KE + p'/rho_0) [+ partial-cell PGF correction]
              + per-grid momentum terms + bottom drag      -> finalize
              + d/dz(A_v du'/dz) + sponge
    dT/dt   = K_h L(T) - K_bih L(L(T)) + d/dz(K_v dT/dz) + sponge

``L(q) = mask_cell div(h_face mask_face grad q) / h`` (thickness weighted, so
``sum(area h dq)`` closes to round-off where ``h`` varies; ``h_face`` is the
continuity face thickness) and ``div`` / ``grad`` are the B2
operator contract (:mod:`legoesm.grids.operator_protocol`).  Vertical
diffusion is explicit only when ``implicit_vertical_mixing`` is off.

Grid-specific numerics are plug-ins (:class:`OceanPEOperators`): geometry and
density (each grid's EOS / land-fill call of the shared
``iterate_eos_and_pressure_anomaly``), face thickness, kinetic energy, the
partial-cell PGF correction, the Coriolis / PV-flux + momentum advection +
viscosity block, cell-to-face averaging, and the returned-tendency convention
(MPAS removes the depth mean and returns it as ``F_slow``; lat-lon returns the
full tendency); terms before ``finalize`` enter MPAS's ``F_slow``, the
explicit ``A_v`` and the sponge after it do not (both models' own order).
Builders: ``latlon_ocean_pe_operators`` in
:mod:`.ocean_pe_latlon_cgrid` and ``mpas_ocean_pe_operators`` in
:mod:`.ocean_pe_mpas`.

KNOWN DIFFERENCES between the grids, kept as plug-ins (voted): continuity face
thickness (lat-lon MIN of the two cells, MPAS MEAN on z*) and the depth-mean
divisor floor (lat-lon ``1e-10``, MPAS ``max(H, min_water_column_m)``).

KNOWN DIFFERENCE from both models (user decision, measured in the tests): the
models' lateral tracer diffusion ``K mask_cell div(mask_face grad q)`` is not
thickness weighted and leaks heat and salt where ``h`` varies; the core uses
the thickness-weighted form ``L`` above.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.ocean.dynamics.ocean_tendency_common import apply_sponge_tracer_relaxation
from legoesm.ocean.physics.mixing import flux_divergence_zero_flux
from legoesm.ocean.vertical import diagnose_w_from_flux_div

Faces = tuple[jax.Array, ...]

#: Floor on layer / interface thickness in the explicit vertical diffusion and
#: bottom drag [m] (MPAS form, voted).
THICKNESS_FLOOR_M = 1e-10


class OceanPECoeffs(NamedTuple):
    """Scalar coefficients of the unified terms, resolved from a grid config."""

    rho_0: float
    K_h: float
    K_bih: float
    K_v: float
    A_v: float
    implicit_vertical_mixing: bool
    bottom_drag_r: float


class OceanPEContext(NamedTuple):
    """Fields the core hands to the per-grid momentum and finalize plug-ins."""

    vel: Faces
    vel_prime: Faces
    h_k: jax.Array
    jacobian: jax.Array
    rho_prime: jax.Array
    w: jax.Array
    h_face: Faces           # continuity face thickness
    h_face_metric: Faces    # depth-mean / PV-flux face thickness


class OceanPEOperators(NamedTuple):
    """B2 grid operators, masks and the per-grid plug-ins."""

    grid_ops: Any
    coeffs: OceanPECoeffs
    dz_ref: jax.Array
    dz_half_ref: jax.Array
    z_coord: Any
    cell_mask: jax.Array
    face_masks: Faces                   # 3-D wet masks, one per velocity family
    face_masks_2d: Faces                # column wet masks
    geometry_density: Callable[[], tuple]           # -> (J, h_k, rho', p'_filled)
    face_thickness: Callable[[jax.Array, Faces], tuple[Faces, Faces]]
    depth_mean_divisor: Callable[[jax.Array], jax.Array]
    cell_to_face: Callable[[jax.Array], Faces]
    kinetic_energy: Callable[[Faces], jax.Array]
    pgf_correction: Callable[[Faces, OceanPEContext], Faces]
    momentum_terms: Callable[[Faces, OceanPEContext], Faces]
    finalize: Callable[[Faces, OceanPEContext], tuple[Faces, Any]]


class OceanPETendencies(NamedTuple):
    dvel: Faces
    dT: jax.Array
    dS: jax.Array
    deta: jax.Array
    extra: Any          # per-grid finalize output (MPAS: F_slow per edge)


def _as_tuple(x: Any) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def continuity(vel: Faces, h_face: Faces, face_masks: Faces, grid_ops: Any,
               z_coord: Any) -> tuple[jax.Array, jax.Array]:
    """Per-level thickness-flux divergence and the diagnosed ``w``."""
    div_k = grid_ops.divergence(*(h * u * m for u, h, m in zip(vel, h_face, face_masks)))
    return div_k, diagnose_w_from_flux_div(div_k, z_coord, thickness_weighted=True)


def baroclinic_split(vel: Faces, h_face: Faces, face_masks_2d: Faces,
                     divisor: Callable[[jax.Array], jax.Array]) -> Faces:
    """Perturbation velocity ``u - sum_k(h u) / divisor(sum_k h)``."""
    out = []
    for u, h, m in zip(vel, h_face, face_masks_2d):
        pair = jnp.sum(jnp.stack([u * h, h], axis=-1), axis=-2)
        out.append(u - (pair[..., 0] / divisor(pair[..., 1]) * m)[..., None])
    return tuple(out)


def thickness_weighted_laplacian(q: jax.Array, grid_ops: Any, h_face: Faces,
                                 face_masks: Faces, h: jax.Array,
                                 cell_mask: jax.Array) -> jax.Array:
    """``mask_cell div(h_face mask_face grad q) / max(h, floor)`` for ``q`` of
    shape ``(*cells, ncol)``; ``sum(area h L(q))`` telescopes to zero."""
    grad = _as_tuple(grid_ops.gradient(q))
    flux = grid_ops.divergence(*(hf * g * m for hf, g, m in zip(h_face, grad, face_masks)))
    return flux / jnp.maximum(h, THICKNESS_FLOOR_M) * cell_mask


def vertical_diffusion(q: jax.Array, coeff: float, dz_ref: jax.Array,
                       dz_half_ref: jax.Array, jacobian: jax.Array) -> jax.Array:
    """Explicit ``d/dz(coeff dq/dz)``, zero flux at surface and bottom.

    ``q`` is ``(..., nlev)`` and ``jacobian`` broadcasts as ``(..., 1)``."""
    if q.shape[-1] < 2:
        return jnp.zeros_like(q)
    dz_half = jnp.maximum(dz_half_ref * jacobian, THICKNESS_FLOOR_M)
    dz = jnp.maximum(dz_ref * jacobian, THICKNESS_FLOOR_M)
    return flux_divergence_zero_flux(coeff * (q[..., :-1] - q[..., 1:]) / dz_half, dz)


def bottom_drag(vel: Faces, j_face: Faces, dz_bottom: jax.Array, r: float,
                face_masks_2d: Faces) -> Faces:
    """Linear drag ``-r u / h`` on the deepest level (z*), zero elsewhere."""
    out = []
    for u, jf, m in zip(vel, j_face, face_masks_2d):
        h_bot = jnp.maximum(dz_bottom * jf, THICKNESS_FLOOR_M)
        out.append(jnp.zeros_like(u).at[..., -1].set(-r * u[..., -1] / h_bot * m))
    return tuple(out)


def ocean_baroclinic_tendencies(vel: Faces, T: jax.Array, S: jax.Array,
                                ops: OceanPEOperators, sponge: Any = None,
                                sponge_vel_ref: Faces | None = None) -> OceanPETendencies:
    """Baroclinic tendencies ``(dvel, dT, dS, deta, extra)`` on any grid.

    ``vel`` is a tuple of face-velocity families ``(*faces, nlev)``; ``T`` and
    ``S`` are ``(*cells, nlev)``.  ``sponge`` (optional ``SpongeForcing``)
    relaxes T and S; ``sponge_vel_ref`` (one reference or ``None`` per velocity
    family) adds ``gamma_face (ref - u)`` with ``sponge.gamma`` horizontal."""
    c = ops.coeffs
    gops = ops.grid_ops
    cell_mask_3d = ops.cell_mask[..., None]

    jacobian, h_k, rho_prime, p_prime = ops.geometry_density()
    h_face, h_face_metric = ops.face_thickness(h_k, vel)
    div_k, w = continuity(vel, h_face, ops.face_masks, gops, ops.z_coord)
    deta = -jnp.sum(div_k, axis=-1) * ops.cell_mask
    vel_prime = baroclinic_split(vel, h_face_metric, ops.face_masks_2d,
                                 ops.depth_mean_divisor)
    ctx = OceanPEContext(vel, vel_prime, h_k, jacobian, rho_prime, w, h_face, h_face_metric)

    # Momentum: -grad(KE + p'/rho_0), per-grid terms, drag, then the grid's
    # returned-tendency convention, then explicit A_v and the sponge.
    grad = _as_tuple(gops.gradient(ops.kinetic_energy(vel) + p_prime / c.rho_0))
    grad = ops.pgf_correction(grad, ctx)
    dvel = ops.momentum_terms(tuple(-g for g in grad), ctx)
    j_face = ops.cell_to_face(jacobian)
    if c.bottom_drag_r > 0:
        drag = bottom_drag(vel, j_face, ops.dz_ref[-1], c.bottom_drag_r, ops.face_masks_2d)
        dvel = tuple(d + b for d, b in zip(dvel, drag))
    dvel, extra = ops.finalize(dvel, ctx)
    if not c.implicit_vertical_mixing:
        dvel = tuple(
            d + vertical_diffusion(up, c.A_v, ops.dz_ref, ops.dz_half_ref, jf[..., None]) * m
            for d, up, jf, m in zip(dvel, vel_prime, j_face, ops.face_masks))

    # Tracers: horizontal Laplacian / biharmonic on the (T, S) stack.
    tr = jnp.stack([T, S], axis=-1)
    flat = tr.reshape(*tr.shape[:-2], -1)
    dflat = jnp.zeros_like(flat)
    if c.K_h > 0 or c.K_bih > 0:
        # Flat column index is 2 k + tracer; a level-broadcast mask (last dim 1) stays.
        def rep(a):
            return a if a.shape[-1] == 1 else jnp.repeat(a, 2, axis=-1)

        hf_rep = tuple(rep(a) for a in h_face)
        fm_rep = tuple(rep(m) for m in ops.face_masks)
        h_rep = rep(h_k)

        def lap(q):
            return thickness_weighted_laplacian(q, gops, hf_rep, fm_rep, h_rep, cell_mask_3d)

        lap1 = lap(flat)
        if c.K_h > 0:
            dflat = dflat + c.K_h * lap1
        if c.K_bih > 0:
            dflat = dflat - c.K_bih * lap(lap1)
    dtr = dflat.reshape(tr.shape)
    if not c.implicit_vertical_mixing:
        dtr = dtr + jax.vmap(
            lambda q: vertical_diffusion(q, c.K_v, ops.dz_ref, ops.dz_half_ref,
                                         jacobian[..., None]),
            in_axes=-1, out_axes=-1)(tr)
    dT, dS = dtr[..., 0], dtr[..., 1]

    if sponge is not None:
        dT, dS = apply_sponge_tracer_relaxation(dT, dS, T, S, sponge, mask=None,
                                                expand_gamma_axis=-1)
        if sponge_vel_ref is not None and any(r is not None for r in sponge_vel_ref):
            if sponge.gamma.ndim == T.ndim:
                raise ValueError(
                    "3-D (full-rank per-cell) SpongeForcing.gamma supports tracer "
                    "relaxation only; a momentum reference needs a horizontal gamma.")
            g_face = ops.cell_to_face(sponge.gamma.astype(T.dtype))
            dvel = tuple(d if ref is None else d + gf[..., None] * (ref.astype(T.dtype) - u)
                         for d, u, ref, gf in zip(dvel, vel, sponge_vel_ref, g_face))

    dvel = tuple(d * m for d, m in zip(dvel, ops.face_masks))
    return OceanPETendencies(dvel, dT * cell_mask_3d, dS * cell_mask_3d, deta, extra)
