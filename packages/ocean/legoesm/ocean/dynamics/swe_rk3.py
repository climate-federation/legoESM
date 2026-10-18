"""NEMO shallow-water-engine explicit free-surface RK3 stages.

This is a selectable program in the shared lat-lon C-grid lane, not a
testcase-specific model.  The three stages transcribe the compiled SWG
``stprk3.f90:141-367`` path.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import dctn, idctn
from legoesm.core.source_rounding import nemo_source_round
from legoesm.grids.latlon import ensure_geometry
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    nemo_lateral_viscosity_coefficients,
    nemo_lateral_viscosity_constant_coefficients,
    nemo_ldf_lap_viscosity_e3_cgrid,
    nemo_vor_ene_vorticity_cgrid,
    pv_flux_ens,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_c2_kinetic_energy,
    nemo_transport_wzv_divergence_level,
)
from legoesm.ocean.eos import nemo_qco_r3t_ratio
from legoesm.ocean.state import ENSVorticityConfig
from legoesm.ocean.vertical import (
    nemo_ldf_reference_e3f,
    nemo_qco_live_face_geometry_from_operands,
    nemo_qco_live_vorticity_e3f_cgrid,
    nemo_qco_resolved_mesh_operands,
    nemo_qco_vorticity_f_cgrid,
)


class SWERK3StageEntry(NamedTuple):
    """The thirteen native-A2D fields in one admitted SWE stage entry."""

    ssh_k: object
    u_k: object
    v_k: object
    r3t_k: object
    r3u_k: object
    r3v_k: object
    r3f: object
    u_before: object
    v_before: object
    utau_before: object
    vtau_before: object
    utau_now: object
    vtau_now: object


SWERK3StageOutput = SWERK3StageEntry

SWE_SURFACE_MODES = ("explicit", "rigid_lid")


class SWERigidLidGeometry(NamedTuple):
    """Static closed-rectangle data of the rigid-lid projection."""

    wet_bounds: tuple[int, int, int, int]
    spacing_m: float
    laplacian_eigenvalue_m2: object


def wet_rectangle_bounds(mask) -> tuple[int, int, int, int]:
    """Inclusive ``(j0, j1, i0, i1)`` of a mask that is one wet rectangle."""
    wet = np.asarray(mask, dtype=bool)
    jj, ii = np.where(wet)
    if wet.ndim != 2 or jj.size == 0:
        raise ValueError("projection mask must be a nonempty 2-D rectangle")
    bounds = int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())
    expected = np.zeros_like(wet)
    expected[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] = True
    if not np.array_equal(wet, expected):
        raise ValueError("projection mask is not one rectangle")
    return bounds


def swe_rigid_lid_geometry(grid, z_coord) -> SWERigidLidGeometry:
    """Projection data of one flat, constant-spacing, closed-rectangle card.

    Static (numpy) so the projection itself stays jit- and grad-safe.
    """
    geom = ensure_geometry(grid)
    wet = np.asarray(z_coord.is_active, dtype=bool)
    if wet.ndim != 3 or wet.shape[-1] != 1:
        raise ValueError("rigid-lid projection requires exactly one layer")
    bounds = wet_rectangle_bounds(wet[..., 0])
    if bounds[0] < 1 or bounds[2] < 1:
        raise ValueError(
            "rigid-lid projection requires a land row and column before "
            "the wet rectangle")
    spacings = np.concatenate([
        np.asarray(getattr(geom, name), dtype=np.float64).ravel()
        for name in ("dx_u", "dy_v", "dx_v", "dy_u")])
    spacing = float(spacings[0])
    if not np.isfinite(spacing) or spacing <= 0.0 \
            or not np.all(spacings == spacing):
        raise ValueError(
            "rigid-lid projection requires one constant grid spacing")
    thickness = np.asarray(z_coord.nemo_e3t_0, dtype=np.float64)[wet]
    if thickness.size == 0 or not np.all(thickness == thickness[0]):
        raise ValueError(
            "rigid-lid projection requires one constant layer thickness")
    j0, j1, i0, i1 = bounds
    ny, nx = j1 - j0 + 1, i1 - i0 + 1
    ky = np.arange(ny, dtype=np.float64)
    kx = np.arange(nx, dtype=np.float64)
    eigenvalue = (
        2.0 * (np.cos(np.pi * ky / ny) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * kx / nx) - 1.0)[None, :]
    ) / spacing ** 2
    eigenvalue[0, 0] = 1.0
    return SWERigidLidGeometry(bounds, spacing, jnp.asarray(eigenvalue))


def swe_rigid_lid_projection(u_native, v_native,
                             geometry: SWERigidLidGeometry):
    """Project native U/V onto the closed-wall non-divergent subspace.

    Solves the Neumann problem ``lap(phi) = div(u, v)`` on the wet rectangle
    exactly with a DCT-II (``phi`` has zero wet-area mean, zero on land) and
    returns ``(u - dphi/dx, v - dphi/dy, phi)``.  ``phi`` carries the input
    units times metres: a velocity potential [m2 s-1] for velocity input, a
    pressure potential [m2 s-2] for an acceleration.  Wall faces are never
    touched, so wall-normal flow stays whatever the input carried (zero on
    the card).  The operator is the jnp twin of the offline
    ``closed_wall_divergent_part``.
    """
    j0, j1, i0, i1 = geometry.wet_bounds
    dx = geometry.spacing_m
    u = jnp.asarray(u_native)
    v = jnp.asarray(v_native)
    divergence = (u - jnp.roll(u, 1, axis=1) + v - jnp.roll(v, 1, axis=0)) / dx
    rhs = divergence[j0:j1 + 1, i0:i1 + 1]
    rhs = rhs - jnp.mean(rhs)
    phi_hat = dctn(rhs, type=2, norm="ortho") / geometry.laplacian_eigenvalue_m2
    phi_hat = phi_hat.at[0, 0].set(0.0)
    phi = idctn(phi_hat, type=2, norm="ortho")
    u_div = jnp.zeros_like(u).at[j0:j1 + 1, i0:i1].set(
        jnp.diff(phi, axis=1) / dx)
    v_div = jnp.zeros_like(v).at[j0:j1, i0:i1 + 1].set(
        jnp.diff(phi, axis=0) / dx)
    phi_full = jnp.zeros_like(u).at[j0:j1 + 1, i0:i1 + 1].set(phi)
    return u - u_div, v - v_div, phi_full


def swe_rigid_lid_stage_output(output, entry, stage, stage_dt, geometry):
    """Rigid-lid closure of one certified stage output.

    The stage ran on a zero-ssh entry, so its pressure gradient was zero and
    every other term is the certified program.  Its ssh update (divergence of
    the non-divergent entry) is round-off; the lid replaces it by exactly
    zero and keeps the entry's zero-ssh thickness ratios.  The stage velocity
    is projected; with a non-divergent before-velocity the removed part is
    ``stage_dt`` times the gradient of the returned pressure potential
    [m2 s-2].
    """
    u_nd, v_nd, potential = swe_rigid_lid_projection(
        output.u_k, output.v_k, geometry)
    phi = potential / stage_dt
    before_u, before_v = output.u_before, output.v_before
    if stage == 3:
        before_u, before_v = u_nd, v_nd
    return output._replace(
        ssh_k=jnp.zeros_like(output.ssh_k), u_k=u_nd, v_k=v_nd,
        r3t_k=entry.r3t_k, r3u_k=entry.r3u_k, r3v_k=entry.r3v_k,
        r3f=entry.r3f, u_before=before_u, v_before=before_v), phi



def _native_to_u(native):
    zero = jnp.zeros_like(native[:, :1])
    return jnp.concatenate([zero, native], axis=1)[..., jnp.newaxis]


def _native_to_v(native):
    zero = jnp.zeros_like(native[:1])
    return jnp.concatenate([zero, native], axis=0)[..., jnp.newaxis]


def _native_north(field):
    return jnp.concatenate([field[1:], jnp.zeros_like(field[:1])], axis=0)


def swe_rotated_potential_forcing(
        potential_f, e2u, e1v, e3u, e3v, umask, vmask, vertex_mask):
    """Map an F-point transport potential to non-divergent U/V acceleration.

    ``potential_f`` is a volume-transport tendency potential (m3 s-2). Its
    matched differences are divided by the live face cross-sections.  The
    returned accelerations therefore make exactly cancelling contributions to
    the native C-grid continuity stencil.  Multiplying by ``vertex_mask``
    imposes zero potential on boundary vertices; this gives zero wall-normal
    force and makes the basin-integrated transport force telescope to zero.

    Coordinates and signs are native NEMO C-grid: positive U is east and
    positive V is north.  A positive northward potential difference produces
    positive (eastward) U force; a positive eastward difference produces
    negative (southward) V force.  Thus the map is the clockwise rotated
    gradient ``(d potential/dy, -d potential/dx)``.
    """
    potential = jnp.asarray(potential_f) * (jnp.asarray(vertex_mask) != 0)
    south = jnp.concatenate(
        [jnp.zeros_like(potential[:1]), potential[:-1]], axis=0)
    west = jnp.concatenate(
        [jnp.zeros_like(potential[:, :1]), potential[:, :-1]], axis=1)
    # East/north convention: +d_y(potential) -> +east U and
    # +d_x(potential) -> -north V (clockwise rotated gradient).
    transport_u = (potential - south) * jnp.asarray(umask)
    transport_v = -(potential - west) * jnp.asarray(vmask)
    wet_u = jnp.asarray(umask) != 0
    wet_v = jnp.asarray(vmask) != 0
    cross_u = jnp.where(
        wet_u, jnp.asarray(e2u) * jnp.asarray(e3u), 1.0)
    cross_v = jnp.where(
        wet_v, jnp.asarray(e1v) * jnp.asarray(e3v), 1.0)
    force_u = jnp.where(wet_u, transport_u / cross_u, 0.0)
    force_v = jnp.where(wet_v, transport_v / cross_v, 0.0)
    return force_u, force_v


def _validate_program(config, z_coord, stage):
    vor = config.vorticity_scheme
    if config.outer_integrator != "swe_rk3":
        raise ValueError(
            "SWE RK3 stage replay requires outer_integrator='swe_rk3'")
    if config.swe_surface not in SWE_SURFACE_MODES:
        raise ValueError(
            "SWE RK3 requires config.swe_surface stated as one of "
            f"{SWE_SURFACE_MODES}; got {config.swe_surface!r}")
    if stage not in (1, 2, 3):
        raise ValueError(f"unknown SWE RK3 stage {stage!r}; expected 1, 2, or 3")
    if (config.momentum_advection != "vector_invariant"
            or config.vertical_momentum_scheme != "nemo_advective"
            or config.ke_gradient_scheme != "c2"):
        raise ValueError(
            "SWE RK3 requires NEMO vector C2 advection (KEG + ZAD)")
    if (not isinstance(vor, ENSVorticityConfig)
            or vor.scheme != "ens_total" or vor.ln_dynvor_msk is not True):
        raise ValueError(
            "SWE RK3 requires total ENS with ln_dynvor_msk=true")
    if int(z_coord.n_levels) != 1:
        raise ValueError("SWE RK3 supports exactly one layer")
    if config.constants.g != config.g:
        raise ValueError("SWE RK3 requires one gravity value in config.constants.g")
    if (config.lateral_viscosity_operator != "nemo_div_curl"
            or config.lateral_viscosity_e3_weighting != "nemo_e3"):
        raise ValueError(
            "SWE RK3 requires NEMO's e3-weighted div-curl viscosity")
    if config.lateral_viscosity_coefficient_source not in (
            "nemo_ldf_c2d", "nemo_constant"):
        raise ValueError(
            "SWE RK3 requires a computed NEMO viscosity coefficient source")


def swe_rk3_stage(entry, before_entry, dt, stage, grid, z_coord, land_mask,
                  config, potential_f=None):
    """Execute one NEMO SWE RK3 stage from native-A2D stage/before entries.

    Source order is the compiled P3 program: zero RHS; C2 KEG then the
    structurally-zero one-layer ZAD; total ENS; inline reduced-gravity
    pressure; ``ssh_nxt``; ``dom_qco_r3c``; before-level velocity update.
    Stage 3 additionally applies NEMO's e3-weighted Laplacian, centred wind
    stress, and linear drag on the whole-step before velocity.  No
    positive-thickness floor is evaluated on this path.
    """
    _validate_program(config, z_coord, stage)
    sr = nemo_source_round
    dtype = jnp.asarray(entry.ssh_k).dtype
    geom = ensure_geometry(grid)
    mask = jnp.asarray(land_mask, dtype=dtype)
    tmask3 = jnp.asarray(z_coord.is_active, dtype=dtype)
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    umask3 = jnp.asarray(umask3, dtype=dtype)
    vmask3 = jnp.asarray(vmask3, dtype=dtype)
    ops = nemo_qco_resolved_mesh_operands(
        z_coord, grid, umask3, vmask3, dtype, 1)

    u = _native_to_u(jnp.asarray(entry.u_k, dtype=dtype))
    v = _native_to_v(jnp.asarray(entry.v_k, dtype=dtype))
    u_before = _native_to_u(jnp.asarray(before_entry.u_k, dtype=dtype))
    v_before = _native_to_v(jnp.asarray(before_entry.v_k, dtype=dtype))

    # key_qco live face thicknesses consume the recorded stage-entry ratios.
    one = jnp.asarray(1.0, dtype=dtype)
    e3u_native = sr(ops.e3u_0 * sr(
        one + jnp.asarray(entry.r3u_k, dtype=dtype)[..., None] * ops.umask3))
    e3v_native = sr(ops.e3v_0 * sr(
        one + jnp.asarray(entry.r3v_k, dtype=dtype)[..., None] * ops.vmask3))
    h_u = jnp.concatenate([e3u_native[:, -1:], e3u_native], axis=1)
    h_v = jnp.concatenate([jnp.zeros_like(e3v_native[:1]), e3v_native], axis=0)
    h_vtx = nemo_qco_live_vorticity_e3f_cgrid(
        entry.ssh_k, z_coord, dtype, nn_e3f_typ=1, grid=grid,
        r3f_override=entry.r3f, apply_fe3mask=False)

    # dyn_adv: C2 kinetic-energy gradient, then one-layer ZAD (= zero).
    ke = nemo_c2_kinetic_energy(u, v)[..., 0]
    r1_e1u = sr(one / jnp.asarray(geom.dx_u[:, 1:], dtype=dtype))
    r1_e2v = sr(one / jnp.asarray(geom.dy_v[1:, :], dtype=dtype))
    keg_u_native = -sr(sr(jnp.roll(ke, -1, axis=1) - ke) * r1_e1u)
    keg_v_native = -sr(sr(_native_north(ke) - ke) * r1_e2v)
    rhs_u = _native_to_u(keg_u_native)
    rhs_v = _native_to_v(keg_v_native)
    rhs_u = sr(rhs_u + jnp.zeros_like(rhs_u))
    rhs_v = sr(rhs_v + jnp.zeros_like(rhs_v))

    # dyn_vor: total ENS, direct e3f division, with the compiled relative mask.
    zeta = nemo_vor_ene_vorticity_cgrid(u, v, grid)
    f_vtx = nemo_qco_vorticity_f_cgrid(z_coord, dtype)
    vtx_mask = compute_vertex_mask(mask, grid=grid)
    vor_u, vor_v = pv_flux_ens(
        zeta, h_vtx, h_v, v, h_u, u, umask3, vmask3, vtx_mask,
        ln_dynvor_msk=True, f_vtx=f_vtx, q_boundary="nemo_live",
        metric_widths=(geom.dx_u, geom.dx_v, geom.dy_u, geom.dy_v),
        metric_reciprocals=(
            sr(one / jnp.asarray(geom.dx_u, dtype=dtype)),
            sr(one / jnp.asarray(geom.dy_v, dtype=dtype))),
    )
    rhs_u = sr(rhs_u + vor_u)
    rhs_v = sr(rhs_v + vor_v)

    # dyn_ldf is called only in stage 3 and reads Kbb for the input velocity
    # and divergence thickness, but Kmm (the current stage) in the output
    # curl denominators (compiled dynldf_lev.f90:130-150).
    if stage == 3:
        before_r3t = jnp.asarray(before_entry.r3t_k, dtype=dtype)
        before_r3u = jnp.asarray(before_entry.r3u_k, dtype=dtype)
        before_r3v = jnp.asarray(before_entry.r3v_k, dtype=dtype)
        e3t_before = sr(ops.e3t_0 * sr(one + before_r3t[..., None]))
        e3u_before_native = sr(
            ops.e3u_0 * sr(one + before_r3u[..., None] * ops.umask3))
        e3v_before_native = sr(
            ops.e3v_0 * sr(one + before_r3v[..., None] * ops.vmask3))
        e3u_before = jnp.concatenate(
            [jnp.zeros_like(e3u_before_native[:, :1]), e3u_before_native], axis=1)
        e3v_before = jnp.concatenate(
            [jnp.zeros_like(e3v_before_native[:1]), e3v_before_native], axis=0)
        reference_e3f = nemo_ldf_reference_e3f(z_coord)
        # r3f has no time-level dimension. dom_qco_r3c overwrites it after
        # every stage, so stage 3 reads the stage-2/current ratio while the
        # divergence operands above remain at Kbb (dynldf_lev.f90:130-150).
        e3f_current = nemo_qco_live_vorticity_e3f_cgrid(
            entry.ssh_k, z_coord, dtype, nn_e3f_typ=1, grid=grid,
            reference_e3f=reference_e3f, r3f_override=entry.r3f)
        if config.lateral_viscosity_coefficient_source == "nemo_constant":
            ahmt, ahmf = nemo_lateral_viscosity_constant_coefficients(
                geom, jnp.asarray(config.lateral_viscosity.A_h, dtype=dtype))
        elif config.lateral_viscosity_coefficient_source == "nemo_ldf_c2d":
            half_uv = (jnp.asarray(config.lateral_viscosity.A_h, dtype=dtype)
                       / (jnp.asarray(geom.radius, dtype=dtype)
                          * jnp.asarray(geom.dlon, dtype=dtype)))
            ahmt, ahmf = nemo_lateral_viscosity_coefficients(geom, half_uv)
        else:  # defended by _validate_program; keep dispatch fail-closed here.
            raise ValueError(
                "unknown SWE RK3 viscosity coefficient source "
                f"{config.lateral_viscosity_coefficient_source!r}")
        ldf_u, ldf_v = nemo_ldf_lap_viscosity_e3_cgrid(
            u_before, v_before, geom, ahmt, ahmf, e3t_before,
            mask=mask, u_mask=umask3, v_mask=vmask3,
            vertex_mask=compute_vertex_mask(mask, grid=grid),
            thickness_operands=(
                e3t_before, e3u_before, e3v_before, e3f_current,
                h_u, h_v))
        rhs_u = sr(rhs_u + ldf_u)
        rhs_v = sr(rhs_v + ldf_v)

    # Inline shallow-water pressure gradient.  Gravity comes only from the card.
    ssh = jnp.asarray(entry.ssh_k, dtype=dtype)
    grav = jnp.asarray(config.constants.g, dtype=dtype)
    hpg_u_native = sr(sr(-grav * sr(jnp.roll(ssh, -1, axis=1) - ssh)) * r1_e1u)
    hpg_v_native = sr(sr(-grav * sr(_native_north(ssh) - ssh)) * r1_e2v)
    zrhs_u_native = hpg_u_native
    zrhs_v_native = hpg_v_native

    # Stage-3-only wind and layer friction.  Stress is positive in the ocean
    # momentum direction; the positive wind source and negative drag sink use
    # NEMO-native U/V. Drag uses whole-step Kbb, not stage K.  Preserve the
    # compiled scalar association before adding zrhs once to the carried RHS
    # (P3 stprk3.f90:319-331).
    if stage == 3:
        r1_rho0 = sr(one / jnp.asarray(config.constants.rho_0, dtype=dtype))
        half_r1_rho0 = sr(jnp.asarray(0.5, dtype=dtype) * r1_rho0)
        tau_u = sr(jnp.asarray(entry.utau_before, dtype=dtype)
                   + jnp.asarray(entry.utau_now, dtype=dtype))
        tau_v = sr(jnp.asarray(entry.vtau_before, dtype=dtype)
                   + jnp.asarray(entry.vtau_now, dtype=dtype))
        wind_u = sr(sr(half_r1_rho0 * tau_u) / e3u_native[..., 0])
        wind_v = sr(sr(half_r1_rho0 * tau_v) / e3v_native[..., 0])
        drag = jnp.asarray(config.bottom_drag.bottom_drag_r, dtype=dtype)
        drag_u = sr(drag * jnp.asarray(before_entry.u_k, dtype=dtype))
        drag_v = sr(drag * jnp.asarray(before_entry.v_k, dtype=dtype))
        zrhs_u_native = sr(sr(zrhs_u_native + wind_u) - drag_u)
        zrhs_v_native = sr(sr(zrhs_v_native + wind_v) - drag_v)

    rhs_u = sr(rhs_u + _native_to_u(zrhs_u_native))
    rhs_v = sr(rhs_v + _native_to_v(zrhs_v_native))

    # The learned lane is an optional additive transport-potential tendency.
    # Keep the ordinary certified program structurally untouched when absent:
    # no zero addition and no changed source-rounding association are allowed.
    if potential_f is not None:
        closure_u, closure_v = swe_rotated_potential_forcing(
            potential_f,
            jnp.asarray(ops.e2u, dtype=dtype),
            jnp.asarray(ops.e1v, dtype=dtype),
            e3u_native[..., 0], e3v_native[..., 0],
            umask3[:, 1:, 0], vmask3[1:, :, 0], vtx_mask[1:, 1:])
        rhs_u = rhs_u + _native_to_u(closure_u)
        rhs_v = rhs_v + _native_to_v(closure_v)

    # ssh_nxt: div_hor(Kbb,Kbb), one live layer, emp=0 on this deck.
    u_native = u[:, 1:, :]
    v_native = v[1:, :, :]
    e2u = jnp.asarray(ops.e2u, dtype=dtype)[..., None]
    e1v = jnp.asarray(ops.e1v, dtype=dtype)[..., None]
    flux_u = sr(sr(e2u * e3u_native) * u_native)
    flux_v = sr(sr(e1v * e3v_native) * v_native)
    flux_u_west = jnp.roll(flux_u, 1, axis=1)
    flux_v_south = jnp.concatenate(
        [jnp.zeros_like(flux_v[:1]), flux_v[:-1]], axis=0)
    live_e3t = sr(ops.e3t_0 * sr(
        one + jnp.asarray(entry.r3t_k, dtype=dtype)[..., None]))
    r1_area_t = sr(one / jnp.asarray(ops.area_t, dtype=dtype))[..., None]
    e3_hdiv = nemo_transport_wzv_divergence_level(
        flux_u, flux_u_west, flux_v, flux_v_south, r1_area_t,
        live_e3t, tmask3)
    divisor = (3.0, 2.0, 1.0)[stage - 1]
    stage_dt = sr(jnp.asarray(dt, dtype=dtype)
                  / jnp.asarray(divisor, dtype=dtype))
    zhdiv = jnp.sum(e3_hdiv, axis=-1)
    ssh_before = jnp.asarray(before_entry.ssh_k, dtype=dtype)
    ssh_after = sr(sr(ssh_before - sr(stage_dt * zhdiv)) * mask)

    # dom_qco_r3c output ratios use the shared, unfloored QCO builders.
    r3t_after = nemo_qco_r3t_ratio(
        z_coord, ssh_after, jnp.asarray(ops.e3t_0[..., 0]) * mask,
        evaluation="nemo_reciprocal")
    face_after = nemo_qco_live_face_geometry_from_operands(
        ssh_after, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)
    _, r3f_after = nemo_qco_live_vorticity_e3f_cgrid(
        ssh_after, z_coord, dtype, nn_e3f_typ=1, grid=grid,
        return_ratio=True, source_associated=True)

    # Vector-form velocity update from Nbb, then native U/V record layout.
    u_after = sr(u_before + sr(stage_dt * rhs_u) * umask3)
    v_after = sr(v_before + sr(stage_dt * rhs_v) * vmask3)
    next_before_u = (u_after[:, 1:, 0] if stage == 3
                     else jnp.asarray(before_entry.u_k, dtype=dtype))
    next_before_v = (v_after[1:, :, 0] if stage == 3
                     else jnp.asarray(before_entry.v_k, dtype=dtype))
    return SWERK3StageOutput(
        ssh_after, u_after[:, 1:, 0], v_after[1:, :, 0], r3t_after,
        face_after.r3u, face_after.r3v, r3f_after,
        next_before_u, next_before_v,
        jnp.asarray(entry.utau_before, dtype=dtype),
        jnp.asarray(entry.vtau_before, dtype=dtype),
        jnp.asarray(entry.utau_now, dtype=dtype),
        jnp.asarray(entry.vtau_now, dtype=dtype),
    )


def swe_rk3_stage1(entry, dt, grid, z_coord, land_mask, config):
    """Backward-compatible stage-1 wrapper; the shared kernel owns arithmetic."""
    return swe_rk3_stage(
        entry, entry, dt, 1, grid, z_coord, land_mask, config)


def _entry_from_state(state, stress_before, stress_now, grid, z_coord, config):
    """Build the native-A2D stage boundary consumed by the production kernel."""
    dtype = state.eta.data.dtype
    mask = state.land_mask.data
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    ops = nemo_qco_resolved_mesh_operands(
        z_coord, grid, umask3, vmask3, dtype, 1)
    r3t = nemo_qco_r3t_ratio(
        z_coord, state.eta.data, jnp.asarray(ops.e3t_0[..., 0]) * mask,
        evaluation="nemo_reciprocal")
    face = nemo_qco_live_face_geometry_from_operands(
        state.eta.data, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)
    _, r3f = nemo_qco_live_vorticity_e3f_cgrid(
        state.eta.data, z_coord, dtype, nn_e3f_typ=1, grid=grid,
        return_ratio=True)
    u = state.u.data[:, 1:, 0]
    v = state.v.data[1:, :, 0]
    return SWERK3StageEntry(
        state.eta.data, u, v, r3t, face.r3u, face.r3v, r3f,
        u, v, stress_before[0], stress_before[1], stress_now[0], stress_now[1])


def swe_rk3_step(state, dt, surface_forcing, grid, z_coord, config,
                 stage_runner=None, stage_potential_fn=None):
    """Advance the shared ocean state with the three-stage SWE RK3 program.

    The surface caller supplies grid-native U/V stresses through
    ``tau_i_native``/``tau_j_native``.  This keeps the shared stage program
    independent of any testcase-specific analytic forcing formula.
    """
    _validate_program(config, z_coord, 1)
    if surface_forcing is None:
        raise ValueError(
            "SWE RK3 step requires grid-native surface stress forcing")
    tau_u = getattr(surface_forcing, "tau_i_native", None)
    tau_v = getattr(surface_forcing, "tau_j_native", None)
    if tau_u is None or tau_v is None:
        raise ValueError(
            "SWE RK3 step requires both tau_i_native and tau_j_native")
    dtype = state.eta.data.dtype
    now = (jnp.asarray(tau_u, dtype=dtype), jnp.asarray(tau_v, dtype=dtype))
    before = (
        now[0] if state.tau_x_prev is None else jnp.asarray(state.tau_x_prev),
        now[1] if state.tau_y_prev is None else jnp.asarray(state.tau_y_prev),
    )
    rigid_lid = config.swe_surface == "rigid_lid"
    if rigid_lid:
        # The lid holds ssh at zero: the boundary entry carries zero ssh and
        # zero thickness ratios whatever eta the state arrived with.
        lid = swe_rigid_lid_geometry(grid, z_coord)
        state = state._replace(
            eta=state.eta.replace(data=jnp.zeros_like(state.eta.data)))
    boundary = _entry_from_state(state, before, now, grid, z_coord, config)
    if stage_runner is None:
        def run(entry, before_entry, stage):
            potential = (None if stage_potential_fn is None
                         else stage_potential_fn(entry, stage))
            return swe_rk3_stage(
                entry, before_entry, dt, stage, grid, z_coord,
                state.land_mask.data, config, potential_f=potential)
    else:
        if stage_potential_fn is not None:
            raise ValueError(
                "stage_potential_fn cannot be combined with stage_runner")

        def run(entry, before_entry, stage):
            return stage_runner(entry, dt, stage, before_entry)
    lid_phi = []
    if rigid_lid:
        certified_run = run

        def run(entry, before_entry, stage):
            stage_dt = jnp.asarray(dt, dtype=dtype) / (3.0, 2.0, 1.0)[stage - 1]
            output, phi = swe_rigid_lid_stage_output(
                certified_run(entry, before_entry, stage), entry, stage,
                stage_dt, lid)
            lid_phi.append(phi)
            return output
    s1 = run(boundary, boundary, 1)
    s2 = run(s1, boundary, 2)
    s3 = run(s2, boundary, 3)
    u = _native_to_u(s3.u_k)
    v = _native_to_v(s3.v_k)
    state = state._replace(
        eta=state.eta.replace(data=s3.ssh_k),
        u=state.u.replace(data=u), v=state.v.replace(data=v),
        tau_x_prev=now[0], tau_y_prev=now[1])
    return state if not rigid_lid else state._replace(rigid_lid_phi=lid_phi[-1])
