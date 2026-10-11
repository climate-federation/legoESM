"""NEMO shallow-water-engine explicit free-surface RK3 stages.

This is a selectable program in the shared lat-lon C-grid lane, not a
testcase-specific model.  The first landed stage transcribes the compiled SWG
``stprk3.f90:164-210`` path; later stages extend the same dispatch.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.core.source_rounding import nemo_source_round
from legoesm.grids.latlon import ensure_geometry
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
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


def _native_to_u(native):
    zero = jnp.zeros_like(native[:, :1])
    return jnp.concatenate([zero, native], axis=1)[..., jnp.newaxis]


def _native_to_v(native):
    zero = jnp.zeros_like(native[:1])
    return jnp.concatenate([zero, native], axis=0)[..., jnp.newaxis]


def _native_north(field):
    return jnp.concatenate([field[1:], jnp.zeros_like(field[:1])], axis=0)


def _validate_stage1(config, z_coord):
    vor = config.vorticity_scheme
    if config.outer_integrator != "swe_rk3":
        raise ValueError(
            "SWE RK3 stage replay requires outer_integrator='swe_rk3'")
    if (config.momentum_advection != "vector_invariant"
            or config.vertical_momentum_scheme != "nemo_advective"
            or config.ke_gradient_scheme != "c2"):
        raise ValueError(
            "SWE RK3 stage 1 requires NEMO vector C2 advection (KEG + ZAD)")
    if (not isinstance(vor, ENSVorticityConfig)
            or vor.scheme != "ens_total" or vor.ln_dynvor_msk is not True):
        raise ValueError(
            "SWE RK3 stage 1 requires total ENS with ln_dynvor_msk=true")
    if int(z_coord.n_levels) != 1:
        raise ValueError("landed SWE RK3 stage 1 supports exactly one layer")
    if config.constants.g != config.g:
        raise ValueError("SWE RK3 requires one gravity value in config.constants.g")


def swe_rk3_stage1(entry, dt, grid, z_coord, land_mask, config):
    """Execute NEMO SWE RK3 stage 1 from an admitted native-A2D entry.

    Source order is the compiled P3 program: zero RHS; C2 KEG then the
    structurally-zero one-layer ZAD; total ENS; inline reduced-gravity
    pressure; ``ssh_nxt``; ``dom_qco_r3c``; before-level velocity update.
    No positive-thickness floor is evaluated on this path.
    """
    _validate_stage1(config, z_coord)
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
    u_before = _native_to_u(jnp.asarray(entry.u_before, dtype=dtype))
    v_before = _native_to_v(jnp.asarray(entry.v_before, dtype=dtype))

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
        r3f_override=entry.r3f)

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

    # Inline shallow-water pressure gradient.  Gravity comes only from the card.
    ssh = jnp.asarray(entry.ssh_k, dtype=dtype)
    grav = jnp.asarray(config.constants.g, dtype=dtype)
    hpg_u_native = sr(sr(-grav * sr(jnp.roll(ssh, -1, axis=1) - ssh)) * r1_e1u)
    hpg_v_native = sr(sr(-grav * sr(_native_north(ssh) - ssh)) * r1_e2v)
    rhs_u = sr(rhs_u + _native_to_u(hpg_u_native))
    rhs_v = sr(rhs_v + _native_to_v(hpg_v_native))

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
    r1_area_t = sr(one / jnp.asarray(ops.area_t, dtype=dtype))
    e3_hdiv = nemo_transport_wzv_divergence_level(
        flux_u, flux_u_west, flux_v, flux_v_south, r1_area_t,
        live_e3t, tmask3)
    stage_dt = sr(jnp.asarray(dt, dtype=dtype) / jnp.asarray(3.0, dtype=dtype))
    zhdiv = jnp.sum(e3_hdiv, axis=-1)
    ssh_after = sr(sr(ssh - sr(stage_dt * zhdiv)) * mask)

    # dom_qco_r3c output ratios use the shared, unfloored QCO builders.
    r3t_after = nemo_qco_r3t_ratio(
        z_coord, ssh_after, jnp.asarray(ops.e3t_0[..., 0]) * mask,
        evaluation="nemo_reciprocal")
    face_after = nemo_qco_live_face_geometry_from_operands(
        ssh_after, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)
    _, r3f_after = nemo_qco_live_vorticity_e3f_cgrid(
        ssh_after, z_coord, dtype, nn_e3f_typ=1, grid=grid,
        return_ratio=True)

    # Vector-form velocity update from Nbb, then native U/V record layout.
    u_after = sr(u_before + sr(stage_dt * rhs_u) * umask3)
    v_after = sr(v_before + sr(stage_dt * rhs_v) * vmask3)
    return SWERK3StageOutput(
        ssh_after, u_after[:, 1:, 0], v_after[1:, :, 0], r3t_after,
        face_after.r3u, face_after.r3v, r3f_after,
        jnp.asarray(entry.u_before, dtype=dtype),
        jnp.asarray(entry.v_before, dtype=dtype),
        jnp.asarray(entry.utau_before, dtype=dtype),
        jnp.asarray(entry.vtau_before, dtype=dtype),
        jnp.asarray(entry.utau_now, dtype=dtype),
        jnp.asarray(entry.vtau_now, dtype=dtype),
    )
