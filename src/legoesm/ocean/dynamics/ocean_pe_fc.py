"""FC-Gram Boussinesq Ocean PE on the cubed-sphere.

Replaces all horizontal operators with FC spectral versions.
Everything else is identical to ocean_pe_cdgrid: Wright EOS, hydrostatic
pressure, layer thickness, w diagnosis, skew-symmetric momentum,
vertical advection, vertical diffusion, land masking.

Divergence damping is applied when fc_config.div_damp_2 or
fc_config.div_damp_4 are nonzero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_fc import FCOperatorConfig, _fc_pad_halo_vector
from legoesm.grids.halo import pad_halo_4d
from legoesm.core.operators_fc_3d import (
    fc_curl_z_3d,
    fc_gradient_x_3d,
    fc_gradient_y_3d,
    fc_divergence_3d,
    fc_hyperdiffusion_3d,
    fc_scalar_advection_3d,
    fc_laplacian_3d,
    fc_divergence_damping_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_4d
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.physics.mixing import vertical_diffusion
from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
)


def ocean_baroclinic_tendencies_fc(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    fc_config: FCOperatorConfig,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
    surface_forcing=None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies with FC spectral operators.

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
    fc_config : FCOperatorConfig
    config : OceanConfig
    physics_fn : callable, optional

    Returns
    -------
    OceanTendencies
    """
    u = state.u.data
    v = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    mask_3d = mask[..., jnp.newaxis]

    g = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS ---
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta_safe, z_coord.dz_ref, J, rho_0, g,
    )
    rho = wright_eos(T, S, p_hydro)
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient ---
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    # Pre-pad (p_prime, K) once and share across both gradient pairs.
    # Both are 3D scalar fields on (6, n, n, nlev); ``pad_halo_4d`` and
    # the FC index-space derivatives treat the trailing axis as a
    # passive batch.  Stack along trailing axis to (6, n, n, nlev, 2),
    # fold to (6, n, n, nlev*2), and run a single halo + paired x/y
    # gradient call.  Halves the halo cost vs the previous Loop 78
    # form (which already shared halo within each gradient pair but
    # not across the two scalar fields).
    K = 0.5 * (u**2 + v**2)
    n_face_pK, n_i_pK, n_j_pK, nlev_pK = p_prime.shape
    _pK_stack = jnp.stack([p_prime, K], axis=-1)
    _pK_flat = _pK_stack.reshape(n_face_pK, n_i_pK, n_j_pK, nlev_pK * 2)
    _pK_pad = pad_halo_4d(_pK_flat, halo=1, interp_offsets=grid.halo_interp_offsets)
    _dpK_dx_flat = fc_gradient_x_3d(_pK_flat, grid, fc_config, padded=_pK_pad)
    _dpK_dy_flat = fc_gradient_y_3d(_pK_flat, grid, fc_config, padded=_pK_pad)
    _dpK_dx = _dpK_dx_flat.reshape(
        _dpK_dx_flat.shape[0], _dpK_dx_flat.shape[1],
        _dpK_dx_flat.shape[2], nlev_pK, 2,
    )
    _dpK_dy = _dpK_dy_flat.reshape(
        _dpK_dy_flat.shape[0], _dpK_dy_flat.shape[1],
        _dpK_dy_flat.shape[2], nlev_pK, 2,
    )
    dp_dx = _dpK_dx[..., 0]
    dK_dx = _dpK_dx[..., 1]
    dp_dy = _dpK_dy[..., 0]
    dK_dy = _dpK_dy[..., 1]

    # --- 4. Diagnose w from flux divergence (batched flux + bare div) ---
    # Both ``flux_div_k`` and ``div_v`` are FC divergences on (u-component,
    # v-component) pairs that share the (6, n, n, nlev) shape and use the
    # same vector halo + metric weights.  Stack the two u-inputs and the
    # two v-inputs along a new trailing axis, fold to (6, n, n, nlev*2),
    # call ``fc_divergence_3d`` once on the thicker tensor, and unfold.
    # The trailing axis is purely passive: the vector rotation in
    # ``_fc_pad_halo_vector`` broadcasts ``cos_angle/sin_angle`` over the
    # trailing axis via ``[..., None]``, the metric weights broadcast the
    # same way, and the FC index-space derivatives operate on axis 1/2
    # only.  2 fc_divergence calls → 1 (one shared vector halo
    # exchange + one fused derivative + one final divide).
    n_face_d, n_i_d, n_j_d, nlev_d = u.shape
    u_masked = u * mask_3d
    v_masked = v * mask_3d
    _div_u_pair = jnp.stack([h_k * u_masked, u_masked], axis=-1)
    _div_v_pair = jnp.stack([h_k * v_masked, v_masked], axis=-1)
    _div_u_pair_flat = _div_u_pair.reshape(n_face_d, n_i_d, n_j_d, nlev_d * 2)
    _div_v_pair_flat = _div_v_pair.reshape(n_face_d, n_i_d, n_j_d, nlev_d * 2)
    # Pre-pad (combined_u, combined_v) ONCE so the divergence and the
    # curl below share the FC vector halo exchange.  ``stack(..., axis=-1)
    # + reshape`` interleaves [h*u, u, h*u, u, ...], so the
    # ``[..., 1::2]`` slot of the padded array is the padded ``u_masked``
    # — exactly what ``fc_curl_z_3d`` needs.  Saves one full
    # ``_fc_pad_halo_vector`` collective per RHS evaluation (a vector
    # halo with cross-face cos/sin rotation, costlier than a scalar
    # halo) — same Loop 134 exploit as the FC laplacian/hyperdiff
    # share.
    _combined_u_pad, _combined_v_pad = _fc_pad_halo_vector(
        _div_u_pair_flat, _div_v_pair_flat, grid,
    )
    _div_pair_flat = fc_divergence_3d(
        _div_u_pair_flat, _div_v_pair_flat,
        grid, fc_config,
        padded=(_combined_u_pad, _combined_v_pad),
    ).reshape(n_face_d, n_i_d, n_j_d, nlev_d, 2)
    flux_div_k = _div_pair_flat[..., 0]
    div_v = _div_pair_flat[..., 1]
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    # --- 5. Vorticity ---
    # Reuse the ``u_masked`` / ``v_masked`` padded slices from the
    # combined halo above — slot index 1 of the interleaved
    # ``[h*u, u, h*u, u, ...]`` layout.
    _u_masked_pad = _combined_u_pad[..., 1::2]
    _v_masked_pad = _combined_v_pad[..., 1::2]
    zeta = fc_curl_z_3d(
        u_masked, v_masked, grid, fc_config,
        padded=(_u_masked_pad, _v_masked_pad),
    )

    # --- 6. Kinetic energy gradient (computed via the batched
    # (p_prime, K) gradient block above — halo and derivative shared
    # with p_prime, halving the cost of each timestep).

    # --- 7. Vector-invariant momentum (skew-symmetric) ---
    # H_total + U_bar + V_bar all reduce ``... * h_k`` over the level
    # axis — fuse into one stacked column reduction.
    _bar_triple = jnp.sum(
        jnp.stack([h_k, u * h_k, v * h_k], axis=-1), axis=-2,
    )
    H_total = jnp.maximum(_bar_triple[..., 0], min_water_col)
    U_bar = _bar_triple[..., 1] / H_total * mask
    V_bar = _bar_triple[..., 2] / H_total * mask
    u_prime = (u - U_bar[..., jnp.newaxis]) * mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * mask_3d
    f_3d = grid.f[..., jnp.newaxis]

    du_dt = (zeta * v + f_3d * v_prime - dK_dx
             - 0.5 * u * div_v - dp_dx / rho_0)
    dv_dt = (-zeta * u - f_3d * u_prime - dK_dy
             - 0.5 * v * div_v - dp_dy / rho_0)

    # --- Divergence damping (only if fc_config requests it) ---
    # ``(u_masked, v_masked)`` halo is already produced by the
    # combined-pad block above (Loop 173) — slot-1 of the interleaved
    # ``[h*u, u, h*u, u, ...]`` layout.  Pass it via ``padded=`` so
    # the inner divergence inside ``fc_divergence_damping_3d`` skips
    # its own ``_fc_pad_halo_vector`` collective (Loop 176).
    if fc_config.div_damp_2 > 0 or fc_config.div_damp_4 > 0:
        du_damp, dv_damp = fc_divergence_damping_3d(
            u_masked, v_masked, grid, fc_config,
            padded=(_u_masked_pad, _v_masked_pad),
        )
        du_dt = du_dt + du_damp
        dv_dt = dv_dt + dv_damp

    # --- 8. Vertical advection of u, v ---
    # Batch (u, v) via leading-axis stack so the velocity-independent
    # shared work (``w_full`` / ``jac_safe`` / ``dz_half``) runs once
    # and the upwind gradient broadcasts across the new axis.  Same
    # leading-axis batching as Loop 142 / 162.
    _uv_va = jnp.stack([u, v], axis=0)
    _uv_va_adv = _vertical_advection_ocean(_uv_va, w, z_coord, J)
    du_dt = du_dt + _uv_va_adv[0]
    dv_dt = dv_dt + _uv_va_adv[1]

    # --- 9. Tracer tendencies ---
    # Stack T, S along a trailing tracer axis and fold it into the level
    # axis so halo-issuing FC operators (fc_scalar_advection_3d,
    # fc_laplacian_3d, fc_hyperdiffusion_3d) — now 4D-native via
    # pad_halo_4d (Loop 65) — run ONCE for both tracers instead of being
    # re-entered under vmap-over-(T,S).  Vertical operators stay
    # per-tracer because they hard-code the vertical axis at -1.
    tracer_stack = jnp.stack([T, S], axis=-1)  # (6, n, n, nlev, 2)
    n_face, n_i, n_j, nlev_t, n_tracers = tracer_stack.shape
    tracer_flat = tracer_stack.reshape(n_face, n_i, n_j, nlev_t * n_tracers)
    # Broadcast masked velocities across the combined (level × tracer)
    # axis.  ``tracer_flat`` reshape interleaves levels and tracers as
    # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]``, so each level's
    # velocity must be duplicated ``n_tracers`` times to align.
    # ``jnp.repeat`` does this directly; ``jnp.tile`` would concatenate
    # the entire array and mis-align tracer ↔ level.
    u_masked = u * mask_3d
    v_masked = v * mask_3d
    if n_tracers == 1:
        u_b, v_b = u_masked, v_masked
    else:
        u_b = jnp.repeat(u_masked, n_tracers, axis=-1)
        v_b = jnp.repeat(v_masked, n_tracers, axis=-1)

    # Pre-pad ``tracer_flat`` ONCE up-front and share the halo across
    # ``fc_scalar_advection_3d`` (Loop 177 — internal x/y gradients
    # share q's halo) AND the Laplacian/hyperdiff branches below.
    # Three halo-issuing calls on the same input collapse to a single
    # ``pad_halo_4d`` collective per RHS evaluation when
    # diffusion is on (Loop 178 extension of Loops 134/177).
    tracer_flat_pad = pad_halo_4d(
        tracer_flat, halo=1, interp_offsets=grid.halo_interp_offsets,
    )
    horiz_flat = fc_scalar_advection_3d(
        tracer_flat, u_b, v_b, grid, fc_config, padded=tracer_flat_pad,
    )
    if physics_fn is None and config.K_h > 0:
        horiz_flat = horiz_flat + fc_laplacian_3d(
            tracer_flat, grid, fc_config, padded=tracer_flat_pad,
        ) * config.K_h
    if physics_fn is None and config.hyperdiff_coeff > 0:
        horiz_flat = horiz_flat + fc_hyperdiffusion_3d(
            tracer_flat, grid, fc_config, config.hyperdiff_coeff,
            padded=tracer_flat_pad,
        )
    horiz_stack = horiz_flat.reshape(n_face, n_i, n_j, nlev_t, n_tracers)

    # Vertical advection per-tracer (vmap over the trailing axis so JAX
    # produces one batched kernel rather than n_tracers unrolled stencils).
    def _vert_adv(q):
        return _vertical_advection_ocean(q, w, z_coord, J)

    vert_adv_stack = jax.vmap(_vert_adv, in_axes=-1, out_axes=-1)(tracer_stack)

    if physics_fn is None and config.K_v > 0:
        def _vdiff(q):
            return vertical_diffusion(q, z_coord, J, config.K_v)

        vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
        tracer_tend_stack = horiz_stack + vert_adv_stack + vdiff_stack
    else:
        tracer_tend_stack = horiz_stack + vert_adv_stack

    dT_dt = tracer_tend_stack[..., 0]
    dS_dt = tracer_tend_stack[..., 1]

    # --- 10. Mixing ---
    # Stack u, v along a trailing axis and fold it into the level dim so
    # the halo-issuing horizontal viscosity operators (fc_laplacian_3d,
    # fc_hyperdiffusion_3d, both 4D-native via pad_halo_4d in Loop 65)
    # run ONCE on the thicker (6, n, n, nlev*2) field instead of issuing
    # two separate halo MPI exchanges per call.  Vertical diffusion stays
    # per-component (axis -1 = nlev hard-coded, no halo).
    if physics_fn is None:
        n_face_v, n_i_v, n_j_v, nlev_v = u.shape
        if config.A_h > 0 or config.hyperdiff_coeff > 0:
            vel_masked_stack = jnp.stack(
                [u * mask_3d, v * mask_3d], axis=-1,
            )  # (6, n, n, nlev, 2)
            vel_masked_flat = vel_masked_stack.reshape(
                n_face_v, n_i_v, n_j_v, nlev_v * 2,
            )
            # Pre-pad ONCE so the explicit Laplacian and the inner
            # Laplacian of the biharmonic hyperdiffusion share the halo
            # on ``vel_masked_flat`` instead of issuing two independent
            # ``pad_halo_4d`` collectives on the same input.  Same
            # halo-sharing pattern as the CD-grid ocean (Loop 133).
            vel_masked_pad = pad_halo_4d(
                vel_masked_flat, halo=1,
                interp_offsets=grid.halo_interp_offsets,
            )
        if config.A_h > 0:
            vel_lap_flat = (
                fc_laplacian_3d(
                    vel_masked_flat, grid, fc_config,
                    padded=vel_masked_pad,
                ) * config.A_h
            )
            vel_lap = vel_lap_flat.reshape(n_face_v, n_i_v, n_j_v, nlev_v, 2)
            du_dt = du_dt + vel_lap[..., 0]
            dv_dt = dv_dt + vel_lap[..., 1]
        if config.A_v > 0:
            def _vdiff_uv(q):
                return vertical_diffusion(q, z_coord, J, config.A_v)

            vel_uv = jnp.stack([u, v], axis=-1)
            vel_vdiff = jax.vmap(_vdiff_uv, in_axes=-1, out_axes=-1)(vel_uv)
            du_dt = du_dt + vel_vdiff[..., 0]
            dv_dt = dv_dt + vel_vdiff[..., 1]

        if config.hyperdiff_coeff > 0:
            vel_hyper_flat = fc_hyperdiffusion_3d(
                vel_masked_flat, grid, fc_config, config.hyperdiff_coeff,
                padded=vel_masked_pad,
            )
            vel_hyper = vel_hyper_flat.reshape(
                n_face_v, n_i_v, n_j_v, nlev_v, 2,
            )
            du_dt = du_dt + vel_hyper[..., 0]
            dv_dt = dv_dt + vel_hyper[..., 1]
    else:
        phys = physics_fn(state, grid, z_coord, surface_forcing)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 12. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 13. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(
            data=jnp.zeros_like(H_bathy), name="dH_bathy_dt",
            dims=dims_2d, units="m/s",
        ),
        dland_mask_dt=Field(
            data=jnp.zeros_like(mask), name="dland_mask_dt",
            dims=dims_2d, units="1/s",
        ),
    )
