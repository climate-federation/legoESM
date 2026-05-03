"""MPAS ocean baroclinic tendencies using TRiSK operators.

Boussinesq hydrostatic primitive equations in vector-invariant form
on MPAS Voronoi (C-grid) meshes. Uses the TRiSK discretization from
Ringler et al. (2010).

Equations (per layer k):
    du/dt = q_e * F_q - grad(KE + p'/ρ₀ + g·η) - w·du'/dz + A_h·del2(u) + B_h·del4(u) - del2(A_smag·del2(u)) + A_v·d²u/dz²
    d(h·T)/dt = -div(h·u·T) + K_h·h·lap(T) - K_bih·h·bilap(T) + K_v·d²T/dz²
    d(h·S)/dt = -div(h·u·S) + K_h·h·lap(S) - K_bih·h·bilap(S) + K_v·d²S/dz²
    dη/dt = -Σ_k div(h_k · u_k)

TRiSK split status (see issue #160)
-----------------------------------
The PV-flux term below uses q = ζ_rel / h (relative vorticity of the
perturbation velocity only), NOT the full PV q = (f+ζ_total)/h_total
required by the Ringler-Thuburn-Skamarock-Klemp energy-conserving
identity. The planetary Coriolis force f × u is applied separately as
a forward-backward (Matsuno) step in the step function. This split
was introduced when closing #103 (MPAS ocean depth-mean Coriolis
double-counting), and it trades one inconsistency for another: the
TRiSK energy-conservation property is preserved only for the
perturbation subsystem, the Rossby-wave β coupling on the barotropic
mode is underrepresented, and the transport paired with q inside
pv_flux_*_conserving_3d is h·u' rather than the continuity-equation
flux h·u_total. Fixing this cleanly requires the MOM6-style
slow-forcing refactor tracked in issue #160.

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Ringler, T. D., et al. (2013). Ocean Modelling, 69, 211-232.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.core.operators_voronoi import (
    apvm_correction_3d,
    biharmonic_vorticity_del4_3d,
    bilaplacian_cell_3d,
    divergence_cell_3d,
    gradient_edge_3d,
    curl_vertex_3d,
    kinetic_energy_cell_3d,
    potential_vorticity_vertex_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    smagorinsky_biharmonic_3d,
    leith_biharmonic_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
    vertex_thickness_3d,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    diagnose_w_from_flux_div,
    vertical_advection_ocean,
    flux_form_vertical_momentum_advection,
)
from legoesm.ocean.freshwater import FreshwaterForcing
from legoesm.ocean.dynamics.ocean_tendency_common import (
    apply_freshwater_virtual_salt_top,
    apply_sponge_tracer_relaxation,
    iterate_eos_and_pressure_anomaly,
)


def mpas_ocean_baroclinic_tendencies(
    state: MPASOceanState,
    mesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig = MPASOceanConfig(),
    freshwater: FreshwaterForcing | None = None,
    physics_fn=None,
    surface_forcing=None,
    sponge=None,
) -> MPASOceanTendencies:
    """Compute baroclinic (slow) tendencies for MPAS ocean.

    Parameters
    ----------
    state : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    freshwater : FreshwaterForcing or None
        Freshwater forcing. If None, no freshwater terms are applied.
    sponge : SpongeForcing, optional
        Sponge layer relaxation fields.

    Returns
    -------
    MPASOceanTendencies
    """
    g = config.g
    rho_0 = config.rho_0

    u_3d = state.u.data          # (nEdges, nlev)
    T_3d = state.T.data          # (nCells, nlev)
    S_3d = state.S.data          # (nCells, nlev)
    eta = state.eta.data         # (nCells,)
    H_bathy = state.H_bathy.data  # (nCells,)
    mask = state.land_mask.data  # (nCells,)

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)

    from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

    def _fill_land_cells_mpas(field_cell, mask_cell):
        return fill_land_cells_mpas(field_cell, mask_cell, c1, c2)

    # ---- Layer thickness and Jacobian ----
    jacobian = compute_ocean_jacobian(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )  # (nCells, nlev)

    # ---- Density and hydrostatic pressure ----
    # Fill land-cell T/S with ocean-neighbor values before EOS so that
    # density on land ≈ ρ₀, preventing spurious ρ' at coastlines.
    # Uses the reference Jacobian (J=1, eta=0): the barotropic solver
    # handles g*grad(eta) and using actual J here would double-count it.
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T_3d, S_3d, mask,
        lambda field: _fill_land_cells_mpas(field, mask),
        eos_fn, z_coord.dz_ref, rho_0, g,
        n_iter=2,
    )

    # Fill land cells in p_prime before gradient_edge so the 2-cell
    # stencil sees smooth values at coastlines.
    p_prime = _fill_land_cells_mpas(p_prime, mask)

    # ---- Edge mask for land boundaries ----
    edge_mask = mask[c1] * mask[c2]  # 1 only if both cells are ocean

    # ---- Depth-averaged velocity and perturbation ----
    # The baroclinic step must operate on PERTURBATION velocity
    # u' = u - u_bar to avoid double-counting with the barotropic
    # solver.  The barotropic solver handles the depth-mean Coriolis,
    # pressure gradient, and KE; the baroclinic step handles only the
    # vertical shear (perturbation) component.
    # (Matches latlon C-grid: ocean_pe_latlon_cgrid.py:256-266)
    h_e_3d = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    # Both ``H_e`` and ``u_bar`` numerator share the ``h_e_3d`` weight
    # on the level axis — fuse into one stacked column reduction.
    _u_pair = jnp.sum(jnp.stack([h_e_3d, u_3d * h_e_3d], axis=-1), axis=1)
    H_e = jnp.maximum(_u_pair[..., 0], config.min_water_column_m)
    u_bar = _u_pair[..., 1] / jnp.maximum(H_e, 1e-10)
    u_bar = u_bar * edge_mask  # (nEdges,)
    u_prime_3d = u_3d - u_bar[:, jnp.newaxis]  # (nEdges, nlev)

    # ---- Thickness flux and vertical velocity ----
    # Compute flux divergence BEFORE momentum tendencies because we need
    # w for vertical advection of momentum (issue #152).
    thickness_flux = u_3d * h_e_3d * edge_mask[:, jnp.newaxis]  # (nEdges, nlev)
    div_flux = divergence_cell_3d(thickness_flux, mesh)  # (nCells, nlev)

    # Diagnose w from full-velocity flux divergence (matching latlon pattern:
    # ocean_pe_latlon_cgrid.py:302-305)
    w = diagnose_w_from_flux_div(
        div_flux, z_coord, thickness_weighted=True,
    )  # (nCells, nlev+1)

    # ---- Momentum tendencies (batched 3D) ----
    # MOM6-style split (#160): compute the FULL nonlinear tendency with
    # TOTAL velocity and TOTAL PV = (f + ζ) / h.  Coriolis enters
    # exclusively through the PV flux — no separate substep needed.
    # The depth-mean → F_slow_u for the barotropic solver; the
    # baroclinic perturbation gets the deviation.

    # Kinetic energy from TOTAL velocity
    ke = kinetic_energy_cell_3d(u_3d, mesh)  # (nCells, nlev)

    # Bernoulli function: KE(u) + p'/rho_0
    bernoulli = ke + p_prime / rho_0  # (nCells, nlev)

    # Pressure gradient + Bernoulli — when scalar tracer diffusion is on
    # (``K_h > 0``) we *also* need ``∇T`` and ``∇S`` for the harmonic
    # diffusion downstream.  All three gradients use the same
    # ``cellsOnEdge`` gather + finite-difference (the trailing axis
    # passes through passively), so concatenate ``bernoulli`` with the
    # ``(T, S)`` tracer pack along the trailing axis and run
    # ``gradient_edge_3d`` once on the thicker tensor.  Saves one
    # gradient call (1 ``cellsOnEdge`` gather + 1 ``dcEdge`` divide) per
    # RHS evaluation when ``K_h > 0`` — same passive trailing-axis
    # exploit as Loop 139 in the MPAS atmosphere PE.
    nlev_B = bernoulli.shape[-1]
    # ``_tracer_grad_flat_pre`` is needed both for the K_h Laplacian
    # *and* for the inner gradient of the K_bih biharmonic, so the
    # batched gradient must include the (T, S) channels when *either*
    # coefficient is non-zero.
    _need_tracer_grad = config.K_h > 0 or config.K_bih > 0
    if _need_tracer_grad:
        _tracer_flat_pre = jnp.stack([T_3d, S_3d], axis=-1).reshape(
            T_3d.shape[0], nlev_B * 2,
        )
        _btr_input = jnp.concatenate(
            [bernoulli, _tracer_flat_pre], axis=-1,
        )  # (nCells, nlev*(1 + n_tracers))
        _btr_grad = gradient_edge_3d(_btr_input, mesh)
        grad_B = _btr_grad[:, :nlev_B]
        _tracer_grad_flat_pre = _btr_grad[:, nlev_B:]
    else:
        grad_B = gradient_edge_3d(bernoulli, mesh)  # (nEdges, nlev)
        _tracer_grad_flat_pre = None

    # PV flux: RELATIVE vorticity only, q = ζ(u)/h. Planetary Coriolis is
    # applied separately (a) as online f·v_t(u_bar) in the barotropic
    # substep and (b) as a forward-backward Matsuno correction on the
    # 3D perturbation in the step() function. This matches the lat-lon
    # C-grid pattern and sidesteps the frozen-Coriolis instability that
    # follows from carrying planetary Coriolis in the baroclinic-step
    # depth-mean forcing (τ ~ 1/f ≈ 0.2 days at mid-latitudes; see #160).
    zero_f = jnp.zeros_like(mesh.fVertex)
    q_relative = potential_vorticity_vertex_3d(
        u_3d, h_k, zero_f, mesh,
    )  # (nVertices, nlev); equals ζ/h_v

    # APVM (Anticipated Potential Vorticity Method, Sadourny & Basdevant
    # 1985; Ringler et al. 2010) upstream-biases q by dt_apvm/2 to damp
    # the ζ-checkerboard null mode of the energy-conserving PV flux.
    # Without it, a 2Δx vortex mode at vertices amplifies through
    # nonlinear interactions (observed: τ ~ 1 d at U=0.2 m/s on 20 km).
    if config.apvm_dt > 0.0:
        q_relative = apvm_correction_3d(q_relative, u_3d, mesh, config.apvm_dt)

    # ``h_e_3d`` is already computed above (line 158); pass it via
    # ``h_edge_3d=`` so ``pv_flux_*_conserving_3d`` skips its internal
    # ``edge_thickness_3d`` (i.e. one redundant ``cellsOnEdge`` gather).
    if config.pv_scheme == "energy":
        pv_flux = pv_flux_energy_conserving_3d(
            u_3d, h_k, q_relative, mesh, h_edge_3d=h_e_3d,
        )
    elif config.pv_scheme == "mixed":
        # Weighted blend: α·F_energy + (1−α)·F_enstrophy. α=1 reverts to
        # pure energy-conserving; α=0 to pure enstrophy-conserving. For the
        # Eady ζ-checkerboard null mode, α ≈ 0.6–0.9 preserves most of the
        # BCI growth rate while inheriting the enstrophy scheme's stability.
        alpha = config.pv_alpha
        pv_flux = (
            alpha * pv_flux_energy_conserving_3d(
                u_3d, h_k, q_relative, mesh, h_edge_3d=h_e_3d,
            )
            + (1.0 - alpha)
              * pv_flux_enstrophy_conserving_3d(
                  u_3d, h_k, q_relative, mesh, h_edge_3d=h_e_3d,
              ))
    else:
        pv_flux = pv_flux_enstrophy_conserving_3d(
            u_3d, h_k, q_relative, mesh, h_edge_3d=h_e_3d,
        )

    # Horizontal viscosity on perturbation velocity (shear, not depth-mean).
    # When both A_h > 0 and B_h > 0, the biharmonic
    # ``vector_laplacian_del4_3d(u) = -∇²(∇²u)``, so its inner ∇² is
    # identical to the explicit A_h Laplacian — compute it once and
    # share between both branches.  Same exploit as Loop 135 for the
    # latlon ocean and Loop 139 for the MPAS atmosphere.  Also adds an
    # ``if A_h > 0`` guard so a configuration with A_h = 0 (Smag-only,
    # Leith-only, or B_h-only) skips the unconditional del2 the
    # previous code paid for and discarded.
    visc = jnp.zeros_like(u_prime_3d)
    if config.A_h > 0 and config.B_h > 0:
        _del2_u_visc = vector_laplacian_del2_3d(u_prime_3d, mesh)
        visc = visc + config.A_h * _del2_u_visc
        # vector_laplacian_del4 = -del2(del2); fold the sign into the
        # subtraction so the arithmetic matches ``+ B_h * del4``.
        visc = visc - config.B_h * vector_laplacian_del2_3d(_del2_u_visc, mesh)
    elif config.A_h > 0:
        visc = visc + config.A_h * vector_laplacian_del2_3d(u_prime_3d, mesh)
    elif config.B_h > 0:
        visc = visc + config.B_h * vector_laplacian_del4_3d(u_prime_3d, mesh)

    # Flow-dependent Smagorinsky biharmonic viscosity
    if config.C_smag > 0:
        visc = visc + smagorinsky_biharmonic_3d(u_prime_3d, mesh, config.C_smag)

    # Flow-dependent Leith biharmonic viscosity
    if getattr(config, "C_leith", 0.0) > 0:
        visc = visc + leith_biharmonic_3d(
            u_prime_3d, mesh, config.C_leith,
            modified=getattr(config, "C_leith_modified", False))

    # Biharmonic dissipation on relative vorticity ζ (scale-selective damping
    # of grid-scale vorticity patterns — notably the ζ-checkerboard null
    # mode of the energy-conserving PV flux).  This is applied to the total
    # velocity u_3d (not u_prime_3d) because ζ is a derived quantity and the
    # full ζ (including the planetary-Coriolis-free baroclinic+barotropic ζ)
    # carries the null-mode amplitude.  Invisible to ``B_h·del4(u)`` because
    # the null mode lives in the kernel of the discrete curl-to-velocity map.
    if config.K_zeta_bih > 0:
        visc = visc + config.K_zeta_bih * biharmonic_vorticity_del4_3d(
            u_3d, mesh)

    # Vertical advection of perturbation momentum (#171 Level-1).
    w_e = 0.5 * (w[c1] + w[c2])  # (nEdges, nlev+1)
    vert_adv_u = flux_form_vertical_momentum_advection(
        u_prime_3d, w_e, h_e_3d,
    )

    # Full nonlinear momentum tendency
    du_dt_full = (-grad_B + pv_flux + visc + vert_adv_u) * edge_mask[:, jnp.newaxis]

    # Bottom drag on full velocity (not perturbation) — the ocean floor
    # sees the total flow.  Applied before F_slow_u computation so the
    # depth-averaged drag enters the barotropic solver via slow forcing.
    # r is in [m/s]: du/dt = -r * u / dz_bottom  (resolution-independent stress).
    if config.bottom_drag_r > 0:
        dz_bot_e = jnp.maximum(h_e_3d[:, -1], 1e-10)
        du_dt_full = du_dt_full.at[:, -1].add(
            -config.bottom_drag_r * u_3d[:, -1] / dz_bot_e * edge_mask)

    # Depth-mean → slow forcing for barotropic solver.
    # du_dt_full now carries only (PGF + relative-vorticity PV flux +
    # viscosity + vertical-advection + bottom-drag), NO planetary
    # Coriolis. So F_slow_u passed to the barotropic solver contains no
    # planetary Coriolis either, and the barotropic substep applies
    # evolving f·v_t(u_bar) online.
    F_slow_u = jnp.sum(du_dt_full * h_e_3d, axis=1) / jnp.maximum(H_e, 1e-10)
    F_slow_u = F_slow_u * edge_mask  # (nEdges,)

    # Baroclinic perturbation = full minus depth-mean. Planetary Coriolis
    # on this perturbation is applied via forward-backward Matsuno in the
    # step() function (see _forward_backward_coriolis_mpas_3d).
    du_dt_3d = (du_dt_full - F_slow_u[:, jnp.newaxis]) * edge_mask[:, jnp.newaxis]

    # ---- Tracer tendencies (diffusion + physics only) ----
    # Horizontal AND vertical tracer advection are handled in the step()
    # function using barotropic-averaged transport (Hallberg 1997, #102, #145).
    # This matches the latlon C-grid pattern (ocean_pe_latlon_cgrid.py).
    #
    # Stack T and S along a trailing axis and fold it into the level dim
    # so the halo-issuing operators (gradient_edge_3d, divergence_cell_3d,
    # bilaplacian_cell_3d) run ONCE on the thicker (nCells, nlev*2)
    # field instead of being called twice per timestep — eliminates the
    # per-tracer kernel duplication.  Vertical diffusion stays per-tracer
    # because ``_vertical_diffusion`` hard-codes the vertical axis at -1.
    h_safe = jnp.maximum(h_k, 1e-10)  # (nCells, nlev)
    tracer_stack = jnp.stack([T_3d, S_3d], axis=-1)  # (nCells, nlev, 2)
    nCells_t, nlev_t, n_tracers = tracer_stack.shape
    tracer_flat = tracer_stack.reshape(nCells_t, nlev_t * n_tracers)

    # Horizontal tracer diffusion: K_h * lap(T,S) per layer.  edge_mask is
    # (nEdges,) and broadcasts across the trailing axis via [:, None].
    # ``_tracer_grad_flat_pre`` was computed alongside ``grad_B`` via the
    # batched gradient block above (Loop 148) — reuse it here so we
    # don't issue a redundant ``gradient_edge_3d`` on the same input.
    #
    # When BOTH K_h and K_bih are active, the K_h Laplacian
    # ``div(grad*edge_mask)`` is identical to the *inner* (raw)
    # Laplacian of the biharmonic ``bilaplacian_cell_3d`` (which is
    # defined as ``laplacian_cell_3d(laplacian_cell_3d(f, mask=mask),
    # mask=mask)`` — its inner step computes ``div(grad(f)*edge_mask)
    # * cell_mask``).  Inline the bilaplacian and share the inner
    # ``div(grad*edge_mask)`` with K_h's Laplacian — saves one
    # ``divergence_cell_3d`` call (1 ``edgesOnCell`` gather + reduce)
    # per RHS evaluation when both coefficients are active.  Same
    # Loop 135 exploit as the latlon ocean K_h+K_bih sharing.
    _inner_lap_div: jnp.ndarray | None = None
    if config.K_h > 0 or config.K_bih > 0:
        grad_flat = _tracer_grad_flat_pre * edge_mask[:, jnp.newaxis]
        _inner_lap_div = divergence_cell_3d(grad_flat, mesh)
    if config.K_h > 0:
        # Reshape to (nCells, nlev, 2) so h_safe and h_k broadcast via [:, :, None].
        div_stack = _inner_lap_div.reshape(nCells_t, nlev_t, n_tracers)
        diff_stack = config.K_h * div_stack / h_safe[..., None] * h_k[..., None]
    else:
        diff_stack = jnp.zeros_like(tracer_stack)

    # Biharmonic tracer diffusion: -K_bih * bilap(T,S).  Same sign convention
    # as the latlon ``bilaplacian_cgrid`` wiring: ``bilaplacian_cell_3d``
    # returns ∇²(∇²f), so the physical dissipation sign is applied at the
    # call site.  The ``mask`` kwarg zeros gradients at coastlines and the
    # intermediate Laplacian on land on both passes; the trailing
    # ``/ h_safe * h_k`` factor is ≈1 on wet cells and a dry-cell safety
    # guard where h_k → 0.
    if config.K_bih > 0:
        # Inline the bilaplacian: inner = ``div(grad*edge_mask) *
        # cell_mask`` (already partially computed above as
        # ``_inner_lap_div``); outer = ``laplacian_cell_3d(inner,
        # mask=mask)``.
        inner_lap_masked = _inner_lap_div * mask[:, jnp.newaxis]
        outer_grad = gradient_edge_3d(inner_lap_masked, mesh) * edge_mask[:, jnp.newaxis]
        outer_div = divergence_cell_3d(outer_grad, mesh)
        bilap_flat = outer_div * mask[:, jnp.newaxis]
        bilap_stack = bilap_flat.reshape(nCells_t, nlev_t, n_tracers)
        diff_stack = diff_stack - (
            config.K_bih * bilap_stack / h_safe[..., None] * h_k[..., None]
        )

    # Mask land cells (mask broadcasts via [..., None, None] over level + tracer)
    diff_stack = diff_stack * mask[:, jnp.newaxis, jnp.newaxis]
    dT_dt_3d = diff_stack[..., 0]
    dS_dt_3d = diff_stack[..., 1]

    # ---- Vertical mixing ----
    dz_half = z_coord.dz_half_ref  # (nlev-1,)
    dz = z_coord.dz_ref  # (nlev,)

    # Vertical viscosity on perturbation velocity: d/dz(A_v * du'/dz)
    du_dt_3d = du_dt_3d + _vertical_diffusion(
        u_prime_3d, dz_half, dz, jacobian=jacobian, coeff=config.A_v, is_edge=True,
        mesh=mesh,
    )

    # Vertical tracer diffusion (per-tracer; axis -1 of the input is nlev,
    # so vmap over the trailing tracer axis to get one batched kernel).
    def _vdiff(q):
        return _vertical_diffusion(
            q, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
            mesh=mesh,
        )

    vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
    vdiff_stack = vdiff_stack * mask[:, jnp.newaxis, jnp.newaxis]
    dT_dt_3d = dT_dt_3d + vdiff_stack[..., 0]
    dS_dt_3d = dS_dt_3d + vdiff_stack[..., 1]

    # ---- Physics (surface forcing, bottom drag, etc.) ----
    if physics_fn is not None:
        phys = physics_fn(state, mesh, z_coord, surface_forcing)
        du_dt_3d = du_dt_3d + phys.du_dt.data
        dT_dt_3d = dT_dt_3d + phys.dT_dt.data
        dS_dt_3d = dS_dt_3d + phys.dS_dt.data

    # ---- Free surface tendency ----
    # deta/dt = -sum_k div(u_k * h_e_k)
    deta_dt = -jnp.sum(div_flux, axis=1) * mask  # (nCells,)

    # ---- Freshwater forcing ----
    # Note: freshwater_eta_tendency is NOT applied to deta_dt here because
    # deta_dt is not used for state update — the barotropic solver handles
    # the free-surface equation (including freshwater via F_slow_eta passed
    # from ocean_model_mpas.py:step()).  Only the virtual salt flux is
    # applied here as a tracer tendency.
    if freshwater is not None and config.freshwater_closure != "none":
        dS_dt_3d = apply_freshwater_virtual_salt_top(
            dS_dt_3d, freshwater, config.S_ref, h_k[:, 0], config.rho_0, mask,
        )

    # ---- Sponge layer relaxation ----
    # Cast sponge arrays to state dtype to prevent float64 promotion when
    # the precision policy stores state in float32 (crashes barotropic scan).
    if sponge is not None:
        dT_dt_3d, dS_dt_3d = apply_sponge_tracer_relaxation(
            dT_dt_3d, dS_dt_3d, T_3d, S_3d, sponge, mask=mask,
            expand_gamma_axis=-1,
        )
        # Edge velocity sponge (if reference velocity provided)
        if sponge.u_ref is not None:
            _dt = T_3d.dtype
            gamma_edge = 0.5 * (sponge.gamma.astype(_dt)[c1] + sponge.gamma.astype(_dt)[c2])
            gamma_edge_3d = gamma_edge[:, jnp.newaxis]
            du_dt_3d = du_dt_3d + gamma_edge_3d * (sponge.u_ref.astype(_dt) - u_3d) * edge_mask[:, jnp.newaxis]

    return MPASOceanTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="degC/s"),
        dS_dt=Field(data=dS_dt_3d, name="dS_dt",
                    dims=("nCells", "nlev"), units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt",
                      dims=("nCells",), units="m/s"),
        F_slow_u=Field(data=F_slow_u, name="F_slow_u",
                       dims=("nEdges",), units="m/s²"),
    )


def _vertical_diffusion(field_3d, dz_half, dz, jacobian, coeff, is_edge, mesh):
    """Compute vertical diffusion d/dz(coeff * df/dz).

    Parameters
    ----------
    field_3d : jax.Array, shape (n, nlev)
    dz_half : jax.Array, shape (nlev-1,)
    dz : jax.Array, shape (nlev,)
    jacobian : jax.Array or None, shape (nCells,) or None
    coeff : float
    is_edge : bool
        If True, field lives on edges (use edge-averaged jacobian).
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (n, nlev)
    """
    nlev = field_3d.shape[1]
    if nlev < 2:
        return jnp.zeros_like(field_3d)

    # Scale dz by jacobian if available
    if jacobian is not None and not is_edge:
        J = jacobian[:, jnp.newaxis]  # (nCells, 1)
    elif jacobian is not None and is_edge:
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        J = 0.5 * (jacobian[c1] + jacobian[c2])  # (nEdges,)
        J = J[:, jnp.newaxis]
    else:
        J = 1.0

    dz_half_actual = dz_half * J if jacobian is not None else jnp.broadcast_to(
        dz_half[jnp.newaxis, :], (field_3d.shape[0], nlev - 1),
    )
    dz_actual = dz * J if jacobian is not None else jnp.broadcast_to(
        dz[jnp.newaxis, :], field_3d.shape,
    )

    # Flux at interfaces: coeff * (f[k] - f[k+1]) / dz_half
    dz_half_safe = jnp.maximum(dz_half_actual, 1e-10)
    flux_interface = coeff * (field_3d[:, :-1] - field_3d[:, 1:]) / dz_half_safe

    # Tendency: (flux[k-1/2] - flux[k+1/2]) / dz[k].  Use ``jnp.pad``
    # to attach the zero-flux top/bottom boundaries — single Pad HLO
    # op vs alloc fresh ``(n, 1)`` zeros and concatenate.
    flux_above = jnp.pad(flux_interface, ((0, 0), (1, 0)))  # (n, nlev)
    flux_below = jnp.pad(flux_interface, ((0, 0), (0, 1)))  # (n, nlev)

    dz_safe = jnp.maximum(dz_actual, 1e-10)
    return (flux_above - flux_below) / dz_safe
