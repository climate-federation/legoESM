"""MPAS barotropic (free-surface) solver on Voronoi meshes.

Split-explicit forward-backward substeps for the barotropic mode.
Updates sea surface height (eta) and depth-averaged normal velocity
(u_bar) using the fast gravity-wave CFL.

Coriolis treatment: ``f·v_t(u_bar)`` is applied at every substep with
the *current-substep* ``u_bar`` (evolving, semi-implicit Heun option).
``F_slow_u`` passed in by the caller is the depth-mean of the full 3D
baroclinic tendency with the planetary-Coriolis part subtracted (see
``ocean_pe_mpas.py``), so the barotropic substep can add online
evolving Coriolis without double-counting. This matches the lat-lon
C-grid pattern. Without this split, a frozen-Coriolis F_slow_u
accumulates over 30 substeps per baroclinic step and grows a
near-inertial numerical mode (τ ~ 1/f, ~0.2 days at mid-latitudes).

References
----------
- Ringler, T. D., et al. (2013). Ocean Modelling, 69, 211-232.
- Higdon, R. L. (2005). J. Comput. Phys., 205(1), 194-224.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    tangential_velocity,
    vector_laplacian_del2,
    edge_thickness as _edge_avg,
)
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate, compute_layer_thickness,
)
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    maxvel_clip,
)
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas


def barotropic_substeps_mpas(
    state,
    mesh,
    z_coord,
    config,
    dt_baro,
    n_substeps,
    F_slow_eta=None,
    F_slow_u=None,
    halo_refresh=None,
):
    """Run barotropic substeps on MPAS Voronoi mesh.

    Forward-backward scheme:
        1. Forward:  eta^{n+1} = eta^n - dt * div(H_e * u_bar^n) + dt * F_slow_eta
        2. Backward: u_bar^{n+1} = u_bar^n + dt * (-g*grad(eta^{n+1}) + f*v_t)

    The 3D baroclinic tendency is already applied to u before calling
    this function, so u_bar is computed from the updated state.  No
    F_slow_u is needed (same as cubed-sphere barotropic.py).

    Parameters
    ----------
    state : MPASOceanState
        State with 3D velocity already updated by baroclinic tendency.
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    dt_baro : float
        Barotropic substep size [s].
    n_substeps : int
        Number of barotropic substeps.
    F_slow_eta : jax.Array or None, shape (nCells,)
        Slow forcing for eta (e.g., freshwater mass flux) [m/s].
    halo_refresh : MPASOceanHaloRefresh, optional
        Distributed in-substep packed halo refresh (stage-correctness
        lever): each substep consumes 2-6 stencil hops of the
        ``(eta, u_bar)`` carry — versus a ``halo_depth=2`` partition
        budget — so the carry is refreshed at substep ENTRY ([B1], one
        packed cell+edge message), the PGF eta after its
        continuity update ([B2]), and the optional div-damp /
        barotropic-viscosity / eta-diffusion chains at their block
        entries.  The scan-constant ``F_slow_*`` rings are refreshed
        ONCE before the scan.  ``None`` (serial) is byte-identical.

    Returns
    -------
    eta_avg : jax.Array, shape (nCells,)
        Time-averaged sea surface height over substeps [m].
    u_bar_avg : jax.Array, shape (nEdges,)
        Time-averaged depth-averaged velocity over substeps [m/s].
    Hu_avg : jax.Array, shape (nEdges,)
        Time-averaged depth-integrated edge transport [m²/s].
        All three are time-averaged to filter fast barotropic gravity
        waves from the baroclinic coupling (Higdon 2005, issue #149).
    """
    g = config.g
    mask = state.land_mask.data  # (nCells,)
    H_bathy = state.H_bathy.data
    eta = state.eta.data
    u_3d = state.u.data  # (nEdges, nlev)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Compute layer thickness and depth-averaged velocity
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )  # (nCells, nlev)

    # Edge layer thickness for each level — min-rule on partial cells
    # so the depth-mean here matches what the implicit-CN solver and
    # the baroclinic step's F_slow_u see.  Using a centered
    # 0.5*(h[c1]+h[c2]) on a step edge lets phantom transport leak
    # through and drives the seamount rest-state explosion.
    partial_cells = isinstance(z_coord, OceanPartialCellCoordinate)

    # Distributed Voronoi (codex 2026-06-11 CRITICAL): with a partition
    # layout armed, ``is_multi_process()`` is now TRUE on this path, so
    # the eta-floor clamp's global sums WOULD allreduce — its local
    # arrays carry halo cells, which an unweighted sum double-counts.
    # Wire the owned mask explicitly (mesh-matched accessor; ``None``
    # on single-rank/global meshes keeps the legacy behavior).
    # The refresh object carries the owned mask on BOTH distributed lanes
    # (MPI layout / SPMD ppermute); the layout accessor stays as the fallback
    # for callers that pass no halo_refresh (the historical MPI path).
    _hr_ow = getattr(halo_refresh, "owned_mask_cells", None)
    if _hr_ow is not None:
        _clamp_ow, _clamp_fg = _hr_ow, True
    else:
        from legoesm.parallel.voronoi_mpi import get_matching_voronoi_layout
        _vl_clamp = get_matching_voronoi_layout(mesh)
        _clamp_ow = (None if _vl_clamp is None
                     else _vl_clamp.owned_mask_cells)
        _clamp_fg = _vl_clamp is not None
    if partial_cells:
        h_e_k = min_cell_to_edge(h_k, mesh)
    else:
        h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)

    # Depth-integrated transport: sum_k(u_k * h_e_k)
    Hu_bar = jnp.sum(u_3d * h_e_k, axis=1)  # (nEdges,)

    # Total water column at edges — min-rule on partial cells (matches
    # ``barotropic_implicit_mpas._edge_H_min_rule``).
    if partial_cells:
        H_e = jnp.maximum(jnp.sum(h_e_k, axis=1), config.min_water_column_m)
    else:
        H_total_cell = eta + H_bathy  # (nCells,)
        H_total_cell = jnp.maximum(H_total_cell, config.min_water_column_m)
        H_e = _edge_avg(H_total_cell, mesh)  # (nEdges,)

    # Depth-averaged velocity (from state that already includes baroclinic tendency).
    # Mask land edges to zero up-front so any stale value at a land
    # edge (e.g. from FP roundoff or spin-up transient) cannot leak
    # into the Coriolis tangential-velocity gather below — TRiSK's
    # ``tangential_velocity`` gathers neighboring edges, so a
    # nonzero land-edge u_bar can contaminate adjacent interior
    # edges' v_t.  Iter-62 audit follow-up to iter-61 fix.
    u_bar = Hu_bar / jnp.maximum(H_e, 1e-10)
    u_bar = u_bar * edge_mask

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    # F_slow_u is PGF+KE+zeta-advection+viscosity+bottom-drag with the
    # planetary Coriolis contribution EXCLUDED (see ocean_pe_mpas.py).
    # The barotropic substep applies online evolving f·v_t(u_bar) below,
    # matching the lat-lon C-grid pattern.
    if F_slow_u is None:
        F_slow_u = jnp.zeros_like(u_bar)
    # [stage-halo B0] Scan-constant rings, refreshed ONCE before the scan
    # (one packed message): F_slow_u's halo carries the neighbor rank's
    # (masked-wrong) tendency depth-means, u_bar's ring was depth-averaged
    # from the 2-hop-consumed post-Coriolis u, and both feed every
    # substep.  F_slow_eta rides along (zeros stay zeros under exchange).
    if halo_refresh is not None:
        (u_bar, F_slow_u), (F_slow_eta,) = halo_refresh.both(
            (u_bar, F_slow_u), (F_slow_eta,))

    # --- Fix 1: Neumann fill for eta before gradient ---
    # Fill land-cell eta with nearest-ocean-neighbor average so that
    # gradient_edge sees smooth fields at coastlines instead of the
    # sharp ocean-to-zero jump from masking.
    def _fill_land_cells_mpas(field_cell, mask_cell):
        return fill_land_cells_mpas(field_cell, mask_cell, c1, c2)

    # --- Fix 2: Barotropic Laplacian diffusion ---
    baro_alpha_val = config.barotropic_diffusion_alpha
    dt_ref = config.barotropic_diffusion_dt_ref
    use_baro_diffusion = baro_alpha_val > 0.0

    if use_baro_diffusion:
        # Edge-centered diffusion coefficient for flux-form diffusion:
        # div(nu_edge * grad(eta)) is exactly conservative (divergence
        # theorem), unlike nu_cell * div(grad(eta)) which leaks volume
        # when cell areas are non-uniform.
        nu_dt_edge = baro_alpha_val * (dt_baro / dt_ref) * (
            0.5 * (mesh.areaCell[c1] + mesh.areaCell[c2])
        )

    # Divergence damping on barotropic velocity: grad(div(u_bar)).
    # Targets the divergent mode that creates the eta checkerboard (#205).
    use_div_damp = config.barotropic_div_damp > 0.0
    if use_div_damp:
        div_damp_coeff = jnp.asarray(
            config.barotropic_div_damp, dtype=eta.dtype,
        ) * (dt_baro / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))
        div_damp_area_edge = 0.5 * (mesh.areaCell[c1] + mesh.areaCell[c2])

    # Barotropic-mode lateral viscosity on u_bar: A_baro * del2(u_bar).
    # Targets the TRiSK rotational null branch on hexagonal C-grids
    # (Thuburn 2008; Ringler et al. 2010), which is invisible to eta
    # diffusion and to divergence damping (the null mode has both
    # ∇·u_bar ≈ 0 and ∇η ≈ 0). MPAS-O production uses an analogous
    # del2 viscosity on the depth-mean velocity (Ringler et al. 2013).
    A_baro_visc = jnp.asarray(config.barotropic_u_viscosity, dtype=eta.dtype)
    use_baro_visc = config.barotropic_u_viscosity > 0.0

    # --- Fix 3: Semi-implicit Coriolis (trapezoidal predictor-corrector) ---
    # On Voronoi meshes, the (u, v_tangential) decomposition doesn't
    # allow a direct Crank-Nicolson solve. Instead, use a trapezoidal
    # predictor-corrector: predict u_star with old v_t, recompute v_t
    # from u_star, and average. This is second-order and avoids the
    # explicit f*dt instability at high latitudes.
    use_semi_implicit = config.semi_implicit_coriolis

    # Eta floor: prevent water column from going below minimum depth.
    # Matches cubed-sphere and lat-lon barotropic solvers.
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = (min_water_col - H_bathy) * mask
    _area_cell = mesh.areaCell  # for mass-conserving floor clamp (#176)

    # Cosine time filter and BEBT/MAXVEL parameters
    bebt = config.bebt
    _maxvel = config.maxvel_barotropic
    use_maxvel = _maxvel > 0.0

    use_cosine_filter = config.barotropic_time_filter == "cosine"
    # ``w_transport`` is the continuity-consistent SM2005 tail-sum transport
    # weight ``tail_j/(n·w_total)`` (NOT a flat 1/n) so the discrete continuity
    # invariant ``div(Hu_avg) == (eta_old - eta_avg)/dt`` holds for both box and
    # cosine (the flat 1/n broke it — worst for cosine).
    # ``n_loop`` (== 2*n_substeps - 1) is the length of the averaging window,
    # which extends past t+dt so that the window is CENTRED on t+dt; the
    # transport weights still close continuity over the physical
    # dt = n_substeps*dt_s.  See compute_filter_weights.
    w_filter, w_total, w_transport, n_loop = compute_filter_weights(
        n_substeps, eta.dtype, use_cosine=use_cosine_filter,
    )

    # Accumulators for time-averaged barotropic fields (issues #145, #149, #102).
    # The substep body deliberately returns transport in eta precision (the
    # continuity equation's dtype).  Seed the scan carry in that same dtype;
    # under split precision u_bar can be f64 while eta is f32, and a
    # zeros_like(u_bar) seed makes lax.scan reject the f64 -> f32 transition.
    Hu_sum = jnp.zeros_like(u_bar, dtype=eta.dtype)
    eta_sum = jnp.zeros_like(eta)
    ubar_sum = jnp.zeros_like(u_bar)

    # Forward-backward substeps via scan
    _eta_dtype = eta.dtype
    _ubar_dtype = u_bar.dtype

    def _substep(carry, wts_i):
        w_i, w_tr_i = wts_i
        eta_c, u_bar_c, Hu_sum_c, eta_sum_c, ubar_sum_c = carry
        # [stage-halo B1] Substep-entry refresh of the carry pair (ONE
        # packed cell+edge message): the previous substep consumed 2-6
        # hops, so the carry ring is stale every iteration.  The AD-safe
        # custom_vjp sendrecv is scan-safe (the latlon band path's
        # per-substep pad is the precedent).
        if halo_refresh is not None:
            (u_bar_c,), (eta_c,) = halo_refresh.both((u_bar_c,), (eta_c,))

        # Total depth at edges (updated with current eta).  On partial
        # cells use min-rule so the per-substep H_e_c matches the
        # per-level ``min_cell_to_edge(h_k)`` convention used by
        # F_slow_u (ocean_pe_mpas.py:239), the implicit-CN solver
        # (barotropic_implicit_mpas.py:133), the reconcile-velocity
        # transport divide (ocean_model_mpas.py:382), and the tracer-
        # flux mass channel (ocean_model_mpas.py:419).  Centered
        # ``_edge_avg`` here would leave a ``(centered − min)·u_bar``
        # residual at every step edge that gets fed back into u_3d
        # via ``Hu_avg = mean(H_e_c·u_bar_c)`` and the reconcile-
        # velocity ``delta_u = (Hu_avg − Hu_3d) / H_e_old``.  Latent
        # bug only — the implicit-CN solver is the production path
        # and is already self-consistent.  Audit 2026-05-04.
        H_c = jnp.maximum(eta_c + H_bathy, config.min_water_column_m)
        if partial_cells:
            H_e_c = jnp.minimum(H_c[c1], H_c[c2])
        else:
            H_e_c = _edge_avg(H_c, mesh)

        # Forward: update eta (continuity + freshwater mass source)
        transport = H_e_c * u_bar_c * edge_mask

        # Accumulate transport for time-averaged tracer advection with the
        # continuity-consistent SM2005 transport weight (NOT a flat 1/n).
        Hu_sum_new = Hu_sum_c + w_tr_i * transport.astype(_eta_dtype)

        eta_next = eta_c - dt_baro * divergence_cell(transport, mesh) * mask + dt_baro * F_slow_eta * mask
        eta_next = _clamp_redistribute(
            eta_next, eta_floor, mask, _area_cell,
            n_iter=config.eta_floor_clamp_iters,
            owned_weight=_clamp_ow, force_global=_clamp_fg,
        )

        # Backward: update u_bar using new eta
        # BEBT: blend new/old eta for semi-implicit PGF (#205)
        eta_pgf = bebt_blend(eta_next, eta_c, bebt)
        # [stage-halo B2] eta_next consumed 2 hops (H_e + divergence);
        # the fill+gradient PGF chain below needs a fresh ring.
        if halo_refresh is not None:
            (eta_pgf,) = halo_refresh.cells(eta_pgf)
        eta_filled = _fill_land_cells_mpas(eta_pgf, mask)
        grad_eta = gradient_edge(eta_filled, mesh)

        # PGF + evolving Coriolis + slow forcing.
        # F_slow_u now has the depth-mean planetary Coriolis subtracted
        # in ocean_pe_mpas.py, so we always apply online f·v_t(u_bar) here
        # — otherwise the barotropic u_bar loses its rotational restoring
        # torque inside the substep loop and a near-inertial numerical
        # mode (τ ~ 1/f) grows on the order of 0.2 days at mid-latitudes.
        # Land-edge zero-out: wrap the FULL update (u_bar_c + dt·tendency)
        # in ``* edge_mask`` so any stale u_bar_c at a land edge is also
        # zeroed each substep.  Previously the parens lay only around the
        # tendency: ``u_bar_c + dt * (...) * mask`` parses as
        # ``u_bar_c + (dt * (...) * mask)``, which masks the tendency
        # but leaves u_bar_c untouched at land edges.  Iter-61 audit fix.
        if use_semi_implicit:
            # Heun predictor-corrector for Coriolis (#172 docs fix)
            v_t_old = tangential_velocity(u_bar_c, mesh)
            u_star = (u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old + F_slow_u
            )) * edge_mask
            v_t_star = tangential_velocity(u_star, mesh)
            u_bar_next = (u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * 0.5 * (v_t_old + v_t_star)
                + F_slow_u
            )) * edge_mask
        else:
            v_t_old = tangential_velocity(u_bar_c, mesh)
            u_bar_next = (u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old + F_slow_u
            )) * edge_mask

        # Divergence damping: add nu * grad(div(u_bar)) (#205).
        if use_div_damp:
            # [stage-halo] div+fill+grad = 3 hops on the just-updated
            # u_bar_next — refresh its ring at block entry.
            if halo_refresh is not None:
                (u_bar_next,) = halo_refresh.edges(u_bar_next)
            div_ubar = divergence_cell(u_bar_next * edge_mask, mesh) * mask
            div_filled = _fill_land_cells_mpas(div_ubar, mask)
            grad_div = gradient_edge(div_filled, mesh)
            u_bar_next = (
                u_bar_next + div_damp_coeff * div_damp_area_edge * grad_div
            ) * edge_mask

        # Barotropic-mode lateral viscosity on u_bar (TRiSK null branch).
        # Forward-Euler Laplacian: stable while A_baro_visc * dt_baro / dx² < 0.5.
        if use_baro_visc:
            # [stage-halo] del2 = 2 hops on the updated u_bar_next.
            if halo_refresh is not None:
                (u_bar_next,) = halo_refresh.edges(u_bar_next)
            lap_u = vector_laplacian_del2(u_bar_next, mesh)
            u_bar_next = (
                u_bar_next + dt_baro * A_baro_visc * lap_u
            ) * edge_mask

        # Optional barotropic damping (Rayleigh drag)
        if config.barotropic_damping > 0:
            u_bar_next = u_bar_next * (1.0 - dt_baro * config.barotropic_damping)

        # Bottom drag — SINGLE OWNER (finding #6 fix).
        # ``ocean_pe_mpas`` already applies the full bottom drag ``-r·u_bot/h``
        # (with BBL / partial-cell handling) to ``du_dt_full``; its depth-mean
        # is carried into the barotropic mode through ``F_slow_u`` and applied at
        # every substep above.  Re-applying ``implicit_bottom_drag_factor`` here
        # would make the effective barotropic-mode drag ``≈ 2·r/H`` (codex iter-2
        # finding #1).  Drag is therefore owned exclusively by the 3D tendency /
        # F_slow; we do NOT re-apply it here — consistent with the implicit-CN
        # MPAS solver, which already relies on F_slow alone.

        # MAXVEL clipping
        if use_maxvel:
            u_bar_next = maxvel_clip(u_bar_next, _maxvel)

        # Barotropic Laplacian diffusion on eta (flux-form: conservative).
        # Uses div(nu_edge * grad(eta)) instead of nu_cell * div(grad(eta))
        # so that volume is exactly conserved by the divergence theorem.
        if use_baro_diffusion:
            # [stage-halo] fill+grad+div = 3 hops on the updated eta_next.
            if halo_refresh is not None:
                (eta_next,) = halo_refresh.cells(eta_next)
            eta_filled = _fill_land_cells_mpas(eta_next, mask)
            grad_e = gradient_edge(eta_filled, mesh)
            diff_flux = nu_dt_edge * grad_e * edge_mask
            eta_next = (
                eta_next + divergence_cell(diff_flux, mesh)
            ) * mask
            eta_next = _clamp_redistribute(
            eta_next, eta_floor, mask, _area_cell,
            n_iter=config.eta_floor_clamp_iters,
            owned_weight=_clamp_ow, force_global=_clamp_fg,
        )

        # Accumulate eta and u_bar with cosine filter weights
        eta_sum_new = eta_sum_c + w_i * eta_next.astype(_eta_dtype)
        ubar_sum_new = ubar_sum_c + w_i * u_bar_next.astype(_eta_dtype)

        # Cast back to input dtype (mesh ops may promote to float64)
        return (eta_next.astype(_eta_dtype), u_bar_next.astype(_ubar_dtype),
                Hu_sum_new.astype(_eta_dtype),
                eta_sum_new, ubar_sum_new), None

    (eta_new, u_bar_new, Hu_sum_f, eta_sum_f, ubar_sum_f), _ = jax.lax.scan(
        _substep, (eta, u_bar, Hu_sum, eta_sum, ubar_sum),
        (w_filter, w_transport), length=n_loop,
    )

    # Time-averaged barotropic fields.  ``w_transport`` already carries the full
    # continuity-consistent normalisation (SM2005 tail-sum), so the accumulator
    # IS the time-averaged transport ``Hu_avg`` that closes
    # ``div(Hu_avg) == (eta_old - eta_avg)/dt``.
    Hu_avg = Hu_sum_f
    eta_avg = eta_sum_f / w_total   # eta/velocity: cosine or box filtered
    u_bar_avg = ubar_sum_f / w_total

    return eta_avg, u_bar_avg, Hu_avg


def reconcile_3d_velocity(u_3d, u_bar_old, u_bar_new, mesh, mask):
    """Reconcile 3D velocity with updated barotropic velocity.

    Preserves baroclinic structure (deviations from depth-mean) while
    replacing the depth-mean with the barotropic solution:

        u_3d_new = (u_3d - u_bar_old) + u_bar_new

    This ensures depth_avg(u_3d_new) = u_bar_new exactly.

    Parameters
    ----------
    u_3d : jax.Array, shape (nEdges, nlev)
    u_bar_old : jax.Array, shape (nEdges,)
        Depth-averaged velocity BEFORE barotropic substeps.
    u_bar_new : jax.Array, shape (nEdges,)
        Depth-averaged velocity AFTER barotropic substeps.
    mesh : VoronoiMesh
    mask : jax.Array, shape (nCells,)

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Baroclinic deviation + new barotropic mean
    u_prime = u_3d - u_bar_old[:, jnp.newaxis]
    return (u_prime + u_bar_new[:, jnp.newaxis]) * edge_mask[:, jnp.newaxis]
