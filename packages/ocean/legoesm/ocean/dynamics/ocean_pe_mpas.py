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
    divergence_cell_3d,
    gradient_edge_3d,
    curl_vertex_3d,
    kinetic_energy_cell_3d,
    potential_vorticity_vertex_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    smagorinsky_biharmonic_3d,
    smagorinsky_laplacian_3d,
    leith_biharmonic_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
)
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    compute_max_level_edge_bot,
    density_jacobian_pgf_smc03_mpas,
    donor_cell_to_edge,
    min_cell_to_edge,
    partial_cell_pgf_correction_edge,
    vertex_thickness_hybrid,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
    compute_centroid_depth,
    compute_layer_thickness,
    compute_ocean_jacobian,
    diagnose_w_from_flux_div,
    flux_form_vertical_momentum_advection,
)
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    salt_flux_salinity_tendency,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    apply_freshwater_virtual_salt_top,
    apply_sponge_tracer_relaxation,
    iterate_eos_and_pressure_anomaly,
    nemo_drag_r_from_speed_sq,
    validate_bottom_drag_scheme,
)
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas
from legoesm import constants


def mpas_ocean_baroclinic_tendencies(
    state: MPASOceanState,
    mesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig = MPASOceanConfig(),
    freshwater: FreshwaterForcing | None = None,
    physics_fn=None,
    surface_forcing=None,
    sponge=None,
    halo_refresh=None,
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
    halo_refresh : MPASOceanHaloRefresh, optional
        Distributed in-step halo refresh (stage-correctness lever): the
        biharmonic two-pass operators (B_h del4 / Smagorinsky / Leith on
        momentum, K_bih on tracers) consume 4 stencil hops — twice the
        ``halo_depth=2`` partition budget — so their intermediate
        Laplacians are refreshed mid-operator.  ``None`` (serial) is
        byte-identical.

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
    # Depth-dependent reference profile for ρ'.  Two flavors:
    #   * STATIC (preferred): if ``state.rho_ref_z`` was populated at
    #     init, ρ' = ρ − ρ_ref(z) using a frozen profile.  Cuts the
    #     partial-cell PGF residual ~24× without the dynamic version's
    #     positive-feedback drift (project_mpas_etopo_instability.md
    #     §8g).  Always wins when present.
    #   * DYNAMIC (legacy / discouraged): wet-cell mean recomputed every
    #     call (``use_baroclinic_rho_ref=True``).  NaN'd at day 60 on
    #     ETOPO+ico4 because ρ_ref chases T,S drift.  Kept only for
    #     back-compat / comparison.
    _rho_ref_z_static = (
        state.rho_ref_z.data if state.rho_ref_z is not None else None
    )
    _use_dd_rho_ref = getattr(config, "use_baroclinic_rho_ref", False)
    if _use_dd_rho_ref and isinstance(z_coord, OceanPartialCellCoordinate):
        _is_active_3d = z_coord.is_active.astype(T_3d.dtype)
    else:
        _is_active_3d = None
    # Integrate the baroclinic pressure anomaly p' against the actual
    # partial-cell thickness h_k (NEMO ``ln_hpg_zps`` / MITgcm) ONLY for
    # the Adcroft scheme.  This is the matched partner of the
    # ``pgf_scheme == "adcroft"`` Adcroft–Campin face-correction branch
    # below and MUST fire under the same guard: the AC correction reads
    # centroid depths from h_partial, so p' has to be on the h_partial
    # grid too (commit 62913cbe6 fixed that mismatch by enabling
    # ``use_h_actual_pgf`` for the adcroft comparison runs).
    #
    # The bare "centered" scheme carries NO such correction.  Integrating
    # p' on h_partial places p'[k] at each cell's *actual* centroid
    # depth, which differs across a bottom-level step; the centered
    # ``gradient_edge(p')`` then differences pressures at mismatched
    # depths, injecting an uncompensated spurious ∇p' that blows up the
    # rest state on a seamount step (max|u| 5.5e-4 → 1.2e-1 over 1 h).
    # Centered therefore always integrates p' on the dz_ref reference-
    # depth grid (its stable, common-depth treatment) regardless of
    # ``use_h_actual_pgf``.  smc03/ahh08/zero drop p' from the Bernoulli
    # scalar entirely, so the choice is moot for them and they too keep
    # the dz_ref default.  When unset, falls back to dz_ref (legacy z*).
    _use_h_actual_pgf = (
        getattr(config, "use_h_actual_pgf", False)
        and getattr(config, "pgf_scheme", "centered") == "adcroft"
    )
    if _use_h_actual_pgf and isinstance(z_coord, OceanPartialCellCoordinate):
        _h_for_pgf = h_k
    else:
        _h_for_pgf = None
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T_3d, S_3d, mask,
        lambda field: _fill_land_cells_mpas(field, mask),
        eos_fn, z_coord.dz_ref, rho_0, g,
        n_iter=2,
        use_depth_dependent_ref=_use_dd_rho_ref,
        is_active_3d=_is_active_3d,
        h_actual=_h_for_pgf,
        rho_ref_z_static=_rho_ref_z_static,
        allow_baroclinic_f32=True,   # opt-in f32-EOS lever (LEGOESM_BAROCLINIC_F32)
    )

    # Fill land cells in p_prime before gradient_edge so the 2-cell
    # stencil sees smooth values at coastlines.
    p_prime = _fill_land_cells_mpas(p_prime, mask)

    # ---- Edge mask for land boundaries ----
    edge_mask = mask[c1] * mask[c2]  # 1 only if both cells are ocean

    # ---- Per-level edge mask for partial-cell step edges ----
    # On a step edge (one cell deeper than the other), the cell-level
    # ``edge_mask`` is 1 (both cells wet) but the deeper cell's lower
    # levels have no real water on the shallow neighbor's side.
    # ``gradient_edge_3d`` doesn't know that and computes ∇p' there
    # using the FILLED tracer values from the dry side — a non-physical
    # pressure gradient that drives a phantom du_dt at the bottom
    # partial cell on the deeper side, growing 1000×/step (root cause
    # of the seamount rest-state explosion).  Build a per-level mask
    # ``edge_mask_3d[edge, k] = 1`` only when ``k <= maxLevelEdgeBot``
    # and apply it to ``du_dt_full`` so tendencies below the shallower
    # neighbor's seafloor are exactly zero.  On legacy z-star (every
    # column full) ``maxLevelEdgeBot = nlev-1`` everywhere → the mask
    # is identically ``edge_mask[:, None]`` → bit-exact regression.
    if isinstance(z_coord, OceanPartialCellCoordinate):
        bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
        nlev_loc = u_3d.shape[1]
        k_idx = jnp.arange(nlev_loc, dtype=bot_e.dtype)
        edge_mask_3d = (
            (k_idx[None, :] <= bot_e[:, None]).astype(u_3d.dtype)
            * edge_mask[:, None]
        )
    else:
        edge_mask_3d = jnp.broadcast_to(
            edge_mask[:, None], u_3d.shape,
        ).astype(u_3d.dtype)

    # ---- Depth-averaged velocity and perturbation ----
    # The baroclinic step must operate on PERTURBATION velocity
    # u' = u - u_bar to avoid double-counting with the barotropic
    # solver.  The barotropic solver handles the depth-mean Coriolis,
    # pressure gradient, and KE; the baroclinic step handles only the
    # vertical shear (perturbation) component.
    # (Matches latlon C-grid: ocean_pe_latlon_cgrid.py:256-266)
    # ---- Edge thickness on partial cells (P2 of MPAS realistic-geometry plan) ----
    # On a partial-cell coordinate, two distinct edge thicknesses are
    # needed (Petersen 2015 §3.4):
    #   * h_e_3d         — min-rule (MITgcm hFacZ): the flux-closure /
    #     metric thickness used by the depth-averaging weight, PV-flux
    #     normalization, vertical momentum advection cross-section, and
    #     bottom-drag layer thickness.
    #   * h_e_continuity — donor-cell upstream: the *advected* h in
    #     ``div(h u)`` for the continuity equation.  A centered average
    #     leaks thickness from below the seafloor through a step.
    # On the legacy z-star coordinate every column is a full cell, so
    # both reduce to the centered mean ``0.5*(h[c1]+h[c2])`` bit-exactly.
    if isinstance(z_coord, OceanPartialCellCoordinate):
        h_e_3d = min_cell_to_edge(h_k, mesh)
        h_e_continuity = donor_cell_to_edge(h_k, u_3d, mesh)
    else:
        h_e_3d = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
        h_e_continuity = h_e_3d
    # Both ``H_e`` and ``u_bar`` numerator share the ``h_e_3d`` weight
    # on the level axis — fuse into one stacked column reduction.
    # NOTE (#517 item 1/5): NOT routed through the shared
    # depth_average_to_faces / column_depth helpers.  Splitting this
    # fused ``jnp.stack``+single-``jnp.sum`` into two separate reductions
    # changes XLA's fusion in the full MPAS step and drifts the seamount
    # centered-scheme transport at ~1e-10 (caught by
    # test_seamount_centered_stable_over_steps), so the fused form is
    # kept verbatim to stay byte-identical.  H_e additionally is reused
    # by F_slow_u below; u_bar floors the divisor at a bare 1e-10 ON TOP
    # of H_e's min_water_column_m floor (a divergent second floor).
    _u_pair = jnp.sum(jnp.stack([h_e_3d, u_3d * h_e_3d], axis=-1), axis=1)
    H_e = jnp.maximum(_u_pair[..., 0], config.min_water_column_m)
    u_bar = _u_pair[..., 1] / jnp.maximum(H_e, 1e-10)
    u_bar = u_bar * edge_mask  # (nEdges,)
    u_prime_3d = u_3d - u_bar[:, jnp.newaxis]  # (nEdges, nlev)

    # ---- Thickness flux and vertical velocity ----
    # Compute flux divergence BEFORE momentum tendencies because we need
    # w for vertical advection of momentum (issue #152).
    # Use ``h_e_continuity`` (donor-cell upstream) for the conservative
    # mass flux on partial cells; identical to ``h_e_3d`` on z-star.
    thickness_flux = u_3d * h_e_continuity * edge_mask[:, jnp.newaxis]  # (nEdges, nlev)
    div_flux = divergence_cell_3d(thickness_flux, mesh)  # (nCells, nlev)

    # Diagnose w from full-velocity flux divergence (matching latlon pattern:
    # ocean_pe_latlon_cgrid.py:302-305)
    w = diagnose_w_from_flux_div(
        div_flux, z_coord, thickness_weighted=True,
    )  # (nCells, nlev+1)

    # ---- Momentum tendencies (batched 3D) ----
    # Split-explicit Coriolis approach: the PV flux below uses ONLY
    # relative vorticity q = ζ/h (zero_f at line 454), NOT the full
    # PV (f+ζ)/h.  Planetary Coriolis f×u is applied SEPARATELY:
    #   - f×u_bar: online in the barotropic solver (Heun scheme)
    #   - f×u_prime: forward-backward Matsuno in step()
    # This is the standard split-explicit convention (matches POP,
    # MPAS-Ocean production).  F_slow_u therefore contains NO
    # planetary Coriolis — see comment at line ~643.
    #
    # NOTE: issue #160 tracks a potential future refactor to the
    # MOM6-style approach where full PV = (f+ζ)/h enters through
    # the PV flux and no separate Coriolis substep is needed.
    # That is NOT what is currently implemented.

    # Kinetic energy from TOTAL velocity.  Multiply by edge_mask first
    # (defense-in-depth per the ocean-modeling audit) so dry-edge u
    # contributions are zeroed out before squaring; suppresses spurious
    # coastal Kelvin-wave ringing if upstream code lets nonzero u into
    # a dry edge.  edge_mask is 0 or 1 → the multiplication is a no-op
    # on flat-bottom z-star where dry edges already carry u=0.
    ke = kinetic_energy_cell_3d(
        u_3d * edge_mask[:, jnp.newaxis], mesh,
    )  # (nCells, nlev)

    # PGF scheme dispatch — looked up early so the SMC03 / AHH08 paths
    # can split ``p'/rho_0`` out of the batched Bernoulli gradient.
    # Allowed values (see ``MPASOceanConfig.pgf_scheme`` docstring):
    #   "centered" : bare ``gradient_edge(p'/rho_0)`` (default; correct
    #                on z-star, has O(1 cm/s) shelf-break residual on
    #                partial cells).
    #   "adcroft"  : centered + Adcroft & Campin (2004) face-correction
    #                additive term (eliminates the partial-cell
    #                cancellation error to leading order).
    #   "smc03"    : Shchepetkin & McWilliams (2003) density-Jacobian
    #                PGF (per-column harmonic-slope ρ(z) reconstruction
    #                evaluated at a face-reference depth).
    #   "ahh08"    : Adcroft, Hallberg & Hill (2008) analytic finite-
    #                volume PGF.  Closed-form ``∫p dz`` per cell using
    #                the Wright EOS rational form; differences
    #                face-averaged pressures over the common wet face.
    #                Machine-zero rest state on partial cells regardless
    #                of step structure (the property SMC03 only achieves
    #                on linear ρ).
    pgf_scheme = getattr(config, "pgf_scheme", "centered")
    use_smc03 = (
        pgf_scheme == "smc03"
        and isinstance(z_coord, OceanPartialCellCoordinate)
    )
    use_ahh08 = (
        pgf_scheme == "ahh08"
        and isinstance(z_coord, OceanPartialCellCoordinate)
    )
    # ``"zero"`` is a diagnostic-only scheme that drops both ``p'/rho_0``
    # and any partial-cell correction from the momentum tendency.  Used
    # to test whether PGF is the energy injector for a given mode (per
    # §8f's monkey-patch experiment; now exposed as a production-style
    # config so the §8j-era PGF=0 test on audit-fixed runs can be
    # reproducible).  Should NEVER be set in production: removes the
    # restoring force that drives the entire baroclinic dynamics.
    use_zero_pgf = pgf_scheme == "zero"
    # AHH08 evaluates the Wright EOS analytically and is therefore tied
    # to the Wright EOS path.  Linear EOS users should stay on adcroft
    # / smc03 (the analytic integral simplifies trivially to a quadratic
    # but adds nothing — already linear-exact under SMC03).
    if use_ahh08 and getattr(config, "eos", "wright") != "wright":
        raise ValueError(
            f"pgf_scheme='ahh08' requires eos='wright'; got eos="
            f"{config.eos!r}.  Switch to a different PGF scheme or "
            f"the Wright EOS.",
        )

    # Bernoulli scalar.  Default = KE + p'/rho_0 so the centered/adcroft
    # paths get both the kinetic-energy gradient and the bare pressure
    # gradient in a single batched ``gradient_edge_3d`` call.  Under
    # SMC03 the pressure gradient is computed by a separate column-aware
    # operator (no longer a gradient-of-a-scalar), so we drop p'/rho_0
    # here and add it back as ``pgf_smc03 / rho_0`` after the batched
    # call.  ``ke`` and ``p_prime`` share shape ``(nCells, nlev)`` so
    # the downstream slicing/reshape is unaffected.
    if use_smc03 or use_ahh08 or use_zero_pgf:
        # AHH08 and SMC03 both compute the pressure gradient directly
        # (not as a gradient-of-a-scalar), so drop p'/rho_0 from the
        # Bernoulli scalar and add the scheme's PGF acceleration after
        # the batched gradient call.  ``"zero"`` drops it for the
        # different reason of producing zero PGF acceleration entirely
        # (diagnostic only).
        bernoulli = ke                      # (nCells, nlev)
    else:
        bernoulli = ke + p_prime / rho_0    # (nCells, nlev)

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

    # ---- Partial-cell PGF correction dispatch (P3 / P3c) ----
    # ``pgf_scheme`` is resolved above the Bernoulli build so the SMC03
    # path can drop ``p'/rho_0`` from the batched gradient.  Three
    # branches (matching that resolution; ``use_smc03`` already encodes
    # the partial-cell precondition):
    #
    #   "adcroft"  : add Adcroft & Campin (2004) face correction to
    #                ``grad_B = ∇(KE + p'/rho_0)``.  Bit-exact zero on
    #                full cells, so flat-bottom z-star is unaffected.
    #                Centroid_depth uses the eta=0 reference (matching
    #                the rho_prime / p_prime reference; using live eta
    #                breaks the rest-state machine-zero claim once eta
    #                evolves — see ``ocean_pe_latlon_cgrid.py:961``).
    #   "smc03"    : ``grad_B`` currently holds only ``∇KE`` (Bernoulli
    #                was rebuilt without p'/rho_0); add the SMC03
    #                column-aware density-Jacobian PGF acceleration
    #                ``pgf_smc03 / rho_0``.  Required for stability
    #                over real bathymetry (the Adcroft correction is a
    #                thin spike at partial-cell interfaces that drives
    #                a 2Δz vertical mode on ETOPO; SMC03's per-column
    #                ρ(z) reconstruction is smooth in z).  See
    #                ``docs/ocean/experiments/density_jacobian_pgf_mpas.md``.
    #   "centered" : no correction; ``grad_B`` already contains the
    #                bare ``∇(KE + p'/rho_0)``.
    if pgf_scheme == "adcroft" and isinstance(z_coord, OceanPartialCellCoordinate):
        centroid_depth = compute_centroid_depth(
            jnp.zeros_like(eta), H_bathy, z_coord,
        )
        ac_correction = partial_cell_pgf_correction_edge(
            centroid_depth, rho_prime, mesh, g, rho_0,
        )
        grad_B = grad_B + ac_correction
    elif use_smc03:
        pgf_smc03 = density_jacobian_pgf_smc03_mpas(
            rho_prime, z_coord.h_partial, z_coord.is_active, mesh, g,
        )                                       # (nEdges, nlev) [Pa/m]
        grad_B = grad_B + pgf_smc03 / rho_0     # acceleration [m/s^2]
    elif use_ahh08:
        # AHH08 needs T, S directly (not rho_prime) — the Wright EOS
        # is evaluated internally by the analytic integrator.  Pass
        # h_partial unconditionally; on z-star (no partial cells)
        # ``use_ahh08`` is False so we never reach this branch.
        from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
            density_jacobian_pgf_ahh08_mpas,
        )
        pgf_ahh08 = density_jacobian_pgf_ahh08_mpas(
            T_3d, S_3d, z_coord.h_partial, mesh, g,
        )                                       # (nEdges, nlev) [Pa/m]
        grad_B = grad_B + pgf_ahh08 / rho_0     # acceleration [m/s^2]
    elif pgf_scheme not in ("centered", "adcroft", "smc03", "ahh08", "zero"):
        raise ValueError(
            f"pgf_scheme={pgf_scheme!r} unsupported on MPAS "
            f"(allowed: 'centered', 'adcroft', 'smc03', 'ahh08', 'zero')",
        )

    # PV flux: RELATIVE vorticity only, q = ζ(u)/h. Planetary Coriolis is
    # applied separately (a) as online f·v_t(u_bar) in the barotropic
    # substep and (b) as a forward-backward Matsuno correction on the
    # 3D perturbation in the step() function. This matches the lat-lon
    # C-grid pattern and sidesteps the frozen-Coriolis instability that
    # follows from carrying planetary Coriolis in the baroclinic-step
    # depth-mean forcing (τ ~ 1/f ≈ 0.2 days at mid-latitudes; see #160).
    #
    # P4 of MPAS realistic-geometry plan: on partial-cell coordinates,
    # use the hybrid vertex thickness (active-renormalized kite-area
    # mean in the interior, min-over-active fallback at coasts) for
    # the q = ζ/h_v normalization.  Pure kite-mean lets a thin partial
    # vertex transmit O(1) PV flux from a deep neighbor and drive
    # spurious coastal currents (per the dycore audit).  The
    # downstream ``du_dt_full * edge_mask`` already zeros the
    # perimeter contribution from edges with one dry cell, so we do
    # NOT additionally mask the Thuburn tangential reconstruction
    # (per audit recommendation; keeps the discrete summation-by-
    # parts identity intact on the wet sub-mesh).  Default
    # ``pv_scheme="enstrophy"`` (already in MPASOceanConfig) is the
    # right choice for realistic bathymetry per the audit.
    zero_f = jnp.zeros_like(mesh.fVertex)
    if isinstance(z_coord, OceanPartialCellCoordinate):
        zeta = curl_vertex_3d(u_3d, mesh)
        h_v_hybrid = vertex_thickness_hybrid(
            h_k, mesh, alpha=config.vertex_thickness_alpha,
        )
        h_v_safe = jnp.maximum(h_v_hybrid, 1.0e-10)
        q_relative = (zeta + zero_f[:, None]) / h_v_safe
    else:
        q_relative = potential_vorticity_vertex_3d(
            u_3d, h_k, zero_f, mesh,
        )  # (nVertices, nlev); equals ζ/h_v_kite

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
    elif config.pv_scheme == "enstrophy":
        pv_flux = pv_flux_enstrophy_conserving_3d(
            u_3d, h_k, q_relative, mesh, h_edge_3d=h_e_3d,
        )
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy', 'mixed'."
        )

    # Horizontal viscosity on TOTAL velocity (not perturbation).
    # The depth-average of the viscous tendency enters F_slow and damps
    # the barotropic mode.  This matches MOM6, NEMO, POP, and the
    # lat-lon implementation (Hallberg 1997).  Previously acted on
    # u_prime_3d, which zeroed the barotropic viscous contribution.
    # On MPAS, K_zeta_bih already provided barotropic damping (it acts
    # on u_3d), so the effect is secondary — but switching to u_3d
    # for A_h/B_h/C_smag is the correct formulation.
    visc = jnp.zeros_like(u_3d)
    # Per-edge equatorial-boost factor for A_h (and B_h).  Boosts
    # damping at low latitudes where the implicit-CN solver's Coriolis
    # restoring fails (f→0).  Diagnosed in project_mpas_etopo_
    # instability.md §"equatorial mode": top-100 hot-spot edges
    # cluster at mean |lat|=22°, peaking in equatorial Pacific.
    _eq_boost = config.equatorial_visc_boost
    if _eq_boost > 0:
        # Tight Gaussian centered at equator (default sigma=5° matches lat-lon
        # production config).  Previous cos²(lat) was too wide — it
        # overdamped real mid-latitude dynamics while the instability is
        # confined to |lat| < 10°.
        if config.equatorial_visc_sigma_deg <= 0:
            raise ValueError(
                "equatorial_visc_sigma_deg must be > 0 when "
                "equatorial_visc_boost > 0 (Gaussian width divides latEdge; "
                f"got {config.equatorial_visc_sigma_deg})."
            )
        _sigma_rad = jnp.radians(config.equatorial_visc_sigma_deg)
        _lat_e = mesh.latEdge.astype(u_3d.dtype)
        _gauss = jnp.exp(-0.5 * (_lat_e / _sigma_rad) ** 2)
        _lat_factor = (1.0 + _eq_boost * _gauss)[:, jnp.newaxis]  # (nEdges, 1)
    else:
        _lat_factor = 1.0
    _mid_refresh = None if halo_refresh is None else halo_refresh.edges
    if config.A_h > 0 and config.B_h > 0:
        _del2_u_visc = vector_laplacian_del2_3d(u_3d, mesh)
        visc = visc + config.A_h * _lat_factor * _del2_u_visc
        # vector_laplacian_del4 = -del2(del2); fold the sign into the
        # subtraction so the arithmetic matches ``+ B_h * del4``.
        # [stage-halo T1] the shared intermediate feeds a SECOND del2
        # (4 hops total > halo_depth) — refresh its ring first; the A_h
        # term above already consumed only 2 hops and stays as-is.
        _del2_for_del4 = _del2_u_visc
        if _mid_refresh is not None:
            (_del2_for_del4,) = _mid_refresh(_del2_for_del4)
        visc = visc - config.B_h * _lat_factor * vector_laplacian_del2_3d(_del2_for_del4, mesh)
    elif config.A_h > 0:
        visc = visc + config.A_h * _lat_factor * vector_laplacian_del2_3d(u_3d, mesh)
    elif config.B_h > 0:
        visc = visc + config.B_h * _lat_factor * vector_laplacian_del4_3d(
            u_3d, mesh, mid_refresh=_mid_refresh)

    # Flow-dependent Smagorinsky biharmonic viscosity
    if config.C_smag > 0:
        visc = visc + smagorinsky_biharmonic_3d(
            u_3d, mesh, config.C_smag, mid_refresh=_mid_refresh)

    # Flow-dependent Smagorinsky Laplacian viscosity
    # Unlike the biharmonic variant, this damps ALL scales where strain
    # is large (not just grid-scale).  Effective at coarse resolution
    # (~120 km) where the biharmonic Smagorinsky produces zero.
    # Matches the lat-lon ``C_smag_lap`` scheme.
    if getattr(config, "C_smag_lap", 0.0) > 0:
        visc = visc + _lat_factor * smagorinsky_laplacian_3d(
            u_3d, mesh, config.C_smag_lap)

    # Flow-dependent Leith biharmonic viscosity
    if getattr(config, "C_leith", 0.0) > 0:
        visc = visc + leith_biharmonic_3d(
            u_3d, mesh, config.C_leith,
            modified=getattr(config, "C_leith_modified", False),
            mid_refresh=_mid_refresh)

    # Biharmonic dissipation on relative vorticity ζ (scale-selective damping
    # of grid-scale vorticity patterns — notably the ζ-checkerboard null
    # mode of the energy-conserving PV flux).  This is applied to the total
    # velocity u_3d (not u_prime_3d) because ζ is a derived quantity and the
    # full ζ (including the planetary-Coriolis-free baroclinic+barotropic ζ)
    # carries the null-mode amplitude.  Invisible to ``B_h·del4(u)`` because
    # the null mode lives in the kernel of the discrete curl-to-velocity map.
    if config.K_zeta_bih > 0:
        # [stage-halo T3] vertex-channel mid-refresh: the curl -> vertex-
        # Laplacian -> tangential-gradient chain is 3 hops (codex r1 #2 —
        # the NEMO-match recipe runs K_zeta_bih=1e14, so without this the
        # stage-correct claim was false for the production config).
        visc = visc + config.K_zeta_bih * biharmonic_vorticity_del4_3d(
            u_3d, mesh,
            mid_refresh=(None if halo_refresh is None
                         else halo_refresh.vertices))

    # Vertical advection of perturbation momentum (#171 Level-1).
    w_e = 0.5 * (w[c1] + w[c2])  # (nEdges, nlev+1)
    vert_adv_u = flux_form_vertical_momentum_advection(
        u_prime_3d, w_e, h_e_3d,
    )

    # Full nonlinear momentum tendency.  Use the per-level edge mask
    # (zero below the shallower neighbor's seafloor on partial cells)
    # so spurious ∇p' from dry-cell-filled tracers can't drive a
    # phantom tendency at the bottom partial cell of the deeper column.
    du_dt_full = (-grad_B + pv_flux + visc + vert_adv_u) * edge_mask_3d

    # Bottom drag on full velocity (not perturbation) — the ocean floor
    # sees the total flow.  Applied before F_slow_u computation so the
    # depth-averaged drag enters the barotropic solver via slow forcing.
    # r is in [m/s]: du/dt = -r * u / dz_bottom  (resolution-independent stress).
    #
    # On partial-cell coordinates the ocean floor sits at the per-edge
    # bottom level ``maxLevelEdgeBot = min(bot[c1], bot[c2])``, which
    # generally differs from ``nlev-1``.  Applying drag at level
    # ``nlev-1`` unconditionally silently skips drag on every edge
    # whose true bottom is shallower (h_e at nlev-1 is 0 there → the
    # 1e-10 floor masks the divide and the drag tendency is ~0). P3.5
    # of the realistic-geometry plan: scatter the drag to each edge's
    # actual bottom level via a one-hot expansion.
    # Bottom drag is always applied explicitly here so that its
    # depth-mean enters F_slow_u (the barotropic forcing).  The
    # implicit vertical path in step() does NOT double-count —
    # it handles only vertical viscosity/diffusivity, not drag.
    _drag_scheme = validate_bottom_drag_scheme(
        str(getattr(config, "bottom_drag_scheme", "legacy")))
    if config.bottom_drag_r > 0 or _drag_scheme != "legacy":
        H_BBL = getattr(config, "bottom_drag_bbl_thickness", 0.0)
        # Quadratic-with-floor drag (MOM6 DRAG_BG_VEL).  When
        # ``bottom_drag_bg_velocity > 0``, the effective drag coefficient
        # scales with velocity: ``r_eff = (r/u_bg) * sqrt(u² + u_bg²)``.
        # Recovers linear ``r`` at |u|→0, quadratic ``Cd·|u|`` at high
        # speed.  u_bg=0 → bit-exact legacy linear drag.
        _u_bg = float(getattr(config, "bottom_drag_bg_velocity", 0.0))
        if _drag_scheme != "legacy":
            # NEMO zdfdrg drag law (np_non_lin / np_loglayer) on the Voronoi
            # mesh, with NEMO's exact operator placement: rCdU is evaluated
            # PER CELL (t-point analogue) from that cell's own bottom speed
            # and bottom thickness — |U|² = 2·KE (Ringler discrete kinetic
            # energy, the Voronoi analogue of the t-point 2-component
            # average) and h = the cell's bottom thickness (e3t(mbkt)) —
            # and the resulting coefficient is THEN 2-point averaged to the
            # edge over cellsOnEdge (NEMO dynzdf's rCdU face average).
            # Averaging the inputs instead would not commute with the
            # nonlinear Cd(h)·√(s²+ke0) on slopes (codex r1 #3).
            if isinstance(z_coord, OceanPartialCellCoordinate):
                _bot_c = jnp.maximum(z_coord.bottom_level, 0)[:, jnp.newaxis]
                _ke_bot = jnp.take_along_axis(ke, _bot_c, axis=1)[:, 0]
                _h_bot_c = jnp.take_along_axis(h_k, _bot_c, axis=1)[:, 0]
            else:
                _ke_bot = ke[:, -1]
                _h_bot_c = h_k[:, -1]
            _r_cell = nemo_drag_r_from_speed_sq(
                2.0 * _ke_bot, _h_bot_c,
                scheme=_drag_scheme,
                cd0=float(getattr(config, "bottom_drag_cd0", 1.0e-3)),
                cd_max=float(getattr(config, "bottom_drag_cdmax", 0.1)),
                z0=float(getattr(config, "bottom_drag_z0", 3.0e-3)),
                ke0=float(getattr(config, "bottom_drag_ke0", 2.5e-3)),
                von_karman=constants.kappa_von_karman,
            )
            _c1 = mesh.cellsOnEdge[0]
            _c2 = mesh.cellsOnEdge[1]
            _r_eff = (0.5 * (_r_cell[_c1] + _r_cell[_c2])
                      )[:, jnp.newaxis]  # (nEdges, 1) — broadcasts over levels
        elif _u_bg > 0.0:
            _Cd_eq = config.bottom_drag_r / _u_bg
            # Compute per-edge, per-level effective r from the speed at
            # each level (the drag sees the FULL velocity, not just bottom).
            _speed_sq = u_3d ** 2  # edge-normal component squared
            _r_eff = _Cd_eq * jnp.sqrt(_speed_sq + _u_bg ** 2)  # (nEdges, nlev)
        else:
            _r_eff = config.bottom_drag_r  # scalar, legacy linear

        if H_BBL > 0:
            # Distributed BBL drag (Killworth & Edwards 1999, MOM6
            # ``BBL_thick_min``): spread drag over a fixed Ekman thickness
            # near the seafloor instead of applying ``r·u/h_partial_bot``
            # to a single (possibly very thin) partial cell.  Required
            # on partial-cell coordinates with realistic bathymetry —
            # the thinnest ETOPO+ico4 partial-cell bottom is 0.28 m, so
            # the legacy ``r·dt/h_bot`` ratio reaches 2 at ``r=1.1e-3``,
            # ``dt=500`` and the explicit drag sign-reverses ``u`` (per
            # the 2026-05-03 diagnostic).  Bounds per-cell drag by
            # ``r·u/H_BBL``.  Grid-agnostic helper shared with lat-lon
            # C-grid PE.  Multiply by ``edge_mask`` to zero dry edges.
            #
            # For quadratic drag (_r_eff is per-edge-level), we inline
            # the BBL computation instead of calling the scalar-r helper.
            if isinstance(_r_eff, jnp.ndarray) and _r_eff.ndim >= 2:
                # Per-level r_eff — inline the BBL overlap computation
                z_half = jnp.concatenate(
                    [jnp.zeros((h_e_3d.shape[0], 1), dtype=h_e_3d.dtype),
                     -jnp.cumsum(h_e_3d, axis=-1)], axis=-1)
                z_top = z_half[:, :-1]; z_bot = z_half[:, 1:]
                z_seafloor = z_half[:, -1:]; bbl_top = z_seafloor + H_BBL
                overlap = jnp.maximum(0.0,
                    jnp.minimum(z_top, bbl_top) - jnp.maximum(z_bot, z_seafloor))
                h_safe_bbl = jnp.maximum(h_e_3d, 1e-10)
                drag_3d = -_r_eff * u_3d * overlap / (h_safe_bbl * H_BBL)
            else:
                from legoesm.ocean.dynamics.ocean_tendency_common import (
                    bbl_distributed_drag_face_column,
                )
                drag_3d = bbl_distributed_drag_face_column(
                    u_3d, h_e_3d, _r_eff, H_BBL,
                )
            du_dt_full = du_dt_full + drag_3d * edge_mask[:, jnp.newaxis]
        elif isinstance(z_coord, OceanPartialCellCoordinate):
            # Legacy single-cell drag at maxLevelEdgeBot.
            # WARNING: CFL-violates at thin partial cells (see
            # docs/ocean/experiments/density_jacobian_pgf_mpas.md §8a).
            # Prefer ``bottom_drag_bbl_thickness > 0`` on real bathymetry.
            bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
            # Edges with at least one dry neighbor have bot_e < 0 (since
            # dry cells have bottom_level = -1); mask via valid_edge below.
            bot_e_safe = jnp.maximum(bot_e, 0)
            nlev_u = u_3d.shape[1]
            is_bot = (
                jnp.arange(nlev_u, dtype=bot_e.dtype)[None, :]
                == bot_e_safe[:, None]
            )  # (nEdges, nlev) bool, exactly one True per edge
            valid_edge = (bot_e >= 0) & (edge_mask > 0.5)
            valid_edge_dt = valid_edge.astype(u_3d.dtype)
            h_at_bot = jnp.sum(
                jnp.where(is_bot, h_e_3d, 0.0), axis=1,
            )
            u_at_bot = jnp.sum(
                jnp.where(is_bot, u_3d, 0.0), axis=1,
            )
            dz_bot_e = jnp.maximum(h_at_bot, 1.0e-10)
            # For quadratic drag, compute r_eff at the bottom level
            if isinstance(_r_eff, jnp.ndarray) and _r_eff.ndim >= 2:
                _r_at_bot = jnp.sum(jnp.where(is_bot, _r_eff, 0.0), axis=1)
            else:
                _r_at_bot = _r_eff
            drag_at_bot = (
                -_r_at_bot * u_at_bot / dz_bot_e * valid_edge_dt
            )
            du_dt_full = du_dt_full + (
                drag_at_bot[:, None] * is_bot.astype(du_dt_full.dtype)
            )
        else:
            # Legacy z-star: every column has a full bottom cell at nlev-1.
            dz_bot_e = jnp.maximum(h_e_3d[:, -1], 1e-10)
            if isinstance(_r_eff, jnp.ndarray) and _r_eff.ndim >= 2:
                _r_bot = _r_eff[:, -1]
            else:
                _r_bot = _r_eff
            du_dt_full = du_dt_full.at[:, -1].add(
                -_r_bot * u_3d[:, -1] / dz_bot_e * edge_mask)

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
    du_dt_3d = (du_dt_full - F_slow_u[:, jnp.newaxis]) * edge_mask_3d

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
        # Use per-level edge mask on partial cells: zero the tracer
        # gradient at levels below the shallower neighbor's seafloor.
        # Without this, the gradient sees T_active vs T=0 (inactive
        # levels are zero-filled) and injects a massive spurious flux
        # into thin partial cells at step edges.
        # ``_tracer_grad_flat_pre`` is (nEdges, nlev*n_tracers) with
        # interleaved layout [gradT_0, gradS_0, gradT_1, gradS_1, ...].
        # Repeat mask per-level to match: [mask_0, mask_0, mask_1, mask_1, ...].
        _tracer_edge_mask = jnp.repeat(edge_mask_3d, n_tracers, axis=1)  # (nEdges, nlev*2)
        grad_flat = _tracer_grad_flat_pre * _tracer_edge_mask
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
        # Inline the bilaplacian: inner = ``div(grad*edge_mask_3d) *
        # cell_mask`` (already partially computed above as
        # ``_inner_lap_div``); outer = ``laplacian_cell_3d(inner,
        # mask=mask)``.
        inner_lap_masked = _inner_lap_div * mask[:, jnp.newaxis]
        # [stage-halo T2] the bilaplacian's inner div(grad) already
        # consumed 2 hops; the outer grad->div pair needs a fresh ring
        # (the K_h branch's use of _inner_lap_div is a FINAL tendency —
        # no further hops — so only this copy is refreshed).
        if halo_refresh is not None:
            (inner_lap_masked,) = halo_refresh.cells(inner_lap_masked)
        outer_grad = gradient_edge_3d(inner_lap_masked, mesh) * _tracer_edge_mask
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
    # When ``implicit_vertical_mixing`` is enabled (default), vertical
    # viscosity/diffusivity is applied as a backward-Euler implicit solve
    # in ``step()`` (unconditionally stable).  The explicit tendency here
    # is skipped.  The legacy explicit path is kept for back-compat when
    # the flag is False.
    if not config.implicit_vertical_mixing:
        dz_half = z_coord.dz_half_ref  # (nlev-1,)
        dz = z_coord.dz_ref  # (nlev,)

        # Vertical viscosity on perturbation velocity: d/dz(A_v * du'/dz)
        # Mask by edge_mask_3d to zero sub-seafloor levels on partial cells.
        # Without this, the diffusion operator sees the full dz_ref*J metric
        # (nonzero at all levels) and injects momentum below the seafloor at
        # step edges, creating a positive feedback that drives SSH blow-up
        # when A_v is large (e.g., KPP-produced viscosity).
        du_dt_3d = du_dt_3d + _vertical_diffusion(
            u_prime_3d, dz_half, dz, jacobian=jacobian, coeff=config.A_v,
            is_edge=True, mesh=mesh,
        ) * edge_mask_3d

        # Vertical tracer diffusion (per-tracer; axis -1 of the input is nlev,
        # so vmap over the trailing tracer axis to get one batched kernel).
        def _vdiff(q):
            return _vertical_diffusion(
                q, dz_half, dz, jacobian=jacobian, coeff=config.K_v,
                is_edge=False, mesh=mesh,
            )

        vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
        vdiff_stack = vdiff_stack * mask[:, jnp.newaxis, jnp.newaxis]
        dT_dt_3d = dT_dt_3d + vdiff_stack[..., 0]
        dS_dt_3d = dS_dt_3d + vdiff_stack[..., 1]

    # ---- Physics (surface forcing, bottom drag, etc.) ----
    if physics_fn is not None:
        phys = physics_fn(state, mesh, z_coord, surface_forcing)
        # Mask physics du_dt by edge_mask_3d: KPP (and any future physics
        # producing edge tendencies) uses dz_ref*J_edge as its vertical
        # metric, which is nonzero at ALL levels including below the
        # shallower neighbor's seafloor on partial cells.  Without this
        # mask, the diffusion leaks momentum into sub-seafloor levels at
        # step edges.  That leaked momentum is not removed by any
        # subsequent masking (Coriolis and reconciliation use 2D
        # edge_mask only), accumulates exponentially via the KPP
        # diffusion feedback (sub-seafloor u creates gradients that
        # feed the next step's diffusion), and ultimately drives the
        # SSH explosion observed when KPP is activated on ETOPO.
        du_dt_3d = du_dt_3d + phys.du_dt.data * edge_mask_3d
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
        # When ``normalize_freshwater`` is on, remove the global area-mean of the
        # net freshwater flux so the virtual-salt closure conserves GLOBAL SALT
        # (the same correction the free-surface eta path applies for volume in
        # ocean_model_mpas.step) -- without it an unbalanced ∮(P-E+R) drifts the
        # mean salinity even though volume is conserved.
        #
        # FAIL-FAST under multi-rank MPAS (codex round-2 #1/#2, round-3 placement):
        # the salt freshwater normalization below AND the sibling eta
        # normalization (ocean_model_mpas.step) are rank-local area-means with NO
        # owned-cell mask, so an MPI Voronoi run would silently halo-double-count
        # and apply inconsistent volume vs salt corrections.  Guard the SINGLE
        # reduction SOURCE here (the public ``MPASOceanModel.tendencies`` entry,
        # which ``_step_impl`` also routes through first) so a DIRECT tendency
        # call is refused too — not just a full step.  ``is_multi_process()``
        # alone misses the layout-less Voronoi MPI path, so also check
        # ``mpi_world_size()`` (the established MPAS fail-fast predicate).  Inert
        # single-rank.  Remove when owned-mask plumbing lands on both paths
        # (the freshwater helper already exposes ``owned_mask`` +
        # ``global_sum_if_distributed``).
        if bool(getattr(config, "normalize_freshwater", False)):
            import jax as _jax

            from legoesm.parallel.reductions import (
                is_multi_process, mpi_world_size,
            )
            # jax.process_count() covers the jax.distributed multi-controller
            # mode explicitly: is_multi_process() is deliberately mpi4jax-only
            # (reduction routing), but this is a REFUSAL guard — any
            # multi-process shape must trip it until owned-mask plumbing
            # lands (codex round-4).
            if (is_multi_process() or mpi_world_size() > 1
                    or _jax.process_count() > 1):
                raise NotImplementedError(
                    "normalize_freshwater=True under multi-rank MPAS is not yet "
                    "supported: the top-layer-salt and eta freshwater means are "
                    "rank-local (no owned-cell mask), so an MPI Voronoi run would "
                    "silently apply halo-double-counted and inconsistent volume "
                    "vs salt corrections.  Run single-rank, or set "
                    "normalize_freshwater=False, until owned-mask plumbing lands "
                    "on both paths."
                )
        # NEMO-style runoff depth spreading (sbcrnf rn_dep_max): the runoff
        # channel dilutes the top `_spread_arg` metres — a FLAT scalar
        # (runoff_depth_spread_m) OR the PER-CELL ln_rnf_depth_ini map
        # (runoff_depth_spread_map: small Arctic/Siberian rivers stay
        # near-surface for more shelf freshening).  Other channels stay at
        # the top cell; column-integral conservation unchanged.  Grid-agnostic
        # helper: the MPAS state is the flattened (nCells,) / (nCells, nlev)
        # analogue of the C-grid's per-column arrays.  Static config gate ->
        # legacy top-cell path stays untraced/bit-exact when neither is set.
        from legoesm.ocean.freshwater import resolve_runoff_spread_arg
        _spread_arg = resolve_runoff_spread_arg(config)
        if _spread_arg is not None:
            from legoesm.ocean.freshwater import (
                runoff_spread_virtual_salt_tendency_3d,
            )
            dS_fw_3d = runoff_spread_virtual_salt_tendency_3d(
                freshwater, config.S_ref, h_k, config.rho_0, mask,
                runoff_spread_m=_spread_arg, area=mesh.areaCell,
                normalize=bool(getattr(config, "normalize_freshwater", False)),
            )
            dS_dt_3d = dS_dt_3d + (dS_fw_3d * mask[:, None]).astype(
                dS_dt_3d.dtype)
        else:
            dS_dt_3d = apply_freshwater_virtual_salt_top(
                dS_dt_3d, freshwater, config.S_ref, h_k[:, 0], config.rho_0, mask,
                area=mesh.areaCell,
                normalize=bool(getattr(config, "normalize_freshwater", False)),
            )

    # ---- Real salt-mass flux (e.g. sea-ice brine rejection) ----
    # A top-layer salinity SOURCE distinct from the freshwater virtual-salt
    # dilution above: dS/dt = salt_flux*1e3/(rho_0*h_top).  Applied HERE (not in
    # physics_fn) so it shares the SAME canonical floored top-layer thickness
    # ``h_k[:, 0]`` the tracer update integrates mass against — so the injected
    # salt MASS equals salt_flux even on shallow/floored partial-top cells, and
    # it matches the freshwater virtual-salt scaling exactly.
    _sf_salt = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None
    if _sf_salt is not None:
        # Gate by surface-forcing scheme (fail closed).  ``surface_forcing`` is a
        # multi-consumer struct (KPP also reads it for buoyancy); the real salt-
        # mass SOURCE is only the coupler-driven path, so apply it ONLY under
        # ``scheme="external"`` (two-way coupling) or ``"none"`` (the bare
        # surface_forcing pass-through, e.g. OMIP).  Under prescribed/restoring/
        # combined the ocean has its own surface forcing, so a passed salt_flux
        # is a misconfiguration — raise rather than silently corrupt salinity.
        _phys = getattr(config, "physics", None)
        _sf_scheme = (
            getattr(_phys.surface_forcing, "scheme", "none")
            if _phys is not None and getattr(_phys, "surface_forcing", None) is not None
            else "none"
        )
        if _sf_scheme not in ("none", "external"):
            raise ValueError(
                f"surface_forcing.salt_flux supplied under surface_forcing "
                f"scheme {_sf_scheme!r}: the real salt-mass source is only "
                f"consumed under the coupler-driven 'external' (or 'none') "
                f"scheme. Use scheme='external' for two-way salt coupling, or "
                f"omit salt_flux.",
            )
        dS_salt = salt_flux_salinity_tendency(
            jnp.asarray(_sf_salt, dS_dt_3d.dtype), h_k[:, 0], config.rho_0,
        )
        dS_dt_3d = dS_dt_3d.at[:, 0].add(dS_salt * mask)

    # ---- Sponge layer relaxation ----
    # Cast sponge arrays to state dtype to prevent float64 promotion when
    # the precision policy stores state in float32 (crashes barotropic scan).
    if sponge is not None:
        dT_dt_3d, dS_dt_3d = apply_sponge_tracer_relaxation(
            dT_dt_3d, dS_dt_3d, T_3d, S_3d, sponge, mask=mask,
            expand_gamma_axis=-1,
        )
        # Edge velocity sponge (if reference velocity provided).
        # EXT-N1 guard (mirrors ocean_pe_latlon_cgrid._bc_sponge_relaxation):
        # a full-rank (per-cell, z-varying) gamma supports TRACER relaxation
        # only.  The momentum sponge below averages gamma to edges and then
        # broadcasts over the vertical with ``gamma_edge[:, None]``; a
        # ``(nCells, nlev)`` gamma would silently broadcast to
        # ``(nEdges, nlev, nlev)`` (a wrong-shaped momentum increment) instead.
        # Static shapes ⇒ trace-time error, never a silently wrong broadcast.
        if sponge.u_ref is not None:
            if sponge.gamma.ndim == T_3d.ndim:
                raise ValueError(
                    "3-D (full-rank per-cell) SpongeForcing.gamma supports "
                    "tracer relaxation only; u_ref requires a 1-D "
                    "(nCells,) horizontal gamma (the MPAS momentum sponge "
                    "averages gamma to edges and broadcasts over the "
                    "vertical).")
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
