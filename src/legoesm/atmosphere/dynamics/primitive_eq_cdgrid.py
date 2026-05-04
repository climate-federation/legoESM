"""FV3-inspired Hydrostatic Primitive Equations on the cubed-sphere (D-grid dynamics).

**Fidelity status: stabilized research path, not a faithful FV3 port.**

Key similarities to FV3:
- D-grid prognostic winds (cell corners, (6, n+1, n+1, nlev))
- Arakawa-Lamb gradient at D-grid corners
- Exact circulation-based vorticity
- C-grid mass flux for transport

Key differences from faithful FV3:
- Uses RK3 time integration (FV3 uses forward-backward splitting)
- Uses edge-midpoint stagger with corner averaging (FV3 uses true D-grid)
- Halo exchange uses interpolation (FV3 uses exact tile-edge coupling)
- No Lagrangian vertical coordinate (FV3 uses vertically Lagrangian remapping)

Operator staggering
-------------------
- Momentum:  D-grid prognostic, C-grid diagnostic (for KE / mass flux)
- Vorticity: cell centres (from D-grid circulation)
- Bernoulli / pressure gradient: Arakawa-Lamb gradient at D-grid corners
- Divergence: C-grid flux-form (exact mass conservation)
- Scalar diffusion: cell-centre (proper inter-face halo exchange)
- Wind diffusion: D-grid Laplacian with halo

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Simmons & Burridge (1981): Energy and Angular-Momentum Conserving Scheme
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    FV3HydrostaticState,
    FV3HydrostaticTendencies,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    dgrid_to_center_vector,
    cgrid_divergence,
    dgrid_vorticity,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
)
from legoesm.core.operators_3d import (
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
    laplacian_compact_3d as _laplacian_compact_3d,
)
from legoesm.core.conservation import (
    zero_mean_tendency,
    fix_mass_hydrostatic,
    fix_mass_hydrostatic_target,
)
from legoesm.core.operators import (
    global_integral,
    hyperdiffusion,
    laplacian_compact,
)
from legoesm.parallel.cubesphere_exchange import (
    packed_pad_halo_4d as _packed_pad_halo_4d_spmd,
)
from legoesm.parallel.halo_exchange import (
    packed_pad_halo_mpi_4d as _packed_pad_halo_4d_mpi,
)
from legoesm.core.precision import _resolve_dtype, cast_pytree
from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_sigma_dot_and_total,
    compute_mass_flux_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.core.precision import _resolve_dtype, cast_pytree
from legoesm.core.operators import (
    global_integral,
    hyperdiffusion,
    laplacian_compact,
)
from legoesm.core.operators_cdgrid import _overlapped_arakawa_lamb_gradient
from legoesm.core.conservation import (
    fix_mass_hydrostatic,
    fix_mass_hydrostatic_target,
)
from legoesm.core.state import HydrostaticTendencies
from legoesm.grids.halo import (
    pad_halo_4d as _pad_halo_4d_module,
    pad_halo_vector,
    pad_halo_vector_4d,
)
from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
from legoesm.parallel.halo_exchange import packed_pad_halo_mpi_4d
from legoesm import constants


# ==============================================================================
# Configuration
# ==============================================================================

class CDGridPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-D grid hydrostatic PE model.

    For explicit time integration without semi-implicit gravity wave
    treatment, set ``n_barotropic_substeps > 1`` to subcycle the
    barotropic (external gravity wave) mode.  The barotropic substep
    advances surface pressure and the column-mean divergent flow with
    dt_baro = dt / n_barotropic_substeps, while the baroclinic modes
    (temperature, internal wind structure) evolve on the full dt.

    Alternatively, set ``implicit_grav_wave_damping > 0`` to apply a
    linearized implicit correction to p_s after each step, which damps
    the fastest gravity wave mode without substeps.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    div_damp_coeff: float = 0.0   # Divergence damping coefficient [m^2/s]
    implicit_grav_wave_damping: float = 0.0
        # Implicit damping factor for the external gravity wave mode.
        # Applied as an exponential filter: ps_new *= exp(-alpha * dt * lap(ps))
        # where alpha = implicit_grav_wave_damping.
        # Typical value: 0.5 * c_grav^2 * dt / dx^2 where c_grav ~ 300 m/s.
        # This is a simplified semi-implicit treatment that selectively
        # damps divergent modes without a full barotropic solve.
    T_min: float = 50.0            # Temperature floor [K] (positivity protection)
    p_floor: float = 100.0         # Pressure floor [Pa] for adiabatic heating (limits 1/p)
    sponge_sigma: float = 0.15     # Rayleigh sponge activates above this sigma
    sponge_tau_sec: float = 3600.0 # e-folding time at model top [s]
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    T_diss_coeff: float = 0.0
        # Velocity-dependent Laplacian dissipation for the temperature
        # equation.  Mimics upwind advection's built-in diffusivity:
        #   nu_T = T_diss_coeff * |v| * dx
        # Typically not needed when A_h (Laplacian viscosity) is used,
        # since A_h already damps intermediate-scale T noise.
        # Typical range when used: 0.1-0.5.  0 disables (default).
    zero_mean_ps_tendency: bool = True
        # Apply zero_mean_tendency() to dp_s/dt every RK stage.
        # Ensures exact mass conservation to machine precision but
        # requires a global reduction (MPI allreduce when distributed).
        # Disable for pure performance benchmarks to eliminate sync.
    use_async_halo: bool = False
        # Enable interior/boundary split for compute-communication
        # overlap in MPI mode.  Computes stencils on interior points
        # before halo exchange completes, then recomputes boundary
        # points after.  Overhead: boundary fraction (~8% at C96,
        # ~17% at C48) of redundant compute.  Benefit: hides MPI
        # latency behind interior compute.  Off by default.


# ==============================================================================
# FV3 D-grid tendency function (core implementation)
# ==============================================================================

def fv3_hydrostatic_tendencies(
    state: FV3HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: FV3HydrostaticTendencies | None = None,
    physics_tendency_cc: HydrostaticTendencies | None = None,
) -> FV3HydrostaticTendencies:
    """Compute tendencies for the FV3 hydrostatic PE with D-grid winds.

    The prognostic momentum is stored at D-grid cell corners.  The C-grid
    velocities are diagnosed from the D-grid winds for mass flux and KE
    computation.

    Parameters
    ----------
    state : FV3HydrostaticState
        Prognostic state with D-grid winds.
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid
    config : CDGridPrimitiveEquationConfig
    physics_tendency : FV3HydrostaticTendencies, optional
        Pre-converted D-grid physics tendency.  Existing API; added
        directly to the dycore tendencies (no extra halo).
    physics_tendency_cc : HydrostaticTendencies, optional
        Iter-65: cell-centre physics tendency.  When provided, the
        ``du_dt`` / ``dv_dt`` components ride iter-64's batched
        corner interpolation rather than firing their own halo
        collective in the caller.  The caller passes this instead of
        ``physics_tendency`` to skip the standalone corner-interp
        halo for the physics u/v contribution.  ``dT_dt`` /
        ``dp_s_dt`` from this object are added directly (cell-centre,
        no interp needed).

    Returns
    -------
    FV3HydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u_d = state.u_d.data   # (6, n+1, n+1, nlev) — D-grid
    v_d = state.v_d.data
    T = state.T.data       # (6, n, n, nlev)
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa

    # Positivity protections
    T = jnp.maximum(T, config.T_min)
    p_s = jnp.clip(p_s, config.p_floor, 2.0e6)

    # --- 1. D-grid to C-grid ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    # u_c: (6, n+1, n, nlev),  v_c: (6, n, n+1, nlev)

    # Cell-centre velocities from D-grid (orthogonal basis, for KE)
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)

    # --- 2. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 3. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 4. KE at cell centres from C-grid velocities ---
    KE = 0.5 * (u_cell ** 2 + v_cell ** 2)

    # --- 5. Bernoulli function B = KE + Phi (cell centres) ---
    B = KE + Phi

    # --- 6. D-grid vorticity at cell centres via circulation ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)  # (6, n, n, nlev)

    # === Stage-level packed halo exchange (merged iter-58/60) ===
    # Pack ALL cell-field halos for the stage into ONE collective:
    # always {ζ, B, 1/T, T, ln_ps_3d}; conditionally {u_cell, v_cell}
    # for hyperdiff and {hybrid_factor} for the hybrid PGF correction.
    #
    # Iter-58 merge: previously two separate packed exchanges fired
    # at different points in the function.  Iter-59 added ln_ps_hi
    # reuse for the high-precision PGF.  Iter-60 adds the hybrid
    # factor ``B_full * p_s / p_full`` to the pack so the PGF
    # correction's corner interpolation skips its standalone halo too.
    ln_ps = jnp.log(p_s)
    inv_T = 1.0 / T
    ln_ps_3d = ln_ps[..., jnp.newaxis]  # (6, n, n, 1) — rides the pack
    if _hybrid:
        _hybrid_factor = sigma_coord.B_full * p_s[..., None] / p_full  # (6, n, n, nlev)
    else:
        _hybrid_factor = None
    # Iter-61: when divergence damping is active, compute ``div_v`` via
    # the C-grid divergence operator BEFORE the merged stage halo and
    # pack it so the corresponding ``_arakawa_lamb_gradient(div_v)``
    # below skips its own standalone halo collective.  ``cgrid_divergence``
    # is local (no halo) and depends only on (u_c, v_c) which are
    # already in scope.  Note: ``div_v`` is also used downstream by
    # ``compute_mass_flux_hybrid`` / ``compute_sigma_dot_and_total`` —
    # those consume the unpadded interior values so this hoist is
    # transparent to them.
    _need_div_pad = config.div_damp_coeff > 0
    if _need_div_pad:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)
    else:
        div_v = None  # computed lazily below if not div-damped
    from legoesm.grids.halo import _halo_backend
    _needs_uv_pad = config.A_h > 0 or config.hyperdiff_coeff > 0
    # Packed MPI/SPMD halos now apply the duogrid kinked-to-extended
    # remap when `duogrid=dg` is passed (iter-84).  Fall through to
    # None pre-pads only when neither backend is active.
    _pe_dg = grid.duogrid
    if _halo_backend == "spmd":
        from legoesm.parallel.cubesphere_exchange import _spmd_mesh
        _zeta_pad, _B_pad, _invT_pad = packed_pad_halo_4d(
            zeta, B, inv_T, mesh=_spmd_mesh, duogrid=_pe_dg,
        )
    elif _halo_backend == "mpi":
        from legoesm.grids.halo import _mpi_topology
        _zeta_pad, _B_pad, _invT_pad = packed_pad_halo_mpi_4d(
            zeta, B, inv_T, topology=_mpi_topology, duogrid=_pe_dg,
        )
    else:
        _zeta_pad = _B_pad = _invT_pad = None  # operators do own exchange

    # Pre-pad slots for fields whose merged-stage exchange now happens
    # in section 11 (T / ln_ps_3d / u_cell / v_cell) and for the hybrid-
    # factor / div_v halos that are no longer packed at this point in
    # the function.  Initialising to ``None`` lets each downstream
    # operator fall back to its own standalone halo collective via
    # ``_pad_halo_auto`` — same numerical result, one extra exchange.
    # Without these defaults the references at lines ~343/~366/~403
    # raise ``UnboundLocalError`` (regression introduced when the
    # iter-58/60 merged-stage halo was split apart).
    _lnps_pad = None
    _hf_pad = None
    _div_v_pad = None

    # Vorticity interpolated to D-grid corners, absolute vorticity = ζ_corner + f_corner
    zeta_corner = (_interp_center_to_corner(zeta, cdgrid, padded=_zeta_pad)
                   + cdgrid.f_corner[..., None])

    # --- 7. Bernoulli gradient at D-grid corners (Arakawa-Lamb) ---
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid, padded=_B_pad)

    # --- 8. Pressure gradient correction at D-grid corners ---
    # Promote to higher precision for the PGF computation to avoid
    # catastrophic cancellation (large p terms, small gradient).
    # Use result_type to only upcast (never downcast from current dtype).
    _pg_dt = jnp.result_type(ln_ps.dtype, _resolve_dtype("atm_pressure_gradient", "compute"))
    ln_ps_hi = ln_ps.astype(_pg_dt)
    # Iter-59: when the merged stage halo (iter-58) already exchanged
    # ``ln_ps`` at the PGF compute precision (i.e. ``ln_ps.dtype == _pg_dt``),
    # reuse ``_lnps_pad`` here instead of re-exchanging ln_ps standalone.
    # Saves one halo collective per RK3 stage on the SPMD / MPI cubed-
    # sphere FV3 PE float64 path.  Falls through to the legacy 2D-halo
    # path when dtypes differ (e.g. mixed float32 state with float64 PGF
    # compute precision) so no precision is lost.
    if _lnps_pad is not None and ln_ps.dtype == _pg_dt:
        _lnps_pad_hi = _lnps_pad[..., 0]  # (6, n+2, n+2) at PGF precision
        dln_dx_hi, dln_dy_perp_hi = _arakawa_lamb_gradient(
            ln_ps_hi, cdgrid, padded=_lnps_pad_hi,
        )
    else:
        dln_dx_hi, dln_dy_perp_hi = _arakawa_lamb_gradient(ln_ps_hi, cdgrid)  # 2D, separate exchange
    # Harmonic mean for T at corners suppresses spurious PGF from high-n T.
    T_corner = 1.0 / _interp_center_to_corner(inv_T, cdgrid, padded=_invT_pad)
    T_corner_hi = T_corner.astype(_pg_dt)
    pg_corr_x = (R_d * T_corner_hi * dln_dx_hi[..., None]).astype(u_d.dtype)
    pg_corr_y_perp = (R_d * T_corner_hi * dln_dy_perp_hi[..., None]).astype(v_d.dtype)

    # Hybrid coordinate correction: in sigma coords grad_eta(ln p) = grad(ln p_s),
    # but in hybrid coords grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    # Near the model top B -> 0 (pure pressure levels), so the PGF correction
    # from surface pressure should vanish.  Without this factor the model
    # develops spurious upper-level heating and eventually blows up.
    if _hybrid:
        # ``_hybrid_factor`` and ``_hf_pad`` are produced by the merged
        # halo exchange above (iter-60).  Reuse the pre-padded halo so
        # the corner interpolation skips its own standalone collective.
        _hf_corner = _interp_center_to_corner(
            _hybrid_factor, cdgrid, padded=_hf_pad,
        )
        pg_corr_x = pg_corr_x * _hf_corner
        pg_corr_y_perp = pg_corr_y_perp * _hf_corner

    # --- 9. D-grid momentum tendencies ---
    du_d_dt = zeta_corner * v_d - dB_dx - pg_corr_x
    dv_d_dt = -zeta_corner * u_d - dB_dy_perp - pg_corr_y_perp

    # --- 10a. C-grid divergence for continuity + (optional) damping ---
    # ``cgrid_divergence`` is identical regardless of caller, so compute
    # it once here and feed both the optional divergence-damping block
    # below and the continuity / mass-flux block in section 10b.  The
    # previous code computed ``cgrid_divergence(u_c, v_c, cdgrid)``
    # twice when ``div_damp_coeff > 0`` — one full halo exchange + PPM
    # pass per RHS evaluation.  Drop the duplicate.
    # Iter-61: when div_damp is active, ``div_v`` was already computed
    # before the merged stage halo and packed alongside the cell fields,
    # so reuse it here.  Without div_damp, compute lazily — no need to
    # pad it since downstream operators (``compute_mass_flux_hybrid``,
    # ``compute_sigma_dot_and_total``) consume the unpadded interior
    # values.
    if div_v is None:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)

    # Divergence damping at D-grid
    if config.div_damp_coeff > 0:
        if config.use_async_halo and _halo_backend == "mpi":
            ddiv_dx, ddiv_dy_perp = _overlapped_arakawa_lamb_gradient(
                div_v, cdgrid,
            )
        else:
            # Iter-61: feed the pre-padded ``_div_v_pad`` from the
            # merged stage halo so the A-L gradient skips its own halo
            # exchange.  Falls through to a standalone exchange on the
            # single-device (None) path or the async-overlap path above.
            ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(
                div_v, cdgrid, padded=_div_v_pad,
            )
        du_d_dt = du_d_dt + config.div_damp_coeff * ddiv_dx
        dv_d_dt = dv_d_dt + config.div_damp_coeff * ddiv_dy_perp

    # --- 10b. Surface pressure tendency and vertical motion ---

    if _hybrid:
        # Reuse the column-sum that ``compute_mass_flux_hybrid``
        # already produces internally instead of recomputing
        # ``jnp.sum(div_v * dp, axis=-1)`` ourselves.  Saves one
        # cross-level reduction per RK3 stage on every horizontal-shard
        # configuration that touches the hybrid path.
        mass_flux, _D_total_p_full = compute_mass_flux_hybrid(
            div_v, p_s, sigma_coord,
        )
        dp_s_dt_data = -_D_total_p_full[..., 0] / sigma_coord.B_range
        if config.zero_mean_ps_tendency:
            dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        # Vertical advection of D-grid winds: interpolate (u_d, v_d)
        # together to cell centres (single 4-point average instead of
        # two), compute vertical advection per-component, then batch
        # the back-interpolation to D-grid corners (single halo + 4-pt
        # average instead of two).  Same trailing-axis-as-passive-batch
        # pattern as Loops 113/114.
        n_face_uv, n_i_uv, n_j_uv, nlev_uv = u_d.shape[0], u_d.shape[1] - 1, u_d.shape[2] - 1, u_d.shape[3]
        _uv_d = jnp.stack([u_d, v_d], axis=-1)
        _uv_cc_flat = _interp_corner_to_center(
            _uv_d.reshape(*_uv_d.shape[:-2], nlev_uv * 2),
        )
        _uv_cc = _uv_cc_flat.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2)
        # Batch ``vertical_advection_hybrid`` over (u_cc, v_cc, T) by
        # adding a leading axis: T uses the same ``mass_flux``,
        # ``p_s``, and ``sigma_coord`` as (u, v), so ``F_full`` and
        # ``p_full`` are computed once and broadcast over the new (3,)
        # axis instead of being computed twice (once for the (u, v)
        # batch and once for T standalone).  3 calls → 1.  Loop 168
        # extends Loop 142.
        _uvT_cc_lead = jnp.stack(
            [_uv_cc[..., 0], _uv_cc[..., 1], T], axis=0,
        )  # (3, face, i, j, nlev)
        _vert_adv_uvT_lead = vertical_advection_hybrid(
            _uvT_cc_lead, mass_flux, p_s, sigma_coord,
        )
        _vert_adv_uv_cc = jnp.moveaxis(_vert_adv_uvT_lead[:2], 0, -1)
        vert_adv_T = _vert_adv_uvT_lead[2]
        # Iter-64: defer the cell-center → D-grid corner interpolation
        # of ``_vert_adv_uv_cc`` until the diffusion section so it can
        # be batched with the (lap_uv, hyperdiff_uv) corner interps
        # into a single halo collective.

        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)
    else:
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        # Iter-52: ``compute_sigma_dot_and_total`` runs the cumsum
        # exactly once and returns both σ̇ and ``D_total = Σ div · Δσ``,
        # rather than recomputing ``jnp.sum(div_v * dsigma, ...)``
        # separately and then ``compute_sigma_dot`` doing its own
        # cumsum.  Saves one cross-level collective per RK3 stage on
        # the non-hybrid σ-coordinate path.
        sigma_dot, _D_total_full = compute_sigma_dot_and_total(
            div_v, sigma_coord,
        )
        dp_s_dt_data = -p_s * _D_total_full[..., 0] / sigma_range
        if config.zero_mean_ps_tendency:
            dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        # Vertical advection of D-grid winds via cell-centre interpolation.
        # Batch (u_d, v_d) → (u_cc, v_cc) and (vert_adv_u_cc,
        # vert_adv_v_cc) → (vert_adv_u_d, vert_adv_v_d) — same pattern
        # as the hybrid branch above.
        n_face_uv, n_i_uv, n_j_uv, nlev_uv = u_d.shape[0], u_d.shape[1] - 1, u_d.shape[2] - 1, u_d.shape[3]
        _uv_d = jnp.stack([u_d, v_d], axis=-1)
        _uv_cc_flat = _interp_corner_to_center(
            _uv_d.reshape(*_uv_d.shape[:-2], nlev_uv * 2),
        )
        _uv_cc = _uv_cc_flat.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2)
        # Batch ``vertical_advection`` over (u_cc, v_cc, T) by adding a
        # leading axis.  T shares ``sigma_dot`` / ``sigma_coord`` with
        # (u, v), so the velocity-independent shared work is computed
        # once and broadcast over the (3,) leading axis.  3 calls → 1.
        # Loop 168 extends Loop 142.
        _uvT_cc_lead = jnp.stack(
            [_uv_cc[..., 0], _uv_cc[..., 1], T], axis=0,
        )  # (3, face, i, j, nlev)
        _vert_adv_uvT_lead = vertical_advection(
            _uvT_cc_lead, sigma_dot, sigma_coord,
        )
        _vert_adv_uv_cc = jnp.moveaxis(_vert_adv_uvT_lead[:2], 0, -1)
        vert_adv_T = _vert_adv_uvT_lead[2]
        # Iter-64: defer the cell-center → D-grid corner interpolation
        # of ``_vert_adv_uv_cc`` (mirrors the hybrid branch above).

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)

    # Iter-64: ``vert_adv_uv_d`` will be added to ``du_d_dt``/``dv_d_dt``
    # in section 12c below alongside the lap_uv and hyperdiff_uv
    # contributions, so the three corner interpolations share a single
    # halo collective.

    # --- 11. Thermodynamic equation ---
    # Horizontal advection: centred advection using cell-centre velocities.
    #
    # Loop 189 — promote ``ln_ps`` to ``(6, n, n, 1)`` and pack it
    # into the same packed halo exchange as ``T`` (and, when
    # ``_needs_uv_pad``, ``u_cell`` / ``v_cell``).  This eliminates
    # the two standalone halo exchanges that previously fired inside
    # ``gradient_x(ln_ps)`` and ``gradient_y(ln_ps)`` (each issued
    # its own ``pad_halo``) — they are now replaced by sliced gradients
    # of the shared 4D padded field.  Saves 2 halo exchanges (or, on
    # SPMD/MPI, packs ln_ps into an existing collective at the cost
    # of ``9*4`` extra bytes per face boundary).
    _needs_uv_pad = config.A_h > 0 or config.hyperdiff_coeff > 0
    _pad_halo_4d = _pad_halo_4d_module
    # Packed MPI halo now applies duogrid remap (iter-84) so we use it
    # even when duogrid is active.  Non-MPI fallback does per-field pads
    # with duogrid routing preserved.
    _pe_dg = grid.duogrid
    # ``ln_ps_3d`` carries a singleton trailing axis on purpose so it
    # can ride the same packed exchange as the full 3D fields.  This is
    # explicitly supported by ``packed_pad_halo_mpi_4d`` (and its SPMD
    # twin), whose docstring promises that "all fields must share the
    # same (6, n, n) spatial prefix; the trailing axis (levels/channels)
    # can differ" — the helper concatenates along the trailing axis and
    # splits per-field on return (halo_exchange.py:1128-1133).  Loop 189
    # added the optimisation; restoring it here preserves the
    # ``∇(ln p_s)`` halo without paying for two extra exchanges.  The
    # MPI branch is taken whenever the backend is MPI (regardless of
    # ``_needs_uv_pad``) so the lnps halo is never silently routed
    # through a non-packed code path under MPI.
    ln_ps_3d = ln_ps[..., jnp.newaxis]  # (6, n, n, 1)
    if _halo_backend == "mpi":
        from legoesm.grids.halo import _mpi_topology
        if _needs_uv_pad:
            _T_pad, _u_cc_pad, _v_cc_pad, _lnps_pad = packed_pad_halo_mpi_4d(
                T, u_cell, v_cell, ln_ps_3d,
                topology=_mpi_topology, duogrid=_pe_dg,
            )
        else:
            _T_pad, _lnps_pad = packed_pad_halo_mpi_4d(
                T, ln_ps_3d, topology=_mpi_topology, duogrid=_pe_dg,
            )
            _u_cc_pad = _v_cc_pad = None
    else:
        # Route through duogrid remap when duogrid is active on the grid,
        # matching the pattern used by _arakawa_lamb_gradient via
        # `_pad_halo_auto`.
        _pe_offs = None if _pe_dg is not None else grid.halo_interp_offsets
        _T_pad = _pad_halo_4d(T, interp_offsets=_pe_offs, duogrid=_pe_dg)
        _lnps_pad = _pad_halo_4d(ln_ps_3d, interp_offsets=_pe_offs, duogrid=_pe_dg)
        if _needs_uv_pad:
            _u_cc_pad = _pad_halo_4d(u_cell, interp_offsets=_pe_offs, duogrid=_pe_dg)
            _v_cc_pad = _pad_halo_4d(v_cell, interp_offsets=_pe_offs, duogrid=_pe_dg)

    dT_dx = _gradient_x_3d(T, grid, padded=_T_pad)
    dT_dy = _gradient_y_3d(T, grid, padded=_T_pad)
    horiz_adv_T = -(u_cell * dT_dx + v_cell * dT_dy)

    # Adiabatic heating: kappa * T * omega / p
    # ln_ps gradient at cell centres for the v.grad(ln ps) correction.
    # Use the 3D variants with the shared ``_lnps_pad`` from exchange
    # #2 — saves the two halo exchanges that the 2D ``gradient_x`` /
    # ``gradient_y`` would otherwise fire (Loop 189).
    dln_ps_dx = _gradient_x_3d(ln_ps_3d, grid, padded=_lnps_pad)[..., 0]  # (6, n, n)
    dln_ps_dy = _gradient_y_3d(ln_ps_3d, grid, padded=_lnps_pad)[..., 0]
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = u_cell * dln_ps_dx[..., None] + v_cell * dln_ps_dy[..., None]
    # In sigma coords: grad_eta(ln p) = grad(ln p_s).
    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    # Apply the same factor to the adiabatic v.grad(ln p_s) correction.
    if _hybrid:
        v_dot_grad_lnps = v_dot_grad_lnps * (sigma_coord.B_full * p_s[..., None] / p_adiab)
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 12. Diffusion ---
    # 12a. Laplacian viscosity on D-grid winds
    #
    # Apply the compact Laplacian at cell centres (full-strength damping
    # on all modes) and interpolate the tendency back to D-grid corners.
    # This avoids the corner-centre-corner round-trip of _laplacian_dgrid
    # which attenuates the grid-scale mode to near zero.
    # Pre-padded {T, u_cell, v_cell} from exchange #2 eliminate
    # redundant halo exchanges in the Laplacian calls.
    # Stack {u_cell, v_cell, T} along a trailing axis and fold into the
    # level dim so a single Laplacian (12a) and a single hyperdiffusion
    # (12b) run on the thicker (6, n, n, nlev*3) field.  The existing
    # pre-padded arrays (from packed_pad_halo_4d / packed_pad_halo_mpi_4d
    # at exchange #2) are stacked the same way so the inner laplacians
    # still skip their halo (no extra MPI cost), while the outer ∇² of
    # hyperdiffusion shares its outer-halo sweep across all 3 fields
    # instead of paying it 3 times.
    _need_uvT_stack = config.A_h > 0 or config.hyperdiff_coeff > 0
    if _need_uvT_stack:
        n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT = u_cell.shape
        _uvT_stack = jnp.stack([u_cell, v_cell, T], axis=-1)
        _uvT_flat = _uvT_stack.reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT * 3)
        _uvT_pad_stack = jnp.stack([_u_cc_pad, _v_cc_pad, _T_pad], axis=-1)
        _pad_pre = _uvT_pad_stack.shape[:3]
        _uvT_pad_flat = _uvT_pad_stack.reshape(*_pad_pre, nlev_uvT * 3)

    # When BOTH A_h and hyperdiff_coeff are active, compute the inner
    # ∇²(_uvT_flat) (compact stencil) ONCE and reuse it for both the
    # explicit Laplacian (Section 12a) and the inner stage of the
    # biharmonic hyperdiffusion (Section 12b).  Saves one full
    # ``_laplacian_compact_3d`` call (i.e. one extra arithmetic ∇² pass
    # over the (u, v, T) trio) per RHS evaluation — same idea as the
    # ocean Loop 135 K_h+K_bih sharing.
    _lap_flat: jax.Array | None = None
    if config.A_h > 0 or config.hyperdiff_coeff > 0:
        _lap_flat = _laplacian_compact_3d(
            _uvT_flat, grid, padded=_uvT_pad_flat,
        )

    # --- 11b. Velocity-dependent temperature dissipation ---
    # Lifted below the (u, v, T) Laplacian above so that, when ``A_h``
    # or ``hyperdiff_coeff`` is also active, the T component of the
    # batched ∇²(uvT) is reused — one fewer ``_laplacian_compact_3d``
    # call (i.e. one extra ∇² over T standalone).  When the (u, v, T)
    # batch did not run, fall back to a standalone ``∇²(T)``.
    if config.T_diss_coeff > 0:
        wind_speed = jnp.sqrt(u_cell**2 + v_cell**2)
        dx_local = grid.dx[..., None]
        nu_T = config.T_diss_coeff * wind_speed * dx_local
        if _lap_flat is not None:
            # Reuse the T slice of the batched compact Laplacian.
            lap_T = _lap_flat.reshape(
                n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3,
            )[..., 2]
        else:
            lap_T = _laplacian_compact_3d(T, grid, padded=_T_pad)
        dT_dt_data = dT_dt_data + nu_T * lap_T

    # Iter-63/64: collect ALL (u, v) cell-center → D-grid corner
    # interpolations of the stage and batch them into ONE halo +
    # 4-pt corner average.  ``_vert_adv_uv_cc`` (always present) plus
    # optionally ``lap_uvT[..., :2]`` (when A_h > 0) and
    # ``hyperdiff_uvT[..., :2]`` (when hyperdiff_coeff > 0).  Each
    # ``_interp_center_to_corner`` would otherwise do its own halo
    # exchange; the batched version stacks the pairs along the trailing
    # axis and exchanges in one halo.  Saves 1 collective per stage
    # in the typical hyperdiff-only config; 2 per stage when both
    # A_h and hyperdiff are active.
    lap_uvT = None
    hyperdiff_uvT = None
    if config.A_h > 0:
        lap_uvT = _lap_flat.reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3)
    if config.hyperdiff_coeff > 0:
        hyperdiff_flat = _hyperdiffusion_3d(
            _uvT_flat, grid, config.hyperdiff_coeff,
            padded=_uvT_pad_flat,
            # When ``_lap_flat`` was already computed above (A_h > 0),
            # feed it in as the inner ∇² so the biharmonic skips a
            # redundant compact-stencil pass.
            inner_lap=_lap_flat,
        )
        hyperdiff_uvT = hyperdiff_flat.reshape(
            n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3,
        )

    # Build the cell-center batch: vert_adv (always) + lap (if A_h>0)
    # + hyperdiff (if hyperdiff_coeff>0) + phys_cc.du/dv (iter-65,
    # when ``physics_tendency_cc`` is provided), each shape
    # (n_face, n, n, nlev*2) along the trailing axis.
    _uv_corner_blocks = [
        _vert_adv_uv_cc.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2),
    ]
    if lap_uvT is not None:
        _uv_corner_blocks.append(
            lap_uvT[..., :2].reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT * 2)
        )
    if hyperdiff_uvT is not None:
        _uv_corner_blocks.append(
            hyperdiff_uvT[..., :2].reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT * 2)
        )
    if physics_tendency_cc is not None and physics_tendency_cc.du_dt is not None:
        _phys_uv_cc = jnp.stack(
            [physics_tendency_cc.du_dt.data, physics_tendency_cc.dv_dt.data],
            axis=-1,
        ).reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
        _uv_corner_blocks.append(_phys_uv_cc)
    _n_blocks = len(_uv_corner_blocks)
    if _n_blocks == 1:
        _combined_uv_cc = _uv_corner_blocks[0]
    else:
        _combined_uv_cc = jnp.concatenate(_uv_corner_blocks, axis=-1)
    _combined_uv_d_flat = _interp_center_to_corner(_combined_uv_cc, cdgrid)
    _combined_uv_d = _combined_uv_d_flat.reshape(
        _combined_uv_d_flat.shape[0], _combined_uv_d_flat.shape[1],
        _combined_uv_d_flat.shape[2], _n_blocks, nlev_uv, 2,
    )  # (n_face_d, n_i_d, n_j_d, n_blocks, nlev, 2-uv)
    _block_idx = 0
    _vert_adv_uv_d = _combined_uv_d[:, :, :, _block_idx]
    _block_idx += 1
    _lap_uv_d = None
    if lap_uvT is not None:
        _lap_uv_d = _combined_uv_d[:, :, :, _block_idx]
        _block_idx += 1
    _hd_uv_d = None
    if hyperdiff_uvT is not None:
        _hd_uv_d = _combined_uv_d[:, :, :, _block_idx]
        _block_idx += 1
    _phys_uv_d = None
    if (physics_tendency_cc is not None
            and physics_tendency_cc.du_dt is not None):
        _phys_uv_d = _combined_uv_d[:, :, :, _block_idx]
        _block_idx += 1

    # Add the vert_adv contribution to du_d_dt/dv_d_dt now that the
    # corner interpolation is done.
    du_d_dt = du_d_dt + _vert_adv_uv_d[..., 0]
    dv_d_dt = dv_d_dt + _vert_adv_uv_d[..., 1]

    if config.A_h > 0:
        du_d_dt = du_d_dt + config.A_h * _lap_uv_d[..., 0]
        dv_d_dt = dv_d_dt + config.A_h * _lap_uv_d[..., 1]
        dT_dt_data = dT_dt_data + config.A_h * lap_uvT[..., 2]

    if config.hyperdiff_coeff > 0:
        du_d_dt = du_d_dt + _hd_uv_d[..., 0]
        dv_d_dt = dv_d_dt + _hd_uv_d[..., 1]
        dT_dt_data = dT_dt_data + hyperdiff_uvT[..., 2]

    # Surface pressure hyperdiffusion (cell-centre)
    if config.hyperdiff_ps_coeff > 0:
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 13. Upper-atmosphere Rayleigh sponge (D-grid) ---
    if config.sponge_tau_sec > 0 and config.sponge_sigma > 0:
        sigma_full = sigma_coord.sigma_full
        sponge_frac = jnp.clip(
            (config.sponge_sigma - sigma_full) / config.sponge_sigma, 0.0, 1.0
        )
        sponge_rate = sponge_frac**2 / config.sponge_tau_sec
        du_d_dt = du_d_dt - sponge_rate * u_d
        dv_d_dt = dv_d_dt - sponge_rate * v_d

    # --- 14. Physics ---
    if physics_tendency is not None:
        du_d_dt = du_d_dt + physics_tendency.du_d_dt.data
        dv_d_dt = dv_d_dt + physics_tendency.dv_d_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data
    # Iter-65: cell-centre physics ``du_dt`` / ``dv_dt`` rode the
    # iter-64 batch above; ``dT_dt`` / ``dp_s_dt`` are added directly
    # (cell-centre, no corner interp needed).
    if physics_tendency_cc is not None:
        if physics_tendency_cc.du_dt is not None and _phys_uv_d is not None:
            du_d_dt = du_d_dt + _phys_uv_d[..., 0]
            dv_d_dt = dv_d_dt + _phys_uv_d[..., 1]
        dT_dt_data = dT_dt_data + physics_tendency_cc.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency_cc.dp_s_dt.data

    dims_3d_corner = ("face", "x", "y", "level")
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return FV3HydrostaticTendencies(
        du_d_dt=Field(data=du_d_dt, name="du_d_dt", dims=dims_3d_corner, units="m/s^2"),
        dv_d_dt=Field(data=dv_d_dt, name="dv_d_dt", dims=dims_3d_corner, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


# ==============================================================================
# Adapter: D-grid to cell-centre conversion
# ==============================================================================

def fv3_to_hydrostatic(
    state: FV3HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> HydrostaticState:
    """Convert FV3 D-grid state to cell-centre HydrostaticState.

    Uses corner-to-centre interpolation for the wind components.
    Batches (u_d, v_d) into a single ``_interp_corner_to_center`` call
    so the 4-point average kernel runs once on the thicker stacked
    tensor instead of twice — same passive-trailing-axis pattern used
    in the tendency function.
    """
    u_d = state.u_d.data
    v_d = state.v_d.data
    n_face_a, n_corner_i, n_corner_j, nlev_a = u_d.shape
    _uv_d = jnp.stack([u_d, v_d], axis=-1)  # (face, n+1, n+1, nlev, 2)
    _uv_cc_flat = _interp_corner_to_center(
        _uv_d.reshape(n_face_a, n_corner_i, n_corner_j, nlev_a * 2),
    )
    _uv_cc = _uv_cc_flat.reshape(
        _uv_cc_flat.shape[0], _uv_cc_flat.shape[1], _uv_cc_flat.shape[2],
        nlev_a, 2,
    )
    u_cc = _uv_cc[..., 0]
    v_cc = _uv_cc[..., 1]
    return HydrostaticState(
        u=state.u_d.replace(data=u_cc, name="u"),
        v=state.v_d.replace(data=v_cc, name="v"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=state.tracers,
    )


# ==============================================================================
# Model class
# ==============================================================================

class CDGridPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic PE model on the cubed-sphere with FV3 C-D grid dynamics.

    Prognostic winds live on the D-grid (cell corners).  The model
    accepts and returns ``FV3HydrostaticState`` from ``step()``.

    For backward compatibility with code that passes ``HydrostaticState``
    (cell-centre winds), use ``step_cell_centre()`` or ``fv3_to_hydrostatic``.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : CDGridPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: CDGridPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CDGridPrimitiveEquationConfig()
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self._target_mass = None

    def _sync_dgrid_boundary(self, state: FV3HydrostaticState):
        """No-op: cross-face continuity is handled by halo exchange.

        The FV3 approach relies on halo exchange in the d2a2c operators
        (dgrid_to_cgrid, cdgrid_momentum_tendencies) to handle cross-
        face data, not explicit boundary syncing.  Any explicit sync
        (whether averaging or owner-copy) acts as edge-selective
        dissipation that seeds spurious v-wind in steady-state flows.
        """
        return state

    def tendencies(
        self,
        state,
        physics_tendency=None,
    ):
        """Compute tendencies, accepting FV3HydrostaticState or HydrostaticState."""
        return cdgrid_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.cdgrid,
            self.config, physics_tendency,
        )

    def step(self, state, dt, physics_fn=None):
        """Advance one time step.

        Accepts ``FV3HydrostaticState`` (D-grid prognostic winds).
        For legacy ``HydrostaticState`` input, uses the cell-centre
        adapter path.

        Parameters
        ----------
        state : FV3HydrostaticState or HydrostaticState
        dt : float
        physics_fn : callable, optional

        Returns
        -------
        Same type as input state.
        """
        # Precompute target mass outside JIT boundary (host-side only).
        # This avoids writing traced values into persistent object attributes
        # inside a jit-compiled method.
        if (self.config.use_conservation_fixer and self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = global_integral(state.p_s, self.grid)

        if isinstance(state, FV3HydrostaticState):
            return self._step_fv3(state, dt, physics_fn=physics_fn)
        # Legacy cell-centre state path
        return self._step_cell_centre(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_fv3(
        self,
        state: FV3HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> FV3HydrostaticState:
        """Advance one time step with D-grid prognostic winds."""
        state = cast_pytree(state, None, "compute")
        cdgrid = self.cdgrid

        def tendency_fn(s):
            phys_cc = None
            if physics_fn is not None:
                # Convert D-grid state to cell-centre for physics.
                # Iter-65: pass the cell-centre physics tendency
                # directly to fv3_hydrostatic_tendencies via
                # ``physics_tendency_cc`` so its (du_dt, dv_dt)
                # corner interpolation is batched with the iter-64
                # vert_adv/lap/hyperdiff corner interp — saves the
                # standalone halo collective that the old conversion
                # path emitted here.
                s_cc = fv3_to_hydrostatic(s, cdgrid)
                _phys_result = physics_fn(s_cc, self.grid, self.sigma_coord)
                phys_cc = _phys_result[0] if type(_phys_result) is tuple else _phys_result

            tend = fv3_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, cdgrid,
                self.config,
                physics_tendency=None,
                physics_tendency_cc=phys_cc,
            )
            # Return an FV3HydrostaticState-shaped pytree with tendency data
            # so that the time integrator's tree_map works correctly.
            return FV3HydrostaticState(
                u_d=s.u_d.replace(data=tend.du_d_dt.data),
                v_d=s.v_d.replace(data=tend.dv_d_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Synchronize D-grid boundary corners across cubed-sphere faces
        state_new = self._sync_dgrid_boundary(state_new)

        # Implicit gravity wave damping — post-step Laplacian diffusion on p_s.
        if self.config.implicit_grav_wave_damping > 0:
            alpha = self.config.implicit_grav_wave_damping
            lap_ps = laplacian_compact(state_new.p_s.data, self.grid)
            p_s_damped = state_new.p_s.data + alpha * dt * lap_ps
            p_s_damped = jnp.maximum(p_s_damped, self.config.p_floor)
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_damped),
            )

        # Conservation fixer (operates on p_s which is at cell centres).
        # ``fix_mass_hydrostatic_target`` / ``fix_mass_hydrostatic`` only
        # touch ``state.p_s`` and call ``state._replace(p_s=...)``, so
        # we can pass the FV3 state directly.  Going through
        # ``fv3_to_hydrostatic`` ran a corner-to-centre halo exchange +
        # 4-point average on (u_d, v_d) just to throw the wind result
        # away — wasted compute and an extra collective on the SPMD
        # cubed-sphere path.
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                # _target_mass is precomputed in step() outside the JIT boundary.
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
                state_h_new = fv3_to_hydrostatic(state_new, cdgrid)
                state_h_old = fv3_to_hydrostatic(state, cdgrid)
                state_h_fixed = fix_mass_hydrostatic(
                    state_h_new, state_h_old, self.grid,
                )

        return cast_pytree(state_new, None, "storage")

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_cell_centre(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        """Advance one step, accepting and returning cell-centre state.

        This is a convenience wrapper for code that still works with
        ``HydrostaticState``.  Cell-centre winds are interpolated to
        D-grid corners at entry and back to cell centres at exit;
        internally the dycore operates entirely on D-grid winds.
        """
        # Batch the cell-centre → D-grid interpolation for u, v: same
        # passive-trailing-axis pattern as the tendency function and
        # ``fv3_to_hydrostatic`` adapter.  One halo + one 4-point
        # average shared between u and v.
        _u_in = state.u.data
        _v_in = state.v.data
        _ni_face, _ni_i, _ni_j, _ni_lev = _u_in.shape
        _uv_in = jnp.stack([_u_in, _v_in], axis=-1)
        _uv_d_flat = _interp_center_to_corner(
            _uv_in.reshape(_ni_face, _ni_i, _ni_j, _ni_lev * 2),
            self.cdgrid,
        )
        _uv_d = _uv_d_flat.reshape(
            _uv_d_flat.shape[0], _uv_d_flat.shape[1], _uv_d_flat.shape[2],
            _ni_lev, 2,
        )
        u_d = _uv_d[..., 0]
        v_d = _uv_d[..., 1]
        fv3_state = FV3HydrostaticState(
            u_d=state.u.replace(data=u_d, name="u_d"),
            v_d=state.v.replace(data=v_d, name="v_d"),
            T=state.T,
            p_s=state.p_s,
            phis=state.phis,
            tracers=state.tracers,
        )
        fv3_new = self._step_fv3(fv3_state, dt, physics_fn=physics_fn)
        return fv3_to_hydrostatic(fv3_new, self.cdgrid)

    # Backward-compatible aliases
    step_cell_centre = _step_cell_centre

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)


# ==============================================================================
# Backward-compatible aliases referenced by __init__.py
# ==============================================================================

def cdgrid_hydrostatic_tendencies(
    state,
    grid: CubedSphereGrid,
    sigma_coord,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency=None,
):
    """Compute hydrostatic tendencies, accepting either HydrostaticState or FV3HydrostaticState.

    If given a HydrostaticState (cell-centre winds), converts to D-grid internally,
    calls fv3_hydrostatic_tendencies, and returns HydrostaticTendencies (cell-centre).

    If given a FV3HydrostaticState, delegates directly to fv3_hydrostatic_tendencies.
    """
    if isinstance(state, FV3HydrostaticState):
        return fv3_hydrostatic_tendencies(state, grid, sigma_coord, cdgrid, config, physics_tendency)

    # HydrostaticState path: convert cell-centre -> D-grid (batched).
    _u_in = state.u.data
    _v_in = state.v.data
    _ni_face, _ni_i, _ni_j, _ni_lev = _u_in.shape
    _uv_in = jnp.stack([_u_in, _v_in], axis=-1)
    _uv_d_flat = _interp_center_to_corner(
        _uv_in.reshape(_ni_face, _ni_i, _ni_j, _ni_lev * 2), cdgrid,
    )
    _uv_d = _uv_d_flat.reshape(
        _uv_d_flat.shape[0], _uv_d_flat.shape[1], _uv_d_flat.shape[2],
        _ni_lev, 2,
    )
    u_d = _uv_d[..., 0]
    v_d = _uv_d[..., 1]
    fv3_state = FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=getattr(state, 'tracers', None),
    )
    fv3_tend = fv3_hydrostatic_tendencies(fv3_state, grid, sigma_coord, cdgrid, config, physics_tendency)

    # Convert D-grid tendencies back to cell-centre (batched).
    _du_d = fv3_tend.du_d_dt.data
    _dv_d = fv3_tend.dv_d_dt.data
    _nd_face, _nd_i, _nd_j, _nd_lev = _du_d.shape
    _duv_d = jnp.stack([_du_d, _dv_d], axis=-1)
    _duv_cc_flat = _interp_corner_to_center(
        _duv_d.reshape(_nd_face, _nd_i, _nd_j, _nd_lev * 2),
    )
    _duv_cc = _duv_cc_flat.reshape(
        _duv_cc_flat.shape[0], _duv_cc_flat.shape[1], _duv_cc_flat.shape[2],
        _nd_lev, 2,
    )
    du_cc = _duv_cc[..., 0]
    dv_cc = _duv_cc[..., 1]

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticTendencies(
        du_dt=Field(data=du_cc, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_cc, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=fv3_tend.dT_dt,
        dp_s_dt=fv3_tend.dp_s_dt,
        dphis_dt=fv3_tend.dphis_dt,
    )


def hydrostatic_to_fv3(
    state: HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> FV3HydrostaticState:
    """Convert cell-centre HydrostaticState to FV3 D-grid state.

    Uses **vector**-aware halo exchange + 4-point averaging — the
    cell-centre ``(u, v)`` are local-east/local-north components, and
    the local basis rotates across face boundaries.  Scalar
    ``_interp_center_to_corner`` (which uses scalar ``pad_halo``)
    averages ``u`` from a face into a neighbouring face's edge
    without rotating into that face's basis, which leaves a non-zero
    spurious divergence at every cube edge — initialising the JW
    baroclinic-wave IC through that path injected ``|div_v| ~ 5e-5
    s^-1`` at t=0 and produced a 970 Pa surface-pressure shift in
    one 300 s step.  This was the structural cause of the C24 BCW
    blow-up at day 0.35 (issue tracked in scaling.md §1.b).

    The shallow-water ``_sw_to_cdgrid`` helper used the correct
    vector-aware path; this routine now mirrors it.
    """

    _u_in = state.u.data
    _v_in = state.v.data
    base = cdgrid.base
    if _u_in.ndim == 4:
        u_pad, v_pad = pad_halo_vector_4d(
            _u_in, _v_in,
            base.cos_angle, base.sin_angle,
            base.cos_angle_padded, base.sin_angle_padded,
            interp_offsets=base.halo_interp_offsets,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                      + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                      + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    else:
        u_pad, v_pad = pad_halo_vector(
            _u_in, _v_in,
            base.cos_angle, base.sin_angle,
            base.cos_angle_padded, base.sin_angle_padded,
            interp_offsets=base.halo_interp_offsets,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                      + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                      + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])

    return FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=getattr(state, 'tracers', None),
    )
