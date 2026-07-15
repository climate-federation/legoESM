"""FV3-inspired Hydrostatic PE on cubed-sphere (D-grid). Research path, not a faithful FV3 port.

Lin (2004), Putman & Lin (2007), Simmons & Burridge (1981).
Differs from faithful FV3: RK3 (not forward-backward), interpolated halo,
no Lagrangian vertical coord.
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
    center_to_dgrid_vector,
    cgrid_divergence,
    cgrid_flux_divergence_sync,
    cgrid_interp_cc_to_faces_local,
    pad_halo_auto,
    dgrid_vorticity,
    arakawa_lamb_gradient,
    interp_center_to_corner,
    interp_corner_to_center,
)
from legoesm.core.operators_3d import (
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    divergence_3d as _divergence_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
    laplacian_compact_3d as _laplacian_compact_3d,
)
from legoesm.core.conservation import (
    zero_mean_tendency,
    fix_ps_mass,
    fix_ps_mass_target,
)
from legoesm.core.operators import (
    global_integral,
    hyperdiffusion,
    laplacian_compact,
)
from legoesm.core.precision import resolve_dtype, cast_pytree
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
    compute_sigma_dot_from_cumsum,
    compute_mass_flux_from_cumsum,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    IntegrationMixin,
    refuse_unthreaded_stateful_physics,
)
from legoesm.core.operators_cdgrid import overlapped_arakawa_lamb_gradient
from legoesm.grids.halo import (
    pad_halo_4d as _pad_halo_4d_module,
    pad_halo_vector,
    pad_halo_vector_4d,
)
from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
from legoesm.parallel.halo_exchange import packed_pad_halo_mpi_4d
from legoesm.atmosphere.dynamics.shared.tracer_transport import (
    advective_tracer_tendency,
)
from legoesm import constants


# ==============================================================================
# Configuration
# ==============================================================================

class CDGridPrimitiveEquationConfig(NamedTuple):
    """Config for C-D grid hydrostatic PE.

    Set ``n_barotropic_substeps > 1`` to subcycle external gravity waves,
    or ``implicit_grav_wave_damping > 0`` for a simplified semi-implicit p_s damping.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]; FV3_3D iter 33-35: needs ~2e7 at C72
    smagorinsky_cs: float = 0.0
        # FV3_3D iter 57/58: opt-in Smagorinsky A_h. A_h_total = A_h + cs*dx²*|D|. Typical 0.1-0.4.
    ah_d_con: float = 0.0
        # FV3_3D iter 225: KE→heat d_con for Smagorinsky A_h. dT = -ah_d_con*(u·du+v·dv)/c_pd at corners.
        # Default 0.0 = baseline; gated by A_h>0. FV3 default 1.0.
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    div_damp_coeff: float = 0.0   # Divergence damping coefficient [m^2/s]
    implicit_grav_wave_damping: float = 0.0
        # Simplified semi-implicit damping. ps *= exp(-alpha*dt*lap(ps)). Typical 0.5*c_grav²*dt/dx².
    T_min: float = 50.0            # Temperature floor [K]
    p_floor: float = 100.0         # Pressure floor [Pa] for adiabatic 1/p
    sponge_sigma: float = 0.15     # Rayleigh sponge above this sigma
    sponge_tau_sec: float = 3600.0 # e-folding time at model top [s]
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    T_diss_coeff: float = 0.0
        # Velocity-dependent T diffusion: nu_T = coeff*|v|*dx. Typical 0.1-0.5 when used.
    zero_mean_ps_tendency: bool = True
        # Apply zero_mean_tendency() to dp_s/dt every RK stage. Requires MPI allreduce.
    use_async_halo: bool = False
        # MPI interior/boundary split for compute-comm overlap.
    use_fv3_lin_pgf: bool = False
        # FV3 Lin (1997) cross-product PGF. FV3_3D iter 4: INERT — needs forward-backward stepping
        # for stability with RK3 (CFL-incompatible). Retained for future iter.
    div_damp_dddmp: float = 0.0
        # FV3_3D iter 5: adaptive Smag div damping. FV3 sw_core.F90:1720
        # damp = da_min_c * max(d2_bg, min(0.20, dddmp*|div|)). FV3 default 0.2.
    div_damp_d_con: float = 0.0
        # FV3_3D iter 223: KE→heat d_con for cell-centre div_damp.
        # dT = -coeff*(u·du+v·dv)/c_pd. Default 0.0; gated by div_damp_coeff>0. FV3 default 1.0.
    damp_v: float = 0.0
        # FV3_3D iter 12: post-step del-n vorticity damping via SW backbone
        # fv3_del6_vorticity_damping (sw_core.F90:1948-1999). Applied once per step AFTER RK3.
        # Iter1009 SW uses 0.030.
    nord_v: int = 2
        # Order of post-step vorticity damping (0=del-2, 1=del-4, 2=del-6). FV3 default 2.
    damp_v_d_con: float = 0.0
        # FV3_3D iter 208: KE→heat d_con for damp_v (FV3 sw_core.F90:1953-1990).
        # ΔKE = u·du + 0.5du² + v·dv + 0.5dv²; ΔT = -coeff*ΔKE/c_pd at corners → centres.
        # iter 338: opt-in metric-aware form via use_fv3_metric_aware_d_con.
        # PE T is true temperature (no FV3 pkz factor needed; iter-246).
        # FV3_3D iter 338/344/347-352: metric-aware port at all 8 PE+NH d_con sites.
    delt_max: float = 0.0
        # FV3_3D iter 218/219: per-step heating cap (FV3 dyn_core.F90:1774). Clips |dT| to bdt*delt_max
        # (K). PE: skip k=0,1; cap k>=2. NH: 0.1x at k=0, 0.5x at k=1, 1x k>=2. FV3 default 1.0.
    corner_div_damp_d_con: float = 0.0
        # FV3_3D iter 221: KE→heat d_con for corner-div damp. dT/dt = -coeff*(u·du+v·dv)/c_pd.
        # Gated by corner_div_damp_d2_bg>0. FV3 default 1.0. Capped by iter-218/219 delt_max.
    use_fv3_cross_face_du_proj: bool = False
        # FV3_3D iter 370/384: cross-face halo for damp_v wind-increment projection. Requires
        # duogrid=True to actually transfer cross-face values (NO-OP without duogrid).
    use_fv3_metric_aware_d_con: bool = False
        # FV3_3D iter 338: metric-aware d_con at damp_v_d_con site (FV3 sw_core.F90:1956-1985 with
        # rsin2/cosa_s). Other d_con sites stay with simpler form. Equivalent in orthogonal limit.
    d_con_top_zero_levels: int = 0
        # FV3_3D iter 433 (mirror of NH 431/432): zero d_con heating in top N levels.
        # Port of FV3 dyn_core.F90:790/800/804 d_con_k=0. Wired at 4 PE d_con sites.
    use_fv3_a2b_zeta_corner: bool = False
        # FV3_3D iter 14: a2b_ord4 4th-order A→B interp for ζ_corner only (other corner interps
        # stay 2nd-order; iter-9: swapping all breaks operator balance).
    corner_div_damp_d2_bg: float = 0.0
        # FV3_3D iter 16: B-grid corner-div adaptive damping (FV3 sw_core.F90:1720).
        # damp = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt)). Stable range 0 to ~0.005.
        # iter-17 optimum HS C36: 0.0005.
    corner_div_damp_dddmp: float = 0.20
        # FV3 sw_core.F90 default 0.20. Active when corner_div_damp_d2_bg>0.
    corner_div_damp_d4_bg: float = 0.0
        # FV3_3D iter 18: del-(2*(nord+1)) corner-div damp (FV3 sw_core.F90:1809-1817).
        # dd8 = (da_min_c*d4_bg)^(nord+1); vort = damp2*delpc + dd8*divg_d_iter. FV3 typical d4_bg=0.16, nord=2.
    corner_div_damp_nord: int = 0
        # 0=del-2, 1=del-4, 2=del-6. Active when corner_div_damp_d4_bg>0.
    rf_tau_days: float = 0.0
        # FV3_3D iter 449 (PE mirror of NH 448): Ray_fast (FV3 dyn_core.F90:2922-3020).
        # rff(k) = 1/(1 + dt/(tau*86400)*sin²(...)²); u_d,v_d *= rff for pfull<rf_cutoff_pa.
        # tau in DAYS. Typical 5-15 days. PE: no w.
    rf_cutoff_pa: float = 3000.0
        # FV3 default rf_cutoff=3.0e2 Pa = 30 hPa.
    heat_source_del2_iters: int = 0
        # FV3_3D iter 458 (PE mirror of NH 457): del-2 smoothing of _d_con_sum heat source.
    heat_source_del2_coeff: float = 0.20
        # FV3 cnst_0p20 from dyn_core.F90.
    use_fv3_sponge_damp_v: bool = False
        # FV3_3D iter 443 (PE mirror of NH 442): sponge boost of damp_v at k=0,1 (NOT k=2). FV3
        # dyn_core.F90:786-787,796-797 damp_vt=0.5*d2_divg. Linear damp^(nord_v+1) scaling.
    corner_div_damp_d2_bg_k2: float = 0.0
        # FV3_3D iter 439: per-level sponge boost at k=1,k=2 (FV3 dyn_core.F90:792,802).
        # k=1 if >0.01: d2_divg=max(d2_bg,d2_bg_k2); k=2 if >0.05: max(d2_bg, 0.2*d2_bg_k2).
        # FV3 namelist 4.0 not portable, see iter-452.
    corner_div_damp_d2_bg_k1: float = 0.0
        # FV3_3D iter 438: per-level sponge boost at k=0 (FV3 dyn_core.F90:780).
        # d2_divg = max(0.01, d2_bg, d2_bg_k1). FV3 namelist 4.0 not portable, see iter-452;
        # legoESM use ~1e-4 with d2_bg=5e-4.
    corner_div_damp_fv3_vector_fill: bool = False
        # FV3_3D iter 23: FV3-faithful vector cube-vertex fill (sw_core.F90:1762). NO-OP at nord=1.
    corner_div_damp_dt_proxy: float = 200.0
        # FV3_3D iter 188: dt fallback for adaptive cap when dt_actual not passed. PE typical 50-200s.
    sponge_implicit: bool = False
        # Issue-#273 throughput work: when True, skip the explicit
        # ``-α u`` Rayleigh-sponge tendency contribution inside
        # ``fv3_hydrostatic_tendencies`` and apply the sponge as an
        # operator-split multiplicative damping
        # ``u ← u · exp(-α dt)`` after the RK3 integrator inside
        # ``CDGridPrimitiveEquationModel._step_fv3``.  Unconditionally
        # stable for any ``α · dt > 0`` — takes the sponge out of
        # the explicit-CFL budget so future work on issue #273 can
        # push ``dt`` upward without re-tuning ``sponge_tau_sec``.
        # Default ``False`` keeps the legacy tendency-form path
        # bit-exact.  Field appended to the end of the NamedTuple
        # to preserve positional construction for legacy call sites.
    implicit_grav_wave_use_pcg: bool = False
        # Issue-#273 throughput work, Phase 3 of the Hoskins–Simmons
        # FV3 D-grid port.  When True and
        # ``implicit_grav_wave_damping > 0``, the post-RK3 surface-
        # pressure correction switches from the legacy explicit
        # forward-Euler diffusion ``p_s ← p_s + α dt ∇²p_s``
        # (conditionally stable at ``α dt / dx² < 0.5``) to an
        # implicit Helmholtz solve
        # ``(I − α dt ∇²) p_s_new = p_s_explicit`` via
        # ``jax.scipy.sparse.linalg.cg``.  The cubed-sphere D-grid
        # ``cdgrid_scalar_laplacian`` is built on a nearest-copy
        # halo exchange so it is FV-adjoint-symmetric + negative
        # semi-definite under the area-weighted inner product, and
        # a ``M^{1/2}·A·M^{-1/2}`` shim casts that to a Euclidean-
        # SPD operator inside ``cg_helmholtz_solve``.  Production
        # tolerance ``1e-10`` reached in ~10 CG iterations at
        # ``α dt / dx² ≤ 5``.
        #
        # Solver contract.  When CG fails to reach tolerance within
        # ``maxiter=200`` (extreme coefficients or pathological
        # metrics), the production wrapper falls back JAX-safely to
        # *no damping for this step* (``p_s`` left unchanged) and
        # surfaces a ``RuntimeWarning`` via ``jax.debug.callback``
        # — falling back to the legacy explicit path at the same
        # coefficient would re-introduce the CFL instability the
        # implicit path was meant to suppress.
        #
        # AD: ``jax.scipy.sparse.linalg.cg`` installs an implicit-
        # function-theorem VJP, so ``jax.grad`` flows cleanly
        # through this branch (subject to the warm-start being
        # detached from the gradient).
        #
        # Default ``False`` keeps the legacy explicit-diffusion
        # path bit-exact.
    p_ceil: float = 2.0e6          # Surface-pressure ceiling [Pa] (~20-bar overflow guard for omega/p). Appended last to preserve positional ABI.
    moisture_flux_form: bool = False
        # #771: when True, transport the state tracers with the mass-conserving
        # flux-form substep (flux_form_tracer_step, co-transported δp) AFTER RK3
        # instead of the in-RK3 advective -(u·∇q) form.  Fixes the cube
        # column-water non-conservation / day-150 blow-up.  Default False keeps
        # the advective path bit-exact.  Appended last (positional ABI).


def validate_corner_div_damp_nord(nord: int) -> None:
    """FV3_3D iter 890: explicit nord range validation.

    FV3 namelist `nord` is documented integer in {0, 1, 2, 3}:
      0  — del-2 only (no higher-order Laplacian)
      1  — del-4 (one Laplacian iteration)
      2  — del-6 (two Laplacian iterations)
      3  — del-8 (three Laplacian iterations)

    legoESM tested range: nord ∈ {0, 1, 2, 3} (regression-guarded
    in iter-886/887/888/889).  Higher values are not FV3-canonical
    and not regression-tested; raise to prevent silent misuse.

    Raises
    ------
    ValueError
        If nord < 0 or nord > 3.
    """
    if not isinstance(nord, (int,)) or nord < 0 or nord > 3:
        raise ValueError(
            f"corner_div_damp_nord={nord!r} outside FV3 namelist "
            f"range {{0, 1, 2, 3}}.  legoESM only regression-tested "
            f"for these values (iter 886-889).  Use 0 (del-2 only), "
            f"1 (del-4, FV3 default), 2 (del-6), or 3 (del-8)."
        )


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
    dt_actual: float | jax.Array | None = None,
) -> FV3HydrostaticTendencies:
    """Compute tendencies for FV3 hydrostatic PE with D-grid winds.

    physics_tendency_cc (iter-65): cell-centre physics; du/dv ride iter-64 batched corner interp.
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
    p_s = jnp.clip(p_s, config.p_floor, config.p_ceil)

    # --- 1. D-grid to C-grid ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    # u_c: (6, n+1, n, nlev),  v_c: (6, n, n+1, nlev)

    # Cell-centre velocities from D-grid (orthogonal basis, for KE)
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)

    # --- 2. Pressure at full levels + layer thickness dp (continuity) ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)  # (6, n, n, nlev)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        # astype: keep dp in the state dtype when the σ arrays are f32.
        dp = p_s[..., jnp.newaxis] * sigma_coord.dsigma.astype(p_s.dtype)

    # --- 3. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 4. KE at cell centres from C-grid velocities ---
    KE = 0.5 * (u_cell ** 2 + v_cell ** 2)

    # --- 5. Bernoulli function B = KE + Phi (cell centres) ---
    # FV3_3D iter 4: Lin (1997) cross-product PGF not stable with RK3 (needs forward-backward);
    # use_fv3_lin_pgf flag inert here.
    B = KE + Phi

    # --- 6. D-grid vorticity at cell centres via circulation ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)  # (6, n, n, nlev)

    # Stage-level packed halo (iter-58/60): {ζ, B, 1/T} always; ln_ps_3d/hybrid_factor/u_cell/v_cell conditional.
    ln_ps = jnp.log(p_s)
    inv_T = 1.0 / T
    ln_ps_3d = ln_ps[..., jnp.newaxis]  # (6, n, n, 1) — rides the pack
    if _hybrid:
        _hybrid_factor = sigma_coord.B_full * p_s[..., None] / p_full  # (6, n, n, nlev)
    else:
        _hybrid_factor = None
    # iter-61: precompute div_v for div_damp so A-L gradient skips standalone halo
    # (continuity is flux-form div(dp·v) — div_v is ONLY needed for div damping)
    _need_div_pad = config.div_damp_coeff > 0
    if _need_div_pad:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)
    else:
        div_v = None
    from legoesm.grids.halo import get_halo_backend
    _halo_backend = get_halo_backend()
    _needs_uv_pad = config.A_h > 0 or config.hyperdiff_coeff > 0
    # iter-84: packed MPI/SPMD halos apply duogrid kinked-to-extended remap when duogrid=dg
    _pe_dg = grid.duogrid
    if _halo_backend == "spmd":
        from legoesm.parallel.cubesphere_exchange import get_spmd_mesh
        _spmd_mesh = get_spmd_mesh()
        # FV3_3D 2026-06-04: thread ``interp_offsets`` exactly as the MPI and
        # single-device paths do (when duogrid is off).  Previously the SPMD
        # path dropped it, so the packed exchange NEAREST-COPIED cross-face
        # halos instead of applying the 3-point Lagrange correction — a
        # ~0.6% global divergence from the single-device reference at 1 step
        # (the cube-edge halo error propagates through the dynamics).  This
        # broke the 16 ``test_cubed_sphere_spmd_step`` parity cases.
        _pe_offs_zeta = None if _pe_dg is not None else grid.halo_interp_offsets
        # iter-58/60: ln_ps_3d (+ hybrid_factor when hybrid) RIDE this stage
        # pack so the sec-8 PGF gradient + sec-8 hybrid-coord correction reuse
        # the merged halo instead of taking 1-2 SEPARATE collectives/substep
        # — reviving the iter-59/60 _lnps_pad/_hf_pad reuse branches below, which
        # were dead because the slots were never filled (the comment at the
        # ln_ps_3d definition said it "rides the pack" but it did not).  Mirrors
        # the NH sibling compressible_euler_cdgrid.py:320 {K, pi_prime} pack.
        # dp rides the pack too: the flux-form continuity (sec 10b) needs
        # dp at the C-grid faces — same fused-entry-exchange trick as the
        # lat-lon PE's _dp_lat_pad.
        _pe_pack = [zeta, B, inv_T, ln_ps_3d, dp]
        _pe_opt = []                       # names of optional trailing fields
        if _hybrid:
            _pe_pack.append(_hybrid_factor); _pe_opt.append("hf")
        if _need_div_pad:                  # div_v rides too (div-damp AL grad)
            _pe_pack.append(div_v); _pe_opt.append("div")
        _pe_pieces = packed_pad_halo_4d(
            *_pe_pack, mesh=_spmd_mesh, duogrid=_pe_dg,
            interp_offsets=_pe_offs_zeta,
        )
        _zeta_pad, _B_pad, _invT_pad, _lnps_pad, _dp_pad = _pe_pieces[:5]
        _opt = dict(zip(_pe_opt, _pe_pieces[5:]))
        _hf_pad = _opt.get("hf")
        _div_v_pad = _opt.get("div")
    elif _halo_backend == "mpi":
        from legoesm.grids.halo import get_mpi_topology
        # FV3_3D iter-1041: pass interp_offsets when duogrid is off so the
        # packed MPI exchange Lagrange-remaps halos rather than nearest-
        # copying — matches the single-device path which threads
        # ``grid.halo_interp_offsets`` here.
        _pe_offs_zeta = None if _pe_dg is not None else grid.halo_interp_offsets
        # iter-58/60/61: ride ln_ps_3d (+ hybrid_factor + div_v) on this pack —
        # see SPMD note.
        # dp rides the pack (flux-form continuity) — see SPMD note.
        _pe_pack = [zeta, B, inv_T, ln_ps_3d, dp]
        _pe_opt = []
        if _hybrid:
            _pe_pack.append(_hybrid_factor); _pe_opt.append("hf")
        if _need_div_pad:
            _pe_pack.append(div_v); _pe_opt.append("div")
        _pe_pieces = packed_pad_halo_mpi_4d(
            *_pe_pack, topology=get_mpi_topology(), duogrid=_pe_dg,
            interp_offsets=_pe_offs_zeta,
        )
        _zeta_pad, _B_pad, _invT_pad, _lnps_pad, _dp_pad = _pe_pieces[:5]
        _opt = dict(zip(_pe_opt, _pe_pieces[5:]))
        _hf_pad = _opt.get("hf")
        _div_v_pad = _opt.get("div")
    else:
        # operators do own exchange (single-device); no merged stage halo.
        _zeta_pad = _B_pad = _invT_pad = _lnps_pad = _hf_pad = None
        _div_v_pad = None
        _dp_pad = None

    # _lnps_pad/_hf_pad/_div_v_pad are now filled by the stage pack above
    # (SPMD/MPI) or None (single-device → per-op halo).

    # FV3_3D iter 14/190: optional a2b_ord4 for ζ_corner; shared with iter-187 smag_vort cap (sw_core.F90:1795)
    _need_zeta_a2b_for_smag = (
        config.corner_div_damp_d2_bg > 0.0
        and config.corner_div_damp_d4_bg > 0.0
        and config.corner_div_damp_nord > 0
    )
    _need_zeta_a2b = config.use_fv3_a2b_zeta_corner or _need_zeta_a2b_for_smag
    _zeta_a2b_ord4: jax.Array | None = None
    if _need_zeta_a2b:
        from legoesm.core.operators_cdgrid import (
            interp_center_to_corner_a2b_ord4,
        )
        # FV3_3D iter-1043: ``interp_center_to_corner_a2b_ord4`` is
        # shape-polymorphic (axes-1/2 slicing, trailing axes broadcast)
        # and ``_pad_halo_auto_h2`` already dispatches to ``pad_halo_4d``
        # for 4D input.  Calling it directly on 4D avoids a ``jax.vmap``
        # that would wrap ``pad_halo`` under MPI — same fix as NH iter-1043.
        _zeta_a2b_ord4 = interp_center_to_corner_a2b_ord4(zeta, cdgrid)

    if config.use_fv3_a2b_zeta_corner:
        zeta_corner_relative = _zeta_a2b_ord4
    else:
        zeta_corner_relative = interp_center_to_corner(
            zeta, cdgrid, padded=_zeta_pad,
        )
    zeta_corner = zeta_corner_relative + cdgrid.f_corner[..., None]

    # --- 7. Bernoulli gradient at D-grid corners (Arakawa-Lamb) ---
    dB_dx, dB_dy_perp = arakawa_lamb_gradient(B, cdgrid, padded=_B_pad)

    # --- 8. Pressure gradient correction at D-grid corners ---
    # Higher precision for PGF to avoid catastrophic cancellation
    _pg_dt = jnp.result_type(ln_ps.dtype, resolve_dtype("atm_pressure_gradient", "compute"))
    ln_ps_hi = ln_ps.astype(_pg_dt)
    # iter-59: reuse _lnps_pad from merged stage halo when dtype matches
    if _lnps_pad is not None and ln_ps.dtype == _pg_dt:
        _lnps_pad_hi = _lnps_pad[..., 0]  # (6, n+2, n+2) at PGF precision
        dln_dx_hi, dln_dy_perp_hi = arakawa_lamb_gradient(
            ln_ps_hi, cdgrid, padded=_lnps_pad_hi,
        )
    else:
        dln_dx_hi, dln_dy_perp_hi = arakawa_lamb_gradient(ln_ps_hi, cdgrid)  # 2D, separate exchange
    # Harmonic mean for T at corners suppresses spurious PGF from high-n T.
    T_corner = 1.0 / interp_center_to_corner(inv_T, cdgrid, padded=_invT_pad)
    T_corner_hi = T_corner.astype(_pg_dt)
    pg_corr_x = (R_d * T_corner_hi * dln_dx_hi[..., None]).astype(u_d.dtype)
    pg_corr_y_perp = (R_d * T_corner_hi * dln_dy_perp_hi[..., None]).astype(v_d.dtype)

    # Hybrid coord: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s). Required at model top.
    if _hybrid:
        # iter-60: reuse pre-padded _hf_pad from merged halo
        _hf_corner = interp_center_to_corner(
            _hybrid_factor, cdgrid, padded=_hf_pad,
        )
        pg_corr_x = pg_corr_x * _hf_corner
        pg_corr_y_perp = pg_corr_y_perp * _hf_corner

    # --- 9. D-grid momentum tendencies ---
    du_d_dt = zeta_corner * v_d - dB_dx - pg_corr_x
    dv_d_dt = -zeta_corner * u_d - dB_dy_perp - pg_corr_y_perp

    # --- 10a. C-grid divergence damping ---
    # div_v was precomputed at the stage pack iff div_damp_coeff > 0 (the
    # flux-form continuity in 10b no longer consumes it).
    if config.div_damp_coeff > 0:
        if config.use_async_halo and _halo_backend == "mpi":
            ddiv_dx, ddiv_dy_perp = overlapped_arakawa_lamb_gradient(
                div_v, cdgrid,
            )
        else:
            ddiv_dx, ddiv_dy_perp = arakawa_lamb_gradient(
                div_v, cdgrid, padded=_div_v_pad,
            )
        # FV3_3D iter 5: adaptive Smag damp (sw_core.F90:1720)
        # damp = da_min_c * max(d2_bg, min(0.20, dddmp*|div|))
        if config.div_damp_dddmp > 0:
            _da_min_c = jnp.min(cdgrid.area_corner)
            _d2_bg = config.div_damp_coeff / _da_min_c
            _div_abs_corner = interp_center_to_corner(
                jnp.abs(div_v), cdgrid,
            )                                                  # (6, n+1, n+1, nlev)
            _adaptive_coeff = _da_min_c * jnp.maximum(
                _d2_bg,
                jnp.minimum(0.20, config.div_damp_dddmp * _div_abs_corner),
            )                                                  # (6, n+1, n+1, nlev)
            _du_d_dt_dd = _adaptive_coeff * ddiv_dx
            _dv_d_dt_dd = _adaptive_coeff * ddiv_dy_perp
        else:
            _du_d_dt_dd = config.div_damp_coeff * ddiv_dx
            _dv_d_dt_dd = config.div_damp_coeff * ddiv_dy_perp
        du_d_dt = du_d_dt + _du_d_dt_dd
        dv_d_dt = dv_d_dt + _dv_d_dt_dd

        # FV3_3D iter 223: KE→heat d_con for cell-centre div_damp; dT/dt = -coeff*(u·du+v·dv)/c_pd
        if config.div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 349: metric-aware form at PE cell-centre div_damp
                _u_d_n = 0.5 * (u_d[:, :-1, :, :] + u_d[:, 1:, :, :])
                _v_d_n = 0.5 * (v_d[:, :, :-1, :] + v_d[:, :, 1:, :])
                _du_n = 0.5 * (
                    _du_d_dt_dd[:, :-1, :, :] + _du_d_dt_dd[:, 1:, :, :]
                )
                _dv_n = 0.5 * (
                    _dv_d_dt_dd[:, :, :-1, :] + _dv_d_dt_dd[:, :, 1:, :]
                )
                _ubs = _du_n[:, :, :-1, :]
                _ubn = _du_n[:, :, 1:, :]
                _vbw = _dv_n[:, :-1, :, :]
                _vbe = _dv_n[:, 1:, :, :]
                _us = _u_d_n[:, :, :-1, :]
                _un = _u_d_n[:, :, 1:, :]
                _vw = _v_d_n[:, :-1, :, :]
                _ve = _v_d_n[:, 1:, :, :]
                _u2 = _us + _un
                _du2 = _ubs + _ubn
                _v2 = _vw + _ve
                _dv2 = _vbw + _vbe
                _cosa = cdgrid.cosa_cell[..., None]
                _rsin2 = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_m = 0.25 * _rsin2 * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (
                        _us * _ubs + _un * _ubn
                        + _vw * _vbw + _ve * _vbe
                    )
                    - _cosa * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
                _dT_dt_dd_cc = (
                    -config.div_damp_d_con
                    * _dKE_dt_cc_m
                    / constants.c_pd
                )
            else:
                _dKE_dt_corner_dd = (
                    u_d * _du_d_dt_dd + v_d * _dv_d_dt_dd
                )
                _dT_dt_dd_cc = (
                    -config.div_damp_d_con
                    * interp_corner_to_center(_dKE_dt_corner_dd)
                    / constants.c_pd
                )
        else:
            _dT_dt_dd_cc = None
    else:
        _dT_dt_dd_cc = None

    # FV3_3D iter 16: B-grid corner-div damping (FV3 sw_core.F90:1641-1724).
    # ke(i,j) += damp*delpc(i,j); momentum -= grad(ke). Differs from cell-centre div_damp above.
    if config.corner_div_damp_d2_bg > 0.0:
        from legoesm.core._fv3_divergence_corner import (
            fv3_divergence_corner_3d,
        )
        # Step 1: B-grid corner divergence (Fortran delpc)
        delpc = fv3_divergence_corner_3d(u_d, v_d, cdgrid)  # (6, n+1, n+1, nlev)

        # Step 2: adaptive damp at corners (FV3 sw_core.F90:1720)
        # damp = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt))
        _da_min_c = jnp.min(cdgrid.area_corner)
        _delpc_abs = jnp.abs(delpc)
        # iter-189: prefer dt_actual when provided; fall back to dt_proxy (iter-188)
        if dt_actual is not None:
            _dt_approx = dt_actual
        else:
            _dt_approx = config.corner_div_damp_dt_proxy
        _damp_corner = _da_min_c * jnp.maximum(
            config.corner_div_damp_d2_bg,
            jnp.minimum(
                0.20, config.corner_div_damp_dddmp * _delpc_abs * _dt_approx,
            ),
        )                                                  # (6, n+1, n+1, nlev)

        # FV3_3D iter 438/439/446: per-level sponge boost via shared helper (FV3 dyn_core.F90:780,792,802)
        from legoesm.core.fv3_sponge_boost import (
            apply_top_sponge_damp_boost as _shared_boost,
        )
        _damp_corner = _shared_boost(
            _damp_corner, _da_min_c,
            config.corner_div_damp_d2_bg,
            config.corner_div_damp_d2_bg_k1,
            config.corner_div_damp_d2_bg_k2,
        )

        # FV3_3D iter 18: del-(2*(nord+1)) damping (FV3 sw_core.F90:1725-1822, nord>0 path)
        # dd8 = (da_min_c*d4_bg)^(nord+1); ke_corr = damp2*delpc + dd8*divg_d
        # FV3_3D iter 893: nord-loop preserved inline (a 1-ULP trace-reorder
        # would break the iter-22 bit-for-bit test).
        if config.corner_div_damp_d4_bg > 0.0 and config.corner_div_damp_nord > 0:
            from legoesm.core._fv3_divergence_corner import (
                fv3_corner_laplacian_iteration,
            )
            _vfill = config.corner_div_damp_fv3_vector_fill

            # FV3_3D iter-1044: ``fv3_corner_laplacian_iteration`` is now
            # 4D-native (mirror of NH iter-1044).  Direct call avoids
            # ``jax.vmap`` around ``pad_halo`` which fires mpi4jax's
            # sendrecv batch-axis assertion under MPI.
            _delpc_initial = delpc
            _divg_d_iter = delpc
            for _ in range(config.corner_div_damp_nord):
                _divg_d_iter = fv3_corner_laplacian_iteration(
                    _divg_d_iter, cdgrid, apply_vector_corner_fill=_vfill,
                )

            # FV3_3D iter 187: smag_vort cap for nord>=1 (FV3 sw_core.F90:1797-1809).
            # smag_vort = |dt|*sqrt(delpc² + wk_corner²); wk_corner = a2b_ord4(zeta_relative).
            # iter-181/183: double-where pattern for grad-safe sqrt at rest state.
            # iter-190: reuses _zeta_a2b_ord4 from earlier site.
            _zeta_smag_corner = _zeta_a2b_ord4              # (6, n+1, n+1, nlev)
            _smag_arg = _delpc_initial ** 2 + _zeta_smag_corner ** 2
            _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
            _smag_root = jnp.where(
                _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
            )
            _smag_vort = jnp.abs(_dt_approx) * _smag_root    # (6, n+1, n+1, nlev)
            _damp_corner = _da_min_c * jnp.maximum(
                config.corner_div_damp_d2_bg,
                jnp.minimum(0.20, config.corner_div_damp_dddmp * _smag_vort),
            )                                                # (6, n+1, n+1, nlev)

            _dd8 = jnp.asarray(
                (_da_min_c * config.corner_div_damp_d4_bg)
                ** (config.corner_div_damp_nord + 1),
                dtype=delpc.dtype,
            )
            _ke_correction = (
                _damp_corner * _delpc_initial + _dd8 * _divg_d_iter
            )                                               # (6, n+1, n+1, nlev)
        else:
            _ke_correction = _damp_corner * delpc          # (6, n+1, n+1, nlev)

        # Step 4: gradient at D-grid corners. FV3_3D iter 333: PE ke_correction halo via duogrid
        # remap (mirror of NH iter-325 fix for fv_duogrid.F90 Lagrange-extended halo)
        _pe_dg_ke = grid.duogrid
        from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d_fn
        _ke_pad = _pad_halo_4d_fn(
            _ke_correction, duogrid=_pe_dg_ke,
        )                                                  # (6, n+3, n+3, nlev)

        # Centred difference at corner (i,j): ∂x ke = (ke_pad[i+2,j+1] - ke_pad[i,j+1])/(2*dx_corner)
        _dke_dx_pad = (_ke_pad[:, 2:, 1:-1, :] - _ke_pad[:, :-2, 1:-1, :])
        _dke_dy_pad = (_ke_pad[:, 1:-1, 2:, :] - _ke_pad[:, 1:-1, :-2, :])

        # 2*dx, 2*dy at corners via padded cdgrid.dxc/dyc (face-centred)
        _dx_corner_uface = jnp.pad(
            cdgrid.dxc, [(0, 0), (0, 0), (0, 1)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _dy_corner_vface = jnp.pad(
            cdgrid.dyc, [(0, 0), (0, 1), (0, 0)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _two_dx = 2.0 * _dx_corner_uface[..., None]         # (6, n+1, n+1, 1)
        _two_dy = 2.0 * _dy_corner_vface[..., None]

        # u -= grad(ke); dt_approx cancels (iter-16 damping is per-step rate)
        _du_d_dt_cdd = -_dke_dx_pad / _two_dx
        _dv_d_dt_cdd = -_dke_dy_pad / _two_dy
        du_d_dt = du_d_dt + _du_d_dt_cdd
        dv_d_dt = dv_d_dt + _dv_d_dt_cdd

        # FV3_3D iter 221: KE→heat d_con for corner-div damp (FV3 sw_core.F90:1085-1086, dyn_core.F90:1764-1779)
        # dT/dt = -d_con*(u·du+v·dv)/c_pd. iter-218/219 delt_max applied in section 12.
        if config.corner_div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 347: metric-aware form at corner-div d_con site
                _u_d_n = 0.5 * (u_d[:, :-1, :, :] + u_d[:, 1:, :, :])
                _v_d_n = 0.5 * (v_d[:, :, :-1, :] + v_d[:, :, 1:, :])
                _du_n = 0.5 * (
                    _du_d_dt_cdd[:, :-1, :, :] + _du_d_dt_cdd[:, 1:, :, :]
                )
                _dv_n = 0.5 * (
                    _dv_d_dt_cdd[:, :, :-1, :] + _dv_d_dt_cdd[:, :, 1:, :]
                )
                _ubs = _du_n[:, :, :-1, :]
                _ubn = _du_n[:, :, 1:, :]
                _vbw = _dv_n[:, :-1, :, :]
                _vbe = _dv_n[:, 1:, :, :]
                _us = _u_d_n[:, :, :-1, :]
                _un = _u_d_n[:, :, 1:, :]
                _vw = _v_d_n[:, :-1, :, :]
                _ve = _v_d_n[:, 1:, :, :]
                _gys = _us * _ubs
                _gyn = _un * _ubn
                _gxw = _vw * _vbw
                _gxe = _ve * _vbe
                _u2 = _us + _un
                _du2 = _ubs + _ubn
                _v2 = _vw + _ve
                _dv2 = _vbw + _vbe
                _cosa = cdgrid.cosa_cell[..., None]
                _rsin2 = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_m = 0.25 * _rsin2 * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (_gys + _gyn + _gxw + _gxe)
                    - _cosa * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
                _dT_dt_cdd_cc = (
                    -config.corner_div_damp_d_con
                    * _dKE_dt_cc_m
                    / constants.c_pd
                )
            else:
                _dKE_dt_corner_cdd = (
                    u_d * _du_d_dt_cdd + v_d * _dv_d_dt_cdd
                )
                _dT_dt_cdd_cc = (
                    -config.corner_div_damp_d_con
                    * interp_corner_to_center(_dKE_dt_corner_cdd)
                    / constants.c_pd
                )
        else:
            _dT_dt_cdd_cc = None

    else:
        _dT_dt_cdd_cc = None

    # --- 10b. Surface pressure tendency and vertical motion ---
    # Iter-1: skip per-stage zero_mean_tendency when end-step fix_mass active (saves 3-4 allreduces/step)
    _apply_zero_mean_per_stage = (
        config.zero_mean_ps_tendency
        and not (config.use_conservation_fixer and config.fix_mass)
    )

    # Flux-form continuity (both branches): interpolate dp to the C-grid
    # faces (2-point average, halo-consistent cross-face pad — the cube
    # analogue of the lat-lon PE's interp_cell_to_uface/vface) and take the
    # exact flux-form divergence div(dp_k·v).  The previous advective
    # closure div(v)·dp_k differs wherever ∇p_s ≠ 0 and leaves an
    # O(v·∇p_s) residual in the GLOBAL ∫dp_s/dt·dA budget that only the
    # mass fixer masked; the flux form telescopes (face fluxes cancel in
    # pairs), so ∫dp_s/dt·dA vanishes to seam-halo precision (exactly,
    # with duogrid seam-flux sync).  Sign convention: the divergence is
    # +div; positive divergence (mass export) ⇒ dp_s/dt < 0 below.
    if _dp_pad is None:  # single-device: per-op halo (MPI/SPMD: stage pack)
        _dp_pad = pad_halo_auto(dp, cdgrid)  # (6, n+2, n+2, nlev)
    dp_u, dp_v = cgrid_interp_cc_to_faces_local(_dp_pad)
    div_dp = cgrid_flux_divergence_sync(
        dp_u, dp_v, u_c, v_c, cdgrid)  # (6, n, n, nlev)
    # cumsum reused for BOTH dp_s_dt (last entry) and the σ̇/mass-flux
    # integration below (iter-52/54 pattern: one cross-level collective).
    _cumsum_dp = jnp.cumsum(div_dp, axis=-1)  # (6, n, n, nlev)
    _D_total_p = _cumsum_dp[..., -1:]         # (6, n, n, 1)  [Pa/s]

    if _hybrid:
        # Hybrid closure: B_range · dp_s/dt = -Σ_k div(dp_k·v).
        dp_s_dt_data = -_D_total_p[..., 0] / sigma_coord.B_range
        if _apply_zero_mean_per_stage:
            dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        # Flux-form mass flux from the SAME cumsum (shared boundary closure).
        mass_flux = compute_mass_flux_from_cumsum(
            _cumsum_dp, _D_total_p, sigma_coord,
        )

        # Loop 113/114: batch (u_d, v_d) corner→centre and back
        n_face_uv, n_i_uv, n_j_uv, nlev_uv = u_d.shape[0], u_d.shape[1] - 1, u_d.shape[2] - 1, u_d.shape[3]
        _uv_d = jnp.stack([u_d, v_d], axis=-1)
        _uv_cc_flat = interp_corner_to_center(
            _uv_d.reshape(*_uv_d.shape[:-2], nlev_uv * 2),
        )
        _uv_cc = _uv_cc_flat.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2)
        # Loop 168: batch vertical_advection_hybrid over (u_cc, v_cc, T); shared mass_flux/p_s
        _uvT_cc_lead = jnp.stack(
            [_uv_cc[..., 0], _uv_cc[..., 1], T], axis=0,
        )  # (3, face, i, j, nlev)
        _vert_adv_uvT_lead = vertical_advection_hybrid(
            _uvT_cc_lead, mass_flux, p_s, sigma_coord,
        )
        _vert_adv_uv_cc = jnp.moveaxis(_vert_adv_uvT_lead[:2], 0, -1)
        vert_adv_T = _vert_adv_uvT_lead[2]
        # iter-64: defer _vert_adv_uv_cc corner interp to section 12c (batched with diff)

        # Tracer vertical advection uses the SAME hybrid mass flux as T (bound
        # via default args so the closure captures this branch's arrays).
        def _tracer_vert_fn(_q1, _mf=mass_flux, _ps=p_s):
            return vertical_advection_hybrid(_q1, _mf, _ps, sigma_coord)

        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)
    else:
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        # σ closure: (1 - σ_top) · dp_s/dt = -Σ_k div(dp_k·v), dp_k = p_s·Δσ_k
        # (div_dp already carries the p_s factor — no extra p_s multiply).
        dp_s_dt_data = -_D_total_p[..., 0] / sigma_range
        if _apply_zero_mean_per_stage:
            dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        # Flux-form σ̇ from the SAME cumsum (shared closure; mirrors the
        # lat-lon C-grid reference implementation).
        sigma_dot = compute_sigma_dot_from_cumsum(
            _cumsum_dp, _D_total_p, p_s, sigma_coord,
        )

        n_face_uv, n_i_uv, n_j_uv, nlev_uv = u_d.shape[0], u_d.shape[1] - 1, u_d.shape[2] - 1, u_d.shape[3]
        _uv_d = jnp.stack([u_d, v_d], axis=-1)
        _uv_cc_flat = interp_corner_to_center(
            _uv_d.reshape(*_uv_d.shape[:-2], nlev_uv * 2),
        )
        _uv_cc = _uv_cc_flat.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2)
        # Loop 168: batch vertical_advection over (u_cc, v_cc, T)
        _uvT_cc_lead = jnp.stack(
            [_uv_cc[..., 0], _uv_cc[..., 1], T], axis=0,
        )  # (3, face, i, j, nlev)
        _vert_adv_uvT_lead = vertical_advection(
            _uvT_cc_lead, sigma_dot, sigma_coord,
        )
        _vert_adv_uv_cc = jnp.moveaxis(_vert_adv_uvT_lead[:2], 0, -1)
        vert_adv_T = _vert_adv_uvT_lead[2]

        # Tracer vertical advection uses the SAME sigma_dot as T (bound via a
        # default arg so the closure captures this branch's array).
        def _tracer_vert_fn(_q1, _sd=sigma_dot):
            return vertical_advection(_q1, _sd, sigma_coord)

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)

    # iter-64: vert_adv_uv_d added with lap_uv/hyperdiff_uv at section 12c (shared halo)

    # --- 11. Thermodynamic equation ---
    # Loop 189: pack ln_ps_3d with T (and u/v if needed) — saves 2 halo exchanges
    _needs_uv_pad = config.A_h > 0 or config.hyperdiff_coeff > 0
    _pad_halo_4d = _pad_halo_4d_module
    _pe_dg = grid.duogrid
    ln_ps_3d = ln_ps[..., jnp.newaxis]  # (6, n, n, 1)
    # Scalars T + ln(ps): packed halo (MPI/SPMD) / per-field (single-device).
    if _halo_backend == "mpi":
        from legoesm.grids.halo import get_mpi_topology
        # FV3_3D iter-1041: thread interp_offsets through the packed MPI
        # exchange to match the single-device Lagrange remap when duogrid off.
        _pe_offs = None if _pe_dg is not None else grid.halo_interp_offsets
        _T_pad, _lnps_pad = packed_pad_halo_mpi_4d(
            T, ln_ps_3d, topology=get_mpi_topology(), duogrid=_pe_dg,
            interp_offsets=_pe_offs,
        )
    elif _halo_backend == "spmd":
        # Pack T + ln(ps) into ONE collective, mirroring the MPI branch
        # (the SPMD path previously did 2 separate exchanges — each is a
        # full cross-node latency on Gloo-TCP, clock job 8473330: 0.4-4.9
        # ms/permute, no pipelining; same zeta/B/invT pattern above).
        from legoesm.parallel.cubesphere_exchange import get_spmd_mesh
        _pe_offs = None if _pe_dg is not None else grid.halo_interp_offsets
        _T_pad, _lnps_pad = packed_pad_halo_4d(
            T, ln_ps_3d, mesh=get_spmd_mesh(), duogrid=_pe_dg,
            interp_offsets=_pe_offs,
        )
    else:
        # Duogrid remap when active (matches pad_halo_auto pattern)
        _pe_offs = None if _pe_dg is not None else grid.halo_interp_offsets
        _T_pad = _pad_halo_4d(T, interp_offsets=_pe_offs, duogrid=_pe_dg)
        _lnps_pad = _pad_halo_4d(ln_ps_3d, interp_offsets=_pe_offs, duogrid=_pe_dg)
    # u_cell/v_cell are face-local VECTOR components — pad with the rotation-aware
    # vector halo in BOTH paths.  fv3_faithful (iter-14): `pad_halo_vector_4d`
    # auto-dispatches to the MPI/SPMD backend (rotate→pad→rotate), so the ∇²/∇⁴
    # diffusion stencil reads ROTATED neighbour-face halos at cube panel seams on
    # every backend.  The MPI path previously packed (u,v) as SCALARS, which fed
    # UNROTATED seam halos into the diffusion → the cube panel-edge imprint
    # returned under MPI (single-device was already fixed).
    if _needs_uv_pad:
        _u_cc_pad, _v_cc_pad = pad_halo_vector_4d(
            u_cell, v_cell,
            cdgrid.base.cos_angle, cdgrid.base.sin_angle,
            cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
            interp_offsets=_pe_offs,
            duogrid=_pe_dg,
        )
    else:
        _u_cc_pad = _v_cc_pad = None

    dT_dx = _gradient_x_3d(T, grid, padded=_T_pad)
    dT_dy = _gradient_y_3d(T, grid, padded=_T_pad)
    horiz_adv_T = -(u_cell * dT_dx + v_cell * dT_dy)

    # Adiabatic: kappa * T * omega / p + kappa*T*v.grad(ln p_s) correction
    # Loop 189: use 3D gradients with shared _lnps_pad
    dln_ps_dx = _gradient_x_3d(ln_ps_3d, grid, padded=_lnps_pad)[..., 0]  # (6, n, n)
    dln_ps_dy = _gradient_y_3d(ln_ps_3d, grid, padded=_lnps_pad)[..., 0]
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = u_cell * dln_ps_dx[..., None] + v_cell * dln_ps_dy[..., None]
    # Hybrid: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s)
    if _hybrid:
        v_dot_grad_lnps = v_dot_grad_lnps * (sigma_coord.B_full * p_s[..., None] / p_adiab)
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 11c. Tracer advection (advective form, consistent with T) ---
    # Dynamics tracer tendency via the SHARED cubed-sphere advective-tracer core
    # (one batched cube halo pad; same numerics as TracerTransportModel and the
    # T transport above).  Physics tracer tendencies (e.g. Kessler warm-rain via
    # physics_tendency_cc.tracer_tendencies) are added in the physics block
    # below.  ``None`` when the state is dry (no tracers).
    _dtracers = None
    if state.tracers:
        _tnames = list(state.tracers)
        _q_packed = jnp.stack(
            [state.tracers[k].data for k in _tnames], axis=-1,
        )  # (6, n, n, nlev, n_tracers)
        # #771: with flux-form moisture the HORIZONTAL transport is done as a
        # mass-conserving post-RK3 substep (step()), so the in-RK3 tendency
        # keeps only the (mass-flux-consistent) vertical advection here; the
        # advective -(u·∇q) horizontal term is skipped to avoid double transport.
        _dq_packed = advective_tracer_tendency(
            _q_packed, u_cell, v_cell, grid, _tracer_vert_fn,
            horizontal=not config.moisture_flux_form,
        )
        _dtracers = {k: _dq_packed[..., i] for i, k in enumerate(_tnames)}

    # FV3_3D iter 239: aggregate 3 tendency-based d_con sources after A_h block, then cap once
    # (FV3 sw_core.F90 + dyn_core.F90:1764-1779)

    # --- 12. Diffusion ---
    # 12a: compact Laplacian at centres avoids laplacian_dgrid corner-centre roundtrip attenuation.
    # Stack {u_cell, v_cell, T} along trailing axis for batched ∇²/hyperdiff (3 calls → 1).
    _need_uvT_stack = config.A_h > 0 or config.hyperdiff_coeff > 0
    if _need_uvT_stack:
        n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT = u_cell.shape
        _uvT_stack = jnp.stack([u_cell, v_cell, T], axis=-1)
        _uvT_flat = _uvT_stack.reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT * 3)
        _uvT_pad_stack = jnp.stack([_u_cc_pad, _v_cc_pad, _T_pad], axis=-1)
        _pad_pre = _uvT_pad_stack.shape[:3]
        _uvT_pad_flat = _uvT_pad_stack.reshape(*_pad_pre, nlev_uvT * 3)

    # Share inner ∇²(uvT) between Laplacian (12a) and biharmonic hyperdiff (12b). Ocean Loop 135 pattern.
    _lap_flat: jax.Array | None = None
    if config.A_h > 0 or config.hyperdiff_coeff > 0:
        _lap_flat = _laplacian_compact_3d(
            _uvT_flat, grid, padded=_uvT_pad_flat,
        )

    # --- 11b. T-diss reuses T slice of batched ∇²(uvT) when available ---
    if config.T_diss_coeff > 0:
        # FV3_3D iter 182: double-where for grad-safe sqrt at rest state
        _ws_sq = u_cell ** 2 + v_cell ** 2
        _safe_ws_sq = jnp.where(_ws_sq > 0.0, _ws_sq, 1.0)
        wind_speed = jnp.where(
            _ws_sq > 0.0, jnp.sqrt(_safe_ws_sq), 0.0,
        )
        dx_local = grid.dx[..., None]
        nu_T = config.T_diss_coeff * wind_speed * dx_local
        if _lap_flat is not None:
            lap_T = _lap_flat.reshape(
                n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3,
            )[..., 2]
        else:
            lap_T = _laplacian_compact_3d(T, grid, padded=_T_pad)
        dT_dt_data = dT_dt_data + nu_T * lap_T

    # Iter-63/64: batch all (u, v) cell-centre→corner interps into one halo
    lap_uvT = None
    hyperdiff_uvT = None
    if config.A_h > 0:
        lap_uvT = _lap_flat.reshape(n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3)
    if config.hyperdiff_coeff > 0:
        # fv3_faithful (iter-14): the shared ``_hyperdiffusion_3d`` scalar-pads
        # the inner ∇²(u,v) result for its OUTER ∇², re-injecting an UNROTATED
        # panel-seam halo into the wind biharmonic (the inner ∇² is already
        # vector-halo'd above).  Build ∇⁴(u,v)=∇²(∇²(u,v)) with a VECTOR halo on
        # the inner ∇²(u,v); T keeps the scalar shared path (a true scalar).
        _hd_offs = None if _pe_dg is not None else grid.halo_interp_offsets
        _lap_r = _lap_flat.reshape(
            n_face_uvT, n_i_uvT, n_j_uvT, nlev_uvT, 3,
        )
        _il_u, _il_v, _il_T = _lap_r[..., 0], _lap_r[..., 1], _lap_r[..., 2]
        _il_u_pad, _il_v_pad = pad_halo_vector_4d(
            _il_u, _il_v,
            cdgrid.base.cos_angle, cdgrid.base.sin_angle,
            cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
            interp_offsets=_hd_offs, duogrid=_pe_dg,
        )
        _hd_u = -config.hyperdiff_coeff * _divergence_3d(
            _gradient_x_3d(_il_u, grid, padded=_il_u_pad),
            _gradient_y_3d(_il_u, grid, padded=_il_u_pad), grid,
        )
        _hd_v = -config.hyperdiff_coeff * _divergence_3d(
            _gradient_x_3d(_il_v, grid, padded=_il_v_pad),
            _gradient_y_3d(_il_v, grid, padded=_il_v_pad), grid,
        )
        _hd_T = _hyperdiffusion_3d(
            T, grid, config.hyperdiff_coeff, inner_lap=_il_T,
        )
        hyperdiff_uvT = jnp.stack([_hd_u, _hd_v, _hd_T], axis=-1)

    # Lift each cc VECTOR tendency block (vert_adv + lap + hyperdiff + physics)
    # to D-grid corners.  fv3_faithful (iter-14): these are face-local (u, v)
    # vector increments, so the cc→corner interp must ROTATE components across
    # cube panel seams.  The previous batched ``interp_center_to_corner`` on
    # the concatenated (u, v) treated them as scalars and seam-blended without
    # rotation (the same bug fixed for the wind lift), re-injecting a cube-edge
    # imprint into every diffusion/physics tendency.  Use ``center_to_dgrid_vector``
    # per block.  Output convention preserved: ``_*_uv_d[..., 0]`` = u_d,
    # ``[..., 1]`` = v_d at corners (6, n+1, n+1, nlev, 2).
    def _lift_uv_cc(_u_cc, _v_cc):
        _u_d, _v_d = center_to_dgrid_vector(_u_cc, _v_cc, cdgrid)
        return jnp.stack([_u_d, _v_d], axis=-1)

    _vert_adv_uv_d = _lift_uv_cc(
        _vert_adv_uv_cc[..., 0], _vert_adv_uv_cc[..., 1])
    _lap_uv_d = None
    if lap_uvT is not None:
        _lap_uv_d = _lift_uv_cc(lap_uvT[..., 0], lap_uvT[..., 1])
    _hd_uv_d = None
    if hyperdiff_uvT is not None:
        _hd_uv_d = _lift_uv_cc(hyperdiff_uvT[..., 0], hyperdiff_uvT[..., 1])
    _phys_uv_d = None
    if (physics_tendency_cc is not None
            and physics_tendency_cc.du_dt is not None):
        _phys_uv_d = _lift_uv_cc(
            physics_tendency_cc.du_dt.data, physics_tendency_cc.dv_dt.data)

    # vert_adv contribution
    du_d_dt = du_d_dt + _vert_adv_uv_d[..., 0]
    dv_d_dt = dv_d_dt + _vert_adv_uv_d[..., 1]

    if config.A_h > 0:
        # FV3_3D iter 58: Smag adaptive A_h at corners; du/dt += (A_h + ah_smag) * lap_u
        if config.smagorinsky_cs > 0.0:
            from legoesm.core._smagorinsky_visc import (
                compute_smagorinsky_ah_3d,
            )
            _ah_smag_corner = compute_smagorinsky_ah_3d(
                u_d, v_d, cdgrid, config.smagorinsky_cs,
            )                                              # (6, n+1, n+1, nlev)
            _ah_eff_corner = config.A_h + _ah_smag_corner
            _du_d_dt_ah = _ah_eff_corner * _lap_uv_d[..., 0]
            _dv_d_dt_ah = _ah_eff_corner * _lap_uv_d[..., 1]
            du_d_dt = du_d_dt + _du_d_dt_ah
            dv_d_dt = dv_d_dt + _dv_d_dt_ah
            # T at centres: interpolate ah_smag from corners (4-pt avg)
            _ah_smag_cell = 0.25 * (
                _ah_smag_corner[:, :-1, :-1, :]
                + _ah_smag_corner[:, 1:, :-1, :]
                + _ah_smag_corner[:, :-1, 1:, :]
                + _ah_smag_corner[:, 1:, 1:, :]
            )                                              # (6, n, n, nlev)
            _ah_eff_cell = config.A_h + _ah_smag_cell
            dT_dt_data = dT_dt_data + _ah_eff_cell * lap_uvT[..., 2]
        else:
            _du_d_dt_ah = config.A_h * _lap_uv_d[..., 0]
            _dv_d_dt_ah = config.A_h * _lap_uv_d[..., 1]
            du_d_dt = du_d_dt + _du_d_dt_ah
            dv_d_dt = dv_d_dt + _dv_d_dt_ah
            dT_dt_data = dT_dt_data + config.A_h * lap_uvT[..., 2]

        # FV3_3D iter 225: KE→heat d_con for A_h Laplacian; dT/dt = -ah_d_con*(u·du+v·dv)/c_pd
        if config.ah_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 351: metric-aware form at PE A_h d_con
                _u_d_n = 0.5 * (u_d[:, :-1, :, :] + u_d[:, 1:, :, :])
                _v_d_n = 0.5 * (v_d[:, :, :-1, :] + v_d[:, :, 1:, :])
                _du_n = 0.5 * (
                    _du_d_dt_ah[:, :-1, :, :] + _du_d_dt_ah[:, 1:, :, :]
                )
                _dv_n = 0.5 * (
                    _dv_d_dt_ah[:, :, :-1, :] + _dv_d_dt_ah[:, :, 1:, :]
                )
                _ubs = _du_n[:, :, :-1, :]
                _ubn = _du_n[:, :, 1:, :]
                _vbw = _dv_n[:, :-1, :, :]
                _vbe = _dv_n[:, 1:, :, :]
                _us = _u_d_n[:, :, :-1, :]
                _un = _u_d_n[:, :, 1:, :]
                _vw = _v_d_n[:, :-1, :, :]
                _ve = _v_d_n[:, 1:, :, :]
                _u2 = _us + _un
                _du2 = _ubs + _ubn
                _v2 = _vw + _ve
                _dv2 = _vbw + _vbe
                _cosa = cdgrid.cosa_cell[..., None]
                _rsin2 = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_m = 0.25 * _rsin2 * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (
                        _us * _ubs + _un * _ubn
                        + _vw * _vbw + _ve * _vbe
                    )
                    - _cosa * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
                _dT_dt_ah_cc = (
                    -config.ah_d_con
                    * _dKE_dt_cc_m
                    / constants.c_pd
                )
            else:
                _dKE_dt_corner_ah = (
                    u_d * _du_d_dt_ah + v_d * _dv_d_dt_ah
                )
                _dT_dt_ah_cc = (
                    -config.ah_d_con
                    * interp_corner_to_center(_dKE_dt_corner_ah)
                    / constants.c_pd
                )
        else:
            _dT_dt_ah_cc = None
    else:
        _dT_dt_ah_cc = None

    # FV3_3D iter 239: aggregate 3 d_con sources (iter-221/223/225), cap once with delt_max
    _d_con_sum = None
    for _contrib in (_dT_dt_cdd_cc, _dT_dt_dd_cc, _dT_dt_ah_cc):
        if _contrib is not None:
            _d_con_sum = (
                _contrib if _d_con_sum is None
                else _d_con_sum + _contrib
            )
    if _d_con_sum is not None:
        # FV3_3D iter 433: sponge-zero top N levels (PE mirror of NH 432)
        if config.d_con_top_zero_levels > 0:
            _nlev_zsp = _d_con_sum.shape[-1]
            _k_idx_zsp = jnp.arange(_nlev_zsp)
            _d_con_mask_sp = jnp.where(
                _k_idx_zsp < config.d_con_top_zero_levels,
                0.0, 1.0,
            )
            _d_con_sum = _d_con_sum * _d_con_mask_sp[None, None, None, :]
        # FV3_3D iter 458 (PE mirror of NH 457): del-2 smoothing of heat_source
        if config.heat_source_del2_iters > 0:
            _da_min_hs_pe = jnp.min(cdgrid.area_corner)
            _cd_hs_pe = config.heat_source_del2_coeff * _da_min_hs_pe
            for _ in range(config.heat_source_del2_iters):
                _lap_hs_pe = _laplacian_compact_3d(_d_con_sum, grid)
                _d_con_sum = _d_con_sum + _cd_hs_pe * _lap_hs_pe
        if config.delt_max > 0.0:
            # Sponge-aware cap: k=0,1 uncapped, k>=2 capped to delt_max K/s
            _nlev = _d_con_sum.shape[-1]
            _k_idx = jnp.arange(_nlev)
            _cap_per_level = jnp.where(
                _k_idx < 2, jnp.inf, config.delt_max,
            )
            _cap_b = _cap_per_level[None, None, None, :]
            _d_con_sum = jnp.clip(_d_con_sum, -_cap_b, _cap_b)
        dT_dt_data = dT_dt_data + _d_con_sum

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
    #
    # Two paths:
    #
    # * ``config.sponge_implicit = False`` (default) — legacy
    #   explicit-tendency form ``du/dt = -α u``.  Conditionally
    #   stable: forward Euler diverges at ``α · dt > 2`` and
    #   SSP-RK3 around ``α · dt ≳ 2.5``.  At production parameters
    #   (τ = 3600 s, dt = 150 s, peak α ≈ 2.8e-4 s⁻¹) the margin
    #   is comfortable, so this path stays bit-exact for legacy
    #   configs and direct callers of ``fv3_hydrostatic_tendencies``.
    #
    # * ``config.sponge_implicit = True`` — operator-split path.
    #   The sponge contribution is *omitted* from the tendency and
    #   instead applied once per macro step as the analytic
    #   multiplicative damping ``u ← u · exp(-α dt)`` inside
    #   ``CDGridPrimitiveEquationModel._step_fv3``.  Unconditionally
    #   stable for any ``α · dt > 0`` — takes the sponge out of
    #   the explicit-CFL budget so issue-#273 throughput work can
    #   push ``dt`` upward without re-tuning ``sponge_tau_sec``.
    #
    # The naive ``expm1(-α dt) / dt · u`` "effective-tendency"
    # trick is intentionally *not* used: SSP-RK3 re-evaluates the
    # tendency on each stage's state, so the multi-stage update
    # saturates at ``u_new ≈ u_old / 3`` for ``α dt → ∞`` instead
    # of damping to zero (Codex review iter-1 catch).
    if (config.sponge_tau_sec > 0
            and config.sponge_sigma > 0
            and not config.sponge_implicit):
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
    # Iter-65: cc du/dv rode iter-64 batch; dT/dp_s added directly
    if physics_tendency_cc is not None:
        if physics_tendency_cc.du_dt is not None and _phys_uv_d is not None:
            du_d_dt = du_d_dt + _phys_uv_d[..., 0]
            dv_d_dt = dv_d_dt + _phys_uv_d[..., 1]
        dT_dt_data = dT_dt_data + physics_tendency_cc.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency_cc.dp_s_dt.data

    # Physics tracer tendencies (e.g. Kessler warm-rain) → add to the dynamics
    # tracer tendency, for tracers already prognostic in the state (introducing
    # a new key here would break the RK integrator's pytree structure).  Both
    # tendency containers carry an optional tracer_tendencies dict.
    if _dtracers is not None:
        for _src in (physics_tendency, physics_tendency_cc):
            _tt = None if _src is None else _src.tracer_tendencies
            if _tt:
                for _k, _f in _tt.items():
                    if _k in _dtracers:
                        _dtracers[_k] = _dtracers[_k] + _f.data

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
        tracer_tendencies=(
            None if _dtracers is None else {
                k: Field(data=v, name=f"d{k}_dt", dims=dims_3d, units="kg/kg/s")
                for k, v in _dtracers.items()
            }
        ),
    )


# ==============================================================================
# Adapter: D-grid to cell-centre conversion
# ==============================================================================

def fv3_to_hydrostatic(
    state: FV3HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> HydrostaticState:
    """Convert FV3 D-grid state to cell-centre HydrostaticState (batched corner→centre)."""
    u_d = state.u_d.data
    v_d = state.v_d.data
    n_face_a, n_corner_i, n_corner_j, nlev_a = u_d.shape
    _uv_d = jnp.stack([u_d, v_d], axis=-1)  # (face, n+1, n+1, nlev, 2)
    _uv_cc_flat = interp_corner_to_center(
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
    """Hydrostatic PE model on cubed-sphere with FV3 C-D grid. Prognostic D-grid winds."""

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: CDGridPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CDGridPrimitiveEquationConfig()
        # FV3_3D iter 902: enforce iter-890 nord range validation at
        # model construction (fail-fast vs silent misuse).
        validate_corner_div_damp_nord(self.config.corner_div_damp_nord)
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self._target_mass = None
        # Operator-split physics carry (issue #413, mirrors the MPAS PE):
        # ``step(..., phys_state=...)`` stashes the updated ``PhysicsState``
        # here so the caller can feed it back next step, while ``step``
        # keeps returning the state only (public contract unchanged).
        self._phys_state = None

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-20; mirrors iter-18 API)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-20; iter-19 API)."""
        self._target_mass = target_mass

    def compute_mass(self, state) -> jax.Array:
        """Compute global ``∫ p_s dA`` in fp64 via ``global_integral``.

        iter-21: API parity with the MPAS PE (iter-11), lat-lon PE
        (iter-2/12), and spectral PE (iter-3) ``compute_mass``
        helpers.  Reuses the existing fp64-clean ``global_integral``
        path used by ``step()`` for the initial-mass snapshot.
        """
        return global_integral(state.p_s, self.grid)

    def _sync_dgrid_boundary(self, state: FV3HydrostaticState):
        """No-op: cross-face continuity via halo exchange (explicit sync seeds spurious v-wind)."""
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

    def step(self, state, dt, physics_fn=None, phys_state=None):
        """Advance one step. Accepts FV3HydrostaticState or HydrostaticState.

        ``phys_state`` is the operator-split physics carry (prognostic
        TKE / convection / GWD ``PhysicsState``, issue #413).  When
        supplied, ``physics_fn`` is called with it (4-arg contract) and
        the updated carry is stashed on ``self._phys_state`` for the
        caller to feed back next step — mirroring the MPAS PE.  Carry
        semantics: every RK stage's physics evaluation receives the
        STEP-INPUT carry; the carry-out comes from one extra physics
        evaluation on the post-step state (its tendencies are
        discarded), so the prognostic fields advance exactly once per
        ``dt`` and the carry-out is consistent with the state it
        accompanies into the next step.  ``phys_state=None`` (default)
        is byte-identical to the legacy 3-arg physics call.
        """
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="CDGrid step()")
        target_mass = None
        if (self.config.use_conservation_fixer and self.config.fix_mass
                and self.config.anchor_mass_to_initial):
            target_mass = self._target_mass
            if target_mass is None:
                target_mass = global_integral(state.p_s, self.grid)
                if not isinstance(target_mass, jax.core.Tracer):
                    # Designed eager path: cache the concrete t=0 mass so
                    # later segments keep anchoring to the same constant.
                    self._target_mass = target_mass
                # Traced path (step() called inside an OUTER jit/scan —
                # e.g. make_sharded_step, probe/bench scan drivers):
                # NEVER cache — a tracer stored on self leaks into the
                # next trace (UnexpectedTracerError).  Thread the
                # per-call pre-step mass instead; the fixer then
                # telescopes post-step mass back to the initial mass,
                # matching the non-anchor branch's semantics.

        if isinstance(state, FV3HydrostaticState):
            state_new, self._phys_state = self._step_fv3(
                state, dt, physics_fn=physics_fn, target_mass=target_mass,
                phys_state=phys_state)
            return state_new
        # Legacy cell-centre state path
        return self.step_cell_centre(
            state, dt, physics_fn=physics_fn, target_mass=target_mass,
            phys_state=phys_state)

    @staticmethod
    def _flux_form_scatter_blocked(backend, topo, spmd_mesh, state_lead) -> bool:
        """True iff the ``moisture_flux_form`` substep must fail-closed.

        The #811 unblock has BOTH halves now: (1) the mass-conservation
        reductions are allreduce-aware (``conservation.global_face_sum_if_
        scattered``), and (2) the transport + wind reconstruction are 4D-halo
        (``transport_step_4d`` / ``d2a2c_vect_4d`` — ONE ``pad_halo_4d`` /
        ``pad_halo_vector_4d`` per exchange for all levels, so there is no
        ``vmap(pad_halo)`` cross-face ``sendrecv``).  So:

        * MPI FACE-ONLY (single-rank, REPLICATED, or face-SCATTERED with
          ``tiling == (1, 1)``): **allowed** — certified replicated-vs-scattered
          equivalent (fwd + grad) by
          ``tests/distributed/test_cube_face_scatter_mpi.py`` (#811 / #771).
        * MPI SUB-FACE TILING (``tiling != (1, 1)``): fail-closed — the 4D halos
          reject ``interp_offsets`` under sub-face tiling
          (``pad_halo_mpi_4d``), so this path is unsupported (codex #811 review).
        * SPMD: still fail-closed whenever a mesh is active.  The 4D halos
          dispatch to explicit SPMD exchanges, but that combination is
          unvalidated here (#811 SPMD follow-up).

        ``state_lead`` is retained for signature + unit-test stability.
        Extracted + pure so the guard is unit-tested (a rename then breaks the
        test loudly instead of silently disabling it).
        """
        del state_lead
        if backend == "mpi":
            # Face-only MPI is supported; sub-face tiling is not (the 4D halo
            # rejects interp_offsets when tiling != (1, 1)).
            tiling = getattr(topo, "tiling", (1, 1))
            return tiling is not None and tuple(tiling) != (1, 1)
        if backend == "spmd" and spmd_mesh is not None:
            return True
        return False

    def _flux_form_moisture_substep(self, state_in, state_out, dt):
        """#771: mass-conserving flux-form HORIZONTAL tracer transport as a
        post-RK3 substep.

        The RK3 tendency already applied the (mass-flux-consistent) vertical
        advection + physics with the advective horizontal ``-(u·∇q)`` skipped
        (``moisture_flux_form``).  Here the horizontal transport is done by the
        conserving, free-stream-preserving :func:`flux_form_tracer_step`.

        DESIGN / LIMITATIONS (this is a 1st-order operator split, NOT a fully
        continuity-consistent FV transport — the hybrid PE core carries mass via
        ``p_s`` + a ``div_v`` continuity + a mass fixer, not via this flux
        operator, so there is no shared horizontal mass-flux to advect against):

        * ``flux_form_tracer_step`` co-transports its own ``δp★`` (used only to
          keep ``q★`` free-stream-preserving) and we DISCARD it; the tracer is
          then reconciled onto the dynamics' post-step ``δp`` by a per-tracer
          GLOBAL multiplicative mass fixer, so ``∫ area·δp·q`` is conserved on
          the model grid to fp-accuracy (``scale`` ≈ 1).  The end-to-end cube
          test (``test_cdgrid_flux_form_moisture``) validates this: ~3000× less
          column-water drift than the advective form under a divergent wind.
        * The transport step itself is monotone/positive-definite; the tiny
          global rescale (``scale`` ≈ 1) is NOT strictly monotone (a
          ``scale > 1`` can lift an existing maximum) but preserves
          nonnegativity when the tracer mass is nonnegative — which is why the
          input ``q`` is floored at 0 below (any negative left by RK3
          vertical/physics is unphysical noise and would otherwise poison both
          the conserved target and the later ``q >= 0`` clip).

        A fully continuity-consistent scheme would require making the PE
        continuity itself FV-flux-form (transport ``δp`` with the same operator)
        — a much larger dycore change tracked separately (#771 follow-up).
        """
        from legoesm.core.fv3_sw_core import d2a2c_vect_4d
        from legoesm.atmosphere.dynamics.shared.flux_form_tracer_transport import (
            flux_form_tracer_step,
        )
        from legoesm.core.conservation import (
            conservation_accumulator,
            global_face_sum_if_scattered,
        )

        # MPI face-scatter is now SUPPORTED (#811): the mass reductions are
        # allreduce-aware (global_face_sum_if_scattered), the transport is
        # 4D-halo (transport_step_4d — one pad_halo_4d per exchange for all
        # levels), and the wind reconstruction is 4D (d2a2c_vect_4d — one
        # pad_halo_vector_4d).  Certified replicated-vs-scattered equivalent by
        # tests/distributed/test_cube_face_scatter_mpi.py.  SPMD face-sharding is
        # still fail-closed (the 4D halos dispatch to explicit SPMD exchanges,
        # but that path is unvalidated here — #811 SPMD follow-up).  Static-config
        # check at trace time, so the raise is a hard compile-time refusal.
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
        )
        if self._flux_form_scatter_blocked(
                get_halo_backend(), get_mpi_topology(), get_spmd_mesh(),
                state_out.u_d.data.shape[0]):
            raise NotImplementedError(
                "moisture_flux_form is not supported under SPMD face-sharding: "
                "the 4D-halo transport + wind reconstruction dispatch to explicit "
                "SPMD exchanges but that path is unvalidated. Run MPI "
                "(single-rank / replicated / face-scatter, all supported) or "
                "single-device SPMD (#811 SPMD follow-up).")

        cdgrid = self.cdgrid
        sigma = self.sigma_coord
        tnames = list(state_out.tracers)

        # Contravariant transport winds from the STEP-INPUT corner D-grid winds
        # (the SW split transports mass with the step-input winds): corner ->
        # edge (0.5-average, as in the del-n path) -> d2a2c_vect_4d (ONE vector
        # halo for ALL levels, so the reconstruction is MPI face-scatter safe —
        # no vmap(pad_halo_vector); #811).
        u_d = state_in.u_d.data
        v_d = state_in.v_d.data
        u_edge = 0.5 * (u_d[:, :-1, :, :] + u_d[:, 1:, :, :])   # (6,n,n+1,nlev)
        v_edge = 0.5 * (v_d[:, :, :-1, :] + v_d[:, :, 1:, :])   # (6,n+1,n,nlev)

        _o = d2a2c_vect_4d(u_edge, v_edge, cdgrid)
        ut, vt = _o[4], _o[5]                                  # ut, vt contravar

        # Layer mass on the dynamics' post-step p_s (what the moisture lives
        # on).  ``layer_thickness_dp`` is polymorphic over the sigma / hybrid
        # coordinate (matches how the dycore forms δp).
        p_s1 = jnp.clip(
            state_out.p_s.data, self.config.p_floor, self.config.p_ceil)
        delp = sigma.layer_thickness_dp(p_s1)                  # (6,n,n,nlev)
        _dt = delp.dtype

        # Floor at 0 BEFORE transport (see docstring): a monotone flux-form step
        # keeps a nonnegative input nonnegative, so the final q>=0 clip becomes a
        # true no-op and the conserved target below is a real (nonnegative) mass
        # — not corrupted by unphysical RK3 negatives that the later clip would
        # otherwise silently remove, breaking conservation.
        q = jnp.maximum(
            jnp.stack([state_out.tracers[k].data for k in tnames], axis=-1),
            0.0).astype(_dt)                                   # (6,n,n,nlev,ntr)

        q_new, _delp_star = flux_form_tracer_step(
            q, delp, ut.astype(_dt), vt.astype(_dt), dt, cdgrid)

        # Global per-tracer mass fixer: reconcile q_new (conservative on the
        # discarded δp★ ≈ delp) onto the dynamics' delp so ∫area·delp·q is
        # preserved to fp-accuracy on the model grid.  scale ≈ 1 (a tiny global
        # correction — not strictly monotone, but nonnegativity-preserving).
        acc = conservation_accumulator()
        w = (cdgrid.base.area[..., None, None].astype(acc)
             * delp[..., None].astype(acc))                    # (6,n,n,nlev,1)
        tgt = jnp.sum(w * q.astype(acc), axis=(0, 1, 2, 3))     # (ntr,)
        cur = jnp.sum(w * q_new.astype(acc), axis=(0, 1, 2, 3))
        # Under MPI face-scatter tgt/cur are owned-face PARTIALS; allreduce the
        # (numerator, denominator) pair to the true global tracer masses BEFORE
        # the per-tracer scale — else each rank divides by its own partial and
        # the rescales diverge, silently breaking global conservation.  Identity
        # on single-rank / replicated / SPMD, keyed off cdgrid.base.area (#811).
        # differentiable_broadcast=True: this scale rescales EVERY owned face
        # (q_fixed = q_new * scale), so the reduction's VJP must allreduce the
        # cotangent — else the cross-rank gradient through the shared scale is
        # dropped (the same uniform ~1e-3 cotangent leak the flux_form_tracer_step
        # rescale had; the full-step gradient gate exposes THIS second one, #811).
        tgt, cur = global_face_sum_if_scattered(
            jnp.stack([tgt, cur], axis=0), cdgrid.base.area,
            differentiable_broadcast=True)
        scale = (tgt / jnp.maximum(cur, jnp.asarray(1e-30, acc))).astype(_dt)
        q_fixed = q_new * scale

        return state_out._replace(tracers={
            k: state_out.tracers[k].replace(
                data=q_fixed[..., i].astype(state_out.tracers[k].data.dtype))
            for i, k in enumerate(tnames)
        })

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_fv3(
        self,
        state: FV3HydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass=None,
        phys_state=None,
    ) -> tuple:
        """Advance one time step with D-grid prognostic winds.

        Returns ``(state_new, phys_state_out)``.  ``phys_state_out`` is
        the input carry when no carry/physics is active; see
        :meth:`step` for the per-RK-stage carry semantics.
        """
        state = cast_pytree(state, None, "compute")
        cdgrid = self.cdgrid

        def _eval_physics(s_cc):
            """Physics with the step-input carry (4-arg contract when
            a carry is threaded; legacy 3-arg call otherwise)."""
            if phys_state is not None:
                return physics_fn(s_cc, self.grid, self.sigma_coord,
                                  phys_state)
            return physics_fn(s_cc, self.grid, self.sigma_coord)

        def tendency_fn(s):
            phys_cc = None
            if physics_fn is not None:
                # Iter-65: pass cc physics via physics_tendency_cc (rides iter-64 batched corner interp)
                s_cc = fv3_to_hydrostatic(s, cdgrid)
                _phys_result = _eval_physics(s_cc)
                phys_cc = _phys_result[0] if type(_phys_result) is tuple else _phys_result

            # iter-189: dt_actual for corner-div adaptive cap
            tend = fv3_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, cdgrid,
                self.config,
                physics_tendency=None,
                physics_tendency_cc=phys_cc,
                dt_actual=dt,
            )
            # Return FV3HydrostaticState-shaped pytree for integrator tree_map.
            # Tracer tendencies ride as a matching tracers dict (same keys + Field
            # aux as the input state) so the SSP-RK3 ``state + dt·tendency``
            # tree_map advances q_v/q_c/q_r alongside the prognostic fields.
            _tracer_tend = None
            if s.tracers is not None and tend.tracer_tendencies is not None:
                _tracer_tend = {
                    k: s.tracers[k].replace(data=tend.tracer_tendencies[k].data)
                    for k in s.tracers
                }
            return FV3HydrostaticState(
                u_d=s.u_d.replace(data=tend.du_d_dt.data),
                v_d=s.v_d.replace(data=tend.dv_d_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=_tracer_tend,
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Operator-split upper-atmosphere Rayleigh sponge — opt-in
        # path (``config.sponge_implicit = True``).  Replaces the
        # explicit ``-α u`` tendency contribution that is skipped
        # inside ``fv3_hydrostatic_tendencies`` when this flag is
        # set, with the analytic multiplicative damping
        # ``u_new = u_old · exp(−sponge_rate · dt)``.  Exact integrator
        # of the linear ODE ``du/dt = −α u``, unconditionally stable
        # for any ``α · dt > 0`` — takes the sponge out of the
        # explicit-CFL budget so issue-#273 throughput work can
        # push ``dt`` upward without re-tuning ``sponge_tau_sec``.
        #
        # Default ``sponge_implicit = False`` keeps the legacy
        # tendency-form path bit-exact (see the matching branch in
        # ``fv3_hydrostatic_tendencies``).
        if (self.config.sponge_tau_sec > 0
                and self.config.sponge_sigma > 0
                and self.config.sponge_implicit):
            sigma_full = self.sigma_coord.sigma_full
            sponge_frac = jnp.clip(
                (self.config.sponge_sigma - sigma_full)
                / self.config.sponge_sigma,
                0.0, 1.0,
            )
            sponge_rate = sponge_frac**2 / self.config.sponge_tau_sec
            damp_factor = jnp.exp(-sponge_rate * dt)
            state_new = state_new._replace(
                u_d=state_new.u_d.replace(
                    data=state_new.u_d.data * damp_factor,
                ),
                v_d=state_new.v_d.replace(
                    data=state_new.v_d.data * damp_factor,
                ),
            )

        # Synchronize D-grid boundary corners across cubed-sphere faces
        state_new = self._sync_dgrid_boundary(state_new)

        # FV3_3D iter 12: post-step del-n vorticity damp (SW backbone, FV3 sw_core.F90:1948-1999)
        if self.config.damp_v > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                fv3_del6_vorticity_damping,
            )
            # Convert C-D corners (6, n+1, n+1, nlev) to FV3 normal D-grid per level
            u_corner = state_new.u_d.data
            v_corner = state_new.v_d.data
            u_normal = 0.5 * (u_corner[:, :-1, :, :] + u_corner[:, 1:, :, :])
            v_normal = 0.5 * (v_corner[:, :, :-1, :] + v_corner[:, :, 1:, :])

            # FV3 damp coeff: da_min_c = global min B-grid corner area
            da_min_c = jnp.min(self.cdgrid.area_corner)
            damp_step = (self.config.damp_v * da_min_c) ** (
                self.config.nord_v + 1
            )

            # FV3_3D iter-1045: ``fv3_del6_vorticity_damping`` is now
            # 4D-native (3D static metrics broadcast via ``[..., None]``;
            # halo dispatched to ``pad_halo_4d``).  Direct call avoids
            # ``jax.vmap`` around ``pad_halo`` under MPI.
            du_normal, dv_normal = fv3_del6_vorticity_damping(
                u_normal, v_normal, damp=damp_step,
                nord=self.config.nord_v, cdgrid=self.cdgrid,
            )

            # FV3_3D iter 443 (PE mirror of NH 442): sponge boost at k=0,k=1 (NOT k=2)
            if self.config.use_fv3_sponge_damp_v:
                from legoesm.core.fv3_sponge_boost import (
                    apply_top_sponge_field_scale as _shared_pscale_v,
                )
                du_normal = _shared_pscale_v(
                    du_normal, self.config.damp_v,
                    self.config.nord_v, factor=0.5,
                    d2_bg=self.config.corner_div_damp_d2_bg,
                    d2_bg_k1=self.config.corner_div_damp_d2_bg_k1,
                    d2_bg_k2=self.config.corner_div_damp_d2_bg_k2,
                    apply_at_k2=False,
                )
                dv_normal = _shared_pscale_v(
                    dv_normal, self.config.damp_v,
                    self.config.nord_v, factor=0.5,
                    d2_bg=self.config.corner_div_damp_d2_bg,
                    d2_bg_k1=self.config.corner_div_damp_d2_bg_k1,
                    d2_bg_k2=self.config.corner_div_damp_d2_bg_k2,
                    apply_at_k2=False,
                )

            # Project FV3 normal → corners by mode='edge' padding + avg
            # FV3_3D iter 370: opt cross-face halo via pad_halo_4d (duogrid-aware)
            # iter-1046: ``du_normal``/``dv_normal`` are non-square
            # ``(6, n, n+1, nlev)``/``(6, n+1, n, nlev)``; the cubed-
            # sphere ``pad_halo_4d`` MPI path assumes square ``(6, n, n,
            # nlev)`` and crashes on non-square data without duogrid.
            # Fall through to ``mode='edge'`` only when MPI is active
            # AND duogrid is off (the iter-370 regression test depends
            # on the non-square local diff being non-zero, so we keep
            # the call on the local backend).
            # FV3_3D iter-1083: see NH counterpart for the local-vs-
            # MPI dispatch via pad_halo_dgrid_vector_4d{,_replicated_mpi}.
            from legoesm.grids.halo import get_halo_backend as _ghb_pe
            if self.config.use_fv3_cross_face_du_proj:
                if _ghb_pe() == "mpi":
                    from legoesm.grids.halo import get_mpi_topology
                    from legoesm.grids.dgrid_halo import (
                        pad_halo_dgrid_vector_4d_replicated_mpi,
                    )
                    du_full, dv_full = pad_halo_dgrid_vector_4d_replicated_mpi(
                        du_normal, dv_normal, get_mpi_topology(),
                    )
                else:
                    from legoesm.grids.dgrid_halo import (
                        pad_halo_dgrid_vector_4d,
                    )
                    du_full, dv_full = pad_halo_dgrid_vector_4d(
                        du_normal, dv_normal,
                    )
                du_pad = du_full[:, :, 1:-1, :]
                dv_pad = dv_full[:, 1:-1, :, :]
            else:
                du_pad = jnp.pad(
                    du_normal, [(0, 0), (1, 1), (0, 0), (0, 0)],
                    mode="edge",
                )
                dv_pad = jnp.pad(
                    dv_normal, [(0, 0), (0, 0), (1, 1), (0, 0)],
                    mode="edge",
                )
            du_corner = 0.5 * (du_pad[:, :-1, :, :] + du_pad[:, 1:, :, :])
            dv_corner = 0.5 * (dv_pad[:, :, :-1, :] + dv_pad[:, :, 1:, :])

            # FV3_3D iter 208: KE→heat d_con for damp_v (FV3 sw_core.F90:1953-1990)
            # ΔKE = u·du + 0.5du² + v·dv + 0.5dv²; ΔT = -coeff*ΔKE/c_pd at corners → centres
            if self.config.damp_v_d_con > 0.0:
                from legoesm.core.operators_cdgrid import (
                    interp_corner_to_center,
                )
                if self.config.use_fv3_metric_aware_d_con:
                    # FV3_3D iter 338/344: metric-aware form (cosa_cell/rsin2_cell)
                    ub_s = du_normal[:, :, :-1, :]   # (6,n,n,nlev)
                    ub_n = du_normal[:, :, 1:, :]
                    vb_w = dv_normal[:, :-1, :, :]
                    vb_e = dv_normal[:, 1:, :, :]
                    u_s = u_normal[:, :, :-1, :]
                    u_n = u_normal[:, :, 1:, :]
                    v_w = v_normal[:, :-1, :, :]
                    v_e = v_normal[:, 1:, :, :]
                    gy_s = u_s * ub_s
                    gy_n = u_n * ub_n
                    gx_w = v_w * vb_w
                    gx_e = v_e * vb_e
                    u2 = u_s + u_n
                    du2 = ub_s + ub_n
                    v2 = v_w + v_e
                    dv2 = vb_w + vb_e
                    cosa_b = self.cdgrid.cosa_cell[..., None]
                    rsin2_b = self.cdgrid.rsin2_cell[..., None]

                    dKE_cc_metric = 0.25 * rsin2_b * (
                        ub_s ** 2 + ub_n ** 2 + vb_w ** 2 + vb_e ** 2
                        + 2.0 * (gy_s + gy_n + gx_w + gx_e)
                        - cosa_b * (u2 * dv2 + v2 * du2 + du2 * dv2)
                    )
                    dT = -self.config.damp_v_d_con * dKE_cc_metric / constants.c_pd
                else:
                    dKE_corner = (
                        u_corner * du_corner + 0.5 * du_corner ** 2
                        + v_corner * dv_corner + 0.5 * dv_corner ** 2
                    )
                    dKE_cc = interp_corner_to_center(dKE_corner)
                    dT = -self.config.damp_v_d_con * dKE_cc / constants.c_pd
                # FV3_3D iter 433 (PE mirror of NH 431): sponge-zero d_con top N levels
                if self.config.d_con_top_zero_levels > 0:
                    _nlev_zv = dT.shape[-1]
                    _k_idx_zv = jnp.arange(_nlev_zv)
                    _d_con_mask_v = jnp.where(
                        _k_idx_zv < self.config.d_con_top_zero_levels,
                        0.0, 1.0,
                    )
                    dT = dT * _d_con_mask_v[None, None, None, :]
                # FV3_3D iter 218/219: per-step |dT| cap (FV3 dyn_core.F90:1764-1776)
                # PE: skip cap for k<2 (top sponge); cap rest to dt*delt_max
                if self.config.delt_max > 0.0:
                    nlev = dT.shape[-1]
                    k_idx = jnp.arange(nlev)
                    cap_per_level = jnp.where(
                        k_idx < 2, jnp.inf,
                        dt * self.config.delt_max,
                    )
                    cap_b = cap_per_level[None, None, None, :]
                    dT = jnp.clip(dT, -cap_b, cap_b)
                state_new = state_new._replace(
                    u_d=state_new.u_d.replace(data=u_corner + du_corner),
                    v_d=state_new.v_d.replace(data=v_corner + dv_corner),
                    T=state_new.T.replace(data=state_new.T.data + dT),
                )
            else:
                state_new = state_new._replace(
                    u_d=state_new.u_d.replace(data=u_corner + du_corner),
                    v_d=state_new.v_d.replace(data=v_corner + dv_corner),
                )

        # Implicit gravity wave damping — post-step Laplacian operation on p_s.
        #
        # Default path: explicit forward-Euler diffusion
        # ``p_s ← p_s + α dt ∇²p_s``.  Conditionally stable at
        # ``α dt / dx² < 0.5``.
        #
        # ``implicit_grav_wave_use_pcg = True`` switches to a
        # Phase-3 implicit Helmholtz solve
        # ``(I − α dt ∇²) p_s_new = p_s_explicit`` using
        # ``cg_helmholtz_solve``.  The FV-adjoint-symmetric
        # ``cdgrid_scalar_laplacian`` (built on a nearest-copy halo
        # exchange) makes the operator M-symmetric and negative
        # semi-definite under the area-weighted inner product, and a
        # ``M^{1/2}·A·M^{-1/2}`` shim casts that to a Euclidean-SPD
        # operator so ``jax.scipy.sparse.linalg.cg`` is well-defined.
        # CG also installs an implicit-function-theorem VJP, so
        # ``jax.grad`` flows cleanly through the solve.
        if self.config.implicit_grav_wave_damping > 0:
            alpha = self.config.implicit_grav_wave_damping
            if self.config.implicit_grav_wave_use_pcg:
                from legoesm.atmosphere.dynamics.gcm.semi_implicit_cdgrid import (
                    cg_helmholtz_solve,
                )
                # Production tolerance 1e-10 — CG reaches it in ~10
                # iterations at α dt / dx² ≤ 5, two orders of
                # magnitude tighter than the Phase-2 Richardson
                # tol=1e-6.  When the solver fails to reach
                # tolerance within ``maxiter`` (e.g. at extreme
                # ``α dt / dx² ≫ 100`` or with ill-conditioned
                # metrics), fall back JAX-safely to *no damping
                # for this step* — applying explicit forward-Euler
                # at the same coefficient would violate its CFL
                # bound and amplify the instability the implicit
                # path was meant to suppress.  A ``RuntimeWarning``
                # is surfaced via ``jax.debug.callback`` so the
                # user can lower ``dt`` or ``α``.
                _pcg_tol = 1.0e-10
                _p_s_implicit, _rel_res = cg_helmholtz_solve(
                    state_new.p_s.data,
                    coeff=alpha * dt,
                    cdgrid=self.cdgrid,
                    tol=_pcg_tol,
                    maxiter=200,
                    return_residual=True,
                )
                _converged = _rel_res <= _pcg_tol
                def _maybe_audit(rel_res):
                    def _warn(rel):
                        import warnings
                        warnings.warn(
                            f"implicit_grav_wave CG failed to reach "
                            f"tol={_pcg_tol:.0e} "
                            f"(rel_res={float(rel):.3e}); leaving "
                            f"p_s unchanged for this step.  Reduce "
                            f"``implicit_grav_wave_damping``, reduce "
                            f"``dt``, or raise ``maxiter`` in "
                            f"``cg_helmholtz_solve``.",
                            RuntimeWarning,
                        )
                    jax.debug.callback(_warn, rel_res)
                jax.lax.cond(
                    _converged,
                    lambda _: None,
                    _maybe_audit,
                    _rel_res,
                )
                p_s_damped = jnp.where(
                    _converged, _p_s_implicit, state_new.p_s.data,
                )
            else:
                lap_ps = laplacian_compact(state_new.p_s.data, self.grid)
                p_s_damped = state_new.p_s.data + alpha * dt * lap_ps
            p_s_damped = jnp.maximum(p_s_damped, self.config.p_floor)
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_damped),
            )

        # Iter-2: raw-array fix_ps_mass skips fv3_to_hydrostatic roundtrip
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                # Anchor target threaded from step() (concrete cached t=0
                # mass on the eager path; per-call value under an outer
                # jit).  Direct callers of _step_fv3 / step_cell_centre
                # pass None: honour a concrete set_target_mass(...) /
                # cached snapshot first (pre-fix behaviour), then fall
                # back to the pre-step mass, which telescopes to the
                # initial mass exactly like the non-anchor branch below.
                # Tracer-cached values are never read (the step() fix
                # never stores them; defensive guard regardless).
                if target_mass is not None:
                    _target = target_mass
                elif (self._target_mass is not None
                        and not isinstance(self._target_mass,
                                           jax.core.Tracer)):
                    _target = self._target_mass
                else:
                    _target = global_integral(state.p_s, self.grid)
                p_s_fixed = fix_ps_mass_target(
                    state_new.p_s.data, _target, self.grid,
                )
            else:
                p_s_fixed = fix_ps_mass(
                    state_new.p_s.data, state.p_s.data, self.grid,
                )
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_fixed),
            )

        # FV3_3D iter 449 (PE mirror of NH 448): Ray_fast. pfull(k) = (A_full+B_full)*p_ref
        if self.config.rf_tau_days > 0.0:
            from legoesm.core.fv3_rayleigh_fast import (
                compute_rff_profile,
            )
            _coord = self.sigma_coord
            _pfull_pe = (_coord.A_full + _coord.B_full) * _coord.p_ref
            _ptop_pe = _pfull_pe[0]
            _rff_pe = compute_rff_profile(
                _pfull_pe, ptop=_ptop_pe,
                rf_cutoff=self.config.rf_cutoff_pa,
                tau_days=self.config.rf_tau_days,
                dt=dt,
            )
            _rff_pe_b = _rff_pe[None, None, None, :]
            state_new = state_new._replace(
                u_d=state_new.u_d.replace(
                    data=state_new.u_d.data * _rff_pe_b,
                ),
                v_d=state_new.v_d.replace(
                    data=state_new.v_d.data * _rff_pe_b,
                ),
            )

        # #771: mass-conserving flux-form HORIZONTAL moisture transport as a
        # post-RK3 substep — replaces the in-RK3 advective -(u·∇q) (skipped in
        # the tendency when moisture_flux_form is set).  Runs on the final
        # (post-mass-fixer) p_s and the STEP-INPUT winds; it co-transports δp,
        # then a per-tracer global fixer reconciles onto the dynamics' δp so
        # column water conserves.  Being monotone, it makes the q>=0 clip below
        # a no-op safety net rather than a one-signed moisture source.
        if self.config.moisture_flux_form and state_new.tracers is not None:
            state_new = self._flux_form_moisture_substep(state, state_new, dt)

        # Prognostic tracer floor: advective tracer transport is not
        # positive-definite, so clamp q >= 0 AFTER the RK update (the physics
        # adapters clip their INPUT; this is the prognostic floor the moist
        # forcing's contract relies on).  Tracers are untouched by the
        # wind/sponge/vorticity-damping post-steps above.
        if state_new.tracers is not None:
            state_new = state_new._replace(tracers={
                _k: _f.replace(data=jnp.maximum(_f.data, 0.0))
                for _k, _f in state_new.tracers.items()
            })

        state_out = cast_pytree(state_new, None, "storage")

        # Operator-split physics carry (issue #413): one extra physics
        # evaluation on the POST-STEP state produces the carry-out
        # (tendencies discarded) — the prognostic fields advance exactly
        # once per dt and the carry-out is consistent with the returned
        # state.  Threading the carry through the RK combination instead
        # would be wrong: stateful kernels return REPLACEMENT values
        # (e.g. the implicit TKE solve), which RK stage weights would
        # corrupt.  Python-level gate: with no carry this block is never
        # traced and the step is byte-identical to the legacy path.
        phys_state_out = phys_state
        if physics_fn is not None and phys_state is not None:
            _pr = _eval_physics(fv3_to_hydrostatic(state_out, cdgrid))
            if type(_pr) is tuple and len(_pr) > 1:
                phys_state_out = _pr[1]

        return state_out, phys_state_out

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_cell_centre(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass=None,
        phys_state=None,
    ) -> tuple:
        """Cell-centre wrapper: cc winds → D-grid corners (entry); back to cc (exit).

        Returns ``(state_new, phys_state_out)`` — internal contract;
        the public :meth:`step_cell_centre` unpacks it.
        """
        # cc → D-grid interp for (u, v).
        # fv3_faithful (iter-14): the winds are a VECTOR, so the cc→corner
        # interp must rotate face-local components across panel seams.  The
        # previous ``interp_center_to_corner`` on the stacked (u, v) treated
        # them as two SCALARS and blended seam-crossing components WITHOUT
        # rotation, producing D-grid winds that were ~167×/83× rougher at
        # panel edges than the interior → a ~229× rougher relative vorticity
        # → the cube discrete-balance v-imprint (steady jet developed 3.4 m/s
        # spurious v).  ``center_to_dgrid_vector`` is the rotation-aware
        # (vector) interpolation and is the inverse of the vector-aware exit
        # (``dgrid_to_center_vector`` in ``fv3_to_hydrostatic``); with it the
        # wind/vorticity edge-roughness drops to ~1× (interior level).
        _u_in = state.u.data
        _v_in = state.v.data
        u_d, v_d = center_to_dgrid_vector(_u_in, _v_in, self.cdgrid)
        fv3_state = FV3HydrostaticState(
            u_d=state.u.replace(data=u_d, name="u_d"),
            v_d=state.v.replace(data=v_d, name="v_d"),
            T=state.T,
            p_s=state.p_s,
            phis=state.phis,
            tracers=state.tracers,
        )
        fv3_new, phys_state_out = self._step_fv3(
            fv3_state, dt, physics_fn=physics_fn, target_mass=target_mass,
            phys_state=phys_state)
        return fv3_to_hydrostatic(fv3_new, self.cdgrid), phys_state_out

    def step_cell_centre(self, state, dt, physics_fn=None,
                         target_mass=None, phys_state=None):
        """Public cell-centre step: returns the new state only (legacy
        contract); the updated physics carry is stashed on
        ``self._phys_state`` (issue #413, mirrors :meth:`step`)."""
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="CDGrid step_cell_centre()")
        state_new, self._phys_state = self._step_cell_centre(
            state, dt, physics_fn=physics_fn, target_mass=target_mass,
            phys_state=phys_state)
        return state_new

    def step_with_physics(self, state, dt, physics_fn=None,
                          phys_state=None):
        return self.step(state, dt, physics_fn=physics_fn,
                         phys_state=phys_state)


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
    """Hydrostatic tendencies. HydrostaticState → cc; FV3HydrostaticState → D-grid."""
    if isinstance(state, FV3HydrostaticState):
        return fv3_hydrostatic_tendencies(state, grid, sigma_coord, cdgrid, config, physics_tendency)

    # HydrostaticState path: convert cell-centre -> D-grid.
    # fv3_faithful (iter-14): the winds are a VECTOR — use the rotation-aware
    # ``center_to_dgrid_vector`` (matching ``_step_cell_centre``), NOT a scalar
    # ``interp_center_to_corner`` on the stacked (u, v), which blends face-local
    # components across panel seams without rotation and re-injects the
    # cube-edge vorticity imprint.  Keeps ``step`` and ``tendencies`` consistent.
    u_d, v_d = center_to_dgrid_vector(state.u.data, state.v.data, cdgrid)
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
    _duv_cc_flat = interp_corner_to_center(
        _duv_d.reshape(_nd_face, _nd_i, _nd_j, _nd_lev * 2),
    )
    _duv_cc = _duv_cc_flat.reshape(
        _duv_cc_flat.shape[0], _duv_cc_flat.shape[1], _duv_cc_flat.shape[2],
        _nd_lev, 2,
    )
    du_cc = _duv_cc[..., 0]
    dv_cc = _duv_cc[..., 1]

    dims_3d = ("face", "x", "y", "level")
    return HydrostaticTendencies(
        du_dt=Field(data=du_cc, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_cc, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=fv3_tend.dT_dt,
        dp_s_dt=fv3_tend.dp_s_dt,
        dphis_dt=fv3_tend.dphis_dt,
        # Tracers are cell-centre scalars (no corner→centre conversion needed),
        # so the FV3 advective tracer tendencies pass through unchanged — codex
        # caught this being dropped on the cell-centre tendencies API.
        tracer_tendencies=fv3_tend.tracer_tendencies,
    )


def hydrostatic_to_fv3(
    state: HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> FV3HydrostaticState:
    """Convert cc HydrostaticState to FV3 D-grid (vector-aware halo + 4-pt avg).

    Scalar interp leaves spurious cube-edge divergence (~5e-5 s^-1) — caused C24 BCW
    blow-up. Mirrors SW _sw_to_cdgrid path.
    """

    _u_in = state.u.data
    _v_in = state.v.data
    base = cdgrid.base
    # fv3_faithful (iter-14): thread the duogrid through the vector halo so this
    # cc→D-grid entry lift matches the duogrid-aware center_to_dgrid_vector used
    # in _step_cell_centre / tendencies() (suppress interp_offsets when duogrid
    # is active).  No-op when duogrid is off (the matrix), where base.duogrid is
    # None and interp_offsets stays base.halo_interp_offsets.
    _ho_dg = base.duogrid
    _ho_offs = None if _ho_dg is not None else base.halo_interp_offsets
    if _u_in.ndim == 4:
        u_pad, v_pad = pad_halo_vector_4d(
            _u_in, _v_in,
            base.cos_angle, base.sin_angle,
            base.cos_angle_padded, base.sin_angle_padded,
            interp_offsets=_ho_offs,
            duogrid=_ho_dg,
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


def make_fv3_faithful_pe_config(**overrides) -> CDGridPrimitiveEquationConfig:
    """FV3_3D iter 392: factory for FV3-faithful PE config.

    Pair with ``use_duogrid=True`` so ``iter-370 use_fv3_cross_face_du_proj``
    has effect (iter-384).  Enables iter-14/338/370/433/436/437/451/459/443.

    FV3-fidelity flags:

    Enabled by default:

    - ``use_fv3_a2b_zeta_corner``: FV3 4th-order A→B ζ corner interp
      (iter-14, mirror of SW sw_core.F90:a2b_ord4).
    - ``use_fv3_metric_aware_d_con``: metric-aware d_con form
      (iter-338, cosa_s/rsin2 form at all 5 d_con sites).

    Disabled by default (opt-in via ``overrides``):

    - ``use_fv3_cross_face_du_proj``: cross-face halo for damp_v
      wind projection (iter-370).  Disabled as of iter-1072
      (non-square halo silent corruption — see NH factory docstring
      + FV3_3D.md iter-1072).

    Plus production knobs: ``d_con_top_zero_levels``, ``delt_max``,
    ``nord_v``, ``corner_div_damp_nord``, ``corner_div_damp_d4_bg``,
    ``heat_source_del2_iters``, ``use_fv3_sponge_damp_v``.

    Pass ``overrides`` kwargs to override any default.
    """
    defaults = dict(
        use_fv3_a2b_zeta_corner=True,
        use_fv3_metric_aware_d_con=True,
        # FV3_3D iter-1079: enabled.  See NH factory note —
        # pad_halo_dgrid_vector_4d (iter-1078) gives all 24 directed
        # edges bit-for-bit FV3-faithful.
        use_fv3_cross_face_du_proj=True,
        d_con_top_zero_levels=2,
        delt_max=1.0,
        nord_v=1,
        corner_div_damp_nord=1,
        corner_div_damp_d4_bg=0.16,
        heat_source_del2_iters=2,   # FV3 nf_ke at nord=1
        # iter-452: d2_bg_k1/k2 left at 0.0 — see NH factory note.
        use_fv3_sponge_damp_v=True,
    )
    defaults.update(overrides)
    return CDGridPrimitiveEquationConfig(**defaults)


def make_legoesm_pe_min_edge_config(**overrides) -> CDGridPrimitiveEquationConfig:
    """FV3_3D iter 468 (PE mirror of NH iter-467): min-edge factory.

    Turns OFF 3 flags that hurt NH θ′ ratio (iter-465/466): metric_aware_d_con,
    heat_source_del2, d_con_top_zero. NH-tuned; PE sweep not done — empirical only.
    """
    edge_min_overrides = dict(
        use_fv3_metric_aware_d_con=False,
        heat_source_del2_iters=0,
        d_con_top_zero_levels=0,
    )
    edge_min_overrides.update(overrides)
    return make_fv3_faithful_pe_config(**edge_min_overrides)


def make_legoesm_pe_min_edge_aggressive_config(
    **overrides
) -> CDGridPrimitiveEquationConfig:
    """FV3_3D iter 484: PE mirror of NH iter-483 aggressive factory.

    Stacks iter-466 hurting-flag drops + iter-481 corner_div boost
    (d2_bg=5e-2).  iter-469/470 found PE T/u_d insensitive to most
    factory flags at C8 — this aggressive config is unlikely to help
    much on PE compared to NH, but provided for API symmetry with
    NH iter-483.  Trade-off: over-damps physical waves.
    """
    aggressive_overrides = dict(
        use_fv3_metric_aware_d_con=False,
        heat_source_del2_iters=0,
        d_con_top_zero_levels=0,
        corner_div_damp_d2_bg=5e-2,
    )
    aggressive_overrides.update(overrides)
    return make_fv3_faithful_pe_config(**aggressive_overrides)
