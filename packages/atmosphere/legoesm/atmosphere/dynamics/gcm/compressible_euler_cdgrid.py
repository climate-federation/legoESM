"""FV3-inspired C-D grid Non-Hydrostatic Compressible Euler on cubed-sphere.

State stored at cell centres for physics compatibility (faithful FV3 NH would store on D-grid edges).
Lin (2004).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.core.operators_3d import (
    divergence_3d,
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
    laplacian_compact_3d,
    vertical_advection_height,
)
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector,
    dgrid_to_cgrid,
    dgrid_to_center_vector,
    dgrid_vorticity,
    cgrid_mass_flux_divergence,
    cgrid_divergence,
    arakawa_lamb_gradient,
    interp_center_to_corner,
    interp_corner_to_center,
    laplacian_dgrid,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.integration import (
    IntegrationMixin,
    refuse_unthreaded_stateful_physics,
)
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    compute_exner_perturbation,
    sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
    CompressibleEulerConfig,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.cfl import estimate_min_dx_cubed_sphere
from legoesm.core.conservation import (
    compute_nh_dry_mass,
    fix_mass_nonhydrostatic,
)
from legoesm.grids.cubed_sphere import apply_small_earth_scaling
from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d_module
from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
from legoesm.parallel.halo_exchange import packed_pad_halo_mpi_4d
from legoesm import constants
import logging


# --- FV3 dynamical-core constants (GFDL FV3 sw_core.F90 / fv_arrays.F90) ---
# ``cnst_0p20`` is FV3's fixed adaptive-divergence-damping (Smagorinsky)
# coefficient ceiling: damp = da_min * max(d2_bg, min(0.20, dddmp*|div|*dt)).
# It is a published FV3 constant, NOT a tunable scheme knob (the tunable knobs
# are the ``*_d2_bg`` / ``*_dddmp`` config fields multiplied against it).
_FV3_CNST_0P20: float = 0.20


class CDGridCompressibleEulerConfig(NamedTuple):
    """Config for C-D grid NH CE. Lin 2004, Putman & Lin 2007. No Hollingsworth-Kallberg."""
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    sponge_width: float = 10000.0
    sponge_coeff: float = 0.05
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True
    semi_implicit_acoustic: bool = False
    outer_integrator: str = "ssp_rk3"
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False
    acoustic_off_centering: float = 0.0   # Off-centering beta; 0=centered, 0.1 long runs
    # FV3_3D iter 168: corner-div damping (mirror of PE iter-16/18; FV3 sw_core.F90:1641-1822 d_sw5)
    corner_div_damp_d2_bg: float = 0.0
    corner_div_damp_dddmp: float = 0.20
    corner_div_damp_d4_bg: float = 0.0
    corner_div_damp_nord: int = 0
    heat_source_del2_iters: int = 0
        # FV3_3D iter 457: del-2 smoothing of _d_con_sum (FV3 dyn_core.F90:1755-1756).
        # nf_ke=min(3,nord+1); FV3 nord=1 → nf_ke=2.
    heat_source_del2_coeff: float = 0.20
        # FV3 cnst_0p20.
    rf_tau_days: float = 0.0
        # FV3_3D iter 448: Ray_fast (FV3 dyn_core.F90:2922).
        # rff(k) = 1/(1 + dt/tau0*sin²(...)²); u,v,w *= rff for pfull<rf_cutoff_pa.
        # tau in DAYS. Typical 5-15 days.
    rf_cutoff_pa: float = 3000.0
        # FV3 default rf_cutoff=3.0e2 Pa = 30 hPa.
    use_fv3_sponge_damp_v: bool = False
        # FV3_3D iter 442: sponge boost of NH damp_v at k=0,1 (NOT k=2). FV3 dyn_core.F90:786-787,796-797.
        # damp_vt = 0.5*d2_divg. Linear damp^(nord_v+1) scaling.
    use_fv3_sponge_damp_w: bool = False
        # FV3_3D iter 441: sponge boost of NH damp_w (FV3 dyn_core.F90:782,793,803). damp_w = d2_divg.
        # (boosted/damp_w)^(nord_w+1) post-step scaling. Levels: k=0 (d2_bg_k1>0), k=1 (k2>0.01), k=2 (k2>0.05).
    w_safety_cap: float = 0.0
        # FV3_3D iter 584: optional |w| safety cap (FV3 fv_mapz.F90:51 w_max=90).
        # 0.0 = disabled (default).  When > 0, clip w_half to [-w_safety_cap, +w_safety_cap]
        # post-step.  Note: FV3's w_limiter cascades excess to neighbor level for
        # momentum conservation; legoESM applies simple clip on half-level w because
        # the Lagrangian-to-Eulerian remap context (where FV3 applies it) is absent here.
        # Use for runaway-w protection in stress tests; not part of FV3-faithful path.
    w_safety_cap_min: float = 0.0
        # Negative cap (downward).  0.0 means use -w_safety_cap symmetrically.
        # Set explicitly to override (FV3 uses asymmetric: w_max=90, w_min=-60).
    corner_div_damp_d2_bg_k1: float = 0.0
        # NH mirror of PE iter-438 (FV3 dyn_core.F90:780). FV3 namelist 4.0 not portable, see iter-452.
    corner_div_damp_d2_bg_k2: float = 0.0
        # NH mirror of PE iter-439 (FV3 dyn_core.F90:792,802). 0.01/0.05 thresholds absolute.
    corner_div_damp_fv3_vector_fill: bool = False
    corner_div_damp_dt_proxy: float = 10.0
    # NH outer dt 10s typical (PE uses 200); only matters when adaptive cap active
    # FV3_3D iter 169: post-step del-n vorticity damp (mirror of PE iter-12; FV3 sw_core.F90:1948-1999)
    damp_v: float = 0.0
    nord_v: int = 2
    # 0=del-2, 1=del-4, 2=del-6. FV3 default 2. Gated by damp_v>0.
    # FV3_3D iter 209 (NH mirror of PE 208): KE→heat for damp_v.
    # Δθ_p = -coeff*(u·du + 0.5du² + v·dv + 0.5dv²)/(c_pd*Π_ref). Π_ref from HeightCoordinate.
    damp_v_d_con: float = 0.0
    # FV3_3D iter 170: a2b_ord4 for ζ_corner only (mirror of PE iter-14)
    use_fv3_a2b_zeta_corner: bool = False
    # FV3_3D iter 171: cell-centre div damp (mirror of PE iter-5; FV3 sw_core.F90:1720)
    # damp = da_min_c * max(d2_bg, min(0.20, dddmp*|div|))
    div_damp_coeff: float = 0.0
    div_damp_dddmp: float = 0.0
    div_damp_d_con: float = 0.0
        # FV3_3D iter 224 (NH mirror of PE 223): KE→heat for cell-centre div_damp.
        # dθ_p/dt += -coeff*(u·du+v·dv)/(c_pd*Π_ref). Gated by div_damp_coeff>0.
    # FV3_3D iter 173: opt-in async halo for div damp A-L gradient (MPI only)
    use_async_halo: bool = False
    # FV3_3D iter 180: Smag A_h (mirror of PE iter 57-59); range ~0.1-0.4
    smagorinsky_cs: float = 0.0
    ah_d_con: float = 0.0
        # FV3_3D iter 226 (NH mirror of PE 225): KE→heat for A_h Laplacian; dθ_p ∝ u·du+v·dv.
        #
        #     dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
        #     dθ_p/dt += -ah_d_con * (dKE/dt) / (c_pd * Π_ref)
        #
        # at corners, projected to cell centres via
        # ``interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE ``A_h > 0``.
    # FV3-faithful post-step del-(2*(nord_w+1)) damping for vertical
    # velocity ``w`` (FV3_3D iter 193).  Faithful port of FV3
    # ``sw_core.F90:1080-1086`` (in ``d_sw1``)::
    #
    #     damp4 = (damp_w * da_min_c) ** (nord_w + 1)
    #     call del6_vt_flux(nord_w, ..., damp4, w, wk, fx2, fy2, ...)
    #     dw = (fx2[i,j] - fx2[i+1,j] + fy2[i,j] - fy2[i,j+1]) * rarea
    #     w += dw
    #
    # Reuses the SW backbone ``del6_vt_flux`` from
    # ``legoesm.core.fv3_del6_vt_flux``.  Applied ONCE per full
    # timestep AFTER the split-explicit acoustic update (mirrors
    # iter-169 ``damp_v`` post-step pattern).  Default 0.0 preserves
    # baseline bit-for-bit (Python-static branch).  FV3 AM4 production
    # default is ``damp_w=0.30 + nord_w=2`` (del-6).  Complementary to
    # the legoESM-native ``hyperdiff_w_coeff`` biharmonic — users
    # typically choose one mechanism, not both.
    damp_w: float = 0.0
    nord_w: int = 2
    # FV3-faithful KE→heat conversion for iter-193 ``damp_w`` damping
    # (FV3_3D iter 203, refined in iter-207 with Π Exner factor).
    # Faithful port of FV3 ``sw_core.F90:1086``::
    #
    #     heat_source = -d_con * dw * (w + 0.5*dw)
    #
    # which represents the negative of the change in kinetic energy
    # density per unit mass (``-ΔKE_w = -(w*dw + 0.5*dw²)``).  When
    # ``damp_w`` removes KE from ``w`` (``dw`` opposite sign to ``w``),
    # heat_source is POSITIVE — the lost KE is deposited as heat in
    # the temperature field, preserving total energy.
    #
    # iter-207: Π_ref from HeightCoordinate (FV3-faithful within ref-state linearization).
    # heat computed at half-levels (where w/dw live), averaged to full-levels for θ_p.
    damp_w_d_con: float = 0.0
    delt_max: float = 0.0
        # FV3_3D iter 218: per-step heating cap (FV3 dyn_core.F90:1774). NH form: |Δθ_p·Π_ref|≤dt·delt_max.
    corner_div_damp_d_con: float = 0.0
        # FV3_3D iter 222 (NH mirror of PE 221): KE→heat for corner-div damp.
        # dθ_p/dt = -coeff*(u·du+v·dv)/(c_pd*Π_ref).
    use_fv3_cross_face_du_proj: bool = False
        # FV3_3D iter 370 (NH mirror of PE 370): cross-face halo for damp_v wind projection.
        # iter-384: NO-OP without duogrid=True.
    use_fv3_metric_aware_d_con: bool = False
        # FV3_3D iter 339 (NH mirror of PE 338): metric-aware d_con at damp_v_d_con site.
        # Composes with use_fv3_d_con_cv + use_fv3_dynamic_exner.
    use_fv3_dynamic_exner: bool = False
        # FV3_3D iter 336: dynamic Exner Π_total = Π_ref + π' (FV3 live pkz).
        # iter-394: Π_total ≡ FV3 pkz under EOS. Wired at 3 slow-tendency d_con sites.
        # Post-acoustic sites keep Π_ref. NH-only (PE uses T directly).
    d_con_top_zero_levels: int = 0
        # FV3_3D iter 431: sponge-zero d_con top N levels (FV3 dyn_core.F90:773-805).
        # Wired at all 5 NH d_con sites via _d_con_sum mask (iter-432).
    use_fv3_vector_halo_uv: bool = False
        # FV3_3D iter 328: vector halo for cc → D-grid (u, v). Scalar halo leaves O(1) basis-mismatch
        # at cube edges (face-local e_x/e_y differ). center_to_dgrid_vector rotates via pad_halo_vector
        # (FV3 ext_vector, fv_duogrid.F90:626-975). PE unaffected (winds at corners).
    use_fv3_a2b_ord4_vector_uv: bool = False
        # FV3_3D iter 697: 4th-order a2b_ord4 PPM+Lagrange cascade for cc → D-grid (u, v).
        # Requires use_fv3_vector_halo_uv=True.  Halo=2 vector pad + 4th-order corner cascade
        # (FV3 a2b_edge.F90:a2b_ord4 duogrid path).  iter-698 empirical: -25.8% θ′ edge
        # ratio at C8; iter-699 verified -21.9% at C16.  Promoted to factory ON.
    use_fv3_a2b_ord4_theta_corner: bool = False
        # FV3_3D iter 700: 4th-order a2b_ord4 cc → B-grid corner for θ_total in the
        # c_p · θ_corner · dπ Coriolis-pressure term (line ~383).  Same 4th-order
        # scalar cascade used by use_fv3_a2b_zeta_corner (iter-170 PE) and the
        # iter-696 vector path.  Default OFF; impact measurement pending.
    use_fv3_d_con_cv: bool = False
        # FV3_3D iter 320: c_v denominator for NH d_con (FV3 dyn_core.F90:1795 cv_air branch).
        # NH conserves internal energy c_v·T; c_pd under-heats by c_v/c_p≈0.714 (~40%). PE unaffected.


def _apply_top_sponge_damp_boost(damp_corner, da_min_c, config):
    """FV3_3D iter 446: wrapper adapting iter-440 NH callers to shared helper."""
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_damp_boost as _shared,
    )
    return _shared(
        damp_corner, da_min_c,
        config.corner_div_damp_d2_bg,
        config.corner_div_damp_d2_bg_k1,
        config.corner_div_damp_d2_bg_k2,
    )


def cdgrid_compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    cdgrid: CubedSphereCDGrid,
    config: CDGridCompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
    dt_actual: float | jax.Array | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies using C-D grid operators."""
    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data

    c_p = constants.c_pd
    rho_0 = height_coord.rho_ref
    theta_0 = height_coord.theta_ref
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    J = terrain_metric.jacobian

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- 1. Exner perturbation and horizontal PGF ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)
    # FV3_3D iter 336: dynamic Exner Π_total = Π_ref + π' for slow-tendency d_con denominators
    if config.use_fv3_dynamic_exner:
        _exner_eff_b = (
            height_coord.exner_ref[None, None, None, :]
            + pi_prime
        )
    else:
        _exner_eff_b = height_coord.exner_ref[None, None, None, :]

    # --- 2. Convert to D-grid ---
    # Batched (u,v) cc→corner via single halo + 4-pt avg
    # FV3_3D iter 328: vector-aware center_to_dgrid_vector rotates across cube faces (default False = scalar)
    n_face_uv, n_i_uv, n_j_uv, nlev_uv = u.shape
    if config.use_fv3_vector_halo_uv:
        u_d, v_d = center_to_dgrid_vector(
            u, v, cdgrid,
            use_fv3_a2b_ord4=config.use_fv3_a2b_ord4_vector_uv,
        )
    else:
        _uv_stack = jnp.stack([u, v], axis=-1)  # (6, n, n, nlev, 2)
        _uv_flat = _uv_stack.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
        _uv_d_flat = interp_center_to_corner(_uv_flat, cdgrid)
        _uv_d = _uv_d_flat.reshape(
            _uv_d_flat.shape[0], _uv_d_flat.shape[1], _uv_d_flat.shape[2],
            nlev_uv, 2,
        )
        u_d = _uv_d[..., 0]
        v_d = _uv_d[..., 1]

    # --- 3. C-grid velocities ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # --- 4. Vorticity (cell centres) ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

    # --- 5. KE at cell centres from D-grid (orthogonal basis) ---
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    K = 0.5 * (u_cc ** 2 + v_cc ** 2)

    # --- 6. Gradients at D-grid corners ---
    # Pack K + pi_prime into single halo. FV3_3D iter 325: duogrid remap (PE iter-84 mirror;
    # required for FV3 a2b_edge 4th-order corner accuracy at edges).
    _nh_dg = grid.duogrid
    from legoesm.grids.halo import get_halo_backend as _ghb_step6
    _hb_step6 = _ghb_step6()
    if _hb_step6 == "spmd":
        from legoesm.parallel.cubesphere_exchange import get_spmd_mesh
        _spmd_mesh_step6 = get_spmd_mesh()
        _K_pad_step6, _pi_pad_step6 = packed_pad_halo_4d(
            K, pi_prime, mesh=_spmd_mesh_step6, duogrid=_nh_dg,
        )
    elif _hb_step6 == "mpi":
        from legoesm.grids.halo import get_mpi_topology as _gmt_step6
        _mpi_topo_step6 = _gmt_step6()
        # FV3_3D iter-1041: pass interp_offsets when duogrid is off so the
        # packed MPI (K, pi_prime) exchange Lagrange-remaps halos rather
        # than nearest-copying — matches the local path which has
        # ``operators`` doing per-field ``pad_halo_4d(interp_offsets=...)``.
        _nh_offs_step6 = (
            None if _nh_dg is not None else grid.halo_interp_offsets
        )
        _K_pad_step6, _pi_pad_step6 = packed_pad_halo_mpi_4d(
            K, pi_prime, topology=_mpi_topo_step6, duogrid=_nh_dg,
            interp_offsets=_nh_offs_step6,
        )
    else:
        _K_pad_step6 = _pi_pad_step6 = None

    # Batch 2 A-L gradients (K, pi_prime) on stacked trailing axis (2 calls → 1)
    n_face_kp, n_i_kp, n_j_kp, nlev_kp = K.shape
    _kp_stack = jnp.stack([K, pi_prime], axis=-1)  # (6, n, n, nlev, 2)
    _kp_flat = _kp_stack.reshape(n_face_kp, n_i_kp, n_j_kp, nlev_kp * 2)
    if _K_pad_step6 is not None:
        _kp_pad_stack = jnp.stack([_K_pad_step6, _pi_pad_step6], axis=-1)
        _kp_pad_flat = _kp_pad_stack.reshape(
            _kp_pad_stack.shape[0], _kp_pad_stack.shape[1],
            _kp_pad_stack.shape[2], nlev_kp * 2,
        )
    else:
        _kp_pad_flat = None
    _dKpi_dx_flat, _dKpi_dy_perp_flat = arakawa_lamb_gradient(
        _kp_flat, cdgrid, padded=_kp_pad_flat,
    )
    _dKpi_dx = _dKpi_dx_flat.reshape(
        _dKpi_dx_flat.shape[0], _dKpi_dx_flat.shape[1],
        _dKpi_dx_flat.shape[2], nlev_kp, 2,
    )
    _dKpi_dy_perp = _dKpi_dy_perp_flat.reshape(
        _dKpi_dy_perp_flat.shape[0], _dKpi_dy_perp_flat.shape[1],
        _dKpi_dy_perp_flat.shape[2], nlev_kp, 2,
    )
    dK_dx = _dKpi_dx[..., 0]
    dpi_dx = _dKpi_dx[..., 1]
    dK_dy_perp = _dKpi_dy_perp[..., 0]
    dpi_dy_perp = _dKpi_dy_perp[..., 1]

    # --- 7. D-grid momentum tendencies ---
    # iter-74: interp ζ only; add f_corner directly (interp(f_cc)≠f_corner gives O(dx²) Coriolis err).
    # FV3_3D iter 170/190: a2b_ord4 for ζ_corner (mirror of PE 14); shared with iter-187 smag_vort cap.
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
        # shape-polymorphic (axis-1/2 slicing, trailing axes broadcast)
        # and ``_pad_halo_auto_h2`` already dispatches to
        # ``pad_halo_4d`` for 4D input.  Calling it directly on the 4D
        # ``zeta`` avoids a ``jax.vmap`` that would wrap ``pad_halo``
        # under MPI — mpi4jax's sendrecv batching rule asserts matching
        # batch axes and fires when sendrecv runs inside vmap.
        _zeta_a2b_ord4 = interp_center_to_corner_a2b_ord4(zeta, cdgrid)

    if config.use_fv3_a2b_zeta_corner:
        zeta_corner = _zeta_a2b_ord4
    else:
        zeta_corner = interp_center_to_corner(zeta, cdgrid)
    if config.use_coriolis:
        abs_vor_corner = zeta_corner + cdgrid.f_corner[..., None]
    else:
        abs_vor_corner = zeta_corner
    if config.use_fv3_a2b_ord4_theta_corner:
        from legoesm.core.operators_cdgrid import (
            interp_center_to_corner_a2b_ord4 as _icc_a2b_ord4_theta,
        )
        # FV3_3D iter-1043: same lift-out-of-vmap rationale as the
        # ``zeta`` a2b path above.
        theta_corner = _icc_a2b_ord4_theta(theta_total, cdgrid)
    else:
        theta_corner = interp_center_to_corner(theta_total, cdgrid)

    du_d_dt = abs_vor_corner * v_d - dK_dx - c_p * theta_corner * dpi_dx
    dv_d_dt = -abs_vor_corner * u_d - dK_dy_perp - c_p * theta_corner * dpi_dy_perp

    # FV3_3D iter 171: cell-centre div damp (mirror of PE iter-5; FV3 sw_core.F90:1720)
    _need_div_damp = config.div_damp_coeff > 0.0
    if _need_div_damp:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)         # (6, n, n, nlev)
        # iter-173: async-halo overlap under MPI
        from legoesm.grids.halo import get_halo_backend as _ghb_div
        if config.use_async_halo and _ghb_div() == "mpi":
            from legoesm.core.operators_cdgrid import (
                overlapped_arakawa_lamb_gradient,
            )
            ddiv_dx, ddiv_dy_perp = overlapped_arakawa_lamb_gradient(
                div_v, cdgrid,
            )
        else:
            ddiv_dx, ddiv_dy_perp = arakawa_lamb_gradient(div_v, cdgrid)
        if config.div_damp_dddmp > 0.0:
            # FV3 sw_core.F90:1720 adaptive Smagorinsky formulation.
            # ``da_min_c`` = global min B-grid corner area.
            _da_min_c = jnp.min(cdgrid.area_corner)
            _d2_bg = config.div_damp_coeff / _da_min_c
            _div_abs_corner = interp_center_to_corner(
                jnp.abs(div_v), cdgrid,
            )
            _adaptive_coeff = _da_min_c * jnp.maximum(
                _d2_bg,
                jnp.minimum(
                    _FV3_CNST_0P20, config.div_damp_dddmp * _div_abs_corner,
                ),
            )
            _du_d_dt_dd = _adaptive_coeff * ddiv_dx
            _dv_d_dt_dd = _adaptive_coeff * ddiv_dy_perp
        else:
            _du_d_dt_dd = config.div_damp_coeff * ddiv_dx
            _dv_d_dt_dd = config.div_damp_coeff * ddiv_dy_perp
        du_d_dt = du_d_dt + _du_d_dt_dd
        dv_d_dt = dv_d_dt + _dv_d_dt_dd

        # FV3_3D iter 224: NH mirror of PE iter-223 cell-centre
        # div_damp d_con.  Compute heat tendency and stash for
        # accumulation into dtheta_p_dt later.
        if config.div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 350: NH mirror of PE iter-349 metric
                # form at cell-centre div_damp d_con site.
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
                _cosa_dd = cdgrid.cosa_cell[..., None]
                _rsin2_dd = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_dd = 0.25 * _rsin2_dd * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (
                        _us * _ubs + _un * _ubn
                        + _vw * _vbw + _ve * _vbe
                    )
                    - _cosa_dd * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
            else:
                _dKE_dt_corner_dd = (
                    u_d * _du_d_dt_dd + v_d * _dv_d_dt_dd
                )
                _dKE_dt_cc_dd = interp_corner_to_center(
                    _dKE_dt_corner_dd,
                )
            _cx_dd = (
                constants.c_vd if config.use_fv3_d_con_cv
                else constants.c_pd
            )
            _dtheta_p_dt_dd_cc = (
                -config.div_damp_d_con
                * _dKE_dt_cc_dd
                / (_cx_dd * _exner_eff_b)
            )
        else:
            _dtheta_p_dt_dd_cc = None
    else:
        div_v = None  # computed lazily by theta block below
        _dtheta_p_dt_dd_cc = None

    # Laplacian viscosity — batch (u_d, v_d) into a single
    # ``laplacian_dgrid`` call by stacking along a trailing axis and
    # folding into the level dim.  ``laplacian_dgrid`` is now
    # 4D-native (single ``pad_halo_4d`` for all "levels"), so the
    # paired call shares one halo exchange and one compact ∇² across
    # both wind components — same passive-trailing-axis pattern as the
    # corner interps and other dycore batches.
    if config.A_h > 0:
        n_face_vl, n_id_vl, n_jd_vl, nlev_vl = u_d.shape
        _uv_d_lap_stack = jnp.stack([u_d, v_d], axis=-1)
        _uv_d_lap_flat = _uv_d_lap_stack.reshape(
            n_face_vl, n_id_vl, n_jd_vl, nlev_vl * 2,
        )
        _uv_d_lap_out = laplacian_dgrid(_uv_d_lap_flat, cdgrid).reshape(
            n_face_vl, n_id_vl, n_jd_vl, nlev_vl, 2,
        )
        # FV3_3D iter 180: optional Smagorinsky-style adaptive A_h
        # added on top of the static coefficient.  Same wiring as
        # PE iter 58.  When ``smagorinsky_cs > 0``, compute the
        # per-cell adaptive field at corners via the existing
        # ``compute_smagorinsky_ah_3d`` helper and combine.
        if config.smagorinsky_cs > 0.0:
            from legoesm.core._smagorinsky_visc import (
                compute_smagorinsky_ah_3d,
            )
            _ah_smag_corner = compute_smagorinsky_ah_3d(
                u_d, v_d, cdgrid, config.smagorinsky_cs,
            )                                              # (6, n+1, n+1, nlev)
            _ah_eff_corner = config.A_h + _ah_smag_corner
            _du_d_dt_ah = _ah_eff_corner * _uv_d_lap_out[..., 0]
            _dv_d_dt_ah = _ah_eff_corner * _uv_d_lap_out[..., 1]
        else:
            _du_d_dt_ah = config.A_h * _uv_d_lap_out[..., 0]
            _dv_d_dt_ah = config.A_h * _uv_d_lap_out[..., 1]
        du_d_dt = du_d_dt + _du_d_dt_ah
        dv_d_dt = dv_d_dt + _dv_d_dt_ah

        # FV3_3D iter 226 (NH mirror of PE 225): A_h d_con; dθ_p/dt = -ah_d_con*(u·du+v·dv)/(c_pd*Π_ref)
        if config.ah_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 352: NH metric form at A_h d_con site
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
                _cosa_ah = cdgrid.cosa_cell[..., None]
                _rsin2_ah = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_ah = 0.25 * _rsin2_ah * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (
                        _us * _ubs + _un * _ubn
                        + _vw * _vbw + _ve * _vbe
                    )
                    - _cosa_ah * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
            else:
                _dKE_dt_corner_ah = (
                    u_d * _du_d_dt_ah + v_d * _dv_d_dt_ah
                )
                _dKE_dt_cc_ah = interp_corner_to_center(
                    _dKE_dt_corner_ah,
                )
            _cx_ah = (
                constants.c_vd if config.use_fv3_d_con_cv
                else constants.c_pd
            )
            _dtheta_p_dt_ah_cc = (
                -config.ah_d_con
                * _dKE_dt_cc_ah
                / (_cx_ah * _exner_eff_b)
            )
        else:
            _dtheta_p_dt_ah_cc = None
    else:
        _dtheta_p_dt_ah_cc = None

    # FV3_3D iter 168: B-grid corner-div damp (mirror of PE iter-16/18; FV3 sw_core.F90:1641-1822 d_sw5)
    if config.corner_div_damp_d2_bg > 0.0:
        from legoesm.core._fv3_divergence_corner import (
            fv3_divergence_corner_3d,
        )
        delpc = fv3_divergence_corner_3d(u_d, v_d, cdgrid)  # (6, n+1, n+1, nlev)

        # Step 2: adaptive damp (FV3 sw_core.F90:1720). iter-189: prefer dt_actual
        _da_min_c = jnp.min(cdgrid.area_corner)
        _delpc_abs = jnp.abs(delpc)
        if dt_actual is not None:
            _dt_approx = dt_actual
        else:
            _dt_approx = config.corner_div_damp_dt_proxy
        _damp_corner = _da_min_c * jnp.maximum(
            config.corner_div_damp_d2_bg,
            jnp.minimum(
                _FV3_CNST_0P20, config.corner_div_damp_dddmp * _delpc_abs * _dt_approx,
            ),
        )                                                  # (6, n+1, n+1, nlev)
        _damp_corner = _apply_top_sponge_damp_boost(
            _damp_corner, _da_min_c, config,
        )

        # Step 3: del-(2*(nord+1)) damp (FV3 sw_core.F90:1725-1822, nord>0)
        # FV3_3D iter 893: nord-loop preserved inline (a 1-ULP trace-reorder
        # would break the iter-22 bit-for-bit test; mirror of PE-side rationale).
        if config.corner_div_damp_d4_bg > 0.0 and config.corner_div_damp_nord > 0:
            from legoesm.core._fv3_divergence_corner import (
                fv3_corner_laplacian_iteration,
            )
            _vfill = config.corner_div_damp_fv3_vector_fill

            # FV3_3D iter-1044: ``fv3_corner_laplacian_iteration`` is now
            # shape-polymorphic (3D and 4D dispatched at pad_halo step).
            # Calling it directly on the 4D ``delpc`` field avoids a
            # ``jax.vmap`` that would wrap ``pad_halo`` under MPI — same
            # mpi4jax sendrecv batch-axis fix as iter-1042 / iter-1043.
            _delpc_initial = delpc
            _divg_d_iter = delpc
            for _ in range(config.corner_div_damp_nord):
                _divg_d_iter = fv3_corner_laplacian_iteration(
                    _divg_d_iter, cdgrid, apply_vector_corner_fill=_vfill,
                )

            # FV3_3D iter 187: smag_vort cap for nord>=1 (FV3 sw_core.F90:1797-1809).
            # smag_vort = |dt|*sqrt(delpc² + ζ²); iter-181/183 double-where guards sqrt(0).
            # iter-190: reuse _zeta_a2b_ord4 from iter-170 site
            _zeta_smag_corner = _zeta_a2b_ord4              # (6, n+1, n+1, nlev)
            _smag_arg = _delpc_initial ** 2 + _zeta_smag_corner ** 2
            _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
            _smag_root = jnp.where(
                _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
            )
            _smag_vort = jnp.abs(_dt_approx) * _smag_root    # (6, n+1, n+1, nlev)
            _damp_corner = _da_min_c * jnp.maximum(
                config.corner_div_damp_d2_bg,
                jnp.minimum(_FV3_CNST_0P20, config.corner_div_damp_dddmp * _smag_vort),
            )                                                # (6, n+1, n+1, nlev)
            _damp_corner = _apply_top_sponge_damp_boost(
                _damp_corner, _da_min_c, config,
            )

            _dd8 = jnp.asarray(
                (_da_min_c * config.corner_div_damp_d4_bg)
                ** (config.corner_div_damp_nord + 1),
                dtype=delpc.dtype,
            )
            _ke_correction = (
                _damp_corner * _delpc_initial + _dd8 * _divg_d_iter
            )                                              # (6, n+1, n+1, nlev)
        else:
            _ke_correction = _damp_corner * delpc          # (6, n+1, n+1, nlev)

        # Step 4: gradient at corners. FV3_3D iter 325: duogrid remap for ke_correction halo
        # (mirror of PE iter-84; required for FV3 a2b_edge 4th-order edge accuracy)
        _ke_pad = _pad_halo_4d_module(
            _ke_correction, duogrid=_nh_dg,
        )                                                   # (6, n+3, n+3, nlev)

        _dke_dx_pad = (_ke_pad[:, 2:, 1:-1, :] - _ke_pad[:, :-2, 1:-1, :])
        _dke_dy_pad = (_ke_pad[:, 1:-1, 2:, :] - _ke_pad[:, 1:-1, :-2, :])

        _dx_corner_uface = jnp.pad(
            cdgrid.dxc, [(0, 0), (0, 0), (0, 1)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _dy_corner_vface = jnp.pad(
            cdgrid.dyc, [(0, 0), (0, 1), (0, 0)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _two_dx = 2.0 * _dx_corner_uface[..., None]
        _two_dy = 2.0 * _dy_corner_vface[..., None]

        _du_d_dt_cdd = -_dke_dx_pad / _two_dx
        _dv_d_dt_cdd = -_dke_dy_pad / _two_dy
        du_d_dt = du_d_dt + _du_d_dt_cdd
        dv_d_dt = dv_d_dt + _dv_d_dt_cdd

        # FV3_3D iter 222 (NH mirror of PE 221): corner-div d_con KE→heat
        if config.corner_div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 348: NH metric form at corner_div_d_con
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
                _cosa_cdd = cdgrid.cosa_cell[..., None]
                _rsin2_cdd = cdgrid.rsin2_cell[..., None]
                _dKE_dt_cc_cdd = 0.25 * _rsin2_cdd * (
                    _ubs ** 2 + _ubn ** 2 + _vbw ** 2 + _vbe ** 2
                    + 2.0 * (_gys + _gyn + _gxw + _gxe)
                    - _cosa_cdd * (_u2 * _dv2 + _v2 * _du2 + _du2 * _dv2)
                )
            else:
                _dKE_dt_corner_cdd = (
                    u_d * _du_d_dt_cdd + v_d * _dv_d_dt_cdd
                )
                _dKE_dt_cc_cdd = interp_corner_to_center(
                    _dKE_dt_corner_cdd,
                )
            _cx_cdd = (
                constants.c_vd if config.use_fv3_d_con_cv
                else constants.c_pd
            )
            _dtheta_p_dt_cdd_cc = (
                -config.corner_div_damp_d_con
                * _dKE_dt_cc_cdd
                / (_cx_cdd * _exner_eff_b)
            )
        else:
            _dtheta_p_dt_cdd_cc = None

    else:
        _dtheta_p_dt_cdd_cc = None

    # --- 8. Convert back to cell-centre ---
    # Batch (du_d_dt, dv_d_dt) corner→cc (4-pt avg, no halo)
    _duv_d_dt = jnp.stack([du_d_dt, dv_d_dt], axis=-1)  # (..., 2)
    _duv_d_dt_flat = _duv_d_dt.reshape(
        _duv_d_dt.shape[0], _duv_d_dt.shape[1], _duv_d_dt.shape[2],
        nlev_uv * 2,
    )
    _duv_dt = interp_corner_to_center(_duv_d_dt_flat).reshape(
        n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2,
    )
    du_dt = _duv_dt[..., 0]
    dv_dt = _duv_dt[..., 1]

    # --- 9. Vertical advection of u, v ---
    # Loops 137/141: batch 2 vertical_advection_height calls via leading axis
    _uv_va = jnp.stack([u, v], axis=0)
    _uv_va_adv = vertical_advection_height(_uv_va, w, dz, dz_half, J)
    du_dt = du_dt + _uv_va_adv[0]
    dv_dt = dv_dt + _uv_va_adv[1]

    # --- 10. Theta: advective form -v·∇θ = -∇·(θv) + θ∇·v (θ not a conserved density) ---
    # iter-171: reuse div_v from cell-centre div_damp block if available
    if div_v is None:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)

    # --- 10/11/12. (theta, rho, tracers) flux divergence batched via PPM trailing axis ---
    n_face_tr, n_i_tr, n_j_tr, nlev_tr = theta_total.shape
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    n_total = 2 + n_tracers  # theta + rho + tracers

    # Terrain-following z* (J = dz/dz*, horizontally varying): continuity
    # is conservative in the pseudo-density J·ρ (FV3's delp analogue):
    # dρ'/dt = -(1/J) ∇·(J ρ v).  Weight the advected scalar so the SAME
    # PPM flux path discretizes J·ρ at the C-grid faces; the divergence
    # is divided by the cell J below.  J ≡ 1 (flat) is bit-identical.
    J_rho_total = J[..., None] * rho_total

    if n_tracers > 0:
        combined_stack = jnp.concatenate(
            [
                jnp.stack([theta_total, J_rho_total], axis=-1),  # (..., nlev, 2)
                tracers,  # (..., nlev, n_tracers)
            ], axis=-1,
        )  # (..., nlev, n_total)
    else:
        combined_stack = jnp.stack(
            [theta_total, J_rho_total], axis=-1,
        )  # (..., nlev, 2)

    combined_flat = combined_stack.reshape(
        n_face_tr, n_i_tr, n_j_tr, nlev_tr * n_total,
    )
    if n_total == 1:
        u_c_b, v_c_b = u_c, v_c
    else:
        u_c_b = jnp.repeat(u_c, n_total, axis=-1)
        v_c_b = jnp.repeat(v_c, n_total, axis=-1)
    flux_combined_flat = cgrid_mass_flux_divergence(
        combined_flat, u_c_b, v_c_b, cdgrid,
    )
    flux_combined = flux_combined_flat.reshape(
        n_face_tr, n_i_tr, n_j_tr, nlev_tr, n_total,
    )
    # Theta uses advective form: -∇·(θv) + θ·∇·v.
    dtheta_p_dt = flux_combined[..., 0] + theta_total * div_v

    # FV3_3D iter 240 (NH mirror of PE 239): aggregate 3 d_con (iter-222/224/226), cap with delt_max
    # NH cap policy: k=0 → 0.1×, k=1 → 0.5×, k>=2 → 1× (FV3 cv_air sw_core.F90:1782-1786). θ_p space: ÷Π_ref.
    _d_con_sum = None
    for _contrib in (
        _dtheta_p_dt_cdd_cc, _dtheta_p_dt_dd_cc, _dtheta_p_dt_ah_cc,
    ):
        if _contrib is not None:
            _d_con_sum = (
                _contrib if _d_con_sum is None
                else _d_con_sum + _contrib
            )
    if _d_con_sum is not None:
        # FV3_3D iter 432: sponge-zero d_con top N levels
        if config.d_con_top_zero_levels > 0:
            _nlev_zs = _d_con_sum.shape[-1]
            _k_idx_zs = jnp.arange(_nlev_zs)
            _d_con_mask_s = jnp.where(
                _k_idx_zs < config.d_con_top_zero_levels,
                0.0, 1.0,
            )
            _d_con_sum = _d_con_sum * _d_con_mask_s[None, None, None, :]
        # FV3_3D iter 457: optional FV3-faithful del-2 smoothing
        # of aggregate heat_source.  Port of FV3 ``dyn_core.F90:
        # 1755-1756`` ``del2_cubed(heat_source, cnst_0p20*da_min,
        # ..., nf_ke)``.  Each iteration:
        # ``heat_source += coeff * da_min * ∇²heat_source``
        # which is a forward-Euler diffusion step.
        if config.heat_source_del2_iters > 0:
            _da_min_hs = jnp.min(cdgrid.area_corner)
            _cd_hs = config.heat_source_del2_coeff * _da_min_hs
            for _ in range(config.heat_source_del2_iters):
                _lap_hs = laplacian_compact_3d(_d_con_sum, grid)
                _d_con_sum = _d_con_sum + _cd_hs * _lap_hs
        if config.delt_max > 0.0:
            _nlev_d = _d_con_sum.shape[-1]
            _k_idx = jnp.arange(_nlev_d)
            _sponge_factor = jnp.where(
                _k_idx == 0, 0.1,
                jnp.where(_k_idx == 1, 0.5, 1.0),
            )
            # FV3_3D iter 397: use Π_total (= Π_ref + π') under
            # ``use_fv3_dynamic_exner=True`` for the per-step ΔT
            # cap derivation ``|Δθ_p · Π| ≤ delt_max · dt``.
            # Consistent with iter-336/337 d_con denominator
            # wiring.  Default False keeps frozen exner_ref.
            if config.use_fv3_dynamic_exner:
                # _exner_eff_b is shape (6, n, n, nlev) — cell-
                # centred Π_total.  Apply sponge factor as 1D
                # mask along the nlev axis.
                _sf_b = _sponge_factor[None, None, None, :]
                _delt_theta_b = (
                    config.delt_max * _sf_b / _exner_eff_b
                )
            else:
                _delt_theta_per_level = (
                    config.delt_max
                    * _sponge_factor
                    / height_coord.exner_ref
                )
                _delt_theta_b = _delt_theta_per_level[
                    None, None, None, :
                ]
            _d_con_sum = jnp.clip(
                _d_con_sum, -_delt_theta_b, _delt_theta_b,
            )
        dtheta_p_dt = dtheta_p_dt + _d_con_sum
    # Rho uses pure flux form in z*: dρ'/dt = -(1/J) ∇·(J ρ v)
    # (continuity; ``cgrid_mass_flux_divergence`` already returns the
    # NEGATIVE divergence, so net outflow of the J-weighted flux lowers
    # rho).  The vertical leg carries its 1/J in the acoustic substeps.
    drho_p_dt = flux_combined[..., 1] / J[..., None]

    # Tracer advection (advective form).  Vertical advection still runs
    # per-tracer via ``vmap`` over the trailing axis so JAX produces one
    # batched kernel.
    if n_tracers > 0:
        horiz = (
            flux_combined[..., 2:]                # (..., nlev, n_tracers)
            + tracers * div_v[..., None]          # advective-form correction
        )

        def _vert_one(q):
            return vertical_advection_height(q, w, dz, dz_half, J)

        vert = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(tracers)
        dtracers_dt = horiz + vert
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 13. Hyperdiffusion ---
    # ``hyperdiffusion_3d(field, grid, coeff)`` is defined as
    # ``-coeff * ∇²(∇²(field))`` where each ∇² runs the cubed-sphere
    # 4D-native compact stencil (single ``pad_halo_4d`` per call).
    # When *both* ``hyperdiff_coeff`` (applied to (u, v, theta_p))
    # and ``hyperdiff_rho_coeff`` (applied to rho_p) are non-zero,
    # the two operator chains share the same biharmonic structure
    # but use different coefficients on the outer step.  Inline the
    # operator and stack ALL four fields along a trailing axis: a
    # single inner ∇² and a single outer ∇² serve all four fields,
    # then per-field coefficients are applied at the very end.
    # 4 ∇² evaluations → 2 (one inner, one outer) per RHS evaluation
    # when both coefficients are active.  When only one coefficient
    # is active, fall back to the existing path (3-field or 1-field).
    _coeff_uvT = config.hyperdiff_coeff
    _coeff_rho = config.hyperdiff_rho_coeff
    if _coeff_uvT > 0 and _coeff_rho > 0:
        n_face_h, n_i_h, n_j_h, nlev_h = u.shape
        _hyper_stack = jnp.stack(
            [u, v, theta_p, rho_p], axis=-1,
        )  # (6, n, n, nlev, 4)
        _hyper_flat = _hyper_stack.reshape(n_face_h, n_i_h, n_j_h, nlev_h * 4)
        # Inner ∇² (compact stencil) — shared across all four fields.
        _lap1 = laplacian_compact_3d(_hyper_flat, grid)
        # Outer ∇² = div(grad).  Pad ``_lap1`` once and feed it to
        # both gradient ops (saves 1 ``pad_halo_4d`` per call).
        # iter-169: use the imported ``_pad_halo_4d_module`` alias —
        _dg = getattr(grid, 'duogrid', None)
        _offsets = None if _dg is not None else grid.halo_interp_offsets
        _lap1_pad = _pad_halo_4d_module(_lap1, interp_offsets=_offsets, duogrid=_dg)
        _gx = gradient_x_3d(_lap1, grid, padded=_lap1_pad)
        _gy = gradient_y_3d(_lap1, grid, padded=_lap1_pad)
        _lap2 = divergence_3d(_gx, _gy, grid).reshape(
            n_face_h, n_i_h, n_j_h, nlev_h, 4,
        )
        # Per-field hyperdiff coefficients
        du_dt = du_dt - _coeff_uvT * _lap2[..., 0]
        dv_dt = dv_dt - _coeff_uvT * _lap2[..., 1]
        dtheta_p_dt = dtheta_p_dt - _coeff_uvT * _lap2[..., 2]
        drho_p_dt = drho_p_dt - _coeff_rho * _lap2[..., 3]
    elif _coeff_uvT > 0:
        n_face_h, n_i_h, n_j_h, nlev_h = u.shape
        hyper_stack = jnp.stack(
            [u, v, theta_p], axis=-1,
        )  # (6, n, n, nlev, 3)
        hyper_flat = hyper_stack.reshape(n_face_h, n_i_h, n_j_h, nlev_h * 3)
        hyper_out_flat = hyperdiffusion_3d(
            hyper_flat, grid, _coeff_uvT,
        )
        hyper_out = hyper_out_flat.reshape(n_face_h, n_i_h, n_j_h, nlev_h, 3)
        du_dt = du_dt + hyper_out[..., 0]
        dv_dt = dv_dt + hyper_out[..., 1]
        dtheta_p_dt = dtheta_p_dt + hyper_out[..., 2]
    elif _coeff_rho > 0:
        drho_p_dt = drho_p_dt + hyperdiffusion_3d(
            rho_p, grid, _coeff_rho,
        )

    # --- 14. Sponge layer ---
    sponge = sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge * u
    dv_dt = dv_dt - sponge * v
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 15. w tendency (slow: horizontal advection) ---
    # Pre-pad w_full for shared halo across gradient_x/y
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    _dg_w = getattr(grid, 'duogrid', None)
    _offsets_w = None if _dg_w is not None else grid.halo_interp_offsets
    _w_full_pad = _pad_halo_4d_module(w_full, interp_offsets=_offsets_w, duogrid=_dg_w)
    dw_dx = gradient_x_3d(w_full, grid, padded=_w_full_pad)
    dw_dy = gradient_y_3d(w_full, grid, padded=_w_full_pad)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    # Pad zero at top/bottom interfaces (rigid BC)
    pad_axes_w = ((0, 0),) * (w.ndim - 1)
    horiz_adv_w_half = jnp.pad(
        0.5 * (horiz_adv_w[..., :-1] + horiz_adv_w[..., 1:]),
        (*pad_axes_w, (1, 1)),
    )

    dw_dt = horiz_adv_w_half - sponge_half * w
    if config.hyperdiff_w_coeff > 0:
        dw_dt = dw_dt + hyperdiffusion_3d(w, grid, config.hyperdiff_w_coeff)

    # --- 16. Physics ---
    if physics_tendency is not None:
        du_dt = du_dt + physics_tendency.du_dt.data
        dv_dt = dv_dt + physics_tendency.dv_dt.data
        dw_dt = dw_dt + physics_tendency.dw_dt.data
        dtheta_p_dt = dtheta_p_dt + physics_tendency.dtheta_prime_dt.data
        drho_p_dt = drho_p_dt + physics_tendency.drho_prime_dt.data
        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=dw_dt, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=dtheta_p_dt, name="dtheta_prime_dt", dims=dims_3d, units="K/s",
        ),
        drho_prime_dt=Field(
            data=drho_p_dt, name="drho_prime_dt", dims=dims_3d, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.phis.data),
            name="dphis_dt", dims=dims_2d, units="m^2/s^3",
        ),
        dtracers_dt=Field(
            data=dtracers_dt, name="dtracers_dt", dims=dims_tr, units="1/s",
        ),
    )


class CDGridCompressibleEulerModel(IntegrationMixin):
    """FV3-style C-D grid non-hydrostatic compressible Euler model."""

    def __init__(
        self,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CDGridCompressibleEulerConfig | None = None,
    ):
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or CDGridCompressibleEulerConfig()
        # FV3_3D iter 902: enforce iter-890 nord range validation at
        # model construction (fail-fast vs silent misuse).  Mirror of
        # PE primitive_eq_cdgrid.py site.
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            validate_corner_div_damp_nord,
        )
        validate_corner_div_damp_nord(self.config.corner_div_damp_nord)
        self._target_mass = None

        # iter-20: anchor-mass API parity (see iter-18 / iter-19 SW twins).

        if self.config.small_earth_factor != 1.0:
            grid = apply_small_earth_scaling(grid, self.config.small_earth_factor)
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)

        # Acoustic CFL check at construction time (outside JIT)
        _ce_logger = logging.getLogger("legoesm.compressible_euler")
        dx_min = estimate_min_dx_cubed_sphere(
            grid.n, getattr(grid, 'radius', constants.R_earth),
        )
        c_sound = float(jnp.sqrt(
            constants.c_pd / constants.c_vd * constants.R_d * 300.0
        ))
        self._dx_min = dx_min
        self._c_sound = c_sound
        _ce_logger.info(
            f"  Acoustic check: dx_min={dx_min/1000:.0f}km, c_s={c_sound:.0f}m/s, "
            f"n_substeps={self.config.n_acoustic_substeps}"
        )

    def tendencies(
        self,
        state: NonHydrostaticState,
        physics_tendency: NonHydrostaticTendencies | None = None,
    ) -> NonHydrostaticTendencies:
        return cdgrid_compressible_euler_slow_tendencies(
            state, self.grid, self.height_coord, self.terrain_metric,
            self.cdgrid, self.config, physics_tendency,
        )

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-20; mirrors iter-18 API)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-20; iter-19 API)."""
        self._target_mass = target_mass

    def compute_dry_mass(self, state: NonHydrostaticState) -> jax.Array:
        """Global dry mass ``∫ J · (rho_ref + rho') · dz · dA`` (fp64).

        iter-21: API parity with the MPAS NH (iter-8) and spectral NH
        (iter-9) ``compute_dry_mass`` helpers; underlying helper is
        ``core.conservation.compute_nh_dry_mass``.
        """
        return compute_nh_dry_mass(
            state.rho_prime.data, self.height_coord,
            self.terrain_metric, self.grid,
        )

    def step(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """Advance one step using split-explicit RK3 with C-D grid transport."""
        # NH CD-grid does not thread a PhysicsState carry — refuse a
        # stateful make_physics(model_type="nonhydrostatic") fn rather than
        # silently reseed its prognostic fields every step (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="NH CDGrid step()")
        # Precompute target mass outside JIT boundary
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = self.compute_dry_mass(state)
        return self._step_jitted(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jitted(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """JIT-compiled step core."""
        acoustic_cfg = CompressibleEulerConfig(
            g=self.config.g,
            n_acoustic_substeps=self.config.n_acoustic_substeps,
            semi_implicit_acoustic=self.config.semi_implicit_acoustic,
            acoustic_off_centering=self.config.acoustic_off_centering,
        )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            # iter-189: dt_actual for corner-div adaptive cap
            tend = cdgrid_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.cdgrid, self.config, phys, dt_actual=dt,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, acoustic_cfg,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, acoustic_cfg,
            )

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        # FV3_3D iter 169 (mirror of PE 12): post-step del-n vorticity damp (FV3 sw_core.F90:1948-1999)
        # NH stores u/v at centres: lift to corners → FV3 damp → project back
        if self.config.damp_v > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                fv3_del6_vorticity_damping,
            )
            # Lift cc (u, v) → D-grid corners (batched corner interp)
            u_cc_new = state_new.u.data
            v_cc_new = state_new.v.data
            n_face_dv, n_id_dv, n_jd_dv, nlev_dv = u_cc_new.shape
            _uv_cc_stack = jnp.stack([u_cc_new, v_cc_new], axis=-1)
            _uv_cc_flat = _uv_cc_stack.reshape(
                n_face_dv, n_id_dv, n_jd_dv, nlev_dv * 2,
            )
            _uv_d_flat = interp_center_to_corner(_uv_cc_flat, self.cdgrid)
            _uv_d = _uv_d_flat.reshape(
                _uv_d_flat.shape[0], _uv_d_flat.shape[1],
                _uv_d_flat.shape[2], nlev_dv, 2,
            )
            u_corner = _uv_d[..., 0]            # (6, n+1, n+1, nlev)
            v_corner = _uv_d[..., 1]

            # C-D corners → FV3 normal D-grid per level
            u_normal = 0.5 * (u_corner[:, :-1, :, :] + u_corner[:, 1:, :, :])
            v_normal = 0.5 * (v_corner[:, :, :-1, :] + v_corner[:, :, 1:, :])

            # FV3 damp coeff
            da_min_c = jnp.min(self.cdgrid.area_corner)
            damp_step = (self.config.damp_v * da_min_c) ** (
                self.config.nord_v + 1
            )

            # FV3_3D iter-1045: ``fv3_del6_vorticity_damping`` is now
            # 4D-native (3D static metrics broadcast via ``[..., None]``;
            # halo dispatched to ``pad_halo_4d``).  Direct call avoids
            # ``jax.vmap`` around ``pad_halo`` under MPI — same pattern
            # as iter-1042 / iter-1043 / iter-1044.
            du_normal, dv_normal = fv3_del6_vorticity_damping(
                u_normal, v_normal, damp=damp_step,
                nord=self.config.nord_v, cdgrid=self.cdgrid,
            )

            # FV3_3D iter 442/447: sponge boost of damp_v at k=0,1 (NOT k=2). FV3 damp_vt = 0.5*d2_divg.
            if self.config.use_fv3_sponge_damp_v:
                from legoesm.core.fv3_sponge_boost import (
                    apply_top_sponge_field_scale as _shared_scale_v,
                )
                du_normal = _shared_scale_v(
                    du_normal, self.config.damp_v,
                    self.config.nord_v, factor=0.5,
                    d2_bg=self.config.corner_div_damp_d2_bg,
                    d2_bg_k1=self.config.corner_div_damp_d2_bg_k1,
                    d2_bg_k2=self.config.corner_div_damp_d2_bg_k2,
                    apply_at_k2=False,
                )
                dv_normal = _shared_scale_v(
                    dv_normal, self.config.damp_v,
                    self.config.nord_v, factor=0.5,
                    d2_bg=self.config.corner_div_damp_d2_bg,
                    d2_bg_k1=self.config.corner_div_damp_d2_bg_k1,
                    d2_bg_k2=self.config.corner_div_damp_d2_bg_k2,
                    apply_at_k2=False,
                )

            # Project FV3 normal → corners (mode='edge' pad + avg)
            # FV3_3D iter 370 (mirror of PE 370): cross-face halo.
            #
            # iter-1046 fix: gate the `pad_halo_4d` call.  ``du_normal``
            # / ``dv_normal`` are NON-SQUARE ``(6, n, n+1, nlev)`` /
            # ``(6, n+1, n, nlev)`` fields.  The cubed-sphere
            # ``pad_halo_4d`` MPI path is built for square
            # ``(6, n, n, nlev)`` (uses precomputed connectivity tables
            # / ``interp_offsets`` of shape ``(6, 4, n)``); feeding
            # non-square data through the MPI scatter / interp helpers
            # crashes ``_place_strip_4d`` with a ``(n+1, nlev)`` vs
            # ``(n, nlev)`` broadcast mismatch.  Under the local
            # backend the non-square call DOES still produce a
            # measurable diff vs ``mode='edge'`` (the iter-370
            # regression test depends on this), so we can't drop the
            # call universally — only fall through to ``mode='edge'``
            # when the MPI backend is active without duogrid.  With
            # duogrid=True the post-pad remap reshapes correctly
            # under both backends.
            # FV3_3D iter-1083: route (du_normal, dv_normal) through
            # the dgrid vector halo.  Local backend: iter-1078
            # pad_halo_dgrid_vector_4d (full 24/24 bit-for-bit).
            # MPI backend: iter-1083 pad_halo_dgrid_vector_4d_replicated_mpi
            # (batched-per-peer sendrecv per
            # _pad_halo_mpi_face_only_4d pattern, deadlock-free at
            # np ∈ {2, 3, 6}; bit-for-bit against single-device
            # reference on owned faces).
            from legoesm.grids.halo import get_halo_backend as _ghb_nh
            if self.config.use_fv3_cross_face_du_proj:
                if _ghb_nh() == "mpi":
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

            # Project corner increments → cell centres (batched)
            _duv_corner = jnp.stack([du_corner, dv_corner], axis=-1)
            _duv_corner_flat = _duv_corner.reshape(
                _duv_corner.shape[0], _duv_corner.shape[1],
                _duv_corner.shape[2], nlev_dv * 2,
            )
            _duv_cc_flat = interp_corner_to_center(_duv_corner_flat)
            _duv_cc = _duv_cc_flat.reshape(
                n_face_dv, n_id_dv, n_jd_dv, nlev_dv, 2,
            )
            du_cc = _duv_cc[..., 0]
            dv_cc = _duv_cc[..., 1]

            # FV3_3D iter 209 (NH mirror of PE 208): KE→heat for damp_v. Δθ_p = -coeff*ΔKE/(c_pd*Π_ref)
            if self.config.damp_v_d_con > 0.0:
                from legoesm import constants
                # FV3_3D iter 339/344 (mirror of PE 338): metric-aware form (cosa_cell/rsin2_cell)
                if self.config.use_fv3_metric_aware_d_con:
                    ub_s = du_normal[:, :, :-1, :]
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
                    dKE_cc = 0.25 * rsin2_b * (
                        ub_s ** 2 + ub_n ** 2 + vb_w ** 2 + vb_e ** 2
                        + 2.0 * (gy_s + gy_n + gx_w + gx_e)
                        - cosa_b * (u2 * dv2 + v2 * du2 + du2 * dv2)
                    )
                else:
                    dKE_cc = (
                        u_cc_new * du_cc + 0.5 * du_cc ** 2
                        + v_cc_new * dv_cc + 0.5 * dv_cc ** 2
                    )
                # FV3_3D iter 337: optional dynamic Exner at the
                # post-acoustic damp_v_d_con site.  Mirrors iter-336
                # slow-tendency wiring but recomputes ``π'`` from the
                # current post-acoustic state (theta_p + rho_p) since
                # the slow_tendencies fn's ``pi_prime`` is out of
                # scope in ``step()``.
                _exner_ref_b1 = self.height_coord.exner_ref[
                    None, None, None, :
                ]
                if self.config.use_fv3_dynamic_exner:
                    _pi_prime_dv = compute_exner_perturbation(
                        state_new.rho_prime.data,
                        state_new.theta_prime.data,
                        self.height_coord,
                    )
                    _exner_eff_dv = _exner_ref_b1 + _pi_prime_dv
                else:
                    _exner_eff_dv = _exner_ref_b1
                _cx_dv = (
                    constants.c_vd if self.config.use_fv3_d_con_cv
                    else constants.c_pd
                )
                dtheta_p = -self.config.damp_v_d_con * dKE_cc / (
                    _cx_dv * _exner_eff_dv
                )
                # FV3_3D iter 431: sponge-zero d_con top N levels (FV3 dyn_core.F90:790/800/804)
                if self.config.d_con_top_zero_levels > 0:
                    _nlev_zd = dtheta_p.shape[-1]
                    _k_idx_zd = jnp.arange(_nlev_zd)
                    _d_con_mask = jnp.where(
                        _k_idx_zd < self.config.d_con_top_zero_levels,
                        0.0, 1.0,
                    )
                    dtheta_p = dtheta_p * _d_con_mask[None, None, None, :]
                # FV3_3D iter 218/219: |Δθ_p*Π| cap. NH cv_air: k=0 → 0.1×, k=1 → 0.5×, else 1×
                if self.config.delt_max > 0.0:
                    nlev = dtheta_p.shape[-1]
                    k_idx = jnp.arange(nlev)
                    sponge_factor = jnp.where(
                        k_idx == 0, 0.1,
                        jnp.where(k_idx == 1, 0.5, 1.0),
                    )
                    # FV3_3D iter 398: dyn_exner-aware cap
                    if self.config.use_fv3_dynamic_exner:
                        sf_b = sponge_factor[None, None, None, :]
                        delt_theta_b = (
                            dt * self.config.delt_max
                            * sf_b / _exner_eff_dv
                        )
                    else:
                        delt_theta_per_level = (
                            dt * self.config.delt_max
                            * sponge_factor
                            / self.height_coord.exner_ref
                        )
                        delt_theta_b = delt_theta_per_level[
                            None, None, None, :
                        ]
                    dtheta_p = jnp.clip(
                        dtheta_p, -delt_theta_b, delt_theta_b,
                    )
                state_new = state_new._replace(
                    u=state_new.u.replace(data=u_cc_new + du_cc),
                    v=state_new.v.replace(data=v_cc_new + dv_cc),
                    theta_prime=state_new.theta_prime.replace(
                        data=state_new.theta_prime.data + dtheta_p,
                    ),
                )
            else:
                state_new = state_new._replace(
                    u=state_new.u.replace(data=u_cc_new + du_cc),
                    v=state_new.v.replace(data=v_cc_new + dv_cc),
                )

        # FV3_3D iter 193 (mirror of iter-169 damp_v): post-step del-(2*(nord_w+1)) damp on w
        # FV3 sw_core.F90:1080-1086 d_sw1. w on half-levels — vmap del6_vt_flux.
        if self.config.damp_w > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                compute_del6_metrics, del6_vt_flux,
            )
            del6_u_w, del6_v_w = compute_del6_metrics(self.cdgrid)
            rarea_w = 1.0 / self.cdgrid.base.area    # (6, n, n)
            da_min_c_w = jnp.min(self.cdgrid.area_corner)
            damp_step_w = (self.config.damp_w * da_min_c_w) ** (
                self.config.nord_w + 1
            )

            # FV3_3D iter-1045: ``del6_vt_flux`` is now 4D-native.
            # Direct call on the full half-level field avoids ``jax.vmap``
            # around ``pad_halo`` under MPI.  Output ``fx2``/``fy2`` are
            # 4D (6, n+1, n, nlev_half) and (6, n, n+1, nlev_half).
            w_new_data = state_new.w.data        # (6, n, n, nlev_half)
            fx2_w, fy2_w = del6_vt_flux(
                w_new_data, damp=damp_step_w, nord=self.config.nord_w,
                del6_u=del6_u_w, del6_v=del6_v_w, rarea=rarea_w,
                cdgrid=self.cdgrid,
            )
            # FV3 flux convention: fx2[w]-fx2[e]; fy2[s]-fy2[n].
            # Broadcast 3D rarea_w against 4D net-flux.
            dw = (
                fx2_w[:, :-1, :, :] - fx2_w[:, 1:, :, :]
                + fy2_w[:, :, :-1, :] - fy2_w[:, :, 1:, :]
            ) * rarea_w[..., None]               # (6, n, n, nlev_half)

            # FV3_3D iter 441/447: sponge boost of damp_w at k=0/1/2. Factor 1.0 (FV3 damp_w = d2_divg).
            if self.config.use_fv3_sponge_damp_w:
                from legoesm.core.fv3_sponge_boost import (
                    apply_top_sponge_field_scale as _shared_scale,
                )
                dw = _shared_scale(
                    dw, self.config.damp_w, self.config.nord_w,
                    factor=1.0,
                    d2_bg=self.config.corner_div_damp_d2_bg,
                    d2_bg_k1=self.config.corner_div_damp_d2_bg_k1,
                    d2_bg_k2=self.config.corner_div_damp_d2_bg_k2,
                    apply_at_k2=True,
                )

            # FV3_3D iter 203: KE→heat for damp_w (FV3 sw_core.F90:1086).
            # heat_source = -d_con*dw*(w + 0.5dw); Δθ_p = heat/(c_pd*Π). Half-levels → full-levels.
            if self.config.damp_w_d_con > 0.0:
                from legoesm import constants
                heat_half = -self.config.damp_w_d_con * dw * (
                    w_new_data + 0.5 * dw
                )                                  # (6, n, n, nlev_half)
                heat_full = 0.5 * (
                    heat_half[..., :-1] + heat_half[..., 1:]
                )                                  # (6, n, n, nlev)
                # iter-207/337: ÷Π_ref (or Π_total when dynamic_exner). Recompute π' here.
                _exner_ref_b2 = self.height_coord.exner_ref[
                    None, None, None, :
                ]
                if self.config.use_fv3_dynamic_exner:
                    _pi_prime_dw = compute_exner_perturbation(
                        state_new.rho_prime.data,
                        state_new.theta_prime.data,
                        self.height_coord,
                    )
                    _exner_eff_dw = _exner_ref_b2 + _pi_prime_dw
                else:
                    _exner_eff_dw = _exner_ref_b2
                _cx_dw = (
                    constants.c_vd if self.config.use_fv3_d_con_cv
                    else constants.c_pd
                )
                dtheta_p = heat_full / (
                    _cx_dw * _exner_eff_dw
                )
                # FV3_3D iter 432: sponge-zero d_con top N levels (mirror of iter-431)
                if self.config.d_con_top_zero_levels > 0:
                    _nlev_zw = dtheta_p.shape[-1]
                    _k_idx_zw = jnp.arange(_nlev_zw)
                    _d_con_mask_w = jnp.where(
                        _k_idx_zw < self.config.d_con_top_zero_levels,
                        0.0, 1.0,
                    )
                    dtheta_p = dtheta_p * _d_con_mask_w[None, None, None, :]
                # FV3_3D iter 218/219: |Δθ_p*Π| cap. NH cv_air: k=0→0.1*, k=1→0.5*, else 1*
                if self.config.delt_max > 0.0:
                    nlev = dtheta_p.shape[-1]
                    k_idx = jnp.arange(nlev)
                    sponge_factor = jnp.where(
                        k_idx == 0, 0.1,
                        jnp.where(k_idx == 1, 0.5, 1.0),
                    )
                    # iter-398: dyn_exner-aware cap
                    if self.config.use_fv3_dynamic_exner:
                        sf_b = sponge_factor[None, None, None, :]
                        delt_theta_b = (
                            dt * self.config.delt_max
                            * sf_b / _exner_eff_dw
                        )
                    else:
                        delt_theta_per_level = (
                            dt * self.config.delt_max
                            * sponge_factor
                            / self.height_coord.exner_ref
                        )
                        delt_theta_b = delt_theta_per_level[
                            None, None, None, :
                        ]
                    dtheta_p = jnp.clip(
                        dtheta_p, -delt_theta_b, delt_theta_b,
                    )
                state_new = state_new._replace(
                    w=state_new.w.replace(data=w_new_data + dw),
                    theta_prime=state_new.theta_prime.replace(
                        data=state_new.theta_prime.data + dtheta_p,
                    ),
                )
            else:
                state_new = state_new._replace(
                    w=state_new.w.replace(data=w_new_data + dw),
                )

        if self.config.fix_mass:
            # _target_mass precomputed outside JIT
            target = self._target_mass if self.config.anchor_mass_to_initial else (
                compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            )
            state_new = fix_mass_nonhydrostatic(
                state_new, target, self.height_coord,
                self.terrain_metric, self.grid,
            )

        # FV3_3D iter 448: Ray_fast. Applied to u, v, w at end of step.
        if self.config.rf_tau_days > 0.0:
            from legoesm.core.fv3_rayleigh_fast import (
                compute_rff_profile, pfull_from_exner,
            )
            _pfull = pfull_from_exner(self.height_coord.exner_ref)
            _ptop = _pfull[0]   # traced scalar (top reference pressure)
            _rff = compute_rff_profile(
                _pfull, ptop=_ptop,
                rf_cutoff=self.config.rf_cutoff_pa,
                tau_days=self.config.rf_tau_days,
                dt=dt,
            )
            _rff_b = _rff[None, None, None, :]  # (1,1,1,nlev)
            state_new = state_new._replace(
                u=state_new.u.replace(
                    data=state_new.u.data * _rff_b,
                ),
                v=state_new.v.replace(
                    data=state_new.v.data * _rff_b,
                ),
            )
            # w on half-levels: apply rff[k] to w[k+1]; w[0] = model-top BC
            _rff_half = jnp.concatenate(
                [jnp.ones((1,)), _rff], axis=0,
            )                              # (nlev+1,)
            _rff_half_b = _rff_half[None, None, None, :]
            state_new = state_new._replace(
                w=state_new.w.replace(
                    data=state_new.w.data * _rff_half_b,
                ),
            )

        # FV3_3D iter 584: optional |w| safety cap (FV3 fv_mapz.F90:51 w_max=90).
        if self.config.w_safety_cap > 0.0:
            _w_max = self.config.w_safety_cap
            _w_min = (-self.config.w_safety_cap_min
                      if self.config.w_safety_cap_min > 0.0
                      else -_w_max)
            state_new = state_new._replace(
                w=state_new.w.replace(
                    data=jnp.clip(state_new.w.data, _w_min, _w_max),
                ),
            )

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)

    # integrate() and integrate_scan() inherited from IntegrationMixin


def make_fv3_component_fidelity_nh_config(
        **overrides) -> CDGridCompressibleEulerConfig:
    """Factory for the FV3 COMPONENT-FIDELITY NH config (iter 392 lineage).

    NAMING (phase 5 of the FV3-native roadmap): enables the individually
    FV3-validated COMPONENTS inside legoESM's stabilized solver.  NOT a
    full FV3 implementation: RK3 time integration (FV3: forward-backward
    acoustic splitting), interpolated halos, cell-centred state with a
    non-Lagrangian vertical (FV3: D-grid covariant winds + Lagrangian
    remapping).  Full-fidelity claims are gated on the phase-4 native core
    (tests/atmosphere/dycore/unit/test_fv3_naming_phase5.py).

    Pair with ``use_duogrid=True`` so iter-325 halo wiring + iter-370
    ``use_fv3_cross_face_du_proj`` transfer values.  Enables
    iter-320/328/336/339/370/431/436/437/451/459/441/442.

    FV3-fidelity flags:

    Enabled by default:

    - ``use_fv3_d_con_cv``: FV3 c_vd branch at d_con sites (iter-320,
      KE→heat conversion with c_v_air instead of c_p_air).
    - ``use_fv3_vector_halo_uv``: vector halo for u/v cell→corner
      interp (iter-328).
    - ``use_fv3_a2b_ord4_vector_uv``: 4th-order a2b corner interp
      for u/v (iter-698, -25.8% θ′ edge ratio at C8).
    - ``use_fv3_dynamic_exner``: dynamic ``Π = Π_ref + π'`` at all
      d_con sites + delt_max caps (iter-336).
    - ``use_fv3_metric_aware_d_con``: metric-aware d_con form
      (iter-339, ``cosa_s/rsin2`` at all 5 d_con sites).

    Disabled by default (opt-in via ``overrides``):

    - ``use_fv3_cross_face_du_proj``: cross-face halo for damp_v
      wind projection (iter-370).  Disabled as of iter-1072 — the
      non-square ``du_normal`` / ``dv_normal`` data (shapes
      ``(6, n+1, n, nlev)`` and ``(6, n, n+1, nlev)``) is silently
      corrupted by ``pad_halo_4d`` (probe verified: NORTH halo
      zeros).  Tracked as iter-1046 non-square halo follow-up.

    Plus production knobs: ``d_con_top_zero_levels``, ``delt_max``,
    ``nord_v``, ``corner_div_damp_nord``, ``corner_div_damp_d4_bg``,
    ``heat_source_del2_iters``, ``use_fv3_sponge_damp_v``,
    ``use_fv3_sponge_damp_w``.

    Pass ``overrides`` kwargs to override any default.
    """
    defaults = dict(
        use_fv3_d_con_cv=True,
        use_fv3_vector_halo_uv=True,
        use_fv3_a2b_ord4_vector_uv=True,   # iter-698: -25.8% θ′ edge ratio at C8
        use_fv3_dynamic_exner=True,
        use_fv3_metric_aware_d_con=True,
        # FV3_3D iter-1079: enabled by default.  Routes (du_normal,
        # dv_normal) through pad_halo_dgrid_vector_4d (iter-1078) —
        # all 24 directed cubed-sphere edges bit-for-bit FV3-faithful
        # via iter-1076 same-axis (16/24) + iter-1078 DGRID_NE
        # component swap (8/24 axis-swap edges).
        use_fv3_cross_face_du_proj=True,
        d_con_top_zero_levels=2,
        delt_max=1.0,
        nord_v=1,
        corner_div_damp_nord=1,
        corner_div_damp_d4_bg=0.16,
        heat_source_del2_iters=2,   # FV3 nf_ke at nord=1
        # iter-452: d2_bg_k1/k2 left at 0.0 — FV3 4.0/2.0 not portable
        use_fv3_sponge_damp_w=True,    # harmless when d2_bg_k* = 0
        use_fv3_sponge_damp_v=True,
    )
    defaults.update(overrides)
    return CDGridCompressibleEulerConfig(**defaults)


def make_fv3_faithful_nh_config(**overrides) -> CDGridCompressibleEulerConfig:
    """Deprecated alias of :func:`make_fv3_component_fidelity_nh_config`.

    The old name overclaimed: the configuration is component-fidelity, not
    a full FV3 implementation (phase-5 naming correction).
    """
    import warnings

    warnings.warn(
        "make_fv3_faithful_nh_config is a deprecated alias: the "
        "configuration is FV3 COMPONENT-fidelity (RK3 + interpolated "
        "halos + cell-centred, non-Lagrangian state), not a full FV3 "
        "implementation. Use make_fv3_component_fidelity_nh_config.",
        FutureWarning,
        stacklevel=2,
    )
    return make_fv3_component_fidelity_nh_config(**overrides)


def make_legoesm_nh_min_edge_config(**overrides) -> CDGridCompressibleEulerConfig:
    """FV3_3D iter 467: NH config for min cube-edge artifact in legoESM.

    iter-466 at C8+duogrid: turns OFF 3 flags hurting θ′ ratio:
    metric_aware_d_con (Δ=-1.6), heat_source_del2 (Δ=-1.4), d_con_top_zero (Δ=-0.6).
    Measured 50.4% reduction at C8+duogrid+3 steps. legoESM-specific calibration.
    """
    # Start from FV3-faithful factory, override 3 iter-466 hurters
    edge_min_overrides = dict(
        use_fv3_metric_aware_d_con=False,
        heat_source_del2_iters=0,
        d_con_top_zero_levels=0,
    )
    # User overrides > iter-466 overrides > factory defaults
    edge_min_overrides.update(overrides)
    return make_fv3_component_fidelity_nh_config(**edge_min_overrides)


def make_legoesm_nh_min_edge_aggressive_config(
    **overrides
) -> CDGridCompressibleEulerConfig:
    """FV3_3D iter 483: aggressive NH edge-min config.

    Stacks iter-466 hurting-flag drops + iter-481 corner_div boost
    (d2_bg=5e-2, 100× factory).  iter-482 measured 59.5% edge ratio
    reduction (3.50 → 1.42×) — composite is 18.7% better than
    min_edge alone.

    Trade-off: high corner_div_damp_d2_bg=5e-2 over-damps physical
    waves more than the factory default.  Use only when minimizing
    cube-edge artifacts at small scale (e.g., C8/C16 visualizations)
    is the priority, not climate accuracy.
    """
    aggressive_overrides = dict(
        use_fv3_metric_aware_d_con=False,
        heat_source_del2_iters=0,
        d_con_top_zero_levels=0,
        corner_div_damp_d2_bg=5e-2,
    )
    aggressive_overrides.update(overrides)
    return make_fv3_component_fidelity_nh_config(**aggressive_overrides)
