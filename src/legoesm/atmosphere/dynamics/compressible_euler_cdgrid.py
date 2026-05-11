"""FV3-inspired C-D grid Non-Hydrostatic Compressible Euler on the cubed-sphere.

**Fidelity status: stabilized research path, not a faithful FV3 port.**

Uses the same C-D grid discretisation as the shallow water and PE solvers:

* D-grid winds (cell corners) are prognostic (internal conversion from cell-centre).
* C-grid velocities (cell edges) are diagnosed for mass/scalar transport.
* Vorticity from circulation (exact on D-grid).
* Bernoulli/pressure gradient via Arakawa-Lamb at D-grid corners.
* Scalar transport (theta, rho, tracers) via C-grid upwind mass flux.
* Acoustic substeps for vertically propagating sound waves.

**State is stored at cell centres** for compatibility with the existing physics
infrastructure. Velocities are converted to D-grid for momentum computation
and back to cell-centre for output. A faithful FV3 NH path would store winds
on D-grid edges throughout and require a physics coupler that consumes D-grid
winds.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
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
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _laplacian_dgrid,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
    _sponge_profile,
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


class CDGridCompressibleEulerConfig(NamedTuple):
    """Configuration for the C-D grid non-hydrostatic CE model.

    This is the recommended cubed-sphere non-hydrostatic solver for
    production AMIP/CMIP simulations.  Uses FV3-style C-D grid staggering
    (Lin 2004, Putman & Lin 2007) which eliminates the Hollingsworth-Kallberg
    instability that affects cell-centre solvers.
    """
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
    acoustic_off_centering: float = 0.0   # Off-centering beta for acoustic damping
                                          # 0.0 = centered, 0.1 = recommended for long runs
    # FV3-faithful corner-divergence damping (FV3_3D iter 168).
    # Mirrors the iter-16/iter-18 wiring in
    # ``CDGridPrimitiveEquationConfig`` so the same FV3 mechanism
    # (sw_core.F90:1641-1822, ``d_sw5``) is available on the
    # non-hydrostatic 3D path.  Default 0.0 preserves baseline.
    corner_div_damp_d2_bg: float = 0.0
    corner_div_damp_dddmp: float = 0.20
    corner_div_damp_d4_bg: float = 0.0
    corner_div_damp_nord: int = 0
    corner_div_damp_fv3_vector_fill: bool = False
    corner_div_damp_dt_proxy: float = 10.0
    # Adaptive-cap dt scale for the FV3 ``min(0.20, dddmp*|delpc|*dt)``
    # branch.  PE path uses 200.0 (typical HS hybrid dt); the NH path
    # runs with much smaller outer dt under split-explicit acoustic
    # substepping, so the default here is 10.0 (typical NH outer dt).
    # The d2_bg floor dominates in HS-like regimes regardless; this
    # constant matters only when the adaptive cap is active.
    # FV3-faithful post-step del-n vorticity damping (FV3_3D iter 169).
    # Mirrors the iter-12 wiring in ``CDGridPrimitiveEquationConfig``.
    # Faithful port of FV3 ``sw_core.F90:1948-1999``: applied ONCE per
    # full timestep, AFTER the split-explicit acoustic update, as a
    # discrete wind correction ``u += fy2 / dx``.  Reuses the SW
    # backbone ``fv3_del6_vorticity_damping`` from
    # ``legoesm.core.fv3_del6_vt_flux``.
    damp_v: float = 0.0
    nord_v: int = 2
    # Order of the post-step vorticity damping (0=del-2, 1=del-4,
    # 2=del-6).  FV3 default is 2 (del-6).  Only active when
    # ``damp_v > 0``.
    # FV3-faithful KE→heat conversion for iter-169 damp_v (FV3_3D
    # iter 209, NH mirror of PE iter-208).  When ``damp_v`` removes
    # KE from (u, v) cell-centre winds via the post-step (du_cc,
    # dv_cc) increments, the lost KE is converted to heat in θ_p
    # for energy conservation:
    #   ΔKE_cc = u * du + 0.5*du² + v * dv + 0.5*dv²
    #   Δθ_p = -damp_v_d_con * ΔKE_cc / (c_pd * Π_ref)
    # Π_ref comes from ``HeightCoordinate.exner_ref`` (matches
    # iter-207 refinement of the iter-203 damp_w_d_con).  Default
    # 0.0 preserves baseline (gated INSIDE iter-169 ``damp_v > 0``).
    # FV3 production default is 1.0.
    damp_v_d_con: float = 0.0
    # FV3-faithful 4th-order A→B interpolation for ζ corner
    # (FV3_3D iter 170).  Mirrors the iter-14 wiring in
    # ``CDGridPrimitiveEquationConfig``.  Reuses the SW backbone
    # ``_interp_center_to_corner_a2b_ord4`` (port of FV3
    # ``a2b_edge.F90:a2b_ord4``).  When True, only the ζ_corner
    # step uses a2b_ord4; θ_corner stays with the legacy 2nd-order
    # 4-point average (iter-9 in PE established that swapping ALL
    # corner interpolations breaks discrete operator balance).
    # Default False preserves baseline.
    use_fv3_a2b_zeta_corner: bool = False
    # FV3-faithful cell-centre divergence damping (FV3_3D iter 171).
    # Mirrors the iter-5 wiring in ``CDGridPrimitiveEquationConfig``.
    # Faithful port of FV3 ``sw_core.F90:1720``::
    #
    #     damp = da_min_c * max(d2_bg, min(0.20, dddmp * |div|))
    #
    # where ``d2_bg = div_damp_coeff / da_min_c``.  When ``dddmp == 0``,
    # the damping is constant (``div_damp_coeff``) — matches the
    # legacy SW production path.  When ``dddmp > 0``, it becomes
    # adaptive (stronger at panel-boundary cells with large |div|).
    # Default 0.0 preserves baseline (no div damping at all in NH).
    div_damp_coeff: float = 0.0
    div_damp_dddmp: float = 0.0
    div_damp_d_con: float = 0.0
        # NH mirror of PE iter-223 cell-centre div_damp d_con
        # (FV3_3D iter 224).  When iter-171 cell-centre div_damp
        # removes KE from (u_d, v_d) via
        # ``du_d_dt += coeff * ddiv_dx``, the lost KE is converted
        # to heat in θ_p.  Heat tendency formula:
        #
        #     dKE/dt_corner = u_d * du_d_dt_dd + v_d * dv_d_dt_dd
        #     dθ_p/dt += -div_damp_d_con * (dKE/dt) /
        #                (c_pd * Π_ref)
        #
        # at corners, projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE
        # ``div_damp_coeff > 0``.
    # FV3_3D iter 173: opt-in async halo overlap for the
    # ``_arakawa_lamb_gradient(div_v)`` call in the iter-171
    # cell-centre div damping block.  Mirrors the PE wiring at
    # ``primitive_eq_cdgrid.py:594-598``.  When True AND the halo
    # backend is "mpi", dispatches to
    # ``_overlapped_arakawa_lamb_gradient`` which overlaps the halo
    # exchange with local computation; falls through to the standard
    # ``_arakawa_lamb_gradient`` on single-device or SPMD backends.
    # Default False preserves baseline.  No effect unless ``div_damp_coeff > 0``.
    use_async_halo: bool = False
    # FV3-style adaptive Smagorinsky A_h (FV3_3D iter 180).  Mirrors
    # the PE iter 57-59 wiring in ``CDGridPrimitiveEquationConfig``.
    # When > 0 AND ``A_h > 0``, an adaptive Smagorinsky coefficient
    # (proportional to the local strain rate × dx²) is added to the
    # static ``A_h`` value at each corner cell.  Reuses the existing
    # ``legoesm.core._smagorinsky_visc.compute_smagorinsky_ah_3d``
    # helper.  Default 0.0 preserves baseline; PE-tested useful range
    # ~0.1-0.4 (iter 60).  No effect when ``A_h == 0``.
    smagorinsky_cs: float = 0.0
    ah_d_con: float = 0.0
        # NH mirror of PE iter-225 Smagorinsky-A_h d_con (FV3_3D
        # iter 226).  When iter-180 A_h Laplacian removes KE from
        # (u_d, v_d) via ``du_d_dt += A_h_eff * lap_u``, the lost
        # KE is converted to heat in θ_p:
        #
        #     dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
        #     dθ_p/dt += -ah_d_con * (dKE/dt) / (c_pd * Π_ref)
        #
        # at corners, projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
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
    # Reuses the SW backbone ``_del6_vt_flux`` from
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
    # Conversion to ``θ_p`` (NH prognostic temperature):
    # ``ΔE_internal = c_pd * ΔT``; ``θ = T / Π``; therefore
    # ``Δθ ≈ ΔT / Π = heat / (c_pd * Π)``.  iter-207 uses
    # ``exner_ref`` (Π_ref at full levels from ``HeightCoordinate``)
    # for the conversion — FV3-faithful within the reference-state
    # linearization.  Π_ref is exact at p=p_ref (surface) and drops
    # to ~0.5 at the model top, so this refinement gives the right
    # heat partition across the column (vs the iter-203 Π=1
    # approximation that under-heated aloft).
    #
    # Heat is computed at half-levels (where ``w`` and ``dw`` live)
    # and averaged to full-levels for the ``θ_p`` increment.
    #
    # Default 0.0 preserves baseline bit-for-bit (Python-static
    # gate).  FV3 production default is ``d_con = 1.0``.  Active
    # only when ``damp_w > 0``.
    damp_w_d_con: float = 0.0
    delt_max: float = 0.0
        # FV3-faithful per-step cap on dissipative heating magnitude
        # (FV3_3D iter 218).  Faithful port of the FV3 ``delt_max``
        # limiter in ``dyn_core.F90:1774`` (cp branch).  In the NH
        # path our state stores potential temperature perturbation
        # ``θ_p ≈ T/Π`` rather than T, so the cap is applied
        # equivalently as ``|Δθ_p * Π_ref| ≤ dt * delt_max``.  Acts
        # on the iter-203 ``damp_w_d_con`` and iter-209
        # ``damp_v_d_con`` blocks.  Default 0.0 disables the cap
        # (preserves baseline bit-for-bit).  FV3 production default
        # is 1.0 K/s.  Differentiable via ``jnp.clip``.
    corner_div_damp_d_con: float = 0.0
        # NH mirror of PE iter-221 corner-div damp d_con (FV3_3D
        # iter 222).  When the iter-168 corner-divergence damping
        # removes KE from (u_d, v_d) via the tendency
        # ``du_d_dt -= ∇x(damp*delpc) / 2dx_corner``, the lost KE
        # is converted to heat in θ_p.  Heat tendency formula:
        #
        #     dKE/dt_corner = u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd
        #     dθ_p/dt += -corner_div_damp_d_con * (dKE/dt) /
        #                (c_pd * Π_ref)
        #
        # computed at corners then projected to cell centres via
        # ``_interp_corner_to_center``.  Π_ref from
        # ``HeightCoordinate.exner_ref`` (matches iter-207
        # refinement).  Default 0.0 preserves bit-for-bit baseline;
        # gated INSIDE ``corner_div_damp_d2_bg > 0``.  FV3
        # production default is 1.0.
    use_fv3_cross_face_du_proj: bool = False
        # FV3-faithful cross-face halo for iter-169 damp_v
        # post-step wind-increment projection back to corners
        # (FV3_3D iter 370, NH mirror of PE iter-370).  Default
        # mode='edge' (same-face); True uses pad_halo_4d
        # (duogrid-aware) for cross-face value.
        # **iter 384 finding**: flag is a NO-OP when grid was
        # constructed with ``use_duogrid=False`` (pair with
        # ``create_cubed_sphere(..., use_duogrid=True)`` for
        # actual cross-face VALUE transfer).
    use_fv3_metric_aware_d_con: bool = False
        # FV3-faithful metric-aware d_con KE→heat form for the
        # NH iter-209 ``damp_v_d_con`` site (FV3_3D iter 339, mirror
        # of PE iter-338).  Port of FV3 ``sw_core.F90:1956-1985``
        # metric form using cell-centre ``cosa_cell`` /
        # ``rsin2_cell`` non-orthogonality metrics + cell-centre
        # ``rdxa`` / ``rdya`` broadcast to edge stagger (1st-order;
        # FV3 has edge-native ``rdx`` / ``rdy``).  When True, the
        # NH damp_v_d_con site swaps the iter-209 simpler form for
        # the FV3-faithful metric form (FV3-faithful LOCAL heat
        # distribution at cube edges where ``cosa_s ≠ 0``).
        # Composes with ``use_fv3_d_con_cv`` (cv heat capacity) +
        # ``use_fv3_dynamic_exner`` (live Π).  Default False
        # preserves bit-for-bit baseline.
    use_fv3_dynamic_exner: bool = False
        # FV3-faithful DYNAMIC Exner factor for the NH d_con KE→heat
        # conversion (FV3_3D iter 336).  iter-207 used frozen
        # ``exner_ref`` (Π_ref at full levels from ``HeightCoordinate``)
        # in the d_con denominator ``c_x · Π_ref`` for the iter-203 /
        # 209 / 222 / 224 / 226 NH d_con sites — a reference-state
        # linearization that is ~30 % under-heating aloft (where
        # actual p deviates from p_ref).  FV3 NH path
        # (``dyn_core.F90:1769`` cv branch + line 1796) uses live
        # ``pkz`` computed from current pressure (``rdg * delp /
        # delz * pt``).  When True, the 3 SLOW-TENDENCY d_con sites
        # (corner_div, cell-centre div_damp, A_h) replace ``Π_ref``
        # with ``Π_total = Π_ref + π_prime`` where ``π_prime`` is
        # the Exner perturbation already computed at slow_tendencies
        # step 1 (line 382).  Default False preserves bit-for-bit
        # baseline.  Post-acoustic d_con sites (damp_v, damp_w) keep
        # ``Π_ref`` (no live π_prime in scope at step()-method post-
        # acoustic site without recomputation).  PE path uses actual
        # T (no Exner factor), so flag is NH-only.
        # **iter 394 audit**: ``Π_total = Π_ref + π'`` from
        # ``compute_exner_perturbation`` (in
        # ``atmosphere/dynamics/compressible_euler.py``) returns
        # the FULL NONLINEAR Exner — not a linearization.
        # ``π_total = (R_d · ρ · θ / p_0)^(R_d/c_v)`` algebraically
        # equals ``(p/p_0)^kappa`` (= FV3 ``pkz``) under EOS
        # ``p = R_d · ρ · T``.  So this flag's ``Π_total`` is the
        # SAME quantity as FV3's ``pkz``; no fidelity gap once
        # enabled (iter-336 / iter-337 sites also extended to
        # post-acoustic).
    d_con_top_zero_levels: int = 0
        # FV3-faithful sponge-layer zeroing of d_con KE→heat
        # (FV3_3D iter 431).  Port of FV3 ``dyn_core.F90:773-805``
        # ``d_con_k = 0`` for the top sponge levels (k=1,2,3
        # conditional on ``d2_bg_k1``/``d2_bg_k2``).  When > 0,
        # zeros the d_con heat tendency for the top N vertical
        # levels (model-top = lowest k-index after flip).  Default
        # 0 preserves bit-for-bit baseline (no zeroing).  FV3
        # production behaviour is N=1-3 depending on sponge
        # configuration.  Currently wired ONLY at the post-
        # acoustic damp_v d_con site (iter-209 mirror); remaining
        # 4 sites (damp_w, corner_div, div_damp, A_h) pending in
        # future iters to match FV3's ``d_con_k`` uniform
        # zeroing semantics.
    use_fv3_vector_halo_uv: bool = False
        # FV3-faithful vector halo for the cell-centre → D-grid corner
        # interpolation of (u, v) (FV3_3D iter 328).  Default False
        # uses ``_interp_center_to_corner`` on a passive stacked
        # ``(u, v)`` axis — this applies SCALAR halo (with duogrid
        # routing if active) but does NOT rotate the (u, v) face-local
        # components across cube-face boundaries.  At cube edges the
        # neighbouring face's e_x / e_y basis differs from the local
        # face's, so a scalar halo treats the components as untransformed
        # field values, leaving an O(1) basis-mismatch error at cube
        # edges that contributes directly to NH cube imprint in u, v.
        # When True, switches to ``center_to_dgrid_vector`` which uses
        # ``pad_halo_vector`` (FV3 ``ext_vector`` analogue at
        # ``fv_duogrid.F90:626-975``): rotates the (u, v) components
        # to the neighbouring face's basis BEFORE the 4-point average
        # to corners, preserving discrete vector continuity at cube
        # edges.  Default False preserves bit-for-bit baseline; opt
        # in for FV3-faithful vector halo at cube edges.  PE path
        # already stores winds at corners (no center-to-corner
        # interpolation, no vector halo gap).
    use_fv3_d_con_cv: bool = False
        # FV3-faithful heat-capacity factor for the NH d_con KE→heat
        # conversion (FV3_3D iter 320).  FV3 ``dyn_core.F90:1795``
        # divides ``heat_source`` by ``cv_air * delp`` in the
        # ``hydrostatic = .false.`` branch (cv = c_p − R_d ≈ 717 J/kg/K)
        # because compressible NH dynamics conserves total energy with
        # internal energy ``c_v · T`` (constant volume) — not enthalpy
        # ``c_p · T`` (constant pressure, hydrostatic limit).  legoESM
        # NH iter-203/207/209/222/224/226 ports inherited the simpler
        # ``c_pd`` denominator from PE, which UNDER-HEATS by ``c_v/c_p
        # ≈ 0.714`` (~40 % under-heating relative to FV3 NH).  When
        # ``True``, all 5 NH d_con sites (damp_w, damp_v, corner_div,
        # cell-centre div_damp, ah) divide by ``c_vd`` instead of
        # ``c_pd``, matching FV3's NH-branch convention.  Default
        # ``False`` preserves bit-for-bit baseline; opt in for
        # FV3-faithful heating partition.  PE path is unaffected (PE
        # uses ``c_pd`` which is FV3-faithful for the hydrostatic
        # branch ``cp_air``).


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
    """Compute slow (advective) tendencies using C-D grid operators.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    cdgrid : CubedSphereCDGrid
    config : CDGridCompressibleEulerConfig
    physics_tendency : NonHydrostaticTendencies, optional

    Returns
    -------
    NonHydrostaticTendencies
    """
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
    # FV3_3D iter 336: optional FV3-faithful dynamic Exner Π_total =
    # Π_ref + π' for the SLOW-TENDENCY d_con denominators.  When
    # ``use_fv3_dynamic_exner = False`` (default), keep frozen
    # ``Π_ref`` (iter-207 refinement); when True, use
    # ``Π_ref + π'`` per FV3 ``dyn_core.F90:1769`` ``cv_air`` /
    # ``cp_air`` branches that divide by live ``pkz``.  Cell-centre
    # broadcast for slow-tendency 4D sites.
    if config.use_fv3_dynamic_exner:
        _exner_eff_b = (
            height_coord.exner_ref[None, None, None, :]
            + pi_prime
        )
    else:
        _exner_eff_b = height_coord.exner_ref[None, None, None, :]

    # --- 2. Convert to D-grid ---
    # Stack (u, v) along a trailing axis and fold into the level dim so
    # a single ``_interp_center_to_corner`` (one halo exchange + one
    # 4-point average) handles both components, replacing two separate
    # calls each with their own halo.  Same passive-trailing-axis
    # pattern as the SH and divergence batching loops.
    # FV3_3D iter 328: when ``use_fv3_vector_halo_uv = True``, switch
    # to the vector-aware ``center_to_dgrid_vector`` which rotates
    # (u, v) face-local components across cube-face boundaries via
    # ``pad_halo_vector`` (FV3 ``ext_vector`` analogue at
    # ``fv_duogrid.F90:626-975``).  Default False preserves the
    # passive-stack scalar halo path bit-for-bit.
    n_face_uv, n_i_uv, n_j_uv, nlev_uv = u.shape
    if config.use_fv3_vector_halo_uv:
        u_d, v_d = center_to_dgrid_vector(u, v, cdgrid)
    else:
        _uv_stack = jnp.stack([u, v], axis=-1)  # (6, n, n, nlev, 2)
        _uv_flat = _uv_stack.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
        _uv_d_flat = _interp_center_to_corner(_uv_flat, cdgrid)
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
    # Pack K and pi_prime into a single halo exchange under SPMD/MPI;
    # under the local backend each operator does its own exchange (same
    # as before).  ``_arakawa_lamb_gradient`` takes ``padded=`` to skip
    # its internal halo when supplied.
    # FV3_3D iter 325: thread the duogrid kinked-to-extended remap
    # through the K + pi_prime packed halo so cube-edge gradient
    # values match the PE iter-84 path.  Without this the NH K and
    # pi_prime halos silently bypassed duogrid while PE applied it,
    # leaving the FV3 ``a2b_edge`` 4th-order corner accuracy partial
    # on the NH path (a real cube-imprint contributor at edges).
    _nh_dg = grid.duogrid
    from legoesm.grids.halo import _halo_backend as _hb_step6
    if _hb_step6 == "spmd":
        from legoesm.parallel.cubesphere_exchange import _spmd_mesh as _spmd_mesh_step6
        _K_pad_step6, _pi_pad_step6 = packed_pad_halo_4d(
            K, pi_prime, mesh=_spmd_mesh_step6, duogrid=_nh_dg,
        )
    elif _hb_step6 == "mpi":
        from legoesm.grids.halo import _mpi_topology as _mpi_topo_step6
        _K_pad_step6, _pi_pad_step6 = packed_pad_halo_mpi_4d(
            K, pi_prime, topology=_mpi_topo_step6, duogrid=_nh_dg,
        )
    else:
        _K_pad_step6 = _pi_pad_step6 = None

    # Batch the two Arakawa-Lamb gradients (K and pi_prime) into a
    # single call on a stacked tensor — the operator treats the
    # trailing axis as a passive batch (the 4-point finite difference
    # and the metric-matrix multiplication broadcast over the trailing
    # dim).  Stack the pre-padded inputs the same way so the local
    # backend (no pre-pad) issues only one halo exchange instead of
    # two.  2 gradient calls → 1.
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
    _dKpi_dx_flat, _dKpi_dy_perp_flat = _arakawa_lamb_gradient(
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
    # Interpolate ζ only; add f_corner directly (FV3 stores f at corners).
    # Previously abs_vor = ζ + f was interpolated as a single field; because
    # the 4-point interpolator is linear but sin(lat) is not,
    # interp(f_cc) ≠ f_corner introduced an O(dx²) Coriolis error at corners.
    # Matches iter-74 fix in cdgrid_momentum_tendencies.
    # FV3_3D iter 170: optionally use FV3-faithful 4th-order A→B
    # interpolation for ζ_corner (port of FV3 ``a2b_edge.F90:a2b_ord4``).
    # When False (default), use the legacy 2nd-order 4-point average.
    # Mirrors PE iter-14.
    #
    # FV3_3D iter 190: factor a2b_ord4(zeta) into a single local so
    # the iter-170 ``zeta_corner`` site AND the iter-187
    # ``_zeta_smag_corner`` site (FV3 smag_vort cap) share one halo-2
    # exchange when BOTH flags are active.  Mirrors the PE iter-190
    # change.
    _need_zeta_a2b_for_smag = (
        config.corner_div_damp_d2_bg > 0.0
        and config.corner_div_damp_d4_bg > 0.0
        and config.corner_div_damp_nord > 0
    )
    _need_zeta_a2b = config.use_fv3_a2b_zeta_corner or _need_zeta_a2b_for_smag
    _zeta_a2b_ord4: jax.Array | None = None
    if _need_zeta_a2b:
        from legoesm.core.operators_cdgrid import (
            _interp_center_to_corner_a2b_ord4,
        )
        # a2b_ord4 takes 3D shape (6, n, n) — vmap over level axis.
        if zeta.ndim == 4:
            _zeta_a2b_ord4 = jax.vmap(
                lambda lev: _interp_center_to_corner_a2b_ord4(lev, cdgrid),
                in_axes=-1, out_axes=-1,
            )(zeta)
        else:
            _zeta_a2b_ord4 = _interp_center_to_corner_a2b_ord4(zeta, cdgrid)

    if config.use_fv3_a2b_zeta_corner:
        zeta_corner = _zeta_a2b_ord4
    else:
        zeta_corner = _interp_center_to_corner(zeta, cdgrid)
    if config.use_coriolis:
        abs_vor_corner = zeta_corner + cdgrid.f_corner[..., None]
    else:
        abs_vor_corner = zeta_corner
    theta_corner = _interp_center_to_corner(theta_total, cdgrid)

    du_d_dt = abs_vor_corner * v_d - dK_dx - c_p * theta_corner * dpi_dx
    dv_d_dt = -abs_vor_corner * u_d - dK_dy_perp - c_p * theta_corner * dpi_dy_perp

    # FV3_3D iter 171: optional cell-centre divergence damping
    # (FV3 sw_core.F90:1720).  Mirrors the PE iter-5 wiring.  ``div_v``
    # is also used by the theta-equation advective-form correction
    # below (line ~520 in legacy ordering); compute once here so the
    # downstream consumer can reuse it.  Lazy: skip when neither
    # consumer needs it.
    _need_div_damp = config.div_damp_coeff > 0.0
    if _need_div_damp:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)         # (6, n, n, nlev)
        # iter-173: opt-in async-halo overlap on the gradient call
        # under MPI; falls through to the standard A-L gradient on
        # single-device / SPMD.  Mirrors the PE wiring.
        from legoesm.grids.halo import _halo_backend as _hb_div
        if config.use_async_halo and _hb_div == "mpi":
            from legoesm.core.operators_cdgrid import (
                _overlapped_arakawa_lamb_gradient,
            )
            ddiv_dx, ddiv_dy_perp = _overlapped_arakawa_lamb_gradient(
                div_v, cdgrid,
            )
        else:
            ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_v, cdgrid)
        if config.div_damp_dddmp > 0.0:
            # FV3 sw_core.F90:1720 adaptive Smagorinsky formulation.
            # ``da_min_c`` = global min B-grid corner area.
            _da_min_c = jnp.min(cdgrid.area_corner)
            _d2_bg = config.div_damp_coeff / _da_min_c
            _div_abs_corner = _interp_center_to_corner(
                jnp.abs(div_v), cdgrid,
            )
            _adaptive_coeff = _da_min_c * jnp.maximum(
                _d2_bg,
                jnp.minimum(
                    0.20, config.div_damp_dddmp * _div_abs_corner,
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
                _dKE_dt_cc_dd = _interp_corner_to_center(
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
    # ``_laplacian_dgrid`` call by stacking along a trailing axis and
    # folding into the level dim.  ``_laplacian_dgrid`` is now
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
        _uv_d_lap_out = _laplacian_dgrid(_uv_d_lap_flat, cdgrid).reshape(
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

        # FV3_3D iter 226: NH mirror of PE iter-225 A_h d_con.
        # Compute heat tendency and stash for accumulation into
        # dtheta_p_dt later.  Heat tendency formula (per second,
        # leading order; iter-207 Π_ref refinement):
        #
        #     dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
        #     dθ_p/dt += -ah_d_con * (dKE/dt) / (c_pd * Π_ref)
        if config.ah_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 352: NH mirror of PE iter-351 metric
                # form at A_h d_con site.
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
                _dKE_dt_cc_ah = _interp_corner_to_center(
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

    # FV3_3D iter 168: optional FV3-faithful B-grid corner-divergence
    # damping (FV3 d_sw5 sw_core.F90:1641-1822).  Direct port of the
    # iter-16/iter-18 wiring in
    # ``primitive_eq_cdgrid.py::fv3_hydrostatic_tendencies`` (lines
    # 635-771).  Uses the existing iter-15 helper
    # ``_fv3_divergence_corner.fv3_divergence_corner_3d`` and the
    # iter-18 ``fv3_corner_laplacian_iteration`` for the higher-order
    # nord >= 1 path.  Shares the exact FV3 formula:
    #
    #   delpc       = corner divergence (B-grid, sin_sg + corner removal)
    #   damp        = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt))
    #   divg_d_iter = (Laplacian)^nord(delpc)            [if nord > 0]
    #   dd8         = (da_min_c * d4_bg) ** (nord + 1)   [if d4_bg > 0]
    #   ke_corr     = damp * delpc + dd8 * divg_d_iter
    #   du -= grad_x(ke_corr) ; dv -= grad_y(ke_corr)
    #
    # Default ``corner_div_damp_d2_bg=0.0`` preserves baseline
    # bit-for-bit (Python-static branch).
    if config.corner_div_damp_d2_bg > 0.0:
        from legoesm.core._fv3_divergence_corner import (
            fv3_divergence_corner_3d,
        )
        # Step 1: B-grid corner divergence (Fortran ``delpc``).
        delpc = fv3_divergence_corner_3d(u_d, v_d, cdgrid)  # (6, n+1, n+1, nlev)

        # Step 2: adaptive damping coefficient at corners — FV3
        # sw_core.F90:1720 formula:
        #   damp = da_min_c * max(d2_bg, min(0.20, dddmp * |delpc| * dt))
        # NH path uses split-explicit acoustic substepping with much
        # smaller outer dt than the PE path; ``corner_div_damp_dt_proxy``
        # defaults to 10.0 (vs 200.0 in PE) to keep the adaptive cap
        # active at the right scale.  d2_bg floor dominates in HS-like
        # regimes regardless.
        # iter-189: prefer the actual integration ``dt`` (passed by
        # ``model.step`` as ``dt_actual=dt``) when provided.  Direct
        # callers without ``dt_actual`` fall back to ``config.dt_proxy``
        # — preserves backward compatibility for unit tests.
        _da_min_c = jnp.min(cdgrid.area_corner)
        _delpc_abs = jnp.abs(delpc)
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

        # Step 3: optional higher-order del-(2*(nord+1)) damping (FV3
        # ``nord > 0`` path, sw_core.F90:1725-1822).  Same Python-static
        # gate (``and``) as the PE path so when EITHER d4_bg == 0 OR
        # nord == 0 the iter-16-equivalent del-2-only path runs.
        if config.corner_div_damp_d4_bg > 0.0 and config.corner_div_damp_nord > 0:
            from legoesm.core._fv3_divergence_corner import (
                fv3_corner_laplacian_iteration,
            )
            _vfill = config.corner_div_damp_fv3_vector_fill

            def _lap_per_level(field_3d):
                return jax.vmap(
                    lambda lev: fv3_corner_laplacian_iteration(
                        lev, cdgrid, apply_vector_corner_fill=_vfill,
                    ),
                    in_axes=-1, out_axes=-1,
                )(field_3d)

            _delpc_initial = delpc
            _divg_d_iter = delpc
            for _ in range(config.corner_div_damp_nord):
                _divg_d_iter = _lap_per_level(_divg_d_iter)

            # FV3_3D iter 187: faithful port of FV3 sw_core.F90:1797-1809
            # smag_vort adaptive cap for nord >= 1 (FV3 uses |delpc|*dt
            # only at nord=0; nord >= 1 uses |dt|*sqrt(delpc² + ζ²)).
            # Reuses ``zeta`` (line 250, RELATIVE vorticity, not the
            # absolute Coriolis-augmented ``zeta_corner`` later) lifted
            # to corners via ``_interp_center_to_corner_a2b_ord4``
            # (FV3 ``a2b_ord4`` for ``wk → vort`` at line 1795).  Same
            # iter-181/183 double-where pattern guards sqrt(0) at rest.
            #
            # FV3_3D iter 190: reuse the ``_zeta_a2b_ord4`` precomputed
            # at the iter-170 site so both sites share a single halo-2
            # exchange when both flags are active.
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
            )                                              # (6, n+1, n+1, nlev)
        else:
            _ke_correction = _damp_corner * delpc          # (6, n+1, n+1, nlev)

        # Step 4: gradient at D-grid corners.  Halo-pad the
        # ke-correction so the i±1 / j±1 reads at face-boundary corners
        # pick up the neighbouring panel.  Centred difference at
        # corner (i, j); 2*dx denominator uses dxc / dyc averaged.
        # FV3_3D iter 325: route the ke_correction halo through the
        # duogrid kinked-to-extended remap so the cube-edge gradient
        # at the FV3 corner-divergence damping site matches the PE
        # iter-84 + iter-1184/1188 path.  Without duogrid, the corner
        # i+1 / i-1 reads at the cube edge see the cube-projected
        # halo cell instead of the duogrid-corrected value, leaving
        # an O(dx²) bias at panel boundaries that contributes to
        # cube imprint in u/v.
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

        # FV3_3D iter 222: NH mirror of PE iter-221 corner-div
        # damp KE→heat conversion.  Compute heat tendency at
        # corners then project to cell centres for later
        # accumulation into dtheta_p_dt.
        if config.corner_div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 348: NH mirror of PE iter-347 metric
                # form at corner_div_d_con site.  Project corners
                # to edge stagger via 2-pt average, apply iter-339
                # metric form structure at tendency rate.
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
                _dKE_dt_cc_cdd = _interp_corner_to_center(
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
    # Batch (du_d_dt, dv_d_dt) corner-to-center interp.  Same
    # passive-trailing-axis pattern; ``_interp_corner_to_center`` is a
    # 4-point average with no halo, so this saves one kernel launch.
    # Cell-centre shape captured at line 153 — ``_interp_corner_to_center``
    # outputs cell-centre.  Replaces undefined ``_at`` placeholders.
    _duv_d_dt = jnp.stack([du_d_dt, dv_d_dt], axis=-1)  # (..., 2)
    _duv_d_dt_flat = _duv_d_dt.reshape(
        _duv_d_dt.shape[0], _duv_d_dt.shape[1], _duv_d_dt.shape[2],
        nlev_uv * 2,
    )
    _duv_dt = _interp_corner_to_center(_duv_d_dt_flat).reshape(
        n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2,
    )
    du_dt = _duv_dt[..., 0]
    dv_dt = _duv_dt[..., 1]

    # --- 9. Vertical advection of u, v ---
    # Batch the two ``vertical_advection_height`` calls by stacking
    # (u, v) along a new leading axis.  ``w_full`` / ``w_star``
    # depend only on (w, dz, dz_half, J) so they are computed once
    # and the trailing-axis ``[..., :-1] - [..., 1:]`` gradient
    # broadcasts across the new axis.  Two passes through the
    # vertical-advection kernel collapse to one — same trailing/leading
    # axis batching as Loops 137/141.
    _uv_va = jnp.stack([u, v], axis=0)
    _uv_va_adv = vertical_advection_height(_uv_va, w, dz, dz_half, J)
    du_dt = du_dt + _uv_va_adv[0]
    dv_dt = dv_dt + _uv_va_adv[1]

    # --- 10. Theta equation: advective form -v·∇θ ---
    # The θ equation uses advective form (not divergence/flux form) because
    # θ is NOT a conserved density — it satisfies dθ/dt = 0, not ∂(ρθ)/∂t = -∇·(ρθv).
    # Advective form = flux divergence + θ·div(v):  -v·∇θ = -∇·(θv) + θ∇·v
    # iter-171: reuse div_v from the cell-centre div_damp block above
    # if it was already computed; otherwise compute lazily here.
    if div_v is None:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)

    # --- 10/11/12. (theta, rho, tracers) flux divergence (batched) ---
    # ``cgrid_mass_flux_divergence`` issues a ``pad_halo_4d`` on its
    # scalar input and runs the PPM reconstruction along the trailing
    # axis as a passive batch.  Stack ``(theta_total, rho_total)`` and
    # any prognostic tracers along that trailing axis, fold into the
    # level dim, and run a single PPM transport call instead of two.
    # ``u_c``, ``v_c``, and ``div_v`` are shared; ``jnp.repeat`` builds
    # the matching velocity broadcast for the interleaved (level ×
    # scalar) trailing axis.
    n_face_tr, n_i_tr, n_j_tr, nlev_tr = theta_total.shape
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    n_total = 2 + n_tracers  # theta + rho + tracers

    if n_tracers > 0:
        combined_stack = jnp.concatenate(
            [
                jnp.stack([theta_total, rho_total], axis=-1),  # (..., nlev, 2)
                tracers,  # (..., nlev, n_tracers)
            ], axis=-1,
        )  # (..., nlev, n_total)
    else:
        combined_stack = jnp.stack(
            [theta_total, rho_total], axis=-1,
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

    # FV3_3D iter 240 (NH mirror of PE iter-239): aggregate the
    # 3 tendency-based d_con contributions (iter-222 corner-div,
    # iter-224 cell-centre div_damp, iter-226 A_h) and apply the
    # iter-218/219 sponge-aware ``delt_max`` cap on the
    # AGGREGATE.  Mirrors FV3 ``dyn_core.F90:1764-1779``: all
    # KE-removal sources accumulate into a single heat_source
    # then dyn_core.F90 caps once with the per-level
    # (sponge-aware) ``delt_max`` limiter.  NH cap policy:
    # k=0 → 0.1× delt_max, k=1 → 0.5×, k≥2 → 1× (FV3 cv_air
    # branch sw_core.F90:1782-1786).  Cap is in θ_p space, so
    # divide by ``Π_ref`` to convert from a T-space ``delt_max``
    # bound (matches the iter-219 single-mechanism pattern).
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
    # Rho uses pure flux form: -∇·(ρv) + 0 (continuity).
    drho_p_dt = flux_combined[..., 1]

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
        # the bare ``pad_halo_4d`` call below was an F821 NameError
        # (no top-level import) that would crash this branch when
        # ``_lap_uvT_div2 > 0``.  The previous local alias
        # ``_pad_halo_4d_uvtr = _pad_halo_4d_module`` was assigned
        # but never used.
        _dg = getattr(grid, 'duogrid', None)
        _offsets = None if _dg is not None else grid.halo_interp_offsets
        _lap1_pad = _pad_halo_4d_module(_lap1, interp_offsets=_offsets, duogrid=_dg)
        _gx = gradient_x_3d(_lap1, grid, padded=_lap1_pad)
        _gy = gradient_y_3d(_lap1, grid, padded=_lap1_pad)
        _lap2 = divergence_3d(_gx, _gy, grid).reshape(
            n_face_h, n_i_h, n_j_h, nlev_h, 4,
        )
        # Apply per-field hyperdiffusion coefficients.
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
    sponge = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge * u
    dv_dt = dv_dt - sponge * v
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 15. w tendency (slow: horizontal advection) ---
    # Pre-pad ``w_full`` once and pass to both gradient_x_3d /
    # gradient_y_3d via ``padded=`` so they share the halo MPI exchange.
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    # iter-169: same F821 fix as above — bare ``pad_halo_4d`` would
    # NameError; use the imported ``_pad_halo_4d_module`` directly.
    _dg_w = getattr(grid, 'duogrid', None)
    _offsets_w = None if _dg_w is not None else grid.halo_interp_offsets
    _w_full_pad = _pad_halo_4d_module(w_full, interp_offsets=_offsets_w, duogrid=_dg_w)
    dw_dx = gradient_x_3d(w_full, grid, padded=_w_full_pad)
    dw_dy = gradient_y_3d(w_full, grid, padded=_w_full_pad)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    # Pad zero at top/bottom interfaces (rigid BC).  Single Pad HLO op
    # replaces alloc-zeros + scatter.
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
    """FV3-style C-D grid non-hydrostatic compressible Euler model.

    Parameters
    ----------
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : CDGridCompressibleEulerConfig, optional
    """

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
        self._target_mass = None

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

    def step(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3 with C-D grid transport.

        This non-jitted wrapper precomputes target mass outside the JIT
        boundary, then delegates to the jitted ``_step_jitted``.
        """
        # Precompute target mass outside JIT boundary (host-side only).
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = compute_nh_dry_mass(
                state.rho_prime.data, self.height_coord,
                self.terrain_metric, self.grid,
            )
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
            # FV3_3D iter 189: pass the actual integration dt so
            # the iter-168 / iter-187 corner-div damping adaptive cap
            # uses the real dt instead of ``config.corner_div_damp_dt_proxy``.
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

        # FV3_3D iter 169: optional post-step del-n vorticity damping
        # (FV3 sw_core.F90:1948-1999).  Direct port of the iter-12
        # wiring in ``primitive_eq_cdgrid.py:1361-1415``.  Reuses the
        # SW backbone ``fv3_del6_vorticity_damping``.  NH-specific
        # adjustment: NH stores u/v at CELL CENTRES, so we lift to
        # D-grid corners, run the FV3-normal-D-grid damping, then
        # project back to cell centres (PE stores at corners and skips
        # the cell-centre lift/project).  Default ``damp_v=0.0``
        # preserves baseline bit-for-bit (Python-static branch).
        if self.config.damp_v > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                fv3_del6_vorticity_damping,
            )
            # Lift cell-centre (u, v) to D-grid corners for the FV3
            # vorticity-damping step.  Stack & batch the corner interp
            # for a single halo collective + single 4-point average.
            u_cc_new = state_new.u.data
            v_cc_new = state_new.v.data
            n_face_dv, n_id_dv, n_jd_dv, nlev_dv = u_cc_new.shape
            _uv_cc_stack = jnp.stack([u_cc_new, v_cc_new], axis=-1)
            _uv_cc_flat = _uv_cc_stack.reshape(
                n_face_dv, n_id_dv, n_jd_dv, nlev_dv * 2,
            )
            _uv_d_flat = _interp_center_to_corner(_uv_cc_flat, self.cdgrid)
            _uv_d = _uv_d_flat.reshape(
                _uv_d_flat.shape[0], _uv_d_flat.shape[1],
                _uv_d_flat.shape[2], nlev_dv, 2,
            )
            u_corner = _uv_d[..., 0]            # (6, n+1, n+1, nlev)
            v_corner = _uv_d[..., 1]

            # Convert C-D corners (6, n+1, n+1, nlev) to FV3 normal
            # D-grid layout per level: u at v-interfaces (6, n, n+1,
            # nlev), v at u-interfaces (6, n+1, n, nlev).
            u_normal = 0.5 * (u_corner[:, :-1, :, :] + u_corner[:, 1:, :, :])
            v_normal = 0.5 * (v_corner[:, :, :-1, :] + v_corner[:, :, 1:, :])

            # FV3 damp coefficient (matches PE/SW pattern).
            da_min_c = jnp.min(self.cdgrid.area_corner)
            damp_step = (self.config.damp_v * da_min_c) ** (
                self.config.nord_v + 1
            )

            # Apply per-level via vmap.
            def _per_level(args):
                u_lev, v_lev = args
                return fv3_del6_vorticity_damping(
                    u_lev, v_lev, damp=damp_step,
                    nord=self.config.nord_v, cdgrid=self.cdgrid,
                )

            u_normal_t = jnp.moveaxis(u_normal, -1, 0)
            v_normal_t = jnp.moveaxis(v_normal, -1, 0)
            du_normal_t, dv_normal_t = jax.vmap(_per_level)(
                (u_normal_t, v_normal_t),
            )
            du_normal = jnp.moveaxis(du_normal_t, 0, -1)
            dv_normal = jnp.moveaxis(dv_normal_t, 0, -1)

            # Project wind increments from FV3 normal D-grid back to
            # corners (mode='edge' padding then averaging — inverse of
            # the corner→face averaging used at the start).
            # FV3_3D iter 370: optional cross-face halo (mirror of
            # PE iter-370 wiring).
            if self.config.use_fv3_cross_face_du_proj:
                _dg = self.grid.duogrid
                du_full = _pad_halo_4d_module(du_normal, duogrid=_dg)
                du_pad = du_full[:, :, 1:-1, :]
                dv_full = _pad_halo_4d_module(dv_normal, duogrid=_dg)
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

            # Project corner wind increments back to cell centres for
            # NH state storage.  Batch (du, dv) into a single
            # ``_interp_corner_to_center`` call.
            _duv_corner = jnp.stack([du_corner, dv_corner], axis=-1)
            _duv_corner_flat = _duv_corner.reshape(
                _duv_corner.shape[0], _duv_corner.shape[1],
                _duv_corner.shape[2], nlev_dv * 2,
            )
            _duv_cc_flat = _interp_corner_to_center(_duv_corner_flat)
            _duv_cc = _duv_cc_flat.reshape(
                n_face_dv, n_id_dv, n_jd_dv, nlev_dv, 2,
            )
            du_cc = _duv_cc[..., 0]
            dv_cc = _duv_cc[..., 1]

            # FV3_3D iter 209: optional KE→heat conversion for the
            # iter-169 damp_v wind increments (NH mirror of PE
            # iter-208).  ΔKE_cc = u*du + 0.5*du² + v*dv + 0.5*dv²;
            # heat = -damp_v_d_con * ΔKE_cc; Δθ_p = heat / (c_pd *
            # Π_ref) using exner_ref (matches iter-207 refinement).
            if self.config.damp_v_d_con > 0.0:
                from legoesm import constants
                # FV3_3D iter 339: optional metric-aware d_con form
                # (mirror of PE iter-338).  Uses edge-stagger
                # normalized variables + cell-centre rsin2 /
                # cosa_s metrics.  Edge-rdx/rdy approximated by
                # cell-centre rdxa/rdya broadcast (1st-order).
                if self.config.use_fv3_metric_aware_d_con:
                    # FV3_3D iter 339 (iter-344 scaling fix):
                    # metric-aware form using cell-centre rsin2 +
                    # cosa metrics directly on (du, dv) — drops
                    # the FV3 rdx/rdy normalization (cell-centre
                    # rdxa/rdya broadcast under-resolves at ~1e-6
                    # → numerical zero in float64).  Retains the
                    # FV3-faithful metric structure (cube-edge
                    # cosa_s correction).
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
                # FV3_3D iter 431: optional FV3-faithful sponge
                # zeroing of d_con heating in top N levels.
                # Mirrors FV3 ``dyn_core.F90:790/800/804``
                # ``d_con_k = 0`` for k=1,2,3 sponge levels.
                # ``d_con_top_zero_levels = 0`` (default) → no
                # change.
                if self.config.d_con_top_zero_levels > 0:
                    _nlev_zd = dtheta_p.shape[-1]
                    _k_idx_zd = jnp.arange(_nlev_zd)
                    _d_con_mask = jnp.where(
                        _k_idx_zd < self.config.d_con_top_zero_levels,
                        0.0, 1.0,
                    )
                    dtheta_p = dtheta_p * _d_con_mask[None, None, None, :]
                # FV3_3D iter 218/219: optional per-step cap on
                # |Δθ_p*Π| (equivalent to capping |ΔT| to
                # dt*delt_max in T-space).  FV3 dyn_core.F90:1782-
                # 1786 (cv_air branch) tightens the cap on the top
                # 2 sponge layers: k=1 → 0.1*delt, k=2 → 0.5*delt
                # (FV3 1-based; our 0-based: k=0 → 0.1*, k=1 → 0.5*).
                if self.config.delt_max > 0.0:
                    nlev = dtheta_p.shape[-1]
                    k_idx = jnp.arange(nlev)
                    sponge_factor = jnp.where(
                        k_idx == 0, 0.1,
                        jnp.where(k_idx == 1, 0.5, 1.0),
                    )
                    # FV3_3D iter 398: dyn_exner-aware cap (mirror
                    # of iter-397 aggregate cap fix).
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

        # FV3_3D iter 193: optional post-step del-(2*(nord_w+1))
        # damping of vertical velocity ``w``.  Faithful port of FV3
        # ``sw_core.F90:1080-1086`` (``d_sw1``):
        #     damp4 = (damp_w * da_min_c) ** (nord_w + 1)
        #     call del6_vt_flux(nord_w, ..., damp4, w, ..., fx2, fy2, ...)
        #     dw = (fx2[i,j] - fx2[i+1,j] + fy2[i,j] - fy2[i,j+1]) * rarea
        #     w += dw
        # Mirrors the iter-169 ``damp_v`` post-step pattern.  ``w`` in
        # the NH path is on half-levels (shape ``(6, n, n, nlev+1)``);
        # we vmap ``_del6_vt_flux`` over that axis.
        if self.config.damp_w > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                compute_del6_metrics, _del6_vt_flux,
            )
            del6_u_w, del6_v_w = compute_del6_metrics(self.cdgrid)
            rarea_w = 1.0 / self.cdgrid.base.area    # (6, n, n)
            da_min_c_w = jnp.min(self.cdgrid.area_corner)
            damp_step_w = (self.config.damp_w * da_min_c_w) ** (
                self.config.nord_w + 1
            )

            def _per_half_level(w_2d):
                # w_2d shape (6, n, n).  Returns (fx2, fy2).
                return _del6_vt_flux(
                    w_2d, damp=damp_step_w, nord=self.config.nord_w,
                    del6_u=del6_u_w, del6_v=del6_v_w, rarea=rarea_w,
                    cdgrid=self.cdgrid,
                )

            # Vmap over half-level axis.  state_new.w has shape
            # ``(6, n, n, nlev_half)``; move axis to leading.
            w_new_data = state_new.w.data        # (6, n, n, nlev_half)
            w_t = jnp.moveaxis(w_new_data, -1, 0)
            fx2_t, fy2_t = jax.vmap(_per_half_level)(w_t)
            # Reconstruct shapes per FV3 convention:
            #   fx2 shape (n+1, n)  → fx2[west] - fx2[east]
            #   fy2 shape (n, n+1)  → fy2[south] - fy2[north]
            # Both fluxes already include the ``damp`` factor (per
            # iter-937 in `_del6_vt_flux`).
            dw_t = (
                fx2_t[..., :-1, :] - fx2_t[..., 1:, :]
                + fy2_t[..., :-1] - fy2_t[..., 1:]
            ) * rarea_w[None, ...]               # (nlev_half, 6, n, n)
            dw = jnp.moveaxis(dw_t, 0, -1)       # (6, n, n, nlev_half)

            # FV3_3D iter 203: optional KE→heat conversion for the
            # damp_w wind change.  Faithful port of FV3
            # ``sw_core.F90:1086``::
            #
            #     heat_source = -d_con * dw * (w + 0.5*dw)
            #
            # which is the negative of ΔKE_w = w*dw + 0.5*dw².  When
            # damp_w removes KE from w, heat_source is positive and
            # the lost KE is added as heat to the temperature field
            # (energy conservation).  Conversion to θ_p uses the
            # simplified Δθ_p = heat / c_pd (Π Exner factor approxed
            # as 1.0; see iter-203 docstring for the trade-off).
            # Heat is at half-levels (with w, dw); average to
            # full-levels for the θ_p increment.
            if self.config.damp_w_d_con > 0.0:
                from legoesm import constants
                heat_half = -self.config.damp_w_d_con * dw * (
                    w_new_data + 0.5 * dw
                )                                  # (6, n, n, nlev_half)
                heat_full = 0.5 * (
                    heat_half[..., :-1] + heat_half[..., 1:]
                )                                  # (6, n, n, nlev)
                # iter-207: divide by Π_ref (exner_ref from
                # height_coord) in addition to c_pd, so ``Δθ ≈ ΔT/Π``
                # is FV3-faithful within the reference-state
                # linearization.  exner_ref is shape (nlev,);
                # broadcast to (6, n, n, nlev) by adding three
                # leading singleton axes.
                # FV3_3D iter 337: optional dynamic Exner at the
                # post-acoustic damp_w_d_con site.  Mirrors the
                # damp_v_d_con wiring above; recomputes ``π'`` from
                # the current state since slow_tendencies' pi_prime
                # is out of scope in ``step()``.
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
                # FV3_3D iter 218/219: optional per-step cap on
                # |Δθ_p*Π|.  Sponge-layer factors per FV3
                # dyn_core.F90:1782-1786 (cv_air branch):
                # k=0 → 0.1*delt, k=1 → 0.5*delt, else 1.0*delt.
                if self.config.delt_max > 0.0:
                    nlev = dtheta_p.shape[-1]
                    k_idx = jnp.arange(nlev)
                    sponge_factor = jnp.where(
                        k_idx == 0, 0.1,
                        jnp.where(k_idx == 1, 0.5, 1.0),
                    )
                    # FV3_3D iter 398: dyn_exner-aware cap.
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
            # _target_mass is precomputed in step() outside the JIT boundary.
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

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)

    # integrate() and integrate_scan() inherited from IntegrationMixin


def make_fv3_faithful_nh_config(**overrides) -> CDGridCompressibleEulerConfig:
    """FV3_3D iter 392: factory for FV3-faithful NH config.

    Enables every NH FV3-fidelity flag at production-recommended
    values.  Pair with ``create_cubed_sphere(..., use_duogrid=True)``
    so the iter-325 NH halo wiring + iter-370 cross_face flag
    actually transfer cross-face values.

    Includes:
        * ``use_fv3_d_con_cv = True`` (iter-320)
        * ``use_fv3_vector_halo_uv = True`` (iter-328)
        * ``use_fv3_dynamic_exner = True`` (iter-336/337)
        * ``use_fv3_metric_aware_d_con = True`` (iter-339/344)
        * ``use_fv3_cross_face_du_proj = True`` (iter-370)

    Parameters
    ----------
    **overrides
        Any config field can be overridden.

    Returns
    -------
    CDGridCompressibleEulerConfig
    """
    defaults = dict(
        use_fv3_d_con_cv=True,
        use_fv3_vector_halo_uv=True,
        use_fv3_dynamic_exner=True,
        use_fv3_metric_aware_d_con=True,
        use_fv3_cross_face_du_proj=True,
    )
    defaults.update(overrides)
    return CDGridCompressibleEulerConfig(**defaults)
