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
    fix_ps_mass,
    fix_ps_mass_target,
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
from legoesm.core.operators_cdgrid import _overlapped_arakawa_lamb_gradient
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
        # FV3_3D iter 33-35 finding: at C72+ resolutions the matrix's
        # ``_laplacian_visc_cube`` default UNDERESTIMATES A_h by ~10x.
        # The interior synoptic-scale unstable mode (iter-32 diagnosis)
        # needs A_h ~ 2.0e+07 at C72.  Iter 33 confirmed stability
        # with that value; iter 34 added LEGOESM_AH_SCALE env var to
        # the matrix as a workaround.  See FV3_3D.md iter 33-35.
    smagorinsky_cs: float = 0.0
        # FV3_3D iter 57/58 finding: opt-in Smagorinsky-style adaptive
        # Laplacian viscosity.  When > 0 (typical 0.1-0.4), an
        # additional A_h field is computed from the local strain-
        # rate tensor of (u, v) and added to the static ``A_h``:
        #   A_h_total(i,j) = A_h + smagorinsky_cs * dx² * |D|(i,j)
        # where ``|D| = sqrt(D11² + 2*D12² + D22²)``.  The adaptive
        # term auto-scales with local flow strain, addressing the
        # iter-51 codex meta-review concern that the iter-33
        # 10x-A_h calibration is case-specific.  Default 0.0 = off,
        # bit-for-bit baseline.  See FV3_3D.md iter 57/58 for the
        # closure derivation.
    ah_d_con: float = 0.0
        # FV3-faithful KE→heat conversion for the iter-57/58
        # Smagorinsky-augmented A_h Laplacian on (u_d, v_d)
        # (FV3_3D iter 225).  Mirrors iter-208/221/223 d_con but
        # for the A_h tendency.  Heat tendency formula (per
        # second, leading order in dt):
        #
        #     dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
        #     dT/dt += -ah_d_con * (dKE/dt) / c_pd
        #
        # at corners, projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE ``A_h > 0``.  FV3
        # production default is 1.0.
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
    use_fv3_lin_pgf: bool = False
        # FV3-faithful Lin (1997) cross-product hydrostatic
        # pressure-gradient force.  When True, the standard
        # ``(dB/dx + pg_corr_x)`` block (where ``B = KE + Φ`` and
        # ``pg_corr_x = R_d*T_corner*∇(ln p_s)``) is replaced by:
        #   1. Compute KE-only Bernoulli ``B_KE = KE``.
        #   2. ``dKE/dx, dKE/dy_perp`` via the existing A-L gradient.
        #   3. Compute the Lin (1997) cross-product PGF at C-grid faces
        #      via ``_fv3_lin_pgf.fv3_lin1997_pgf_3d_cgrid`` (faithful
        #      port of GFDL FV3 ``dyn_core.F90:p_grad_c``).
        #   4. Project to D-grid corners via the 2-point average in
        #      ``project_cgrid_pgf_to_dgrid_corners``.
        #   5. ``du_d/dt = ζ_corner*v_d - dKE/dx + pgf_x_d`` (the
        #      cross-product carries the correct Fortran sign so it is
        #      ADDED, not subtracted).
        # The Lin formulation gives EXACT hydrostatic cancellation by
        # construction — the cross-product is identically zero in any
        # column where (gz_W, pkc_W) = (gz_E, pkc_E), regardless of
        # discretisation.  Our existing split formulation cancels in
        # continuum but not at panel boundaries where halo
        # interpolation amplifies via the A-L Cartesian matrix.
        # FV3_3D iter 4 found that wiring this into RK3 produces a
        # CFL-incompatible scheme (max\|v\| 13× baseline by day 10,
        # NaN by day 30) because the cross-product PGF at C-grid
        # faces (projected to corners) is not in discrete balance
        # with the rotational ζ × v term at corners with the A-L
        # gradient.  The Lin PGF requires forward-backward time
        # stepping (FV3's native scheme) to be stable.  Currently
        # INERT — flag retained for the future forward-backward
        # iteration.
    div_damp_dddmp: float = 0.0
        # FV3-faithful adaptive Smagorinsky-style divergence damping
        # coefficient.  Faithful port of FV3 sw_core.F90:1720 formula
        # ``damp = da_min_c * max(d2_bg, min(0.20, dddmp * abs(div)))``
        # where ``d2_bg = div_damp_coeff / da_min_c`` is the
        # dimensionless background coefficient.  When ``dddmp > 0``,
        # divergence damping becomes ADAPTIVE: stronger where local
        # divergence is large (e.g., spurious divergence at panel
        # boundary cells from halo amplification) and weaker in the
        # smooth interior.  Default 0.0 preserves the existing
        # constant-coefficient behaviour.  FV3 default is 0.2; the
        # iter1009 dual-target SW calibration uses 0.0625 + factor 8
        # (no Smagorinsky).  Useful range for HS C36 hybrid:
        # 0.05-0.20 (test before raising).  FV3_3D iter 5.
    div_damp_d_con: float = 0.0
        # FV3-faithful KE→heat conversion for the iter-5 cell-
        # centre divergence damping (FV3_3D iter 223).  Mirrors
        # iter-208/221 d_con but for the cell-centre div_damp
        # tendency.  Heat tendency formula (per second, leading
        # order in dt):
        #
        #     dKE/dt = u_d * du_d_dt_dd + v_d * dv_d_dt_dd
        #     dT/dt += -div_damp_d_con * (dKE/dt) / c_pd
        #
        # at corners, then projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE
        # ``div_damp_coeff > 0``.  FV3 production default is 1.0.
    damp_v: float = 0.0
        # FV3-faithful POST-STEP del-n vorticity damping coefficient,
        # reusing the SW backbone ``fv3_del6_vorticity_damping`` from
        # ``legoesm.core.fv3_del6_vt_flux``.  Faithful port of
        # ``sw_core.F90:1948-1999``: applied ONCE per full timestep,
        # AFTER the main RK3 update, as a discrete wind correction
        # ``u += fy2 / dx``.  Same pattern as
        # ``shallow_water_fv3_cdgrid.py:1141``.  When ``damp_v > 0``,
        # the 3D step calls ``fv3_del6_vorticity_damping`` per level
        # (vmap over nlev) on FV3 normal-D-grid layout (averaging from
        # our (n+1, n+1) C-D corner state to (n, n+1) and (n+1, n)
        # FV3 face midpoints), then projects the resulting wind
        # increments back to corners by mode='edge' padding.  Default
        # 0.0 preserves the legacy 3D dycore behaviour.  Iter1009 SW
        # production uses ``damp_v=0.030``.  FV3_3D iter 12 adopts
        # this opt-in mechanism for the 3D path.
    nord_v: int = 2
        # Order of the post-step vorticity damping (0=del-2, 1=del-4,
        # 2=del-6).  FV3 default is 2 (del-6).  Used only when
        # ``damp_v > 0``.
    damp_v_d_con: float = 0.0
        # FV3-faithful KE→heat conversion for iter-12 ``damp_v``
        # damping (FV3_3D iter 208).  Port of the d_con block in
        # FV3 ``sw_core.F90:1953-1990``: when ``damp_v`` removes
        # KE from (u_d, v_d) via the post-step wind increments
        # (du_corner, dv_corner), the lost KE is converted to
        # heat in T (energy conservation):
        #
        #     ΔKE_per_mass = u_d * du + 0.5*du² + v_d * dv + 0.5*dv²
        #     ΔT = -damp_v_d_con * ΔKE / c_pd
        #
        # Computed at corners then projected to cell centres
        # via ``_interp_corner_to_center`` (4-point average) for
        # the T (cell-centre) update.  Default 0.0 preserves
        # baseline bit-for-bit (gated INSIDE the iter-12
        # ``damp_v > 0`` block).  FV3 production default is 1.0.
        # **iter 338 update**: opt-in port of the FV3 metric-aware
        # d_con form is now available via
        # ``use_fv3_metric_aware_d_con`` (see field below).  The
        # default ``False`` retains the iter-208 simpler form for
        # bit-for-bit baseline.
        # **Known fidelity gap (iter 238 audit, iter 324 update)**:
        # FV3's actual d_con formula at sw_core.F90:1980 uses a
        # metric-aware variant:
        # ``heat = -damp * rsin2 * (sum(ub², vb²) + 2*(gx+gy fluxes)
        # - cosa_s * cross_terms)`` where ``rsin2`` and ``cosa_s``
        # are the cubed-sphere C-grid non-orthogonality metrics.
        # Our simpler ``u·du + 0.5·du²`` form is equivalent in the
        # orthogonal-grid limit and conserves GLOBAL energy
        # exactly; LOCAL heat distribution differs at cube edges
        # where ``cosa_s ≠ 0``.  For HS/climate-mean diagnostics
        # this distinction is invisible.
        # **Prerequisites NOW available** (iter 324):
        # ``CubedSphereCDGrid.cosa_cell`` (FV3 ``cosa_s``) and
        # ``rsin2_cell`` (FV3 ``rsin2``) exist as cell-centre
        # fields.  See iter-323 for the symmetric c_pd→c_vd
        # FV3-fidelity port (NH only).
        # **FIDELITY GAP CLOSED** (iter 338/344 + iter 347-352):
        # opt-in ``use_fv3_metric_aware_d_con: bool = False`` flag
        # now wires the FV3-faithful ``cosa_cell``/``rsin2_cell``
        # metric form at ALL 8 PE+NH d_con sites (damp_v +
        # corner_div + div_damp + A_h × PE+NH).  Default False
        # preserves bit-for-bit iter-208 simpler form.  Edge-rdx /
        # rdy normalization dropped (cell-centre rdxa/rdya
        # numerical zero at C8); equivalent to iter-208 in the
        # orthogonal limit, adds cosa_s edge correction at cube
        # vertices.
        # **iter 246 audit (PE pkz factor)**: FV3 ``dyn_core.F90:
        # 1768`` divides ``heat_source`` by ``c_pd * delp * pkz``
        # to compute the per-step ΔT, where ``pkz`` is the local
        # Exner.  Our PE iter-208 port divides only by ``c_pd``
        # — no pkz factor, since PE's prognostic T is the actual
        # temperature (FV3's ``pt`` is ``c_p*T/pkz`` so the
        # division cancels the pkz internally).  The legoESM
        # PE T is layer-mean temperature directly, so no Exner
        # division is needed.  Verified consistent through
        # iter-228 bit-for-bit formula tests + iter-243 global
        # energy conservation.  No fidelity gap; the pkz factor
        # in FV3 is a state-variable convention difference.
        # NH has its own iter-209 damp_v_d_con + iter-203
        # damp_w_d_con (no PE w field, hydrostatic).
    delt_max: float = 0.0
        # FV3-faithful per-step cap on dissipative heating magnitude
        # (FV3_3D iter 218).  Faithful port of the FV3 ``delt_max``
        # limiter in ``dyn_core.F90:1774`` (cp branch):
        #
        #     pt += sign(min(|bdt*delt_max|, |dtmp|), dtmp) / pkz
        #
        # which clips the per-step temperature change ``dtmp`` to
        # magnitude ``bdt * delt_max`` (Kelvin), preserving the sign
        # of dtmp.  ``bdt`` is the model big timestep ``dt``.  Acts
        # on the iter-208 ``damp_v_d_con`` block.  Default 0.0
        # disables the cap (preserves baseline bit-for-bit when
        # ``damp_v_d_con > 0``).  FV3 production default is 1.0
        # K/s.  Differentiable everywhere via ``jnp.clip``.
        # iter-219 added FV3 sponge-aware behavior: PE skips the
        # cap entirely for k=0,1 (top 2 sponge layers, FV3 cp_air
        # branch) while applying it to k>=2; NH applies a tighter
        # cap (0.1*delt_max at k=0, 0.5* at k=1, 1* at k>=2).
    corner_div_damp_d_con: float = 0.0
        # FV3-faithful KE→heat conversion for the iter-16/18
        # corner-divergence damping (FV3_3D iter 221).  When the
        # corner-div mechanism removes KE from (u_d, v_d) via the
        # tendency ``du_d_dt -= ∇x(damp*delpc) / 2dx``, the lost KE
        # is converted to heat in T (energy conservation).  Mirrors
        # the iter-208 pattern but for corner-div instead of
        # damp_v.  Heat tendency formula (per second, leading
        # order; the 0.5*du² term is O(dt) and dropped in the
        # tendency form):
        #
        #     dKE/dt = u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd
        #     dT/dt += -corner_div_damp_d_con * (dKE/dt) / c_pd
        #
        # where du_d_dt_cdd, dv_d_dt_cdd are the corner-div damp
        # contribution to the wind tendency.  Computed at corners
        # then projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE the iter-16
        # ``corner_div_damp_d2_bg > 0`` block.  FV3 production
        # default is ``d_con = 1.0``.  The iter-218/219 sponge-
        # aware ``delt_max`` cap also applies to this heating term.
    use_fv3_cross_face_du_proj: bool = False
        # FV3-faithful cross-face halo for the iter-12 ``damp_v``
        # post-step wind-increment projection back to corners
        # (FV3_3D iter 370).  Default ``mode='edge'`` (same-face
        # extension) leaves an O(dx) bias at cube edges where the
        # neighbor face's du/dv value differs from the local face's.
        # When True, swaps to ``pad_halo_4d`` (with duogrid routing
        # if active) which reads the neighbor face's adjacent
        # column.  Approximation: pad_halo_4d's interpolation
        # assumes cell-centre stagger but du_normal is at edge
        # stagger — the cross-face value is off by 0.5 cell.
        # Better than mode='edge' (cross-face VALUE-aware) but not
        # bit-for-bit FV3 (which uses edge-native cubed_a2d_halo).
        # Default False preserves bit-for-bit baseline.
        # **iter 384 finding**: flag is a NO-OP when grid was
        # constructed with ``use_duogrid=False`` because non-duogrid
        # pad_halo's interp_offsets at edge stagger don't differ
        # from mode='edge' at this site.  For cross-face VALUES to
        # actually transfer across faces, pair with
        # ``create_cubed_sphere(..., use_duogrid=True)``.
    use_fv3_metric_aware_d_con: bool = False
        # FV3-faithful metric-aware d_con KE→heat form for the
        # iter-208 ``damp_v_d_con`` site (FV3_3D iter 338).  Port of
        # FV3 ``sw_core.F90:1956-1985`` block:
        #
        #     heat_cc = -0.25 * d_con * rsin2 * (
        #         sum_4_edges(ub², vb²) + 2*sum_4_edges(gy, gx)
        #         - cosa_s * (u2*dv2 + v2*du2 + du2*dv2))
        #
        # where ub = du * rdx, fy = u * rdx, gy = fy * ub (analogous
        # for vb, gx) and rsin2 / cosa_s are the cubed-sphere C-grid
        # non-orthogonality metrics at the cell centre.  The simpler
        # iter-208 form ``ΔKE = u·du + 0.5·du² + ...`` is equivalent
        # in the orthogonal-grid limit and conserves GLOBAL energy
        # exactly; the metric-aware form gives the FV3-faithful
        # LOCAL distribution at cube edges where ``cosa_s ≠ 0``.
        # When True, swaps the iter-208 form for the metric-aware
        # form at the damp_v_d_con site only (corner_div /
        # div_damp / A_h d_con sites stay with the simpler form
        # since FV3 has separate KE accounting paths for those).
        # Uses cell-centre ``rdxa / rdya`` broadcast to edge stagger
        # (1st-order approximation; FV3 has edge-native ``rdx /
        # rdy``).  Default False preserves bit-for-bit baseline.
    d_con_top_zero_levels: int = 0
        # FV3-faithful sponge-layer zeroing of d_con KE→heat
        # (FV3_3D iter 433, PE mirror of NH iter-431/432).
        # Port of FV3 ``dyn_core.F90:790/800/804`` ``d_con_k = 0``
        # for the top sponge levels.  When > 0, zeros the d_con
        # heat tendency for the top N vertical levels (model-
        # top = lowest k-index).  Default 0 preserves bit-for-
        # bit baseline.  Wired at all 4 PE d_con sites:
        # post-acoustic damp_v (mirror of NH iter-431) +
        # aggregate ``_d_con_sum`` covering 3 slow-tendency
        # contributions (corner_div, div_damp, A_h; mirror of
        # NH iter-432).
    use_fv3_a2b_zeta_corner: bool = False
        # FV3-faithful 4th-order A→B interpolation for the relative
        # vorticity ``ζ`` from cell centres to D-grid corners (the
        # rotational ζ × v term in the momentum tendency).  Reuses the
        # SW backbone ``_interp_center_to_corner_a2b_ord4`` from
        # ``operators_cdgrid.py`` (port of FV3 ``a2b_edge.F90:a2b_ord4``
        # — tensor-product 4th-order Lagrange).  When True, only the
        # ζ_corner step uses a2b_ord4; T_corner, hybrid_factor and
        # other corner interpolations stay with the legacy 2nd-order
        # 4-point average.  Default False preserves the legacy
        # behaviour.  iter-9 established that swapping ALL corner
        # interpolations breaks discrete operator balance (max winds
        # 2.7× larger); iter 14 tests whether a TARGETED swap of
        # zeta_corner only is stable.
    corner_div_damp_d2_bg: float = 0.0
        # FV3-faithful B-grid corner-divergence adaptive damping
        # coefficient (FV3 ``d2_bg`` parameter at sw_core.F90:1720).
        # When > 0, compute the FV3 ``divergence_corner`` (sw_core.F90:
        # 2124) via the iter-15 helper ``_fv3_divergence_corner.
        # fv3_divergence_corner_3d`` and apply an adaptive Smagorinsky-
        # style damping at D-grid corners:
        #
        #   delpc = corner divergence (B-grid)
        #   damp  = da_min_c * max(d2_bg, min(0.20, dddmp * |delpc| * dt))
        #   du   -= ∇_x(damp * delpc) at corners
        #
        # Faithful port of FV3 d_sw5 lines 1720-1724.  This pairs
        # naturally with the iter-12 post-step ``damp_v``: iter-12 is
        # the del-n vorticity damping; iter-16 is the in-step
        # corner-divergence damping.  Together they reproduce FV3's
        # canonical damping pair for cube-edge artifact suppression.
        #
        # Default 0.0 preserves baseline.  FV3 production default
        # is 0.0625 with dddmp = 0.2; iter-1009 SW uses (8 * 0.0625,
        # 0.0) — i.e. background-only without Smagorinsky.  Stable
        # range for our 3D path: 0.0 to ~0.005 (FV3's 0.0625 default
        # is too aggressive for our C-D + RK3 architecture).  iter-17
        # 30-day scan identified ``corner_div_damp_d2_bg = 0.0005``
        # as the OPTIMUM for HS C36 hybrid: -79 % mid-level cube
        # imprint, -35 % max\|v\|, mass drift ~6e-10.
    corner_div_damp_dddmp: float = 0.20
        # Companion Smagorinsky coefficient for ``corner_div_damp_d2_bg``.
        # Faithful FV3 default is 0.20 (sw_core.F90).  Active only
        # when ``corner_div_damp_d2_bg > 0``.
    corner_div_damp_d4_bg: float = 0.0
        # FV3-faithful HIGHER-ORDER corner-divergence damping
        # coefficient (FV3 ``d4_bg``, sw_core.F90:1809-1817).  When
        # > 0 AND ``corner_div_damp_nord > 0``, the iter-16 del-2
        # corner-divergence damping is supplemented with a del-
        # ``(2*(nord+1))`` term:
        #
        #   delpc = corner divergence (B-grid) at full-step
        #   divg_d = (Laplacian)^nord(delpc)
        #   damp2 = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt))
        #   dd8   = (da_min_c * d4_bg)^(nord+1)
        #   vort  = damp2 * delpc + dd8 * divg_d
        #   du   -= grad(vort)
        #
        # The Laplacian iteration is taken at the corner-staggered
        # B-grid, faithful to FV3's d_sw5 inner-loop sequence
        # (sw_core.F90:1747-1769): take gradient of divg_d to face
        # midpoints, take divergence back to corners, with
        # corner-removal at sw/se/ne/nw at each iteration.
        # FV3 production typical values: d4_bg=0.16, nord=2 (del-6
        # damping).  Default 0.0 + nord=0 preserves the iter-16
        # del-2-only behaviour.
    corner_div_damp_nord: int = 0
        # Order of the higher-order corner-divergence damping
        # iteration.  0 = del-2 only (iter-16 behaviour);
        # 1 = del-4 (one Laplacian iteration);
        # 2 = del-6 (two Laplacian iterations, FV3 d_sw5 default).
        # Active only when ``corner_div_damp_d4_bg > 0``.
    rf_tau_days: float = 0.0
        # FV3-faithful fast Rayleigh friction timescale (PE
        # mirror of NH iter-448).  Port of FV3
        # ``dyn_core.F90:2922-3020`` ``Ray_fast``::
        #     rff(k) = dt/(tau*86400) * sin²(π/2 · log(...))²
        #     rff(k) = 1 / (1 + rff(k))
        #     u_d, v_d *= rff   for pfull(k) < rf_cutoff_pa
        # ``tau`` in DAYS.  Default 0.0 = bit-for-bit baseline.
        # FV3 production default 0.0 (RF off); typical 5-15 days.
        # PE has no w; only u_d, v_d are damped.
    rf_cutoff_pa: float = 3000.0
        # Cutoff pressure for PE Rayleigh friction (FV3 default
        # ``rf_cutoff = 3.0e2`` Pa = 30 hPa).
    use_fv3_sponge_damp_v: bool = False
        # FV3-faithful sponge boost of PE ``damp_v`` (vorticity
        # damping) at top sponge levels (FV3_3D iter 443, PE
        # mirror of NH iter-442).  Ports FV3
        # ``dyn_core.F90:786-787, 796-797`` ``damp_vt = 0.5 *
        # d2_divg`` at sponge layers k=0, k=1 (FV3 does NOT
        # extend to k=2).  Linear ``damp^(nord_v+1)`` scaling
        # trick: ``(du, dv) *= (0.5 * boosted / damp_v)^
        # (nord_v+1)``.  Gated on ``corner_div_damp_d2_bg_k1``
        # (k=0) and ``corner_div_damp_d2_bg_k2 > 0.01`` (k=1).
        # Default False = bit-for-bit baseline.
    corner_div_damp_d2_bg_k2: float = 0.0
        # FV3-faithful per-level sponge boost of the corner-
        # divergence d2_bg coefficient at k=1 and k=2 (FV3_3D
        # iter 439).  Port of FV3 ``dyn_core.F90:792, 802``::
        #
        #     ! k=2 (1-based, our k=1 0-based):
        #     if (d2_bg_k2 > 0.01)
        #         d2_divg = max(d2_bg, d2_bg_k2)
        #     ! k=3 (1-based, our k=2 0-based):
        #     if (d2_bg_k2 > 0.05)
        #         d2_divg = max(d2_bg, 0.2 * d2_bg_k2)
        #
        # When > 0.01, the k=1 level damping coefficient is
        # overridden with ``da_min_c * max(d2_bg, d2_bg_k2)``.
        # When > 0.05, the k=2 level is also overridden with
        # ``da_min_c * max(d2_bg, 0.2 * d2_bg_k2)``.  FV3
        # production value is 2.0.  Default 0.0 preserves
        # bit-for-bit baseline.  Wired on PE only (NH pending).
    corner_div_damp_d2_bg_k1: float = 0.0
        # FV3-faithful per-level sponge boost of the corner-
        # divergence d2_bg coefficient at the topmost level
        # (FV3_3D iter 438).  Port of FV3 ``dyn_core.F90:780``::
        #
        #     d2_divg = max(0.01, d2_bg, d2_bg_k1)   ! k=1
        #
        # When > 0 (and ``corner_div_damp_d2_bg > 0``), the k=0
        # level (model top) of the corner-divergence adaptive
        # damping coefficient is OVERRIDDEN with
        # ``max(corner_div_damp_d2_bg, corner_div_damp_d2_bg_k1)`` —
        # ignoring the adaptive ``dddmp*|delpc|*dt`` term and the
        # 0.20 cap.  FV3 production default value is 4.0 (large
        # boost over typical interior d2_bg = 0.0625).  Default
        # 0.0 here preserves bit-for-bit baseline (no boost).
        # Future iters: extend to k=1 (``d2_bg_k2``), then to NH.
    corner_div_damp_fv3_vector_fill: bool = False
        # FV3-fully-faithful vector cube-vertex fill
        # (``fill_corners(vc, uc, VECTOR=true, DGRID=true)``,
        # sw_core.F90:1762).  Active path uses the wider FV3 D-grid
        # layout for vc / uc inside ``fv3_corner_laplacian_iteration``.
        # At ``nord = 1`` (the only currently-supported active path)
        # the cells fill_corners writes to are NOT read by the
        # divergence operator at nt = 0, so this flag is mathematically
        # a no-op and bit-for-bit preserves iter-18 behaviour
        # (proven by ``test_corner_laplacian_vector_fill_is_noop_for_nord1``
        # via 5 random seeds + 54 deterministic impulse positions +
        # nonuniform-metric stress test).  Iter 23+ may extend this
        # to nord >= 2 where the flag would have functional effect.
        # Default False — gives bit-for-bit iter-18.
    corner_div_damp_dt_proxy: float = 200.0
        # FV3_3D iter 188: parity with NH ``corner_div_damp_dt_proxy``
        # field.  PE outer dt is typically 50-200s for HS production
        # (see iter-72 LEGOESM_HS_CUBE_DT_CFL auto-mode); 200.0 was
        # the hardcoded value at the iter-16 wiring site (line 669)
        # since this default is the iter-33 ah_x10+dt=200 reference.
        # Used in BOTH the iter-16 nord=0 cap (``dddmp * |delpc * dt|``)
        # and the iter-187 nord >= 1 smag_vort cap
        # (``dddmp * |dt| * sqrt(delpc² + ζ²)``).  Default 200.0
        # preserves existing behaviour bit-for-bit.


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
    # FV3_3D iter 4 attempt: setting ``B = KE`` (drop Φ) and replacing
    # the split (∇B + pg_corr) with FV3 Lin (1997) cross-product PGF
    # gave a CFL-incompatible scheme — the cross-product PGF at C-grid
    # faces (projected to D-grid corners via 2-point average) is not in
    # discrete balance with the rotational ζ × v term computed at
    # corners with the A-L gradient.  Result on HS C36 hybrid: max\|v\|
    # 8 m/s by day 10 (vs 0.6 baseline), 27 m/s by day 15, NaN by day
    # 30.  See FV3_3D.md iteration 4 for detail.  Lin PGF requires
    # forward-backward time stepping (FV3's native scheme) for
    # stability with this 2-step balance — RK3 cannot recover the
    # cross-step cancellation.  ``config.use_fv3_lin_pgf`` is retained
    # for forward use but does not currently affect this function.
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
    # FV3_3D iter 14: optionally use FV3-faithful 4th-order A→B for
    # the zeta corner interpolation (SW-backbone reuse).  Default
    # uses the legacy 2nd-order 4-point average.
    #
    # FV3_3D iter 190: factor the a2b_ord4(zeta) into a single local
    # so the iter-170 ``zeta_corner_relative`` site AND the iter-187
    # ``_zeta_smag_corner`` site (FV3 smag_vort cap, sw_core.F90:1795)
    # share one halo-2 exchange when BOTH flags are active.  Closes
    # iter-187 codex review concern 3 (extra unmerged halo).
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
        zeta_corner_relative = _zeta_a2b_ord4
    else:
        zeta_corner_relative = _interp_center_to_corner(
            zeta, cdgrid, padded=_zeta_pad,
        )
    zeta_corner = zeta_corner_relative + cdgrid.f_corner[..., None]

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
        # FV3_3D iter 5: optional adaptive Smagorinsky-style damping.
        # Faithful port of sw_core.F90:1720
        # ``damp = da_min_c * max(d2_bg, min(0.20, dddmp * abs(div)))``.
        # When ``dddmp > 0``, the constant ``div_damp_coeff`` is
        # replaced by a per-cell coefficient that boosts damping at
        # cells with large |div| (typically the panel-boundary halo-
        # error cells driving the cube imprint).  Background floor
        # preserves smooth-interior behaviour.
        if config.div_damp_dddmp > 0:
            # Match the SW path's Fortran-faithful formulation in
            # ``cdgrid_momentum_tendencies`` (operators_cdgrid.py:1732).
            # ``da_min_c`` is the global minimum corner area
            # (Fortran ``gridstruct%da_min_c``); broadcast over levels.
            _da_min_c = jnp.min(cdgrid.area_corner)
            _d2_bg = config.div_damp_coeff / _da_min_c
            _div_abs_corner = _interp_center_to_corner(
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

        # FV3_3D iter 223: optional KE→heat d_con conversion for
        # the iter-5 cell-centre div_damp wind tendency (FV3
        # sw_core.F90 cell-centre div_damp d_con contribution).
        # Mirrors iter-221 corner-div d_con but uses the cell-
        # centre div_damp tendency instead.  Heat tendency formula
        # (per second, leading order in dt):
        #
        #     dKE/dt = u_d * du_d_dt_dd + v_d * dv_d_dt_dd
        #     dT/dt += -div_damp_d_con * (dKE/dt) / c_pd
        #
        # Default 0.0 preserves bit-for-bit baseline; gated INSIDE
        # the iter-5 ``div_damp_coeff > 0`` block.  FV3 production
        # default is 1.0.
        if config.div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 349: metric-aware form at PE
                # cell-centre div_damp d_con site.  Same pattern as
                # iter-347 corner_div.
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
                    * _interp_corner_to_center(_dKE_dt_corner_dd)
                    / constants.c_pd
                )
        else:
            _dT_dt_dd_cc = None
    else:
        _dT_dt_dd_cc = None

    # FV3_3D iter 16: optional adaptive B-grid corner-divergence
    # damping (FV3 d_sw5 sw_core.F90:1641-1724).  Faithful port using
    # iter-15's ``fv3_divergence_corner_3d`` helper.  This ADDS a
    # ke-correction at corners and applies its gradient to the
    # momentum tendency, matching the Fortran sequence
    # ``ke(i, j) += damp * delpc(i, j)`` followed by the d_sw1/d_sw6
    # KE-gradient update of (u, v).
    #
    # Differs from the existing cell-centre div_damp above (which
    # uses A-grid div_v): this ADDS the FV3 corner-staggered B-grid
    # divergence with sin_sg edge metric and corner removal, designed
    # to target panel-boundary halo amplification specifically.
    if config.corner_div_damp_d2_bg > 0.0:
        from legoesm.core._fv3_divergence_corner import (
            fv3_divergence_corner_3d,
        )
        # Step 1: B-grid corner divergence (Fortran ``delpc``).
        delpc = fv3_divergence_corner_3d(u_d, v_d, cdgrid)  # (6, n+1, n+1, nlev)

        # Step 2: adaptive damping coefficient at corners — FV3
        # sw_core.F90:1720 formula:
        #   damp = da_min_c * max(d2_bg, min(0.20, dddmp * |delpc| * dt))
        # Note: FV3 multiplies by dt because ``delpc`` is per-second
        # divergence and ``dddmp * delpc * dt`` is the dimensionless
        # CFL-scaled damping factor.  iter-189 plumbs the actual
        # integration ``dt`` (passed by ``model.step`` as
        # ``dt_actual=dt``) into both the iter-16 nord=0 cap and the
        # iter-187 nord >= 1 smag_vort cap.  Direct callers that
        # don't pass ``dt_actual`` fall back to the iter-188
        # ``config.corner_div_damp_dt_proxy`` (default 200.0, matching
        # the previously hardcoded value) — preserves backward
        # compatibility for unit tests calling the tendency function
        # directly.  The d2_bg floor dominates in HS regimes anyway
        # (see iter-5 adaptive analysis).
        _da_min_c = jnp.min(cdgrid.area_corner)
        _delpc_abs = jnp.abs(delpc)
        # iter-189: prefer the actual integration dt when provided.
        # The ``is not None`` check happens at trace time — production
        # callers always pass a value; direct test callers always pass
        # None.  No mixed call pattern, so JIT trace cache stays
        # single-entry per caller.
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

        # FV3_3D iter 438/439/446: per-level sponge boost at
        # k=0 / k=1 / k=2 via shared helper.  See
        # ``legoesm.core.fv3_sponge_boost`` for the FV3
        # ``dyn_core.F90:780, 792, 802`` port (k=0 from
        # d2_bg_k1; k=1 if d2_bg_k2>0.01; k=2 if d2_bg_k2>0.05
        # with 0.2 factor).
        from legoesm.core.fv3_sponge_boost import (
            apply_top_sponge_damp_boost as _shared_boost,
        )
        _damp_corner = _shared_boost(
            _damp_corner, _da_min_c,
            config.corner_div_damp_d2_bg,
            config.corner_div_damp_d2_bg_k1,
            config.corner_div_damp_d2_bg_k2,
        )

        # FV3_3D iter 18: optional higher-order del-(2*(nord+1))
        # damping (FV3 d_sw5 ``nord > 0`` path, sw_core.F90:1725-1822).
        # The corner-staggered ``delpc`` is iterated through a
        # B-grid Laplacian operator ``nord`` times, then mixed back
        # into the ke-correction with coefficient ``dd8``.
        #
        # FV3 formula (sw_core.F90:1809, 1817):
        #   dd8       = (da_min_c * d4_bg) ** (nord + 1)
        #   damp2     = da_min_c * max(d2_bg, min(0.20, dddmp * vort_smag))
        #   ke_corr   = damp2 * delpc_initial + dd8 * divg_d_iterated
        #
        # Bit-for-bit baseline guarantee: this branch is gated by a
        # Python-static ``and`` of two config knobs, so when EITHER
        # ``corner_div_damp_d4_bg == 0`` OR ``corner_div_damp_nord == 0``
        # the iter-16 del-2-only path runs unchanged.  No new code is
        # traced when the higher-order path is disabled.
        if config.corner_div_damp_d4_bg > 0.0 and config.corner_div_damp_nord > 0:
            from legoesm.core._fv3_divergence_corner import (
                fv3_corner_laplacian_iteration,
            )
            # Lift the per-level vmapped helper outside the loop so
            # the same compiled XLA primitive is reused across the
            # ``nord`` Python iterations.
            _vfill = config.corner_div_damp_fv3_vector_fill

            def _lap_per_level(field_3d):
                return jax.vmap(
                    lambda lev: fv3_corner_laplacian_iteration(
                        lev, cdgrid, apply_vector_corner_fill=_vfill,
                    ),
                    in_axes=-1, out_axes=-1,
                )(field_3d)

            _delpc_initial = delpc                          # FV3 ``delpc`` saved
            _divg_d_iter = delpc
            for _ in range(config.corner_div_damp_nord):
                _divg_d_iter = _lap_per_level(_divg_d_iter)

            # FV3_3D iter 187: faithful port of FV3 sw_core.F90:1797-1809
            # smag_vort adaptive cap for the nord >= 1 branch.  FV3
            # uses TWO different formulas for ``damp2``:
            #   nord = 0 (line 1722): damp = ... dddmp * |delpc * dt|
            #   nord >= 1 (line 1797-1809):
            #       wk_corner = a2b_ord4(zeta_relative)
            #       smag_vort = |dt| * sqrt(delpc² + wk_corner²)
            #       damp2     = ... dddmp * smag_vort
            # The legoESM 3D path previously used the nord=0 form for
            # ALL nord — faithful for nord=0 but NOT for nord >= 1.
            #
            # Uses RELATIVE vorticity ``zeta`` (line 442) — NOT the
            # absolute ``zeta_corner = zeta_relative + f_corner``
            # later in the tendency function.  FV3 ``wk`` is relative
            # vorticity (sw_core.F90:1789-1793 ``wk = circulation /
            # area``).  ``a2b_ord4`` lifts it to corners exactly as
            # FV3 does (sw_core.F90:1795 ``a2b_ord4(wk, vort, ...)``),
            # independent of the user-facing ``use_fv3_a2b_zeta_corner``
            # flag (which controls only the rotational term).
            #
            # Iter-181/183 double-where pattern: ``sqrt(0+0)`` has an
            # undefined gradient, breaking ``jax.grad`` at the rest
            # state.  Mask the branch so the backward pass passes
            # through the safe value.
            #
            # FV3_3D iter 190: reuse the ``_zeta_a2b_ord4`` computed
            # earlier (line ~528) — the iter-170 site and this iter-187
            # site share the same a2b_ord4(zeta) under bit-for-bit
            # equivalence.  When ``_need_zeta_a2b_for_smag`` is True
            # (which the surrounding gates establish) the variable is
            # guaranteed to be non-None at this site.
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

            # ``dd8`` follows the FV3 formulation exactly.  Cast through
            # delpc's dtype so that f32 / f64 paths stay consistent.
            _dd8 = jnp.asarray(
                (_da_min_c * config.corner_div_damp_d4_bg)
                ** (config.corner_div_damp_nord + 1),
                dtype=delpc.dtype,
            )
            _ke_correction = (
                _damp_corner * _delpc_initial + _dd8 * _divg_d_iter
            )                                               # (6, n+1, n+1, nlev)
        else:
            # iter-16 del-2 only path.  Bit-for-bit unchanged.
            _ke_correction = _damp_corner * delpc          # (6, n+1, n+1, nlev)

        # Step 4: gradient at D-grid corners.  FV3 normal D-grid uses
        # ``u(i, j) -= dt * (ke(i+1, j) - ke(i, j)) * rdxc`` — a
        # 2-point face-centred difference with u sitting between two
        # corners in i.  Our C-D grid has u_d AT the corner (i, j);
        # the natural translation is the centred difference using
        # padded corner values.  Halo-pad the ke-correction so the
        # i±1 / j±1 reads at face-boundary corners pick up the
        # neighbouring panel.
        # FV3_3D iter 333: route the PE ke_correction halo through
        # the duogrid kinked-to-extended remap so the cube-edge
        # gradient at the FV3 corner-divergence damping site matches
        # the FV3 ``fv_duogrid.F90`` Lagrange-extended halo.  PE-side
        # mirror of the NH iter-325 fix (same bypass: silent
        # cube-projected halo cells contributing O(dx²) bias at panel
        # boundaries).  Closes the symmetric PE/NH ke_correction halo
        # gap.
        _pe_dg_ke = grid.duogrid
        from legoesm.grids.halo import pad_halo_4d as _pad_halo_4d_fn
        _ke_pad = _pad_halo_4d_fn(
            _ke_correction, duogrid=_pe_dg_ke,
        )                                                  # (6, n+3, n+3, nlev)

        # Centred difference at corner (i, j) ∈ [0, n] × [0, n]:
        #   ∂x ke at (i, j) = (ke_pad[i+2, j+1] - ke_pad[i, j+1]) / (2*dx_at_corner)
        # In our padded layout (halo=1), corner index (i, j) maps to
        # padded[i+1, j+1]; (i-1, j) → padded[i, j+1]; (i+1, j) →
        # padded[i+2, j+1].  The 2*dx denominator uses dx_corner.
        _dke_dx_pad = (_ke_pad[:, 2:, 1:-1, :] - _ke_pad[:, :-2, 1:-1, :])
        _dke_dy_pad = (_ke_pad[:, 1:-1, 2:, :] - _ke_pad[:, 1:-1, :-2, :])

        # 2*dx and 2*dy at corners — use cdgrid.dxc / dyc averaged.
        # cdgrid.dxc is at u-faces (n+1, n); cdgrid.dyc is at v-faces
        # (n, n+1).  Corner dx/dy approximation: average two adjacent
        # face dxc/dyc.  For the centred difference's 2*dx
        # denominator at corner (i, j) we want the cell-centre-to-
        # cell-centre distance, ≈ 2 * (dx_corner / 2) = dx_corner
        # where dx_corner is the corner-to-corner distance (which is
        # the cell width).  Use cdgrid.base.dx as a sufficient
        # approximation (corner ≈ cell width on smooth grid).
        _dx_corner_uface = jnp.pad(
            cdgrid.dxc, [(0, 0), (0, 0), (0, 1)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _dy_corner_vface = jnp.pad(
            cdgrid.dyc, [(0, 0), (0, 1), (0, 0)], mode="edge",
        )                                                  # (6, n+1, n+1)
        _two_dx = 2.0 * _dx_corner_uface[..., None]         # (6, n+1, n+1, 1)
        _two_dy = 2.0 * _dy_corner_vface[..., None]

        # Subtract gradient (FV3 sign: u -= grad(ke), so as a tendency:
        # du/dt -= grad(ke_correction) / dt_approx → simplified to a
        # direct subtraction since the dt_approx cancels with our
        # tendency convention (the iter-16 damping is meant as a
        # per-step rate equivalent).
        _du_d_dt_cdd = -_dke_dx_pad / _two_dx
        _dv_d_dt_cdd = -_dke_dy_pad / _two_dy
        du_d_dt = du_d_dt + _du_d_dt_cdd
        dv_d_dt = dv_d_dt + _dv_d_dt_cdd

        # FV3_3D iter 221: optional KE→heat conversion for the
        # corner-divergence damping wind tendency.  FV3 sw_core.F90
        # line 1085-1086 accumulates ``ke_correction * delpc`` into
        # ``heat_source`` which then appears in ``dyn_core.F90``
        # line 1764-1779's d_con block.  Mirror in our tendency
        # form: heat tendency = -d_con * (u_d * du_dt + v_d * dv_dt)
        # / c_pd at corners, then projected to cell centres for the
        # T tendency.  Leading-order in dt; the 0.5*du² term that
        # FV3 carries in the discrete form is O(dt) and dropped in
        # this RK3-compatible tendency port.  Default 0.0
        # preserves bit-for-bit baseline.  The iter-218/219 sponge-
        # aware ``delt_max`` cap is applied below in the ``T``
        # tendency builder (section 12) where dT_dt is finalized.
        if config.corner_div_damp_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 347: metric-aware form at corner-div
                # d_con site.  Projects corner-stored u_d, v_d,
                # du_d_dt_cdd, dv_d_dt_cdd to edge stagger and
                # applies the iter-338 cosa/rsin2 metric correction
                # (same structure as damp_v_d_con but for tendency
                # rate per-sec).
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
                    * _interp_corner_to_center(_dKE_dt_corner_cdd)
                    / constants.c_pd
                )
        else:
            _dT_dt_cdd_cc = None

    else:
        _dT_dt_cdd_cc = None

    # --- 10b. Surface pressure tendency and vertical motion ---

    # Iter 1 gating (re-applied after merge): skip per-stage
    # ``zero_mean_tendency`` when end-step ``fix_mass`` is active.
    # The end-step mass fixer enforces conservation per step in a
    # single allreduce; per-stage zero-mean would add 3-4 allreduces
    # per SSP-RK3 step for an effect the end-step fixer corrects.
    _apply_zero_mean_per_stage = (
        config.zero_mean_ps_tendency
        and not (config.use_conservation_fixer and config.fix_mass)
    )

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
        if _apply_zero_mean_per_stage:
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
        if _apply_zero_mean_per_stage:
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

    # FV3_3D iter 239: aggregation moved to after the A_h block
    # so all 3 tendency-based d_con contributions
    # (corner-div, cell-centre div_damp, A_h) sum up and the
    # iter-218/219 sponge-aware ``delt_max`` cap is applied to
    # the AGGREGATE (FV3 sw_core.F90 + dyn_core.F90:1764-1779
    # accumulate heat_source from all sources then cap once).

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
        # FV3_3D iter 182: use the same JAX double-where trick as
        # iter 181 to make the gradient through ``sqrt`` finite at
        # zero strain (rest state).  Forward pass bit-for-bit
        # unchanged: at any nonzero ``u² + v²`` the result is exactly
        # ``sqrt(u²+v²)``; at zero exactly 0.  Backward pass:
        # gradient finite (zero) at rest state instead of NaN.
        _ws_sq = u_cell ** 2 + v_cell ** 2
        _safe_ws_sq = jnp.where(_ws_sq > 0.0, _ws_sq, 1.0)
        wind_speed = jnp.where(
            _ws_sq > 0.0, jnp.sqrt(_safe_ws_sq), 0.0,
        )
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
        # FV3_3D iter 58: optional Smagorinsky-style adaptive A_h
        # added on top of the static ``config.A_h`` constant.  When
        # ``config.smagorinsky_cs > 0``, compute the per-cell adaptive
        # field at corners and add it to the static value.
        if config.smagorinsky_cs > 0.0:
            from legoesm.core._smagorinsky_visc import (
                compute_smagorinsky_ah_3d,
            )
            _ah_smag_corner = compute_smagorinsky_ah_3d(
                u_d, v_d, cdgrid, config.smagorinsky_cs,
            )                                              # (6, n+1, n+1, nlev)
            # _lap_uv_d is at corners (same shape).  Combined:
            #   du/dt += (A_h + ah_smag) * lap_u
            _ah_eff_corner = config.A_h + _ah_smag_corner
            _du_d_dt_ah = _ah_eff_corner * _lap_uv_d[..., 0]
            _dv_d_dt_ah = _ah_eff_corner * _lap_uv_d[..., 1]
            du_d_dt = du_d_dt + _du_d_dt_ah
            dv_d_dt = dv_d_dt + _dv_d_dt_ah
            # T is at cell centers.  Interpolate ah_smag from corners
            # via 4-point average; add to static A_h for cell-centred
            # T tendency.
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

        # FV3_3D iter 225: optional KE→heat d_con conversion for
        # the iter-57/58 Smagorinsky-A_h Laplacian on (u_d, v_d).
        # Mirrors iter-208/221/223 d_con but for the A_h tendency.
        # Heat tendency formula (per second, leading order):
        #
        #     dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
        #     dT/dt += -ah_d_con * (dKE/dt) / c_pd
        #
        # at corners, projected to cell centres via
        # ``_interp_corner_to_center``.  Default 0.0 preserves
        # bit-for-bit baseline; gated INSIDE ``A_h > 0``.  FV3
        # production default is 1.0.
        if config.ah_d_con > 0.0:
            if config.use_fv3_metric_aware_d_con:
                # FV3_3D iter 351: metric-aware form at PE A_h d_con.
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
                    * _interp_corner_to_center(_dKE_dt_corner_ah)
                    / constants.c_pd
                )
        else:
            _dT_dt_ah_cc = None
    else:
        _dT_dt_ah_cc = None

    # FV3_3D iter 239: aggregate the 3 tendency-based d_con
    # contributions (iter-221 corner-div, iter-223 cell-centre
    # div_damp, iter-225 A_h) and apply the iter-218/219 sponge-
    # aware ``delt_max`` cap on the AGGREGATE.  Matches the FV3
    # heat_source pattern: all KE-removal mechanisms accumulate
    # into a single heat_source then dyn_core.F90 caps once with
    # the per-level (sponge-aware) ``delt_max`` limiter.
    _d_con_sum = None
    for _contrib in (_dT_dt_cdd_cc, _dT_dt_dd_cc, _dT_dt_ah_cc):
        if _contrib is not None:
            _d_con_sum = (
                _contrib if _d_con_sum is None
                else _d_con_sum + _contrib
            )
    if _d_con_sum is not None:
        # FV3_3D iter 433: optional FV3-faithful sponge zeroing
        # of d_con heating in top N levels — aggregate covers
        # 3 slow-tendency PE sites (corner_div, div_damp, A_h)
        # via single mask, mirror of NH iter-432.
        if config.d_con_top_zero_levels > 0:
            _nlev_zsp = _d_con_sum.shape[-1]
            _k_idx_zsp = jnp.arange(_nlev_zsp)
            _d_con_mask_sp = jnp.where(
                _k_idx_zsp < config.d_con_top_zero_levels,
                0.0, 1.0,
            )
            _d_con_sum = _d_con_sum * _d_con_mask_sp[None, None, None, :]
        if config.delt_max > 0.0:
            # Sponge-aware tendency cap: PE k=0,1 are uncapped
            # (jnp.inf), k>=2 are capped to ``delt_max`` K/s.
            # Equivalent to a per-step ΔT cap of
            # ``dt * delt_max`` after the integrator advances.
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

            # FV3_3D iter 189: pass the actual integration dt so
            # the iter-16 / iter-187 corner-div damping adaptive cap
            # uses the real dt instead of ``config.corner_div_damp_dt_proxy``.
            tend = fv3_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, cdgrid,
                self.config,
                physics_tendency=None,
                physics_tendency_cc=phys_cc,
                dt_actual=dt,
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

        # FV3_3D iter 12: optional post-step del-n vorticity damping
        # reusing the SW backbone ``fv3_del6_vorticity_damping``.
        # Faithful port of FV3 ``sw_core.F90:1948-1999``: applied ONCE
        # per full timestep AFTER the RK3 update, NOT inside the RK3
        # tendency function.  Same pattern as
        # ``shallow_water_fv3_cdgrid.py:1141``.
        if self.config.damp_v > 0.0:
            from legoesm.core.fv3_del6_vt_flux import (
                fv3_del6_vorticity_damping,
            )
            # Convert C-D grid winds at corners (6, n+1, n+1, nlev) to
            # FV3 normal D-grid layout per level: u at v-interfaces
            # (6, n, n+1, nlev), v at u-interfaces (6, n+1, n, nlev).
            # Average pairs of corners along the appropriate axis.
            u_corner = state_new.u_d.data
            v_corner = state_new.v_d.data
            u_normal = 0.5 * (u_corner[:, :-1, :, :] + u_corner[:, 1:, :, :])
            v_normal = 0.5 * (v_corner[:, :, :-1, :] + v_corner[:, :, 1:, :])

            # Compute the FV3 damp coefficient (matches SW pattern).
            # da_min_c = global min of B-grid corner area.
            da_min_c = jnp.min(self.cdgrid.area_corner)
            damp_step = (self.config.damp_v * da_min_c) ** (
                self.config.nord_v + 1
            )

            # Apply per-level via vmap.  The 2D del6_vt_flux function
            # operates on (6, n, n+1) and (6, n+1, n) per call.
            def _per_level(args):
                u_lev, v_lev = args
                return fv3_del6_vorticity_damping(
                    u_lev, v_lev, damp=damp_step,
                    nord=self.config.nord_v, cdgrid=self.cdgrid,
                )

            u_normal_t = jnp.moveaxis(u_normal, -1, 0)  # (nlev, 6, n, n+1)
            v_normal_t = jnp.moveaxis(v_normal, -1, 0)  # (nlev, 6, n+1, n)
            du_normal_t, dv_normal_t = jax.vmap(_per_level)(
                (u_normal_t, v_normal_t),
            )
            du_normal = jnp.moveaxis(du_normal_t, 0, -1)
            dv_normal = jnp.moveaxis(dv_normal_t, 0, -1)

            # FV3_3D iter 443/447: PE mirror of NH iter-442 via
            # shared helper.  FV3 ``damp_vt = 0.5 * d2_divg`` at
            # k=0, k=1 (NOT k=2).
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

            # Project wind increments from FV3 normal D-grid back to
            # corners (6, n+1, n+1, nlev) by mode='edge' padding then
            # averaging — the inverse of the corner→face averaging
            # used at the start.  At the cube-face boundary the edge
            # repeat preserves the increment magnitude.
            # FV3_3D iter 370: optional cross-face halo for
            # du_normal/dv_normal via pad_halo_4d (duogrid-aware)
            # instead of same-face mode='edge'.
            if self.config.use_fv3_cross_face_du_proj:
                from legoesm.grids.halo import pad_halo_4d as _pad_h4
                _dg = self.grid.duogrid
                du_full = _pad_h4(du_normal, duogrid=_dg)
                # (6, n+2, n+3, nlev) → slice axis=2 to (n+1)
                du_pad = du_full[:, :, 1:-1, :]
                dv_full = _pad_h4(dv_normal, duogrid=_dg)
                # (6, n+3, n+2, nlev) → slice axis=1 to (n+1)
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

            # FV3_3D iter 208: optional KE→heat conversion for the
            # iter-12 damp_v wind increments.  Faithful port of FV3
            # sw_core.F90:1953-1990 d_con block.  Corner-div, cell-
            # centre div_damp, and Smagorinsky-A_h d_con are now
            # also ported (iter-221, iter-223, iter-225 PE; iter-222,
            # iter-224, iter-226 NH); see the d_con knob cluster in
            # ``CDGridPrimitiveEquationConfig`` / ``...EulerConfig``.
            #
            # KE change per unit mass at corners:
            #     ΔKE = u_d * du + 0.5*du² + v_d * dv + 0.5*dv²
            # Heat = -ΔKE; ΔT = -damp_v_d_con * ΔKE / c_pd.
            # T is at cell centres → project ΔKE_corner via
            # ``_interp_corner_to_center``.
            if self.config.damp_v_d_con > 0.0:
                from legoesm.core.operators_cdgrid import (
                    _interp_corner_to_center,
                )
                if self.config.use_fv3_metric_aware_d_con:
                    # FV3_3D iter 338 (iter-344 scaling fix):
                    # metric-aware d_con form using cell-centre
                    # ``rsin2_cell`` + ``cosa_cell`` non-orthogonality
                    # correction.  Drops the FV3 rdx/rdy normalization
                    # (which our cell-centre rdxa/rdya broadcast
                    # under-resolves at ~1e-6 magnitude → numerical
                    # zero in float64).  Retains the
                    # FV3-faithful metric structure:
                    #     dKE = rsin2 * (
                    #         sum_4_edges(du², dv²) +
                    #         2*sum_4_edges(u·du, v·dv) -
                    #         cosa_s * (u*dv + v*du + du*dv crosses))
                    # Equivalent to iter-208 simpler form in the
                    # orthogonal-grid limit (cosa_s = 0, rsin2 = 1);
                    # adds cube-edge non-orthogonality correction
                    # via cosa_s ≠ 0 + rsin2 > 1.
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
                    dKE_cc = _interp_corner_to_center(dKE_corner)
                    dT = -self.config.damp_v_d_con * dKE_cc / constants.c_pd
                # FV3_3D iter 433: optional FV3-faithful sponge
                # zeroing of d_con heating in top N levels (PE
                # mirror of NH iter-431).
                if self.config.d_con_top_zero_levels > 0:
                    _nlev_zv = dT.shape[-1]
                    _k_idx_zv = jnp.arange(_nlev_zv)
                    _d_con_mask_v = jnp.where(
                        _k_idx_zv < self.config.d_con_top_zero_levels,
                        0.0, 1.0,
                    )
                    dT = dT * _d_con_mask_v[None, None, None, :]
                # FV3_3D iter 218/219: optional per-step cap on |dT|.
                # FV3 dyn_core.F90:1764-1776 (cp_air branch) skips
                # the cap entirely for the top 2 sponge layers
                # (``k<3`` in FV3's 1-based indexing, ``k<2`` in our
                # 0-based indexing where ``k=0`` is the model top).
                # delt_max=0 disables the cap globally.
                if self.config.delt_max > 0.0:
                    nlev = dT.shape[-1]
                    k_idx = jnp.arange(nlev)
                    # Top 2 sponge layers get an effectively infinite
                    # cap (no clipping); the rest get dt*delt_max.
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

        # Implicit gravity wave damping — post-step Laplacian diffusion on p_s.
        if self.config.implicit_grav_wave_damping > 0:
            alpha = self.config.implicit_grav_wave_damping
            lap_ps = laplacian_compact(state_new.p_s.data, self.grid)
            p_s_damped = state_new.p_s.data + alpha * dt * lap_ps
            p_s_damped = jnp.maximum(p_s_damped, self.config.p_floor)
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_damped),
            )

        # Conservation fixer (operates on p_s which is at cell centres,
        # in *both* hydrostatic and FV3 D-grid layouts).  Iter 2 routes
        # through the raw-array ``fix_ps_mass`` / ``fix_ps_mass_target``
        # helpers in :mod:`legoesm.core.conservation`, which act
        # directly on the surface-pressure data and skip the
        # ``fv3_to_hydrostatic`` round-trip — that round-trip would run
        # an expensive corner-to-centre halo exchange + 4-point average
        # on (u_d, v_d) only to discard the wind result, since only
        # ``p_s`` is fixed here.
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                # _target_mass is precomputed in step() outside the JIT boundary.
                p_s_fixed = fix_ps_mass_target(
                    state_new.p_s.data, self._target_mass, self.grid,
                )
            else:
                p_s_fixed = fix_ps_mass(
                    state_new.p_s.data, state.p_s.data, self.grid,
                )
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_fixed),
            )

        # FV3_3D iter 449: optional PE FV3 ``Ray_fast`` (PE
        # mirror of NH iter-448).  Reference profile pfull(k)
        # = (A_full[k] + B_full[k]) * p_ref.
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


def make_fv3_faithful_pe_config(**overrides) -> CDGridPrimitiveEquationConfig:
    """FV3_3D iter 392: factory for FV3-faithful PE config.

    Enables every PE FV3-fidelity flag at production-recommended
    values.  Pair with ``create_cubed_sphere(..., use_duogrid=True)``
    so the iter-370 ``use_fv3_cross_face_du_proj`` flag has effect
    (per iter-384 finding).

    Includes:
        * ``use_fv3_a2b_zeta_corner = True`` (iter-14)
        * ``use_fv3_metric_aware_d_con = True`` (iter-338/344)
        * ``use_fv3_cross_face_du_proj = True`` (iter-370)
        * ``d_con_top_zero_levels = 2`` (iter-433/434)
        * ``delt_max = 1.0`` (FV3 ``fv_arrays.F90`` production
          default — iter-436)
        * ``nord_v = 1`` (FV3 ``nord=1`` del-4 vorticity damping
          production default — iter-437)
        * ``corner_div_damp_nord = 1`` (FV3 ``nord=1`` del-4
          corner-div damping production default — iter-437)
        * ``corner_div_damp_d4_bg = 0.16`` (FV3 ``d4_bg``
          production default — iter-451)
        * ``corner_div_damp_d2_bg_k1 = 4.0`` (FV3 sponge boost
          — iter-444)
        * ``corner_div_damp_d2_bg_k2 = 2.0`` (FV3 sponge boost
          — iter-444)
        * ``use_fv3_sponge_damp_v = True`` (iter-443/444; PE
          has no damp_w)

    The ``d_con_top_zero_levels=2`` matches FV3 production
    behaviour under the typical sponge namelist
    (``d2_bg_k1=0.16, d2_bg_k2=0.05``): FV3
    ``dyn_core.F90:790/800/804`` zeros ``d_con_k`` at the top
    2 levels.  Set ``d_con_top_zero_levels=0`` explicitly to
    recover the iter-208/221/223/225 baseline (no sponge
    zeroing).

    Parameters
    ----------
    **overrides
        Any config field can be overridden.

    Returns
    -------
    CDGridPrimitiveEquationConfig
    """
    defaults = dict(
        use_fv3_a2b_zeta_corner=True,
        use_fv3_metric_aware_d_con=True,
        use_fv3_cross_face_du_proj=True,
        d_con_top_zero_levels=2,
        delt_max=1.0,
        nord_v=1,
        corner_div_damp_nord=1,
        corner_div_damp_d4_bg=0.16,
        # iter-452: d2_bg_k1/k2 left at 0.0 — see NH factory note.
        use_fv3_sponge_damp_v=True,
    )
    defaults.update(overrides)
    return CDGridPrimitiveEquationConfig(**defaults)
