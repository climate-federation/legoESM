"""FV3-style C-D grid Shallow Water Equations on the cubed-sphere.

True staggered C-D grid discretisation following Lin (2004):

* D-grid winds (cell corners, shape ``(6, n+1, n+1)``) are prognostic.
* C-grid velocities (cell edges) are diagnosed for mass transport.
* Vorticity is computed from circulation around cell boundaries
  (exact on the D-grid, avoids Hollingsworth-Kallberg instability).
* Bernoulli gradient uses Arakawa-Lamb 4-point formula at corners.
* Mass transport uses PPM (Piecewise Parabolic Method) for 4th-order
  accurate face reconstruction.
* D-to-C grid conversion includes non-orthogonality correction.
* Divergence damping with adaptive Smagorinsky scaling.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Colella & Woodward (1984): The Piecewise Parabolic Method
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    cgrid_mass_flux_divergence,
    cdgrid_momentum_tendencies,
    _extrapolate_boundary_corners,
    fv3_sw_tendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


# ==============================================================================
# State and Config
# ==============================================================================

class CDGridShallowWaterState(NamedTuple):
    """Shallow water state on the FV3 C-D grid.

    h : (6, n, n) -- height at cell centres
    u_d : (6, n+1, n+1) -- x-velocity at cell corners (D-grid)
    v_d : (6, n+1, n+1) -- y-velocity at cell corners (D-grid)
    h_s : (6, n, n) -- surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array


class CDGridShallowWaterConfig(NamedTuple):
    """Configuration for C-D grid shallow water model."""
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0  # Biharmonic hyperdiffusion
    div_damp: float = 0.0
    # Divergence damping coefficient [m^2/s].
    #
    # USED by three live call paths that all share this config field.
    # Line numbers in the same file are intentionally omitted below
    # because in-file line numbers shift whenever this comment is
    # itself edited, which made earlier iterations of this note (iter-
    # 178, 179) out-of-date the moment they were committed.  Refer by
    # function / class name for the in-file callers and by
    # module:function name for the cross-file operators; `grep` will
    # find them reliably.
    #
    #   1. `cdgrid_momentum_tendencies` (in legoesm.core.operators_cdgrid)
    #      — called from `cdgrid_shallow_water_tendencies` in this
    #      file, which is invoked by `CDGridShallowWaterModel.
    #      tendencies` and `.step`.
    #   2. `fv3_sw_tendencies` (in legoesm.core.operators_cdgrid) —
    #      the A-L RK3 PRODUCTION path used by
    #      `FV3EdgeShallowWaterModel` when `use_experimental_csw=
    #      False` (the default).
    #   3. `fv3_csw_tendencies` (in legoesm.core.fv3_sw_core) — the
    #      experimental CSW path dispatched when
    #      `use_experimental_csw=True`.
    #
    # IGNORED by the FB chain (`fv3_forward_backward_step` /
    # `fv3_fb_sw_step` / `_d_sw_native` in legoesm.core.fv3_sw_core):
    # `div_damp` is accepted and forwarded, but `_d_sw_native` never
    # reads it.  The FB chain's divergence damping comes from the
    # d_sw5 coefficients (`d2_bg`/`dddmp`/`d4_bg`/`nord`) below.  See
    # the `_d_sw_native` docstring for the split.
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"
    use_experimental_csw: bool = False  # EXPERIMENTAL: C-grid tendencies via RK3 (NOT the FV3 forward-backward scheme). Known unstable.
    boundary_fix: bool = True  # Replace boundary corner tendencies with interior

    # FV3 d_sw5 corner divergence damping knobs (fv_arrays.F90 defaults).
    d2_bg: float = 0.0         # Background del-2 coefficient
    dddmp: float = 0.0         # Adaptive Smagorinsky coefficient
    d4_bg: float = 0.16        # Background del-4+ coefficient
    nord: int = 1              # Damping order: 0=del-2, 1=del-4, 2=del-6

    # FV3 d_sw6 vorticity damping (vtdm4, do_vort_damp in Fortran).
    damp_v: float = 0.0        # Vorticity damping coefficient
    # FV3 derives nord_v(k) = min(2, flagstruct%nord) at runtime
    # (dyn_core.F90:757,1258).  Sentinel -1 means "auto-derive from nord";
    # the FB wrapper substitutes min(2, nord) at step time to honour the
    # Fortran convention instead of locking nord_v to a fixed default.
    nord_v: int = -1           # Vorticity damping order (-1 = derive)

    # Iter-766: Fortran `a2b_ord4` 3-pt cube-corner average in A-L
    # gradient cube-corner halo (a2b_edge.F90:385-388).  Direction-
    # neutral; replaces the 2-pt edge-halo-only average at the 4
    # cube-vertex halo cells per face in `_arakawa_lamb_gradient`.
    # Default OFF; FALSIFIED as a mode-A fix in iter-766 measurement
    # (1.89× W2 v_ll_Linf blowup at C36 dt=300).  Retained as an
    # opt-in diagnostic path for future cube-corner ablation studies.
    # Do NOT combine with `fortran_dir_aware_corners` — they both
    # overwrite the same 4 cube-corner halo cells and the iter-766
    # mutation is silently discarded by iter-765's subsequent p1/p2
    # construction.
    fortran_a2b_corner_avg: bool = False

    # Iter-767: Fortran `fill_corners_agrid_r8` VECTOR cube-corner
    # fill (fv_mp_mod.F90:1433-1457 with mySign=-1).  Overwrites the
    # 4 cube-vertex halo cells of (u_cc_pad, v_cc_pad) output by
    # `pad_halo_vector` with Fortran's direct cross-component swap +
    # sign flip (`u ← ±v`, `v ← ±u`).  Fortran uses this for non-
    # duogrid cube-corner fills because the rotate-pad-rotate scalar
    # chain is inconsistent at the 3-face cube vertex where face-
    # local grid angle is discontinuous.  Default OFF pending
    # iter-767 W2 measurement.
    fortran_vector_corner_fill: bool = False

    # Iter-769: optionally skip the 4 cube-corner cells in the
    # boundary_fix smoothing.  The cascaded row-0/col-0 (and row-n
    # /col-n) smoothing gives the corner cells a DOUBLE update —
    # effectively a 4-point average of the 2×2 block at the corner.
    # Since iter-762/768 localize mode A at cells adjacent to the 8
    # cube vertices, this flag isolates whether the cascaded corner
    # smoothing contributes to mode A.  Fortran has no post-tendency
    # smoothing at all (Codex iter-769 review), so setting this to
    # True moves boundary_fix closer to Fortran behavior at the 4
    # corner cells while keeping the stabilizer at non-corner
    # boundary cells (iter-511's load-bearing scope).  Default False
    # pending iter-769 W2 measurement.
    boundary_fix_skip_corners: bool = False

    # Iter-862 / iter-869b / iter-871c: opt-in flags for the legacy
    # FB-chain corner fixes (Fortran d_sw4 corner-KE override and
    # d_sw5 cube-vertex corner corrections).  Both default OFF
    # because the right-hand-side data uses mode='edge' same-face
    # halo at cube vertices, which iter-870 confirmed differs from
    # a cross-face proxy.  Until the iter-872+ cross-face D-grid
    # edge halo helper validates the RHS, both flags should remain
    # False.  When True (and a legacy non-bounded-domain grid),
    # they fire from the FB-chain entry points (`fv3_fb_sw_step` →
    # `_d_sw_native`).  Production `fv3_sw_tendencies` does NOT
    # invoke either flag (does not call `_d_sw_native`).
    apply_legacy_d_sw4_corner_ke_fix: bool = False
    apply_legacy_d_sw5_corner_corrections: bool = False

    # Iter-888c (Codex iter-888b stop-time fix): expose iter-888's
    # Fortran s11/s14/s15 boundary-formula opt-in (tp_core.F90:614-628
    # left, :632-647 right) on the canonical FB MODEL config, so users
    # of `FV3FBShallowWaterModel` can enable the new path WITHOUT
    # bypassing the model class and calling `fv3_fb_sw_step` directly.
    # Pre-iter-888c the kwarg was reachable from the function-level
    # FB chain (iter-888b plumbing) but the model class did NOT
    # forward it.  Default OFF preserves bit-identical behaviour for
    # existing FB model callers.  Like the other `apply_legacy_*`
    # flags above, this fires only on the FB chain; production
    # `FV3EdgeShallowWaterModel` (default `use_experimental_csw=False`
    # → `fv3_sw_tendencies` → `_ppm_reconstruct_1d` in
    # `operators_cdgrid.py`) does NOT consume this field.
    apply_fortran_xppm_boundary: bool = False

    # Iter-900 (default OFF): when True (and apply_fortran_xppm_boundary
    # is also True), `_ppm_reconstruct_1d` swaps the LEFT-side cube-
    # edge boundary overrides to Fortran-faithful formulas at the
    # corrected q_face indices [2,3,4]:
    #   q_face[2] = al(0) = c1*q1(-2)+c2*q1(-1)+c3*q1(0)
    #   q_face[3] = al(1) = xt clipped using q1(-1..2)
    #   q_face[4] = al(2) = c3*q1(1)+c2*q1(2)+c1*q1(3)
    # iter-892 has a known 1-cell shift bug at the LEFT slots that
    # iter-893 nonetheless landed as production (W2 v_ll_Linf 0.159
    # -> 0.132 m/s).  iter-900 keeps iter-892 as the production
    # default; flipping requires iter-901+ measurement that the
    # strict-Fortran path is at least as good.  See iter-899 entry in
    # `docs/fv3_fortran_fidelity_review.md` for the index-map analysis.
    # RIGHT-side overrides are unchanged in iter-900 (deferred to
    # iter-901+).
    #
    # SCOPE (iter-903b clarification): consumed by
    # `FV3EdgeShallowWaterModel.step` ONLY — that model forwards into
    # `fv3_sw_tendencies` -> `cgrid_mass_flux_divergence` ->
    # `_ppm_reconstruct_1d` where the override fires.  The
    # `FV3FBShallowWaterModel.step` path uses the FB chain
    # (`fv3_fb_sw_step` -> `_ppm_1d` in `fv_tp_2d.py`) which does NOT
    # consume `_ppm_reconstruct_1d`; setting this flag while using
    # `FV3FBShallowWaterModel` is detected at runtime and a clear
    # warning is emitted (see the FB step entry point).
    fortran_faithful_ppm_left: bool = False

    # Iter-903 (default OFF): symmetric counterpart to
    # `fortran_faithful_ppm_left`.  When True (and
    # apply_fortran_xppm_boundary is also True),
    # `_ppm_reconstruct_1d` swaps the RIGHT-side cube-edge boundary
    # overrides to Fortran-faithful formulas at the corrected
    # q_face indices [n+2, n+3]:
    #   q_face[n+2] = al(npx-1) = c1*q1(npx-3)+c2*q1(npx-2)+c3*q1(npx-1)
    #   q_face[n+3] = al(npx)   = xt clipped using q1(npx-2..npx+1)
    #     (PARTIAL faithfulness — q1(npx+1) is mode='edge' replica
    #      under halo=2; q1(npx+2) for al(npx+1) is unavailable).
    # Iter-892 has the same 1-cell shift bug on RIGHT as on LEFT
    # (placed c1/c2/c3 + xt-clipped formulas at q_face[n+1, n+2]).
    # iter-903 keeps iter-892 as production default; flipping
    # requires iter-904+ measurement that the strict-Fortran path
    # is at least as good.  See iter-899 entry in
    # `docs/fv3_fortran_fidelity_review.md` for the index-map
    # analysis.  iter-892's q_face[n+1] override is REMOVED in this
    # branch — that slot reverts to the standard 4th-order interior
    # stencil because Fortran has no boundary override at al(n-1).
    #
    # SCOPE (iter-903b clarification): same as
    # `fortran_faithful_ppm_left` — consumed by
    # `FV3EdgeShallowWaterModel.step` ONLY.  Setting this flag while
    # using `FV3FBShallowWaterModel.step` is detected at runtime and
    # a clear warning is emitted.
    fortran_faithful_ppm_right: bool = False

    # Iter-904 (default OFF): when True, the production
    # `FV3EdgeShallowWaterModel.step` swaps the height-tendency path
    # in `fv3_sw_tendencies` from the iter-892 CDGrid PPM
    # (fv3_d2cc -> fv3_cc2c -> cgrid_mass_flux_divergence) to the
    # true-FV3 d_sw1 finite-volume transport
    # (_d2a2c_vect -> compute_transport_quantities -> transport_step
    # in `fv3_sw_core.py` / `fv_tp_2d.py`), then derives
    # dh_dt = (h_new - h) / dt for the SSP-RK3 caller.  Momentum
    # tendencies (du_d_dt, dv_d_dt) are UNCHANGED in iter-904 — only
    # the mass transport path is swapped.
    #
    # Motivation: the iter-892 CDGrid PPM is NOT true FV3
    # (Arakawa-Lamb pressure gradient + tuned div_damp + non-FV3
    # boundary_fix).  True FV3 uses c_sw + d_sw1/d_sw4/d_sw5/d_sw6
    # forward-backward chain (`dyn_core.F90:489`, `sw_core.F90:79`).
    # iter-904 isolates whether the d_sw1 mass transport alone (kept
    # in an SSP-RK3 wrapper) reduces the W2 v-wind bias from
    # incomplete geostrophic cancellation.  Default-OFF preserves
    # iter-892/iter-893 production behavior bit-for-bit.
    #
    # SCOPE (iter-904): consumed by `FV3EdgeShallowWaterModel.step`
    # ONLY.  `FV3FBShallowWaterModel.step` already uses the FB chain
    # natively and does not benefit from the SSP-RK3 d_sw1 wrap.
    # iter-903b/c warning pattern is reused: setting this flag while
    # using FV3FBShallowWaterModel emits a UserWarning at __init__.
    use_fv3_dsw1_mass_transport: bool = False

    # Iter-905 (default OFF): when True, the production
    # `FV3EdgeShallowWaterModel.step` splits the time integration:
    #   - Mass (h): updated ONCE per full dt via the true-FV3 d_sw1
    #     finite-volume transport (`_d2a2c_vect` -> `transport_step`),
    #     held fixed across the RK3 stages for momentum.
    #   - Momentum (u_d, v_d): SSP-RK3 stages on momentum tendencies
    #     ONLY, with `h` fixed at the IC h (mass tendency forced to
    #     zero in tendency_fn so RK3 doesn't double-update mass).
    # This addresses the iter-904/iter-904b structural finding that
    # `use_fv3_dsw1_mass_transport=True` inside RK3 produces ~3x mass
    # advection per step (because each stage calls transport_step at
    # full dt).  Splitting forces mass to be advected exactly once
    # per dt.
    #
    # Limitation: holding h fixed across RK3 stages breaks the
    # momentum-mass conservation coupling that RK3 normally provides.
    # Stage Bernoulli function uses the IC h instead of an evolving
    # estimate; this is approximately first-order accurate in mass-
    # momentum interactions but second/third-order in pure momentum
    # advection.  iter-905 is intentionally an EXPERIMENTAL middle
    # ground between iter-904's full-RK3 wrap and a true FB chain
    # (which would replace dispatch_integrator entirely).
    #
    # SCOPE: consumed by `FV3EdgeShallowWaterModel.step` ONLY.
    # `FV3FBShallowWaterModel.__init__` extends its iter-903b warning
    # to flag this as well.
    use_split_mass_momentum_integration: bool = False

    # Iter-909 (default OFF): when True, the production
    # `fv3_sw_tendencies` divergence-damping path multiplies the
    # adaptive_coeff by `cube_edge_div_damp_factor` (default 0.5)
    # at face boundary cells (i in [0..band-1] U [n-band..n-1] and
    # j similarly), keeping full strength at the deep interior.
    # `cube_edge_div_damp_band` (default 2) controls the band depth.
    #
    # Motivation per iter-907/908/908b: div_damp is the dominant
    # contributor to W2 D-grid hot-spot magnitude (29x more than
    # boundary_fix), but a 1-D global coefficient sweep does not
    # improve W2 below the iter-892 baseline (8x is in a wide flat
    # plateau; 16x+ NaN's).  iter-909 tests whether targeting the
    # softer adaptive_coeff at hot-spot cells specifically (without
    # changing it at deep interior) yields a different trade-off.
    #
    # Default OFF preserves iter-892/iter-893 production behaviour.
    # SCOPE: consumed by `FV3EdgeShallowWaterModel.step` ONLY;
    # `FV3FBShallowWaterModel.__init__` warns if set on FB chain.
    cube_edge_softer_div_damp: bool = False
    cube_edge_div_damp_factor: float = 0.5
    cube_edge_div_damp_band: int = 2

    # Iter-872c-take3 (Codex pass-3): production divergence-damping
    # `dddmp` coefficient (Fortran `flagstruct%dddmp`,
    # fv_arrays.F90:360).  Default 0.2 preserves pre-iter-872
    # production behaviour bit-for-bit on the canonical W2/W5/
    # cosine-bell matrix configs that all rely on the historic
    # hardcoded 0.2 value.
    #
    # SCOPE — this field is consumed by `FV3EdgeShallowWaterModel.
    # step` ONLY (forwarded to `fv3_sw_tendencies`).  It is
    # deliberately NOT forwarded by `cdgrid_shallow_water_tendencies`
    # to avoid silently injecting adaptive Smagorinsky into
    # `CDGridShallowWaterModel` default-config users (Codex pass-3
    # finding).  Advanced `CDGridShallowWaterModel` callers wanting
    # adaptive Smagorinsky should pass `dddmp` directly to
    # `cdgrid_momentum_tendencies`, which has a Fortran-strict 0.0
    # default kwarg added in iter-872c.
    #
    # The FB chain passes `dddmp` to `_d_sw5_corner_divergence`
    # directly via the separate `dddmp` field above.
    dddmp_prod: float = 0.2

    # Iter-926: optional Fortran-style d_sw5 corner-divergence damping
    # applied as a POST-RK3 step (analogous to the existing damp_v
    # post-step hook).  Default OFF preserves iter-893 production
    # behaviour bit-for-bit.
    #
    # When True, after the RK3 main step completes,
    # `FV3EdgeShallowWaterModel.step` calls
    # `_d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, dt,
    # d2_bg=config.d2_bg, dddmp=config.dddmp, d4_bg=config.d4_bg,
    # nord=config.nord)` and applies the resulting `ke_damping`
    # corner field as a per-step KE-gradient wind correction
    # (sw_core.F90:1942):
    #   u_new = u_new + (ke_damp(i,j) - ke_damp(i+1,j)) / dx
    #   v_new = v_new + (ke_damp(i,j) - ke_damp(i,j+1)) / dy
    # This is NOT an RK3 sampled tendency — it's a discrete per-step
    # wind update applied OUTSIDE the integrator, exactly as the
    # Fortran d_sw5 → d_sw6 chain does (KE update then wind update).
    #
    # SCOPE: consumed by `FV3EdgeShallowWaterModel.step` ONLY.  The
    # `FV3FBShallowWaterModel` already routes through the true-FV3
    # d_sw1/d_sw4/d_sw5/d_sw6 chain natively, so the flag is
    # redundant there and a UserWarning fires at FB model
    # construction time (iter-903c-style).
    #
    # Coefficients: uses the existing `d2_bg`/`dddmp`/`d4_bg`/`nord`
    # config fields (Fortran defaults d2_bg=0, dddmp=0, d4_bg=0.16,
    # nord=1 → del-4 background damping).  These are independent of
    # production's `div_damp`/`dddmp_prod` (which feed the cell-
    # centred A-L gradient damping inside `fv3_sw_tendencies`).
    use_fv3_dsw5_corner_damping: bool = False


# ==============================================================================
# Tendencies
# ==============================================================================

def cdgrid_shallow_water_tendencies(
    state: CDGridShallowWaterState,
    cdgrid: CubedSphereCDGrid,
    config: CDGridShallowWaterConfig = CDGridShallowWaterConfig(),
):
    """Compute C-D grid shallow water tendencies.

    Uses PPM transport for mass flux and d2a2c with non-orthogonality
    correction for D-to-C grid conversion.

    Parameters
    ----------
    state : CDGridShallowWaterState
    cdgrid : CubedSphereCDGrid
    config : CDGridShallowWaterConfig

    Returns
    -------
    (dh_dt, du_d_dt, dv_d_dt) : tuple of jax.Array
    """
    h, u_d, v_d, h_s = state

    # Iter-872c-take5 (Codex pass-5): warn for direct callers that
    # set `dddmp_prod` non-default — `cdgrid_shallow_water_tendencies`
    # deliberately does NOT forward `dddmp_prod` to
    # `cdgrid_momentum_tendencies` (Codex pass-3 fix), so a non-
    # default value is silently ignored here.  The model-class
    # warning at `CDGridShallowWaterModel.__init__` catches the
    # common case but not direct functional callers.
    _default_dddmp_prod = CDGridShallowWaterConfig._field_defaults[
        "dddmp_prod"]
    if config.dddmp_prod != _default_dddmp_prod:
        import warnings
        warnings.warn(
            f"`cdgrid_shallow_water_tendencies` called with "
            f"`config.dddmp_prod={config.dddmp_prod!r}` (non-default).  "
            f"This function deliberately does NOT forward "
            f"`dddmp_prod` to `cdgrid_momentum_tendencies` (Codex "
            f"pass-3 fix to avoid silent default-config behaviour "
            f"changes); the setting will be ignored on this code "
            f"path.  `dddmp_prod` is consumed by "
            f"`FV3EdgeShallowWaterModel.step` only.  Pass `dddmp` "
            f"directly to `cdgrid_momentum_tendencies` if adaptive "
            f"Smagorinsky is required on this code path.",
            stacklevel=2,
        )

    # 1. Mass transport via C-grid velocities (with non-orth correction)
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    # 2. Momentum tendencies (vector-invariant form with div damping)
    # Iter-872c-take3 (Codex pass-3): `dddmp_prod` is deliberately
    # NOT forwarded here.  `dddmp_prod` is scoped to
    # `FV3EdgeShallowWaterModel` only; forwarding it from the
    # shared config to `cdgrid_momentum_tendencies` would silently
    # change `CDGridShallowWaterModel(default_config)` behaviour
    # (Codex pass-3 finding).  Advanced `CDGridShallowWaterModel`
    # users wanting adaptive Smagorinsky should pass `dddmp`
    # directly to `cdgrid_momentum_tendencies` (Fortran-strict 0.0
    # default kwarg added in iter-872c).
    du_d_dt, dv_d_dt = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=config.g, A_h=config.A_h,
        hyperdiff_coeff=config.hyperdiff_coeff,
        div_damp=config.div_damp,
    )

    # Boundary-corner fix: halo interpolation gives O(dx) gradient error
    # at all face-boundary corners (12-60x larger than interior).
    # Replace with nearest-interior values that have O(dx^2) accuracy.
    n = cdgrid.n
    du_d_dt, dv_d_dt = _extrapolate_boundary_corners(du_d_dt, dv_d_dt, n)

    return dh_dt, du_d_dt, dv_d_dt


# ==============================================================================
# Model class
# ==============================================================================

class CDGridShallowWaterModel(IntegrationMixin):
    """FV3-style C-D grid shallow water model on the cubed-sphere.

    Features PPM transport, non-orthogonality-corrected d2a2c, and
    adaptive divergence damping. No ad-hoc boundary smoothing needed.

    Parameters
    ----------
    grid : CubedSphereGrid
        Base cubed-sphere grid (cell-centre metrics).
    config : CDGridShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: CDGridShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None
        # Iter-872c-take4 (Codex pass-4): warn if `dddmp_prod` is set
        # to a non-default value, because `CDGridShallowWaterModel`
        # routes through `cdgrid_shallow_water_tendencies` which
        # deliberately does NOT forward `dddmp_prod` (Codex pass-3
        # finding).  Without this warning a user could set
        # `dddmp_prod=0.4` and silently get the same numerics as
        # `dddmp_prod=0.2` — a reproducibility hazard.
        _default_dddmp_prod = CDGridShallowWaterConfig._field_defaults[
            "dddmp_prod"]
        if self.config.dddmp_prod != _default_dddmp_prod:
            import warnings
            warnings.warn(
                f"CDGridShallowWaterConfig.dddmp_prod="
                f"{self.config.dddmp_prod!r} is set on a "
                f"CDGridShallowWaterModel instance, but this model's "
                f"tendency path (`cdgrid_shallow_water_tendencies`) "
                f"deliberately does NOT forward `dddmp_prod` to "
                f"`cdgrid_momentum_tendencies` (Codex pass-3 fix).  "
                f"The setting will be silently ignored.  "
                f"`dddmp_prod` is consumed by "
                f"`FV3EdgeShallowWaterModel.step` only.  Advanced "
                f"`CDGridShallowWaterModel` users wanting adaptive "
                f"Smagorinsky should pass `dddmp` directly to "
                f"`cdgrid_momentum_tendencies` instead.",
                stacklevel=2,
            )

    def set_initial_mass(self, state: CDGridShallowWaterState):
        """Anchor conservation fixer to initial state mass."""
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    def _sync_dgrid_boundary(self, state: CDGridShallowWaterState):
        """Owner-based sync of D-grid corner winds at shared edges.

        For each shared face edge, the lower face index is the "owner".
        The non-owner face COPIES the owner's geographic wind — no
        averaging.  This ensures bitwise-identical corner values at
        shared boundaries without the edge-selective dissipation that
        pairwise averaging introduces.

        Vertices (shared by 3 faces) are owned by the lowest face index.
        """
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        u_d, v_d = state.u_d, state.v_d
        n = self.grid.n
        ca_c = self.cdgrid.cos_angle_corner
        sa_c = self.cdgrid.sin_angle_corner

        # Convert all corners to geographic
        ue = ca_c * u_d - sa_c * v_d
        vn = sa_c * u_d + ca_c * v_d

        def _get_strip(arr, face, edge):
            if edge == WEST:    return arr[face, 0, :]
            elif edge == EAST:  return arr[face, n, :]
            elif edge == SOUTH: return arr[face, :, 0]
            else:               return arr[face, :, n]

        # Edge sync: non-owner copies from owner (lower face index)
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                if nbr_face < face:
                    # nbr_face owns → copy FROM neighbor
                    nbr_ue = _get_strip(ue, nbr_face, nbr_edge)
                    nbr_vn = _get_strip(vn, nbr_face, nbr_edge)
                    if is_reversed:
                        nbr_ue = nbr_ue[::-1]
                        nbr_vn = nbr_vn[::-1]
                    if edge == WEST:
                        ue = ue.at[face, 0, :].set(nbr_ue)
                        vn = vn.at[face, 0, :].set(nbr_vn)
                    elif edge == EAST:
                        ue = ue.at[face, n, :].set(nbr_ue)
                        vn = vn.at[face, n, :].set(nbr_vn)
                    elif edge == SOUTH:
                        ue = ue.at[face, :, 0].set(nbr_ue)
                        vn = vn.at[face, :, 0].set(nbr_vn)
                    else:
                        ue = ue.at[face, :, n].set(nbr_ue)
                        vn = vn.at[face, :, n].set(nbr_vn)

        # Vertex sync: lowest face index owns
        _vtx = [
            [(0, 0, 0), (3, n, 0), (5, 0, n)],
            [(0, n, 0), (1, 0, 0), (5, n, n)],
            [(0, 0, n), (3, n, n), (4, 0, 0)],
            [(0, n, n), (1, 0, n), (4, n, 0)],
            [(1, n, 0), (2, 0, 0), (5, n, 0)],
            [(1, n, n), (2, 0, n), (4, n, n)],
            [(2, n, 0), (3, 0, 0), (5, 0, 0)],
            [(2, n, n), (3, 0, n), (4, 0, n)],
        ]
        for vtx in _vtx:
            owner = vtx[0]  # already sorted by face index
            ue_own = ue[owner[0], owner[1], owner[2]]
            vn_own = vn[owner[0], owner[1], owner[2]]
            for f, i, j in vtx[1:]:
                ue = ue.at[f, i, j].set(ue_own)
                vn = vn.at[f, i, j].set(vn_own)

        # Convert back to face-local
        u_d_new = ca_c * ue + sa_c * vn
        v_d_new = -sa_c * ue + ca_c * vn
        return state._replace(u_d=u_d_new, v_d=v_d_new)

    def tendencies(self, state: CDGridShallowWaterState):
        """Compute tendencies (pure function wrapper)."""
        return cdgrid_shallow_water_tendencies(
            state, self.cdgrid, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: CDGridShallowWaterState, dt: float,
    ) -> CDGridShallowWaterState:
        """Advance one time step using SSP-RK3."""
        from legoesm.core.precision import cast_pytree
        # Cast state to compute precision at the boundary.
        state_c = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            dh, du, dv = cdgrid_shallow_water_tendencies(
                s, self.cdgrid, self.config,
            )
            return CDGridShallowWaterState(
                h=dh, u_d=du, v_d=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state_c, tendency_fn, dt, self.config.time_integrator,
        )

        # Owner-based sync: once per time step, after integrator
        state_new = self._sync_dgrid_boundary(state_new)

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        # Cast back to storage precision.
        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# FV3 Forward-Backward Shallow Water Model (EXPERIMENTAL — DO NOT USE)
# ==============================================================================

class FV3FBShallowWaterModel:
    """EXPERIMENTAL: FV3 forward-backward shallow water model.

    **NOT PRODUCTION-READY.** Known unstable (85 m/s v-wind after 1 day,
    3% mass error). Use ``FV3EdgeShallowWaterModel`` with the default
    ``use_experimental_csw=False`` for production work.

    This model uses the three-phase FV3 forward-backward scheme from
    ``fv3_sw_core.fv3_fb_sw_step``:
    1. c_sw: C-grid half-step (KE + vorticity, forward)
    2. p_grad_c: pressure gradient at C-grid (backward, using h_star)
    3. d_sw: D-grid full-step (mass transport + wind update, no A-L gradient)

    Blocker: The forward-backward coupling is unstable for finite dt without
    additional dissipation at the c_sw/d_sw interface. A faithful port would
    require FV3's exact dissipation control (del2/del4 at specific phases).

    Parameters
    ----------
    grid : CubedSphereGrid
    config : CDGridShallowWaterConfig, optional
    """

    def __init__(self, grid, config=None):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None
        # Iter-903c (Codex iter-903b stop-time fix): emit the
        # iter-900/iter-903 ignored-flag warning at MODEL CONSTRUCTION
        # time (not inside `step`).  Pre-iter-903c the warning lived
        # inside `@jax.jit step`, which means it only fired at TRACE
        # time — first call would emit, subsequent JIT-cached calls
        # would not.  Codex correctly flagged this as a "JIT-trace-
        # time only" guard.  Moving to `__init__` makes the warning
        # fire deterministically once per model construction,
        # independent of JIT timing.
        if (self.config.fortran_faithful_ppm_left
                or self.config.fortran_faithful_ppm_right
                or self.config.use_fv3_dsw1_mass_transport
                or self.config.use_split_mass_momentum_integration
                or self.config.cube_edge_softer_div_damp
                or self.config.use_fv3_dsw5_corner_damping):
            import warnings
            ignored = []
            if self.config.fortran_faithful_ppm_left:
                ignored.append("fortran_faithful_ppm_left")
            if self.config.fortran_faithful_ppm_right:
                ignored.append("fortran_faithful_ppm_right")
            if self.config.use_fv3_dsw1_mass_transport:
                ignored.append("use_fv3_dsw1_mass_transport")
            if self.config.use_split_mass_momentum_integration:
                ignored.append("use_split_mass_momentum_integration")
            if self.config.cube_edge_softer_div_damp:
                ignored.append("cube_edge_softer_div_damp")
            if self.config.use_fv3_dsw5_corner_damping:
                ignored.append("use_fv3_dsw5_corner_damping")
            warnings.warn(
                f"FV3FBShallowWaterModel ignores config flag(s) "
                f"{', '.join(ignored)}: the FB chain (fv3_fb_sw_step) "
                f"already routes through the true-FV3 d_sw1/d_sw4/"
                f"d_sw5/d_sw6 chain natively, so iter-900/iter-903/"
                f"iter-904/iter-926 production-only opt-ins have no "
                f"effect here.  These flags are specific to "
                f"FV3EdgeShallowWaterModel (Arakawa-Lamb + RK3 path "
                f"with selective FV3-style swaps).  Either switch to "
                f"FV3EdgeShallowWaterModel, or unset the flag(s) on "
                f"this config to silence the warning.",
                UserWarning, stacklevel=2)

    def set_initial_mass(self, state):
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step using FV3 forward-backward."""
        from legoesm.core.precision import cast_pytree
        from legoesm.core.fv3_sw_core import fv3_fb_sw_step

        state_c = cast_pytree(state, None, "compute")

        # FV3 dyn_core.F90:757,1258 derives nord_v(k) = min(2, nord) at
        # runtime when vorticity damping is active.  Honour that convention
        # by substituting when the user left nord_v at sentinel -1.
        nord_v = (min(2, self.config.nord) if self.config.nord_v < 0
                  else self.config.nord_v)

        h_new, u_new, v_new = fv3_fb_sw_step(
            state_c.h, state_c.u_d, state_c.v_d, state_c.h_s,
            self.cdgrid, dt, g=self.config.g,
            div_damp=self.config.div_damp,
            d2_bg=self.config.d2_bg,
            dddmp=self.config.dddmp,
            d4_bg=self.config.d4_bg,
            nord=self.config.nord,
            damp_v=self.config.damp_v,
            nord_v=nord_v,
            # Iter-871c: forward iter-869b/iter-871b opt-in flags from
            # the config so users can enable the FB-chain corner fixes
            # via `CDGridShallowWaterConfig`.  Default OFF preserves
            # bit-identical behaviour for existing callers; setting
            # either flag True at config-time threads through the FB
            # entry point to the `_d_sw_native` corner-fix sites.
            apply_legacy_d_sw4_corner_ke_fix=(
                self.config.apply_legacy_d_sw4_corner_ke_fix),
            apply_legacy_d_sw5_corner_corrections=(
                self.config.apply_legacy_d_sw5_corner_corrections),
            # Iter-888c (Codex iter-888b stop-time fix): forward the
            # iter-888 Fortran s11/s14/s15 boundary-formula opt-in
            # (tp_core.F90:614-628, :632-647) so FB-model users can
            # enable it via `CDGridShallowWaterConfig`.  Default OFF.
            apply_fortran_xppm_boundary=(
                self.config.apply_fortran_xppm_boundary),
        )

        state_new = FV3EdgeShallowWaterState(
            h=h_new, u_d=u_new, v_d=v_new, h_s=state_c.h_s)

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# FV3 edge-midpoint D-grid shallow water model
# ==============================================================================

class FV3EdgeShallowWaterState(NamedTuple):
    """Shallow water state with FV3 edge-midpoint D-grid stagger.

    h   : (6, n, n)   -- height at cell centres
    u_d : (6, n, n+1) -- x-velocity at x-edge midpoints
    v_d : (6, n+1, n) -- y-velocity at y-edge midpoints
    h_s : (6, n, n)   -- surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array


class FV3EdgeShallowWaterModel(IntegrationMixin):
    """PRODUCTION shallow water model with FV3-inspired edge-midpoint D-grid stagger.

    This is a **stabilized research path**, not a faithful FV3 port.
    Key differences from FV3:
    - Uses Arakawa-Lamb 4-point gradient (FV3 uses 2-point c_sw gradient)
    - Uses RK3 time integration (FV3 uses forward-backward splitting)
    - Edge-midpoint stagger avoids boundary sync (FV3 uses tile-edge coupling)

    Edge-midpoint D-grid winds sit half a cell from any face boundary,
    eliminating boundary sync entirely and removing edge artifacts.

    Momentum tendencies are computed at cell corners (compact stencil)
    and averaged to edge-midpoint positions.
    """

    def __init__(self, grid, config=None):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None

    def set_initial_mass(self, state):
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step."""
        from legoesm.core.precision import cast_pytree
        state = cast_pytree(state, None, "compute")

        if self.config.use_experimental_csw:
            # EXPERIMENTAL — known unstable (NaN by step ~50).
            # This wraps fv3_csw_tendencies in RK3, NOT the actual
            # forward-backward step (fv3_forward_backward_step).
            # See docs/cubed_sphere_edge_artifacts.md iterations 7-14.
            import warnings
            warnings.warn(
                "use_experimental_csw=True is experimental and known unstable. "
                "It runs fv3_csw_tendencies (C-grid half only) through "
                "RK3, NOT the actual FV3 forward-backward scheme "
                "(fv3_forward_backward_step). See "
                "docs/cubed_sphere_edge_artifacts.md.",
                stacklevel=2,
            )
            from legoesm.core.fv3_sw_core import fv3_csw_tendencies

            def tendency_fn_csw(s):
                dh, du, dv = fv3_csw_tendencies(
                    s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                    g=self.config.g,
                    div_damp=self.config.div_damp,
                    hyperdiff_coeff=self.config.hyperdiff_coeff,
                )
                return FV3EdgeShallowWaterState(
                    h=dh, u_d=du, v_d=dv,
                    h_s=jnp.zeros_like(s.h_s),
                )

            state_new = dispatch_integrator(
                state, tendency_fn_csw, dt, self.config.time_integrator,
            )
        else:
            def tendency_fn(s):
                dh, du, dv = fv3_sw_tendencies(
                    s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                    g=self.config.g,
                    div_damp=self.config.div_damp,
                    hyperdiff_coeff=self.config.hyperdiff_coeff,
                    boundary_fix=self.config.boundary_fix,
                    boundary_fix_skip_corners=(
                        self.config.boundary_fix_skip_corners),
                    fortran_a2b_corner_avg=self.config.fortran_a2b_corner_avg,
                    fortran_vector_corner_fill=(
                        self.config.fortran_vector_corner_fill),
                    dddmp=self.config.dddmp_prod,
                    # Iter-889: forward iter-888c's `apply_fortran_xppm_boundary`
                    # config field to `fv3_sw_tendencies` so the production
                    # CDGrid path picks up Fortran's iord<7 cube-edge
                    # boundary overrides (tp_core.F90:357-369) when the
                    # user opts in.  iter-888c's "production-path inert"
                    # contract is now superseded — production CDGrid
                    # responds to the flag from iter-889 onward.
                    apply_fortran_xppm_boundary=(
                        self.config.apply_fortran_xppm_boundary),
                    # Iter-900: forward fortran_faithful_ppm_left to the
                    # production CDGrid PPM so the LEFT cube-edge override
                    # uses Fortran's actual al(0)/al(1)/al(2) recipes at
                    # the corrected q_face indices when opted in.  Default
                    # OFF preserves iter-892/iter-893 production behavior.
                    fortran_faithful_ppm_left=(
                        self.config.fortran_faithful_ppm_left),
                    # Iter-903: forward the symmetric RIGHT-side
                    # Fortran-faithful flag to the production CDGrid PPM.
                    fortran_faithful_ppm_right=(
                        self.config.fortran_faithful_ppm_right),
                    # Iter-904: forward the true-FV3 d_sw1 mass-
                    # transport opt-in.  When True, fv3_sw_tendencies
                    # routes height-tendency through transport_step
                    # (Lin-Rood FV3-style) instead of
                    # cgrid_mass_flux_divergence, and requires `dt` in
                    # scope.  Only the production model forwards this
                    # flag; FV3FBShallowWaterModel uses the FB chain
                    # natively and emits a __init__ warning if the
                    # flag is set there.
                    use_fv3_dsw1_mass_transport=(
                        self.config.use_fv3_dsw1_mass_transport),
                    dt=dt,
                    # Iter-904b (Codex iter-904 stop-time fix):
                    # forward FV3 d_sw1 mass-transport damping
                    # (nord_v, damp_v) per `sw_core.F90:886-887`.
                    # Iter-904c (Codex iter-904b stop-time fix):
                    # use the SAME nord_v sentinel resolution as the
                    # FB chain (`dyn_core.F90:757,1258`):
                    # `nord_v = min(2, nord) if nord_v<0 else nord_v`.
                    # iter-904b incorrectly hardcoded `2` when
                    # nord_v<0, ignoring `self.config.nord`.
                    dsw1_nord=(
                        min(2, self.config.nord)
                        if self.config.nord_v < 0
                        else self.config.nord_v),
                    dsw1_damp_c=self.config.damp_v,
                    # Iter-909: forward cube-edge-aware adaptive_coeff
                    # softening for the production divergence-damping
                    # path.  Default-OFF preserves iter-892/iter-893
                    # bit-for-bit.
                    cube_edge_softer_div_damp=(
                        self.config.cube_edge_softer_div_damp),
                    cube_edge_div_damp_factor=(
                        self.config.cube_edge_div_damp_factor),
                    cube_edge_div_damp_band=(
                        self.config.cube_edge_div_damp_band),
                )
                return FV3EdgeShallowWaterState(
                    h=dh, u_d=du, v_d=dv,
                    h_s=jnp.zeros_like(s.h_s),
                )

            if self.config.use_split_mass_momentum_integration:
                # Iter-905: split mass+momentum integration.  Mass via
                # transport_step ONCE outside RK3; momentum via RK3
                # with h held fixed at the IC throughout the 3 stages.
                from legoesm.core.fv3_sw_core import _d2a2c_vect
                from legoesm.core.fv_tp_2d import transport_step
                _, _, _, _, ut0, vt0 = _d2a2c_vect(
                    state.u_d, state.v_d, self.cdgrid)
                eff_nord = (min(2, self.config.nord)
                            if self.config.nord_v < 0
                            else self.config.nord_v)
                h_new_split = transport_step(
                    state.h, ut0, vt0, dt, self.cdgrid,
                    nord=eff_nord, damp_c=self.config.damp_v,
                    apply_fortran_xppm_boundary=(
                        self.config.apply_fortran_xppm_boundary))
                # Wrap tendency_fn to force mass tendency to zero AND
                # to use the IC h for the Bernoulli function regardless
                # of which RK3 stage we're in.  This decouples mass
                # from the RK3 stages.
                h_held = state.h
                def tendency_fn_momentum_only(s):
                    s_held = FV3EdgeShallowWaterState(
                        h=h_held, u_d=s.u_d, v_d=s.v_d, h_s=s.h_s)
                    inner = tendency_fn(s_held)
                    return FV3EdgeShallowWaterState(
                        h=jnp.zeros_like(s.h),
                        u_d=inner.u_d, v_d=inner.v_d,
                        h_s=jnp.zeros_like(s.h_s))
                state_after_momentum = dispatch_integrator(
                    state, tendency_fn_momentum_only, dt,
                    self.config.time_integrator,
                )
                state_new = FV3EdgeShallowWaterState(
                    h=h_new_split,
                    u_d=state_after_momentum.u_d,
                    v_d=state_after_momentum.v_d,
                    h_s=state.h_s)
            else:
                state_new = dispatch_integrator(
                    state, tendency_fn, dt,
                    self.config.time_integrator,
                )

            # Fortran-faithful POST-STEP del-n vorticity damping
            # (sw_core.F90:1948-1999).  Fortran applies del6_vt_flux
            # ONCE per full timestep, AFTER the main d_sw6 update, as
            # `u += fy2 / dx`.  This is NOT a continuous tendency — it
            # is a discrete step update applied outside the RK3 loop.
            # Iter-755 refactored this from a tendency-form (iter-754)
            # to the Fortran-faithful post-step form to avoid RK3-
            # multiplied damping semantics.
            if self.config.damp_v > 0.0:
                from legoesm.core.fv3_del6_vt_flux import (
                    fv3_del6_vorticity_damping)
                eff_nord_v = (min(2, self.config.nord)
                              if self.config.nord_v < 0
                              else self.config.nord_v)
                # Fortran gridstruct%da_min_c = min(area_c) where
                # area_c is the B-GRID CORNER (dual-cell) area, NOT
                # the A-grid cell area.  Defined at
                # fv_grid_utils.F90:743:
                #   global_mx_c(area_c(is:ie,js:je), ..., da_min_c, ...)
                # In our cdgrid convention, `area_corner` (shape
                # (6, n+1, n+1)) is the B-grid dual-cell area.  Iter-
                # 755b fix: use area_corner, not base.area.
                da_min_c = jnp.min(self.cdgrid.area_corner)
                damp_step = (self.config.damp_v * da_min_c) ** (eff_nord_v + 1)
                du_step, dv_step = fv3_del6_vorticity_damping(
                    state_new.u_d, state_new.v_d,
                    damp=damp_step, nord=eff_nord_v, cdgrid=self.cdgrid,
                )
                state_new = state_new._replace(
                    u_d=state_new.u_d + du_step,
                    v_d=state_new.v_d + dv_step,
                )

            # Iter-926: optional Fortran-style d_sw5 corner-divergence
            # damping as a POST-RK3 wind correction (sw_core.F90:1641-
            # 1944 d_sw5 → d_sw6 KE-update structure).  Default OFF
            # preserves iter-893 production behaviour bit-for-bit.
            #
            # Fortran d_sw5 computes ke_damping = damp * delpc at
            # D-grid corners and adds it to ke_corner before the
            # d_sw6 wind update u += (ke(i,j) - ke(i+1,j)) / dx.
            # Per user iter-926 brief: "apply it only as a full-step
            # post-RK3 correction in FV3EdgeShallowWaterModel.step,
            # analogous to the existing post-step damp_v hook.  This
            # matters because Fortran d_sw5 is per-step KE/wind-
            # update structure, not a continuous RK3 tendency."
            if self.config.use_fv3_dsw5_corner_damping:
                from legoesm.core.fv3_sw_core import (
                    _d_sw5_corner_divergence, _d2a2c_vect)
                _EPS = 1e-30
                ua, va, _, _, _, _ = _d2a2c_vect(
                    state_new.u_d, state_new.v_d, self.cdgrid)
                ke_damping = _d_sw5_corner_divergence(
                    state_new.u_d, state_new.v_d, ua, va,
                    self.cdgrid, dt,
                    d2_bg=self.config.d2_bg,
                    dddmp=self.config.dddmp,
                    d4_bg=self.config.d4_bg,
                    nord=self.config.nord,
                    apply_legacy_corner_corrections=False,
                )
                # KE-gradient → wind correction (Fortran d_sw6
                # sw_core.F90:1942):
                #   u(i,j) += (ke(i,j) - ke(i+1,j)) / dx_u
                #   v(i,j) += (ke(i,j) - ke(i,j+1)) / dy_v
                dx_u = self.cdgrid.dx_edge_y   # (6, n, n+1) — at u_d
                dy_v = self.cdgrid.dy_edge_x   # (6, n+1, n) — at v_d
                ke_diff_u = (ke_damping[:, :-1, :]
                             - ke_damping[:, 1:, :])    # (6, n, n+1)
                ke_diff_v = (ke_damping[:, :, :-1]
                             - ke_damping[:, :, 1:])    # (6, n+1, n)
                state_new = state_new._replace(
                    u_d=state_new.u_d + ke_diff_u / jnp.maximum(dx_u, _EPS),
                    v_d=state_new.v_d + ke_diff_v / jnp.maximum(dy_v, _EPS),
                )

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")
