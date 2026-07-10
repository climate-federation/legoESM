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

from legoesm.core.fv3_sw_core import fv3_fb_sw_step
from legoesm.core.precision import cast_pytree
# SW state pytrees now live in core (shared with the ocean barotropic solver).
from legoesm.core.shallow_water_state import (
    CDGridShallowWaterState,
    FV3EdgeShallowWaterState,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    cgrid_mass_flux_divergence,
    cdgrid_momentum_tendencies,
    extrapolate_boundary_corners,
    fv3_sw_tendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import (
    CONNECTIVITY,
    EAST,
    NORTH,
    SOUTH,
    WEST,
)
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.core.conservation import (
    conservation_accumulator, global_area_sum, batch_global_area_sums,
    shard_invariant_cube_face_sum, cube_faces_are_whole_on_shards,
)


def _cube_area_sum(x, area):
    """Shard-count-invariant ``sum(x * area)`` on a cube ``(6, n, n)`` field
    (issue #852): reduces each whole face then combines the 6 partials in a
    fixed order so the sum is bit-identical across whole-face decompositions.
    Falls back to the plain sum off-cube or on a face-split (tiled) mesh.
    """
    prod = x * area
    if (prod.ndim == 3 and prod.shape[0] == 6
            and cube_faces_are_whole_on_shards()):
        return shard_invariant_cube_face_sum(prod)
    return jnp.sum(prod)
from legoesm.core.fv3_sw_core import (
    d2a2c_vect,
    d_sw5_corner_divergence,
    fb_v_d_to_covariant,
    fv3_csw_tendencies,
)
from legoesm.core.fv_tp_2d import transport_step
from legoesm.core.fv3_del6_vt_flux import fv3_del6_vorticity_damping
from legoesm import constants


# --- numerics floor (one-off; not a tunable scheme coefficient) ---
# Physical lower bound on the shallow-water fluid depth ``h`` [m].  ``h`` is a
# layer thickness and MUST stay >= 0; a uniform additive mass-conservation
# correction can otherwise drive a thin layer negative (mass through a "hole").
# Exactly zero is the physical floor; with mass_target > 0 the floor-then-
# renormalize below is provably positive (see ``_apply_mass_conserving_floor``).
_H_FLOOR_M: float = 0.0  # coeff-ok: physical non-negativity floor for depth h


def _apply_mass_conserving_floor(
    h, area, total_area, mass_target, mass_new, h_floor=_H_FLOOR_M,
):
    """Mass-conserving, positivity-preserving shallow-water depth fixer.

    Sign / conservation convention (h is a fluid DEPTH, positive-up; mass =
    ``∫ h dA`` is the conserved scalar):

    1. Uniform additive correction ``c = (mass_target - mass_new)/total_area``
       so ``∫ (h+c) dA == mass_target`` EXACTLY (the legacy behaviour; preserves
       horizontal gradients of h, important for differentiability).
    2. Floor at ``h_floor`` so ``h >= h_floor`` (no negative depth / no mass
       leak through a thin layer).  Flooring INJECTS a non-negative mass
       ``deficit = ∫ (h_floored - (h+c)) dA``.
    3. Remove exactly ``deficit`` by proportionally shrinking the headroom
       ``(h_floored - h_floor)`` of the cells ABOVE the floor, so the column
       mass returns to ``mass_target`` while every cell stays >= ``h_floor``.

    Conservation: ``∫ h_out dA == mass_target`` to accumulator precision (the
    flooring-injected mass is removed term-for-term).  Positivity: with
    ``h_floor == 0`` and ``mass_target > 0`` the renorm ``scale = mass_target /
    excess_mass ∈ (0, 1]`` so ``h_out = scale * (h+c) >= 0`` is guaranteed.

    Differentiability: gradients are FINITE everywhere (the divide uses a safe
    denominator so the empty-headroom boundary cannot seed a NaN adjoint).  The
    map has the usual finite-subgradient KINKS of ``maximum``/``where`` at floor
    onset (``h_add == h_floor``) and at ``deficit == 0`` — it is sub-
    differentiable, not C¹, like every other clip/floor in the dycore.  The
    no-flooring fast path (``deficit == 0``) returns the legacy additive result
    BIT-IDENTICALLY via ``jnp.where`` so existing W2/W5 regression baselines are
    unchanged on the normal, never-negative path.
    """
    correction = (mass_target - mass_new) / total_area
    h_add = h + correction
    h_floored = jnp.maximum(h_add, h_floor)
    # Mass injected by the floor (>= 0); must be removed to conserve.  Both
    # global area sums use the shard-count-invariant cube reduction (#852) so
    # the renormalization scale is decomposition-independent on the floor branch.
    deficit = _cube_area_sum(h_floored - h_add, area)
    # Headroom above the floor that can absorb the removal.
    excess = jnp.maximum(h_floored - h_floor, 0.0)
    excess_mass = _cube_area_sum(excess, area)
    # Proportional shrink of the headroom; guard the empty-headroom case.
    # Use a SAFE denominator BEFORE the divide so reverse-mode AD never sees a
    # 0/0 (a bare ``deficit/excess_mass`` inside ``jnp.where`` still evaluates
    # the divide on the dead branch and seeds NaN adjoints at ``excess_mass==0``
    # — the feasible dry-column boundary h_add==0, mass_target==0).
    # ``maximum(scale, 0)`` keeps h >= h_floor even in the (physically
    # impossible for mass_target>0, h_floor=0) regime deficit > excess_mass.
    has_headroom = excess_mass > 0.0
    excess_mass_safe = jnp.where(has_headroom, excess_mass, 1.0)
    scale = jnp.where(
        has_headroom,
        jnp.maximum(1.0 - deficit / excess_mass_safe, 0.0),
        1.0,
    )
    h_renorm = h_floor + excess * scale
    # Bit-identical to the legacy additive result when nothing was floored.
    return jnp.where(deficit > 0.0, h_renorm, h_add)


# ==============================================================================
# State and Config
# ==============================================================================

# CDGridShallowWaterState moved to legoesm.core.shallow_water_state (imported above).


class CDGridShallowWaterConfig(NamedTuple):
    """Configuration for C-D grid shallow water model.

    Iter-1009 dual-target preset (production CDGrid path):
        from tests.test_iter921_w2_v_vs_h_pareto_sentinel import _div_damp_cube
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=10.0 * _div_damp_cube(N),  # iter-1009: 10* (vs iter-893's 8*)
            boundary_fix=True,
            damp_v=0.04, nord_v=2,             # iter-1009: 0.04 (vs iter-893's 0.06)
            apply_fortran_xppm_boundary=True,
        )
    With this calibration on FV3EdgeShallowWaterModel + N=36 + dt=300:
        W2 1-day v_ll_Linf = 0.1147 m/s (≤ 0.119 acceptance) ✓
        cosine bell day-1: 5/5 sentinels pass
        W5 day 5: h_min=3885 m, speed_max=68.6 m/s (artifact-free) ✓
    See `tests/test_iter1002_w2_target_met.py` for the pinned sentinels.
    """
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
    # cube-vertex halo cells per face in `arakawa_lamb_gradient`.
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
    # (d2a2c_vect -> compute_transport_quantities -> transport_step
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
    #     finite-volume transport (`d2a2c_vect` -> `transport_step`),
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
    # The FB chain passes `dddmp` to `d_sw5_corner_divergence`
    # directly via the separate `dddmp` field above.
    dddmp_prod: float = 0.2

    # Iter-926/iter-927: optional Fortran-style d_sw5 corner-divergence
    # damping applied as a POST-RK3 wind correction.  Default OFF
    # preserves iter-893 production bit-for-bit.
    #
    # When True (iter-927 REPLACEMENT semantics):
    # 1. Production cell-centre `adaptive_coeff*grad(div)` damping is
    #    SKIPPED (`div_damp` forced to 0 inside `fv3_sw_tendencies`).
    # 2. After the RK3 main step, `d_sw5_corner_divergence(u_d, v_d,
    #    ua, va, cdgrid, dt, d2_bg=config.d2_bg, dddmp=config.dddmp,
    #    d4_bg=config.d4_bg, nord=config.nord)` is computed and
    #    applied as a per-step KE-gradient wind correction:
    #      u_new += (ke_damp(i,j) - ke_damp(i+1,j)) / dx
    #      v_new += (ke_damp(i,j) - ke_damp(i,j+1)) / dy
    # This is NOT an RK3 sampled tendency — it's a discrete per-step
    # update outside the integrator, exactly as Fortran d_sw5 → d_sw6
    # does (sw_core.F90:1641-1944).  In FV3 there is no separate
    # "div_damp" from d_sw5; d_sw5 IS the divergence damping.
    #
    # MEASUREMENT (iter-927).  Both ADDITIVE (iter-926b: keep
    # production div_damp + add d_sw5) and REPLACEMENT (iter-927:
    # skip production div_damp + d_sw5 only) variants REJECTED on
    # acceptance criterion (≥10 % v_ll_Linf improvement, <5 % h_L2
    # regression):
    #   ADDITIVE  :  v_ll_Linf +1608 %, h_L2 +577 %.
    #   REPLACEMENT:  v_ll_Linf  +637 %, h_L2 +238 %.
    # Replacement is LESS BAD than additive (it doesn't double-damp)
    # but still rejects.  Reason: the Python A-L+RK3 operator family
    # has its own structural cancellation balance with cell-centre
    # `grad(div)` damping; switching to corner d_sw5 KE-add structure
    # disrupts that balance because the other operators (vorticity,
    # B-function, Coriolis) are still in the A-L family.  The full
    # FV3 fix requires the FB chain (currently unstable).
    #
    # SCOPE: consumed by `FV3EdgeShallowWaterModel.step` ONLY.  The
    # `FV3FBShallowWaterModel` routes through the true-FV3 d_sw1/
    # d_sw4/d_sw5/d_sw6 chain natively, so the flag is redundant
    # there and a UserWarning fires at FB model construction time.
    #
    # Coefficients: uses the existing `d2_bg`/`dddmp`/`d4_bg`/`nord`
    # config fields.  Default coefficients (d2_bg=0, dddmp=0,
    # d4_bg=0.16, nord=1) give Fortran-default del-4 background
    # damping.  Setting `d2_bg=div_damp/da_min_c, dddmp=0.2, nord=0`
    # mimics production's adaptive Smagorinsky regime in d_sw5
    # corner form.
    use_fv3_dsw5_corner_damping: bool = False


def iter1009_dual_target_config(
    n: int,
    div_damp_factor: float = 8.0,
    damp_v: float = 0.030,
    hyperdiff_coeff: float = 0.0,
) -> CDGridShallowWaterConfig:
    """Iter-1009/1021/1030 dual-target preset: W2 ≤ 0.119 m/s + W5 day-5 artifact-free.

    On `FV3EdgeShallowWaterModel` at N=36 dt=300 s this preset achieves:
        W2 1-day v_ll_Linf = 0.1138 m/s (≤ 0.119 acceptance) ✓
        W5 day-5 (h_min=3888 m, speed_max=45.1 m/s) ✓ (artifact-free)
        Cosine bell day-1: 5/5 production metrics within ±5% ✓

    Calibration evolution:
    - iter-1009: (div=10, damp_v=0.04) → v_ll=0.1147, W5 spd=68.6
    - iter-1021: (div=9, damp_v=0.035)  → v_ll=0.1137, W5 spd=53.3
    - iter-1030: (div=8, damp_v=0.030)  → v_ll=0.1138, W5 spd=45.1

    Iter-1030 is the best W5 day-5 stability margin (45.1 m/s vs the
    80 m/s threshold = 44% headroom) at essentially equal W2.

    See `tests/test_iter1002_w2_target_met.py` for the pinned sentinels
    and `docs/fv3_fortran_fidelity_review.md` (iter-985..1030) for the
    full calibration narrative.

    Resolution caveat: C36 is uniquely the dual-target sweet spot.
    C24 fails W2 (0.1835); C48 fails W5 day-5 (speed > 200 m/s)
    regardless of (div_damp, damp_v) tuning.

    Parameters
    ----------
    n : int
        Cubed-sphere face cells per side.  C36 is the validated
        dual-target reference.
    div_damp_factor : float, default 8.0
        Multiplier on `_div_damp_cube(n)`.  Iter-1030 measured 8.0*
        as the W5-best factor — slightly looser div damping allows
        the Rossby wave train to develop with less spurious damping.
    damp_v : float, default 0.030
        Vorticity damping coefficient.  Iter-1030 measured 0.030 as
        the W5-best damp_v (lower than iter-1009's 0.06 and
        iter-1021's 0.035).
    hyperdiff_coeff : float, default 0.0
        Biharmonic (del-4) hyperdiffusion coefficient [m^4/s].  Off
        by default (iter-1030 dual-target preset deliberately uses no
        hyperdiff because W2 / W5 day-5 are stable without it).  For
        LONGER runs (W5 full 15-day, W6 full 14-day Rossby-Haurwitz),
        the matrix runner sets ``hyperdiff_coeff=_hyperdiff_cube(n)``
        — without that, cube W5/W6 blow up at days 14.58 / 9.03
        respectively while latlon W5/W6 stay stable.  See
        ``new_test_dycores.md`` iter-31/33.

    Returns
    -------
    CDGridShallowWaterConfig
        Pre-populated with the iter-1009/1021 calibration plus
        `apply_fortran_xppm_boundary=True` and `boundary_fix=True`.
        Includes optional biharmonic hyperdiffusion via the
        `hyperdiff_coeff` kwarg (iter-35; default 0.0 preserves the
        original iter-1030 dual-target behavior).
    """
    # _div_damp_cube(n) = 1.5e7 * (48/n)^2 — same formula as
    # tests/test_iter921_w2_v_vs_h_pareto_sentinel.py.  Inlined here
    # to keep `src` independent of `tests`.
    div_damp_base = 1.5e7 * (48.0 / n) ** 2

    # Iter-1017 Codex review: warn on non-C36.  Calibration is
    # validated at N=36 (iter-985..1012); other resolutions may
    # fail one or both targets.  See iter-1012 entry.
    if n != 36:
        import warnings
        warnings.warn(
            f"`iter1009_dual_target_config(n={n})` calibration was "
            f"validated only at N=36 dt=300 s.  At other resolutions "
            f"the dual W2/W5 target is NOT guaranteed: iter-1012 "
            f"measured C24 fails W2 (v_ll=0.1835), C48 fails W5 "
            f"day-5 (speed > 100 m/s).  Use this preset on C36 for "
            f"the documented dual-target acceptance, or recalibrate "
            f"`div_damp_factor`/`damp_v` for your target N.",
            UserWarning, stacklevel=2,
        )

    return CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        div_damp=div_damp_factor * div_damp_base,
        boundary_fix=True,
        damp_v=damp_v, nord_v=2,
        apply_fortran_xppm_boundary=True,
    )


# ---------------------------------------------------------------------------
# Resolution-scaling helpers used by both the production matrix runner and
# the ``legoesm test williamson`` CLI.  Iter-1030 calibration values; matching
# constants in ``scripts/run_atmosphere_test_matrix.py`` (``_hyperdiff_cube``,
# ``_div_damp_cube``) and ``tests/test_iter921_w2_v_vs_h_pareto_sentinel.py``
# (``_div_damp_cube``) must stay in sync — these are the single source of
# truth and the script-local mirrors should defer to these helpers.
# ---------------------------------------------------------------------------


def _require_positive_int(value, name: str) -> int:
    """Coerce ``value`` to a positive ``int`` or raise ``ValueError``.

    Uses ``operator.index`` so a ``float``/``np.float64`` (e.g. ``2.5``) is
    rejected rather than silently truncated, and excludes ``bool`` (which is
    an ``int`` subclass) — the contract is a genuine positive integer.
    """
    import operator
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        ivalue = operator.index(value)
    except TypeError:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}") from None
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


def _validate_cube_resolution(n: int) -> int:
    return _require_positive_int(n, "cubed-sphere resolution `n`")


def cdgrid_hyperdiff_cube(
    n: int, ref_n: int = 48, ref_coeff: float = 1.0e16,
    scaling_exponent: int = 4,
) -> float:
    """Biharmonic hyperdiffusion coefficient [m^4/s] for a C-N cubed-sphere
    grid, scaled with ``(ref_n/n)^scaling_exponent``.

    Default ``ref_n=48``, ``ref_coeff=1e16``, ``scaling_exponent=4`` is the
    iter-1030 calibration (validated at C36/C48/C72 in the matrix runner — see
    ``scripts/run_atmosphere_test_matrix.py`` and the ``new_test_dycores``
    log).  Reuse this helper rather than re-deriving ``1e16``/``ref_n`` inline.

    ``scaling_exponent`` selects the resolution law:

    - ``4`` (default): keeps ``hyperdiff_coeff * dx^-4`` — i.e. the grid-scale
      biharmonic *damping time* — constant across resolutions.
    - ``2`` (FV3 ``d_sw5`` corner-damping style, the ``cdgrid_div_damp_cube``
      law): delivers progressively MORE absolute grid-scale damping as ``n``
      grows.  For colliding modons (#753) the collision drives an enstrophy
      cascade whose grid-scale delivery rate rises with resolution, so the
      ``^4`` law is under-damped at the face seams at C96+; ``^2`` gives the
      empirically-needed ~4x at C96 (= (96/48)^2).

    Every ``scaling_exponent`` returns ``ref_coeff`` at ``n == ref_n``, so the
    default C48 calibration is exponent-invariant (C48 runs are unchanged).
    """
    scaling_exponent = _require_positive_int(scaling_exponent, "scaling_exponent")
    n_int = _validate_cube_resolution(n)
    return ref_coeff * (ref_n / n_int) ** scaling_exponent


def cdgrid_div_damp_cube(
    n: int, ref_n: int = 48, ref_coeff: float = 1.5e7,
) -> float:
    """Base divergence-damping coefficient for a cubed-sphere C-D grid,
    scaled with ``(ref_n/n)^2`` (FV3-style; see ``d_sw5`` corner damping).
    Iter-1030 calibration value ``1.5e7`` at C48.

    The validated dual-target preset ``iter1009_dual_target_config`` uses
    ``8.0 * cdgrid_div_damp_cube(n)`` at C36; the
    ``williamson_cli_calibration`` preset below uses ``2.0 *`` to stay
    stable at C24 and C48 (the 8× factor is C36-specific and overshoots
    at C48 — see iter1009 docstring).
    """
    n_int = _validate_cube_resolution(n)
    return ref_coeff * (ref_n / n_int) ** 2


# --- Colliding-modons (#521/#753) validated stable-config defaults ---
# SINGLE SOURCE OF TRUTH consumed by BOTH the matrix runner (``test_num == 8``
# env-knob defaults) and ``scripts/validate/run_colliding_modons.py`` (CLI
# defaults), so the two cannot drift.  #800 was exactly that drift: the driver
# defaulted to no-duogrid + ``hyperdiff_factor=0.0`` + ``damp_v=0.030`` while the
# matrix used the tuned stable config, so a plain driver run erupted at the cube
# seams.  The 2026-07-03 #521 sweep optimum (duogrid + ``damp_v=0.010`` +
# ``1.0x`` biharmonic; ``0.5x`` erupts at the collision transient, ``0x`` blows
# up ~day 40 even with duogrid) is stable 100 days at C36/C48.
MODON_DIV_DAMP_FACTOR: float = 8.0
MODON_DAMP_V: float = 0.010
MODON_HYPERDIFF_FACTOR: float = 1.0
# Biharmonic backstop resolution law for the modon case (#753/#800).  ``2`` ==
# the ``(ref/n)^2`` FV3 div-damp law; ``4`` == the ``(ref/n)^4``
# grid-scale-damping-time-constant law.  Every exponent returns ``ref_coeff`` at
# ``n == ref_n == 48``, so C48 is exponent-invariant; the two laws differ only
# off the reference (C96: ^2 gives 4x the ^4 backstop = (96/48)^2; C36: ^2 gives
# 0.56x — i.e. LESS damping, better core preservation, still stable).
#
# Default is ``2``: the ``(ref/n)^4`` law is under-damped at the C96+ face seams
# (the modon collision drives an enstrophy cascade whose grid-scale delivery
# rate rises with resolution), so C96 ERUPTS under ^4 (max|u| 309-449 m/s by
# day ~5-50) but is stable under ^2.  Validated 100 days, mass drift 0.00e+00:
# C36 (^2 = 0.56x, the regression gate) PASS max|u| ~9 m/s; C96 (^2 = 4x) PASS
# max|u| ~18-40 m/s through the collision; C48 invariant.  See #753 item 1.
# ``4`` remains selectable (env/CLI) for the pre-#753 byte-identical behaviour.
MODON_HYPERDIFF_SCALING: int = 2


def williamson_cli_calibration(
    n: int,
    div_damp_factor: float = 2.0,
) -> CDGridShallowWaterConfig:
    """Pre-built ``CDGridShallowWaterConfig`` for the
    ``legoesm test williamson`` CLI (#269).

    Differences from ``iter1009_dual_target_config`` (which targets C36
    dual W2/W5 acceptance and is unstable at C48 with its 8×
    divergence-damping factor):
      * Uses ``cdgrid_hyperdiff_cube(n)`` (= iter-1030 calibration) by
        default so the height/v-wind edge artifacts that motivated #269
        get the production-quality diffusion bound — the prior CLI used
        a heuristic ``1e-4 * mean_dx**4 / dt`` that under-damped at C48.
      * Uses a gentler 2× ``cdgrid_div_damp_cube(n)`` so the CLI stays
        stable at C24/C48 (8× blows up W5 day-5 at C48, per the
        ``iter1009_dual_target_config`` docstring caveat).

    Other flags match ``iter1009_dual_target_config``:
      ``damp_v=0.030``, ``nord_v=2``, ``boundary_fix=True``,
      ``apply_fortran_xppm_boundary=True``, ``use_conservation_fixer=True``.

    Parameters
    ----------
    n : int
        Cubed-sphere face cells per side.
    div_damp_factor : float, default 2.0
        Multiplier on ``cdgrid_div_damp_cube(n)``.  Lower = looser
        damping (more accurate, less robust); higher = more damping
        (cleaner artifacts, may blow up at coarse/fine resolution).
    """
    return CDGridShallowWaterConfig(
        hyperdiff_coeff=cdgrid_hyperdiff_cube(n),
        div_damp=div_damp_factor * cdgrid_div_damp_cube(n),
        damp_v=0.030,
        nord_v=2,
        boundary_fix=True,
        apply_fortran_xppm_boundary=True,
        use_conservation_fixer=True,
    )


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
    du_d_dt, dv_d_dt = extrapolate_boundary_corners(du_d_dt, dv_d_dt, n)

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
        # iter-5: fp64 budget accumulator — fp32 reductions on ~6·N²
        # cubed-sphere arrays leak ~N·eps noise into the anchor and
        # produced ~10^-7 spurious "mass drift" in W5.  Matches the
        # cubed-sphere PE ``batch_global_area_sums`` precision.
        self._target_mass = global_area_sum(state.h, self.cdgrid.base)

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-20; mirrors iter-18 API)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-20; iter-19 API)."""
        self._target_mass = target_mass

    def compute_mass(self, state) -> jax.Array:
        """Global ``∫ h dA`` (fp64).  iter-21: API parity with PE / NH twins."""
        return global_area_sum(state.h, self.cdgrid.base)

    def _sync_dgrid_boundary(self, state: CDGridShallowWaterState):
        """Owner-based sync of D-grid corner winds at shared edges.

        For each shared face edge, the lower face index is the "owner".
        The non-owner face COPIES the owner's geographic wind — no
        averaging.  This ensures bitwise-identical corner values at
        shared boundaries without the edge-selective dissipation that
        pairwise averaging introduces.

        Vertices (shared by 3 faces) are owned by the lowest face index.

        FV3_3D iter-1052: MPI dispatch.  Under ``_halo_backend == "mpi"``
        the single-device logic reads ``ue[nbr_face, ...]`` directly,
        which under MPI replicated mode returns stale values for
        non-owned faces.  The MPI variant uses batched-per-peer
        sendrecv (iter-1051 pattern) for edge sync and an ``allreduce
        SUM`` for the 8 cube-vertex broadcasts.
        """
        from legoesm.grids.halo import get_halo_backend
        if get_halo_backend() == "mpi":
            return self._sync_dgrid_boundary_mpi(state)
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

    def _sync_dgrid_boundary_mpi(self, state: CDGridShallowWaterState):
        """FV3_3D iter-1052: MPI-aware variant of :meth:`_sync_dgrid_boundary`.

        Edge sync uses the iter-1051 batched-per-peer ``sendrecv``
        pattern: each rank packs its boundary strips (``ue``, ``vn``)
        for ALL cross-rank edges with a given peer into one buffer,
        and exchanges with that peer in a single ``sendrecv``.
        Each side's recv buffer contains the peer's strips for the
        same shared edges; the non-owner side (higher face index)
        overwrites its local value with the owner's strip.

        Vertex sync uses ``mpi4jax.allreduce(SUM)`` on a per-rank
        ``(8, 2)`` array where each rank fills the owner-face value
        for vertices it owns and zeros elsewhere.  After the
        allreduce every rank has the owner value for all 8 cube
        vertices and applies them to the 3 face-positions per
        vertex.  Sum is correct because each vertex has exactly
        ONE owner_face → one rank contributes a non-zero value.

        Owned-face contract preserved: only owned-face cells are
        written; non-owned face state retains its pre-sync value
        (same contract as iter-1040+ MPI tests).
        """
        from collections import defaultdict
        from legoesm.grids.halo import get_mpi_topology
        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
        try:
            import mpi4jax
            from mpi4py import MPI as _MPI
        except ImportError as exc:
            raise ImportError(
                "MPI _sync_dgrid_boundary_mpi requires mpi4jax + mpi4py."
            ) from exc
        topology = get_mpi_topology()
        # FV3_3D iter-1052 (codex F-15): face-only MPI only.  Sub-face
        # tiled mode would need tile-local D-grid sync logic — same
        # limitation as iter-1040+ ``pad_halo_mpi(interp_offsets=...)``.
        if topology.tiling != (1, 1):
            raise NotImplementedError(
                "CDGridShallowWaterModel._sync_dgrid_boundary_mpi only "
                "supports face-only MPI (tiling=(1, 1)); got tiling="
                f"{topology.tiling}.  Sub-face tiling needs tile-local "
                "edge / vertex sync logic which is not yet derived."
            )
        sendrecv = get_sendrecv_vjp(mpi4jax)
        comm = _MPI.COMM_WORLD
        rank = topology.rank
        n = self.grid.n
        # CDGridShallowWaterState stores winds at corners with shape
        # ``(6, n+1, n+1)`` — boundary strips have length ``n+1``.
        strip_len = n + 1

        u_d, v_d = state.u_d, state.v_d
        ca_c = self.cdgrid.cos_angle_corner
        sa_c = self.cdgrid.sin_angle_corner
        ue = ca_c * u_d - sa_c * v_d
        vn = sa_c * u_d + ca_c * v_d

        def _get_strip(arr, face, edge):
            if edge == WEST:    return arr[face, 0, :]
            elif edge == EAST:  return arr[face, n, :]
            elif edge == SOUTH: return arr[face, :, 0]
            else:               return arr[face, :, n]

        def _is_owned(f):
            return f in topology.local_face_ids

        # === EDGE SYNC: batched-per-peer ===
        # Classify each (owned_face, edge) where nbr_face != owned_face.
        # Group cross-rank edges by peer rank.  Each shared edge has
        # ONE entry per rank's view; both sides agree on canonical
        # ordering by sorting on the LOWER face index (then edge).
        cross_rank_edges = defaultdict(list)
        local_edges_nonowner = []
        for face in topology.local_face_ids:
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                if nbr_face >= face:
                    # face is OWNER for this edge (or self-loop, n/a).
                    # Non-owner side handles the write; if non-owner
                    # is on same rank, queue it as local-nonowner case.
                    if _is_owned(nbr_face):
                        # Local owner-case: nothing to do (the peer
                        # face on this rank will copy from us).
                        pass
                    else:
                        # Cross-rank owner-case: send our strip to
                        # the non-owner's rank (already covered when
                        # iterating from that side).
                        nbr_rank = topology.neighbor_ranks.get((face, edge))
                        if nbr_rank is not None and nbr_rank != rank:
                            cross_rank_edges[nbr_rank].append(
                                (face, edge, nbr_face, nbr_edge, rev,
                                 True)  # is_owner_local
                            )
                else:
                    # face is NON-OWNER (nbr_face < face).
                    if _is_owned(nbr_face):
                        # Local non-owner case: direct copy from nbr_face.
                        local_edges_nonowner.append(
                            (face, edge, nbr_face, nbr_edge, rev)
                        )
                    else:
                        nbr_rank = topology.neighbor_ranks.get((face, edge))
                        if nbr_rank is not None:
                            cross_rank_edges[nbr_rank].append(
                                (face, edge, nbr_face, nbr_edge, rev,
                                 False)  # is_owner_local
                            )

        # Per peer: pack send strips (ue, vn) in canonical order.
        # Canonical key: (min(my_face, nbr_face), min(my_edge, nbr_edge_at_min_face)).
        # Both ranks compute same key for the same shared edge.
        def _canon_key(entry):
            f, e, nf, ne, rev, owner_local = entry
            if f < nf:
                return (f, e)
            return (nf, ne)

        nbr_recv_strips = {}  # (face, edge) -> (ue_strip, vn_strip)
        # FV3_3D iter-1053: iterate peers in ascending rank order so
        # ALL ranks issue their sendrecv calls in the same global
        # peer-sequence — avoids cyclic-wait deadlock at np=6 where
        # each rank has 4 peers and dict-insertion order would
        # otherwise produce ``rank 0 → rank 3 → rank 2 → rank 1 →
        # rank 0`` cyclic blocking.  Sorted ordering ensures rank A
        # talks to rank B in the same iteration on both sides.
        for peer_rank in sorted(cross_rank_edges.keys()):
            entries = cross_rank_edges[peer_rank]
            ordered = sorted(entries, key=_canon_key)
            # Each side packs its OWN side's strips at each shared edge.
            # Even though only owner's strip is needed, packing both
            # makes shape symmetric → mpi4jax.sendrecv with equal-shape
            # send/recv works.
            ue_parts = []
            vn_parts = []
            for f, e, nf, ne, rev, owner_local in ordered:
                ue_parts.append(_get_strip(ue, f, e))
                vn_parts.append(_get_strip(vn, f, e))
            send_ue = jnp.concatenate(ue_parts, axis=0)
            send_vn = jnp.concatenate(vn_parts, axis=0)
            send_buf = jnp.concatenate([send_ue, send_vn], axis=0)
            send_tag = rank
            recv_tag = peer_rank
            recv_buf = sendrecv(
                send_buf, jnp.zeros_like(send_buf),
                peer_rank, peer_rank,
                send_tag, recv_tag, comm,
            )
            # Unpack: first half = peer's ue strips; second half = peer's vn.
            total_strip_size = send_ue.shape[0]
            recv_ue = recv_buf[:total_strip_size]
            recv_vn = recv_buf[total_strip_size:]
            offset = 0
            for f, e, nf, ne, rev, owner_local in ordered:
                chunk_ue = recv_ue[offset:offset + strip_len]
                chunk_vn = recv_vn[offset:offset + strip_len]
                offset += strip_len
                if not owner_local:
                    # Apply reversal — peer's strip is in nbr_face's
                    # coord frame; flip if shared edge has axis swap.
                    if rev:
                        chunk_ue = chunk_ue[::-1]
                        chunk_vn = chunk_vn[::-1]
                    nbr_recv_strips[(f, e)] = (chunk_ue, chunk_vn)

        # Pre-extract local non-owner case strips BEFORE writes.
        local_nonowner_strips = {}
        for f, e, nf, ne, rev in local_edges_nonowner:
            s_ue = _get_strip(ue, nf, ne)
            s_vn = _get_strip(vn, nf, ne)
            if rev:
                s_ue = s_ue[::-1]
                s_vn = s_vn[::-1]
            local_nonowner_strips[(f, e)] = (s_ue, s_vn)

        # Write back: for each owned face's non-owner edge, overwrite.
        def _set_edge(arr, face, edge, strip):
            if edge == WEST:
                return arr.at[face, 0, :].set(strip)
            elif edge == EAST:
                return arr.at[face, n, :].set(strip)
            elif edge == SOUTH:
                return arr.at[face, :, 0].set(strip)
            else:
                return arr.at[face, :, n].set(strip)

        for f, e, nf, ne, rev in local_edges_nonowner:
            s_ue, s_vn = local_nonowner_strips[(f, e)]
            ue = _set_edge(ue, f, e, s_ue)
            vn = _set_edge(vn, f, e, s_vn)
        for (f, e), (s_ue, s_vn) in nbr_recv_strips.items():
            ue = _set_edge(ue, f, e, s_ue)
            vn = _set_edge(vn, f, e, s_vn)

        # === VERTEX SYNC: allreduce(SUM) ===
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
        # Each rank fills owner-face value for vertices it owns.
        my_vtx = jnp.zeros((8, 2), dtype=ue.dtype)
        for i, vtx in enumerate(_vtx):
            owner_face, oi, oj = vtx[0]
            if _is_owned(owner_face):
                my_vtx = my_vtx.at[i, 0].set(ue[owner_face, oi, oj])
                my_vtx = my_vtx.at[i, 1].set(vn[owner_face, oi, oj])
        # Allreduce SUM: exactly one rank contributes non-zero per row.
        # mpi4jax.allreduce returns (result, token) or just result; use
        # ``mpi4jax_array_result`` helper to normalize.
        from legoesm.parallel.reductions import mpi4jax_array_result
        all_vtx = mpi4jax_array_result(
            mpi4jax.allreduce(my_vtx, op=_MPI.SUM, comm=comm)
        )
        # Apply: for each vertex, set all owned face positions.
        for i, vtx in enumerate(_vtx):
            ue_own = all_vtx[i, 0]
            vn_own = all_vtx[i, 1]
            for f, ii, jj in vtx:
                if _is_owned(f):
                    ue = ue.at[f, ii, jj].set(ue_own)
                    vn = vn.at[f, ii, jj].set(vn_own)

        # Convert back to face-local.  ca_c * ue + sa_c * vn for u_d;
        # -sa_c * ue + ca_c * vn for v_d.  This applies to all 6
        # faces — non-owned face values retain pre-sync content.
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
            # iter-5: fp64 budget accumulator (see set_initial_mass).
            acc = conservation_accumulator()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
                mass_new = global_area_sum(state_new.h, self.cdgrid.base)
            else:
                # Shard-count-invariant batched reduction (#852): mirrors the PE
                # fixer (fix_ps_mass -> batch_global_area_sums) so the SW cube
                # mass fixer is decomposition-independent too (the bare stacked
                # jnp.sum reduced per-shard then all-reduced — float32
                # non-associative, shard-count-dependent).
                mass_target, mass_new = batch_global_area_sums(
                    [state.h, state_new.h], self.cdgrid.base,
                )
            # iter-5: fp64 budget accumulator promotes the add.  Floor-then-
            # renormalize keeps h >= 0 (no mass leak through a thin layer)
            # while conserving column mass exactly; BIT-IDENTICAL to the
            # legacy additive correction on the normal never-negative path.
            h_fixed = _apply_mass_conserving_floor(
                state_new.h, area, total_area, mass_target, mass_new,
            )
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
        # iter-5: fp64 budget accumulator — fp32 reductions on ~6·N²
        # cubed-sphere arrays leak ~N·eps noise into the anchor and
        # produced ~10^-7 spurious "mass drift" in W5.  Matches the
        # cubed-sphere PE ``batch_global_area_sums`` precision.
        self._target_mass = global_area_sum(state.h, self.cdgrid.base)

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-20; mirrors iter-18 API).

        After this, the next ``step()`` falls back to the pre-state path.
        Call ``set_initial_mass(state)`` to re-anchor.
        """
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-20; iter-19 API)."""
        self._target_mass = target_mass

    def compute_mass(self, state) -> jax.Array:
        """Global ``∫ h dA`` (fp64).  iter-21: API parity with PE / NH twins."""
        return global_area_sum(state.h, self.cdgrid.base)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step using FV3 forward-backward."""
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
            # iter-5: fp64 budget accumulator (see set_initial_mass).
            acc = conservation_accumulator()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
                mass_new = global_area_sum(state_new.h, self.cdgrid.base)
            else:
                # Shard-count-invariant batched reduction (#852): mirrors the PE
                # fixer (fix_ps_mass -> batch_global_area_sums) so the SW cube
                # mass fixer is decomposition-independent too.
                mass_target, mass_new = batch_global_area_sums(
                    [state.h, state_new.h], self.cdgrid.base,
                )
            # iter-5: fp64 budget accumulator promotes the add.  Floor-then-
            # renormalize keeps h >= 0 (no mass leak through a thin layer)
            # while conserving column mass exactly; BIT-IDENTICAL to the
            # legacy additive correction on the normal never-negative path.
            h_fixed = _apply_mass_conserving_floor(
                state_new.h, area, total_area, mass_target, mass_new,
            )
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# FV3 edge-midpoint D-grid shallow water model
# ==============================================================================

# FV3EdgeShallowWaterState moved to legoesm.core.shallow_water_state (imported above).


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
        # iter-5: fp64 budget accumulator — fp32 reductions on ~6·N²
        # cubed-sphere arrays leak ~N·eps noise into the anchor and
        # produced ~10^-7 spurious "mass drift" in W5.  Matches the
        # cubed-sphere PE ``batch_global_area_sums`` precision.
        self._target_mass = global_area_sum(state.h, self.cdgrid.base)

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-20; mirrors iter-18 API).

        After this, the next ``step()`` falls back to the pre-state path.
        Call ``set_initial_mass(state)`` to re-anchor.
        """
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-20; iter-19 API)."""
        self._target_mass = target_mass

    def compute_mass(self, state) -> jax.Array:
        """Global ``∫ h dA`` (fp64).  iter-21: API parity with PE / NH twins."""
        return global_area_sum(state.h, self.cdgrid.base)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step."""
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
            # Iter-927: when `use_fv3_dsw5_corner_damping=True`, REPLACE
            # the production cell-centre A-L `adaptive_coeff*grad(div)`
            # damping with Fortran's corner d_sw5 damping (applied as a
            # post-RK3 hook below).  This is Fortran-faithful: in FV3
            # there is no separate "div_damp" tendency — d_sw5 IS the
            # divergence damping (sw_core.F90:1641-1944).  Adding both
            # (iter-926b) was REJECTED catastrophically (W2 +1608 %).
            # Replacing keeps the structure Fortran-faithful.  Default
            # OFF preserves iter-893 production bit-for-bit.
            _div_damp_for_tendency = (
                0.0 if self.config.use_fv3_dsw5_corner_damping
                else self.config.div_damp)

            def tendency_fn(s):
                dh, du, dv = fv3_sw_tendencies(
                    s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                    g=self.config.g,
                    div_damp=_div_damp_for_tendency,
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
                _, _, _, _, ut0, vt0 = d2a2c_vect(
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
                _EPS = 1e-30
                # Convention (2026-07-10 review): d_sw5_corner_divergence +
                # d2a2c_vect are COVARIANT-convention (FV3 Fortran verbatim);
                # the state winds are the model's ORTHOGONAL pair — convert v
                # at entry (u identical in both).  The hook only consumes the
                # scalar ke_damping, so no exit conversion is needed.
                v_cov = fb_v_d_to_covariant(
                    state_new.u_d, state_new.v_d, self.cdgrid)
                ua, va, _, _, _, _ = d2a2c_vect(
                    state_new.u_d, v_cov, self.cdgrid)
                ke_damping = d_sw5_corner_divergence(
                    state_new.u_d, v_cov, ua, va,
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
            # iter-5: fp64 budget accumulator (see set_initial_mass).
            acc = conservation_accumulator()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
                mass_new = global_area_sum(state_new.h, self.cdgrid.base)
            else:
                # Shard-count-invariant batched reduction (#852): mirrors the PE
                # fixer (fix_ps_mass -> batch_global_area_sums) so the SW cube
                # mass fixer is decomposition-independent too.
                mass_target, mass_new = batch_global_area_sums(
                    [state.h, state_new.h], self.cdgrid.base,
                )
            # iter-5: fp64 budget accumulator promotes the add.  Floor-then-
            # renormalize keeps h >= 0 (no mass leak through a thin layer)
            # while conserving column mass exactly; BIT-IDENTICAL to the
            # legacy additive correction on the normal never-negative path.
            h_fixed = _apply_mass_conserving_floor(
                state_new.h, area, total_area, mass_target, mass_new,
            )
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# Shared shallow-water barotropic core providers (registered for the 3-D ocean).
#
# The 3-D ocean's barotropic (free-surface) substep is itself a shallow-water
# problem and runs through this validated FV3 SW core.  These builders are
# REGISTERED in the foundational ``legoesm.registry.SW_BAROTROPIC_REGISTRY`` so
# ``ocean.dynamics.ocean_model`` can RESOLVE one by ``barotropic_staggering``
# name and reuse the SW dycore WITHOUT importing the atmosphere component — the
# sharing flows through the registry, keeping the components independent.
# ==============================================================================

def _build_fv3sw_barotropic_model(grid, cdgrid, ocean_config):
    """Build the corner-staggered (C-D grid) SW core for the ocean barotropic."""
    sw_cfg = iter1009_dual_target_config(
        cdgrid.n,
        div_damp_factor=ocean_config.barotropic_sw_div_damp_factor,
        damp_v=ocean_config.barotropic_sw_damp_v,
    )._replace(g=ocean_config.g, fix_mass=False)
    model = CDGridShallowWaterModel(grid, sw_cfg)
    model.cdgrid = cdgrid
    return model


def _build_fv3edge_barotropic_model(grid, cdgrid, ocean_config):
    """Build the FV3 edge-staggered SW core for the ocean barotropic."""
    sw_cfg = iter1009_dual_target_config(
        cdgrid.n,
        div_damp_factor=ocean_config.barotropic_sw_div_damp_factor,
        damp_v=ocean_config.barotropic_sw_damp_v,
    )._replace(g=ocean_config.g, fix_mass=False)
    model = FV3EdgeShallowWaterModel(grid, sw_cfg)
    model.cdgrid = cdgrid
    return model


_SW_BAROTROPIC_REGISTERED = False


def _register_sw_barotropic_builders() -> None:
    """Register the SW barotropic-core builders (idempotent; called on import)."""
    global _SW_BAROTROPIC_REGISTERED
    if _SW_BAROTROPIC_REGISTERED:
        return
    from legoesm.registry import SW_BAROTROPIC_REGISTRY

    SW_BAROTROPIC_REGISTRY.register(
        "fv3sw", _build_fv3sw_barotropic_model, overwrite=True
    )
    SW_BAROTROPIC_REGISTRY.register(
        "fv3edge", _build_fv3edge_barotropic_model, overwrite=True
    )
    _SW_BAROTROPIC_REGISTERED = True


_register_sw_barotropic_builders()
