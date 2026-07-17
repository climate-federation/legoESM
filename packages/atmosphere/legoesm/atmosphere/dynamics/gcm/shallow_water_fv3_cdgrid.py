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

import warnings
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
    #      `FV3EdgeShallowWaterModel`.
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
    # `FV3EdgeShallowWaterModel` (`fv3_sw_tendencies` →
    # `_ppm_reconstruct_1d` in `operators_cdgrid.py`) does NOT consume
    # this field.
    apply_fortran_xppm_boundary: bool = False

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

    # RESERVED (fail-loud): d_sw5-style del-4/del-6 background
    # divergence damping on the production path.  0.0 = OFF (the only
    # valid value today; bit-identical).  A nonzero value RAISES in
    # fv3_sw_tendencies — the tendency-form insertion is invalid under
    # RK3 (dt-multiplied; every 2026-07-17 probe went NaN by step 100;
    # codex damping r1 P1-1).  The valid implementation is a POST-STEP
    # staged operator port of the certified d_sw5 translation
    # (fv3_native_d_sw.d_sw5) — this field is kept so that port has a
    # config home.  Authoritative duo case values for it: nord=2,
    # d4_bg=0.12 (Zenodo 8327578 input.nml).
    d4_bg_prod: float = 0.0

    # CORNER-LOCALIZED del-n vorticity damping (NON-FV3 stabilizer,
    # boundary_fix class).  When > damp_v: the post-step del-n damping
    # near the 8 cube-vertex regions uses THIS coefficient while the
    # interior keeps damp_v — separating vertex-mode suppression from
    # global core erosion (2026-07-17 modon sweep evidence).  0.0 = OFF
    # (bit-identical).  Mask: linear ramp from 1 inside
    # corner_damp_radius cells of a face corner to 0 beyond
    # radius+ramp cells (index-space; static, precomputed).
    corner_damp_v: float = 0.0
    corner_damp_radius: float = 4.0    # cells: full-strength zone
    corner_damp_ramp: float = 3.0      # cells: linear taper width

    # Damping order for d4_bg_prod: 1 = del-4, 2 = del-6.  The
    # authoritative duo case configs (Zenodo 8327578 input.nml) run
    # nord=2, d4_bg=0.12.
    d4_nord_prod: int = 1


def iter1009_dual_target_config(
    n: int,
    div_damp_factor: float = 8.0,
    damp_v: float = 0.030,
    hyperdiff_coeff: float = 0.0,
    d4_bg_prod: float = 0.0,
    corner_damp_v: float = 0.0,
    d4_nord_prod: int = 1,
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
        d4_bg_prod=d4_bg_prod,
        corner_damp_v=corner_damp_v,
        d4_nord_prod=d4_nord_prod,
    )


def fb_m1_preset_config(
    damp_v: float = 0.02,
    dddmp: float = 0.2,
    d4_bg: float = 0.16,
) -> CDGridShallowWaterConfig:
    """Phase-1 M1 validated FB damping preset for ``FV3FBShallowWaterModel``.

    ``nord=1, d4_bg=0.16, dddmp=0.2, damp_v=0.02, nord_v=2`` — the Phase-0
    (PR #967) preset validated on the 120-day colliding-modon run (mass
    drift ~1e-15).  The FB chain reads ONLY the d_sw5/d_sw6 knobs
    (``d2_bg/dddmp/d4_bg/nord`` + ``damp_v/nord_v``); ``div_damp`` and
    ``hyperdiff_coeff`` are production-path fields the FB step never
    consumes, so they stay 0 here.  ``nord_v`` is pinned explicitly to 2
    because the sentinel ``-1`` derives ``min(2, nord)=1`` for ``nord=1``
    (the sentinel trap documented in the M1 program notes).

    The tuning kwargs (``damp_v``/``dddmp``/``d4_bg``) exist for the M1
    per-case calibration sweeps; defaults are the validated preset.

    See ``docs/architecture/fv3_single_implementation_program.md`` (Phase 1
    M1) for the calibration table produced with this preset.
    """
    return CDGridShallowWaterConfig(
        nord=1, d4_bg=d4_bg, dddmp=dddmp, d2_bg=0.0,
        damp_v=damp_v, nord_v=2,
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

def _warn_if_not_fv3_native_grid(model_name: str, grid) -> None:
    """Phase-1 FV3-native compatibility migration.

    FV3-named models constructed on a non-ED grid keep working (the historic
    equiangular behavior is an explicitly preserved legacy choice) but warn:
    FV3's native grid is gnomonic_ed (``fv_arrays.F90`` default
    ``grid_type=0``), and the FV3-native path will require it.
    """
    if grid.gnomonic_form != "ed":
        warnings.warn(
            f"{model_name} was constructed on the legacy "
            f"{grid.gnomonic_form!r} cubed-sphere grid. FV3's native grid is "
            "gnomonic_ed (grid_type=0); build it with "
            "create_fv3_native_cubed_sphere(n) or "
            "create_cubed_sphere(n, gnomonic='ed'). The FV3-native path "
            "will make ED the default for FV3-named models.",
            FutureWarning,
            stacklevel=3,
        )


class FV3FBShallowWaterModel:
    """EXPERIMENTAL: FV3 forward-backward shallow water model.

    **NOT PRODUCTION-READY.** Known unstable (85 m/s v-wind after 1 day,
    3% mass error). Use ``FV3EdgeShallowWaterModel`` for production work.

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

    def __init__(self, grid, config=None, *, fv3_native_angles=False):
        _warn_if_not_fv3_native_grid("FV3FBShallowWaterModel", grid)
        self.grid = grid
        # The forward-backward core is the FV3-native grid's intended
        # consumer: unlike the A-L production solver (whose operators are
        # tuned to the legacy single-sided seam angles), the FB chain can
        # opt into fv3_native_angles — the exact grid_utils_init cross-face
        # seam cosa_u/v, sina_u/v (cubed_sphere_cdgrid.py:639).  Default
        # False keeps the legacy seam angles for the equiangular baseline.
        #
        # The seam angles are an ED-gnomonic concept: create_cubed_sphere_
        # cdgrid applies fv3_native_angles ONLY when gnomonic_form=="ed"
        # (silently ignoring it otherwise).  Reject the silent-fallback
        # combination loudly here so a caller cannot believe they got
        # native angles on an equiangular grid (codex p4c FB-review P2).
        if fv3_native_angles and getattr(grid, "gnomonic_form", None) != "ed":
            raise ValueError(
                "fv3_native_angles requires an ED gnomonic grid "
                f"(gnomonic_form='ed'); got "
                f"{getattr(grid, 'gnomonic_form', None)!r}. Build the grid "
                "with create_fv3_native_cubed_sphere(...).")
        self.cdgrid = create_cubed_sphere_cdgrid(
            grid, fv3_native_angles=fv3_native_angles)
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
        _warn_if_not_fv3_native_grid("FV3EdgeShallowWaterModel", grid)
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

    def _corner_damp_masks(self):
        """Static corner-proximity masks at the D staggerings.

        Index-space Chebyshev distance to the nearest of the four face
        corners; 1 within `corner_damp_radius` cells, linear taper to 0
        over `corner_damp_ramp` more.  Computed once per model (numpy,
        constant-folded under jit)."""
        # Cache NUMPY arrays only — caching jnp arrays built inside a
        # traced step leaks tracers through the instance attribute
        # (side-effect; the modon C1 arm error).  jnp.asarray of the
        # cached numpy constants folds at trace time.
        cached = getattr(self, "_corner_damp_masks_np", None)
        if cached is None:
            import numpy as _np

            n = self.cdgrid.base.n
            r = float(self.config.corner_damp_radius)
            w = max(float(self.config.corner_damp_ramp), 1.0e-6)
            if r + w >= n / 2:
                import warnings

                warnings.warn(
                    f"corner_damp_v: radius+ramp = {r + w:g} cells covers "
                    f"half the face edge at n={n} — the corner zones "
                    "overlap and the damping is effectively GLOBAL at "
                    "this resolution (codex damping r1 P2-3)",
                    stacklevel=2)

            def mask(shape_ij):
                ni, nj = shape_ij
                ii = _np.arange(ni)[:, None]
                jj = _np.arange(nj)[None, :]
                d = _np.full((ni, nj), _np.inf)
                for ci in (0, ni - 1):
                    for cj in (0, nj - 1):
                        d = _np.minimum(
                            d,
                            _np.maximum(_np.abs(ii - ci), _np.abs(jj - cj)))
                m = _np.clip((r + w - d) / w, 0.0, 1.0)
                return _np.broadcast_to(m, (6, ni, nj)).copy()

            n_ = self.cdgrid.base.n
            cached = (mask((n_, n_ + 1)), mask((n_ + 1, n_)))
            self._corner_damp_masks_np = cached
        return jnp.asarray(cached[0]), jnp.asarray(cached[1])

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step."""
        state = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            dh, du, dv = fv3_sw_tendencies(
                s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                g=self.config.g,
                div_damp=self.config.div_damp,
                hyperdiff_coeff=self.config.hyperdiff_coeff,
                boundary_fix=self.config.boundary_fix,
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
                d4_bg=self.config.d4_bg_prod,
                d4_nord=self.config.d4_nord_prod,
            )
            return FV3EdgeShallowWaterState(
                h=dh, u_d=du, v_d=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

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
        if self.config.damp_v > 0.0 or self.config.corner_damp_v > 0.0:
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
            du_step = dv_step = 0.0
            if self.config.damp_v > 0.0:
                damp_step = (self.config.damp_v * da_min_c) ** (eff_nord_v + 1)
                du_step, dv_step = fv3_del6_vorticity_damping(
                    state_new.u_d, state_new.v_d,
                    damp=damp_step, nord=eff_nord_v, cdgrid=self.cdgrid,
                )
            if self.config.corner_damp_v > self.config.damp_v:
                # CORNER-LOCALIZED del-n vorticity damping — a NON-FV3
                # stabilizer (same class as boundary_fix): the cube-
                # vertex vorticity modes need the full damp_v that
                # erodes vortex cores everywhere else (2026-07-17 modon
                # sweep: damp_v=0 stays stable 100 d but grows 8-fold
                # corner-symmetric artifacts; damp_v=0.010 keeps
                # corners clean but decays the cores 23->5 m/s).  Blend
                # the certified damping evaluated at the CORNER
                # coefficient into the update via a static corner-
                # proximity mask, so the interior keeps the low global
                # coefficient.
                damp_hi = (self.config.corner_damp_v * da_min_c
                           ) ** (eff_nord_v + 1)
                du_hi, dv_hi = fv3_del6_vorticity_damping(
                    state_new.u_d, state_new.v_d,
                    damp=damp_hi, nord=eff_nord_v, cdgrid=self.cdgrid,
                )
                m_u, m_v = self._corner_damp_masks()
                du_step = du_step * (1.0 - m_u) + du_hi * m_u
                dv_step = dv_step * (1.0 - m_v) + dv_hi * m_v
            state_new = state_new._replace(
                u_d=state_new.u_d + du_step,
                v_d=state_new.v_d + dv_step,
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
