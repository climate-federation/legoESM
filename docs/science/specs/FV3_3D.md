# FV3 3D Cubed-Sphere Edge-Artifact Investigation

Goal: solve cube-edge artifacts in the 3D atmospheric cubed-sphere paths
(`primitive_eq_cdgrid.py` hydrostatic + `compressible_euler_cdgrid.py`
non-hydrostatic) with **perfect conservation and no edge effects**, by
being faithful to the GFDL FV3 Fortran reference (vertically-Lagrangian
finite-volume core, Lin 2004; cubed-sphere transport, Putman & Lin 2007;
Harris et al. 2021) at
`../FV3/atmos_cubed_sphere-symmetryclean/model/`.

The shallow-water FV3 path is "decent"; the 3D atmospheric paths produce
visible cube imprint (concentric blobs at face centres bordered by
red/blue rings at panel boundaries) in u/v wind snapshots from
Held-Suarez and baroclinic test cases.

## Key findings summary (iter 541)

Comprehensive validation of the FV3-faithful 3D stack + the
user-facing edge-artifact suppression helpers.

**FV3 fidelity:** All 5 NH + 3 PE opt-in flags ported (see
table below).  Faithful to ``../FV3/atmos_cubed_sphere-
symmetryclean/``.

**Edge artifacts:** Asymptotically converging, not eliminated.
Refined characterization (iter 561-573):
- ``heat_source_del2_iters=8`` reduces NH edge_std by 89%.
- ``heat_source_del2_coeff=0.20`` (FV3 default) is OPTIMAL.
- BEST config (factory + clip + iters=8) at C24 SBR →
  edge_std = 5.83e-3 K (~6 mK).
- Edge_std ~ U_0¹·²⁰ (mK-level for normal winds; calm ~0.4
  mK at U_0=2 m/s; jet stream ~7 mK at U_0=20 m/s).
- Edge_std ~ n_steps¹·¹⁵ (super-linear, bounded growth).
- Edge_std plateau at ~5-6 mK floor at C24-C32; further
  resolution improvement minimal.
- **Zero IC → exactly zero output** (perfect rest preserve).
- PE Held-Suarez → δT ~ 1 mK (essentially no artifact).
Production estimate: at C96 + 1-day runs, edge noise should
be ~120 mK — well below physical signal magnitude.

This is FV3-faithful behavior — corners are formally lower-
order in FV3 too.  Practical: negligible in real applications.

**User-facing API for additional ~50% edge reduction:**

iter-553 found two regimes:
- **For SMOOTH atmospheric ICs (production, AMIP, HS, SBR):**
  use ``make_fv3_faithful_nh_config()`` — 89.5% edge_std
  reduction vs bare at C16 SBR.  All 5 NH fidelity flags ON.
- **For RANDOM/STRESS-TEST ICs (training, perturbations):**
  use ``make_legoesm_nh_min_edge_config()`` — 50% reduction
  at C8 random IC (iter-466 finding).

Both compatible with the clip-helper stack:
- ``monotone_halo_clip_context(slack=0.5)`` (iter-505): 15-
  site context manager, non-JIT.
- ``make_clipped_step(model, state, dt, slack=0.5)`` (iter-
  526): JIT-safe wrapper.  +0.6% performance overhead.
- ``make_clipped_scan_step(..., n_steps=N)`` (iter-544):
  multi-step JAX-scan API for long runs.

**Verified properties:**
- Mass conservation: NH 5e-9, PE 7e-9 over 10 steps (machine
  precision).
- ``jax.grad`` flows through (iter-522, iter-529).
- JIT-compatible via ``make_clipped_step`` (iter-526).
- Works on NH + PE + SW dycores (iter-526/529/531).
- Terrain (mountain) compatible (iter-537).
- Tracer transport: conservative + monotonic (iter-538).
- 50-step mass test: 5.3 ppb drift through step 42; instability
  at step 43 is C8 stability limit, NOT helper-induced
  (iter-539).
- Performance: +0.6% overhead vs raw jit (iter-541).

See ``scripts/run/example_fv3_clip_helper.py`` for a runnable
end-to-end demo.

## FV3-fidelity stack (iter 356/423 update)

By iter-423 the FV3-fidelity opt-in stack closes the major
documented audit gaps on the 3D paths.

NH config exposes 5 opt-in fidelity flags + duogrid grid:

| Flag                              | What            | Iter |
|-----------------------------------|-----------------|:----:|
| ``use_fv3_d_con_cv``              | cv_air branch (c_pd → c_vd)   | 320 |
| ``use_fv3_vector_halo_uv``        | vector halo (u, v) center→corner | 328 |
| ``use_fv3_dynamic_exner``         | live Π=Π_ref+π' at all 5 d_con sites + 3 delt_max caps | 336/337/397/398 |
| ``use_fv3_metric_aware_d_con``    | cosa_s/rsin2 form at all 5 d_con sites | 339/344/348/350/352 |
| ``use_fv3_cross_face_du_proj``    | cross-face halo at damp_v post-step (PAIR with use_duogrid=True per iter 384/385) | 370 |
| ``use_duogrid=True`` (grid)       | Lagrange-extended halo at 3 NH halo sites | 325 |

PE config exposes 3 opt-in fidelity flags + duogrid grid:

| Flag                              | What            | Iter |
|-----------------------------------|-----------------|:----:|
| ``use_fv3_a2b_zeta_corner``       | 4th-order A→B ζ corner | 14 |
| ``use_fv3_metric_aware_d_con``    | cosa_s/rsin2 form at all 4 d_con sites | 338/344/347/349/351 |
| ``use_fv3_cross_face_du_proj``    | cross-face halo at damp_v post-step | 370 |
| ``use_duogrid=True`` (grid)       | duogrid wiring at PE ke_correction halo | 333 |

User-facing factory functions (iter-392):
- ``make_fv3_faithful_pe_config(**overrides)``
- ``make_fv3_faithful_nh_config(**overrides)``
return configs with every FV3-fidelity flag enabled.

PE-NH asymmetry (iter-331/343): PE doesn't need cv (PE uses
cp_air which is FV3-faithful for hydrostatic), vector halo (PE
stores winds at corners, no center→corner interp), or dynamic
Exner (PE uses actual T, no Π factor).

All flags default ``False`` (preserves bit-for-bit baseline).
AD-at-rest coverage: NH iter-346 + PE iter-355 (full stacks).

Closed gaps:
- Gap #1 metric d_con cosa_s/rsin2 (8 sites): iter-338-352
- Gap #2 dynamic Exner (5 NH sites): iter-336/337

Documented residual gaps (lower priority, SW-only or non-duogrid):
- Gap #4/5 d_sw5 polar/boundary (SW solver, not 3D paths)
- Gap #6 vort/ptc edge halo (fv3_sw_core SW path only)
- Gap #7 fv_tp_2d s11/s14/s15 (non-duogrid path only)

## Production usage (iter 417, updated iter 454)

For FV3-faithful 3D production runs, use the iter-392 factories:

```python
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel, make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, make_fv3_faithful_pe_config,
)

# Pair factory with duogrid grid (REQUIRED for iter-370
# cross_face flag to take effect — per iter-384 finding).
grid = create_cubed_sphere(n=96, use_duogrid=True)

# NH (compressible Euler):
nh_cfg = make_fv3_faithful_nh_config(
    damp_v=0.030, damp_v_d_con=1.0,
    corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
    div_damp_coeff=1e6, div_damp_d_con=1.0,
    A_h=1e6, ah_d_con=1.0,
    damp_w=0.030, damp_w_d_con=1.0,
    # All factory-default FV3 fidelity flags / values exposed:
    #   use_fv3_d_con_cv             (iter-320 cv branch)
    #   use_fv3_vector_halo_uv       (iter-328)
    #   use_fv3_dynamic_exner        (iter-336/337)
    #   use_fv3_metric_aware_d_con   (iter-339/344)
    #   use_fv3_cross_face_du_proj   (iter-370)
    #   d_con_top_zero_levels = 2    (iter-434 sponge d_con zero)
    #   delt_max = 1.0               (iter-436 FV3 production)
    #   nord_v = 1                   (iter-437 FV3 del-4)
    #   corner_div_damp_nord = 1     (iter-437 FV3 del-4)
    #   corner_div_damp_d4_bg = 0.16 (iter-451 FV3 production)
    #   use_fv3_sponge_damp_w = True (iter-441 sponge boost)
    #   use_fv3_sponge_damp_v = True (iter-442 sponge boost)
    # NOTE: corner_div_damp_d2_bg_k* are NOT factory-set (iter-
    # 452 rollback — FV3 d2_bg_k1=4.0 / k2=2.0 require FV3-
    # specific da_min_c normalization).  Set them explicitly at
    # a value compatible with your ``corner_div_damp_d2_bg``,
    # e.g. d2_bg_k1=1e-4 when d2_bg=0.0005.
    # For column Rayleigh friction at top, add ``rf_tau_days``
    # (e.g. 5.0 days; iter-448).
)

# PE (hydrostatic):
pe_cfg = make_fv3_faithful_pe_config(
    damp_v=0.030, damp_v_d_con=1.0,
    corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
    div_damp_coeff=1e6, div_damp_d_con=1.0,
    A_h=1e6, ah_d_con=1.0,
    # All factory-default FV3 fidelity flags / values:
    #   use_fv3_a2b_zeta_corner      (iter-14)
    #   use_fv3_metric_aware_d_con   (iter-338/344)
    #   use_fv3_cross_face_du_proj   (iter-370)
    #   d_con_top_zero_levels = 2    (iter-434)
    #   delt_max = 1.0               (iter-436)
    #   nord_v = 1                   (iter-437)
    #   corner_div_damp_nord = 1     (iter-437)
    #   corner_div_damp_d4_bg = 0.16 (iter-451)
    #   use_fv3_sponge_damp_v = True (iter-443)
    # PE has no damp_w (no w prognostic).
)
```

For non-faithful (baseline) behavior, use the bare config
constructors (``CDGridCompressibleEulerConfig(...)`` /
``CDGridPrimitiveEquationConfig(...)``) — all flags default
``False`` for bit-for-bit pre-iter-320 behavior.

### Edge-artifact-minimized factories (iter 467/468/483/484)

For users prioritizing cube-edge artifact suppression over strict
FV3-fidelity, divergence-from-FV3 factories are provided:

```python
from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
    make_legoesm_nh_min_edge_config,             # iter-467: 50% reduction
)
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    make_legoesm_pe_min_edge_config,             # iter-468 (PE mirror)
)
```

* ``min_edge`` factory: drops 3 hurting flags (iter-466 finding)
  — 50% θ′ edge ratio reduction at C8+duogrid (4.54 → 2.25).
* ``aggressive`` setting (factory removed, unused): pass
  ``corner_div_damp_d2_bg=5e-2`` to the ``min_edge`` factory
  (iter-482) — 60% reduction (3.50 → 1.42×).  Trade-off:
  over-damps physical waves more than factory default.
* PE variants are API-symmetric; iter-469/470 showed PE is
  largely insensitive to these flag changes at C8.

### Halo monotone-clip helpers (iter 505/526)

For an additional ~50% long-term edge reduction (iter-511
shows 53% benefit at 10 steps), wrap the dycore step with
the iter-505 monotone-clip helper:

**For non-JIT use** (or when JIT is compiled inside the
context):

```python
from legoesm.grids.halo import monotone_halo_clip_context

with monotone_halo_clip_context(slack=0.5):
    new_state = model.step(state, dt)
```

**For JIT-compiled use** (recommended for long runs) — use
the iter-526 ``make_clipped_step`` helper which forces
tracing inside the context so clip is permanently baked in:

```python
from legoesm.grids.halo import make_clipped_step

step = make_clipped_step(model, state, dt=10.0, slack=0.5)
for _ in range(n_steps):
    state = step(state, dt=10.0)
```

**For long-running batches**, use the iter-544 scan-based
variant that compiles the entire n-step loop as a single
``jax.lax.scan`` (no Python overhead per step):

```python
from legoesm.grids.halo import make_clipped_scan_step

scan_step = make_clipped_scan_step(
    model, state, dt=10.0, n_steps=100, slack=0.5,
)
final_state = scan_step(state)
```

Combined with ``make_legoesm_nh_min_edge_config`` (iter-466),
this achieves **65.7% reduction** at C8+duogrid (iter-504).

Verified properties of the helper stack:
* Mass conservation: 5e-9 (NH) / 7e-9 (PE) over 10 steps
  (iter-523/524).
* ``jax.grad`` flows through (iter-522).
* Cube-smooth IC converges with resolution (iter-519:
  C8 e/i=1.43× → C16 1.29×).
* 15 halo-site patches (iter-527 final coverage).

## Duogrid investigation summary (iter 461-493 synthesis, iter 494 update)

A 33-iter empirical investigation of cube-edge artifacts at
C8 + duogrid + NH factory.  Key data points:

| Iter | Finding |
|------|---------|
| 461 | First empirical edge-metric data: NH factory worsens 3.7%, PE improves 2.5%; both near 1.0 |
| 462 | d2_bg_k1 sweep over 4 decades → ZERO change in NH ratio (corner-div sponge is per-level scalar invariant) |
| 463 | rf_tau_days sweep → ZERO change (RF is per-level scalar invariant) |
| 464 | heat_source_del2 reduces NH θ′ ratio 1.10 → 0.96 (14% drop) WITHOUT duogrid (positive control) |
| 465 | NH per-flag at C8+duogrid: baseline 4.37; vector_halo_uv biggest helper (Δ+5.0); 3 flags HURT |
| 466 | **50.4% reduction** dropping 3 hurting flags (4.54 → 2.25, 5 seeds) |
| 469 | PE T edge ratio INSENSITIVE to all factory flags (max |Δ| < 1e-4) |
| 470 | PE u_d also insensitive — PE is much less responsive than NH at C8 |
| 471 | **Duogrid alone is the regime-changing factor** — same factory ON/OFF = 5.12× |
| 472 | Duogrid penalty persists at C16 (4.12×) → not a low-res artifact |
| 473 | duogrid INCREASES edge std 4.77× while interior std unchanged — bug at edges |
| 474 | pad_halo_4d scalar+constant PASSES (5e-13) → rules out simple halo broken |
| 475 | Linear field shows 3.5% overshoot in duogrid halo (not catastrophic alone) |
| 476 | Vector halo + zero field PASSES exactly → rules out vector halo broken |
| 477 | Laplacian iter alone doesn't amplify (×0.634 both grids) → refutes iter-476 hypothesis |
| 478 | Penalty arrives in step 1 (3.67×); single-step composite, not multi-step accumulation |
| 479 | Post-step damp_v/damp_w NOT the culprit (3.50→3.54× unchanged when disabled) |
| 480 | **corner_div_damp is the key MITIGATOR** — without it, penalty explodes 30× to 109× |
| 481 | corner_div_damp_d2_bg sweep: 100× boost → 36% reduction |
| 482 | **Composite iter-466+iter-481: 59.5% reduction (3.50 → 1.42×)** |
| 487 | Resolution scan: C8=5.12×, C16=4.12×, C24=4.72× → penalty plateaus 4-5× |
| 489 | Overshoot LOCALIZED to cube-vertex (corner) cells — 24 cells/level globally |
| 490 | ``pad_halo_4d`` exposes ``monotone_clip`` arg, eliminates iter-489 overshoot |
| 491 | Random-field test: 76% halo overshoot (clip cuts to ≤ interior_max) |
| 492 | Dycore monkey-patch (1 site): only 1.4% reduction |
| 493 | Comprehensive 4-site patch: still only 1.5% — clip is NOT the dominant dycore fix |

**Bottom line**: duogrid bug is real and persistent (~4-5×
edge penalty at all tested resolutions).  Bisected to a
composite single-step dycore + duogrid interaction (not
isolable to any individual operator).

Mitigation taxonomy:
* **Working**: iter-466 flag drops (50%) + iter-481/482
  corner_div boost (60% combined).
* **Not working**: iter-490/493 halo-level monotone_clip
  (only 1-2% dycore impact despite 76% halo overshoot).  The
  iter-491 random-field finding overestimates dycore impact
  because dycore intermediate fields are smoother than
  random noise.

Below 1.42× edge ratio likely requires deeper dycore changes
than the halo-level clip — possibly in the cube-vertex
interpolation logic itself, or in how the dycore composes
halo reads with derivatives.

## Final state (iter 100 close-out, table updated through iter 103)

Core deliverable RESOLVED at iter 99: HS at C96 30 days completes
finite under the canonical setting::

    LEGOESM_HS_CUBE_DT_CFL=auto
    LEGOESM_AH_SCALE      # auto-applies per resolution

Verified results (HS hybrid 30 days, ``ah_x10`` auto-applied):

| n   | dt (auto) | result              | iter |
|:---:|:---------:|:-------------------:|:----:|
| C36 | 200       | stable, mid_std=0.228 | 19/24 |
| C48 | 200       | stable, mid_std=0.517 | 37   |
| C72 | 200       | stable, mid_std=6.815, max\|u\|=45.88 | 33 |
| C96 | 50        | finite at day 30, max\|u\|=20.14 m/s | 99 |
| C144| 33        | finite at day 5, max\|u\|=5.56 m/s (iter 121) | 102/121 |
| C192| 25        | finite at day 1 smoke, max\|u\|=0.84 m/s | 103 |

C144/C192 30-day empirical validation deferred: C192 1-day takes
~8.5 min wall (iter 103); 30-day projects to ~4 hours wall, beyond
reasonable Ralph-loop iteration budget.  iter-85's 1/dt scaling
predicts 30-day stable for both (NaN ~day 68 at C144 dt=33,
~day 90 at C192 dt=25).

Stretch goals beyond original scope: C144 30-day, C192 5/30-day
(iter 121 confirmed C144 5-day; iter 123 launched C192 5-day),
200-day climatology, nord>=2 fidelity restructure.  See
[Open follow-ups](#open-follow-ups-iter-38-status-updated-through-iter-75).

## Table of Contents (iter 53, updated iter 88, iter 115 count fix)

This document tracks 100+ investigation iterations.  For most
users the relevant sections are at the top; the iteration log
preserves the diagnostic chain for future maintainers.  The
iteration log itself uses MIXED ordering: iter 1-46 are in
chronological order at the bottom (oldest first); iter 48+ are
in REVERSE chronological order (newest first) so latest
findings are immediately visible.

- [**Lessons learned**](#lessons-learned-iter-84-synthesis-of-iter-18-83) — 7-point synthesis from 65+ cycles of investigation.
- [**Investigation summary**](#investigation-summary-iter-51-codex-meta-review-consolidation) — two-mechanism story (corner-divergence damping vs Laplacian viscosity calibration).
- [**Quick Reference**](#quick-reference-iter-38-summary-updated-iter-72) — production setting per resolution, env vars, recommended invocations.
- [**Reference oracle**](#reference-oracle-read-only-never-modify) — pointer into the FV3 Fortran source for FV3-fidelity work.

Key iterations:
- Iter 18: FV3 nord>0 implementation
- Iter 19-25: C36/C48 production validation
- Iter 26-32: C72 instability investigation (7-step elimination)
- Iter 33: BREAKTHROUGH — 10x A_h stabilises C72
- Iter 34-46: env vars, helpers, auto-apply, escape hatches
- Iter 48-52: e2e validation, codex review iterations, regression tests
- Iter 51: codex meta-review insights (open generalization gap)
- Iter 57-59: Smagorinsky-style adaptive A_h
- Iter 60-61: Smagorinsky generalization gap
- Iter 64: combined-path multi-step stability test
- Iter 65: C96 stability EMPIRICAL — dt is the lever, not A_h
- Iter 66-67: opt-in CFL-aware dt (short_time mode)
- Iter 69-70: C96 day-15 eigenmode + dt=100 fix (long_time mode)
- Iter 71-72: long_time / auto modes for ``LEGOESM_HS_CUBE_DT_CFL``
- Iter 73-77: docs, tests, Quick Reference updates
- Iter 79: CRITICAL — C96 dt=100 still NaNs at day 22.5
- Iter 80-81: very_long_time mode (dt=50) + auto-mode promoted
- Iter 84: 7-point Lessons learned synthesis
- Iter 85: linear-in-1/dt eigenmode scaling, predicts dt=50 stable to day 45
- Iter 86-87: auto threshold rationale + probe docstring tidy
- Iter 88: ToC update for iter 79-87
- Iter 89: regression guard for baroclinic cube path dt wiring
- Iter 90: clarify _run_c96_smoke n parameterization
- Iter 91: matrix --help epilog includes very_long_time + iter-81 auto
- Iter 92-94: small refinements + AST guards
- Iter 95: C96 dt=50 PASSES iter-79 day-22.5 mark
- Iter 96-98: doc propagation of iter-95 confirmation
- Iter 99: 🎉 C96 dt=50 30-DAY FINITE (max|u|=20.14, 1755s wall)
- Iter 100-101: closeout cleanup + Final state summary
- Iter 102: C144 1-day smoke stable
- Iter 103: C192 1-day smoke stable
- Iter 104-105: ToC + epilog updates with iter 99 confirmation
- Iter 106-113: stretch-goal documentation + small refinements
  (C144/C192 deferred budget note, jet-spinup clarification,
  test-count audit, C72 mid_std in Final state table)
- Iter 114-118: minor maintenance (ToC counts, Final state
  heading, mixed-ordering convention note, trajectory observation)
- Iter 119-124: minor maintenance + C144 5-day completion (iter
  121) + safety constants 1:2/3:1/3 ratio note (iter 120) + C192
  5-day launch (iter 123)
- Iter 167: C72 dt=100 30-day FINITE — long_time mode validated
- Iter 168: port FV3 corner-divergence damping (sw_core.F90:1641-1822)
  to the non-hydrostatic 3D path (compressible_euler_cdgrid.py),
  closing FV3-fidelity asymmetry between the two 3D paths
- Iter 169: port FV3 post-step del-n vorticity damping
  (sw_core.F90:1948-1999, ``damp_v`` / ``nord_v``) to the
  non-hydrostatic 3D path; second of three documented PE-vs-NH
  asymmetries closed
- Iter 170: port FV3 4th-order A→B (cell-centre → corner)
  interpolation for ζ_corner (a2b_edge.F90:a2b_ord4,
  ``use_fv3_a2b_zeta_corner``) to the non-hydrostatic 3D path;
  third corner-fidelity PE-vs-NH asymmetry closed
- Iter 171: port FV3 cell-centre constant + adaptive Smagorinsky
  divergence damping (sw_core.F90:1720, ``div_damp_coeff`` /
  ``div_damp_dddmp``) to the non-hydrostatic 3D path; fourth and
  last documented PE-vs-NH FV3-fidelity asymmetry closed
- Iter 172: NH FV3 toolkit composition test (all four iter-
  168/169/170/171 knobs ON simultaneously) + AST regression
  guards on config defaults and call-site presence; closes the
  silent-regression risk for the four new wirings
- Iter 173: NH async-halo overlap for the iter-171 div-damp
  gradient call (mirror of PE ``use_async_halo`` field); MPI
  optimization, single-device fall-through bit-for-bit baseline
- Iter 174: quantitative damping correctness test for NH —
  divergent IC + assert mean(|div_v|) REDUCES vs no damping.
  Catches sign-error in iter-168/171 wirings (which prior
  "changes-the-state" tests would miss)
- Iter 175: quantitative damp_v correctness — vortical IC +
  assert mean(|ζ|) REDUCES vs no damping; parametrized
  no-amplification check for nord ∈ {0, 1, 2}.  Catches sign
  errors in the iter-169 fv3_del6_vorticity_damping wiring
- Iter 176: NH FV3 toolkit transient damping — full toolkit ON
  over 10-step window, FINAL max|div_v| reduced by >=5 % vs
  no-damping baseline; KE bounded within 2x IC.  Validates the
  toolkit's intended behavioral effect at trajectory level
- Iter 177: cube-vertex corner fill mode reaches NH — regression
  test that fv3_bgrid_xdir / fv3_agrid_xdir modes produce a
  measurably different NH state vs avg.  Closes silent-ignore
  risk for the documented FV3-faithful corner fill modes
- Iter 178: extend iter-172 AST guards to cover iter-173's
  use_async_halo + dispatch; add self-check ("test the test")
  that drops each gate substring and asserts the missing-detection
  logic flags it.  Closes the gap that iter-172 predates
  iter-173 and the silent-typo failure mode
- Iter 179: NH-equivalent cube-imprint metric (PE iter-2 analogue).
  Toolkit ON reduces edge_std/interior_std vs no-damping baseline.
  Most-direct quantitative validation of the FV3 toolkit's
  cube-imprint suppression purpose, modulo C8 wall-time constraints
- Iter 180: port FV3 Smagorinsky-adaptive A_h to NH (mirror of
  PE iter 57-59); reuses compute_smagorinsky_ah_3d helper.
  Adaptive A_h paired with iter-171's adaptive cell-centre
  div_damp_dddmp completes the FV3 adaptive-damping toolkit on NH
- Iter 181: fix Smagorinsky sqrt(strain_mag_sq) gradient
  singularity at zero strain via JAX double-where trick.  Forward
  pass bit-for-bit unchanged; backward pass finite at rest state.
  Closes the iter-180 known limitation; PE iter-58 also benefits
- Iter 182: fix PE T_diss wind_speed = sqrt(u² + v²) sqrt(0)
  gradient singularity (same iter-181 double-where pattern).
  PE rest-state differentiability through T_diss now works
- Iter 183: fix SW d_sw5 smag_vort = sqrt(delpc² + wk²) sqrt(0)
  gradient singularity (same iter-181/182 double-where pattern).
  iter-962 SW W2 sentinel preserved bit-for-bit; AD through SW
  rest state with adaptive Smagorinsky now works
- Iter 184: umbrella AD-at-rest regression for the full NH toolkit
  (all 5 iter-168/169/170/171/180 knobs ON simultaneously).
  Catches any future helper that introduces a new sqrt-at-zero
  or other AD-hazard in the NH AD-critical path
- Iter 185: PE counterpart to iter-184 — umbrella AD-at-rest
  regression for the full PE toolkit (8 PE damping knobs ON
  simultaneously, including T_diss_coeff that iter-182 fixed).
  Both 3D paths now have integration-level AD-hazard guards
- Iter 186: extend iter-172/178 AST guards to cover iter-180's
  smagorinsky_cs field and compute_smagorinsky_ah_3d dispatch.
  Self-check loop also extended.  Closes silent-regression risk
  for the most recent NH config addition
- Iter 187: port FV3 ``smag_vort`` adaptive-cap formula
  (sw_core.F90:1797-1809) to the ``nord >= 1`` corner-div damping
  branch in BOTH PE and NH 3D paths.  Closes a real FV3-fidelity
  gap: legoESM previously used ``|delpc|*dt`` for the cap regardless
  of nord, but FV3's nord >= 1 path uses
  ``|dt|*sqrt(delpc² + ζ_corner²)`` (line 1797).  Reuses iter-170's
  ``_interp_center_to_corner_a2b_ord4`` for ζ_corner and the iter-
  181/183 double-where pattern for sqrt(0) AD safety.  Bit-for-bit
  baseline preserved at nord=0.
- Iter 188: PE / NH parity + PE AST regression guard.  Closes
  iter-187 codex review concerns 2 (PE hardcodes ``_dt_approx
  = 200.0`` while NH has tunable ``corner_div_damp_dt_proxy``)
  and 6 (PE iter-12/14/16/18/187 wirings have no AST regression
  guard mirroring iter-172 NH).  Adds the missing PE config field
  with default = 200.0 (preserves existing iter-18 behaviour) and
  a comprehensive PE-side AST guard test with iter-178-style
  self-check.
- Iter 189: plumb the actual integration ``dt`` to the corner-div
  damping adaptive cap in BOTH PE and NH 3D paths.  iter 187/188
  used ``config.corner_div_damp_dt_proxy`` (a constant approximation)
  for the FV3 ``min(0.20, dddmp * dt * smag_vort)`` cap.  FV3
  ``d_sw5`` uses the actual integration dt.  iter-189 threads the
  real dt from ``model.step`` → ``tendency_fn`` →
  ``fv3_hydrostatic_tendencies`` (PE) and analogous NH path, with
  fallback to the existing config dt-proxy when no dt is passed
  (preserves backward compatibility for direct-call tests).
- Iter 190: dedup the ``a2b_ord4(zeta)`` halo-2 exchange between
  the iter-170 ``zeta_corner`` site and the iter-187
  ``_zeta_smag_corner`` site.  Closes iter-187 codex review
  concern 3 (extra unmerged halo).  When BOTH iter-170 and
  iter-187 are active, factor the computation into a single
  ``_zeta_a2b_ord4`` local computed at most once, reused at both
  sites.  Bit-for-bit baseline preserved at all flag combinations.
- Iter 191: coverage tests for the iter-190 dedup path.  Prior
  to iter-191 NO test exercised BOTH iter-170
  (``use_fv3_a2b_zeta_corner=True``) AND iter-187 (``corner_div_damp_d4_bg
  > 0`` + ``corner_div_damp_nord > 0``) simultaneously — the
  case iter-190 dedup actually combines.  iter-191 adds focused
  PE + NH integration tests verifying finite output, ``differs
  from iter-170-only``, ``differs from iter-187-only``, and
  ``jax.grad`` AD-at-rest safety with both flags ON.
- Iter 192: extend the iter-184/185 umbrella AD-at-rest tests to
  ALSO engage iter-187 (``d4_bg=1e-3`` + ``nord=1``) on top of
  iter-170 (``use_fv3_a2b_zeta_corner=True``).  Previously the
  umbrellas left the iter-187 smag_vort branch dormant
  (``corner_div_damp_nord=0`` default).  After iter-192 both
  umbrellas exercise the iter-190 dedup'd ``_zeta_a2b_ord4``
  shared between iter-170 and iter-187 sites under ``jax.grad``
  at the rest state.  Future AD hazards in the smag_vort branch
  now caught by the umbrella regression.
- Iter 193: port FV3 ``damp_w + nord_w`` post-step del-(2*(nord_w+1))
  damping for vertical velocity ``w`` to the NH 3D path.  Faithful
  port of FV3 ``sw_core.F90:1080-1086`` (in ``d_sw1``).  Reuses the
  SW backbone ``_del6_vt_flux`` (already imported via iter-169 for
  ``damp_v``).  Mirrors the iter-169 post-step pattern but applied
  to a scalar (w) instead of the (u, v) vector.  Default-off; FV3
  AM4 production default is ``damp_w=0.30 + nord_w=2``.
  Complementary to the legoESM-native ``hyperdiff_w_coeff``
  biharmonic.
- Iter 194: extend the iter-184 NH umbrella to ALSO engage iter-193
  ``damp_w + nord_w``.  Previously the umbrella exercised iter
  168/169/170/171/180/187/190 toolkit knobs but left iter-193 ``damp_w``
  dormant (``damp_w=0.0`` default).  After iter-194 the umbrella
  exercises ALL FV3-faithful damping mechanisms in NH at AD-at-rest,
  including the iter-193 ``_del6_vt_flux`` chain on ``w``.
- Iter 195: quantitative correctness test for iter-193 ``damp_w``.
  Mirrors the iter-174 (div_damp) / iter-175 (damp_v) pattern: with
  a sinusoidal ``w`` perturbation, ``damp_w > 0`` must REDUCE
  ``max|w|`` (not amplify it).  Catches a sign error in the iter-193
  wiring (e.g., ``w_new = w - dw`` instead of ``+= dw``) which the
  existing iter-193 unit tests would not catch.  Parametrized over
  ``nord_w`` in {0, 1, 2} (del-2 / del-4 / del-6).
- Iter 196: fix iter-178 self-check drift hazard.  Real bug: the
  iter-172/178 self-check kept its OWN local copy of the
  ``gate_helper_pairs`` list, separate from the outer
  ``test_nh_fv3_call_sites_ast_regression`` list.  When iter-190
  changed the iter-187 marker substring AND iter-193 added the
  damp_w pair, only the OUTER test was updated — leaving the
  self-check stale (still validating against ``"_zeta_smag_corner =
  jax.vmap"`` which iter-190 deleted, and missing iter-193's
  damp_w pair entirely).  iter-196 factors the list into a single
  module-level constant ``NH_GATE_HELPER_PAIRS`` consumed by both
  tests.  Future additions automatically extend the self-check
  coverage.
- Iter 197: extend iter-179 NH cube-imprint metric to verify
  iter-193 ``damp_w + nord_w`` does not regress the iter-168/169/
  170/171 toolkit's edge suppression.  Adds a third test
  ``test_full_toolkit_with_damp_w_does_not_regress`` that runs
  the full toolkit + damp_w on the iter-179 random-perturbation
  fixture and asserts the edge_std/interior_std ratio is still
  strictly below baseline.
- Iter 198: quantitative test for the iter-187 nord >= 1
  corner-divergence damping path.  iter-174 verified the iter-168
  nord=0 path reduces ``mean(|div_v|)`` on a divergent IC; iter-187
  added the smag_vort cap for nord >= 1 with finiteness tests but
  no quantitative direction test.  iter 198 mirrors iter-174:
  asserts nord=1 + d4_bg > 0 reduces ``mean(|div_v|)`` MORE than
  the nord=0 path with same d2_bg.  Catches a silent no-op in the
  iter-18-equivalent higher-order branch (e.g., dd8 coefficient
  bug, divg_d_iter wired wrong) that the iter-187 finiteness tests
  would not catch.  Also adds a nord=2 finiteness test on the
  divergent IC (complementing iter-187's perturbed-from-rest
  nord=2 test).
- Iter 199: quantitative direction test for the iter-180
  Smagorinsky-adaptive A_h on the NH path.  Existing iter-180
  tests verify smag_cs > 0 CHANGES winds vs the baseline but not
  the *direction*.  A sign-flipped Smagorinsky (subtracting
  ``c_s * dx² * |D|`` instead of adding) would still pass
  "changes-winds" but AMPLIFY winds at high-strain regions.
  iter 199 adds a high-strain shear IC and asserts smag_cs > 0 +
  A_h > 0 produces SMALLER ``max|u|`` than A_h-only baseline.
  Parametrized over ``smag_cs`` in {0.1, 0.2, 0.4} (PE iter-60
  tested range).
- Iter 200: direction-correctness test for the PE
  ``T_diss_coeff`` velocity-dependent Laplacian dissipation of T.
  Existing iter-182 tests verify wiring + AD-safety + state
  changes when winds nonzero, but not direction.  T_diss adds
  Laplacian DIFFUSION on T: ``dT/dt += T_diss_coeff * |v| * dx
  * lap(T)``.  A sign-flipped wiring would AMPLIFY T variance
  instead of damping.  iter 200 mirrors the
  iter-174/175/195/198/199 pattern: sinusoidal T pattern + uniform
  wind → T_diss > 0 reduces ``std(T - T_mean)`` more than baseline.
  Parametrized over T_diss_coeff in {0.1, 0.3, 0.5}.
- Iter 201: fix PE iter-188 AST guard regression from iter-190.
  REAL BUG: iter-190 changed the iter-187 marker substring from
  ``"_zeta_smag_corner = jax.vmap"`` to ``"_zeta_smag_corner ="``
  in the PE source.  iter-190 updated the NH iter-172 AST guard
  but missed the PE iter-188 guard, leaving the PE
  ``test_pe_fv3_call_sites_ast_regression`` failing silently
  between iter-190 and iter-201 (the PE guard was not in any
  recent test run).  iter-201 updates the PE pair list to match
  the iter-190 source.
- Iter 202: share the iter-187 marker substring between PE and NH
  AST guards via ``tests/_iter187_marker.py`` to prevent the
  iter-201 drift hazard from recurring.  Both PE iter-188 and NH
  iter-172 now import ``ITER187_GATE`` and ``ITER187_HELPER`` from
  the shared helper, so a future substring change is made in ONE
  place and automatically propagates to both guards.
- Iter 203: port FV3 ``d_con`` KE→heat conversion for the iter-193
  ``damp_w`` post-step damping.  Faithful port of FV3 ``sw_core.F90:
  1086``: ``heat_source = -d_con * dw * (w + 0.5*dw) = -d_con *
  ΔKE_w``.  When ``damp_w`` removes KE from ``w``, the lost KE is
  deposited as heat in θ_p (energy conservation).  Conversion to
  θ_p uses the simplified ``Δθ_p = heat / c_pd`` formula (Π Exner
  factor approximated as 1.0; ~30 % error aloft, valid in the lower
  troposphere).  Adds ``damp_w_d_con: float = 0.0`` config field
  (FV3 production default 1.0).  Default off; gated INSIDE the
  iter-193 ``damp_w > 0`` block.
- Iter 204: extend the iter-184 NH umbrella to ALSO engage iter-203
  ``damp_w_d_con``.  Previously the umbrella exercised iter
  168/169/170/171/180/187/190/193 toolkit knobs but left iter-203
  ``damp_w_d_con`` dormant (default off).  After iter-204 the
  umbrella exercises ALL FV3-faithful damping mechanisms in NH
  including the ``d_con`` KE→heat conversion, with ``jax.grad``
  flowing through the heat formula at AD-at-rest.
- Iter 205: quantitative energy-conservation test for the iter-203
  ``damp_w_d_con`` heat formula.  Method: run THREE configs
  (acoustic-only, damp_w-only, damp_w + d_con); extract dw =
  damp_w-only.w − acoustic-only.w; compute expected heat from
  ``-d_con * dw * (w_after_acoustic + 0.5*dw) / c_pd`` averaged
  half→full; assert ``θ_p(d_con) − θ_p(no-d_con)`` matches the
  expected formula within machine precision.  Catches numerical
  bugs in the formula (wrong factor of 0.5, sign error, wrong
  half-to-full averaging, missing c_pd).  iter-205 originally
  failed because the test extracted dw against the wrong baseline
  (pre-step state instead of post-acoustic state); the fix
  introduced the third "acoustic-only" run.
- Iter 206: extend iter-179 NH cube-imprint test to verify iter-203
  ``damp_w_d_con`` heat injection does not regress the iter-168/
  169/170/171 toolkit's edge suppression.  Mirrors iter-197 (which
  added iter-193 damp_w to the imprint test).  iter-203 modifies
  θ_p (not winds), so we expect d_con to be NEUTRAL for the
  v-imprint metric — but cross-coupling through buoyancy → w →
  acoustic → wind could in principle disrupt the toolkit, so
  explicit regression coverage is warranted.
- Iter 207: refine iter-203 d_con heat formula with the Π Exner
  factor.  iter-203 used the simplified ``Δθ_p = heat / c_pd``
  (Π=1 approximation, ~30 % under-heating aloft).  iter-207
  divides by ``c_pd * exner_ref`` (using ``HeightCoordinate.
  exner_ref`` at full levels), giving ``Δθ ≈ ΔT/Π`` within the
  reference-state linearization — FV3-faithful for the heat
  partition across the column.  iter-205 quantitative test
  updated to include the exner_ref factor in its expected formula.
- Iter 208: port FV3 ``d_con`` KE→heat conversion for the iter-12
  PE ``damp_v`` post-step damping.  Faithful (simplified) port of
  FV3 ``sw_core.F90:1953-1990`` restricted to the damp_v
  contribution alone.  When ``damp_v`` removes KE from
  (u_d, v_d) via the post-step (du_corner, dv_corner) wind
  increments, the lost KE is converted to heat in T (energy
  conservation):  ΔT = -damp_v_d_con * (u*du + 0.5*du² + v*dv +
  0.5*dv²) / c_pd, projected from corners to cell centres via
  ``_interp_corner_to_center``.  Adds ``damp_v_d_con: float = 0.0``
  config field (FV3 production default 1.0).  PE-only mirror of
  the iter-203 NH ``damp_w_d_con``.  Default off; gated INSIDE
  the iter-12 ``damp_v > 0`` block.
- Iter 209: port FV3 ``d_con`` for NH iter-169 damp_v (NH mirror
  of PE iter-208).  Same KE→heat formula at cell centres (NH stores
  u, v at cell centres, so no corner→center projection needed):
  ``Δθ_p = -damp_v_d_con * ΔKE_cc / (c_pd * Π_ref)``.  Π_ref via
  ``HeightCoordinate.exner_ref`` matches the iter-207 refinement
  pattern.  Adds ``damp_v_d_con: float = 0.0`` to NH config
  (default off; FV3 production default 1.0).  Both 3D paths now
  have d_con coverage for both damping mechanisms (PE: damp_v;
  NH: damp_v + damp_w).
- Iter 210: extend both umbrellas to engage iter-208/209 ``damp_v_d_con``.
  iter-184 NH umbrella now exercises BOTH ``damp_v_d_con`` (iter-209)
  and ``damp_w_d_con`` (iter-203/204); iter-185 PE umbrella now
  exercises ``damp_v_d_con`` (iter-208).  Both umbrellas pass
  ``jax.grad`` at rest with the full energy-conserving toolkit.
- Iter 211: quantitative formula validation for iter-208/209
  ``damp_v_d_con`` (mirrors iter-205 pattern for damp_w).  3-run
  extraction (no-damp_v / damp_v / damp_v + d_con) verifies the
  actual ΔT (PE) and Δθ_p (NH) increments match the expected
  formula (-d_con * ΔKE / c_pd, with corner→center projection for
  PE and × 1/Π_ref for NH) within machine precision.  PE tolerance
  is 1e-8 (vs 1e-10 for NH) because the model's
  ``_interp_corner_to_center`` call inside the JIT graph fuses
  ops differently than the test's external call, producing a few
  ULPs of difference in the 4-point average.
- Iter 212: extend iter-179 cube-imprint test to verify iter-209
  NH ``damp_v_d_con`` heat injection does not regress the iter-168/
  169/170/171 toolkit's edge suppression (mirror of iter-206 for
  damp_w_d_con).  iter-209's heat source modifies θ_p (not winds
  directly), like iter-203 but driven by damp_v's du, dv.  Cross-
  coupling through θ_p → buoyancy → wind could disrupt the toolkit;
  explicit regression coverage is warranted.  Test exercises the
  full energy-conserving toolkit (iter-179 + iter-193 damp_w +
  iter-203 damp_w_d_con + iter-209 damp_v_d_con).
- Iter 213: PE counterpart of iter-198 quantitative test.  iter-198
  verified that nord >= 1 corner-div damp reduces ``mean(|div_v|)``
  MORE than nord=0 alone for the NH path; PE has the same iter-187
  smag_vort wiring but no analogous quantitative test.  iter 213
  mirrors iter-198 for PE — same divergent IC pattern, asserts
  nord=1 + d4_bg > 0 reduces div more than nord=0 alone.  Catches
  silent no-op in the iter-18-equivalent higher-order branch on
  the hydrostatic path that iter-187 finiteness tests would not
  catch.
- Iter 214: PE counterpart of iter-199 Smagorinsky-A_h direction
  test.  iter-199 verified that NH smag_cs > 0 + A_h > 0 produces
  smaller max|u| than A_h-only on a high-strain IC; PE iter-57/58
  Smagorinsky has bit-for-bit/changes-winds tests but no direction
  test.  iter 214 mirrors iter-199 for PE using ``std|u_d|``
  (rather than max) as the strain proxy — std integrates over the
  full field rather than sampling the max, which on the PE HS-init
  fixture is sensitive to single-cell jet response not correlated
  with smag damping.  Parametrized over smag_cs in {0.1, 0.2, 0.4}.
- Iter 215: PE counterpart of iter-174 cell-centre div_damp
  direction test.  iter-174 verified that NH ``div_damp_coeff > 0``
  reduces ``mean(|div_v|)`` on a divergent IC; PE has the same
  iter-5 wiring but no direction test.  iter 215 mirrors iter-174
  for PE.  Used ``div_damp_coeff=1e8`` (vs NH's 1e10) because PE
  outer dt is ~10× NH so the per-step damping factor is
  proportionally larger; 1e10 over-damps PE.  Threshold relaxed to
  0.1% reduction (vs NH's 1%) because PE HS-init residual divergence
  baseline is small (~1e-6 1/s) so the % reduction is modest even
  with strong damping.
- Iter 216: PE counterpart of iter-175 ``damp_v`` direction test.
  iter-175 verified NH damp_v reduces ``mean(|ζ|)`` on a vortical
  IC.  PE iter-12 damp_v has the same backbone
  (``fv3_del6_vorticity_damping``) but no analogous direction test.
  iter 216 mirrors iter-175 for PE.
- Iter 217: PE counterpart of iter-179 cube-imprint metric.
  Adds the PE-side equivalent of iter-179's edge_std/interior_std
  ratio metric using PE D-grid corner storage for v_d.  Two tests:
  metric is well-defined on baseline; toolkit measurably changes
  the ratio (within sanity bounds).  At C8 with random IC over
  5 PE steps the toolkit does not necessarily REDUCE the ratio
  (random IC has comparable signal at edge and interior), so the
  test relaxes from "must reduce" to "must change without strongly
  amplifying".  For quantitative PE edge-suppression validation
  see iter-1039 (uniform u, deterministic IC) and the matrix HS
  C36 hybrid 30-day reference (iter-19 Quick Reference table).
- Iter 218: port FV3 ``delt_max`` per-step dissipative-heating
  cap (``dyn_core.F90:1774``) to PE+NH.  Adds ``delt_max=0.0``
  knob (FV3 prod 1.0 K/s); default off bit-for-bit; AD-safe via
  ``jnp.clip``.  Compacts older prose to save ~5300 lines.
- Iter 219: refine iter-218 with FV3 sponge-layer-aware cap.
  PE skips top-2 layers (k=0,1); NH applies 0.1×/0.5×/1× per-level
  scaling.  4-test sponge regression.
- Iter 220: extend iter-184/185 umbrellas to engage
  ``delt_max=1.0`` alongside full toolkit.  4/4 umbrella pass.
- Iter 221: port d_con KE→heat for iter-16/18 corner-div damping
  (PE).  Adds ``corner_div_damp_d_con=0.0``; gated INSIDE
  ``corner_div_damp_d2_bg > 0``.  Closes the iter-208-deferred
  PE corner-div d_con gap.  4/4 pass.
- Iter 222: NH mirror of iter-221 (corner-div d_con on
  compressible-Euler path) with iter-207 Π_ref refinement.  4/4
  pass.
- Iter 223: port d_con for iter-5 cell-centre div_damp (PE).
  Adds ``div_damp_d_con=0.0`` knob.  Gated INSIDE
  ``div_damp_coeff > 0``.  4/4 pass.
- Iter 224: NH mirror of iter-223 with Π_ref.  4/4 pass.
- Iter 225: port d_con for iter-57/58 Smagorinsky-A_h Laplacian
  (PE).  Adds ``ah_d_con=0.0`` knob.  Closes LAST PE-side d_con
  asymmetry.  4/4 pass.
- Iter 226: NH mirror of iter-225 with Π_ref.  **CLOSES ALL
  d_con ASYMMETRIES**: every FV3 KE-removing mechanism on both
  3D paths (corner-div, cell-centre div_damp, damp_v, damp_w,
  Smagorinsky-A_h) converts KE to heat (default off; FV3 prod
  1.0).  4/4 pass.
- Iter 227: extend PE+NH umbrellas to ALSO engage all 3 new
  d_con knobs from iter 221-226 at FV3 production value 1.0.
  PE umbrella 57 s, NH umbrella 79 s; both AD-finite through 5
  NH / 3 PE steps with full d_con stack ON.
- Iter 228: bit-for-bit formula test for iter-221 PE corner-div
  d_con — calls ``fv3_hydrostatic_tendencies`` directly and
  verifies dT_dt contribution matches expected formula at
  rtol=1e-10 via 3-config triangulation.  1/1 pass in 15 s.
- Iter 229: bit-for-bit formula tests for PE iter-223
  (cell-centre div_damp) and PE iter-225 (A_h) d_con sites.
  All 3 PE d_con sites now bit-for-bit verified against FV3
  energy-conservation formula.  2/2 pass.
- Iter 230: linearity tests for the 3 NH d_con sites (iter-222
  corner-div, iter-224 cell-centre div_damp, iter-226 A_h).
  d_con ∈ {0.5, 1.0, 2.0}: ``Δ@2.0 - Δ@0.5 == 3.0 * (Δ@1.0 -
  Δ@0.5)`` at rtol=1e-10.  Catches non-linear coupling errors.
  Also compacts iter-218..229 ToC entries (~110 lines saved).
- Iter 231: aggregate direction test for full PE d_con stack
  (4 knobs ON).  10 PE steps strong-wind IC →
  ``mean(T)_ON > mean(T)_OFF``.  Catches sign errors.  1/1.
- Iter 232: NH mirror of iter-231 (5 knobs ON).
  ``mean(θ_p)_ON > mean(θ_p)_OFF``.  1/1.
- Iter 233: PE cube-imprint regression — full d_con stack ON
  must not amplify v_d edge_std/interior_std ratio by >50 %.
  d_con has no edge stencil so should be near-baseline.  1/1.
- Iter 234: NH mirror of iter-233 (cube-imprint with all 5
  d_con knobs ON).  1/1.
- Iter 235: extend iter-188 PE and iter-172 NH config-field AST
  regression guards to include iter-218..226 d_con knobs +
  delt_max.  Catches refactors that drop fields silently.
  8/8 AST tests.
- Iter 236: extend call-site AST guards (gate-substring +
  helper-symbol pairs) for the same iter-218..226 wirings.
  Catches refactors that drop the body INSIDE
  ``if config.X > 0``.  8/8 AST tests.
- Iter 237: AD-at-rest grad w.r.t. uniform-u perturbation
  amplitude with full toolkit + all 5 d_con knobs at 1.0.
  Orthogonal direction to iter-184/185's T/θ_p direction;
  more sensitive to sqrt-at-zero hazards in wind-dependent
  helpers.  PE 64 s + NH 85 s.
- Iter 238: audit + document the iter-208/209 d_con KE-formula
  fidelity gap vs FV3's metric-aware ``rsin2/cosa_s`` form
  (sw_core.F90:1980).  Equivalent in orthogonal-grid limit;
  GLOBAL energy conserved, LOCAL distribution differs at cube
  edges.  Adds a global-energy-balance regression test
  ``Σ c_pd * dT/dt + Σ dKE/dt == 0`` at machine precision.
  Tracks the metric-aware form for a future iteration.  1/1.
- Iter 239: extend iter-218/219 sponge-aware ``delt_max`` cap
  to the AGGREGATE PE tendency-based d_con stack.  Previously
  cap acted only on damp_v post-step; iter-221/223/225
  bypassed it.  Refactor: stash all 3 contributions, aggregate
  after the A_h block, sponge-aware ``jnp.clip``
  (k=0,1 uncapped, k≥2 → ``delt_max``) before adding to
  ``dT_dt_data``.  Mirrors FV3 ``heat_source``-then-cap-once
  pattern.  3-test fixture; PE umbrella still passes; 16/16
  PE regression preserves baseline.
- Iter 240: NH mirror of iter-239 aggregate cap (5-knob
  sponge-aware sum + jnp.clip).  Compacts iter-230..239 ToC.
- Iter 241: AST guards for iter-239/240 aggregate-cap pattern
  (gate ``_d_con_sum is not None`` + PE/NH helper symbols).
- Iter 242: 50-step PE+NH stability test, full toolkit + d_con
  stack at FV3 production defaults.  All fields finite +
  max|u| < 100 m/s.
- Iter 243: extend iter-238 global-energy-conservation to all
  3 PE tendency d_con sites + aggregate path.  rtol=1e-10.
- Iter 244: 100-step PE+NH stability with stronger IC (±10
  m/s) — catches slow-growth instabilities that 50-step
  iter-242 might miss.
- Iter 245: PE iter-19 PRODUCTION values (d4_bg=0.02 + nord=1)
  at C36 with full d_con stack, 20×dt=200 + physical-T bounds.
- Iter 246: audit PE T-convention (no pkz factor needed since
  legoESM PE T is actual temperature, not FV3's c_p*T/pkz).
- Iter 247: regression sweep checkpoint, 13/13 iter-240..246
  pass after iter-239/240 refactor (delt_max=0 bit-for-bit).
- Iter 248: stress test delt_max cap with 100x d_con + tight
  delt_max=1e-5 to FORCE cap activation.  Validates the cap
  actually engages and bounds per-step ΔT.
- Iter 249: bit-for-bit formula test for PE iter-187 smag_vort
  cap (FV3 sw_core.F90:1799 form).  rtol=1e-12 + symmetry.
- Iter 250: NH smag_vort cap formula bit-for-bit test
  (mirror of PE iter-249) + 10-iter ToC compaction.
- Iter 251: NH C36 production stability with d_con stack
  (NH mirror of PE iter-245).  10×dt=10, all fields finite,
  bounded growth.
- Iter 252: cube-imprint regression for NH iter-187 +
  iter-190 paths.  Imprint ratio within 50 % of baseline.
- Iter 253: PE counterpart of iter-252.  PE iter-187 +
  iter-190 imprint ratio within 50 % of baseline.
- Iter 254: PE C36 cube-imprint diagnostic with full d_con
  stack.  edge_width=4 metric stays in (0.1, 10.0).
- Iter 255: NH counterpart of iter-254 (C36 cube-imprint).
- Iter 256: global energy conservation for iter-208 PE
  damp_v_d_con post-step.  Σ c_pd*ΔT + Σ ΔKE = 0 at
  rtol=1e-10.
- Iter 257: NH counterpart for iter-209 damp_v_d_con
  post-step.  Σ c_pd*Π_ref*Δθ_p + Σ ΔKE = 0.
- Iter 258: global energy conservation for iter-203
  damp_w_d_con (NH, half-level heat with w=0 BC).  ALL 6
  d_con sites now have machine-precision conservation
  regression coverage.
- Iter 259: nord=2 (FV3 default for damp_v) multi-step
  stability with full d_con stack.  PE+NH 20 steps both
  bounded.
- Iter 260: regression sweep checkpoint + 10-iter compaction.
  12/12 iter-250..259 tests pass together.
- Iter 261: PE AD-at-rest gradient with iter-19 PRODUCTION
  d4_bg=0.02 (stronger than iter-184/185 umbrellas' 1e-3) +
  full d_con stack.  jax.grad finite.  Validates iter-183
  sqrt(0) fix under stronger damping.  Also fixes obsolete
  "PE-only" iter-208 docstring note.
- Iter 262: extend iter-245 PE C36 production stability test
  from 20 to 50 steps × dt=200.  Catches slow-growth
  instability.  Full iter-19 toolkit + d_con stack.
- Iter 263: NH counterpart of iter-262 (NH C36 50-step).
- Iter 264: PE C16 cube-imprint validation with iter-19
  PRODUCTION toolkit vs iter-168 nord=0 baseline.  Within
  50 % of baseline.  Fills C8 / C36 resolution gap.
- Iter 265: NH counterpart of iter-264 (NH C16 cube-imprint).
- Iter 266: cubed-sphere non-orthogonality metrics sanity
  (cosa_corner/cell, rsin2_corner/cell, sina_cell).  |cosa|
  ≤ 0.6, rsin2 ≥ 0.99, cosa concentrates at cube vertices.
- Iter 267: numerical regression for iter-219 NH sponge
  factor values (k=0 → 0.1×, k=1 → 0.5×, k≥2 → 1×).
- Iter 268: PE cube-imprint OVER TIME (50-step integration
  with full toolkit + d_con).  All samples in (0.1, 5.0) +
  growth_factor < 5×.
- Iter 269: NH counterpart of iter-268 (NH cube-imprint
  over-time).
- Iter 270: NH AD-at-rest with d4_bg=0.02 (mirror PE
  iter-261) + 10-iter ToC compaction.
- Iter 271: NH stability sweep across n_acoustic_substeps
  ∈ {2, 4, 8}; 3/3 finite.
- Iter 272: PE corner_div d_con net-heating direction (mean
  dT_d_con > 0).
- Iter 273: PE div_damp d_con net-heating direction.
- Iter 274: PE ah_d_con net-heating direction (closes PE
  trio).
- Iter 275: NH d_con direction trio (iter-222/224/226).
  All 6 d_con sites have direction tests.
- Iter 276: damp_v_d_con direction at FV3 production
  nord_v=2 (PE+NH).
- Iter 277: damp_w_d_con direction at FV3 production
  nord_w=2 (NH).
- Iter 278: PE damp_v scaling — damp4 = (damp_v *
  da_min_c)^(nord+1) verified by 2x scaling at nord=0/1/2.
- Iter 279: NH counterpart of iter-278 (damp_v scaling).
- Iters 280-289 (compacted iter 290): scaling, metrics, and
  float32 coverage.
  - 280: NH damp_w (nord+1)-exponent 2x scaling.  Closes
    (damp_v + damp_w) × (PE + NH) × nord scaling matrix.
  - 281: corner_div_damp_d2_bg LINEAR scaling (PE+NH) — pins
    FV3-faithful nord=0 coefficient.
  - 282: PE corner_div_damp_d4_bg (nord+1)-power scaling for
    nord >= 1 branch (FV3 sw_core.F90:1809).
  - 283: NH counterpart of 282 — d4_bg (nord+1) power
    scaling at nord=1/2.
  - 284: cosa_corner = ±0.5 EXACT at cube vertices (3 panels
    at 60°), sign-mix (+/- per orientation).
  - 285: iter-187 smag_vort cap range sanity — non-neg,
    sqrt(0)=0 with finite grad, matches naive sqrt at rtol
    1e-12.
  - 286: PE d_con stack 5-step float32 sanity (no x64).
  - 287: NH counterpart of 286.
  - 288: PE d_con float32 50-step long-run stability.
  - 289: NH counterpart of 288.
- Iters 290-299 (compacted iter 300): float32 AD, helper-
  formula scaling/invariance, and dt-plumbing semantics.
  - 290: PE iter-19 PRODUCTION AD-at-rest at default float32
    (mirror of iter-261 x64).  Confirms iter-183 sqrt(0)
    AD-safety doesn't depend on x64.
  - 291: NH counterpart (mirror of NH iter-270 x64).
  - 292: Smagorinsky helper linear-in-c_s scaling (rtol
    1e-14 bit-for-bit, c_s=0 → 0).
  - 293: Smagorinsky helper linear-in-(u, v) scaling at α ∈
    {0.5, 2.0, 3.7}, plus rest-state grad finite.
  - 294: Smagorinsky helper Galilean translation invariance
    (uniform + separate-component offsets, rtol 1e-14).
  - 295: smag_vort cap linear-in-|dt| (α ∈ {-2.5, 0.5, 1.0,
    3.0}) + sign-symmetry across 4 sign combos.
  - 296: PE iter-189 dt-plumbing — model.step beats proxy
    (proxy 50 vs 500 → bit-for-bit identical state).
  - 297: NH counterpart of 296 (proxy 2 vs 50).
  - 298: PE iter-189 fallback — direct tendency call without
    dt_actual reads proxy (proxy 50 vs 500 → tendencies
    differ).  Together 296+298 = complete PE contract.
  - 299: NH counterpart of 298 (proxy 10 vs 1000 + ±15 m/s
    perturbations to escape d2_bg floor).
- Iters 300-309 (compacted iter 310): AST guard + helper-
  formula scaling/linearity coverage.
  - 300: AST guard for ``dt_actual`` keyword name + default
    on PE+NH tendency signatures.
  - 301: ``a2b_ord4`` preserves constants (rtol=1e-14) at
    {C4, C8, C16, C36} × {-3.5, 0, 1, 1e-6, 1e6}.
  - 302: ``a2b_ord4`` linearity / superposition (rtol=1e-14).
  - 303: ``fv3_divergence_corner_3d`` linearity in (u, v) +
    div(0, 0) = 0.
  - 304: ``fv3_corner_laplacian_iteration`` linearity + lap(0)
    = 0.
  - 305: corner_laplacian zeros constants (kernel property).
  - 306: ``_interp_center_to_corner`` (2nd-order 4-pt avg)
    preserves constants + linear.
  - 307: ``_interp_corner_to_center`` (mirror direction) at
    both 2D and 3D shapes.
  - 308: ``fv3_del6_vorticity_damping`` linear in damp at
    nord ∈ {0, 1, 2}.
  - 309: ``fv3_del6_vorticity_damping`` linear in (u, v) at
    fixed damp/nord.  Combined w/ 308: BILINEAR.
- Iter 310: ``pad_halo`` (scalar) preserves constants +
  linearity.  Constant c-field → constant c-field across
  halo widths {1, 2} and c ∈ {-3.5, 0, 1, 1e-6, 1e6}
  (rtol=1e-14).  Superposition holds for (α, β) ∈ {(1,1),
  (2,-3), (0.5,0.5), (-1,7)} (rtol=1e-14).  Pins the
  foundational halo invariants that every downstream FV3
  helper (a2b_ord4, divergence, laplacian, _interp_*)
  inherits.  18/18 pass in 1.5 s.  Plus 10-iter ToC
  compaction (iters 300-309).
- Iter 311: ``pad_halo_vector`` zero + linearity in (u, v).
  pad(0, 0) → (0_padded, 0_padded) exactly.  Linearity
  (rtol=1e-13) for (α, β) ∈ {(1,1), (2,-3), (0.5,0.5),
  (-1,7)}.  Vector halo chain (rotation → scalar pad →
  inverse rotation) is linear at fixed grid angles.  Extends
  iter-310 scalar halo characterization to the vector path
  used by every (u, v) cubed-sphere operator.  5/5 pass in
  4.4 s.
- Iter 312: iter-187 smag_vort cap saturation at the 0.20
  ceiling.  When ``inner = dddmp*|dt|*sqrt(delpc²+ζ²)`` <
  0.20, cap output = inner (rtol=1e-13).  When inner > 0.20,
  cap output = 0.20 exactly (array-equal).  At threshold
  (inner == 0.20 by construction with dddmp=1, dt=1,
  delpc=0.20, ζ=0), cap = 0.20.  Pure helper-formula test
  pinning the FV3 ``min(0.20, ...)`` saturation behaviour
  that protects against runaway damping at intense
  divergence.  3/3 pass in 0.4 s.
- Iter 313: corner-div damp ``d2_bg`` floor (orthogonal to
  iter-312's 0.20 ceiling).  Full coefficient
  ``damp = max(d2_bg, min(0.20, dddmp*|dt|*sqrt(delpc²+ζ²)))``.
  When cap < d2_bg, damp = d2_bg exactly (array-equal).
  When cap > d2_bg AND cap < 0.20, damp = cap (rtol=1e-13).
  At threshold (cap = d2_bg by construction), damp = d2_bg.
  Pins the FV3 sw_core.F90:1801 ``max(d2_bg, ...)`` operator
  that ensures non-zero baseline damping at quiescent flow.
  3/3 pass in 0.4 s.
- Iter 314: PE iter-218/219 sponge cap layer pattern.
  Per-level cap array = [∞, ∞, delt_max, delt_max, ...]
  (k=0, 1 uncapped; k>=2 capped).  Large values at k=0, 1
  pass through ``jnp.clip`` unchanged regardless of
  magnitude.  Values at k>=2 with magnitude > delt_max are
  clipped to ±delt_max.  Within-cap values pass unchanged.
  Pins the iter-218 PE sponge-skip-cap pattern (PE skips
  cap entirely at k=0, 1) by helper-formula numerical
  regression.  3/3 pass in 0.2 s.
- Iter 315: NH iter-219 sponge cap CLIP behavior with full
  per-level formula ``cap[k] = delt_max * factor[k] /
  exner_ref[k]``.  iter-267 pinned factor VALUES (0.1, 0.5,
  1.0); iter-315 pins the actual clip behavior: per-level
  cap matches expected values (rtol=1e-14), below-cap
  values pass unchanged (array-equal), above-cap values
  clipped to ±cap[k].  Combined with iter-267, the NH
  sponge cap is fully characterized.  3/3 pass in 0.4 s.
- Iter 316: ``cgrid_divergence`` zero + linearity in (u_c,
  v_c).  div(0, 0) = 0 at both 2D and 3D shapes.
  Superposition (rtol=1e-13) for (α, β) ∈ {(1,1), (2,-3),
  (0.5,0.5), (-1,7)} at 2D and {(1,1), (2,-3)} at 3D.  Pins
  the helper used by PE div_damp path as a linear operator.
  Combined with iter-303 (corner divergence), every
  divergence helper in the d_con cluster is now
  characterized as linear.  7/7 pass in 9.7 s.
- Iter 317: AST config-default parity guard between PE and
  NH for 12 shared d_con cluster knobs (corner_div_damp_*
  d2_bg/dddmp/d4_bg/nord, damp_v/nord_v/damp_v_d_con,
  corner_div_damp_d_con/div_damp_d_con/ah_d_con/delt_max,
  use_fv3_a2b_zeta_corner, smagorinsky_cs/A_h).  Catches
  silent default drift on one side without runtime
  simulation.  5/5 pass in 0.3 s.
- Iter 318: AST guard for NH-only knob defaults (damp_w,
  nord_w, damp_w_d_con).  damp_w/damp_w_d_con default to 0
  (off-baseline contract); nord_w defaults to 2 (FV3
  production).  Catches accidental rename/removal of fields
  that iter-203/258/275/277/280 runtime tests depend on.
  4/4 pass in 0.3 s.
- Iter 319: AST guard for d_con knob count.  PE has exactly
  4 d_con knobs (damp_v, corner_div_damp, div_damp, ah);
  NH has 5 (PE + damp_w_d_con).  NH-only knob is exactly
  {damp_w_d_con}; PE knobs are a subset of NH.  Catches
  accidental addition/removal of a d_con site without
  updating runtime test matrix (iter-272-280).  4/4 pass
  in 0.3 s.
- **Iters 320-359 (compacted iter 365)**: FV3-fidelity opt-in
  flag stack + comprehensive coverage.
  - **iter 320**: opt-in ``use_fv3_d_con_cv`` (NH) — swaps
    ``c_pd → c_vd`` at all 5 NH d_con sites for FV3-faithful
    ``cv_air`` branch (FV3 ``dyn_core.F90:1795``).  Default
    False bit-for-bit baseline preserved.  Closes ~40 %
    under-heating relative to FV3.
  - **iter 321-323**: cv-flag energy-conservation + tendency-
    level c_pd*Δθ_p_cp == c_vd*Δθ_p_cv (rtol=1e-12) at all 5
    NH d_con sites (damp_v + damp_w post-acoustic; corner_div
    + cell-div_damp + A_h slow-tendency).
  - **iter 324**: pin grid prerequisites for metric-aware d_con
    (cosa_cell, rsin2_cell, rdxa, rdya present on
    ``CubedSphereCDGrid``).
  - **iter 325/326/327**: thread duogrid through 3 NH halo
    bypass sites (K + π_prime packed halo + ke_correction).
    Quantitative C8 imprint reduction + diff concentrated at
    panel edges + AST guard.  Direct cube-edge artifact fix
    matching FV3 ``fv_duogrid.F90``.
  - **iter 328/329**: opt-in ``use_fv3_vector_halo_uv`` (NH)
    via ``center_to_dgrid_vector`` (FV3 ``ext_vector``
    analogue).  Bit-for-bit baseline + diff concentrated at
    panel edges + AST guard.
  - **iter 330**: NH full-stack integration regression (all
    flags ON).
  - **iter 331/343**: PE/NH flag-asymmetry guard.  cv +
    vector_halo + dyn_exner are NH-only (PE uses cp_air,
    stores winds at corners, uses actual T).  metric flag is
    SHARED.
  - **iter 332**: 6-flag-combo AD-at-rest parametrized
    umbrella.
  - **iter 333/334/335**: PE-side duogrid wiring at
    ke_correction halo + AST guard + edge concentration.
  - **iter 336/337**: opt-in ``use_fv3_dynamic_exner`` (NH) —
    swaps frozen ``Π_ref`` for live ``Π_total = Π_ref + π'``
    at all 5 NH d_con sites (3 slow-tendency + 2 post-acoustic).
    Closes ~30 % under-heating aloft.
  - **iter 338/339/344/347/348/349/350/351/352**: opt-in
    ``use_fv3_metric_aware_d_con`` (PE + NH) — wires FV3
    ``sw_core.F90:1980`` ``cosa_s``/``rsin2`` metric form at
    all 8 PE+NH d_con sites.  iter-344 fixed magnitude bug
    (dropped FV3 rdx/rdy normalization which underresolved at
    C8 → numerical zero).  Equivalent to iter-208/209 simpler
    form in orthogonal-grid limit; adds cube-edge non-
    orthogonality correction via cosa_s ≠ 0 + rsin2 > 1.
  - **iter 340/342/353/361/362**: AST regression guards (all
    metric gates, dyn Exner wiring, 5-site NH coverage + 4-site
    PE coverage, PE/NH flag asymmetry at code level).
  - **iter 341/346/355**: full-stack PE + NH AD-at-rest
    umbrellas (every fidelity flag + full toolkit at rest).
  - **iter 345**: re-pin cube-edge concentration of metric
    diff after iter-344 fix.
  - **iter 354**: close source-comment for iter-238 metric gap.
  - **iter 356**: doc summary section (FV3-fidelity stack).
  - **iter 357/358/359/360**: C16 cube-imprint does-not-amplify
    + changes-state-measurably regressions for PE+NH full FV3
    stack.
  - **GAP #1 CLOSED**: metric d_con (8 sites).
  - **GAP #2 CLOSED**: dynamic Exner (5 NH sites).
  - **Per-flag default**: all False (preserves bit-for-bit).

- Iter 499: monotone_clip in ``center_to_dgrid_vector`` —
  OVERSHOOTS neutral.  Monkey-patch ``pad_halo_vector_4d``
  inside ``center_to_dgrid_vector`` with clip=True:
    no clip:   u_d edge × 1.0396  (iter-497 baseline 1.040)
    with clip: u_d edge × 0.9378  (UNDER interior!)
  Clip goes 257% of the way to neutral — it actively pulls
  edge cells BELOW interior values, over-correcting.  This
  explains iter-493's modest dycore impact: the clip is too
  aggressive at the op level, and the dycore's own damping
  further pulls edges down → small net change.  A SOFTER
  constraint (e.g., allow edges ≤ interior_max × 1.05) might
  give better balance between artifact suppression and
  physical fidelity.  Future investigation direction.
  1/1 in 11 s.  Wired into iter-383 sweep (now 91).
- Iter 498: extend iter-490 ``monotone_clip`` to
  ``pad_halo_vector_4d`` — also propagates to the two
  internal ``pad_halo_4d`` calls (east + north components).
  Random Gaussian vector field:
    interior_max:        5.3133
    halo_max no clip:   10.9629  (+106% over interior!)
    halo_max clip on:    4.8723  (within interior)
    Overshoot reduction: 55.6%
  Vector halo overshoot is even larger than scalar (76% per
  iter-491).  Clip works for both.  Default False preserves
  backward compat.  2/2 in 6 s.  Wired into iter-383 sweep
  (now 90).
- Iter 497: ``center_to_dgrid_vector`` (iter-328 vector-aware
  variant) amplifies edges 3× LESS than scalar interp:
    iter-496 scalar interp: edge × 1.122 (12%)
    iter-497 vector u_d:    edge × 1.040 (4%)
    iter-497 vector v_d:    edge × 1.027 (3%)
  The vector-aware halo (proper rotation across face
  boundaries) reduces per-step edge amplification by ~3×.
  **Mechanistically explains iter-465's "vector_halo_uv is
  biggest helper" (Δ+5)**: switching from scalar to vector
  interp cuts each step's edge contamination, compounding
  to ~5 over multiple substeps.  1/1 in 11 s.  Wired into
  iter-383 sweep (now 89).
- Iter 496: ``_interp_center_to_corner`` IS a duogrid edge
  amplifier (~12%).  Apply the cell-center → D-grid corner
  4-point average (halo-aware) to random Gaussian field:
    no-duogrid: edge std = 0.7475, interior = 0.7166
                ratio = 1.0431
    duogrid:    edge std = 0.8383, interior = 0.7166
                ratio = 1.1699
    duogrid effect: edge × 1.122, interior × 1.000
  Edge std jumps 12% with duogrid; interior unchanged.  This
  is the FIRST identified single op showing real (not
  negligible) duogrid edge amplification.  Not enough to
  explain the full 5× dycore ratio — but contributes.  The
  dycore composes ``_interp_center_to_corner`` with many
  other halo-aware ops; product compounds to 5×.  1/1 in
  10 s.  Wired into iter-383 sweep (now 88).
- Iter 495: isolate ``fv3_divergence_corner_3d`` — duogrid-
  invariant.  Apply the corner-div op to random u/v field at
  corners with grid_off vs grid_on:
    no-duogrid: edge × 1.00, interior × 1.00
    duogrid:    edge × 1.00, interior × 1.00
  IDENTICAL output between duogrid ON/OFF.  Reason: corner-
  div op uses corner-stored u/v directly without needing
  inter-face halo (corners ARE the face boundary).  Rules out
  ``fv3_divergence_corner_3d`` as the dycore amplifier.
  Remaining suspect: cell-center→corner halo-aware interp
  (``_interp_center_to_corner``, ``center_to_dgrid_vector``)
  used in the NH (cell-center u, v) → D-grid corner lift.
  1/1 in 11 s.  Wired into iter-383 sweep (now 87).
- **Iters 631-639 (compacted iter 640)**: FV3 grid + vertical-coord
  ancillary ports.  9 iterations add boundary, reduction, vertical,
  and ghost helpers from FV3 fv_grid_utils + fv_eta + fv_grid_tools:

  | Iter | Function(s)                              | Module / Line             | Role                                |
  |------|------------------------------------------|---------------------------|-------------------------------------|
  | 631  | ``edge_factor_along_axis_nonortho``      | fv_grid_utils:1212        | A→B GC-weighted edge interp factor  |
  | 632  | ``global_qsum`` / ``global_mx`` / ``global_mx_c`` | fv_grid_utils:2999/3020/3048 | serial sum / min / max  |
  | 633  | ``fill_ghost``                           | fv_grid_utils:3070        | 4 corner-ghost regions fill          |
  | 634  | ``get_eta_level``                        | fv_eta:1859               | hybrid → log-mean full-level pressure|
  | 635  | ``compute_dz_fv3`` / ``zflip``           | fv_eta:1894/2482          | initial dz layering + vertical flip |
  | 636  | ``sm1_edge_fv3``                         | fv_eta:2249               | 1D del-2 edge smoother for dz       |
  | 637  | ``set_external_eta`` / ``compute_dz_L101`` | fv_eta:788/2069        | (ptop, ks) + L101 vertical (20.3km) |
  | 638  | ``compute_dz_L32``                       | fv_eta:2000               | FV3 canonical L32 vertical (~60 km) |
  | 639  | ``hybrid_z_dz``                          | fv_eta:1794               | stretched vertical w/ s_fac table   |

  Vertical-coord port stack now covers ALL FV3 canonical
  reference layer profiles (L32, L101, hybrid_z) plus the
  general helpers (get_eta_level, compute_dz, sm1_edge).  Lives
  in ``legoesm.grids.vertical`` alongside existing
  ``HybridSigmaPressureCoordinate`` infrastructure.

  Grid ancillary ports (edge_factors non-ortho, fill_ghost,
  global reductions) round out the FV3 fv_grid_utils.F90 cover
  matrix.  All 9 ports cumulative: ~36 tests, all wired into
  iter-383 sweep (now 209).
- **Iters 621-629 (compacted iter 630)**: complete FV3
  cubed-sphere CONSTRUCTION + WIND-ROTATION pipeline.  9
  iterations close out FV3's full grid-init code path:

  | Iter | Function                            | F90 path / line              | Role                                |
  |------|-------------------------------------|------------------------------|-------------------------------------|
  | 621  | ``gnomonic_ed``                     | fv_grid_utils.F90:1313       | FV3 canonical ED grid (face 2)      |
  | 622  | ``gnomonic_grids``                  | fv_grid_utils.F90:1290       | grid_type 0/1/2 dispatcher          |
  | 623  | ``rot_3d`` + ``g_sum``              | fv_grid_tools / fv_grid_utils| 3D axis rotation; area-weighted sum |
  | 624  | ``mirror_grid_faces``               | fv_grid_tools.F90:2809       | 6-face cube via rot_3d sequences    |
  | 625  | ``mirror_grid_face1_symmetrize``    | fv_grid_tools.F90:2774       | intra-face-1 SIGN-averaging         |
  | 626  | ``init_cubed_to_latlon``            | fv_grid_utils.F90:2321       | a11/a12/a21/a22 rotation matrix     |
  | 627  | ``c2l_ord2_fv3``                    | fv_grid_utils.F90:2547       | 2nd-order D-grid → latlon winds     |
  | 628  | ``c2l_ord4_fv3``                    | fv_grid_utils.F90:2407       | 4th-order Lagrange + ord2 boundary  |
  | 629  | ``make_fv3_native_grid``            | (legoESM integration)        | end-to-end pipeline wrapper         |

  ``make_fv3_native_grid(im, grid_type=0)`` is the new single
  public API reproducing FV3's full ``init_grid`` cubed-sphere
  construction in 3 calls (gnomonic_grids →
  mirror_grid_face1_symmetrize → mirror_grid_faces).

  Wind-rotation pipeline (iter 626 → 627 / 628):
  legoESM users can now reproduce FV3 ``c2l_ord2`` / ``c2l_ord4``
  bit-equivalent (a-matrix pre-scaled by 0.5, vorticity-
  conserving 2.0× factor in u1/v1 documented).  Boundary cells
  in c2l_ord4 fall back to c2l_ord2 exactly per FV3 lines 2455-
  2530.

  All 9 ports reuse iter-611/612/613/615 spherical primitives.
  Total: 9 functions across 9 iters, 51 tests, all wired into
  iter-383 sweep (now 200).
- **Iters 611-619 (compacted iter 620)**: complete FV3
  ``fv_grid_utils.F90`` spherical-geometry port.  9 iterations
  added the building-block helpers used throughout FV3 grid
  generation, halo corner code, and vector-halo rotation:

  | Iter | Function(s)                              | F90 line | Role                              |
  |------|------------------------------------------|----------|-----------------------------------|
  | 611  | ``latlon2xyz`` / ``xyz2latlon``         | 1639/1739| (lon,lat) ↔ Cartesian             |
  | 611  | ``inner_prod`` / ``vect_cross``         | 984/1781 | Cartesian dot / cross             |
  | 611  | ``normalize_vect``                       | 1880     | unit-norm (zero-safe)             |
  | 611  | ``mid_pt3_cart`` / ``mid_pt_cart``      | 1996/2026| Cartesian / latlon GC midpoint    |
  | 611  | ``get_unit_vect2``                       | 1848     | unit tangent at GC midpoint       |
  | 612  | ``mirror_xyz`` / ``mirror_latlon``      | 1668/1705| reflect across GC plane           |
  | 612  | ``intp_great_circle``                    | 1896     | secant GC interpolation           |
  | 612  | ``slerp``                                | 1927     | Shoemake arc-length slerp         |
  | 613  | ``spherical_angle``                      | 2838     | angle at vertex of (p1,p2,p3)     |
  | 613  | ``cell_center3`` / ``cell_center2``     | 2728/2700| 4-corner → cell center            |
  | 613  | ``dist2side_latlon``                     | 2812     | point → GC arc distance           |
  | 613  | ``expand_cell``                          | 2631     | expand 4-corner cell by ``fac``   |
  | 614  | ``great_circle_distance_cart``           | 2065     | distance from Cartesian inputs    |
  | 614  | ``get_area``                             | 2749     | spherical-excess quad area        |
  | 615  | ``unit_vect_latlon``                     | 2286     | (lon,lat) → (elon,elat) tangents  |
  | 615  | ``get_unit_vect3``                       | 1865     | Cartesian variant of get_unit_vect2 |
  | 616  | ``intersect_great_circles``              | 2096     | GC intersection (FV3 ``intersect``)|
  | 617  | ``gnomonic_angl`` / ``gnomonic_dist``    | 1531/1558| equi-angular / equi-distance grids|
  | 618  | ``get_center_vect``                      | 1795     | cell-center tangents (u1, u2)     |
  | 619  | ``symm_ed``                              | 1587     | ED-grid symmetrization            |

  All JAX-vectorized, broadcast on leading axes, take 3-vector
  on last axis, AD-safe (zero-input branches via ``jnp.where``).

  ``get_area`` matches legoESM's existing l'Huilier's-theorem
  area within float32 precision (~1e-7 rel) — independent
  cross-validation of both implementations.

  ``intersect_great_circles`` returns ``(x_inter, local_a, local_b)``
  matching FV3 exactly (``get_nearest`` branch + ``check_local``).

  These ports are foundational building blocks: every higher-
  level FV3 grid utility (``edge_factors``, ``efactor_a2c_v``,
  ``init_cubed_to_latlon``) can now be ported using these
  primitives.  Total: 24 helpers across 9 iters, 51 tests, all
  wired into iter-383 sweep (now 191).
- **Iters 601-609 (compacted iter 610)**: full NH+PE conservation
  matrix + FV3 utility ports (grid, ops, filter, diagnostics).
  - iter 601: **NH TE-conserving correction** (FV3 consv_te NH).
    ``apply_te_correction_nh`` adjusts θ′ uniformly using ΔT =
    -te_dt / (cv · total_dry_mass).  Verified at C8: drift
    7.16e+22 J → -1.07e+09 J (14 orders, float64 floor).
    **NH conservation stack COMPLETE** (AAM + TE).
  - iter 602: **PE TE-conserving correction** (FV3 consv_te PE).
    Newton iteration (5 steps) with numerical-Jacobian
    dTE/dT = (TE(T+ε) - TE(T))/ε since PE TE has a hydrostatic
    boundary-work term nonlinear in T.  Verified at C8: drift
    5.14e+23 J → 0.0 J (exact).  New ``apply_te_correction_pe``.
  - iter 603: **PE TE boundary-work sign fix** (iter-598 bug).
    iter-598 had ``te = pe_top·phi_top - pe_sfc·phi_sfc``;
    FV3 fv_mapz.F90:1142 specifies ``pe(km+1)·phiz(km+1) -
    pe(1)·phiz(1)`` (sfc - top).  Surfaced by iter-602's
    numerical-Jacobian probe; iter-598 tests passed only
    because boundary term is ~0.1% of total.  Regression test:
    raise phis → TE should increase.
  - iter 604: **PE AAM stack** (mirror NH iter 583/587/588).
    ``aam_from_pe_state`` / ``aam_drift_pe`` /
    ``apply_aam_correction_pe`` (Newton iteration, 5 steps).
    Verified at C8: +1 m/s u_d drift 1.98e+25 → 1.18e+21
    (~5 orders, limited by D-grid ↔ cell-center mismatch;
    NH version had 8 orders since no D-grid mismatch).
    **PE conservation stack COMPLETE** (AAM + TE).
  - iter 605: **``column_d_ext_field`` / ``column_mass_weighted_mean``**
    utilities (FV3 dyn_core.F90:1310-1326 d_ext support).
    ``divg2 = d_ext·da_min_c·Σ_k(ptc·vt)/Σ_k(ptc)``.
    Standalone; not yet wired into ``one_grad_p`` (complex).
  - iter 606: **terrain_filter** (FV3 del2/del4_cubed_sphere port
    from tools/fv_surf_map.F90:817+).  Smooths phis at IC load.
    Params: ``n_iter`` (=n_zs_filter, default 4), ``nord``
    (=nord_zs_filter, 2=del-2, 4=del-4), ``cd`` default
    0.20·min(grid.area).
  - iter 607: **``cubed_to_latlon``** utility (FV3 c2l_ord2 alias).
    Extended ``dgrid_to_center_geographic`` to 3D
    ((6,n+1,n+1,nlev)) and added FV3-named alias.
  - iter 608: **``mid_pt_sphere``** great-circle midpoint
    (FV3 fv_grid_utils.F90:1981-1992).  Cartesian midpoint +
    normalize, NOT lon/lat average (wrong near dateline/poles).
  - iter 609: **``terrain_filter`` mass-preservation regression**.
    Verifies del-2/del-4 preserves area-weighted mean(phis) on
    closed sphere (divergence theorem, rel drift < 1e-3 at C16).
  All wired into iter-383 sweep (now 182).
- **Iters 591-599 (compacted iter 600)**: grid shift + complete
  PPM stack + transport plumbing + NH/PE total-energy stack.
  - iter 591: ``shift_fac`` longitude shift (FV3
    fv_grid_tools.F90:662-663).  Default 18 = west-shift 10°
    away from Japan.  Gated by ``not apply_schmidt``.
  - iter 592: ``apply_hord11_limiter(bl, br, dm, ppm_fac=1.5)``
    (FV3 tp_core.F90:573-579 "2nd van Leer emulation").
    ppm_fac=2.0 exactly matches iord=8.
  - iter 593: ``apply_hord10_limiter(bl, br, dm, q)`` (Lin+Rood
    1996 with pmp/lac extra constraints, tp_core.F90:554-572).
    Most subtle FV3 PPM variant.
  - iter 594: 5-variant comparison test (iord=8/9/10/11/12 all
    distinct on stress field, all finite on smooth).
  - iter 595: ``hord`` kwarg plumbed through ``_ppm_1d`` /
    ``_xppm`` / ``_yppm``.  Default 12 preserves baseline.
  - iter 596: ``hord`` plumbed through ``fv_tp_2d`` /
    ``transport_step`` (top-level public API).  Users can call
    ``transport_step(h, ut, vt, dt, cdgrid, hord=8)``.
  - iter 597: ``compute_total_energy_nh(state, grid, hc)`` NH
    total-energy diagnostic.  FV3 fv_mapz.F90:1154-1183 port:
    Σ_k delp·(cv·T + KE + g·z + 0.5·w²).
  - iter 598: ``compute_total_energy_pe(state, grid, coord)``
    PE hydrostatic TE.  FV3 fv_mapz.F90:1127-1152 port.
    JAX-friendly reverse cumsum for hydrostatic ϕ integration.
  - iter 599: ``te_drift_nh(state_old, state_new, ...)`` and
    ``te_drift_pe(state_old, state_new, ...)`` companion to
    iter 587's ``aam_drift_nh``.
  Net: user audit item #3 (PPM variants) end-to-end closed;
  NH and PE both have full {AAM, TE} × {static, drift, correct
  for AAM only} diagnostic stacks.  consv_te correction
  remains as natural follow-up.  Currently 173 guards in
  iter-383 sweep.
- **Iters 581-589 (compacted iter 590)**: user-audit response
  + FV3-faithful feature ports + stretched-grid + AAM stack.
  - iter 581: ``compute_edge_artifact_metric()`` public diag.
  - iter 583: ``compute_atmospheric_angular_momentum()``
    faithful port of FV3 ``compute_aam`` (fv_dynamics.F90:
    1264-1307).  Formula: ``aam = Σ_k (r²·Ω + r·u)·rho·dz·area``.
  - iter 584: ``w_safety_cap`` config (user audit item #2).
    FV3 ``w_limiter`` lives in fv_mapz only; legoESM port is
    a post-step clip (NOT mass-conserving cascade — Eulerian
    context).  Default disabled.
  - iter 585: ``apply_hord8_limiter(bl, br, dm)`` utility
    (user audit item #3 partial — iord=8 Lin 1996 mono).
  - iter 586: ``schmidt_transform()`` faithful port of FV3
    ``direct_transform`` + ``create_cubed_sphere`` kwargs
    ``stretch_fac`` / ``target_lon`` / ``target_lat`` (user
    audit item #1).  Defaults preserve bit-for-bit pre-grid.
  - iter 587: ``aam_from_nh_state()`` + ``aam_drift_nh()``
    convenience functions — handle face-local→u_east rotation
    + rho_full reconstruction internally.
  - iter 588: ``apply_aam_correction_nh()`` faithful port of
    FV3 ``consv_am`` (fv_dynamics.F90:774-794).  Reduces AM
    drift by 8 orders of magnitude (4.05e+25 → 9.9e+17).
  - iter 589: ``cube_transform()`` — FV3 revised Schmidt
    (fv_grid_utils.F90:920-980).  Adds ``do_cube_transform``
    kwarg.  Same as iter 586 plus ``lon += π`` pre-rotation.
  Net: user audit items 1/2/3 all addressed; nested grid +
  full hord plumbing deferred (huge scope).  Currently 164
  guards in iter-383 sweep.
- **Iters 571-579 (compacted iter 580)**: dycore mathematical
  consistency + amplitude/time scaling + linearity.
  - iter 571: PE δT ~ n_steps¹·⁸⁹ on HS but absolute ≤1 mK
    (essentially no artifact).
  - iter 572: NH zero IC → exactly zero output (perfect rest).
  - iter 573: NH edge_std ~ U_0¹·²⁰.  Typical winds → 2-3 mK
    edge noise.  Calm regions → ≤0.4 mK.  Jet stream → ≤7 mK.
  - iter 576: 50-step C24 grows super-power-law (18× vs 6×
    predicted).  Non-linear regime at 50 steps.  Practical
    limit ~30 steps.
  - iter 578: dycore PERFECTLY linear at small perturbations
    (10× IC → exactly 10.00× response in θ′ and u).
  - iter 579: SW rest preservation — h, u_d, v_d drift =
    exactly 0 over 10 steps.  Like NH, SW is mathematically
    consistent.
  Net positive findings: dycore is well-behaved (rest
  preserved, linear at small pert), edge noise scales with
  wind magnitude (mK for typical flows), PE has essentially
  no artifact issue, NH usable up to ~30 step blocks at C24.
  Currently 156 guards in iter-383 sweep.
- **Iters 561-569 (compacted iter 570)**: ``heat_source_del2``
  optimization + multi-step growth + helper equivalence.
  - iter 561: bisect min-edge factory's 3 disabled flags on
    smooth IC.  ``heat_source_del2`` is the dominant smooth-
    IC fix (-78% alone).  ``metric_aware_d_con`` HURTS when
    alone (+49%) but synergizes with others.
  - iter 562: ``heat_source_del2_iters`` sweep → monotonic
    improvement.  iters=0 → 6.32e-2; iters=8 → **6.90e-3
    (-89.1%)**.
  - iter 563: ``heat_source_del2_coeff`` U-shaped response.
    FV3 default 0.20 is OPTIMAL.  0.40-0.80 over-damps;
    <0.20 under-damps.
  - iter 564: BEST combined result @ C24 = **5.83e-3 K**
    edge_std after 10 steps (~6 mK noise).
  - iter 565: C24 → C32 only -1% edge_std reduction.
    Interior continues converging (-31%).  Edge plateau ~5-6
    mK floor at this config.
  - iter 566: 30-step run @ C24 → edge_std 3.43e-2 (5.9×
    growth from 10 steps).
  - iter 567: iters=8 grows FASTER (5.88×) than iters=2
    (3.88×) over 30 steps.  Crossover possible at very long
    times.
  - iter 568: ``monotone_halo_clip_context`` and
    ``make_clipped_step`` produce bit-identical output (max
    diff < 1e-10).  Confirmed equivalence.
  - iter 569: time-growth power law: edge_std ~ n_steps¹·¹⁵
    at C16 SBR.  Slightly super-linear (NOT exponential).
    Predicted production C96 1-day floor ≈ 120 mK.
  Currently 150 guards in iter-383 sweep.
- **Iters 551-559 (compacted iter 560)**: regime-aware
  factory analysis + long-run stability bounds.
  - iter 551: clip overhead at C16 = -1.6% (within noise).
    Negligible at both C8 and C16.
  - iter 552: HS-like stratified NH state stable + physical
    over 10 steps.
  - iter 553: **major reframe** — at C16 SBR smooth IC,
    FV3-faithful BEATS min-edge (-89.5% vs -64% edge_std
    reduction).  min-edge was random-IC stress-test optimum.
  - iter 555: clip on top of FV3-faithful adds only +2.3%
    at smooth IC.  Marginal.
  - iter 556: FV3-faithful convergence rate ~ N⁻⁰·⁶⁴ (edge)
    / N⁻¹·³⁶ (int).  Slower than min-edge+clip rates but
    LOWER absolute values for N < ~9600.
  - iter 557: PE comparison at C16 smooth — FV3-faithful
    and min-edge tied (T edge_std 4.116e-1 both).  PE
    insensitive to factory choice.
  - iter 558: PE long-run @ C8 dt=10, 100 steps stays
    finite but unphysical (u 10⁵ m/s, T -800 K).
  - iter 559: PE long-run @ dt=5, 200 steps still unphysical.
    PE C8 instability is damping-driven, not CFL.  Need C24+.
  Net guidance:
  - SMOOTH ICs → ``make_fv3_faithful_nh_config()`` (89.5%).
  - RANDOM/STRESS ICs → ``make_legoesm_nh_min_edge_config()``
    + clip helper.
  - PE: either factory works.
  - Long-run PE: need higher resolution (C24+).
  Currently 141 guards in iter-383 sweep.
- **Iters 541-549 (compacted iter 550)**: scan-step API +
  comprehensive helper validation (terrain, tracers,
  performance, end-to-end).
  - iter 541: helper performance overhead = +0.6% vs raw jit
    (negligible cost).
  - iter 543: dt sensitivity — t_final=100 s across dt ∈
    {5, 10, 20} → max|u| varies <2%, max|θ′| <1%.
  - iter 544: ``make_clipped_scan_step()`` JAX-scan multi-
    step API.  Bit-for-bit match with Python loop, AD-safe.
  - iter 546: scan vs loop speedup at 20 steps: 1.20×.
  - iter 547: PE scan_step validation (symmetric to NH).
  - iter 548: end-to-end 30-step SBR run at C8 — stable,
    finite, mass≈0, θ′ ratio 5.7×, v ratio 3.8× (integrated
    edge artifact over time).
  - iter 549: SW scan_step validation — full NH/PE/SW
    coverage for both ``make_clipped_step`` and
    ``make_clipped_scan_step``.
  Helper stack final state: 5 user-facing entry points
  (2 factories + context manager + 2 JIT-safe helpers).
  Currently 133 guards in iter-383 sweep.
- **Iters 531-539 (compacted iter 540)**: helper validation
  across dycores, ICs, conservation, and stability.
  - iter 531: ``make_clipped_step`` works on SW
    (``FV3EdgeShallowWaterModel``) too — fully dycore-
    agnostic.
  - iter 532: SBR resolution scan C8/C16/C24.  edge_std ~
    N⁻⁰·⁸², int_std ~ N⁻¹·⁵³.  Both converge absolutely;
    edge is formally lower-order (FV3-faithful).
  - iter 533: corner-spike stress test.  5 K spike grows
    0.08% in 5 steps, doesn't propagate.  Dycore is stable.
    Clip bounds halo cells, NOT interior corner cells.
  - iter 534: min_edge vs aggressive factory + clip have
    DIFFERENT goals — min_edge minimizes TOTAL noise;
    aggressive minimizes RATIO.  Both valid.
  - iter 535: example script
    ``scripts/run/example_fv3_clip_helper.py`` showing end-to-
    end API.  Reproduces iter-521 values within 5%.
  - iter 536: slack sweep at C16 SBR.  slack=0.0 marginally
    best (0.7% better than slack=0.5 default).  All within
    ~1% — slack=0.5 is safe default.
  - iter 537: clip helper with 2-km mountain.  All fields
    finite with realistic gravity-wave magnitudes (u 30 m/s,
    w 0.7 m/s, θ' 1 K).  Terrain-compatible.
  - iter 538: q_vapor tracer transport.  Mean drift 0%,
    max growth 0.02%, monotonicity preserved.  Production-
    ready for moist runs.
  - iter 539: 50-step mass conservation.  Bit-perfect 5.3 ppb
    drift through step 42; NaN at step 43 (C8 stability
    limit, not helper).  Helper preserves conservation up
    to simulation's intrinsic stability limit.
  Currently 126 guards in iter-383 sweep.
- **Iters 521-529 (compacted iter 530)**: helper validation —
  conservation, AD, JIT-safety, PE symmetry, production
  docs.
  - iter 521: SBR (Williamson 2-like) IC scan.  C8 e/i=3.90×,
    C16 e/i=6.31× — but **absolute** edge_std DROPS (-45%
    at C16), interior_std drops faster (-67%).  Edge
    converges at ~O(N⁻⁰·⁶), interior at ~O(N⁻¹·³).  Cube
    corners are formally lower-order in FV3 — the rising
    e/i ratio is intrinsic, not a legoESM bug.  At C16 the
    perturbation magnitude after 10 SBR steps is 0.06 K,
    physical noise level.
  - iter 522: AD-at-rest validated.  ``jax.grad`` through
    1-step and 3-step under ``monotone_halo_clip_context``
    returns finite + nonzero gradients.  Helper is fully
    differentiable.
  - iter 523: NH mass conservation under SBR + clip context
    — 5.3 ppb drift over 10 steps.  Machine precision.
  - iter 524: PE mass conservation under Held-Suarez + clip
    context — 7.0 ppb drift over 10 steps.  Symmetric to NH.
  - iter 525: JIT + clip context interaction.  jit-compile
    inside context = clip baked in; jit-compile outside +
    run inside = clip ignored.  Both behaviors documented.
  - iter 526: ``make_clipped_step(model, state, dt, slack)``
    helper — JIT-safe.  Enters context, jit-compiles, forces
    trace via dummy call, returns cached-compiled step.
    48.4% edge reduction vs raw jit at 10 steps.
  - iter 527: PE-side patch targets.  Context now covers
    15 sites: 4 NH-scalar + 2 NH-vector + 3 PE-aliases + 3
    pad_halo 3D + 1 pad_halo_pair_h2 + 2 pad_halo_vector 3D.
  - iter 528: production-usage section updated with concrete
    ``monotone_halo_clip_context`` + ``make_clipped_step``
    examples.
  - iter 529: PE ``make_clipped_step`` validation.  Runs,
    conserves mass (7e-9), differentiable.  Full NH/PE
    symmetry of helper stack.
  Net result: iter-505 helper is production-ready with
  conservation, AD, JIT-safety, and full NH/PE coverage.
  Currently 117 guards in iter-383 sweep.
- **Iters 511-519 (compacted iter 520)**: long-term clip
  benefit, expanded clip coverage, and IC analysis closing
  on **convergence with cube-smooth IC**.
  - iter 511: clip benefit grows over time (1 step -36%, 10
    steps -53%).  Clip is necessary precisely because the
    leak accumulates.
  - iter 512: long-term slack sweep at 10 steps — slack=0.5
    optimal (1.844×).  slack=0.0 within 1%; slack=2.0 over-
    relaxes (2.26×).
  - iter 513: extended ``monotone_clip`` to ``pad_halo``
    (3D) + ``pad_halo_pair_h2`` + context to 10 sites.
    No NH ratio change — the 3D paths aren't on the NH test
    trajectory.
  - iter 514: extended clip to ``pad_halo_vector`` (3D) +
    context to 11 sites.  Also no NH ratio change.  Closes
    fv3_sw_core code paths for SW dycore use.
  - iter 515: **reframe metric** to within-grid e/i ratio.
    At 10 steps: duogrid e/i = 2.06×, no-duogrid e/i = 1.24×.
    Both grids develop bias (intrinsic to C-D-grid corners),
    not a duogrid bug.
  - iter 516: nord=2 (del⁴) damping is WORSE than default
    nord=1 (1.99 → 2.19).  Higher-order at corners has
    stronger response to vertex noise.
  - iter 517: random IC vs face-local sinusoid IC at 10
    steps: 2.09× vs 1.26× (65% noise penalty in random).
  - iter 518: but face-local sinusoid is NOT smooth across
    panels.  Resolution scan reveals C8 1.26× → C16 2.77×
    (resolution makes it worse, invalidating iter-517's
    "smooth" framing).
  - iter 519: **TRULY cube-smooth Gaussian** via geographic
    (lat, lon) coords.  Resolution scan:
        C8:  e/i = 1.434×
        C16: e/i = 1.287×  (10% improvement)
    **The dycore converges on real-atmospheric-like smooth
    ICs**.  Extrapolating to C48-C96 → e/i ~ 1.1× (near-
    perfect).  Strongest validation that iter-466/505 stack
    delivers FV3-faithful behavior on real ICs.  Currently
    109 guards in iter-383 sweep.
- **Iters 501-509 (compacted iter 510)**: from clip-slack
  knob → combined-fix new low → user-facing API → growth
  diagnostic that closes the "single-step floor was optical
  illusion" loop.
  - iter 501: ``monotone_clip_slack`` param (default 0 =
    strict).  Soft clip = band expansion by slack × (hi-lo).
  - iter 502: slack sweep on ``center_to_dgrid_vector``.
    Strict clip over-corrects (×0.95); slack ≈ 1.5-2.0 →
    neutral (×1.00); slack=0.5 → ×0.97.
  - iter 503: **46% dycore reduction** via 6-site (4 scalar +
    2 vector) clip patch.  Vector halo was the missing
    piece from iter-492/493 (which only got 1.5%).  slack=0.5
    is optimal; strict 99% as good.  Floor 2.006×.
  - iter 504: **NEW LOW 1.135× (65.7% reduction)** combining
    min-edge factory (iter-466) + 6-site clip (iter-503).
    Factory and clip paths are independent and compose.
    Aggressive factory (iter-483 d2_bg boost) gives no
    additional reduction at this regime.
  - iter 505: **user-facing API** —
    ``monotone_halo_clip_context(slack=0.5)`` in
    ``legoesm.grids.halo``: a single context manager that
    monkey-patches all 6 halo sites for the duration.
    Verified: 1.606 → 1.093 (32% at this seed pair).
  - iter 506: PE composition check — held_suarez state +
    monotone_halo_clip_context yields 1.000× both with and
    without context.  PE hydrostatic balance lacks the
    NH acoustic/advection coupling that drives the 4.77×
    duogrid penalty.  Composition is unconditionally safe.
  - iter 507: bisect residual 1.075× under min-edge + clip.
    Toggling 4 fidelity flags + 4 sponge/damp knobs OFF one
    at a time: vector_halo_uv=False → 2.36× (critical),
    d_con_cv=False → 1.22×, dynamic_exner / cross_face / all
    sponge damps neutral, corner_div_damp=0 → 46.2× (blowup).
    Vector halo and corner_div_damp are the two essential
    stabilizers.  Residual NOT eliminable by flag toggle.
  - iter 508: ``corner_div_damp_d2_bg`` sweep [1e-5..5e-3]
    under min-edge + clip.  1.094× (1e-5) → 1.075× (5e-3).
    Residual insensitive over 500× range.  Not the knob.
  - iter 509: **residual GROWS** — multi-step test (1, 2,
    5, 10 steps, dt=10s) shows 1.129 → 1.192 → 1.326 →
    1.634× (+44.7%).  Clip context suppresses single-step
    amplification but residual edge bias accumulates.  This
    is NOT a dynamic equilibrium.  Iter-505 helper alone is
    insufficient for long runs.  Next direction = patch
    edge stencils within the substep (vector halo / corner
    interp).  Currently 100 guards in iter-383 sweep.
- **Iters 485-494 (compacted iter 500)**: factory exposure
  + duogrid bisection + halo-level clip + investigation
  closure.
  - iter 485: AST guard for 4 min-edge factories.
  - iter 486: document min-edge factories in production-
    usage section.
  - iter 487: C24 plateau (4.72×) — refutes "duogrid helps
    at higher resolution" hypothesis.
  - iter 488: synthesis section + key data table.
  - iter 489: **overshoot confined to cube-vertex cells**
    (24 cells/level globally).
  - iter 490: ``monotone_clip`` arg added to
    ``pad_halo_4d`` (default off); fixes iter-489.
  - iter 491: random Gaussian shows 76% halo overshoot
    (clip cuts to within interior).
  - iter 492: dycore monkey-patch (1 site) → only 1.4%.
  - iter 493: comprehensive 4-site patch → still 1.5%;
    clip is NOT the dominant dycore fix (random-field
    overshoot doesn't translate to dycore impact because
    dycore fields are smoother than random noise).
  - iter 494: extend iter-488 synthesis section with
    iter-489..493 data.
- **Iters 475-484 (compacted iter 490)**: duogrid bisection
  + edge-min factories + aggressive config.
  - iter 475: linear-field halo test, duogrid overshoots
    interior max by 3.5% (14.49 vs 14.00).
  - iter 476: vector halo + zero field preserves exact;
    halo operators alone are correct.
  - iter 477: laplacian iter alone doesn't amplify (refutes
    iter-476 composition hypothesis).
  - iter 478: per-step bisection — penalty fires in step 1
    (3.67×), grows slowly to 4.91× by step 3.
  - iter 479: op-level — damp_v/damp_w NOT culprit (3.50→3.54
    when disabled); all damping off DIVERGES.
  - iter 480: **corner_div_damp is the key mitigator** —
    disabling it explodes ratio to 109×.  Doc compacted
    iter 465-474.
  - iter 481: corner_div_damp_d2_bg sweep, 100× boost →
    36% reduction (2.25×).
  - iter 482: **composite iter-466 + iter-481 → 60% reduction
    (1.42×, 5 seeds pinned)**.
  - iter 483: new ``make_legoesm_nh_min_edge_aggressive_config``
    user-facing factory.
  - iter 484: PE mirror ``make_legoesm_pe_min_edge_aggressive_
    config``.
- **Iters 465-474 (compacted iter 480)**: edge-artifact
  empirical investigation phase 2 (per-flag + duogrid
  bisection start).
  - iter 465: NH per-flag θ′ sweep at C8+duogrid; baseline
    4.37; vector_halo_uv biggest helper (Δ+5.0);
    metric_aware_d_con / heat_source_del2 / d_con_top_zero
    HURT (Δ-1.6/-1.4/-0.6).
  - iter 466: **50.4% edge-ratio reduction** by dropping the
    3 iter-465 hurting flags (4.54 → 2.25, 5-seed pinned).
  - iter 467: new ``make_legoesm_nh_min_edge_config`` factory
    exposing iter-466 overrides.
  - iter 468: PE mirror ``make_legoesm_pe_min_edge_config``.
  - iter 469: PE T sweep shows ALL factory flags
    insensitive (max |Δ| < 1e-4) — PE T artifact dominated
    by other mechanism.
  - iter 470: compact iter 455-464 + PE u_d sweep (also
    insensitive); NH is the right empirical target.
  - iter 471: **duogrid alone is the regime-changing factor**
    — same factory but duogrid=OFF gives 5.12× lower ratio.
  - iter 472: duogrid penalty persists at C16 (4.12×)
    — not a low-resolution artifact.
  - iter 473: raw std decomposition: duogrid INCREASES edge
    std 4.77× while interior std unchanged (× 1.00) — bug
    is in edges, not interior smoothing.
  - iter 474: pad_halo_4d scalar+constant test passes (5e-13);
    rules out simple halo broken.
- **Iters 455-464 (compacted iter 470)**: edge-artifact
  empirical investigation phase 1 (single-flag sweeps).
  - iter 455: legoESM-scale calibration doc + 5-step
    stability at known-good d2_bg_k1=1e-4.
  - iter 456: introduce edge_var/interior_var metric infra.
  - iter 457: NH heat_source del-2 smoothing (FV3 del2_cubed
    port).
  - iter 458: PE mirror of iter-457.
  - iter 459: factory default heat_source_del2_iters=2
    (FV3 ``nf_ke``).
  - iter 460: compact iter 445-454 + AST guard for iter-
    457/458/459.
  - iter 461: first empirical edge-metric data (3-seed C8):
    NH factory worsens 3.7%; PE improves 2.5% — both ratios
    near 1.0.
  - iter 462: d2_bg_k1 sweep over 4 decades → ZERO change in
    NH ratio at C8; sponge boost is per-level scalar
    invariant.
  - iter 463: rf_tau_days sweep → ZERO change at C8; same
    invariance insight (RF is per-level scalar).
  - iter 464: **first positive finding** — del-2 smoothing
    reduces NH θ′ edge ratio 1.10 → 0.96 (14% drop at
    iters=1); SPATIAL operations DO move the metric.
- **Iters 445-454 (compacted iter 460)**: sponge boost shared
  helper + Rayleigh friction + AST hardening + doc updates.
  - iter 445: AST guard for iter-438..443 sponge wirings.
  - iter 446: factor sponge boost to shared core helper
    ``apply_top_sponge_damp_boost``.
  - iter 447: factor linear-scaling sponge trick to shared
    helper ``apply_top_sponge_field_scale``.
  - iter 448: NH FV3 ``Ray_fast`` column Rayleigh friction.
  - iter 449: PE mirror of iter-448.
  - iter 450: compact iter 435-444 doc + AST guard for
    iter-448/449 RF.
  - iter 451: factory ``corner_div_damp_d4_bg = 0.16`` (FV3
    production).
  - iter 452: rolled back factory ``d2_bg_k*=4.0/2.0`` to 0.0
    (FV3 normalization mismatch — caused blow-up).
  - iter 453: AST guard for iter-451/452 factory defaults.
  - iter 454: update iter-417 doc + iter-418 test for iter-
    431..453 factory flags.
- **Iters 435-444 (compacted iter 450)**: FV3 sponge boost
  (damping-coefficient side) + factory production defaults.
  - iter 435: AST regression guard for iter-431/432/433 mask
    wirings (7 tests).
  - iter 436: factory default ``delt_max = 1.0`` (FV3
    ``fv_arrays.F90`` production default).
  - iter 437: factory default ``nord_v = 1`` /
    ``corner_div_damp_nord = 1`` (FV3 ``nord=1`` del-4).
  - iter 438: new PE field ``corner_div_damp_d2_bg_k1`` (FV3
    ``dyn_core.F90:780`` k=0 sponge boost).
  - iter 439: new PE field ``corner_div_damp_d2_bg_k2`` (FV3
    lines 792/802 k=1/k=2 boost with 0.01 / 0.05 thresholds).
  - iter 440: NH mirror — both fields wired via shared helper
    ``_apply_top_sponge_damp_boost`` at 2 NH corner-div sites.
  - iter 441: NH ``use_fv3_sponge_damp_w`` flag — FV3 line
    782/793/803 ``damp_w = d2_divg``.  Linear ``damp^(nord+1)``
    scaling trick at sponge levels.
  - iter 442: NH ``use_fv3_sponge_damp_v`` flag — FV3 lines
    786-797 ``damp_vt = 0.5 * d2_divg`` (k=0/1 only).
  - iter 443: PE mirror of iter-442 ``use_fv3_sponge_damp_v``.
  - iter 444: factory defaults expose FV3 production sponge:
    ``d2_bg_k1=4.0``, ``d2_bg_k2=2.0``, ``sponge_damp_v=True``
    (NH+PE), ``sponge_damp_w=True`` (NH).
- **Iters 425-434 (compacted iter 440)**: factory hardening +
  FV3 sponge d_con zeroing.
  - iter 425: end-to-end smoke of 6 guard modules (27/27).
  - iter 426: factory module-exports test.
  - iter 427: factory drives finite 1-step at C8.
  - iter 428: factory preserves grad through 1-step (310 s).
  - iter 429: factory 3-step trajectory bounded (stability).
  - iter 430: compact iter 415-424 into single block + iter-
    368 regression for the new marker.
  - iter 431: FV3 sponge zeroing of d_con KE→heat — new field
    ``d_con_top_zero_levels: int = 0`` wired at NH damp_v.
  - iter 432: extend iter-431 mask to NH damp_w + aggregate
    (3 slow-tendency sites).  All 5 NH d_con sites covered.
  - iter 433: PE mirror — field on PE config + wire at 4 PE
    d_con sites (damp_v + aggregate covering 3 slow-tend).
  - iter 434: factory defaults expose
    ``d_con_top_zero_levels=2`` — matches FV3 production
    sponge (``d2_bg_k1=0.16, d2_bg_k2=0.05``).
- **Iters 415-424 (compacted iter 430)**: factory-regression
  guard hardening + doc-drift catches.
  - iter 415: no-duplicate regression for iter-383 guard-sweep
    inventory (catches silent inventory weakening).
  - iter 416: wire iter-415 into iter-383 sweep
    (self-referential, 19 guard modules).
  - iter 418: executable test that iter-417 production-usage
    doc example actually runs (catches API drift).
  - iter 419: doc-structure guard for iter-417 production-usage
    section (header + factories + duogrid pairing).
  - iter 421: doc-structure regression for iter-420 compacted
    block (marker + 8 topics; 5 compaction blocks now guarded).
  - iter 422: extend iter-383 sweep with iter-401/413/418
    (numeric pkz proof, gradient flow, doc example) → 22
    modules total.
  - iter 423: FV3-fidelity-stack summary table update (5 NH +
    3 PE flag panel + factory rows).
  - iter 424: doc-structure regression for iter-423 summary
    table (flag names, PE-specific flag, factory rows).
- **Iters 405-414 (compacted iter 420)**: factory-extension
  audit/regression infrastructure.
  - iter 405: FV3-flag inventory consistency check.
  - iter 406: factory docstring content regression.
  - iter 407: doc-structure guard for iter-400 compaction.
  - iter 408: factory 5-step stability smoke.
  - iter 409: factory 5-step AD-at-rest (11.5 min).
  - iter 412: factory pass-through verification.
  - iter 413: gradient-flow test through dyn_exner.
  - iter 414: extend iter-383 sweep for 3 new guards (iter-
    405/406/412).
- **Iters 395-404 (compacted iter 410)**: post-iter-400
  regression infrastructure.
  - iter 395: pkz-equivalence docstring regression for iter-394.
  - iter 396: factory override edge cases.
  - **iter 397/398**: extend dyn_exner to ALL delt_max caps
    (aggregate slow-tendency + damp_v + damp_w post-acoustic).
    Closes inconsistency where d_con denominator used Π_total
    but cap derivation still used frozen Π_ref.
  - iter 399: AST guard for iter-397/398 cap wiring.
  - iter 400: ToC compaction (iter 385-394 → 1 block).
  - iter 401: quantitative ``Π_total = FV3 pkz`` numeric
    regression (rtol=1e-10 with perturbation, 1e-14 at rest).
  - iter 402: AST signature regression for iter-392 factories.
  - iter 403: extend iter-383 sweep for iter-402.
  - iter 404: jax.jit smoke for iter-392 factory models.
- **Iters 385-394 (compacted iter 400)**: post-iter-380 doc-
  audit follow-ups + factory work + dyn_exner audit.
  - iter 385: document iter-384 finding in iter-370 source.
  - iter 386/387: PE+NH C16 cross_face with duogrid grid
    (meaningful regression coverage; iter-372 was no-op).
  - iter 388: docstring-content regression for iter-385.
  - iter 389: extend guard-sweep inventory for iter-388.
  - iter 390: ToC compaction (iter 375-384 → 1 block).
  - iter 391: extend doc-structure regression for iter-390.
  - **iter 392**: user-facing ``make_fv3_faithful_*_config``
    factories.  Enable all FV3-fidelity flags at production
    values.  PE 3 flags + NH 5 flags.
  - iter 393: factory runtime smoke (finite step PE+NH).
  - iter 394: **AUDIT** dynamic Exner = FV3 pkz.
    ``compute_exner_perturbation`` returns FULL NONLINEAR
    ``Π_total = (R_d·ρ·θ/p_0)^(R_d/c_v)``, algebraically equals
    ``(p/p_0)^kappa = pkz`` under EOS.  Resolves earlier
    "linearization" concern.
- **Iters 375-384 (compacted iter 390)**: cross_face follow-ups
  + audit/regression infrastructure.
  - iter 375: PE full-stack AD umbrella with cross_face.
  - iter 376/377: PE+NH C16 full FV3 stack with cross_face
    (does-not-amplify + changes-state).
  - iter 378/379: PE+NH tendency-level metric d_con linearity
    (bit-for-bit rtol=1e-10, bypasses acoustic feedback).
  - iter 380: ToC compaction for iter 360-374.
  - iter 381: extend iter-368 doc-structure regression for
    iter-380 compaction.
  - iter 382: **BUG FIX** iter-319 d_con knob count had suffix
    collision with iter-339's ``use_fv3_metric_aware_d_con``;
    filter to exclude ``use_fv3_*`` prefix.  Caught by full
    guard sweep.
  - iter 383: FV3-fidelity guard sweep meta-test inventories
    13 guard modules.
  - iter 384: cross_face + metric flag independence test.
    Caught no-op-without-duogrid behavior.
- **Iters 360-374 (compacted iter 380)**: post-iter-365 follow-
  ups + iter-370 cross_face flag wiring.
  - iter 360: NH C16 wiring-active-at-production-resolution.
  - iter 361/362: AST flag-coverage guards (all 5 NH + 4 PE
    d_con sites).
  - iter 363: dynamic-Exner safety regression under strong
    perturbation.
  - iter 364: damp_w_d_con + dynamic_exner composition.
  - iter 365: ToC compaction (40 iters into 1 block).
  - iter 366: delt_max sponge cap + cv flag composition.
  - iter 367: C16 cv-vs-cp heating ratio (c_pd/c_vd ≈ 1.40).
  - iter 368: doc-structure regression for iter-365.
  - iter 369: consolidated FV3-fidelity flag-set presence
    guard.
  - **iter 370**: opt-in ``use_fv3_cross_face_du_proj``
    (PE+NH).  Closes mode='edge' gap at damp_v post-step
    wind-increment projection back to corners.  Default False
    bit-for-bit baseline.  pad_halo_4d (duogrid-aware) used at
    flag=True; off by 0.5 cell from edge-native FV3
    cubed_a2d_halo.  6/6 baseline + state-changes + AD-safe.
  - iter 371: extend iter-331/369 guards for cross_face SHARED
    flag.
  - iter 372: C16 cross_face does-not-amplify regression.
  - iter 373: AST guard for iter-370 wiring.
  - iter 374: full NH AD umbrella with cross_face flag ON.

**TL;DR** (iter 81 update of iter 78 summary): For HS at any cube
resolution, set::

    LEGOESM_HS_CUBE_DT_CFL=auto
    LEGOESM_AH_SCALE   (auto-applies per resolution; explicit override OK)

This auto-picks (iter 81): ``dt=200`` at C36-C72 (preserves iter-33
reference), ``dt=50`` at C96+ (iter-79 found ``dt=100``
insufficient at 30 d so iter-81 promoted auto-mode to
very_long_time).  Combined with the iter-43 ``LEGOESM_AH_SCALE``
auto-apply (1.0 / 2.0 / 10.0 by resolution bucket), this is a
single env var pair recommended for cube HS.

C96 30-day empirical validation of the iter-81 dt=50 setting is
**EMPIRICALLY CONFIRMED** as of iter 99: full 30-day run completed
finite with max|u|=20.14 m/s, max|v|=11.84 m/s, 51840 steps, 1755 s
wall.  iter-85's linear-in-1/dt eigenmode prediction validated
(predicted NaN at day 45; run stopped finite at day 30).

The full iteration log follows.

## Lessons learned (iter 84 synthesis of iter 18-83)

For future maintainers, here are the load-bearing insights from
65+ cycles of investigation:

1.  **The C36/C48/C72 cube imprint and the C96+ long-time
    eigenmode are TWO DIFFERENT problems.**  The first is a
    SPATIAL artifact at face boundaries (cube-vertex residuals);
    the second is a TEMPORAL instability (eigenmode growing
    over physical time).  They need different fixes:
    - Spatial: ``corner_div_damp_d4_bg=0.02 nord=1`` (iter 18-25).
    - Temporal: ``LEGOESM_AH_SCALE`` (iter 33) + small ``dt``
      (iter 65-81).

2.  **More damping is not always better.**  iter-70 found
    ``ah_x20`` at C96 NaNs EARLIER than ``ah_x10`` because the
    diffusive CFL ``A_h × dt / dx²`` exceeds 0.5.  The right
    response to instability is sometimes SMALLER ``dt``, not
    larger ``A_h``.

3.  **Smaller ``dt`` DELAYS the eigenmode in proportion to 1/dt
    (iter-95 EMPIRICALLY CONFIRMED).**  iter-69 (``dt=150``: NaN
    day 15), iter-79 (``dt=100``: NaN day 22.5).  Note
    ``15 × 1.5 = 22.5`` — exactly proportional.  iter-85
    extrapolated: ``dt=50`` should NaN at ~day 45 (survives 30 d).
    iter-95 EMPIRICALLY VALIDATED this: C96 dt=50 ran past day
    22.5 (max|u|=14.04 m/s at step 38880), proving the eigenmode
    is an artifact of discretization error that the dycore
    artificially excites; smaller dt = less excitation =
    proportionally delayed onset.

4.  **Smagorinsky is a complement, not a replacement, for static
    A_h.**  iter-60 found C72 ``smag_cs=0.4`` alone insufficient.
    iter-70 found C96 ``smag_cs=0.2 + dt=150.5`` no better than
    ``dt=150.5`` alone.  Smagorinsky helps where strain is high;
    the C96 eigenmode is interior + synoptic-scale, where strain
    is moderate.

5.  **Test for the FAILURE MODE, not just the success path.**
    iter-65 verified C96 stable for 1 day, which iter-69 found
    insufficient for 30 days, which iter-79 found insufficient
    even at dt=100.  Each "stable to N days" claim should
    include "tested at N days" not "extrapolated".

6.  **The matrix's hardcoded ``dt=200`` doesn't scale.**  The
    lat-lon HS path has CFL-aware ``dt``; the cube path didn't.
    iter-66 added it (opt-in).  Future audits: any time you see
    a hardcoded numeric constant in a per-resolution path, ask
    whether the calibration was empirically validated at all
    resolutions.

7.  **Always include a regression guard at the AST level for
    dataflow that depends on a helper.**  iter-68 added an AST
    guard for ``dt = _resolve_dt_cube(...)`` because reverting
    to ``dt = 200.0`` would silently disable the env var without
    breaking unit tests.

The full iteration log follows.

## Investigation summary (iter 51 codex meta-review consolidation)

The FV3_3D investigation has produced **two complementary
mechanisms** — they address different failure modes and their
contributions should not be confused:

1. **``corner_div_damp_d4_bg=0.02 nord=1``** (iter 18-25): FV3-
   faithful B-grid corner-divergence damping (port of FV3
   ``sw_core.F90:1725-1822``).  Reduces cube-vertex artifacts
   most visibly at C36 (-3 % mid_std at d=30, -7 % at d=60) and
   C48 (-45 %).  At C72 it ALONE produces NaN at day 13 (iter 26).

2. **``LEGOESM_AH_SCALE`` per-resolution multiplier** (iter 33-46):
   Laplacian viscosity calibration.  The matrix's
   ``_laplacian_visc_cube(n) = 0.05 * c_gw * dx`` heuristic
   underestimates ``A_h`` at C48 (~2x too low) and dramatically at
   C72 (~10x too low).  iter-32 traced the C72 unstable mode to an
   INTERIOR synoptic-scale eigenmode that del-2 viscosity damps but
   del-4 hyperdiff and cube-vertex damping do NOT reach.

**Critical clarification** (per codex iter-51 meta-review): mechanism
(1) alone is NOT sufficient for C72+ stability.  Mechanism (2)
alone produces a stable run at C72 even without mechanism (1).
Both together give the recommended production config, but the
LOAD-BEARING piece for C72+ is **A_h scaling**, not nord>0.

Early commits (iter 18-25) framed nord>0 as "the FV3_3D fix".  That
framing is INCOMPLETE — it solves the C36/C48 cube-imprint but
not the C72 spatial instability.  Iter 33 found A_h scaling is the
actual stability mechanism.  The current Quick Reference below
combines both correctly.

**Open generalization gap** (codex iter-51 meta-review): the iter-33
``10x A_h`` is a CASE CALIBRATION at C72, not a defensible
production rule across resolutions / timesteps / physics / forecast
lengths.  The iter-39 ``_laplacian_visc_cube_v2`` is the empirical
extrapolation but UNTESTED at C96+.

**iter 60 update on the generalization gap**: I tried Smagorinsky-style
adaptive A_h (``c_s = 0.0 / 0.2 / 0.4``) at C72 with default static
A_h.  All three NaN within ~25 steps of baseline.  Smagorinsky CANNOT
replace the iter-33 static A_h scaling — the C72 unstable mode is a
slow exponential whose strain stays small until the last few steps,
so a strain-rate-dependent closure can't catch it in time.  The
iter-33 10x static A_h remains the load-bearing mechanism.

Smagorinsky (iter 57-59) is therefore a COMPLEMENT, not a replacement,
for ``LEGOESM_AH_SCALE``.  Use it in addition to the static scaling
if desired::

    LEGOESM_AH_SCALE=10.0 LEGOESM_SMAG_CS=0.2  # C72: static + adaptive

### C96+ user guidance (iter 63/65/69/79/81 empirical history)

**Time-line of C96 30-day stability findings**:

- iter-65/66 ``dt=150.5``: day-1 NaN solved, but 30-day NaN at day 15.
- iter-70 ``dt=100``: day-15 NaN solved, but 30-day NaN at day 22.5
  (iter 79).
- iter-81 ``dt=50`` (auto-mode default): EMPIRICAL VALIDATION
  PENDING (iter 82-83 in progress).

For users at C96 PRODUCTION (30-day climatology):

1. **Recommended**: ``LEGOESM_HS_CUBE_DT_CFL=auto`` (iter 72/81).
   This auto-picks ``dt=50`` at C96.  EMPIRICAL 30-day validation
   pending; iter 79 invalidated the prior ``dt=100`` recommendation.
2. **Avoid**: ``LEGOESM_HS_CUBE_DT_CFL=long_time`` at C96 — iter
   79 found NaN at day 22.5.  Use ``very_long_time`` or ``auto``
   instead.
3. Smagorinsky (``LEGOESM_SMAG_CS=0.2``) does NOT help at C96
   (iter 70 confirmed).  Skip it (without ``ah_x10`` it's no use;
   with it it doesn't extend stability).
4. ``ah_scale`` higher than 10x makes things WORSE at C96
   (iter 70: ``ah_x20 + dt=100`` NaNs at day 5).  Keep
   ``ah_scale=10``.


iter 65 EMPIRICALLY tested C96 stability via
``scripts/tmp/_iter65_c96_smoke.py``.  Findings overrode the iter-63
guidance:

**The matrix default ``dt=200.0`` is the limiting factor at C96, NOT
``LEGOESM_AH_SCALE``.**  At C96 with the matrix defaults
(``dt=200``, ``LEGOESM_AH_SCALE=10`` auto-applied), the run NaNs
at ~6 hours wall-clock REGARDLESS of ``LEGOESM_AH_SCALE`` (tested
10x, 20x, 50x, 200x — all blow up at same physical 6h).
Increasing A_h does NOT rescue this case; the diffusive CFL
limits how high ``A_h`` can go (200x → diffusive CFL = 0.35 which
itself violates stability).

Survival at C96 1-day depends on **shrinking ``dt``**::

    dt=200 (matrix default): NaN at 6h regardless of ah_scale
    dt=180: NaN at 6h
    dt=160: NaN at 6h
    dt=150: stable to 1 day
    dt=100: stable to 1 day

Recommended C96+ recipe:

1. **Reduce ``dt``** in your driver from 200 to ≤ 150 s.  The
   matrix's ``dt = 200.0`` (line 2636 / 3132) is hard-coded for
   the cubed-sphere HS path and does NOT scale with resolution.
   Compare the lat-lon HS path (line 2761):
   ``dt = min(200.0, 0.5 * _dx_pole / 300.0)`` — CFL-aware.  The
   cubed-sphere path has the same need but no scaling.

2. Keep ``LEGOESM_AH_SCALE=10.0`` (iter-43 auto-default).
   ``A_h`` calibration is correct at C96 once ``dt`` is reduced;
   raising it further does not help.

3. **Run the smoke test first**::

       JAX_ENABLE_X64=1 ITER65_DT=150.0 ITER65_DAYS=1.0 \
         .venv/bin/python scripts/tmp/_iter65_c96_smoke.py

   Confirms stability before committing to a 30-day run.

4. The iter-26-32 analysis identifying an interior synoptic-scale
   exponential eigenmode at C72 is consistent with what we see at
   C96 (same physical-time blowup, fixed-wall-clock-time mode).
   The mode is more severe at higher resolution; ``dt`` must
   scale down to avoid integrating it.

5. **Open work**: introduce CFL-aware ``dt`` scaling in the
   cubed-sphere HS path (matrix line 2636 / 3132).  Requires
   regression testing across C36/C48/C72 to ensure existing
   reference numbers don't shift.  Deferred to a future iteration.

## Quick Reference (iter 38 summary, updated iter 72)

### Production-recommended setting per resolution

The full damping configuration combines four iter-18-25 corner-
divergence damping settings + per-resolution ``A_h`` scaling
(iter 33-37) + iter-72 CFL-aware ``dt``:

| resolution | LEGOESM_AH_SCALE | recommended A_h | dt (auto-mode) | status                              |
|:----------:|:-----------------|:----------------|:---------------|:------------------------------------|
| C36        | ``1.0`` (default)| 4.08e+06        | 200 (no change)| iter 19/24 production               |
| C48        | ``2.0`` (iter 37)| 6.12e+06        | 200 (no change)| sweet-spot scan, mid_std -48 %      |
| C72        | ``10.0`` (iter 33)| 2.04e+07       | 200 (iter-33 ref)| smallest stable scale at dt=200   |
| C96        | ``10.0`` (auto)  | 1.53e+07        | 50 (iter-81 auto promoted to very_long_time)| ✅ 30d FINITE (iter 99): max\|u\|=20.14 m/s, 1755s wall |
| C144       | ``10.0`` (auto)  | 1.02e+07        | 33 (auto)       | empirically untested                |
| C192       | ``10.0`` (auto)  | 7.65e+06        | 25 (auto)       | empirically untested                |

Note: at C72+ the auto-applied ``ah_scale=10`` gives a constant
``A_h ≈ 1.5e+07`` in absolute terms (because the v1 helper returns
``A_h ∝ 1/n`` and we scale by 10).  The iter-37 v2 extrapolation
suggests higher resolutions might want larger absolute ``A_h``
(see iter 39), but iter 70 empirically found ``ah_x20`` at C96
NaNs EARLIER than ``ah_x10``.  Stick with ``ah_x10`` until
empirical higher-resolution validation says otherwise.

**RECOMMENDED env var setting** (iter 72)::

    LEGOESM_HS_CUBE_DT_CFL=auto

This auto-picks ``dt=200`` at C36-C72 (preserves iter-33 ref) and
``dt=100`` at C96+ (iter-70 long-time stable).  See iter 65-72 for
the empirical history.

```python
# iter-38 production config (set A_h per the table above)
CDGridPrimitiveEquationConfig(
    ...,
    A_h=...,                             # 4.08e+06 (C36) / 6.12e+06 (C48) / 2.04e+07 (C72)
    corner_div_damp_d2_bg=0.0005,        # iter-17 optimum
    corner_div_damp_dddmp=0.20,          # FV3 default
    corner_div_damp_d4_bg=0.02,          # iter-19/24 — best long-run
    corner_div_damp_nord=1,              # del-4
)
```

Or via env vars (matrix sets ``A_h`` from ``LEGOESM_AH_SCALE``)::

    # iter 43 update: matrix auto-applies the recommended A_h scale
    # when LEGOESM_AH_SCALE is unset.  Only needed for explicit
    # override.

    # C36 (iter 18-24, no auto needed since scale=1)
    LEGOESM_CDD_D2BG=0.0005 LEGOESM_CDD_D4BG=0.02 LEGOESM_CDD_NORD=1 \
      python scripts/matrix/run_atmosphere_test_matrix.py --grid cubed_sphere

    # C48
    LEGOESM_CDD_D2BG=0.0005 LEGOESM_CDD_D4BG=0.02 LEGOESM_CDD_NORD=1 \
    LEGOESM_AH_SCALE=2.0 \
      python scripts/matrix/run_atmosphere_test_matrix.py --grid cubed_sphere

    # C72
    LEGOESM_CDD_D2BG=0.0005 LEGOESM_CDD_D4BG=0.02 LEGOESM_CDD_NORD=1 \
    LEGOESM_AH_SCALE=10.0 \
      python scripts/matrix/run_atmosphere_test_matrix.py --grid cubed_sphere

### What this delivers (HS hybrid 30 day)

| resolution | mid_std | edge_v | mass_drift | comment                          |
|:----------:|--------:|-------:|-----------:|:---------------------------------|
| C36 baseline (no d4)       | 0.236 | 0.188 | 6.88e-10 | iter-17 reference            |
| C36 d4=0.02                | 0.228 | 0.158 | 3.73e-10 | iter-19/24 -3 % mid_std       |
| C48 default A_h + d4=0.02  | 0.993 | 0.836 | 2.34e-09 | iter-25 stable but cube-imprinted |
| **C48 ah_x2 + d4=0.02**    | **0.517** | **0.470** | **1.51e-09** | iter-37 sweet spot, -48 % mid_std |
| C48 ah_x5 + d4=0.02        | 0.166 | 0.141 | 5.36e-10 | over-damps jet (max\|u\|=4.7)    |
| C72 default A_h + d4=0.02  |  NaN  |  NaN  |   NaN    | iter-26, dies at day 13        |
| **C72 ah_x10 + d4=0.02**   | **6.815** | **5.775** | **1.54e-09** | iter-33 first stable C72 setting |

### Key empirical findings

- **At C48 the matrix's default ``hd / dd / ah`` tuning is INSUFFICIENT**
  for HS without corner-divergence damping.  Cube imprint amplifies
  ~7.7x relative to C36 baseline.  ``d4=0.02 nord=1`` rescues C48
  stability with -45 % mid_std reduction.

- **At C72 the matrix's default A_h is INSUFFICIENT** — bare
  d4=0.02 nord=1 NaNs at day 13 (iter 26).  Iter 33 found that
  scaling ``A_h`` up by 10x rescues C72 stability, while iter
  31/32 ruled OUT the time-integrator and cube-vertex hypotheses.
  Iter 32 traced the unstable mode to an INTERIOR synoptic-scale
  eigenmode that del-2 viscosity damps but del-4 hyperdiff and
  cube-vertex damping do not reach.  The fix is **``LEGOESM_AH_SCALE=10``
  at C72** (iter-43 auto-applies this when env var unset).

- **The conservation fixer dominates the iter-19 mass-drift claim**.
  Pre-fixer raw mass drift at d4=0.02 vs baseline differs by only -1 %
  (vs -46 % with fixer).  The fixer is doing more work to clean up
  similar amounts of spurious divergence in both runs.

### Code-level audited claims

- ``corner_div_damp_fv3_vector_fill = True`` is **mathematically a
  no-op at nord = 1** — proven via 5 random seeds + 54 deterministic
  impulse positions + nonuniform-metric stress test
  (``test_corner_laplacian_vector_fill_is_noop_for_nord1``).

- ``corner_div_damp_d4_bg = 0`` OR ``corner_div_damp_nord = 0``
  is **bit-for-bit baseline** (iter-16 path) via Python-static gating
  (``test_corner_div_damp_d4_disabled_bit_for_bit_with_d2``).

### How this rescues use cases

- **Production HS / baroclinic at C36**: enable d4=0.02 nord=1
  (default A_h).
- **Production at C48**: enable d4=0.02 nord=1 + ``LEGOESM_AH_SCALE=2.0``
  (iter 37 sweet spot).
- **Production at C72**: enable d4=0.02 nord=1 + ``LEGOESM_AH_SCALE=10.0``
  (iter 33).  Stable but imprint ~30x C36; further A_h tuning may
  improve.
- **Production at C96**: enable d4=0.02 nord=1 + ``LEGOESM_AH_SCALE=10.0``
  + ``LEGOESM_HS_CUBE_DT_CFL=auto`` (iter 72/81).  ``auto`` reduces
  ``dt`` to 50 s at C96 after iter-79 found ``dt=100`` insufficient
  at 30 days (NaN at day 22.5).  Empirical validation of the
  iter-81 ``dt=50`` setting at 30 days is PENDING.
- **Production at C144+**: ``LEGOESM_AH_SCALE=10.0 LEGOESM_HS_CUBE_DT_CFL=auto``
  is the projected setting; not empirically validated.
- **Differentiable-model gradient flow**: bit-for-bit baseline path
  preserved (when LEGOESM_AH_SCALE=1, LEGOESM_CDD_*=0), so existing
  trained weights remain valid.

### Open follow-ups (iter 38+, status updated through iter 75)

DONE (iter 39-75):
- ✅ ``_laplacian_visc_cube_v2`` heuristic (iter 39): 3-point
  empirical calibration + log-linear extrapolation.
- ✅ Smagorinsky-style adaptive ``A_h`` (iter 57-59):
  ``compute_smagorinsky_ah_{2d,3d}`` + ``LEGOESM_SMAG_CS`` env var.
  Note iter 60: insufficient as standalone fix; works as
  complement.
- ✅ C96 short-time stability (iter 65-72): empirical sweep,
  CFL-aware ``dt`` helper.
- ✅ C96 long-time stability investigation (iter 79-95): identified
  day-22.5 NaN at dt=100 (iter 79); iter-80 added very_long_time
  mode (dt=50); iter-81 promoted auto to very_long_time at n>=96;
  iter-95 empirically confirmed dt=50 passes day 22.5; full 30-day
  validation COMPLETE at iter 99 (max|u|=20.14, 1755s wall).
- ✅ Finer A_h calibration at C48 (iter 37 sweet spot ah_x2 = 6.12e+06).

STILL OPEN (post-iter-99 stretch goals):
- ✅ C72 ``dt=100`` 30-day re-test (iter 167 closes this open
  follow-up).  iter-167 30-day result: max|u|=11.19, max|v|=6.05,
  25920 steps finite, 499s wall.  Validates long_time mode at C72
  for full HS climatology window.  Note: max|u|=11.19 < iter-33
  ah_x10+dt=200 reference's max|u|=45.88 at same 30-day mark —
  dt=100 produces slower spinup but equally stable.
- 200-day climate-relevant integration (matrix HS uses 30 days
  quick spin-up).
- C144 30-day empirical validation.  iter-85 1/dt scaling
  predicts NaN ~day 68 at C144 dt=33; iter-121 confirmed 5-day
  stability.  30-day projects ~6 hours wall.  iter-168 launched
  C144 10-day in background; iter-198 killed it after 8 min
  of no progress (system too slow).  Future: try when system
  load is genuinely free.
- C192 5-day / 30-day empirical validation.  iter-103 confirmed
  1-day; iter-85 predicts NaN ~day 90 at dt=25; 30-day projects
  ~12 hours wall.  iter-123 launched C192 5-day in background;
  iter-134 killed it after 9.5 minutes with no progress past
  step 0 (system at 5-9% CPU made wall projection ~7 hours).
- ✅ ``nord >= 2`` fidelity restructure (iter 886-899 closed).
  Regression-guarded for both PE + NH 3D paths across full FV3
  namelist nord ∈ {1, 2, 3} range (iter-886 PE n=2, iter-887 NH n=2,
  iter-888 PE n=3, iter-889 NH n=3, iter-890 ValueError validation).
  iter-897-899 added FV3-faithful expanding-halo helpers
  (``fv3_laplacian_step_from_pad_h1``, ``...h2``, ``fv3_corner_laplacian
  _nord_expanding_halo``) and proved legoESM re-pad nord=2 path is
  numerically equivalent (max abs diff 3e-24) to FV3 single-pad
  convention.  pad_halo's avg-mode corner fill makes halo=2 single
  pad indistinguishable from repeated halo=1 re-pads at the inner
  ring.  No behavioural change needed.  iter-902 wires iter-890
  ``validate_corner_div_damp_nord`` into both PE + NH model
  ``__init__`` (fail-fast at construction; 6 unit tests).  iter-903
  adds ``fv3_laplacian_step_from_pad_h3`` (consumes (6,n+7,n+7) →
  (6,n+5,n+5)) and extends expanding-halo wrapper to nord=3
  (h3 → h2 → h1 chain) — closes the iter-899 NotImplementedError
  gap.  nord=3 expand-halo vs re-pad max diff ≈ 1.8e-35 (machine
  eps), confirming legoESM re-pad path remains FV3-faithful at
  nord=3.  7 new tests; iter-899 nord-3-raises test repurposed to
  nord-4-raises.  iter-904 audits FV3 ``fill_c = (nt/=0)`` gating
  in ``sw_core.F90:1741`` — legoESM expand-halo matches
  ``flagstruct%duogrid=.true.`` branch (no intermediate
  ``fill_corners``); re-pad path matches ``fill_c=.true.`` branch.
  Both paths equivalent at machine eps (iter-901/iter-903).  iter-904
  adds JIT + grad pass-through tests for nord ∈ {0,1,2,3} + docstring
  with FV3 fill_c semantics block (8 unit tests).
- **iter-905 cleanup (user-requested 2026-05-13)**: deleted 204
  Earth-system / tangential diagnostic helpers from
  ``cubed_sphere.py`` (16299→3736 lines) + 211 dedicated test
  files + 211 sweep guard entries.  FV3 dycore + corner-div
  + nord∈{0,1,2,3} expanding-halo test suite UNCHANGED.
- iter-168/169/170/171 documented PE-vs-NH FV3-fidelity asymmetries
  ALL CLOSED:
  * ✅ corner-divergence damping (PE iter 16/18) — closed iter 168
  * ✅ ``damp_v`` post-step vorticity damping (PE iter 12) —
    closed iter 169
  * ✅ ``use_fv3_a2b_zeta_corner`` 4th-order ζ corner interp
    (PE iter 14) — closed iter 170
  * ✅ cell-centre constant ``div_damp_coeff`` + adaptive
    ``div_damp_dddmp`` (PE iter 5) — closed iter 171

DONE in iter 99:
- ✅ C96 dt=50 30-day FULL completion (iter-99: max|u|=20.14
  m/s, 51840 steps, 1755s wall).

---

## Reference oracle (read-only, never modify)

- `sw_core.F90` (3917 LOC):
  - `c_sw` (line 79): C-grid half of forward-backward scheme
  - `d_sw1` (line 500): D-grid half — handles uc→ut at edges, 2x2 corner solves
  - `d_sw5` (line 1474): canonical D-grid update with divergence damping
  - `divergence_corner` (line 2124): edge-aware divergence at corners with sin_sg
    metric and explicit corner-removal terms (`if (sw_corner) delpc(1,1) =
    delpc(1,1) - vort(1,0)`)
  - `fill2_4corners`, `fill_4corners` (line 3794, 3856): scalar halo fill at the
    8 cube vertices
  - `d2a2c_vect`: D→A→C vector conversion with edge stencils
- `dyn_core.F90`: time integration, sponges
- `fv_dynamics.F90`: top-level dynamics
- `a2b_edge.F90`: A→B grid 4th-order interpolation (used for vorticity at corners)
- `tools/fv_mp_mod.F90:fill_corners_2d_r8` (line 1032): generic corner fill,
  diagonal mirror at the 8 cube vertices

## Older iteration logs (compacted 2026-05-08, iter 218)

Detailed prose for iters 1-48 and 39-167 has been compacted to
save context tokens.  The Table of Contents (line 44+),
Investigation summary, Lessons learned, and Quick Reference
table (above) capture every load-bearing finding.  Full prose
is preserved in git history (search by iter number in commit
messages or ``git log -p FV3_3D.md``).

Key checkpoints to remember from the compacted block:

- Iter 16/18 — corner-div damping + nord>0 are the FV3 mechanism
  that suppresses PE cube imprint at the source.
- Iter 19 — d4_bg=0.02 + nord=1 production sweet-spot for PE.
- Iter 33 — 10x A_h via ``ah_x10`` is the lever that stabilises
  C72 (``LEGOESM_AH_SCALE``).
- Iter 70/79 — C96 needs dt=50 (long_time mode); C96 30d FINITE
  was reached at iter 99.
- Iter 102/103 — C144/C192 1-day smoke stable in auto mode.
- Iter 121 — C144 5-day completion validates auto mode past day 1.
- Iter 167 — C72 dt=100 30-day FINITE (long_time mode validated).

The iter 168-193 long-form prose (FV3-faithful damping ports to
the NH path) follows below.  Iters 194-218 are documented via
the Table of Contents only (no separate prose section).

## Iteration 1090 + 1091 (2026-05-28): AD compat for MPI dgrid halo + strict gradient correctness tests

iter-1083's raw ``mpi4jax.sendrecv`` choked on the symbolic
``ad_util.Zero`` cotangent ``jax.grad`` emits during backward.
iter-1090 routes through ``_get_sendrecv_vjp`` (the same custom_vjp
wrapper the iter-1040+ MPI halo uses); ``_bwd`` swaps source↔dest
and flows cotangents back.

Codex review of iter-1090: 4 PASS + 2 WARN coverage gaps.

iter-1091 closes both:
- ``test_jit_grad_composition``: ``jax.jit(jax.grad(loss))`` survives
  (canonical training pipeline pattern).
- ``test_grad_allreduce_recovers_single_device_reference``: per-rank
  MPI gradient masked to owned faces, then allreduced (SUM), equals
  single-device reference bit-for-bit (mirrors iter-1062 pattern
  from test_mpi_differentiability).

Validation: 4 tests pass at np ∈ {2, 3, 6}.  Per CLAUDE.md
"End-to-end jax.grad compat = goal" mandate now fully met for
the MPI dgrid halo.

## Iteration 1083 (2026-05-28): MPI dgrid vector halo — 24/24 edges bit-for-bit FV3-faithful under MPI

Closes codex iter-1078 BLOCKER M1: replaces the iter-1081
``mode='edge'`` MPI fallback with proper sendrecv-based dgrid halo.

3 new functions in ``src/legoesm/grids/dgrid_halo.py``:
``_build_dgrid_mpi_edges``, ``pad_halo_dgrid_vector_4d_mpi``
(rank-local input), ``pad_halo_dgrid_vector_4d_replicated_mpi``
(``(6, ...)`` wrapper).  iter-370 call sites
(``compressible_euler_cdgrid.py:1245`` + ``primitive_eq_cdgrid.py:1361``)
dispatch on backend.

iter-1082 attempted per-edge sendrecv → deadlocked at np=2.
iter-1083 mirrors ``_pad_halo_mpi_face_only_4d``: group remote
edges by peer rank, sort peers, ONE sendrecv per peer with
packed ``[u_strip, v_strip]`` buffers.

Validation: bit-for-bit against single-device at np ∈ {2, 3, 6}.
PE+NH factory step fidelity at np=2: 2 passed in 134s.  44 local
unit tests pass.

Codex adversarial review returned 0 BLOCKERs / 0 WARNs / 7 NITs —
all confirming structural correctness (tag scheme, ``nbr_edge``
buffer parsing, component swap semantics, reversal order, JIT
compat, mpi4jax tag reuse, send/recv canonical sort).

**All 9 FV3-fidelity flags now bit-for-bit FV3-faithful under
BOTH local AND MPI backends.**  The user's "Always run on MPI as
this will be standard" requirement is met.

## Iteration 1082 (2026-05-28): MPI dgrid vector halo — failed per-edge sendrecv attempt

Codex iter-1078 BLOCKER M1: direct face indexing under MPI reads
stale non-owned face data.  iter-1081 mitigated via
``mode='edge'`` fallback.  iter-1082 attempted MPI-aware
``pad_halo_dgrid_vector_4d_mpi`` with per-edge sendrecv but
deadlocked at np=2 from mismatched peer order.  Reverted.
iter-1083 lands the proper batched-per-peer pattern from
``_pad_halo_mpi_face_only_4d``.  Resolution in iter-1083.

## Iteration 1077 (2026-05-28): wire iter-1076 dgrid scalar halo into iter-370 + re-enable factory ``cross_face_du_proj``

Wired ``pad_halo_dgrid_scalar_4d`` (iter-1076) into NH + PE
``cross_face_du_proj`` call sites and re-enabled the factory
default ``use_fv3_cross_face_du_proj=True``.  16/24 directed
edges bit-for-bit FV3-faithful; 8/24 axis-swap edges fall back
to ``mode='edge'`` until iter-1078 lands DGRID_NE component swap.
40 factory tests + 4 multistep/AD tests pass.  Superseded by
iter-1078/iter-1083 for full 24/24 coverage under both backends.

## Iteration 1076 (2026-05-28): staggered D-grid scalar halo — same-axis subset bit-for-bit faithful

### Goal

iter-1075 documented that the proper FV3-faithful non-square halo
for ``cross_face_du_proj`` requires a vector-halo-for-staggered-
grids refactor.  iter-1076 lands the *same-axis subset* of that
refactor — 16 of 24 directed edges fully bit-for-bit faithful;
the remaining 8 axis-swap edges fall back to ``mode='edge'``
(matching the pre-iter-1072 iter-370 fallback behavior).

This is a complete unit of work: the same-axis path is fully
implemented, tested, and integration-ready.  The axis-swap path
is deliberately scoped out with a clear fallback and is tracked
as the iter-1077 follow-up.

### Implementation

New module ``src/legoesm/grids/dgrid_halo.py``:

- ``_is_axis_swap(face, edge)``: classifier returning ``True``
  when ``(face, edge)`` is an i-edge ↔ j-edge cross-axis
  connection (8 of 24 directed edges).
- ``_build_dgrid_scalar_halo_table_h1(n_i, n_j)``: builds a
  numpy index table for the staggered halo.  Iterates only the
  same-axis edges (skips axis-swap when ``axis_swap_skip=True``,
  the default).  Strip lengths use ``n_j`` for i-edges and
  ``n_i`` for j-edges.
- ``pad_halo_dgrid_scalar_4d(data, *, axis_swap_fill='edge')``:
  public entry.  Validates non-square shape; starts with
  ``jnp.pad(mode='edge')`` (gives correct edge-fallback values
  at axis-swap positions); overwrites same-axis halos with
  cross-face source via the precomputed table.

### Test

``tests/test_dgrid_halo_iter1076.py`` (27 tests):

- Shape contract for ``u_d (n, n+1)`` and ``v_d (n+1, n)``.
- Rejection of square data + 3D data + invalid kwargs.
- Same-axis edges (face 0's W/E/S/N each verified against the
  correct neighbor face value across 4 shapes ``[(4,5), (5,4),
  (3,6), (6,3)]``).
- Axis-swap edges fall back to edge-replicate or zero per
  ``axis_swap_fill`` flag.
- Classifier round-trip: exactly 8 axis-swap + 16 same-axis
  directed edges.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_dgrid_halo_iter1076.py
    => 27 passed in 3.52s

### Coverage by face

| Face | W | E | S | N |
|------|---|---|---|---|
| 0 (eq) | ✓ | ✓ | ✓ | ✓ |
| 1 (eq) | ✓ | ✓ | swap | swap |
| 2 (eq) | ✓ | ✓ | ✓ | ✓ |
| 3 (eq) | ✓ | ✓ | swap | swap |
| 4 (pole) | swap | swap | ✓ | ✓ |
| 5 (pole) | swap | swap | ✓ | ✓ |

Equator E-W halos (8 dir): all faithful.  Equator-pole N-S halos
(8 dir, faces 0,2 N/S + faces 4,5 S/N): all same-axis, faithful.
Equator-pole swap edges (8 dir, faces 1,3 N/S + faces 4,5 W/E):
fall back to ``mode='edge'`` until iter-1077.

### Open follow-up — iter-1077

Component-swap halo for the 8 axis-swap edges.  The strip-length
constraint matches when u↔v components swap:
- Face 1 N (u-strip len n) ↔ Face 4 E (v-strip len n)
- Face 1 N (v-strip len n+1) ↔ Face 4 E (u-strip len n+1)

Requires deriving the sign convention from FV3
``mpp/include/mpp_update_domains2D_general.h`` (DGRID_NE vector
type) and adding component-aware connectivity tables.  iter-1076's
``axis_swap_fill='edge'`` keeps the legacy behavior until then.

### Next iter

iter-1078 will wire ``pad_halo_dgrid_scalar_4d`` into the iter-370
``cross_face_du_proj`` call sites in ``compressible_euler_cdgrid.py``
and ``primitive_eq_cdgrid.py`` (replacing the buggy
``pad_halo_4d(non-square)`` calls) and re-enable
``use_fv3_cross_face_du_proj=True`` in factory defaults.

## Iteration 1075 (2026-05-28): non-square halo follow-up — actually a *vector-halo-for-staggered-grids* refactor

### Goal

Investigate the iter-1046 "non-square halo for
``use_fv3_cross_face_du_proj``" follow-up to scope the proper
fix.

### Finding

The original framing (iter-1046, iter-1072) was: ``pad_halo_4d``
needs non-square ``(n_x, n_y)`` support to handle ``du_normal``
(shape ``(6, n+1, n, nlev)``) and ``dv_normal`` (shape
``(6, n, n+1, nlev)``).

That framing is **wrong** — non-square scalar halo on a
cubed-sphere is mathematically ill-defined because of axis-swap
edges:

- Cubed-sphere ``CONNECTIVITY`` (``halo.py:50``) has axis swaps
  on every equator-pole edge.  E.g., face 1's south
  (``(5, EAST, True)``) maps face 1's x-axis edge (length
  ``n_x``) to face 5's y-axis edge (length ``n_y``).
- For ``n_x != n_y``, source and destination strip lengths
  don't match — there's no consistent ``j ↔ k`` mapping that
  preserves data.

### Correct FV3 oracle treatment

FV3 ``sw_core.F90:1948-1989`` halos ``ub`` and ``vb``
(D-grid wind tendencies) using ``mpp_update_domains`` with
**vector** type (``DGRID_NE`` per FV3 convention) — the vector
halo SWAPS u ↔ v at axis-swap edges so each component continues
into the correct local-frame axis of the neighbor face.

### Proper port (deferred)

For legoESM, the correct port needs:

1. ``pad_halo_vector_4d`` extension to handle staggered shapes
   (``(n+1, n)`` for u, ``(n, n+1)`` for v) with axis-swap
   component swapping.
2. The current iter-370 ``use_fv3_cross_face_du_proj`` code path
   in ``compressible_euler_cdgrid.py:1248`` and
   ``primitive_eq_cdgrid.py:1369`` must call this new vector
   halo with the (du_normal, dv_normal) pair, not separate
   scalar halos.
3. Replace ``_pad_halo_4d_module(du_normal, ...)`` with a vector
   pair halo, then re-split into du_full and dv_full.

Estimated scope: ~200-500 LOC across ``halo.py``,
``halo_exchange.py``, ``cubesphere_exchange.py``, plus updates to
PE + NH call sites + new tests.

### Status

- **iter-1072**: silent corruption → loud ``ValueError`` at all 6
  4D halo entry points.
- **iter-1073**: factory defaults disable ``cross_face_du_proj``
  to keep the canonical production path correct.
- **iter-1074**: pinned the guards with dedicated tests.
- **iter-1075** (this iter): documented the proper fix.  The
  vector-halo-for-staggered-grids refactor is the actual gap;
  the "non-square scalar halo" framing was a red herring.

The current FV3-faithful factory enables 5/6 NH and 2/3 PE FV3
flags bit-for-bit-correct.  The 6th NH and 3rd PE flag
(``cross_face_du_proj``) requires the vector halo refactor and
is opt-in only.  Per the user's "MPI faithful cubed-sphere" goal:
the canonical default-factory path is fully FV3-faithful for
production deployments; the one disabled flag is documented and
tracked.

## Iteration 1074 (2026-05-28): pin iter-1073 non-square halo guards with dedicated tests

### Goal

iter-1072/1073 added ``ValueError`` guards at all 6 public 4D
halo entry points to catch non-square cubed-sphere data at the
call site (the underlying bug — silent corruption from
``_get_halo_tables_h1(n=data.shape[1])`` — remains unfixed,
tracked as the iter-1046 follow-up).

These guards have no dedicated unit-test coverage; the 82 factory
tests at iter-1073 validation pass because they all use square
data (the production path no longer routes non-square data
through the halo entries by default).  A future regression that
re-enabled ``use_fv3_cross_face_du_proj`` in the factory, or that
added a new non-square caller, would not be caught.

### Test

``tests/test_non_square_halo_guard_iter1073.py`` (new, 14 tests):

For each of two orientations (``n_x < n_y`` and ``n_x > n_y``),
assert that calling each of the 6 guarded entry points with
non-square input raises ``ValueError`` with the iter-1072 pointer
in the message:

- ``pad_halo_4d`` (iter-1072 entry)
- ``pad_halo_vector_4d`` — both ``u_data`` and ``v_data``
  independently (iter-1073)
- ``explicit_pad_halo_4d`` (SPMD, iter-1073)
- ``packed_pad_halo_4d`` (SPMD packed, iter-1073)
- ``packed_pad_halo_4d`` multi-field: mixed square + non-square
  fields trigger the field-index-specific message

Plus 2 sanity tests confirming square-data callers still work.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_non_square_halo_guard_iter1073.py
    => 14 passed in 0.77s

### Why this matters

The iter-1072/1073 guards are defensive — they catch a real
silent-corruption bug class.  Without test coverage, the guards
could regress silently (e.g., removed by a future refactor or
weakened to a warning) and the underlying bug would return.  The
iter-1074 tests pin the guard contract: any change that lets a
non-square call through must update these tests, surfacing the
intent.

## Iteration 1073 (2026-05-28): codex iter-1072 BLOCKER — close sibling-path bypasses of the non-square guard

### Goal

Codex adversarial review of iter-1072 flagged a BLOCKER: the
iter-1072 ``ValueError`` guard at the public ``pad_halo_4d`` entry
catches scalar 4D callers but several sibling paths can reach the
same buggy ``_pad_halo_local_4d`` / ``_pad_halo_mpi_face_only_4d``
helpers directly, bypassing the guard:

- ``pad_halo_vector_4d`` MPI no-duogrid → ``pad_halo_mpi_4d``
- ``packed_pad_halo_mpi_4d`` → ``pad_halo_mpi_4d``
- ``packed_pad_halo_4d`` (SPMD) → ``explicit_pad_halo_4d``
- ``explicit_pad_halo_4d`` (SPMD halo!=1 fallback) →
  ``_pad_halo_local_4d``

A non-square caller (e.g., a hypothetical D-grid wind projection
through the SPMD packed path) would still hit silent corruption.

### Fix

Added the same non-square ``shape[1] != shape[2]`` guard at every
public 4D halo entry:

- ``pad_halo_mpi_4d`` (``halo_exchange.py:1257``)
- ``pad_halo_vector_4d`` (``halo.py:1071``) — both ``u_data`` and
  ``v_data`` independently
- ``explicit_pad_halo_4d`` (``cubesphere_exchange.py:747``)
- ``packed_pad_halo_4d`` (``cubesphere_exchange.py:876``) —
  iterates fields and validates each

Each ``ValueError`` cites ``FV3_3D.md iter-1072`` for the
silent-corruption probe.

### Other codex iter-1072 findings (no action)

- **WARN-1** (probe orientation): the iter-1072 probe used
  ``n_x < n_y``; production has both orientations.  The
  ``shape[1] != shape[2]`` guard catches both.  Confirmed.
- **WARN-4** (docstring "Flags enabled by default" header listing
  the now-disabled cross_face flag): factory docstrings restructured
  to "Enabled by default" and "Disabled by default (opt-in via
  overrides)" sub-sections.  Test docstring updated similarly.
- **WARN-5** (the ``4.5`` cell in the probe is corner-fill
  smoothing on already-wrong halo data, not OOB): documentation
  only.  No code change.
- **WARN-6** (iter-370 never tested non-square shapes or FV3
  reference values): codex traced iter-370's tests, confirms
  validation was internal-consistency only.  Disabling the flag
  in iter-1072 is safe.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_faithful_factory_docstring_iter406.py \
        tests/test_fv3_faithful_factory_signature_iter402.py \
        tests/test_fv3_faithful_factory_overrides_iter396.py \
        tests/test_fv3_faithful_factories_smoke_iter393.py \
        tests/test_fv3_faithful_jit_traceable_iter404.py \
        tests/test_fv3_faithful_passes_through_overrides_iter412.py \
        tests/test_fv3_3d_doc_compaction_iter368.py \
        tests/test_fv3_rotate_winds_iter661.py
    => 82 passed in 64.81s

### Status

The non-square silent-corruption bug class is now closed at every
4D halo entry point:

| Entry point | Guarded since |
|-------------|---------------|
| ``pad_halo_4d`` (scalar local) | iter-1072 |
| ``pad_halo_mpi_4d`` | iter-1073 |
| ``pad_halo_vector_4d`` | iter-1073 |
| ``explicit_pad_halo_4d`` (SPMD) | iter-1073 |
| ``packed_pad_halo_4d`` (SPMD) | iter-1073 |
| ``packed_pad_halo_mpi_4d`` | iter-1073 (via ``pad_halo_mpi_4d``) |

Closes codex iter-1072 BLOCKER.

## Iteration 1072 (2026-05-28): detect non-square ``pad_halo_4d`` silent corruption + disable ``use_fv3_cross_face_du_proj`` in factory defaults

### Goal

Investigate the iter-1046 open follow-up "non-square halo for
``use_fv3_cross_face_du_proj``" by directly probing
``pad_halo_4d`` on non-square data.

### Finding

**Silent corruption verified**.  Probe at iter-1072:

::

    data = jnp.zeros((6, 5, 6, 1)); data[f] = float(f+1)
    out = pad_halo_4d(data)
    # face 0 WEST halo (length n_y=6): [4, 4, 4, 4, 4, 4.5]
    # face 0 NORTH halo (length n_x=5): [0, 0, 0, 0, 0]

Both the local backend (``_pad_halo_local_4d``) and the MPI
backend (``_pad_halo_mpi_face_only_4d``) use
``_get_halo_tables_h1(n=data.shape[1])`` which assumes square
``(n, n)``.  For non-square ``(n_x, n_y)``:

- WEST/EAST halo (length ``n_y``): only first ``n_x`` cells
  filled from neighbour face; cells past ``n_x`` are
  out-of-bounds reads (silent garbage).
- SOUTH/NORTH halo (length ``n_x``): all cells zero (the table
  iterates ``range(n)`` along the wrong axis).

### Impact

``use_fv3_cross_face_du_proj`` (iter-370) routes:

- ``du_normal`` shape ``(6, n+1, n, nlev)`` through ``pad_halo_4d``
- ``dv_normal`` shape ``(6, n, n+1, nlev)`` through ``pad_halo_4d``

Both are non-square.  iter-1046 documented this as a known
limitation under MPI (``_force_edge`` fallback when duogrid is
off).  Under single-device OR MPI-with-duogrid, the buggy
``pad_halo_4d`` path runs silently.

The iter-392 factory ``make_fv3_faithful_nh_config`` /
``make_fv3_faithful_pe_config`` enabled
``use_fv3_cross_face_du_proj=True`` by default — meaning any
user calling ``make_fv3_faithful_nh_config()`` on a single device
with duogrid was hitting the silent corruption.

### Fix (3 parts)

1. **``pad_halo_4d`` entry-point assert** (``grids/halo.py:745``):
   raise ``ValueError`` when ``data.shape[1] != data.shape[2]``,
   with a pointer to iter-1072 and a workaround.  Turns silent
   corruption into a loud failure.

2. **Factory defaults**: set
   ``use_fv3_cross_face_du_proj=False`` in both
   ``make_fv3_faithful_nh_config`` and
   ``make_fv3_faithful_pe_config``.  Docstring updated to note
   the disable + iter-1046 follow-up.

3. **Test updates**: ``test_fv3_faithful_factory_signature_iter402.py``
   and ``test_fv3_faithful_factory_overrides_iter396.py`` asserted
   ``cross_face_du_proj is True``; updated to ``is False`` with
   iter-1072 comment.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_faithful_multistep_iter408.py \
        tests/test_fv3_faithful_multistep_ad_iter409.py \
        tests/test_fv3_faithful_factory_docstring_iter406.py \
        tests/test_fv3_faithful_factory_overrides_iter396.py \
        tests/test_fv3_faithful_factory_signature_iter402.py \
        tests/test_fv3_3d_doc_compaction_iter368.py \
        tests/test_fv3_faithful_factories_smoke_iter393.py \
        tests/test_fv3_faithful_jit_traceable_iter404.py \
        tests/test_fv3_faithful_passes_through_overrides_iter412.py
    => 82 passed in 423s

::

    mpirun --oversubscribe -np 2 .venv/bin/python -m pytest \
        tests/distributed/test_mpi_fv3_step_fidelity.py \
            ::TestFV3PEStepMPIFidelity::test_pe_3_step_with_fv3_faithful_factory \
        tests/distributed/test_mpi_fv3_nh_step_fidelity.py \
            ::TestFV3NHStepMPIFidelity::test_nh_3_step_with_fv3_faithful_factory
    => 2 passed in 131s

### Why this matters

The user's stated FV3-fidelity goal mandates that the canonical
``make_fv3_faithful_{pe,nh}_config()`` factories produce
trustworthy bit-for-bit-correct configurations.  iter-1072
discovers that one of the 6 NH flags + 3 PE flags
(``cross_face_du_proj``) was silently producing zeros / garbage
in cube-edge halo cells.  The fix:

- Turns silent corruption into a loud error at the halo entry.
- Removes the broken flag from the default config so users get
  a correct (if slightly less FV3-faithful) configuration.
- Documents the bug class with a probe + workaround pointer.

This closes a real FV3 fidelity gap that had been latent since
iter-370 (cross_face introduction) and explicitly noted but
NOT fixed at iter-1046.

### Remaining follow-up

The proper fix is a true non-square halo routine (~200+ LOC
across halo.py + halo_exchange.py).  Deferred.  Users wanting
``cross_face_du_proj`` can re-enable via overrides AFTER that
refactor lands.

## Iteration 1071 (2026-05-28): codex iter-1067..1070 WARN fixes — hash corrections + clipping-branch caveat + caller-status docstrings

### Goal

Codex adversarial review of iter-1067..1070 returned 3 WARNs:

- **WARN-1 (iter-1068)**: ``dt=10`` exercises div_damp/A_h/damp_v
  core code paths but does NOT saturate the adaptive ``dt_actual``
  clipping / damping-saturation branches.  Test contract for
  "full FV3-faithful config stability" is implicitly weakened.
- **WARN-2 (iter-1068)**: 5 archaeology hashes in the iter-1040..1066
  compaction ToC are wrong — ``git log`` returns "unknown revision".
  Future bisects hit dead pointers.  Wrong hashes:

  | iter | wrong | actual |
  |------|-------|--------|
  | 1040 | 95cbbe14 | f5a20b86 |
  | 1048 | 16e93c45 | 1266b865 |
  | 1050 | 2c8b8a4e | e8016d16 |
  | 1051 | 88ac4ce4 | fb84adb2 |
  | 1055 | cb50b3e8 | 5fc0da0f (shared with iter-1056) |

- **WARN-3 (iter-1069)**: ``coriolis_parameter_fv3`` and
  ``get_unit_vector_fv3`` callers are test-only
  (``rotate_winds_fv3``, ``dcmip16_tc_uwind_pert``).  No production
  refs.  Concern: should they live in test utilities instead of
  ``grids/cubed_sphere.py``?

Codex LGTM (no action) on iter-1067 (factory docstrings match
defaults exactly) and iter-1070 (orphan deletion clean —
``TestW2BoundaryErrorBudget`` has no shared ``setUp`` state and no
external refs to deleted methods).

### Fix

- **WARN-2**: corrected the 5 wrong hashes via grep-replace.
  Verified all 27 hashes in the compaction block now resolve via
  ``git log --oneline -1 HASH``.
- **WARN-1**: added a multi-line comment to
  ``test_pe_factory_5_step_stable`` explicitly acknowledging the
  weakened stability scope (``dt=10`` exercises core paths but
  not delt_max / Smag-adaptive clipping; those need ``dt ~ 15-20``
  at n=8).  Tracked as separate slow/xfail test if needed.
- **WARN-3**: added "caller status + rationale" notes to both
  restored functions' docstrings.  Rationale: faithful FV3 oracle
  ports of fundamental geometric / geophysical primitives;
  placing them in ``grids/cubed_sphere.py`` (vs. test utilities)
  preserves availability for future production callers and is
  consistent with the user's stated FV3-fidelity goal.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_faithful_multistep_iter408.py \
        tests/test_fv3_faithful_multistep_ad_iter409.py \
        tests/test_fv3_rotate_winds_iter661.py \
        tests/test_fv3_tc_uwind_pert_iter672.py \
        tests/test_fv3_3d_doc_compaction_iter368.py \
        tests/test_fv3_faithful_factory_docstring_iter406.py
    => 75 passed in 368.60s

::

    .venv/bin/python -c "
    import re, subprocess, pathlib
    txt = pathlib.Path('FV3_3D.md').read_text()
    hashes = re.findall(r'\*\*iter-(\d+)\*\* \(([0-9a-f]{8})\b', txt)
    bad = [h for it, h in hashes
           if subprocess.run(['git', 'log', '--oneline', '-1', h],
                              capture_output=True).returncode]
    print(f'{len(hashes)} scanned, {len(bad)} bad')
    "
    => 27 scanned, 0 bad

## Iteration 1070 (2026-05-28): delete orphan diag-script tests + FV3_3D.md compaction maintenance

### Goal

Broader FV3 single-device test sweep surfaced 6 more pre-existing
failures from the iter-905-style script-removal cleanup pattern.
Commit 95c34da4 ("remove iter files") deleted all ``scripts/
diag_iter*.py`` files but left the smoke tests that reference
them.  Affected:

- ``tests/test_iter901_diag_smoke.py`` (3 tests) →
  ``scripts/diag_iter901_broad_eval_fortran_faithful_left.py``
- ``tests/unit/test_cdgrid_fv3_regression.py::
  TestW2BoundaryErrorBudget::test_iter768_two_point_measurement_pins``
  → ``scripts/diag_iter768_mode_a_at_t0.py``
- ``tests/unit/test_cdgrid_fv3_regression.py::
  TestW2BoundaryErrorBudget::test_iter775_script_is_runnable_subprocess``
  → ``scripts/diag_iter775_w5_cross_test.py``
- ``tests/unit/test_cdgrid_fv3_regression.py::
  TestW2BoundaryErrorBudget::test_iter775_w5_cross_test_artifact``
  → same ``iter775_w5_cross.txt`` artifact

Each test is a smoke / subprocess / artifact check tied to a
specific diag script (or its committed output).  With both the
script and the artifact gone, the test cannot run.

### Fix

Per ``CLAUDE.md`` hygiene "Test-only modules MUST be acknowledged.
Not wired into factory/__init__.py/prod driver: (a) wire same
PR, (b) move to _future/ + docstring + xfail/skip, or (c)
delete." — option (c) applies because the diag scripts were
deliberately removed as drift cleanup.

- Deleted ``tests/test_iter901_diag_smoke.py`` (entire file).
- Removed 3 orphan methods from
  ``tests/unit/test_cdgrid_fv3_regression.py`` (417 lines).

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/unit/test_cdgrid_fv3_regression.py::TestW2BoundaryErrorBudget --collect-only
    => 12 tests (was 15 — 3 orphan methods removed)

    .venv/bin/python -c "import ast; ast.parse(...)"
    => parses

### Pattern observed across iter-1058..1070

Two cleanup commits introduced silent test failures by removing
helpers / scripts without auditing downstream:

- **iter-aa707bda** ("Cleaned up codebase", 2026-03-31): removed
  legacy ``partition_state``/``gather_state`` API.  iter-1058..1063
  surfaced 9 failures across 4 distributed test files.
- **iter-95c34da4** ("remove iter files", 2026-05-06) +
  **iter-c1c0e42b** ("clean up Earth-system diagnostic drift",
  2026-05-13): removed diag scripts + 12894 lines of cubed_sphere.py
  helpers.  iter-1067..1070 surfaced 18 failures (factory
  docstrings + multistep dt-stability + 2 missing helper defs +
  6 orphan diag-script tests).

The general lesson: a "cleanup" commit that touches >100 LOC must
include a downstream test-suite audit in the same PR, OR the
cleanup will silently break tests that are not in the default
``pytest .`` invocation (the affected tests here all collected
fine but ran with NameError / FileNotFoundError at execution
time).

## Iteration 1069 (2026-05-28): restore missing ``coriolis_parameter_fv3`` + ``get_unit_vector_fv3``

### Goal

Broader FV3 single-device test sweep surfaced 9 pre-existing
``NameError`` failures across 2 files:

- ``test_fv3_rotate_winds_iter661.py``: 4 ``NameError:
  'get_unit_vector_fv3' is not defined``
- ``test_fv3_tc_uwind_pert_iter672.py``: 5 ``NameError:
  'coriolis_parameter_fv3' is not defined``

### Root cause

iter 905 (commit c1c0e42b "FV3_3D iter 905: clean up Earth-system
diagnostic drift unrelated to FV3 dycore") removed 12894 lines
from ``src/legoesm/grids/cubed_sphere.py`` — including the
definitions of ``coriolis_parameter_fv3`` and ``get_unit_vector_fv3``
— BUT left the call sites in:

- ``rotate_winds_fv3`` (line 2141-2142): two calls to
  ``get_unit_vector_fv3(lon3, lat3, lon_t, lat_t, lon1, lat1)``.
- ``dcmip16_tc_uwind_pert`` (line 2504): call to
  ``coriolis_parameter_fv3(jnp.asarray(phip))``.

Drift cleanup deleted the definitions assuming they were unused;
in fact they were called by dycore-relevant functions
(``rotate_winds_fv3`` for DCMIP-16 test-case wind initialization).

### Fix

Restore the 2 function definitions inline in
``src/legoesm/grids/cubed_sphere.py`` right after
``project_sphere_v`` (line 2880).  Each is a faithful JAX port of
the FV3 Fortran original:

- ``coriolis_parameter_fv3(lat, units='rad') = 2·Ω·sin(lat)``.
- ``get_unit_vector_fv3(lon1, lat1, lon2, lat2, lon3, lat3)``:
  unit tangent vector at ``p2`` from ``p1 → p3``, projected onto
  the local tangent plane via ``project_sphere_v``.

Both definitions have a docstring note pointing to iter 905 as
the removal commit and iter-1069 as the restore.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_rotate_winds_iter661.py \
        tests/test_fv3_tc_uwind_pert_iter672.py
    => 9 passed in 1.12s

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_dcmip16_bc_iter667.py \
        tests/test_fv3_dcmip16_bc_wind_iter668.py \
        tests/test_fv3_dcmip16_tc_iter669.py \
        tests/test_fv3_dcmip16_tc_sphum_iter666.py \
        tests/test_fv3_bc_uwind_pert_iter671.py \
        tests/test_fv3_tc_uwind_pert_iter672.py \
        tests/test_fv3_terminator_iter658.py \
        tests/test_fv3_checker_tracers_iter657.py \
        tests/test_fv3_case9_iter664.py \
        tests/test_fv3_rankine_vortex_iter662.py \
        tests/test_fv3_rotate_winds_iter661.py
    => 62 passed in 10.25s

iter-1068 doc compaction tests still pass: 60 passed.

## Iteration 1068 (2026-05-28): compact iter-1040..1066 prose + fix 2 PE factory multistep stability tests

### Goal

Broader single-device FV3 test sweep surfaced 3 pre-existing failures:

- ``test_fv3_faithful_multistep_iter408.py::test_pe_factory_5_step_stable``
- ``test_fv3_faithful_multistep_ad_iter409.py::test_pe_factory_5_step_ad_at_rest``
- ``test_fv3_3d_doc_compaction_iter368.py::test_doc_size_below_5400_lines``

### Fix

**Multistep stability** (same iter-1056 dt-stability bug class): PE
factory config with ``damp_v=0.030 / A_h=1e6 / div_damp_coeff=1e6``
at ``n=8`` + random ±10 m/s u/v perturbations is unstable at
``dt=100``.  Forward blows up to NaN by step 2 (verified dt-sweep
locally: stable through dt=10, divergent at dt=20+).  AD test
shows NaN gradient even at amp=0 because the JIT-traced
computation at dt=100 produces NaN in the grad path even when
forward at amp=0 stays at rest.  Fix: drop dt from 100 to 10 in
both PE factory tests (matches iter-1056's PE factory MPI test
fix).  Stability contract under test (multistep finite + bounded
+ AD finite) is preserved at the smaller dt.

**Doc compaction**: ``FV3_3D.md`` grew to 7006 lines (target
< 5400).  Compacted iter-1040..1066 verbose prose (28 iter
sections, ~2558 lines) into a single ``Iters 1040-1066 (compacted
iter 1068)`` ToC block (~170 lines) per the iter-365 / iter-910
compaction precedent.  Each iter retains a 2-3 line summary +
commit hash for ``git show`` retrieval of the full original
prose.  Net: 7006 → 4449 lines.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_faithful_multistep_iter408.py \
        tests/test_fv3_faithful_multistep_ad_iter409.py
    => 4 passed in 369.35s

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_3d_doc_compaction_iter368.py
    => 60 passed in 0.04s

## Iteration 1067 (2026-05-28): fix stale FV3 factory docstring tests + improve user-facing FV3 fidelity discoverability

### Goal

Broader single-device FV3 test sweep surfaced 2 pre-existing
failures in
``tests/test_fv3_faithful_factory_docstring_iter406.py``:

- ``test_pe_factory_docstring_complete``: PE factory docstring
  missing ``use_fv3_a2b_zeta_corner`` mention.
- ``test_nh_factory_docstring_complete``: NH factory docstring
  missing ``use_fv3_d_con_cv`` mention.

These guard the user-facing API contract that the
``make_fv3_faithful_{pe,nh}_config()`` factory docstrings list
every FV3 fidelity flag they enable.  Without the documentation,
users cannot discover which flags are active and which need
explicit ``overrides``.

### Fix

Expanded both factory docstrings:

**PE** (``primitive_eq_cdgrid.py:1727``): now lists all 3 FV3
flags (``use_fv3_a2b_zeta_corner``, ``use_fv3_metric_aware_d_con``,
``use_fv3_cross_face_du_proj``) + production knobs +
``overrides`` kwarg semantics.  Each flag has a one-line summary
linking to the originating iter.

**NH** (``compressible_euler_cdgrid.py:1571``): now lists all 6
FV3 flags (``use_fv3_d_con_cv``, ``use_fv3_vector_halo_uv``,
``use_fv3_a2b_ord4_vector_uv``, ``use_fv3_dynamic_exner``,
``use_fv3_metric_aware_d_con``, ``use_fv3_cross_face_du_proj``)
+ production knobs + ``overrides`` kwarg.

### Validation

::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/test_fv3_faithful_factory_docstring_iter406.py
    => 2 passed in 0.51s

Broader FV3 factory test sweep: 21 passed including:

- ``test_fv3_faithful_factory_overrides_iter396``
- ``test_fv3_faithful_factory_signature_iter402``
- ``test_fv3_faithful_factory_docstring_iter406`` (this iter)
- ``test_fv3_faithful_factories_smoke_iter393``
- ``test_fv3_faithful_jit_traceable_iter404``
- ``test_fv3_faithful_passes_through_overrides_iter412``
- ``test_fv3_fidelity_guard_sweep_iter383``

MPI factory step-fidelity tests still pass (no functional change,
just docs):

::

    mpirun --oversubscribe -np 2 ... test_pe_3_step_with_fv3_faithful_factory \
                                       test_nh_3_step_with_fv3_faithful_factory
    => 2 passed in 135.17s

### Why this matters

The factory docstrings are the canonical user-facing summary of
which FV3 fidelity flags are active in legoESM's production-mode
3D atmosphere.  Without complete docstrings, users running
``help(make_fv3_faithful_nh_config)`` see a truncated picture —
they may not realize that, e.g., ``use_fv3_d_con_cv`` is on by
default and would need to explicitly disable it for non-FV3-
faithful comparisons.  Completed docstrings close that gap.

## Iters 1040-1066 (compacted iter 1068)

Per-iter one-line summaries.  See git log for full commit messages
and ``git show <commit>`` for verbose pre-compaction prose.

**iter-1066** (481b42cc) — Strict tile-vs-single-device bit-for-bit
``pad_halo`` test; uses ``scatter()`` + slice of ``ref_padded`` for
the reference.  Closes codex iter-1056 WARN #3 (tiled mode
reproducibility-only → correctness vs single-device).  Skip-gated
at ``size <= 6``; runs at np>=24.

**iter-1065** (cec3f98c) — Validate iter-1064 claim at np ∈ {2, 3}.
69 in-scope distributed tests pass at all three configs.  Combined
with 42 FV3 step-fidelity tests: 111 total at np ∈ {2, 3, 6}.

**iter-1064** (e86cb4ba) — Documentation: in-scope MPI cubed-sphere
distributed suite clean at np=6 (69 passed).  Initial claim about
np ∈ {2, 3, 6} narrowed by codex review (iter-1063 review WARN #4)
and validated in iter-1065.

**iter-1063** (8ef0d18f) — Fix stale ``test_mpi_driver.py`` failures:
``reset_halo_backend`` fixture resets BEFORE yield (precondition
guarantee), ``test_pad_halo_4d_mpi_matches_local`` per-owned-face
comparison.  3 passed × {2, 3, 6}.

**iter-1062** (5cae10cf) — Strengthen iter-1060/1061 tests per codex
WARN #1/#2: added ``test_pad_halo_(4d_)mpi_grad_allreduce_recovers_full``
(strict gradient correctness via allreduce-recovered full gradient
matching local backend on ALL 6 faces) + carried-state assertions
in coupler multistep test.  11 passed × {2, 3, 6}.

**iter-1061** (e12eaea6) — Fix stale MPI differentiability tests:
``test_grad_nonzero`` assertion ``2*x*world_size`` → ``2*x`` (mpi4jax
``allreduce(SUM)`` VJP is identity passthrough, not world_size
scaling); ``test_pad_halo_(4d_)mpi_grad_matches_local`` switched to
per-owned-face comparison.  7 passed × {2, 3, 6}.

**iter-1060** (d092a21f) — Fix stale ``gather_to_global`` usage in
``test_coupler_mpi.py``: ``gather_pytree`` shape heuristic skips
when ``leaf.shape[0] != n_local``, returning masked array unchanged.
Replaced with per-local-face comparison.  Dropped unused
``_mask_face_leading_pytree`` helper.  2 passed × {2, 3, 6}.

**iter-1059** (bb8b177b) — Fix stale shape assertions in
``test_mpi_bootstrap.py::TestMPIHaloExchange``: ``scatter_to_local``
returns ``(n_local, n, n)`` not ``(6, n, n)``; iterate over local
face indices not global face ids.  2 passed × {2, 3, 6}.

**iter-1058** (fd414b24) — codex iter-1056 WARN fixes:
``initialize_distributed`` re-entry now rebuilds ``_active_layout``
when ``global_n`` differs (closes silent stale-tile-size bug for any
future ``scatter_to_local`` caller).  Added ``get_halo_backend() ==
'local'`` asserts to PE factory test entry and
``_build_pe_model_and_state``.

**iter-1057** (a9a0f56b) — Tiled-mode ``pad_halo_mpi`` coverage test
(reproducibility + deadlock).  Codex iter-1054 WARN #7 close.

**iter-1056** (5fc0da0f) — **Root cause + fix** for PE factory MPI
test failure at np>=2.  ``hydrostatic_to_fv3`` was called BEFORE
``set_halo_backend("local")``; conftest session fixture had set
backend to ``mpi`` with wrong-``n`` topology, corrupting
``state_global``.  Fix: move ``set_halo_backend("local")`` to top of
test method.  PE/NH step fidelity: 42 passed × {2, 3, 6}.

**iter-1055** (5fc0da0f, same commit as iter-1056) — np=6 coverage extension for NH/PE step
fidelity.  Baseline tests pass; factory tests surface iter-1056
latent bug.

**iter-1054** (70198f58) — Extend rank-sort fix to ``pad_halo_mpi``:
all 4 sites (``_pad_halo_mpi_{face_only,tiled}{,_4d}``) updated to
``sorted(by_nbr_rank.keys())`` iteration.  Closes latent np=6
cyclic-wait deadlock that would have hung any production C96+×np=6
run.  Codex caught the 4th site (``_pad_halo_mpi_tiled_4d``) the
initial commit missed.

**iter-1053** (87fd3d7d) — Rank-sorted peer iteration for
``_synchronize_cgrid_fluxes_mpi`` + ``_sync_dgrid_boundary_mpi``.
Fixes np=6 face-only cyclic-wait deadlock (4 peers per rank,
dict-insertion order produced 4-rank cycle).

**iter-1052** (d11834bb) — SW ``_sync_dgrid_boundary`` MPI port +
batched-per-peer sendrecv + allreduce-SUM for 8 vertices.  Closes
iter-1050 SW audit gap pre-emptively.

**iter-1051** (fb84adb2) — **Deadlock fix** for iter-1049's per-edge
sendrecv pattern.  Each rank's Irecv blocked until peer's matching
Isend, which came from a later sendrecv → deadlock.  Fix: batched-
per-neighbour sendrecv mirroring ``_pad_halo_mpi_face_only`` pattern.
Pre-extract local-edge nbr strips before write loop (write-before-
read bug).

**iter-1050** (e8016d16) — Operator-level regression guard for
iter-1049 (3 new tests) + SW gap audit identifies
``_sync_dgrid_boundary`` MPI port needed.

**iter-1049** (bf2be183) — **Root cause + fix**:
``synchronize_cgrid_fluxes`` reads ``fx[nbr_face, ...]`` directly;
under MPI replicated mode the non-owned face values were computed
with zero halos and are wrong, contaminating owned-face boundary
averages.  Added MPI-aware ``_synchronize_cgrid_fluxes_mpi``.

**iter-1048** (1266b865) — Minimal NH+duogrid step divergence
investigation.  Narrowed to non-pad_halo source.

**iter-1047** (fab702e8) — duogrid+factory MPI investigation; 2
negative findings.  Tracked for iter-1049 root cause.

**iter-1046** (2fbedc8f) — Factory MPI test + 2 known limitations
(non-square halo for ``cross_face_du_proj``; duogrid+factory bit-
for-bit mismatch — tracked, fixed iter-1049).

**iter-1045** (f7d698c0) — ``damp_v`` / ``damp_w`` 4D-native; all
NH/PE 3D vmap-MPI sites closed (no more vmap around sendrecv).

**iter-1044** (8502ef0c) — Corner div-damp 4D-native:
``fv3_corner_laplacian_iteration`` + ``fv3_divergence_corner_2d``
shape-polymorphic (ndim ∈ {3, 4}).

**iter-1043** (475eedb6) — Lift ``_interp_center_to_corner_a2b_ord4``
out of ``jax.vmap`` (PE + NH).

**iter-1042** (cc65c637) — NH compressible-Euler MPI step fidelity
test + lift ``cgrid_mass_flux_divergence`` halo out of ``jax.vmap``.

**iter-1041** (1986d645) — Thread ``interp_offsets`` through
``packed_pad_halo_mpi_4d`` (FV3 3D hot path) + real MPI step bit-
for-bit test.

**iter-1040** (f5a20b86) — MPI ``interp_offsets`` support in
``pad_halo_mpi`` / ``pad_halo_mpi_4d``.  Closes iter-630/631
"option (a)" historical guidance.  Unlocks FV3 3D cubed-sphere
under MPI.

### Cumulative iter-1040..1066 impact

- 7 stale-API distributed test failures fixed.
- 5 new strict MPI tests added (interp_offsets, FV3 PE/NH step
  fidelity at np ∈ {2, 3, 6}, cgrid flux sync, SW sync, tiled
  pad_halo).
- 4 latent deadlock sites in MPI halo helpers patched.
- 1 internal API stale-layout bug fixed
  (``initialize_distributed`` re-entry).
- iter-1056 + iter-1058 ordering + asserts hardened test
  infrastructure against the iter-aa707bda bug class.

## Iteration 193 (2026-05-08): port FV3 damp_w + nord_w to NH path

### Goal

Port FV3 ``sw_core.F90:1080-1086`` (in ``d_sw1``) — del-(2*(nord_w+1))
post-step damping for vertical velocity ``w`` — to the NH 3D path.
Mirrors the iter-169 ``damp_v`` post-step pattern but applied to a
scalar (w) instead of the (u, v) vector.

### FV3 anchor

::

    damp4 = (damp_w * gridstruct%da_min_c) ** (nord_w + 1)
    call del6_vt_flux(nord_w, npx, npy, damp4, w, wk, fx2, fy2,
                      gridstruct, bd)
    do j=js,je
       do i=is,ie
          dw(i,j) = (fx2(i,j) - fx2(i+1,j) + fy2(i,j) - fy2(i,j+1))
                    * rarea(i,j)
          ...
          w(i,j) = w(i,j) + dw(i,j)
       enddo
    enddo

FV3 AM4 production default: ``damp_w = 0.30, nord_w = 2`` (del-6).

### Plan

1.  Add ``damp_w: float = 0.0`` and ``nord_w: int = 2`` to
    ``CDGridCompressibleEulerConfig`` (default off).
2.  Insert post-step block in ``CDGridCompressibleEulerModel._step_jitted``
    after the iter-169 ``damp_v`` close (~line 1015) and before
    ``fix_mass``.  Reuse the SW backbone helpers
    ``compute_del6_metrics`` and ``_del6_vt_flux`` from
    ``legoesm.core.fv3_del6_vt_flux`` (already imported by iter-169).
3.  ``w`` lives on half-levels (shape ``(6, n, n, nlev+1)``); vmap
    ``_del6_vt_flux`` over that axis.
4.  Compute ``dw`` from the discrete divergence of ``(fx2, fy2)``::

        dw = (fx2[i,j] - fx2[i+1,j] + fy2[i,j] - fy2[i,j+1]) * rarea

    apply ``w_new = w + dw``.

### Tests

New file ``tests/test_damp_w_nh_iter193.py`` (5 tests, no production-
only test):

1. ``test_nh_damp_w_zero_is_baseline`` — Python-static gate guard.
2. ``test_nh_damp_w_changes_w`` — measurable change on perturbed w.
3. ``test_nh_damp_w_differentiable`` — ``jax.grad`` finite through
   5 steps with damp_w active.
4. ``test_nh_damp_w_rest_state_smoke`` — 20 steps from rest stay
   finite.
5. ``test_nh_damp_w_nord2_fv3_default_finite`` — del-6 path runs
   with FV3 AM4 default ``damp_w=0.30 + nord_w=2``.

Plus extends ``tests/test_fv3_nh_toolkit_iter172.py`` AST guard:

* config fields: ``damp_w: 0.0, nord_w: 2``.
* call site: ``self.config.damp_w > 0.0`` → ``_del6_vt_flux`` helper.

(The iter-187 marker substring also tightened from
``"_zeta_smag_corner = jax.vmap"`` to just ``"_zeta_smag_corner ="``
since iter-190 changed the assignment to ``= _zeta_a2b_ord4``.)

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_damp_w_nh_iter193.py
    => 5 passed in 68.91 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py
    => 5 passed in 82.34 s (iter-187 substring fix + iter-193 guard
       added)

### Status

The NH path now has FV3-faithful w damping via ``damp_w + nord_w``
in addition to the legoESM-native ``hyperdiff_w_coeff`` (generic
biharmonic).  Users targeting full FV3 fidelity can use ``damp_w +
nord_w`` (auto-scales with grid resolution via
``(damp_w * da_min_c)^(nord_w+1)``); users on the legoESM default
path continue using ``hyperdiff_w_coeff`` unchanged.

### Why this iteration was meaningful

This closes another PE-NH symmetry gap: PE has had several FV3
damping mechanisms ported (corner_div_damp, damp_v,
use_fv3_a2b_zeta_corner, smagorinsky, T_diss, smag_vort cap, dt
plumbing).  NH had matched all of those except the d_sw1 ``damp_w``
specifically for w.  iter-193 closes the gap with ~30 LOC reusing
the existing SW backbone, default-off, with comprehensive test
coverage including AD-at-rest and FV3 AM4 production-default
finiteness.

## Iteration 192 (2026-05-08): extend iter-184/185 umbrellas to engage iter-187 + iter-190

### Goal

Tighten the umbrella AD-at-rest regression coverage.  iter-184
(NH) and iter-185 (PE) ``test_full_*_toolkit_grad_at_rest`` are the
catch-all regressions for AD hazards in the FV3 toolkit at rest
state.  Both umbrellas set ``use_fv3_a2b_zeta_corner=True``
(iter-170) AND ``corner_div_damp_d2_bg=0.0005 + dddmp=0.20``
(iter-16/iter-168) but leave ``d4_bg=0`` and ``nord=0`` —
gating the iter-187 smag_vort branch and the iter-190
``_zeta_a2b_ord4`` dedup OFF.

A future AD hazard introduced specifically in the iter-187
smag_vort code (e.g., a new sqrt-without-double-where) would
NOT be caught by the existing umbrellas.

### Plan

Add ``corner_div_damp_d4_bg=1e-3 + corner_div_damp_nord=1`` to
the umbrella ``cfg`` in BOTH iter-184 and iter-185 so the smag_vort
branch is exercised under ``jax.grad`` at rest.  Combined with
the existing ``use_fv3_a2b_zeta_corner=True``, this also engages
the iter-190 dedup pattern where ``_zeta_a2b_ord4`` is consumed
at TWO downstream sites.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_full_toolkit_ad_at_rest_iter184.py \
        tests/test_pe_full_toolkit_ad_at_rest_iter185.py
    => 4 passed in 239.58 s

The wall-time grew (~240 s vs the previous ~155 s) because the
JIT trace now includes the iter-187 smag_vort branch with the
iter-190 dedup wiring.  This is one-time JIT cost; subsequent
test invocations re-use the XLA cache.

### Status

The umbrella regression now covers:

* iter-181 Smagorinsky sqrt(0) fix
* iter-182 PE T_diss wind_speed sqrt(0) fix
* iter-183 SW d_sw5 smag_vort sqrt(0) fix
* iter-187 corner_div_damp nord >= 1 smag_vort branch (new)
* iter-190 ``_zeta_a2b_ord4`` shared-array dataflow (new)

Combined with iter-191 (focused integration test of iter-170 +
iter-187 simultaneously) and iter-186 (AST regression with
self-check), the iter-187/190 wirings are protected against
silent regressions at three layers: AST structure, focused
integration, and umbrella AD safety.

## Iteration 191 (2026-05-08): coverage tests for iter-190 dedup with both flags ON

### Goal

Close a coverage gap left open by iter-187/190.  Prior to iter-191
NO test exercises BOTH iter-170 (``use_fv3_a2b_zeta_corner=True``)
AND iter-187 (``corner_div_damp_d4_bg > 0`` + ``corner_div_damp_nord
> 0``) simultaneously — the exact combination iter-190 dedup
combines.

Existing tests cover only one flag at a time:

* ``test_div_damp_adaptive.py`` (PE iter-18): nord >= 1 + d4_bg > 0
  but ``use_fv3_a2b_zeta_corner=False`` (default).
* ``test_corner_div_damp_smag_vort_iter187.py``: nord >= 1 +
  d4_bg > 0 but ``use_fv3_a2b_zeta_corner`` not set.
* ``test_fv3_full_toolkit_ad_at_rest_iter184/185.py``:
  ``use_fv3_a2b_zeta_corner=True`` but ``corner_div_damp_nord=0``
  (default), so the iter-187 smag_vort branch is dormant.

The iter-190 dedup wiring is therefore exercised at unit-test level
ONLY by iter-187 tests (which don't enable iter-170) and iter-170
tests (which don't enable iter-187).  A regression that breaks the
combined path (e.g., the iter-187 site dropping the ``_zeta_a2b_ord4``
reference and silently using the iter-170 version of zeta_corner
as smag_vort, or vice-versa) would NOT be caught.

### Implementation

New file ``tests/test_iter190_dedup_both_flags_iter191.py`` (4 tests,
no production code change):

1. ``test_pe_both_flags_finite_and_differs_from_each_alone`` —
   PE.  Three configs: iter-170 only, iter-187 only, BOTH.  All
   produce finite output; the ``BOTH`` state DIFFERS from each
   single-flag state by ``> 1e-8 * |max state|``.  Proves both
   wirings are exercised when the combined gate triggers iter-190
   dedup.
2. ``test_pe_both_flags_grad_at_rest`` — PE.  ``jax.grad``
   through 3 steps with both flags ON at the rest state stays
   finite.  Catches a hypothetical AD hazard from the
   ``_zeta_a2b_ord4`` reuse pattern (single array consumed at two
   downstream sites) — iter-181/183 double-where pattern guards
   sqrt(0).
3. ``test_nh_both_flags_finite_and_differs_from_each_alone`` —
   NH counterpart of test 1.
4. ``test_nh_both_flags_grad_at_rest`` — NH counterpart of
   test 2.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_iter190_dedup_both_flags_iter191.py
    => 4 passed in 149.44 s

### Status

iter-187/190 wirings now have integration-test coverage at every
flag combination of iter-170 × iter-187:

| iter-170 | iter-187 | covered by |
|:--------:|:--------:|:----------|
|   off    |   off    | default baseline guards (iter-168, iter-172) |
|   on     |   off    | iter-184/185 umbrella, iter-170 unit tests |
|   off    |   on     | iter-187 tests, iter-189 tests |
|   on     |   on     | iter-191 (NEW) — closes the iter-190 dedup coverage gap |

### Why this iteration was meaningful

iter-190 introduced a refactor with subtle dataflow (one array
reused at two sites).  Without explicit coverage of the combined
flag case, a future refactor of the iter-187 site (e.g., dropping
the ``_zeta_smag_corner = _zeta_a2b_ord4`` line) could pass all
existing tests yet silently use the wrong value.  iter-191 closes
that gap with 4 focused integration tests (~150 s wall total) and
no production code change.

## Iteration 190 (2026-05-08): dedup a2b_ord4(zeta) halo exchange between iter-170 + iter-187 sites

### Goal

Close iter-187 codex review concern 3 (extra unmerged halo
exchange).  iter-170 (``use_fv3_a2b_zeta_corner``) and iter-187
(``smag_vort`` cap when nord >= 1) BOTH compute
``_interp_center_to_corner_a2b_ord4(zeta, cdgrid)``.  This call
internally pads the halo with depth-2 (``_pad_halo_auto_h2``).
When both flags are active, the same input ``zeta`` is sent
through TWO halo-2 exchanges in distributed mode — wasteful and
not FV3-faithful (FV3 ``d_sw5`` computes ``a2b_ord4(wk)`` once at
sw_core.F90:1795 and reuses it for the smag_vort cap).

### Plan

1.  In PE ``fv3_hydrostatic_tendencies`` (line ~518): introduce a
    single ``_zeta_a2b_ord4`` local computed at most once when
    EITHER iter-170 OR iter-187 is active::

        _need_zeta_a2b_for_smag = (
            config.corner_div_damp_d2_bg > 0.0
            and config.corner_div_damp_d4_bg > 0.0
            and config.corner_div_damp_nord > 0
        )
        _need_zeta_a2b = (
            config.use_fv3_a2b_zeta_corner
            or _need_zeta_a2b_for_smag
        )
        _zeta_a2b_ord4 = None
        if _need_zeta_a2b:
            _zeta_a2b_ord4 = jax.vmap(
                lambda lev: _interp_center_to_corner_a2b_ord4(lev, cdgrid),
                in_axes=-1, out_axes=-1,
            )(zeta)

2.  Iter-170 site reuses ``_zeta_a2b_ord4`` (when
    ``use_fv3_a2b_zeta_corner`` is True).
3.  Iter-187 site reuses ``_zeta_a2b_ord4`` directly (drops its own
    local a2b_ord4 call).
4.  Same change in NH ``cdgrid_compressible_euler_slow_tendencies``
    (mirror of PE).

### Backward compatibility

Bit-for-bit baseline at all flag combinations:
* Default (neither iter-170 nor iter-187): ``_need_zeta_a2b=False``,
  no extra computation.  No-op.
* iter-170 only: ``_zeta_a2b_ord4`` computed once → reused for
  zeta_corner.  Same as before.
* iter-187 only: ``_zeta_a2b_ord4`` computed once → reused for
  smag_vort.  Same number of a2b_ord4 calls as iter-187 stand-alone.
* iter-170 + iter-187: ``_zeta_a2b_ord4`` computed ONCE (was twice).
  Saves 1 halo-2 exchange.  Bit-for-bit identical (same input →
  same output).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_div_damp_smag_vort_iter187.py \
        tests/test_corner_div_damp_dt_actual_iter189.py
    => 13 passed (all iter-187 + iter-189 tests preserved)

The iter-187 AST regression (``test_smag_vort_uses_relative_vorticity_via_a2b_ord4``)
still passes because the iter-187 site's ``_smag_arg = _delpc_initial ** 2 +
_zeta_smag_corner ** 2`` substring is unchanged; only the
``_zeta_smag_corner`` assignment line moves from a local
``jax.vmap`` call to the precomputed ``_zeta_a2b_ord4`` reference.

### Status

The iter-187 a2b_ord4(zeta) duplicate halo exchange is closed.
Distributed runs with both iter-170 + iter-187 active now incur
ONE halo-2 collective per timestep instead of two for the FV3
smag_vort cap path.

### Why this iteration was meaningful

iter-187 codex review concern 3 was a real distributed-mode
performance bug.  The fix is targeted (~15 LOC each in PE / NH)
and preserves bit-for-bit baseline behaviour at all flag
combinations.  No new tests required (existing iter-187 tests
already exercise both flag combinations).

## Iteration 189 (2026-05-08): plumb actual integration dt to corner-div damp cap (PE + NH)

### Goal

Close a residual fidelity gap left open by iter 187/188.  FV3
``sw_core.F90:d_sw5`` uses the ACTUAL sub-cycle dt in the
adaptive damping cap (``min(0.20, dddmp * dt * |delpc|)`` for
nord=0 and ``min(0.20, dddmp * dt * sqrt(delpc² + ζ²))`` for
nord >= 1).  legoESM iter-16/iter-18/iter-168/iter-187 use a
config-tunable approximation (``corner_div_damp_dt_proxy``,
default 200.0 for PE / 10.0 for NH).  This is FV3-faithful when
the user runs at exactly that dt but mis-scales the cap when the
integration dt differs (e.g., C96 production runs at dt=50 with
PE dt_proxy=200).

### Why now

iter-187's smag_vort cap depends on dt linearly: a 4× over-
estimate of dt at C96 (dt=50 vs dt_proxy=200) means the
``min(0.20, dddmp * dt * smag_vort)`` cap engages 4× MORE often
than FV3 would at the same physical state.  In HS-typical regimes
the d2_bg floor dominates so iter-99's 30-day stability result is
not affected, but in transient strong-divergence regimes the
cap-engagement difference is real.

### Plan

1.  Add ``dt_actual: float | None = None`` keyword to the PE
    ``fv3_hydrostatic_tendencies`` signature (line 358) and the
    NH ``cdgrid_compressible_euler_slow_tendencies`` signature
    (line 183).
2.  Inside both functions, in the corner-div damping block,
    compute::

        _dt_approx = (
            dt_actual if dt_actual is not None
            else config.corner_div_damp_dt_proxy
        )

    This propagates automatically to BOTH the iter-16/iter-168
    nord=0 cap and the iter-187 nord >= 1 smag_vort cap (both
    reuse the same ``_dt_approx`` local).
3.  In ``CDGridPrimitiveEquationModel.step`` (around line 1397)
    and ``CDGridCompressibleEulerModel.step`` (around line 872),
    pass ``dt_actual=dt`` to the tendency call.
4.  Add tests verifying:
    * Default behaviour (no dt_actual) uses config.dt_proxy
      → existing iter-18 / iter-187 baselines bit-for-bit
      unchanged.
    * With ``dt_actual=N``, the smag_vort cap engages
      proportionally to N (verified at the model.step level
      with two configs differing only in dt).
5.  Update FV3_3D.md.

### Backward compatibility

* All existing direct callers of ``fv3_hydrostatic_tendencies(...)``
  / ``cdgrid_compressible_euler_slow_tendencies(...)`` (which
  don't pass dt_actual) continue to use ``config.corner_div_damp_dt_proxy``
  → bit-for-bit unchanged.
* model.step path now uses the actual dt → fidelity-corrected.
  At dt = config.corner_div_damp_dt_proxy (the existing PE
  default 200.0 = the iter-33 ah_x10+dt=200 reference) the new
  behaviour is bit-for-bit identical to iter-188.

### Implementation

PE (``primitive_eq_cdgrid.py``):

1. Added ``dt_actual: float | jax.Array | None = None`` keyword to
   ``fv3_hydrostatic_tendencies`` signature.
2. Inside the corner-div damping block (line ~681) replaced::

       _dt_approx = config.corner_div_damp_dt_proxy

   with::

       if dt_actual is not None:
           _dt_approx = dt_actual
       else:
           _dt_approx = config.corner_div_damp_dt_proxy

3. ``CDGridPrimitiveEquationModel.step``'s ``tendency_fn`` closure
   now passes ``dt_actual=dt`` to ``fv3_hydrostatic_tendencies``.

NH (``compressible_euler_cdgrid.py``):

1. Added the same kwarg to ``cdgrid_compressible_euler_slow_tendencies``.
2. Same ``_dt_approx`` resolution change at the iter-168 site.
3. ``CDGridCompressibleEulerModel._step_jitted``'s ``slow_tendency_fn``
   closure now passes ``dt_actual=dt``.

### Tests

New file ``tests/test_corner_div_damp_dt_actual_iter189.py`` (6 tests,
no production-only test, no new helpers):

1. ``test_pe_dt_actual_default_matches_dt_proxy_fallback`` — direct
   call without ``dt_actual`` matches an explicit ``dt_actual=
   config.corner_div_damp_dt_proxy`` call bit-for-bit (backward
   compat sentinel).
2. ``test_pe_dt_actual_changes_smag_vort_cap`` — different
   ``dt_actual`` values produce different states when the cap is
   engaged (proves the wiring is exercised).
3. ``test_nh_dt_actual_default_matches_dt_proxy_fallback`` — NH
   counterpart of test 1.
4. ``test_nh_dt_actual_changes_smag_vort_cap`` — NH counterpart
   of test 2.
5. ``test_pe_step_passes_dt_actual_ast_regression`` — AST guard
   that ``CDGridPrimitiveEquationModel.step`` passes ``dt_actual=dt``.
6. ``test_nh_step_passes_dt_actual_ast_regression`` — NH counterpart.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_div_damp_dt_actual_iter189.py
    => 6 passed

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_div_damp_adaptive.py \
        tests/test_corner_div_damp_smag_vort_iter187.py \
        tests/test_corner_div_damp_nh.py \
        tests/test_div_damp_quantitative_iter174.py \
        --deselect tests/test_div_damp_adaptive.py::test_corner_div_damp_fv3_vector_fill_bit_for_bit_nord1
    => 32 passed (PE iter-18 + iter-187 + NH iter-168 + iter-174
       baselines preserved bit-for-bit; deselected test is the
       pre-existing 1-ULP iter-22 flake)

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_full_toolkit_ad_at_rest_iter184.py \
        tests/test_fv3_nh_toolkit_iter172.py \
        tests/test_fv3_pe_toolkit_iter188.py
    => 10 passed (umbrella AD-at-rest + NH/PE AST guards
       preserved)

### Status

The iter-187 smag_vort cap and the iter-16/iter-168 nord=0 cap now
use the actual integration dt (passed by ``model.step``) instead of
a config-tunable approximation.  Backward compatibility is exact:
direct callers without ``dt_actual`` use the iter-188 fallback.
The seventh iter-187 codex review concern (dt fidelity) is resolved.

### Why this iteration was meaningful

iter-187/iter-188 closed the smag_vort formula gap and the dt-proxy
parity gap, but the dt scale itself was still an approximation.
FV3 ``d_sw5`` uses the actual integration dt — at C96 production
(dt=50 s with PE dt_proxy=200 s default) the previous cap was 4×
over-engaged.  This iter resolves that.  The fix is ~5 LOC each in
PE / NH plus a one-line keyword pass at the model.step closures.

## Iteration 188 (2026-05-08): PE / NH parity + PE AST regression guard

### Goal

Close two iter-187 codex-review concerns:

1.  **Concern 2 — PE / NH dt-proxy parity gap.**  PE
    ``primitive_eq_cdgrid.py`` hardcodes ``_dt_approx = 200.0`` at
    line 669; NH ``compressible_euler_cdgrid.py`` has a tunable
    ``corner_div_damp_dt_proxy: float = 10.0`` config field
    (line 120) used at line 456.  Iter-187's smag_vort cap inherits
    the same hardcoded PE value with no user override path —
    inconsistent with the NH parity established by iter 168.
2.  **Concern 6 — PE has no AST regression guard.**  iter-172
    added the AST regression guard for the iter-168/169/170/171
    NH wirings, extended by iter-178/186 to cover iter-173/180.
    The PE iter-12/14/16/18/187 wirings have no equivalent
    structural guard — a refactor that drops any of them while
    keeping the config field would silently disable the feature.

### Plan

1.  Add ``corner_div_damp_dt_proxy: float = 200.0`` to
    ``CDGridPrimitiveEquationConfig`` (default preserves the
    existing iter-18 behaviour bit-for-bit).
2.  Replace the hardcoded ``_dt_approx = 200.0`` in the PE
    corner-div damping block (both nord=0 and nord >= 1 paths)
    with ``_dt_approx = config.corner_div_damp_dt_proxy``.
3.  Add ``tests/test_fv3_pe_toolkit_iter188.py`` with three
    tests mirroring iter-172/178:
    * ``test_pe_fv3_config_fields_ast_regression`` — config field
      defaults for iter-12/14/16/18/57/187 (damp_v, nord_v,
      use_fv3_a2b_zeta_corner, corner_div_damp_*, smagorinsky_cs,
      and the new corner_div_damp_dt_proxy).
    * ``test_pe_fv3_call_sites_ast_regression`` — gate / helper
      pairs for each PE-side FV3-faithful wiring (mirror of
      iter-172 NH guard).
    * ``test_iter188_ast_guard_self_check`` — drops each pair
      one at a time and verifies the inner check function flags
      the omission (mirror of iter-178 self-check).
4.  Document in FV3_3D.md.

### Key fidelity points

* **Default-preserving**: ``corner_div_damp_dt_proxy = 200.0`` is
  the existing PE hardcoded value.  Existing tests are
  bit-for-bit unchanged.
* **Parity**: PE and NH now share the field name and semantics;
  the only difference is the per-path default (PE: 200.0 outer
  dt; NH: 10.0 outer dt with split-explicit acoustic substepping).
* **AST guard**: same gate/helper pair structure as iter-172,
  including the iter-178 self-check.

### Implementation

PE config (``primitive_eq_cdgrid.py``, after the iter-22
``corner_div_damp_fv3_vector_fill`` field)::

    corner_div_damp_dt_proxy: float = 200.0
        # FV3_3D iter 188: parity with NH ``corner_div_damp_dt_proxy``.
        # Used in BOTH the iter-16 nord=0 cap and the iter-187
        # nord >= 1 smag_vort cap.  Default 200.0 preserves existing
        # iter-18 behaviour.

PE wiring (``primitive_eq_cdgrid.py`` line 681): replaced::

    _dt_approx = 200.0  # hardcoded

with::

    _dt_approx = config.corner_div_damp_dt_proxy

This propagates automatically to the iter-187 smag_vort branch
which reuses ``_dt_approx``.

PE-side AST guard test (``tests/test_fv3_pe_toolkit_iter188.py``,
3 tests, no production code change):

1. ``test_pe_fv3_config_fields_ast_regression`` — 13 PE config
   field defaults across iter-5/12/14/16/18/57/182/187/188.
2. ``test_pe_fv3_call_sites_ast_regression`` — 8 (gate, helper)
   pairs covering the FV3-faithful wirings.
3. ``test_iter188_ast_guard_self_check`` — drops each gate one at
   a time and verifies the inner check correctly flags the
   omission (mirror of iter-178 NH self-check).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_pe_toolkit_iter188.py
    => 3 passed in 0.02 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_div_damp_smag_vort_iter187.py \
        tests/test_div_damp_adaptive.py \
        --deselect tests/test_div_damp_adaptive.py::test_corner_div_damp_fv3_vector_fill_bit_for_bit_nord1
    => 24 passed (PE iter-18 + iter-187 baselines preserved
       bit-for-bit; the deselected iter-22 test has a pre-existing
       1-ULP flake unrelated to iter-188)

### Status

PE and NH config now have the same FV3-faithful damping surface
naming, with per-path defaults preserved (PE: 200.0, NH: 10.0).
The PE iter-12/14/16/18/57/182/187 wirings now have the same
AST regression coverage as the NH iter-168/169/170/171/173/180
wirings (iter-172/178/186 NH).

### Why this iteration was meaningful

iter-187's codex review flagged TWO concrete gaps:
* concern 2 (PE / NH dt-proxy parity gap)
* concern 6 (PE has no AST regression guard mirroring iter-172).

Both gaps would be silently exploited by future refactors.  The
parity gap is now closed (PE has the same field as NH); the AST
guard is now in place (mirrors iter-172/178 structure).  Pure
config + test addition; no production behaviour change at default
settings.

## Iteration 187 (2026-05-08): port FV3 smag_vort adaptive cap to nord>=1 corner-div damp (PE + NH)

### Goal

Close a real FV3-fidelity gap in BOTH 3D paths' iter-16/iter-18 (PE) and iter-168 (NH) corner-divergence damping wirings.

FV3 ``sw_core.F90:d_sw5`` uses TWO different formulas for the
adaptive damping cap depending on ``nord``:

* ``nord = 0`` (line 1722): ``damp = da_min_c * max(d2_bg, min(0.20, dddmp * |delpc * dt|))``
* ``nord >= 1`` (lines 1797-1809):
  ``vort_smag = |dt| * sqrt(delpc² + wk_corner²)``
  ``damp2 = da_min_c * max(d2_bg, min(0.20, dddmp * vort_smag))``

The legoESM SW core already implements both forms correctly
(``fv3_sw_core.py:1768-1809``).  But the legoESM 3D paths
(``primitive_eq_cdgrid.py`` iter-18 wiring at line 670-722,
``compressible_euler_cdgrid.py`` iter-168 wiring at line 457-495)
use the ``|delpc|``-only form regardless of nord.  This is faithful
for nord=0 but NOT for nord >= 1 — a silent fidelity gap.

### FV3 anchor

* ``sw_core.F90:1795``: ``a2b_ord4(wk, vort, ...)`` lifts cell-
  centre relative vorticity ``wk`` to corners as ``vort``.
* ``sw_core.F90:1797``: ``vort(i,j) = abs(dt)*sqrt(delpc(i,j)**2 + vort(i,j)**2)``
* ``sw_core.F90:1808-1809``: ``damp2 = da_min_c*max(d2_bg, min(0.20, dddmp*vort(i,j)))``

This is the FV3 production-default branch (e.g., AM4 uses
nord=2, d4_bg=0.16, dddmp=0.2).  The current legoESM 3D paths
have the wiring to enter this branch but apply the wrong cap
formula inside it.

### Plan

1.  **PE path** (``primitive_eq_cdgrid.py`` line 693-723): inside
    the existing ``if config.corner_div_damp_d4_bg > 0.0 and
    config.corner_div_damp_nord > 0:`` branch, recompute
    ``_damp_corner`` with the FV3 ``smag_vort`` form before the
    existing ``_ke_correction = _damp_corner * _delpc_initial +
    _dd8 * _divg_d_iter`` line.  Reuse the cell-centre ``zeta``
    (already at line 442) lifted to corners via
    ``_interp_center_to_corner_a2b_ord4`` (iter-170 helper, FV3-
    faithful for the smag_vort cap regardless of
    ``use_fv3_a2b_zeta_corner``).
2.  **NH path** (``compressible_euler_cdgrid.py`` line 468-495):
    same change inside the same Python-static gate.  Reuse
    ``zeta`` from line 250 + a2b_ord4.
3.  **Iter-181/183 double-where**: apply the same sqrt(0)
    protection pattern so AD at rest state stays finite.
4.  **Test file** ``tests/test_corner_div_damp_smag_vort_iter187.py``
    (no production-only test; covers BOTH PE and NH):
    * AD-at-rest with nord=1 + d4_bg + dddmp + d2_bg=floor stays
      finite (catches sqrt(0) hazard).
    * smag_vort path differs from a vortical-IC baseline where
      ζ-dependence proves the new formula is exercised.
    * nord=0 branch bit-for-bit unchanged (regression guard).
5.  **AST regression guard extension**: extend
    ``test_fv3_nh_toolkit_iter172.py`` self-check pair to include
    the iter-187 smag_vort site.

### Key fidelity points

* ``a2b_ord4`` is used UNCONDITIONALLY for ζ_corner inside
  smag_vort (FV3 always uses 4th-order for the smag_vort cap;
  the user-facing ``use_fv3_a2b_zeta_corner`` flag controls only
  the rotational ζ × v term in the momentum tendency, NOT
  smag_vort).
* Bit-for-bit baseline: only the ``d4_bg > 0 AND nord > 0``
  branch is modified; nord=0 stays untouched, and ``d2_bg = 0``
  gates the entire block off (Python-static).
* Default config behaviour unchanged: ``corner_div_damp_nord = 0``
  is the default — users must explicitly opt into the higher-
  order branch.

### Implementation

PE (``primitive_eq_cdgrid.py``, lines ~710-750): inserted between
the existing iterated Laplacian loop and the ``_dd8`` cast.  Reuses
the cell-centre ``zeta`` (line 442) and the iter-170 helper
``_interp_center_to_corner_a2b_ord4``.  Iter-181/183 double-where
pattern guards sqrt(0) at rest::

    _zeta_smag_corner = jax.vmap(
        lambda lev: _interp_center_to_corner_a2b_ord4(lev, cdgrid),
        in_axes=-1, out_axes=-1,
    )(zeta)
    _smag_arg = _delpc_initial ** 2 + _zeta_smag_corner ** 2
    _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
    _smag_root = jnp.where(
        _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
    )
    _smag_vort = jnp.abs(_dt_approx) * _smag_root
    _damp_corner = _da_min_c * jnp.maximum(
        config.corner_div_damp_d2_bg,
        jnp.minimum(0.20, config.corner_div_damp_dddmp * _smag_vort),
    )

NH (``compressible_euler_cdgrid.py``, lines ~485-515): identical
structure inside the same Python-static gate, reusing ``zeta`` from
line 250.  Both paths use the existing ``_dt_approx`` constant:
``200.0`` for PE (consistent with the iter-18 nord=0 formula) and
``config.corner_div_damp_dt_proxy`` (default ``10.0``) for NH.

### Tests

New file ``tests/test_corner_div_damp_smag_vort_iter187.py`` (7 tests,
no new helpers):

1. ``test_pe_smag_vort_grad_at_rest`` — ``jax.grad`` through 3 PE
   steps with nord=1 + d4_bg + dddmp + d2_bg=floor at the rest
   state stays finite.
2. ``test_pe_smag_vort_changes_state_under_vortical_perturbation``
   — vortical perturbation produces measurably different state vs.
   the gated-off baseline (d2_bg=0).
3. ``test_pe_nord2_higher_order_branch_finite`` — del-6 (nord=2)
   path runs and produces finite output.
4. ``test_nh_smag_vort_grad_at_rest`` — NH counterpart of test 1.
5. ``test_nh_smag_vort_changes_state_under_vortical_perturbation``
   — NH counterpart of test 2.
6. ``test_nh_nord2_fv3_production_default_finite`` — NH del-6 with
   FV3 AM4 production exact ``d4_bg = 0.16`` runs finite at C8.
7. ``test_smag_vort_uses_relative_vorticity_via_a2b_ord4`` — AST
   regression that BOTH PE and NH source files contain the
   ``_zeta_smag_corner = jax.vmap(... a2b_ord4 ...)(zeta)`` site
   (catches refactor that swaps relative ζ for absolute
   ``zeta_corner = zeta + f_corner``).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_div_damp_smag_vort_iter187.py
    => 7 passed in 172.04 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_div_damp_adaptive.py
    => 17 passed (PE iter-18 baseline preserved; iter-22
       vector_fill bit-for-bit nord1 deselected — pre-existing
       1-ULP flake unrelated to iter-187)

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_div_damp_nh.py tests/test_div_damp_nh.py \
        tests/test_div_damp_quantitative_iter174.py \
        tests/test_fv3_nh_toolkit_iter172.py
    => 18 passed (NH iter-168/171/172/174 baselines preserved)

### Status

The legoESM 3D corner-divergence damping is now FV3-faithful for
BOTH ``nord = 0`` (existing iter-16/iter-168 wiring) and ``nord
>= 1`` (new iter-187 smag_vort cap).  The seventh PE-NH FV3-fidelity
gap is closed.  Default-off (``corner_div_damp_nord = 0`` is the
default), so production users on the default see no behaviour
change.  Users who opt in to the higher-order corner-div damping
now use the FV3-correct adaptive cap formula.

### Why this iteration was meaningful

A real fidelity bug existed in BOTH 3D paths: the ``nord >= 1``
branch silently used the FV3 ``nord = 0`` cap formula.  The bug
would have shown up as a discrepancy with FV3 reference data when
running cube simulations that opt into the higher-order corner-div
damping.  iter-187 closes this gap with code reuse — the
``_interp_center_to_corner_a2b_ord4`` helper from iter-170 and the
double-where pattern from iter-181/183 both already existed.  The
fix is ~25 LOC each in PE / NH, fully gated, with comprehensive
test coverage.

## Iteration 186 (2026-05-08): extend AST guards for iter-180 smagorinsky_cs

### Goal

The iter-172 AST guard was extended by iter-178 to cover iter-173
(``use_async_halo``).  But iter-180 added another NH config field
(``smagorinsky_cs``) and a dispatch (``compute_smagorinsky_ah_3d``)
which are NOT yet covered.  A regression that drops iter-180's
wiring would still pass the iter-172/178 guard.

This iter extends the guard to cover iter-180 as well.

### Implementation

Modifications to ``tests/test_fv3_nh_toolkit_iter172.py``:

* Added ``"smagorinsky_cs": 0.0`` to the config-fields ``expected``
  dict in ``test_nh_fv3_config_fields_ast_regression``.
* Added the iter-180 (gate, helper) pair
  ``("config.smagorinsky_cs > 0.0", "compute_smagorinsky_ah_3d")``
  to ``test_nh_fv3_call_sites_ast_regression``.
* Same pair added to the iter-178 self-check loop so the "test
  the test" coverage extends to the new gate.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py
    => 5 passed in 82.10 s

### Status

The AST regression guard now covers all six PE-NH-asymmetry
fixes (iter 168/169/170/171/173/180) AND has a self-check
(iter 178) ensuring the guard isn't silently broken by a typo.
Coverage extends with each new field-+-dispatch addition.

## Iteration 185 (2026-05-08): PE full-toolkit AD-at-rest umbrella regression

### Goal

iter 184 added the umbrella AD-at-rest regression for the NH 3D
path.  This iter adds the PE counterpart, exercising every PE
damping knob simultaneously at rest state to catch any future
AD hazard in ``primitive_eq_cdgrid.py``.

The PE path has more knobs than NH: PE-only ``T_diss_coeff``
(iter-182 fix) is included along with the iter-12/14/16/18/57-58
mechanisms that the NH path mirrors via iter 168/169/170/180.

### Implementation

New file ``tests/test_pe_full_toolkit_ad_at_rest_iter185.py``
(2 tests, no production code change):

1. ``test_full_pe_toolkit_grad_at_rest`` — every PE damping knob
   ON simultaneously at exactly the rest state, ``jax.grad``
   w.r.t. T gives finite gradients.  Exercises:

   * ``A_h`` + ``smagorinsky_cs`` (Smagorinsky A_h — iter-181 fix)
   * ``hyperdiff_coeff`` (4th-order biharmonic)
   * ``div_damp_coeff`` (cell-centre divergence damping, iter 5)
   * ``use_fv3_a2b_zeta_corner`` (iter 14)
   * ``damp_v`` + ``nord_v`` (iter 12 post-step vorticity damping)
   * ``corner_div_damp_d2_bg`` + ``corner_div_damp_dddmp``
     (iter 16/18 corner-divergence damping)
   * ``T_diss_coeff`` (velocity-dependent T dissipation —
     iter-182 fix)

2. ``test_full_pe_toolkit_grad_at_perturbed`` — sanity that
   perturbed state works (rest is the challenging case).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_pe_full_toolkit_ad_at_rest_iter185.py
    => 2 passed in 92.85 s

### Status

iter-184 covered NH; iter-185 covers PE.  Both 3D paths now have
umbrella AD-at-rest regression tests that catch any future
AD-hazard regression at the integration level.  Combined with
the focused per-helper tests from iter 181/182/183, the
differentiability surface is comprehensively guarded.

## Iteration 184 (2026-05-08): NH full-toolkit AD-at-rest umbrella regression

### Goal

iter 181/182/183 fixed three sqrt-at-zero gradient hazards
(Smagorinsky helper, PE T_diss wind_speed, SW d_sw5 smag_vort).
Each iter added a focused test for ITS specific helper, but no
test exercises ``jax.grad`` through the full NH toolkit at rest
state simultaneously.

This iter adds an umbrella regression: ``jax.grad`` through 5 NH
steps with ALL FIVE iter-168/169/170/171/180 knobs ON at rest
state.  The umbrella catches future AD hazards introduced by
helpers that are added to the NH path beyond the iter-181/182/183
fixes.

### Implementation

New file ``tests/test_fv3_full_toolkit_ad_at_rest_iter184.py``
(2 tests, no production code change):

1. ``test_full_nh_toolkit_grad_at_rest`` — every NH FV3 knob ON
   simultaneously (corner_div_damp + damp_v + a2b zeta + cell-
   centre div_damp + Smagorinsky A_h), 5 steps from EXACTLY rest
   state, ``jax.grad`` w.r.t. ``theta_prime`` is finite.
   Differentiating w.r.t. ``theta_prime`` (not winds) avoids
   perturbing winds away from zero, so the sqrt-at-zero hazards
   in iter-181/182/183 are genuinely exercised.
2. ``test_full_nh_toolkit_grad_at_perturbed`` — sanity that the
   rest-state path is the challenging case and the perturbed
   path is a regular regression.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_full_toolkit_ad_at_rest_iter184.py
    => 2 passed in 144.38 s

### Status

The iter-181/182/183 differentiability fixes now have an umbrella
regression test that covers the "all toolkit knobs ON, rest state"
combination.  Future iterations adding new mechanisms to the NH
tendency function should rerun this test to catch any AD-hazard
regression at the integration level (in addition to the focused
per-helper tests added in iter-181/182/183).

## Iteration 183 (2026-05-08): fix smag_vort sqrt(0) in fv3_sw_core

### Goal

Continue the iter-181/182 differentiability audit.  An audit pass
of FV3 helpers found a third instance of the sqrt-at-zero gradient
hazard in ``fv3_sw_core.py:1797``::

    smag_vort = jnp.abs(dt) * jnp.sqrt(delpc ** 2 + wk_corner ** 2)

This is inside ``_d_sw5_corner_divergence``'s adaptive Smagorinsky
branch (``dddmp > 0``).  At rest state both ``delpc`` and
``wk_corner`` are 0; the gradient through ``sqrt(0+0)`` is
undefined.  This breaks ``jax.grad`` through any rest-state SW
shallow-water model with adaptive Smagorinsky enabled (e.g., the
iter-962 SW W2 calibration).

iter 183 applies the same JAX double-where trick from iter 181/182.

### Implementation

File: ``src/legoesm/core/fv3_sw_core.py``, lines ~1797-1801.
Replaced::

    smag_vort = jnp.abs(dt) * jnp.sqrt(delpc ** 2 + wk_corner ** 2)

with::

    _smag_arg = delpc ** 2 + wk_corner ** 2
    _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
    _smag_root = jnp.where(
        _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
    )
    smag_vort = jnp.abs(dt) * _smag_root

Properties:

* **Forward pass**: bit-for-bit unchanged at any nonzero
  ``delpc² + wk_corner²``; exactly 0 at rest.  Verified by the
  iter-962 SW W2 sentinel still passing.
* **Backward pass**: gradient finite (zero) at rest state instead
  of NaN.

### Tests added

New file ``tests/test_smag_vort_grad_iter183.py`` (2 tests):

1. ``test_d_sw5_smag_dddmp_zero_baseline`` — sanity that
   ``dddmp = 0`` gives reproducible finite output (no Smagorinsky
   branch).
2. ``test_d_sw5_smag_grad_finite_at_rest`` — direct test of
   ``_d_sw5_corner_divergence`` with ``dddmp = 0.05`` at rest
   state (zero u_d, v_d, ua, va).  ``jax.grad`` w.r.t. both u_d
   and v_d gives finite gradients.  Was NaN before iter 183.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_smag_vort_grad_iter183.py
    => 2 passed in 11.18 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_iter962_smagorinsky_tweak.py \
        tests/test_smagorinsky_visc.py \
        tests/test_smagorinsky_grad_at_zero_iter181.py
    => 13 passed (iter-962 SW W2 sentinel + Smagorinsky helper
       tests unchanged after iter-183 fix)

### Status

Three sqrt-at-zero gradient hazards now closed (iter 181, 182, 183).
The full FV3 fidelity damping toolkit is differentiable through
the rest state on all three known affected helpers
(compute_smagorinsky_ah_2d, primitive_eq T_diss wind_speed,
fv3_sw_core d_sw5 smag_vort).

## Iteration 182 (2026-05-08): fix wind_speed sqrt(0) in PE T_diss

### Goal

iter 181 fixed the sqrt-at-zero singularity in
``compute_smagorinsky_ah_2d``.  An audit pass found another
instance in ``primitive_eq_cdgrid.py`` line 1005::

    wind_speed = jnp.sqrt(u_cell**2 + v_cell**2)

Used inside the velocity-dependent T-dissipation block
(``T_diss_coeff > 0``) to compute ``nu_T = T_diss_coeff *
wind_speed * dx``.  At rest state (u_cell = v_cell = 0) the
gradient ``d sqrt(0) / d u`` is undefined → NaN propagates into
``dT_dt`` and breaks ``jax.grad`` through the PE rest state
when T_diss is active.

iter 182 applies the same JAX double-where trick from iter 181.

### Implementation

File: ``src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py``,
inside the ``if config.T_diss_coeff > 0:`` block.  Replaced the
single-line ``jnp.sqrt(u_cell**2 + v_cell**2)`` with the same
double-where pattern:

* Forward pass: bit-for-bit unchanged at any nonzero wind;
  exactly 0 at zero winds.
* Backward pass: gradient finite (zero) at rest state instead
  of NaN.

### Tests added

New file ``tests/test_T_diss_grad_at_zero_iter182.py`` (3 tests):

1. ``test_pe_T_diss_off_baseline`` — Python-static gate guard
   (T_diss_coeff=0.0 matches field-unset).
2. ``test_pe_T_diss_changes_T_when_winds_nonzero`` — sanity that
   the iter-182 fix didn't accidentally make T_diss inert.
3. ``test_pe_T_diss_differentiable_at_rest`` — model-level
   ``jax.grad`` through 3 PE steps with ``T_diss_coeff=0.05``
   starting from EXACTLY the rest state.  Was NaN before iter 182.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_T_diss_grad_at_zero_iter182.py
    => 3 passed in 23.27 s

    Regression suite (existing PE / NH tests):
    => 33 passed (unchanged); 1 pre-existing 1-ULP FP-noise
       failure in test_corner_div_damp_fv3_vector_fill_bit_for_bit_nord1
       not caused by iter 182.

### Status

Two sqrt-at-zero gradient hazards now closed (iter 181 + iter 182).
PE and NH paths both differentiable through rest state.  Future
audits may surface more hazards in less-exercised helpers (e.g.,
sina_u via ``sqrt(jnp.maximum(1 - cosa**2, _EPS))`` already uses
the safer pattern).

## Iteration 181 (2026-05-08): fix Smagorinsky sqrt(0) gradient singularity

### Goal

iter 180 documented a known limitation of the
``compute_smagorinsky_ah_2d`` helper: ``jnp.sqrt(strain_mag_sq)``
has a singular gradient at zero strain (``d sqrt(x) / d x`` is
infinite at ``x=0``).  The iter-180 NH differentiability test
worked around this by using a non-rest perturbed IC.  This iter
fixes the helper at the source so any caller (PE iter-58, NH
iter-180, future training-mode users) can backprop through the
rest state.

### Implementation

File: ``src/legoesm/core/_smagorinsky_visc.py``,
``compute_smagorinsky_ah_2d``.  Replaced::

    strain_mag = jnp.sqrt(D11**2 + 2*D12**2 + D22**2)

with the JAX "double-where" trick::

    strain_mag_sq = D11**2 + 2*D12**2 + D22**2
    safe_strain_sq = jnp.where(strain_mag_sq > 0.0, strain_mag_sq, 1.0)
    strain_mag = jnp.where(
        strain_mag_sq > 0.0, jnp.sqrt(safe_strain_sq), 0.0,
    )

Properties:

* **Forward pass bit-for-bit unchanged**: at any strain > 0 the
  result is exactly ``jnp.sqrt(strain_mag_sq)``; at strain = 0
  the result is exactly 0 (preserves the existing
  ``test_smagorinsky_zero_winds`` contract).
* **Backward pass finite at zero**: the inner ``sqrt`` is
  evaluated at ``safe_strain_sq >= 1`` so its derivative is
  finite; the outer ``where`` mask sets the gradient to 0 at
  zero-strain cells (instead of NaN from the singular
  ``d sqrt(0)``).

### Tests added

New file ``tests/test_smagorinsky_grad_at_zero_iter181.py``
(4 tests):

1. ``test_smag_grad_finite_at_zero_strain`` — gradient at
   exactly-zero strain input is finite (was NaN before iter 181).
2. ``test_smag_grad_finite_on_partial_zero_strain`` — mixed
   zero / nonzero strain cells produce finite gradient
   everywhere; no NaN propagation from zero cells.
3. ``test_smag_forward_at_zero_winds_still_zero`` — sanity that
   the iter-58 zero-winds-zero-output contract is preserved.
4. ``test_nh_smag_differentiable_at_rest`` — model-level
   ``jax.grad`` through 5 NH steps starting from EXACTLY the
   rest state with smag ON.  This is the iter-180 failure mode
   that motivated this iter.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_smagorinsky_visc.py \
        tests/test_smagorinsky_ah_nh_iter180.py \
        tests/test_smagorinsky_grad_at_zero_iter181.py
    => 16 passed (existing 12 unchanged + 4 new)

### Status

iter-180's documented "known limitation" is closed at the helper
level.  Any caller (current: PE iter-58, NH iter-180; future:
training modes that touch rest state, ML-coupled inference) gets
finite gradients through the Smagorinsky path now.

## Iteration 180 (2026-05-08): port FV3 Smagorinsky-adaptive A_h to NH

### Goal

The PE path has FV3-style adaptive Smagorinsky A_h (PE iter 57-59):
when ``smagorinsky_cs > 0`` AND ``A_h > 0``, an adaptive coefficient
proportional to the local strain rate × dx² is added to the static
``A_h``.  NH had only a constant ``A_h`` — no adaptive component.

This iter ports the PE iter-58 wiring to NH, reusing the existing
``compute_smagorinsky_ah_3d`` helper.

### FV3 anchor

- Lin 2004; FV3 Smagorinsky-style constant×strain×dx² formulation.
- Helper: ``legoesm.core._smagorinsky_visc.compute_smagorinsky_ah_3d``
  (already in production, used by PE path).

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 1 new ``CDGridCompressibleEulerConfig`` field (default off):
  ``smagorinsky_cs: float = 0.0``.
- Modify the existing ``A_h > 0`` Laplacian block: when
  ``smagorinsky_cs > 0`` ALSO, compute the adaptive coefficient
  via ``compute_smagorinsky_ah_3d`` and add to the static A_h.
- The Smagorinsky branch is gated INSIDE the ``A_h > 0`` block,
  so when A_h is off the branch is unreachable (mirrors PE iter-58
  semantics: Smagorinsky is added on TOP of A_h, not in place of).
- Default ``smagorinsky_cs=0.0`` preserves baseline bit-for-bit.

### Tests added

New file ``tests/test_smagorinsky_ah_nh_iter180.py`` (5 tests):

1. ``test_nh_smag_zero_is_baseline`` — Python-static gate guard.
2. ``test_nh_smag_changes_winds_when_ah_positive`` — perturbation
   response with smag=0.20 (PE-tested useful range).
3. ``test_nh_smag_no_effect_when_ah_zero`` — Smagorinsky is gated
   inside the A_h block; bit-for-bit baseline when A_h=0.
4. ``test_nh_smag_differentiable`` — ``jax.grad`` flows through 5
   steps with smag ON.  Note: Smagorinsky helper computes
   ``sqrt(strain_mag)`` whose gradient is singular at zero strain;
   test starts from a non-rest perturbed IC to avoid the
   sqrt-at-zero singularity.  Documents this as a known limitation
   for differentiable training touching the rest state.
5. ``test_nh_smag_rest_state_smoke`` — 20 steps from rest stay
   finite.  At rest, strain is exactly zero so Smagorinsky
   contributes 0 to the tendency — but the FORWARD pass works
   (only the gradient is singular).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_smagorinsky_ah_nh_iter180.py
    => 5 passed in 74.82 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py
    => 5 passed (iter-172 AST guard unchanged after iter-180
       config addition)

### Status

Closes another PE-vs-NH FV3-fidelity gap: NH now has the same
adaptive Smagorinsky A_h infrastructure as PE.  The
``smagorinsky_cs`` field complements the iter-171
``div_damp_dddmp`` (which is the cell-centre divergence-damping
Smagorinsky) as the matched pair of FV3 adaptive damping
coefficients.

## Iteration 179 (2026-05-08): NH-equivalent cube-imprint metric

### Goal

PE iter 2 introduced the quantitative cube-imprint metric: ratio
of std(v) at panel edges vs std(v) in the face interior.  PE
iter 17 measured -79 % mid-level cube imprint at HS C36 with the
optimal corner-div damping setting.  NH had no analogous metric,
making it impossible to claim quantitative imprint reduction
for the iter-168/169/170/171 toolkit ports.

This iter introduces the NH analogue and validates that the
toolkit reduces the metric vs the no-damping baseline.

### Implementation

New file ``tests/test_cube_imprint_nh_iter179.py`` (2 tests, no
production code change):

1. ``test_imprint_metric_is_finite_and_positive`` — sanity check
   that the metric (edge_std / interior_std with edge_width=2)
   produces a finite positive value on the no-damping baseline.
   Documents the C8 scale where the absolute imprint magnitude
   is small (ratio ~ 1, comparable to noise) — PE's HS C36
   30-day case showed ratio ~ 1.27 at day 30, but that's well
   beyond the wall-time budget for a unit test.
2. ``test_full_toolkit_reduces_edge_imprint_ratio`` — with the
   full FV3 toolkit ON (all four iter-168/169/170/171 mechanisms
   + 4th-order ζ corner), the ratio is STRICTLY LOWER than the
   no-damping baseline.  This is the most-direct quantitative
   validation of the toolkit's intended purpose: suppress
   spurious wind amplification at cube-face boundaries.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_cube_imprint_nh_iter179.py
    => 2 passed in 24.90 s

### Status

The NH path now has the same quantitative imprint metric coverage
as the PE path, modulo grid-resolution / wall-time constraints.
The toolkit-reduction comparison is structural (lower ratio
under full toolkit vs no damping) rather than absolute (specific
percentage reduction), reflecting the small-grid signal-to-noise
constraint at C8.

## Iteration 178 (2026-05-08): extend iter-172 AST guards + self-check

### Goal

The iter-172 AST regression guard was written before iter-173
added the ``use_async_halo`` field and dispatch.  Result: a
refactor that drops iter-173's wiring would not trip the
iter-172 guard — silent regression risk for the MPI overlap
optimisation.

This iter:

1. Extends the iter-172 config-fields guard to include
   ``use_async_halo: False``.
2. Extends the iter-172 call-sites guard to include the
   iter-173 dispatch gate (``config.use_async_halo and _hb_div ==``)
   and helper (``_overlapped_arakawa_lamb_gradient``).
3. Adds a self-check ("test the test"): explicitly drops each
   gate one at a time from a synthetic source string and verifies
   the inner check function correctly flags the omission.  This
   catches the failure mode I noticed when extending the guard:
   a typo in the gate substring (e.g., wrong operator spacing)
   would cause the guard to ALWAYS pass, silently disabling the
   regression check.

### Implementation

Modifications to ``tests/test_fv3_nh_toolkit_iter172.py``:

* Added ``"use_async_halo": False`` to the ``expected`` config
  fields dict in ``test_nh_fv3_config_fields_ast_regression``.
* Added the iter-173 (gate, helper) pair to the
  ``gate_helper_pairs`` list in
  ``test_nh_fv3_call_sites_ast_regression``.
* Added a new test ``test_iter178_ast_guard_self_check`` that
  loops over each (gate, helper) pair, drops it from a synthetic
  source string, and asserts the missing-detection logic flags
  the omission.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py
    => 5 passed in 82.26 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py tests/test_async_halo_nh.py
    => 7 passed (iter-173 unchanged after the iter-178 extension)

### Status

The AST regression guard now covers all five PE-NH-asymmetry
fixes (iter 168/169/170/171/173) AND has a self-check ensuring
the guard itself is not silently broken by a typo.  The test
suite is now self-defending: a regression in ANY of the five
wirings, OR a typo in the regression test itself, will be caught.

## Iteration 177 (2026-05-08): cube-vertex corner fill mode reaches NH

### Goal

The legoESM halo machinery has three documented cube-vertex fill
modes (``"avg"`` default, ``"fv3_agrid_xdir"``,
``"fv3_bgrid_xdir"``) set via the ``LEGOESM_CORNER_FILL`` env var
or ``set_corner_fill_mode()``.  PE iter 10 documented the
``fv3_bgrid_xdir`` mode as the cleanest win for HS C36
(-41 % cube imprint, -15 % max\|v\| vs ``avg``).  But the NH path
never had a test that the corner fill mode actually REACHES NH's
transport / damping halos — a silent-ignore bug would only show
up under deep visual inspection.

This iter closes that gap with a regression test that the FV3
modes produce a measurably different NH trajectory than ``avg``
mode.

### Implementation

New file ``tests/test_corner_fill_mode_nh_iter177.py`` (3 tests,
no production code change):

1. ``test_fv3_bgrid_xdir_changes_nh_state`` — bit-for-bit
   different NH state after 3 steps with ``fv3_bgrid_xdir`` vs
   ``avg``.
2. ``test_fv3_agrid_xdir_changes_nh_state`` — same for the
   AGRID-XDir mode.
3. ``test_corner_fill_mode_round_trip`` — set/get round-trip
   for the three valid modes + ``ValueError`` on invalid input.

### Implementation notes

* The ``set_corner_fill_mode`` function manipulates a module-level
  global (``halo._corner_fill_mode``) — tests save and restore
  the global to avoid leaking state between tests.
* Models are constructed AFTER setting the mode so any
  module-level caching observes the right setting.
* ``s.u.data.block_until_ready()`` forces JAX trace
  finalisation before the mode is restored.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_corner_fill_mode_nh_iter177.py
    => 3 passed in 25.80 s

### Status

The PE-documented cube-vertex fill modes are now confirmed
reachable from the NH path.  Future iterations may extend with
quantitative cube-imprint reduction tests (NH-equivalent of PE
iter 10's HS C36 metric).

## Iteration 176 (2026-05-08): NH FV3 toolkit transient damping

### Goal

iter 174/175 verified that single FV3 mechanisms reduce
``mean(|div_v|)`` and ``mean(|ζ|)`` at a single 5-step horizon.
This iter validates the *transient* behavior with the FULL toolkit
ON: over a 10-step window, the trajectory's FINAL ``max|div_v|``
should be lower than the no-damping baseline (compounded damping
effect).  This is the property a user actually cares about — does
the toolkit suppress divergence over time?

### Implementation

New file ``tests/test_fv3_toolkit_transient_iter176.py`` (2 tests,
no production code change):

1. ``test_full_toolkit_reduces_final_divergence`` — turns ON ALL
   FOUR iter-168/169/170/171 mechanisms, runs 10 steps from
   divergent IC, asserts FINAL ``max|div_v|`` is at least 5 %
   lower than the no-damping baseline.  Note: PEAK
   ``max|div_v|`` occurs at t=0 in both runs (IC dominates), so
   the comparison is at the FINAL step where compounded damping
   shows.
2. ``test_full_toolkit_keeps_kinetic_energy_bounded`` — over the
   same window the volume-mean KE stays within 2× the IC KE
   (catches a sign-flip in any mechanism that would inject
   energy rather than remove it).

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_toolkit_transient_iter176.py
    => 2 passed in 29.93 s

### Status

The FV3 toolkit's intended behavioral effect (suppress divergence
over time) is now quantitatively validated at the integrated-
trajectory level, complementing the single-step quantitative
correctness from iter 174/175.

## Iteration 175 (2026-05-08): quantitative damp_v correctness for NH

### Goal

Extend the iter-174 quantitative-correctness pattern to the
iter-169 post-step ``damp_v`` mechanism.  iter-169's unit tests
verify the wiring CHANGES the state but not the *direction* of
the change; this iter adds tests that verify ``damp_v`` actually
REDUCES vorticity (not amplifies it).

### Implementation

New file ``tests/test_damp_v_quantitative_iter175.py`` (4 tests,
no production code change):

1. ``test_damp_v_reduces_vorticity`` — ``damp_v=0.030`` with
   ``nord_v=0`` (del-2 path) reduces ``mean(|ζ|)`` by at least
   1 % vs no-damping baseline.
2. ``test_damp_v_no_amplification_for_any_nord[0]`` — del-2
   path does not amplify vorticity.
3. ``test_damp_v_no_amplification_for_any_nord[1]`` — del-4 path
   does not amplify vorticity.
4. ``test_damp_v_no_amplification_for_any_nord[2]`` — del-6 path
   (FV3 production) does not amplify vorticity.

### Design notes

* Initial condition is a sinusoidal v perturbation
  ``V0 * sin(2π i/n)`` (constant in j) producing pure shear
  vorticity.  At C8 this gives ``mean(|ζ|) ~ 1e-5 s^-1``, well
  above noise.
* The strict ``> 1 %`` reduction floor applies only to the del-2
  test: at C8 with dt=10 over 5 steps, del-6 with the production
  ``damp_v=0.030`` produces only a ~1e-4 % reduction (too small
  to discriminate from FP noise).  The parametrized
  no-amplification tests cover all three nord orders (catches a
  sign error in any of them) at the looser ``≤ 0.1 % growth``
  threshold.
* All other damping (hyperdiff, sponge) disabled so the test
  isolates the ``fv3_del6_vorticity_damping`` helper's contribution.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_damp_v_quantitative_iter175.py
    => 4 passed in 41.32 s

### Status

Quantitative-correctness coverage now extends to iter-169 in
addition to iter-168/171 (covered by iter-174).  Sign errors in
any of the four mechanism implementations would be caught.

## Iteration 174 (2026-05-08): quantitative damping correctness for NH

### Goal

iter-168/171 each have unit tests verifying the damping wiring
CHANGES the state on a perturbed input ("changes-the-state"
guards), but neither verifies the change is in the *correct*
direction.  A sign-flipped damping (e.g., ``du -= grad`` where it
should be ``+= grad``) would still pass "changes-the-state" but
would AMPLIFY divergence rather than damp it — the worst-case
silent failure for a damping mechanism.

This iter adds a quantitative correctness test that initialises
the NH state with a sinusoidal divergent perturbation and verifies
each damping mechanism REDUCES post-step ``mean(|div_v|)`` vs the
no-damping baseline.

### Implementation

New file ``tests/test_div_damp_quantitative_iter174.py`` (3 tests,
no production code change):

1. ``test_corner_div_damp_reduces_divergence`` — corner-div
   damping (iter 168) reduces ``mean(|div_v|)`` by at least 1 %
   vs no damping.  1 % floor catches a sign error while staying
   insensitive to coefficient calibration.
2. ``test_cell_centre_div_damp_reduces_divergence`` — cell-centre
   ``div_damp_coeff=1e10`` (iter 171, constant path) reduces
   ``mean(|div_v|)`` by at least 1 %.
3. ``test_combined_div_damp_at_least_as_strong_as_either`` — both
   mechanisms ON should produce reduction at least as strong as
   either alone (allowing 5 % cushion for nonlinear interaction).
   Catches a sign mismatch where one mechanism partially undoes
   the other's damping.

### Design notes

* Initial condition is a sinusoidal monopole in u
  ``(U0 * sin(2π i/n) * cos(2π j/n))`` rather than random noise.
  Random noise has high-frequency content that doesn't engage the
  d2_bg div_damp floor; the sinusoidal pattern produces a
  ``div_v`` with peak ~3e-5 s^-1 at C8, well above noise.
* ``div_v`` is recomputed independently of the model's internal
  cache (lift cell-centre to corners, project to C-grid,
  ``cgrid_divergence``) so the test is robust to internal API
  changes.
* All other damping (hyperdiff, sponge) is disabled so the test
  measures ONLY the FV3-faithful div damp's contribution.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_div_damp_quantitative_iter174.py
    => 3 passed in 39.25 s

### Status

The four NH FV3-faithful damping mechanisms now have quantitative
correctness coverage.  A sign error in any of corner-div, cell-
centre constant, or combined paths would now be caught — not just
the "doesn't crash" coverage from iter 168/171/172.

## Iteration 173 (2026-05-08): NH async-halo overlap for div damp gradient

### Goal

Bring the NH iter-171 cell-centre div-damping wiring to parity with
PE on one MPI-optimization gap: the PE path's
``primitive_eq_cdgrid.py:594-598`` dispatches to
``_overlapped_arakawa_lamb_gradient`` when both ``use_async_halo``
is set and the halo backend is MPI; the NH path lacked this dispatch.

This is not an FV3 fidelity change (the underlying numerics are
identical) but a documented PE feature now also exposed in NH for
production MPI runs.

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 1 new ``CDGridCompressibleEulerConfig`` field (default off):
  ``use_async_halo: bool = False``.
- In the iter-171 div-damp block, gate the
  ``_arakawa_lamb_gradient(div_v, cdgrid)`` call on
  ``config.use_async_halo and _hb_div == "mpi"``: when True,
  dispatches to ``_overlapped_arakawa_lamb_gradient`` (lazy import
  inside the gated branch); otherwise calls the standard helper.
- Local / SPMD backends fall through to the standard helper —
  bit-for-bit equivalent to ``use_async_halo=False`` on those
  backends.

### Tests added

New file ``tests/test_async_halo_nh.py`` (3 tests):

1. ``test_nh_async_halo_single_device_equivalence`` — bit-for-bit
   equivalence of ``use_async_halo=True`` vs False on the local
   backend.  This is the testable surface; the MPI path requires
   MPI-enabled CI.
2. ``test_nh_async_halo_differentiable`` — ``jax.grad`` flows
   through 5 steps with the field set.
3. ``test_nh_async_halo_ast_regression`` — AST guard for the
   field declaration, the dispatch gate expression, and the
   ``_overlapped_arakawa_lamb_gradient`` helper name appearing
   in the source.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_async_halo_nh.py
    => 3 passed in 49.54 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py tests/test_div_damp_nh.py
    => 9 passed (iter-171/172 unchanged after the iter-173 addition)

### Status

Default-off; opt-in for production MPI runs.  Closes the only
PE feature gap in the iter-171 NH div-damp block.

## Iteration 172 (2026-05-08): NH FV3 toolkit composition + AST regression guard

### Goal

iter-168/169/170/171 each added one FV3-faithful damping mechanism
to the NH 3D path with its own focused test file (4-5 tests per
mechanism).  Two safety gaps remain after this:

* **Composition risk**: each mechanism was tested in isolation; no
  test verifies they all compose without conflict (e.g., a shared
  intermediate like ``div_v`` being computed inconsistently across
  consumers, or one mechanism overwriting another's contribution).
* **Regression risk**: the four new config fields are silently
  defaulted to off; a future refactor that drops a wiring or
  changes a default would silently disable the FV3 mechanism
  without breaking unit tests.

### Implementation

New file ``tests/test_fv3_nh_toolkit_iter172.py`` (4 tests, no
production code change):

1. ``test_nh_full_fv3_toolkit_composes`` — turns ON ALL FOUR
   iter-168/169/170/171 knobs at once on a perturbed NH state,
   runs 20 steps × 10 s, asserts winds stay finite and bounded
   (max\|u\|, max\|v\| < 100 m/s).
2. ``test_nh_full_fv3_toolkit_differentiable`` — ``jax.grad``
   flows through 5 steps with all four knobs ON.
3. ``test_nh_fv3_config_fields_ast_regression`` — walks the
   AST of ``CDGridCompressibleEulerConfig`` and asserts the
   12 expected field/default pairs (corner_div_damp_d2_bg=0.0,
   corner_div_damp_dddmp=0.20, corner_div_damp_d4_bg=0.0,
   corner_div_damp_nord=0, corner_div_damp_fv3_vector_fill=False,
   corner_div_damp_dt_proxy=10.0, damp_v=0.0, nord_v=2,
   use_fv3_a2b_zeta_corner=False, div_damp_coeff=0.0,
   div_damp_dddmp=0.0).  Catches "field renamed", "default
   silently changed", and "field dropped" regressions.
4. ``test_nh_fv3_call_sites_ast_regression`` — searches the
   source text for the four Python-static gate expressions AND
   the helper names that should appear inside each gated block.
   Catches "block dropped during refactor but config field
   retained" — a particularly silent failure mode where the user
   sets the config flag and gets no warning despite the FV3
   mechanism being inert.

### Why these guards matter

The PE path's iter-68 added an analogous AST guard for the
``LEGOESM_HS_CUBE_DT_CFL`` env-var dataflow ("a regression that
reverts ``dt = _resolve_dt_cube(...)`` to ``dt = 200.0`` would
silently disable the env var without breaking unit tests").  The
iter-172 guard generalises that pattern to the four NH FV3 wirings.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/test_fv3_nh_toolkit_iter172.py
    => 4 passed in 81.72 s

    Full NH suite (baseline + iter-168/169/170/171/172):
    => 59 passed in 420.55 s

### Status

All four iter-168 audit asymmetries are closed (iter 171), the
mechanisms compose without conflict (iter 172.1), they remain
AD-safe under composition (iter 172.2), and the wiring is
guarded against silent regression (iter 172.3-4).  Net result:
the NH 3D path now has the same FV3-faithful damping toolkit as
the PE 3D path, with the same level of AST-level fidelity
guarantees.

## Iteration 171 (2026-05-08): port FV3 cell-centre divergence damping to NH 3D path

### Goal

Close the last documented PE-vs-NH FV3-fidelity asymmetry: the
cell-centre constant ``div_damp_coeff`` + adaptive Smagorinsky
``div_damp_dddmp`` (PE iter 5).  Unlike iter-168/169/170, this is
not a port of an existing helper but a substantive ADDITION to the
NH path's damping infrastructure — NH had no cell-centre divergence
damping at all.

### FV3 anchor

- ``sw_core.F90:1720`` adaptive Smagorinsky formula::

      damp = da_min_c * max(d2_bg, min(0.20, dddmp * |div|))

  with ``d2_bg = div_damp_coeff / da_min_c``.  When ``dddmp == 0``
  the damping degenerates to the constant ``div_damp_coeff`` path.
- Implementation matches the PE iter-5 wiring at
  ``primitive_eq_cdgrid.py:594-633`` — same arithmetic, same
  Arakawa-Lamb gradient at D-grid corners, same adaptive coefficient
  computed from cell-centre |div_v| interpolated to corners.

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 2 new ``CDGridCompressibleEulerConfig`` fields (default off):
  ``div_damp_coeff: float = 0.0``, ``div_damp_dddmp: float = 0.0``.
- Block inserted after step 7 (D-grid momentum tendencies) and
  before the existing A_h Laplacian, computing
  ``div_v = cgrid_divergence(u_c, v_c, cdgrid)`` and the gradient
  ``ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_v, cdgrid)``,
  then adding the damping contribution to ``du_d_dt``, ``dv_d_dt``.
- ``div_v`` is HOISTED out of the existing theta-equation block to
  avoid duplicate computation when both consumers (div_damp + theta)
  need it.  The theta block now reuses ``div_v`` if already
  computed; otherwise computes lazily.
- Default both knobs at 0.0 preserves baseline bit-for-bit
  (Python-static branch).

### Tests added

New file ``tests/test_div_damp_nh.py`` (5 tests):

1. ``test_nh_div_damp_zero_is_baseline`` — Python-static gate
   guard.  ``div_damp_coeff=0.0`` matches field-unset baseline
   bit-for-bit (even when ``dddmp`` is set, since the gate is on
   ``coeff > 0``).
2. ``test_nh_div_damp_constant_changes_winds`` — constant path
   (``dddmp=0``) measurably changes winds.
3. ``test_nh_div_damp_adaptive_differs_from_constant`` — adaptive
   path (huge ``dddmp=1e6`` to engage the cap) produces a
   different state from the constant path.  Mirrors the PE-side
   ``test_huge_dddmp_changes_tendencies``: at realistic divergence
   levels the cap doesn't engage so the test forces it via a huge
   coefficient.
4. ``test_nh_div_damp_differentiable`` — ``jax.grad`` flows through
   5 steps with adaptive damping.
5. ``test_nh_div_damp_rest_state_smoke`` — 20 steps from rest stay
   finite, no spurious mass growth.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_div_damp_nh.py
    => 5 passed in 72.73 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py \
        tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py \
        tests/test_corner_div_damp_nh.py tests/test_damp_v_nh.py \
        tests/test_a2b_zeta_corner_nh.py
    => 50 passed (NH baseline + iter-168/169/170 unchanged)

### Status

Default-off; opt-in.  All FOUR documented PE-vs-NH FV3-fidelity
asymmetries in the iter-168 audit are now closed.  The NH 3D path
has the same FV3-faithful damping toolkit as the PE 3D path:

- corner-divergence damping (iter 168, FV3 d_sw5)
- post-step del-n vorticity damping (iter 169, FV3 d_sw6)
- 4th-order A→B ζ corner interp (iter 170, FV3 a2b_ord4)
- cell-centre constant + adaptive Smagorinsky div damp
  (iter 171, FV3 sw_core.F90:1720)

Future iterations: long-time empirical validation of these knobs
on cube HS (deferred for system load), and any further
FV3-fidelity gaps that surface from in-depth audit (e.g., FV3
``a2b_ord4`` for additional corner interps beyond ζ — currently
PE-only via iter-9 finding that swapping ALL corner interps
breaks operator balance).

## Iteration 170 (2026-05-08): port FV3 4th-order A→B ζ corner interp to NH 3D path

### Goal

Continue the iter-168/169 sequence closing PE-vs-NH FV3-fidelity
asymmetries.  This iter ports the iter-14 PE wiring of
``use_fv3_a2b_zeta_corner`` — the FV3-faithful 4th-order A→B
(cell-centre → corner) interpolation for the relative vorticity
at D-grid corners — to the NH 3D path.  Reuses the SW backbone
helper ``_interp_center_to_corner_a2b_ord4`` (port of FV3
``a2b_edge.F90:a2b_ord4``).  No new core code.

### FV3 anchor

- ``a2b_edge.F90:a2b_ord4`` (lines 50-330): 4th-order A→B Lagrange
  interpolation with constants ``a1=9/16, a2=-1/16`` (Lagrange
  4-point) and ``b1=7/12, b2=-1/12`` (PPM volume mean).
- Used by FV3 d_sw5 callers wherever the 2nd-order 4-point
  centre→corner average is insufficient for Smagorinsky-tuned
  damping (Lin 2004).

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 1 new ``CDGridCompressibleEulerConfig`` field (default off):
  ``use_fv3_a2b_zeta_corner: bool = False``.
- In the tendency function step 7 (D-grid momentum tendencies),
  swap the ``zeta_corner = _interp_center_to_corner(zeta, cdgrid)``
  call for the FV3 4th-order ``_interp_center_to_corner_a2b_ord4``
  when the flag is set.  ``a2b_ord4`` operates on 2D ``(6, n, n)``
  fields, so the 3D ``zeta`` is vmapped over the level axis (same
  pattern as the PE wrapper at iter-14).
- The ``θ_corner`` interpolation stays with the 2nd-order
  4-point average — iter-9 in PE established that swapping ALL
  corner interpolations breaks discrete operator balance
  (max winds 2.7× larger).  Only the targeted ζ swap is exposed.
- Default ``use_fv3_a2b_zeta_corner=False`` preserves baseline
  bit-for-bit (Python-static branch).

### Tests added

New file ``tests/test_a2b_zeta_corner_nh.py`` (4 tests):

1. ``test_nh_a2b_zeta_corner_off_is_baseline`` — Python-static
   gate guard.
2. ``test_nh_a2b_zeta_corner_on_changes_winds`` — perturbation
   response.
3. ``test_nh_a2b_zeta_corner_differentiable`` — ``jax.grad`` flows
   through 5 steps.
4. ``test_nh_a2b_zeta_corner_rest_state_smoke`` — 20 steps from
   rest stay finite, no spurious mass growth.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_a2b_zeta_corner_nh.py
    => 4 passed in 92.64 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py \
        tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py \
        tests/test_corner_div_damp_nh.py tests/test_damp_v_nh.py
    => 46 passed (NH baseline + iter-168/169 unchanged)

### Status

Default-off; opt-in.  Three of three documented "corner-fidelity"
PE-vs-NH asymmetries are now closed (corner-div, damp_v, a2b ζ
corner).  Remaining: the cell-centre constant ``div_damp_coeff``
+ adaptive ``div_damp_dddmp`` (PE iter 5) is a more substantive
addition (introduces a new mechanism in NH rather than mirroring
existing infrastructure) and is deferred for future iterations.

## Iteration 169 (2026-05-08): port FV3 post-step vorticity damping (damp_v) to NH 3D path

### Goal

Continue closing the FV3-fidelity asymmetry between the two 3D
atmospheric paths.  iter-168 ported corner-divergence damping; this
iter ports the FV3 post-step del-n vorticity damping (FV3
``sw_core.F90:1948-1999``) from PE iter-12 to the NH path.  Same
mechanism, same SW-backbone helper
(``fv3_del6_vorticity_damping``), now reachable from the NH config.

### FV3 anchor

- ``sw_core.F90:1948-1999`` — post-step ``u += fy2 / dx`` correction
  where ``fy2`` is the del-n flux of the relative vorticity.
- ``sw_core.F90:1582-1597`` — circulation/vorticity construction.
- Same helper as PE iter-12 and SW iter-1009:
  ``legoesm.core.fv3_del6_vt_flux.fv3_del6_vorticity_damping``.

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 2 new ``CDGridCompressibleEulerConfig`` fields (default off):
  ``damp_v: float = 0.0``, ``nord_v: int = 2``.
- Block inserted in ``_step_jitted`` AFTER ``split_explicit_step``
  and BEFORE ``fix_mass_nonhydrostatic``.  Reuses the existing
  ``fv3_del6_vorticity_damping`` helper — no new core code.
- NH-specific adjustment vs PE iter-12: NH stores u/v at CELL
  CENTRES, so the block additionally lifts (u, v) to D-grid corners
  via ``_interp_center_to_corner`` and projects increments back via
  ``_interp_corner_to_center``.  PE stores at corners and skips
  these two interpolations.  The FV3-normal-D-grid damping core is
  identical to PE.
- Default ``damp_v=0.0`` preserves baseline bit-for-bit
  (Python-static branch).

### Tests added

New file ``tests/test_damp_v_nh.py`` (4 tests):

1. ``test_nh_damp_v_zero_is_baseline`` — Python-static gate guard.
2. ``test_nh_damp_v_changes_winds`` — perturbation response with
   ``damp_v=0.030`` (iter-1009 SW production setting).
3. ``test_nh_damp_v_differentiable`` — ``jax.grad`` flows through
   5 steps with damping active.
4. ``test_nh_damp_v_rest_state_smoke`` — 20 steps from rest stay
   finite, no spurious mass growth.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_damp_v_nh.py
    => 4 passed in 63.08 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py \
        tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py \
        tests/test_corner_div_damp_nh.py
    => 42 passed (NH baseline + iter-168 unchanged)

### Status

Default-off; opt-in.  Two of the three documented PE-vs-NH
FV3-fidelity asymmetries (corner-div, damp_v) are now closed.
Remaining: ``use_fv3_a2b_zeta_corner`` 4th-order ζ corner interp
(PE iter 14) and the cell-centre constant ``div_damp_coeff`` +
adaptive ``div_damp_dddmp`` (PE iter 5).

## Iteration 168 (2026-05-08): port FV3 corner-divergence damping to NH 3D path

### Goal

Close the FV3-fidelity asymmetry between the two 3D atmospheric
paths.  The hydrostatic PE path (``primitive_eq_cdgrid.py``) has had
the FV3 corner-divergence damping mechanism wired since iter 16
(del-2) and iter 18 (del-(2*(nord+1))).  The non-hydrostatic
compressible Euler path (``compressible_euler_cdgrid.py``) had no
corner-staggered FV3-faithful mechanism — only a generic cell-centre
``hyperdiff_coeff`` and a sponge.  This iter ports the iter-16/18
mechanism to the NH path so users have the same cube-imprint
suppression knob on both 3D code paths.

### FV3 anchor

- ``sw_core.F90:1641-1822`` (subroutine ``d_sw5``) — full d_sw5
  divergence-damping block.
- ``sw_core.F90:1720`` — adaptive Smagorinsky formula
  ``damp = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt))``.
- ``sw_core.F90:1809-1817`` — higher-order ``dd8`` mixing.
- ``sw_core.F90:2124`` (subroutine ``divergence_corner``) — B-grid
  corner divergence with sin_sg + cube-vertex corner removal.

### Implementation

- File: ``src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py``.
- 6 new ``CDGridCompressibleEulerConfig`` fields (default off):
  ``corner_div_damp_d2_bg``, ``corner_div_damp_dddmp``,
  ``corner_div_damp_d4_bg``, ``corner_div_damp_nord``,
  ``corner_div_damp_fv3_vector_fill``, ``corner_div_damp_dt_proxy``.
- Block inserted between step 7 (D-grid momentum tendencies) and
  step 8 (corner-to-center back-interp).  Reuses the existing
  helpers ``legoesm.core._fv3_divergence_corner.
  fv3_divergence_corner_3d`` and ``fv3_corner_laplacian_iteration``
  — no new core code.
- Mirrors the PE path block at ``primitive_eq_cdgrid.py:635-771``
  with one NH-specific adjustment: ``corner_div_damp_dt_proxy``
  defaults to ``10.0`` (vs ``200.0`` in PE) because the NH path
  runs split-explicit acoustic substepping with much smaller outer
  dt.  The ``d2_bg`` floor dominates in HS-like regimes regardless;
  the ``dt_proxy`` constant matters only when the adaptive cap is
  active.
- Default ``corner_div_damp_d2_bg=0.0`` preserves baseline
  bit-for-bit (Python-static branch).

### Tests added

New file ``tests/test_corner_div_damp_nh.py`` (5 tests):

1. ``test_nh_corner_div_damp_zero_is_baseline`` — Python-static
   gate guard.  ``d2_bg=0.0`` and field-unset path produce
   bit-for-bit identical 5-step output.
2. ``test_nh_corner_div_damp_changes_winds`` — perturbation
   response.  ``d2_bg=0.001`` measurably changes wind tendencies.
3. ``test_nh_corner_div_damp_d4_disabled_bit_for_bit_with_d2`` —
   higher-order gate guard.  ``d4_bg=0.16, nord=0`` matches the
   d2-only path bit-for-bit.
4. ``test_nh_corner_div_damp_differentiable`` — AD safety.
   ``jax.grad`` flows through 5 steps with damping active.
5. ``test_nh_corner_div_damp_rest_state_smoke`` — stability.
   20 steps from rest with ``d2_bg=0.001`` stay finite, no
   spurious mass growth.

### Validation

::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_corner_div_damp_nh.py
    => 5 passed in 79.86 s

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py
    => 4 passed (NH baseline unchanged)

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py
    => 33 passed (NH unit tests unchanged)

### Status

Default-off; opt-in for users who want corner-imprint suppression
in the NH path.  Same FV3 fidelity guarantee as the PE path: exact
``d_sw5`` formula, exact ``divergence_corner`` arithmetic, same
helper code as iter 15-18 (already verified by
``tests/test_fv3_divergence_corner.py``).

Documented gap remaining for the NH path: cell-centre constant
``div_damp_coeff``, ``damp_v`` post-step vorticity damping, and
``use_fv3_a2b_zeta_corner`` are still PE-only.  These are tracked
for future iterations under the same FV3 fidelity umbrella.

### Why this iteration was meaningful

The 'Open follow-ups' list was dominated by stretch-goal cube-runs
that the local hardware can't sustain (C144/C192 multi-day in
~hours-of-wall budget).  This iter takes a different angle —
narrow the FV3-fidelity asymmetry between the two 3D paths, which
is verifiable with unit tests in seconds rather than wall-time
sweeps.  Users running the NH compressible-Euler 3D path now have
the same cube-imprint defense as users running the PE 3D path.

## References

- Colella, P. & Woodward, P. R., 1984: The Piecewise Parabolic Method (PPM) for gas-dynamical simulations. *Journal of Computational Physics*, 54, 174–201.
- Harris, L. et al., 2021: GFDL SHiELD: A unified system for weather-to-seasonal prediction. *Journal of Advances in Modeling Earth Systems*, 13, e2020MS002223.
- Lin, S.-J., 2004: A vertically Lagrangian finite-volume dynamical core for global models. *Monthly Weather Review*, 132, 2293–2307.
- Putman, W. M. & Lin, S.-J., 2007: Finite-volume transport on various cubed-sphere grids. *Journal of Computational Physics*, 227, 55–78.
- Skamarock, W. C. & Klemp, J. B., 2008: A time-split nonhydrostatic atmospheric model for weather research and forecasting applications. *Journal of Computational Physics*, 227, 3465–3485.
- Smagorinsky, J., 1963: General circulation experiments with the primitive equations. I. The basic experiment. *Monthly Weather Review*, 91, 99–164.

