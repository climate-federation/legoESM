# Cubed-sphere shallow-water test suite — issues #504 / #505 / #506

Resolution record for the cubed-sphere Williamson / cosine-bell suite
(GitHub `climate-federation/legoESM` #504–#506). Reference scheme: GFDL FV3
`sw_core.F90` / `test_cases.F90`.

## #504 — cosine-bell corner distortion (RESOLVED)

**Root cause.** The cube transport built the cosine-bell mass flux from the
contravariant `d2a2c` chain (`uc,vc → ut,vt → xfx = dt·ut·dy·sin_sg`). That
chain is **not** geometric-conservation-law (GCL) consistent at the cube
face-boundary seam (the FV3-faithful `ut = uc/sin_sg` override at `i=1/i=n-1`),
so a *non-divergent* wind acquires spurious area-flux divergence — 27× at the
8 cube vertices, a stripe at the seam. A constant `h≡1` advected by the
analytic solid-body wind develops `max|h−1| = 0.52` in 2 days, fragmenting the
bell at corner crossings (α=π/4 L2 = 0.93, 11° drift).

**Fix.** Build the cosine-bell mass flux as the **exact discrete curl of a
corner streamfunction** (FV3 `test_cases.F90` `wind_field=0`):

    xfx[i,j] = dt·(ψ[i,j] − ψ[i,j+1])     yfx[i,j] = dt·(ψ[i+1,j] − ψ[i,j])

so the discrete divergence telescopes to machine zero for **any** ψ.
New `fv_tp_2d.streamfunction_mass_fluxes` + `streamfunction_transport_step`
(+ shared `_finalize_transport`); `cosine_bell.rotation_streamfunction`.
Wired in the `run_cosine_bell` cube path **only** — W2/W5/W6 use the
prognostic path, so zero regression there.

**Result.** Free-stream divergence 0.52 → 3.7e-7; α=π/4 L2 0.93 → 0.115
(1-day cross-grid metric; 0.19 over a full 12-day revolution with diffusion);
drift 11° → 0.3°. Visually coherent, no corner artifacts.
Tests: `tests/core/test_streamfunction_freestream.py`,
`tests/test_atmosphere_cross_grid_plots.py::TestIssue504`.

## #505 — Williamson-2 comparison panel (RESOLVED)

Added the `v` meridional panel to
`ATMOSPHERE_COMPARISON_FIELDS['williamson2']`. The near-zero steady-state `v`
is the cube-imprint canary (see Visual verification). `TestIssue505`.

## #506a — Williamson-6 surface pressure (RESOLVED)

`p_s = ρ_air·g·(h + h_s)` injected into the W6 snapshots + a `williamson6`
comparison panel. `TestIssue506SurfacePressure`. Also #504a: `alpha`
threaded through `run_cosine_bell`; permanent `cosine_bell_a0` (α=0) matrix
case on all four grids.

## #506b — Case-6 (Rossby–Haurwitz wave-4) long-run symmetry (DISPOSITION)

**Observation.** The wave-4 RH solution loses its 4-fold longitudinal
symmetry; the issue reported breakdown ~day 20, "C96 worse than C48".

**Current state.** The shipped code holds symmetry to ~day 22–34 at C48
(off-(m=4) power fraction crosses 1e-2 at ~day 22, fully broken by ~day 56) —
*better* than the issue's reported ~day 20.

**Why this is the correct endpoint (not a remaining bug):**

1. **Physical instability — cube is not the outlier.** The RH wave-4 is
   barotropically unstable (Thuburn & Li 2000); every model breaks it down on
   a timescale set by its numerical diffusion. `diag_506b_crossgrid_case6.py`
   integrates W6 to 40 d on four grids at matched ~2° and measures the day the
   off-(m=4) power fraction crosses 1e-2:

   | grid | resolution | breakdown day | note |
   |------|-----------|---------------|------|
   | icosahedral | ico6 | **24** | breaks *earlier* than cube |
   | cubed-sphere | C48 | **28** | |
   | spectral | T63 | (blew up d26) | untuned high-res dt, not symmetry loss |
   | lat-lon | 96×192 | >40 (none) | heavy polar/implicit diffusion |

   The cube tracks the icosahedral grid almost exactly and breaks *later* than
   it — so the breakdown is grid-generic instability, not a cube GCL pathology.
   The lat-lon outlier preserves symmetry only because of its much heavier
   implicit dissipation (more damping → later breakdown, not "more correct").
   Full multi-week symmetry preservation is therefore partly *unphysical*.

2. **The cube residual is FV3-faithful.** The remaining seam GCL error is the
   `ut = uc/sin_sg` face override — a faithful port of `sw_core.F90`. FV3
   itself is not free-stream-exact on the cube. The #504 streamfunction trick
   does **not** transfer to the prognostic path (no analytic ψ). A principled
   GCL rewrite of shared `d_sw` would **deviate from the reference scheme**
   and touch the tuned W2/W5 path — a faithfulness regression, not a fix.

3. **"C96 worse than C48" is no longer reproduced.** The original issue's
   resolution-dependent complaint is the sharpest test of a cube-specific
   pathology. `diag_506b_cube_scaling.py` runs W6 to 40 d at C48 and C96 with
   the runner's own resolution-scaled dt/diffusion:

   | resolution | breakdown day | off-(m=4) @ day40 |
   |-----------|---------------|-------------------|
   | C48 | 28 | 0.340 |
   | C96 | **28** | **0.193** |

   Both break at the *same* physical day; C96 is in fact *cleaner* at day 40,
   not worse. The resolution-dependent symptom the issue reported was a
   property of the pre-fix code (div-damp scaling + C36-only calibration) and
   is gone in the shipped code.

4. **The prognostic advective free-stream correction is rejected with
   numbers.** `diag_506_advective_freestream.py` converts the flux-form update
   to advective form (`h += (div(F) − h·div(F(1)))/area`), which IS
   free-stream-exact without a streamfunction. Measured on the cube cosine
   bell with the original d2a2c winds:

   | | free-stream `max|h−1|` (2 d) | bell α=π/4 L2 (12 d) |
   |--|------------------------------|----------------------|
   | flux-form (FV3) | 0.517 | 0.925 |
   | advective correction | **2.9e-7** | **0.934** |

   It fixes free-stream by 6 orders **but does not improve transport
   accuracy** (L2 0.925 → 0.934, marginally worse) — free-stream preservation
   ≠ accuracy. And it abandons the conservative flux-form FV3 relies on
   (mass only recoverable via the global rescaler, not local flux closure),
   so it is both a conservation and a faithfulness regression. Rejected.

5. **Isolated levers exhausted.** `div_damp ∝ (48/n)²` is the correct del-2
   scaling; the hyperdiff sweep (factor 2/4/8/16) lowers early asymmetry ~4×
   but does not move the ~day-33 breakdown onset.

**Verdict.** #506b is closed as *working-as-intended / reference-faithful*.
The residual is intrinsic barotropic instability plus FV3's own GCL error;
the cube is not the worst grid, C96 is not worse than C48, and the two
free-stream "fixes" available (streamfunction — prescribed-wind only;
advective form — non-conservative + no accuracy gain) cannot fix the
prognostic path without deviating from the reference. Evidence:
`scripts/tmp/diagnostic/diag_506_case6_symmetry.py`,
`diag_506b_fv3_sinsg_gcl.py` (FV3 edge `sin_sg` changes GCL by −0%),
`diag_506b_crossgrid_case6.py`, `diag_506b_cube_scaling.py` (C48≡C96 day-28),
`diag_506_advective_freestream.py`, `results/diag_506_ext/`.

This disposition was challenged by codex adversarial review (verdict
needs-attention → all three findings addressed): C96 scaling measured
(finding 1), advective candidate quantified + rejected (finding 2), and the
#504 flux-sign convention confirmed code≡doc with a direct sign assertion
`tests/core/test_streamfunction_freestream.py::test_flux_sign_convention`
(finding 3).

## Long-run cosine-bell faithfulness (issue #521 context, 2026-06-20)

Verified the cube cosine-bell transport over **10 revolutions (120 days)** at C48,
both orientations, reusing the production streamfunction path
(`scripts/tmp/diag_cosine_bell_longrun.py`). Reference: FV3 `test_cases.F90`
`wind_field=0` builds the mass flux as the corner-streamfunction discrete curl
(`uc=-(ψ[i,j+1]-ψ[i,j])/dy`, `vc=(ψ[i+1,j]-ψ[i,j])/dx`) — exactly
`fv_tp_2d.streamfunction_mass_fluxes`; the PPM is `tp_core.F90` hord=10 (Huynh
2nd-constraint) with the `is==1`/`ie+1==npx` one-sided edge stencil
(`apply_fortran_xppm_boundary`). Faithful by construction.

| @ rev10 (120 d), C48 | α=π/4 (corners) | α=0 (edges) |
|----------------------|-----------------|-------------|
| free-stream `max\|h−1\|` | **1.55e-6 (flat, no rev-over-rev growth)** | **1.55e-6 (flat)** |
| bell L2 vs IC | 0.45 | 0.41 |
| peak height (init 988) | 477 | 712 |
| mass error | ~1e-6 (bounded) | ~1e-6 |
| min h | 0 (monotone, no undershoot) | 0 |
| centroid drift | 2.66° | 5.35° |

**Conclusions (faithful endpoint, no remaining bug):**
1. **No edge/corner GCL artifact — spatially localized, not just global.**
   Free-stream `h≡1` stays at the roundoff floor for the full 120 days
   (C48 1.55e-6, C96 1.19e-7), and that residual is **uniform**: edge-ring max ==
   interior max == face-corner max (ratio 1.00). A seam/corner leak would both
   accumulate over revolutions AND concentrate on the boundary ring; it does
   neither. (pre-#504 d2a2c flux: 0.52 in 2 days, striped at the seam.) The
   no-rescaler regression test (`mass_target=None`) confirms this is not the
   global mass fixer masking a local leak — `max|h−1|` is a local max and would
   expose any sign-cancelling seam pair.
2. **The distortion CONVERGES with resolution ⇒ it is numerical diffusion, not a
   bug or a fixed cube artifact.** Doubling C48→C96 (α=π/4): rev-1 peak loss
   16%→5%, 1-rev L2 0.114→0.041 (~2.8×, between 2nd and 3rd order), drift
   0.31°→0.07° (4.4×). A GCL/seam artifact would NOT clean up at ~2nd order.

   | | C48 rev1 | C96 rev1 | order |
   |--|----------|----------|-------|
   | bell L2 (α=π/4) | 0.114 | 0.041 | ~1.5 |
   | peak loss | 16% | 5% | |
   | drift | 0.31° | 0.07° | ~2.1 |

3. **The corner orientation is no longer the outlier.** α=π/4 (corner-crossing)
   L2 0.45 vs α=0 (edge-only) 0.41 (pre-#504 it was 0.93 vs 0.19); edge-crossing
   in fact *drifts more* (5.35° vs 2.66°). Consistent with — though not by itself
   proof of — grid-generic dispersion; the resolution convergence in (2) is the
   decisive evidence. Single-rev L2 matches PL07 / Lauritzen monotone-PPM at this
   resolution. Chasing zero decay would require a non-monotone or non-FV3 limiter
   — a faithfulness regression, rejected.

Scope (what is and isn't certified): free-stream `h≡1` certifies GCL /
divergence-freeness across all seams/corners for all time; the α=π/4 cosine bell
certifies a real non-constant gradient transported *over* the corners. Neither
probes filamentary tracers or sign-changing fields — the colliding-modons test
(issue #521) is the intended nonlinear seam stress test (now implemented and
matrix-registered; see the #521 section below). Regression pin:
`test_streamfunction_freestream.py::test_freestream_bounded_over_many_revolutions`
(h≡1 over 2 revolutions, `max|h−1|` < 1e-5). Diagnostic:
`scripts/tmp/diag_cosine_bell_longrun.py`. Codex-reviewed (5 adversarial
findings; resolution-scaling + residual-localization added to address them).

## #521 — Colliding modons (nonlinear SW test case, 2026-06-20)

Implemented the FV3 case-8 "soliton twin-vortex" / JAMES Colliding-Modons test
(`doi:10.1002/2017MS000965`). Faithful port of `tools/test_cases.F90` case-8
(codex-reviewed, 6/6 axes, no bugs):
- Non-rotating (f=0), flat free surface h≡5000 m (`gh0=5e3·g`, `delp/g`).
- Two equatorial zonal-wind Gaussians `±50·exp(−(r/750 km)²)` at 90°E (westerly)
  and 270°E (easterly); `r` = great-circle distance to the edge midpoint.
- Reuses the validated cube zonal-wind edge projection (`u_d=cos_angle_edge_x·u_e`,
  `v_d=−sin_angle_edge_y·u_e`) — same as the W2 / cosine-bell cube IC; FV3's
  `inner_prod(e1/e2, ex)` of the east vector is exactly that metric rotation.
- Non-rotating handled as experiment glue: `cdgrid._replace(f_corner=0)` (the SW
  core's only Coriolis use is `f_corner + rarea_c·vort`), mirroring `f0=fC=0`.

`tests/test_cases/colliding_modons.py` (`colliding_modons_cdgrid`, the FV3
edge-midpoint assembler; all-grid IC kernels live in the same module) +
`tests/atmosphere/shallow_water/unit/test_colliding_modons_run.py` (IC
faithfulness + short non-rotating prognostic run: stable, mass-conserving,
modons evolve, h>0). Driver `scripts/validate/run_colliding_modons.py`
(collision/conservation/symmetry diagnostics).

**Status — matrix-registered, stable 100 days at C36/C48/C96 (#521 + #753
item 1 landed; #753 item-2 return metric open).** The case is a standard
matrix-runner case (`test_num == 8`, `--only sw --test =colliding_modons
--grid cubed_sphere`). The validated stable config is the shared `MODON_*`
constants (single source of truth for the matrix and `run_colliding_modons.py`,
so they cannot drift — the #800 desync class): `use_duogrid=True` cube seams,
`damp_v=0.010`, `hyperdiff_factor=1.0×`, and the **`(ref/n)^2` biharmonic law**
(`MODON_HYPERDIFF_SCALING=2`).

- **The (ref/n)^2 default (#753 item 1).** The earlier `(ref/n)^4`
  grid-scale-damping-time-constant law is under-damped at the C96+ face seams
  (the collision drives an enstrophy cascade whose grid-scale delivery rate
  rises with resolution), so **C96 erupts under `^4`** (max|u| 309–449 m/s by
  day ~5–50) but is stable under `^2` (which delivers 4× the backstop at C96 =
  (96/48)²). Every exponent returns `ref_coeff` at `n == ref_n == 48`, so **C48
  is exponent-invariant** (byte-identical to the pre-#753 code there); at C36
  `^2` gives 0.56× — LESS damping than `^4`, so the flip cannot over-damp the
  coarse cores. `LEGOESM_SW_MODON_HYPERDIFF_SCALING=4` opts back into `^4`.
- **100-day validation (matrix runner, non-rotating, mass fixer on).** C36
  (`^2` = 0.56×, the regression gate): **PASS, mass drift 0.00e+00**, max|u|
  ~9 m/s. C96 (`^2` = 4×, the fix): **PASS, mass drift 0.00e+00**, max|u|
  oscillates ~18–40 m/s through the collision (day-100 ~33) — it does NOT decay
  monotonically toward zero, i.e. the cores survive the run (not over-damped to
  death). C48: invariant, PASS. C192 (`^2` = 16×): also **PASS, mass drift
  0.00e+00**, max|u| ~40 m/s at day 100 — the law holds at the highest
  resolution probed. Head-to-head, C96 under the old `^4` default erupts to
  max|u| 274–410 m/s by day 5–10 (energy_err 2.4e-2, cores destroyed).

**Automated coverage caveat — C96 is the resolution that exercises the fix.**
The default matrix cube resolution is **C36** (`GRID_RESOLUTIONS["cubed_sphere"]`),
which is stable under BOTH `^2` and `^4`, so the standard
`--only sw --test =colliding_modons --grid cubed_sphere` run does NOT exercise
the eruption. The fast unit tests
(`test_run_colliding_modons_cli.py`, `test_williamson_cli_calibration.py`,
`test_matrix_nh_cube_parity_ast_guard.py`) pin the constant + the coefficient
ratios, so they catch a silent revert of the default to `^4` — but the
*physical* C96 seam stability rests on the **100-day matrix integration at
`--resolution C96`** above, which must be run in the nightly/manual matrix (the
same convention as the cube visual-regression gate). A future maintainer should
not assume the default C36 run guards the C96 fix.

OPEN — **#753 item 2 (return metric, resolution-bound).** The matrix PASS gate
is finite + blow-up-threshold + conservation; it certifies seam stability, NOT
the exact ~100-day return-to-initial-position (the driver prints the tracked
vortex longitudes `lon_W`/`lon_E`, but there is no coded return-window/core-
amplitude acceptance yet). Closing the day-100 return gap needs resolved 750 km
cores → C96/C192 core-strength preservation, tracked separately in #753.

## Visual verification (no artifacts)

`scripts/validate/visual_regression.py --check`: SSIM=1.0000, hamming=0,
edge_ratio=1.349 == baseline. W2 `v`-wind native cube shows the known ~1%
cube-imprint residual (steady state, no growth); W5 wind speed and the Case-6
height/`v` fields show no new edge stripes or corner spikes.
