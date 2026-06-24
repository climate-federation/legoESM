# internal_tide (#576) — RESOLVED (Flat-y + ImplicitFreeSurface). Design & history.

## STATUS: REPRODUCED 2026-06-24
M2 b' pattern_corr **+0.99 (day 0.1) → +0.94 (0.5) → +0.81 (1.0) → +0.67 (1.2)**,
amplitudes within ~1.5× (from corr≈0 + blow-up). Fix = `meridionally_flat` config
field (Oceananigans `Flat`-y: ∂/∂y≡0) on the `implicit_cn`/`ImplicitFreeSurface`
stack. Productionized: `LatLonCGridOceanConfig.meridionally_flat` (default-off,
bit-identical), wired through the comparison + canonical config; tests
`test_meridionally_flat_inertial_oscillation` + the comparator. Residual: late-time
corr decay (0.94→0.51 over days 1.4→2.0) = advection/dispersion vs oracle WENO7
(refinement). The long "Arakawa–Lamb / spatial null mode" thread below is SUPERSEDED
(it was an `implicit_cn` split-coupling instability, not a spatial scheme defect).

---

# internal_tide (#576) — barotropic Coriolis null-mode fix: design & plan (history)

Actionable handoff from the 20-iteration diagnostic (ralph log
`.claude/ralph_internal_tide_staircase_task.md`; memory `project_internal_tide_pgf`).
This is the *one* remaining piece of work that closes Oceananigans `internal_tide`
fidelity — and, because it is the same root, the Silvestri §5 turbulent blow-up.

## ⚠️⚠️⚠️ RESOLUTION PATH (from the Oceananigans SOURCE, not a search)

Looked up the oracle's actual time stepper in the Oceananigans source
(`Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl`,
`barotropic_pressure_correction.jl`):
- **Timestepper = `:QuasiAdamsBashforth2`** (AB2 of the full momentum RHS, Coriolis
  INCLUDED).
- **Free surface = `ImplicitFreeSurface`** (the `XYRegularStaticRG` default; the
  internal_tide grid is x-regular / y-Flat, so it is NOT split-explicit).
- Scheme: AB2 momentum → solve η implicitly → `correct_barotropic_mode!` subtracts
  the barotropic pressure gradient UNIFORMLY at every level (`u -= g·Δt·∂ₓη`). No
  forward-backward Coriolis in the η solve, no `_cori_fac` gating, no `F_slow`
  Coriolis routing — i.e. far SIMPLER than legoESM's `implicit_cn` predictor-corrector.

**Result:** legoESM's `barotropic_solver="implicit_unsplit"` (MITgcm-faithful: full
velocity solved implicitly with a uniform surface-pressure correction) is the
faithful match to Oceananigans' `ImplicitFreeSurface`, and it is **STABLE** where
`implicit_cn` blows up: the periodic-y flat-bottom inertial oscillation stays
bounded and tracks the analytic solution for 2 full periods (u: 0.20→0.00→−0.21→
0.00→+0.20, max|u|≈0.20); flat-bottom + tidal forcing also stable (day 1, max|u|
0.77). So the instability was the `implicit_cn` split-coupling of the implicit FS
with the explicit/FB Coriolis — NOT a spatial null mode and NOT needing Arakawa–Lamb.
(All the earlier "spatial null mode / AL redesign" conclusions are SUPERSEDED.)

**Remaining (narrow, well-defined): complete the y-periodic wiring for the bump
path.** Isolation with `implicit_unsplit`: bump is STABLE walled (day 0.5,
max|u|=0.495) but blows up PERIODIC even homogeneous (day 0.08). So it is NOT a
partial-cell-numerics or stratification problem — it is the y-periodic boundary
wiring, which was only validated FLAT-bottom (the 4 sites: zero_polar_lat_ends,
pad_with_pole_bc_lat, compute_face_masks, compute_face_masks_3d). The bump path
exercises more meridional-boundary operators that are not yet periodic-wired. And
the match genuinely NEEDS periodic-y: walled `implicit_unsplit` internal_tide runs
stably 2 days but corr≈0 / under-generates (the geostrophic-adjustment topology
mismatch). So the ONLY remaining work is the coherent y-periodic operator sweep
(Step 1 of the Plan), now with the stepper question SETTLED (`implicit_unsplit`).

### Updated plan (stepper settled)
1. Set `barotropic_solver="implicit_unsplit"` for the Oceananigans internal_tide
   recipe (faithful to `ImplicitFreeSurface`; fixes the core instability).
2. Complete the coherent y-periodic meridional-boundary conversion for the bump /
   partial-cell path (find the not-yet-wrapped operators that the bump exercises:
   the partial-cell PGF, the unsplit-solve N/S boundary, the topographic continuity).
3. Run the comparator (periodic-y + implicit_unsplit): target M2 b' corr ≥ 0.6.
NO Arakawa–Lamb / novel-Coriolis work is needed — that whole line is superseded.

## Verified root cause (historical — see RESOLUTION PATH above for the actual fix)

The Oceananigans `internal_tide` oracle is `topology=(Periodic, Flat, Bounded)` —
meridionally **unbounded** (true 2-D x–z). legoESM's lat-lon beta-plane C-grid is
meridionally **closed** (walls the N/S v-faces). Because the barotropic Rossby
radius √(gH)/f ≈ 1358 km ≫ basin (256 km), a uniform barotropic flow in the closed
basin **geostrophically adjusts** (u retained as a ~+0.4 m/s DC current) instead of
freely oscillating — so the prescribed M2 tide advects a steady lee wake over the
ridge instead of radiating an internal tide, and b' decorrelates from the oracle.

Matching the oracle therefore needs a **meridionally-periodic (y-re-entrant)**
configuration. A 4-site gated PoC (`set_meridionally_periodic()`) confirms this:
with periodic-y the barotropic mode follows the **analytic inertial oscillation**
exactly through a quarter period (u: 0.20→0.145→0.005 vs analytic 0.20→0.143→0.005).

**But** periodic-y re-admits the **C-grid 2Δy barotropic Coriolis null mode** — the
meridional analog of the 2Δx mode `tests/.../test_barotropic_coriolis_null_mode.py`
pins, and the same mode that blows up Silvestri §5. The closed-basin walls were
suppressing it. It is a 2Δy checkerboard in v (verified: 2Δy-rough/|v|=1.83, u stays
x-uniform) that **grows unbounded** and NaNs the run by ~day 0.2.

### What it is NOT (all empirically ruled out, iters 16–20)
- NOT a time-integrator instability: `ab2_epsilon` 0.1/0.5/1.0 → identical blow-up;
  both AB2 (`explicit_ab2`) and FB (`matsuno_split`) blow up → purely **spatial**.
- NOT fixable by dissipation: `A_h` 0–200 (baroclinic, the mode is barotropic),
  `barotropic_diffusion_alpha`, operator-split del²y (on v; on u+v), and an in-solve
  **predictor** biharmonic smoother (ν≤0.0625, nulls a pure 2Δ mode each step) all
  fail — the mode is distributed across the barotropic predictor+corrector+3-D
  `u_prime`+η and the Coriolis re-seeds it every step.
- NOT the 3-D momentum scheme: `weno5`/`vector_invariant` (enstrophy-conserving PV
  flux) blow up too — under `explicit_ab2`+`implicit_cn` the barotropic Coriolis
  comes from `F_slow` + the solver, not the 3-D vorticity flux.
- NOT the boundary wiring: **f=0 periodic is perfectly stable** (max|u|=0.200) — the
  4-site periodic wiring is correct; the instability appears only with f≠0.

## ⚠️ CORRECTED MECHANISM (standalone SWE proof) — it is NOT a pure spatial null mode

A minimal standalone C-grid barotropic SWE solver (`scripts/tmp/_baro_swe_coriolis.py`)
shows the **naive** 4-pt-average Coriolis is PERFECTLY STABLE + tracks the analytic
inertial oscillation for 2 periods when the gravity wave is time-resolved (dt=20s,
CFL≈0.7) — even with a seeded 2Δy mode. So the earlier "purely spatial null mode"
conclusion (iters 16–20; `ab2_epsilon` had no effect because it's the OUTER AB2,
not the barotropic split) is WRONG. The real instability is the **implicit-free-
surface (dt=300s ≫ gravity CFL) + explicit/split-timescale Coriolis** coupling for
the wildly time-under-resolved 2Δ gravity mode (ω_2Δ·dt ≈ 33). legoESM's solvers all
mishandle this: `implicit_cn`+`explicit_ab2` routes Coriolis through F_slow (split
timescale, gated off in the solve); `matsuno_split` DOUBLE-COUNTS (3-D Matsuno
sub-step + barotropic-solver FB term → wrong rotation u=0.137 vs analytic 0.001);
`explicit_substep`+`explicit_ab2` gates the in-substep Coriolis off (split). NONE do
the clean "Coriolis resolved/coupled WITH gravity, counted ONCE" that the stable
standalone does.

## ⚠️⚠️ FURTHER CORRECTION (implementation attempt) — the growing mode is dominantly BAROCLINIC

Implementation probing (z-structure of the model's growing 2Δy mode at t/Tin=0.4):
`max|v_baroclinic_dev|=2.14` vs `max|v_barotropic|=0.79`; v z-profile
`[-2.44, -0.49, 0.28, 0.37]` (strongly z-varying). The mode lives DOMINANTLY in the
3-D baroclinic `u_prime` (= `u_star − U_old`), which `u_new_3d = u_prime + U_new`
carries straight past the barotropic solver — explaining why NO barotropic-solver
knob touches it (`barotropic_implicit_theta_eta/pgf` 0.5–1.0: zero effect; same for
every solver/scheme/ab2_epsilon). The standalone BAROTROPIC SWE (`_baro_swe_*.py`)
therefore can NOT reproduce it (it has no baroclinic d.o.f.; its own f=0 blow-up is
a different, missing-consistency-term artifact). The real instability is the **3-D
Coriolis 2Δy null mode in the BAROCLINIC momentum** (the `coriolis_cgrid` 4-pt
average in `du_dt` / `_forward_backward_coriolis_3d`), exposed by periodic-y and
coupled through the free surface. So the fix IS a spatial enstrophy-conserving
(Arakawa–Lamb) Coriolis — but applied to the **full 3-D momentum**, not just the
barotropic solver. This is genuine multi-day dycore work; the standalone testbed
route is a dead end (needs the full 3-D + free-surface model to reproduce).

## The fix (exact, minimal locus)

**REVISED:** make the barotropic Coriolis–gravity coupling STABLE at large dt —
either (a) a clean forward-backward in-substep Coriolis counted once (the
Oceananigans/ROMS split-explicit way: Coriolis applied IN the barotropic substeps
with gravity, NO F_slow Coriolis, NO 3-D Matsuno double-count), or (b) a semi-
implicit (CN/FB) Coriolis coupled with the implicit free surface in `implicit_cn`
(`_cori_fac=1` in the predictor, with Coriolis REMOVED from F_slow so it is counted
once). The standalone forward-backward Coriolis+gravity reference (stable, correct)
is the target. NOT an Arakawa–Lamb spatial redesign (the spatial scheme is fine).

Replace the barotropic-solver Coriolis with an **Arakawa–Lamb energy/enstrophy-
conserving** discretization for the 2-D barotropic (U, V, η) system, so the 2Δ mode
has proper restoring and never grows. Locus: `barotropic_implicit_latlon_cgrid.py`
predictor — the 4-pt `V_at_u = 0.25(V[i]+V[i+1]+V_west[i]+V_west[i+1])` and
`U_pred_at_v` f-averages (the null space), **and** the `F_slow` depth-mean Coriolis
routing under `explicit_ab2` (the face-f form f_u≠f_v leaks energy into the null
mode). The 3-D `_bc_pv_flux` enstrophy-conserving machinery is the template, applied
to the 2-D barotropic system.

Note: simply enabling the existing `coriolis_energy_conserving` (Sadourny vertex-f)
on the solver predictor does NOT fix it (tested: worse) — the linear-barotropic AL
form must be derived/implemented specifically, not the existing 3-D option re-used
as-is.

## Plan

1. **Y-periodic grid feature** (necessary, ~half done): promote the gated global
   `set_meridionally_periodic()` to a real config/grid flag; complete the coherent
   meridional-boundary conversion (scalar+vector halo with NO pole-fold sign flip,
   the Coriolis interp, divergence/gradient, the implicit Helmholtz N/S operator,
   polar filter) in ONE pass — partial conversion destabilizes (iter 15b). Defer
   MPI/SPMD band-periodic exchange; local backend first.
2. **AL barotropic Coriolis** (the actual fix): derive + implement the enstrophy-
   conserving 2-D barotropic Coriolis; wire gated into the predictor + F_slow path;
   verify it is null-mode-free (no growth on the 2Δy checkerboard).
3. **AD + default-off bit-identity**: both features gated; existing walled/global/
   tripolar/ACC configs bit-identical; AD finite through the new operators.

## Acceptance gates (the fix is done only when ALL hold)
- **Analytic**: y-periodic flat-bottom inertial oscillation stays **bounded for a
  full inertial period** AND tracks U0·cos(ft) (extend
  `test_barotropic_inertial_oscillation.py`).
- **Null-mode test** `test_barotropic_coriolis_null_mode.py` still green (the fix
  must not re-introduce the 2Δx mode it pins).
- **internal_tide comparator** `compare_oceananigans_internal_tide.py` (periodic-y,
  `MOM_ADV=flux_form`): time-varying M2 b' **pattern_corr ≥ 0.6** vs oracle, rms
  within ~2×, stable over ≥2 days (no NaN).
- **No regression**: Silvestri §5 survival, ACC stability, partial-cell/PGF suites.

## Why it matters beyond internal_tide
Same root as Silvestri §5 (branch `fix/silvestri-turbulent-dissipation`). One
null-mode-free barotropic Coriolis closes both. See `oceananigans_reproduction_
scoreboard.md` row 2d and `project_phase_g_recipe_fidelity`.
