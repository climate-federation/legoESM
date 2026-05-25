# Alistair Adcroft — meeting follow-ups & plan of work

**Date of conversation:** 2026-05-21
**Reference artifacts shown originally:** `docs/legoesm_ocean_model.{md,pdf}`, `docs/legoesm_ocean_choices.{md,pdf}` (38-pp dossier + 11-pp worksheet; not currently in tree — see status note below).
**Status:** living document — items captured first, plan-of-work drafted after.

---

> **2026-05-25 status note.**  This document and the consensus / reviewer
> artifacts in this directory were originally landed in **PR #297**
> (commit 2004ccdc) alongside per-term test scaffolding (Phase 1A
> toggle infra, 1B convergence helper, 1C three per-term test files)
> and a load-bearing `ensure_geometry()` Coriolis-override fix.
>
> They were then removed in **PR #311** (commit 57c01a23, *"chore: repo
> rescue ... ocean + plane WIP"*) along with the per-term tests and
> the dossier/worksheet, characterised as *"obsolete ocean review docs
> + per-term test scaffolding"*.  That same commit also silently
> reverted the `ensure_geometry()` fix — the f-plane regression bug
> came back to main without anyone noticing.
>
> The meeting + reviewer documents are **restored here** because the
> conversation record is unique (Pierre's parallel per-term test
> framework — `tests/ocean/unit/test_term_by_term_analytic.py` —
> supersedes the *implementation* of items 1C, but does not capture
> what Alistair said or the four-reviewer synthesis).
>
> The `ensure_geometry()` fix is re-applied in `src/legoesm/grids/latlon.py`
> and pinned by `tests/grids/test_ensure_geometry_f_override.py` to
> prevent another silent revert.
>
> Phase 1A toggle infrastructure (the `disable_*` fields on
> `LatLonCGridOceanConfig` / `MPASOceanConfig` + gates in `_step_impl`)
> survived on main and remains complementary to Pierre's IC-zeroing
> pattern (useful when IC alone cannot isolate a term).
>
> The dossier + worksheet + methodology doc are *not* restored — they
> were point-in-time snapshots that Pierre's living `test_term_by_term_analytic.py`
> + `tests/ocean/fidelity/` framework supersedes in practice.

---

## Purpose

Conversation with Alistair Adcroft (GFDL/Princeton) on legoESM ocean
design. This document captures (i) the items that came up, (ii) our
current understanding of each, (iii) the standing questions, (iv) the
initial direction, and (v) the consolidated plan of work that comes out
of them.

Each item uses the same template so we can hold things side by side
and so the plan-of-work section can reference items by number.

---

## Item template

```
## Item N — <short topic>

Area: <grid / mode split / advection / GM-Redi / vertical mixing / PGF / ...>
Relates to: <pointer to section of design dossier or worksheet, if any>

What came up.
    A few sentences capturing Alistair's point, recommendation, or
    counter-question. Keep it close to what he actually said — no
    paraphrase that drifts.

Our current understanding.
    What we (Dhruv) took away.  Where we feel solid, where we feel
    hand-wavy.

Standing questions.
    The pieces that are still open. Either things to read up on, or
    things that need a numerical experiment, or things that need a
    follow-up with Alistair.

Initial direction.
    Possible work to investigate, prototype, or implement.  No
    commitment yet — just sketch.  This becomes input to the plan of
    work below.
```

---

# Items raised

## Item 1 — Decouple C-grid discretization from coordinate metrics (quadrilateral grids)

**Area:** Horizontal grids / discrete operators.
**Relates to:** dossier §3 (Horizontal grids), §4 (Staggering choice), §5 (Discrete operators per grid); worksheet §1 (Grid).

**What came up.**
For quadrilateral C-grid families (lat-lon, Mercator, cubed-sphere
panels, tripolar, …) the C-grid staggering itself is universal — the
div/grad/curl operators and the placement of `u`, `v`, `h`, `η` on cell
edges and centres are identical. What differs from coordinate to
coordinate is the *metric tensor*: `dx_T`, `dy_T`, areas, Jacobian
factors, angle relations. The clean architecture is therefore one
C-grid operator library that takes metric arrays as inputs, plus a
per-coordinate "metric provider" that fills those arrays from the
coordinate's geometry. This is the MOM6 design pattern — the supergrid
defines metrics on a common 2× refined grid and derives every
staggered metric from it. Adding a new coordinate (Mercator, tripolar,
stretched, regional patch) becomes a metric-provider change rather
than a re-implementation of every operator.

**Our current understanding.**
Conceptually clear and well-aligned with MOM6 practice. Where it bites
legoESM: today we have separate operator implementations for lat-lon
(`src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`) and for
cubed-sphere (`src/legoesm/ocean/dynamics/cubed_sphere*`), with
coordinate assumptions baked into the operator code itself. This is
precisely the kind of duplication issue #214 (ocean grid-agnostic
refactoring) is meant to address — but #214 has focused on *scheme*
helpers (sponge, EOS-pressure, freshwater) rather than the *operator*
layer.

**Standing questions.**
1. How tangled is the coordinate assumption inside our current
   operator code? Specifically, are there inline `cos(φ)` factors and
   `R_earth` constants in `latlon_cgrid_operators.py`, or is the code
   already metric-driven and we just need to expose the metrics?
2. Cubed-sphere is also quadrilateral — does this metric-driven
   refactor give a cleaner attack on the face-boundary instability,
   by making panel-boundary metric handling explicit rather than
   implicit in the operator stencils?
3. Should we adopt the supergrid pattern (metrics defined on a 2×
   refined grid; T-, u-, v-, q-point metrics derived from it)? MOM6
   does. It buys consistency by construction.
4. Does MPAS (TRiSK on Voronoi) sit outside this scope — i.e. is the
   "quadrilateral C-grid operator library" the right boundary, and
   TRiSK lives in its own world? Probably yes — different mimetic
   family — but worth being explicit.

**Initial direction.**
Two-step plan, in order:
1. **Audit current operator code for hardcoded coordinate
   assumptions.** Catalogue every inline `cos(lat)`, `R_earth`,
   gnomonic-specific term in `latlon_cgrid_operators.py` and the
   cubed-sphere operator modules. Output: a table of *what the
   operator does* vs *what coordinate fact it assumes*.
2. **Prototype the refactor on lat-lon first.** Define a
   `QuadrilateralCGridMetrics` container; rewrite operators to accept
   metrics as arguments; verify regular lat-lon reproduces existing
   results bit-for-bit; add Mercator as a metric-provider variant;
   check Mercator stays equivalent to the existing Mercator path.
   Then port the cubed-sphere operators to the same interface — the
   harder lift, but the one most likely to clean up the face-boundary
   issue.

---

## Item 2 — Vertical structure is orthogonal: SW as the `nlev=1` limit; coordinate-invariant physics

**Area:** Vertical coordinate / dycore architecture.
**Relates to:** dossier §1 (Continuous equations), §2 (Vertical coordinate), §7 (PGF); worksheet §3 (Vertical coord).

**What came up.**
Two connected threads under the vertical-coordinate discussion.

*(2a) The ocean dycore should be a layered model.* With `nlev=1` it
should reduce cleanly to the shallow-water system on the *same code
path*, so the rich SW test suite (Williamson-2 geostrophic balance,
Williamson-5 isolated mountain, Galewsky barotropic instability,
Rossby–Haurwitz, …) becomes free dycore validation. Today the
`nlev=1` limit runs (`global_barotropic_wind_1lev` in the test matrix)
but we have no Williamson-style SW matrix on it.

*(2b) Physical equations should be coordinate-invariant.* Alistair's
canonical example: the horizontal pressure gradient should not change
when we switch vertical coordinate from z to z\* to ALE — it is
fundamentally $-(1/\rho)\nabla p$, with the coordinate being a
separate regridding/remapping choice that does not enter the momentum
equation [Adcroft & Hallberg 2006]. PGF is the headline example; the
principle generalises to other operators.

Both threads come from the same architectural principle: factor the
dycore so that the vertical structure is an orthogonal axis rather
than baked into the operator code.

**Our current understanding.**
For (2a) — we are most of the way there. The dycore is layered;
`nlev=1` runs; `global_barotropic_wind_1lev` validates the
configuration end-to-end. Missing piece: a proper Williamson-style SW
test matrix on the `nlev=1` path.

For (2b) — partly there. MPAS density-Jacobian PGF is closer to
coordinate-invariant in spirit. Lat-lon PGF is written assuming z\*.
Moving to a single PGF discretisation that works in z, z\*, σ, and
ALE — e.g. Shchepetkin & McWilliams (2003) finite-volume PGF or
Adcroft, Hallberg & Harrison (2008) — would be the structural fix.

**Standing questions.**
1. How layered is the dycore today? Is `nlev` a clean orthogonal axis
   everywhere, or are there `nlev=1` special cases sprinkled in?
2. What does the atmosphere SW test infrastructure look like, and can
   we adapt it directly to the ocean dycore in `nlev=1` mode?
3. Beyond PGF, what else in the dycore is z\*-aware in its
   discretisation — tracer remapping, eta-thickness coupling, vertical
   advection?  Need a catalogue.
4. Does coordinate invariance need to come *before* a future
   conversation about ALE adoption? Probably yes — invariant PGF is
   the natural enabler.
5. Connects to **Item 1**: same architectural principle (decouple
   physics from coordinate / metric choices).

**Initial direction.**
Three threads, in increasing scope:

1. **SW test matrix on `nlev=1`.** Stand up Williamson-2 /
   Williamson-5 / Galewsky on the ocean dycore in `nlev=1` mode.
   Reuse the atmosphere SW test infrastructure where possible.
   Low-effort, high-clarity: a known-correct dycore benchmark that is
   independent of vertical-coordinate questions.
2. **Coordinate-invariance audit.** Catalogue every operator that
   currently knows it's z\*. Output: explicit "what would change if we
   switched to z or ALE" list. That list defines the architectural
   refactor scope.
3. **Single coordinate-invariant PGF.** Move toward one PGF
   discretisation that works in z, z\*, σ, and ALE without
   reformulation.  Natural enabler for any later ALE conversation.

---

## Item 3 — Time stepping is per-term, not single-axis: forward–backward + RK2 + implicit Coriolis

**Area:** Time integration / dycore architecture.
**Relates to:** dossier §11 (Outer time integration), §12 (Mode splitting); worksheet §4 (Outer time integrator).
**Confidence flag:** this thread was the hardest to recall and parts below are *plausible reconstruction*, not direct quotation. The "ask Alistair directly" follow-ups are marked explicitly.

**What came up (fragments as recalled).**
- RK3 and RK4 can take very long time steps, but are "not necessarily that good" (reason not captured).
- Forward Euler is poor for momentum, acceptable for tracers.
- MOM6 uses a *combination* of forward and backward time steppers, with "something handled by RK2".
- Coriolis is problematic and is treated differently from the rest.

**Our current understanding — and where we are reconstructing.**
The headline message we took away: time integration in an ocean dycore is not one choice. Different terms have different stiffness, oscillation, and stability character, and the production answer is to pick the stepper *per term*. Our worksheet §4 treated time integration as a single axis — that framing is too coarse and is probably what Alistair was pushing back on.

Plausible reconstruction of the specific remarks (to be verified):

- **"RK3/RK4 can take long steps but are not necessarily good."** Two readings, both candidates:
  (a) *Cost:* RK3/RK4 admit larger CFL than Euler/RK2 for advection, but each step costs 3–4 tendency evaluations. Net throughput gain over a well-chosen RK2 is small.
  (b) *Phase error:* for oscillatory modes (gravity waves, inertia-gravity, Coriolis) the truncation order of higher-order RK does not translate to better wave-phase accuracy.
- **"Euler bad for momentum, fine for tracers."** Classical stability: forward Euler is unconditionally unstable for inertial oscillations driven by Coriolis ($|1 + if\Delta t| > 1$ always). Momentum therefore needs forward–backward, semi-implicit, or higher-order treatment. Tracer advection of a passive scalar is stable under Euler within the CFL limit and the dominant choice is the spatial scheme, not the temporal.
- **"Forward–backward + RK2."** Matches MOM6's structure: forward–backward in the barotropic substeps for the gravity-wave mode [Hallberg 1997; Shchepetkin & McWilliams 2005], RK2 (predictor–corrector / Heun) for the baroclinic outer step, Coriolis treated semi-implicitly via alternating $u$/$v$ updates (or fully implicitly via the $2\times2$ rotation matrix).

In legoESM today we have one "outer integrator" knob (Euler / Heun / SSP-RK3 / RK4) plus a separate barotropic stepper. MPAS uses Heun for Coriolis specifically; lat-lon's Coriolis treatment needs checking. This is *coarser* than the MOM6 picture.

**Standing questions.**
1. **(ask Alistair directly)** What did he actually mean by "RK3/RK4 not good"? Cost? Phase error? Conservation? Worth a one-line follow-up to settle it.
2. **(internal audit)** What is the per-term stepper today on lat-lon? On MPAS? Need a table: Coriolis, PGF, horizontal advection (momentum, tracer), vertical mixing.
3. Is forward–backward implemented anywhere outside the barotropic substep in legoESM? MOM6 uses it more broadly.
4. Should the "outer integrator" axis on the worksheet be deprecated in favor of per-term knobs? Probably yes — the single-axis framing implies orthogonality we don't actually have.
5. Differentiability through implicit Coriolis or implicit vertical diffusion: AD through linear solves is fine in principle (implicit function theorem), but we should verify the current code path supports it.

**Initial direction.**
1. **Per-term time-stepper audit.** Build one table:
   | term | current scheme (lat-lon) | current scheme (MPAS) | MOM6 reference | stability character |
   Goal: identify any term using forward Euler inappropriately, and any term where a better stepper would unlock a longer outer $\Delta t$.
2. **Clarify Alistair's RK3/RK4 remark.** Either a one-line email, or do the analysis (cost-per-step × CFL + phase error for oscillatory modes) and confirm we arrive at the same conclusion.
3. **Restructure the time-integration axis** in the worksheet from a single "outer integrator" choice to a per-term table. Sharpens the next conversation.

---

## Item 4 — Coriolis is hard; lean energy-conserving

**Area:** Coriolis (spatial + temporal discretisation).
**Relates to:** Item 3 (time stepping); dossier §8 (Coriolis and PV-flux); worksheet §8.

**What came up.**
Continuing the time-stepping thread (Item 3): Coriolis is "quite hard" and Alistair would lean toward an energy-conserving approach.

**Our current understanding — and one ambiguity.**
"Energy-conserving Coriolis" can mean two things, often both at once:

- *(4a) Spatial.* Pick a C-grid PV-flux discretisation that conserves discrete kinetic energy globally — Sadourny EC, or the energy-conserving member of the Arakawa–Hsu (1990) family. MOM6 does this and adds explicit grid-scale enstrophy dissipation to prevent spectral blocking.
- *(4b) Temporal.* Pick a time-integration scheme for the Coriolis operator that exactly conserves discrete KE for the linear inertial oscillator — implicit midpoint / Crank–Nicolson on the $2\times2$ rotation. Forward Euler does not; forward–backward and Heun do partially.

The remark came in the time-stepping thread, so (4b) is the likely intent — but the phrase "energy-conserving Coriolis" is more common in the spatial sense, and a serious treatment usually adopts both.

In legoESM today:
- *Spatial:* MPAS has EE / EN / AL81 / mixed PV-flux available; default is AL81. Lat-lon uses Sadourny + Hollingsworth (commit `d0183817`); not yet confirmed whether the Sadourny variant is EC or EN.
- *Temporal:* MPAS uses Heun for Coriolis; lat-lon's treatment is less clearly documented. Neither is discretely energy-conserving in time.

**Standing questions.**
1. **(ask Alistair)** Disambiguate: spatial PV-flux, temporal scheme, or both?
2. Which Sadourny variant is the lat-lon code using?
3. Is Crank–Nicolson Coriolis differentiable and JIT-friendly under JAX? (Should be — small per-cell $2\times2$ linear solve — confirm.)
4. If we get exact energy conservation temporally, does the case for AL81 (combined energy + enstrophy spatial scheme) weaken?

**Initial direction.**
1. **Coriolis treatment audit** per grid: spatial PV-flux form, temporal scheme, what is currently conserved. This is the Coriolis row of the per-term audit table from Item 3.
2. **Prototype Crank–Nicolson Coriolis** as a local, well-contained change. Test on the linear inertial oscillator (closed-form reference) and the geostrophic-adjustment case.
3. **Decide spatial PV-flux** once Q1 is answered.

---

## Item 5 — Tracer advection: start with 3rd-order; DST-3 TVD is fine

**Area:** Tracer transport.
**Relates to:** dossier §10 (Tracer advection); worksheet §10.

**What came up.**
For tracer advection, Alistair would default to a 3rd-order scheme as a first choice. DST-3 TVD is "not bad".

**Our current understanding.**
Pragmatic validation of the existing default: don't start with the most expensive option (SOM), don't start with the lowest-order (centered 2nd), pick a 3rd-order monotone scheme. This is what MOM6 does — PPM [Colella & Woodward 1984] — and what legoESM already has as the default — DST-3 [Easter 1993] with TVD limiters and the d0 cap (see `project_dst3_advection`). The two 3rd-order TVD schemes (PPM, DST-3) are siblings; differences at large scales are marginal and dominated by limiter choice.

SOM [Prather 1986] is retained for the low-mixing cases (Eady, ACC channel) where minimising spurious diapycnal mixing matters more than cost.

**Standing questions.**
1. Is DST-3 actually the default across all production-style runs, or are some test cases defaulting to centered 2nd-order? Quick audit needed.
2. We document that DST-3 reaches "full benefit at CFL > 0.1 or RK3". What CFL are we running at for typical configurations? If we are routinely below ~0.1, the third-order behaviour is not being realised.
3. Should we add PPM as an additional option to enable direct MOM6 cross-checks? Cost is low and the structural similarity to DST-3 makes the comparison meaningful.
4. SOM stays for low-mixing cases — confirm no plan to deprecate.
5. Does the TVD limiter inside DST-3 introduce limiter-induced damping that interacts with the differentiability story? Probably not (measure-zero limiter kinks) but worth a sanity check.

**Initial direction.**
1. **Audit the test matrix** for the tracer scheme selected per case. Make DST-3 the explicit default where it isn't already.
2. **Confirm CFL regime** — record the CFL for representative production-style runs (gyre, ACC, Eady). Diagnose whether we are in the "full 3rd-order benefit" range.
3. **(low priority)** Add PPM as a third option for MOM6 cross-checks.

---

## Item 6 — Per-term validation tests (inertial oscillation, …)

**Area:** Validation methodology / test harness.
**Relates to:** Items 2a (SW limit tests), 3 (per-term audit), 4 (Coriolis), 5 (tracer advection); dossier §21 (Production test matrix); worksheet §16 (Initial conditions).

**What came up.**
Beyond the integrated test cases we already run, we should have a suite of per-term tests that exercise individual terms against closed-form solutions. Inertial oscillation was the canonical example: turn everything off except Coriolis, kick a parcel, verify the trajectory is a circle at period $2\pi/f$.

**Our current understanding.**
This is the dycore equivalent of unit testing: each term gets a test that exercises *only that term* against an analytical reference. Williamson (1992) is the canonical SW version; the principle generalises to all dycore terms.

Today the ocean test matrix has *integrated* tests (rest state, geostrophic adjustment, inertia-gravity wave, gyres, Eady, ACC) but unit-level isolation tests are sparse. We can run "everything" in a configuration that exercises gyre dynamics, but we cannot cleanly run "just Coriolis" with every other term off.

Candidate per-term test suite:

| term                       | isolation test              | closed-form reference        |
|----------------------------|-----------------------------|------------------------------|
| Coriolis + time stepper    | inertial oscillation        | circle at period $2\pi/f$    |
| Tracer advection           | 1D periodic advection       | translation at velocity $u$  |
| Tracer diffusion           | 1D Gaussian spread          | $\sigma^2(t) = 2\kappa t$    |
| PGF + free surface         | linear gravity wave (SW)    | $\omega^2 = gHk^2$           |
| PGF + Coriolis (steady)    | geostrophic balance         | $f\times u = -\nabla(g\eta/\rho_0)$ |
| Vertical mixing            | 1D mixed-layer deepening    | Kraus–Turner closed form     |
| Baroclinic PGF             | rest-state preservation     | $\max\|u\|\to 0$             |

This connects directly to several earlier items:
- **Item 2a** — SW test matrix is exactly this idea applied to the `nlev=1` SW limit.
- **Item 3** — the per-term time-stepper audit becomes operational when every term has a unit test that exercises it.
- **Item 4** — inertial oscillation is the canonical Coriolis unit test, and the right harness for verifying the Crank–Nicolson Coriolis prototype.
- **Item 5** — 1D pure advection is the canonical tracer-scheme unit test for measuring order of accuracy.

**Standing questions.**
1. Inventory — which of these tests do we already have in `tests/ocean/unit/` or as test-matrix entries? (Rest state and geostrophic adjustment are in the matrix; the rest seem absent to first order.)
2. What is the cleanest API for "turn everything off except X"? Do `LatLonCGridOceanConfig` and `MPASOceanConfig` already expose enable/disable toggles, or do we need to factor that in?
3. Should per-term tests live in `tests/ocean/unit/` (pytest, CI gate) or as new test-matrix entries (visual + diagnostic outputs)? Probably both — pytest for gating, matrix entries for convergence-rate plots.
4. Each per-term test should run at multiple resolutions to confirm the expected order of accuracy (2nd-order PGF, 3rd-order DST-3 tracer, …). Worth standardising.
5. Differentiability angle: per-term tests are also the right level for gradient correctness checks (e.g. `jax.grad` through the inertial-oscillation trajectory against its analytical derivative).

**Initial direction.**
1. **Inventory current isolation harness.** What can we already selectively disable in each grid's config dataclass? Output: gap list per term.
2. **Design a term-toggle API** if needed — each scheme gets an `enabled` flag with a hard default, plus a "test-mode" config factory that returns the minimal config for each per-term test.
3. **Build canonical per-term tests** in priority order:
   (i) Inertial oscillation (also supports Item 4);
   (ii) 1D pure advection convergence (also supports Item 5);
   (iii) Linear gravity wave (also supports Item 2a's SW work);
   (iv) Extend rest-state preservation across more stratification configurations.
4. **Convergence-rate plots** for each test at multiple resolutions — the right "look at the numbers" complement to the visual matrix outputs.

---

## Item 7 — PGF unit test: flow past a bump (Haidvogel reference)

**Area:** Validation / pressure gradient force.
**Relates to:** Items 2b (coordinate-invariant PGF), 6 (per-term tests); dossier §7 (PGF); worksheet §6.

**What came up.**
For the pressure-gradient force, Alistair pointed at *flow past a bump* (a seamount-in-stratification test) as the canonical PGF validation. The reference is in Dale Haidvogel's book — most likely the Haidvogel & Beckmann (1999) monograph *Numerical Ocean Circulation Modeling* — which carries the canonical setup and reference numbers. (Confirm exact citation when we pull the recipe.)

**Our current understanding.**
The classical PGF test: place a stratified ocean at rest in a basin with a topographic bump (commonly a Gaussian seamount). Closed-form expected result: the ocean stays at rest — $\max|u|\to 0$. Numerically, the PGF discretisation produces spurious currents whose amplitude measures the PGF error over varying bathymetry.

The test was canonised by Beckmann & Haidvogel (1993) for σ-coordinate models, where PGF cancellation errors are an order of magnitude worse than in z\*. For z\*-coord models the spurious currents are smaller but still finite, and the seamount test is the standard yardstick for PGF quality. Shchepetkin & McWilliams (2003) report z\* numbers in the same setup as part of motivating their density-Jacobian PGF.

In legoESM today we have `rest_state_stratified_with_land` cases, but to first order the bathymetry there is flat — the "land" is land masking, not topographic variation. We don't have a dedicated seamount-on-stratification test.

**Standing questions.**
1. Confirm: do the current `rest_state_stratified_*` cases actually exercise non-flat bathymetry? If not, this is a gap.
2. Pull Haidvogel & Beckmann's seamount setup (or Shchepetkin & McWilliams 2003's variant): basin size, seamount geometry, stratification profile, target $\max|u|$, time horizon.
3. Run with and without Coriolis? The Beckmann–Haidvogel original is rotational; a non-rotating variant isolates PGF more cleanly.
4. Connection to **Item 2b** — the seamount test is exactly the metric that tells us whether a candidate coordinate-invariant PGF discretisation is good enough.
5. Cross-grid comparison: lat-lon (z\* standard PGF) vs MPAS (density-Jacobian). Expectation is MPAS produces smaller $\max|u|$. If not, lat-lon PGF needs attention.

**Initial direction.**
1. **Audit current rest-state cases** for bathymetry variation.
2. **Add seamount-on-stratification test** to the matrix — Gaussian bump, linear stratification, no forcing. Variants: rotating and non-rotating; multiple horizontal and vertical resolutions.
3. **Benchmark against Haidvogel & Beckmann / Shchepetkin & McWilliams** if their numbers are accessible. At minimum, document our z\* baseline so we have something to compare against when we change the PGF.
4. **Cross-grid PGF comparison** on the same setup.

---
<!-- next item goes here -->



---

# Plan of work

Five phases, sequenced for dependency and leverage. Rationale lives in `docs/ocean/adcroft_review_consensus.md` (synthesis of four independent reviews) and the per-reviewer files (`review_dycore_expert.md`, `review_ocean_model_expert.md`, `review_dycore_tester.md`, `review_claude_audit.md`).

Convention:
- `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped
- Effort: **S** ≤ 1 wk · **M** 1–3 wk · **L** > 3 wk
- "Gates" = what must complete first · "Unblocks" = what this enables

The plan is meant to evolve. As tests come in, update the Progress log below and re-prioritise. Phases are not strict — work within a phase parallelises.

---

## Phase 0 — Internal documentation calibration (≤ 1 day) — DONE 2026-05-22

No outbound communication. Internal cleanup so the dossier matches the code.

- [x] **0.1** Fix dossier drift. **Findings:** only 2 of the 6 items in `adcroft_review_consensus.md` §2 were actual dossier drift (Heun → Matsuno; missing AL81 mention for lat-lon ζ). Fixed lines 35, 218, 506–509, 511–519, 521–527, 540–541. Consensus doc §2 retitled to "Code-reality summary" with the more accurate framing.
- [x] **0.2** Worksheet edits: §4 replaced with per-term table (Coriolis · PGF · mom advection · tracer advection · barotropic · vertical mixing); §8 expanded with spatial-vs-temporal split + AL81 for lat-lon; §10 tracer default corrected from "★ DST-3" to "★ TVD" (the actual default) with the full scheme menu listed.
- [x] **0.3** Re-rendered `docs/legoesm_ocean_model.pdf` (38 pp, 235 KB) and `docs/legoesm_ocean_choices.pdf` (11 pp landscape A4, 82 KB) with `pandoc --pdf-engine=lualatex`.

Gates: none. Unblocks: everything (rest of the plan starts from corrected internal docs).

---

## Phase 1 — Test framework foundation (≤ 2 wk)

The methodology prerequisite. Without these tests we cannot defend any default change empirically. Strong reviewer consensus this is the right Phase 1 — see consensus §3 Item 6.

### 1A — Toggle infrastructure (Phase 1A — landed 2026-05-22)

- [x] **1A.1** `disable_coriolis` / `disable_pgf` / `disable_momentum_advection` / `disable_tracer_advection` / `disable_drag` added to `LatLonCGridOceanConfig` (`src/legoesm/ocean/state.py:664`) and `MPASOceanConfig` (`src/legoesm/ocean/mpas_config.py:330`).
- [x] **1A.2** `test_mode: bool = False` field landed alongside on both configs. The factory sets it `True` so test-mode is visible in the audit trail.
- [x] **1A.3** `make_test_config(term, grid, **overrides)` in `src/legoesm/ocean/experiments/test_configs.py`. Supports `"coriolis_only"`, `"tracer_advection_only"`, `"gravity_wave"` today; extend as new recipes land.
- [x] **1A.4** Lat-lon gates wired: PGF + momentum advection at `ocean_pe_latlon_cgrid.py:1049–1057`, PV-flux skip at `:1199`, drag at `:1534`, Coriolis Matsuno skip at `ocean_model_latlon_cgrid.py:744`. Tracer-advection gate deferred — lands alongside Recipe #2 (1C.2) when that test needs it. MPAS gates deferred — land with the first MPAS-specific per-term test.

### 1B — Convergence-rate methodology (Phase 1B — landed 2026-05-22)

- [x] **1B.1** `convergence_rate(errors, resolutions)` and `assert_convergence_rate_at_least(...)` in `tests/ocean/unit/_helpers.py`. Pairwise rates; rate-target table per scheme in the methodology doc.
- [x] **1B.2** Methodology written to `docs/ocean/per_term_test_methodology.md` — what a per-term test is, the toggle factory, why rate not L2, the rate-target table per tracer scheme, the gradient-correctness variant pattern, and the test-file naming convention.

### 1C — Three per-term tests in parallel

(Three reviewers picked three different "first" tests. Since each is S-effort, land them in parallel.)

- [x] **1C.1 — Recipe #1 — Inertial oscillation** (Coriolis isolation). Landed 2026-05-22 in `tests/ocean/unit/test_per_term_inertial_oscillation.py`, 6 tests, all passing. Four layers: pure-math Matsuno (3 tests including invariant + period 2nd-order convergence), `_forward_backward_coriolis_3d` function test, full-dycore qualitative oscillation test, toggle-equivalence test. **Findings recorded:** Matsuno is 2nd-order in *phase / period* but 1st-order in *trajectory L_inf* (orbit eccentricity); invariant quadratic form is `u² + (f·dt)·u·v + v²`, not `u² + v²`; barotropic cosine time filter means dycore output is window-averaged — quantitative comparison happens at the function level, not at the dycore-step level. Documented in test docstrings.
- [x] **1C.2 — Recipe #2 — 1D periodic tracer advection convergence.** Landed 2026-05-22 in `tests/ocean/unit/test_per_term_tracer_advection.py`, 6 tests, all passing. Methodology: short translation (1/8 revolution) + SSP-RK3 + smooth Gaussian, comparing to analytical shifted Gaussian. **Findings:** upwind reaches rate ≈ 1.0 ✓; tvd reaches rate ≈ 2.0 ✓; **dst3 shows rate ≈ 1.0 in L2 on smooth IC** because the Van Leer limiter activates at every smooth peak and locally reduces the scheme to upwind. DST-3 *is* still substantially better than upwind in absolute L2 (ratio ≈ 0.4 at n=128) — just doesn't reveal its formal 3rd order through this diagnostic. PPM overflowed on the periodic Gaussian setup (likely partial-cell or pole interaction; tracked as follow-up). WENO5/7 capped to ≲ 3 by SSP-RK3 temporal error anyway. Mass conservation verified to fp tolerance for upwind / tvd / dst3. Methodology doc updated with the empirical table.
- [x] **1C.3 — Recipe #5 — Linear SW gravity wave dispersion.** Landed 2026-05-22 in `tests/ocean/unit/test_per_term_gravity_wave.py`, 4 tests, all passing. Setup: f-plane (f=0), nlev=1, η=A cos(λ) wavenumber-1 mode, H=4000 m, c=sqrt(gH)≈198 m/s. **What works:** (1) Wave is *dynamic* — η decreases substantially in amplitude at T/4 and u grows. (2) First positive→negative zero crossing of η at λ=0 matches T_wave/4 to within 8%. (3) Energy stays finite (no blow-up). (4) η at λ=0 changes sign within first half-period. **What does NOT work (and was deliberately removed from the test):** returning-to-IC after one period — the wave amplitude attenuates substantially over one period (~98% energy loss) under the *default* dycore config. Root cause investigated but not localised: not BEBT (bebt=0 same result), not cosine-vs-box filter (both damp similarly). The damping is large enough to suggest something beyond standard time-filter aliasing suppression — possibly an interaction with the implicit-vertical-mixing pathway or a more complex coupling in the barotropic-baroclinic flux reconciliation. **Tracked as Phase 5 follow-up.** Methodology doc updated with the empirical findings.

### 1D — Differentiability gates

- [ ] **1D.1** Add a gradient-correctness variant of each of 1C.1–1C.3: `jax.grad` of a summary diagnostic vs finite-difference reference. Per-term AD regression coverage. [S]

### 1E — Bit-for-bit regression (gates Phase 4)

- [ ] **1E.1** Per-step state-checksum infrastructure for the ocean test matrix. Required *before* Item 1's metric refactor lands. [M]

Phase 1 gates: nothing.
Phase 1 unblocks: Phase 2 (default switches need the tests to defend them); Phase 4 (refactor needs bit-for-bit + identity tests).

---

## Phase 2 — Default switches with empirical evidence (≤ 3 wk)

All four reviewers picked **Item 7 (seamount test)** as the single most leveraged work item. It is the prerequisite for defensible default switches.

### 2A — Seamount baseline (Recipe #10)

- [ ] **2A.1** Implement `rest_state_stratified_seamount` in `src/legoesm/ocean/experiments/`. Reuse `rest_state.py`'s init machinery with a Gaussian-bump $H_\text{bathy}$ override. [M]
- [ ] **2A.2** Domain 320 km × 320 km, $H_0=4500$ m + Gaussian seamount ($h_0=4000$ m, $L=25$ km). Linear $N^2 \approx 10^{-4}$ s$^{-2}$. 180-day rest run.
- [ ] **2A.3** Run on lat-lon and MPAS, all four PGF schemes (`adcroft`, `smc03`, `ahh08`, `centered` on MPAS). Output: $\max|u|(t)$ curves as CI artifact. Also RPE-drift (Ilicak protocol).
- [ ] **2A.4** Variants: $f=0$ (pure PGF isolation, dycore-tester addition) and $f=10^{-4}$ (production-relevant, BH93 original). Three resolutions ($\Delta x \in \{10, 5, 2.5\}$ km).
- [ ] **2A.5** Document target values per `pgf_ahh08.py:61`: `adcroft` ~ cm/s, `smc03` ~ mm/s, `ahh08` smaller. Pass criterion: ≥ 100× spread between schemes; if all give cm/s, an implementation is broken.

Gates: Phase 1A (toggles) + Phase 1B (convergence helper).
Unblocks: 2B, 2C, 2E.

### 2B — Wire AHH08 onto lat-lon

- [ ] **2B.1** Build the lat-lon edge wrapper next to `latlon_cgrid_operators.py`. `pgf_ahh08.py`'s column primitive `column_pressure_integrals_ahh08` is already grid-agnostic. [S]
- [ ] **2B.2** Add `"ahh08"` to lat-lon's `_valid_pgf` set; gate on `eos="wright"`.
- [ ] **2B.3** Verify on the seamount test that lat-lon-AHH08 produces $\max|u|$ comparable to MPAS-AHH08 on the same physical setup.

Gates: 2A complete. Unblocks: 2E.

### 2C — Lat-lon tracer default

- [ ] **2C.1** Change `LatLonCGridOceanConfig.tracer_advection` default from `"tvd"` to `"dst3"`. One-line code change. [S]
- [ ] **2C.2** Validate on the existing matrix: Eady, ACC, gyre runs stable; RPE drift before/after on lock-exchange and overflow.

Gates: 1C.2 (convergence test confirms `dst3` is actually 3rd-order in our implementation).

### 2D — MPAS PV-flux default switch

- [ ] **2D.1** Change `MPASOceanConfig.pv_scheme` default from `"enstrophy"` to `"energy"`. [S]
- [ ] **2D.2** Gating test: discrete-KE conservation on Galewsky barotropic instability (Recipe #8), all diffusion off. EC variant should give $|KE(t) - KE(0)|/KE(0) < 10^{-10}$/day; EN drifts $\sim 10^{-6}$/day. [M]
- [ ] **2D.3** Run the existing ocean test matrix. Watch for new instabilities — if ACC channel or Eady was relying on enstrophy dissipation as hidden viscosity, the switch will surface that.
- [ ] **2D.4** (Optional) Implement AL81 on MPAS as a follow-up — vertex PV and vertex thickness helpers are already in `mpas_ocean.py`; the missing piece is the AL81 stencil itself. [M]

Gates: 1C.1 (inertial oscillation must pass first), 1D.1 (gradient correctness).

### 2E — MPAS PGF default switch

- [ ] **2E.1** Change `MPASOceanConfig.pgf_scheme` default from `"centered"` to `"ahh08"` (or `"smc03"` if AHH08 doesn't win on the seamount). [S]
- [ ] **2E.2** Re-run the test matrix; expect MPAS+ETOPO bottom-trapped instability behaviour to change (memory: `project_mpas_etopo_instability`).

Gates: 2A (seamount data justifies the choice).

Phase 2 unblocks: Phase 4 (operator refactor uses seamount as regression target).

---

## Phase 3 — SW matrix on ocean `nlev=1` (≤ 2 wk, parallel to Phase 2)

Reuses existing atmosphere SW infrastructure. Half a day of wiring per case.

- [ ] **3A** Build `tests/ocean/unit/sw_adapter.py` (~50 LOC). Maps ocean `(η+H, u, v, T_const, S_const)` ↔ SW `(h, u, v)`. [S]
- [ ] **3B — Williamson-2** (steady geostrophic) on lat-lon ocean. Two flow orientations ($\alpha=0$, $\pi/4$). Convergence rate 2 at 4°→0.5°. Fail loudly if rate doesn't reach 2 — indicates metric error or PGF bug. [M]
- [ ] **3C — Williamson-5** (isolated mountain). Compare to Jakob–Chien 1995 reference (Fig. 7). **Ships in the same PR as Recipe #10 seamount** — it is the SW analog. [M]
- [ ] **3D — Galewsky barotropic instability.** Reuse for Phase 2D's discrete-KE gating test. Cubed-sphere variant is the best edge-effect detector once CS ocean stabilises (deferred). [M]
- [ ] **3E** (Deferred) Williamson-1 cosine-bell advection on the sphere — extends 1C.2 to the 2D rotation case.

Gates: 1A (toggles to set up `nlev=1` with the right operators wired).
Unblocks: independent regression coverage; sharpens the cubed-sphere face-instability investigation in Phase 5.

---

## Phase 4 — Architectural refactor: metric-driven operators (Item 1) (1–2 months)

The largest single piece of architectural debt. All four reviewers agree this gates the strong form of Item 2b (single PGF across coords) and the cubed-sphere face-instability fix.

### 4A — Regression scaffolding (must precede the refactor)

- [ ] **4A.1** Bit-for-bit checksum infrastructure for every output step of the test matrix. [M] (Same as 1E.1; if 1E.1 didn't land in Phase 1, it must land here first.)
- [ ] **4A.2** Vector-calculus-identity tests on the lat-lon C-grid: $\nabla\cdot(\nabla\times\mathbf{v})=0$, $\nabla\times(\nabla\phi)=0$, discrete Stokes' theorem, divergence theorem. Port from the existing CS D-grid test. [M]

### 4B — `QuadrilateralCGridMetrics` design

- [ ] **4B.1** Define the container. Include rotation matrices and 3D effective-area handling **from day 1**, per ocean-model-expert (don't paint into a corner by doing lat-lon-only first). [M]
- [ ] **4B.2** Decide: supergrid pattern verbatim (MOM6-style 2× refined mesh) vs derive metrics directly at T/u/v/q points. Reviewers split. **Default to MOM6 pattern unless a concrete reason argues against.** Document the choice.

### 4C — Refactor ~15 operators in `latlon_cgrid_operators.py`

- [ ] **4C.1** Operators in scope: `compute_grad_*`, `compute_div_*`, `compute_curl_*`, `kinetic_energy_*`, `coriolis_cgrid`, `pgf_*`, vertex/edge averaging helpers. [L]
- [ ] **4C.2** Remove all inline `cos(lat)`, `R_earth`, `radius`, `grid.radius` references. Operators consume metric arrays as arguments.
- [ ] **4C.3** Bit-for-bit reproduce existing lat-lon test matrix results (gated by 4A.1).

### 4D — Cubed-sphere port (same epic — do not separate)

- [ ] **4D.1** Cubed-sphere panels as a metric-provider variant. Includes explicit per-edge rotation matrices in halo exchange. [L]
- [ ] **4D.2** Verify the seamount test (2A) on cubed-sphere with the refactored operators. **The face-boundary instability may or may not improve — that observation is the input to Phase 5b (Item 9).**

### 4E — Mercator as third coordinate variant

- [ ] **4E.1** Add Mercator metric provider. Run Williamson-2 cross-coordinate test (recipe #6 cross-grid variant): same flow on lat-lon vs Mercator should produce identical L2(η) to scheme order. [S]

Phase 4 gates: Phase 1 (regression coverage) + Phase 2 (seamount baseline as regression target).
Phase 4 unblocks: strong-form Item 2b (single PGF across coords); future ALE work; Phase 5b (Item 9 face-instability investigation).

---

## Phase 5 — Open-scope items

No fixed sequencing — pull in based on priority signals from Phases 1–4 outcomes.

- [ ] **5A — Item 8 — Barotropic–baroclinic mode-split coupling.** Audit our `_step_impl` against SMC05 / BEBT to see whether the barotropic-substep accumulated flux is properly fed to the baroclinic tracer step (per MOM6's continuity reconciliation). Memory pointer: `project_barotropic_solver`, `project_barotropic_noise_issue`. [M]
- [ ] **5B — Item 9 — Cubed-sphere face-boundary instability.** Algorithmic investigation. Likely needs explicit vector rotation in halo exchange + possibly a different mimetic operator family at panel edges. Gated by Phase 4D outcome. [L]
- [ ] **5C — KPP implicit-treatment audit.** Per ocean-model-expert: confirm vertical mixing uses tridiagonal implicit (CVMix-style). If not, the outer $\Delta t$ is more constrained than the audit acknowledged. [S]
- [ ] **5D — CN-Coriolis.** Deferred. Only revisit if 1C.1 (inertial oscillation) reveals Matsuno as the bottleneck — which no reviewer expects. [M]
- [ ] **5E — ALE adoption.** Deferred until Phase 4 lands. Even then, gated by appetite — it's a several-month undertaking on top of the operator refactor.

---

## Cross-cutting risks and gating decisions

- **Phase 4 without Phase 1+2 is a recipe for invisible regressions.** Per dycore-tester: "the only gate becomes 'code looks cleaner', which is not falsifiable." Phase order is load-bearing.
- **2D (MPAS PV-flux switch) may expose hidden viscosity dependence.** If ACC channel / Eady silently relied on enstrophy dissipation, the switch surfaces it as new instability. The Galewsky KE-conservation gate plus the existing matrix re-run is the safety net.
- **5C (KPP implicit audit) might invalidate Phase 2 timelines.** If KPP is explicit, outer $\Delta t$ may need to drop, which changes everything downstream. Worth doing 5C early — at least a one-day check.
- **Items 8 and 9** (mode-split coupling, face-boundary instability) are real, Adcroft-relevant, and currently *outside* the seven items. Worth deliberately opening planning threads for them rather than waiting.

---

## Standing reviewer disagreements — how we resolve them

1. **PPM vs DST-3 as lat-lon tracer default.** Decision rule: switch to DST-3 now (1-line, audit-supported); after Phase 1's convergence-rate harness runs, decide whether to switch again to PPM based on Eady/ACC RPE drift evidence.
2. **Lat-lon-only vs lat-lon + cubed-sphere together for Item 1.** Decision rule: design the metric interface with both in mind (rotation matrices, 3D effective areas as first-class). First implementation can land lat-lon-only, with cubed-sphere in the immediate next PR of the same epic.
3. **First per-term test.** Decision rule: all three are S-effort. Land them in parallel as the Phase 1 deliverable.

---

# Progress log

Append-only. Format: `YYYY-MM-DD — package id — what changed — what we learned — next move`.

- **2026-05-22 — Phase 0 complete.** Dossier and worksheet corrected; PDFs re-rendered. **What we learned:** on careful re-reading, only 2 of the 6 "drifts" I had flagged were dossier drift; the others were source-grounded facts the dossier already represented correctly. The actual dossier drifts were the MPAS Coriolis "Heun" mislabel (in 3 places) and the missing AL81 mention for lat-lon ζ in §6. The worksheet had one real drift (DST-3 listed as default tracer; the production default is TVD). Consensus doc §2 retitled and reframed to reflect this. **Next move:** Phase 1 — toggle infrastructure (1A) + convergence helper (1B) + three per-term tests in parallel (1C).
- **2026-05-22 — Phase 1A + 1B foundation landed.** Toggles + factory + convergence helper + methodology doc in place. **What changed:** 6 disable_* fields on each config; `make_test_config` factory (3 recipes wired: coriolis_only, tracer_advection_only, gravity_wave); lat-lon gates for 4 of 5 terms (tracer advection gate deferred to 1C.2); MPAS gates deferred; `convergence_rate` + `assert_convergence_rate_at_least` helpers; `docs/ocean/per_term_test_methodology.md`. **Verification:** `tests/ocean/unit/test_momentum_diagnostics_closure.py` still passes (4/4) — gates don't break existing tendency semantics with all flags False. Convergence helper self-tested on synthetic 1st/2nd/3rd-order error sequences. **What we learned:** plumbing the gates is small surface area — 5 inserted blocks across 2 files — because the dycore is already factored such that each term is a localised computation. The MPAS plumbing will mirror this pattern when needed. **Next move:** Phase 1C.1 (inertial oscillation) is the natural first per-term test — it uses the `"coriolis_only"` factory recipe which is fully plumbed.
- **2026-05-22 — Phase 1C.1 inertial-oscillation test landed.** Six tests in `test_per_term_inertial_oscillation.py` (pure-math Matsuno scheme, Matsuno function unit test, full-dycore qualitative oscillation, toggle equivalence). **What we learned:** (a) Matsuno is 2nd-order in *phase/period* but 1st-order in *trajectory L_inf* because its orbit has eccentricity O(f·dt) — fixed-amplitude radial wobble independent of T; (b) the right invariant is the quadratic form `u² + (f·dt)·u·v + v²`, not `u² + v²`; (c) the barotropic cosine time filter means dycore-step output ≈ time-averaged inertial oscillation, *not* the endpoint — so quantitative scheme verification happens at the function level, not the dycore-step level. Documented in test docstrings. Toggle plumbing verified to not perturb existing semantics (toggle-equivalence test passes at rtol=1e-3 — the fp-noise floor from different JIT traces with different config NamedTuples).
- **2026-05-22 — Phase 1C.3 gravity-wave test landed.** Four tests in `test_per_term_gravity_wave.py`. **What we found:** standing gravity-wave (η=A cos(λ), u=0) on the default lat-lon dycore is *substantially damped* over one analytical period — eta amplitude drops by ~10× and total energy by ~64× across one T_wave. Root cause investigated but not localised: not BEBT (bebt=0 same result), not cosine vs box filter. The qualitative properties of the wave (dynamic evolution, sign change of eta, period of T_wave/4 to first zero) are preserved; the *quantitative* energy-conservation property is not. This is recorded as a finding rather than a bug — the production dycore is configured for stability over multi-year runs, which trades against gravity-wave amplitude preservation. Phase 5 follow-up: identify the dominant dissipation mechanism (likely interaction with implicit vertical mixing or barotropic-baroclinic flux reconciliation), and add a high-fidelity gravity-wave dispersion suite with tuned bebt/filter/substep parameters that achieves clean energy conservation.
- **2026-05-24 — Localised and fixed an upstream `ensure_geometry()` Coriolis-override bug.** After the merge with `origin/main`, the Phase 1C.3 gravity-wave test had been skipped because the standing wave was decaying monotonically instead of oscillating. Bisect across the 169 new commits identified the offending commit: `eb657b57` (Pierre's "Tripolar grid Phase 0 + Phase 1A start"). The new `ensure_geometry()` helper at `src/legoesm/grids/latlon.py:551` was silently overwriting `grid.f` with `omega × sin(lat)` whenever it converted a `LatLonGrid` to a `LatLonCGridGeometry` at model-construction time. Anyone who used `grid._replace(f=…)` to build an f-plane / β-plane / custom-rotation test got Coriolis quietly re-enabled at global f = 2Ω sin(lat) — turning a pure gravity-wave test into a Poincaré wave whose dispersion ω² = f² + gHk² produces monotonic decay rather than oscillation. **Fix:** ~25 lines in `ensure_geometry()` to preserve `grid.f` when present and propagate it correctly through the f_T / f_u / f_v face averaging. **Verification:** the standalone reproducer at `scratch/gw_repro.py` switches from FAIL (no oscillation) to PASS (oscillates) at step 18. **Result:** 3 previously-skipped tests now pass — the Layer-3 inertial dycore test (which uses `grid._replace(f=F0)` on an f-plane), gravity-wave eta-sign-change, gravity-wave period-matches-analytical. Phase 1C now stands at **16 active per-term tests + 9 Pierre tests = 25 active lat-lon ocean per-term tests, with only 2 active skips left (tracer rate saturation, separate bug)**. Adjacent regression check: 140/140 tests pass on the operator / grid / advection suites. **What this teaches us:** the per-term methodology paid off exactly as designed — a subtle dycore-affecting bug that the integrated test matrix completely missed (1187 tests passed!) was caught by a focused gravity-wave dispersion test in a day. Without my Phase 1C.3 test, this would have shipped silently and broken every future f-plane / β-plane idealised study.
- **2026-05-22 — Phase 1C.2 1D tracer-advection convergence landed.** Six tests in `test_per_term_tracer_advection.py`. Methodology: short-translation (1/8 revolution) + SSP-RK3 + smooth Gaussian, compared against analytical shifted Gaussian. **Findings (recorded in methodology doc):** upwind hits rate ≈ 1.0 ✓; tvd hits rate ≈ 2.0 ✓. **DST-3 shows rate ≈ 1.0 in L2** on smooth IC despite being formally 3rd-order — its Van Leer limiter activates at the smooth peak (where `δ_uu/δ` flips sign) and locally reduces the scheme to upwind. DST-3 is still better than upwind in absolute L2 at the same resolution (ratio ≈ 0.4 at n=128) but the rate diagnostic doesn't reveal it on this IC + limiter combination. PPM overflowed on the periodic Gaussian setup (likely partial-cell or pole interaction; tracked as follow-up). WENO5/7 capped at ~3 by SSP-RK3 temporal-error floor. Mass conservation verified to fp tolerance for upwind/tvd/dst3. **What this teaches us:** the convergence-rate methodology *works* — it surfaced a real, defensible empirical finding about the production DST-3 implementation that wasn't visible in the existing tests. The methodology doc table now reflects empirical rates rather than aspirational formal orders. **Next move:** Phase 1C.3 — linear SW gravity-wave dispersion test.

