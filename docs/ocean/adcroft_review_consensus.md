# Adcroft follow-ups — multi-agent review consensus

**Status:** synthesis of four independent code-grounded reviews — three persona-driven agents (dycore-expert, ocean-model-expert, dycore-tester) plus an in-conversation audit. All four reviewers read the same primary inputs (`adcroft_followups.md`, `legoesm_ocean_model.md`, `legoesm_ocean_choices.md`) and the same source tree. Per-reviewer files: `review_dycore_expert.md`, `review_ocean_model_expert.md`, `review_dycore_tester.md`, `review_claude_audit.md`.

**Date:** 2026-05-21.

---

## 1. The single most important finding

**Capability is good. Defaults and tests trail badly.**

Every reviewer arrived at this independently from a different lens. The schemes Alistair would recommend are mostly already in `src/legoesm/ocean/` — they are just not the production defaults and they are not validated against per-term closed-form references. The remediation across most of the seven items is closer to "switch defaults + build the harness that defends the choice" than "implement new science".

Our own design dossier (`docs/legoesm_ocean_model.md`) had a small amount of documentation drift against `src/legoesm/ocean/`. On careful re-reading during Phase 0, **only 2 of the 6 items in §2 were actual dossier drift** (the MPAS "Heun" mislabel and the missing AL81 mention for lat-lon ζ flux); one was worksheet drift (tracer default), the remaining three were already correctly described in the dossier. The table in §2 is kept as the consolidated *code-reality summary* the reviewers used to recalibrate their assessments. Fixing the actual drifts is on us; no outbound action. We also have a few places where it's possible we misheard Alistair himself (§2bis); these stay flagged for our own record only.

---

## 2. Code-reality summary (and the actual dossier drift)

Six code-grounded facts the reviewers built on. On careful re-reading during Phase 0:
- **Items 1, 2 were actual dossier drift** and have been corrected in `docs/legoesm_ocean_model.md`.
- **Item 4 was worksheet drift** (dossier was correct; worksheet showed wrong default) — fixed in `docs/legoesm_ocean_choices.md`.
- **Items 3, 5, 6** were already represented correctly in the dossier; they appear here because they bear on the work plan, not because the dossier was wrong.

None of this is a science bug. No outbound communication to Alistair is needed.

| # | Dossier said | Reality (source-grounded) |
|---|---|---|
| 1 | Lat-lon Coriolis = Sadourny EC + Hollingsworth | Lat-lon Coriolis = **Arakawa–Lamb 1981 (AL81)** 12-point triad with Le Sommer / Stewart-Dellar partial-cell weighting, plus Hollingsworth correction. `pv_flux_al81_partial_cell` at `latlon_cgrid_operators.py:2435`, wired at `ocean_pe_latlon_cgrid.py:1213`. Sadourny EC was the *historical* form; it blew up on ETOPO (NaN by day 19) and was replaced. |
| 2 | MPAS Coriolis uses Heun | Both lat-lon **and MPAS** use forward–backward Matsuno (`_forward_backward_coriolis_3d`, `_forward_backward_coriolis_mpas_3d`). |
| 3 | MPAS PV-flux default = AL81 | MPAS default is `pv_scheme="enstrophy"`. Options are `{"energy","enstrophy","mixed"}`. **AL81 is not in the MPAS dispatch.** |
| 4 | Tracer default = DST-3 | Lat-lon default is `"tvd"` (2nd-order). MPAS default is `"upwind"` (1st-order). DST-3 is available on lat-lon, not on MPAS. PPM is available on lat-lon, not on MPAS. |
| 5 | Lat-lon PGF options = `{"adcroft","smc03"}` | True — but **AHH08 is implemented (`pgf_ahh08.py`) and not exposed on lat-lon**. Lat-lon `_valid_pgf` excludes it for no documented reason. |
| 6 | MPAS PGF default = density-Jacobian | MPAS default is `"centered"`. SMC03 and AHH08 are *available* but not default. |

**Status:** items 1, 2 corrected in the dossier (Phase 0.1); item 4 worksheet drift corrected in the choices doc (Phase 0.2). Items 3, 5, 6 were already accurate in the dossier; the points remain in the table because they describe the production state and are relevant to the work plan. No outbound communication.

---

## 2bis. Where the meeting record itself is uncertain

It's possible that some of what was captured in `adcroft_followups.md` reflects mishearing or rough reconstruction on our side rather than Alistair's actual position. We don't need to clear these up with him — they're flagged so we don't over-attribute specific recommendations to him and so we know which questions to re-open if a future conversation goes back to these topics.

Points where we may have misheard or filled in detail that wasn't there:

- **Item 3 — "RK3/RK4 not necessarily good".** Already flagged as fragmentary recollection in the follow-up doc. We don't know whether the objection was cost, phase error, conservation, or something else. The standing question is the right one to ask if we re-open the topic; we should not assume the remark applies as stated.
- **Item 3 — "Forward + backward + RK2" combination.** Reconstructed. The MOM6 architecture we mapped this onto (forward–backward barotropic + RK2 baroclinic + Matsuno Coriolis) is plausible, but we are not confident Alistair literally described it that way.
- **Item 4 — "Energy-conserving Coriolis".** Genuinely ambiguous between spatial (Sadourny EC / Arakawa–Hsu / AL81) and temporal (Crank–Nicolson on the rotation). The reviewers landed on "almost certainly spatial; CN-Coriolis is misallocated effort", but the temporal reading is not ruled out by the meeting alone.
- **Item 5 — "DST-3 TVD is not bad".** Clear endorsement of *some* 3rd-order TVD scheme. Less clear whether it endorses DST-3 specifically over PPM, or whether it was a general "3rd-order monotone is the right neighbourhood" comment.
- **Item 7 — "Haidvogel's book".** Explicit "may have a solution in" framing. The likely reference is Haidvogel & Beckmann (1999), *Numerical Ocean Circulation Modeling*; the canonical seamount setup is Beckmann & Haidvogel (1993). We should not over-attribute specific recipe details to Alistair.

These uncertainties do not change the work plan in §5 — the per-term tests and seamount benchmark settle the empirical questions regardless of which reading is correct.

---

## 3. Consensus map across the seven items

Format: stance / level of reviewer agreement / where the four reviewers diverged.

### Item 1 — Decouple operators from coordinate metrics

- **Consensus stance:** AGREE the issue is real; the architecture has 127 inline `cos`/`R_earth`/`radius` references in `latlon_cgrid_operators.py` across 2810 lines. The MOM6 supergrid pattern is the right target.
- **Reviewer agreement:** 4/4 reviewers agree on the diagnosis. Strong opinions on sequencing.
- **Disagreements surfaced:**
  - Should lat-lon-only refactor come first, or lat-lon + cubed-sphere together? Ocean-model-expert pushes hard for **both together** — "the lat-lon-to-Mercator refactor is too easy to expose the real interface design questions; cubed-sphere is where the abstraction earns its keep". Dycore-expert says scope can be narrowed to lat-lon first because MPAS-TRiSK lives in a different mimetic family.
  - Should the supergrid be adopted verbatim (ocean-model-expert: yes, copy MOM6's pattern with rotation matrices and 3D effective areas from day 1) or built incrementally (dycore-expert: define metric arrays at T/u/v/q points first, add supergrid only if a third grid family forces it)?
  - When should Item 1 land relative to Item 7? Dycore-expert: **Item 7 first**, use seamount baseline as the regression gate. Ocean-model-expert: agrees. Dycore-tester: bit-for-bit regression on the full matrix must precede the refactor.
- **Recommended path (consensus):**
  1. Land Item 7 seamount baseline first (it becomes the regression target).
  2. Build the operator/metric interface for ~15 functions in `latlon_cgrid_operators.py` (compute_grad/div/curl, kinetic_energy, coriolis_cgrid, pgf_*, vertex/edge averaging).
  3. Prototype on lat-lon + cubed-sphere together — explicitly include rotation matrices and 3D effective-area handling.
  4. Bit-for-bit reproduce current results; add vector-calculus-identity tests.
  5. Add Mercator as the second coordinate variant.

### Item 2a — Shallow-water limit at `nlev=1`

- **Consensus stance:** AGREE strongly; this is one of the highest-value cheap actions in the audit.
- **Reviewer agreement:** 4/4. Audit confirms zero `nlev==1` special cases in `ocean/`; Williamson SW infrastructure exists at `tests/atmosphere/shallow_water/`, `tests/test_cases/williamson*.py`, `tests/unit/test_williamson2_cdgrid.py`. Reuse is wiring, not derivation.
- **Refinements added by reviewers (not in original item):**
  - Ocean-model-expert: ocean SW ≠ atmosphere SW. Free-surface vs rigid-lid changes interpretation of Williamson-6 (Rossby–Haurwitz). Williamson-5 (isolated mountain) is the *SW analog of the seamount test* — should ship in the same PR as Item 7.
  - Dycore-tester: also wire Galewsky barotropic instability. On cubed-sphere it's the single best edge-effect detector once CS ocean stabilises.
  - Dycore-expert: same priority ordering — Williamson-2 first, Williamson-5 second, Galewsky third. Skip MPAS Williamson port until lat-lon SW is landing.
- **Recommended path:** Williamson-2 + Williamson-5 + Galewsky on lat-lon ocean `nlev=1` via a 50-LOC SW adapter (`tests/ocean/unit/sw_adapter.py`). MPAS-Voronoi SW deferred.

### Item 2b — Coordinate-invariant PGF

- **Consensus stance:** AGREE; *the infrastructure is already in better shape than the dossier said*.
- **Reviewer agreement:** 4/4 on the diagnosis. AHH08 (Adcroft–Hallberg–Harrison 2008) and SMC03 (Shchepetkin–McWilliams 2003) are both implemented and selectable.
- **The real problems:**
  - AHH08 not exposed on lat-lon — easy fix (`pgf_ahh08.py` is grid-agnostic in its core; the column primitive `column_pressure_integrals_ahh08` is vertical-only; add the lat-lon edge wrapper + `_valid_pgf` entry).
  - MPAS default is `"centered"` — empirically the worst of the four available schemes. Per `pgf_ahh08.py:61` docstring on ETOPO+ico4: `"adcroft"` ~1.4e-6, `"smc03"` ~8.6e-8, AHH08 better still.
- **Disagreements surfaced:**
  - Sequencing: ocean-model-expert argues Item 2b *strong form* (one PGF across all vertical coordinates) is **downstream of Item 1** (because PGF composes with horizontal operators that are coordinate-baked). Dycore-expert is similar but less strict. Dycore-tester says the test that *measures* coord-invariance is Item 7 itself.
  - Discretization-invariance vs form-invariance: ocean-model-expert raises a technical pushback — "coordinate-invariant" can mean form-invariant (same code path, any vertical coord) or discretization-invariant (same truncation error). The latter is impossible in general — z\* PGF errors over steep bathymetry will always exceed isopycnal PGF errors. AHH08 is the best form-invariant scheme; discretization-invariance is a *limit*, not a target. The dossier conflates these.
- **Recommended path:**
  1. Wire AHH08 onto lat-lon (≤1 day).
  2. Land Item 7 seamount test on all four PGF schemes; pick defaults from the data.
  3. Defer the strong-form Item 2b ("single PGF across all coords") until Item 1 lands.

### Item 3 — Per-term time stepping

- **Consensus stance:** AGREE on the per-term framing; the worksheet's single "outer integrator" axis is a **category error** and should be deprecated.
- **Reviewer agreement:** 4/4. All reviewers note that the dycore *already does* per-term stepping in `_step_impl` (forward–backward Matsuno for Coriolis, sub-step for barotropic, RK-family for tracer). The framework just isn't exposed at the config layer.
- **`src/legoesm/timestepping/dispatch.py`:** consensus says **do not** wire the ocean into it. The dispatch module is appropriate for the atmosphere's monolithic RK stepper; ocean is structurally split-explicit and benefits from its own per-grid `_step_impl`.
- **Disagreements surfaced on Alistair's "RK3/RK4 not necessarily good" remark:**
  - Dycore-expert: probably both (a) cost — RK3 buys ~1.4× CFL for 3× tendency cost vs RK2 — and (b) phase error for oscillatory modes is no better than well-tuned Matsuno+forward–backward.
  - Ocean-model-expert: **PUSH-BACK**. Phase error of RK3 is strictly better than RK2 on inertia-gravity modes (Durran 1991). SSP-RK3 has positivity-preservation properties valuable for differentiable workflows. **Do not deprecate SSP-RK3 from the dispatch.**
  - Both reviewers agree: a one-line follow-up email to Alistair to disambiguate. Suggested framing: "Are you objecting to cost, phase error, or conservation properties? Each has different implications for us."
- **One additional finding from ocean-model-expert:** ask whether KPP vertical mixing is *implicit* (CVMix-style tridiagonal). If not, that is a higher-priority fix than CN Coriolis — explicit KPP at $K_v=0.1$ m²/s, $\Delta z=5$ m caps $\Delta t < 125$ s.
- **Recommended path:**
  1. Replace worksheet §4 with a per-term table. Documentation work; the code already does the right thing.
  2. Keep SSP-RK3 in the dispatch.
  3. Confirm KPP path is implicit; if not, prioritise.
  4. Send Alistair the disambiguation email on RK3/RK4.

### Item 4 — Coriolis: energy-conserving

- **Consensus stance:** AGREE on the spatial principle; consensus PUSH-BACK on temporal CN-Coriolis as a priority.
- **Reviewer agreement:** 4/4 — though framed differently.
- **Spatial side:**
  - Lat-lon is **already AL81** (best-in-class on quadrilateral C-grid). Nothing to do.
  - MPAS default `pv_scheme="enstrophy"` is the *opposite* of Alistair's recommendation. **All reviewers agree this is the single highest-leverage configuration change in the audit.** Switch the default to `"energy"`.
  - Implementing AL81 on MPAS is a 1-week task per dycore-expert; possibly the right longer-term answer.
- **Temporal side:**
  - All reviewers push back against prioritising CN-Coriolis. Matsuno already has $|G|^2 = 1 + O((f\Delta t)^4)$ amplification per step; at typical ocean $f\Delta t \lesssim 0.1$, KE drift is sub-1% per multi-year integration. CN buys exact conservation at the cost of a per-cell 2×2 implicit solve and *worse* phase error at moderate Courant numbers (Durran 1991 §2.3).
  - Recommendation: do not allocate eng-weeks to CN-Coriolis until the inertial-oscillation test (Recipe #1, below) reveals it as the bottleneck — which no reviewer expects.
- **Verification path:** the inertial-oscillation test (Recipe #1) discriminates Matsuno from CN at $f\Delta t = 0.036$: Matsuno KE drift $\sim 10^{-3}$ per inertial period; CN $< 10^{-12}$. If the empirical numbers come in better than predicted, the question settles itself.
- **Recommended path:**
  1. Switch MPAS default to `pv_scheme="energy"`; validate against the existing matrix.
  2. Implement AL81 on MPAS (mixed scheme with sensible default α). Optionally — defer if "energy" suffices.
  3. Do *not* prioritise CN-Coriolis. Park the implementation behind the inertial-oscillation test outcome.
  4. Fix the dossier (§2 internal documentation drift).

### Item 5 — Tracer advection

- **Consensus stance:** AGREE with Alistair's "3rd-order TVD" target; PUSH-BACK on dossier's characterisation.
- **Reviewer agreement:** 4/4 on the diagnosis; some disagreement on PPM vs DST-3.
- **Capability/default gap is large:**
  - Lat-lon offers `{upwind, tvd, ppm_fct, ppm, dst3, dst3_multidim, som, weno5, weno7}` — 9 schemes. Default is `"tvd"` (2nd-order).
  - **MPAS offers only `{upwind, tvd}`** (dycore-expert and audit both confirm). Default is `"upwind"` (1st-order). At $\sim$100 km Voronoi resolution and $|u|\sim 0.1$ m/s, that's $K_\text{implicit} \sim 5\times 10^3$ m²/s — GM-eddy-flux scale of *numerical* noise, before any explicit diffusion is added.
  - The dossier's claim that DST-3 is the default *anywhere* is wrong.
- **Disagreements surfaced:**
  - PPM vs DST-3: ocean-model-expert PUSHES BACK on the dossier's "siblings, marginal" framing. PPM (Colella–Woodward 1984) has built-in monotonicity-preserving parabolic reconstruction with a well-characterised limiter; DST-3 + TVD limiter clips extrema harder. On Eady/ACC sharp fronts the difference matters. Citation: Griffies, Pacanowski & Hallberg (2000) on spurious diapycnal mixing. MOM6 ships PPM.
  - SOM as a default: ocean-model-expert pushes back. Six extra prognostic moments per tracer → AD checkpointing footprint blows up. Keep SOM as a research option, not a production default.
  - The `project_dst3_advection` note "full benefit at CFL > 0.1" — ocean-model-expert reads it sharply: at low CFL DST-3 degenerates toward 1st-order. PPM does not have this CFL-dependence. If our production CFLs are ~0.05, DST-3 is barely earning its keep.
- **Recommended path:**
  1. **Lat-lon: change default to `"dst3"`** today (1-line change). Validate Eady/ACC RPE drift before/after.
  2. **MPAS: implement DST-3 on Voronoi** (port MPAS-O's unstructured-PPM reconstruction; well-trodden literature). Make it the default once validated.
  3. **Add RPE-drift diagnostic** to the test matrix (Ilicak et al. 2012 protocol on lock-exchange and overflow). This is how you actually defend a tracer-scheme choice; convergence-rate alone is insufficient.
  4. Document SOM use-case explicitly: low-mixing experiments (Eady, ACC) only. Not a production default.

### Item 6 — Per-term tests

- **Consensus stance:** AGREE strongly. This is the methodological piece that ties everything else together, and it is the largest gap.
- **Reviewer agreement:** 4/4. All reviewers note `tests/ocean/` has zero Williamson SW tests, zero inertial-oscillation tests, zero 1D advection convergence tests, zero diffusion convergence tests, zero seamount tests. The matrix has only integrated tests.
- **Term-toggle infrastructure:** partial. Working: GM/Redi (`None` disables), Smag/Leith (zero coeff), APVM (`apvm_dt=0`), MAXVEL (zero), KPP via subconfig. **Missing:** clean `enable_coriolis`, `enable_pgf`, `enable_momentum_advection`, `enable_tracer_advection`, `enable_drag` flags.
- **Disagreements surfaced:**
  - Naming of toggle: dycore-expert says `enabled: bool = True` per scheme. Ocean-model-expert prefers `disable_*` flags with a separate `test_mode: bool = False` that production runs must NOT have set, so "test mode" is loud in the audit trail.
  - First test priority:
    - Dycore-expert: **linear gravity wave first** (cheapest; runs on `nlev=1`; validates PGF + free surface + barotropic substep simultaneously).
    - Dycore-tester: **1D pure-advection convergence first** (smallest test that unlocks the entire methodology; we ship 9 tracer schemes with zero convergence validation).
    - Ocean-model-expert: **inertial oscillation first** ("smallest test that exercises the largest number of code paths"; settles the CN-vs-Matsuno question).
  - The three positions are not mutually exclusive — all three tests are S-effort (≤1 week each) and can land in parallel.
- **Cross-cutting refinements (all reviewers added):**
  - **Convergence rate, not absolute L2, is the right pass criterion.** Threshold drifts with grid / noise; convergence rate is the invariant.
  - **Time-convergence twin of every space-convergence test** (vary $\Delta t$ at fixed $\Delta x$). Exposes the actual stepper.
  - **Gradient-correctness variant of every per-term test** — compare `jax.grad` to analytic derivative. This is *legoESM's* differentiable contribution over MOM6's framework.
  - **Discrete KE conservation on Galewsky jet** (all diffusion off) is the gating test for the MPAS `pv_scheme="energy"` switch in Item 4. Specific number: EC variant gives $|KE(t)-KE(0)|/KE(0) < 10^{-10}$/day; EN drifts at $\sim 10^{-6}$/day.
- **Recommended path:**
  1. Land `enable_*` (or `disable_*`) flags in `LatLonCGridOceanConfig` and `MPASOceanConfig` with `test_mode` guard.
  2. Build `make_test_config(term=..., grid=...)` factory.
  3. Land first three tests in parallel: inertial oscillation, 1D advection convergence, linear gravity wave.
  4. Standardise `convergence_rate(L2_list, n_list)` helper in `tests/ocean/unit/_helpers.py`.
  5. Each test ships with a gradient-correctness variant.

### Item 7 — Flow past a bump (seamount)

- **Consensus stance:** AGREE strongly. **All four reviewers independently picked this as the single most leveraged work item.**
- **Reviewer agreement:** 4/4 — the only item where all reviewers converged on "this is *the* one".
- **Why this dominates:**
  - It is the *only* canonical PGF discriminator (Beckmann & Haidvogel 1993). 33 years old, never run on legoESM.
  - It converts every PGF design choice from literature claim to measured scalar on our code, our grids, our bathymetry.
  - Five of seven items get pulled forward by this: Item 2b (which PGF default), Item 1 (regression target for the metric refactor), Item 4 (clean baseline for the MPAS PV-flux switch), Item 2a (Williamson-5 is the SW analog), Item 6 (canonical per-term PGF test).
- **Recipe (consensus, drawn from dycore-tester recipe #10):**
  - **Domain:** 320 km × 320 km, $H_0=4500$ m + Gaussian seamount $h_s = 4000\exp(-(r/L)^2)$, $L=25$ km. Seamount peaks at $z=-500$ m.
  - **Resolution sweep:** $\Delta x \in \{10, 5, 2.5\}$ km. Vertical: $n_\text{lev}=20$ z\* layers.
  - **Stratification:** $T(z) = 5 + 15z/H_0$ °C (linear), constant $S$, giving $N^2 \approx 10^{-4}$ s$^{-2}$ (matches BH93 §3).
  - **IC:** $u=v=0$, $\eta=0$, hydrostatic. No forcing, no relaxation.
  - **Variants:** $f=0$ (pure PGF isolation) AND $f=10^{-4}$ s$^{-1}$ (rotating; production-relevant). Beckmann–Haidvogel original is rotating.
  - **Time:** 180 days. Diagnostic: $\max|u|(t)$ daily, plus RPE drift (Ilicak protocol).
  - **Target values** (per `pgf_ahh08.py:61` and Shchepetkin–McWilliams 2003):
    - `pgf_scheme="adcroft"` (z\* standard): $\max|u| \sim$ 1 cm/s.
    - `pgf_scheme="smc03"`: $\max|u| \sim$ 1 mm/s.
    - `pgf_scheme="ahh08"` (requires `eos="wright"`): $\max|u|$ smaller still.
  - **Pass criterion:** ≥100× spread between schemes expected. If all three give $\max|u| \sim$ cm/s, an implementation is broken.
- **Refinements added by reviewers:**
  - Ocean-model-expert: two stratifications (linear $N^2$ AND exponential thermocline); two bump geometries (smooth Gaussian AND steeper sloped); cross-grid comparison (lat-lon adcroft vs MPAS AHH08 — if MPAS doesn't win, AHH08 wiring is broken).
  - Dycore-tester: convergence rate as the verification (max|u| should *decrease* with resolution for correct PGF, *increase* for buggy PGF).
  - Both: Williamson-5 (SW mountain test) in the same PR — it is the SW limit of the same physics.
  - AD consistency check at the seamount setup (gradient through PGF over varying bathymetry is a non-trivial AD probe).

---

## 4. Findings beyond the seven items

### Item 8 candidate — Barotropic–baroclinic mode-split coupling

Proposed by dycore-expert. Memory notes `project_barotropic_solver` and `project_barotropic_noise_issue` confirm this is open, real, and Adcroft-relevant (BEBT in Shchepetkin & McWilliams 2005; Adcroft co-authored relevant follow-ups). The mode-split coupling determines how barotropic substep updates of $\eta$ are reconciled with baroclinic-step tracer fluxes — if done wrong, you get either spurious mass-conservation drift (the symptom we see in lat-lon barotropic-mode noise) or hidden viscosity. None of the seven items addresses this. Worth a deliberate add to the next conversation.

### Item 9 candidate — Cubed-sphere face-boundary instability

Proposed by ocean-model-expert. "Elephant in the room" — the metric-driven refactor (Item 1) is the right architectural piece, but face-boundary instability may require an algorithmic change beyond metric cleanup: explicit vector rotation in halo exchange, possibly a different mimetic operator family at panel edges, possibly a PV-flux stencil that doesn't straddle face edges. This is a *separate* item from Item 1 because the architectural cleanup may not be sufficient.

### Differentiability-specific considerations

Ocean-model-expert raised this and it does not appear in the original seven items. The trade-off space for legoESM is *different* from MOM6's because we care about:
1. **Gradient stability through limiters** — favors PPM over DST-3, favors smooth tapers everywhere (sigmoid, tanh, not hard `jnp.clip`).
2. **Checkpointing cost in long barotropic subcycles** — favors fewer prognostic variables → argues *against* SOM as default.
3. **AD through implicit solves** — should use implicit-function-theorem differentiation on any future implicit Coriolis or implicit vertical mixing, not unrolled iterative solvers.

These considerations should appear in the design dossier (`legoesm_ocean_model.md` §20) but currently get a single section without the trade-off implications enumerated.

---

## 5. Prioritised plan of work

Sequenced for maximum leverage and minimum dependency conflict. Phases are not strict — work within a phase can parallelise.

### Phase 0 — Internal documentation calibration (now)

- [ ] Fix the six points in §2 in `docs/legoesm_ocean_model.md` (the dossier). Internal corrections; no outbound communication.
- [ ] Update worksheet §4 from single-axis "outer integrator" to a per-term table.
- [ ] Carry the §2bis "meeting uncertainties" forward — they don't need to be cleared up with Alistair now, but they should be remembered the next time these topics come up.

### Phase 1 — Test framework foundation (≤2 weeks)

The methodological prerequisite. Without this, every later default change is rhetoric.

- [ ] **`enable_*` (or `disable_*`) flags** in `LatLonCGridOceanConfig` and `MPASOceanConfig`. Per ocean-model-expert: pair with a `test_mode: bool = False` that production refuses.
- [ ] **`make_test_config(term=..., grid=...)` factory** in `ocean/experiments/test_configs.py`.
- [ ] **`convergence_rate(L2_list, n_list)` helper** in `tests/ocean/unit/_helpers.py`.
- [ ] Three per-term tests in parallel (S effort each):
  - [ ] **Recipe #1 — Inertial oscillation** (Coriolis isolation, settles the CN-vs-Matsuno question).
  - [ ] **Recipe #2 — 1D periodic tracer advection convergence** (smallest test that unlocks the methodology; gates Item 5 default switch).
  - [ ] **Recipe #5 — Linear SW gravity wave dispersion** (cheapest; validates PGF + free surface + barotropic substep).
- [ ] Gradient-correctness variant of each test.

### Phase 2 — Default switches with empirical evidence (≤3 weeks)

- [ ] **Recipe #10 — Beckmann–Haidvogel seamount** (M effort). Two stratifications, two bump shapes, $f=0$ and $f=10^{-4}$ variants, three resolutions. Run all four PGF schemes; produce the `max|u|` vs PGF scheme curve as a CI artifact.
- [ ] **Wire AHH08 onto lat-lon** (≤1 day). Add to `_valid_pgf`; build the edge wrapper; verify on the seamount test.
- [ ] **Switch lat-lon tracer default** from `"tvd"` to `"dst3"`. Validate on Eady, ACC RPE drift.
- [ ] **Switch MPAS PV-flux default** from `"enstrophy"` to `"energy"`. Gate test: Recipe #8 (Galewsky discrete-KE conservation, all diffusion off).
- [ ] **Switch MPAS PGF default** from `"centered"` to `"ahh08"` (if seamount results support it) or `"smc03"` (fallback).

### Phase 3 — SW matrix on the ocean (≤2 weeks, parallel to Phase 2)

- [ ] SW adapter (`tests/ocean/unit/sw_adapter.py`) — ~50 LOC.
- [ ] Williamson-2 on lat-lon ocean `nlev=1`. Convergence rate, two flow orientations ($\alpha=0$ and $\alpha=\pi/4$).
- [ ] Williamson-5 (isolated mountain) on lat-lon ocean `nlev=1`. Ships in the same PR as Recipe #10 (Item 7).
- [ ] Galewsky on lat-lon ocean `nlev=1`. Continuous-regression coverage and the discrete-KE test in Item 4.
- [ ] Discrete-KE conservation test (`Recipe #8` from dycore-tester table).

### Phase 4 — Architectural refactor (1–2 months)

Gated on Phase 1 (regression coverage) and Phase 2 (seamount baseline).

- [ ] **Bit-for-bit regression infrastructure** before any refactor. Per-step state checksums; CI fails on any divergence.
- [ ] **Vector-calculus-identity tests** ported from the existing `tests/unit/test_vector_calculus_identities.py` (CS D-grid) onto the lat-lon `QuadrilateralCGridMetrics`.
- [ ] **Define `QuadrilateralCGridMetrics`** container. Per ocean-model-expert: include rotation matrices and 3D effective-area handling from day one. Mercator and cubed-sphere panel coordinates become metric-provider variants.
- [ ] **Refactor 15 operators** in `latlon_cgrid_operators.py` to consume metrics as arguments. No inline `cos`/`R_earth`/`radius`.
- [ ] **Prototype on lat-lon + cubed-sphere together** (ocean-model-expert push) — the abstraction earns its keep on cubed-sphere, not on Mercator.
- [ ] **Mercator as third coordinate variant** — bit-for-bit reproduction of existing Mercator path.

### Phase 5 — Items beyond Alistair's seven (open scope)

- [ ] **Item 8 — Barotropic–baroclinic mode-split coupling.** Audit current implementation against SMC05 / BEBT. Open scope.
- [ ] **Item 9 — Cubed-sphere face-boundary instability.** Algorithmic investigation; may require new mimetic operator at panel edges. Long-horizon.
- [ ] **KPP implicit-treatment audit** (ocean-model-expert raised). Confirm vertical mixing uses tridiagonal implicit solver à la CVMix. If not, prioritise.
- [ ] **CN-Coriolis** — deferred. Only after Recipe #1 reveals Matsuno as a bottleneck.
- [ ] **ALE adoption** — deferred until Item 1 + coord-invariant PGF land.

---

## 6. Risks and gating decisions

- **The Phase 4 refactor without Phase 1 + Phase 2 in place is high-risk.** Per dycore-tester: "the only gate is 'code looks cleaner', which is not falsifiable." Phase order is load-bearing.
- **The MPAS PV-flux default switch (`enstrophy` → `energy`) may expose hidden dependencies.** If ACC channel or Eady runs were relying on enstrophy dissipation as hidden viscosity, the switch will manifest as new instabilities. The Galewsky discrete-KE gating test plus the existing matrix re-run is the safety net.
- **Implicit-KPP question is unresolved.** If vertical mixing is currently *explicit* with $K_v \sim 0.1$ m²/s, our outer $\Delta t$ is constrained more tightly than the audit acknowledges. Worth an early audit before committing to Phase 2 timelines.
- **Bit-for-bit regression must precede Item 1.** Dycore-tester is explicit on this. The "checksum every state in CI" infrastructure is its own work item and should be Phase 1.

---

## 7. Where the reviewers disagree — and what to do about it

Three substantive disagreements where consensus did not form. Worth being explicit:

1. **PPM vs DST-3 as lat-lon default.** Ocean-model-expert: PPM strictly better for differentiable workflows. Dycore-expert: marginal difference. Dycore-tester: convergence-rate test will tell us. **Decision rule:** switch to DST-3 now (1-line change, audit-supported); run the convergence-rate harness; switch to PPM if the data warrants.

2. **Lat-lon-only vs lat-lon + cubed-sphere together for Item 1.** Ocean-model-expert: together, because the cubed-sphere panel boundaries are where the abstraction earns its keep. Dycore-expert: lat-lon first, less surface area. **Decision rule:** start the *interface design* with both in mind (rotation matrices, 3D effective areas as first-class) but the *first implementation* can be lat-lon-only, with cubed-sphere conversion as the immediate next step in the same epic.

3. **First per-term test to land.** Three reviewers, three different "first" recommendations: gravity wave (dycore-expert), 1D advection convergence (dycore-tester), inertial oscillation (ocean-model-expert). **Decision rule:** all three are S-effort; land them in parallel as the Phase 1 deliverable.

---

## 8. What the user should do next

In priority order:

1. **Read this document end-to-end** and the four reviewer files. Flag anywhere the synthesis misrepresents a reviewer.
2. **Fix the dossier internally** (§2). Six corrections to land in `docs/legoesm_ocean_model.md` on its next revision. No outbound communication. The §2bis uncertainties stay on our own record.
3. **Choose Phase 1 owner.** The test framework foundation is the gating piece. Anyone working on it should own §3 Item 6.
4. **Decide on Phase 4 timing.** The architectural refactor will absorb 1–2 months. Phasing it after Phase 2's seamount baseline is the consensus, but the calendar choice is yours.
5. **Open Items 8 and 9 as separate planning threads.** They are real and Adcroft-relevant.

This synthesis should be treated as a *living* document — re-open it as Phase 1/2 results come in, since several of the standing questions resolve empirically once tests land.
