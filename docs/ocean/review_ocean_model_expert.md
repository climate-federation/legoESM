# Ocean-model-expert independent review

*Reviewer persona: senior ocean dycore engineer, MOM6/MPAS/ROMS background, Adcroft-adjacent but willing to disagree.*
*Built on: `docs/ocean/adcroft_followups.md` (7 items), `docs/ocean/review_claude_audit.md` (code audit).*

---

## Headline

Alistair's seven items form a coherent ocean-numerics manifesto, and I broadly endorse it — but the implementation gap is **smaller and more uneven than either the dossier or the audit suggests**. The lat-lon PV-flux is already AL81 (energy *and* enstrophy conserving, 12-point partial-cell — `pv_flux_al81_partial_cell` at `latlon_cgrid_operators.py:2435`), not bare Sadourny; AHH08 and SMC03 are wired; DST-3, PPM, SOM, WENO5/7 are all in the menu. The actual debt is concentrated in three places: (1) **the operator layer is coordinate-baked** (Item 1), which blocks the cleanest version of Items 2b and the cubed-sphere face-instability fix; (2) **defaults trail capabilities** on tracer advection and MPAS PV-flux; (3) **the validation harness has no clean per-term isolation** (Items 6, 7) — which is exactly what Alistair was reaching for. I would push back on the framing of "energy-conserving Coriolis" as a temporal goal (the spatial choice is what controls aliasing-driven blowups; the temporal CN refinement is marginal in our regime), and I would *strongly* push back on demoting RK3/SSP-RK3 — for our differentiable use case, conservation-of-positivity properties of SSP integrators are more valuable than the dossier reconstruction allows. The single highest-leverage move is **Item 7 (seamount-on-stratification rest state)**, because it converts every other PGF and PV-flux choice from rhetoric into a measurable yardstick.

---

## Per-item assessment

### Item 1 — Coordinate-baked operators

- **Stance:** AGREE. This is the highest-leverage architectural piece, and also the one we are most behind on. The audit's number (2810 lines, 127 inline `cos`/`R_earth`/`radius` refs in `latlon_cgrid_operators.py`) understates the problem because it doesn't reveal that the cubed-sphere operators have the *same* pathology with worse symptoms.

MOM6's "supergrid" pattern is the correct reference (Adcroft et al. 2019). The supergrid is a 2× refined logically-rectangular mesh on which all metric quantities — $dx_T, dy_T, dx_u, dy_u, dx_v, dy_v, dx_q, dy_q$, cell areas, edge lengths, angle-rotation matrices — are *derived once at grid construction*. Operators take the metrics as input arrays and never touch geometry. Crucially, the same operator code runs unchanged across:
- regular lat-lon (with $\cos\phi$ metric)
- Mercator (with isotropic metric scaled by $1/\cos\phi$)
- tripolar (NEMO ORCA-style, supergrid encodes the pole pair)
- cubed-sphere (per-panel supergrids with rotation matrices on edges)
- regional patches (supergrid is just a finite rectangle of the global one)

This is the *cleanest* architectural pattern for a quadrilateral C-grid hierarchy. Our current code reaches into `grid.radius` and computes `cos(lat)` inline, which means every coordinate change is a code change.

Two ocean-specific concerns the doc doesn't surface:

1. **Vector rotation at panel boundaries.** On the cubed-sphere, when you halo-exchange $u, v$ across a panel edge, the basis vectors of the source and target panels differ. The metric-driven refactor must therefore include a per-edge rotation matrix (call it $R_\text{edge}$, $2\times2$) on each velocity halo. This is *the* root cause family for cubed-sphere face-boundary noise across every ocean model that has tried this (FV3-ocean prototypes had it; OMARE-cs had it; we have it — see the open issue on face-boundary exponential instability). The metric-driven refactor is the correct moment to introduce $R_\text{edge}$ as a first-class object, not an afterthought.

2. **Partial bottom cells modify the metric.** Adcroft, Hill & Marshall (1997) — partial cells mean the *bottom* cell area depends on bathymetry, and that area enters every flux-divergence operator. If the metric layer abstracts horizontal areas but the bottom cell does its own thing, you've recreated the duplication problem one level deeper. The supergrid abstraction should naturally accommodate 3D effective areas $A(i,j,k) = A_\text{H}(i,j) \cdot f_\text{bot}(i,j,k)$.

I would also push back on the "Mercator first, cubed-sphere later" sequencing implied by the doc. The lat-lon-to-Mercator refactor is *too easy* — both are logically rectangular, metrics are diagonal — to expose the real interface design questions. Doing Mercator first risks producing an abstraction that is just "lat-lon with a different $\cos\phi$" and that breaks on cubed-sphere when the rotation matrices arrive. **Do one regular C-grid coordinate (lat-lon) plus one panelized coordinate (cubed-sphere with at least one face edge) in the first prototype.** That is what tells you whether the abstraction is right. Tripolar can wait.

A last point on scope: MPAS (TRiSK on Voronoi) is correctly outside the scope of the quadrilateral C-grid operator library. The mimetic-discrete-exterior-calculus framework that TRiSK descends from has a *different* primitive operator algebra (Thuburn et al. 2009; Ringler et al. 2010). Conflating them is a Bad Idea. Keep MPAS its own world; share only the high-level dycore orchestration.

**Recommendations:**
1. **Define the `QuadrilateralCGridMetrics` container with rotation matrices from day one.** Use MOM6's supergrid pattern verbatim; we're not going to do better than that design.
2. **Prototype on lat-lon + cubed-sphere together**, not lat-lon then cubed-sphere. The cubed-sphere is where the abstraction earns its keep.
3. **Build partial-bottom-cell support into the metric layer**, not as a post-hoc patch.
4. **Treat this as the prerequisite for Items 2b (coord-invariant PGF) and the cubed-sphere face-instability fix.** Sequence accordingly.

### Item 2 — Layered + coordinate-invariant

#### 2a. SW as the `nlev=1` limit

- **Stance:** AGREE in principle, AGREE-WITH-NUANCE in execution. The hard part is interpretation, not coding.

The audit's finding — zero `nlev == 1` special cases — confirms the dycore is genuinely orthogonal in the vertical axis. Wiring the existing Williamson harness (which lives at `tests/test_cases/williamson*.py` and `tests/atmosphere/shallow_water/test_cases/williamson_mpas.py`) onto the ocean `nlev=1` path is straightforward — half a day of engineering once you have the right factory.

But the ocean SW reduction is *not* the same beast as the atmosphere SW. Critical differences that should be in the test interpretation:

1. **Free surface vs rigid lid.** Atmosphere SW typically has rigid-lid (height as a tracer), ocean SW has free-surface elevation $\eta$. Williamson-2 (geostrophic balance) and Williamson-5 (isolated mountain) translate cleanly. Williamson-6 (Rossby–Haurwitz) involves higher harmonic balance and is sensitive to the lid choice.

2. **Coriolis spatial scheme.** Atmosphere SW tests on the cubed-sphere are typically run with Sadourny EC. Ocean lat-lon will run them with AL81. *That's a feature.* Williamson-2 on AL81 should be slightly worse on the energy diagnostic and slightly better on enstrophy than on plain Sadourny EC — and that's a direct check that the spatial scheme is doing what it claims.

3. **Galewsky barotropic instability** is the most useful one not on the dossier list. The Galewsky et al. (2004) jet is a clean test of whether the PV-flux scheme generates spurious mixing at the jet edge. AL81's energy-and-enstrophy property predicts cleaner downstream wave-breaking patterns than Sadourny EN. Run it.

4. **The PGF in SW limit is trivial** — it's $-g\nabla\eta$, with no thermal-wind component. So Item 2b's PGF improvements *cannot* be tested in the SW limit. Don't oversell what the SW matrix tells you about the full model.

5. **Don't expect SW tests to catch face-boundary instability on the cubed-sphere.** Williamson-2 is too smooth to excite the grid-scale modes that drive our day-19 ETOPO blowup. Add Williamson-5 (isolated mountain) — that one *will* exercise topographic boundary handling and is the right SW analog of the seamount test.

**Recommendations:**
1. Wire Williamson-2, Williamson-5, Galewsky onto the ocean `nlev=1` path. Use the *same* test infrastructure as atmosphere, not a duplicate.
2. Document explicitly which SW tests probe which production schemes (Coriolis/PV-flux: yes; PGF: no; tracer advection: only the SW height equation, not 3D tracers).
3. Treat the SW matrix as **continuous regression coverage**, not as a substitute for the per-term unit tests in Item 6.

#### 2b. Coordinate-invariant PGF

- **Stance:** AGREE strongly with the principle; AGREE-WITH-NUANCE on execution priority.

Alistair's framing is unambiguous and correct: PGF is $-(1/\rho)\nabla p$, the coordinate is a regridding choice, the discretization should not care. Adcroft & Hallberg (2006), Shchepetkin & McWilliams (2003), and AHH08 are the canonical references. Our codebase has AHH08 and SMC03 wired on MPAS; SMC03 on lat-lon — better than the dossier said.

The execution sequence matters here, and I differ from the doc:

- The doc's "Item 2b ... close to done" framing is half right. **The implementations exist, but they have not been benchmarked on the test that would tell you which to default to.** That test is Item 7 (seamount). Defaulting to AHH08 on MPAS without measuring SMC03 vs AHH08 on a controlled bump is a defensible engineering call but not a scientifically defended one.

- **Lat-lon AHH08 wiring is the lowest-hanging fruit.** `pgf_ahh08.py` is grid-agnostic in its core; the column-pressure-integral routine (`column_pressure_integrals_ahh08`, line 242) is a vertical operation orthogonal to horizontal grid topology. Adding `"ahh08"` to lat-lon's `_valid_pgf` set should be a half-day of work plus tests.

- The **deeper architectural piece** — making PGF identical across z*, ALE, σ, isopycnal — is conditional on Item 1 (metric-driven operators). You cannot have one PGF discretization across vertical coordinates if the *horizontal* operators it composes with are coordinate-baked. So 2b in its strong form is downstream of 1.

A specific technical pushback: the "coordinate-invariant PGF" can be interpreted two ways. (a) **Form-invariant**: the same code path handles any vertical coordinate choice via metric tensors. (b) **Discretization-invariant**: the truncation error of PGF is the same across coordinates. (b) is *impossible* in general — z* PGF errors over steep bathymetry will always be larger than isopycnal PGF errors, because the cancellation is between near-equal quantities. AHH08 is the best (a) you can build; (b) is a limit, not a target. The dossier conflates these.

**Recommendations:**
1. **Wire AHH08 onto lat-lon.** Half a day of work. Unlocks the cross-grid comparison.
2. **Run Item 7 first**, then choose defaults.
3. **Defer the "single PGF across z*/ALE/σ" goal until Item 1 lands.** Doing it before Item 1 creates two refactors that have to be re-merged.

#### 2b. Coordinate-invariant PGF

- **Stance:** AGREE-WITH-NUANCE
- *(to be filled)*

### Item 3 — Per-term time stepping

- **Stance:** AGREE on the per-term framing; PUSH-BACK on the reconstructed "RK3/RK4 are not good" remark.

The per-term framing is the right one. Hallberg (1997), Higdon (2005), and Shchepetkin & McWilliams (2005) all converged on the same picture: barotropic mode uses forward–backward (with cosine/Williams time-filtering on the average); baroclinic outer step uses RK2 or AB3; Coriolis uses a stable rotation-preserving scheme (Matsuno or semi-implicit); vertical mixing is implicit (necessary for stability with KPP-magnitude diffusivities); tracer advection uses its own scheme appropriate to the limiter family. There is no single "outer integrator" that works for all of these — that framing is genuinely too coarse.

The audit's finding that lat-lon and MPAS both use forward–backward Matsuno for Coriolis (not Heun as the dossier said) is consistent with this. **Where the audit is right and important**: the worksheet §4's single "outer integrator" axis should be deprecated. The right axis is a per-term table, and most of the per-term structure is already in `_step_impl` — it just isn't exposed as configuration.

Now the push-back. The reconstructed "RK3/RK4 not good" remark needs interrogation, not just a one-line follow-up. There are three readings, and they have different implications:

(a) **Cost vs CFL gain.** This is true for the *non-stiff* terms but misleading. RK3-SSP (Shu & Osher 1988) admits an effective Courant number near 1.0 for the linear advection operator, vs RK2's 0.5. So the throughput ratio is 1.5/3 = 0.5 — RK3 *is* worse per cost unit for advection. **But for differentiable training**, the SSP property (no overshooting under TVD limiters in forward mode) preserves positivity *along the gradient path*, which RK2 does not. This is meaningful for tracer positivity constraints in our use case.

(b) **Phase error for oscillatory modes.** Durran (1991) and Williamson (1994) showed RK3 phase error is $O(\Delta t^3)$ vs RK2's $O(\Delta t^2)$ — so RK3 is *strictly better* for inertia-gravity modes at fixed $\Delta t$. The "not good" framing is wrong here.

(c) **Conservation.** RK3/SSP-RK3 don't conserve energy exactly for the rotation operator, but neither does RK2. Higher order doesn't break conservation — it preserves more digits of it.

My reading: the remark, if accurately reconstructed, is about MOM6's design economics (RK2 + forward-backward + Matsuno covers the production regime well, and adding RK3 to MOM6 would multiply tendency-evaluation cost without comparable scientific gain). That's a defensible MOM6 design choice. It is **not** a general statement that RK3 is bad — and for our differentiable use case, the SSP-RK3 option in `src/legoesm/timestepping/dispatch.py` should *not* be deprecated. Keep it available.

A second issue the doc doesn't mention: **implicit vertical mixing matters more than the temporal scheme on Coriolis.** KPP diffusivities can be $10^{-1}$ m²/s in active mixed layers; explicit treatment requires $\Delta t < \Delta z^2 / (2 K_v)$, which for $\Delta z = 5$ m, $K_v = 0.1$ m²/s gives $\Delta t < 125$ s — at the limit of barotropic substep ratios. CVMix (used by MOM6/MPAS) handles vertical mixing implicitly via tridiagonal solves. If we don't, we either run cripplingly short outer steps or accept noisy mixed layers. Confirm whether our KPP path is implicit; if not, that's a higher-priority fix than CN Coriolis.

**Recommendations:**
1. **Replace worksheet §4's "outer integrator" axis with a per-term table.** This is documentation work, not code work — the code already does the right thing.
2. **Keep SSP-RK3 in the dispatch.** Do not deprecate it. It earns its keep for AD with positive-tracer constraints.
3. **Confirm vertical mixing is implicit.** If not, this is the time-stepping fix that actually matters.
4. **Ask Alistair to clarify "RK3/RK4 not good" — but with a prepared counter-position**, not as a fishing question.

### Item 4 — Energy-conserving Coriolis

- **Stance:** AGREE-WITH-NUANCE on the spatial scheme; PUSH-BACK on the temporal framing.

The audit understates the lat-lon spatial state. `latlon_cgrid_operators.py:2435` is `pv_flux_al81_partial_cell` — a 12-point Arakawa–Lamb (1981) triad PV flux *with* the Le Sommer / Stewart–Dellar (2016) partial-cell weighting that conserves both KE and potential enstrophy on stepped bathymetry. This is not Sadourny EC. It is exactly the scheme Alistair would name as the "lean energy-conserving" pick on a quadrilateral C-grid — i.e., MOM6's Arakawa–Hsu (1990) is in the same family. The dossier's "Sadourny EC" label and the audit's "Sadourny EC + Hollingsworth" are both shorthand for code that is actually one rung up.

So on lat-lon there is essentially nothing to do spatially. On MPAS the audit is right: default `pv_scheme="enstrophy"` is wrong-footed if you take Alistair seriously. Switch it to `"energy"` and validate. The Ringler et al. (2010) MPAS energy-conserving PV flux is the right default; the enstrophy variant is the one you fall back to only if you have a separate spectral-blocking control (we don't — we damp via APVM/Smagorinsky, which is a band-aid).

Where I **push back**: framing Crank–Nicolson Coriolis as an "energy-conserving Coriolis" deliverable is misallocated effort. (a) On the discrete inertial oscillator, *forward–backward* (Matsuno) — which both grids already use — already has neutral amplification: $|G|^2 = 1 + O((f\Delta t)^4)$, not just partial conservation. The energy drift is $\sim (f\Delta t)^4$ per step, negligible for $f\Delta t \lesssim 0.1$. CN buys you machine-precision conservation but at the cost of a $2\times2$ implicit solve per cell per substep, and on tilted-${\beta}$ regimes the phase error is *worse* than Matsuno at moderate Courant numbers (see Durran 1991 §2.3 on linear oscillator schemes). (b) The real KE leak in our regime is from spectral blocking and from advection — Megann (2018) tracer-variance diagnostics on MOM6 show advection-driven dissipation is 2–3 orders of magnitude larger than Coriolis-driven dissipation. Spending eng-weeks on CN Coriolis is rearranging deck chairs.

**Recommendations:**
1. Change MPAS default to `pv_scheme="energy"`. Run the existing matrix; if anything breaks (e.g., ACC channel, Eady), it's a sign we were relying on enstrophy dissipation as hidden viscosity.
2. Add a one-line audit note that lat-lon is AL81, not Sadourny — the dossier and audit are both misleading.
3. **Do not** prioritize CN Coriolis. Re-allocate that eng-time to Item 7.

### Item 5 — Tracer advection

- **Stance:** AGREE with Alistair on the 3rd-order target; AGREE-WITH-NUANCE on DST-3 specifically; PUSH-BACK on the current MPAS default of `"upwind"` which is indefensible at any climate-relevant resolution.

Alistair's "3rd-order TVD is fine" is a pragmatic call that I would phrase more sharply: at $1°$ resolution, the difference between $1^\text{st}$-order upwind and a $3^\text{rd}$-order monotone scheme is on the order of an apparent $\kappa_\text{diapycnal} \sim 10^{-3}$ vs $10^{-5}$ m²/s — *two orders of magnitude*. Griffies, Pacanowski & Hallberg (2000) made this point definitively. Megann (2018) and Holmes et al. (2021) confirmed it on MOM6. **Upwind is not a "default" in any production ocean model written after 1995.** That MPAS is shipping `"upwind"` is a soft bug.

On PPM vs DST-3: I disagree mildly with the doc's framing that they are "siblings ... differences at large scales are marginal." PPM (Colella–Woodward 1984, as implemented by MOM6) has *built-in* monotonicity-preserving parabolic reconstruction with a well-characterized limiter; DST-3 (Easter 1993) is a third-order upstream-biased linear scheme paired with a Zalesak-style TVD limiter. In practice on terrain-following or z-star with sharp fronts (overflows, ACC standing meanders), PPM tends to have less limiter-induced flatness at extrema because the parabolic reconstruction concentrates the limiting near sign changes; DST-3 with TVD limiters tends to clip extrema harder. The Eady and ACC runs in our repo would expose this if instrumented — RPE drift would tell the story.

A second push-back: SOM (Prather 1986) is overrated as a "low-mixing case" tool *for differentiable use*. SOM carries sub-grid moments — mean, slopes, curvatures — which means six extra prognostic fields per tracer. The gradient graph through SOM is correspondingly six times larger and the checkpointing footprint blows up. For differentiable workflows you want PPM or DST-3 plus ALE remapping, not SOM. MOM6's choice to keep SOM only for specific tracers reflects the same trade-off.

The CFL footnote in `project_dst3_advection` ("full benefit at CFL > 0.1 or RK3") deserves a sharper read: at low CFL the third-order phase error becomes the dominant accuracy term, and the scheme degenerates toward first-order behavior in the implicit-dissipation sense (this is Easter 1993's own observation). If our production CFLs are ~0.05 (likely given the small barotropic subcycle ratios), DST-3 is barely earning its keep. PPM does not have this CFL-dependence — it is genuinely 3rd-order at all CFL within stability.

**Recommendations:**
1. **MPAS default → PPM (or DST-3 if PPM not yet wired on Voronoi).** Document Eady and ACC RPE drift before/after as evidence.
2. **Lat-lon default → DST-3 today, PPM after Eady/ACC measurement.** "tvd" (2nd-order) is acceptable for a debug default but not a production default.
3. **Add RPE-drift diagnostic to the test matrix** (Ilicak et al. 2012 protocol on lock-exchange and overflow). This is how you actually defend a tracer-scheme choice; the order-of-accuracy plot alone is insufficient.
4. **Keep SOM as a research/diagnostic option, not a recommended default for any production run.**

### Item 6 — Per-term tests

- **Stance:** AGREE strongly. This is the methodological piece that ties everything else together, and it is currently the largest gap.

The audit confirms the bad news: no inertial-oscillation test, no 1D pure-advection convergence test, no 1D diffusion test, no seamount rest-state test. We have rest-state preservation and geostrophic adjustment, and that's basically it for analytical-reference tests. Compared to MOM6's test suite (which includes Hallberg's analytical-reference framework and the full Ilicak 2012 protocol) and MPAS-O's COMPAS-style unit tests, we are running open-loop.

The doc's candidate test list is good. I would refine it with ocean-specific considerations:

1. **Inertial oscillation: run two variants.** (i) Single-cell parcel with full Coriolis on, everything else off — closed-form circle at $2\pi/f$. (ii) Larger-domain inertia-gravity wave initialization with $f \neq 0$ — tests Coriolis-PGF coupling, not just rotation. The first variant catches sign errors; the second catches dispersion-relation errors.

2. **Tracer advection: convergence rate matters more than the absolute value.** A single 1D pure-advection test at one resolution tells you the scheme is wired correctly. The *rate* at which error decreases with resolution tells you whether you're getting 3rd-order behavior or, more commonly, 1st-order behavior because the limiter is constantly active. Petersen, Jacobsen, Ringler & Hecht (2015) on MPAS-O's tests is the right template.

3. **Tracer diffusion: not just Gaussian spread, but explicit $\sigma^2 = 2\kappa t$ slope check.** The constant-$\kappa$ Gaussian self-similarity is a 2-parameter identity, and you should verify *both* parameters (decay amplitude and width-vs-time slope) to detect e.g. wrong $\kappa$ scaling or missing $1/2$ factor.

4. **Linear gravity wave (SW): use the explicit dispersion relation $\omega^2 = gHk^2$.** Initialize a single wavenumber, measure the phase speed at multiple resolutions. This is the test that distinguishes "C-grid Coriolis is approximately implemented" from "C-grid Coriolis has the right dispersion at $f\Delta x / c \sim 1$" — i.e., the famous A/B/C/D grid dispersion comparison from Arakawa & Lamb (1977).

5. **Geostrophic balance: not a single test but a hierarchy.** (i) Linear EOS + flat bottom — analytic check on $f \times u = -g \nabla \eta$. (ii) Nonlinear EOS + flat bottom — checks thermal wind. (iii) Nonlinear EOS + bumpy bottom — this is *Item 7*, and it should not be separated from "geostrophic balance."

6. **Mixed-layer deepening: Kraus–Turner closed form is the wrong reference.** The Kraus–Turner result is itself a 1970s-era parameterization, not a fluid-dynamics analytic solution. The right reference is the entrainment law $h \sim t^{1/2}$ for a constant surface buoyancy flux on an initially linearly-stratified ocean (Niiler & Kraus 1977; Pollard, Rhines & Thompson 1973). This tests whether your KPP entrains at the *correct rate*, not just the correct functional form.

The bigger structural point Alistair is making, which the doc gets right but understates: **per-term tests are the right scale for differentiability checking too.** Gradient correctness through the inertial oscillator (compare `jax.grad` to the analytic derivative of the circular trajectory) is the cleanest unit test for "is autodiff working through this term?" Once you have those, you have a regression net for every future change — not just for accuracy, but for the AD path.

A specific pushback on the "term toggle API" framing: I would push for `disable_*` flags rather than `enabled` flags on the term-level config, with the explicit convention that *production runs reject any config with a disabled term*. The point is to make "test-mode" configurations *loud* in the audit trail. A bare `enabled: bool = True` invites silent misuse.

**Recommendations:**
1. **Build the per-term test suite in two passes:** (a) "smoke tests" at one resolution (catch wiring errors), (b) "convergence-rate tests" at three resolutions (catch order-of-accuracy regressions). (b) is the one that earns its keep for choosing defaults.
2. **Add gradient-correctness variants** that compare `jax.grad` to the analytical derivative on each test.
3. **Make `disable_*` flags loud** — a separate config flag like `test_mode: bool = False` should be required to disable any production term.
4. **Inertial oscillation first**, before any other dycore work. It's the smallest test that exercises the largest number of code paths.

### Item 7 — Flow past a bump

- **Stance:** AGREE strongly. This is the single most actionable test in the entire follow-up list.

The Beckmann & Haidvogel (1993) seamount test is the canonical PGF discriminator, and it does something no other test in our matrix does: it converts a discretization choice into a single scalar (max|u| over a fixed integration window) that *cannot* be gamed by tuning viscosity or damping. Either the PGF cancellation works on terrain, or it doesn't.

The expected ordering, drawn from the published literature on z-coordinate and z* models with similar resolution:
- Plain centered/Adcroft PGF in z* with a Gaussian seamount: max|u| ~ $10^{-2}$ to $10^{-3}$ m/s (Shchepetkin & McWilliams 2003, Table 1 region).
- SMC03 (density Jacobian): max|u| ~ $10^{-5}$ m/s — 1–2 orders better.
- AHH08 (finite-volume with analytic vertical integration): max|u| ~ $10^{-7}$ m/s for linear EOS, slightly worse for Wright but still ~ $10^{-6}$ m/s.

Our `pgf_ahh08.py:61` docstring's "ETOPO+ico4: adcroft ~1.4e-6, smc03 ~8.6e-8" is consistent with this ordering, *but* ETOPO+ico4 is a forced run with real topography — it conflates the PGF error with everything else happening (Coriolis, advection, drag). A controlled seamount-at-rest test is **the** clean measurement.

A specific design recommendation that diverges from the dossier's "Gaussian bump + linear stratification" sketch:
- **Use two stratifications**: (i) a linear $N^2$ profile (Beckmann–Haidvogel 1993 original), (ii) an exponential-thermocline profile (Shchepetkin–McWilliams 2003) which loads the PGF cancellation more heavily in the upper ocean where most of our "real" applications live.
- **Use two bump geometries**: a smooth Gaussian (smooth-coefficient cancellation regime) and a steeper sloped seamount with peak slope $\sim 0.1$ (DOME-like, exercises the limiter regime where finite-volume schemes pull ahead).
- **Run non-rotating and rotating**. The non-rotating case isolates PGF; the rotating case is the production-relevant regime, but errors there mix PGF with Coriolis/geostrophic adjustment of any spurious flow.
- **Vertical resolution sweep**: 30, 60, 90 levels. The PGF error per level should decrease as $O(\Delta z^2)$ for SMC03 and faster for AHH08 — this is the convergence-rate evidence that distinguishes schemes.

The cross-grid comparison matters more than the dossier suggests. If MPAS-AHH08 doesn't outperform lat-lon-Adcroft by at least an order on the same seamount, something is broken — likely in the lat-lon AHH08 wiring (which doesn't exist today) or in how the lat-lon operators handle non-flat bathymetry at the AL81 vertex weights.

**Recommendations:**
1. **Build this test before Item 1's refactor.** It's the baseline-establishing measurement that lets you defend any operator-layer change later.
2. **Wire AHH08 onto lat-lon.** `pgf_ahh08.py` is grid-agnostic in spirit; the lat-lon `_valid_pgf` set excludes it artificially. Same machinery, just needs the dispatch entry.
3. **Make this the gating test for the production PGF default**, replacing the current implicit defaults.
4. **Document RPE drift alongside max|u|.** Petersen et al. (2015) point out that max|u| can be misleading if the spurious flow is geostrophically adjusted to a steady (incorrect) state — RPE drift captures the energy injection rate.

---

## Where I would push back against Alistair

I have evidence to disagree with Alistair on four points. None are showstoppers; the disagreements are about *priority and framing* rather than substance.

1. **"Energy-conserving Coriolis" as a temporal goal (Crank–Nicolson).** Matsuno on the discrete inertial oscillator already has neutral amplification to leading order; the energy drift per step is $O((f\Delta t)^4)$, which for typical ocean $f\Delta t \lesssim 0.1$ is below all other error sources. CN buys exact conservation at the cost of a per-cell $2\times2$ implicit solve and worse phase error at moderate Courant numbers (Durran 1991 §2.3). The eng-effort is better spent on the spatial PV-flux choice on MPAS (which *is* wrong) and on Item 7 (which gives you a measurable handle on every PGF/Coriolis trade-off).

2. **"DST-3 TVD is fine."** It is fine, but PPM is better, *especially* for differentiable workflows. The TVD limiter in DST-3 produces sharper clipping at extrema, which means more spurious diapycnal mixing on tilted isopycnals (the Ilicak et al. 2012 lock-exchange RPE diagnostic would expose this). PPM's parabolic reconstruction is more limiter-friendly. Alistair would not actually object — MOM6's own default is PPM — but the doc's reconstruction implies near-equivalence, which I disagree with.

3. **RK3/RK4 dismissal (if accurately reconstructed).** SSP-RK3 is the right outer integrator for differentiable training with positivity constraints. The "cost per CFL" argument is correct in absolute throughput but ignores the value of forward-mode positivity preservation along the gradient path. Do not deprecate SSP-RK3 from `dispatch.py` based on this remark.

4. **Sequencing of Item 2b before Item 1.** The doc's "Item 2b is close to done" framing risks defaulting to AHH08 on MPAS before measuring it on the seamount test, and risks landing a "coordinate-invariant" PGF that still composes with coordinate-baked horizontal operators. The clean sequence is: Item 7 (seamount baseline) → Item 1 (metric-driven operators) → Item 2b (one PGF across coordinates). Item 2b done before Items 7 and 1 leaves debt.

A meta-disagreement: the doc treats the seven items as roughly parallel work streams. They are not. Item 7 is a measurement that gates Items 2b and 4. Item 1 is a refactor that gates Item 2b's strong form. Item 6 is the methodology that makes every other item's claim defensible. The dependency graph matters more than the item count.

---

## Cross-cutting observations

1. **The lat-lon Coriolis spatial scheme is AL81, not Sadourny EC.** Both the dossier and the audit's headline mis-state this. `pv_flux_al81_partial_cell` (`latlon_cgrid_operators.py:2435`) is the 12-point Arakawa–Lamb 1981 triad with partial-cell weighting (Le Sommer et al. 2009 / Stewart & Dellar 2016). This is one rung better than the documented state. The MPAS default (`pv_scheme="enstrophy"`) is *worse* than the doc implies because the audit didn't surface that the lat-lon side is already best-in-class.

2. **The validation harness is the single biggest debt, not the code.** The architectural compliance with Alistair's framework is actually pretty good — AHH08, SMC03, AL81, DST-3, PPM, SOM, WENO5/7 are all wired. What we cannot do today is *defend* any of those choices empirically because we have no inertial-oscillation, no seamount, no Ilicak RPE-drift, no 1D convergence-rate tests. Building the harness first makes every subsequent default change defensible.

3. **Defaults vs capabilities is a deliberate-versus-accident question.** Several of the trailing defaults (MPAS `"upwind"` tracer, MPAS `"enstrophy"` PV, lat-lon `"tvd"` tracer, `"adcroft"` PGF) look like *historical* defaults that were set when only those options were wired and never updated. None of these would survive a code-review pass that asked "what is the production-relevant choice today?" A short PR cycle to bring defaults in line with capabilities would close a lot of nominal-vs-actual gaps cheaply.

4. **Differentiability shapes the trade-off space differently than MOM6's.** Alistair's framework is built around production cost and Fortran-codebase ergonomics. Our framework should weight (a) gradient stability through limiters (favors PPM over DST-3, favors smooth tapers everywhere), (b) checkpointing cost in long barotropic subcycles (favors fewer prognostic variables — argues *against* SOM as default), (c) AD through implicit solves (we should use implicit-function-theorem differentiation on any future implicit Coriolis or implicit mixing). These considerations don't appear in the doc, and they should.

5. **The cubed-sphere face-boundary instability is the elephant in the room** and it is not on the 7-item list. The metric-driven refactor (Item 1) is the right architectural piece, but the face-boundary problem may require an algorithmic change beyond metric-cleanup — possibly a different mimetic operator family at panel edges, possibly explicit vector rotation in halo exchange, possibly a different PV-flux stencil that doesn't straddle face edges. This deserves its own item.

---

## One change that would matter most

**Build the seamount-on-stratification rest-state test (Item 7) and make it a gating test for the production PGF default.** This is the single most leveraged piece of work because it converts every PGF design choice from a literature-cited assertion into a measurable scalar (max|u|, RPE drift) on our own code, our own grids, our own bathymetry. Once this test exists:

- The lat-lon-vs-MPAS PGF comparison becomes a graph instead of an argument.
- Defaulting from `"adcroft"` / `"centered"` to `"smc03"` or `"ahh08"` becomes a one-line config change with evidence attached.
- The Item 1 metric-driven refactor gains a non-trivial regression target (any operator-layer change must preserve or improve the seamount max|u|).
- The Item 2b "coordinate-invariant PGF" claim gets a quantitative criterion (max|u| should be the same across coord choices in the rest state — by construction, since the ocean is at rest).
- The MPAS PV-flux default change (`"enstrophy"` → `"energy"`) can be evaluated on the same setup, since spurious flows generated by PGF errors are subsequently advected by the PV-flux scheme, and energy-conserving PV flux should not amplify them as the enstrophy variant might.

The implementation cost is small (Gaussian bump bathymetry override on the existing `rest_state.py` machinery, one or two stratification profiles, time-loop with $\max\|u\|$ and RPE diagnostics) — probably less than a week of focused work including the test wiring. The downstream value is that *every other item in Alistair's list becomes defensible with evidence instead of advocacy*. That is the change with the highest ratio of decision-leverage to engineering cost in the entire follow-up list.

---

## Key references cited

**C-grid and PV-flux discretization:**
- Arakawa & Lamb (1977), *Methods Comput. Phys.* 17 — foundational C-grid.
- Arakawa & Lamb (1981), *MWR* 109 — combined energy + enstrophy PV flux. **Our lat-lon scheme.**
- Sadourny (1975), *JAS* 32 — EC and EN forms.
- Arakawa & Hsu (1990), *MWR* 118 — MOM6's PV-flux family.
- Ringler, Thuburn, Klemp & Skamarock (2010) — TRiSK / MPAS.
- Le Sommer et al. (2009); Stewart & Dellar (2016) — AL81 partial-cell weights.

**Pressure gradient force:**
- Adcroft, Hallberg & Harrison (2008) — **AHH08**, finite-volume PGF.
- Shchepetkin & McWilliams (2003) — **SMC03**, density-Jacobian PGF.
- Beckmann & Haidvogel (1993) — seamount test (canonical PGF validation).
- Haidvogel & Beckmann (1999), *Numerical Ocean Circulation Modeling*.
- Adcroft & Hallberg (2006) — ALE / coordinate-invariance framework.
- Mellor, Oey & Ezer (1998); Haney (1991) — sigma PGF errors.

**Time stepping:**
- Hallberg (1997), *MWR* 125 — split-explicit barotropic stability.
- Higdon (2005) — two-level time stepping for layered models.
- Shchepetkin & McWilliams (2005) — ROMS split-explicit with weighted averaging.
- Durran (1991), *MWR* 119 — linear oscillator schemes (Matsuno vs CN vs RK).
- Shu & Osher (1988); Williamson (1994) — SSP-RK and RK phase error.

**Tracer transport:**
- Colella & Woodward (1984) — PPM.
- Easter (1993) — DST-3 with TVD limiters.
- Prather (1986) — SOM (second-order moments).
- Lin & Rood (1996) — multidim flux-form transport, FV3 foundation.
- Griffies, Pacanowski & Hallberg (2000) — spurious diapycnal mixing analysis.
- Ilicak, Adcroft, Griffies & Hallberg (2012) — RPE-drift diagnostic protocol.
- Megann (2018); Holmes et al. (2021) — production diagnostic.
- Petersen, Jacobsen, Ringler & Hecht (2015) — MPAS-O transport convergence-rate tests.

**SW limit and validation:**
- Williamson, Drake, Hack, Jakob & Swarztrauber (1992) — canonical SW test suite.
- Galewsky, Scott & Polvani (2004) — barotropic instability test.
- Niiler & Kraus (1977); Pollard, Rhines & Thompson (1973) — mixed-layer entrainment laws.

**Vertical coordinates and ocean dycore architecture:**
- Adcroft, Hill & Marshall (1997) — partial bottom cells.
- Adcroft & Campin (2004) — z*.
- Adcroft et al. (2019) — MOM6 architecture and supergrid pattern.
- Ronchi, Iacono & Paolucci (1996); Putman & Lin (2007); Harris & Lin (2013) — cubed-sphere.

---

*End of independent review.*
