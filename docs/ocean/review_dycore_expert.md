# Dycore-expert independent review

*Author: Claude in the dycore-expert persona. Independent of the
`review_claude_audit.md` author. Builds on that audit's
quantitative findings; lens is "would I sign off on this dycore for
production climate use?"*

## Headline

legoESM ocean is closer to Alistair's framework than the dossier
suggests *in capability* but is materially out of alignment *in
defaults and in test coverage*. The architecture has the right
ingredients on the shelf — SMC03 and AHH08 PGF, Sadourny EC Coriolis
on lat-lon, Matsuno on both grids, DST-3/PPM/SOM/WENO5/WENO7 tracer
schemes — but the production knobs ship as `tvd`/`upwind` tracers,
enstrophy-conserving PV flux on MPAS, and `adcroft`/`centered` PGF.
The single biggest piece of architectural debt is not on Alistair's
list explicitly but is what makes Item 1 a real refactor and Item 2b
non-trivial: the lat-lon C-grid operators have 127 inline metric
references with no `MetricBundle` boundary, so "swap the coordinate"
is a 2810-line edit, not a 50-line one. Items 4 (Coriolis) and 7
(seamount) are higher-value than the items themselves convey: they
are the two tests that would, between them, *settle* most of the
other questions.

## Per-item assessment

### Item 1 — Coordinate-baked operators

- **Stance:** AGREE; the framing understates the scope.

The audit's "127 inline `cos`/`R_earth`/`radius` refs in 2810 lines"
is the right number to internalise. In an FV3 or MOM6-style codebase
the equivalent count inside `sw_core.F90` or `MOM_lateral_mixing_coeffs.F90`
is near zero — metrics are state, not computation, and operators
consume `dxT, dyT, IdxCu, IdyCv, areaT` etc. as arrays. legoESM's
lat-lon operators conflate three things that should be separate:
(i) the C-grid stencil (a topological fact), (ii) the metric tensor
(a per-grid array), and (iii) the coordinate identity (currently
hardcoded as `cos(lat)*R_earth*dlon`). The MOM6 supergrid pattern is
the right target, *but the supergrid itself is not necessary* — what
matters is the operator/metric boundary. The cubed-sphere ocean path
is currently a separate codebase from lat-lon (different files,
different operators), which doubles the surface area. A
`QuadrilateralCGridMetrics` container, even if initially produced
*from* the existing `LatLonGrid` and `CubedSphereGrid`, would let the
operator layer be unified, and that unification is what makes
Mercator and tripolar live as data not code.

Recommendations:
1. Make the operator/metric boundary explicit *before* attempting
   coordinate refactors. The boundary is `compute_grad_*`,
   `compute_div_*`, `compute_curl_*`, `kinetic_energy_*`,
   `coriolis_cgrid`, `pgf_*`, plus the vertex/edge averaging
   helpers — about 15 functions in `latlon_cgrid_operators.py`.
   Pass metrics in; do not let operators reach into `grid.radius`
   or compute `cos(lat)`.
2. Skip the supergrid as a first pass. Define metric arrays directly
   at `T`, `u`, `v`, `q` points; bit-for-bit reproduce current
   lat-lon. Add the supergrid only if a third grid family forces it.
3. Treat MPAS (TRiSK) as out of scope for this refactor. The mimetic
   family is different (per-edge length, area, cell connectivity), and
   conflating it with the quadrilateral operator library buys nothing.
   The right level for grid-agnosticism between MPAS and quadrilateral
   C-grid lives one layer up, in the dycore step function, not in the
   operators.

### Item 2 — Layered + coordinate-invariant

#### 2a — Shallow-water limit at `nlev=1`

- **Stance:** AGREE strongly; this is the cheapest high-value action
  in the audit.

The zero hits for `nlev == 1` special-cases is the right outcome — it
means the ocean dycore is genuinely layered, not a 1-layer SW model
with extra levels bolted on. This is non-trivial; many codes
(including older MOM6 branches) have `if (nlev == 1)` litter at
exactly the points the dycore most needs to be uniform. legoESM has
already done that work. So Williamson on `nlev=1` ocean is *not* a
re-derivation, it is a test-harness wiring task.

The atmosphere SW harness at `tests/atmosphere/shallow_water/` and
`tests/unit/test_williamson2_cdgrid.py` is the right scaffold to
borrow. Williamson 2 (steady geostrophic) catches PGF + Coriolis
discretisation errors; Williamson 5 (mountain) catches PGF over
topography (which is also Item 7's seamount in a SW guise);
Galewsky catches barotropic instability — which is exactly the
class of bug the ACC-channel and Eady experiments are sensitive to.
Adding these three tests on the ocean `nlev=1` path would do more
to harden the lat-lon dycore than three months of "look at the
gyre output and squint" debugging.

The MPAS path needs its own version. MPAS-A has the standard
Williamson suite (Skamarock et al. 2012); porting that to the
MPAS-Voronoi ocean dycore is harder because the test cases must be
re-initialised on the Voronoi mesh, but the closed-form references
are the same.

Recommendations:
1. Williamson 2 on the lat-lon ocean `nlev=1` path — first priority.
   This is the steady geostrophic balance test, runs in seconds, and
   measures `max|u_error|` after long integration.
2. Williamson 5 (zonal flow over an isolated mountain) — second
   priority. This *is* the seamount test in SW form (see Item 7) and
   should ship in the same PR as the Item 7 seamount test.
3. Galewsky barotropic instability — third priority. The high-quality
   reference solution makes it the right test for "does the C-grid
   handle nonlinear vorticity dynamics correctly", which is where
   AL81 vs energy vs enstrophy PV-flux choices show up.
4. Skip the MPAS Williamson port until after the lat-lon SW tests
   are landing artifacts. The MPAS-A Voronoi SW infrastructure can
   be borrowed in a separate PR.

#### 2b — Coordinate-invariant PGF

- **Stance:** AGREE; status is far better than the dossier said but
  defaults are wrong.

The infrastructure is genuinely good. `pgf_smc03.py` and
`pgf_ahh08.py` both exist; AHH08 with the Wright EOS gives
machine-precision rest-state preservation on partial cells per its
own derivation (the analytic ρ(z) integral cancels exactly between
columns on horizontally-uniform stratification). For coordinate
invariance, AHH08 is the strongest single PGF: by construction it
works the same in z, z*, σ, ALE because the integral is taken over
the *physical* z-range of the wet cell, with the coordinate only
defining the cell bounds. SMC03 is a strong second; the
density-Jacobian form is largely coordinate-invariant in practice.

The two real issues:

(i) **AHH08 is not exposed on lat-lon.** The valid set is
`{"adcroft", "smc03"}`, default `"adcroft"`. There is no good reason
not to add AHH08 once the operator/metric refactor (Item 1) is at
least scoped, since AHH08's column primitives are already
grid-neutral (`pgf_ahh08.py` is generic; the edge wrapper just needs
a lat-lon version next to `mpas_partial_cell_helpers.py`).

(ii) **MPAS default is `"centered"`.** Per AHH08's own docstring, on
ETOPO+ico4 centered gives `~1.3e-4 m/s/step` rest-state PGF residual
versus AHH08's machine-zero. This is a four-orders-of-magnitude error
budget that propagates into spurious bottom-trapped currents — the
audit notes this matches `project_mpas_etopo_instability.md`'s
"algorithm-class limit" finding. Centered PGF is the wrong production
choice and AHH08 is sitting on the shelf.

The bigger statement: coordinate invariance is *the* design
principle that buys ALE later. Once PGF is AHH08, switching from z*
to ALE is a regridding/remapping change in the vertical layer
construction, not a dycore rewrite. That is worth saying explicitly
in the dossier — Alistair will recognise it.

Recommendations:
1. Switch MPAS production default to `pgf_scheme="ahh08"` immediately
   (after Item 7 seamount test confirms the win). Keep `"centered"` as
   a regression option.
2. Add AHH08 to lat-lon: define the edge wrapper in
   `latlon_cgrid_operators.py`, expand `_valid_pgf` to include
   `"ahh08"`, gate on `eos="wright"`.
3. Audit the rest of the dycore for z*-aware operators: tracer
   remapping, `compute_ocean_jacobian`, vertical advection, eta-h
   coupling. Coordinate invariance is only meaningful if it goes
   beyond PGF.

### Item 3 — Per-term time stepping

- **Stance:** AGREE; the dycore already does this — the worksheet
  framing is the problem.

The "outer integrator" axis on the worksheet is a category error. No
serious ocean dycore picks one stepper for the whole RHS. MOM6's
structure is canonical: forward–backward in the barotropic substeps
(Hallberg 1997), a predictor–corrector outer step, alternating
forward–backward Coriolis on momentum, leapfrog with Robert–Asselin
or a Heun-on-tracer for the slow tracer step. ROMS, NEMO, MOM5 all
land in the same neighbourhood. The choice is *per term*, and the
production design decision is which combination of per-term schemes
yields a stable, accurate, energy-budgeted whole. Alistair's
"RK3/RK4 not necessarily good" is, I am almost certain, about both
items he mentioned: (a) cost — RK3 buys ~1.4× CFL for 3× tendency
evaluations versus RK2; (b) more importantly, the rotational-mode
phase error of higher-order RK is *not* better than a properly
tuned forward–backward in the gravity-wave subcycle. The 4th-order
phase error of RK4 on a `iωΔt` eigenvalue is `O((ωΔt)^5)` *amplitude*
but the *phase* error is still `O((ωΔt)^4)`, which is the same scaling
as Matsuno on a per-substep basis once you account for the substep
ratio. This is the FV3 community's lived experience too: SSP-RK3
on the Lagrangian-vertical dynamics is fine, but the production
gain comes from the acoustic substepping, not the outer RK order.

legoESM's reality: the per-term stepping is already in `_step_impl`
for both grids, just not exposed at the config layer. The worksheet
single-axis framing leaks into discussions and makes the audit harder
than it should be. The `src/legoesm/timestepping/dispatch.py` module
exists but is unused by the ocean — that is fine; the ocean does not
benefit from a unified `dispatch_integrator` because its step is
*structurally* split-explicit.

Recommendations:
1. Replace the worksheet's single "outer integrator" axis with a
   per-term table: `{Coriolis, PGF, hadv-mom, vadv-mom, hadv-tracer,
   vadv-tracer, vmix-mom, vmix-tracer, barotropic}` × `{scheme,
   reference, stability character, what limits its dt}`. Make it
   describe what the code already does.
2. Do *not* wire ocean to `timestepping.dispatch`. The dispatch module
   is a useful abstraction for the atmosphere's monolithic RK stepper,
   not for split-explicit ocean.
3. Email Alistair the explicit two-part RK3/RK4 read (cost ratio +
   phase error), and ask him to confirm. The answer affects whether
   the per-term audit takes 1 week (just document and expose) or 4
   weeks (add CN options where they help).

### Item 4 — Energy-conserving Coriolis

- **Stance:** AGREE on spatial EC; PUSH-BACK on temporal CN as a
  priority; AGREE that MPAS default is wrong.

The (4a) spatial vs (4b) temporal split in the follow-up doc is the
right disambiguation, and the answer is different on each grid.

*Lat-lon is fine.* `coriolis_cgrid` at `latlon_cgrid_operators.py:381`
is the Sadourny (1975) EC form — four-point average of `v` to a
`u`-point and vice versa — with the Hollingsworth correction added
in commit `d0183817`. This is exactly what Alistair recommends
spatially. Forward–backward (Matsuno) is *almost* energy-neutral on
the linear rotation operator (amplification factor of `1 + O((fΔt)^3)`
per step) — for `fΔt ~ 0.05` this is `~10^-4` per step, fine on
multi-year integrations. Crank–Nicolson Coriolis is genuinely exact
in discrete KE, but the upgrade from Matsuno is sub-1% at typical
ocean `fΔt` and the residual error is dominated by other terms
(spatial truncation in the EC averaging, mode-split coupling errors).
I would not prioritise CN-Coriolis.

*MPAS spatial PV-flux is misaligned and is the highest-leverage
single config change in the audit.* `pv_scheme="enstrophy"` is the
opposite recommendation: it conserves enstrophy at the cost of leaking
energy at the grid scale, which on a Voronoi mesh manifests as the
`ζ`-checkerboard mode the APVM is meant to damp. The config docstring
acknowledges this ("avoids the ζ-checkerboard null mode of the
energy-conserving scheme") — that is a backwards justification. The
right fix is to use the EC PV flux *and* control the null mode with a
genuine biharmonic-`ζ` damper (already implemented as `K_zeta_bih`)
or with the AL81 mixed scheme. The audit notes AL81 is not in the
dispatch — that is a one-week implementation against the existing
Ringler et al. (2010) literature; the helper functions for vertex
PV and vertex thickness are already there.

The dossier's "MPAS Coriolis uses Heun" is wrong: both grids use
Matsuno (`_forward_backward_coriolis_mpas_3d`). This is a
documentation bug, not a dycore bug.

Recommendations:
1. Implement AL81 PV-flux on MPAS (`mixed` with sensible default α);
   make it the new MPAS default once validated on the existing matrix.
   If AL81 is too invasive, *temporarily* switch the default to
   `"energy"` with `K_zeta_bih > 0` on, and treat `"enstrophy"` as a
   diagnostic option.
2. Do *not* prioritise CN-Coriolis. Defer until the inertial-oscillation
   test (Item 6) reveals it as the bottleneck — which I do not expect.
3. Fix the dossier: lat-lon and MPAS both use Matsuno-family
   forward–backward Coriolis. Heun is reserved for the outer step on
   the (atmosphere) shallow-water tests, not the ocean dycores.

### Item 5 — Tracer advection

- **Stance:** AGREE on the recommendation; PUSH-BACK on the
  characterisation of where we are.

Alistair's "3rd-order TVD is fine; DST-3 not bad" is the standard
production take — MOM6 ships PPM-FCT (Colella-Woodward 1984 with
Zalesak FCT), NEMO ships UBS or weighted MUSCL, ROMS ships MPDATA,
all roughly third-order with monotonicity preservation. legoESM has
the schemes: DST-3, PPM, PPM-FCT, DST-3-multidim, SOM, WENO5/7. The
audit reports lat-lon default is `"tvd"` (2nd-order) and MPAS default
is `"upwind"` (1st-order). The lat-lon dispatch table at
`ocean_model_latlon_cgrid.py:129–197` confirms this. The MPAS
dispatch at `ocean_model_mpas.py:166,657` reveals a sharper problem:
**MPAS supports only `upwind` and `tvd`** — no DST-3, no PPM, no SOM,
no WENO. The dossier's claim that DST-3 is the default *anywhere* is
wrong, and the MPAS scheme list is narrower than lat-lon by 6 options.

This is the second-largest capability/default gap in the audit (after
the MPAS PV-flux default). 1st-order upwind on a production ocean
tracer is not defensible — the implicit diffusion is `~|u|·Δx/2`,
which on ico4 (Δx ~ 100 km) gives `K_implicit ~ 5e3 m^2/s` for
`|u|~0.1 m/s`. That is GM-eddy-flux-scale numerical noise, before
any explicit `K_h` is added.

DST-3 vs PPM: they are siblings. DST-3 is slightly cheaper, slightly
less monotone; PPM is more robust on sharp fronts. For ocean tracers
(T, S smooth on most scales) either is fine. Adding PPM as a config
option on lat-lon is already done; making it the default on MPAS is
real work because the Voronoi PPM stencil requires the
unstructured-mesh PPM reconstruction (this is what MPAS-O upstream
does; we would be porting their algorithm).

The `project_dst3_advection` note about "full 3rd-order benefit at
CFL > 0.1 or with RK3" is real and is the right caveat. Most ocean
runs are at CFL ~ 0.3 for advection so DST-3 *is* in its sweet spot;
that note is more relevant for very-low-CFL idealised tests.

SOM stays for the Eady and ACC channel (low-mixing) cases.

Recommendations:
1. Lat-lon: change default `tracer_advection` from `"tvd"` to
   `"dst3"`. Run the production matrix; verify Eady, ACC, gyre, AMIP
   stay stable. This is a 1-line change with a finite-effort
   validation tail.
2. MPAS: implement DST-3 on Voronoi (port the MPAS-O reconstruction;
   the unstructured-PPM literature is well-trodden). Make it the
   default once validated. If that is too much work in the current
   window, at minimum switch the MPAS default to `"tvd"` — anything
   higher-order than `"upwind"`.
3. Document the SOM use case explicitly: when minimising spurious
   diapycnal mixing is the goal (Eady, ACC channel, internal-wave
   studies), default SOM. The current scheme selection is per-config
   already; just write down the recommendation.

### Item 6 — Per-term tests

- **Stance:** AGREE; the framing is right but the priority within the
  table needs reordering.

Per-term tests are the right unit of dycore testing. The audit
confirms partial term-toggle coverage (GM/Redi `None`, Smag/Leith
zero-coefficient, MAXVEL clip, APVM) and zero coverage on the harder
toggles (Coriolis, PGF, momentum advection). The "test-mode config
factory" pattern Alistair implied is the right answer: each scheme
gets an `enabled: bool = True` field, plus a `make_minimal_config(
term: Literal[...])` factory in `ocean/experiments/test_configs.py`.

The candidate test list in the follow-up doc is good but the
ordering by *dycore-leverage* should be:

1. **Linear gravity wave** (`ω² = gHk²`). The cheapest test, runs on
   `nlev=1`, validates PGF + free surface + the barotropic substep
   simultaneously. Should be done *before* anything else because
   gravity-wave dispersion is the first thing that breaks when
   barotropic-mode-split coupling is wrong (`project_barotropic_solver`).
2. **Inertial oscillation.** Closed-form (circle at period `2π/f`),
   tests Coriolis + time-stepper in isolation. Direct support for the
   Matsuno-vs-CN Coriolis question (Item 4).
3. **Geostrophic balance preservation** (= Williamson 2 at nlev=1,
   so this is Item 2a). Tests PGF + Coriolis + EOS together.
4. **Seamount rest-state** = Item 7. The PGF-over-topography test.
5. **1D pure tracer advection** with sinusoidal IC, measure order of
   accuracy at multiple resolutions. Validates Item 5.
6. **1D Gaussian diffusion**, validates the diffusion operators
   (cheap, mostly catches regressions).
7. **Kraus–Turner mixed-layer deepening**, validates vertical mixing.

(7) is *not* a dycore-leverage test — it is a physics-parameterisation
test, and the dycore-expert lens treats it as separate from the
dynamical-core test suite. The first six are dycore.

The differentiability angle in the follow-up is well-taken: the same
tests double as gradient-correctness tests. `jax.grad` on the inertial
oscillation trajectory has a closed-form derivative; that is the right
unit-test for the JAX/autodiff path through the Coriolis solver.

Recommendations:
1. Build the test-mode config factory first. Pattern:
   `make_minimal_config(term="coriolis", grid="latlon") -> Config`
   returns a config with everything off except Coriolis, with a
   sensible `dt`, `nlev`, domain. This is the prerequisite for the
   tests themselves.
2. Land the linear gravity wave + inertial oscillation tests in the
   *same* PR as the test-mode factory. Both have closed-form
   references and both run in <1s.
3. Make per-term tests *gradient-checked*: each test verifies
   `jax.grad(final_state_summary)(initial_state)` against an
   analytical gradient. This is the contribution legoESM can make
   over and above what MOM6/MPAS-O have.
4. Keep the integrated tests (gyre, ACC, Eady) as the *system* tests;
   add per-term tests as the *unit* tests. Both layers needed.

### Item 7 — Flow past a bump

- **Stance:** AGREE; this is the single most diagnostic test the
  ocean dycore is missing.

Beckmann & Haidvogel (1993) and its later extensions (Shchepetkin &
McWilliams 2003, Adcroft & Hallberg 2006) are *the* canonical PGF
benchmark, and they are missing from legoESM. The audit confirms
`rest_state_stratified_with_land` runs with scalar `H_max=5500`, i.e.
flat bathymetry — so the existing rest-state preservation tests
exercise the EOS-pressure path but not the PGF-over-topography path.
That is the path that matters, because PGF cancellation errors over
varying bathymetry are *the* mechanism that converts a sigma- or z*-
coordinate ocean from numerically clean to numerically polluted.

The test is also a forcing function on every other Adcroft item:

- **Item 2b:** the seamount test is the metric that distinguishes
  `centered`/`adcroft`/`smc03`/`ahh08`. The numbers
  (1.3e-4 → 1.4e-6 → 8.6e-8 → ~1e-12) from `pgf_ahh08.py:61` exist
  *because someone already ran a seamount-like benchmark*; landing
  the test puts those numbers in CI.
- **Item 1:** when operators are coordinate-invariant, the seamount
  test passes on lat-lon, Mercator, cubed-sphere with the same
  closed-form expectation. That is how you *know* the metric
  refactor preserved correctness.
- **Item 6:** seamount-on-rest-stratification *is* the canonical
  per-term test for "PGF in isolation". Drop forcing, drop Coriolis
  (or run both variants), measure `max|u|(t)`.

The reference setup I would adopt (off the top of my head, confirm
with Haidvogel & Beckmann 1999 monograph or the BH93 paper itself):
basin ~320 km × 320 km, Gaussian seamount of 1 km amplitude on a
4 km flat-bottom basin, exponential stratification with
`N^2(z) ~ 1e-4 e^(z/1000)` s^-2, no surface forcing, no Coriolis (or
optional f=1e-4), run 10 days, report `max|u|(t)` and the spurious
KE budget.

Recommendations:
1. Implement `rest_state_stratified_seamount` as a new test-matrix
   entry in `src/legoesm/ocean/experiments/`, reusing
   `rest_state.py`'s init machinery with a Gaussian-bump `H_bathy`
   override.
2. Run all four PGF schemes (`centered`, `adcroft`, `smc03`, `ahh08`)
   on identical setup. Make the four `max|u|(t)` curves a CI artifact;
   the ordering must match the docstring claims.
3. Run with both `f=0` (pure PGF test) and `f=1e-4` (geostrophic
   adjustment over topography). The latter is a more demanding
   integrated test and connects to Item 4.
4. Cross-grid: same test on MPAS-Voronoi (ico4 resolution) — this
   is also the right benchmark for the AHH08 default switch.

## Cross-cutting observations

1. **The capability/default gap is the dominant story.** Across Items
   2b, 4, 5, the "better" scheme is already in source. SMC03 + AHH08
   PGF, Sadourny EC + Hollingsworth, DST-3/PPM/SOM/WENO — all
   implemented. What is *defaulted* (`"adcroft"`/`"centered"`,
   `"enstrophy"`, `"tvd"`/`"upwind"`) is the conservative end of every
   axis. Adcroft will read this as "you wrote the algorithms but
   don't use them" — which is mostly fair. The remediation is mostly
   a `default = "X"` change plus the seamount test (Item 7) to prove
   the win empirically.
2. **MPAS lags lat-lon on tracer advection but leads on PGF.** MPAS
   has only `upwind`/`tvd` (vs lat-lon's 9 options) but is the only
   grid with AHH08. The asymmetry is accidental, not principled.
3. **The cubed-sphere ocean is invisible in this review.** The audit
   focuses on lat-lon and MPAS-Voronoi. The cubed-sphere ocean
   (`ocean_pe_cdgrid.py` etc.) has its own face-boundary instability
   story (see `project_cubesphere_ocean_instability.md`). Whatever
   architectural conclusion is reached for lat-lon must also reach
   the cubed-sphere code or the duplication grows.
4. **No memory of `nlev=1` special-casing is unusually good.** Most
   ocean dycores have it; legoESM does not. Williamson on the ocean
   dycore is a real, cheap, scientific win — not a "nice to have".
5. **Documentation debt is now a scientific risk, not just a tidiness
   issue.** The dossier sent into the Adcroft conversation overstated
   alignment on three items (AL81, DST-3 default, Heun Coriolis).
   That risks the next conversation being calibrated against the
   wrong baseline. An erratum is worth the half-day before the next
   exchange.

## Disagreements with the items as framed

- **Item 1 understates the operator-vs-metric refactor.** It is
  framed as a clean two-step plan (audit, then prototype). The
  reality is that the operator surface area is ~15 functions and
  ~2810 lines with deeply tangled coordinate assumptions. A clean
  refactor takes weeks, not days, and risks regressions on every
  validated experiment. The plan needs an explicit bit-for-bit
  reproduction phase before any new coordinate is added.
- **Item 3 misses that the *outer integrator* abstraction is harmful.**
  The follow-up doc treats it as a worksheet-cleanup issue. From the
  dycore-expert lens, presenting time integration as a single axis is
  a *design* mistake that leaks into the API. The fix is not just to
  rewrite the worksheet — it is to delete the "outer integrator"
  config field from the dossier mental model entirely and replace it
  with the per-term table.
- **Item 4's spatial-vs-temporal split is right, but the priority is
  inverted.** The doc treats both (4a) and (4b) as live work. From
  the dycore-expert standpoint, (4a) on MPAS is high priority
  (change the default) and (4b) is low priority (Matsuno is fine).
  The follow-up doc gives them equal weight; they should not.
- **Item 5's framing as "DST-3 is fine" understates the MPAS gap.**
  The audit reveals MPAS has only upwind/TVD — that is a
  scheme-coverage gap, not a default-choice gap. The follow-up doc
  treats it as a default-choice issue.
- **Item 6's candidate list mixes dycore and physics tests.** The
  Kraus–Turner mixed-layer test is a physics-parameterisation test,
  not a dycore test. Separating the two changes priority and
  ownership.
- **Item 7 is undersold.** It is the single most diagnostic test
  the dycore is missing. The follow-up doc lists it as "Item 7" of 7;
  in a sensible priority ordering it would be in the top three,
  alongside Item 1 and Item 2a.
- **Missing item: barotropic-baroclinic mode-split coupling.** None
  of the 7 items address how the barotropic substep couples to the
  baroclinic slow step. Memory note `project_barotropic_solver` and
  `project_barotropic_noise_issue` confirm this is open, real, and
  Adcroft-relevant (BEBT comes from Shchepetkin & McWilliams 2005,
  Adcroft co-authored relevant follow-ups). Worth adding as Item 8
  before the next conversation.

## One refactor that would matter most

**Land the seamount rest-state test (Item 7) and make every PGF
scheme run on it as a CI artifact.** This is not what most people
would pick — the operator/metric refactor (Item 1) is bigger and
more architecturally satisfying. But Item 1 is months of work and
its payoff is *future flexibility*. The seamount test is one or two
weeks and its payoff is *immediate scientific confidence*. It is
also the lever that disambiguates the most other open questions:

- It tells you whether AHH08 actually wins (Item 2b) — and gives you
  the receipt to switch defaults.
- It is the right benchmark for "did the operator refactor preserve
  correctness" (Item 1).
- It connects directly to `project_mpas_etopo_instability` — the
  Adcroft framework is precisely the language for explaining whether
  the ETOPO bottom-trapped instability is an algorithm-class limit
  or a PGF-residual issue, and the seamount test is the discriminator.
- It generalises naturally to Williamson 5 (mountain in SW) for the
  `nlev=1` test matrix (Item 2a).
- It is the canonical example of a per-term test (Item 6).

Five of seven items get pulled forward by this one piece of work,
and it lands a benchmark number that can be cited in *every*
subsequent PGF / vertical-coordinate / mode-split conversation. The
operator/metric refactor (Item 1) follows naturally once the test is
in CI: it becomes the gate that says "yes, the refactor preserved
the dycore". Doing Item 1 *before* Item 7 inverts that ordering and
forces the refactor to be its own validator, which is exactly the
trap that broke `project_mpas_etopo_instability` for so long.
