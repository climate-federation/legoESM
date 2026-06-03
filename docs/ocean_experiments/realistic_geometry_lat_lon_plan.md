# Realistic coastlines + bathymetry on the lat-lon C-grid ocean — implementation plan

**Status (2026-05-02):** Phases 0, 1, 2, **3a**, 3.5 **complete**;
Phases **3b, 3c, 3d, 3e deferred** (see "Deferred scope" below);
Phase 4(a) smoke-test, Phase 4(b) 20-yr spinup, Phase 4(c) 50-yr
spinup with cos²(lat) + polar cap **complete**; **Phase 4(c) extension
to 100 yr complete** (yr 99.98, max|u|=1.21 m/s, equilibrated, AMOC
absent — lines up the Phase 4(d) forcing-protocol baseline); Phase 5
cross-comparison plotting **deferred** (will fold into Phase 4(d)
forcing-experiment matrix).  See
``realistic_geometry_phase4_results.md`` for full results and
``realistic_geometry_forcing_literature_review.md`` for the Phase 4(d)
forward plan derived from the 2026-05-02 OMIP/CORE literature review.

The momentum-field residual originally attributed to "partial-cell
q-noise + cold-start imbalance" was re-diagnosed in the 2026-05-01
evening session as a 2Δy zonal-jet computational mode (in *u*) driven
by the realistic-geometry config dropping A_h 20× from the idealised
default.  The realistic config can't simply restore A_h=2e5 because
that triggers a separate Arctic high-lat partial-cell instability
(single-cell runaway at lat 82.5° in 15 days).  Standard MITgcm/
MOM6/NEMO fix (cos²(lat) A_h scaling, lat>80° polar cap) landed.
AL81 audit identified six defects but a discrete energy + enstrophy
budget test confirmed they are stylistic / minor, not bugs.

## Deferred scope

The 2026-05-02 PR ships Phases 0–3a, 3.5, 4(a)–4(c).  The following
items from the original plan are explicitly **deferred** rather than
silently dropped:

- **Phase 3b — Tilted Gaussian ridge (45° rotated)**.  Low expected
  info-gain given Phase 4(c) ran 100 yr cleanly without grid-alignment
  artefacts in the bulk interior (only the 2Δy mode at the equator,
  fixed by cos²(lat)).  Revisit only if Phase 4(d) forcing changes
  re-expose alignment-sensitive structure.
- **Phase 3c — Munk gyre with 45° diagonal eastern boundary**.
  Medium info-gain — would catch any latent corner-cell bug under
  sustained flow.  Recommended before any future "real coastline +
  diagonal western boundary current" claim (e.g. Gulf Stream
  separation diagnostics).  Not blocking for the global-overturning
  experiment matrix.
- **Phase 3d — Circular island in closed basin**.  Low info-gain;
  Phase 4(c) ran with multiple real islands (Antarctica, Australia,
  Madagascar, etc.) without conservation drift.
- **Phase 3e — Two-basin with narrow strait + Whitehead 1989
  hydraulic-control comparison**.  Medium info-gain; would calibrate
  our expectation for Indonesian Throughflow + Drake exchange in
  Phase 4(d).  Recommended **before** publishing any throughflow
  number from the realistic-geometry runs.
- **Phase 5 — Cross-run comparison package** (Drake transport,
  AMOC pathway, η spectrum on real bathy, topographic Rossby).
  Folded into the Phase 4(d) experiment matrix in
  ``realistic_geometry_forcing_literature_review.md`` so it has
  multiple configs to compare, not a single-config baseline.

The 2026-05-02 lit review identified **forcing protocol** (SST
restoring → prescribed heat flux, add idealised E−P + β_S, Bryan-
Lewis κ_v) as the highest-leverage path to credible AMOC, not
boundary-handling.  The deferred 3b–3e tests retain their original
scientific value but are no longer on the critical path to the
"realistic Atlantic overturning at 5°" milestone.

The momentum-field residual originally attributed to "partial-cell
q-noise + cold-start imbalance" was re-diagnosed in the 2026-05-01
evening session as a 2Δy zonal-jet computational mode (in *u*, peaking
at the equator) driven by the realistic-geometry config dropping A_h
20× from the idealised default.  The realistic config can't simply
restore A_h=2e5 because that triggers a separate Arctic high-lat
partial-cell instability (single-cell runaway at lat 82.5° in 15
days).  Standard MITgcm/MOM6/NEMO fix (cos²(lat) A_h scaling) landed
this session.  AL81 audit identified six defects but D2 (WENO5 vs
AL81 = -15-20%) bounds their combined effect; a discrete energy +
enstrophy budget test will discriminate "real theoretical hole" from
"alternative stylistic choice" before any AL81 fix work proceeds.

### Phase 4(b) first-pass result (2026-05-01)

Run: ``results/ocean/global_overturning_realistic_geometry/`` —
ETOPO 5° (36×72), MEO r=0.2, A_h=1e4, K_GM=K_Redi=800,
drag=2.5e-3, B_h=5e9, SMC03 PGF, AL81 PV-flux, implicit-CN
barotropic.  50-yr targeted, killed at year 20.8 with sufficient
qualitative data.  Restarts saved at years 0, 5, 10, 15, 20.

- ✅ Stratification, SST pattern, SSH pattern, mass conservation,
  no NaN, bottom-T trend correct.
- ❌ No deep AMOC-like MOC cell forms; barotropic streamfunction
  dominated by coastal noise (no coherent gyres); surface speeds
  3–5× realistic with pathological striping in equatorial Pacific.

The full SMC03 + AL81 + h_vtx min-rule + MEO + closure stack is
**stable shippable infrastructure**.  Gap to production-quality
dynamics is a combination of (a) residual partial-cell q-noise in
the AL81 stencil (likely missing corner-triad ENE corrections),
and (b) absence of production spinup machinery (Levitus/WOA IC,
forcing ramp, viscosity ramp).  See results doc for the prioritised
next-session plan.

### Resumed (was: paused at Phase 3.5/4 pending partial cells)

The pause documented below was lifted on 2026-05-01 by the merge of
the partial-cells branch (``ocean-partial-cells``) and the
density-Jacobian PGF branch (``ocean-pgf-smc03``, S&M 2003) into this
PR.  Phase 3.5 (the headline "lat-lon C-grid runs the seamount
stress test cleanly on real ETOPO" gate that motivated the pause):

- **Beckmann-Haidvogel seamount, 30-day rest-state**:
  smoothing=5, r_max=0.54, drag=1e−3 →
  ``|u|max = 1.5 mm/s`` (target < 5 mm/s).  **PASS.**  Down from
  the legacy Adcroft-only path's 99 mm/s 2Δz computational mode.
  Rest-state PGF residual 60× smaller than Adcroft (1.1e−8 vs
  6.7e−7 m/s²).
- **Real ETOPO 30-day rest-state, 3°/20-level**:
  Adcroft baseline never reaches the < 50 mm/s criterion (390 mm/s
  on day 1) and crashes day 12.  SMC03 satisfies the < 50 mm/s
  criterion through day 12 (peak 49 mm/s); a non-PGF global-domain
  instability (likely coastal computational mode at irregular
  coastlines / thin-cell implicit-CN solver pathology) drives both
  schemes to NaN later (Adcroft day 12, SMC03 day 19).  The
  remaining 30-day gap is downstream of the PGF and out-of-scope
  for the SMC03 work.

See ``partial_cells_results.md`` for the full result table, the
iteration log (initial Option A → Option B → C1 ``min(z_c)`` fix),
and the on-branch research artefacts
(``pgf_production_models_research.md`` documenting MOM6/ROMS/NEMO
production PGF approaches, ``pgf_smc03_code_review.md`` documenting
the C1 bug and its fix).

**Phase 4-5 remaining work**: Wolfe-Cessi-style spinup on real
ETOPO and eddy-permitting experiments.  Both gated on closing the
30-day NaN-free instability noted above — likely horizontal
viscosity / biharmonic / Smagorinsky tuning, coastal sponge, or
MEO-style additional smoothing.  A diagnostic spike to localise
the instability (homogeneous T,S 1-day ETOPO run to separate
"stratification × coastline" from "pure dynamics × coastline") is
the recommended next step.

---

### Original pause note (2026-04-30, kept for history)

**Why paused**: Phase 3a's empirical regime boundary (model passes
seamount stress at r_max < 0.24) plus Phase 3a's MEO sweep on real
ETOPO showed that getting r_max below the model's stability bound at
2.5° resolution requires r_target ≤ 0.03 — corresponding to a +32%
ocean volume change that effectively eliminates continental shelves
and slopes.  The plan's "realistic geometry" promise is undermined
if we land Phase 4 with a near-flat-bottom bathymetry.

The chosen mitigation path (per the d-J PGF decision gate the plan
explicitly flagged) is **z\* + partial cells** — the modern MOM6 /
MITgcm production approach.  Partial cells eliminate the PGF
cancellation problem in the bulk of the water column by keeping
full-cell `z_k` constant horizontally; only the partial bottom cell
needs special-case PGF treatment.  This is the production-grade
solution that ~half of all CMIP-class ocean models use.

The realistic-geometry plan resumes at Phase 3.5 once
`partial_cells_plan.md` ships and lat-lon C-grid runs the seamount
stress test cleanly on real ETOPO.  Phases 0–3a artefacts (MEO
machinery, idealised-coastline tests, ETOPO ingest, regime-boundary
characterisation) all remain valuable infrastructure regardless of
the coordinate change.

## Motivation

Every ocean experiment in legoESM to date uses idealized geometry:

- **Closed rectangular channels** — Eady, Silvestri baroclinic jet
- **Zonally-periodic bands with a single Gaussian ridge** — ACC channel, Zhang 2024
- **"Global" ocean with flat 4000-m bottom and stylized coastlines** —
  the Wolfe-Cessi-inspired global overturning experiment that produced
  our 50-yr lat-lon and (in-flight) MPAS reference runs

None of this exercises the failure modes that dominate production
ocean modelling:

- Sill overflows (Denmark Strait, Faroe-Bank Channel) where dense water
  actually forms
- Real western boundary currents constrained by realistic coastlines
- Drake Passage as a geographic choke point with real bathymetry
- Indonesian Throughflow as a Pacific ↔ Indian connector
- Topographic Rossby waves over continental slopes
- AMOC return flow along a real North Atlantic basin
- Pressure gradient errors over steep topography (the classical
  z-coordinate ocean numerical hazard)
- Barotropic-mode ↔ steep-bathymetry interaction (which is precisely
  where chequerboard noise lives — see
  `docs/issues/barotropic_mode_noise.md`)

`docs/ocean_boundary_conditions_analysis.md` already enumerates failure
modes our boundary-handling stack might exhibit under realistic
geometry — corner cell inconsistencies at diagonal coastlines, metric
amplification near steep topography, ordering dependencies between
Neumann fill and conservation fixers — but **none of these have been
tested**.

The strategic logic: the lat-lon C-grid stack is our highest-confidence
configuration.  Stress-testing it under realistic geometry now (a) is
the cheapest path to a credible "this is a real ocean model" result,
and (b) hardens the boundary-handling foundation that any future
investment (tripole grid, sea ice, OMIP forcing, atmosphere coupling)
will sit on.  Building tripole or OMIP without first proving realistic
geometry on lat-lon means bringing up multiple unknowns simultaneously
and bundling several months of debugging into one branch.

## Scope

The plan delivers, in order:

1. A **bathymetry + coastline data ingest** that converts
   ETOPO/GEBCO global bathymetry and a standard land mask onto a
   lat-lon C-grid.
2. **Variable-bathymetry support** validated end-to-end:
   per-cell `H_bathy`, partial bottom cells, z* coordinate behaviour
   under variable depth, every vertical operator handles partial
   cells correctly.
3. **Realistic-coastline support** validated end-to-end:
   `replace_land_mask` flow correctness on diagonal/curved coastlines,
   conservation invariants, no spurious mass leak through coastline
   diagonals.
4. **Steep-bathymetry stress tests** that characterise pressure
   gradient errors and chequerboard amplification on realistic slopes,
   with mitigations (bathymetry smoothing, density-Jacobian PGF if
   needed) tested and quantified.
5. **A 50-yr global overturning run** on realistic geometry —
   Wolfe-Cessi-style idealized forcing, real bathymetry + real
   coastlines — comparable to the 50-yr flat-bottom lat-lon reference.
6. **Diagnostics**: Drake transport with real Drake Passage geometry,
   AMOC pathway with real Atlantic, topographic Rossby response, sill
   exchange where resolved.

The plan **does not** include:

- OMIP / CORE-II / JRA55-do realistic forcing.  Forcing stays
  idealized (two-belt wind + cosine SST restoring) so the only
  varying input from the 50-yr reference is geometry.  Realistic
  forcing is a separate, downstream plan.
- Tripole grid.  This is the prerequisite step before tripole; see
  `docs/ocean_experiments/tripole_grid_plan.md`.
- Sea ice.  Deferred until after the tripole port.
- Atmosphere coupling.  Out of scope.

## Strategic positioning

This plan is **Step 1** of a two-step "make legoESM a credible global
ocean model" arc:

| Step | Plan | Delivers |
|---|---|---|
| 1 | this plan | Realistic geometry on validated lat-lon stack; stress-tested boundary handling |
| 2 | `tripole_grid_plan.md` | Production-class structured grid; credible Arctic; OMIP-ready foundation |

Doing them in this order means tripole inherits a battle-tested
boundary-handling stack rather than bringing up two unknowns
simultaneously.

## Scientific contract

The realistic-geometry port must demonstrate:

1. **Rest state on realistic geometry.**  Initialise at rest with
   stratified T/S on real bathymetry + real coastlines, integrate
   1 sim-day with `barotropic_solver="implicit_cn"`.  Require
   `|η|max ≤ 1e-10 m`, `|u|max ≤ 1e-10 m/s`.  This is the single
   sharpest test for boundary-handling bugs under non-trivial
   geometry.
2. **Mass conservation.**  Total ocean volume invariant to round-off
   across a 1-yr integration on realistic geometry.
3. **Tracer conservation.**  Closed-domain heat and salt totals
   invariant to round-off under no-restoring no-wind forcing.
4. **Pressure gradient sanity.**  Spurious flow over the Mid-Atlantic
   Ridge from a rest state with realistic stratification stays below
   a documented bound (target: < 1 cm/s peak after 30 days, with the
   chosen smoothing).
5. **Drake transport.**  The 50-yr realistic-geometry global
   overturning run produces a Drake transport in the same sign and
   physical regime as published estimates (~120–170 Sv eastward for
   the real ACC, though our coarse resolution + idealised forcing
   means we expect a *physically credible* not *quantitatively
   correct* answer).
6. **AMOC pathway.**  A meridional-overturning streamfunction with
   the topology expected from a real Atlantic basin (sinking in the
   North Atlantic, return flow at depth, upwelling in the Southern
   Ocean), distinct from the artefactual two-cell pattern that flat-
   bottom + stylised-continents produces.
7. **No regression on flat-bottom path.**  All existing flat-bottom
   experiments (Eady, ACC channel, flat-bottom global overturning)
   remain bit-exact under the variable-bathymetry code path with
   `H_bathy = const`.

## API surface

Goal: callers of the existing experiment configs add one extra step
(load bathymetry) and the rest is unchanged.

```python
# New: src/legoesm/ocean/bathymetry.py
def load_bathymetry(
    source: Literal["etopo2022", "gebco2024", "etopo1"],
    grid: LatLonCGridGeometry,
    *,
    smoothing: BathymetrySmoothingConfig | None = None,
    min_depth_m: float = 50.0,        # cells shallower → land
    cache_dir: Path | None = None,
) -> tuple[H_bathy, land_mask]:
    """Conservatively interpolate global bathymetry onto the model grid.

    Returns
    -------
    H_bathy : (nlat, nlon) float — positive depths in metres.  Zero on land.
    land_mask : (nlat, nlon) bool — True for ocean, False for land.

    The smoothing config controls the trade-off between fidelity to
    raw bathymetry and pressure-gradient error over steep slopes.
    Default is a Shapiro-2 filter applied N times where N is chosen
    to keep dh/dx · g · drho / rho_0 below a target threshold.
    """
```

`LatLonCGridGeometry` already carries an `H_bathy` field; this plan
exercises the path that fills it from data instead of `H_max` constant.
`replace_land_mask` already exists and atomically updates u_mask,
v_mask; this plan exercises and stress-tests it.

```python
# New: src/legoesm/ocean/experiments/global_overturning_realistic.py
@dataclass
class GlobalOverturningRealisticConfig(GlobalOverturningConfig):
    """Wolfe-Cessi-style global overturning with realistic geometry.

    Inherits forcing parameters from GlobalOverturningConfig.  Adds
    bathymetry source + smoothing config.  Initial T/S stratification
    built on realistic H_bathy rather than flat bottom.
    """
    bathymetry_source: str = "etopo2022"
    bathymetry_smoothing: BathymetrySmoothingConfig = ...
    coastline_min_depth_m: float = 50.0
```

## Implementation phasing

### Phase 0 — Bathymetry and coastline data ingest  (≈ 4 days)

**Decisions before starting:**

- Bathymetry source: **ETOPO2022 (15 arc-sec)** is the recommended
  default — newer than ETOPO1, freely distributed by NOAA, and
  consistent with what most ocean modelling groups currently use.
  GEBCO 2024 is the alternative if a specific comparison study
  motivates it.
- Initial resolution: **72×144 (2.5°)**.  The current 36×72 (5°)
  cannot resolve Drake Passage (~6° wide) or Indonesian Throughflow
  meaningfully.  72×144 places ~3 cells across Drake.  144×288
  (1.25°) is the next step up but ~4× the cost — defer until Phase 4
  validates at 2.5°.
- Vertical levels: stay at **20 levels** for first validation; increase
  to 30–40 only if sill-depth resolution becomes a blocker in Phase 5.

**Tasks:**

- Write `src/legoesm/ocean/bathymetry.py`:
  - `_load_etopo2022(cache_dir)` — NOAA NetCDF reader with HTTPS
    download + local cache.
  - `_conservative_remap(raw_lat, raw_lon, raw_depth, grid)` —
    cell-area-weighted average of raw bathymetry onto the model grid.
    Handles wrap-around in longitude.
  - `load_bathymetry(...)` — top-level entry point.
  - `_diagnose_bathymetry(H_bathy, land_mask)` — sanity diagnostics:
    total ocean area, max/mean depth, histogram of slopes, list of
    suspiciously narrow straits.
- Write `src/legoesm/ocean/coastlines.py`:
  - `derive_land_mask_from_bathymetry(H_bathy, min_depth_m)` —
    conservative land mask.
  - `_apply_topology_fixes(land_mask)` — close known under-resolved
    straits if their resolution is below threshold; document each
    fix.
- Diagnostic: `scripts/diagnose_realistic_geometry.py` produces
  `results/realistic_geometry_check/{bathymetry,coastline,slopes}.png`
  for visual inspection.

**Decision gate**:
- Bathymetry loads, conservatively remapped onto 72×144 grid;
  total ocean area within 0.5% of the geographic value (~3.6e8 km²).
- Land mask shows recognisable continents (Eurasia, Americas, Africa,
  Antarctica) in the right places.
- Slope histogram identifies the 1% steepest cells (these are
  the candidates for Phase 3 stress tests).
- No solver code touched yet.

### Phase 1 — Variable-bathymetry validation  (≈ 1 week)

The codebase nominally supports variable `H_bathy` already.  This
phase **proves it works correctly** end-to-end.

**Tasks:**

- Audit the vertical coordinate construction (`create_ocean_z_star`)
  under variable `H_bathy`.  Specifically:
  - Partial bottom cells: layer thickness `dz_k(i, j)` for the
    deepest active level may be < `dz_deep`.
  - z* compression: under non-zero `eta`, the layer thickness scales
    by `(H + eta) / H` — verify this works correctly per cell.
- Audit every operator that touches the vertical:
  - `compute_ocean_rho_and_pressure` — hydrostatic integration over
    variable depth
  - `vertical_diffusion` — implicit tridiagonal solver under
    variable column depth
  - Tracer flux divergence — bottom flux must be zero on partial
    cells
  - PGF — depth-integrated form must be consistent with
    per-level form when bathymetry varies
- Add unit tests:
  - `test_partial_cell_rest_state.py`: rest state on a 2D step
    bathymetry profile (deep + shallow + step), 1 sim-day,
    machine-precision preservation.
  - `test_variable_bathymetry_conservation.py`: closed-domain
    tracer conservation on a smoothly-varying H_bathy.
  - `test_z_star_compression.py`: z* layer thickness scales
    correctly under non-zero η.

**Decision gate**:
- All three new unit tests pass at machine precision.
- Existing flat-bottom test matrix runs bit-exact.

### Phase 2 — Realistic coastline validation  (≈ 4 days)

The land mask + face mask machinery exists (`replace_land_mask`,
`u_mask`, `v_mask`) but is exercised only on simple convex domains.
This phase stress-tests it on diagonal and curved coastlines.

**Tasks:**

- Add unit tests:
  - `test_diagonal_coastline_rest_state.py`: rest state on a domain
    with a 45° diagonal coastline; require machine-precision
    preservation.  Catches corner-cell flux leak.
  - `test_island_topology.py`: rest state on a domain with an
    isolated island; tracer conservation over closed domain.
  - `test_realistic_landmask_consistency.py`: load 72×144 realistic
    coastline, assert `u_mask = land_mask AND land_mask_east_neighbor`
    and similar for v_mask, T_mask everywhere.
- Add a runtime check (gated on `enable_runtime_checks`) that
  asserts: `total_mass_flux_through_land == 0` to round-off after
  every `step()`.
- Document any topology fix applied (e.g. "Bering Strait closed
  because <1 cell at 2.5° resolution") in
  `docs/ocean_experiments/realistic_geometry_topology_fixes.md`.

**Decision gate**:
- Diagonal coastline + island tests pass at machine precision.
- Realistic-mask consistency assertion holds globally.
- Topology-fix doc lists every modification to the raw mask.

### Phase 3 — Idealized geometry ladder + steep-bathymetry stress  (≈ 12 days)

The fundamental design decision: rather than confront all
boundary-handling failure modes simultaneously in a single realistic-
geometry run, **isolate each failure mode with a small dedicated
test**.  This adds ~5 days to Phase 3 over the original "single PGF
stress test" scope, but buys clean root-cause attribution if Phase 4
misbehaves.  Without this, a single ETOPO-run blowup is genuinely
hard to debug (multiple unknowns coupled).

The five tests cover five distinct failure modes; together they
exhaust the boundary-handling concerns we expect realistic geometry
to expose.

| Mode | Test |
|---|---|
| PGF over steep slope | 3a Beckmann-Haidvogel seamount |
| Grid alignment / numerical anisotropy | 3b Tilted Gaussian ridge |
| Corner cells under sustained flow | 3c Munk gyre, diagonal coastline |
| Closed-loop topology (islands) | 3d Circular island in closed basin |
| Sill / strait exchange + connected basins | 3e Two-basin with strait |

Each test gets a short result write-up.  The collection lands at
`docs/ocean_experiments/idealized_geometry_validation.md` (built
during this phase, not now).

#### 3a — Beckmann-Haidvogel seamount  (≈ 3 days)

The canonical PGF stress test for z-coordinate ocean models
(Beckmann & Haidvogel 1993).  Stratified rest state in a periodic
box with a single Gaussian seamount; zero forcing; integrate 30
days.  Spurious velocity peak is the diagnostic — every z-coord
ocean paper reports this number, so we have published bounds to
compare against.

- Setup: 100×50 km box, 30 levels, central Gaussian seamount with
  height ~50% of total depth; uniform N²; rest state.
- Mitigation hierarchy (Shapiro smoothing first, density-Jacobian
  PGF only if smoothing fails — same as the original Phase 3 design).
- Mode: **screening**.  Pass criterion: peak `|u|` after 30 days
  ≤ 5 mm/s (consistent with well-formed z-coord schemes).

#### 3b — Tilted Gaussian ridge  (≈ 2 days)

Take the existing ACC-channel single-ridge experiment and rotate
the ridge 45° relative to the grid.  Re-run.  Compare flow
structure to the grid-aligned baseline.

- Setup: same as the existing ACC channel test (no new code beyond
  the ridge orientation).
- Mode: **screening**.  Pass criterion: zonal-mean transport and
  meridional structure within 5% of grid-aligned baseline.

#### 3c — Munk gyre, diagonal eastern boundary  (≈ 2 days)

Wind-driven double gyre in a closed rectangular basin whose
eastern wall runs at 45°.  This is the only test that exercises
corner cells *under sustained flow* — bug modes here are
invisible to rest-state tests.

- Setup: 4000×2000 km basin, 20 levels, two-belt wind stress
  (Munk 1950 standard), Wright EOS, no GM/Redi.
- Mode: **screening + qualitative comparison**.  Pass criterion:
  western boundary current intensification structure
  qualitatively matches the rectangular-basin Munk solution
  (no anomalous flow at the diagonal corner).

#### 3d — Circular island in closed rectangular basin  (≈ 1 day)

Catches non-simply-connected ocean topology bugs that 3c can't
reach (e.g. halo-wrap failures around an island, circulation-
conservation around a closed contour).

- Setup: closed rectangular basin with a single circular island in
  the centre, wind-driven.
- Mode: **screening**.  Pass criterion: closed-domain mass + tracer
  conservation to round-off; circulation around the island contour
  consistent with Kelvin's theorem under the applied forcing.

#### 3e — Two-basin with narrow strait  (≈ 4 days, full diagnostic)

The most scientifically substantive of the five tests.  Sill / strait
exchange is the dominant bathymetry-coastline coupling we'll see in
realistic geometry; characterising it cleanly here pays off when we
interpret the realistic-geometry Phase 4 run.

- Setup: two closed rectangular basins connected by a narrow
  channel (~3 cells wide); apply different surface T* in each
  basin (e.g. cold north, warm south); spin up.
- Diagnostic: exchange-flow magnitude, baroclinic structure across
  the strait, mass balance closure.  Compare to two-layer
  hydraulic-control theory (Whitehead 1989) where applicable.
- Mode: **full scientific diagnostic** — this one earns the
  investment because the dynamics is novel for our stack and we'll
  encounter it again in the realistic Atlantic basin.

#### Phase 3 decision gate

- All five tests pass their respective pass criteria.
- 3a's peak spurious velocity, with chosen Shapiro smoothing, is
  documented and ≤ 5 mm/s.
- 3e's exchange-flow diagnostic is consistent with two-layer
  hydraulic-control prediction within a documented tolerance.
- If any test fails: root-cause investigation, fix, re-run that
  single test.  **Phase 4 does not start until all five pass.**
- If 3a forces density-Jacobian PGF (smoothing insufficient at
  realistic-bathymetry slopes), that decision goes through a
  separate review — do NOT silently bake it into the plan.

### Phase 3.5 — Realistic-geometry smoke test  (≈ 3–4 days)

The Phase 3 ladder catches each failure mode in isolation, but does
not test whether they appear simultaneously when full ETOPO bathymetry
+ real coastlines are combined.  This phase bridges the gap before
the multi-hour Phase 4 commitment.

**Tasks:**

- 30-day rest-state run on the real ETOPO bathymetry + real coastlines
  product from Phase 0 (with the Shapiro smoothing chosen in Phase 3a).
  Stratified T initial condition, zero forcing.
  - Diagnostic: peak `|u|` over time, regional max-velocity maps focused
    on the Mid-Atlantic Ridge, East Pacific Rise, Drake Passage shelf
    break, continental shelves around Antarctica.
- 90-day forced run with the Wolfe-Cessi-style two-belt wind + cosine
  SST restoring on the same realistic geometry.
  - Diagnostic: `|η|`, `|u|`, T-range bounded; mass + heat conservation
    to round-off; targeted regional checks at known PGF-hard regions
    (each gets a documented bound).
- A short driver script analogous to the Phase 4 driver but capped at
  90 days, output to
  `results/ocean/realistic_geometry_smoke_test/`.

**Decision gate:**

- 30-day rest-state run produces `|u|max ≤ 5 mm/s` (same threshold as
  Phase 3a Beckmann-Haidvogel).  Regional bounds met everywhere
  identified in Phase 0's slope-histogram diagnostic.
- 90-day forced run completes cleanly with documented diagnostics, OR
  identifies a specific regional issue that gets fixed before Phase 4
  starts.
- If the forced run shows local instability or persistent unphysical
  flow at a known PGF-hard region, the fix is to revisit Phase 0's
  smoothing parameters or apply a regional `H_bathy` floor (documented),
  not to silently increase global smoothing.

This phase exists specifically to **avoid wasting a multi-hour Phase 4
run** on a regression that the Phase 3 idealized ladder happened not
to expose.  Cost: ~3–4 days for ~6 h of additional model runtime
(30+90 days at dt=600s + diagnostics).

### Phase 4 — 50-yr global overturning on realistic geometry  (≈ 1 week + wall time)

**Status (2026-05-01):** First-pass complete — see
``realistic_geometry_phase4_results.md``.  Run reached year 20.8
without NaN; thermodynamics correct, momentum noisy.  Next-session
priorities (in order of cost/info-gain):

1. Audit ``pv_flux_al81_partial_cell`` for missing corner-triad ENE
   corrections (Sadourny-Salmon energy-enstrophy form).  Diagnose
   via discrete energy + enstrophy budgets vs flat-bottom reference.
2. Forcing ramp: τ_wind=0 + long SST τ_T for first sim-year, ramp
   over 6 months to production values.
3. Equilibrated initial T/S from Levitus/WOA climatology rather
   than analytic exp(z).
4. Spinup viscosity ramp: A_h=5e5 for first year → 1e4 after gyres
   set up.
5. Re-run 50-100 yr with above.

If after (1)-(5) partial-cell noise still dominates, fall back to
PLM-in-(T, S) PGF upgrade flagged in
``pgf_production_models_research.md``.

**Gated on Phases 3 and 3.5**: do not start Phase 4 until 3a–3e all
pass and 3.5's smoke tests complete cleanly.  This is a multi-hour
wall-clock investment; we want every distinct boundary-handling failure
mode characterised and resolved beforehand, *and* the realistic
combined geometry verified to integrate stably for at least a
full season.

Mirror the lat-lon flat-bottom 50-yr workflow but with realistic
geometry.

**Tasks:**

- Write `scripts/global_overturning/run_global_overturning_realistic_implicit_spinup.py`
  (10 yr) and `..._50yr_implicit_continuation.py` (40 yr).  Same
  driver structure as the flat-bottom counterparts; only the geometry
  + initial T/S construction differs.
- Initial conditions: cosine-latitude SST profile applied as
  surface T*; subsurface T computed by the same exponential decay,
  but capped at the local bathymetry.
- Forcing: same idealized two-belt wind + cosine SST restoring as
  the flat-bottom reference (so geometry is the only difference).
- Same 5-yr restart cadence, same dt (600 s with implicit_cn).
- Output: `results/ocean/global_overturning_realistic_50yr_implicit/`.

**Decision gate**:
- Run completes without blowup.
- `|η|max`, `|u|max`, T-range stay in physically credible bounds
  throughout (similar to flat-bottom run, modulo expected geometric
  modifications).
- Mass + heat conservation diagnostics close to round-off.

### Phase 5 — Diagnostics and comparison  (≈ 1 week)

**Tasks:**

- Drake transport time series + 50-yr mean.  Compare:
  - Flat-bottom 50-yr lat-lon reference (`global_overturning_50yr_implicit`)
  - Realistic-geometry 50-yr (this run)
  - MPAS 50-yr (today's running run, when complete)
- AMOC streamfunction:
  - Realistic geometry should show clear sinking in North Atlantic,
    return at depth, upwelling in Southern Ocean.
  - Flat-bottom shows a more symmetric two-cell pattern.
- Topographic Rossby response: spectral analysis of mid-depth flow
  variability over Mid-Atlantic Ridge.
- Sill diagnostics where resolved: Denmark Strait throughflow if
  resolved at 2.5° (likely marginally; re-check at 1.25° later).
- Spectrum of `eta`: characterise the chequerboard component on
  realistic geometry.  Does the implicit-CN solver still suppress it?
  Or does interaction with topography re-introduce noise?
  (`docs/issues/barotropic_mode_noise.md` Crit 2 with realistic
  bathymetry — new data point.)
- All comparison plots saved to
  `results/ocean/realistic_vs_flat_comparison/`.

**Decision gate**:
- A summary diagnostic page (analogous to the global overturning
  Phase 1.5 diagnostic suite) that demonstrates which features of
  the circulation are robustly captured and which are
  resolution/forcing limited.
- Documentation of any failure modes surfaced for the tripole plan
  to inherit.

## Validation hierarchy

In ascending order of difficulty:

| Test | What it catches | Phase |
|---|---|---|
| Bathymetry data load | NetCDF reader bug | 0 |
| Conservative remap area sum | Remapping bug | 0 |
| Land mask continent placement | Coastline derivation bug | 0 |
| Partial bottom cell rest state | z* / partial cell handling | 1 |
| Diagonal coastline rest state | Corner cell flux leak | 2 |
| Island topology conservation | Closed-domain consistency | 2 |
| Realistic-mask consistency assertion | u/v/T mask alignment | 2 |
| 3a Beckmann-Haidvogel seamount | PGF over steep slope | 3 |
| 3b Tilted Gaussian ridge | Grid-alignment / numerical anisotropy | 3 |
| 3c Munk gyre, diagonal eastern boundary | Corner cells under sustained flow | 3 |
| 3d Circular island in closed basin | Closed-loop topology / island handling | 3 |
| 3e Two-basin with strait | Sill exchange + connected-basin dynamics | 3 |
| 3.5 — 30d real-geometry rest state | Combined-geometry steady stability | 3.5 |
| 3.5 — 90d real-geometry forced run | Combined-geometry forced stability | 3.5 |
| 50-yr realistic geometry run | End-to-end production readiness | 4 |
| Drake transport vs published estimates | Big-picture circulation correctness | 5 |
| AMOC pathway topology | Realistic basin connectivity | 5 |

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| PGF errors on real bathymetry require density-Jacobian PGF — non-trivial port | High | Phase 3 explicitly tests smoothing first; only invest in d-J PGF if smoothing fails.  Decision gate forces the conversation. |
| Under-resolved straits (Bering, Indonesian) at 2.5° produce non-physical basin disconnection | Medium | Phase 2 documents every topology fix; defer to higher resolution (1.25°) for science questions where these matter |
| Conservation fixers behave differently on realistic vs rectangular domains | Medium | Phase 1 + 2 add explicit conservation tests on realistic geometry |
| Implicit-CN solver convergence degrades on realistic bathymetry (Helmholtz operator condition number changes) | Medium | Phase 5 monitors PCG iteration counts; tune `pcg_tol` / `pcg_maxiter` as needed |
| Chequerboard noise re-emerges on realistic geometry (Crit 2 was validated only on flat bottom) | Medium | Phase 5 spectrum diagnostic catches this; barotropic-mode lateral viscosity follow-up (already on the books) is the fix |
| Stratification + bathymetry interaction produces unphysical bottom water | Medium | Phase 4 diagnostics include bottom-cell T/S; if bottom water gets too cold/salty, revisit convection scheme on realistic geometry |
| ETOPO/GEBCO data files large + downloading at runtime is fragile | Low | Cache to user-configurable directory; document data source URL; provide pre-cached small-resolution version for CI |
| Existing experiments break under variable bathymetry path | Low | Phase 1 enforces bit-exact regression with `H_bathy = const`; flat-bottom path stays untouched |

## Reuse map

| Component | Reuse | Notes |
|---|---|---|
| EOS (Wright, linear) | 100% | column-local |
| KPP, Richardson, constant vertical mixing | 100% | column-local |
| Convection (enhanced_diffusion) | 100% | column-local |
| Surface forcing (combined, restoring, prescribed) | 100% | per-cell |
| Time integration (segment runner, scan blocks) | 100% | grid-agnostic |
| Restart I/O | 100% | grid-agnostic |
| Channel packing for ML | 100% | shape-preserving |
| Conservation diagnostics | 100% | already metric-aware |
| Visbeck adaptive κ | 100% | column-local |
| GM/Redi centered + triads | ~95% | uses bathymetry-aware bottom boundary |
| All advection schemes | ~95% | already mask-aware; need partial-cell verification |
| `latlon_cgrid_operators.py` | ~95% | partial-cell handling at bottom levels |
| Implicit-CN barotropic solver | 100% | Helmholtz operator already takes `H_bathy` |
| `LatLonCGridOceanConfig` | 100% | no schema change |
| `LatLonCGridOceanState` | 100% | no schema change |
| `replace_land_mask` helper | 100% | exists; this plan stress-tests it |
| `GlobalOverturningConfig` + forcings | ~95% | extended to take bathymetry source |

What is genuinely new:

| Component | LOC estimate | Notes |
|---|---|---|
| `src/legoesm/ocean/bathymetry.py` | ~400 | NetCDF reader + conservative remap + smoothing |
| `src/legoesm/ocean/coastlines.py` | ~150 | mask derivation + topology fixes |
| Realistic-geometry experiment config + driver | ~250 | `global_overturning_realistic.py` + spinup/continuation scripts |
| New unit tests (partial cell, diagonal coast, island, etc.) | ~600 | comprehensive boundary tests |
| Phase 3 idealized-geometry test drivers (3a–3e) | ~700 | seamount, tilted ridge, Munk diagonal, island, two-basin |
| `docs/ocean_experiments/idealized_geometry_validation.md` | ~prose | Phase 3 consolidated results |
| Diagnostic comparison scripts | ~300 | Phase 5 plotting |
| Topology-fix documentation | ~50 LOC + prose | per-fix justification |

**Total new LOC: ~1750.**  Comparable in magnitude to the tripole plan
(~1500 LOC), but front-loaded with data-wrangling rather than
operator refactoring.

## Calendar estimate

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Bathymetry + coastline data ingest | 4 days | 4 d |
| 1 — Variable-bathymetry validation | 1 week | 9 d |
| 2 — Realistic coastline validation | 4 days | 13 d |
| 3 — Idealized geometry ladder + steep-bathymetry stress | 12 days | 25 d |
|   ↳ 3a Beckmann-Haidvogel seamount | 3 d | |
|   ↳ 3b Tilted Gaussian ridge | 2 d | |
|   ↳ 3c Munk gyre, diagonal coastline | 2 d | |
|   ↳ 3d Circular island | 1 d | |
|   ↳ 3e Two-basin with strait (full diagnostic) | 4 d | |
| 3.5 — Realistic-geometry smoke test | 3–4 days | 29 d |
| 4 — 50-yr realistic-geometry run (gated on 3 + 3.5) | 1 week + ~6 h wall | 34 d |
| 5 — Diagnostics + comparison | 1 week | 39 d |

**Total: ~5.5 weeks** of focused work plus the 50-yr wall-clock time
(now ~6 h on the implicit-CN path at dt=600 s, may need adjusting
for variable bathymetry CFL).

If Phase 3 reveals smoothing is insufficient and the
density-Jacobian PGF port becomes necessary, add ~2 weeks.

## Reference material

- **Beckmann & Haidvogel (1993)** — "Numerical simulation of flow
  around a tall isolated seamount." *J. Phys. Oceanogr.* 23,
  1736–1753.  The canonical PGF stress test (Phase 3a).
- **Munk (1950)** — "On the wind-driven ocean circulation."
  *J. Meteor.* 7, 80–93.  Reference solution for Phase 3c gyre.
- **Whitehead (1989)** — "Internal hydraulic control in
  rotating fluids — applications to oceans." *Geophys. Astrophys.
  Fluid Dyn.* 48, 169–192.  Reference for Phase 3e exchange-flow
  diagnostic.
- **Sandwell et al. (2014)** — ETOPO1 global bathymetry; one
  precursor of current ETOPO products.
- **NCEI (2022)** — ETOPO 2022, NOAA's current global topography +
  bathymetry product (15 arc-sec).
- **GEBCO Bathymetric Compilation Group (2024)** — alternative
  global bathymetry, also 15 arc-sec.
- **Shchepetkin and McWilliams (2003)** — density-Jacobian PGF; the
  reference if Phase 3 smoothing is insufficient.
- **Adcroft, Hallberg, Harrison (2008)** — partial bottom cells in
  z-coordinate models.
- **Wolfe & Cessi (2010, 2011, 2014)** — the idealized global
  overturning configuration we already match in flat-bottom form.
- `docs/ocean_boundary_conditions_analysis.md` — already enumerates
  the failure modes this plan is designed to surface.
- `docs/issues/barotropic_mode_noise.md` — Crit 2 chequerboard
  validation; this plan re-runs that diagnostic on realistic
  bathymetry.

## Open questions for decision before Phase 0

1. **Resolution: 72×144 (2.5°) for first pass, or jump straight to
   144×288 (1.25°)?**  Recommendation: 72×144 for Phases 0–4; bump
   to 144×288 only for the production Phase 4 / 5 run if 2.5°
   results are clearly resolution-limited.
2. **Bathymetry source: ETOPO 2022 vs GEBCO 2024?**  Recommendation:
   ETOPO 2022 by default (NOAA-distributed, well-documented, current
   most-used product).  GEBCO 2024 as an option for sensitivity
   studies later.
3. **Topology fixes for under-resolved straits**: close them
   (default), or leave them open with extreme bathymetry?
   Recommendation: close them at 2.5° (Bering Strait, Indonesian
   Throughflow are under-resolved) and document; revisit at 1.25°
   where they may resolve.
4. **Vertical levels: 20 (current) or upgrade?**  Recommendation:
   stay at 20 for first pass; upgrade only if Phase 5 diagnostics
   show clear sill-depth resolution issues.
5. **PGF mitigation: budget for Shapiro smoothing alone, or
   pre-commit to density-Jacobian if needed?**  Recommendation:
   budget Shapiro only; treat density-Jacobian as a separate plan
   if Phase 3 forces it.
