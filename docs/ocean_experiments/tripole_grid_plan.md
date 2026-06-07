# Tripole grid for the lat-lon C-grid ocean — implementation plan

**Status (2026-04-29):** Scoping only.  No implementation in tree.

## Motivation

The lat-lon C-grid ocean is the highest-confidence configuration in
legoESM (50-yr global-overturning + GM/Redi triads run validated on
2026-04-28).  Its single fundamental limitation as a global ocean grid
is the **convergent-meridian polar problem**:

- At the north pole, `Δx = R·cos(φ)·Δλ → 0`, forcing tiny `Δt` for
  internal-gravity-wave and tracer-advection CFL even with the implicit
  barotropic solver removing the gravity-wave constraint.
- The pole point is a geometric singularity needing special-case
  handling and ad-hoc polar filtering — awkward in the presence of
  coastlines and bathymetry.
- The current 36×72 (5°) grid hides this by having a coarse
  representation of the Arctic where the singularity is benign.  At any
  resolution where the Arctic is dynamically relevant (≤1°), the polar
  problem dominates time-step and stability.

The **tripole grid** (Murray 1996; Madec & Imbard 1996) is the standard
production-ocean solution to this problem.  Every CMIP-class structured
ocean model uses some version of it (MOM6 OM4, NEMO ORCA1/ORCA025,
POP/CESM gx1v6, MITgcm gx1v).  It moves the northern coordinate
singularities off the ocean and onto land — typically two displaced
poles over Canada/Greenland and Russia/Siberia — while preserving the
2D structured-array layout that makes the existing lat-lon C-grid code
fast, JIT-clean, and ML/AD-friendly.

The choice between **tripole on lat-lon** and **C-grid cubed-sphere**
as the next ocean-grid investment is laid out in `docs/research/`
(decision summarised in MEMORY).  Tripole wins on effort
(~10× less), reuse (~80% of the lat-lon C-grid stack), risk (the math
is solved; grid files are public), and scientific payoff (production-
grade global ocean compatible with MPAS for cross-grid validation).

## Scope

The plan delivers, in order:

1. A `tripole_grid` module that loads ESMF/NetCDF tripolar grid files
   (ORCA1 or MOM6 OM4) into the existing `LatLonCGridOcean*` data
   structures, with per-cell metric arrays.
2. A metric-array refactor of `latlon_cgrid_operators.py` that is
   **bit-exact equivalent** to the current implementation when fed
   regular lat-lon metrics, and correct for arbitrary orthogonal
   curvilinear grids.
3. A **fold halo exchange** wrapper handling the bipolar-cap seam,
   for both scalars and vectors.
4. **Vector rotation** in halo exchange across cells whose local
   `(i, j)` axes are not geographically `(east, north)`.
5. Re-validation of the existing lat-lon test matrix on a tripole
   grid: rest-state preservation, conservation, GM/Redi triads,
   global overturning, Eady BCI.
6. A **production tripole run** (ORCA1) of the global overturning
   experiment, comparable to the lat-lon-square 50-yr reference.

The plan **does not** include:

- Sea ice on tripole (deferred — sea ice is the natural follow-on;
  separate plan).
- Atmospheric coupling on tripole (deferred — atmosphere stays on
  cubed-sphere; coupler already does cross-grid regridding).
- Variable-resolution tripole (e.g. tropical refinement).  ORCA-style
  uniform tripole only.

## Scientific contract

The tripole port must reproduce, on equivalent test cases, the
behaviours the lat-lon implementation already validates:

1. **Rest state** — initialised at rest with stratified T/S, integrate
   1 sim-day with `barotropic_solver="implicit_cn"`.  `|η|`, `|u|` must
   stay at machine precision (≤1e-12 m, ≤1e-12 m/s).  This is the
   single sharpest test for halo-exchange bugs, fold errors, and
   metric mistakes.
2. **Mass conservation** — total ocean volume invariant to round-off
   across a 1-yr integration.
3. **Tracer conservation** — total heat / salt invariant under closed
   surface forcing (no restoring, no wind) to round-off.
4. **GM/Redi triads** — Eady BCI run on tripole reproduces lat-lon
   `eady_gm_redi_triads` results within numerical tolerance south of
   the bipolar cap; Arctic Eady test (new — runs the cap region in
   isolation) shows triad cancellation works through the fold.
5. **Geostrophic adjustment** — zonal jet relaxes to geostrophy with
   the correct adjustment timescale; no spurious noise at the
   65°N transition or the bipolar fold.
6. **Williamson 2 analogue** — ocean rest state with bottom topography
   (height-anomaly geostrophic balance) preserves balance to round-off,
   including in the bipolar cap.
7. **Global overturning equivalence** — re-run of the
   `run_global_overturning_implicit_spinup` driver on ORCA1 reproduces
   the lat-lon-square 50-yr Drake transport, MOC, and zonal-mean T to
   within agreed tolerance south of 65°N (i.e. wherever the grids
   coincide), and produces a *credibly resolved* Arctic overturning
   rather than the degenerate-pole behaviour of regular lat-lon.

## API surface

The intent is that **no calling code outside `tripole_grid.py` and the
operator/halo modules needs to know whether the grid is regular
lat-lon or tripole**.  All grid-specific information is carried in
metric arrays attached to the existing `LatLonCGridOcean*` structures.

```python
# New: src/legoesm/grids/tripole.py
def create_tripole_grid(
    grid_file: str | Path,        # ESMF NetCDF path (ORCA1, OM4, etc.)
    *,
    cap_lat: float | None = None, # auto-detected from file metadata
) -> LatLonCGridGeometry:
    """Load a tripole grid file into a LatLonCGridGeometry with
    per-cell dx_T, dy_T, dx_u, dy_u, dx_v, dy_v, area_T, area_u, area_v,
    plus a fold descriptor and vector-rotation angles in the bipolar cap.

    Compatible drop-in for create_latlon_grid() — returns the same
    NamedTuple type with extra fields populated.  Regular lat-lon
    grids return zeros for rotation angles and a no-op fold descriptor.
    """
```

The existing `create_latlon_grid()` stays as the regular-lat-lon
constructor; it is updated to populate the new metric-array fields
analytically (from `cos(φ)·Δλ` etc.) so that the metric-aware
operators give bit-exact identical results to today's analytic path.

## Implementation phasing

### Phase 0 — Grid file selection and reader  (≈ 3 days)

- Pick the canonical tripole grid file.  Recommended: **ORCA1**
  (NEMO 1°, 362×292) — most widely used in CMIP, freely distributed
  by ECMWF, has clean ESMF NetCDF.  Fallback: **OM4 0.5°**
  (576×576 ish) from GFDL.
- Write `src/legoesm/grids/tripole.py`:
  - `_read_esmf_tripole(path)` — NetCDF reader, returns lat/lon at
    cell centers and corners, areas, fold descriptor.
  - `create_tripole_grid(path, ...)` — converts to
    `LatLonCGridGeometry` with metric arrays and fold/rotation info.
  - `_visualize_tripole(grid)` — diagnostic plot of cell areas and
    the bipolar cap; saved to `results/tripole_grid_check.png`.
- Add fields to `LatLonCGridGeometry`:
  - `dx_T, dy_T` per cell  `(nlat, nlon)`
  - `dx_u, dy_u` at u-points `(nlat, nlon)`
  - `dx_v, dy_v` at v-points `(nlat, nlon)`
  - `area_T, area_u, area_v` per cell `(nlat, nlon)`
  - `cos_alpha, sin_alpha` per u/v point — rotation from local i-axis
    to geographic east (zero outside cap)
  - `fold_descriptor` — small NamedTuple with cap row index, fold
    permutation, vector sign flags
- Update `create_latlon_grid()` to populate the new fields analytically
  (so regular-lat-lon path is metric-array but bit-exact equivalent).

**Decision gate**: ORCA1 grid loads cleanly; areas sum to 4πR² to
≤1e-6 relative; cell-area histogram visualised; fold seam identified
and matches grid-file metadata.  No solver code touched yet.

### Phase 1 — Metric-aware operator refactor  (≈ 1 week)

- Refactor `latlon_cgrid_operators.py` so every operator takes
  per-cell metric arrays from `LatLonCGridGeometry` rather than
  computing `cos(φ)` internally.  This is mostly mechanical — the
  current code already carries `cos(φ)` factors symbolically.
- Touched operators:
  - `gradient_T_to_u`, `gradient_T_to_v`
  - `divergence_uv_to_T`
  - `curl_uv_to_corner` (vorticity)
  - `laplacian_T`, `laplacian_u`, `laplacian_v`, `biharmonic_*`
  - PGF (depth-integrated and per-level)
  - Coriolis (4-point average)
- Validation:
  - Existing `tests/ocean/` test suite must pass **bit-exact**
    against current main on regular-lat-lon path.
  - New unit test `test_metric_array_equivalence.py`: run each
    operator on an analytic test field with the regular-lat-lon
    metric arrays, compare to today's analytic-path output —
    require equality to 1 ULP.

**Decision gate**: full lat-lon ocean test matrix bit-exact under
metric-array path.  No tripole code yet.

### Phase 2 — Fold halo exchange  (≈ 1 week)

This is the conceptually new piece.  The bipolar cap row j=jmax is a
seam where cell (i, jmax) is the same physical location as cell
(N_i − i + 1, jmax).  Halo exchange across the fold has to:

- For scalars: copy data with i-axis reversed.
- For vectors: copy with i-axis reversed AND sign flip on both
  components (the local east/north on the two halves of the seam are
  antiparallel).

Implementation:

- `src/legoesm/ocean/halo/tripole_fold.py`:
  - `apply_fold_scalar(field_4d, fold_descriptor)` — scalar variant.
  - `apply_fold_vector(u_field, v_field, fold_descriptor)` — vector
    variant with rotation-aware sign flip.
  - `pad_halo_tripole_4d(field, geometry)` — wrapper around the
    existing `pad_halo_4d` that also applies the fold for the north
    halo when `geometry.fold_descriptor.is_active`.
- Replace direct calls to `pad_halo_4d` with `pad_halo_tripole_4d` in
  the lat-lon C-grid operators (no-op on regular lat-lon).
- Same for `pad_halo_vector_4d`.

Validation:

- `test_fold_round_trip.py`:
  - Scalar field `T(i, j)` initialised with a known pattern; pad halo,
    extract north halo, verify it equals `T[::-1, jmax-k]`.
  - Vector field `(u, v)` initialised with a uniform geographic-east
    flow; pad halo, verify across the fold the components have the
    correct sign convention (i.e. the rotated representation of "east"
    on the other side of the fold is recovered).
- `test_fold_conservation.py`: random initial T/S on tripole, advect
  zero-velocity for one step (no flux), verify total tracer
  invariant to round-off (in particular at the fold).

**Decision gate**: fold halo passes round-trip and conservation
tests; no other operator code touched yet.

### Phase 3 — Vector rotation in cap  (≈ 4 days)

Inside the bipolar cap, the local `(i, j)` axes are not aligned with
geographic `(east, north)`.  Per-edge rotation angles
`(cos α, sin α)` are computed once at grid-file load and stored in
geometry.

- For halo exchanges crossing into a cell whose neighbour has a
  different rotation (i.e. anywhere inside the cap or at the
  cap boundary), rotate vector components into the receiving cell's
  local frame before storing in the halo.
- Outside the cap, all rotation angles are zero — the path becomes
  a no-op.
- Affects: `pad_halo_vector_4d` (now needs geometry, not just shape).

Validation:

- `test_uniform_geographic_flow.py`: initialise with constant
  geographic-east `u_geo` everywhere.  In the cap, this corresponds
  to spatially-varying `(u_local, v_local) = (cos α · u_geo,
  −sin α · u_geo)`.  After one halo exchange, `(u_local, v_local)`
  must remain the correct spatial representation of `u_geo` —
  divergence of the flow must be zero to round-off.
- `test_rotation_consistency.py`: rotation matrices on adjacent cells
  must compose to a single rotation when chained across the cap.

**Decision gate**: uniform geographic-east flow has zero divergence
across the entire grid (cap included).

### Phase 4 — Ocean model integration  (≈ 1 week)

- Wire metric-aware operators + tripole halo into
  `LatLonCGridOceanModel`.  This is mostly mechanical: no new logic,
  just ensuring every halo-exchange call site uses the geometry-
  aware variant and every operator takes metric arrays.
- Touched files:
  - `ocean_model_latlon_cgrid.py`
  - `ocean_pe_latlon_cgrid.py`
  - `barotropic_latlon_cgrid.py`
  - `barotropic_implicit_latlon_cgrid.py` — including the PCG
    Helmholtz inner loop (which does halo exchanges per iteration —
    cost matters; verify no recompilation per iter)
  - GM/Redi (`gm_redi_latlon_cgrid.py`) — slope computation crosses
    halos at the cap boundary
  - All advection schemes (upwind, TVD, SOM, DST-3, PPM-FCT, WENO)
- Add `enable_runtime_checks` path: assert no tracer flux through
  fold seam land cells; assert vector rotation is unitary.

Validation:

- **Rest-state test** (the key gate): tripole grid, stratified T/S,
  zero forcing, integrate 1 sim-day with implicit_cn solver.
  Require `|η|max ≤ 1e-12 m`, `|u|max ≤ 1e-12 m/s`.  This catches
  any halo, metric, or fold error.
- All existing lat-lon ocean tests, run on regular lat-lon, remain
  bit-exact.

**Decision gate**: rest state preserved at machine precision on
ORCA1.  Existing lat-lon tests bit-exact.

### Phase 5 — Validation suite  (≈ 1.5 weeks)

Run the existing test matrix on tripole and characterise differences:

- Eady BCI (with tripole grid using Arctic-side initial condition):
  - Centered GM/Redi: front structure, KE growth match lat-lon to
    ~5%.
  - Triad GM/Redi: T-drift through the fold below 1e-9 K/12 days
    (lat-lon reference ~1e-12 K/12 days).
- ACC channel (tropical-band — south of cap, should match lat-lon
  bit-exact since metrics are identical there).
- Geostrophic adjustment in the cap.
- Munk gyre wrapping the cap (tests fold under sustained flow).
- Williamson-style ocean rest state on ORCA1 with bathymetry.

Each test gets a comparison plot in
`results/ocean/tripole_validation/<test>/`.

**Decision gate**: comparison plots show no spurious cap structure;
quantitative metrics within agreed tolerance.  Any discrepancy
> 5% south of cap is a bug to investigate.

### Phase 6 — Production global-overturning bring-up  (≈ 1 week)

- Adapt `run_global_overturning_implicit_spinup.py` and
  `run_global_overturning_50yr_implicit_continuation.py` to take a
  grid argument and run on ORCA1 (1°, 362×292).
- 10-yr fresh spinup + 40-yr continuation, exactly mirroring the
  lat-lon 50-yr pattern.
- Diagnostics:
  - Drake transport
  - MOC zonal-mean
  - Arctic overturning (NEW — could not be diagnosed on 36×72
    because the pole was degenerate)
  - Bipolar-cap velocity / eta health (no spurious cap features)

**Decision gate**: 50-yr ORCA1 run completes with healthy
diagnostics; Arctic structure is credible (no degeneracy at the
fold seam); Drake transport in the same physical regime as the
lat-lon 36×72 50-yr run.

## Validation hierarchy

In ascending order of difficulty / scientific value:

| Test | What it catches | Phase |
|---|---|---|
| Grid file load + area sum | NetCDF reader bug | 0 |
| Operator bit-exact on regular lat-lon | Metric-array refactor regression | 1 |
| Fold halo round-trip (scalar) | Fold permutation bug | 2 |
| Fold halo round-trip (vector + sign) | Vector fold bug | 2 |
| Uniform geographic flow zero-divergence | Rotation bug | 3 |
| Rest-state preservation 1 day | Any halo / metric / fold bug | 4 |
| Existing lat-lon test matrix bit-exact | Phase 1–4 regression | 4 |
| Eady BCI south of cap matches lat-lon | Operator correctness | 5 |
| Triad GM/Redi T-drift through fold | GM/Redi-on-fold correctness | 5 |
| 50-yr ORCA1 global overturning | End-to-end production readiness | 6 |

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Fold halo for higher-order schemes (SOM has 9 moments per cell — folding moments correctly is non-obvious) | High | Phase 5 explicitly tests SOM / DST-3 / WENO across fold; if any scheme fails, fall back to TVD on the cap row |
| Vector rotation cost in PCG iterations (the implicit barotropic solver does halo exchanges per iteration — recompilation or per-iter overhead matters) | Medium | Verify rotation arrays are static (not retraced); benchmark PCG iter cost with and without fold |
| GM/Redi triads at the fold (cross-fold triads need orientation-aware stencil) | Medium | Phase 5 tests this directly; if triads-through-fold fails, ship Phase 6 with `slope_scheme="centered"` (matches MPAS choice) |
| Land mask consistency at the fold seam (Murray 1996 grids treat coastline at the seam carefully — easy to introduce a one-cell leak) | Medium | Use ORCA1's distributed land mask file directly; do not regenerate |
| ESMF NetCDF reader brittleness (different conventions across OM4, ORCA, gx1v) | Low | Phase 0 picks ORCA1 specifically because its convention is the cleanest; OM4 fallback is a separate reader |
| Existing lat-lon tests don't actually exercise all halo patterns we now need (so "bit-exact" passes but tripole still fails) | Medium | Phase 1 validation runs the metric-array path on regular lat-lon AND adds new unit tests for each operator on the analytic test field |
| GPU / TPU paths break on fold halo (the index gather is irregular) | Low | The fold is a simple `i → N_i − i + 1` reverse — pure JAX `lax.gather` or array reverse, no host callbacks |

## Reuse map

What comes out of the existing lat-lon C-grid investment unchanged:

| Component | Reuse | Notes |
|---|---|---|
| EOS (Wright, linear) | 100% | column-local |
| KPP, Richardson, constant vertical mixing | 100% | column-local |
| Convection (enhanced_diffusion) | 100% | column-local |
| Surface forcing (combined, restoring, prescribed) | 100% | per-cell |
| Time integration (segment runner, scan blocks) | 100% | grid-agnostic |
| Restart I/O | 100% | grid-agnostic |
| Channel packing for ML | 100% | shape-preserving |
| Conservation diagnostics | 100% | grid-agnostic with metric arrays |
| Visbeck adaptive κ | 100% | column-local |
| GM/Redi centered + triads | ~90% | slope across fold needs Phase 2 halo |
| All advection schemes (upwind, TVD, SOM, DST-3, PPM-FCT, WENO) | ~85% | flux skeleton same; need fold-aware halo |
| `latlon_cgrid_operators.py` | ~95% | metric refactor only |
| Implicit-CN barotropic solver | ~95% | PCG framework unchanged; matrix apply uses tripole operators |
| `LatLonCGridOceanConfig` | 100% | no schema change |
| `LatLonCGridOceanState` | 100% | no schema change |
| Test matrix infrastructure | 100% | adds tripole as a grid axis |

What is genuinely new:

| Component | LOC estimate | Notes |
|---|---|---|
| `src/legoesm/grids/tripole.py` | ~400 | NetCDF reader + metric assembly |
| Fold halo wrapper | ~200 | scalar + vector |
| Vector rotation utility | ~150 | per-edge angles + halo integration |
| Metric-array refactor diff | ~300 | mostly mechanical |
| New tests | ~500 | round-trip, rotation, rest-state |

**Total new LOC: ~1500.**  Compare: the lat-lon C-grid migration was
~3k LOC, and the C-grid cubed-sphere effort is estimated at ~5–8k LOC
of new code.

## Calendar estimate

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Grid file selection + reader | 3 days | 3 d |
| 1 — Metric-aware operator refactor | 1 week | 8 d |
| 2 — Fold halo exchange | 1 week | 13 d |
| 3 — Vector rotation in cap | 4 days | 17 d |
| 4 — Ocean model integration + rest state | 1 week | 22 d |
| 5 — Validation suite | 1.5 weeks | 30 d |
| 6 — Production ORCA1 50-yr | 1 week | 35 d |

**Total: 7 weeks of focused work** (~1.5 months).  Compare: C-grid
cubed-sphere is estimated at 3–4 months.

## Reference material

- **Murray, R. J. (1996)** "Explicit generation of orthogonal grids
  for ocean models." *J. Comput. Phys.* 126, 251–273.  The standard
  derivation of the bipolar-cap construction.
- **Madec, G. and Imbard, M. (1996)** "A global ocean mesh to
  overcome the North Pole singularity." *Climate Dyn.* 12, 381–388.
  ORCA grid family.
- **NEMO ORCA grid files**: distributed by ECMWF and the NEMO
  consortium; ORCA1 is the standard 1° benchmark.
- **MOM6 OM4 grid files**: distributed with MOM6; tripolar with
  bipoles over Canada and Russia.
- `docs/md_files/ocean_grid_staggering.md` — staggering trade-offs for the
  C-grid path (this plan extends the lat-lon C-grid leg, not the
  cubed-sphere leg).
- `docs/md_files/LATLON_CGRID_MIGRATION.md` — pattern this plan mirrors.

## Open questions for decision before Phase 0

1. **ORCA1 vs OM4 OM4 0.5°**: ORCA1 is the recommended starting
   point (1°, simpler reader, more widely used).  OM4 if higher
   resolution is wanted from the start.
2. **Drop existing `create_latlon_grid()` support, or keep as
   metric-array fallback?** — recommendation: keep, as the
   regular-lat-lon path is the bit-exact validation target for
   Phase 1 and remains useful for development tests.
3. **Sea ice on tripole now or later?** — recommendation: later.
   The Arctic is the natural sea-ice domain, but the dynamics+physics
   port should land first to de-risk the seam handling.
