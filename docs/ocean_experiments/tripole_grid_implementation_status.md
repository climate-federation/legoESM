# Tripolar Grid Implementation Status

Companion to `tripole_grid_plan.md` (the 2026-04-29 scoping document).
This doc tracks what has actually been implemented on `feature/latlon-pole-fix`,
what was learned during bring-up, and what remains.

**Last updated:** 2026-05-20
**Branch:** merged to `main`

## TL;DR

- **Phases 0–6a complete.** Tripolar infrastructure landed, validated
  on a synthetic grid (39 unit tests), brought up on real eORCA1, and
  exercised in a clean 20-year ORCA1 production run.
- **20-yr ORCA1 production run** (`tripole_orca1_etopo_20yr_production/`)
  completed in 14.5 h wall on a single GPU at dt=600 s with 81 quarterly
  restarts/snapshots, no NaNs, final-state `max|u|≈1.02 m/s`,
  `|eta|≤1.7 m`, T ∈ [0, 27] °C.
- **Cap-region cascade** flagged here on 2026-05-08 (the original
  remaining blocker) was resolved by the mid-May fix sequence
  documented under "Phase 6a — Resolution" below.
- The 1 km `dx` floor is now an argument on `create_tripole_grid`
  (`min_dx_m`, default 1000.0) instead of a runner-side `_replace(...)`.

## What got built (Phases 0–5)

### Phase 0 — Grid infrastructure
**Files:** `src/legoesm/grids/latlon.py`, `src/legoesm/grids/tripole.py`

- `LatLonCGridGeometry` NamedTuple with per-cell metric arrays at all
  C-grid stagger points (T, u, v, q): `dx_T`, `dy_T`, `area_T`, `dx_u`,
  `dy_u`, `dx_v`, `dy_v`, `area_q`. Plus rotation angles
  `cos_alpha_u/v`, `sin_alpha_u/v` and a `FoldDescriptor`.
- Legacy compat fields (`cos_lat`, `sin_lat`, `lat`, `lon`, `dlon`, `dlat`)
  retained so downstream code that accesses them works unchanged.
- `FoldDescriptor`: `is_active`, `fold_j`, `cap_j`, `perm_T`, `perm_v`,
  `vector_sign_u/v`.
- `create_latlon_geometry()` — analytical constructor; bit-exact match to
  `create_latlon_grid()` for all metric values used by operators.
- `ensure_geometry(grid)` — converter; called once at model-construction
  time so existing call sites (`create_latlon_grid()`) continue to work.
- `create_tripole_grid(path)` — NEMO mesh_mask NetCDF reader.
- `create_synthetic_tripole(n_lat, n_lon)` — regular lat-lon with active
  fold descriptor, used for unit testing.
- `download_orca1_grid()` — fetches eORCA1 (484 MB) from Zenodo
  (`https://zenodo.org/records/4436658/files/eORCA1.2_mesh_mask.nc`).

### Phase 1A — Metric-array operator refactor
**File:** `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` (and friends)

Every operator that previously computed `cos(lat)`, `dlon`, `dlat` inline
now reads pre-computed per-cell metrics from the geometry. Dispatch is
via the sentinel `grid.dlat == 0.0` (set by tripolar grids):

- `gradient_x_cgrid`, `gradient_y_cgrid`, `divergence_cgrid`,
  `curl_vertex_cgrid`, `coriolis_cgrid`, `_gradient_curl_to_u/v`,
  `partial_cell_pgf_correction_x/y`, `density_jacobian_pgf_smc03_x/y`,
  `_make_diag_preconditioner` (barotropic_implicit), KE-gradient and
  velocity-divergence in `ocean_pe_latlon_cgrid.py`, `dst3_to_u/v_points`
  in `advection.py`, Zalesak FCT.
- Regular lat-lon (where `dlat > 0`) takes the legacy scalar/1D path —
  bit-exact backward compat.

### Phase 1B — North-boundary abstraction
**File:** `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`

Replaced ~20 hardcoded `jnp.pad(interior, ((1,1), …))` wall-BC patterns
with helpers:

- `_pad_ns_zero(interior)` — south & north both zero (wall BC).
- `pad_ns_scalar(interior, grid)` — south wall, north = wall on regular
  lat-lon, fold-reflected on tripolar.
- `pad_ns_vector_u/v(interior, grid)` — same but with sign flip across
  the fold for vector components.
- `pad_ns_vector_pair(u_int, v_int, grid)` — combined fold + rotation
  for (u, v) pairs (Phase 3 vector rotation).

### Phase 2 — Fold halo exchange
The fold logic lives inside `pad_ns_*` rather than a separate module. The
fold operation is pure `jnp` array indexing (`field[fold_j, perm]`) —
fully JIT/AD compatible, no host callbacks. T-fold convention (i-reversal
permutation `perm[i] = n_lon - 1 - i`).

### Phase 3 — Vector rotation in bipolar cap
`pad_ns_vector_pair` applies the combined transformation:
```
u_dest = -cos(Δα) · u_src - sin(Δα) · v_src
v_dest =  sin(Δα) · u_src - cos(Δα) · v_src
```
where `Δα = α_dest - α_source`. With zero rotation this reduces to the
sign flip `(-u, -v)`, matching the existing fold behaviour.

### Phase 4 — Ocean model integration
- `LatLonCGridOceanModel.__init__` calls `ensure_geometry(grid)` once.
- All downstream operators see `LatLonCGridGeometry`.
- Migrated remaining `jnp.pad` sites in `ocean_pe_latlon_cgrid.py`,
  `barotropic_latlon_cgrid.py`, `barotropic_implicit_latlon_cgrid.py`,
  `advection.py` to `_pad_ns_zero`.
- Forward-backward Coriolis Pad sites, strain_rate, stress_divergence,
  Leith viscosity, PV flux pads, advection v-face schemes — all migrated.

### Phase 5 — Validation suite
**File:** `tests/ocean/unit/test_tripole_fold.py` — 21 new tests:

- Fold round-trip (scalar i-reversal, vector sign flips, involution
  property, 3D fields, wrap-column staggering).
- Operators on synthetic tripole produce finite results with nonzero
  north boundary (gradient_y, divergence, curl, Coriolis, Laplacian).
- Backward-compat: `pad_ns_*` on regular lat-lon = `jnp.pad` bit-exact.
- Vector rotation: zero-rotation parity, rotation mixes u/v at fold,
  rotation preserves kinetic energy (orthogonality), 3D fields work.
- **Rest-state test on synthetic tripole**: `max|eta| = 0`, `max|u| = 0`.
- **Wind-forced stability**: 5 steps under velocity kick — all finite.
- **Differentiability**: `jax.grad` through tripolar timestep returns
  finite gradients.

### Stats
- 8 commits (the Phase-5-final state at commit `a1955f8f`).
- ~2500 LOC added/modified.
- 39 tests passing (18 pre-existing + 21 tripole).
- Zero regressions in the broader ocean test suite.

## Phase 6a — ORCA1 production bring-up

**Goal:** Run the global overturning experiment (wind + SST restoring) on
the eORCA1 tripolar grid (332×362, ~1°) using NEMO's native bathymetry
and land mask.

**Runner:** `scripts/run/global_overturning/run_tripole_20yr.py` (the runner
referenced earlier in this doc — `run_global_overturning_tripole.py` —
was the bring-up version with the 4 documented bugs and was deleted on
2026-05-20 after the production runner stabilised.)

**Grid file:** `data/grids/eORCA1.2_mesh_mask.nc` (484 MB, downloaded from
Zenodo via `download_orca1_grid()`).

**Bathymetry reconstruction:**
```python
H_bathy[j,i] = gdept_1d[mbathy[j,i]]   # last active depth level
```
where `mbathy` is the integer level-count field and `gdept_1d` is the
1D depth coordinate from the mesh_mask file.

### Bugs found during bring-up

#### Bug 1: `prescribed_surface_forcing` C-grid shape mismatch
**Commit:** `ccf6c3ab`
**File:** `src/legoesm/ocean/physics/surface_forcing/prescribed.py`

Function returned `du_dt` with T-point shape `(n_lat, n_lon, nlev)` but u
lives at u-points `(n_lat, n_lon+1, nlev)` on the lat-lon C-grid. Same for
`dv_dt`/v-points. This silent shape mismatch was masked at coarse
resolution (5° tests pass) but blew up at ORCA1 resolution.

**Fix:** Detect C-grid via shape comparison (`u.shape[1] == T.shape[1]+1`),
then interpolate T→u (zonal average with periodic wrap) and T→v
(meridional average with zero pole pad).

A-grid and cubed-sphere callers (where u/v share T's shape) take the
no-op path — backward compatible.

#### Bug 2: Helmholtz preconditioner per-face metrics
**Commit:** `9a3c5ad3`
**File:** `src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py`

The Phase 1A migration of `_make_diag_preconditioner` took
`grid.dx_u[:, 0]` etc. — only column 0 of the 2D metric arrays. On regular
lat-lon all columns are identical so this works. On tripolar, dx varies
substantially in longitude (especially in the bipolar cap), so column-0
values are unrepresentative. The preconditioner returns a wildly wrong
inverse-diagonal, causing PCG to diverge or converge to a bad solution
during the implicit barotropic solve.

**Fix:** On the tripolar path, use the full per-face 2D metrics:
```
dy_u_E = grid.dy_u[:, 1:],  dy_u_W = grid.dy_u[:, :-1]
dx_u_E = grid.dx_u[:, 1:],  dx_u_W = grid.dx_u[:, :-1]
dx_v_N = grid.dx_v[1:, :],  dx_v_S = grid.dx_v[:-1, :]
dy_v_N = grid.dy_v[1:, :],  dy_v_S = grid.dy_v[:-1, :]

diag_zonal = (H_u_E * dy_u_E / dx_u_E + H_u_W * dy_u_W / dx_u_W) / area
diag_merid = (H_v_N * dx_v_N / dy_v_N + H_v_S * dx_v_S / dy_v_S) / area
```
Regular lat-lon path unchanged for bit-exact backward compat.

#### Bug 3: Surface forcing land-cell zeroing
**Commit:** `9a3c5ad3`
**File:** `src/legoesm/ocean/physics/surface_forcing/prescribed.py`

On land cells `H_bathy = 0` → `jacobian = 0` → `dz_0 = 0`. The previous
`1/(rho * jnp.maximum(dz_0, 1e-10))` clamp gave finite-but-huge values
(~1e7) on land. Fine when `du_dt` is masked at the point of use, but
catastrophic when interpolated to neighbouring u/v faces (Bug 1's fix) —
ocean cells adjacent to coast get contaminated by the huge land values.

**Fix:**
```python
is_ocean = dz_0 > 1.0e-3   # > 1 mm cell thickness ⇒ ocean
inv_rho_dz = jnp.where(is_ocean, 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10)), 0.0)
```
Applied to all three surface tendency formulas (momentum, heat, freshwater).

#### Bug 4: Shelf cell masking
**Commit:** `9a3c5ad3`
**File:** `scripts/run/global_overturning/run_tripole_20yr.py` (then `run_global_overturning_tripole.py`)

z-star vertical coordinate scales the first-layer thickness as
`dz_ref[0] · H_bathy / H_max`. On a 27 m shelf cell with `H_max = 5500 m`
and `dz_ref[0] = 26.2 m`, this gives `dz_0 = 0.13 m` (13 cm), making
surface forcing `du/dt = tau / (rho · dz_0)` ~200× the deep-ocean value.

This is correct z-star behaviour but incompatible with naive prescribed
wind on a real grid that contains shelf cells. The proper fix is a hybrid
z/z-star coordinate or partial cells; for now we mask cells with
`H_bathy < 500 m` as land. **Step 1 max|u| dropped from 0.46 m/s to
0.024 m/s** with this fix (vs analytical ~4×10⁻³ m/s; 6× over-estimate is
reasonable for geometric variation).

#### 1 km dx floor — lifted into `create_tripole_grid`
**Commit (lift):** 2026-05-20 cleanup
**File:** `src/legoesm/grids/tripole.py`

eORCA's design places the displaced "northern poles" on Canada and
Russia (land) — the cells with tiny `dx` at the bipolar fold seam are
all land. On the raw eORCA1.2 mesh ~2.4 % of cells have `dx_T <
1000 m`, and the minimum `dx_v`/`dy_v` reaches **0 m** (degenerate
seam cells), which is a hard NaN trigger via division, not merely a
CFL slow-down. The previous runner-side `_replace(...)` floor was
therefore load-bearing.

The clamp is now an argument on `create_tripole_grid` — `min_dx_m`
(default `1000.0`, pass `0.0` to disable) — applied uniformly to
`dx_T, dy_T, dx_u, dy_u, dx_v, dy_v`, with `area_T, area_q` clamped
to `min_dx_m**2`. Bit-exact with the old runner-side post-construction
floor (verified on eORCA1.2 mesh, 2026-05-20). Callers no longer need
to know about it.

### Phase 6a — Resolution (cap-region cascade fixed)

The cap-region cascade flagged on 2026-05-08 was closed by a sequence
of mid-May fixes. The argmax-`|u|` had been migrating from -51° (step
1, normal ocean) to ~75–77°N (steps 3+, bipolar cap) during the
cascade; the root causes turned out to be Hypothesis 3 (a hidden
column-0 metric extraction in the strain/stress operators) and a
preconditioner dtype-promotion bug, not the viscosity / PCG /
time-step hypotheses speculated about above.

Commits that landed the resolution (newest first):

- `860a54ed` (2026-05-19) — Fix `sys.path` in tripolar runner: point
  to `src/` not repo root.
- `c2f5ba46` (2026-05-18) — Runner: rewrite with ORCA bathy + 80°N
  cap + linear EOS + production config. `H_min=200` removes dangerous
  shallow cap cells (the "27 m shelves next to 5500 m cliffs in the
  bipolar cap" called out as Bug 4 above). Adcroft PGF + partial cells.
- `1fb210df` (2026-05-18) — Fix `strain_rate_cgrid` +
  `stress_divergence_cgrid` for tripolar 2D metrics. The remaining
  column-0 metric extraction (Hypothesis 3) — caught.
- `bb51083b` (2026-05-18) — Fix preconditioner dtype promotion on
  tripolar geometry. PCG was silently downcasting to float32 in the
  cap, where the order-of-magnitude `dx` variation needs full
  double-precision arithmetic to converge.
- `5deaa2bb` (2026-05-14) — Fix `gradient_y_cgrid` broadcast for
  tripolar + variable-`dy` merge.

**Production validation:** 20-yr eORCA1 idealized forced run
(`tripole_orca1_etopo_20yr_production/`) — see TL;DR above. 81
quarterly restarts, no NaNs end to end, scientifically reasonable
ACC/AMOC structure visible in the cross-grid comparison plots
(`comparison_idealized_*` and `comparison_idealized_sections_*` in
`~/saved_legoESM_data/`).

### Reproducing the 20-yr production run

```bash
JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_tripole_20yr.py \
    --grid data/grids/eORCA1.2_mesh_mask.nc \
    --etopo data/bathymetry/etopo_1deg.nc --years 20
```

Output lands under `results/tripole_orca1_etopo_20yr/`; the
2026-05-19/20 run was archived to
`~/saved_legoESM_data/tripole_orca1_etopo_20yr_production/`.

## Architectural decisions

These are choices that should NOT be revisited without good reason — they
shaped the entire phased rollout:

1. **`dlat == 0.0` as tripolar sentinel.** Operators check
   `grid.dlat == 0.0` to dispatch to the per-cell metric path. On regular
   lat-lon, `dlat > 0` always; on tripolar, `dlat = 0.0` (literal sentinel
   set in `create_tripole_grid`). This avoids `isinstance` checks and
   keeps the code branch-free in the JIT trace.
2. **Geometry conversion at model init, not per-operator-call.** The
   ocean model constructor calls `ensure_geometry(grid)` once. All
   operators receive the converted geometry. This means call sites that
   pass `LatLonGrid` (e.g. existing tests) work unchanged.
3. **Fold logic inside `pad_ns_*`, not a separate module.** Originally the
   plan called for a `tripole_fold.py` module. In practice the fold
   operation (i-reversal + sign flip) is so simple that putting it in the
   pad helpers was cleaner — operators don't need to know whether the
   fold is active.
4. **Synthetic tripole for all unit tests.** Tests use
   `create_synthetic_tripole()` (regular lat-lon with active
   `FoldDescriptor`) rather than ORCA1, because ORCA1 is 484 MB. This
   means the unit tests don't catch real-grid issues — those have to be
   caught at integration time. Phase 6a is the result.

## File index

- `src/legoesm/grids/latlon.py` — `LatLonCGridGeometry`, `FoldDescriptor`,
  `ensure_geometry`, `create_latlon_geometry`.
- `src/legoesm/grids/tripole.py` — `create_tripole_grid`,
  `create_synthetic_tripole`, `download_orca1_grid`, `_read_nemo_mesh_mask`,
  `_detect_fold`, `_compute_rotation_angles`.
- `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` —
  `_pad_ns_zero`, `pad_ns_scalar`, `pad_ns_vector_u/v`,
  `pad_ns_vector_pair`, all migrated operators.
- `src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py` —
  `_make_diag_preconditioner` (per-face metrics on tripolar).
- `src/legoesm/ocean/physics/surface_forcing/prescribed.py` —
  C-grid T→u/v interpolation, land-cell zeroing.
- `tests/ocean/unit/test_tripole_fold.py` — 21 fold/rotation/integration
  tests.
- `scripts/run/global_overturning/run_tripole_20yr.py` — ORCA1
  production runner used for the 20-year reference run. The bring-up
  variant `run_global_overturning_tripole.py` plus the
  `_diagnose_tripole_blowup.py`, `_probe_fold_cells.py`,
  `_probe_rest_state.py`, `_probe_rest_uniform.py` probe scripts
  were deleted on 2026-05-20 after the production runner stabilised.

## See also

- `docs/ocean_experiments/tripole_grid_plan.md` — original 2026-04-29
  scoping document. The phasing in this implementation status corresponds
  to the phases in that plan.
