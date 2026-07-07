# Global land-carbon initial-condition map — design (Stage A)

**Status:** design, awaiting review.
**Depends on:** the semi-analytic soil-C spin-up
(`legoesm.land.carbon.spinup.analytic_slow_pool_equilibrium`, PR #823).
**Follow-up:** Stage B (global carbon parameter training warm-started from this
map, against a *modular* multi-observation loss) is a separate spec.

## Goal

Produce a **global, gridded `CarbonState` initial-condition map** ("finidat")
for the DifferLand prognostic carbon pools, so global land-carbon runs (and the
Stage-B training) start *at equilibrium* instead of cold — without spinning up
every grid cell for millennia. The method: **sample the (PFT × climate) space
that occurs globally, spin each sampled archetype to a verified semi-analytic
equilibrium, and assign each grid cell the cover-weighted mix of its
archetypes' equilibria.**

Baseline resolution **1°**; the protocol is resolution-agnostic (finer grids
change only the per-cell inputs, not the archetype machinery).

## Why archetypes (not per-cell spin-up)

A per-cell semi-analytic spin-up would still run one coupled column per land
cell (~15k at 1°). The carbon equilibrium depends on the cell only through its
**PFT parameters** and its **climate** (which set GPP, allocation, and
temperature/moisture-controlled turnover). Cells with the same PFT and
near-identical climate reach the same equilibrium. So we cluster the occupied
(PFT × climate) space into O(hundreds) archetypes, spin those, and map back —
1–2 orders of magnitude fewer columns, and a reusable `(PFT,climate)→pools`
lookup.

## Inputs (data already in the repo)

- **PFT cover**: CLM5 surfdata (`land.surface_data.sources.clm5_surfdata`) —
  per-cell 17-PFT weights aligned with `CLM5_PFT_NAMES`.
- **PFT parameters**: `land.param_providers` / `clm5_pft_table` — per-PFT
  Vc_max25, LCMA, root_depth, θ_wp/fc, etc. (the same table Stage B trains).
- **Climate**: ERA5/AMIP climatology (`tools.forcing.amip`) reduced to per-cell
  features (see below).
- **Soil texture**: HWSD (`land.surface_data.sources.hwsd2`) for hydraulics.

## Climate features (archetype axis)

Per cell, from the ERA5/AMIP monthly climatology:

- **MAT** — mean-annual 2 m air temperature [K].
- **MAP** — mean-annual precipitation [kg m⁻² yr⁻¹].
- **T seasonality** — amplitude of the mean annual T cycle (½·(max−min monthly
  T)) [K]; sets the growing-season length / phenology forcing.
- **Aridity index** — MAP / PET-proxy (PET from net radiation, Priestley–Taylor
  or a radiation proxy) [–]; separates wet from dry same-MAT climates.
- **Mean shortwave** — mean-annual down-SW [W m⁻²]; sets the light level.

These five drive both the clustering and the representative forcing.

## Architecture — three components + a validator

### 1. Archetype builder — `legoesm.land.carbon.global_init.build_archetypes`

- **In**: per-cell PFT weights `(ncell, n_pft)`, per-cell climate features
  `(ncell, 5)`, per-cell soil-texture class, a land mask.
- Per PFT that occurs anywhere, k-means-cluster the climate features of the
  cells where that PFT has weight > `w_min` into `K` bins (K per-PFT, capped;
  standardised features). Each cluster centroid + its modal soil texture is one
  **archetype**.
- **Out**:
  - `ArchetypeTable`: for each archetype — PFT id, climate-feature centroid,
    representative forcing parameters, soil-texture class, `MultiLayerLandConfig`.
  - `cell_to_archetypes`: sparse `(ncell)` list of `(archetype_id, weight)` where
    `weight = cover_fraction × (cell in that PFT's cluster)`; weights per cell
    sum to the vegetated fraction.
- **Testable in isolation**: deterministic clustering (fixed seed), every
  occupied (PFT, cell) maps to exactly one archetype, weights reproduce cover.

### 2. Archetype equilibration — `global_init.equilibrate_archetypes`

- Reuses the **semi-analytic spin-up** and a NEW climate-feature forcing
  generator `land.climate_forcing.make_climatological_forcing(MAT, T_seasonal,
  T_diurnal, sw_mean, precip_rate, doy, hour)` — a generalisation of
  `make_synthetic_lmip_forcing` parameterised by **real climate features**
  instead of latitude (single source; `lmip_forcing` becomes the
  latitude-preset caller).
- For each archetype (vmap/scan over archetypes — hundreds of tiny columns),
  run the coupled land+carbon transient (`--carbon-spinup semi_analytic`
  protocol) to a verified equilibrium; assert the verify-segment drift is small.
- **Out**: `archetype_id → equilibrium CarbonState` (6 pools) + the diagnostic
  GPP/NPP/SOC of each archetype (for QC vs biome ranges).

### 3. Global mapper — `global_init.map_to_grid`

- Per cell, `CarbonState_cell = Σ_a weight_a · pools(archetype_a)` (cover-
  weighted mix of its archetypes' equilibria); bare-soil fraction contributes
  zero vegetation pools. Conserves by construction (linear mix).
- **Out**: gridded `CarbonState` `(ncell, 6)`; written as a restart/"finidat"
  (`.npz`/NetCDF) plus the `ArchetypeTable` + lookup, so Stage B and any global
  run load it directly.

### 4. Validator — `scripts/validate/global_carbon_ic_map.py`

The map's *point* is that it is at equilibrium. No new global driver is needed:
`step_multilayer_land` already batches columns (`ncol`), so we validate on a
**representative sample of cells** (a batch, not the full 15k grid):

- Draw a stratified sample of land cells (across PFT/climate), load their mapped
  `CarbonState`, run a short (~5-yr) coupled land+carbon segment forced by each
  cell's climatology; the **carbon drift → 0** (report %/yr per pool, vs a
  cold-start control which drifts strongly). A single-PFT cell should be at its
  archetype's equilibrium; a mixed cell tests how well the cover-weighted mix of
  equilibria approximates the true equilibrium.
- Per-archetype SOC/biomass/GPP vs published biome ranges (reuse the Stage-0
  `LITERATURE` gates) — a realism smoke on the archetypes.
- Coverage: every land cell is assigned; weights sum to cover; no NaN/negative
  pools.

## Data flow

CLM5 PFT + ERA5 climate + HWSD soil → **build_archetypes** → ArchetypeTable +
cell→archetype weights → **equilibrate_archetypes** (semi-analytic) → archetype
equilibria → **map_to_grid** → global `CarbonState` finidat → (Stage B / global
runs) → **validator** confirms drift→0.

## Where it lives

- `packages/land/legoesm/land/carbon/global_init.py` — builder/equilibrate/mapper.
- `packages/land/legoesm/land/climate_forcing.py` — climate-feature forcing
  (refactor `lmip_forcing` onto it).
- `scripts/data/build_global_carbon_ic.py` — driver (loads maps, writes finidat).
- `scripts/validate/global_carbon_ic_map.py` — the drift/realism validator.
- Tests mirror under `tests/land/...`; every new module gets a direct test.

## Testing

- Unit: clustering determinism + coverage; climate-forcing (features → sane
  diurnal/seasonal cycle; reduces to the latitude preset); mapper conservation +
  weighted-mix correctness; equilibration reuses the tested spin-up.
- Integration: a tiny synthetic world (few PFTs × few climate cells) → archetypes
  → equilibria → map → assert per-cell pools equal the cover-weighted archetype
  equilibria and global drift is ~0 over a short verify run.
- Compute on Ginsburg via sbatch (login-node policy); the archetype spin-ups are
  cheap, the 1° validator segment is the heaviest piece.

## Explicitly out of scope (Stage B / later)

- The training loop + the **modular observation-loss registry** (SOC, LAI/
  biomass, SIF, δ¹³C-SOC) — separate spec. Stage A only produces the warm-start.
- A model-side SIF diagnostic (needed before a SIF loss term) — Stage B.
- Fractional-PFT sub-grid tiling beyond the cover-weighted pool mix.
- Observational calibration of the archetype climate binning (K selection is a
  tunable with a documented default).

## Open defaults (chosen; changeable in review)

- Dominant clustering per PFT with cover-weighted **fractional** mixing at map
  time (not dominant-PFT-only) — more accurate, matches `PFTParamProvider`.
- `K` climate bins per PFT: default 12, capped by occupied-cell count.
- Representative forcing synthesised from climate features (v1), not per-archetype
  ERA5 annual cycles (v2) — lighter + resolution-agnostic.
