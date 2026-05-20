# Tripolar OMIP-style run on eORCA1 — implementation plan

**Status (2026-05-20):** Plan accepted. Step 1 (smoke test) in progress.

Companion to `tripole_grid_plan.md` (the 2026-04-29 scoping document)
and `tripole_grid_implementation_status.md` (the bring-up record,
Phase 6a closed 2026-05-20).

## Goal

Run the same JRA55-DO OMIP-style integration on the **tripolar
eORCA1** grid as the existing MPAS OMIP runs (`mpas_jra55_etopo_*`).
Same forcing recipe (`jra55_do_tropical`, LY09 bulk fluxes), same
SSS-restoring + freeze-cap, eventually WOA18 initialization for a
real climate run. Final deliverable: a 20-yr (then 100-yr) eORCA1
tripolar OMIP run alongside the `mpas_jra55_etopo_100yr_*` series,
plottable with the same comparison scripts.

## Why this is feasible without large code changes

The 2026-05-20 cleanup left the tripolar pieces almost ready:

1. **`LatLonCGridOceanModel.step()`** already accepts
   `surface_forcing=` (`src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1278`)
   — the same external-forcing path MPAS uses. The model itself
   needs no changes; tripolar uses this same model.
2. **`_setup_jra55_forcing_state` + `_jra55_step`** in
   `scripts/run_omip.py` already implement JRA55-cache → bulk-flux →
   `surface_forcing` for both `latlon` and `mpas`. Currently gated
   to `_supported_jra55_grids = ("latlon", "mpas")` at line 878.
3. **`create_tripole_grid(min_dx_m=1000.0)`** loads eORCA1.2 with
   per-cell metrics + fold descriptor + rotation angles. Used by
   `run_tripole_20yr.py`.
4. **ETOPO bathymetry + partial cells + z-star** for the tripolar
   C-grid is validated by the 20-yr idealized production run.
5. **Vector rotation hooks** (`cos_alpha_u`, `sin_alpha_u`,
   `cos_alpha_v`, `sin_alpha_v`) are present on the geometry — used
   to rotate u/v for the comparison plots, so JRA55 geographic
   winds → grid-aligned `tau_i` / `tau_j` is mechanical.

## What needs adding (file-by-file)

### A. Dispatch tripolar through `run_omip.py`

- **`_parse_resolution`** (`run_omip.py:294`): add
  `elif grid_type == "tripole"`. Resolution mapping likely just
  `{"eorca1": "data/grids/eORCA1.2_mesh_mask.nc"}`.
- **`_create_setup`** (`run_omip.py:349`): add `elif grid_type ==
  "tripole"` branch that builds `create_tripole_grid(...)`, ETOPO
  bathymetry (lift the proven recipe out of
  `run_tripole_20yr.py:63-67`), z-star, `LatLonCGridOceanConfig`
  matching the MPAS-JRA55 dissipation defaults (`A_h=1e5`,
  `A_v=1e-4`, `K_v=1e-5`, `pgf_scheme="adcroft"`,
  `barotropic_solver="implicit_cn"`, `normalize_freshwater=True`,
  GM/Redi, etc.). Surface forcing = `SurfaceForcingConfig(scheme="none")`
  to match the MPAS JRA55 path.
- **`_init_rest_state`** (`run_omip.py:694`): add tripolar branch —
  call `rest_state_latlon_cgrid_ocean(geom, ..., H_bathy_override=...)`.
  The existing `latlon` branch already supports tripolar geometries
  via Phase 6a work.
- **Output naming**: tripolar runs land at
  `results/<tag>/tripole/eorca1/` mirroring `results/<tag>/mpas/ico5/`.

### B. JRA55 cache → tripolar T-points

- **`_setup_jra55_forcing_state`** (`run_omip.py:854`): widen
  `_supported_jra55_grids` to `("latlon", "mpas", "tripole")`.
  Inside, the per-grid bit is computing the destination lat/lon
  for regridding the JRA55 cache. For tripolar that's
  `np.degrees(geom.lat_T)`, `np.degrees(geom.lon_T)` (2D arrays on
  the geom). The JRA55 cache is on a regular 0.5625° atmospheric
  grid; need to verify whether the existing regridder is
  grid-agnostic or has 1-D lat/lon assumptions.
- **`_jra55_step`** (`run_omip.py`, near line 851): unchanged in
  spirit, but verify whether it indexes any field as `[..., j]`
  vs `[..., j, i]`. C-grid tripolar arrays are 2D `(n_lat, n_lon)`,
  same as the regular latlon C-grid, so this should "just work"
  once the regridder produces a 2D output.

### C. Wind-stress rotation at the fold

JRA55 gives `tau_east`, `tau_north` in geographic coordinates. The
C-grid u-equation needs `tau_i` (grid-aligned), so:

```
tau_i_at_u  = tau_east * cos_alpha_u + tau_north * sin_alpha_u
tau_j_at_v  = -tau_east * sin_alpha_v + tau_north * cos_alpha_v
```

This is mostly identity south of the cap (`cos_alpha=1`,
`sin_alpha=0`) so it has zero impact outside the bipolar fold.
**Where this hooks in**: either inside
`bulk_formula_surface_forcing` (cleanest — then it's done for any
tripolar caller), or in `_jra55_step` right before the result is
packed for `surface_forcing`. Either way it's a 5-line patch; the
bulk-formula path with a `geom`-aware dispatch is preferred.

### D. Freeze-cap + SSS restoring + freshwater normalization

Pre-existing knobs in the MPAS path:

- **Freeze-cap**: SST minimum clamp at
  `T_freeze_ocean = 271.35 K`. Already implemented for MPAS; need
  to verify it lives in a grid-agnostic place
  (`physics/surface_forcing/integration.py`?) or whether it's
  MPAS-specific. If MPAS-specific, factor into shared utility.
- **SSS restoring**: applied via `surface_forcing` pipeline outside
  the JRA55 path. Confirm `LatLonCGridOceanConfig` has the same
  fields as `MPASOceanConfig.physics.surface_forcing.restoring`.
- **`normalize_freshwater`**: yes on `MPASOceanConfig`. Confirm
  `LatLonCGridOceanConfig` has it — if not, port the global
  freshwater normalization (commit `16ac1e86`) to the C-grid model.

### E. WOA18 initialization (climate run, not smoke test)

- Interpolate WOA18 T(z), S(z) from regular 1° lat-lon onto
  tripolar T-points + z-star levels. The MPAS init path does this;
  the cubed-sphere does too. For tripolar, the destination is
  `geom.lat_T, geom.lon_T` 2D — the same KDTree/IDW regridder used
  in B works.
- Opt-in via existing `--woa-t`, `--woa-s` flags; when absent, fall
  back to the analytical cold start (`T_water_init_C=20`).

## Sequenced delivery

### Step 1 — Smoke test (1 day from rest)

Get a tripolar JRA55 run to step cleanly for **1 day** from rest,
with all the plumbing wired (no WOA, no freeze-cap, no SSS-restoring
tuning). This catches every wiring bug:

- Tripolar dispatch through `_create_setup` / `_init_rest_state`.
- JRA55 cache → tripolar regrid.
- Wind-stress rotation at the fold.
- `surface_forcing` argument actually reaches
  `LatLonCGridOceanModel.step()`.

Pass criteria: completes without NaN, `max|u| < 1 m/s`, no shape
errors at the fold.

### Step 2 — Stability (10-yr nudge spinup)

Run `tripole_jra55_etopo_10yr_nudge`, the analog of
`mpas_jra55_etopo_10yr_nudge_ico6`: WOA T/S nudging on top of JRA55
forcing, A_h=1e5, KPP, GM/Redi 2400, full production knobs. Pass:
completes 10 yr, `max|u| < 3 m/s` throughout, no NaNs, SST in
[-1.8, 35] °C.

### Step 3 — Production 100-yr

Restart from the 10-yr spinup, run to year 100
(`tripole_jra55_etopo_100yr`). Save quarterly restarts +
snapshots. Cadence matches `mpas_jra55_etopo_100yr_woa_v5`.

### Step 4 — Comparison plots

Extend `compare_etopo_idealized_4runs.py` +
`compare_etopo_idealized_sections.py`: add `"jra55_3way"` set in
`RUN_SETS` with `(MPAS ico5, MPAS ico6, Tripolar ORCA1)`. The
plotting infrastructure already handles tripolar (used last
conversation).

## Risks / things worth pre-validating

- **Bipolar-cap freshwater closure** under JRA55 P-E + river runoff.
  Fold halo exchange must conserve freshwater mass across the
  seam; 20-yr idealized run had no freshwater flux so this is
  untested.
- **Bulk-formula at the cap**: bulk fluxes use 10-m winds + 2-m air
  T/q. Near the displaced poles those are JRA55 values at high
  latitudes (geographic), no special structure — should be fine,
  but worth sanity-checking the first 30 days for unexpected
  fluxes near the seam.
- **PCG conditioning at the cap under JRA55 wind**: the cascade
  resolved in May was under idealized forcing. JRA55 winds are an
  order of magnitude more intense in storms; if PCG was marginal,
  this is where it'd surface. Mitigation: same
  `barotropic_implicit_pcg_tol=1e-10`, `maxiter=300` defaults as
  MPAS.
- **WOA18 interpolation onto curvilinear grid**: at the cap,
  regular WOA18 cells become very oblique relative to the eORCA1
  grid. Use the same conservative regridder MPAS uses, or KDTree
  IDW as a first cut for the smoke test.

## Open decisions (recorded after agreement)

1. **Extend `run_omip.py` rather than fork** a focused
   `run_omip_tripole.py`. Same as how MPAS was added on top of an
   existing latlon-grid script.
2. **Resolution beyond ORCA1**: not in scope for this plan.
3. **GPU budget**: 20-yr idealized used 1 GPU, 14.5 h wall. JRA55
   adds a ~3 GB cache + a few surface arrays; fits on a V100S.
4. **This doc** is the durable handoff record; status updates
   land here, not in conversation.
