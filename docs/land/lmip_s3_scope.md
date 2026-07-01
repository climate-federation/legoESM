# LMIP (TRENDY S3) — Scoping Document & Workplan

**Status:** active · **Branch:** `land-dev-followup` · **Last updated:** 2026-06-30

This document scopes what it takes to run an LMIP land simulation in legoESM and
lays out the workplan for the first push. "LMIP" here means the **TRENDY S3**
factorial scenario (Sitch et al., 2024, *Global Biogeochemical Cycles*,
doi:10.1029/2024GB008102).

---

## 1. Objective

Stand up the machinery for a TRENDY-style land run. The **end goal** (multi-push)
is an S3 land-carbon simulation producing annual NBP. **This first push** is
deliberately narrower: get **real reanalysis meteorology driving the existing
biophysical land model**, validated, before any carbon work begins.

---

## 2. The S3 protocol (target definition)

From the TRENDY-v11 protocol (Sitch et al. 2024, Table 2 / §2.3):

- **S3 = all forcings transient** over 1700→2021: atmospheric CO₂, climate,
  N deposition, N fertilizer, **and** land-use/land-cover change (LULCC).
  (For contrast: S0 = all pre-industrial; S1 = only CO₂ transient; S2 = CO₂ +
  climate transient, LULCC fixed at 1700. The attribution differences are
  S1 = CO₂ fertilization, S2−S1 = climate, **S3−S2 = E_LUC**.)
- **Spin-up:** CO₂ and land cover at 1700 levels (CO₂ = 276.59 ppm); climate from
  1901–1920 cycled until **soil-carbon pools reach equilibrium**.
- **Transient meteorology:** 1700–1920 reuse 1901–1920 climate repeatedly;
  1921–2021 use the actual-year climate.
- **Climate forcing:** CRU-JRA, 6-hourly, 0.5°. Variables: Tmin/Tmax/Tmean,
  precip, specific humidity, surface pressure, u/v wind, downward longwave
  (dlwrf), downward shortwave (dswrf). (Optional: diffuse-radiation fraction.)
- **Land-use forcing:** LUH2-GCB states + transitions, wood harvest.
- **N forcing:** N deposition (1850–2014 historical + RCP8.5 to 2021),
  N fertilizer (post-1910).
- **Required output:** annual NBP (PgC/yr), globally and for
  N-extratropics (>30°N) / tropics / S-extratropics (<30°S).
- **Budget-inclusion gates:** (1) steady state after spin-up — |NBP| offset
  < 0.10 PgC/yr, drift < 0.05 PgC/yr/century; (2) S_LAND−E_LUC is a sink in the
  1990s–2000s; (3) E_LUC is a source.

**Implication:** a *faithful* S3 is fundamentally a carbon-cycle simulation with a
soil-carbon spin-up and land-use bookkeeping. That is the multi-push horizon, not
this push.

---

## 3. Current capability inventory (ground truth, 2026-06-30)

Canonical source: `packages/land/legoesm/land/`. (`src/legoesm/land/` is a stale
empty skeleton — ignore.)

| Area | Status | Evidence |
|---|---|---|
| Photosynthesis / GPP | ✅ real — Farquhar C3/C4 + LUE | `canopy/photosynthesis.py`, `carbon/carbon_cycle.py` |
| Ra, Rh, allocation, phenology, NEE | ✅ real — DALEC990 6-pool, conserves C to 1e-9 | `carbon/carbon_cycle.py`, `tests/land/unit/test_carbon_cycle.py` |
| Biophysics (soil T, hydrology, snow, SEB, canopy) | ✅ real | `multilayer_land.py`, `soil_thermal.py`, `richards.py`, `snow_budget.py`, `surface_scheme/`, `canopy/` |
| Surface/boundary data (soil, PFT, LAI) | ✅ real — HWSD2 + CLM5 producer→surfdata | `land/surface_data/`, `land/boundary_data/` |
| Prognostic LAI | 🟡 exists, opt-in/off | `surface_scheme/two_leaf_canopy.py:compute_prognostic_lai` |
| Reanalysis met forcing reader | ❌ absent — drivers use synthetic forcing | — |
| Transient CO₂ | ❌ hard-coded (412 ppmv) | drivers' `_make_forcing`/`make_global_forcing` |
| Global multi-year production land driver | ❌ absent | `run_lmip.py` (single-point, carbon off), `run_lmip_smoke.py` (global, idealized, short) |
| Soil-C spin-up protocol | ❌ absent | — |
| NBP diagnostics | ❌ absent | — |
| N cycle (dep + fertilizer) | ❌ absent | — |
| LULCC → E_LUC | ❌ absent (schema can carry transient `pft_frac`, no flux logic) | `surface_data/schema.py` |
| Multi-pool soil C (CENTURY/RothC) | 🟡 single bulk SOM pool | `carbon/carbon_cycle.py` |

**Key takeaway:** the expensive part (a tested carbon core) already exists. The
gap to a TRENDY run is **plumbing** — realistic forcing, a production driver,
spin-up, diagnostics — not new biogeochemistry.

### 3.1 The `run_lmip_smoke` configuration (this push's testbed)

Confirmed in source:
- Carbon scheme = **`"none"`** (default `CarbonConfig()`, `carbon/config.py:66`) →
  **biophysics only** (energy + water + snow + soil temperature); **no DALEC, no NBP**.
- LAI = **prescribed CLM5 monthly climatology**, interpolated per day-of-year via
  `interp_monthly` (`global_surface_data.py:657`), which does `mod(doy, 365)` with a
  Dec→Jan wrap → the seasonal LAI cycle **repeats every year with no interannual
  variability**.
- Time integration = `jax.lax.scan`; forcing currently synthetic
  (`make_global_forcing`).

---

## 4. Scope decisions (locked for this push)

| Decision | Choice |
|---|---|
| Target | **WS-1 forcing enabler only**, then re-scope WS-2+ |
| Carbon | **None this push** — use the `run_lmip_smoke` biophysics-only config; carbon scheme decided later |
| Grid | **latlon first** |
| Resolution | **Coarse global ~2°** first (cheap to iterate); 0.5° later |
| Forcing dataset | **CRU-JRA** (TRENDY-native), already staged on glade/Derecho |
| N cycle | Deferred (carbon-only is a valid TRENDY config) |
| LULCC / E_LUC | Deferred |
| Transient CO₂ | Deferred (constant present-day CO₂ ok for biophysics; CO₂ only enters via canopy stomata) |

---

## 5. Development workflow

- **All development is local** (Mac, `/Users/linnia/Desktop/LEAP/legoESM`).
  The user provides an **example CRU-JRA met-forcing NetCDF** (and any other
  example files needed) so readers and unit tests run locally against real-shaped
  data plus small synthetic fixtures.
- **Testing and production runs happen on Derecho**, against the full CRU-JRA
  archive on glade.
- The build/validate seam: M1–M2 (reader, regrid, time disaggregation) are built
  and unit-tested locally; M3 (real-data global smoke at ~2°) runs on Derecho.

---

## 6. WS-1 workplan — CRU-JRA forcing → biophysics-only land (latlon)

**Design doctrine (CLAUDE.md, non-negotiable):** forcing is an **explicit
per-step `lax.scan` input** (the land analog of `SegmentForcing`), never closed
over — time-varying data in a closure freezes it and triggers recompiles. The
reader pre-stages forcing into a device pytree streamed through `scan`.

### M1 — CRU-JRA reader + regrid + variable map  ✅ DONE (codex review pending)
**New module:** `packages/land/legoesm/land/forcing/cru_jra.py` + `solar.py`
(mirror `packages/ocean/legoesm/ocean/forcing/jra55_do.py`). Tests:
`tests/land/forcing/test_cru_jra.py`, `test_solar.py` (14 pass, incl. an
opt-in real-data test that validated against the example 2023 files).
Ratchet-clean (inline-coeff). Confirmed the CLM dual time-stamp convention
(Solr start-stamped, TPQWL/Prec midpoint-stamped) holds in the real files.

- Open/cache CRU-JRA NetCDF (zarr cache like the ocean forcing path).
- Regrid 0.5° → ~2° latlon using existing `boundary_data/_internals.py`
  (no new regridder).
- Variable map → `AtmToSurface` (`packages/core/legoesm/core/coupling_fields.py`):

  | CRU-JRA | `AtmToSurface` field |
  |---|---|
  | tmp | `T_lowest` |
  | spfh | `q_lowest` |
  | pre | `precip_total` (+ `precip_snow` by T-threshold partition, reuse existing) |
  | pres | `p_lowest`, `p_surface` |
  | ugrd / vgrd | `u_lowest` / `v_lowest` |
  | dlwrf | `lw_down` |
  | dswrf | `sw_down` |
  | (derived) | `cos_zenith` (solar geometry), `rho_lowest` (from p,T,q via `thermo`), `co2_ppmv` (constant this push) |

- Units/constants only via `legoesm.thermo` / `legoesm.constants` (no literals).
- **Test:** direct unit test against a small synthetic CRU-JRA-shaped fixture
  (variable presence, units, regrid shape/conservation, `AtmToSurface` assembly).

### M2 — 6-hourly → model-dt disaggregation  ✅ DONE
`disaggregate_forcing(cols, lat, lon, model_times_s, ...)` in `cru_jra.py`
returns a stacked `AtmToSurface` (shape `(n_steps, ncol)`) = the explicit
per-step scan input (SegmentForcing doctrine), built on the host.  Default land
`dt`=1 h (6 sub-steps), `--dt 1800` → 30 min (12 sub-steps).  Tests:
`tests/land/forcing/test_disaggregate.py` (7 pass) — **SW energy conservation at
both dt** (per-interval sub-step mean == 6 h input, rtol 1e-9), precip mass
conservation, linear-interp correctness, diurnal shape (sunrise→0, noon peak),
night-window guard (no NaN).

CRU-JRA is 6-hourly (21600 s); at land `dt`=1 h → 6 sub-steps (30 min → 12).
**Per-variable rules (NOT blanket linear interpolation):**

| Variable | Nature | Rule |
|---|---|---|
| tmp, spfh, pres, u, v | instantaneous | **linear interpolation** between bracketing 6h values |
| dlwrf | interval-mean flux | linear interpolation |
| **dswrf** | interval-mean flux | **solar-zenith-weighted, conserving the 6h mean** |
| pre (precip) | interval-mean rate | **hold constant over the interval** (conservative) |

- **SW disaggregation:** distribute the 6-hourly mean SW across the 12 sub-steps
  weighted by `cos(zenith)`, **normalized so the sub-step mean reproduces the
  6-hourly mean** (energy-conserving). Reuse the existing solar geometry
  (`make_global_forcing` declination/zenith, consistent with
  `canopy/radiative_transfer.py`) — do not re-derive zenith.
  **Night-window guard:** if `Σcos(zenith)=0` over a window but the 6h mean > 0
  (terminator averaging), fall back gracefully (no divide-by-zero).
- **Convention flag:** CRU-JRA mixes instantaneous fields (tmp, spfh, pres, wind)
  with interval-mean fluxes (dswrf, dlwrf, pre). Pin the time-stamp convention
  explicitly in the reader and assert it — mis-stamping a mean flux skews the
  diurnal cycle and the surface energy/water budget.
- Implement the calendar logic: 1700–1920 reuse 1901–1920; 1921+ actual year.
  (Deferred until multi-year runs matter, but the disaggregator is structured for it.)
- **Pre-impl:** check whether the ocean OMIP/CORE forcing already has an
  SW-disaggregation helper to factor (`ocean/forcing/`, OMIP forcing path) before
  writing a new one.
- **Tests:** (a) **SW energy conservation** — 12-sub-step mean = 6h input mean;
  (b) **diurnal shape** — zero SW at night, peak near local noon;
  (c) precip mass conservation over an interval; (d) linear-interp correctness.

### M3 — Wire into the biophysics-only driver + validate  ✅ DONE (Derecho batch ready)
Local end-to-end synthetic-forcing smoke against the REAL 31 MB CLM5 surfdata
(`data/legoesm_surfdata_c250617.nc`) at 24×48 latlon PASSED (470 land cells,
T_sfc 266–349 K, sensible/latent fluxes finite, NetCDF written).  Derecho
turnkey artifacts under `scripts/cluster/derecho_lmip/`:

- `lmip_biophys.pbs` — PBS batch (single node, 8 CPU, 32 GB, 30 min wall).
  Overridable via `qsub -v` (YEAR, RESOLUTION, DT, N_STEPS, SURFACE_SCHEME…).
  Fails fast with a clear message if data isn't staged.
- `README.md` — one-time env setup + each-run recipe (`download_lmip_data.sh` →
  `qsub`).

Submit from a Derecho login node after `download_lmip_data.sh --year 1920`.
**New driver:** `scripts/run/run_lmip_biophys.py` (copied from `run_lmip_smoke.py`,
which is kept untouched as the synthetic-forcing test).  Same config (carbon
`none`, prescribed 1-yr-cycle LAI); default `dt`=1 h (`--dt 1800` for 30 min),
default latlon `--resolution 90` (~2°).  Forcing staged via the new
`cru_jra.stage_forcing` (load bracketing slices → NaN-aware regrid → disaggregate
→ stacked `AtmToSurface`) and streamed through `lax.scan`.
**Real-data fix:** CRU-JRA is land-only (ocean = NaN, ~57 %); the forcing regrid
MUST use `regrid_scalar_nan_aware` (plain IDW NaN-poisons every coastal land
cell).  Validated on the real 2023 Desktop files at ~2°: 7297 land cols, T
215–321 K, SW 0–1140 W/m², physical LW/q/wind/rho.  Tests:
`tests/land/integration/test_run_lmip_biophys.py` (end-to-end synthetic smoke) +
`test_disaggregate.py::test_stage_forcing_real_data_physical` (opt-in real data).
NEXT: run on Derecho against the glade CRU-JRA archive; codex review.

### M3 (original notes) — Wire into the biophysics-only driver + validate (Derecho)
- Swap synthetic `make_global_forcing` → real CRU-JRA forcing in the
  `run_lmip_smoke` `lax.scan` path, keeping the **exact** `run_lmip_smoke` config
  (`MultiLayerLandConfig`, prescribed 1-year-cycle LAI, `carbon="none"`), on the
  **latlon** grid at ~2°.
- Run a short global smoke on Derecho against the full CRU-JRA archive on glade.
- **Acceptance / validation:** no NaNs/Infs; physically sane sensible & latent heat
  fluxes; reasonable Bowen ratios by climate zone; soil temperature in a physical
  range; surface energy/water budgets close; spatial fields free of grid artifacts.
  Compare against the synthetic-forcing baseline to confirm the forcing path is the
  only change.
- **Codex adversarial review** (mandatory — new forcing module touching the AD/scan
  path); iterate per the iterate-with-codex loop until clean.

### WS-1 acceptance criteria
1. `cru_jra.py` reads the example NetCDF and produces a valid `AtmToSurface`
   pytree on a ~2° latlon grid (unit-tested locally).
2. 6h→30min disaggregation passes SW energy-conservation + diurnal-shape +
   precip-conservation tests.
3. A ~2° latlon biophysics-only global smoke runs end-to-end on Derecho under real
   CRU-JRA forcing with physically sane diagnostics.
4. Codex review clean.

---

## 7. Deferred workstreams (post-WS-1, to be re-scoped)

- **WS-2 — Production driver + transient CO₂ + spin-up.** A real global
  multi-year land driver (merge `run_lmip_smoke`'s `lax.scan` global path with
  `run_lmip`'s restart/checkpoint), transient CO₂ series, and a soil-C
  equilibrium spin-up protocol (cycle 1901–1920). Enables S0 (control) and S2-like
  (climate+CO₂ on fixed-1700 cover) → the natural sink S_LAND.
- **WS-3 — Carbon ON + NBP diagnostics.** Enable a carbon scheme (DALEC or an
  alternative TBD) and prognostic LAI; NBP/NEP aggregation to annual global +
  latitude bands. (Note: with carbon on, prognostic LAI should be enabled or the
  CO₂-fertilization signal is muted — moot while biophysics-only.)
- **WS-4 — LULCC → E_LUC (full S3).** Transient land cover from LUH2-GCB, carbon
  bookkeeping on cover change (clear/transfer vegetation pools, product/litter
  pools, wood harvest). Gives full S3 and S3−S2 = E_LUC. Hardest workstream;
  carbon conservation across transitions is the critical correctness concern.
- **Optional fidelity (TRENDY-valid without):** N cycle, multi-pool soil carbon
  (CENTURY/RothC), fire, dynamic vegetation.

---

## 7b. Data staging

Inputs live under the gitignored `data/` and are staged by
`scripts/data/download_lmip_data.sh` (modelled on `download_omip_data.sh`):

- **CRU-JRA forcing** → `data/crujra/` — **symlinked** from the glade copy
  `/glade/campaign/cesm/cesmdata/inputdata/atm/datm7/atm_forcing.datm7.CRUJRA.0.5d.c20260129/three_stream`.
  Files: `clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5.<stream>.<year>.nc`
  (no suffix; year 1920 available). The reader takes `prefix=`/`suffix=`; driver
  flags `--prefix`/`--suffix`/`--year` default to these.
- **Surfdata** (`legoesm_surfdata_c250617.nc`, full: soil + CLM5 PFT/LAI/cover) →
  `data/` — **downloaded from Zenodo** (record 21087964). Regridded to the model
  grid at run time by the driver; no build step.

`run_lmip_biophys.py` defaults point at this layout (`--surfdata
data/legoesm_surfdata_c250617.nc`, `--forcing-dir data/crujra`, `--year 1920`).

## 8. Open items / inputs needed

- [ ] **Example CRU-JRA met-forcing NetCDF** (user to provide) — for local reader
      development and unit fixtures.
- [ ] **glade path + file/variable naming convention** for the full CRU-JRA
      archive (e.g. per-variable per-year files vs. combined) — for M3 on Derecho.
- [ ] Any other example boundary files the reader/regrid needs locally.

---

## 9. References

- Sitch, S., et al. (2024). Trends and Drivers of Terrestrial Sources and Sinks of
  Carbon Dioxide: An Overview of the TRENDY Project. *Global Biogeochemical Cycles*,
  38, e2024GB008102. doi:10.1029/2024GB008102.
- Related branch `run_fluxnet`: point-scale PLUMBER2/AmeriFlux forcing reader
  (`src/legoesm/land/fluxnet_forcing.py`, old `src/` layout) — reference for
  variable mapping and `AtmToSurface` wiring.
