# AmeriFlux Land-Only Runner Plan (`scripts/fluxnet/`)

**Date:** 2026-06-03
**Status:** Scoped
**Branch:** `run_fluxnet`
**Scope:** A versioned, reproducible runner for land-only legoESM simulations at
AmeriFlux fluxtower sites with prescribed PLUMBER2 meteorology and tower-flux
evaluation. Adopts the design philosophy of
[`lesommer/legoESM-ocean-runners`](https://github.com/lesommer/legoESM-ocean-runners).
**Flagship site:** US-UMB (University of Michigan Biological Station, deciduous
broadleaf forest, ~45.56°N, −84.71°W).

---

## 1. Motivation

Running the land model today means hand-editing `scripts/run_lmip.py` (796 lines,
~40 argparse flags) and its built-in `_make_forcing` **synthetic** diurnal+seasonal
forcing. This is the land-side equivalent of the problem the ocean-runners repo
was built to solve for `run_omip.py` (3687 lines): no versioned configurations,
no provenance trail, no way to reproduce a past run.

We adopt the ocean-runners philosophy — **named templates, init≠run separation,
reproducibility tags, a data catalog, and one-command reproduction** — and add the
two axes that land-only tower simulations require and the ocean runners did not:

1. **Prescribed observational forcing.** Real gap-filled half-hourly tower
   meteorology, not synthetic. This replaces `_make_forcing` with a PLUMBER2
   NetCDF → `AtmToSurface` reader.
2. **Observational validation targets.** Tower-measured sensible/latent heat and
   carbon fluxes (`Qh`, `Qle`, `NEE`, `GPP`) are carried alongside the forcing
   and scored against the model — the PLUMBER2/FLUXNET land-evaluation protocol.
   (The ocean analogue was "validate against NEMO"; here it is the tower.)

### Decisions locked (2026-06-03)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo layout | **In-tree** under `scripts/fluxnet/` | Shares venv + CI; no external commit-pinning machinery. Provenance cost: must record working-tree dirty state in the tag (see §6). |
| Forcing/eval data product | **PLUMBER2** | De-facto land-eval standard. NetCDF, fully gap-filled half-hourly met, ships LAI, ships matching flux obs. Near-1:1 mapping to `AtmToSurface`. |
| v1 site scope | **One flagship site** (US-UMB) | Get the full pipeline green + validated end-to-end, then template-expand. Mirrors ocean's "one OMIP run-tested, rest init-only." |

---

## 2. The Six Pillars, Mapped to Land/Fluxnet

| Pillar | Ocean-runners mechanism | Land/fluxnet mechanism |
|--------|-------------------------|------------------------|
| Named templates | `templates/<category>/<name>.yaml` | `scripts/fluxnet/templates/fluxnet/<SITE>.yaml` over a shared `_base.yaml` |
| Init ≠ run | `init_experiment.py` → run dir + `run.sh`; `-o dot.path=val` overrides | same CLI; materializes a self-contained dir with frozen resolved config |
| Provenance | `experiment.tag` (INI): commit, patches, py/jax | `experiment.tag`: commit + **dirty flag**, py/jax, full resolved config, data checksums |
| Reproduction | `reproduce.py tag` → checkout → patches → verify data | `reproduce.py tag` → checkout (warn on dirty) → verify data → rerun frozen config |
| Data catalog | `data_catalog.yaml` + `fetch_data.py check/fetch` | same; PLUMBER2 site → URL + sha256 + local path + license |
| Patches + status | `patches/`, `project_status.md` | no `patches/` (in-tree); `project_status.md` per-site run-status table |

---

## 3. Grounding — What Exists Today

| Component | Location | Role here |
|-----------|----------|-----------|
| Single-point land driver (synthetic forcing) | `scripts/run_lmip.py` | **Reference example only** — a test script, likely to be rewritten/retired. Read it to understand the call pattern (`step_multilayer_land`, config assembly, diagnostic cadence) and lift the solar-zenith *formula*, but do **not** depend on or import from it. The fluxnet runner is written fresh. |
| Column-pure land step | `step_multilayer_land(state, forcing, config, U_MIN, dt, lat=, carbon_state=, doy=)` in `multilayer_land.py` | Unchanged entry point the runner calls. |
| Forcing contract | `AtmToSurface` (`coupler/coupling_fields.py:14`) — 15 fields | Target type the PLUMBER2 reader produces per timestep. |
| Soil texture presets, PFT lookup | `run_lmip.py:88-164` | Shared by both runners; factor out. |
| Prescribed-LAI path | `land_canopy_amip_plan.md` (`lai_forcing.interpolate_lai`) | PLUMBER2 ships LAI → a fluxnet site is the cleanest first consumer of that canopy work. |

**No FLUXNET/AmeriFlux handling exists anywhere** in `src/`, `scripts/`, `tests/`,
or `docs/` today (verified by grep, 2026-06-03). Clean slate.

---

## 4. Directory Layout

```
scripts/fluxnet/
├── README.md                  # the 6-pillar workflow (mirrors ocean-runners README)
├── project_status.md          # per-site run-status table (run-tested / init-only)
├── data_catalog.yaml          # PLUMBER2 site → URL + sha256 + local path + license
├── templates/
│   ├── _base.yaml             # shared defaults: spin-up cycles, dt, output, validation
│   └── fluxnet/
│       └── US-UMB.yaml        # flagship site (overrides _base)
├── runners/
│   ├── run_fluxnet_site.py    # driver: resolved config → run → diagnostics + experiment.tag
│   └── io.py                  # shared NetCDF append / restart / daily-accumulation (factored from run_lmip)
└── tools/
    ├── init_experiment.py     # template (+ -o overrides) → self-contained run dir + run.sh
    ├── fetch_data.py          # check / fetch PLUMBER2 files against data_catalog.yaml
    ├── reproduce.py           # experiment.tag → checkout commit → verify data → rerun
    └── validate_templates.py  # template schema check against legoESM config NamedTuples

src/legoesm/land/
└── fluxnet_forcing.py         # NEW (tested): PLUMBER2 NetCDF → AtmToSurface(t); reuses thermo/_shared

scripts/plot_fluxnet.py        # diagnostics + tower-obs overlay (extend plot_lmip.py, don't fork)
tests/land/test_fluxnet_forcing.py   # Tier-0 reader test
```

**Why the reader lives in `src/`, not `scripts/`.** CLAUDE.md requires every new
`.py` to carry a direct unit test and forbids re-deriving humidity/density/
saturation. The reader derives `rho_lowest` (virtual-T ideal gas →
`atmosphere.physics._shared`), partitions `precip_snow` by a temperature
threshold, and computes `cos_zenith` (solar geometry — formula referenced from
`run_lmip._make_forcing:211-223`, reimplemented here as the single source).
That is reusable, testable physics →
`src/legoesm/land/fluxnet_forcing.py`. The runner script is thin I/O glue.

---

## 5. PLUMBER2 → `AtmToSurface` Mapping

PLUMBER2 ships per-site NetCDF (`<SITE>_..._Met.nc`, `<SITE>_..._Flux.nc`) with
half-hourly, fully gap-filled fields. Most map with **no unit conversion**:

| `AtmToSurface` field | PLUMBER2 Met var | Transform |
|----------------------|------------------|-----------|
| `T_lowest` | `Tair` | none (already K) |
| `q_lowest` | `Qair` | none (already specific humidity, kg/kg) |
| `sw_down` | `SWdown` | none |
| `lw_down` | `LWdown` | none (PLUMBER2 always gap-fills) |
| `precip_total` | `Precip` | none (already kg/m²/s) |
| `u_lowest` | `Wind` | scalar wind → `u`; `v_lowest = 0` |
| `p_lowest`, `p_surface` | `PSurf` | both = PSurf (single tower level) |
| `co2_ppmv` | `CO2air` | none (fallback to config constant if absent) |
| `precip_snow` | — | **derive**: `where(Tair < T_snow_thresh, Precip, 0)`; mass-conserving |
| `rho_lowest` | — | **derive**: ideal gas from PSurf, Tair, Qair via `atmosphere.physics._shared` |
| `cos_zenith` | — | **derive**: solar geometry from site lat/lon + timestamp |
| `has_radiation`, `has_precipitation` | — | 1.0 (PLUMBER2 fully gap-filled) |

**Site metadata** from NetCDF global attributes — `latitude`, `longitude`,
`elevation`, `reference_height`, `IGBP_veg` — drives the template's PFT/texture
resolution and the zenith calculation. **Eval targets** (`Qh`, `Qle`, `NEE`,
`GPP`, `Qg` from `*_Flux.nc`) are loaded but **never fed to the model**; they are
scoring-only.

**LAI.** PLUMBER2 carries an `LAI` series → fed through the canopy plan's
`lai_forcing.interpolate_lai` path when the canopy work lands. v1 may run with
static PFT albedo if the canopy path is not yet wired; the reader still surfaces
`LAI(t)` so the consumer is ready.

---

## 6. SiteConfig Schema and Template Sketch

### `SiteConfig` (loaded from YAML, dot-overridable via `-o`)

```yaml
# scripts/fluxnet/templates/_base.yaml — shared defaults
site:
  id: null                      # required; set by site template (e.g. US-UMB)
  # lat/lon/elevation/reference_height/pft default to NetCDF attrs;
  # template may override explicitly.

forcing:
  product: plumber2             # only product in v1
  met_file: null                # resolved from data_catalog by site id
  snow_temp_threshold_K: 273.15 # = constants.T_freeze (rain/snow partition)
  co2_fallback_ppmv: 412.0      # used only if CO2air absent in file

soil:
  texture: loam                 # USDA class → run_lmip soil-texture presets
  n_layers: 10
  total_depth_m: 3.0
  growth_factor: 1.5

land:
  bulk_scheme: most
  snow_albedo_feedback: true
  carbon_scheme: none           # "none" until canopy/DALEC path is wired

time:
  dt_s: 1800.0                  # match PLUMBER2 half-hourly cadence

spinup:
  cycles: 15                    # loop the FULL multi-year forcing record N× to equilibrate
  convergence_dT_K: 0.05        # end-of-cycle soil-column ΔT threshold
  convergence_dtheta: 0.002     # end-of-cycle soil-column Δθ threshold
  output: false                 # suppress diagnostics during spin-up

evaluation:
  enabled: true
  flux_file: null               # resolved from data_catalog by site id
  metrics: [nme, corr, bias]    # PLUMBER2-style scores on Qh, Qle

output:
  dir: null                     # required at init time
  diag_interval_days: 1
  checkpoint_days: 100
```

```yaml
# scripts/fluxnet/templates/fluxnet/US-UMB.yaml — flagship
extends: ../_base.yaml
site:
  id: US-UMB                    # University of Michigan Biological Station
  pft: broadleaf_deciduous_temperate_tree   # IGBP DBF → CLM5 PFT
soil:
  texture: sandy_loam           # glacial sandy soils at UMBS
spinup:
  cycles: 20                    # deep-soil thermal memory in a cold-winter forest
```

`SiteConfig` is a legoesm-style `NamedTuple` (in `runners/run_fluxnet_site.py` or
`tools/`); the YAML loader resolves `extends`, applies `-o dot.path=value`
overrides, fills `lat/lon/.../pft` from NetCDF attributes where the template left
them null, and emits the **frozen resolved config** that is both run and embedded
in `experiment.tag`.

### `experiment.tag` (INI) for an in-tree runner

```ini
[legoESM]
commit = <git rev-parse HEAD>
dirty  = true|false              ; in-tree honesty cost — working tree may differ
tree_sha256 = <hash of tracked diffs>   ; only if dirty

[reproducibility]
python_version = 3.x
jax_version = 0.4.x
jax_enable_x64 = 1

[data]
US-UMB_Met.nc  = <sha256>
US-UMB_Flux.nc = <sha256>

[config]
resolved = <full post-override YAML, inline or sidecar path>
```

`reproduce.py` checks out `commit` (warns loudly if the original run was `dirty`),
re-verifies data checksums via `fetch_data.py`, and reruns the frozen resolved
config — no template lookup, so a later template edit can't silently change the
reproduction.

---

## 7. Spin-up + Evaluation Protocol

Standard PLUMBER2/FLUXNET land protocol, encoded in `_base.yaml` and
`runners/run_fluxnet_site.py`:

1. **Spin-up.** Cycle the **full multi-year forcing record** `spinup.cycles` times
   (always all available years, in chronological order, never a single
   representative year) to equilibrate soil moisture and temperature.
   Diagnostics suppressed.
2. **Convergence check.** Log end-of-cycle soil-column |ΔT| and |Δθ| vs the
   previous cycle against `convergence_dT_K` / `convergence_dtheta`. Report
   whether spin-up actually converged — never silently assume it did.
3. **Evaluation pass.** One final loop through the record at full diagnostic
   cadence; this pass is what gets scored.

---

## 8. Validation Tiers (Acceptance Gates)

| Tier | Gate | Pass criterion |
|------|------|----------------|
| **0 — reader** | M1 | `test_fluxnet_forcing.py`: round-trip a real `*_Met.nc`; every `AtmToSurface` field finite and in physical range; derived `q` consistent with stored `Qair`; `cos_zenith ∈ [0,1]`; `precip_snow ≤ precip_total`. |
| **1 — single-site smoke** | M2 | Flagship short run, no blowup; per-step surface energy + water closure (reuse closure checks from canopy plan §9). |
| **2 — spin-up convergence** | M3 | Soil T/θ drift between final cycles below threshold; convergence logged. |
| **3 — tower evaluation** | M4 | Modeled vs observed `Qh`/`Qle` diurnal + seasonal; PLUMBER2-style NME/corr/bias; sanity bar = beat trivial PLUMBER2 baseline predictors. |

---

## 9. Milestones

Each milestone is single-PR-sized and leaves the tree runnable.

### M0 — Scaffolding
- `scripts/fluxnet/` skeleton, `_base.yaml`, `SiteConfig` schema + YAML loader
  (resolve `extends`, apply `-o` dot-overrides), `validate_templates.py`.
- No data, no run yet.

### M1 — Forcing reader *(parallel with M0)*
- `src/legoesm/land/fluxnet_forcing.py`: PLUMBER2 `*_Met.nc` → `AtmToSurface(t)`.
- Reuses `legoesm.thermo`, `atmosphere.physics._shared` (density), and the
  solar-zenith code lifted from `run_lmip._make_forcing` (single source — have
  `run_lmip` import it after the move).
- **Validation:** Tier 0.

### M2 — Runner + flagship template
- `runners/run_fluxnet_site.py` consuming the resolved config.
- `runners/io.py`: NetCDF append / restart / daily accumulation, written fresh
  for this runner (use `run_lmip.py:285-454` only as a structural reference).
- `templates/fluxnet/US-UMB.yaml`; `data_catalog.yaml` entry + `fetch_data.py`.
- First green end-to-end run. **Validation:** Tier 1.

### M3 — Spin-up + provenance
- `spinup.py` forcing-cycling + convergence log.
- `experiment.tag` writer (commit + dirty flag, py/jax, resolved config, data
  checksums).
- `init_experiment.py` materializes self-contained run dirs + `run.sh`.
- **Validation:** Tier 2.

### M4 — Evaluation
- `evaluation.py`: tower-obs scoring (NME/corr/bias on `Qh`, `Qle`).
- Extend `plot_lmip.py` → `plot_fluxnet.py` with obs overlay.
- `reproduce.py`.
- **Validation:** Tier 3.

### M5 — Template expansion
- Add sites covering contrasting PFTs (grass, evergreen needleleaf, crop,
  tropical); `project_status.md` status table (run-tested / init-only).

```
M0 ──► M2 ──► M3 ──► M4 ──► M5
   ╲   ╱   scaffold→run→provenance→eval→sites
    M1 (reader, parallel)
```

---

## 10. Reuse / Non-Duplication (CLAUDE.md)

`run_lmip.py` is a reference example, **not** an import target — the fluxnet
runner is written fresh. The non-duplication obligation is against the **`src/`
library**, not against `run_lmip`:

- **Solar zenith**: implement once in `fluxnet_forcing.py` (formula referenced
  from `run_lmip._make_forcing:211-223`); that becomes the single source.
- **Soil-texture presets** and **PFT lookup**: if still needed, promote to a
  small `src/legoesm/land/` helper rather than copying `run_lmip`'s private dicts.
- **Diagnostic/NetCDF/restart**: written fresh in `runners/io.py`.
- **Humidity / density / saturation**: never re-derive — `legoesm.thermo` +
  `atmosphere.physics._shared`. No inline Tetens/Magnus, no `611.2*exp(...)`.
- **Constants**: `from legoesm import constants` — `T_freeze`, `sigma_sb`, `S_0`,
  etc. No `273.15` for the rain/snow threshold.
- **LAI**: route through the canopy plan's `lai_forcing.interpolate_lai`; the
  fluxnet site is its first consumer.

---

## 11. Open Questions

1. ~~Multi-year forcing cycling vs single-year repeat.~~ **Resolved 2026-06-03:**
   spin-up always cycles the full multi-year record in chronological order;
   never a single representative year.
2. **CO2 forcing.** Use PLUMBER2 `CO2air` (observed, time-varying) or a fixed
   `co2_fallback_ppmv`? Default to observed when present.
3. **Canopy coupling readiness.** Whether v1 US-UMB runs with the new multi-layer
   canopy (`land_canopy_amip_plan.md`) or static-PFT albedo depends on canopy M4
   landing. The reader surfaces `LAI(t)` either way.
4. **Reference-height adjustment.** Tower `reference_height` differs from the
   model's lowest-level assumption; decide whether to log-adjust wind/T/q to a
   common height or accept the tower height as the lowest model level. Default:
   accept tower height (single-level land forcing convention).

---

## 12. Progress Log / Session Handoff

### Status as of 2026-06-03

| Milestone | State |
|-----------|-------|
| **M1 — Forcing reader** | ✅ **Done + tested** (10 Tier-0 tests passing) |
| M0 — Scaffolding (templates, `SiteConfig`, `validate_templates.py`) | Not started |
| M2 — Runner + US-UMB template | Not started |
| M3–M5 | Not started |

Note: M1 was completed ahead of M0 because the forcing reader is the
foundational, self-contained physics piece and needs no scaffolding to validate.

### Files created this session

| File | What it is |
|------|-----------|
| `docs/run_fluxnet_plan.md` | This plan (the source of truth). |
| `src/legoesm/land/fluxnet_forcing.py` | **M1 deliverable.** PLUMBER2 reader: loader + source-agnostic derivations + `AtmToSurface` builder + PLUMBER2 adapter + slice helper. |
| `tests/land/unit/test_fluxnet_forcing.py` | Tier-0 test (hermetic synthetic-NetCDF fixture, no download). |

### `fluxnet_forcing.py` public API (built, tested)

- `Plumber2ForcingConfig` — variable names + unit knobs (`tair_offset`, `precip_scale`, `co2_fallback_ppmv`, `time_is_local`).
- `Plumber2Site`, `Plumber2Forcing` (has `.n_time`, `.n_years`).
- `load_plumber2(met_path, flux_path=None, config=...)` → `Plumber2Forcing` (eager full-record load; ~40 MB for 15 yr; file closed after read).
- **Source-agnostic derivations** (plain arrays in): `partition_precip_snow`, `air_density` (→ `_shared.compute_rho`), `solar_zenith` (→ `radiation.solar.cos_zenith_angle`).
- `build_atm_to_surface_series(...)` → `AtmToSurface` with `(n_time, 1)` fields (scan-friendly).
- `plumber2_to_atm_series(data)` — thin PLUMBER2 shim.
- `forcing_at_index(series, i)` — `(n_time,1)`→`(1,)` slice; spin-up uses `i = step % n_time`.

### How to run the test (env quirks matter)

There is **no `.venv`**; this machine uses conda `base` python, and `legoesm`
is **not pip-installed** — run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 python3 -m pytest tests/land/unit/test_fluxnet_forcing.py -v
```

**Known gotcha — pre-existing circular import.** Importing `legoesm.land.*`
*before* the coupler triggers a cycle
(`land.config → surface_scheme → coupler → atmosphere.physics → driver →
coupled_config → land.config`). It is **not** caused by our code. Workarounds:
under pytest, `tests/conftest.py` initializes the package first (so tests pass);
in ad-hoc scripts, import `legoesm.coupler.coupling_fields` before
`legoesm.land.*` (this is what `run_lmip.py` does). A proper fix (defer the
coupler import in `land/__init__.py`) is out of scope for this feature.

### Decisions locked

- In-tree runner under `scripts/fluxnet/` (not a separate repo).
- PLUMBER2 forcing/eval product; US-UMB flagship site.
- Spin-up cycles the **full multi-year record** in chronological order (never a single year).
- `run_lmip.py` is a **reference example only** — do not import from it; the runner is written fresh.
- Architecture: **source-specific readers + shared derivation layer** (the house pattern in `forcing/`); when source #2 (ERA5 / coupler history) lands, lift the derivation free-functions into `forcing/` and add a `forcing.source=...` dispatch with raise-on-unknown.

### Next step (recommended)

Build the **M2 runner run-loop** (cycle forcing → `step_multilayer_land` →
diagnostics) and validate it on the synthetic `build_atm_to_surface_series`
output — **no data download needed**; the real US-UMB file drops in afterward.
Alternatives: M0 scaffolding (templates/`SiteConfig`), or data-first
(`data_catalog.yaml` + `fetch_data.py`, which needs the US-UMB PLUMBER2 file
location). The runner must call `step_multilayer_land(state, forcing, config,
U_MIN, dt, lat=, doy=)` — read `run_lmip.py:507-630` for the call pattern (but
write fresh).
