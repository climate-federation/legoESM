# Land Canopy + Global AMIP Upgrade Plan

**Date:** 2026-05-20 (decisions locked 2026-05-22; **re-baselined 2026-06-12** — see §0)
**Status:** Re-baselined. Canopy *physics* is done in two interchangeable
flavors (DifferBESS two-leaf, in-tree; CLM-ML plugin, optional dependency).
The vendoring approach (original M2) is **superseded**. The remaining work to
hit the AMIP albedo/energy-budget goal is **M5 (global surface-data loader) +
M6 (AMIP wiring + spin-up)**, both of which are *surface-scheme-agnostic*.
**Scope:** Prescribed-LAI, spatially-resolved land surface for AMIP, fed by a
global surface-data loader that drives **any** of the three surface schemes
(Slab / DifferBESS two-leaf / CLM-ML) behind the shared `SurfaceFluxOutput`
contract.
**Canopy implementations (both already built, sharing `SurfaceFluxOutput`):**
1. **DifferBESS two-leaf** — `src/legoesm/land/{canopy,surface_scheme}/`, in-tree,
   merged on `land/stable`. Sunlit/shaded EB, Farquhar C3/C4, prognostic
   LAI-dependent canopy albedo. Default target for AMIP v1 (no external dep).
2. **CLM-ML plugin** — `src/legoesm/land/canopy/clm_ml_interface.py` on
   `origin/aya/land`, an adapter to the *external* optional
   [AyaLahlou/clm-ml-jax](https://github.com/AyaLahlou/clm-ml-jax) (BSD-3)
   package (`pip install ".[canopy]"`). Richer multi-layer physics; unmerged.

---

## 0. Status update (2026-06-12) — re-baseline

This plan was written (2026-05-20) assuming the canopy did not exist yet and
that *vendoring* CLM-ML in-tree (M2) was the largest milestone. Both
assumptions are now obsolete:

- **Canopy physics is done — twice.** A DifferBESS-derived two-leaf canopy
  (sunlit/shaded EB, Farquhar C3/C4, Monin-Obukhov, Newton+Picard closure,
  prognostic LAI) was built and merged on `land/stable` across the 6-phase
  "Canopy refactor" (2026-04-11 → 06-03). Separately, `origin/aya/land` added
  a CLM-ML canopy *plugin* (`clm_ml_interface.py`, 757 LOC). Both produce the
  shared `SurfaceFluxOutput` and are dispatched in `step_multilayer_land` by
  `isinstance` on the surface-scheme config — so they are interchangeable.
- **Vendoring (original M2) is superseded.** The clm-ml work pivoted from
  "copy 25 modules into `src/legoesm/land/multilayer_canopy/`" to "depend on
  the external `clm_ml_jax` package via a thin lazy-imported adapter." There is
  **no** `src/legoesm/land/multilayer_canopy/` and there should not be one.
- **The real gap is spatial data + driver wiring, not physics.** Today
  `scripts/run_amip.py` runs **slab-land with `dynamic_albedo=False`**; land
  albedo is a 3-band zonal proxy (0.15/0.20/0.25) blended with snow. Neither
  canopy is reachable from AMIP, and **no branch** has a global PFT/LAI/
  background-albedo loader. That loader (**M5**) plus AMIP wiring (**M6**) are
  the entire remaining path to spatially-accurate albedo + a closed surface
  energy budget.

**Design directive for the re-baseline (locked 2026-06-12):** everything built
in M5 must be **surface-scheme-agnostic** — it loads geophysical surface data
(PFT fractions, monthly LAI, VIS/NIR background albedo, canopy height) and maps
it onto the model grid *without* assuming which scheme consumes it. Slab,
DifferBESS two-leaf, and CLM-ML must all be drivable from the same loaded
fields. Scheme selection becomes a **runtime config choice** in M6, not a data
or loader choice. See revised §10 (M5/M6) and §12.

---

## 1. Motivation

The current `step_multilayer_land` is physically sophisticated below ground
(Richards + Johansen thermal + Farquhar stomata) but has no above-ground canopy
representation. Vegetation influences the simulation only through four static
PFT scalars: albedo, emissivity, z0, and root depth. The consequences for a
global AMIP run:

- **Albedo bias**: deciduous forests keep their summer albedo year-round (no leaf-off cycle)
- **Overestimated bare-soil evaporation**: all precipitation hits the soil directly; no canopy interception pool that re-evaporates rapidly
- **Underestimated stomatal regulation**: `stomata.enabled = False` by default, so transpiration is governed by moisture stress alone with no photosynthetic coupling
- **LAI nowhere**: there is no path to get seasonally varying LAI into the model without enabling the full DALEC carbon cycle

Rather than build a big-leaf canopy from scratch, we **vendor a JAX port of
Bonan's CLM-ml_v2.CHATS multi-layer canopy** (clm-ml-jax). This buys us, in
a single integration step, the full canopy biogeophysics stack used in
state-of-the-art ESM land components: multi-layer solar + longwave radiative
transfer, sunlit/shaded leaf-level energy balance, roughness-sublayer
turbulence (Harman-Finnigan), coupled Farquhar+Ball-Berry/Medlyn
photosynthesis-stomata, multi-layer canopy water storage, and plant
hydraulics. Prescribed LAI drives the leaf-area-density profile.

This is a deliberate scope inversion vs the original "as simply as possible"
plan: the *integration* work is bigger (vendor + rewire + adapter), but the
*physics* is research-grade from day one and there is no v1/v2 canopy upgrade
on the roadmap.

---

## 2. Scope — What Is and Is Not Included

### In scope

| Feature | Rationale |
|---------|-----------|
| Vendor `clm-ml-jax` under `src/legoesm/land/multilayer_canopy/` | Foundation; full Bonan multi-layer canopy biogeophysics |
| Rewire vendored constants to `legoesm.constants` | CLAUDE.md compliance; eliminates `MLclm_varcon`-style duplicates |
| Rewire vendored saturation thermodynamics to `legoesm.thermo` | CLAUDE.md compliance; no inline Tetens/Magnus |
| Replace vendored namelist driver with legoesm config NamedTuples | Plays nicely with existing config schema and `LandSurfaceParams` |
| Multi-layer canopy state (per-layer T_leaf, q_leaf, canopy_water profile, sunlit/shaded) | Comes with CLM-ML; replaces single-leaf state |
| Multi-layer solar + longwave RT, roughness-sublayer turbulence | Comes with CLM-ML |
| Coupled Farquhar + Ball-Berry/Medlyn photosynthesis-stomata | Comes with CLM-ML (replaces legoesm's existing `carbon/stomata.py` for the canopy path) |
| Plant hydraulics | Comes with CLM-ML |
| Prescribed monthly LAI climatology (global × PFT) | Drives leaf-area density profile in canopy |
| Adapter: `AtmToSurface` ↔ CLM-ML inputs ↔ `TileResponse` | New module; only way to plug vendored canopy into the legoesm coupler |
| Extended PFT table (leaf clumping, per-band leaf ρ/τ, leaf-N profile, hydraulic traits) | CLM-ML needs these; current 12-column `_CLM5_PFT_TABLE_RAW` is insufficient |
| Global PFT map reader + LAI interpolator | Infrastructure for spatial heterogeneity (tile-ready, dominant-PFT in v1) |
| AMIP driver updates: pass LAI forcing, cold-start canopy state, run 20-yr coupled spin-up | Wire new physics into `run_amip.py` |

### Explicitly out of scope

| Feature | Why deferred |
|---------|--------------|
| Sub-grid topographic wetness / TOPMODEL | Requires DEM-derived wetness index |
| Soil freeze-thaw phase change (ice lens) | Important at high latitudes but complex |
| Nutrient cycling (N/P limitation on GPP) | Carbon-only scope; CLM-ML's leaf-N profile is prescribed, not prognostic |
| Dynamic vegetation / prognostic phenology | LAI is prescribed, not predicted |
| Full DALEC carbon cycle activation | Independent decision; prescribed LAI doesn't require it |
| Offline land spin-up (`run_lmip_global.py`) | Superseded — using 20-yr coupled spin-up in `run_amip.py` |
| Multi-PFT tile blending per column | Deferred to v2; v1 uses dominant-PFT but designs tile-ready |
| CLM-ML's WUE-optimized stomatal path (`gs_type=2`) | Default to Ball-Berry (`gs_type=1`); WUE path adds bisection cost and is rarely used in operational CLM |
| Adding our own LMIP-CHATS regression suite | Trust upstream `tests/fortran_validation/test_golden_jax.py` as the canopy regression gate (locked 2026-05-22) |
| Big-leaf canopy fallback (`canopy.scheme = "bigleaf"`) | Single dispatch path: CLM-ML *is* the canopy. No big-leaf alternative |
| Big-leaf `CanopyConfig`, Beer's-law `canopy_land_albedo`, single-store `canopy_water` field | Replaced by multi-layer state and CLM-ML's own RT |

---

## 3. Current State Summary

### What already works

| Component | Location | Notes |
|-----------|----------|-------|
| Multi-layer soil thermal | `soil_thermal.py` | Johansen conductivity, θ-dependent |
| Richards unsaturated flow | `richards.py` | Celia mixed-form, Picard iteration |
| Root water uptake sink | `multilayer_land.py:304–354` | Exponential profile, per-layer extraction |
| Snow budget | `snow_budget.py` | Energy-limited melt, albedo aging |
| MOST bulk fluxes | `coupler/bulk_flux.py` | Full Monin-Obukhov |
| Farquhar photosynthesis | `carbon/stomata.py` | C3 with T-response, Rubisco + RuBP limited |
| Ball-Berry / Medlyn gs | `carbon/stomata.py` | Fixed-point coupled solver (n=5 iter) |
| Jarvis gs | `carbon/stomata.py` | PAR × T × VPD × soil stress |
| Snow-veg albedo blending | `surface_albedo.py` | Age-decaying snow, latitude-dependent veg α |
| 17-PFT CLM5 table | `surface_params.py` | albedo, emissivity, z0, root_depth, θ_wp, θ_fc |

### What is absent or non-functional

| Gap | Impact |
|-----|--------|
| No canopy water store | All precip infiltrates; interception evaporation missing |
| LAI only from DALEC carbon (`C_fol / LCMA`) | Stomata disabled; albedo static; no seasonal signal |
| No prescribed LAI path | Cannot drive stomatal or albedo physics without activating DALEC |
| `stomata.enabled = False` default | Transpiration = moisture stress only (no A_net / gs coupling) |
| Albedo ignores LAI | Deciduous trees: identical albedo in January and July |
| `f_veg` is an emergent quantity | Vegetation cover proxy is `sum(root_frac × beta_root)`, not LAI-driven |

---

## 4. New Physics — Design

The canopy physics is the vendored CLM-ML port. This section describes what
that gives us and how it slots into the existing legoesm column step.

### 4.1 Prescribed LAI

LAI(t, x) is read from an external monthly climatology (CLM5 `surfdata` is
the preferred source — already PFT-resolved with `MONTHLY_LAI`), linearly
interpolated to the current model day, and passed as an explicit traced
argument to the land step. CLM-ML distributes the column LAI across its
`nlevmlcan ≈ 40` canopy layers using its built-in leaf-area-density profile;
the legoesm code only needs to provide the column-integrated LAI.

**LAI dispatch priority inside `step_multilayer_land`:**
1. Explicit `lai` argument (shape `(ncol,)`) — from global forcing pipeline
2. DALEC carbon state: `lai = carbon_state.C_fol / config.carbon.LCMA` — when differland active
3. Config default: `lai = jnp.full(ncol, config.lai_default)` — single-point runs without forcing

**New module:** `src/legoesm/land/lai_forcing.py`

```python
class LAIForcingConfig(NamedTuple):
    path: str
    varname: str = "MONTHLY_LAI"     # CLM5 surfdata default
    smooth_days: float = 15.0

def interpolate_lai(lai_monthly: jnp.ndarray, doy: float) -> jnp.ndarray:
    """Linear interpolation between monthly means. Returns (ncol,) for v1,
    (ncol, ntile) when tiling lands in v2."""
```

The global per-PFT loader lives in `src/legoesm/land/global_surface_data.py`
(see §8.2) — it is responsible for the disk → grid step; this module only
does temporal interpolation.

---

### 4.2 Multi-layer canopy biogeophysics (CLM-ML)

The vendored canopy provides, in a single integration step:

| Component | Vendored module | Replaces |
|-----------|-----------------|----------|
| Multi-layer solar RT (direct + diffuse, VIS + NIR, sunlit/shaded) | `MLSolarRadiationMod.py` | Static `albedo_land`, would-be Beer's-law albedo |
| Multi-layer longwave RT | `MLLongwaveRadiationMod.py` | Single emissivity surface LW |
| Roughness-sublayer turbulence (Harman-Finnigan) | `MLCanopyTurbulenceMod.py` | Single-level MOST flux (which still runs for soil/snow patch) |
| Per-layer leaf boundary-layer conductance | `MLLeafBoundaryLayerMod.py` | (no analogue) |
| Per-layer leaf energy balance (sunlit + shaded T_leaf, q_leaf) | `MLLeafFluxesMod.py`, `MLCanopyFluxesMod.py` | Single-leaf `T_can ≡ T_soil[0]` assumption |
| Coupled Farquhar + Ball-Berry/Medlyn stomata | `MLLeafPhotosynthesisMod.py` | `legoesm.land.stomata` (kept for non-canopy path; bypassed for canopy) |
| Multi-layer canopy water (interception, drip, dew) | `MLCanopyWaterMod.py` | Would-be single `canopy_water` store |
| Plant hydraulics (ψ_stem, ψ_leaf, xylem cavitation) | `MLPlantHydraulicsMod.py` | (no analogue) |
| Canopy-aware soil temperature top BC | `MLSoilTemperatureMod.py` | Used as canopy-bottom flux into legoesm's `soil_thermal.py` |
| Runge-Kutta sub-stepping inside the canopy time step | `MLRungeKuttaMod.py` | (no analogue) |

**Coupling pattern.** CLM-ML runs as the *upper boundary condition* for the
existing legoesm soil column. The canopy step ingests `AtmToSurface` plus the
top-of-soil state, runs its full multi-layer solve (including its own
sub-stepping), and emits:

- net surface energy fluxes (SH, LH, G_top) for legoesm's coupler
- top-of-soil heat and water fluxes for legoesm's `soil_thermal.py` and `richards.py`
- column-integrated GPP and transpiration for diagnostics

The existing legoesm soil column (Richards + Johansen thermal) continues to
run unchanged below the canopy. This minimizes the surface area where vendored
code touches legoesm code paths.

---

### 4.3 Differentiability posture

The vendored code uses `jit`, `lax.scan`, `lax.fori_loop`, and
`jax.checkpoint` throughout, with IFT-based gradient correction on the
iterative photosynthesis solve (`_ci_solver_scan_ift`) and the WUE bisection.
It is *designed* to be `jax.grad`-safe but is **not advertised as validated
end-to-end** through the full canopy step. We treat full-column AD as a
follow-on goal: differentiable training of canopy parameters is out of scope
for v1 AMIP. We will validate AD per-module as part of M2 acceptance.

---

### 4.4 PFT table extension

The current 12-column `_CLM5_PFT_TABLE_RAW` is insufficient for CLM-ML, which
needs additional canopy parameters. We extend the table — or replace it with
CLM-ML's `MLpftconMod.py` table after rewiring — to add at minimum:

| Parameter | Purpose |
|-----------|---------|
| `leaf_clumping_index` | Modifies effective LAI in multi-layer RT |
| `rho_leaf_vis`, `rho_leaf_nir`, `tau_leaf_vis`, `tau_leaf_nir` | Per-band leaf reflectance and transmittance |
| `chi_leaf` | Leaf-angle distribution (Ross-Goudriaan) |
| `dleaf` | Characteristic leaf dimension (boundary-layer conductance) |
| `Vc_max25_top`, `kn` | Leaf-nitrogen profile (Vc_max decays with canopy depth) |
| `psi_*`, `K_max_*`, vulnerability curve parameters | Plant hydraulics |

The simplest path is to **vendor CLM-ML's `MLpftconMod.py` PFT table as-is**
and have `LandSurfaceParams` look up its values from there, rather than
mechanically merging 20+ canopy columns into our existing 12-column table.
The existing 12-column table remains in use for the non-canopy soil and
hydrology parameters.

---

## 5. New Modules and Files

> **HISTORICAL (superseded 2026-06-12, see §0/§10).** This section describes
> the abandoned in-tree vendoring layout. There is **no**
> `src/legoesm/land/multilayer_canopy/` subpackage and no
> `multilayer_canopy_adapter.py` / `multilayer_canopy_config.py`. The CLM-ML
> path is the single adapter `canopy/clm_ml_interface.py` (`origin/aya/land`)
> against the external optional `clm_ml_jax` package. The only **new** module
> the live plan still calls for is the scheme-agnostic **M5 loader**
> `src/legoesm/land/global_surface_data.py` (see §8.2 and revised §10/M5).
> Retained below for the adapter/PFT/config design reasoning, which still
> informs the loader's per-scheme mappers.

### Vendored subpackage

```
src/legoesm/land/multilayer_canopy/
├── __init__.py                          (legoesm-facing re-exports)
├── ML_pftcon.py                         (was MLpftconMod.py — PFT canopy traits)
├── ML_canopy_state.py                   (was MLCanopyFluxesType.py — per-layer state container)
├── ML_init_vertical.py                  (was MLinitVerticalMod.py — canopy layering, LAI profile)
├── ML_canopy_fluxes.py                  (was MLCanopyFluxesMod.py — main entry point)
├── ML_solar_radiation.py                (was MLSolarRadiationMod.py)
├── ML_longwave_radiation.py             (was MLLongwaveRadiationMod.py)
├── ML_canopy_turbulence.py              (was MLCanopyTurbulenceMod.py)
├── ML_leaf_boundary_layer.py            (was MLLeafBoundaryLayerMod.py)
├── ML_leaf_photosynthesis.py            (was MLLeafPhotosynthesisMod.py)
├── ML_canopy_water.py                   (was MLCanopyWaterMod.py)
├── ML_plant_hydraulics.py               (was MLPlantHydraulicsMod.py)
├── ML_flux_profile_solution.py          (was MLFluxProfileSolutionMod.py)
├── ML_runge_kutta.py                    (was MLRungeKuttaMod.py)
├── ML_canopy_nitrogen_profile.py        (was MLCanopyNitrogenProfileMod.py)
├── ML_leaf_heat_capacity.py             (was MLLeafHeatCapacityMod.py)
├── ML_water_vapor.py                    (was MLWaterVaporMod.py — REWIRE: delegate to legoesm.thermo)
├── ML_math_tools.py                     (was MLMathToolsMod.py)
├── ML_soil_fluxes.py                    (was MLSoilFluxesMod.py)
├── ML_soil_temperature.py               (was MLSoilTemperatureMod.py — used only as canopy-bottom flux source)
├── ML_clm_varctl.py                     (config flags — collapse into legoesm config NamedTuples)
└── rsl_lookup_tables/                   (Harman-Finnigan RSL tables; binary data, unchanged)
```

Files explicitly **dropped** from the vendoring:
- `MLclm_varcon.py` — replaced by imports from `legoesm.constants`
- `MLGetAtmForcingMod.py` — replaced by the adapter (§5.2)
- `MLclm_varpar.py` — folded into `MultiLayerCanopyConfig`
- Everything under upstream's `offline_driver/`, `offline_executable/`, `clm_share/`, `clm_src_*`, `cime_src_share_util/` — not needed; legoesm owns the driver, time stepping, and shared utilities
- Upstream `tests/` are vendored separately under `tests/land/multilayer_canopy/`

### New legoesm-side modules

| File | Purpose |
|------|---------|
| `src/legoesm/land/multilayer_canopy_adapter.py` | `to_canopy_inputs(AtmToSurface, MultiLayerLandState, lai, params) -> ...`; `from_canopy_outputs(...) -> TileResponse + soil-top flux`; thread `lai(t)` to the canopy LAI profile |
| `src/legoesm/land/multilayer_canopy_config.py` | `MultiLayerCanopyConfig` NamedTuple — collapses upstream namelist into legoesm-style config; documents every flag |
| `src/legoesm/land/lai_forcing.py` | `interpolate_lai(monthly_lai, doy)`, `LAIForcingConfig` |
| `src/legoesm/land/global_surface_data.py` | PFT-fraction loader, per-PFT monthly LAI loader, `dominant_pft_collapse` (v1-only helper) — preserves PFT axis on disk |

### Modified legoesm modules

| File | Changes |
|------|---------|
| `src/legoesm/land/state.py` | Add `canopy: MLCanopyState` field to `MultiLayerLandState` — nested NamedTuple with per-layer T_leaf, q_leaf, canopy_water profile (sunlit/shaded), plant-hydraulic ψ |
| `src/legoesm/land/config.py` | Add `lai_default`, `canopy: MultiLayerCanopyConfig` field to `MultiLayerLandConfig`; keep `stomata` for non-canopy path |
| `src/legoesm/land/multilayer_land.py` | Add `lai` argument; replace surface flux + albedo + transpiration block with adapter call into vendored canopy; soil thermal/Richards run as before below canopy |
| `src/legoesm/land/surface_params.py` | Add canopy-table lookup path; existing 12-column table stays for soil/hydrology |

### New scripts

No new top-level scripts. Offline global spin-up is *not* in scope — `run_amip.py`
runs a 20-yr coupled spin-up before the AMIP analysis window opens (see §8). The
existing single-point `scripts/run_lmip.py` remains the regression workhorse for
M1–M4 validation.

### New tests

| File | What it covers |
|------|---------------|
| `tests/land/test_canopy.py` | Unit tests for `step_canopy_water_budget`: mass conservation, W_can bounds, dry-canopy throughfall = P_rain |
| `tests/land/test_lai_forcing.py` | Interpolation correctness, edge cases (doy=0, 365, leap year) |
| `tests/land/test_multilayer_canopy.py` | Integration test: LAI path in `step_multilayer_land`, energy balance, water balance |

---

## 6. Changes to step_multilayer_land — Revised Physics Sequence

```
NEW SIGNATURE:
step_multilayer_land(state, forcing, config, U_min, dt,
                     lat=None, carbon_state=None, doy=0.0,
                     land_params=None, lai=None)

STEP 0 (NEW): Resolve column LAI
  lai_col = lai             (explicit arg)
         or C_fol / LCMA   (differland)
         or config.lai_default (fallback)

STEP 1 (NEW): Build canopy inputs via adapter
  canopy_in = to_canopy_inputs(
      atm=forcing,
      soil_top_T=state.T_soil[..., 0],
      soil_top_theta=state.theta_soil[..., 0],
      snow_depth=state.snow_depth,
      lai_col=lai_col,
      params=canopy_params,                # from LandSurfaceParams + CLM-ML PFT table
      config=config.canopy,
  )

STEP 2 (NEW): Run vendored multi-layer canopy step
  canopy_state_new, canopy_out = ml_canopy_fluxes(
      canopy_state=state.canopy,
      canopy_in=canopy_in,
      config=config.canopy,
      dt=dt,
  )
  # canopy_out contains: SH, LH, G_top (canopy→soil heat flux),
  # P_through (throughfall + drip), E_transp, gpp_col,
  # albedo_vis, albedo_nir (LAI-aware, sunlit/shaded mixed),
  # net_radiation breakdown.

STEP 3 (MOD): Snow budget — driven by canopy_out throughfall + sublimation,
              not bulk-flux surface vapor exchange.

STEP 4 (MOD): Richards top BC = canopy_out.P_through + meltwater
              (replaces what was raw precip - bare-soil evap)
              Root sink uses canopy_out.E_transp_profile.

STEP 5: Richards solver                                  (unchanged)

STEP 6 (MOD): Soil thermal top BC = canopy_out.G_top      (replaces internal
              surface-energy-balance ground flux calculation)

STEP 7: Soil thermal                                     (unchanged)

STEP 8 (MOD): TileResponse fields from canopy_out for the canopy patch;
              snow/bare-soil patches still use the legoesm coupler path.
              New state: state.canopy = canopy_state_new
```

The bulk-flux block (`coupler.bulk_flux.compute_most_fluxes`) is **no longer
called for the vegetated fraction**; CLM-ML's roughness-sublayer turbulence
replaces it. Bulk flux still runs for the snow and bare-soil patches.

---

## 7. Config Schema Changes

### `MultiLayerLandConfig` additions

```python
class MultiLayerLandConfig(NamedTuple):
    # ... all existing fields ...
    lai_default: float = 1.0
    canopy: MultiLayerCanopyConfig = MultiLayerCanopyConfig()
    # stomata stays for the (legacy / non-canopy) path; canopy stomata is
    # handled inside the vendored Farquhar+BB solver.
```

### `MultiLayerCanopyConfig` (new)

```python
class MultiLayerCanopyConfig(NamedTuple):
    # Vertical discretization
    nlevmlcan: int = 40              # Canopy layers
    nlevsoi_can: int = 1             # Soil layers the canopy module is aware of
                                      # (only the top layer; deep soil is legoesm's)
    # Stomatal model selection (upstream gs_type)
    gs_scheme: str = "ball_berry"    # "ball_berry" | "medlyn"
                                      # "wue" (WUE-optimized) deferred — see §2
    # Photosynthesis model
    is_c3_default: bool = True       # PFT table overrides per-column
    # Radiation
    use_clumping: bool = True
    # Sub-stepping inside the canopy step
    n_substeps: int = 5              # RK sub-steps per legoesm dt
    # Plant hydraulics
    plant_hydraulics_on: bool = True
    # Differentiability — use IFT correction on iterative solves
    use_ift_grad: bool = True
    # Tolerances for canopy iterative solves
    ci_solver_n_iter: int = 40       # Fixed scan length for Ci secant solve
    rsl_table_path: str | None = None  # Defaults to vendored RSL table
```

### Upstream flag mapping

Upstream's namelist exposes ~30 flags via `MLclm_varctl.py`. The integration
work is to fold the live ones into `MultiLayerCanopyConfig` above; the rest
(diagnostic-only, restart options, history-output knobs) are dropped or
replaced by legoesm's standard `TileResponse` and restart pathways. Document
the full flag-by-flag mapping in `multilayer_canopy_config.py` docstring.

### `MultiLayerLandState` extension

```python
class MultiLayerLandState(NamedTuple):
    # Existing fields ...
    T_soil: jax.Array
    psi_soil: jax.Array
    theta_soil: jax.Array
    runoff_surface: jax.Array
    runoff_subsurface: jax.Array
    snow_depth: jax.Array
    snow_age: jax.Array
    # New canopy state — nested for clarity and tile-readiness
    canopy: MLCanopyState
```

```python
class MLCanopyState(NamedTuple):
    """Per-column-per-layer canopy state. Shape (ncol, nlevmlcan, 2)
    where the trailing axis is (sunlit, shaded). In v2 with tiling,
    leading axis becomes (ncol, ntile, nlevmlcan, 2)."""
    T_leaf: jax.Array        # [K]
    q_leaf: jax.Array        # [kg/kg]
    canopy_water: jax.Array  # [kg/m^2] per layer
    psi_leaf: jax.Array      # [m]  plant-hydraulic leaf potential
    psi_stem: jax.Array      # [m]
    # Diagnostics carried forward across sub-steps
    A_net: jax.Array         # [µmol/m^2/s] per layer per sun-class
    gs: jax.Array            # [mol/m^2/s] per layer per sun-class
```

### PFT table

CLM-ML's `MLpftconMod.py` vendored as `ML_pftcon.py`. Adds canopy traits
listed in §4.4. The existing 12-column `_CLM5_PFT_TABLE_RAW` is kept
unchanged for soil and hydrology parameters; canopy params come from the
vendored CLM-ML table. `LandSurfaceParams` gains a `canopy_pft_idx` field
linking the two tables.

---

## 8. Spin-Up and AMIP Infrastructure

**Spin-up strategy (locked 2026-05-22).** No offline land spin-up driver.
`run_amip.py` cold-starts the land, runs a 20-yr coupled spin-up phase with the
atmosphere and prescribed SST, then opens the AMIP analysis window. The active
root zone (top 3 m) equilibrates within ~5–10 yr of coupled forcing; the
remaining 10–15 yr give us margin against drift in soil temperature and the
canopy-water store.

### 8.1 Cold-start land IC

`MultiLayerLandState` initial fields (built once at the top of `run_amip.py`):

| Field | Cold-start value |
|-------|-----------------|
| `T_soil[k]` | Latitude-binned annual-mean `T_sfc` from atmosphere IC, uniform with depth |
| `psi_soil`, `theta_soil` | `theta = 0.5 * (theta_wp + theta_fc)` (mid-range), then `psi = psi_from_theta(theta)` |
| `snow_depth` | 0 everywhere (snow will accumulate in the first winter) |
| `snow_age` | 0 |
| `runoff_surface`, `runoff_subsurface` | 0 |
| `canopy.T_leaf[k, sun/sha]` | Equal to `T_soil[0]` (no canopy ↔ soil gradient at t=0) |
| `canopy.q_leaf[k, sun/sha]` | Saturation `q_sat(T_leaf, p_ref)` via `legoesm.thermo` |
| `canopy.canopy_water[k]` | 0 (dry canopy) |
| `canopy.psi_leaf`, `canopy.psi_stem` | `psi_soil[0]` (hydrostatic, no transpiration draw) |
| `canopy.A_net`, `canopy.gs` | 0 (will be overwritten on first call) |

A helper `init_canopy_state_cold(ncol, config)` lives in
`multilayer_canopy/__init__.py` and is called from `run_amip.py` alongside
the existing `init_multilayer_land_state`.

Restart format carries a size-1 tile axis from day one so v2 PFT tiling can land
without a restart-format break.

### 8.2 PFT map and LAI loader (`src/legoesm/land/global_surface_data.py`)

The loader preserves per-PFT data on disk and collapses to dominant-PFT only at
the *driver* level. This is the most important tile-readiness boundary in v1.

```python
def load_pft_fractions(path: str, grid) -> jnp.ndarray:
    """Returns (npft, ncol) area fractions on the model grid.
    v1 caller will argmax to get dominant PFT; v2 will use directly."""

def load_lai_climatology_by_pft(path: str, grid) -> jnp.ndarray:
    """Returns (12, npft, ncol) monthly LAI per PFT on the model grid.
    v1 caller will collapse along the PFT axis using dominant_pft_collapse;
    v2 will index by tile."""

def dominant_pft_collapse(
    fractions: jnp.ndarray,           # (npft, ncol)
    lai_by_pft: jnp.ndarray,          # (12, npft, ncol)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """v1-only helper. Returns (pft_idx (ncol,), lai (12, ncol))."""

def pft_blended_config(pft_idx: jnp.ndarray) -> LandSurfaceParams:
    """Build per-column LandSurfaceParams from dominant PFT table lookup."""
```

In v2 the driver skips `dominant_pft_collapse`, passes `(12, npft, ncol)` LAI
and `(npft, ncol)` `LandSurfaceParams` to a vmap'd `step_multilayer_land`, and
aggregates `TileResponse` by tile fraction in the coupler.

### 8.3 AMIP driver updates (`scripts/run_amip.py`)

1. At startup: load global PFT fractions and per-PFT monthly LAI; call
   `dominant_pft_collapse` to produce `(ncol,)` PFT index and `(12, ncol)` LAI
2. Build per-column `LandSurfaceParams` via the dominant-PFT table lookup
3. Cold-start `MultiLayerLandState` per §8.1
4. Set `MultiLayerLandConfig.stomata = StomataConfig(enabled=True)` (see M3)
5. At each model time step, linearly interpolate monthly LAI to the current DOY
   and thread `lai (ncol,)` through the coupler → `step_multilayer_land`
6. Run the 20-yr coupled spin-up phase with output suppressed or sparse
7. Open the AMIP analysis window; full diagnostics resume

---

## 9. Validation Plan

### Tier 0 — Upstream Fortran-fidelity tests (M2 acceptance gate)

Decision (locked 2026-05-22): trust upstream `clm-ml-jax` validation. We
vendor `tests/fortran_validation/test_golden_jax.py` under
`tests/land/multilayer_canopy/test_clm_ml_fortran_fidelity.py` and require
it to pass against the rewired (constants → `legoesm.constants`, thermo →
`legoesm.thermo`) modules. **Any divergence from the original `test_golden_jax`
results is a M2 blocker — the rewire is wrong.**

We do *not* add our own LMIP-CHATS regression. Failures observed downstream
(e.g., at the AMIP analysis stage) loop back to upstream's test suite as the
authoritative reference.

### Tier 1 — Adapter and column-step tests (M4 acceptance gate)

| Test | Pass criterion |
|------|---------------|
| Energy balance closure (canopy + soil column) | `|G_top_canopy + SH + LH − R_net − ΔU_canopy| < 0.1 W m⁻²` at each step |
| Water balance closure | `ΔS_soil + Σ_layers ΔW_can + Δsnow = P − ET − runoff` to machine precision |
| Zero-LAI behavior | All canopy fluxes → 0; `step_multilayer_land` results match bare-soil baseline (M0) |
| Adapter round-trip | Forcing a single column through `to_canopy_inputs` → vendored entry → `from_canopy_outputs` reproduces a known canopy reference run |
| Sub-stepping convergence | Halving `n_substeps` changes column-mean fluxes by < 0.5% |

### Tier 2 — Seasonal-cycle tests (M5 acceptance gate)

LMIP single-point at:
- Amazon (lat=-3°, lon=-60°, tropical evergreen, BR-Sa1-like): ET should peak in dry season; canopy interception ≈ 15–25% of P_rain
- Boreal Canada (lat=55°, lon=-100°, needleleaf evergreen boreal): LAI near-zero in winter → high albedo from soil/snow blend; stomata effectively closed → near-zero winter transpiration

### Tier 3 — Global validation (M6 acceptance gate)

After 20-yr coupled spin-up:
- Annual mean ET vs GLEAM/MODIS ET (RMSE target < 0.5 mm/day globally)
- Annual mean GPP vs FLUXCOM
- Runoff vs GRDC gauge-measured river discharge
- Surface albedo seasonal cycle vs CERES EBAF (key metric for the original AMIP motivation)

---

## 10. Implementation Milestones

Each milestone is single-PR-sized, validated at a single LMIP point before
going further, and leaves the model in a runnable state.

### M0 — Single-point LMIP baseline *(done)*
- `scripts/run_lmip.py` committed; regression target for all subsequent work.
- `scripts/plot_lmip.py` diagnostic plotter (surface energy, soil T/θ
  depth-time, surface-layer time series) — present in the working tree.

### M1 — Plumb `lai` through `step_multilayer_land` *(done via canopy refactor)*
- The two-leaf canopy consumes LAI (prescribed or prognostic `C_fol/LCMA`).
- Residual: confirm a single `lai_default` resolution order is documented on
  `MultiLayerLandConfig`; not blocking.

### M2 — ~~Vendor CLM-ML in-tree~~ *(SUPERSEDED 2026-06-12)*
**Superseded by the `origin/aya/land` plugin approach.** The clm-ml work did
*not* vendor 25 modules into `src/legoesm/land/multilayer_canopy/`; it added a
single lazy-imported adapter (`canopy/clm_ml_interface.py`, 757 LOC) against
the *external optional* `clm_ml_jax` package (`pip install ".[canopy]"`). Do
**not** create `src/legoesm/land/multilayer_canopy/`. The original M2 vendoring
checklist, the `MLclm_varcon`/`ML_water_vapor` rewire tasks, and §11's
"vendored code under `src/legoesm/`" compliance items **no longer apply** — the
constant/thermo-hygiene burden moves into the *adapter* (already done in the
plugin), not into an in-tree copy of CLM-ML. (See §11 note.)

### M3/M4 — Canopy state + wiring into `step_multilayer_land` *(done — two flavors)*
Both done, independently, by the merged refactor and by `aya/land`:
- **DifferBESS two-leaf** (`land/stable`): `surface_scheme/` abstraction,
  `TwoLeafCanopyConfig`, dispatched in `step_multilayer_land`; canopy↔soil via
  outer Picard; energy/water close; zero-LAI → bare-soil baseline.
- **CLM-ML plugin** (`aya/land`): `CLMMLCanopyConfig`, `CanopyState`,
  `compute_clm_ml_canopy_fluxes` dispatched as a third `isinstance` branch.
- Both emit `SurfaceFluxOutput`; the shared post-flux pipeline (snow, Richards,
  soil thermal, carbon, TileResponse) is scheme-blind.
- **Residual integration work** (carry into M6, not M3/M4): the two branches
  are not yet reconciled on one line; pick the AMIP-v1 scheme and ensure the
  three-way dispatch (Slab / two-leaf / CLM-ML) coexists where AMIP runs.

### M5 — Global surface-data loader (scheme-agnostic, tile-ready) ◄ **NEXT**
This is now the critical path. **Design directive (locked 2026-06-12): the
loader is surface-scheme-agnostic.** It loads geophysical fields and maps them
to the grid; it does *not* know or assume which scheme consumes them.
- `src/legoesm/land/global_surface_data.py` per §8.2.
- Load and regrid to the model grid:
  - `(npft, ncol)` PFT fractions (CLM5 surfdata `PCT_NAT_PFT` / `PCT_CFT`)
  - `(12, npft, ncol)` monthly LAI (CLM5 `MONTHLY_LAI`; MODIS MOD15A2H fallback)
  - `(ncol,)` VIS + NIR **background (soil/snow-free) albedo** — drives Slab's
    background term *and* the canopy `ALB_VIS`/`ALB_NIR` inputs
  - `(npft, ncol)` canopy height (htop/hbot) + SAI
- **Scheme-agnostic output contract.** The loader returns a single
  `GlobalSurfaceData` container of raw geophysical fields. A *separate*, thin,
  per-scheme mapper turns it into that scheme's params:
  - Slab → `land_albedo` / background-albedo override
  - DifferBESS two-leaf → `CanopyLandParams` (`ALB_VIS`, `ALB_NIR`, `LAI`,
    `CI`, PFT→Vcmax25/canopy-height tables)
  - CLM-ML → `LandSurfaceParams` canopy fields (LAI, SAI, htop, hbot)
  No scheme-specific assumptions live in the loader itself.
- `dominant_pft_collapse()` stays the *only* v1-only helper; everything
  upstream preserves the PFT axis (§10a).
- LMIP seasonal-cycle tests (Tier 2 in §9) at Amazon + Boreal Canada, run for
  **each available scheme** to prove the loaded data drives all of them.
- **Validation:** loader regression tests (regrid conservation, PFT-fraction
  sums, LAI seasonal range) + tropical/boreal seasonal-cycle pass.

### M6 — AMIP wiring + scheme selection + coupled spin-up ◄ **after M5**
- `scripts/run_amip.py` gains a `--land-surface-scheme {slab,two_leaf,clm_ml}`
  flag; the loaded `GlobalSurfaceData` feeds the selected scheme via its mapper.
  Default = `two_leaf` (in-tree, no external dep); `slab` stays the fallback;
  `clm_ml` available when the `[canopy]` extra is installed.
- Thread `lai(t)` (monthly, interpolated) through the coupler step.
- Replace the current slab-land `dynamic_albedo=False` path so albedo comes
  from loaded background albedo + snow feedback + (for canopy) prognostic
  canopy albedo.
- Cold-start `MultiLayerLandState` (soil + canopy) per §8.1.
- 20-yr coupled spin-up with sparse monthly diagnostics (drift monitoring).
- Coupler owns tile aggregation (no-op `ntile=1` in v1).
- **Validation (Tier 3 in §9):** surface-albedo seasonal cycle vs CERES EBAF
  (the priority metric), annual-mean ET vs GLEAM, GPP vs FLUXCOM.

```
M0 ──► M1 ──► M2 ───────► M3/M4 ──────► M5 ──────► M6
done    done   SUPERSEDED  done(×2)      global     AMIP wiring
              (plugin, not  two-leaf +   loader     + scheme select
               in-tree)     clm-ml plugin (agnostic) + spin-up
                                          ◄ NEXT     ◄ priority
```

M5 is the critical path to the albedo/energy-budget goal and depends on no
unfinished canopy work — both schemes that consume its output already exist.

---

## 10a. Tile-Ready Design Contract (v1 ships dominant-PFT, v2 will tile)

These constraints are non-negotiable in v1 because violating any of them turns
the v2 tiling work into a rewrite.

- **`step_multilayer_land` stays strictly column-pure.** No internal awareness
  of tiles, no PFT branching, no fraction-weighted aggregation. v2 wraps it in
  `jax.vmap(step_multilayer_land, in_axes=<tile_axis>)` from the driver.
- **State fields stay `(ncol,)` / `(ncol, n_layers)` in v1.** v2 adds a leading
  or trailing `ntile` axis. `canopy_water` (M4) follows the same rule — it is
  a column field in v1, a column-tile field in v2. Richards and soil-thermal
  operators are untouched at tile time.
- **`LandSurfaceParams` semantics are `(ncol,)` in v1, `(ncol, ntile)` in v2.**
  No hardcoded one-PFT-per-column assumptions anywhere except the explicit
  `dominant_pft_collapse` call at the top of `run_amip.py`.
- **The M5 loader preserves per-PFT data on disk.** Dominant-PFT collapse is a
  *separate, named, v1-only* function called by the driver. If we collapse at
  file-read time, M5→v2 becomes a rewrite.
- **Tile aggregation lives in the coupler.** `TileResponse → SurfaceToAtm`
  blending is where fraction-weighted averaging happens. v1 is a trivial
  `ntile=1` reduction; v2 reuses the same code path with real fractions.
  Nothing about `step_multilayer_land` changes when tiling lands.
- **Restart format carries a size-1 tile axis from day one.** Avoids a forced
  restart-format break when tiling lands.

---

## 11. Constants and Compliance Notes

> **Re-baseline note (2026-06-12):** the original M2 in-tree vendoring was
> superseded (§0, §10). The bullets below describing the `MLclm_varcon` /
> `ML_water_vapor` rewire of a vendored `src/legoesm/land/multilayer_canopy/`
> are **historical** — there is no in-tree CLM-ML copy. The live obligations
> are: (a) the DifferBESS two-leaf and the CLM-ML *adapter*
> (`clm_ml_interface.py`) route all constants through `legoesm.constants` and
> all saturation thermo through `legoesm.thermo` (the adapter already does);
> and (b) the **M5 loader** uses `legoesm.constants` for any physical literals
> and adds no hardcoded constants/thermo of its own.

Per CLAUDE.md, vendored code under `src/legoesm/` is **not** exempt from
constant and thermodynamics-hygiene rules. *(Historical — M2's rewire was the
deliverable when vendoring was the plan:)*

- Every physical constant in `MLclm_varcon.py` and every literal scattered
  across the 25 vendored modules must route through `legoesm.constants`.
  Delete `MLclm_varcon.py` after the audit.
- Saturation thermodynamics (Tetens, Magnus, Clausius-Clapeyron, ice
  saturation) must route through `legoesm.thermo`. `ML_water_vapor.py`
  becomes a delegator.
- `getattr(..., 'X', <literal>)` fallbacks and function-signature defaults
  must use `constants.X`, not bare literals (audit-enforced).
- New config NamedTuple field defaults for physical constants should
  reference `constants.X` or carry a `# = constants.X` annotation.
- Every vendored module that is wired into the canopy dispatch chain must
  have at least the upstream Fortran-fidelity test exercising it
  (M2 acceptance gate). Modules that fail to be reached by that test are
  candidates for the explicit-drop list in §5 — don't ship dead code under
  `src/legoesm/`.
- The canopy package is wired into `step_multilayer_land` in M4. After M4
  it must not be "alive only because the upstream test imports it" — the
  legoesm production path through `step_multilayer_land` must exercise the
  full canopy chain.
- `MultiLayerLandConfig` default flips canopy-on; older LMIP regression
  tests that assume bare-soil-equivalent results need explicit
  `MultiLayerCanopyConfig(...)` overrides or `lai=0.0` to recover the
  pre-canopy baseline.

---

## 12. Decisions (locked 2026-05-22) and Remaining Open Questions

### Resolved

1. **Canopy implementation** — ~~Vendor `clm-ml-jax` in-tree.~~ **REVISED
   2026-06-12.** Two interchangeable canopies now exist behind the shared
   `SurfaceFluxOutput`: the **DifferBESS two-leaf** (in-tree, merged on
   `land/stable`, the AMIP-v1 default — no external dependency) and the
   **CLM-ML plugin** (`origin/aya/land`, an adapter to the *external optional*
   `clm_ml_jax` package). The door stays open to both; scheme is a runtime
   choice in M6. CLM-ML is *not* the sole canopy and is *not* vendored in-tree.
2. **Vendoring policy** — ~~Copy in & rewire 25 modules.~~ **REVISED
   2026-06-12.** No in-tree vendoring. CLM-ML is consumed as an optional
   dependency (`pip install ".[canopy]"`) through a lazy-imported adapter that
   owns the `legoesm.constants` / `legoesm.thermo` rewire at the interface
   boundary. Upstream's offline driver / namelist / CLM-share shims are not
   pulled in (the adapter constructs `mlcanopy_type` directly).
3. **Validation gate** — Run upstream `tests/fortran_validation/test_golden_jax.py`
   (vendored unchanged) as the M2 acceptance gate. No new LMIP-CHATS regression
   in legoesm. Trust upstream for Fortran fidelity; legoesm's responsibility
   is the rewire and the adapter.
4. **Plant hydraulics + WUE-gs** — Plant hydraulics ON in v1 (comes with
   "replace" choice). WUE-optimized stomatal path (`gs_type=2`) deferred to
   v2 in favor of Ball-Berry.
5. **Spin-up strategy** — 20-yr coupled spin-up in `run_amip.py` with prescribed
   SST. No offline land driver. `run_lmip_global.py` is *not* built.
6. **Land IC** — Cold start. Latitude-binned annual-mean `T_sfc` for `T_soil`;
   `theta = 0.5*(theta_wp+theta_fc)` for soil moisture; zero snow; canopy
   T_leaf = T_soil[0], canopy_water = 0, ψ_leaf = ψ_soil[0].
7. **PFT cover** — Dominant-PFT in v1, but the M5 loader, `LandSurfaceParams`
   shapes, restart format, and `step_multilayer_land` API are all designed
   tile-ready (see §10a). v2 tiling drops in without state or operator changes.
8. **Stomata default** — Canopy stomata always on (inside the vendored
   Farquhar+BB solver). The pre-existing `StomataConfig` in
   `MultiLayerLandConfig` stays as a separate non-canopy path with its old
   `enabled=False` default; it is not used by the canopy step.

### Still open

1. **LAI data source** — CLM5 surfdata `MONTHLY_LAI` is preferred (already
   PFT-resolved, matches the M5 loader contract). MODIS MOD15A2H is the
   fallback if the CLM5 file is not available on the target machine. Decide
   when M5 starts.
2. **End-to-end `jax.grad`** — Upstream is designed to be AD-safe with IFT
   correction but is not advertised as validated end-to-end. Whether we
   commit to keeping the canopy chain `grad`-compatible (and gate that with
   tests) is a v2 question; not blocking AMIP v1.
3. **Coupled spin-up output cadence** — During the 20-yr spin-up, write sparse
   monthly diagnostics for drift monitoring (recommended), or skip output
   entirely. Decide when M6 starts.
4. **Coupling frequency** — Upstream CLM-ML expects a 30-min coupling step
   (CLM convention). legoesm's atmosphere may step faster; either sub-step the
   canopy at its native cadence inside `step_multilayer_land`, or accept the
   legoesm atmosphere `dt` and let `MultiLayerCanopyConfig.n_substeps` absorb
   the difference. Decide in M3/M4 once we measure stability sensitivity.
