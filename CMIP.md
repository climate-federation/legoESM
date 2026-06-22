# CMIP coupled-realism work log

Goal: make the legoESM **coupled CMIP** simulations physically realistic
(coupled atmosphere + slab/two-layer ocean + land + sea ice; lat-lon, rrtmgp +
sbm convection + morrison microphysics + sundqvist clouds, `ic="standard"`).
Branch `omip-faithful-nemo-comparison`. This file is updated as work proceeds.

Run env (Ginsburg, GPU partition `glab1`, `--account=glab`):
`JAX_ENABLE_X64=1 python scripts/run/run_coupled.py --preset … --grid latlon
--ic standard --radiation rrtmgp --convection sbm --microphysics morrison
--clouds sundqvist …` (always via `sbatch`/`srun`, never the login node).

## Status at a glance
| Quantity | Earth | original | NOW (fixes on) |
|---|---|---|---|
| Planetary albedo | ~30% | 13–14% | **28–30%** |
| OLR (rlut) | ~240 | 388 | ~221–232 |
| CWV | ~25 | 84→14 | **20.7** |
| `<R_TOA>` (area-wtd) | ~0 | −36 (artifact) | **−1.2 … +15** |
| SST drift | ~0 | −9 K/yr | **−7.2 K/yr** (spin-up) |
| LW_net_sfc | ~−55 (CAM) | −110 | −75 … −91 |

## Fixes SHIPPED (committed + pushed)
1. **IC fix** — root cause of low precip / high CWV / OLR=388 was the uniform
   isothermal `ic="default"` scaffold. Fix = `ic="standard"` (constant-lapse-rate
   troposphere + equator-pole gradient), now default for lat-lon + cube. NOT a
   missing large-scale precip scheme. (PR #525)
2. **Cloud calibration + radiative-condensate floor** — prognostic `q_c≈0` made
   clouds optically inert (albedo 13%). Floored radiative condensate at
   `cf·q_c_diagnostic`; calibrated `rh_crit=0.77`, `q_c_diagnostic=1e-3` →
   albedo→29.7%, OLR→232. (PR #528)
3. **Area-weighted diagnostics** (`a1fb79d8c`) — coupled global means used a plain
   `jnp.mean`, polar-biasing every lat-lon metric. Fixed with `area_weighted_mean`
   (`grid.grid_area`). **`<R_TOA>` −36 → −1.24** (the "cold catastrophe" was largely
   a diagnostic artifact; the solar code was already correct, area-integrates to
   S_0/4=340). The SST drift, however, is REAL (spin-up to a too-cold/dry RCE).
4. **Convective cloud-fraction source** (`cd6de9746`, `98f21c304`, `3035badfe`,
   `222adb53c`) — THE tropical cold-bias fix. Diagnosis: the sbm *adjustment*
   convection scheme provides no convective cloud fraction, and the grid-mean-RH
   stratiform schemes give cf≈0 at the tropical RH≈0.71 → convecting tropics had
   NO cloud → surface LW radiated to space (LW_net_sfc −137 W/m², precip 1 mm/day).
   Added an opt-in (`--convective-cloud`) bounded Slingo(1987) cumulus cover tied
   to the (lagged) convective precip, over a high anvil deck, combined by max
   overlap, threaded through the compiled hot-loop (lagged-precip SegmentCarry
   field). Made the anvil an **optically-thin CIRRUS** (`conv_cloud_condensate`
   = 1.5e-4, ~7× < stratiform) so it is LW-active (warms, moistens) but
   SW-transparent (Earth-like albedo). Default-off ⇒ value-identical. Codex-clean,
   106 cloud tests. **v4 result: albedo 28.6%, CWV 14→20.7, SST drift −8.7→−7.2,
   R_TOA→+15** (tuning ladder v1 thick→overcast 79%, v2/v3 anvil→albedo 42%,
   v4 thin cirrus→28.6%).
5. **Surface albedo over land/ice** (`622f02333`) — ocean (0.06) + sea-ice (0.65)
   were correct, but snow-covered LAND was too dark (constant 0.2) because
   `snow_albedo_feedback` defaulted off. Added `--snow-albedo-feedback` →
   lat-varying vegetation albedo (0.15/0.20/0.25) + snow brightening (→0.6–0.8)
   for both slab and multilayer land.
6. **Multilayer land ("land MIP")** — `--preset slab_richards` swaps the crude
   slab bucket for the multilayer Richards soil/veg land (validated offline by
   `scripts/run/run_lmip.py`). Validated coupled (30d): clean, Earth-like albedo
   28.5%, precip 1.5→2.0 (better ET), near energy balance. Caveat: soil cold-spins
   inline (no warm-start bridge; months–years).

## Diagnosis (the residual ~4.5 K cold bias)
The cold bias is **tropical + structural**, NOT a TOA leak (R_TOA≈0 area-weighted)
and NOT solar. Chain: dry tropical column → weak water-vapor greenhouse + no
convective cloud → surface LW to space → cold → drier. The convective thin-cirrus
cloud (fix #4) closes most of it (CWV, LW_net, drift, R_TOA all improved) at
Earth-like albedo. Land surface fixed separately (#5, #6).

## Dead ends (ruled out, don't retry)
- **mass_flux convection** → overcast (albedo 76–79%, R_TOA −298): over-moistens
  → the steep Sundqvist √-curve saturates. Thin-cirrus on sbm is the right path.
- **Gustiness ↑** → over-cools the energy-limited interactive slab (drift −20 K/yr).
- **AMIP (prescribed SST)** → blocked: deck SST file has all-zero coords;
  run_amip.py rrtmgp compile too slow (no gpoint-batch flag). See
  `docs`/memory `cmip_amip_path_blockers`.
- **Present-day (2021) GHG** → confirms the bias is structural not forcing
  (R_TOA −1.2→+0.8 but SST drift unchanged −8.3).

## Combined full-realistic run — DONE (job 8534577, 90d)
The culmination run: thin-cirrus convective cloud + multilayer Richards land +
`--snow-albedo-feedback`, all three session fixes together, 90 days.

| Quantity | Earth | Combined 90d | verdict |
|---|---|---|---|
| Planetary albedo | ~30% | **28.6%** | ✓ Earth-like, no overcast |
| OLR (rlut) | ~240 | 220 | ✓ |
| rsut | ~100 | 80.8 | slightly low |
| `<R_TOA>` (area-wtd) | ~0 | **+17.8** | net-IN ⇒ warming back |
| CWV | ~25 | 18.1 | low (cold column) |
| **tas (global)** | ~288 K | **278.8 K** | **−9 K cold** |
| tos (global) | ~291 K | 287.6 K | ~OK (−3 K) |
| SST drift | ~0 | −7.3 K/yr | spin-up |
| siconc | ~6% | 16.5% | too icy (cold) |
| LW_net_sfc | −55 | −100 | still high |

**Key finding:** the fast radiative physics is now Earth-like (albedo, OLR,
cloud), but the system **overshot cold** — atm column-T fell monotonically
300→254 K over 90d while `<R_TOA>` flipped to **+17.8 W/m² net-IN**. Classic
slab+soil overshoot/lag: TOA already wants to warm but the slow reservoirs
haven't caught up. Critically **tos=287.6 K (warm-ish) but tas=278.8 K (cold)**
⇒ the −9 K cold is **land + high-latitude (sea-ice) concentrated**, NOT the open
ocean: the multilayer soil is cold-spinning inline (no warm-start) and sea ice
over-grew, dragging the global tas down. This is **incomplete spin-up, not a
physics bug** (R_TOA>0 confirms net energy gain). Equilibration timescale: 50 m
slab @ +17.8 W/m² ≈ 3–4 yr to recover; soil months–yr. 90d is far too short.

## Deeper diagnosis — air-sea DECOUPLING + anemic evaporation (2026-06-21)
Splitting the combined-run tas/tos by latitude band exposed the real mechanism:

| band | tas | tos | sea-air gap |
|---|---|---|---|
| tropics (0–20°) | 15.0°C | **27.9°C** | **+12.9** |
| midlat (20–50°) | 12.7°C | 19.7°C | +7.0 |
| highlat (50–90°) | −0.8°C | 6.0°C | +6.8 |

The ocean SST is ~right (tropics 28°C ✓) but the near-surface AIR is ~13K
colder than the ocean **even over open tropical ocean (0% ice)**. With
`hfls`≈40 W/m² (tropical Earth ~120), the warm ocean is barely
evaporating → the air stays cold + dry (CWV 18) → weak vapor greenhouse →
surface LW to space → cold. **The proximate cause is the surface latent-heat
flux.** Math: tropical `hfls=ρ·L_v·Ch·|U|·Δq` with Δq~0.017, Ch=1.5e-3 implies a
surface wind |U|~0.5 m/s — the model's tropical surface winds are near-calm, and
the **constant neutral** bulk scheme has **no convective gustiness**. Over a
13K-unstable ocean the real atmosphere evaporates via free-convection gustiness
(w*); the `coare3`/`large_yeager` MOST schemes include exactly that w* term.

**New lever SHIPPED — `--surface-bulk-scheme {constant,coare3,large_yeager}`**
(opt-in, default `constant` ⇒ byte-identical). Applied CONSISTENTLY across the
three air-sea flux computations so the turbulent heat leaving the ocean matches
the heat entering the atmosphere:
- atmosphere `SurfaceLayerConfig.bulk_scheme` (via `_resolve_turbulence`),
- **slab/two-layer `SimpleOceanConfig.bulk_scheme`** (the actual coupled heat
  budget — new shared `_ocean_turbulent_fluxes` helper, dispatch-hardened),
- coupler `CouplerConfig.bulk_scheme` (radiation skin-T feedback / dynamic ocean).

Codex adversarial review (read-only, 7 findings) caught the key bug: the
slab/two-layer `SimpleOcean` heat budget hard-codes its OWN constant fluxes and
does NOT read `CouplerConfig` — so an atmosphere-only switch would be
**non-conservative**. Fixed by wiring the scheme into `SimpleOceanConfig` too
(HIGH#1). Also: reject `coare3`+`turbulence=none` (HIGH#2, no atm surface layer
to update); CLI help scoped honestly (land/lake/ice tiles keep their own scheme,
MED#4); known 0.98 saline-qsat offset in `ocean_tile_response` only (MED#3, not
the slab budget). Tests: resolver propagation, validate_strict membership +
turbulence guard, coverage ratchet, SimpleOcean MOST-enhances-unstable-flux,
dispatch-hardening raise.
CAUTION (prior dead-end): an *ad-hoc* gustiness bump over-cooled the
energy-limited slab (−20 K/yr); the MOST scheme computes w* self-consistently
from the buoyancy flux. The 30d coare3 experiment will show whether stronger,
*conservative* evaporation warms+moistens the air faster than it cools the slab.

## coare3 surface-flux EXPERIMENT — VALIDATED net win (job 8535204, 30d)
Combined config + `--surface-bulk-scheme coare3` (after fixing a float32 MOST
`fori_loop` carry-dtype crash, commit 372ee6a64). vs the constant-scheme run
(first 30d, area-weighted):

| metric | constant | coare3 | verdict |
|---|---|---|---|
| **SST drift** | −7.2 K/yr | **−3.2 K/yr** | ✓✓ halved |
| hfls (global) | 41 | **55** W/m² | ✓ toward Earth ~80 |
| hfls (tropics) | 10 | **21** | ✓ doubled |
| global tas | 10.4°C | **11.9°C** | ✓ +1.5 K warmer |
| tropics tas–tos gap | +12.9 | **+11.3** | ✓ shrunk |
| CWV / precip | 20.7 / 1.71 | **21.8 / 1.90** | ✓ |
| R_TOA | +15 | +8.7 W/m² | ✓ toward balance |
| **planetary albedo** | 28.6% | **35.4%** | ✗ overshoot |

The evaporation lever works exactly as the air-sea-decoupling diagnosis
predicted: the self-consistent MOST convective-gustiness w* doubles tropical
evaporation, warms+moistens the column, shrinks the tas–tos gap, and **halves
the cold SST drift**. The prior ad-hoc gustiness dead-end (gustiness=5 → −20
K/yr WORSE) is REVERSED here — because (a) the flux is applied conservatively to
BOTH the atmosphere and the slab budget and (b) w* is derived self-consistently
from the buoyancy flux, not bumped.
NEW TENSION: more moisture → more (reflective) low cloud → **albedo 28.6→35.4%**
(too high; the same SW/LW tension as the thin-cirrus tuning). The net is still a
clear improvement (warmer, moister, half the drift), but the albedo overshoot
now caps further warming → the next lever.

### Albedo recovery — coare3 + thinner cloud = BEST config (job 8535243, 30d)
Exposed the cloud SW/LW knob as CLI overrides (`--rh-crit`/`--q-c-diagnostic`/
`--conv-cloud-max`, opt-in, default = CloudConfig default = byte-identical;
commit 96263e5d6) and ran coare3 + `--q-c-diagnostic 5e-4` (cloud optically
thinner ⇒ less SW reflection, still LW-active). Full 3-way (first 30d):

| metric | constant | coare3 | **coare3 + q_c 5e-4** |
|---|---|---|---|
| planetary albedo | 28.6% | 35.4% | **31.3%** ✓ Earth-like (overshoot recovered) |
| **SST drift** | −7.2 | −3.2 | **−2.2 K/yr** ✓✓ best (3.3× better than original) |
| CWV | 20.7 | 21.8 | **22.0** ✓ |
| tropics tas–tos gap | +12.9 | +11.3 | **+11.2** ✓ |
| hfls (global) | 41 | 55 | 51 W/m² |
| R_TOA | +15 | +8.7 | +19.2 W/m² |

**The two levers COMPOSE into the most realistic coupled config to date.** coare3
fixes the air-sea decoupling (stronger evaporation, warmer/moister column,
halves the drift); the thinner cloud recovers the albedo (35→31%, Earth-like)
AND further cuts the drift (less SW reflected → more SW into the surface →
−3.2→−2.2 K/yr). **SST drift cut from the original −7.2 to −2.2 K/yr at Earth-like
albedo.** R_TOA +19 (net-in) is the cost of the thinner cloud — it keeps the
system warming toward equilibrium. RECOMMENDED coupled config:
`--surface-bulk-scheme coare3 --q-c-diagnostic 5e-4` (+ the combined preset).

## Open / next levers (ranked by the combined-run diagnosis)
The combined run isolates the residual to **spin-up of the slow surface
reservoirs**, dominated by cold land + over-grown sea ice (tos warm, tas cold).
So:
1. **Long run to equilibrium** (definitive): R_TOA=+17.8 predicts warm-back;
   a multi-year (≥1 yr) run is the real "is it realistic at equilibrium" test.
   90d=5.7h ⇒ 1yr≈23h (fits 72h walltime). Restart-chaining for multi-year.
2. **Warm-start the surface** so it doesn't overshoot cold from the T=300 IC:
   init slab at observed SST + soil at observed soil-T (kills the artificial
   300→254 cooling transient; the run that converges fastest to realistic).
3. Reduce the sea-ice over-growth (siconc 16.5% vs ~6%): the cold high-lat
   air over-freezes; warmer equilibrium should self-correct, but check the
   ice-albedo feedback isn't latching.
4. Small extra LW trapping (`conv_cloud_max` 0.15→~0.18, albedo headroom to 30%)
   to accelerate warming — secondary to spin-up.

## Key files
- `packages/atmosphere/legoesm/atmosphere/physics/clouds/cloud_fraction.py`
  (`convective_cloud_fraction`, thin-cirrus condensate split) + `config.py`.
- `packages/coupler/legoesm/driver/{compiled_segments,physics_pipeline,model_driver}.py`
  (lagged-precip carry wiring).
- `packages/tools/legoesm/diagnostics/energy_budget.py` (`area_weighted_mean`).
- `scripts/run/run_coupled.py` (`--convective-cloud`, `--snow-albedo-feedback`).
- Memory: `cmip_area_weighting_diag`, `cmip_convective_cloud_wiring`,
  `cmip_land_albedo_multilayer`, `cmip_amip_path_blockers`.
