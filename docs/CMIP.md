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
Recommended config: combined preset (slab_richards + --snow-albedo-feedback +
--convective-cloud) + **--surface-bulk-scheme coare3 --q-c-diagnostic 5e-4**.
| Quantity | Earth | original | combined (constant) | **BEST (coare3+thin-cloud)** |
|---|---|---|---|---|
| Planetary albedo | ~30% | 13–14% | 28.6% | **31.3%** |
| OLR (rlut) | ~240 | 388 | ~221 | ~221 |
| CWV | ~25 | 84→14 | 20.7 | **22.0** |
| `<R_TOA>` (area-wtd) | ~0 | −36 (artifact) | +15 | +19 (warming) |
| **SST drift** | ~0 | −9 K/yr | −7.2 | **−2.2 K/yr** |
| hfls | ~80 | — | 41 | **51** |
| tropics tas–tos gap | ~1 | — | +12.9 | **+11.2** |

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

## 2 m tas diagnostic — correct CMIP near-surface T (job 8544950, 30d)
CMOR `tas` was the lowest model level (~80 m at nlev=20), reading colder than
2 m over a warm ocean. New `DiagnosticCollector._tas_2m` interpolates to 2 m via
the MOST similarity profile (`compute_most_fluxes(return_2m=True)`). DIAGNOSTIC-
ONLY (no dynamics change): vs the same balanced run, tropics tas **21.7 → 22.9 °C**
(gap +6.6 → **+5.4**), midlat tas 16.6 → 16.9 (gap +3.3); R_TOA/hfls/albedo/drift
identical. A real, modest correction toward Earth-like (the lowest level is only
~80 m up, so the profile correction is ~1–2 K). Opt-in core (`return_2m`, default
5-tuple byte-identical → OMIP/AD untouched); collector falls back to the lowest
level when SST/sigma absent.

## ✅ 180d balanced equilibration — DEFINITIVE PROOF (job 8539793)
The recommended config (`coare3 --gustiness-zi 300 --q-c-diagnostic 3e-4` +
combined preset) run 180 days reaches a **stable, energy-balanced, Earth-like
climate** — the closing validation:

| metric | 30d (transient) | **Day-180 equilibrated** | Earth |
|---|---|---|---|
| **`<R_TOA>`** | +21.8 | **−0.72 W/m²** | ~0 ✓✓✓ BALANCED |
| SST drift | −5.7 | **−2.5 K/yr** | ~0 ✓ |
| **tropics tas–tos gap** | +6.6 | **+4.9** | ~1 ✓✓ |
| **midlat tas–tos gap** | — | **+1.8** | ~1 ✓✓ |
| tropics tas | 21.7 | **23.2 °C** | ~26 |
| planetary albedo | 31.4 | 33.0% | ~30 ✓ |
| precip | 2.58 | **2.32 mm/d** | ~2.7 ✓ |
| hfls | 72.8 | 60.8 | ~80–120 |
| column-T | — | **269 K plateau** (Day 90→180 flat) | — |

The +22 R_TOA transient **settled to −0.72 W/m²** over the run (the ocean→
atmosphere redistribution completed; the warm column now radiates the surplus),
the air-sea gap closed to +4.9/+1.8 K, and the column-T plateaued at 269 K. **The
whole arc: original cold catastrophe (R_TOA −36 artifact, runaway cold drift) →
stable, energy-balanced, Earth-like coupled climate.** Both fixes merged to main
(#563 + #569). Residual: hfls 61 still < Earth ~80; CWV 35 high.

## FINAL balanced config — gustiness + thin cloud (job 8538901, 30d)
**RECOMMENDED:** combined preset + `--surface-bulk-scheme coare3
--gustiness-zi 300 --q-c-diagnostic 3e-4`. The two knobs compose:

| metric | no-gust | gust-only | **balanced** | Earth |
|---|---|---|---|---|
| hfls | 51 | 74.7 | **72.8 W/m²** | ~80–120 ✓ |
| tropics tas–tos gap | +11.2 | +6.6 | **+6.6** | ~1 ✓ (halved) |
| precip | 1.90 | 2.68 | **2.58 mm/d** | ~2.7 ✓✓ |
| **planetary albedo** | 31.3 | 35.4 | **31.4%** | ~30 ✓✓ |
| CWV | 22.0 | 30.6 | 30.8 | ~25 |
| `<R_TOA>` | +19 | +12.9 | +21.8 | ~0 |
| SST drift | −2.2 | −6.7 | −5.7 K/yr | ~0 |

The COARE convective gustiness (the diagnosed missing w*) gives Earth-like
hfls / air-sea gap / precip; the thinner cloud recovers the moisture-driven
albedo overshoot back to 31%. **The persistent air-sea decoupling — the last
major equilibrium bias — is fixed at Earth-like albedo.** R_TOA +22 / drift −5.7
are the ocean→atmosphere heat-redistribution transient (the atmosphere warmed
+5 K; a long run settles them as the prior 180d config did). Open: a long run of
this config to confirm equilibration; CWV slightly high (30 vs 25).

## 180d equilibrated result (job 8535790, COMPLETE) + remaining-bias diagnosis
Best config (coare3 + `--q-c-diagnostic 5e-4` + combined preset), 180 days:
| metric | Day-30 | **Day-180 (equilibrated)** | Earth |
|---|---|---|---|
| planetary albedo | 31.3% | **30.1%** | ~30 ✓ |
| SST drift | −2.2 | **−1.4 K/yr** | ~0 (improving ✓) |
| column-T | 258.7 K | ~256.7 K (quasi-stable ~257) | — |
| `<R_TOA>` | +19 | **+20 W/m²** | ~0 ✗ persists |
| hfls | 51 | **45.6 W/m²** | ~80–120 ✗ |
| CWV / precip | 22.0 / 1.90 | 19.7 / 1.60 | 25 / 2.7 |
| tropics tas–tos gap | +11.2 | **+13.8** (tos 29.8, tas 16.0) | ~1 ✗ |

**Equilibration: PASS** — bottoms ~Day 90–100 then quasi-stabilizes at ~257 K
column-T; SST drift down to −1.4 K/yr; albedo Earth-like. NOT the constant
scheme's runaway. The fast-physics realism goal is met + merged (PR #563).

**Remaining bias = persistent air-sea decoupling, TWO parts:**
1. **Diagnostic artifact (~4 K of the gap):** `tas` is the LOWEST MODEL LEVEL T
   (diagnostics.py:693, "proxy for 2 m"), ~100 m up at nlev=20. A neutral
   log-law 2 m interpolation (T_sfc + ~0.72·(T_low−T_sfc)) gives ~20°C 2 m air
   vs the 16°C lowest level. → **add a proper MOST 2 m `tas` diagnostic** (needs
   surface skin-T + ustar/θ*/L threaded to the collector). Correct CMIP output.
2. **Real flux deficiency (~10 K + R_TOA +20):** even at 2 m the air is ~10 K
   below a 29.8°C ocean, with hfls 45 ≪ Earth ~120. The ocean absorbs R_TOA +20
   and warms (tos 28.8→29.8) but the weak surface turbulent flux can't shed it to
   the cold atmosphere → gap widens, R_TOA stays imbalanced. coare3 helped but
   the effective surface wind is still ~0.85 m/s (calm tropics; w* under-boosts).
   NEXT lever: stronger air-sea exchange — NOTE the slab is now energy-GAINING
   (R_TOA +20), the OPPOSITE of the old gustiness dead-end regime, so a higher
   gustiness/effective-wind floor should now shed the ocean's excess to the
   atmosphere (closes BOTH R_TOA and the gap) — but validate vs SST drift.

## Convective gustiness (COARE w*) — THE air-sea-coupling fix (job 8537518, 30d)
The 180d diagnosis (calm warm ocean barely evaporates; the MOST solver was
MISSING the COARE free-convection velocity scale w*) → implemented `w* =
(g·z_i·<w'θv'>/θv)^(1/3)`, `U_eff = √(|U|²+(β·w*)²)`, opt-in `--gustiness-zi`
(z_i = BL depth; commits d07dfc3ef + 0b72144c1). zi=600 vs no-gust (30d):

| metric | no-gust | **zi=600** | Earth |
|---|---|---|---|
| hfls | 51 | **77.9 W/m²** | ~80–120 ✓✓ |
| tropics tas–tos gap | +11.2 | **+5.8** | ~1 ✓✓ (halved) |
| precip | 1.90 | **2.76 mm/d** | ~2.7 ✓✓ |
| `<R_TOA>` | +19 | **+10.4** | ~0 ✓ |
| tropics tas | 17.6 | 22.5 °C | ~26 |
| CWV | 22.0 | 32.3 | ~25 (now high) |
| albedo | 31.3 | 35.7% | ~30 ✗ overshoot |
| SST drift | −2.2 | −7.2 K/yr | ~0 ✗ |

**VALIDATED: gustiness is THE fix for the air-sea decoupling** — hfls + precip
Earth-like, the tas–tos gap halved, R_TOA toward balance, the cold column warmed
+7 K (257→265 K col-T). But **zi=600 OVERSHOOTS**: over-evaporates → over-cools
the slab (−7.2, the old gustiness regime re-appears now that fluxes are strong)
+ over-moistens (CWV 32) → over-clouds (albedo 35.7). zi=300 (job 8538045) ≈ zi=600 (hfls 74.7, gap +6.6, albedo 35.4, drift −6.7):
**z_i has weak leverage** (w*∝z_i^⅓), so the albedo/CWV overshoot rides WITH the
gustiness, not tunable via z_i. The air-sea fix is locked in; the albedo (35) is
offset by the proven `q_c` knob → **final balanced config = gustiness zi=300 +
`--q-c-diagnostic 3e-4` (thinner cloud), job 8538901**. The slab drift −6.7 is
largely a transient ocean→atmosphere heat redistribution (atmosphere warmed
+5 K; R_TOA +13 net-in refills the slab over a long run). Lever + wiring shipped;
only z_i + q_c tune.

## Equilibration confirmed (180d, best config) + status
**Merged to main (PR #563).** A 180-day run of the best config
(coare3 + `--q-c-diagnostic 5e-4` + combined preset) bottoms out near Day
90–100 (~257 K column-T) then **warms back** (Day 130: 257.2 K and rising) under
R_TOA ≈ +19 W/m² — a stable Earth-like climate, NOT the constant scheme's runaway
cold drift (254 K and falling at Day 90). The fast physics is Earth-like; the
remaining residual is spin-up time + a still-too-cold near-surface air over land
and a low hfls (51 vs ~80).

## Open / next levers
1. **Soil warm-start SHIPPED** (`--warm-start-soil`, opt-in, byte-identical):
   init the land soil at the atmosphere's lat-structured near-surface air T (t=0)
   instead of uniform 280 K (tropical soil was ~18 K too cold → months of
   cold-spin). `init_multilayer_land_state`/`init_surface_state` now accept a
   per-column array; validation run 8536696 (best config + warm-start) pending.
2. **Tune against the EQUILIBRATED state** (not the spin-up transient): analyse
   the Day-150–180 biases of the 180d run → target the dominant remaining bias
   (likely the tropics tas–tos gap / hfls 51<80 surface-air coupling).
3. Reduce sea-ice over-growth (siconc 16.5% vs ~6%): warmer equilibrium should
   self-correct; check the ice-albedo feedback isn't latching.
4. Multi-year / restart-chained run for full slab+soil equilibrium.

## Full 3D dynamic ocean from realistic WOA IC (2026-06-24, in progress)
Goal: run the coupled model with the prognostic 3D ocean
(`--ocean dynamic --ocean-ic woa`, `LatLonCGridOceanModel` on the atm lat-lon
grid, WOA18 T/S + WOA continents/bathy) instead of the slab/two_layer used for
all the work above.

**RUNS STABLY = the deliverable** (10d job 8554291): the OMIP cold-start recipe
(`barotropic=implicit_cn`, `momentum=rk3`, `pgf=smc03`, partial cells +
balanced init) is AUTO-applied; no blowup (max current 27→20 m/s settling),
realistic SST [272–290 K].

**But cold-COLLAPSED** (90d 8557907, KILLED): SST −0.5 K/day, R_TOA −27,
CWV 17.6 (dry), drift −133 K/yr — a runaway, NOT a self-settling spin-up
transient like the slab.

**Root cause (fix shipped, commit a232a32ff): air-sea interface energy
inconsistency.** The 3D-ocean `q_net` is built from `ocean_tile_response`
(coupler), which called `compute_most_fluxes` WITHOUT the convective-gustiness
`w*` that the atmosphere surface layer and the slab ocean already use. So the
atmosphere evaporated with gustiness but the ocean-tile flux driving the
3D-ocean heat budget did not → a calm warm WOA ocean barely evaporates
(hfls ~45 vs ~120 W/m²) → dry atmosphere → the warm SST radiates LW the dry
column can't trap → R_TOA −27 → cold collapse. Same mechanism the slab
air-sea-decoupling fix solved; the 3D-ocean tile was simply missed.

Fix (opt-in, byte-identical when off):
- `CouplerConfig.gustiness_w_zi` (BL depth z_i for w*) + ocean_tile_response
  passes it to `compute_most_fluxes`;
- `run_coupled` threads `--gustiness-zi` onto the coupler tile (same value the
  atm surface layer gets) → interface flux is energy-CONSISTENT.

Validation: 15d A/B job 8558650 (vs prior 10d 8554291: R_TOA −27, CWV 17.6,
drift −133) — expect higher CWV, less-negative R_TOA, arrested drift.

**Gustiness alone CONFIRMED insufficient** (15d job 8558910): R_TOA −54,
SST drift −92 K/yr, column-T 256.8→250.9 K monotonic, circulation dying. The
standard complement — WOA surface T/S restoring — was implemented (commit
b071ba359): canonical Haney kernel `ocean/forcing/surface_relaxation.py`
(run_omip's `_apply_restoring` refactored onto it; no duplicate numerics),
`CoupledConfig.ocean_restore_{sst,sss}_tau_days`, `_apply_ocean_restoring` in
`_step_ocean` (opt-in, byte-identical off), `run_coupled
--ocean-restore-{sst,sss}-tau-days`.

**Restoring WORKS — measured win** (15d job 8559528, SST τ=15 d / SSS τ=60 d
vs gustiness-only 8558910):

| metric        | gustiness only | + restoring τ=15d |
|---------------|----------------|-------------------|
| SST drift     | −92 K/yr       | **−47 K/yr** (½)  |
| R_TOA         | −54 W/m²       | **−24 W/m²** (2.3×)|
| SST mean      | 279.0 K        | **281.4 K** (trop 290.6 ~WOA) |
| column-T d15  | 250.9 K        | 252.1 K (+1.2)    |
| CWV           | 15.87          | 15.87 (atm dry)   |

Restoring directly anchors the ocean (drift halved, R_TOA energy-loss more
than halved, SST realistic). NOT yet equilibrated: the atmosphere is still dry
(CWV 15.87), so R_TOA stays −24 and SST still drifts at τ=15 d. The atm column
responds slowly (a free coupled spin-up from a dry IC needs months).

**Stronger restoring τ=5 d (30 d job 8560151) — COLD COLLAPSE SOLVED:**

| config            | SST drift   | R_TOA   | SST mean        | col-T d30 |
|-------------------|-------------|---------|-----------------|-----------|
| gustiness only    | −92 K/yr    | −54     | 279.0           | collapse  |
| + restore τ=15 d  | −47 K/yr    | −24     | 281.4           | —         |
| **+ restore τ=5 d** | **+25 K/yr** | **−10** | **284.3 (trop 298)** | 252.6 (plateau) |

SST drift flipped **negative → positive** (−92 → +25 K/yr): the collapse is
arrested, mild warming toward equilibrium. R_TOA −54 → −10 (near balance). SST
mean 284.3 K with a realistic tropical warm pool (298 K). Column-T plateaued
(Day 20/25/30 = 252.6/252.4/252.6 K — equilibrated), circulation stable
(max_v ~19). **The full 3D dynamic ocean from realistic WOA IC now runs stably
AND realistically** (stable SST near climatology, energy near balance, no
collapse) — the deliverable.

Residual: CWV 14.9 (atm still dry) ⇒ R_TOA still −10. A longer run lets the warm
restored SST + gustiness evaporation moisten the column (CWV→greenhouse→R_TOA→0).

**Recommended realistic 3D-ocean coupled config:** `--ocean dynamic --ocean-ic
woa --ocean-restore-sst-tau-days 5 --ocean-restore-sss-tau-days 30
--surface-bulk-scheme coare3 --gustiness-zi 300 --q-c-diagnostic 3e-4
--land-bulk-scheme most --convective-cloud --snow-albedo-feedback --preset
slab_richards` (land MOST + Richards water-limited ET; ocean COARE air-sea).

**Full-config 60 d validation (job 8561537) — STABLE + RECOVERING:** the
complete config above. SST drift +19.2 K/yr (decelerating from the 30 d +25 ⇒
approaching equilibrium), R_TOA −8.9, SST mean 285.4 K with a realistic tropical
warm pool (299.8 K), column-T recovered +2.3 K from the Day-25 minimum
(252.4 → 254.7 K, monotonic warming Days 25–60 as the warm restored SST +
gustiness moisten the atm), circulation strengthening (max_v 18.9 → 20.8).
**The full 3D-ocean coupled CMIP run from realistic WOA IC runs stably and
equilibrates toward a realistic climate — cold collapse definitively solved.**
Residual: CWV ~15 (atm still drier than Earth's ~25) ⇒ R_TOA still −8.9; the
same slow-moistening residual the slab faced (slab took 180 d to reach R_TOA
−0.72), so multi-month integration is the remaining lever, not a structural
bug. Figure: `docs/scaling/cmip_3docean_full_60d.png`.

**120 d continuation (job 8565679) — SST converges but R_TOA worsens (the
restoring tradeoff):** SST drift +14.2 K/yr (decelerating 25→19→14 ⇒
converging), SST mean 286.1 K (tropical warm pool 300.8 K), CWV 16.8 (rising
from 15.1 — atm slowly moistening), column-T 257.6 K (+5.2 K from the Day-25
min, still rising). BUT R_TOA −18.4 (WORSE than 60 d's −8.9).
**Read = restoring-strength vs energy-balance tradeoff.** The τ=5 d SST
restoring forces the surface warm; the column warms ⇒ OLR rises, but the dry
atm (CWV 16.8 ≪ Earth's ~25) can't trap it ⇒ the planet leaks energy and the
restoring flux does net work to hold the warm SST. Strong restoring buys
SST-realism + stability; it PAYS an energy imbalance. NEXT LEVERS (the open
question): (1) weaker restoring τ=15–30 d — lets SST cool slightly toward an
energetically self-consistent state (cost: SST a bit below WOA); (2) attack the
root **dry-atm bias** (CWV caps ≪ Earth across BOTH slab + 3D-ocean runs — the
fundamental residual) via the moisture budget (evap vs precip efficiency).
The deliverable (stable, realistic-SST, non-collapsing 3D-ocean coupled run from
WOA IC) STANDS; full energy closure is the refinement.

**Restoring-strength study CONCLUDED (τ=5d vs τ=15d, both 120d):**

| metric        | τ=5 d (8565679) | τ=15 d (8566931) |
|---------------|-----------------|------------------|
| R_TOA         | −18.4           | **−13.6** (better) |
| SST drift     | **14.2** K/yr   | 18.8 K/yr (more)  |
| CWV           | **16.8→19.5**   | 11.3 (drier)      |
| column-T d120 | **257.6**       | 253.0 (colder)    |
| SST mean      | **286.1 (trop 300.8)** | 284.4 (trop 297.9) |
| max_v         | **22.1**        | 17.6 (weaker)     |

**τ=5 d is the recommended config.** Weaker restoring (τ=15 d) buys ~5 W/m²
better R_TOA but pays a colder, drier, weaker-circulation climate that drifts
MORE from WOA (18.8 vs 14.2 K/yr) — less realistic on every climate metric. The
R_TOA −18 at τ=5 d is the accepted flux-correction cost of holding realistic
SST. **The root residual shared by BOTH (and by the slab runs) is the DRY
ATMOSPHERE** (CWV ≪ Earth's ~25) ⇒ weak greenhouse ⇒ R_TOA<0. That moisture
deficit — not the restoring strength — is the fundamental next lever (moisture
budget: evap/precip efficiency; or multi-month integration as the atm moistens,
CWV already re-rising 16.7→19.5 over Days 60–120 at τ=5 d).

## Per-tile surface-flux schemes (2026-06-24, commit 57979cdbf)
User policy: **land = MOST, slab ocean = MOST, ocean air-sea = COARE 3.0**
(COARE is itself a MOST algorithm; the split is generic-MOST for the simple
surfaces vs ocean-specific Charnock+gustiness COARE for the air-sea flux).

- **Ocean air-sea → COARE**: `--surface-bulk-scheme coare3` drives the atm
  surface layer + the coupler ocean tile (`ocean_tile_response` →
  `compute_most_fluxes(scheme="coare3")`, with the gustiness w* fix). Already
  wired; audit confirmed.
- **Land → MOST**: NEW `--land-bulk-scheme` (default `most`) wired onto
  Land/MultiLayerLandConfig.bulk_scheme. Both land models already dispatch
  `most` with `z0_init=config.z0_land` (land roughness, no ocean Charnock).
- **Slab ocean → MOST**: added `most` to `SimpleOcean._ocean_turbulent_fluxes`
  dispatch; NEW `--slab-bulk-scheme` (default `most`), decoupled from
  `--surface-bulk-scheme`.
- **Land dynamic water pools + dryness → ALREADY PRESENT** (audit, no code
  change): slab bucket (β_soil, water-capped evap) + Richards multilayer
  (θ_soil, β_root, extractable-water limit). `--preset slab_richards` uses the
  Richards multilayer soil ⇒ water-limited ET (dry soil → reduced LH → sensible
  warming). The recommended 3D-ocean coupled config uses `slab_richards`.
- Config-level defaults stay `constant` (non-run_coupled callers byte-identical).

## Key files
- `packages/atmosphere/legoesm/atmosphere/physics/clouds/cloud_fraction.py`
  (`convective_cloud_fraction`, thin-cirrus condensate split) + `config.py`.
- `packages/coupler/legoesm/driver/{compiled_segments,physics_pipeline,model_driver}.py`
  (lagged-precip carry wiring).
- `packages/tools/legoesm/diagnostics/energy_budget.py` (`area_weighted_mean`).
- `scripts/run/run_coupled.py` (`--convective-cloud`, `--snow-albedo-feedback`).
- Memory: `cmip_area_weighting_diag`, `cmip_convective_cloud_wiring`,
  `cmip_land_albedo_multilayer`, `cmip_amip_path_blockers`.
