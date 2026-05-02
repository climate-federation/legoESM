# Forcing protocol & coarse-resolution circulation: literature review and Phase 4(d) recommendation

**Author**: ocean-model-expert subagent
**Date**: 2026-05-02
**Scope**: research-only. No source edits. Companion to
`realistic_geometry_phase4_results.md` (Phase 4(c) outcome) and
`realistic_geometry_lat_lon_plan.md`.

## Purpose

Phase 4(c) produced our first numerically clean 50-yr realistic-geometry
spinup (max|u| equilibrated at ~1.2 m/s, no NaN, Southern Ocean MOC
cell visible). But the circulation magnitudes are unphysical:
**MOC max ≈ 180 Sv** (observed AMOC ~15–20 Sv; observed Pacific
overturning ~10 Sv) and **BT max ≈ 280 Sv** (observed ACC at Drake
~140 Sv). There is no NADW formation pathway. This document collects
literature evidence on what coarse forced ocean simulations actually
do, why our numbers are off, and which single change would buy the
most.

## 1. Forcing protocols at coarse resolution

The community's standardised forced-ocean intercomparison protocols
have all converged on roughly the same set of surface fields. The
relevant papers:

- **CORE-I / CORE-II (Large & Yeager 2004; 2009; *Climate Dynamics*
  33)** — "Coordinated Ocean-ice Reference Experiments". Defined a
  reanalysis-derived dataset (~6-hour 10m winds + radiation + 2m T,q
  + monthly P, R) with **bulk-formula turbulent fluxes** computed
  *online* from prognostic SST. Five 62-yr cycles is the standard
  spinup. **Griffies et al. (2009, *Ocean Modelling*)** is the
  protocol paper.
- **OMIP-1 (Griffies et al. 2016, *GMD*)** — CORE-II forcing
  formalised for CMIP6.
- **OMIP-2 / JRA55-do (Tsujino et al. 2018 *Ocean Modelling*; 2020
  *GMD*)** — replacement for CORE-II using JRA55 atmospheric
  reanalysis. Higher-frequency winds (3-hourly) → more realistic
  storm tracks, ~25 % stronger AMOC than CORE-II in most models
  (Danabasoglu et al. 2014, *Ocean Modelling*; Tsujino et al. 2020).

Both protocols supply **2D fields**: zonal+meridional wind stress (or
10 m wind for online bulk formula), net heat flux components (SW, LW,
turbulent), precipitation, evaporation (or 2 m q for online bulk),
river runoff (Dai & Trenberth 2002). At 5° the grid cannot resolve
storm tracks anyway, so the *pattern* added by realistic 2D τ matters
more than the spectral content.

### What each "more-realistic" forcing addition buys at 5°

| Addition | Mechanism | Expected effect at 5° |
|---|---|---|
| Realistic 2D τ (vs zonal-mean 2-belt) | Curl pattern → realistic gyre boundaries; stronger τ in Southern Ocean → realistic ACC | Subtropical/subpolar gyre topology emerges; Drake transport drops toward observed once western boundary frictional sink balances input |
| E−P (freshwater) | Halocline in N Pacific (kills NADW there) and N Atlantic (modulates DWF). N. Atlantic salt budget is the main AMOC control. | NADW formation localises to N. Atlantic; AMOC sign correct. *Without* salinity, AMOC site is essentially random. |
| River runoff | Coastal freshening, Arctic export | Modest at 5°; Amazon/Bering buoyancy plumes unresolved |
| Net heat flux (vs SST restoring) | Allows model to make its own SST, doesn't force unphysical convergence | Reduced spurious sinking; restoring forces unrealistic deep heat sink because surface T is held against deep T_initial |
| Sea-ice coupling | Brine rejection → DWF at marginal ice zones (Labrador, Weddell) | Critical for realistic AMOC source water |

The single most important missing field for AMOC realism is
**salinity forcing (E-P)**. Without it, the system has no halocline,
no salt advection feedback (Stommel 1961, *Tellus*), and DWF location
is dictated by whichever pole gets coldest first.

## 2. Mechanistic diagnosis of MOC = 180 Sv

Our setup with zonal-mean τ + cosine SST restoring + no salinity is
essentially the **Wolfe & Cessi (2010, JPO 40)** idealized
configuration moved onto realistic ETOPO. Wolfe & Cessi's own runs
in a flat-bottom rectangular basin with κ_v=1e-5, comparable diffusion,
gave MOC ~10 Sv. Why is ours 18× larger?

### Three coupled drivers

1. **SST restoring + linear EOS + no surface salt = unbounded buoyancy
   sink.** With cosine restoring T*(lat) toward 25→0 °C and τ_T=30 d,
   the implied surface heat flux at high latitudes is ~hundreds of
   W/m² wherever upwelled deep water (T ≈ 2 °C) reaches the surface.
   This drives essentially infinite sinking — the more deep water
   that comes up, the more the restoring fixes it back to T*, the
   more sinking, ad infinitum. With a *prognostic* heat flux (net
   bulk formula), the surface relaxes back toward T* by reducing the
   air-sea ΔT, breaking the feedback. Cessi & Otheguy (2003 *JPO 33*)
   showed analytically that under fixed-T (= τ_T → 0) restoring,
   MOC ∝ (κ_v · L · ΔT)^(1/3) with no upper bound from forcing
   amplitude; under fixed-flux it saturates at the prescribed flux.
   Our τ_T=30 d is "fixed-T" in the relevant limit.
2. **κ_v = 1e-5 + GM/Redi κ=800 sets too-large diffusive overturning.**
   The Bryan (1987 *JPO 17*) scaling MOC ∝ κ_v^(2/3) gives ~12 Sv at
   κ_v=1e-5, but coarse z-coordinate models get *additional* spurious
   diapycnal mixing from advection of ~1e-5 m²/s (Griffies, Pacanowski
   & Hallberg 2000, *Ocean Modelling 2*). Doubling effective κ_v
   gives MOC ×1.6 by Bryan scaling. Park & Bryan (2000, *JPO 30*)
   diagnosed effective κ_v in coarse z-coord models at 5–10× nominal.
3. **Closed Arctic + idealised wind concentrates surface buoyancy
   loss in Southern Ocean and N Atlantic uniformly.** With no
   freshwater cap on N Pacific, our model is free to sink there too;
   our 180 Sv is dominantly Southern Ocean (consistent with the
   Phase 4(c) MOC plot showing the SO cell, not NADW). The W&C 2010
   2-basin idealised geometry doesn't permit this because they have
   only one basin.

### BT max = 280 Sv (vs ACC ~140 Sv)

ACC strength is set by Drake Passage geometry + Southern Ocean wind
stress (Munk-Palmen 1951; Marshall & Radko 2003 *JPO 33*). Our ACC
overshoot has two likely causes: (a) idealised zonal-mean τ peaks at
~50°S without the realistic decay over Drake, putting too much
momentum input where it can't escape via form stress; (b) at 5° the
Drake sill is 1 cell wide and bottom form stress (the dominant ACC
sink in reality, ~80 % of zonal-momentum balance per Masich, Chereskin
& Gille 2015 *JPO 45*) is grossly under-represented. The first is
fixable by realistic τ; the second is a resolution limit.

## 3. Coarse-resolution parameter space (published values)

| Model / config | Resolution | A_h (m²/s) | A_v (m²/s) | κ_v (m²/s) | κ_GM (m²/s) | κ_Redi (m²/s) | Reference |
|---|---|---|---|---|---|---|---|
| **legoESM Phase 4(c)** | **5°** | **2e5 cos²(lat)** | **1e-3** | **1e-5** | **800** | **800** | this branch |
| GFDL-CM2.0 OM3 | 1° (lat-lon) | 1e5 → 5e4 trop | 1e-4 | 0.1–1.4e-5 (Bryan-Lewis) | 600 (depth-dep) | 600 | Griffies et al. 2005 *Ocean Modelling* |
| MOM6-OM4 (CMIP6) | 0.25° | Smag biharm | online (KPP) | 1.5e-5 + tides | 600 (Visbeck) | 600 | Adcroft et al. 2019 *JAMES* |
| NEMO ORCA1 | 1° (tripole) | -1.2e10 (biharm) | 1.2e-4 (Pacanowski-Philander) | 1.5e-5 + tide | 1000 | 1000 | Madec et al. 2022 |
| NEMO ORCA2 | 2° | 4e4 | 1e-4 | 1.2e-5 + tide | 2000 | 2000 | Madec et al. 2008 |
| CESM-POP (CCSM4) | nom. 1° | 1.5e7 (biharm) | KPP | KPP + 0.1e-5 bg | 300–3000 (visb) | 300–3000 | Danabasoglu et al. 2012 *J Climate* |
| FAMOUS (Smith et al. 2008) | 3.75°×2.5° | 6e4 | 1e-4 | 1e-4 (!) | 2000 | 2000 | Smith, Gregory, Osprey 2008 *GMD* |
| UVic ESCM v2.9 | 3.6°×1.8° | 2e5 | 0 (no momentum eqn — frictional geostrophic) | 0.15–1.3e-4 (Bryan-Lewis) | 800 | 800 | Weaver et al. 2001 *Atmos-Ocean* |
| PlaSim-LSG | ~5° | n/a (LSG = uplifted SW) | n/a | ~1e-4 | implicit | implicit | Maier-Reimer 1993 |

**Where we sit**: A_h = 2e5 with cos²(lat) is in the ORCA2 / UVic
range, not unreasonable. **κ_v = 1e-5 is on the LOW end** for 5°.
Coarse models traditionally use 1e-4 (FAMOUS, UVic-equivalent) or
Bryan-Lewis profiles peaking at ~1.3e-4 in the abyss to *compensate*
for the spurious-mixing-induced lack of stratification. **Our κ_v
is set as if we were a 1° model** — too small for 5° physics.

κ_GM = 800 is on the low end (most coarse configs use 1000–2000;
Visbeck flow-dependent kappa typically peaks at 2000–3000). Higher
κ_GM flattens isopycnals more aggressively, weakening the zonal-mean
density gradient that drives the residual MOC overshoot via thermal
wind. Marshall & Radko (2003) and Gent (2011 *Annu Rev Mar Sci*)
both discuss this as the "κ_GM controls AMOC" lever in coarse models.

## 4. Vertical resolution at 16 levels — what breaks

OMIP-2 standard (Tsujino et al. 2020) is **75 levels** with 1 m
surface and ~200 m at depth. CESM-POP runs 60. NEMO ORCA1 runs 75.
Even FAMOUS at 3.75° uses 20 levels and notes (Smith et al. 2008)
that the AMOC is highly sensitive to vertical resolution above 20.

**At 16 levels (5° = ~310 m mean), what breaks**:

1. **Sill overflows are unresolved**. Denmark Strait sill is ~600 m
   deep with 200 m sill-to-floor; we have <1 active level there.
   Legg et al. (2009 *BAMS 90*) showed that sill overflows in
   coarse z-coord models without parameterised entrainment are
   *systematically too dense* and sink to the bottom without
   entraining ambient water. That makes NADW too dense, biasing AMOC
   pathway.
2. **Mixed layer depth resolution**. Polar wintertime MLD reaches
   1500–2500 m in Labrador / Weddell. At 16 levels, the MLD is at
   most 1-cell-resolved, so deep convection is artificially "all or
   nothing": either the column mixes to a fixed level or it doesn't.
3. **Pycnocline thickness is set by 2–3 levels** instead of 8–10,
   which biases isopycnal slopes and hence GM-induced overturning.
4. **Bryan (1984 *JPO 14*)** noted that deep ocean adjustment
   timescales are level-resolution-sensitive: doubling vertical
   levels in the deep ~halves the time-to-equilibrium because the
   diffusive scale is set by Δz²/κ_v.

For our purposes the most acute problem is (2): without realistic
DWF, AMOC has no source. This is *partially* compensable by the
convection scheme (enhanced diffusion when N²<0) but at 16 levels
the source location is poorly determined.

## 5. Spinup protocol: IC choice + ramps

Standard practice (CORE-II protocol; Stouffer et al. 2004 *J Climate*):

- **Initial T, S from WOA / Levitus climatology**, not analytic
  exp(z) on a uniform-S column. Levitus mean state already has the
  thermocline, halocline, and AABW signal — the model adjusts from
  near-equilibrium rather than building stratification from rest.
  With analytic IC + no salinity, our T/S has no NADW progenitor at
  all.
- **Forcing ramp** is *not* universal in CORE-II (which starts from
  Levitus and rams immediately), but is standard for cold-start
  experiments: linear ramp τ from 0 over the first year (Bryan 1984;
  Sausen, Barthel & Hasselmann 1988 *Climate Dynamics* — "tracer
  acceleration" / "asynchronous coupling").
- **Asynchronous / accelerated coupling** (Bryan 1984) — step the
  deep tracer with a larger dt than the surface; standard in 1980s-
  90s ocean spinups, less common now but Stouffer et al. (2004) note
  it cuts spinup time ~10×. Not differentiability-friendly.
- **5 × 62-yr cycles** (CORE-II convention) — the model is judged
  spun-up when surface fields between cycle 4 and cycle 5 differ by
  < 0.1 °C SST, < 1 Sv AMOC, < 0.05 PSU SSS, etc.

For us, the cheap-but-high-leverage step is **WOA/Levitus IC**.
Our Phase 4(b/c) MOC plots show "no NADW pathway forming" because
our IC has no preformed deep water mass to maintain.

## 6. Straits at 5°

Resolution-driven priorities (cf. our `realistic_geometry_topology_fixes.md`):

- **Drake Passage**: critical, marginally resolved (1 cell). Without
  it the ACC closes off and the model is grossly wrong. Our enforced
  3000 m depth + 300 km width is essentially the only choice.
- **Indonesian Throughflow**: dynamically critical for global heat
  budget (Gordon 2005 *Oceanography 18*). Our forced-open 200 km/1500 m
  enforcement is correct.
- **Denmark Strait + Faroe-Bank**: NADW overflow gateway. At 5° our
  enforced 600/800 m depth gives the *signal* but not the entraining
  density structure. The Legg et al. 2009 overflow scheme would
  matter here in production but is out of scope.
- **Bering**: enforced at 40 m. Helsinki et al. (Hu et al. 2012
  *J Climate*) and Wadley & Bigg (2002 *QJRMS*) both show the
  Bering closure/opening is a *first-order* control on N Pacific
  freshwater export and hence on N Atlantic SSS, which controls AMOC.
  In z*, a 40 m water column at lat 65° has a *single* level, ~80 %
  compressed at our default Δz_surface=20 m. **At 5°, our enforced
  Bering is more harmful than helpful**: it admits a numerically
  marginal Pacific→Arctic salt flux that is poorly resolved. UVic
  and FAMOUS (both 3.75–5°) routinely close it.

**Recommendation for 5°**: close Bering. Keep Drake, ITF, Denmark,
Faroe.

## 7. Highest-leverage single change — ranked top 3

The Phase 4(c) MOC=180 / BT=280 problem is **forcing-limited**, not
numerics-limited (the diagnostic ladder closed numerics in the
2026-05-01 evening session). The three highest-leverage fixes:

### #1. Switch from SST restoring to a prescribed net heat flux (or much weaker restoring τ_T)
This breaks the unbounded buoyancy-sink feedback (Cessi & Otheguy
2003). Even keeping idealised cosine spatial structure, replacing
the τ_T=30 d Newton restoring with a *flux* gives the model agency
to set its own surface T and stops the artificial deep heat sink
that drives MOC overshoot. Estimated effect: MOC drops 2–3× to
~60–80 Sv. Cheap to implement (existing surface forcing config
already supports prescribed flux). **This is the single highest-
information-gain change.**

### #2. Add idealised salinity forcing (E−P pattern + restoring or prescribed)
Without a salt budget the model cannot make NADW realistically.
A simple zonal-mean E−P from Da Silva et al. (1994) climatology, or
even a cosine "evaporative tropics, fresh poles" idealisation
(Stommel 1961), turns on the AMOC source-water formation pathway.
Estimated effect: AMOC localises to N. Atlantic; magnitude drops as
the SO cell weakens. Could turn the SO-dominated 180 Sv into a
recognisable two-cell pattern. Medium implementation cost (existing
surface forcing has tau_S; need to add E-P pattern + linear EOS
upgrade to use β_S).

### #3. Increase κ_v toward 1e-4 (Bryan-Lewis profile) and tune κ_GM up to 1500–2000
Match coarse-resolution literature (FAMOUS, UVic, ORCA2) instead of
1° conventions. Bryan-Lewis profile (κ_v = κ_top + (κ_bottom −
κ_top) · arctan(α(z−z_ref))/π) gives surface ~0.3e-4, abyss ~1.3e-4.
This compensates for under-resolved tidal mixing + spurious advective
diffusion. Higher κ_GM flattens isopycnals more aggressively,
draining APE that would otherwise drive MOC overshoot.
Estimated effect on MOC: smaller than #1/#2 (~10–30 % reduction)
but improves the deep stratification, which is currently too weak.
Cheapest of the three — purely a config change.

**Combined effect (#1 + #2 + #3)**: a 2-pole 2-basin Atlantic-like
overturning with magnitudes in the 20–40 Sv range is plausible. Not
quantitatively right (we're still 5° / 16 levels / no sea-ice / no
overflow scheme) but **qualitatively recognisable**, which is the
Phase 4 goal.

## Recommended Phase 4(d) experiment matrix

Ordered by info-gain ÷ cost. Each row is a 50-yr spinup at the same
numerics as Phase 4(c). Wall time budget ~4 h each on the laptop.

| # | Change vs Phase 4(c) | Mechanism | Cost (impl) | Expected deliverable |
|---|---|---|---|---|
| **A** | **κ_v = Bryan-Lewis (0.3 → 1.3e-4); κ_GM = 1500** | Match coarse-res literature; drain APE; compensate spurious mixing | ~half day | First test of "is it just config?". If MOC drops to ~80 Sv, lever ranking confirmed. If not, forcing dominates. |
| **B** | A + replace SST restoring with prescribed cosine *heat flux* (Q*(lat) tuned so 4(c) equilibrium SST is preserved at zero anomaly) | Removes unbounded buoyancy-sink feedback (Cessi-Otheguy 2003) | ~1 day (plumb prescribed Q in surface forcing) | Test of #1 lever. Expected MOC ~60 Sv, BT unchanged. |
| **C** | B + add idealised E−P forcing (cosine pattern, tropical evap / polar precip), enable β_S in linear EOS | Activates salt budget; localises DWF | ~1.5 days (surface S flux + linear EOS β_S) | Test of #2 lever. Expected: 2-pole MOC, NADW pathway, AMOC ~20–30 Sv, recognisably Atlantic-flavoured. **This is the headline Phase 4(d) deliverable.** |
| **D** *(optional, gated on C)* | C + close Bering + drop to 30 levels | Resolve sill overflows + remove a problematic strait | ~2 days | If C still has wrong AMOC pathway, level resolution is the bottleneck. Otherwise skip. |

**Recommended execution order**: A → B → C, abandoning the chain
early if A alone produces a recognisable circulation (unlikely
given the analysis above but cheap to verify).

## Open questions

1. Is there a published *coarse* (≥3°) z-coordinate forced ocean
   spinup that achieves AMOC ~15 Sv from rest with idealised forcing,
   or is the literature unanimous that you need at least bulk-formula
   surface fluxes + E−P to reach observed magnitudes? Our search
   suggests the latter but Wolfe & Cessi 2010, 2011, 2014 series uses
   restoring + idealised wind in 2-basin geometry to reach ~10 Sv —
   we should verify whether their τ_T is closer to flux limit
   (τ_T ~ 1 yr) than ours.
2. The Park & Bryan (2000) finding that effective κ_v is 5–10× nominal
   in z-coord models — has this been re-quantified for partial-cell
   z* implementations like ours? Adcroft et al. 2008 partial cells
   were claimed to reduce spurious mixing but I'm not aware of a
   modern recheck.

## References (cited, with year + journal)

- Adcroft, A., et al. (2019). "The GFDL Global Ocean and Sea Ice Model OM4.0…", *JAMES* 11.
- Bryan, K. (1984). "Accelerating the convergence to equilibrium of ocean-climate models", *JPO* 14.
- Bryan, F. (1987). "Parameter sensitivity of primitive equation ocean GCMs", *JPO* 17.
- Cessi, P. & Otheguy, P. (2003). "Oceanic teleconnections: remote response to decadal wind forcing", *JPO* 33.
- Da Silva, A., Young, A., Levitus, S. (1994). *Atlas of Surface Marine Data 1994*, NOAA Atlas NESDIS.
- Dai, A. & Trenberth, K. (2002). "Estimates of freshwater discharge from continents", *J. Hydrometeorology* 3.
- Danabasoglu, G., et al. (2012). "The CCSM4 Ocean Component", *J. Climate* 25.
- Danabasoglu, G., et al. (2014). "North Atlantic simulations in CORE-II forced ocean models", *Ocean Modelling* 73.
- Gent, P. (2011). "The Gent–McWilliams parameterization: 20/20 hindsight", *Annu. Rev. Mar. Sci.* 3.
- Gordon, A. (2005). "The Indonesian Seas and the Indonesian Throughflow", *Oceanography* 18.
- Griffies, S., et al. (2005). "Formulation of an ocean model for global climate simulations", *Ocean Modelling* 8.
- Griffies, S., et al. (2009). "Coordinated Ocean-ice Reference Experiments (COREs)", *Ocean Modelling* 26.
- Griffies, S., et al. (2016). "OMIP contribution to CMIP6", *GMD* 9.
- Griffies, S., Pacanowski, R. & Hallberg, R. (2000). "Spurious diapycnal mixing", *Ocean Modelling* 2.
- Hu, A., et al. (2012). "Bering Strait throughflow and the AMOC", *J. Climate* 25.
- Large, W. & Yeager, S. (2009). "The global climatology of an interannually varying air-sea flux dataset", *Climate Dynamics* 33.
- Legg, S., et al. (2009). "Improving oceanic overflow representation in climate models", *BAMS* 90.
- Madec, G., et al. (2022). *NEMO ocean engine*, Notes du Pôle de Modélisation IPSL.
- Marshall, J. & Radko, T. (2003). "Residual-mean solutions for the ACC", *JPO* 33.
- Masich, J., Chereskin, T. & Gille, S. (2015). "Topographic form stress in the SO", *JPO* 45.
- Park, Y.-G. & Bryan, K. (2000). "Comparison of thermally driven circulations from a depth-coordinate and an isopycnal-coordinate model", *JPO* 30.
- Sausen, R., Barthel, K., Hasselmann, K. (1988). "Coupled ocean-atmosphere models with flux correction", *Climate Dynamics* 2.
- Smith, R., Gregory, J., Osprey, A. (2008). "A description of the FAMOUS climate model", *GMD* 1.
- Stommel, H. (1961). "Thermohaline convection with two stable regimes of flow", *Tellus* 13.
- Stouffer, R., et al. (2004). "Atlantic ocean response to a slowdown of the THC in a coupled GCM", *J. Climate* 17.
- Tsujino, H., et al. (2018). "JRA-55 based surface dataset for driving ocean-sea-ice models (JRA55-do)", *Ocean Modelling* 130.
- Tsujino, H., et al. (2020). "Evaluation of global ocean–sea-ice model simulations based on the experimental protocols of OMIP phase 2", *GMD* 13.
- Weaver, A., et al. (2001). "The UVic Earth System Climate Model", *Atmosphere-Ocean* 39.
- Wolfe, C. & Cessi, P. (2010). "What sets the strength of the middepth stratification…", *JPO* 40.
