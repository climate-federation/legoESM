# Next implementations — missing land/ocean/cryosphere parameterizations

Gap audit of land / ocean / sea-ice parameterizations vs SOTA models
(NEMO4-SI3, MOM6+CVMix, Oceananigans, CESM-POP; CLM5, JULES, Noah-MP,
ORCHIDEE; CICE6/Icepack). This is a **breadth** list (which schemes exist),
not a fidelity review of existing schemes (see the per-module reviews in
PRs #698 / #708).

**HIGH-priority items are being implemented separately** (Langmuir/Stokes
wave-enhanced ocean mixing; multi-layer land snowpack). This file tracks the
remaining MED / LOW gaps as scoped future work — each should follow the
brainstorm → spec-first contract → codex-adversarial-review → conservation
test → gated (default-off) → PR flow.

Priority key: **M** = medium (clear realism/coupling gain), **L** = low
(niche or coupled-mode-only).

---

## 🌊 Ocean

| Pri | Gap | SOTA reference | Notes / entry point |
|-----|-----|----------------|---------------------|
| M | Prognostic internal-wave-driven mixing (IDEMIX) + lee-wave (Nikurashin) | NEMO/MOM6 IDEMIX, de Lavergne maps | We have only *local* Simmons tidal dissipation in `vertical_mixing/tidal.py`; add a prognostic internal-wave energy budget + non-local dissipation. |
| M | Overflow / dense-water descent (sill plumes, AABW/NADW pathways) | MOM6 overflow, Campin-Goosse | Extends `bbl_adv.py`; downslope dense-plume entrainment. |
| M | Iceberg model (Lagrangian bergs + meltwater/heat spread) | NEMO ICB, MOM6 Berg | SH freshwater/heat budget; couples to `ice_shelf*` + surface_forcing. |
| M | Multi-nutrient / iron-limited BGC (Fe, Si, P, DIC/alkalinity, multi-PFT) | MOM6-COBALT, PISCES | Current `biogeochemistry/npzd.py` is single-N; needed to close the marine carbon cycle. |
| L | Prognostic wave model coupling (WW3) + sea-state-dependent fluxes | coupled ESMs | Provides Stokes drift for Langmuir + roughness. |
| L | Symmetric/frontal submesoscale beyond MLE | MOM6 | `lateral_mixing/mle.py` covers restratification; add SI. |

## 🌱 Land

| Pri | Gap | SOTA reference | Notes / entry point |
|-----|-----|----------------|---------------------|
| M-H | Dynamic vegetation / prognostic phenology (LAI, allocation, mortality) | CLM5-BGC, LPJ, FATES/ED | `carbon/` has pools but no live-vegetation dynamics; required for coupled carbon. |
| M | Nitrogen cycle (C–N coupling: mineralisation, denitrification, N-limited GPP) | CLM5 | GPP in `carbon/carbon_cycle.py` is N-unlimited. |
| M | Groundwater / unconfined aquifer (water table, TOPMODEL baseflow) | CLM5, JULES-TOPMODEL | `richards.py` bottom BC is free-drainage only. |
| M | River routing (runoff → discharge network) | MOSART, RTM, CaMa-Flood | Runoff is applied at the coast, not routed. |
| M | C4 photosynthesis (Collatz) | CLM5, JULES | *Documented gap*: `carbon/stomata.py` runs C4 PFTs through C3 kinetics. |
| M | Two-stream / sunlit-shaded (multilayer) canopy radiation | CLM5 two-big-leaf | Currently big-leaf. |
| L | Wetland inundation + CH4 emissions | CLM5 | |
| L | Fire (combustion, burned area) | CLM5 | |
| L | Crops / land-use / management | CLM5-crop | |
| L | Dust emission (land → atmosphere aerosol) | CLM5 | |
| L | Explicit urban tile + finer sub-grid PFT mosaic | CLM5 | `clm_surface_map.py` blends glacier albedo only. |

## ❄️ Cryosphere (sea ice)

| Pri | Gap | SOTA reference | Notes / entry point |
|-----|-----|----------------|---------------------|
| M | Mushy-layer thermodynamics (prognostic salinity, Turner-Hunke) | CICE6 mushy | We have BL99 (`_future/bitz_lipscomb.py`) + `brine.py`, not the mushy-layer SOTA. |
| M | Form drag (variable air/ocean–ice drag from ridges/ponds/floe edges) | CICE Tsamados 2014 | Momentum exchange uses fixed drag; entry `dynamics.py`. |
| M | Snow-on-ice metamorphism / grain-size (SNICAR-like) albedo aging | CICE Icepack | `snow.py` + `shortwave.py`; albedo feedback. |
| L-M | Floe-size distribution + wave–ice breakup (MIZ, floe-dependent lateral melt) | CICE6 FSD (Roach) | |
| L | Landfast ice (basal/tensile stress, grounding) | CICE landfast | `rheology.py`. |
| L | Sea-ice BGC (ice algae, brine-channel tracers) | CICE zbgc | |
| L | Blowing-snow redistribution on ice | — | |

---

## Suggested ordering (impact × tractability)
1. **C4 photosynthesis** (land) — small, well-specified, closes a documented gap.
2. **Nitrogen cycle** + **dynamic vegetation** (land) — together enable a credible coupled carbon cycle.
3. **Mushy-layer + form drag** (ice) — brings sea-ice to CICE6 level.
4. **IDEMIX internal-wave mixing** + **iron-limited BGC** (ocean).
5. **Groundwater / river routing** (land) — closes the terrestrial water cycle.
