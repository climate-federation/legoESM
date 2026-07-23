# EC-site validation figures — captions

Publication captions for the three figures produced by
`scripts/plot/plot_ec_site_publication.py` (see
`ec_site_evaluation_runbook.md` to regenerate). The rendered high-resolution
figures are checked in alongside this file:

| figure | file |
|--------|------|
| 1 — turbulent energy fluxes | `figures/ec_site/ec_site_energy.png` |
| 2 — carbon, water partition, soil moisture | `figures/ec_site/ec_site_carbon.png` |
| 3 — pooled skill + interannual means | `figures/ec_site/ec_site_summary.png` |

All fluxes are the offline
prognostic multilayer land model with the **two-leaf canopy** surface scheme,
forced by half-hourly FLUXNET/DifferBESS drivers at four eddy-covariance sites:

| site | biome | period |
|------|-------|--------|
| US-MMS | mesic temperate deciduous forest (Morgan Monroe, US) | 2015–2018 |
| DE-Obe | montane evergreen needleleaf forest (Oberbärenburg, DE) | 2015–2018 |
| US-Ton | Mediterranean oak savanna (Tonzi, US) | 2015–2018 |
| DE-Hai | humid temperate deciduous forest (Hainich, DE) | 2010–2013 |

Throughout, **lines are modelled**, **markers are the raw eddy-covariance
observations**, and a **shaded band spans up to the energy-balance-closure-
corrected observation**. Raw turbulent fluxes generically under-close the surface
energy budget; the model closes it, so the honest target is the raw measurement
with the closure correction shown as an uncertainty envelope rather than silently
applied.

---

## Figure 1 — `*_energy.png`: turbulent energy fluxes

**Modelled and observed latent (orange) and sensible (blue) heat flux at four
FLUXNET sites.** Rows are sites; the left column is the mean summer (JJA) diurnal
cycle and the right column is the mean seasonal cycle. Lines are the two-leaf
canopy model; markers are raw half-hourly eddy-covariance observations averaged
into the same composite; the shaded band spans up to the energy-balance-closure-
corrected value. The model reproduces the amplitude and phase of the daytime
latent- and sensible-heat partition across the mesic (US-MMS, DE-Hai),
needleleaf (DE-Obe) and water-limited savanna (US-Ton) regimes, with the raw
observations sitting inside — or just below — the modelled curve as expected from
the closure deficit.

## Figure 2 — `*_carbon.png`: carbon, water partition and soil moisture

**Gross primary productivity, the latent-heat partition, and near-surface soil
moisture.** Rows are sites. Column 1: modelled vs observed GPP (mean seasonal
cycle); column 2: the modelled latent-heat flux split into canopy transpiration
(green) and bare-soil evaporation (orange), showing that the model routes water
loss predominantly through stomatal transpiration in the growing season with a
soil-evaporation background; column 3: modelled vs observed volumetric soil
moisture at the shallowest observed sensor depth. GPP phenology and magnitude
track the observations at the temperate forests; at the Mediterranean savanna the
model captures the spring green-up and summer down-regulation set by soil-moisture
stress.

## Figure 3 — `*_summary.png`: pooled skill and interannual means

**Pooled model–observation skill (top) and per-site year-to-year means
(bottom).** Top row: daily-mean model-vs-observation scatter pooled across all
four sites for latent heat, sensible heat and GPP, against the raw observations.
Model and observations are **co-sampled** (compared only on timesteps where both
are available) so a discrepancy reflects model error and not a sampling mismatch;
the 1:1 line and the Nash–Sutcliffe efficiency (vs raw and, for LE/H, vs the
closure-corrected obs) are annotated. Bottom row: **one panel per site**, each on
its own year axis so that sites with different observational coverage are not
forced onto a shared timeline. Each panel shows annual-mean latent heat (orange)
and sensible heat (blue) on the left axis and GPP (green) on the right axis;
annual means are co-sampled and **month-stratified** (equal weight per month) so a
season-heavy sample does not bias them; dashed open markers are observations, solid
filled markers are the model, and the shaded band is the energy-balance-closure
envelope on the energy fluxes.

Pooled daily skill (Nash–Sutcliffe efficiency vs **raw** obs, co-sampled):
**LE ≈ 0.58**, **H ≈ 0.57**, **GPP ≈ 0.80** (LE / H are 0.65 / 0.68 vs the
closure-corrected obs).

---

### Notes on scope

The panel members are temperate/Mediterranean sites where the two-leaf canopy is
well behaved. The model under-predicts GPP and latent heat at **dense, high-LAI
wet broadleaf forests** (a coupled leaf-energy-balance / stomatal bistability
documented in `two_leaf_canopy_wet_forest_bistability.md`); wet evergreen-broadleaf
and tropical sites are therefore not yet included. US-Var (Vaira Ranch grassland)
is likewise excluded: its prescribed satellite LAI does not drop to zero during
the drought-deciduous dry season, so the model over-transpires late in the season
— a forcing-data limitation, not a scheme error, and one we deliberately do not
mask by tuning the cuticular conductance.
