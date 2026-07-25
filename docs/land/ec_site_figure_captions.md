# EC-site validation figures — captions

Publication caption for the figure produced by
`scripts/plot/plot_ec_site_publication.py` (see
`ec_site_evaluation_runbook.md` to regenerate). The rendered high-resolution
figures are checked in alongside this file:

| figure | file |
|--------|------|
| combined validation figure | `figures/ec_site/ec_site_combined.png` |

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

## The combined figure — `*_combined.png`

**Land-surface validation against eddy-covariance flux towers.** Panels are
labelled (a)–(w). **(a)–(c)** Pooled model–observation skill: daily-mean scatter
across all four sites for latent heat, sensible heat and GPP against the raw
observations, with the 1:1 line, R², Nash–Sutcliffe efficiency (vs raw and, for the
energy fluxes, vs the closure-corrected obs) and mean bias annotated; points are
coloured by site. **Remaining rows, one per site** (US-MMS, DE-Obe, US-Ton,
DE-Hai), each with five panels: the mean summer (JJA) **diurnal cycle** and the
mean **seasonal cycle** of latent (orange) and sensible (blue) heat; the **GPP**
seasonal cycle; the modelled **latent-heat partition** into canopy transpiration
(green) and bare-soil evaporation (orange) with the observed total; and **soil
moisture** at the observed sensor depth and two deeper model layers.

Model and observations are **co-sampled** throughout — compared only on timesteps
where both are available — so a model–observation discrepancy reflects model error
rather than a sampling mismatch.

Pooled daily skill (Nash–Sutcliffe efficiency vs **raw** obs, co-sampled):
**LE ≈ 0.58**, **H ≈ 0.57**, **GPP ≈ 0.80** (LE / H are 0.65 / 0.68 vs the
closure-corrected obs). Raw eddy-covariance fluxes under-close the surface energy
budget by ~22–34 % at these sites; the model closes it by construction, so its LE/H
bias is **+17 %** against the raw obs and **−17 %** against the closure-corrected
obs — i.e. the model sits inside, and near the centre of, the closure-uncertainty
envelope. GPP has no closure ambiguity and carries a **+6 %** pooled bias.

**Interannual variability is deliberately not shown.** With only two to four years
per site, year-to-year means are dominated by sampling noise and cannot validate
interannual variability; the defensible claims here are the diurnal, seasonal and
pooled-daily skill.

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
