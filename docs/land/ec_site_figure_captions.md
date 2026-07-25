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
forced by FLUXNET/DifferBESS drivers (half-hourly; hourly at US-MMS) at four eddy-covariance sites:

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

Publication caption, exactly as used in the manuscript:

> **Land-surface validation against eddy-covariance flux towers.** The offline land
> model with the two-big-leaf canopy scheme (sunlit/shaded) at four FLUXNET sites
> spanning temperate and Mediterranean regimes: US-MMS (mesic deciduous forest),
> DE-Obe (montane needleleaf forest), US-Ton (oak savanna) and DE-Hai (humid
> deciduous forest). One row per site (**a–t**): the mean summer (JJA) diurnal
> cycle and the mean seasonal cycle of latent (orange) and
> sensible (blue) heat; the GPP seasonal cycle; the modelled latent-heat partition
> into canopy transpiration (green) and bare-soil evaporation (orange) with the
> observed total; and soil moisture at the observed sensor depth and two deeper
> model layers. Lines are modelled, markers are raw eddy-covariance observations,
> and shaded bands span up to the energy-balance-closure-corrected value. Model and
> observations are co-sampled, i.e. compared only on timesteps where both are
> available. Pooled daily Nash–Sutcliffe efficiency against the raw observations is
> 0.58 (latent heat), 0.57 (sensible heat) and 0.80 (GPP); against the
> closure-corrected observations the energy fluxes score 0.68 and 0.68. Raw
> eddy-covariance fluxes under-close the surface energy budget by 22–34 % at these
> sites, so the modelled turbulent fluxes lie near the centre of the
> closure-uncertainty envelope (+17 % against raw, −17 % against corrected).

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
