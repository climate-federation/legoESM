# EC-site (flux-tower) evaluation of the two-leaf canopy — runbook

**Audience:** anyone reproducing the offline land site-level validation.
**What it does:** runs the prognostic multilayer land model with the two-leaf canopy
surface scheme at four FLUXNET eddy-covariance sites and compares modelled vs observed
latent heat, sensible heat, gross primary productivity, and soil moisture across the
mean daily cycle, the seasonal cycle, and year to year.

The four sites span climate regimes:

| site | biome | tower height | rooting |
|------|-------|--------------|---------|
| US-MMS | mesic temperate deciduous forest (Morgan Monroe) | 46 m | 2 m |
| DE-Obe | montane evergreen needleleaf forest (Oberbärenburg) | 30 m | 1 m |
| US-Ton | Mediterranean oak savanna (Tonzi, phreatophyte) | 23.5 m | 5 m, 10 m column |
| DE-Hai | humid temperate deciduous forest (Hainich) | 42 m | 2 m |

Per-site tower height and rooting depth live in the `EC_SITE_PHYSICS` table in
`scripts/run/run_ec_site.py`; soil hydraulics come from the per-site texture table
(`scripts/cluster/ec_site/ec_site_soil_texture.csv`). These are the reproducible
defaults — no hand-set environment is required.

---

## A. Reproduce the figures (self-contained — no external data)

The processed model+obs outputs for the four sites are checked in (compressed, ~14 MB
total) under `scripts/validate/ec_site_example_data/`, so the three publication figures
regenerate from the repo alone:

```bash
pip install -e ".[dev]"          # once, if not already installed
JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_publication.py \
    scripts/validate/ec_site_example_data \
    /tmp/ecsite            # output prefix
# -> /tmp/ecsite_energy.png, _carbon.png, _summary.png
```

- **`_energy.png`** — rows = sites; the mean summer daily cycle and the seasonal cycle
  of latent heat (orange) and sensible heat (blue). Markers = raw eddy-covariance
  observations; the shaded band spans up to the energy-balance-closure-corrected value.
- **`_carbon.png`** — gross primary productivity, the latent-heat partition into plant
  transpiration and bare-soil evaporation, and near-surface soil moisture.
- **`_summary.png`** — pooled model–observation scatter (skill vs raw and vs corrected
  obs) and year-to-year means.

Expected pooled daily skill (Nash–Sutcliffe vs raw obs): LE ≈ 0.49, H ≈ 0.56,
GPP ≈ 0.72 (0.62 / 0.68 vs the closure-corrected obs).

The rendered high-resolution figures are also checked in under
`docs/land/figures/ec_site/` (`ec_site_energy.png`, `ec_site_carbon.png`,
`ec_site_summary.png`); see `docs/land/ec_site_figure_captions.md` for captions.

---

## B. Regenerate the model outputs from driver data

To re-run the model (e.g. to add a site or change the canopy config) you need the
DifferBESS half-hourly driver NetCDFs, `<SITE>_driver_v2[_gapfree|_etcorr].nc`
(SW/LW down, T, q, wind, precip, CO2, per-step LAI / Vcmax, and the observed fluxes).
These are large (100–220 MB/site) FLUXNET-derived files and are **not** checked in;
point `--driver-dir` at wherever you have them.

```bash
JAX_ENABLE_X64=1 python scripts/run/run_ec_site_evaluation.py \
    --site DE-Hai --out DE-Hai.nc \
    --driver-dir /path/to/sitelevel/nc \
    --year-lo 2010 --year-hi 2013
```

Run each site (windows used for the checked-in outputs):

| site | `--year-lo --year-hi` |
|------|-----------------------|
| US-MMS | 2015 2018 |
| DE-Obe | 2015 2018 |
| US-Ton | 2015 2018 |
| DE-Hai | 2010 2013 |

then plot the directory of NetCDFs as in section A. The runner writes exactly the
variables the plotter reads (`gpp_mod`, `le_mod`, `le_canopy`, `le_soil`, `h_mod`,
`theta_prof`, and the `*_obs` / `*_obs_corr` counterparts, plus a `valid` mask and
site attrs).

---

## C. What the model does

- **Surface scheme:** two-leaf canopy (`TwoLeafCanopyConfig`) — sunlit/shaded Farquhar
  C3+C4 photosynthesis, Ball-Berry stomata, a Newton leaf-energy-balance closure, and
  a below-canopy Sellers-1992 + Kelvin soil-evaporation resistance.
- **Soil:** the multilayer Richards column + implicit soil-thermal solve; per-site
  van-Genuchten hydraulics from soil texture; free-drainage bottom boundary.
- **Config default:** `stress_b0=False` — the Ball-Berry cuticular intercept is not
  down-regulated by soil-moisture stress, so a live canopy keeps a baseline
  transpiration while a dormant/deciduous canopy self-limits via LAI → 0.
- **Scoring is against RAW eddy-covariance obs**, with the closure-corrected value shown
  as an uncertainty band (raw fluxes under-close the surface energy budget; the model
  closes it, so it sits inside the band).

### Data provenance & caveats

- **The evaluation targets are RAW, never gap-filled.** Gap-filling in the
  `*_driver_v2_gapfree.nc` files applies only to the *meteorological forcing* (the model
  inputs: SW/LW down, T, VPD, wind, pressure, soil T). The *targets* — observed GPP
  (`GPP_DT`), latent heat (`ET`), sensible heat (`H`), and the closure-corrected
  `ET_CORR`/`H_CORR` band — keep their genuine NaN gaps (reader `_obs_masked`: *"no
  gap-fill so the comparison ignores missing tower data"*), and those gaps are excluded
  from every metric. Verified: the target variables are byte-identical between the raw
  `*_driver_v2.nc` and the `*_gapfree.nc` files (0 filled points, 0 value change). The
  `valid` mask further restricts scoring to steps whose *forcing* was genuinely observed
  (pre gap-fill), so filled forcing does not inflate the skill either.
- **⚠ Observed soil moisture (`SWC`) has gap-fill artifacts at some sites** — US-MMS
  (a constant-filled 2018 tail + step), US-Ton (blocky flat segments), DE-Obe (a 2018
  step). `SWC` is used *only* for the prognostic soil initial condition (the first
  finite value, before any artifact) and the observed soil-moisture panel of
  `_carbon.png`; it does **not** drive the fluxes (soil moisture is prognostic). So these
  artifacts affect only the soil-moisture comparison for those sites/years, not the
  LE/H/GPP skill. Treat the `SWC` panel qualitatively at US-MMS/US-Ton/DE-Obe.

## D. Add a site

1. Add the site to `EC_SITE_PHYSICS` in `scripts/run/run_ec_site.py`
   (`{"z_ref": <BADM Reference_height_v>, "root_depth": <m>[, "soil_depth_m": <m>]}`).
2. Run section B for the new site → a NetCDF.
3. Add the site ID to `SITES` (and `SWC_SENSOR_CM` / `PFT_COL`) in
   `scripts/plot/plot_ec_site_publication.py`, then plot.

Note: the model under-predicts GPP/latent heat at **dense, high-LAI wet broadleaf
forests** (a documented coupled-closure bistability — see
`two_leaf_canopy_wet_forest_bistability.md`), so tropical/wet-EBF sites are not yet
suitable panel members.
