# EC-site (flux-tower) evaluation of the two-leaf canopy — runbook

**Audience:** anyone reproducing the offline land site-level validation.
**What it does:** runs the prognostic multilayer land model with the two-leaf canopy
surface scheme at four FLUXNET eddy-covariance sites and compares modelled vs observed
latent heat, sensible heat, gross primary productivity, and soil moisture across the
mean daily cycle and the seasonal cycle.

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

**Tower height is measured; rooting depth is a modeling choice.** The *tower height*
column (`z_ref`) is the observed FLUXNET BADM `Reference_height_v`. The *rooting*
column (`root_depth`), by contrast, is a **modeling choice** — the e-folding depth of
the *prescribed* root-density profile (`root_frac = exp(-z/root_depth)`), set to a
value plausible for the site's PFT and soil profile (deeper for phreatophytes that tap
deep soil / weathered-bedrock water through the dry season) and validated against the
fluxes, **not** a measured rooting depth. The `10 m column` for US-Ton and
`soil_depth_m` are likewise column-depth choices. Treat `root_depth` / `soil_depth_m`
as tunable modeling inputs, not observations.

---

## A. Reproduce the figures (self-contained — no external data)

The processed model+obs outputs for the four sites are checked in (compressed, ~14 MB
total) under `scripts/validate/ec_site_example_data/`, so the publication figure
regenerates from the repo alone:

```bash
pip install -e ".[dev]"          # once, if not already installed
JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_publication.py \
    scripts/validate/ec_site_example_data \
    /tmp/ecsite            # output prefix
# -> /tmp/ecsite_combined.png
```

- **`_combined.png`** — one figure, panels (a)–(t): one row per site with the mean
  summer daily cycle and seasonal cycle of latent/sensible heat, the GPP seasonal
  cycle, the latent-heat partition into transpiration and bare-soil evaporation, and
  soil moisture. Markers = raw eddy-covariance observations; shaded bands span up to
  the energy-balance-closure-corrected value. The pooled skill numbers below are
  printed by the plotter. Interannual and pooled-scatter panels are deliberately
  omitted (2–4 years per site is too few to validate interannual variability).

Expected pooled daily skill (Nash–Sutcliffe vs raw obs, on co-sampled timesteps —
model and observations compared only where both are available): LE ≈ 0.58,
H ≈ 0.57, GPP ≈ 0.80 (LE / H are both 0.68 vs the closure-corrected obs). Model and
observations are co-sampled everywhere (compared only where both are available), so a
discrepancy reflects model error rather than a sampling mismatch.

The rendered high-resolution figure is also checked in at
`docs/land/figures/ec_site/ec_site_combined.png`; see
`docs/land/ec_site_figure_captions.md` for the caption.

---

## B. Re-run the model from the checked-in trimmed drivers

The trimmed driver NetCDFs are **checked in** under
`scripts/validate/ec_site_example_drivers/` — each is the analysis-window subset of
the DifferBESS half-hourly driver, keeping only the variables the model drives on
and is evaluated against (forcing + observed targets + closure band + gap-fill
flags), at the source float64 precision, fully annotated (units / long_name /
description + FLUXNET/DifferBESS provenance). So the model itself re-runs from the
repo alone — no external data:

```bash
JAX_ENABLE_X64=1 python scripts/run/run_ec_site_evaluation.py \
    --site DE-Hai --out DE-Hai.nc \
    --driver-dir scripts/validate/ec_site_example_drivers \
    --year-lo 2010 --year-hi 2013
```

Run each site over its window, then plot the directory of NetCDFs as in section A:

| site | `--year-lo --year-hi` |
|------|-----------------------|
| US-MMS | 2015 2018 |
| DE-Obe | 2015 2018 |
| US-Ton | 2015 2018 |
| DE-Hai | 2010 2013 |

This reproduces the checked-in `ec_site_example_data/*.nc` outputs (and hence the
figures) bit-for-bit. The runner writes exactly the variables the plotter reads
(`gpp_mod`, `le_mod`, `le_canopy`, `le_soil`, `h_mod`, `theta_prof`, the `*_obs` /
`*_obs_corr` counterparts, plus `valid` / `reverted` masks and site attrs).

The trimmed drivers are rebuilt from the full external DifferBESS files (100–220
MB/site, not checked in) with `scripts/data/build_ec_site_example_drivers.py`
(`--src-dir <dir with *_driver_v2_gapfree.nc>`) — use that to add a site or widen a
window.

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
  finite value, before any artifact) and the observed soil-moisture panels; it does **not** drive the fluxes (soil moisture is prognostic). So these
  artifacts affect only the soil-moisture comparison for those sites/years, not the
  LE/H/GPP skill. Treat the `SWC` panel qualitatively at US-MMS/US-Ton/DE-Obe.

## D. Add a site

1. Add the site to `EC_SITE_PHYSICS` in `scripts/run/run_ec_site.py`
   (`{"z_ref": <BADM Reference_height_v>, "root_depth": <m>[, "soil_depth_m": <m>]}`).
2. Run section B for the new site → a NetCDF.
3. Add the site ID to `SITES` (and `SWC_SENSOR_CM` / `PFT_COL`) in
   `scripts/plot/plot_ec_site_publication.py`, then plot (the figure grows by one row).

Note: the model under-predicts GPP/latent heat at **dense, high-LAI wet broadleaf
forests** (a documented coupled-closure bistability — see
`two_leaf_canopy_wet_forest_bistability.md`), so tropical/wet-EBF sites are not yet
suitable panel members.
