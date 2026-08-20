# Land dual-target calibration runbook (evaporation + skin T, bias-first)

The repeatable mechanism behind the 2026-08-17 calibration of BOTH land tiers
against the same ERA5 dual target. Grid/resolution agnostic: the calibrators are
per-column (no horizontal operators), so a new resolution only changes the DATA
step; the training commands are unchanged. Supersedes the skin-T-only parts of
`docs/land_skin_t_bias_reproduce.md` (whose protocol/controlled-comparison rules
still apply).

## Objective (what the loss is, and why)

Primary target = **mean bias, including the monthly (seasonal-cycle) bias** —
RMSE fights natural variability and is held only as a pattern regularizer
(user directive 2026-08-17). Per tier the loss is:

    T-MSE + lam_alb*albedo-MSE + lam_le*LE-MSE
    + lam_tbias_mon * mean_month( area-mean(T - ERA5 skt) )^2     # default 30
    + lam_lebias_mon * mean_month( area-mean(LE - ERA5 LE) )^2    # default 0.3
    [+ multilayer only: soil-moisture MSE, seasonal-amplitude, per-PFT bias,
       annual global-bias (lam_tbias)]

Latent heat is what constrains the canopy-conductance chain (Vc_max25/g1/LCMA,
multilayer, stomata ON by default) and the slab's bucket capacity W_max (its
root-zone-storage stand-in).

**STRICT no-inert-parameters rule** (CLAUDE.md): mode-inactive keys are frozen
out of the trainable pytree (`_inactive_keys` in the multilayer trainer), and
`assert_no_inert` aborts at step 0 on any identically-zero or all-non-finite
gradient leaf, gated on the FULL data. It has caught three real defects so far
(slab `pft_root` never consumed; `snow_zenith` unwired everywhere; a traced
static flag).

## Step 1 — data (this is the only resolution-dependent step)

```bash
# ~2 deg = --stride 8 on the 0.25-deg ARCO axis; 1 deg = --stride 4, etc.
PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_hourly_climatology.py \
    --stride 8 --years 2019 2020 --days 6 16 26 --out results/land/era5_hourly.npz
```

* The fetch includes `mean_surface_latent_heat_flux` (already W/m2, ERA5
  positive-DOWN) and stores it sign-flipped as `slhf_wm2` (positive-up, model
  convention). A tripwire asserts the flipped land annual mean is in
  [5, 120] W/m2 — a sign or units error fails before the file ships.
* Add the short soil-moisture keys the trainer reads (the fetch stores long
  names): rename `volumetric_soil_water_layer_{1,2}` to `swvl{1,2}` in the npz
  (idempotent; `scripts/data/augment_era5_soil_moisture.py` re-fetches if the
  long names are absent).
* A missing LE field is a HARD error in both trainers unless `--lam-le 0` is
  passed — no silent fallback to skin-T-only.

## Step 2 — slab tier (CPU, minutes)

```bash
PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/run/train_land_params_era5.py \
    --diurnal-npz results/land/era5_hourly.npz --iters 400 \
    --out results/land/land_tuned_dual_slab.json
```

2026-08-17 result (~2 deg, full monthly log): monthly skin-T bias 3.46 -> 1.70 K,
monthly LE bias 7.08 -> 3.15 W/m2, T-RMSE 4.56 -> 2.78 K (LE-RMSE 21.4 -> 22.3,
accepted — bias is the target). Annual per-cell (all 5551 cells): skin-T bias
+3.33 -> +1.61 K, LE bias -5.9 -> -1.3 W/m2. Slab knobs cannot improve LE-RMSE;
the dual target's value on this tier is guarding evaporation while temperature
is tuned.

## Step 3 — multilayer tier (GPU, hours)

```bash
# warm start from the production bake; mini-batch SGD is REQUIRED at full grid
# (a full-batch gradient wanders and never beats the warm start — measured);
# chain --init-json to continue while the loss still descends.
PYTHONPATH=. JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda .venv/bin/python \
    scripts/run/train_multilayer_land_era5.py \
    --diurnal-npz results/land/era5_hourly.npz --days 4 --n-sub 100000 \
    --iters 300 --no-prefilter --batch 300 --init-from baked --holdout 0.2 \
    --out results/land/land_tuned_dual_multilayer.json
# continuation: same command with --init-json results/land/land_tuned_dual_multilayer.json
```

At another resolution keep `--days 4` (24-h forcing) and `--batch ~300`; only
`--n-sub 100000` (= all cells) and the npz change. 2026-08-17 checkpoint
(300 it): monthly skin-T bias 1.51 -> 1.20 K, monthly LE bias 13.4 -> 8.8 W/m2,
held-out (1110 never-trained cells) bias +1.61 -> +1.17 K.

## Step 4 — maps (every result gets one)

```bash
PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/plot/plot_land_bias_maps.py \
    --npz results/land/era5_hourly.npz --tier slab|multilayer \
    --calibrated <tuned.json> --days 4 --out <maps.png>
```

Pixel (pcolormesh) maps on the source lat-lon grid, rows = skin-T / albedo / LE
bias, cols = default / calibrated / bias-reduced. The plotter takes per-cell
coordinates from the loader (`data["lat"]`, `data["lon"]`) — the loaders permute
cell order, so never rebuild coordinates from the unpermuted land index.

## Step 5 — ship into AMIP

1. **Bake**: `scripts/experiment/format_land_bake.py <tuned.json>` formats the
   `_TUNED_*_MULTILAYER` constant block for `legoesm/land/clm_surface_map.py`
   (reviewed paste; the slab family `_TUNED_PFT_*` is separate). New
   canopy-conductance constants (Vc_max25/g1/LCMA) need bake wiring + dual
   review before first use.
2. **Land IC**: the LMIP spin-up (`scripts/run/run_lmip_biophys.py --config
   config/lmip/amip_mpas4_spinup.yaml --output-dir <dir>`) — 5 years CRU-JRA on
   the target AMIP mesh under the new bake; the restart loader refuses a
   column-count mismatch, so re-run per mesh.  (`run_land_spinup.py` is the
   idealized-forcing alternative.)
3. **AMIP config**: `config/amip/amip_land_v6.yaml` records the IC + matching
   soil-column flags (`--land-ic <restart> --multilayer-n-layers 8
   --multilayer-soil-depth 6.375` on top of the production config); the slab
   fallback needs no config (its bake family is picked up by
   `--slab-land-active` directly).

## v7 — per-PFT canopy snow masking (2026-08-18)

The v6 NH>55 albedo residual was PFT-structured (forests +0.05 too bright,
tundra/bare/grass -0.05..-0.09 too dark): the canopy-snow-masking signature.
`LandAlbedoConfig.snow_cover_scale` (per-cell, PFT-weighted from the trainable
`pft_snowmask` table, 1.0 = legacy) closes it: the v7 multilayer fit learned
forests <1 / tundra-shrub-crop >1 and the NH mean albedo bias went -0.028 ->
+0.003 with all-global biases skin-T +0.16 K / albedo +0.007 / LE +0.35 W/m2
(held-out skin-T bias +0.14 K).  Baked as `_TUNED_PFT_SNOWMASK_MULTILAYER` and
wired in `clm_multilayer_setup`.  The SLAB v7 fit was REJECTED: with no canopy
its mask acted as one more brightness knob and bought temperature with a
+0.074 global albedo bias — the slab bake stays at v6.  Residual multilayer
spots: boreal-evergreen belt slightly over-bright (temperature-vs-albedo
trade), Sahel/Arabia over-bright (dry-soil brightening), Tibet warm spot
reduced but present.

## Coupled A/B result (2026-08-18) — v7 does NOT transfer; old bake stays coupled default

30-day AMIP Jan-1979, res-4 campaign lane, bitwise-identical land IC and
atmosphere, one variable = the bake.  v7 arm: land LE **collapses 34.7 -> 12.3
W/m2** (obs ~28-32), concentrated in the SNOW-FREE tropics (Amazon/Congo/
Maritime Continent, -40 W/m2), land precip -0.12 mm/day, global precip -0.20;
Antarctica planetary albedo moves strongly toward obs (+0.15), NH high
latitudes brighten past the old arm.  Offline the same parameters are
LE-unbiased — a pure offline->coupled transfer failure (the GLM ship condition
that demanded this A/B was right).

REFUTED mechanism: the calibration-vs-deployment stomata mismatch (Farquhar vs
Jarvis).  Offline discriminator: v7 params give LE 40.8 W/m2 under Farquhar
and **63.7 under Jarvis** — the coupled collapse (less LE) is the OPPOSITE
sign, so the coupled consumption path, not the stomata scheme, drives it.
CAUSE UNKNOWN — next discriminators: instrument the coupled arm's land
beta_soil / q_sfc / absorbed-SW per cell (the compiled non-tiled lane
reconstructs the land humidity from beta_soil, wp/fc/root changed in v7; and
v7's brighter land cuts absorbed energy).  Do NOT flip the coupled default to
v6/v7 until the coupled collapse is explained and fixed; offline artifacts
(tuned JSONs, offline bake) remain valid for offline work.

NOTE: the LMIP spin-up driver does NOT consume the baked tuned tables (two
spin-ups under different bakes produced bitwise-identical restarts) — a
per-bake land IC requires wiring the bake into run_lmip_biophys first.

## Snow-mask follow-ups (GLM review, 2026-08-18 — none blocking, all cheap)

* **Thermostat test**: refit with the temperature terms frozen (albedo-only
  loss); if the mask table diverges (esp. boreal), the fitted values are
  temperature compensation, not burial physics.
* **Bare-soil pin test**: pin bare mask = 1.0 and refit; if global scores
  survive, the 0.92 was optimizer slack (dust-on-snow / warm-season leakage),
  not physics.
* **Two-term canopy snow**: the cover-only mask cannot represent
  intercepted-snow BRIGHTENING (autumn/spring falls near 0 C); the boreal
  ordering (0.99 boreal vs 0.89 temperate, reversed vs obs) suggests the two
  effects cancel inside one knob.  Follow-up: add a canopy-snow albedo term
  g(T, time-since-snowfall).
* **Cover validation**: check the effective snow-cover fraction against
  IMS/MODIS snow-cover obs, not only albedo — a cover knob absorbing an albedo
  error passes the albedo score while degrading cover.
* Boreal-evergreen residual (+0.08 bright, ~15-20 W/m2 local spring
  absorption) is a DOCUMENTED temperature-vs-albedo compensation — chase the
  underlying warm bias (roughness/stability, LAI) before letting albedo pay.

## Review demands before the bake is trusted coupled (GLM, 2026-08-18)

* **Coupled A/B vs the previous bake** (30-day AMIP arm, one variable = the
  bake): TOA imbalance/drift, land skin-T, land LE, precipitation pattern,
  soil-moisture drift.  The offline gains are unproven coupled until this runs.
* **Emissivity floor**: the multilayer fit reached 0.944 on a vegetated PFT
  (physical floor ~0.95); raise the BOUNDS_EXT emissivity floor to 0.95 for
  non-bare PFTs at the next re-tune.
* **Beta/stomata double-count invariant test**: beta-on-flux and stomatal
  conductance must never both throttle the same transpiration flux.
* ERA5 land LE and skin T are themselves model output (HTESSEL), not
  observations — carry that target uncertainty when interpreting residuals.

## Known caveats (open, PLAUSIBLE)

* Vc_max25/LCMA may be collinear on a monthly-mean LE target with prescribed
  LAI (GLM review); a twin experiment / per-biome Jacobian SVD would settle it.
  The fitted values are effective-conductance parameters — do not reuse them as
  photosynthetic capacity in the carbon cycle without that check.
* ERA5 has no irrigation: irrigated basins (Indo-Gangetic, Central Valley,
  North China Plain) bias the LE target dry there.
* Antarctica's residual dark-albedo bias on the slab is its glacier-albedo
  UPPER BOUND (0.75 vs ERA5 ~0.85) — a bounds choice, not a fit failure.

## Calibrate-under-A / deploy-under-B: closed as a DEFECT (2026-08-19)

Separate question from the coupled LE collapse above (whose cause is still
unknown): the tuned tables were being DEPLOYED under a land model they were
never FITTED under.  Three different land models were in the chain.

|                    | offline calibration | land spin-up (IC) | coupled AMIP |
|--------------------|---------------------|-------------------|--------------|
| surface scheme     | SimpleSEB           | **two-leaf canopy** | SimpleSEB |
| stomata            | **Farquhar** (enabled + differland + prescribed carbon state) | **Jarvis** (enabled, carbon "none") | **NONE** (`land_stomatal_beta: false` in all three shipped decks) |
| soil column        | 8 layers / **3.0 m** / growth **1.5** | 8 / 6.375 / 2.0 | 8 / **6.375 m** / growth **2.0** |
| per-PFT tables     | the fit            | **legoesm_surfdata defaults, NOT the bake** | the bake |

`compute_effective_beta` has three branches; the coupled tile took the THIRD
(`beta = beta_soil`, no canopy resistance at all), so the baked Vc_max25 / g1 /
LCMA were INERT — a direct violation of the no-inert-parameters rule.  Flipping
`land_stomatal_beta` alone does not repair it: without the differland scheme AND
a non-None carbon state the same dispatch falls to Jarvis.  The soil columns are
both 8 layers, which is exactly why the restart loader's shape check never
caught the depth difference (top layer 6.1 cm fitted vs 2.5 cm deployed).

### What now exists

* `legoesm.land.config.calibrated_multilayer_setup()` — ONE definition of the
  fitted land model.  `train_multilayer_land_era5.py` derives its module
  constants from it, so a re-tune that EDITS the shared definition goes red in
  `tests/unit/test_land_calibrated_physics.py` instead of silently producing
  tables for a model nothing runs.  It does NOT catch an exploratory fit that
  overrides the mode on the command line (`--no-stomata`, `--bulk constant`):
  those set the module constants after the drift check reads them, so a fit run
  that way is not a bake candidate and must not be baked.
* `--land-calibrated-physics` (AMIP) and `physics.calibrated_land_physics` (LMIP
  spin-up) deploy it; both CHECK the overlapping config keys rather than
  overriding them.  Default OFF — every existing run's NUMERICS are unchanged
  (land restart FILES gain one extra array, the soil layer thicknesses).
* The AMIP driver seeds the PRESCRIBED carbon state (fixed leaf carbon -> fixed
  LAI), which is the third condition the Farquhar branch needs.
* `save_land_restart` / `load_land_restart` record and check the soil layer
  THICKNESSES.  A restart written before this carries none and can only be
  warned about — re-run the spin-up to get a checkable one.
* Configs: `config/amip/amip_land_calibrated.yaml`,
  `config/lmip/amip_mpas4_spinup_calibrated.yaml`.

### Measured, offline, under the calibrated setup

RETRACTION: an earlier version of this table, taken from a 6-column probe, gave
needleleaf and crop the WRONG SIGN (-3 %, -2 %; they are +8 %, +9 %) and
mislabelled arctic grass.  The full 17-PFT sweep below replaces it.  The error
was reading a probe's output against a hand-written label list instead of the
model's own PFT names — the same class as every other mislabelled-table failure
in CLAUDE.md.

All 17 PFTs, single-PFT columns, uniform loam, 700 W/m2 down, T = 298 K,
q = 0.012 kg/kg, CO2 412 ppm, float32.  `ratio` = calibrated (Farquhar) beta
divided by the no-stomata beta the coupled runs use today, so 1.0 = no change and
0.5 = evaporation halved.  Two soil-availability states are shown because the
Farquhar path is a canopy/soil BLEND, not a `min`, so it can raise beta as well
as lower it — which is why the sign is not uniform.

| PFT | LAI | ratio @ wet soil (0.9) | ratio @ dry soil (0.4) |
|-----|-----|------------------------|------------------------|
| bare soil | 3.28 | **0.22** | **0.26** |
| needleleaf evergreen temperate | 2.26 | 1.08 | 1.18 |
| needleleaf evergreen boreal | 2.39 | 0.79 | 0.84 |
| needleleaf deciduous boreal | 2.33 | 1.07 | 1.13 |
| broadleaf evergreen TROPICAL | 3.73 | 0.88 | 0.94 |
| broadleaf evergreen temperate | 3.38 | 1.09 | 1.25 |
| broadleaf deciduous tropical | 3.28 | 0.79 | 0.84 |
| broadleaf deciduous temperate | 2.98 | 1.09 | 1.32 |
| broadleaf deciduous boreal | 3.21 | 1.09 | 1.17 |
| broadleaf evergreen shrub | 5.98 | **0.48** | **0.55** |
| broadleaf deciduous temperate shrub | 3.81 | 1.09 | 1.31 |
| broadleaf deciduous boreal shrub | 2.85 | 0.87 | 0.92 |
| c3 arctic grass | 4.30 | 0.97 | 1.05 |
| c3 grass | 3.80 | 1.00 | 1.07 |
| c4 grass | 5.22 | **0.38** | **0.44** |
| c3 crop | 3.73 | 1.09 | 1.28 |
| c4 crop | 4.26 | 0.79 | 0.86 |
| **mean over vegetated (1-16)** | | **0.91** | **1.01** |

What this says (CONFIRMED for the offline beta, PLAUSIBLE for the coupled run,
which adds the soil column, prognostic moisture and every atmospheric feedback):

* Over VEGETATED land the calibrated model is close to neutral — 9 % less
  evaporation on wet soil, 1 % more on dry soil.
* The large throttles are BARE SOIL (a factor of ~4), evergreen shrub and C4
  grass — i.e. deserts, semi-arid shrubland and tropical savanna.  Watch the
  Sahara/Arabia/Australia response, not the rainforest.
* The tropical rainforest PFT moves -6 to -12 %.  **That cannot explain the
  34.7 -> 12.3 W/m2 coupled collapse recorded above (-65 %)**, so this change is
  neither the cause of that failure nor a plausible cure for it.  It is a
  correctness fix, and the collapse remains an open, separate question.
* Bare soil is throttled by a CANOPY conductance term (its Vcmax25 is 0 and its
  g1 is 1, yet the Beer-law blend still hands it an LAI of 3.28 from the
  prescribed leaf carbon).  Physically a bare cell should have no canopy at all.
  The offline fit compensated around this, but it is a genuine structural wart:
  the prescribed leaf carbon is uniform, so LAI does not go to zero on bare soil.

### Review round 1 (codex, 2026-08-19) — what it caught

* **BLOCKER, fixed**: the unstructured-mesh (MPAS) coupled land closure — the lane
  the AMIP campaign actually runs — called the land step WITHOUT the leaf-carbon
  state, so the calibrated switch would have produced Jarvis on that lane no
  matter what the config said.  Now passed, and
  `test_every_coupled_land_step_passes_the_leaf_carbon_state` scans EVERY coupled
  call site so the next lane fails instead of silently reverting the model.
* **BLOCKER, fixed**: a land restart written before thicknesses were recorded was
  accepted with a warning, which would have let the old 6.375 m IC onto the 3 m
  run.  Under the calibrated switch it is now a hard error.
* **Fixed**: `land_gs_max` was silently discarded by the calibrated setup (now
  gated); the calibrated spin-up config enabled soil freeze/thaw that neither the
  fit nor the coupled run has (now off, with the reasoning in the file); the
  documented spin-up/AMIP pairing used mismatched resolutions and a flag that
  does not exist; the calibrator now BUILDS from the shared definition rather
  than restating it.
* **Not fixed, and it is a real limitation of the fit itself**: the calibration
  FREEZES soil moisture (`_FREEZE_FROM = 0`) while the coupled run evolves it.
  "Same land model" here means the same schemes, parameters and soil column — NOT
  the same soil-moisture treatment, and no config switch can make it so.

### Review round 2 (GLM, 2026-08-19) — the mechanism, not the diff

GLM was asked whether deploying the fitted model coupled will help or hurt.  Its
verdict, and where it changed my mind:

* **The A/B that showed the collapse had stomata OFF IN BOTH ARMS.**  So the
  34.7 -> 12.3 W/m2 difference was carried entirely by the NON-canopy parameters
  — plant wilting point and field capacity, albedo, root depth, soil thermal
  scales, snow.  Any argument about stomatal schemes was reasoning about code
  that never executed in either arm.  This narrows the open question sharply and
  is the single most useful thing either review produced.
* **The earlier "REFUTED" verdict survives, but not for the reason given.**
  Against the model that was ACTUALLY deployed (no stomata), the fitted beta is
  LOWER almost everywhere, i.e. the same sign as the collapse, not the opposite.
  The refutation holds on MAGNITUDE (<=~20 % vs -65 %), not on sign.
* **Bare soil is worse than "self-consistent".**  The compensation the fit found
  is only valid at the frozen equilibrium moisture and the 3 m / 6.1 cm geometry;
  coupled it predicts desert evaporation toward zero and skin temperature up
  several K over Sahara / Arabia / Australia.  Fix the leaf area to zero on bare
  soil in the SHARED definition and re-fit.
* **g1 ~ 10 meets a feedback the fit never saw**: interactive humidity closes a
  positive loop (more evaporation -> higher humidity -> higher conductance ->
  more evaporation) whose gain is linear in g1, so an over-fitted g1 amplifies
  it.  Prognostic moisture then converts that into early column dry-down, i.e.
  too much wet-season evaporation and a late-dry-season crash.
* **Frozen moisture makes the plant wilting point and field capacity ILL-POSED**:
  with moisture static the stress function is sampled at one point, so its shape
  is unidentified and trades freely against the conductance parameters and root
  depth.  Root depth is nearly as bad.  Albedo and emissivity stay identifiable.
* **Geometry sensitivity, ranked**: soil thermal scales worst (the damping depth
  scales as sqrt(layer thickness), and the top layer changes 6.1 -> 2.5 cm, a
  factor ~1.6), then the moisture thresholds (same numbers mean different water
  under a 2.1x larger column), then root depth; albedo / emissivity / snow are
  geometry-independent.

**One GLM claim I checked and do NOT accept.**  It read the probe's GPP of
2e-4 gC/m2/s as an annual mean (6307 gC/m2/yr, 2x observed) and called the
conductance over-fitted on that basis.  The probe forced 700 W/m2 at a solar
zenith cosine of 0.9 — peak noon, not a daily mean.  As an instantaneous rate it
is 16.7 umol CO2/m2/s against an observed peak tropical canopy 20-30, i.e. on the
LOW side.  The over-fit-conductance concern may still be right for other reasons
(the g1 ~ 10 argument stands on its own), but this arithmetic does not support it.

**Where GLM disagrees with the directive.**  It would RE-FIT under the coupled
model rather than deploy the fit coupled, because several fitted parameters are
ill-posed under frozen moisture.  Both routes remove the mismatch, and the shared
definition serves either: re-fitting under the coupled model means editing
`calibrated_multilayer_setup()` and re-running the calibrator, after which the
same switch deploys the result.  The deployment switch shipped here is opt-in and
default-off, so nothing is committed to either route.

### Review round 3 (codex) — the plant model was still not reaching the air

* **BLOCKER, fixed**: on the MESH lane the land tile's solved humidity and fluxes
  are published to the atmosphere only under `mpas_land_beta_soil`, which every
  shipped deck leaves off.  A calibrated run without it would solve the canopy
  conductance and then discard it, the atmosphere keeping a static evaporation
  efficiency of 0.6 — the same inert-parameter defect one layer deeper.  The
  calibrated switch now REQUIRES it on that lane, the overlay sets it, and a test
  pins the refusal.
* **MAJOR, bounded rather than fixed**: the cube / lat-lon pipeline builds the
  atmosphere's surface humidity as root-zone soil moisture times the saturation
  humidity at the land skin temperature.  The tile's SOLVED boundary humidity and
  fluxes are never passed through, so there the fitted canopy conductance reaches
  the air only indirectly, by shifting skin temperature — not as the evaporation
  the tables were fitted to.  Routing the solved response through is a coupler
  change with measured history against the naive version (an earlier humidity
  handoff delivered ~a tenth of the solved flux, which is why the mesh lane moved
  to a FLUX handoff).  Round 4 made this a REFUSAL rather than a caveat: the
  switch is rejected off the mesh lane, with a test.
* Three MINORs fixed: a lat-lon MPI restart could carry an orphan soil-column
  stamp into its next save; the call-site scan treated an explicit
  `carbon_state=None` as passing; the overlay still described the old
  warn-on-unverifiable-restart behaviour.

### Review round 4 (codex)

* **BLOCKER, fixed**: the flux-handoff guard my round-3 gate made MANDATORY raises
  on a rank with no land.  Under cell-partition MPI an ocean-only rank
  legitimately has none, so that raise would have aborted one rank while the
  others waited in the next collective — a deadlock introduced by requiring the
  flag.  It now asks every rank with the same logical-OR allreduce the sibling
  guard already used, so an ocean-only rank simply publishes nothing.
* **Fixed**: the calibrated switch is now REFUSED off the mesh lane rather than
  merely documented as unsupported; a non-multilayer calibrated config no longer
  emits two overlapping errors; both YAMLs' documented commands were missing the
  now-required flux-handoff flag (they would have been refused by my own gate);
  the call-site scan missed a POSITIONAL `None`; the runbook named a config key
  that no longer exists; the flux-handoff field comment still described a
  humidity-only handoff.

### Review round 5 (codex) — my round-4 MPI fix was itself a deadlock

* **BLOCKER, fixed**: the cross-rank vote I added in round 4 sat INSIDE the
  "this rank has no land" branch, so only the land-less ranks called the
  collective while the land ranks walked past it — a deadlock, introduced by the
  fix for a deadlock.  The vote is now unconditional for every mesh rank (the
  condition it hangs off is a config value, identical everywhere).  Two rounds to
  get one collective right is the honest cost of an MPI guard.
* **Fixed**: a calibrated run could start with a turbulence scheme whose kernel
  has no place for the land fluxes and be refused HOURS later at run time; the
  same signature scan the runtime guard uses now runs at config time, with a
  test.  The spin-up gate refused one stomatal override but accepted the other
  two.  The non-mesh refusal advised setting "the individual land keys", which
  cannot reproduce the soil growth factor or the carbon scheme — the message no
  longer claims they can.

### Review round 6 (codex) — the collective is gone, and that was the fix

* **BLOCKER, fixed**: even called by every rank, my cross-rank vote sat AFTER a
  rank-local land-albedo raise, so a rank with land but no albedo could die while
  the ocean-only ranks blocked in the vote.  The answer was to delete the
  collective, not to move it: `validate_strict` already guarantees the flag
  implies the multilayer land, and a globally landless run is refused there too,
  so the runtime guard only ever needed to test a CONFIG value — identical on
  every rank, symmetric by construction, nothing to deadlock.  Three rounds on
  one guard; the lesson is that adding a collective to a code path with
  pre-existing rank-local raises is a hazard in itself.
* **Accepted, not fixed**: asking the turbulence kernel whether it can take the
  land fluxes cold-imports the turbulence stack (~7 s) during validation.  It is
  paid once, only by runs that use the flux handoff, against a job that runs for
  hours.  A lightweight duplicate of the signature scan would reintroduce exactly
  the staleness that scan exists to prevent.

### Review round 7 (codex) — a calibrated run could still have had no land

* **BLOCKER, fixed**: flat topography with no land-mask file on the mesh lane
  passed validation, built no soil column on any rank, and the flux handoff (with
  the calibrated conductance riding it) went silently inert.  The existing
  flat/no-mask rejection was gated on a setting the mesh lane does not use.  Now
  refused at config time, plus a global any-land guard in setup for the case the
  config cannot see (an all-ocean mask FILE) — modelled on the sibling guard that
  already sits there, which is the one place every rank reaches before any
  rank-local raise.
* Two prose-vs-code claims corrected: the drift test does NOT catch an
  exploratory fit that overrides the mode on the command line (such a fit is not
  a bake candidate); the spin-up driver's soil-grid/carbon comments described
  only the non-calibrated path.

### Review round 8 (codex) — the rule that ended four rounds of the same bug

* **BLOCKER, fixed**: the guard's cross-rank vote still sat AFTER the rank-local
  land setup, which can itself raise on one rank and not another (a rank with
  land but no surface data, say) while the ocean-only ranks wait in the vote.
  The vote now happens at the TOP of the land block, before the first line that
  can diverge, and the guard reads its result rather than taking its own.
* **The rule this cost four rounds to learn**: a collective must PRECEDE every
  rank-local raise on its path — being called by every rank is not enough.  Each
  earlier attempt satisfied the weaker condition and still deadlocked.  Adding a
  collective to a code path that already contains rank-local raises is a hazard
  in itself; prefer a config-only test, and when a collective is genuinely needed
  put it before any per-rank work.

### Review round 9 (codex) — and where the MPI thread was cut

* **The collective is gone from my diff entirely.**  Round 9 showed the vote was
  still downstream of file loads and pipeline construction that can raise on one
  rank; making it safe would mean synchronising setup FAILURES across ranks — a
  repo-wide change to how every setup step handles errors, not a land-calibration
  change (the two collectives the repo already has in this position have the same
  property).  So the guard is now rank-local and fatal only for SERIAL runs, with
  and under MPI it says nothing at all (see the GLM note below on why a warning there would have fired on every healthy run).  A guard is not worth
  a deadlock; the config-level checks catch the realistic spellings of the trap.
* Two calibrator fixes: its docstring still described the OLD default (constant
  exchange, soil-only beta) rather than the calibrated one; and it rebuilt the
  stomatal and carbon configs from class defaults while claiming one shared
  definition, so a future non-default field there would have reached the
  deployment and not the fit.  Both now derive from the shared definition, and
  the drift test compares the whole configs rather than the on-off fields.

### GLM round 2 — a warning that fires when nothing is wrong

GLM accepted the rank-local guard as the right trade but caught what codex had
not: with the flux handoff on, EVERY ocean-only rank would have warned on EVERY
healthy run, and a warning that fires when nothing is wrong trains everyone to
ignore it.  Under MPI the guard now says nothing (debug only); the serial hard
error stands.  GLM's named follow-up, recorded but not built: count the land
points in the MASK FILE at config time — rank-symmetric, no collective, and it
closes the one case the config check cannot see today.

### Review round 10 — and one place the two reviewers disagree

* **Fixed**: the soil-column refusals ran only on ranks that own land, so a
  wrong-column initial condition would abort those ranks while the ocean-only
  ones waited in the next collective.  The check now happens UP FRONT, from the
  config and the restart file's own record of its thicknesses — both identical on
  every rank — before any grid, partition or land work.  Reading just that one
  array (`load_land_restart_soil_dz`) is what makes it possible without loading
  the state.
* **Fixed**: "serial" was being tested as "no partition object", but a one-rank
  MPI launch still builds one; it is now the rank COUNT.

**Reviewer disagreement, recorded rather than averaged.**  Codex calls a
multi-rank run with an all-ocean mask FILE a blocker (the flag would be inert and
nothing says so); GLM accepts it, on the grounds that the failure is VISIBLE in
the output — a landless run looks landless — rather than silently wrong.  That is
the question that discriminates, and GLM's reading of it is why the guard stays
rank-local.  Both reviewers independently proposed the same fix: count the land
points in the MASK FILE at config time, which is rank-symmetric and needs no
collective.  It is the named follow-up, not built here.

### Review round 11 (codex)

* **BLOCKER, fixed**: the preflight built the soil grid through the shared
  (JAX-backed) helper, and I had placed it BEFORE the runtime bootstrap — so it
  created a JAX array before the backend, precision and distributed setup were
  chosen.  Moved to just after the bootstrap and still before the first
  rank-divergent step, which keeps one implementation of the grid geometry rather
  than a NumPy copy of it.
* **BLOCKER, fixed**: a missing or broken-symlink land initial condition returned
  quietly from the preflight and failed later, on land-owning ranks only.  It now
  refuses up front.
* **MINOR, fixed**: the soil-column comparison and its tolerance existed in three
  places.  One `soil_dz_matches` now serves the preflight, the checkpoint check
  and the loader, so the before-the-run answer and the during-the-run answer
  cannot drift apart.

### Verdicts

* **codex: CLEAN** after twelve adversarial rounds.  It found, in order: the mesh
  lane's own land step never received the leaf-carbon state; a legacy restart
  accepted on an unverifiable column; the mesh lane discarding the tile's solved
  fluxes entirely; three successive wrong placements of one MPI guard; a
  calibrated run that could have had no land at all; a JAX array created before
  the runtime bootstrap; and a missing initial condition that failed only on
  land-owning ranks.  None of these would have been found by the tests.
* **GLM: SHIP**, with five non-blocking notes.  The one acted on: the off-mesh
  refusal now carries the measurement that justifies it (the earlier humidity
  handoff delivered ~a tenth of the solved flux), so nobody later "fixes" the
  guard by weakening it.  Its other notes — persist the column-generating
  parameters alongside the thicknesses (they are recoverable from the
  thicknesses: count, sum and ratio), and broadcast the preflight's file read
  rather than reading per rank — are recorded and not built.  Confirmed on its
  behalf: the turbulence capability check fails CLOSED (a scheme absent from the
  scan is refused, not admitted).
* Both reviewers were asked the same closing question and agree the switch makes
  a future re-fit under the coupled model EASIER, not harder: the shared
  definition is exactly the thing you would edit to re-target the fit, and the
  soil-column records stop a re-fit's spin-up, calibration and validation runs
  from silently mixing columns.

## Cross-grid AMIP: what it costs, and why the cheap route is REFUTED (2026-08-19)

The calibrated switch runs only on the unstructured (MPAS/Voronoi) lane.  That
lane hands the atmosphere the land tile's SOLVED sensible and latent fluxes, so
the canopy conductance arrives as the number the land model computed.  Every
structured lane instead re-derives the land flux from a scaled saturation
humidity, `beta_soil * q_sat(T_land)`.

**The cheap route was tried and refused by BOTH reviewers, independently.**  The
attempt was to put the canopy into that humidity — replace the soil-only beta
with the land model's effective beta.  It is wrong for reasons neither the tests
nor the offline numbers would have shown:

* `simple_seb` applies its throttle to the humidity GRADIENT, and BYPASSES it
  entirely for snow and dew.  A scaled saturation humidity can therefore reach
  the OPPOSITE SIGN of the flux the land solved (codex).
* The atmosphere re-derives the exchange coefficient with a SCALAR roughness
  (`surface_z0_land`, default 0.1 m) while the calibrated land SEB uses the
  per-column TUNED `land_params.z0`, so the two coefficients differ regardless
  of what humidity is passed (codex).
* Land and atmosphere would then evaluate the same conductance separately, at
  different cadences and from different reconstructions of the sun, so the water
  the soil loses is not the water the air gains — a non-conserving interface
  (GLM: "the actual disease is not double throttling but flux non-closure").

**What cross-grid support actually costs.**  The structured lanes run inside a
compiled scan, so the land tile's solved fluxes have to be CARRIED at the
radiation cadence — a new `SegmentCarry` field, which is the repo's textbook
cross-cutting change (~170 references across ~40 files).  The carry already has a
`held_*` family for exactly this cadence problem, so the shape is known; the cost
is the breadth, not the design.  Until then the switch is refused off the mesh
lane rather than deployed under a coupling that cannot express it.

**Measured while checking this** (17 plant types, calibrated setup, 298 K, moist
soil): the canopy term saturates its cap (`clip(gs/gs_ref)`, gs_ref 0.3
mol/m2/s) for 6 of 17 plant types at midday — temperate forests and crops — and
2 of 17 at 300 W/m2.  The tropical evergreen type does NOT saturate (0.77).  So
the fitted conductance has real sensitivity where the collapse is, and none over
productive temperate canopies at peak light.  GLM's proposed replacement (a
series-resistance factor `1/(1+g_a/g_c)`, which saturates without a cap) is a
change to the SHARED function the offline fit also used, so it is a RE-FIT item,
not a deployment one.

## The soil-water tables did NOT cause the coupled collapse (2026-08-19, measured)

`scripts/validate/land_beta_soil_bake_discriminator.py` (committed) evaluates the
DEPLOYED availability function — the one both A/B arms actually ran — with the
previous versus current wilting point, field capacity and root depth, over the
tropical soil-moisture range, on the 6.375 m column both arms used.

| plant type | ratio (current / previous) |
|---|---|
| broadleaf evergreen tropical | 0.98 - 1.00 |
| broadleaf deciduous tropical | 1.00 - 3.66 |
| bare soil | 1.00 - 1.27 |
| evergreen shrub | 1.00 - 1.83 |
| C4 grass | 1.00 - 1.89 |

In the wet tropics BOTH bakes saturate at 1.0 — no throttle at all, identical.
At the dry end the current tables give MORE water, not less.  A 65 % loss needs a
ratio near 0.35; the smallest tropical ratio is 0.98.  **These tables cannot
account for the collapse.**

Vegetation albedo is out too: the tropical evergreen type got DARKER (albedo
0.1396 -> 0.1177, about +4 W/m2 MORE absorbed), and the global mean PFT albedo
moved -0.0007.  The one large albedo change is the soil-colour scale, 1.0436 ->
1.4017 — a candidate for deserts and semi-arid land, not the wet tropics.

ROOT DEPTH, which the bake also moved, was tested SEPARATELY because a
vertically uniform column hides it completely — the root-zone weighting
integrates the same value at every depth, so the sweep above is blind to it.  On
a DRYING column (top layers near wilting, deep layers wet — the state a coupled
run actually reaches) the current tables still give MORE availability, ratios
0.92 to 2.32, tropical evergreen the only one below 1 and only by 8 %.  The probe
now runs both states for exactly this reason.

**What this does and does not establish** (codex, and it is right): availability
is not evaporation.  What the atmosphere takes up also depends on the humidity
gradient, the surface temperature and the feedbacks between them.  A ratio near 1
rules out these tables throttling the surface DIRECTLY; it does not settle
causality.  The probe prints numbers and no verdict.

### The collapse is a BOWEN FLIP, not lost energy (2026-08-19, measured)

`scripts/validate/amip_land_calibrated_ab.py` (committed) scores a coupled pair
area-weighted by the model's own cell area and split by its own land fraction.
Run against the KNOWN previous pair as a harness self-check, it reproduces the
documented collapse — and shows where the energy went:

| land, area-weighted | old bake | newer bake | change |
|---|---|---|---|
| latent heat | 29.5 | 7.0 | **-22.5 W/m2** |
| sensible heat | 10.2 | 35.9 | **+25.7 W/m2** |
| near-surface T | 270.17 | 270.56 | +0.38 K |
| precipitation | 0.64 | 0.60 | -0.04 mm/day |
| OCEAN latent heat (control) | 74.1 | 76.6 | +2.6 W/m2 |

Sensible heat rises by almost exactly what latent loses.  The surface is
receiving about the same energy and PARTITIONING it differently — which is
consistent with albedo being nearly unchanged, and it rules out an
absorbed-energy explanation.  So the cause is something that moves the Bowen
ratio without moving availability at fixed soil state: the surface resistance
terms, the soil thermal scales (which set how the skin temperature responds), or
emissivity.  (The numbers differ from the 34.7 -> 12.3 recorded earlier because
this scorer uses a strict land mask and the model's own area weights; the
collapse is the same event.)

So the eliminated causes are now: stomata (OFF in both arms), the soil-water
tables including root depth (more availability, not less, in every soil state
tested), vegetation albedo (wrong sign), and absorbed energy (the Bowen flip
conserves it).  The cause remains UNKNOWN.

## COUPLED A/B OF THE CALIBRATED MODEL ITSELF (2026-08-19) — it does NOT transfer

The measurement both reviewers asked for.  Two AMIP runs, res-4 mesh, 5 days,
cold-started soil, identical in everything except the plant model; the mesh lane,
which hands the atmosphere the land tile's SOLVED fluxes.  Scored by
`scripts/validate/amip_land_calibrated_ab.py`, area-weighted by the model's own
cell area and split by its own land fraction.

| land | plant model OFF | plant model ON | change |
|---|---|---|---|
| latent heat | 32.17 | 10.47 | **-21.70 W/m2 (-67 %)** |
| sensible heat | 2.25 | 15.11 | +12.86 W/m2 |
| near-surface T | 271.52 | 271.50 | -0.02 K |
| precipitation | 1.67 | 1.64 | -0.03 mm/day |
| **ocean latent heat (CONTROL)** | 59.26 | 59.43 | **+0.17 W/m2** |
| ocean near-surface T (control) | 288.29 | 288.34 | +0.05 K |

The ocean control barely moves, so the land change is cleanly attributable to the
plant model and not to some other difference between the arms.  Both runs
completed (828 s / 842 s), no NaN.

**This is a transfer failure, and a large one.**  The offline sweep of the same
parameters predicted roughly neutral over vegetated land (-9 % wet soil, +1 %
dry) and -6 to -12 % for tropical forest, with the big throttles confined to bare
soil and shrub.  Coupled, the global land mean falls by TWO THIRDS.  The fitted
parameters do not do coupled what they do offline.

**The signature is the same one the bake A/B showed** — a two-thirds latent
collapse with a compensating sensible rise, at almost unchanged skin temperature.
That earlier collapse happened with stomata OFF IN BOTH ARMS, so the two cannot
share a cause through the plant model; what they share is a hard throttle on the
surface moisture supply.  PLAUSIBLE, not confirmed: the land tile has a common
supply-limiting path that both changes push into.  The bare-soil defect below is
the first place to look, since it throttles by a factor of four on cells with no
plants at all and those cells are a large share of global land.

**CONCLUSION: do not deploy the fitted land model coupled as it stands.**  The
switch stays opt-in and default-off, which it is.  This independently supports
GLM's recommendation to RE-FIT under the coupled configuration rather than
transplant the offline fit — and the shared definition built here is what a
re-fit would re-target, so that route is now cheaper, not harder.

**Caveats, stated because they bound the claim**: 5 days from a cold soil column
is a "does the physics behave" window, not a climate.  Neither absolute value is
comparable to the observed 28-32 W/m2.  The CHANGE between the arms is what this
measures, and that is controlled.

## The land model gives ALL land a forest canopy (2026-08-19, measured, NOT fixed)

Measured through the REAL loader on the staged CLM surfdata, coarse global grid:

| leaf area | what the model uses | what the map says |
|---|---|---|
| 795 cells >90 % bare soil | 2.27 - 5.21 | **0.00** |
| all land, mean | ~3.3 | **0.37** |

The coupled land model runs a PRESCRIBED, spatially UNIFORM leaf-carbon pool and
derives leaf area from it as `C_fol / LCMA`.  Uniform pool means every column
gets essentially the same canopy — deserts, tundra, savanna and dormant cropland
alike.  The real per-cell climatology is ALREADY LOADED (`LandSurfaceParams.LAI`,
PFT-weighted CLM `MONTHLY_LAI`), and its own field comment says it exists "so
barren land gets LAI~0, not a spurious uniform canopy".  The stomatal path does
not read it.

Offline effect, bare soil at soil availability 0.60: evaporation factor 0.143
with the invented canopy against 0.600 with the real one — a fourfold throttle by
stomata that do not exist.

**NOT FIXED, and why.**  The one-line fix (prefer the map) was written, measured
and REVERTED on review.  The offline calibrator supplies NO leaf-area map, so the
fix would put production on the map and the fit on the derived value — the
calibrate-one-way / deploy-another defect this entire document exists to remove,
reintroduced through the data rather than the config.  It also makes the fitted
per-PFT `LCMA` inert in production, since `LCMA` reaches the stomatal path only
through `C_fol / LCMA` (codex).  The drift test here did not catch it because it
compares CONFIG FIELDS, not data sources — extend it when this is taken up.

**It must ship as ONE unit**: give the calibrator the same leaf-area source,
re-fit, then deploy.  Two related defects to fold in at the same time (codex):
the "monthly" climatology is averaged over months before use, so there is no
leaf-off season; and the transient-cover path computes a cover-dependent leaf
area and never puts it on the parameters.

**It is NOT the main term.**  With the fix in, a 5-day coupled arm recovered only
0.8 of 21 W/m2.  Whatever dominates is elsewhere.

## RETRACTION: the -67 % was a confounded comparison (2026-08-19)

The pair reported above as "one variable = the plant model" differed in THREE:

| | control | calibrated |
|---|---|---|
| plant model | off | on |
| soil top layer | 1.18 cm | 6.09 cm (growth 1.5 vs 2.0) |
| land -> atmosphere coupling | **static 0.6 efficiency** | the land tile's SOLVED flux |

The third invalidates it.  With the handoff off, the control never used the land
model's evaporation at all — the atmosphere applied a fixed efficiency.  So the
comparison was a CONSTANT against the LAND MODEL, and the difference was
attributed to plants.  The -67 % does not mean what it was reported to mean, and
the "does not transfer" verdict drawn from it is WITHDRAWN pending the isolated
pair (both arms with the solved-flux handoff and the same soil column, differing
only in stomata).

The controlled-comparison rule in CLAUDE.md exists for exactly this, and the
run was launched without checking the arms differed in one field.

## ISOLATED ONE-VARIABLE PAIR (2026-08-19) — plants DO cut evaporation two-thirds

Re-run after the retraction above.  Both arms hand the atmosphere the land tile's
SOLVED flux, both on the SAME soil column, differing ONLY in whether the stomatal
model is on.  Ocean control clean (-0.39 W/m2).  Both completed, no errors.

| land, area-weighted | plants OFF | plants ON | change |
|---|---|---|---|
| latent heat | 13.59 | 4.51 | **-9.08 W/m2 (-67 %)** |
| sensible heat | 13.29 | 21.33 | +8.04 W/m2 |
| near-surface T | 271.24 | 271.68 | +0.44 K |
| precipitation | 1.73 | 1.52 | -0.21 mm/day |

**The -67 % ratio SURVIVES proper isolation.**  What the earlier confounded pair
got wrong was the BASELINE, not the ratio.  Setting the three configurations side
by side (5-day January, cold soil — the ratio is controlled, the absolutes are
not comparable to an annual mean):

| land latent heat | W/m2 |
|---|---|
| shipped default: atmosphere applies a STATIC 0.6 efficiency, land model bypassed | 32.2 |
| land model's own flux, plants OFF | 13.6 |
| land model's own flux, plants ON | 4.5 |
| observed, annual | 28-32 |

**IT IS NOT THE LEAF-AREA DEFECT.**  This pair runs the JARVIS stomatal model
(no carbon state), and that path calls `compute_stomatal_beta(gs, None, ...)` —
leaf area is not an argument.  So the two-thirds cut happens through the
CONDUCTANCE itself, not through canopy area.  The leaf-area defect is real and
separately recorded, but it is not this.

**The larger finding.**  With plants OFF the land model already delivers 13.6
against 28-32 observed.  The shipped configuration looks closer to observations
only because it BYPASSES the land model and applies a constant.  Two separate
deficits are stacked here — the land model's own evaporation is low before any
plant model is switched on, and the plant model then removes two thirds of what
remains.

Next discriminators, in order: (1) is the conductance itself too small coupled,
or is the atmosphere's exchange coefficient the limiter — compare the land tile's
solved flux against the atmosphere's applied flux per column (the reviewers'
named closure diagnostic, which needs both exported); (2) does the deficit with
plants off trace to soil supply or to the surface resistance.

## RETRACTED: the "rainforest starts below wilting" claim (2026-08-19)

Written and withdrawn the same day.  The claim was that the AMIP cold start fills
soil to 25 % of saturation (`land_soil_moisture_init_frac`), leaving the tropical
plant type below its wilting point.  WRONG on the premise: that setting is only a
FALLBACK, used when the initial state carries no humidity tracer, which never
happens in a real AMIP run.

The soil is actually seeded by `aridity_theta_init`:

    theta_init = theta_wp + clip(RH_near_surface, 0, 1) * (theta_fc - theta_wp)

so availability at day 0 EQUALS the initial atmosphere's near-surface relative
humidity.  Humid columns start near field capacity, arid ones near wilting — by
design, to stop a desert cold-start evaporation runaway (#730).  The tropics start
well watered.

**Proven by the experiment that was supposed to test the claim**: two arms
differing ONLY in that setting (0.25 against 0.5) produced BYTE-IDENTICAL land
latent heat, sensible heat, temperature and precipitation.  The setting is inert.
That is the finding worth keeping — a config knob that reads like a control and is
not one.

**Method note, since this is the second wrong diagnosis in a row**: both came from
reading a config FIELD NAME and reasoning from it, instead of reading the code
that consumes it.  The rule already in CLAUDE.md — prove the path executes before
attributing anything to it — applies to initial conditions exactly as to physics.

## WHAT ACTUALLY STANDS, measured

The isolated pair remains valid: both arms seeded identically, both handing the
atmosphere the land tile's solved flux, same soil column, differing ONLY in the
stomatal model.  Land latent heat 13.59 -> 4.51 W/m2 (-67 %), ocean control clean
(-0.39).  So the stomatal model does cut land evaporation by two thirds in a
5-day January cold start.

**And the open puzzle the user named, unresolved**: the LMIP simulations get good
latent heat and GPP from the SAME land model.  So the difference is in how the
land is driven, not in the land model itself.  The largest untested difference is
the FORCING — LMIP runs on reanalysis, the coupled arm on the model's own
atmosphere, and the soil seed and the stomatal response both key off that
atmosphere's near-surface humidity.  A dry model boundary layer would depress both
at once.  NOT TESTED; stated as the next thing to test, not as a finding.

### The cheapest next measurement (GLM, and it answers the OPEN question)

Evaluate the DEPLOYED soil-availability function beta_soil(theta) — the one both
A/B arms actually ran — with the new vs the old plant wilting point and field
capacity, over the tropical soil-moisture range the coupled run visits.  Deployed
latent heat is beta_soil x potential, so that one curve predicts the change
directly.  It is a pure function evaluation: no run, no GPU, and it targets the
parameters the A/B proves were carrying the collapse.

### Still open (NOT fixed here, each a one-line report)

* **Bare soil is given a canopy.**  The prescribed leaf carbon is UNIFORM, so a
  bare-soil cell gets LAI 3.28 and the Beer-law blend treats it as 81 % canopy
  with a conductance of ~0 — evaporation falls by a factor of 4 where there are
  no plants at all.  The offline fit contains the same wart, so the two are
  CONSISTENT and fixing it here would recreate exactly the calibrate-under-one-
  model / deploy-under-another defect this work removes.  The fix is to scale the
  prescribed leaf carbon by vegetated fraction IN THE SHARED DEFINITION and
  RE-FIT.  This is the largest known distortion in the calibrated model and the
  first thing to change if a coupled arm shows a desert evaporation deficit.

* **The spin-up does not use the baked tables.**  `run_lmip_biophys` builds its
  per-PFT params from `legoesm_surfdata` via `surface_data_to_land_params`, not
  from `clm_multilayer_setup`, so the switch matches the spin-up's PHYSICS but
  not its PARAMETERS.  Routing it through the CLM bake means changing which
  surface dataset the spin-up reads — a separate decision.  The LMIP switch is
  named `calibrated_land_physics` for exactly this reason (codex round 2: a
  switch called `calibrated_land` promises more than it delivers).
* **`pft_smscale` is fitted but not baked.**  The calibrator scales the texture
  porosity per PFT (v7 range 0.84-1.18, i.e. +/-18 %) and clamps
  `theta_sat >= max(theta_r + 0.05, plant_fc + 0.02)`; `clm_hydraulics_config`
  applies neither.  The clamp is not currently binding, but the earlier proof of
  that was WRONG (codex): comparing the global minimum porosity against the
  global maximum field capacity ignores the per-PFT scale that multiplies the
  porosity.  Done per PFT, the tightest case is tropical broadleaf evergreen on
  the driest texture — 0.36 x 0.985 = 0.354 against 0.293 + 0.02 = 0.313, a
  margin of 0.042, positive for all 17 PFTs.
* **`snow_zenith` (0.2 in the tuned JSON) is not baked either.**
* **No coupled A/B has been run** with the calibrated model.  It must be one
  variable against the current coupled default, on a land IC regenerated from
  `amip_mpas4_spinup_calibrated.yaml`.  The existing IC belongs to the 6.375 m
  column and predates thickness recording, so under the calibrated switch it is
  REFUSED, not warned about (a run on the historical column still only warns).
  The same check now covers coupled checkpoints, so a calibrated run cannot be
  resumed from a pre-existing one either.
