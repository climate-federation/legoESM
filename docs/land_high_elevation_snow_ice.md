# High-elevation snow / perennial-ice physics (gaps 1–6)

Runbook for the sub-grid elevation-band snow scheme upgrades that target the
high-elevation land surface biases vs ERA5/MODIS: **albedo too dark** (area-weighted
bias ≈ −0.022 over std_elev > 300 m) and **skin temperature too warm** (≈ +0.7 K).
The elevation bands alone (banded precip phase + melt threshold) were necessary but
not sufficient; these gaps add the missing high-elevation *processes*.

Core module: `packages/land/legoesm/land/snow_bands.py`
Wiring: `packages/land/legoesm/land/multilayer_land.py`
State: `MultiLayerLandState.snow_bands / snow_age_bands / ice_bands`

## Status of the six gaps

| # | Process | Status |
|---|---------|--------|
| 1 | Banded surface energy balance (per-band SEB) | **Implemented + wired + tunable** |
| 2 | Elevation lapse of radiative forcing (SW↑, LW↓) | **Implemented + wired + tunable** |
| 3 | Snow darkening (solar-zenith / grain) | **Implemented + WIRED** (real diurnal cos zenith + insolation-weighted albedo) |
| 4 | Glacier firn/ice reservoir + ablation dark-ice albedo | **Implemented + wired + tunable** |
| 5 | Terrain radiation (sky-view) + blowing-snow | **Implemented** (sky-view LW + blowing-snow sublimation, opt-in) |
| 6 | Meltwater refreezing / frozen-soil cold content | **Implemented** (rain-on-snow refreezing; frozen-SOIL = main's `enable_freeze_thaw`) |

All six merged onto main (`1a8a73e51`) + committed (gap 3 `aa378b627`, gaps 5+6 `25c65242d`).
Grain-size metamorphism = the age decay; dust/BC darkening + per-band terrain geometry
+ multi-layer retained-liquid cold content are the documented remaining refinements.
Deferred within gap 3: the coupled `couple_surface_radiation` radiation-time zenith
recompute (narrow opt-in; the offline calibration path is correct).

## Implemented physics

### Gap 1 — banded surface energy balance (`band_net_radiation`)
Each elevation band carries its own albedo, skin temperature
(`T_skin_k = T_sfc − lapse·dz_k`) and net radiation
`Rn_k = SW↓_k(1−α_k) + ε·LW↓_k − ε·σ·T_skin_k⁴`. The **per-band** `Rn_k` drives
per-band melt; the **area-weighted aggregate** drives the shared soil column
(`G_surface`). This keeps the elevation×albedo covariance (bright cold high bands
reflect the enhanced high-altitude sun) that the old cell-mean radiation averaged
away — the dominant lever for the high-elevation cooling. Turbulent fluxes (SH/LH)
remain cell-mean (the lapse shifts a band's air *and* skin T together, so the
air–skin ΔT the bulk flux sees is ~band-invariant); per-band bulk is the deferred
upgrade.

### Gap 2 — radiative elevation lapse
`SW↓_k = SW↓·(1 + γ_sw·dz_k)` (thinner air aloft) and
`LW↓_k = LW↓ − λ_lw·dz_k` (colder/drier air aloft). Because the band anomalies
`dz_k` are **zero-mean** (equal-area Gaussian quintiles), the linear lapse is a pure
sub-grid *redistribution*: `mean_k SW↓_k == SW↓` exactly — no spurious energy source
in a coupled run. The per-column slope is capped so no band's down-flux can go
negative at extreme relief, and the cap (a per-column scalar × zero-mean dz) keeps
the cell-mean conserved.

### Gap 4 — firn/glacier-ice reservoir (`step_snow_bands`)
Replaces the previous instant-cap→runoff bug (winter accumulation was discharged the
same step). Seasonal snow above `swe_snow_cap` firnifies into a per-band `ice_bands`
reservoir that (a) discharges slowly (`ice/τ`, decadal) as frozen runoff, (b) is
ablated by leftover melt energy once the seasonal snow is gone, and (c) exposes a
dark ablation-ice albedo (`alpha_glacier_ice ≈ 0.3`) where snow-free — the
bright-firn / dark-ice contrast a single glacier albedo misses. Fresh snow always
resets the surface age (brightens), even over a glacier; the perennial darkening now
comes from the separate ice reservoir, not a frozen age clock.
Budget closes: `d(SWE+ice)/dt = snowfall − snow_melt − ice_melt − ice_runoff`;
`snow_melt`→infiltration, `ice_melt + ice_runoff`→runoff, energy sink
`(snow_melt + ice_melt)·L_f`.

## New tunable closures (`ElevationSnowBandConfig`, tier-2 / tier-1)

| Param | Meaning | Default | Bounds |
|-------|---------|---------|--------|
| `lapse_rate_K_m` | band T lapse | 6.0e-3 | 4.5e-3 … 8.0e-3 |
| `sw_elev_grad_per_m` | SW↓ elevation gradient | 3.0e-5 | 0 … 1.2e-4 |
| `lw_elev_lapse_W_m2_per_m` | LW↓ elevation lapse | 2.9e-2 | 0 … 6.0e-2 |
| `alpha_glacier_ice` | exposed ablation-ice albedo | 0.30 | 0.15 … 0.45 |

All exposed in the offline trainer (`train_multilayer_land_era5.py`) as
`elev_lapse / elev_sw_grad / elev_lw_lapse / glac_ice_alb`.

## Deferred, with rationale (honest remainder)

- **Gap 3 (solar-zenith / grain / spectral snow albedo).** The BATS/Dickinson
  zenith brightening is implemented + unit-tested in
  `surface_albedo.snow_albedo(..., cos_zenith=)`, but **not wired into the forward**:
  the offline calibration forcing carries a *constant placeholder* `cos_zenith = 0.5`
  (`train_multilayer_land_era5._pack`), so the effect is inactive and uncalibratable
  offline, and in a coupled run a zenith-baked albedo stored in the lagged
  `TileResponse.albedo` would be reused at the next radiation call's (different)
  zenith — an energy inconsistency. Wiring needs (a) a real diurnal `cos_zenith` in
  the offline forcing with insolation-weighted albedo scoring, and (b) radiation-time
  zenith recompute in the coupled `couple_surface_radiation` path. Grain-size
  metamorphism is approximated by the existing age decay; dust/BC darkening needs
  deposition fields not in the forcing.
- **Gap 5 (terrain radiation + blowing snow).** Needs sub-grid slope/aspect/sky-view
  factors and a wind-redistribution model; the required geometry is not in the CLM
  surfdata used here (only `STD_ELEV`). Lowest-impact tractable subset; deferred.
- **Gap 6 (cold-content meltwater refreezing / frozen-soil latent heat).** Needs a
  prognostic liquid-water-in-pack reservoir to hold and refreeze meltwater; it mainly
  shifts melt *timing* and is second-order for the annual/monthly-mean albedo + skin-T
  targets this calibration scores.

## Adversarial review (iterate-with-codex, gpt-5.5 xhigh)

- Round 1 (gaps 1+2+4): 7 findings, all fixed (post-step albedo/LW consistency,
  conservation-breaking flux clamps, dtype promotion, empty-pack deposition mass leak,
  restart seeding, feedback/lat gate, ice-discharge bound).
- Round 2: 7 fixes confirmed OK; 2 new (negative band LW at extreme relief;
  coupled lagged zenith). Both addressed (per-column slope cap; gap-3 unwired).
- Round 3: **CLEAN** (both round-2 findings confirmed fixed; 7 round-1 fixes intact; 65 tests pass).

## Commands

Re-tune (GPU, warm-start from the baked production params):
```
JAX_PLATFORMS=cuda PYTHONPATH=. .venv/bin/python scripts/run/train_multilayer_land_era5.py \
  --elev-bands --init-from baked --bulk most \
  --diurnal-npz /tmp/era5_hourly.npz --days 3 --n-sub 2000 --lr 5e-3 --iters 250 \
  --out results/land_tuned_multilayer_highelev.json
```
Validate (full grid; global / high-elev / Antarctica / Greenland / boreal + bias maps):
```
JAX_PLATFORMS=cuda PYTHONPATH=. .venv/bin/python scripts/tmp/validate_elevband_impact.py \
  --npz /tmp/era5_hourly.npz --tuned results/land_tuned_multilayer_highelev.json \
  --days 2 --out /tmp/highelev_validation
```

## Re-tune / validation results (2026-07-03)

Full-grid ERA5 (era5_hourly, days=2, 5551 land cells, 1167 high-elev). Re-tune
warm-started from baked, 200 iters, batched — the cached `/tmp/keep_fullgrid.npy`
(seed-0, 5434/5551 cells) was reused to skip the O(n_sub) per-cell prefilter.

**A/B/C validation** (A: baked, bands OFF · B: baked, bands ON = gaps 1,2,4 · C: tuned):

| region | Trmse A→B | Tbias A→B | Armse A→B | Abias A→B |
|--------|-----------|-----------|-----------|-----------|
| global | 2.38→2.36 | +0.42→+0.41 | 0.106→0.105 | −0.011→−0.011 |
| high-elev (std>300 m) | 2.80→**2.74** | +0.73→+0.72 | 0.121→0.120 | −0.021→−0.022 |
| Antarctica | 2.41→2.41 | −0.19→−0.20 | 0.081→0.081 | −0.064→−0.064 |
| Greenland | 2.71→**2.59** | +0.35→**+0.29** | 0.141→**0.136** | −0.063→−0.063 |
| boreal>45N | 1.99→**1.93** | +0.35→+0.35 | 0.162→0.162 | −0.041→−0.041 |

**Findings (honest):**
- Gaps 1,2,4 give a **modest, real skin-T improvement where there is relief** —
  Greenland (Trmse 2.71→2.59, Tbias +0.35→+0.29, Armse 0.141→0.136), high-elev
  (Trmse 2.80→2.74), boreal (1.99→1.93). The banded surface energy balance cools the
  over-warm high sub-grid fractions.
- They do **NOT** fix the two headline biases. **Antarctica is unchanged** (A==B): it
  is a flat ice sheet (band_dz≈0), so the elevation bands cannot touch it; its dark
  albedo (−0.064) is a snow/ice-albedo-*magnitude* problem.
- **The re-tune added nothing (C==B):** the baked params already sit at the global-metric
  optimum, and the high-elevation closures gradient to ~0 (flat cells dominate the
  global loss; band_dz≈0 zeros the lapse terms; the ice reservoir needs a long spin to
  fill). `--batch` also made the per-iter loss minibatch-noisy so BEST-tracking returned
  it0. The 4 new closures did not move off their physical defaults.
- **Root cause of the residual — it is FORCING, not the model.** Independently PROVEN in
  the land-validation program: the offline warm-pole/high-terrain bias is a forcing-
  resolution artifact (coarse diurnal + radiation over ice sheets/terrain); *real* ERA5
  forcing removes it automatically. Offline model re-tuning therefore cannot close it.

**Where the remaining lever lives:**
- The polar **dark-albedo** bias (Antarctica −0.064, Greenland −0.063) is the domain of
  **gap 3 (solar-zenith snow brightening)** — snow is brighter at the low polar sun.
  Wiring it needs (a) a real per-cell diurnal `cos_zenith` in the offline forcing
  (requires longitude for local solar time — the loader currently drops it and uses a
  constant 0.5 placeholder), (b) an insolation-weighted albedo score so night hours do
  not bias the mean, and (c) the coupled radiation-time zenith fix (finding 2). That is
  a metric-redesign, not a knob.
- The **true validation** of the elevation-band physics is a **coupled AMIP run** where
  the model computes its own radiation (removing the forcing-driven residual) and gap 3
  becomes naturally active — not the offline forced calibration.

## Update (2026-07-05): all 6 gaps + main merge + review round

Merged origin/main (`1a8a73e51`, 0 behind) — gaps 1,2,4 re-applied onto main's
`compute_simple_seb_fluxes`/`surface_out` flow; fixed two main regressions (slab lost
`T_rad`; the SEB dropped the per-cell CLM/`pft_alb` albedo under snow feedback). Gap 3
WIRED (`aa378b627`): real diurnal `cos(zenith)` in the trainer loader (Cooper decl. +
lon/local-time) + **insolation-weighted** albedo metric (model + ERA5 target, `sw≥0`
clipped for ssrd artefacts). Gaps 5+6 (`25c65242d`): terrain sky-view LW + blowing-snow
sublimation (opt-in) + rain-on-snow refreezing (frozen-soil = main's `enable_freeze_thaw`).

**Codex adversarial review** of gaps 3/5/6 + the merge re-application: 6 findings, all
fixed (`4b3ef4385`) — sky-view effective LW-up consistency, blowing-snow latent/vapor
export, config-scalar dtype casts, AD-safe blowing-snow branch, canopy+bands rejection,
insolation-weighted albedo *loss*. **Code-review agent: no correctness defects.**

**Re-tune (all gaps active, insolation-weighted metric):** loss 28.99→**28.36**, skin-T
RMSE 2.86→2.83 K (the metric now has a trainable signal — unlike the earlier flat re-tune).

**Global map (ERA5 vs CLM5-default vs tuned+gaps, `plot_land_field_maps.py --elev-bands`):**
skin-T RMSE **3.07→2.81 K** (annual-mean field 2.55→2.18 K); albedo 0.075→0.080
(comparable to actual-CLM). The elevation-band + zenith physics improves the global land
skin-T. Bake still gated on a coupled AMIP eval; the residual is substantially
forcing-driven (see above).
