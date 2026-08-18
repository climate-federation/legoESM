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
