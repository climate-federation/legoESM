# Reproducing the offline land skin-T / albedo bias evaluation + tuning

Fixed-protocol runbook for the ERA5-forced offline `multilayer_land` skin-temperature
and surface-albedo bias maps and the bias-targeted re-tune. **Hold the protocol fixed**
(forcing + its sampling, grid, `--days`, params, metric) whenever comparing runs — a
metric that moved because the sampling/grid changed is a confound, not a result
(see CLAUDE.md § Validation, "Controlled comparison").

## 1. Forcing (ERA5 climatology from ARCO, anonymous GCS)

The canonical forcing is a **24-hour diurnal** monthly climatology (24 hours removes the
Jensen `σT⁴` warm bias that a 4-synoptic-hour sample carries). Two fetch drivers:

- **One-shot** (`scripts/data/fetch_era5_hourly_climatology.py`) — local/fast networks:
  ```
  PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_hourly_climatology.py \
      --stride 8 --years 2019 2020 --days 6 16 26 --out /tmp/era5_hourly.npz   # ~2deg, 24h
  ```
  ARCO streams **per timestep at 0.25° then subsamples**, so `--stride` does NOT reduce
  fetch time; a full 2yr×3day×24h fetch is ~1728 timesteps (~50 min).

- **Chunked / resumable** (`scripts/data/fetch_era5_chunked.py`) — for the 1°/24h grid or
  any short-foreground / no-background-network environment. One partial per month, skips
  completed months, `--merge` assembles without network:
  ```
  # repeat until all 12 months on disk (each call fits a short time budget):
  PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_chunked.py \
      --stride 4 --years 2019 2020 --days 6 16 26 --months 1 2 3 4 5 6 --dir /tmp/ck_1deg
  PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_chunked.py \
      --stride 4 --years 2019 2020 --days 6 16 26 --months 7 8 9 10 11 12 --dir /tmp/ck_1deg
  PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_chunked.py \
      --merge --dir /tmp/ck_1deg --out /tmp/era5_1deg_24h.npz                  # 1deg, 24h
  ```
The CLM surface map (`soil_albedo`, `std_elev`, PFT, texture) auto-regrids to the npz
grid via `clm_surface_map.load_clm_surface`, so the npz grid drives the whole setup — no
separate surface-data step per resolution.

## 2. Region bias eval (fixed protocol)

`scripts/tmp/_eval_regions.py <npz> <params.json> <days> [bands 0/1]` prints the
area-weighted **skin-T bias** and **albedo bias** for Sahara / Tibet / north>60 / global.
Always state (npz, params, days, bands) next to every number.

## 3. Global bias maps

```
PYTHONPATH=. JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda .venv/bin/python \
    scripts/plot/plot_land_field_maps.py --npz /tmp/era5_hourly.npz \
    --tuned results/land_tuned_allgaps.json --days 4 --elev-bands --out /tmp/land_bias_maps
```
Rows: skin-T, albedo. Cols: ERA5 / CLM5-default−ERA5 / tuned−ERA5 (diverging bias maps).
The `# PANEL-BIAS ...` stdout lines report each panel's area-weighted mean bias.

Result on the rigorous forcing (era5_hourly, days=4, `land_tuned_allgaps`): tuning halves
the skin-T warm bias (CLM5-default **+1.51 K → tuned +0.78 K**) and tightens albedo
(+0.024 → +0.011). RMSE 3.03 → 2.57 K.

## 4. Bias-targeted re-tune (`--lam-tbias`)

`train_multilayer_land_era5.py` gained `--lam-tbias` = a global skin-T **bias** penalty
(area-weighted mean of `T_model − T_era`, squared) on top of the RMSE term; logs a
`T-bias` column. Example:
```
PYTHONPATH=. JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda .venv/bin/python \
    scripts/run/train_multilayer_land_era5.py --diurnal-npz /tmp/era5_hourly.npz \
    --days 4 --n-sub 100000 --iters 150 --lam-tbias 30 --no-prefilter \
    --init-from baked --out results/land_tuned_lowbias.json
```
Use the **full grid** (`--n-sub 100000`): a small subsample gives a noisy bias gradient
that overshoots at lr=3e-2 and diverges. Strong penalties (lam≳100) also diverge.

**Finding — the offline warm bias is NOT param-tunable to ≤+0.30 K.** The offline
skin-T is set by the **soil thermal equilibrium**, so it is nearly insensitive to the
coolable params (`pft_emis`, `pft_ch` moved 0.0001 / 0.0000 in the fit); the only param
the bias gradient moves is `snow_dcrit`, which affects snow regions, not the warm
deserts/subtropics that dominate the +0.78 K bias. The residual warm bias is
**forcing-driven** (the standing "real forcing removes the warm bias" result), so the
genuine path to a low skin-T bias is a **coupled run** (model's own radiation), not
offline param tuning. A `+0.30 K` map seen earlier was a *forcing artifact* (a different,
now-lost skin-T target), not a better model — the model itself is unchanged from the
`3888b683b` (July-2) baseline (verified: same +0.79 K on the same forcing).
