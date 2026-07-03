# Compare-to-Reanalysis — Operator Runbook

The end-to-end procedure for the LES-informed `clubb_lite` correction campaign:
**run AMIP/CMIP → compare to ERA5 → spin off LES for the worst columns → diagnose
the closure coefficient → re-run keeping only bias-improving rounds → deploy the
correction → verify the bias improved.** Design + rationale live in
`docs/COMPARE_REANALYSIS.md`; this is just the commands, in order. All paths are
relative to the repo root; `PY=.venv/bin/python`.

Every step is *preflighted on cheap synthetic data* so you confirm the chain works
in your environment **before** committing a multi-day HPC job.

---

## 0. Generate a base config

A schema-valid AMIP `clubb_lite` base `ExperimentConfig` (the production grid +
physics; `turbulence=clubb_lite` is required for the deploy):

```bash
$PY scripts/experiment/write_amip_clubb_lite_config.py configs/amip_clubb_lite.json \
    --resolution 8 --nlev 10
```

> `--resolution 8 --nlev 10` is a **coarse starter** (an 8×16 grid) for the smoke / dry-run
> below — the generator prints a NOTE reminding you to scale `--resolution` / `--nlev` **up**
> for a production ERA5 comparison, and to lower `--dt` with resolution (a too-large `dt` at
> higher resolution diverges; the baseline-divergence guard would then fail the run loud).
> `--days` (default 200) sets the **climatology window** the model time-mean is computed over
> — keep it long enough for a stable mean and **aligned to the ERA5 window** (§6); start from a
> spun-up restart (or run long enough) so the initial transient does not bias the mean.

> **Supply a land-sea mask for a realistic comparison (`--land-mask-path`, iter 454).** WITHOUT
> it the config is **flat (no land)** — the model treats the whole globe as prescribed-SST ocean,
> so over **continents** the comparison to ERA5 (which has land) is apples-to-oranges and would
> **dominate the worst-column ranking** (the LES correction would then chase the missing-land
> artifact, not a closure error). For a realistic run pass `--land-mask-path <land-sea fraction
> file>`; the model then runs land physics over land, and the campaign's `OCEAN_ONLY=1` /
> verify `--ocean-only` can meaningfully **exclude** the land columns (which need a land mask to
> identify — without one they read as ocean and are *not* excluded). A flat config is fine only
> for an aquaplanet smoke; the generator prints this NOTE when no mask is given. The mask file is
> any regular lat-lon NetCDF with a land-sea variable — an **ERA5 `lsm`** invariant (fraction 0–1)
> or a **CMIP6 `sftlf`** (percent 0–100) both work; `load_land_fraction` auto-detects the variable,
> rescales percent→fraction, and regrids to the model grid (the end-to-end `land_mask_path →
> static_land_fraction → ocean_valid_mask` chain is locked by a test, iter 455).
>
> **No real mask yet? Generate a synthetic one (iter 486).** To TEST the realistic ocean-only
> path (the mask actually EXCLUDING land columns, not the flat all-ocean no-op) without a real
> download: `python scripts/data/make_synthetic_land_mask.py /tmp/land.nc` writes an idealized
> lat-lon `lsm` (a NH land band) that regrids to any model grid as a land/ocean mix, then
> `smoke_compare_reanalysis.py --ocean-only --land-mask-path /tmp/land.nc` ranks a strict SUBSET
> (land excluded) through the full harness. NOT a real distribution — for production extract the
> invariant `128_172_lsm` from the local NCAR-RDA ERA5 archive (or a CMIP6 `sftlf`).

## 1. Preflight the SETUP (seconds, no model run)

Confirms config load + ERA5 ingest (regrid + vertical interp) + grid/sigma build +
scheme/method validation. Use `--era5-zarr` to validate **your real** ERA5 store's
ingest (variable names, levels, lat ordering); omit it for a pure environment smoke.
`--mode cmip` preflights the coupled driver.

**Required ERA5 variables** (the loader resolves common aliases, e.g. `temperature`↔`t`):
the 3D state `temperature`, `u_component_of_wind`, `v_component_of_wind`,
`specific_humidity` plus the 2D `surface_pressure` are **required** — a store missing any
of these fails loud listing *all* missing ones at once (it must not silently load as zeros
and corrupt the bias). `skin_temperature` (SST) and `geopotential_at_surface` are optional
(zero-filled if absent; supply SST for accurate worst-column environment tags + `--mode cmip`).

```bash
# environment smoke (synthetic ERA5):
$PY scripts/experiment/smoke_compare_reanalysis.py
# real-ERA5 ingest smoke:
$PY scripts/experiment/smoke_compare_reanalysis.py --era5-zarr /path/to/era5.zarr --mode amip
# validate the REALISTIC flag combo + a non-lat-lon grid (iter 484/485):
$PY scripts/experiment/smoke_compare_reanalysis.py --ocean-only --surface-flux \
    --grid-type cubed_sphere
```

A green `[smoke] PASS` means the turnkey chain is wired. Pass `--ocean-only`/`--surface-flux`
to validate the realistic flag combo (the iter-484 ocean-only flat-mask bug was grid-shape-
specific), and `--grid-type {latlon,cubed_sphere}` to smoke the FULL harness (ERA5
regrid → compare → rank → ocean mask) on a structured non-lat-lon grid before the HPC run.
(`gaussian` is unsupported for clubb_lite — the spectral loop does not thread its physics
state, issue #405; `--align-insolation` is NOT a smoke flag — it needs the offline
`--local-era5-date`, not the Zarr smoke.)

### 1b. Preflight the C_K SENSITIVITY (go/no-go, before a multi-day run)

The LES→C_K loop can only lower a bias the boundary-layer closure CONTROLS; an
idealization-dominated bias (gray radiation / no real SST) is **not** moved by C_K (iter 412),
so a multi-day run on it converges to NO improvement. `ck_sensitivity_vs_era5.py` runs two
short C_K runs (or a `--days-sweep`) against real ERA5 and **exits 0 (GO — the loop can lower
the bias) / 1 (NO-GO — idealization-dominated)**, mirroring the campaign exit-code gate, so the
launch can be guarded:

```bash
# REALISTIC go/no-go (rrtmgp, two C_K runs vs real ERA5) gates the launch:
$PY scripts/experiment/ck_sensitivity_vs_era5.py --local-era5-dir $LOCAL_ERA5_DIR \
    --local-era5-date $DATE --resolution 8 --nlev 20 --radiation rrtmgp --days 10 --c-k 0.4 1.0 \
  && sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
# OR ask if RUN LENGTH is the lever (sweep — NO-GO only if the trend is FLAT):
$PY scripts/experiment/ck_sensitivity_vs_era5.py ... --radiation rrtmgp --days-sweep 1 3 6
```

**Match the campaign's column domain (`--ocean-only`, iter 477).** If the campaign ranks
ocean columns (`OCEAN_ONLY=1`), pass `--ocean-only --land-mask-path $LAND_MASK` (and the same
`--max-land-fraction`) here too, so the go/no-go scores the SAME columns. Without it the
pre-flight measures the TOTAL-column sensitivity over a flat config, where a land-model-driven
bias the closure cannot move dilutes the fraction and can FALSELY gate an ocean-only campaign
NO-GO. `--ocean-only` on a flat model (no land mask) prints a NOTE and is a safe no-op.

NO-GO means: add realism (real radiation/SST) and/or run longer (the `--days-sweep` prints the
~days to the feasibility floor) BEFORE spending HPC hours on the loop. The per-level report
still shows C_K controls the boundary layer even when the free-trop radiation error dominates
the TOTAL bias — it tunes the right place, but needs a realistic free-troposphere to move the
total.

> **Run the `--radiation rrtmgp` pre-flight ON A COMPUTE NODE** (a short `srun`/sbatch), not the
> login node: rrtmgp's radiation graph is compute-heavy (a measured realistic run at
> res 4 / nlev 20 / 2 days did not finish in ~16 min on a login node, whereas the same run
> under the `gray` default — same coupled driver + 0.25° ERA5 ingest — completes interactively).
> The `gray` default is interactive-quick but always reports NO-GO (idealization-dominated), so
> it only confirms "do not run the loop under gray"; the **realistic** go/no-go needs rrtmgp on a
> compute node. The turnkey wrapper `scripts/cluster/compare_reanalysis/preflight_ck_sensitivity.sbatch`
> runs it as a batch job and exits with the go/no-go code, so the campaign launch can be a
> dependency: `jid=$(sbatch --parsable preflight_ck_sensitivity.sbatch); sbatch --dependency=afterok:$jid run_correction_campaign.sbatch`.
> To MATCH an ocean-only campaign, pass the same env vars to the pre-flight wrapper as to the
> campaign one: `OCEAN_ONLY=1 LAND_MASK_PATH=<mask> MAX_LAND_FRACTION=<thr>` (iter 478) — both
> launchers then score the same ocean columns, so the go/no-go reflects the campaign's domain.
> For a CUBED-SPHERE campaign also set `GRID_TYPE=cubed_sphere` (iter 493) so the sensitivity is
> measured on the campaign's grid family (gaussian is unsupported for clubb_lite, issue #405;
> the OSSE pre-flight already uses the exact `--config`, so it is grid-faithful by construction).
>
> *Tip:* run the **gray** pre-flight FIRST as a quick (minutes, interactive) chain sanity-check —
> it reports NO-GO with the per-level BL-localization, so a clean run confirms the real-ERA5
> ingest + model + compare are wired end-to-end before you submit the compute-heavy rrtmgp job
> (validated iter 445: res 4 / nlev 10 / days 1 vs 20170901 ERA5 → exit 1, mean bias 11.25,
> most-controllable level σ=0.97 in the BL).
>
> **The OTHER go/no-go — the perfect-model OSSE (iter 491).** Complementary to the C_K-sensitivity
> pre-flight (which asks "can the loop move the REAL bias?"), the OSSE asks "does the loop RECOVER
> a KNOWN coefficient in a controlled twin?" — it needs NO real ERA5 (the model's own run with a
> known C_K is the pseudo-truth) but runs the model several times, so it is also a compute-node
> batch job. The turnkey wrapper `scripts/cluster/compare_reanalysis/preflight_osse.sbatch` runs
> `run_perfect_model_osse.py` and exits 0 (GO — recovered) / 1 (NO-GO), so it too can gate the
> campaign as a dependency. Set `CONFIG=<base>.json` and (for parity with the realistic campaign)
> `SURFACE_FLUX=1`; `COEFFICIENTS=C_K,Pr_t,C_eps` validates the multi-coefficient campaign, and
> `FINE_RESOLUTION=<N>` validates the cross-grid env-kernel transfer.

## 2. Launch the campaign (resumable, self-requeuing)

Edit the `#SBATCH` account/partition/time + the `CONFIG`/`ERA5`/`OUT` paths at the
top of the launcher, then submit. The job **self-requeues** on the wall-clock limit
(`scontrol requeue` on the pre-timeout signal) and `--resume`s from the last atomic
checkpoint — submit **once** for a multi-day run.

**Set `SPINUP_DAYS`** (env var → `--spinup-days`) for a real climatology run: it discards the
first N days of model time (the un-equilibrated spin-up) from the time-mean BEFORE comparing to
ERA5, so the loop targets the *equilibrated* bias — not a spin-up-contaminated one (iter 445 saw
a 91 K aloft T-bias in a 1-day run that is largely spin-up). Keep `--days` (the CONFIG run length)
enough longer than `SPINUP_DAYS` for a stable post-spin-up mean. (A multi-day `--days` with
`SPINUP_DAYS=0` prints a launch NOTE reminding you.)

> **Insolation season — a tracked comparability limitation (iter 447/449/450).** The AMIP **SST** is
> aligned to the ERA5 date (via the forcing), but by default the radiation **insolation** is NOT:
> the driver derives the day-of-year from `day_to_calendar(day)` and, with `insolation_start_doy`
> unset, the model's insolation is **January-based** (day 0 → Jan 1) — a September comparison runs
> ~winter (off-season) insolation. This biases the **high-latitude** worst-column ranking most.
> Three ways to align it:
> - **`ALIGN_INSOLATION=1 sbatch …` / `--align-insolation` (iter 450, turnkey; CODEX PENDING).**
>   The campaign auto-derives `insolation_start_doy` from `--local-era5-date` and injects it into
>   the base config (so every corrected round carries it). Off by default; fails loud without an
>   offline date. The one-flag way to align an offline run.
> - **`insolation_start_doy=<doy>` in the config (iter 449, the seam; CODEX PENDING).** Maps model
>   day 0 to that noleap day-of-year for the insolation **only** — decoupled from the
>   relative-indexed SST forcing. All seven insolation sites (five `day_to_calendar` day_of_year +
>   two gray `daily_mean_insolation`) route through one driver offset (`_insolation_day`); the
>   integer-day offset shifts the season while preserving the diurnal phase; unset (`None`) is
>   byte-identical to the legacy path (validated by the bitwise restart test). Sep 1 → `244`.
> - **A January ERA5 window (e.g. `20170115`) — fully validated.** Day 0 → Jan 1 aligns BOTH the
>   SST forcing AND the Jan-based insolation with no config change.
>
> Without `--align-insolation`, the campaign prints a launch NOTE (exact day-of-year +
> `insolation_start_doy=<doy>`) when the offline `--local-era5-date` is off-season (Apr–Sep).

```bash
CONFIG=configs/amip_clubb_lite.json ERA5=/path/to/era5.zarr \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
# CMIP (coupled ocean):
MODE=cmip COUPLED_PRESET=aquaplanet CONFIG=... ERA5=... \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
```

The launcher runs its own `--dry-run` preflight first, then the campaign, writing the
corrected-coefficient JSON to `$OUT` and a per-round bias trajectory to the `.out` log.

> **First-round startup (not a hang):** the first round JIT-compiles the dycore + the
> spin-off LES (XLA), so expect **several minutes with no `.out` progress** before the first
> `[campaign] round 1/N …` line appears — compilation, not a stall (subsequent rounds reuse the
> compiled code and the per-round line prints as each completes). For a *quick* check that
> skips the model run, use the synthetic smoke (§1) or a campaign `--dry-run` (both return in
> seconds). A coarse real-data run is dominated by this compile + the realistic spin-off LES
> (`--les-hours`, default 2 h); scale `--les-hours` down only for a smoke, not production.

> **Persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`, iter 453).** With **rrtmgp**
> radiation the JIT-compile is the >16-min bottleneck (§1b), and this launcher **self-requeues**
> on the wall-clock limit — without a cache each requeue recompiles from scratch. The sbatch
> sets `JAX_COMPILATION_CACHE_DIR` (default `results/compare_reanalysis/jax_cache`; override to a
> **persistent**, not node-local, path) so the expensive compile is written once and reused
> across requeues + relaunches. The campaign's `--compilation-cache-dir` flag defaults to that
> env var; `--cache-min-compile-secs` (default 30) caches only the slow compiles. (A different
> per-round C_K still recompiles within a run — compiling once via a *traced* C_K is a tracked
> architecture follow-up; the cache is what de-risks the requeue today.)

### 2b. Fully-offline realistic AMIP (local NCAR-RDA ERA5, no network)

When the cluster has **no outbound network** (no gcsfs / `gs://` Zarr) but mirrors the
NCAR-RDA ERA5 collection `d633006` (`e5.oper.an.{pl,sfc}.*.ll025*.nc` on the regular 0.25°
lat-lon grid), the campaign reads ERA5 **entirely offline**: `LOCAL_ERA5_DIR` (+
`LOCAL_ERA5_DATE=YYYYMMDD`) routes the compare reference through `--local-era5-dir`, and
(AMIP only) `AMIP_FORCING=1` ALSO builds the prescribed SST/sea-ice forcing from the **same**
archive via `--amip-forcing-from-local-era5` — so one archive drives both the boundary
condition *and* the comparison, with no hand-built `dataset=custom` config. Use a realistic
radiation scheme for the empirical run:

```bash
# 0. a realistic AMIP config (rrtmgp radiation, ~20 levels):
$PY scripts/experiment/write_amip_clubb_lite_config.py configs/amip_rrtmgp.json \
  --radiation rrtmgp --nlev 20 --resolution 48
# 1. launch fully-offline (LOCAL_ERA5_* overrides the Zarr ERA5):
LOCAL_ERA5_DIR=/glade/.../ERA5 LOCAL_ERA5_DATE=20170901 AMIP_FORCING=1 \
  CONFIG=configs/amip_rrtmgp.json \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
# A FULLY-REALISTIC multi-day offline run (iter 469 — all the consistency knobs in one launch),
# e.g. a 30-day September climatology over real land (CONFIG built with --land-mask-path):
LOCAL_ERA5_DIR=/glade/.../ERA5 LOCAL_ERA5_DATE=20170901 AMIP_FORCING=1 \
  CONFIG=configs/amip_rrtmgp.json ALIGN_INSOLATION=1 OCEAN_ONLY=1 SURFACE_FLUX=1 \
  SPINUP_DAYS=10 AMIP_FORCING_N_MONTHS=1 ERA5_N_DAYS=30 ERA5_N_TIMES=720 \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
```

The launcher exposes every consistency knob as an env var: `ALIGN_INSOLATION` (season, §2),
`OCEAN_ONLY` + `SURFACE_FLUX` (the realistic LES combo, above), `SPINUP_DAYS` (exclude spin-up),
`AMIP_FORCING_N_MONTHS` (SST coverage), and `ERA5_N_DAYS` + `ERA5_N_TIMES` (match the reference
window to the model `days`). The launch NOTEs (§2/§3) cross-check them and warn on a mismatch.

The launcher's `--dry-run` preflight **builds the forcing** (validating the archive read +
the daily subsample + the output write) and prints an `AMIP forcing: built from … → …` line
confirming the offline boundary condition is in place — *before* the multi-day run. The
monthly-hourly ERA5 single-level files are subsampled to daily (`--amip-forcing-hour-stride`,
default 24) so the loader does not OOM on the full hourly month.

> **Match `--days` to the forcing coverage (iter 458/459).** `--amip-forcing-from-local-era5`
> builds the SST forcing from `--amip-forcing-n-months` **consecutive months** (default **1**, the
> `--local-era5-date`'s chunk). A run **longer** than that coverage cyclically **repeats** the
> forcing (`get_forcing_at_time` wraps over the forcing period — intended for a *full annual*
> cycle), so the SST replays while the insolation advances seasonally and they **desync**. So for a
> multi-month climatology, set `AMIP_FORCING_N_MONTHS` / `--amip-forcing-n-months` to **≈
> ceil(days/31)** so the concatenated forcing covers the whole window (iter 459); the AMIP-default
> `days=200` needs ~7 months. The campaign prints a NOTE when `days` exceeds the (scaled) forcing
> span, so a mismatch is caught before the run.

> **Ocean-only ranking (`OCEAN_ONLY=1` / `--ocean-only`, iter 451).** Over realistic land, the
> AMIP surface is the model's OWN land model, carrying large surface biases (soil moisture, snow,
> skin temperature) the turbulence closure cannot fix — so blind worst-column ranking can spend
> the scarce LES budget on land columns where C_K is the wrong lever. `OCEAN_ONLY=1` restricts the
> ranking to OCEAN columns (where the prescribed SST pins the surface, so a model-vs-ERA5 bias is
> attributable to the atmospheric column incl. the closure). A column is ocean where
> `land_fraction <= --max-land-fraction` (default 0.5); the mask is sourced from the model's own
> `ModelDriver.static_land_fraction()` and fails loud if no ocean columns exist. A no-op on an
> aquaplanet (all-zero land fraction → all columns rankable). Off by default (rank all columns).

> **Recommended realistic LES config: `--ocean-only --surface-flux` (iter 364/465).** The spin-off
> LES develops convective turbulence (needed for a valid C_K diagnosis) from the surface buoyancy
> flux. `--surface-flux` derives that flux from the column SST via the GCM bulk scheme — valid over
> **ocean**. So pair it with `--ocean-only`: without it, a LAND worst-column would get a flux from a
> non-ocean SST (a fill value) → a wrong/invalid diagnosis that wastes the LES budget (the campaign
> prints a NOTE when `--surface-flux` is set with a land mask but no `--ocean-only`). Without
> `--surface-flux`, the ocean-column LES is forced only by large-scale subsidence/shear and may not
> develop turbulence (low diagnosis-validity). For a smoke/aquaplanet, `--surface-flux` alone is
> fine (every column is ocean). Both off by default.

## 3. Read the result

The output JSON is **self-describing**: a `health` block (`improved` / `stalled` /
`no_valid_diagnoses` / `non_finite_bias` / …) + a `summary` (per-round + per-variable
bias) + the corrected per-column field. Plot it:

```bash
$PY scripts/plot/plot_campaign_bias_trajectory.py   $OUT   # global + per-variable bias vs round
$PY scripts/plot/plot_corrected_coefficient_field.py $OUT  # where/how the coefficient changed
```

A `health.status == "improved"` already demonstrates clause 5 **in-sample** (each
accepted round lowered the bias on a real model re-run, under the monotonic gate).

The output also records an `averaging` block (`era5_n_times`, `era5_time_idx`) — the
ERA5 window the bias was measured against. **Window alignment is the operator's
responsibility:** the model run is time-meaned and compared to this ERA5 mean, so they
must represent the *same* climatological period (a `--era5-n-times 1` snapshot vs a
multi-day model mean compares weather to climate — a spurious bias the loop would then
"correct"). Use a multi-time ERA5 climatology (`--era5-n-times N`) matched to the model
averaging window.

> **Offline reference spans only ONE day by default (`--era5-n-days`, iter 460).** The local
> NCAR-RDA path opens a single day's chunk (~24 hourly times), so `--era5-n-times` can average at
> most that one day — a single-day reference vs a multi-day model climatology is still
> weather-vs-climate. For a multi-day offline climatology set `--era5-n-days D` (loads D consecutive
> days from `--local-era5-date`, ~24·D times; concatenated) and `--era5-n-times` up to `24·D`. The
> campaign fails loud if `--era5-time-idx + --era5-n-times` exceeds the loaded coverage, **and
> prints a NOTE (iter 463) when the offline reference window (`era5_n_times/24` days) is much
> shorter than the model `days`** — so the default `--era5-n-times 1` against a multi-day run is
> caught at launch. (Zarr `--era5-zarr` already spans times, so `--era5-n-days` is offline-only.)

### 3b. Diagnose a `stalled` / `no_valid_diagnoses` correction (why didn't the bias move?)

A no-go verdict is NOT necessarily a broken loop — it is usually the **config**. The run prints
and records exactly which, so debug in this order:

1. **Was the bias C_K-INSENSITIVE?** If you launched with **idealized radiation** (`gray`/`none`),
   the run prints a launch `WARNING: radiation='gray' is IDEALIZED ... use radiation='rrtmgp'`,
   and the verdict (`stalled` / OSSE `no_change`) carries a `C_K-INSENSITIVE` note. Under gray the
   bias is idealization-dominated, so a 40 % C_K change moves it only ~0.0025 (vs an ~11 real-ERA5
   bias) — **no C_K can fix it.** Re-run with `--radiation rrtmgp` (the §1b go/no-go uses it).

2. **Did the spin-off LESs develop turbulence?** Each run prints an aggregate
   `LES realism: M/N realistic, K rejected (a× laminar, b× blow-up, …)` line (and an identical
   `les_realism` block in the output JSON). Many `laminar` rejections ⇒ the LES is too short /
   unforced — lengthen `--les-hours` or check the surface forcing (`--surface-flux`).

3. **Realistic LES but still no correction?** Run the single-column debug CLI on a worst column —
   it prints, per column, `realism: REALISTIC — N/M valid diagnosis levels`. `REALISTIC` with a low
   `N` means the rejection is the **diagnosis** (insufficient resolved shear/variance for a
   down-gradient closure, or too few levels below the top sponge), NOT turbulence — refine the LES
   **resolution**, not its duration:

```bash
$PY scripts/run/run_column_les.py --manifest worst.json --restart amip_restart.npz \
    --resolution $RES --nlev $NLEV --method clubb_coefficient --out col_coef.npz
```

`n_diagnoses_valid` in the per-round report (and `..._total` in the summary) is the column count
that received a real correction; `0` with worst columns flagged means every spin-off was rejected —
the cause is one of the three above, NOT the correction logic.

## 4. Preflight the DEPLOY (seconds, no model run)

Confirm the campaign output deploys onto your production base config + grid (the
per-column correction is **grid-verified** so it can't land on the wrong cells) and
see which coefficients changed:

```bash
$PY scripts/experiment/check_campaign_deploy.py \
    --base-config configs/amip_clubb_lite.json --campaign-output $OUT
```

> **Use the EFFECTIVE config as `--base-config` for a runtime-flag campaign (iter 464).** If the
> campaign used runtime flags (`--align-insolation` sets `insolation_start_doy`,
> `--amip-forcing-from-local-era5` injects the SST forcing), those are **not** in the original
> base JSON — so the campaign writes the effective config it actually ran to
> `<out>.effective_config.json` (next to `--out`). Pass **that** as `--base-config` to deploy/re-run
> on the **same window**, so the production run reproduces the SST boundary + insolation season the
> C_K was calibrated on. (A **held-out** verify (§6) re-derives the *window-specific* forcing +
> `insolation_start_doy` for the held-out window, but keeps the non-window settings.)

## 5. Deploy + run a production simulation

The correction is a **runtime** override (it does not serialize), so inject it at run
time. `build_deployed_config` returns the ready-to-run config:

```python
from scripts.experiment.check_campaign_deploy import build_deployed_config
from scripts.run.run_correction_campaign import make_base_driver_builder

cfg, grid = build_deployed_config("configs/amip_clubb_lite.json", "OUT.json")
build_driver, _ = make_base_driver_builder("amip")   # AMIP (prescribed SST)
driver = build_driver(cfg)
driver.run(...)        # the LES-corrected production run; save a restart for step 6
```

For **CMIP** (coupled ocean), pass a RESOLVED preset *object* (not the name string):

```python
from legoesm.driver.coupled_config import PRESETS
build_driver, _ = make_base_driver_builder("cmip", coupled_preset=PRESETS["aquaplanet"]())
```

> **Cross-grid (grid-agnostic) deploy (`build_env_kernel_deployed_override`, iter 475).** To deploy
> a *cheap coarse-grid* calibration on an *expensive fine production grid* by environmental
> similarity, run the campaign with `--feedback-strategy environment` (writes `<out>.env_kernel.json`),
> then run the **target** model once to get its `ColumnState` and call
> `build_env_kernel_deployed_override("<out>.env_kernel.json", "<target_config>.json", target_state)`
> → `(deployed_config, grid, coverage)`. Unlike `build_deployed_config` (same-grid), this evaluates
> the kernel on the target grid's *environment*, so it needs the target model state. Check the
> returned `coverage` (`fraction_in_hull`) — pass `min_fraction_covered=` to fail loud on an
> out-of-hull (near-no-op) transfer — and **validate the transfer first** with
> `run_perfect_model_osse.py --fine-resolution` (the twin go/no-go for this exact path).
>
> **Turnkey CLI preflight (`check_cross_grid_deploy.py`, iter 483).** The command-line analog
> of `check_campaign_deploy.py` for the cross-grid path — no Python needed:
> ```bash
> $PY scripts/experiment/check_cross_grid_deploy.py --kernel <out>.env_kernel.json \
>     --target-base-config <target_config>.json --target-restart <target_run>.npz \
>     --sst-npz <target_sst>.npz   # [--min-fraction-covered 0.8] [--allow-approximate-sst]
> ```
> The target grid/nlev/vertical-coord come from `--target-base-config`; `--sst-npz` supplies the
> kernel's *dominant* predictor (the target run's prescribed/coupled SST) and is REQUIRED unless
> `--allow-approximate-sst` (else it fails loud — a deploy on a fabricated SST is untrustworthy).
> It reports the per-column override range + `coverage` (`fraction_covered`/`fraction_in_hull`/
> `sst_from_model`); like the same-grid path the override is NOT serialized — inject it at runtime
> via `apply_env_kernel_override(kernel, target_env)` in the production driver.

## 6. Verify on a HELD-OUT window (the rigorous check)

Produce two restarts on a time window **not used for training**:
- **deployed_restart** — step 5 (`build_deployed_config` → run), the LES-corrected config;
- **baseline_restart** — the SAME run with the *un-corrected base* config
  (`build_driver(base_cfg)` instead of the deployed `cfg`), so the comparison isolates
  the correction's effect.

Then compare both to ERA5 on that held-out window (no overfitting to the training period):

```bash
$PY scripts/validate/compare_amip_era5.py \
    --restart deployed_restart --baseline-restart baseline_restart \
    --grid-type latlon --resolution 8 --nlev 10 \
    --era5-zarr /path/to/era5.zarr --era5-time-idx <held-out index>
```

It reports the per-variable global bias **before vs after** — the clause-5
"did updating the parameters improve the biases" answer on held-out data.

> **Match the ranking domain (`--ocean-only`, iter 452).** If the campaign ran with
> `OCEAN_ONLY=1`, add `--ocean-only --base-config <the run's config>` here so the held-out
> bias (per-variable, combined, AND the exit-code verdict) is measured over the SAME ocean
> columns the correction optimized — otherwise land columns the closure never touched dilute
> the before/after signal. The mask is sourced from the config's own
> `ModelDriver.static_land_fraction()` (its grid must match `--grid-type/--resolution/--nlev`),
> reusing the campaign's `ocean_valid_mask`. Off by default (global bias, all columns).

> **Match the vertical coordinate to your run.** Pass `--vertical-coord {sigma,hybrid}`
> (default `hybrid`, the `GridConfig` default) and, for hybrid, `--p-top-Pa` to match the
> *checkpoint's* run — they set the pressure levels the model state is placed on. The CLI
> now **verifies these against the restart's recorded config and fails loud on a mismatch**
> (so you cannot silently compare on the wrong levels); a pure-sigma run therefore needs
> `--vertical-coord sigma`. The campaign's `--dry-run` likewise rejects a per-column
> `area_weights`/`valid_mask`/`lat`/`lon` whose shape does not match the reference grid —
> all caught in the cheap preflight, not after a multi-day run.

## Automated gating (exit codes)

Every step exits **0 only on success**, so the whole workflow chains with `&&` (or a
SLURM dependency) and stops at the first genuine failure:

| step | exit 0 (proceed) | non-zero (stop) |
|---|---|---|
| `smoke_compare_reanalysis.py` | the turnkey chain validated | a config/ERA5/dry-run failure |
| `run_perfect_model_osse.py` | `recovered` / `transferred` | `bias_only` / `no_change` / `out_of_hull` / `diverged` |
| `run_correction_campaign.py` | health `improved` | `stalled` / `no_change` / `no_valid_diagnoses` / `non_finite_bias` |
| `check_campaign_deploy.py` | a real per-column correction | a no-op (all-default) output |
| `compare_amip_era5.py --baseline-restart` | the held-out **combined** bias fell | the correction did not generalize |

```bash
$PY .../smoke_compare_reanalysis.py && $PY .../run_perfect_model_osse.py … \
  && sbatch .../run_correction_campaign.sbatch   # the .sbatch already gates on its own dry-run
  # then, after the run: check_campaign_deploy && deploy && compare_amip_era5 --baseline-restart
```

The deployable JSON is written even on a non-zero campaign exit (the `health` block records
why); only the exit *status* gates — a workflow reading the JSON is unaffected.

---

**Cross-resolution deploy:** run the campaign with `--feedback-strategy environment`
to export a `<OUT>.env_kernel.json` that re-evaluates on **any** target grid by
environmental similarity (a cheap low-res campaign deploys on a high-res run). Validate
that transfer in a twin BEFORE paying for the high-res run with
`run_perfect_model_osse.py --fine-resolution <N>`: it learns the kernel on `--config`'s
(coarse) grid and deploys + measures the paired bias change on a fine grid built from the
same base config at resolution `N`. Exit 0 iff the verdict is `transferred` (well-covered
**and** the fine bias fell); `out_of_hull` means the fine climate lies outside the
coarse-sampled environment, so the kernel extrapolates and the change is untrusted.

**Controlled go/no-go before real ERA5:** `scripts/validate/run_perfect_model_osse.py`
runs the loop in an identical-twin (the model's own run with a KNOWN coefficient is
the pseudo-truth) and checks it both lowers the bias and recovers the known parameter.
For the go/no-go to faithfully predict your **realistic** run, pass the SAME
diagnosis-path flags the campaign will use — `--surface-flux` (iter 471) and
`--orographic-forcing` — since they change *how* the closure is diagnosed; `--ocean-only`
is column-selection (orthogonal in a self-consistent twin). For a **simultaneous
multi-coefficient** campaign (`--coefficients C_K,Pr_t,C_eps`), pre-flight it with the OSSE's
matching `--coefficients` mode (iter 472): the truth is the model defaults and the biased
start is each default × `--multi-bias-factor`; the verdict (exit-code-gated) requires **all**
coefficients to recover, so it catches compensating-error interactions a single-C_K go/no-go
misses.

---

## 7. Distributed / HPC-scale (very large MPAS meshes)

For a mesh too large for one rank, run the campaign under MPI: each rank owns a slice
of the global mesh, diagnoses + corrects only its owned cells, and the loop reduces
the bias/verdict collectively. Call the **same code on every rank**; the partition,
the rank-local model wiring, and the collective hooks are all set up internally:

```python
from mpi4py import MPI
from legoesm.training.campaign_summary import campaign_health, summarize_campaign
from legoesm.training.distributed_campaign import assemble_global_campaign_result
from scripts.run.run_correction_campaign import (
    build_campaign_output_dict, build_distributed_mpas_campaign,
)

# 1. run on every rank — return_layout=True hands back the partition for the persist:
result, layout = build_distributed_mpas_campaign(
    global_mesh=mesh, reference=era5_ref, area_weights=mesh.grid_area, n_worst=64,
    build_local_driver=build_local_driver,        # (cfg, local_mesh) -> rank-local driver
    base_atm_config=base_cfg, extract_column_state=extract, sigma=sigma,
    n_iterations=10, return_layout=True)

# 2. assemble the rank-local corrected field into the GLOBAL field (collective):
gres = assemble_global_campaign_result(result, layout, corrected_field="C_K")

# 3. write the deployable JSON on rank 0 ONLY (same format as the single-process OUT.json):
if MPI.COMM_WORLD.Get_rank() == 0:
    summary = summarize_campaign(gres, promotion_key="clubb_lite_C_K")
    out = build_campaign_output_dict(
        gres, grid_provenance={"grid_type": "mpas", "ncol": int(mesh.nCells)},
        summary=summary, health=campaign_health(summary), corrected_field="C_K")
    # _atomic_write_json("OUT.json", out)   -> deploy exactly like steps 4–6 above.
```

The output is the **same** deployable artifact as the single-process path, so the
deploy + held-out verify (steps 4–6) are unchanged.

> **Monitor the spin-off LES realism at HPC scale (§3b, distributed).** The single-process CLIs
> print the `LES realism:` breakdown automatically; the distributed path is library-only, so
> wrap the `run_les` you pass to `make_les_diagnose_fn` in `_RealismCapture` and collective-sum
> the per-rank counts into the GLOBAL breakdown — `0` realistic ⇒ debug per §3b:
>
> ```python
> from scripts.run.run_correction_campaign import (
>     _RealismCapture, realism_summary, reduce_realism_summary_mpi)
> from legoesm.parallel.reductions import global_sum_mpi             # the allreduce-SUM
>
> run_les = _RealismCapture(my_run_forced_les)          # then make_les_diagnose_fn(..., run_les_fn=run_les)
> # ... after the distributed campaign, on EVERY rank: ...
> local = realism_summary(run_les.breakdowns)           # this rank's owned columns (None if no LES)
> if local is not None:
>     g = reduce_realism_summary_mpi(local, global_sum_mpi)   # collective → SAME on every rank
>     if MPI.COMM_WORLD.Get_rank() == 0:
>         print(f"LES realism: {g.n_realistic}/{g.n_total} realistic, {g.n_rejected} rejected")
> ```

> **MPI version requirement (correctness):** the collective reductions need a compatible
> JAX / mpi4jax stack. There are **two compatible generations** (the two packages must be paired
> *within* a generation — a cross pairing is the genuinely-incompatible case):
> - **legacy custom-call:** `jax>=0.8,<0.10` + `mpi4jax>=0.8,<0.9`.
> - **FFI-based (verified iter 333):** `jax 0.10.x` + `mpi4jax 0.9.x` (e.g. `jax==0.10.0` +
>   `mpi4jax==0.9.0.post1`) — the FFI release the legacy note awaited; the distributed
>   compare-reanalysis suite (serial-vs-MPI equivalence + differentiability, 17 tests) passes
>   on it. **Do NOT pin `jax<0.10` if you are on the FFI stack** — that would force the broken
>   `mpi4jax 0.8` (custom-call removed in jax 0.10).
>
> A pairing outside BOTH generations (e.g. `jax 0.10` + `mpi4jax 0.8`, or `jax 0.8` + `mpi4jax
> 0.9`) emits a runtime WARNING from `legoesm.parallel.reductions` and **may produce incorrect
> MPI results**. Confirm your stack with a tiny `mpirun -np 2 … -m pytest tests/distributed/`
> before a multi-day job (set `LEGOESM_MPI_STRICT_COMPAT=1` to make a mismatch a hard error).
> The single-process path does not use MPI and is unaffected.
