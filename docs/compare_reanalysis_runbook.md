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

## 1. Preflight the SETUP (seconds, no model run)

Confirms config load + ERA5 ingest (regrid + vertical interp) + grid/sigma build +
scheme/method validation. Use `--era5-zarr` to validate **your real** ERA5 store's
ingest (variable names, levels, lat ordering); omit it for a pure environment smoke.
`--mode cmip` preflights the coupled driver.

```bash
# environment smoke (synthetic ERA5):
$PY scripts/experiment/smoke_compare_reanalysis.py
# real-ERA5 ingest smoke:
$PY scripts/experiment/smoke_compare_reanalysis.py --era5-zarr /path/to/era5.zarr --mode amip
```

A green `[smoke] PASS` means the turnkey chain is wired.

## 2. Launch the campaign (resumable, self-requeuing)

Edit the `#SBATCH` account/partition/time + the `CONFIG`/`ERA5`/`OUT` paths at the
top of the launcher, then submit. The job **self-requeues** on the wall-clock limit
(`scontrol requeue` on the pre-timeout signal) and `--resume`s from the last atomic
checkpoint — submit **once** for a multi-day run.

```bash
CONFIG=configs/amip_clubb_lite.json ERA5=/path/to/era5.zarr \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
# CMIP (coupled ocean):
MODE=cmip COUPLED_PRESET=aquaplanet CONFIG=... ERA5=... \
  sbatch scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
```

The launcher runs its own `--dry-run` preflight first, then the campaign, writing the
corrected-coefficient JSON to `$OUT` and a per-round bias trajectory to the `.out` log.

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

## 4. Preflight the DEPLOY (seconds, no model run)

Confirm the campaign output deploys onto your production base config + grid (the
per-column correction is **grid-verified** so it can't land on the wrong cells) and
see which coefficients changed:

```bash
$PY scripts/experiment/check_campaign_deploy.py \
    --base-config configs/amip_clubb_lite.json --campaign-output $OUT
```

## 5. Deploy + run a production simulation

The correction is a **runtime** override (it does not serialize), so inject it at run
time. `build_deployed_config` returns the ready-to-run config:

```python
from scripts.experiment.check_campaign_deploy import build_deployed_config
from scripts.run.run_correction_campaign import make_base_driver_builder

cfg, grid = build_deployed_config("configs/amip_clubb_lite.json", "OUT.json")
build_driver, _ = make_base_driver_builder("amip")   # or "cmip", coupled_preset=...
driver = build_driver(cfg)
driver.run(...)        # the LES-corrected production run; save a restart for step 6
```

## 6. Verify on a HELD-OUT window (the rigorous check)

Run the baseline and the deployed model, then compare both to ERA5 on a **different**
time window than training (no overfitting to the training period):

```bash
$PY scripts/validate/compare_amip_era5.py \
    --restart deployed_restart --baseline-restart baseline_restart \
    --grid-type latlon --resolution 8 --nlev 10 \
    --era5-zarr /path/to/era5.zarr --era5-time-idx <held-out index>
```

It reports the per-variable global bias **before vs after** — the clause-5
"did updating the parameters improve the biases" answer on held-out data.

---

**Cross-resolution deploy:** run the campaign with `--feedback-strategy environment`
to export a `<OUT>.env_kernel.json` that re-evaluates on **any** target grid by
environmental similarity (a cheap low-res campaign deploys on a high-res run).

**Controlled go/no-go before real ERA5:** `scripts/validate/run_perfect_model_osse.py`
runs the loop in an identical-twin (the model's own run with a KNOWN coefficient is
the pseudo-truth) and checks it both lowers the bias and recovers the known parameter.
