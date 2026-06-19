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

The output also records an `averaging` block (`era5_n_times`, `era5_time_idx`) — the
ERA5 window the bias was measured against. **Window alignment is the operator's
responsibility:** the model run is time-meaned and compared to this ERA5 mean, so they
must represent the *same* climatological period (a `--era5-n-times 1` snapshot vs a
multi-day model mean compares weather to climate — a spurious bias the loop would then
"correct"). Use a multi-time ERA5 climatology (`--era5-n-times N`) matched to the model
averaging window.

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
build_driver, _ = make_base_driver_builder("amip")   # AMIP (prescribed SST)
driver = build_driver(cfg)
driver.run(...)        # the LES-corrected production run; save a restart for step 6
```

For **CMIP** (coupled ocean), pass a RESOLVED preset *object* (not the name string):

```python
from legoesm.driver.coupled_config import PRESETS
build_driver, _ = make_base_driver_builder("cmip", coupled_preset=PRESETS["aquaplanet"]())
```

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
