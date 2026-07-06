# WeatherBench scale training — runbook (Derecho / Levante)

Data-parallel, multi-node training of legoESM on ERA5, evaluated on WeatherBench-2.
One MPI rank per GPU; each rank trains the same replicated model on a different
ERA5 shard and gradients are averaged across ranks. Three modes:
`physics` (tune the physics closures), `neural_gcm` (hybrid physics + ML — the
competitive lane), `sfno` (pure-ML emulator).

**Everything below is on `main`** (`git pull` first).

---

## 0. One-time setup (per machine)

Check out the repo and set up the conda env following the machine's own README:

- **Derecho** (NCAR, PBS): repo at `/glade/work/$USER/legoESM`; see
  `scripts/cluster/scaling_derecho/README.md` (interactive `qsub -I` conda recipe).
  Account `P08010000`.
- **Levante** (DKRZ, SLURM): repo at `/work/bd1083/$USER/legoESM`; conda env
  `legoesm-gpu`; account `bd1083_gpu`. See `scripts/cluster/scaling_levante/README.md`.

The launchers source the machine's `scripts/cluster/scaling_{derecho,levante}/_env.sh`
(modules, conda, account, and the fabric flags), so you don't set those by hand.

ERA5 streams from the public WeatherBench-2 GCS zarr, so the compute nodes need
outbound internet (Derecho/Levante compute nodes have it). It caches locally under
`.cache/wb_scale_era5`.

---

## 1. FIRST: single-GPU smoke (validate the wiring, ~minutes)

Always run this once before a multi-node job — it confirms the model/ERA5/loss
wiring end-to-end at tiny size (T21, 1 epoch, single GPU):

```bash
# on a single-GPU interactive/short job, from the repo root
python scripts/run/train_weatherbench_scale.py \
    --config config/wb/scale/train_07deg.yaml \
    --mode physics --smoke
```

Success = it loads ERA5, trains 1 epoch (a finite `loss=…` line), and writes a
checkpoint. If it errors, fix that before spending multi-node GPU-hours.

---

## 2. Launch the real multi-node run

Edit the node count at the top of the launcher (`select=<N>:…` for Derecho,
`--nodes=<N>` for Levante), then submit. `WB_MODE` picks the training mode.

**Derecho (PBS):**
```bash
cd /glade/work/$USER/legoESM
WB_MODE=neural_gcm qsub scripts/cluster/derecho/train_wb.pbs
```

**Levante (SLURM):**
```bash
cd /work/bd1083/$USER/legoESM
WB_MODE=neural_gcm sbatch --export=ALL,WB_MODE=neural_gcm scripts/cluster/levante/train_wb.slurm
```

Both launch, per node, 4 ranks × 1 A100 each, running:
```
python scripts/run/train_weatherbench_scale.py \
    --config config/wb/scale/train_07deg.yaml --mode $WB_MODE \
    --out $SCRATCH/wb_scale_$WB_MODE --resume --eval-wb2
```

`--resume` makes a walltime-killed job pick up from the last checkpoint on
resubmit. `--eval-wb2` runs the WeatherBench-2 scorecard on the held-out 2020
test year after training.

---

## 3. Options (CLI flags on the entry, override the YAML)

| Flag | Meaning | Default |
|---|---|---|
| `--mode` | `physics` \| `neural_gcm` \| `sfno` | `neural_gcm` |
| `--resolution` | degrees (0.7 = ~256×512 lat-lon) | `0.7` |
| `--epochs` | training epochs | `40` |
| `--multi-step-hours` | forecast rollout leads, e.g. `6,12` | `6,12` |
| `--lr` / `--optimizer` | learning rate / `adamw`\|`muon` | `3e-4` / `adamw` |
| `--grad-accum` | gradient accumulation (effective batch = ranks × this) | `1` |
| `--out` | checkpoint/output dir | `results/wb_scale` |
| `--resume` / `--eval-wb2` | resume from ckpt / run WB2 scorecard after | off |

Scale is **data-parallel**: more ranks (nodes×4) = larger effective batch +
faster wall-clock. 0.7° fits on one A100-80GB, so you scale for throughput, not
because the model doesn't fit.

---

## 4. Outputs

- `$OUT/epoch_XXXX.eqx` — per-epoch checkpoints (rank 0).
- With `--eval-wb2`: a WeatherBench-2 lead-time scorecard (RMSE + ACC + bias on
  Z500/T850/Q700/U-V/MSLP/… vs ERA5) from the scorer in `evaluations/`.

---

## 5. Honest notes

- **Competitive lane:** this is the **NeuralGCM tier** (hybrid physics + ML,
  0.7°) — competitive with GraphCast on deterministic skill to ~5 days at coarse
  resolution. It is not a 0.25° pure-ML emulator; the WB2 scorecard is the honest
  measuring stick.
- **Validation state:** the cross-rank gradient-average core is unit-verified
  (data-parallel mean == serial batch-mean); the end-to-end mode wiring is what
  the single-GPU smoke (step 1) validates — do not skip it.
- **`neural_gcm`/`sfno`** carry more learnable capacity than `physics`; `physics`
  is the cheapest sanity run.

---

## Related (same repo)

- **WeatherBench-2 scorer** — `evaluations/wb_forecast.py` + `wb_orchestrator.py`:
  turns any trained checkpoint into a WB2 lead-time scorecard. Reused by `--eval-wb2`.
- **T63 swap-and-train sweep (Ginsburg)** — `config/wb/sweep/stage1/` +
  `scripts/run/run_wb_sweep_stage1.py`: which physics scheme (per family) trains
  to the best 6 h forecast. `sbatch --array=0-9 scripts/run/_wb_sweep_stage1_runner.sbatch`,
  then rank with `scripts/validate/aimip_sweep_scorecard.py results/wb_sweep_stage1 config/wb/sweep/stage1/manifest.json`.
