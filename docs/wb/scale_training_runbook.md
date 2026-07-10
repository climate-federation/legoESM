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
`.cache/wb_scale_era5`. The `gs://` reader needs `gcsfs` — included in the `ml`
extra since #817 (`pip install -e ".[dev,ml]"`); an env built with `--extras dev`
alone hits `ModuleNotFoundError: gcsfs` at the first data load.

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

`--smoke` runs **gray radiation** (32×64×8, 1 epoch) so the wiring check compiles
in minutes: the full rrtmgp k-distribution graph's JIT compile alone blows past a
40-min single-GPU walltime inside the differentiable checkpointed rollout, while
gray radiation flows through the identical wiring under test. To smoke the rrtmgp
compile itself (with a bumped walltime), set `smoke_radiation: rrtmgp` in the YAML.

---

## 2. Launch the real multi-node run

Edit the node count at the top of the launcher (`select=<N>:…` for Derecho,
`--nodes=<N>` for Levante), then submit. `WB_MODE` picks the training mode.

**Derecho (PBS):**
```bash
cd /glade/work/$USER/legoESM
WB_MODE=neural_gcm qsub scripts/cluster/derecho/train_wb.pbs
# semi-implicit spectral training core (#817 blocker 1):
WB_MODE=neural_gcm WB_CORE=spectral qsub scripts/cluster/derecho/train_wb.pbs
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
| `--training-core` | `latlon` (explicit C-grid) \| `spectral` (Gaussian **semi-implicit**; #817) | `latlon` |
| `--resolution` | **check only** — must match the YAML grid (180/n_lat); edit the YAML to change resolution | derived from YAML |
| `--epochs` | training epochs | `40` |
| `--multi-step-hours` | forecast rollout leads, e.g. `6,12` | `6,12` |
| `--lr` / `--optimizer` | learning rate / `adamw`\|`muon` | `3e-4` / `adamw` |
| `--grad-accum` | gradient accumulation (effective batch = ranks × this) | `1` |
| `--out` | checkpoint/output dir | `results/wb_scale` |
| `--resume` / `--eval-wb2` | resume from ckpt / run WB2 scorecard after | off |

Scale is **data-parallel**: more ranks (nodes×4) = larger effective batch +
faster wall-clock. 0.7° fits on one A100-80GB, so you scale for throughput, not
because the model doesn't fit.

### The two #817 blockers and their levers

- **Exploding adjoint (blocker 1).** The explicit lat-lon core's training
  adjoint grows ~×1.3 per 30 s step (a numerical mode of the explicit core, not
  weather) → `value_and_grad` NaN past ~6 h even with a finite forward.
  **Lever: `--training-core spectral`** (or `WB_CORE=spectral` on the
  launchers) — the Gaussian/spectral core with the Hoskins–Simmons
  semi-implicit step treats the fast gravity-wave terms implicitly, keeping the
  adjoint bounded, and its grid has no pole cells, so `dt` stays at the
  configured `spectral.dt` (1800 s default) instead of collapsing to seconds.
  Configure via the YAML `spectral:` block (`n_max`, `dt`, `semi_implicit`
  [default true], `si_substeps`, `hyperdiff_coeff`, `spectral_filter_strength`).
  Per the #829 T10 dt-sweep, SI pays off **only where the gravity-wave CFL
  binds** (fine truncation) — at very coarse truncation the explicit spectral
  core is already adjoint-stable, so don't expect a coarse-grid demonstration.
  A model trained on the spectral core does **not** transfer its physics-head
  wiring back to the lat-lon production core 1:1 — score it with the spectral
  scorer path.
- **Full-BPTT memory (blocker 2).** The segment `lax.scan` used to store every
  step's carry (527 GiB physics/gray, 970 GiB neural_gcm/f64 at 0.7°). Fixed on
  `main` by #841: rollouts of ≥256 steps automatically use nested **√N
  checkpointing** (`_sqrt_checkpointed_scan`, exact gradients, O(√N) carry
  storage — 0.7° ≈1 TB → ~26 GB); shorter segments keep per-step remat. No flag
  needed — it engages on step count.

---

## 4. Outputs

- `$OUT/epoch_XXXX.eqx` — per-epoch checkpoints (rank 0).
- With `--eval-wb2`: currently a **pointer hook** (#817 papercut, honest doc):
  it logs the instruction to run `evaluations.wb_orchestrator` against the
  trained checkpoint — the scorecard itself (RMSE + ACC + bias on
  Z500/T850/Q700/U-V/MSLP vs ERA5) is produced by that separate scorer run,
  not inline by the trainer. Wiring the in-process scorecard is tracked as a
  follow-up.

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
