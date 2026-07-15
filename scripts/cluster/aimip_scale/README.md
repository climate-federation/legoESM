# AIMIP at scale — Derecho / Levante

Train the AIMIP variants (`classical`, `column_nn`, `sfno_physics`) at
**~1° resolution (T106, 160×320) on ALL training years 1979–2014** on a single
80 GB A100. This is the scaled-up version of the Ginsburg T63 (~1.9°, subset of
years) runs.

## Parallelism: the MODEL is single-device; TRAINING is data-parallel (#985)

The model step itself is single-device (the spectral SI solve is a dense
`(nlev,nlev)` per-wavenumber system; the SH transforms carry no collectives) —
there is no model-parallel axis. But since PR #1020 the TRAINER shards each
chunk's **samples** across MPI ranks (model replicated, gradients averaged via
mpi4jax host-staged allreduce), giving ~N× wall clock on N GPUs: measured
**~3.4× on 4×A100** (chunk incl. compile; ~4× steady-state), 288 updates/
epoch/rank at 4 ranks. `aimip_data_parallel: true` in the base YAML is inert at
nproc==1; the Derecho launcher auto-detects the rank count from the `-l select`
line (see Run below). `sfno_full` is the one DP-rejected variant.

An 80 GB card is still required per rank — that is what fits T106 (51 200
columns vs T63's 12 288) + 8 levels, autoregressive rollout backprop, and
spatial-surface coefficients. NOTE the in-practice sfno_physics capacity is
`embed_dim=64`/4 blocks with physics off (`variant_sfno_physics.yaml`) — the
96/6 + RRTMGP combination in the original plan OOM'd at gradient init even at
80 GB; the 120 h curriculum phase was cut for the same reason (72 h cap).

Beyond one node the same DP launch extends (`select=2:...` etc.; ppn is
derived), and the other levers remain ensemble (one variant/seed per job) or
the separate SPMD lat-lon path (`run_aimip_latlon.py`, not wired here).

## Data (v2 — dense + curriculum, the ACE2-gap upgrades)

`config/aimip/scale/base_t106_allyears.yaml`:
`train_windows` = 1979–2014 × 4 seasons (Jan/Apr/Jul/Oct), `n_days=2` →
**1152 IC/target pairs** (full ENSO + seasonal cycle). Eval held out on
2015–2017. Prescribed-SST/sea-ice/insolation forcing is on (the
interannual-variability pathway). Needs the ERA5 WeatherBench-2 Zarr reachable
from the compute node (the training driver streams it; set the store in
`legoesm.training.era5_to_state.TrainingERA5Config` or the site cache).

v2 additions (all driven by the base YAML, nothing extra to configure):

- **Chunked streaming** (`aimip_chunk_windows: 18`): the 1152-pair set is
  loaded 18 windows (144 samples) at a time per epoch, so host RAM stays
  bounded; sample shapes are constant across chunks → the jitted train
  step never retraces.
- **Rollout curriculum** (`aimip_rollout_curriculum: 12h×2, 24h×2, 72h×2,
  120h×1` epochs): each phase supervises one autoregressive rollout to
  the phase lead against the ERA5 target at that lead — free-run
  stability is trained in, NeuralGCM-style, instead of bolted on by a
  post-hoc finetune.
- **Budget constraints** (NN paths): the SFNO's global-mean lnps tendency
  is projected out (approximate dry-mass fixer) and q_v is clipped ≥ 0
  after every forced rollout step.
- **Classical GHG pin**: with `aimip_radiation: rrtmgp`, training
  radiation carries the mid-training-period (≈1996) GHG concentrations
  via the traced forcing dict. NN variants get no CO₂ input (AIMIP-1
  convention).

The `variant_*.yaml` overlays MUST sit next to the suite files in
`config/aimip/scale/` — `run_aimip.py --suite` loads
`variant_<name>.yaml` from the suite's own directory and hard-fails
otherwise.

## Run

Edit the site env first — **`scripts/cluster/scaling_{derecho,levante}/_env.sh`**:
`REPO` (repo path on GLADE / Levante work), the conda env with CUDA-JAX, and the
account. Then submit **one job per variant**:

```bash
# Derecho (PBS) — data-parallel, 4 GPUs (the production default).
# SELECT must REPEAT the -l select string: the self-chaining qsub re-applies it
# on every link (it cannot read this job's request). LEGOESM_REPO is required
# when running from a git worktree (and is propagated down the chain).
qsub -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB \
     -v VARIANT=sfno_physics,LEGOESM_REPO=<repo>,SELECT=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB \
     scripts/cluster/aimip_scale/train_aimip_derecho.pbs
# Single-GPU (serial trainer path, byte-identical to the pre-DP behavior):
qsub -v VARIANT=classical,LEGOESM_REPO=<repo> scripts/cluster/aimip_scale/train_aimip_derecho.pbs

# Levante (SLURM) — serial launcher (DP wiring is Derecho-only so far)
SBATCH_ACCOUNT=<proj> VARIANT=sfno_physics \
  sbatch --export=ALL scripts/cluster/aimip_scale/train_aimip_levante.slurm
```

Each Derecho job **self-chains**. How that actually works (fixed 2026-07-15 —
the original layout NEVER chained on a walltime-bound link, because the PBS
walltime kill terminates the script before its resubmit line runs):

- The launcher stops training ITSELF at `LINK_BUDGET_S` (default 11 h 15 m,
  i.e. 45 min before the 12 h cap) via `timeout`; the per-chunk checkpoint
  (`chunk_latest.eqx`: model + optimizer state + position, #972) is already on
  disk, so nothing is lost but the in-flight chunk.
- A budget stop (rc 124/137) or a clean-but-unfinished exit chains the next
  link (up to `CHAIN_MAX`), re-applying `$SELECT` and `$LEGOESM_REPO`. Real
  failures (OOM, crash) stop the chain.
- Every link resumes automatically off `chunk_latest.eqx` / `epoch_*.eqx`; so
  does a manual resubmit of the same qsub line.

Mixing serial and DP links in ONE training run is legal for resume but changes
the effective batch (1 → nranks samples/update) mid-trajectory — pick one mode
per run.

## Monitoring a chain (2-minute daily check)

```bash
qstat -u $USER                                   # one link R or Q at all times
LOG=$(ls -t aimip_scale_t106.o* | head -1)
head -1 "$LOG"                                   # link=N advancing; resume=--resume; nranks as submitted
grep "Saved mid-epoch" "$LOG" | tail -3          # chunk counter advancing
grep -iE "nan|Traceback|RESOURCE_EXHAUSTED" "$LOG" | tail -5
ls -l results/aimip_scale_t106/<variant>/<variant>/   # chunk_latest.eqx mtime fresh
```

Compare losses same-chunk across epochs (chunks are different data — within-
epoch variation is meaningless). Known danger moments needing a deliberate
look: the FIRST chunk of each curriculum phase (new rollout length → full
recompile + a larger memory peak; the 72 h phase is the historical OOM point).
A stale `chunk_latest.eqx` mtime with the job still running = a hang —
investigate, don't wait for walltime.

`aimip_chunk_prefetch` is OFF at T106: the prefetched chunk currently
materialises on the GPU and OOMs the DP first step (#985; re-enable once the
producer pins to host).

## Cost estimate

At T106, RRTMGP at 1152 samples is ~20–30× the Ginsburg T63 per-epoch cost,
and the curriculum's later phases (72 h / 120 h rollouts) cost proportionally
more per sample than the 12 h phase. Budget roughly (7 curriculum epochs):

| variant       | ~12h-phase epoch (A100-80) | full curriculum |
|---------------|----------------------------|-----------------|
| sfno_physics  | ~2–3 h                     | ~40–60 h (4–6 links) |
| column_nn     | ~2–3 h                     | ~40–60 h |
| classical     | ~6–8 h (RRTMGP)            | ~120 h+ (chain) |

The self-chaining handles the walltime; note the LR schedule restarts at each
resume (warned in the log — weights are unaffected). Lower the curriculum
epoch counts in the base, or set `aimip_radiation: gray` for the NN variants
(they replace physics anyway) to cut wall clock.

Lower `aimip_n_epochs` in the base, or set `aimip_radiation: gray` for the NN
variants (they replace physics anyway) to cut wall clock.

## After training

Checkpoints land in `results/aimip_scale_t106/<variant>/<variant>/params.eqx`.
Evaluate with the same AMIP inference / reinit-hindcast / fleet-figure pipeline
used at T63 (`scripts/run/run_aimip_amip_inference.py --variant <v> --ckpt <that>`,
then `scripts/plot/plot_aimip_fleet_annual.py`), pointing `--suite` at the T106
suite so the grid matches the checkpoint.
