# AIMIP at scale — Derecho / Levante

Train the AIMIP variants (`classical`, `column_nn`, `sfno_physics`) at
**~1° resolution (T106, 160×320) on ALL training years 1979–2014** on a single
80 GB A100. This is the scaled-up version of the Ginsburg T63 (~1.9°, subset of
years) runs.

## Why single-GPU (not multi-node)

AIMIP training is **single-device by design**: the spectral semi-implicit
dynamical core all-gathers the vertical levels into a dense `(nlev,nlev)`
per-wavenumber solve, and the spherical-harmonic transforms carry no
collectives. So there is no data/model-parallel axis to shard the training step
across GPUs. "At scale" therefore means a **bigger card** — the 80 GB A100 on
Derecho/Levante fits what Ginsburg's 24–40 GB could not:

- T106 (51 200 columns vs T63's 12 288) + 8 levels,
- RRTMGP radiation (not the gray fallback),
- 2-step autoregressive rollout,
- spatial-surface trainable coefficients,
- higher NN capacity (SFNO `embed_dim=96`/6 blocks; column MLP 512/6).

If you want to go beyond one card, the levers are ensemble (one variant/seed per
GPU, already how the three variants run) or the lat-lon C-grid PE training path
(`run_aimip_latlon.py`), which *can* SPMD-shard — but that path is separate and
not wired to this config.

## Data

`config/aimip/scale/base_t106_allyears.yaml`:
`train_windows` = 1979–2014 × 4 seasons (Jan/Apr/Jul/Oct), `n_days=1` →
**576 IC/target pairs** (full ENSO + seasonal cycle). Eval held out on
2015–2017. Prescribed-SST/sea-ice/insolation forcing is on (the
interannual-variability pathway). Needs the ERA5 WeatherBench-2 Zarr reachable
from the compute node (the training driver streams it; set the store in
`legoesm.training.era5_to_state.TrainingERA5Config` or the site cache).

## Run

Edit the site env first — **`scripts/cluster/scaling_{derecho,levante}/_env.sh`**:
`REPO` (repo path on GLADE / Levante work), the conda env with CUDA-JAX, and the
account. Then submit **one job per variant**:

```bash
# Derecho (PBS)
qsub -v VARIANT=sfno_physics scripts/cluster/aimip_scale/train_aimip_derecho.pbs
qsub -v VARIANT=classical    scripts/cluster/aimip_scale/train_aimip_derecho.pbs
qsub -v VARIANT=column_nn    scripts/cluster/aimip_scale/train_aimip_derecho.pbs

# Levante (SLURM)
SBATCH_ACCOUNT=<proj> VARIANT=sfno_physics \
  sbatch --export=ALL scripts/cluster/aimip_scale/train_aimip_levante.slurm
```

Each job **self-chains**: `run_aimip.py` writes a per-epoch checkpoint and
`--resume` continues at the next epoch, so a 15-epoch T106 run that exceeds one
walltime (12 h Derecho / 8 h Levante) resubmits itself (up to `CHAIN_MAX=12`
links) until `results/aimip_scale_t106/<variant>/<variant>/params.eqx` appears.
A manual resubmit resumes automatically (it detects existing epoch checkpoints).

## Cost estimate

At T106, RRTMGP + 2-step at 576 samples is ~10–15× the Ginsburg T63 per-epoch
cost. Budget roughly:

| variant       | ~time/epoch (A100-80) | 15 epochs |
|---------------|-----------------------|-----------|
| sfno_physics  | ~1–1.5 h              | ~18 h (2 links) |
| column_nn     | ~1–1.5 h              | ~18 h |
| classical     | ~3–4 h (RRTMGP)       | ~50 h (5 links) |

Lower `aimip_n_epochs` in the base, or set `aimip_radiation: gray` for the NN
variants (they replace physics anyway) to cut wall clock.

## After training

Checkpoints land in `results/aimip_scale_t106/<variant>/<variant>/params.eqx`.
Evaluate with the same AMIP inference / reinit-hindcast / fleet-figure pipeline
used at T63 (`scripts/run/run_aimip_amip_inference.py --variant <v> --ckpt <that>`,
then `scripts/plot/plot_aimip_fleet_annual.py`), pointing `--suite` at the T106
suite so the grid matches the checkpoint.
