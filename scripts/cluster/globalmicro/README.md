# globalmicro — reference LES for Morrison warm-rain scaling (Ginsburg)

Generates the warm-cloud LES targets that `scripts/run/train_scm_morrison_scale.py`
trains the local warm-rain scale-net against.

## Layout on Ginsburg
- Repo source: `/burg/glab/users/kl3231/Projects/globalmicro/legoESM` (rsync of this tree)
- SAM case decks: `/burg/glab/users/kl3231/SAM/SAM6.11.1/{DYCOMS_RF01,BOMEX,RICO}` (snd/lsf/sfc)
- Outputs: `/burg/glab/users/kl3231/Projects/globalmicro/results/les_{dycoms,bomex,rico}/`
- GPU env: shared `…/jn2808/.conda/envs/legoesm` (jax 0.9.1 + CUDA12); source via `PYTHONPATH`.

## Run
```bash
# 1. sync source (from local repo root)
rsync -az --delete --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='results/' --exclude='diagnostics/' --exclude='*.zarr/' --exclude='*.nc' \
  --exclude='docs/references/' --exclude='.physics-validator/' \
  ./ ginsburg:/burg/glab/users/kl3231/Projects/globalmicro/legoESM/

# 2. submit (one GPU job per case)
ssh ginsburg
mkdir -p /burg/glab/users/kl3231/Projects/globalmicro/{logs,results}
cd /burg/glab/users/kl3231/Projects/globalmicro/legoESM
sbatch -p short scripts/cluster/globalmicro/run_les_reference.sbatch dycoms
sbatch -p short scripts/cluster/globalmicro/run_les_reference.sbatch bomex
# RICO full GCSS protocol (128^3 x 24h) is too big for a 12h slot — run reduced:
sbatch -p short scripts/cluster/globalmicro/run_les_reference.sbatch rico \
    --nx 64 --ny 64 --nz 100 --hours 10
```
Each writes `<out>/{dycoms,bomex,rico}_les_final.npz` with planar-mean `z, theta, qv,
qc` (+ case metrics), consumed by `train_scm_morrison_scale.py --reference <npz>`.
DYCOMS/BOMEX are non-precipitating (precip≈0); RICO drizzles (~0.3 mm/day) and is the
warm-rain training signal. `glab1` GPUs are often saturated — `-p short` (general,
12 h, RTX 8000) usually starts in seconds.
