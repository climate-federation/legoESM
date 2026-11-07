#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=bare_smoke
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/bare_smoke_%j.log
# Smoke of the bare-mode bench edits (value check, per-rank receipt, shutdown
# barrier): single-process virtual devices, then a 6-rank gloo srun.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$WT/packages/core:$WT/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
cd $WT
echo "sha=$(git rev-parse HEAD) bench_sha256=$(sha256sum scripts/bench/bench_fv3_duo_spmd_scaling.py | cut -c1-16)"
echo "== single-process (6 virtual devices) =="
XLA_FLAGS=--xla_force_host_platform_device_count=6 $PY scripts/bench/bench_fv3_duo_spmd_scaling.py --mode bare --single-process --n-timed 10
echo "single rc=$?"
echo "== 6-rank gloo srun =="
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo
srun --kill-on-bad-exit=0 $PY scripts/bench/bench_fv3_duo_spmd_scaling.py --mode bare --n-timed 30
echo "srun rc=$?"
echo DONE bare_smoke
