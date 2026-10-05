#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=duo_gpu_scal
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# M7 G4, GPU venue: the FV3 duo COLUMN lane (CAM6 C24 deck, dt 300) on 1 / 2 / 6
# A100s, one day each, per-phase timers on.  The face layout shards the six
# cube faces over 2, 3 or 6 devices; Ginsburg A100 nodes carry 2 GPUs, so
#   N_GPUS=1 : one process, one device            (--nodes=1 --gres=gpu:1)
#   N_GPUS=2 : one process, two devices, auto     (--nodes=1 --gres=gpu:2)
#   N_GPUS=6 : six processes, 3 nodes x 2, spmd   (--nodes=3 --ntasks=6
#                                                   --ntasks-per-node=2 --gres=gpu:2)
# Submit with the matching sbatch geometry (see fv3_duo_gaps/b2/jobs.txt).
# EXTRA_FLAGS: one-variable additions (e.g. "--land-canopy-max-iters 10").
set -uo pipefail
N_GPUS=${N_GPUS:?1|2|6}
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
export LEGOESM_REPO="$WT"
export PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export AMIP_DATA=/burg-archive/glab/users/ac5006/legoESM/data/amip
export AMIP_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/legoesm_surfdata_c260716.nc
export CLM_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 LEGOESM_JIT_CACHE_DIR=""
export LEGOESM_PHASE_TIMERS=1
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
OUT=${OUT:-/burg-archive/glab/users/pg2328/fv3_duo_gaps/gpu_scaling/g${N_GPUS}}
mkdir -p "$OUT"
cd "$WT"; echo "sha $(git rev-parse --short HEAD) dirty $(git status --porcelain | grep -v '^??' | wc -l) nodes $SLURM_JOB_NODELIST N_GPUS=$N_GPUS"
source "$WT/config/amip/amip_production.ginsburg.sh"
# A preempted/requeued job (burst partition) resumes from the latest
# checkpoint instead of restarting day 0 (the first 60-day run lost 55 min).
latest=$(ls -1 "$OUT"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1)
RESTART=(); [ -n "$latest" ] && RESTART=(--restart-from "$latest")
echo "latest checkpoint: ${latest:-none}"
ARGS=(scripts/run/run_amip.py --config config/amip/amip_production_fv3duo_c24.yaml
      "${AMIP_PATH_FLAGS[@]}" --days "${DAYS:-1}" --checkpoint-days "${CHECKPOINT_DAYS:-${DAYS:-1}}"
      --diag-days 1 "${RESTART[@]}" ${EXTRA_FLAGS:-} --output "$OUT")
case "$N_GPUS" in
  1|2) srun -N 1 -n 1 "$PY" -u "${ARGS[@]}" ;;
  6)   srun --nodes=3 --ntasks=6 --ntasks-per-node=2 --mpi=pmix --gpu-bind=map_gpu:0,1 \
         --kill-on-bad-exit=1 "$PY" -u "${ARGS[@]}" --distributed --distributed-mode spmd ;;
  *)   echo "N_GPUS=$N_GPUS is not a face-sharding count on 2-GPU nodes"; exit 2 ;;
esac
echo "rc=$?"
grep -E "fv3_duo layout|phase timers|MPAS run COMPLETED" "$OUT"/*.log 2>/dev/null | tail -8
