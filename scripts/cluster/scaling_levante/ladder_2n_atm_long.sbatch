#!/bin/bash -l
# TEMPLATE — submitted via ladder_2n_submit.sh with sbatch --export and
# per-arm -N/-J/-o overrides. Runs ONE (lane, n_dev) strong-scaling point
# with the 2026-08-18 best receipted env for that lane.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:40:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
# Required from --export: LANE (mpas|latlon), NDEV (1..128)
# Optional: SUBDIV (default 9), NLAT (default 2048; NLON = 2*NLAT)
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
# lat-lon overlap flags are receipted (-8%); set BEFORE _env.sh which
# consumes LEGOESM_XLA_OVERLAP at source time. MPAS: overlap null/harmful.
if [ "$LANE" = "latlon" ]; then export LEGOESM_XLA_OVERLAP=1; else export LEGOESM_XLA_OVERLAP=0; fi
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ladder2n"
mkdir -p "$OUTDIR"
SUBDIV="${SUBDIV:-9}"; NLAT="${NLAT:-2048}"; NLON=$((2 * NLAT))
GPN=4; NODES=$(( (NDEV + GPN - 1) / GPN ))
NTPN=$(( NDEV < GPN ? NDEV : GPN ))
echo "=== ladder $LANE ndev=$NDEV nodes=$NODES $(date +%T) git=$(git rev-parse --short HEAD) ==="
# best receipted env, uniform per curve (2026-08-18):
#   both lanes: NCCL mcp2p pair (-13..-18% MPAS@64, -6.1% LL@32, -8% rag@64)
#   mpas: wide halo + ragged 2-collective exchange (5.73 ms receipt)
#   latlon: packed stage-entry exchange (-11.8%@128) + overlap flags
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 NCCL_P2P_NET_CHUNKSIZE=131072
rc=0
if [ "$LANE" = "mpas" ]; then
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=1 JAX_ENABLE_X64=0 \
    timeout 5400 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision "$SUBDIV" --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --halo-strategy ppermute --reorder-for "$NDEV" \
      --out "$OUTDIR/mpas_s${SUBDIV}_n${NDEV}.jsonl" || { echo "ARM FAILED"; rc=1; }
else
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  JAX_ENABLE_X64=0 \
    timeout 5400 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 \
      --steps 60 --warmup 10 \
      --out "$OUTDIR/latlon_${NLAT}_n${NDEV}.jsonl" || { echo "ARM FAILED"; rc=1; }
fi
if [ "$LANE" = "mpas" ]; then F="$OUTDIR/mpas_s${SUBDIV}_n${NDEV}.jsonl"; else F="$OUTDIR/latlon_${NLAT}_n${NDEV}.jsonl"; fi
"$PY" -c "
import json, math, sys
rows = [json.loads(l) for l in open('$F')]
d = rows[-1]
ms = d['steady_median_ms']
assert math.isfinite(ms) and ms > 0 and d.get('n_devices') == $NDEV
print(f'$LANE n=$NDEV: {ms:9.3f} ms  {d.get(\"mcells_per_s\",0)/1000:7.2f} GC/s')" \
  || { echo "receipt invalid"; rc=1; }
echo "DONE rc=$rc"; exit $rc
