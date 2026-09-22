#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=atm_mpas_halo_attrib_%j.log

# Where does the MPAS-atmosphere step go at the strong-scaling plateau?
# At 128 GPUs (subdivision 9, 20,480 cells/device, 26 levels, wide halo) the
# step is ~4.6 ms against a ~1.4 ms compute-only ideal, while the halo's
# SERIALISED LATENCY (one 13-14 round exchange + one allreduce, ~50/40 us
# each) is only ~0.7 ms.  The gap is therefore not latency.  This splits it
# with the halo's own measurement knobs, all on the SAME compiled program:
#   base       : production
#   NOCOMM=1   : every ppermute replaced by the identity  -> removes wire+wait
#   NOSTAGE=1  : the per-round gather/pack/scatter loop skipped as well
#   BALLAST=2  : every halo message sent twice              -> bytes slope
# The NOCOMM/NOSTAGE arms BREAK THE ANSWER and are never valid as physics;
# they are timings of a program with the same kernels and the same local
# region.
#
# PRE-REGISTERED (CLAUDE.md compute discipline):
#   numbers produced : steady step ms per arm at a fixed device count;
#                      wire+wait = base - NOCOMM; staging = NOCOMM - NOSTAGE;
#                      bytes = BALLAST2 - base (one extra copy of the payload).
#   CONFIRMS a staging lever : staging >= 25% of the base step at 128 GPUs
#                      (then fusing the per-round gather/pack/scatter is the
#                      next implementation).
#   REFUTES          : staging < 10% of the step; then the plateau is wire
#                      per round (bytes/padding) and the lever is round count.
#   why not cheaper  : the profiler on this stack does not record the halo
#                      collectives (2026-08-13), and a one-device run is a
#                      different program; only the in-situ knobs split it.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_mpas_halo_attrib}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
echo "nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <nd> <label> <env assignments...>
  local nd="$1" label="$2"; shift 2
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="mpas9_d${nd}_${label}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  echo "=== arm $tag $(date '+%F %T')  env: $*"
  if ! env "$@" timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --subdivision 9 --nlev 26 --lloyd 0 \
        --wide-halo --partition-method sfc --steps 60 --warmup 10 \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

for ND in ${ND_LIST:-128 32}; do
  arm "$ND" base     LEGOESM_MPAS_HALO_NOCOMM=0 LEGOESM_MPAS_HALO_NOSTAGE=0
  arm "$ND" nocomm   LEGOESM_MPAS_HALO_NOCOMM=1
  arm "$ND" nostage  LEGOESM_MPAS_HALO_NOSTAGE=1
  arm "$ND" ballast2 LEGOESM_MPAS_HALO_BALLAST=2
  arm "$ND" base_rep LEGOESM_MPAS_HALO_NOCOMM=0 LEGOESM_MPAS_HALO_NOSTAGE=0   # repeat: noise floor
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
