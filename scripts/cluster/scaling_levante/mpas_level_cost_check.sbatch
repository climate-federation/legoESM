#!/bin/bash -l
#SBATCH --job-name=lev_cost
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=lev_cost_%j.log
# ===========================================================================
# EVERY SCALING NUMBER WE PUBLISH IS TAKEN AT THE MOST EXPENSIVE LEVEL COUNT
# IN THE REPO'S OWN MEASURED TABLE. IS THAT TABLE STILL RIGHT?
#
# grids/vertical.py carries cost per cell per level, picoseconds, measured on
# one A100 in float32 with the icosahedral core at 163,842 cells, 60 timed
# steps, two replicates, spreads at or under 1% (2026-08-19, seven jobs).
# Selected rows:
#
#     levels   26     27     28     30     32     36     40     52
#     ps     4755   3861   4058   4591   1629   1706   1664   1621
#
# The scaling ladder runs --nlev 26 on every MPAS arm. Production AMIP runs
# 40. If that table holds, every published MPAS scaling number is taken on a
# configuration costing roughly 2.9x per cell-level what production costs,
# and 32 levels would be cheaper in TOTAL than 26 despite doing more work.
#
# That matters beyond the absolute numbers. The CPU atmosphere's unexplained
# 64 -> 128 rank slowdown is being chased as a communication problem; if the
# local kernel is anomalously expensive at 26 levels, the compute-to-comm
# balance under test is not production's, and the kink may not survive at 40.
#
# WHAT IS MEASURED, and the first arm is a KNOWN ANSWER. Subdivision 7
# (163,842 cells, the table's own size) at 26, 32 and 40 levels reproduces
# the published ratio or refutes it. Then subdivision 9 at the same counts
# asks whether it survives at the size the ladder actually runs. Four GPUs,
# one node, float32, arms interleaved, two repeats.
#
# FALSIFIABILITY, WRITTEN BEFORE SUBMIT.
#   CONFIRMS: at subdivision 7 the per-cell-level cost at 26 levels exceeds
#     32 and 40 by more than 2x, reproducing the table within its own 1%
#     spreads. Then the table stands and the ladder's level count is a real
#     finding about every number it produced.
#   REFUTES: the ratio at subdivision 7 is near 1. Then the table is stale --
#     it is a year-old measurement on one machine and the warning it feeds
#     says as much -- and the level count is not an issue. Say so and delete
#     the claim rather than carrying it.
#   The subdivision 9 arms are the question the table cannot answer; they are
#     reported whatever subdivision 7 does.
#
# NOTHING IS CHANGED BY THIS JOB. The level count is a configuration choice
# and moving it is a separate decision that belongs to the user.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_level_cost_check.sbatch
# ===========================================================================
set -uo pipefail
trap 'echo "LEV_COST_INCOMPLETE: killed at $(date +%T)"' TERM
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

# The collective-affecting flag is pinned: an inherited overlap setting would
# change the step for reasons that have nothing to do with level count.
export LEGOESM_XLA_OVERLAP=0
export NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=131072

ND=4
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/lev_cost_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "provenance: $($PY -c 'import legoesm.grids.vertical as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) outdir=$OUTDIR"

want=0; ok=0
for REP in 1 2; do
  for SUB in 7 9; do
    for NLEV in 26 32 40; do
      want=$(( want + 1 ))
      TAG="sub${SUB}_nlev${NLEV}_r${REP}"
      echo "=== $TAG $(date +%T) ==="
      if timeout --kill-after=30 600 srun --export=ALL --ntasks="$ND" \
          --ntasks-per-node="$ND" --gpus-per-node=4 --gpu-bind=none \
          --cpu-bind=cores --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / ND ))" \
          --kill-on-bad-exit=1 \
          "$PY" scripts/bench/bench_mpas_spmd_scaling.py --multicontroller \
          --n-devices "$ND" --subdivision "$SUB" --nlev "$NLEV" --lloyd 0 \
          --wide-halo --partition-method sfc --steps 30 --warmup 5 \
          --out "$OUTDIR/${TAG}.jsonl"; then
        ok=$(( ok + 1 ))
      else
        echo "ARM FAILED $TAG rc=$?"
      fi
    done
  done
done

echo "=== done: $ok of $want arms ==="
[ "$ok" -lt "$want" ] && { echo "LEV_COST_INCOMPLETE: $(( want - ok )) missing"; exit 1; }
$PY "$SUBMIT_DIR/scripts/bench/_summarise_level_cost.py" "$OUTDIR"
