#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=fesom_hybrid_ab_%j.log

# Full-model A/B of the HYBRID SSH halo transport (fesom_jax commit 0d1fe62) on
# forca20 at NPES GPUs: the 3-D fields keep the coloured transport, the
# free-surface CG's ~2x127 single-level exchanges go through ONE collective
# (padded dense all_to_all, or ragged) instead of K~14 ppermute rounds.
# Same code, same nodes, same physics as the Nature ladder (nature_fesom.sbatch:
# FULL model, JRA55 1958, PHC IC, dt=240, 150 steps), arms interleaved x2.
# Pre-registered (CLAUDE.md compute discipline):
#   numbers  : per_step ms for coloured / coloured+ssh:padded / coloured+ssh:ragged,
#              two repeats each; plus one short coloured arm with FESOM_CG_ITERS=1
#              that prints the CG trip count per solve (pins "~127 iterations").
#   CONFIRMS : best hybrid <= 0.70 x coloured at NPES=128 (the CG's 2-D exchanges
#              were the dominant term; adopt for the figure, ASK first).
#   REFUTES  : best hybrid >= 0.95 x coloured -> the 2-D transport is not the lever
#              (next: iteration count, i.e. FESOM_CG_CHEB, or the psums).
#   why not cheaper: the per-exchange cost is a fabric-latency property; the
#              micro-bench (jobs 27591010/11) isolates it but cannot say what
#              fraction of the real step it is.
#
#   NPES=128 sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 fesom_hybrid_ab.sbatch
#   NPES=64  sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 16 fesom_hybrid_ab.sbatch
set -uo pipefail
FJ=/work/bd1083/b309178/diffESM/fesom_jax
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
NPES="${NPES:?set NPES}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
STEPS="${STEPS:-150}"
(( NPES == NODES * 4 )) || { echo "ERROR: NPES=$NPES != 4 x $NODES nodes" >&2; exit 1; }
SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/fesom_hybrid_ab_j${SLURM_JOB_ID}}"; mkdir -p "$OUTDIR"
cd "$FJ" || exit 1
export PYTHONPATH="$FJ:${PYTHONPATH:-}"
export JDIST=1 GPUS_PER_NODE=4
unset FESOM_CG_CHEB
SHA=$(git rev-parse --short HEAD)
echo "=== $(date) forca20 npes=$NPES nodes=$SLURM_JOB_NODELIST fesom_jax=$SHA steps=$STEPS outdir=$OUTDIR ==="
run_arm() {  # tag extra-args...
  local tag=$1; shift
  local log="$OUTDIR/forca20_d${NPES}_${tag}.log"
  echo "--- arm $tag $(date '+%T')"
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
       --cpu-bind=none --kill-on-bad-exit=1 --export=ALL \
       "$PY" scripts/bench/bench_forward_scaling.py --name forca20 --mesh-dir "$M" --dist-dir "$D" \
         --npes "$NPES" --steps "$STEPS" --dt 240 --ic-dir "$M" --year 1958 --full 1 "$@" \
         < /dev/null > "$log" 2>&1 || echo "!!! FAILED arm $tag"
  grep -h "^\[bench\] mesh=\|\[bench-finite\]" "$log" | sed "s/^/[$tag] /"
}
for REP in 1 2; do
  run_arm "coloured_r$REP"   --halo coloured
  run_arm "sshpadded_r$REP"  --halo coloured --ssh-halo padded
  run_arm "sshragged_r$REP"  --halo coloured --ssh-halo ragged
done
# trip-count arm: short, diagnostic print on (perturbs timing, so it is NOT an A/B leg)
FESOM_CG_ITERS=1 STEPS=25 run_arm "cgiters_coloured" --halo coloured --steps 25
grep -h "cg-iters" "$OUTDIR/forca20_d${NPES}_cgiters_coloured.log" | awk '{print $2}' | sort | uniq -c | sort -rn | head -5
echo "=== DONE $(date) ==="
