#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=fesom_cgpoly_ab_%j.log

# Chebyshev CG preconditioner (CGPOLY, FESOM_CG_CHEB=3) on top of the HYBRID
# transport (coloured 3-D + padded CG) on forca20 at NPES GPUs — the next lever
# both reviewers ranked after the hybrid (job 27593181: coloured 253.1 ->
# hybrid 227.6/228.3 ms at 64 GPUs, -10%; CG trip count 68-72 per solve).
# Pre-registered (CLAUDE.md compute discipline):
#   numbers  : per_step ms for hybrid+cheb0 vs hybrid+cheb3, x2 interleaved;
#              CG trip counts under both (FESOM_CG_ITERS=1 short arms).
#   CONFIRMS : cheb3 <= 0.85 x cheb0 AND trip count drops >= 2x (the lever
#              cuts iterations, not just per-iteration cost).
#   REFUTES  : cheb3 >= 0.97 x cheb0 -> at forca20 the extra SpMV work eats
#              the iteration cut; next lever is the psum count.
#   why not cheaper: the trade (3 extra SpMV+halo per iteration vs 3x fewer
#              iterations) is regime-dependent; measured -20.7% at CORE2-8
#              and -9.6% at NG5-64 with coloured, never on a hybrid baseline.
#
#   NPES=64  sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 16 fesom_cgpoly_ab.sbatch
#   NPES=128 sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 fesom_cgpoly_ab.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/fesom_cgpoly_ab_j${SLURM_JOB_ID}}"; mkdir -p "$OUTDIR"
cd "$FJ" || exit 1
export PYTHONPATH="$FJ:${PYTHONPATH:-}"
export JDIST=1 GPUS_PER_NODE=4
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
  FESOM_CG_CHEB=0 run_arm "cheb0_r$REP" --halo coloured --ssh-halo padded
  FESOM_CG_CHEB=3 run_arm "cheb3_r$REP" --halo coloured --ssh-halo padded
done
for C in 0 3; do
  FESOM_CG_CHEB=$C FESOM_CG_ITERS=1 run_arm "cgiters_cheb$C" --halo coloured --ssh-halo padded --steps 25
  echo "trip counts cheb=$C:"; grep -h "cg-iters" "$OUTDIR/forca20_d${NPES}_cgiters_cheb$C.log" | awk '{print $2}' | sort | uniq -c | sort -rn | head -4
done
echo "=== DONE $(date) ==="
