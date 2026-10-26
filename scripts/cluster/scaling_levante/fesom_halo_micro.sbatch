#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=fesom_halo_micro_%j.log

# Per-exchange cost of each fesom_jax halo transport on forca20 at NPES GPUs — the
# instrument the full-model ladder cannot give (one number per step mixes ~254 tiny
# 2-D CG exchanges with a few dozen 3-D ones).  Pre-registered question: at 64/128
# GPUs, is the coloured 2-D exchange (K ppermute rounds) >= 2x the best single-
# collective 2-D transport?  CONFIRM -> a hybrid (single-collective 2-D for the CG,
# coloured 3-D) is worth building; REFUTE (within 1.3x) -> the CG's exchange count,
# not its transport, is the lever.
#
#   NPES=128 sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 fesom_halo_micro.sbatch
#   NPES=64  sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 16 fesom_halo_micro.sbatch
set -uo pipefail
FJ=/work/bd1083/b309178/diffESM/fesom_jax
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
NPES="${NPES:?set NPES}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
(( NPES == NODES * 4 )) || { echo "ERROR: NPES=$NPES != 4 x $NODES nodes" >&2; exit 1; }
cd "$FJ" || exit 1
export PYTHONPATH="$FJ:${PYTHONPATH:-}"
export JDIST=1 GPUS_PER_NODE=4
echo "=== $(date) forca20 npes=$NPES nodes=$SLURM_JOB_NODELIST fesom_jax=$(git rev-parse --short HEAD) ==="
for KIND in nod elem; do
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
       --cpu-bind=none --kill-on-bad-exit=1 --export=ALL \
       "$PY" scripts/bench/bench_halo_micro.py --mesh-dir "$M" --dist-dir "$D" \
         --npes "$NPES" --kind "$KIND" --halo padded,coloured,ragged --reps 100 --iters 5 < /dev/null
done
echo "=== done $(date) ==="
