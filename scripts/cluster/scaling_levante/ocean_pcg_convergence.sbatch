#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=ocean_pcg_conv_%j.log

# Does the implicit barotropic solve still converge at M=15 iterations?
#
# PRE-REGISTERED:
#   number produced : true area-weighted relative residual and the free-surface
#                     difference against a 600-iteration reference, per M, on
#                     REAL captured systems.
#   ADOPT M=15 if   : rel_res at M=15 is at or below the precision floor the
#                     reference itself reaches (f32) / <= 1e-8 (f64), AND the
#                     free-surface difference stays under 1e-4 m, AND the
#                     implied continuity defect stays under 0.1 mm/day.
#   REJECT if       : any of those is exceeded at either resolution.
#   why not cheaper : convergence depends on the operator's conditioning, which
#                     is set by the mesh; it must be measured at the production
#                     resolutions, not on a toy system.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUT="$SCRATCH/legoesm_scaling/ocean_pcg_conv"
mkdir -p "$OUT"

PROBE=scripts/validate/ocean_fidelity/barotropic_pcg_convergence.py
for L in 7 8; do
  for PREC in f32 f64; do
    X64=""
    [[ "$PREC" == f64 ]] && X64="--x64"
    echo "=== L$L $PREC ==="
    "$LEGOESM_PYTHON" "$PROBE" --subdivision "$L" --nlev 40 --dt 300 \
      --steps 3 --iters 5,10,15,20,30,45,60 --ref-iters 600 $X64 \
      --out "$OUT/conv_L${L}_${PREC}.json" 2>&1 | tail -40
  done
done
echo "ALL DONE"
