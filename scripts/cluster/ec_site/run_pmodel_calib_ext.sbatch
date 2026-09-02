#!/bin/bash
# P-model kphio x beta_cost screen on the EC panel's TRAINING half.
# Protocol pre-registered in docs/land/pmodel_kphio_beta_calibration_prereg.md
# (diagnostic mode: the soil is prescribed from observations, so this fits the
# canopy parameters only and cannot absorb the free-running soil-stress error).
#
# EXTENSION grid (pre-registered rule 1 fired: the 5x5 screen optimum sat at
# the (0.040, 60) corner -> UNRESOLVED). Extends both parameters toward their
# spec lower bounds (kphio floor 0.02, beta floor 30); the previous corner
# (0.040, 60) is re-run as the OVERLAP CONTROL and must reproduce its screen
# objective. 3x3 x 7 sites = 63 tasks.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=pmcal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-62%21
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/ec_site_calib/logs/%A_%a.out

set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
DRV=/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc
OUT=diagnostics/ec_site_calib

# Pre-registered training half (7 sites) and screen grid.
SITES=(US-MMS FI-Hyy DE-Gri US-Ne1 US-Ton US-Whs FR-Pue)
KPHIO=(0.020 0.030 0.040)
BETA=(30 45 60)

NS=${#SITES[@]}
IDX=$SLURM_ARRAY_TASK_ID
SITE=${SITES[$((IDX % NS))]}
G=$((IDX / NS))                     # grid point 0..8 (extension)
K=${KPHIO[$((G % 3))]}
B=${BETA[$((G / 3))]}

TAG="k${K}_b${B}"
ODIR="$OUT/$TAG"
PFILE="$OUT/params/${TAG}.yaml"
mkdir -p "$ODIR" "$OUT/params" "$OUT/logs"

# Params file per grid point (idempotent; the shared --params loader validates
# both keys against the registry bounds gate).
cat > "$PFILE" <<EOF
land.p_model.kphio: $K
land.p_model.beta_cost: $B
EOF

echo "=== $SITE $TAG task=$IDX HEAD=$(git rev-parse --short HEAD) $(date) ==="
$PY scripts/run/run_ec_site.py --driver-nc "$DRV/${SITE}_driver_v2.nc" \
    --mode diagnostic --out "$ODIR" --soil auto \
    --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model \
    --canopy-g1-source p_model --params "$PFILE"
echo "=== $SITE $TAG rc=$? $(date) ==="
