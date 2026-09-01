#!/bin/bash
# Stage-1 HOLD-OUT evaluation (pre-registered rule 2): the interior optimum
# (kphio=0.040, beta=60) and the default control, on the 7 held-out sites.
# 2 arms x 7 sites = 14 tasks. Run ONCE; the hold-out is never used to refine.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=pmhold
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-13%14
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/ec_site_calib_holdout/logs/%A_%a.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
DRV=/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc
OUT=diagnostics/ec_site_calib_holdout
SITES=(DE-Hai US-Oho US-NR1 DE-Obe US-Var DE-Geb US-SRM)
NS=${#SITES[@]}
SITE=${SITES[$((SLURM_ARRAY_TASK_ID % NS))]}
if [ $((SLURM_ARRAY_TASK_ID / NS)) -eq 0 ]; then TAG=k0.081785_b146; K=0.081785; B=146
else TAG=k0.040_b60; K=0.040; B=60; fi
ODIR=$OUT/$TAG; mkdir -p "$ODIR" "$OUT/params" "$OUT/logs"
PFILE=$OUT/params/$TAG.yaml
printf 'land.p_model.kphio: %s\nland.p_model.beta_cost: %s\n' "$K" "$B" > "$PFILE"
echo "=== $SITE $TAG HEAD=$(git rev-parse --short HEAD) $(date) ==="
$PY scripts/run/run_ec_site.py --driver-nc "$DRV/${SITE}_driver_v2.nc" \
    --mode diagnostic --out "$ODIR" --soil auto \
    --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model \
    --canopy-g1-source p_model --params "$PFILE"
echo "=== $SITE $TAG rc=$? $(date) ==="
