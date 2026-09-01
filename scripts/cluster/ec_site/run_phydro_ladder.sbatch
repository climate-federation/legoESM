#!/bin/bash
# Stage-2 dry-site P-hydro ladder. Grid FIXED from the measured sensitivity
# screen (docs/land/pmodel_kphio_beta_calibration_prereg.md): gplant_mol x
# minlwp_mpa (ranked 1-2; gamma_cost ranked last, dropped). 13 arms x 5 dry
# sites = 65 tasks; prognostic, gap-free drivers, select-best-year.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=phlad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-64%22
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/phydro_ladder/logs/%A_%a.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
DRV=/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc
OUT=diagnostics/phydro_ladder
SITES=(US-SRM US-Whs US-Ton US-Var FR-Pue)
GP=(5e-4 1e-3 2e-3 4e-3)
ML=(-1.0 -1.5 -2.0)
NS=${#SITES[@]}
IDX=$SLURM_ARRAY_TASK_ID
SITE=${SITES[$((IDX % NS))]}
A=$((IDX / NS))                       # 0 = anchor, 1..12 = phydro grid
mkdir -p "$OUT/params" "$OUT/logs"
PM="--canopy-stomatal-model medlyn --canopy-capacity-scheme p_model --canopy-g1-source p_model"
if [ $A -eq 0 ]; then
  TAG=anchor_beta_theta; ODIR=$OUT/$TAG; mkdir -p "$ODIR"
  echo "=== $SITE $TAG HEAD=$(git rev-parse --short HEAD) $(date) ==="
  $PY scripts/run/run_ec_site.py --driver-nc "$DRV/${SITE}_driver_v2_gapfree.nc" \
      --mode prognostic --out "$ODIR" --soil auto --select-best-year $PM
else
  G=$((A - 1))
  GPV=${GP[$((G % 4))]}
  MLV=${ML[$((G / 4))]}
  TAG="gp${GPV}_ml${MLV}"; ODIR=$OUT/$TAG; mkdir -p "$ODIR"
  PFILE=$OUT/params/$TAG.yaml
  printf 'land.phydro.gplant_mol: %s\nland.phydro.minlwp_mpa: %s\n' "$GPV" "$MLV" > "$PFILE"
  echo "=== $SITE $TAG HEAD=$(git rev-parse --short HEAD) $(date) ==="
  $PY scripts/run/run_ec_site.py --driver-nc "$DRV/${SITE}_driver_v2_gapfree.nc" \
      --mode prognostic --out "$ODIR" --soil auto --select-best-year $PM \
      --transpiration-stress phydro --params "$PFILE"
fi
echo "=== $SITE rc=$? $(date) ==="
