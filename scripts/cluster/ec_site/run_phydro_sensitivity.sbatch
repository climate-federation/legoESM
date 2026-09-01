#!/bin/bash
# Stage-2 step 1: WHICH P-hydro parameters carry the dry-site signal?
#
# Both design reviewers refused the assumed gamma_cost x gplant_mol slice:
# those two are partly compensatory (one "water-use aggressiveness" axis),
# while root_biomass_gm2 sets the soil/root resistance and minlwp_mpa sets the
# dry cutoff. So MEASURE the one-at-a-time sensitivity before choosing the 2-D
# ladder slice, instead of asserting it.
#
# Sites: US-Whs (semiarid shrub) and US-Ton (oak savanna) — BOTH stage-1
# TRAINING sites, so this cannot contaminate the stage-1 held-out half
# (US-SRM and US-Var are hold-outs and are deliberately excluded until the
# stage-1 verdict is fixed).
#
# 2 sites x (default + 8 one-at-a-time perturbations) = 18 tasks.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=phsens
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-17%18
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/phydro_sens/logs/%A_%a.out

set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
DRV=/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc
OUT=diagnostics/phydro_sens

SITES=(US-Whs US-Ton)
# name:value pairs, one perturbation each (defaults: gamma 1.0, gplant 4e-3,
# minlwp -2.0, root_biomass 500).  Low/high stay inside the spec bounds.
PERTS=("default:none"
       "gamma_lo:land.phydro.gamma_cost=0.3"
       "gamma_hi:land.phydro.gamma_cost=3.0"
       "gplant_lo:land.phydro.gplant_mol=1.0e-3"
       "gplant_hi:land.phydro.gplant_mol=1.2e-2"
       "minlwp_lo:land.phydro.minlwp_mpa=-4.0"
       "minlwp_hi:land.phydro.minlwp_mpa=-1.0"
       "root_lo:land.phydro.root_biomass_gm2=150"
       "root_hi:land.phydro.root_biomass_gm2=1500")

NS=${#SITES[@]}
IDX=$SLURM_ARRAY_TASK_ID
SITE=${SITES[$((IDX % NS))]}
ENTRY=${PERTS[$((IDX / NS))]}
TAG=${ENTRY%%:*}
KV=${ENTRY#*:}

ODIR="$OUT/$TAG"; mkdir -p "$ODIR" "$OUT/params" "$OUT/logs"
PARG=""
if [ "$KV" != "none" ]; then
  PFILE="$OUT/params/${TAG}.yaml"
  echo "${KV%%=*}: ${KV#*=}" > "$PFILE"
  PARG="--params $PFILE"
fi

echo "=== $SITE $TAG ($KV) HEAD=$(git rev-parse --short HEAD) $(date) ==="
$PY scripts/run/run_ec_site.py --driver-nc "$DRV/${SITE}_driver_v2_gapfree.nc" \
    --mode prognostic --out "$ODIR" --soil auto --select-best-year \
    --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model \
    --canopy-g1-source p_model --transpiration-stress phydro $PARG
echo "=== $SITE $TAG rc=$? $(date) ==="
