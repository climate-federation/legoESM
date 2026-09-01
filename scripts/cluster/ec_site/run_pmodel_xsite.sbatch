#!/bin/bash
# P-model EC-site scoring ladder (research branch PR#1698), DIAGNOSTIC mode:
# prescribed observed soil moisture isolates the canopy switches from the
# soil-hydrology confound.  One array task per site (14-site gap-free panel,
# same manifest as fix_xsite), four one-variable arms:
#   base_bb  : prescribed capacities, table slope, ball_berry  (production default)
#   base_med : prescribed capacities, table slope, medlyn      (isolates stomatal model)
#   cap      : p_model capacities,    table slope, medlyn      (isolates capacities)
#   cap_g1   : p_model capacities, p_model g1,     medlyn      (isolates predicted g1)
# Arm N vs arm N-1 differs in EXACTLY one config field.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ecpm
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-13%14
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/ec_site_pmodel/logs/%A_%a.out

set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
DRV=/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc
MANIFEST=scripts/cluster/ec_site/fix_xsite_sites.txt
OUT=diagnostics/ec_site_pmodel
mkdir -p "$OUT"/{base_bb,base_med,cap,cap_g1,logs}

SITE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
NC=$DRV/${SITE}_driver_v2.nc
echo "=== $SITE task=$SLURM_ARRAY_TASK_ID HEAD=$(git rev-parse --short HEAD) $(date) ==="
[ -f "$NC" ] || { echo "[$SITE] missing driver $NC"; exit 1; }

run_arm () {
  local arm=$1; shift
  echo "[$SITE] arm=$arm $(date)"
  $PY scripts/run/run_ec_site.py --driver-nc "$NC" --mode diagnostic \
      --out "$OUT/$arm" --soil auto "$@"
  echo "[$SITE] arm=$arm rc=$? $(date)"
}

run_arm base_bb
run_arm base_med --canopy-stomatal-model medlyn
run_arm cap      --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model
run_arm cap_g1   --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model \
                 --canopy-g1-source p_model

echo "=== $SITE done $(date) ==="
