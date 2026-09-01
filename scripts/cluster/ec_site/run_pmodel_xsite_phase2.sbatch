#!/bin/bash
# P-model EC-site scoring, phase 2: the switches that need the multilayer
# soil (P-hydro) or pair with phase-1 arms (coordination profile).
#   prog_base   : prognostic, prescribed capacities, medlyn      (anchor)
#   prog_pm     : prognostic, p_model capacities + g1, medlyn    (P-model bundle;
#                 cap-vs-g1 attribution comes from the phase-1 diagnostic ladder)
#   prog_phydro : prog_pm + transpiration_stress=phydro          (ONE variable vs prog_pm)
#   diag_coord  : diagnostic, p_model capacities, coordination profile
#                 (ONE variable vs phase-1 'cap')
# Prognostic arms use the gap-free drivers (free-running soil) with
# --select-best-year, matching the fix_xsite protocol.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ecpm2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-13%14
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/diagnostics/ec_site_pmodel/logs/p2_%A_%a.out

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
mkdir -p "$OUT"/{prog_base,prog_pm,prog_phydro,diag_coord,logs}

SITE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
NC=$DRV/${SITE}_driver_v2.nc
GF=$DRV/${SITE}_driver_v2_gapfree.nc
echo "=== $SITE task=$SLURM_ARRAY_TASK_ID HEAD=$(git rev-parse --short HEAD) $(date) ==="

PM="--canopy-stomatal-model medlyn --canopy-capacity-scheme p_model --canopy-g1-source p_model"

if [ -f "$GF" ]; then
  echo "[$SITE] prog_base $(date)"
  $PY scripts/run/run_ec_site.py --driver-nc "$GF" --mode prognostic \
      --out "$OUT/prog_base" --soil auto --select-best-year \
      --canopy-stomatal-model medlyn
  echo "[$SITE] prog_base rc=$?"
  echo "[$SITE] prog_pm $(date)"
  $PY scripts/run/run_ec_site.py --driver-nc "$GF" --mode prognostic \
      --out "$OUT/prog_pm" --soil auto --select-best-year $PM
  echo "[$SITE] prog_pm rc=$?"
  echo "[$SITE] prog_phydro $(date)"
  $PY scripts/run/run_ec_site.py --driver-nc "$GF" --mode prognostic \
      --out "$OUT/prog_phydro" --soil auto --select-best-year $PM \
      --transpiration-stress phydro
  echo "[$SITE] prog_phydro rc=$?"
else
  echo "[$SITE] no gap-free driver; skipping prognostic arms"
fi

echo "[$SITE] diag_coord $(date)"
$PY scripts/run/run_ec_site.py --driver-nc "$NC" --mode diagnostic \
    --out "$OUT/diag_coord" --soil auto \
    --canopy-stomatal-model medlyn --canopy-capacity-scheme p_model \
    --canopy-vcmax-profile coordination
echo "[$SITE] diag_coord rc=$?"
echo "=== $SITE done $(date) ==="
