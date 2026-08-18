#!/bin/bash
#SBATCH --job-name=relabel3w
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/relabel3w_%j.out
#
# Re-score the three-way comparisons after the LABEL fix.
#
# THE DEFECT: compare_three_way_nemo.py took --label-tripole/--label-mpas but
# used them ONLY for the plots; the console table and report.json always said
# "tripole_vs_nemo" / "mpas_vs_nemo".  A run with FESOM2 in slot B therefore
# printed FESOM2's scores under "mpas_vs_nemo", and the two FESOM2 sections of
# job 9383284 both looked like tripole-vs-MPAS tables.  Only the recorded input
# paths in report.json disambiguated them.
#
# WHAT TO CHECK IN THIS LOG: every pair key must now carry the real arm names
# (tripole_vs_nemo / FESOM2_vs_nemo / tripole_vs_FESOM2, then MPAS_vs_nemo /
# FESOM2_vs_nemo / MPAS_vs_FESOM2).  The NUMBERS must be unchanged from
# 9383284 -- this is a naming fix, not a scoring change.  Any moved number is a
# REGRESSION, not an improvement.
#
# PRE-DECODED from 9383284 by provenance (labels were wrong, values were not):
#   SST rmse vs NEMO, month 1, day 30:  tripole 0.851   MPAS 0.855   FESOM2 1.49
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
FS=$R/fesom_core2_d30/snapshot_day0030.npz
for f in "$FS" "$GT" "$R/xgrid_trp_d30/snapshot_final.npz" "$R/xgrid_mpas_d30/snapshot_final.npz"; do
  [ -r "$f" ] || { echo "FATAL: unreadable $f"; exit 1; }
done

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "tripole vs FESOM2 vs NEMO (month 1)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/xgrid_trp_d30/snapshot_final.npz \
    --mpas "$FS" --label-mpas FESOM2 \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/xgrid_trp_fesom_d30

run_stage "MPAS vs FESOM2 vs NEMO (month 1)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/xgrid_mpas_d30/snapshot_final.npz --label-tripole MPAS \
    --mpas "$FS" --label-mpas FESOM2 \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/xgrid_mpas_fesom_d30

run_stage "tripole vs MPAS vs NEMO (month 1, both default labels)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/xgrid_trp_d30/snapshot_final.npz \
    --mpas $R/xgrid_mpas_d30/snapshot_final.npz \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/xgrid_trp_mpas_d30

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
