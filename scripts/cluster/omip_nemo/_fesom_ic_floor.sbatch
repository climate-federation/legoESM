#!/bin/bash
#SBATCH --job-name=fesomic
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomic_%j.out
#
# IC-FLOOR decomposition of the day-30 legoESM-vs-FESOM2 gap:
#   stage 1: write the FESOM2 IC (PHC3.0 winter on the CORE2 mesh) as a
#            comparator snapshot;
#   stage 2: score IC vs NEMO Jan (single-arm scorecard) -- if this rmse
#            already ~= the day-30 FESOM2-vs-NEMO rmse (SST 1.48), the gap is
#            protocol-dominated, not run divergence;
#   stage 3: three-way FESOM2-IC vs FESOM2-day30 vs NEMO -- how much did 30
#            days of run move the state relative to its own IC?
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$FJX:$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
MI=$REPO/data/fesom2_core2/mesh_ic/core2_mesh_ic
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "write FESOM2 IC snapshot" \
  $PY -u scripts/validate/ocean_fidelity/fesom_ic_snapshot.py \
    --mesh-dir $MI/mesh_core2 --ic-dir $MI/ic_core2 \
    --output $R/fesom_core2_ic

run_stage "IC vs NEMO Jan (single-arm scorecard)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot $R/fesom_core2_ic/snapshot_day0000_ic.npz \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --grid-label "FESOM2 IC (PHC3.0 winter)" \
    --output-dir $R/compare_fesom_ic_m01

run_stage "FESOM2 IC vs FESOM2 day-30 vs NEMO (three-way)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/fesom_core2_ic/snapshot_day0000_ic.npz --label-tripole "FESOM2-IC" \
    --mpas $R/fesom_core2_d30/snapshot_day0030.npz --label-mpas "FESOM2-d30" \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/fesom_ic_vs_d30

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
