#!/bin/bash
#SBATCH --job-name=fesomnyf
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomnyf_%j.out
#
# FESOM2 (fesom-jax) 30-day arm on CORE-II NORMAL-YEAR forcing — the NEMO-
# matched protocol (fesom c74311d + legoESM 6b93b0b04).  ONE variable vs the
# fesom_core2_d30 arm 9369143: the atmospheric forcing (JRA55-do 1958 ->
# the same nyf.zarr the NEMO reference and the tripole/MPAS arms use).
# PHC IC and fesom-side SSS/runoff/chl stay — the residual differences the
# manifest states.
#
# PRE-REGISTERED, day-30 vs NEMO m1 (protocol confound test):
#   FESOM2_vs_nemo SST rmse [was 1.49 vs tripole 0.85]:
#     CONFIRMED protocol-dominated : <= 1.1
#     PARTIAL                      : 1.1 - 1.35
#     REFUTED (model, not forcing) : >= 1.35
#   Watch also the hemispheric dipole (SH_midlat +1.47 / NH_midlat -1.94):
#   a JRA-year seasonal-phase artifact should COLLAPSE under NYF.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
D=$REPO/data/fesom2_core2
NYF=/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf/nyf.zarr
cd $REPO || exit 1
PIN=${1:?usage: sbatch _fesom2_nyf_d30.sbatch <legoESM-sha> <fesom-sha>}
FPIN=${2:?usage: sbatch _fesom2_nyf_d30.sbatch <legoESM-sha> <fesom-sha>}
[ "$(git -C $REPO rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: legoESM HEAD != $PIN"; exit 1; }
[ "$(git -C $FJX rev-parse HEAD)" = "$FPIN" ] || { echo "FATAL: fesom HEAD != $FPIN"; exit 1; }
echo "=== HEAD $(git -C $REPO log --oneline -1)"
echo "=== fesom_jax $(git -C $FJX log --oneline -1)"
[ -d "$NYF" ] || { echo "FATAL: no nyf.zarr at $NYF"; exit 1; }

MI=$D/mesh_ic/core2_mesh_ic
export FESOM_SSS_PATH="$MI/PHC2_salx.nc"
export FESOM_RUNOFF_PATH="$MI/CORE2_runoff.nc"
export FESOM_CHL_PATH="$MI/Sweeney_2005.nc"
export FESOM_PHC_PATH="$MI/phc3.0_winter.nc"
for f in "$FESOM_SSS_PATH" "$FESOM_RUNOFF_PATH" "$FESOM_CHL_PATH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

export PYTHONPATH="$FJX:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_ENABLE_X64=0     # fesom-jax published hindcast runs f32; keep ITS convention
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== gate: NYF reader + driver CLI tests ==="
( cd $FJX && JAX_PLATFORMS=cpu PYTHONPATH=$FJX $PY -m pytest -q \
    fesom_jax/tests/test_nyf_forcing.py -p no:cacheprovider 2>&1 | tail -2 )
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate failed rc=$rc"; exit 1; }

$PY -u scripts/run/run_fesom_core2.py \
  --mesh-dir "$MI/mesh_core2" \
  --ic-dir "$MI/ic_core2" \
  --days 30 --dt 1800 --year 1958 \
  --forcing core2_nyf --nyf-zarr "$NYF" \
  --snapshot-every-days 10 \
  --output $REPO/results/omip_nemo/fesom_nyf_d30
rc=$?
echo "=== run rc=$rc ==="
[ $rc -eq 0 ] || exit $rc

echo "=== score: 3-way vs tripole + NEMO m1 (same protocol as xgrid_trp_fesom_d30) ==="
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
SNAP=$REPO/results/omip_nemo/fesom_nyf_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot"; exit 1; }
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole $REPO/results/omip_nemo/xgrid_trp_d30/snapshot_final.npz \
  --mpas "$SNAP" --label-mpas FESOM2_NYF \
  --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
  --out-dir $REPO/results/omip_nemo/xgrid_trp_fesomnyf_d30 2>&1 | tail -40
echo "=== score rc=$?"
