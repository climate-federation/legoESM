#!/bin/bash
#SBATCH --job-name=arcconv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcconv_%j.out
#
# Localise the Arctic deep-convection columns on NATIVE eORCA1 cells (no
# regridding), for the freshwater A/B pair.  The two arms differ by exactly
# `--freshwater-closure real_freshwater`, so any difference in the column set is
# attributable to that flag.
#
# The probe answers the INSTRUMENT question first: mixed_layer_depth returns the
# bottom depth exactly when a column has no density crossing, so a "deep MLD"
# can be a saturated diagnostic rather than a mixing depth.  Read that split
# before reading any attribution.
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
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

NFAIL=0
for arm in gwcorr fwreal; do
  case $arm in
    gwcorr) LAB="VSF control" ;;
    fwreal) LAB="real freshwater" ;;
  esac
  SNAP=$R/nemolev_trp_${arm}_d90/snapshot_day0090.npz
  echo; echo "########## $arm ($LAB), day 90 vs NEMO month 3"
  if [ ! -r "$SNAP" ]; then echo "SKIP $arm: $SNAP unreadable"; continue; fi
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$SNAP" --label "$LAB" \
    --nemo-gridt "$GT" --nemo-month 3 \
    --out-dir $R/arctic_convection_${arm}_d90
  rc=$?
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
  echo "=== $arm rc=$rc"
done
echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
