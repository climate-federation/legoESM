#!/bin/bash
#SBATCH --job-name=cmptrd2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmptrd2_%j.out
#
# Re-score the vertical-mixing operator on the NEW RUN_TRD2 hourly file, with
# the CORRECTED time pairing (trend r -> state r+1, measured by the pairing
# scan).  Same comparator, new inputs: this is the controlled check that the
# Stage-B2 verdict (T corr 0.98-0.999, S 0.94-0.99) survives the corrected
# anchor -- a right-inverse test is insensitive to pairing by construction,
# so a LARGE change here would mean the reconstruction is doing more work
# than the operator.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
[ -r "$TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc" ] || { echo "FATAL: no TRD2 T file"; exit 1; }

$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $TRD2/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --output-dir $REPO/results/omip_nemo/tendency_match_trd2 --skip-maps
rc=$?
echo "=== rc=$rc ==="
exit $rc
