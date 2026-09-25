#!/bin/bash
#SBATCH --job-name=arcbal
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcbal_%j.out
#
# PER-PROCESS TENDENCY COMPARISON, the instrument that suits the defect.
#
# WHY THIS AND NOT ANOTHER SWITCH A/B.  NEMO's Arctic top-20 m terms (job
# 9389043, RMS) are the SAME ORDER as our whole net erosion:
#     NEMO totad 8.8e-7   NEMO zdf 3.4e-7   |   our net erosion 9.7e-7 psu/s
# NEMO mixes HARD and its terms CANCEL, preserving the halocline; ours fail to
# cancel.  A defect in the BALANCE cannot be found by resizing one term, which
# is what all five refuted candidates did (mixing-length choice, advection
# limiter, surface TKE BC, convection, IC).
#
# This comparator evaluates OUR operators on NEMO'S OWN STATE and compares each
# against NEMO's trend for the same hour -- so any difference is the OPERATOR,
# with the state held identical.  No run of ours is needed.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps \
  --output-dir $REPO/results/omip_nemo/arctic_tendency_balance
echo "=== rc=$? ==="
