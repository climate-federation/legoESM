#!/bin/bash
#SBATCH --job-name=nino3prd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3prd_%j.out
#
# RE-RUN of the Prandtl split after the first pass came back UNREADABLE.
#
# Job 9417293 reported a Pr_ours MEDIAN of exactly 10.00 -- the clamp ceiling
# -- in all sixteen region rows at once, calm and convecting, equator and
# pole.  A statistic that returns its own clamp in every region is a statement
# about the abyss (shear ~ 0 makes Ri enormous and Pr saturates legitimately
# there), not about the thermocline the cold-tongue bias lives in.  It also
# used mean(K)/mean(av) while the 2.311 it is meant to explain is
# rms(K)/rms(av) -- ~4x apart on a heavy-tailed K field, so the two numbers
# were never comparable.
#
# FIXED HERE: rows for the full column AND for interfaces above 300 m; the
# matched rms ratio printed next to the mean ratio; the fraction of interfaces
# sitting ON the Pr clamp printed next to the median.  The runtime control was
# also replaced -- the old one asserted the factorisation identity, which is
# true for any four arrays and could not fail; the new one is NEMO's own
# Prandtl sign (avt <= avm off the floors), scoped to calm columns because EVD
# legitimately inverts it, and it is shown to fire in
# tests/validate/test_prandtl_split_report.py.
#
# READING IT, in the nino3<300m /calm row:
#   KM/avm_r ~ 1, KH/avt_r ~ 2.3, at_ceil small -> the closure is right on
#        momentum and the tracer/momentum SPLIT carries the whole excess.
#   KM/avm_r ~ 2.3 as well                      -> the closure over-mixes both
#        and the Prandtl number is not the lever; go back to the amplitude.
#   at_ceil large even above 300 m              -> our Pr is pinned at its
#        clamp in the thermocline, which is itself the defect to fix.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== git SHA: $(git -C $REPO rev-parse HEAD) ==="
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q tests/validate/test_prandtl_split_report.py 2>&1 | tail -5
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps --mxl-choice 3 \
  --output-dir $REPO/results/omip_nemo/nino3_prandtl_depth 2>&1 \
  | grep -aE "Prandtl split|control|floors detected|^  region|^  arctic|^  nh_mid|^  tropics|^  sh_mid|^  antarctic|^  nino|^  eqpac|READ:|FATAL"
echo "=== rc=${PIPESTATUS[0]} ==="
