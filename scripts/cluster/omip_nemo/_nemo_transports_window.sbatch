#!/bin/bash
#SBATCH --job-name=nemotrp
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemotrp_%j.out
#
# MATCHED-WINDOW circulation reference: NEMO's OWN months 1-3 of year 1 (its
# spin-up from climatological rest, the same protocol our d90 arms run) plus
# the 5-year annual mean for context.  Compares against our gwcorr d90 card:
# ACC 144.6 Sv, AMOC26N -0.13 Sv, MHT NH peak 5.61 PW.  The 5-yr-mean AMOC
# (17.7 Sv) is NOT the comparison target for a 90-day cold start; NEMO March
# year 1 is.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
DC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc

# RUN_REF saves NO monthly grid_U/grid_V (measured: only annual velocity
# files + monthly grid_T exist), so a month-3-of-year-1 matched-window
# reference is IMPOSSIBLE from existing output; it needs a 3-month NEMO
# rerun with monthly U/V/T enabled (open work item).  Until then the only
# defensible statement is: our 90-day cold-start AMOC (~0 Sv) vs NEMO's
# 5-YEAR MEAN (17.7 Sv) is a spin-up-window confound, NOT a defect verdict;
# ACC (144.6 vs 159.3 Sv) is less spin-up-sensitive; our MHT NH peak
# (5.6 PW instantaneous, peak at 0N) vs NEMO 2.44 PW stays SUSPICT but
# unjudgeable at matched windows with current data.
NFAIL=0
echo; echo "########## NEMO 5-year annual mean (the only available reference)"
$PY -u scripts/validate/nemo_transports.py \
  --grid-v $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
  --grid-u $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
  --grid-t $RUN/ORCA1_1y_20000101_20041231_grid_T.nc \
  --domain-cfg $DC
rc=$?; [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
