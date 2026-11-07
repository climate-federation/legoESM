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
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
DC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc

# RUN_REF saves no monthly U/V, but RUN_GATEWAY -- the 90-day COLD-START
# byte-clone of the RUN_REF protocol (same IC, same forcing, same dt) --
# saved 5-DAY U/V/T for months 1-3.  That IS the matched-window circulation
# reference for our d30/d90 cold-start arms: record 5 = days 26-30, record
# 17 = days 86-90.  Reduction caveat: our card numbers are INSTANTANEOUS
# day-30/90 states, NEMO's are 5-day means -- state it next to each number.
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
NFAIL=0
for idx in 5 17; do
  echo; echo "########## NEMO RUN_GATEWAY 5d record $idx (days $((idx*5+1))-$((idx*5+5)) of cold start)"
  $PY -u scripts/validate/nemo_transports.py \
    --grid-v $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
    --grid-u $GW/ORCA1_5d_20000101_20000331_grid_U.nc \
    --grid-t $GW/ORCA1_5d_20000101_20000331_grid_T.nc \
    --domain-cfg $DC --time-idx $idx
  rc=$?; [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
done
echo; echo "########## NEMO 5-year annual mean (spun-up context)"
$PY -u scripts/validate/nemo_transports.py \
  --grid-v $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
  --grid-u $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
  --grid-t $RUN/ORCA1_1y_20000101_20041231_grid_T.nc \
  --domain-cfg $DC
rc=$?; [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
