#!/bin/bash
#SBATCH --job-name=trdclose
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trdclose_%j.out
#
# Validate the instrument BEFORE building the advection/lateral-diffusion
# operator tests on it: do the emitted hourly trends (totad + ldf + zdf)
# account for the step's state change?  Records 1-3 of the new RUN_TRD2
# hourly file.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TF=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc
[ -r "$TF" ] || { echo "FATAL: $TF unreadable"; exit 1; }
NFAIL=0
for r in 1 2 3; do
  echo; echo "########## record $r"
  $PY -u scripts/validate/ocean_fidelity/nemo_trend_closure.py --tfile "$TF" --rec $r
  rc=$?; [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
done
echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
