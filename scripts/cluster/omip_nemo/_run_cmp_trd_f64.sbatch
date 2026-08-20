#!/bin/bash
#SBATCH --job-name=cmptrd64
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmptrd64_%j.out
#
# Re-score the tendency-matching comparator after the float64 fix: the f32
# Stage-B solve fabricated ~1e-4 PSU/s trends in EVD columns (alpha ~ 3e5 in
# float32).  Expect the EVD-salt nrmse to collapse from 13-46; whatever
# remains is a REAL operator/semantics residual worth chasing.
# The worst-column dump runs after it with the same f64 inputs.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "comparator (f64) rec 1" \
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $TRD/keep_run3_1h_T.nc \
    --ufile $TRD/keep_run3_1h_U.nc \
    --vfile $TRD/keep_run3_1h_V.nc \
    --rec 1 --output-dir $REPO/results/omip_nemo/tendency_match_f64

run_stage "worst-column dump (f64)" \
  $PY -u scripts/validate/ocean_fidelity/tendency_stageb_worst_column.py \
    --tfile $TRD/keep_run3_1h_T.nc \
    --ufile $TRD/keep_run3_1h_U.nc \
    --vfile $TRD/keep_run3_1h_V.nc \
    --rec 1

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
