#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --job-name=ifs_cumastrn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
# Build the oracle and run its cases.  usage (repo root):
#   sbatch --wait scripts/validate/oracle/ifs_cumastrn/run.sh <build dir> <case...>
set -euo pipefail
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
B=${1:?build dir}; shift
PY=${PY:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}
bash "$REPO/scripts/validate/oracle/ifs_cumastrn/build.sh" "$B"
P="$REPO/src"; for p in core atmosphere coupler ice land ml ocean tools; do P="$P:$REPO/packages/$p"; done
export PYTHONPATH=$P:$REPO JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
cat "$B/MD5SUMS"
for c in "$@"; do
  $PY "$REPO/scripts/validate/oracle/ifs_cumastrn/make_fixture.py" "$B/ifs_cumastrn.x" "$B/work" $c
done
