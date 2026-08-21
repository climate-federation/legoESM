#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3triage
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_gate_triage_%j.log
# Runs scripts/validate/fv3_gradient_gate_triage.py -- the five discriminators for
# the five standing gate failures. Every block pre-registers what confirms and
# what refutes it, so the log is readable without the source.
set -uo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
  echo "PREFLIGHT FAILED: $REPO is dirty ($DIRTY files); a measurement that"
  echo "cannot be attributed to a commit is not a measurement."
  exit 3
fi
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd "$REPO"
$PY scripts/validate/fv3_gradient_gate_triage.py
echo "rc=$?"
