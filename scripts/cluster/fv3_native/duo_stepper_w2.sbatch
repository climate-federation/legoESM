#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=duow2
#SBATCH --time=11:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/duow2_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu
N=${1:-24}
DT=${2:-450}
EXTFLAG=${3:-}
SUFFIX=""
if [ "$EXTFLAG" = "ext" ]; then EXTFLAG="--ext-bundle"; SUFFIX="_ext"; fi
OUT=/burg-archive/glab/users/pg2328/fv3_recon/w2_duo_stepper_c${N}${SUFFIX}.npz
$PY $REPO/scripts/validate/fv3_native/run_duo_stepper_w2.py \
    --n $N --dt $DT --days 5 --out $OUT $EXTFLAG ${4:+--vector-corner $4}
# gate FAIL verdict is informational at this distance from the target,
# but an infrastructure failure must fail the job (no bare || true —
# codex ext r1 P1-3).  The gate prints a verdict line on every clean
# evaluation; a nonzero exit WITHOUT that line = crash.
GATE_LOG=$(mktemp)
GATE_RC=0
$PY $REPO/scripts/validate/fv3_native/w2_duo_oracle_gate.py $OUT --res $N \
    2>&1 | tee "$GATE_LOG" || GATE_RC=$?
if [ "$GATE_RC" -ne 0 ] && ! grep -q "W2_DUO_TARGET_GATE:" "$GATE_LOG"; then
    echo "GATE INFRA FAILURE rc=$GATE_RC"; rm -f "$GATE_LOG"; exit 1
fi
rm -f "$GATE_LOG"
echo DUOW2_DONE
