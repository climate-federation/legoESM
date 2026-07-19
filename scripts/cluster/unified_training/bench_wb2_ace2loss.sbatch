#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=bench_wb2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/bench_wb2_%j.out
# WB2 headline benchmark of the trained ACE2-loss variants: per-variant
# scorecard.json (RMSE/ACC vs persistence + climatology floors), then the
# combined scorecard PNG against the published WB2 SOTA overlay (D8).
#   sbatch scripts/cluster/unified_training/bench_wb2_ace2loss.sbatch
set -euo pipefail

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SUITE="${SUITE:-config/aimip/ace2/suite.yaml}"
LEADS="${LEADS:-24,72,120,240}"
OUT_ROOT="${OUT_ROOT:-results/bench_wb2_ace2loss}"
mkdir -p "$OUT_ROOT"

declare -A CKPT=(
  [classical]=results/aimip_ace2loss/classical/params.eqx
  [column_nn]=results/aimip_ace2loss_column_nn/column_nn/params.eqx
  [sfno_physics]=results/aimip_ace2loss_sfno_physics/sfno_physics/params.eqx
)

SCORECARD_ARGS=()
for v in classical column_nn sfno_physics; do
  ck="${CKPT[$v]}"
  if [ ! -f "$ck" ]; then
    echo "[bench-wb2] SKIP $v: no checkpoint at $ck"
    continue
  fi
  out="$OUT_ROOT/${v}_scorecard.json"
  echo "[bench-wb2] $(date) eval $v ckpt=$ck"
  "$UT_PYTHON" -u scripts/validate/run_aimip_wb2_eval.py \
    --variant "$v" --suite "$SUITE" --checkpoint "$ck" \
    --leads "$LEADS" --out "$out"
  SCORECARD_ARGS+=(--scorecard "${v}=${out}")
done

if [ "${#SCORECARD_ARGS[@]}" -eq 0 ]; then
  echo "[bench-wb2] no scorecards produced" >&2
  exit 1
fi

"$UT_PYTHON" -u scripts/plot/plot_wb_scorecard.py \
  "${SCORECARD_ARGS[@]}" \
  --sota-csv config/wb/sota/wb2_headline_rmse.csv \
  --metric rmse \
  --out "$OUT_ROOT/wb_scorecard_ace2loss.png"
echo "[bench-wb2] done $(date): $OUT_ROOT/wb_scorecard_ace2loss.png"
