#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=bench_cmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/bench_cmp_%j.out
# Controlled WB2 A/B: baseline ace2 (fixed 2-step loss) vs curriculum
# (rollout-length curriculum + bias penalties). Same eval protocol
# (leads/year/grid/metric); the only differences are the training suite.
# Scores classical + column_nn from BOTH campaigns onto one scorecard so the
# per-field/lead delta is directly visible.
#   sbatch scripts/cluster/unified_training/bench_wb2_curriculum_compare.sbatch
set -uo pipefail

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

LEADS="${LEADS:-24,72,120,240}"
OUT_ROOT="${OUT_ROOT:-results/bench_wb2_curriculum}"
mkdir -p "$OUT_ROOT"

# label -> "suite|result_dir/variant" (suite drives the eval model skeleton).
# Ladder: base (fixed 2-step) -> curr (rollout curriculum, constant surface)
# -> curr_most (curriculum + MOST surface closure w/ trainable global MOST
# coeffs + z0h + local z0). classical only for the MOST arm (surface scheme
# is N/A when an NN replaces physics).
declare -A ENTRY=(
  [classical_base]="config/aimip/ace2/suite.yaml|results/aimip_ace2loss/classical"
  [classical_curr]="config/aimip/ace2/suite_curriculum.yaml|results/aimip_ace2curr/classical"
  [classical_curr_most]="config/aimip/ace2/suite_curriculum_most.yaml|results/aimip_ace2curr_most/classical"
  [classical_most_long]="config/aimip/ace2/suite_curriculum_most_long.yaml|results/aimip_ace2curr_most_long/classical"
  [column_nn_base]="config/aimip/ace2/suite.yaml|results/aimip_ace2loss/column_nn"
  [column_nn_curr]="config/aimip/ace2/suite_curriculum.yaml|results/aimip_ace2curr/column_nn"
  [column_nn_v2]="config/aimip/ace2/suite_curriculum_v2.yaml|results/aimip_ace2curr_v2/column_nn"
)
# variant token for --variant (strip the label suffix).
_variant_of() { case "$1" in classical_*) echo classical;; column_nn_*) echo column_nn;; esac; }

SCORECARD_ARGS=()
for label in classical_base classical_curr classical_curr_most classical_most_long column_nn_base column_nn_curr column_nn_v2; do
  suite="${ENTRY[$label]%%|*}"; dir="${ENTRY[$label]##*|}"
  ck="$dir/params.eqx"; ema="$dir/params_ema.eqx"
  [ -f "$ema" ] && ck="$ema"
  if [ ! -f "$ck" ]; then
    echo "[bench-cmp] SKIP $label: no checkpoint at $ck"
    continue
  fi
  v="$(_variant_of "$label")"
  out="$OUT_ROOT/${label}_scorecard.json"
  echo "[bench-cmp] $(date) eval $label (variant=$v) ckpt=$ck"
  if "$UT_PYTHON" -u scripts/validate/run_aimip_wb2_eval.py \
      --variant "$v" --suite "$suite" --checkpoint "$ck" \
      --leads "$LEADS" --out "$out"; then
    SCORECARD_ARGS+=(--scorecard "${label}=${out}")
  else
    echo "[bench-cmp] WARN: eval FAILED for $label — skipping in the plot" >&2
  fi
done

if [ "${#SCORECARD_ARGS[@]}" -eq 0 ]; then
  echo "[bench-cmp] no scorecards produced" >&2
  exit 1
fi

"$UT_PYTHON" -u scripts/plot/plot_wb_scorecard.py \
  "${SCORECARD_ARGS[@]}" \
  --sota-csv config/wb/sota/wb2_headline_rmse.csv \
  --metric rmse \
  --out "$OUT_ROOT/wb_scorecard_curriculum_vs_base.png"
echo "[bench-cmp] done $(date): $OUT_ROOT/wb_scorecard_curriculum_vs_base.png"
