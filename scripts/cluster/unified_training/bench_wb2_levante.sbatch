#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --job-name=wb2_bench
#SBATCH --output=logs/wb2/bench_%j.out
# DKRZ Levante twin of bench_wb2_curriculum_compare.sbatch (which is
# Ginsburg/burg-only: /burg-archive paths, partition burst, account glab).
#
# Scores the three headline AIMIP arms onto ONE WB2 scorecard and overlays the
# published SOTA leaderboard (config/wb/sota/wb2_headline_rmse.csv):
#   classical  — sweep-winner physics + MOST surface closure with the trainable
#                LOCAL z0 land field (aimip_surface_bulk_scheme: most)
#   column_nn  — per-column MLP as the whole physics, v2 rollout curriculum
#   sfno_full  — full SFNO emulator (no dycore, no physics) = the ACE2 analogue
#
# NOT a controlled single-variable comparison: each arm carries its own best
# available training recipe (different curricula / loss presets). What IS held
# fixed is the EVAL protocol — same leads, same eval year, same n_inits, same
# WB2 target grid, same metric — so each arm's distance to SOTA is meaningful
# even though the arms' training recipes differ. Arm-vs-arm deltas are NOT
# attributable to the variant alone.
#
#   sbatch scripts/cluster/unified_training/bench_wb2_levante.sbatch
set -uo pipefail

REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${LEGOESM_PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO" || exit 2

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MPLBACKEND=Agg

# Accepts '+'-separated as well as ','-separated, because `sbatch --export=ALL,K=V`
# SPLITS ITS LIST ON COMMAS: `LEADS=6,12,24` arrives as `LEADS=6` plus two stray
# assignments, and the job then silently scores one lead. Same trap the
# submit_levante.sh WB_MODES convention already documents.
LEADS="${LEADS:-24+72+120+240}"
LEADS="${LEADS//+/,}"
EVAL_YEAR="${EVAL_YEAR:-2017}"
N_INITS="${N_INITS:-8}"
# Init spacing. The U-Cast arms were scored at 60 inits x 144 h (see
# score_ucast_arms.sbatch); a 3-arm table that mixes 8x24 h with 60x144 h is a
# protocol confound, not a result, so the stride has to be settable here too.
STRIDE="${STRIDE:-24}"
OUT_ROOT="${OUT_ROOT:-results/bench_wb2_levante}"
mkdir -p "$OUT_ROOT" logs/wb2

# label -> "suite|checkpoint_dir|variant"
declare -A ENTRY=(
  [classical_most_long]="config/aimip/ace2/suite_curriculum_most_long.yaml|results/aimip_ace2curr_most_long/classical|classical"
  [column_nn_v2]="config/aimip/ace2/suite_curriculum_v2.yaml|results/aimip_ace2curr_v2/column_nn|column_nn"
  [sfno_full_ace2]="config/aimip/scale/suite_sfno_full_t63_ace2.yaml|results/aimip_ace2_sfno_full_t63/sfno_full|sfno_full"
  [sfno_full_ace2_allyears]="config/aimip/scale/suite_sfno_full_t63_allyears_ace2.yaml|results/aimip_ace2_sfno_full_t63_allyears/sfno_full|sfno_full"
  # U-Cast campaign winner (Muon + 10:1 Muon/AdamW LR ratio, maxdata tier):
  # z500 24 h 16.35 m, best sfno arm we have. This is the row the classical and
  # column_nn arms have to be compared against, not the older ace2 arms.
  # Eval-only A/B against column_nn_v2: SAME checkpoint, dry-mass anchor on.
  # Scored at the v2 row's protocol (8 inits x 24 h) so the pair is controlled.
  [column_nn_v2_massfix]="config/aimip/ace2/suite_curriculum_v2_massfix.yaml|results/aimip_ace2curr_v2/column_nn|column_nn"
  # column_nn retrained WITH the dry-mass anchor (job 26794955). Training loss
  # at matched epoch/lead: 120 h 85.02 -> 63.38 and 83.25 -> 61.67 (-26%).
  [column_nn_anchored]="config/aimip/ace2/suite_curriculum_v2_anchored.yaml|results/aimip_cnn_anchored/column_nn|column_nn"
  [sfno_full_muonlr]="config/aimip/scale/suite_sfno_full_ucast_muonlr.yaml|results/aimip_ucast_muonlr_sfno_full_t63/sfno_full|sfno_full"
)
# The two sfno_full arms ARE a controlled pair (identical except training-set
# size); the classical / column_nn arms are not controlled against them.
LABELS="${LABELS:-classical_most_long column_nn_v2 sfno_full_ace2 sfno_full_ace2_allyears}"

# One PNG per LABELS set. Submitting one job per arm (the documented way to
# avoid sbatch's comma-splitting) meant every job wrote the SAME filename, so
# the surviving plot was whichever single-arm job finished last (codex).
PLOT_OUT="$OUT_ROOT/wb_scorecard_$(echo $LABELS | tr ' ' '+')_vs_sota.png"

SCORECARD_ARGS=()
for label in $LABELS; do
  IFS='|' read -r suite dir variant <<< "${ENTRY[$label]}"
  # Evaluate-the-EMA doctrine (D3): prefer params_ema.eqx when the run wrote one.
  # And prefer a VALIDATION-SELECTED checkpoint above that: the last epoch is
  # not the best one. Scanning the muonlr arm's epochs on the validation year
  # 2015 gave 20.11/20.92/19.52/17.81/17.41/21.65 m z500 at 24 h for epochs
  # 45/50/53/55/57/59 — epoch 57 wins and the FINAL epoch is 24 % worse, an
  # ordering that reproduces on the held-out 2017 test year (16.35 vs 20.26).
  # Training early-stopping never caught it because the validation LOSS stayed
  # flat while forecast skill degraded. params_selected_ema.json next to the
  # checkpoint records which epoch was chosen and on what.
  ck="$dir/params.eqx"
  [ -f "$dir/params_ema.eqx" ] && ck="$dir/params_ema.eqx"
  [ -f "$dir/params_selected_ema.eqx" ] && ck="$dir/params_selected_ema.eqx"
  if [ ! -f "$ck" ]; then
    echo "[bench] SKIP $label: no checkpoint at $ck"
    continue
  fi
  out="$OUT_ROOT/${label}_scorecard.json"
  echo "[bench] $(date) eval $label (variant=$variant) ckpt=$ck"
  if "$PY" -u scripts/validate/run_aimip_wb2_eval.py \
      --variant "$variant" --suite "$suite" --checkpoint "$ck" \
      --leads "$LEADS" --eval-year "$EVAL_YEAR" --n-inits "$N_INITS" \
      --init-stride-hours "$STRIDE" \
      --out "$out"; then
    SCORECARD_ARGS+=(--scorecard "${label}=${out}")
  else
    echo "[bench] WARN: eval FAILED for $label — skipping in the plot" >&2
  fi
done

if [ "${#SCORECARD_ARGS[@]}" -eq 0 ]; then
  echo "[bench] no scorecards produced" >&2
  exit 1
fi

"$PY" -u scripts/plot/plot_wb_scorecard.py \
  "${SCORECARD_ARGS[@]}" \
  --sota-csv config/wb/sota/wb2_headline_rmse.csv \
  --metric rmse \
  --out "$PLOT_OUT"
echo "[bench] done $(date): $PLOT_OUT"
