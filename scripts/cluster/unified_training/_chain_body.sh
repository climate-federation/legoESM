#!/usr/bin/env bash
# Shared chain body for unified WB+AIMIP training launchers (design D7).
#
# Factored from scripts/cluster/aimip_scale/train_aimip_derecho.pbs (the #942 /
# #985-hardened resume+chain logic) so Ginsburg/Derecho/Levante wrappers stay
# thin. The site wrapper must define, BEFORE sourcing this file:
#
#   PY             python interpreter
#   REPO           repo root (cwd is already $REPO)
#   CAMPAIGN       aimip | wb        (selects the driver)
#   SUITE          suite/campaign YAML path (repo-relative)
#   OUT            checkpoint root for this run (per-variant/mode subdirs below)
#   VARIANT        aimip variant (aimip) or comma modes (wb)
#   LINK_BUDGET_S  seconds to let the driver run this link (walltime - drain)
#   CHAIN, CHAIN_MAX  current link index / max links
#   resubmit_self()   function that resubmits this wrapper with
#                     CHAIN=$((CHAIN+1)) RESUME=1 (site scheduler specific)
#
# Chaining contract: a link chains only if it exited cleanly and the run is
# incomplete, or hit the link budget WITH a new resumable checkpoint
# (epoch_*.eqx / chunk_latest.eqx). A budget-bound link with no new
# checkpoint stops (an epoch longer than the budget with no mid-epoch
# checkpointing would otherwise restart epoch 0 forever).
set -uo pipefail

RESUME_FLAG=""
if ls "$OUT"/*/epoch_*.eqx >/dev/null 2>&1 \
   || ls "$OUT"/*/chunk_latest.eqx >/dev/null 2>&1 \
   || [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
fi

case "$CAMPAIGN" in
  aimip)
    DRIVER=(scripts/run/run_aimip.py --suite "$SUITE" --variants "$VARIANT")
    # run_aimip writes <out>/<variant>/params.eqx after the final epoch.
    FINAL="$OUT/$VARIANT/params.eqx"
    ;;
  wb)
    # WB campaign: chain the TRAIN stage only; eval/plot are a separate
    # single-process job (eval_*.sbatch) once training completes.
    DRIVER=(scripts/run/run_weatherbench_campaign.py --config "$SUITE" \
            --modes "$VARIANT" --stages train --out-root "$OUT")
    # The WB trainer has no single DONE marker; chain until CHAIN_MAX or a
    # clean exit with no budget kill (a completed train stage exits 0 fast
    # on resubmit because all epochs exist).
    FINAL=""
    ;;
  *)
    echo "[unified-train] Unknown CAMPAIGN '$CAMPAIGN' (aimip|wb)" >&2
    exit 2
    ;;
esac

echo "[unified-train] $(date) campaign=$CAMPAIGN variant=$VARIANT link=$CHAIN suite=$SUITE resume=${RESUME_FLAG:-no}"

_ckpt_sig() {
  ls -t "$OUT"/*/epoch_*.eqx "$OUT"/*/chunk_latest.eqx 2>/dev/null \
    | head -1 | xargs -r stat -c '%n:%Y' 2>/dev/null
}
SIG_BEFORE="$(_ckpt_sig)"

# shellcheck disable=SC2086
timeout -k 120 "${LINK_BUDGET_S}s" "$PY" -u "${DRIVER[@]}" $RESUME_FLAG
RC=$?

if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  if [ "$(_ckpt_sig)" != "$SIG_BEFORE" ]; then
    echo "[unified-train] hit link budget ${LINK_BUDGET_S}s (rc=$RC), checkpoint advanced -> chaining"
    RC=0
  else
    echo "[unified-train] hit link budget (rc=$RC) but NO new resumable checkpoint -> NOT chaining (epoch-0 restart-loop guard)."
    RC=1
  fi
fi
echo "[unified-train] driver rc=$RC $(date)"

if [ "$RC" -eq 0 ] && { [ -z "$FINAL" ] || [ ! -f "$FINAL" ]; } && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  if [ -n "$FINAL" ] || [ "$(_ckpt_sig)" != "$SIG_BEFORE" ] || [ "$CHAIN" -eq 0 ]; then
    echo "[unified-train] not complete -> chaining link $((CHAIN+1))/$CHAIN_MAX"
    resubmit_self
  else
    echo "[unified-train] clean exit with no checkpoint movement -> treating as complete (no chain)"
  fi
elif [ -n "$FINAL" ] && [ -f "$FINAL" ]; then
  echo "[unified-train] DONE: $FINAL"
else
  echo "[unified-train] stop (rc=$RC or chain limit $CHAIN_MAX)"
fi
