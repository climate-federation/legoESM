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
#   VARIANT        aimip variant (aimip) or '+'-separated WB modes (wb;
#                  '+' because ',' cannot ride sbatch --export / qsub -v)
#   LINK_BUDGET_S  seconds to let the driver run this link (walltime - drain)
#   CHAIN, CHAIN_MAX  current link index / max links
#   resubmit_self()   function that resubmits this wrapper with
#                     CHAIN=$((CHAIN+1)) RESUME=1 (site scheduler specific)
#
# OUT may be left unset: for aimip it is derived FROM THE SUITE YAML
# (run_aimip writes to the suite's output_dir — a launcher-invented OUT
# would watch the wrong tree and never see checkpoints); for wb it defaults
# under results/unified_wb/ and is passed to the driver via --out-root.
#
# Chaining contract (aimip): a link chains only if it exited cleanly and
# params.eqx is absent, or hit the link budget WITH a new resumable
# checkpoint (epoch_*.eqx / chunk_latest.eqx). A budget-bound link with no
# new checkpoint stops (an epoch longer than the budget with no mid-epoch
# checkpointing would otherwise restart epoch 0 forever).
# WB is SINGLE-LINK: the WB trainer has no --resume implementation (a
# restart retrains from scratch), so chaining it would burn walltime
# re-running epoch 0 — the wb path refuses to chain until the trainer
# grows resume support.
set -uo pipefail

case "$CAMPAIGN" in
  aimip)
    # run_aimip writes to the SUITE's output_dir; derive OUT from it so the
    # resume/chain logic watches the same tree the driver writes.
    SUITE_OUT=$("$PY" -c "
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
print(d.get('output_dir') or '')
" "$SUITE")
    if [ -z "$SUITE_OUT" ]; then
      echo "[unified-train] suite $SUITE carries no output_dir — cannot derive OUT" >&2
      exit 2
    fi
    if [ -n "${OUT:-}" ] && [ "$OUT" != "$SUITE_OUT" ]; then
      echo "[unified-train] WARNING: OUT=$OUT ignored; run_aimip writes to the suite's output_dir=$SUITE_OUT"
    fi
    OUT="$SUITE_OUT"
    # Resume detection is keyed on THIS variant's own dir ($OUT/$VARIANT),
    # NOT $OUT/* — variants share one suite output_dir, and a sibling
    # variant's checkpoints must not flag resume for this one (which would
    # make run_aimip try to load an incompatible/stale checkpoint, e.g. a
    # column_nn arch that changed input channels). RESUME=1 still forces it.
    RESUME_FLAG=""
    if ls "$OUT/$VARIANT"/epoch_*.eqx >/dev/null 2>&1 \
       || ls "$OUT/$VARIANT"/chunk_latest.eqx >/dev/null 2>&1 \
       || [ "${RESUME:-0}" = "1" ]; then
      RESUME_FLAG="--resume"
    fi
    DRIVER=(scripts/run/run_aimip.py --suite "$SUITE" --variants "$VARIANT")
    # run_aimip writes <out>/<variant>/params.eqx after the final epoch.
    FINAL="$OUT/$VARIANT/params.eqx"
    WATCH_DIR="$OUT/$VARIANT"   # chain-progress signature, this variant only
    ;;
  wb)
    OUT="${OUT:-results/unified_wb/$(basename "${SUITE%.*}")}"
    RESUME_FLAG=""   # WB trainer has no resume — never pass --resume
    WATCH_DIR="$OUT"
    WB_MODES="${VARIANT//+/,}"
    DRIVER=(scripts/run/run_weatherbench_campaign.py --config "$SUITE" \
            --modes "$WB_MODES" --stages train --out-root "$OUT")
    FINAL=""
    if [ "$CHAIN" -gt 0 ]; then
      echo "[unified-train] wb campaign is single-link (no trainer resume); refusing chained link $CHAIN" >&2
      exit 2
    fi
    CHAIN_MAX=0   # single link: never resubmit (see header)
    ;;
  *)
    echo "[unified-train] Unknown CAMPAIGN '$CAMPAIGN' (aimip|wb)" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
echo "[unified-train] $(date) campaign=$CAMPAIGN variant=$VARIANT link=$CHAIN suite=$SUITE out=$OUT resume=${RESUME_FLAG:-no}"

# Nanosecond mtime + size: two same-second checkpoint replacements must not
# read as "no progress" (codex MED).
_ckpt_sig() {
  ls -t "$WATCH_DIR"/epoch_*.eqx "$WATCH_DIR"/chunk_latest.eqx 2>/dev/null \
    | head -1 | xargs -r stat -c '%n:%.Y:%s' 2>/dev/null
}
SIG_BEFORE="$(_ckpt_sig)"

# shellcheck disable=SC2086
timeout -k 120 "${LINK_BUDGET_S}s" "$PY" -u "${DRIVER[@]}" $RESUME_FLAG
RC=$?

if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  if [ "$CAMPAIGN" = "aimip" ] && [ "$(_ckpt_sig)" != "$SIG_BEFORE" ]; then
    echo "[unified-train] hit link budget ${LINK_BUDGET_S}s (rc=$RC), checkpoint advanced -> chaining"
    RC=0
  else
    echo "[unified-train] hit link budget (rc=$RC) with no resumable progress -> NOT chaining (epoch-0 restart-loop guard)."
    RC=1
  fi
fi
echo "[unified-train] driver rc=$RC $(date)"

if [ "$CAMPAIGN" = "aimip" ] && [ "$RC" -eq 0 ] && [ ! -f "$FINAL" ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  echo "[unified-train] not complete -> chaining link $((CHAIN+1))/$CHAIN_MAX"
  resubmit_self
elif [ -n "$FINAL" ] && [ -f "$FINAL" ]; then
  echo "[unified-train] DONE: $FINAL"
else
  echo "[unified-train] stop (rc=$RC, campaign=$CAMPAIGN, chain limit $CHAIN_MAX)"
fi
