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
    WATCH_DIR="$OUT"
    WB_MODES="${VARIANT//+/,}"
    # The WB trainer restores parameters, optimizer state and the frozen-leaf
    # set from its last completed epoch, so this campaign chains like the aimip
    # one. It used to be pinned to a single link and to never pass --resume,
    # which is why a 12-epoch T63 run could not finish inside a 12-hour
    # walltime: every link began again at epoch 0.
    # Each mode has its own $OUT/<mode> directory and finds its OWN latest
    # epoch, so a partially-trained fleet resumes per mode; a mode with no
    # checkpoint simply starts at zero.
    # Decided below, once _ckpt_sig exists: asking whether a PARAMETER file is
    # present would advertise a resumed run where the trainer, which needs the
    # optimizer state and a valid manifest too, will quietly start at epoch 0.
    RESUME_FROM_SIG=1
    RESUME_FLAG=""
    DRIVER=(scripts/run/run_weatherbench_campaign.py --config "$SUITE" \
            --modes "$WB_MODES" --stages train --out-root "$OUT")
    FINAL=""
    ;;
  *)
    echo "[unified-train] Unknown CAMPAIGN '$CAMPAIGN' (aimip|wb)" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"

# Nanosecond mtime + size: two same-second checkpoint replacements must not
# read as "no progress" (codex MED).
# The wb campaign trains several families in ONE invocation and gives each its
# own $OUT/<mode> directory, so the signature has to look one level down as
# well; globbing only the top level saw no progress and refused to chain.
# It watches the MANIFEST, which the trainer writes LAST: the parameter file
# lands first, so a link killed while serialising the optimizer state would
# otherwise report progress, chain, and hand the next link a checkpoint it
# refuses to resume from — an epoch-0 restart wearing a chained run's name.
# aimip has no manifests, so its own flat globs are unchanged.
_ckpt_sig() {
  # wb ASKS THE TRAINER what a complete epoch is rather than re-deciding here.
  # A shell test of its own (a grep for a substring, a glob for the parameter
  # file) drifts from the trainer's: it counted a truncated manifest as
  # progress, so the wrapper chained and the next link refused, and the pair
  # looped while reporting a chained run.
  if [ "$CAMPAIGN" = "wb" ]; then
    "$PY" "$REPO/scripts/run/train_weatherbench_scale.py" \
      --print-latest-complete "$WATCH_DIR" 2>/dev/null
    return
  fi
  ls -t "$WATCH_DIR"/epoch_*.eqx "$WATCH_DIR"/chunk_latest.eqx 2>/dev/null \
    | head -1 | xargs -r stat -c '%n:%.Y:%s' 2>/dev/null
}
SIG_BEFORE="$(_ckpt_sig)"
# One definition of "there is something to resume from", shared with the
# trainer: a non-empty signature means a COMPLETE epoch exists.
if [ "${RESUME_FROM_SIG:-0}" = "1" ]; then
  if [ -n "$SIG_BEFORE" ] || [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
  fi
fi

echo "[unified-train] $(date) campaign=$CAMPAIGN variant=$VARIANT link=$CHAIN suite=$SUITE out=$OUT resume=${RESUME_FLAG:-no}"

# shellcheck disable=SC2086
timeout -k 120 "${LINK_BUDGET_S}s" "$PY" -u "${DRIVER[@]}" $RESUME_FLAG
RC=$?

TIMED_OUT=0
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  SIG_AFTER="$(_ckpt_sig)"
  # An EMPTY signature is never progress. Without this, a query that failed
  # for any transient reason would read as a change from a previous non-empty
  # signature and chain a link with nothing to resume from.
  if [ -n "$SIG_AFTER" ] && [ "$SIG_AFTER" != "$SIG_BEFORE" ]; then
    TIMED_OUT=1
    echo "[unified-train] hit link budget ${LINK_BUDGET_S}s (rc=$RC), checkpoint advanced -> chaining"
    RC=0
  else
    echo "[unified-train] hit link budget (rc=$RC) with no resumable progress -> NOT chaining (epoch-0 restart-loop guard)."
    RC=1
  fi
fi
echo "[unified-train] driver rc=$RC $(date)"

# A campaign with a single FINAL artifact (aimip) chains until that file
# appears.  wb has none — its train stage simply returns — so it chains only
# when the link was CUT SHORT by the walltime with checkpoint progress to show
# for it.  Chaining a wb link that exited normally would retrain a finished
# campaign CHAIN_MAX more times.
if [ "$RC" -eq 0 ] && [ "$CHAIN" -lt "$CHAIN_MAX" ] \
   && { { [ -n "$FINAL" ] && [ ! -f "$FINAL" ]; } \
        || { [ -z "$FINAL" ] && [ "$TIMED_OUT" -eq 1 ]; }; }; then
  echo "[unified-train] not complete -> chaining link $((CHAIN+1))/$CHAIN_MAX"
  resubmit_self
elif [ -n "$FINAL" ] && [ -f "$FINAL" ]; then
  echo "[unified-train] DONE: $FINAL"
else
  echo "[unified-train] stop (rc=$RC, campaign=$CAMPAIGN, chain limit $CHAIN_MAX)"
fi
