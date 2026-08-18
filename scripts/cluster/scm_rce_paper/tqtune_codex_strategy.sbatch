#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_cdxs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_cdxs_%j.out
set +e
# ---------------------------------------------------------------------------
# Adversarial review of the STRATEGY (not yet of code) for tuning every
# convection scheme's parameters to the CRM temperature and humidity profiles.
# Run BEFORE any compute is spent, because the objective definition decides
# what ~120 GPU-free CPU-hours of tuning will optimise.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"
  break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
echo "CODEX_HOME=$CODEX_HOME"
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91

cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW OF A TUNING STRATEGY, before any compute is spent.

READ FIRST: docs/atmosphere/scm_rce_tq_tuning_strategy.md

Then read the code it proposes to change:
  packages/ml/legoesm/training/scm_rce_metrics.py
  scripts/run/run_scm_rce_campaign.py   (objective_value, _candidate_values,
                                         tune_category_winner, _score_profiles,
                                         build_reference_profiles)
  scripts/run/run_scm_rce_convection_intercomparison.py  (checkpoint signature,
                                         evaluate_scheme, CSV_FIELDS)
  packages/ml/legoesm/training/param_collector.py  (tiers)

GOAL: the campaign must tune each convection scheme's parameters so the
single-column RCE equilibrium TEMPERATURE and WATER VAPOUR profiles match an
external CRM reference, and must then report a defensible before/after T and q
ranking. Your job is to find the ways this plan produces a confident WRONG
answer, not to approve it.

ANSWER THESE, each with file:line where code is involved:

1. THE OBJECTIVE. Is `sqrt((T^2 + qv^2)/2)` with each term normalised by the
   CRM's own mass-weighted standard deviation a defensible target? Compare
   against fixed physical tolerances (1 K, 1 g/kg). Which choice can make a
   scheme win for the wrong reason? Be concrete about the resulting T-vs-q
   weight given the reference is a tropical RCE column.

2. HUMIDITY WEIGHTING — the part I most expect to be wrong. A mass-weighted
   ABSOLUTE q_v RMSE is dominated by the boundary layer because q_v spans ~3
   decades vertically and mass weighting adds another low-level factor. State
   quantitatively (from the reference profile shape) what fraction of the
   metric the free troposphere above ~5 km can contribute. Then say which
   target is right: absolute q_v, log q_v, relative humidity, or a
   pressure-weighted-but-variable-normalised form. Name the failure mode of
   each. If you recommend relative humidity, say exactly which saturation
   formulation must be used so the SCM and the CRM are compared like for like.

3. THE TIER. `tier="aggressive"` adds tunable_tier 3. Read param_collector and
   the __param_spec__ blocks of the convection schemes
   (packages/atmosphere/legoesm/atmosphere/physics/convection/*.py). Is any
   tier-2 or tier-3 convection parameter actually a NUMERICS knob whose tuning
   would make the result a statement about the discretisation rather than the
   physics? Name them. Conversely, is any genuinely physical closure parameter
   sitting at tier 0 and therefore excluded from a campaign that claims to tune
   "all parameters"? Name those too.

4. THE SEARCH. `_candidate_values` is a one-at-a-time sweep at the 25 % and
   75 % points followed by joint uniform (or log-uniform) random draws, best-so-
   far acceptance, budget 12 evals per parameter clamped to [48, 240]. For a
   19-parameter scheme, quantify how badly this under-samples, and say whether
   the resulting cross-scheme ranking is confounded by parameter COUNT rather
   than by scheme quality. If it is, name the cheapest fix that removes the
   confound (equal budget? budget per parameter? a local refinement stage?
   report a search-budget sensitivity?).

5. OVERFITTING. The score is a 5-day average at day 95-100 of a 100-day
   integration, single column, fixed SST, no large-scale forcing. What can a
   240-evaluation search buy that is not physics? Name the specific diagnostic
   that would EXPOSE it (drift, water budget closure, a held-out window, a
   held-out SST). Which of these is cheap enough to be worth running?

6. WIRING RISKS. The checkpoint signature already includes `objective`; the
   plan adds a tier. Trace what else must enter the signature so a stale
   checkpoint cannot be silently reused. Is there any other path by which a
   run scored under one objective could be merged into a table labelled with
   another?

7. FALSIFICATION. The strategy declares three checks (T and q RMSE must fall
   for schemes with live parameters; a 0-parameter scheme must be bit-identical
   before and after; a T/q-vs-condensate trade-off must be visible). Are these
   sufficient to catch a mis-wired objective? Name any check that CANNOT FAIL
   as written, and add the ones that are missing.

Reply as a numbered list of concrete defects and concrete recommendations. If
one of my proposals is right, say so in one line and move on -- do not pad. If
you believe the whole framing is wrong, say that first.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN - do not treat this job as a review"
echo ALLDONE
