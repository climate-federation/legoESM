#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_cdx3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_cdx3_%j.out
set +e
# ---------------------------------------------------------------------------
# Adversarial review of the CODE that implements the T/q tuning objective.
# The strategy round already ran (job 9417292) and its findings are in the
# diff; this round reviews the implementation of those findings.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
BASE=${BASE:-f05638ffd5e2d08b9b43d36690b5925eea2e3ad6}
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
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
echo "BASE=$BASE"

codex exec --sandbox read-only -C $WT "$(cat <<PROMPT
ADVERSARIAL REVIEW, ROUND THREE. The diff is \`git diff $BASE..HEAD\`. Round two
of this review already ran against \$BASE and its findings are IMPLEMENTED in
this diff; your job is to find what the implementation of those fixes got
wrong, plus anything the earlier rounds missed.

WHAT CHANGED SINCE \$BASE, and why:
* The humidity variable the objective minimises is now selectable and defaults
  to \`logq\` (fractional humidity error, no saturation curve) instead of
  relative humidity, because RH re-weights temperature through q_sat(T) by a
  measured 2-4x and makes a compensating warm-and-moist bias the exact optimum.
  Both terms are always reported.
* weighted_rmse now zeroes the residual where the weight is zero (0 * NaN).
* The tropospheric mask is derived from the REFERENCE cold point plus a buffer
  rather than a fixed 100 hPa bound.
* The thermo score's own noise floor is reported from the spread across the
  last four non-overlapping analysis windows of the same run.
* tune_category_winner returns a fourth value, a statistics dict, because
  tune_evals is a budget of PROPOSALS and was being reported as if it were
  columns integrated.
* A shipped default outside its declared bounds is now a hard error.
* The markdown summary and the CSV rank by, and name, the objective and
  parameter set from each row's own signature.
* A seed-agreement reporter compares two independent seeds: rank churn, each
  score difference against that run's own noise floor, and whether the seeds
  landed on DISTANT parameter values with similar scores.
* The probe now calls the campaign's own pressure and RH helpers; its RH row
  moved from 27.9 % to 38.9 % as a result.

FIND, with file:line:

A. \`logq\`: is \`ln(max(q, floor))\` scored correctly against a reference that
   is ALSO floored? What happens when the model column is dry and the reference
   is not, or vice versa, and is the resulting penalty bounded in a way that
   lets a candidate win by drying out? Check the floor value against the
   reference's actual minimum.
B. The cold-point mask: \`tropospheric_min_pressure\` searches p < 30000 Pa and
   takes an argmin. Can it pick a level that is not the tropopause (an inversion,
   a noisy profile, a monotonically decreasing T to the model top)? It ran and
   produced 16191 Pa from a cold point at 14191 Pa on the SAM reference -- is
   that the real cold point, and is a mask that keeps 35 of 74 levels scoring
   the layer the campaign claims to score?
C. The window-spread noise floor: is the sample std over 2-4 windows a
   defensible estimate, are the windows genuinely non-overlapping, and does the
   loop handle a run shorter than four windows correctly? Does anything DIVIDE
   by it?
D. The tune_stats accounting: \`unique_evals\` is read as len(cache) AFTER each
   evaluation. Is that the count it claims to be, given the default run is
   inserted before the loop and the driver may share a cache across schemes?
E. The seed-agreement reporter: can its pair-swap count, its rank shift, or its
   noise comparison be wrong for a row with a non-finite score, a scheme present
   in one seed only, or a parameter whose bounds differ between seeds?
F. Anything in the diff that produces a WRONG NUMBER rather than an error, and
   any NEW test that cannot fail.

Numbered list, file:line, one-line fix each. Say explicitly where a risk is not
a defect. Do not pad.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN - do not treat this job as a review"
echo ALLDONE
