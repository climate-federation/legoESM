#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3mutcensus
#SBATCH --time=11:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_mutation_census_%j.log
# Mutation-sensitivity census for the FV3 duo-grid JAX lane.
#
# WHAT THIS ANSWERS
#   docs/atmosphere/reviews/glm_review_strategy_2026-08-14.md finding 4,
#   BLOCKER: "zero demonstrated detection power ... not one of ~500 unit gates
#   went red on a code defect".  This job injects semantic mutations that each
#   violate a named, grep-verified line of the PINNED oracle and measures
#   whether the suite goes red, and which gate goes red first.
#
#   Non-zero exit is the POINT.  rc=1 means at least one real defect is
#   invisible to every gate; rc=2 means the instrument itself is broken (a null
#   control went red, or a mutation failed to apply/shadow) and no verdict in
#   the run should be believed.
#
# TRAPS (both earned on this cluster, both already cost a run)
#   * SLURM sets TMPDIR=/local and it is reaped mid-job -- pytest --basetemp is
#     forced onto a burg path by --work below.
#   * The census NEVER edits the worktree: mutants are staged into --work as a
#     symlink farm and shadow the real modules via PYTHONPATH.  A job killed
#     mid-run therefore cannot leave mutated code on disk for the next job to
#     certify.  Do not "improve" this into an edit-and-restore.
set -uo pipefail

# REPO is derived from the script's SUBMITTED path, never edited per run.
#
# Two earlier attempts failed and both are worth recording:
#   * editing REPO= inside a pinned worktree makes it dirty, so the preflight
#     below (correctly) refuses -- that killed job 9408879;
#   * ${BASH_SOURCE[0]} does NOT work under sbatch: SLURM copies the batch
#     script into its spool dir, so it resolved to /cm/local/apps/slurm and the
#     job ran with REPO pointing at the scheduler -- that killed job 9409086.
# scontrol reports the path actually submitted, which is the only one that
# refers to the worktree.  Falls back to BASH_SOURCE for a bare `bash` run.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
if [ ! -e "$REPO/.git" ]; then
  echo "PREFLIGHT FAILED: REPO=$REPO is not a git worktree (resolved from"
  echo "$_SELF). Refusing rather than running against the wrong tree."
  exit 5
fi
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUNDIR=/burg-archive/glab/users/pg2328/fv3_duo_gaps
WORK=$RUNDIR/mutcensus_$SLURM_JOB_ID
OUT=$RUNDIR/fv3_mutation_census_$SLURM_JOB_ID.md
mkdir -p "$WORK"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "host: $(hostname)"
echo "work: $WORK"
cd "$REPO"

# ---------------------------------------------------------------- PREFLIGHT
# Copied verbatim from scripts/cluster/fv3_native/jax_lane_measure.sbatch
# (earned 2026-08-14, twice in five minutes).  `cp` is aliased to prompt on
# this machine, so BOTH a plain `cp` and a `cp -f` into a pinned worktree
# silently did nothing -- and a job was submitted against a stale file, which
# would have been reported as a result.  The fix is structural: COMMIT FIRST,
# create the worktree at that SHA, and let git place the files.
#
# For THIS job the preflight is load-bearing twice over: the census certifies
# the GATES, so a log that cannot be attributed to a commit certifies nothing.
DIRTY=$(git status --porcelain | wc -l)
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty files: $DIRTY"
if [ "$DIRTY" -ne 0 ]; then
  echo "PREFLIGHT FAILED: worktree is DIRTY. A pinned run must be reproducible"
  echo "from its SHA alone; an uncommitted file means the log cannot be"
  echo "attributed to any commit. Commit, re-create the worktree, resubmit."
  git status --porcelain | head -20
  exit 3
fi

ORACLE=/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean
echo "oracle tree: $ORACLE"
if [ ! -f "$ORACLE/model/dyn_core.F90" ]; then
  echo "PREFLIGHT FAILED: the pinned oracle tree is missing. Every catalogue"
  echo "citation is against THAT tree (three copies exist and their line"
  echo "numbers differ -- STATE lesson 3)."
  exit 3
fi
# Spot-check two citations the catalogue leans on hardest, so a moved oracle
# cannot silently invalidate the whole matrix.
echo "--- oracle citation spot-check ---"
sed -n '856p'  "$ORACLE/model/dyn_core.F90" | grep -q 'iq==1 .or. iq==4' \
  && echo "  OK   dyn_core.F90:856 iq guard" \
  || { echo "  FAIL dyn_core.F90:856 moved"; exit 3; }
sed -n '1336p' "$ORACLE/model/dyn_core.F90" | grep -q 'ext_scalar(delp' \
  && echo "  OK   dyn_core.F90:1336 post-step delp refresh" \
  || { echo "  FAIL dyn_core.F90:1336 moved"; exit 3; }

CENSUS=$REPO/scripts/validate/fv3_native/mutation_census.py

echo "=== 0. ruff ==="
$PY -m ruff check "$CENSUS" "$REPO/tests/grids/test_fv3_mutation_census.py"
echo "    ruff rc=$?"

# The census DECIDES what we believe about the gates, so it is exactly the code
# where a bug manufactures a confident wrong verdict.  Its own gates run first
# and abort the job -- no census number is quoted from an unvalidated
# instrument (CLAUDE.md: validate the instrument before quoting its number).
echo "=== 0b. the census's own gates ==="
$PY -m pytest "$REPO/tests/grids/test_fv3_mutation_census.py" -q \
     --basetemp="$WORK/pytest_basetemp" --tb=short
SELF=$?
echo "    self-test rc=$SELF"
if [ "$SELF" -ne 0 ]; then
  echo "ABORT: the census's own tests are red. Its verdicts are untrusted."
  exit 4
fi

# ---------------------------------------------------------------- DRY RUN
# Seconds, no JAX.  Stages every mutant and asserts each anchor occurs EXACTLY
# once in the module it names.  A stale catalogue is caught here instead of
# after ten hours of pytest, and an anchor that matches zero or two sites is a
# hard stop: a mutation that cannot be applied is worse than none.
echo "=== 1. dry run: verify every anchor still applies ==="
$PY "$CENSUS" --repo "$REPO" --work "$WORK" \
     --out "$RUNDIR/fv3_mutation_census_dryrun_$SLURM_JOB_ID.md" \
     --python "$PY" --dry-run
DRY=$?
echo "    dry-run rc=$DRY"
if [ "$DRY" -ne 0 ]; then
  echo "ABORT: the catalogue does not apply to this SHA. Fix the anchors."
  exit "$DRY"
fi

# ---------------------------------------------------------------- CENSUS
# Cost control: per mutation the SMALLEST plausible selection runs first and
# the ladder only escalates on green (fast -> the module's own gate file ->
# every lane gate file).  Baselines are cached per selection, so the shared
# ones are paid once.
echo "=== 2. census ==="
$PY "$CENSUS" --repo "$REPO" --work "$WORK" --out "$OUT" --python "$PY" \
     --timeout 5400
RC=$?
echo "    census rc=$RC"

echo "=== 3. report ==="
cat "$OUT" 2>/dev/null || echo "  (no report written)"

echo "=== 4. cleanup ==="
# junit xml and the report are kept; the mutant symlink farms are not.
rm -rf "$WORK/mutants" "$WORK/pytest_basetemp"

case "$RC" in
  0) echo "VERDICT: every expected-observable mutation was caught." ;;
  1) echo "VERDICT: at least one semantic defect is INVISIBLE to every gate." ;;
  2) echo "VERDICT: this run CANNOT ANSWER -- either the instrument is broken"
     echo "         (null control red / staging / shadow) or the escalation"
     echo "         budget ran out.  Read the report before quoting anything." ;;
  4) echo "VERDICT: the census's OWN tests are red; it was not run." ;;
  *) echo "VERDICT: unexpected exit $RC." ;;
esac
echo "FV3_MUTATION_CENSUS_DONE rc=$RC"
exit "$RC"
