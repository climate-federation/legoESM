#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_cdxc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_cdxc_%j.out
set +e
# ---------------------------------------------------------------------------
# Adversarial review of the CODE that implements the T/q tuning objective.
# The strategy round already ran (job 9417292) and its findings are in the
# diff; this round reviews the implementation of those findings.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
BASE=${BASE:-6c578bda0f446b4be9949e4ead5afda3b8838a9b}
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
ADVERSARIAL CODE REVIEW. The diff under review is \`git diff $BASE..HEAD\` in
this worktree. Read it in full, and read the surrounding code it changes -- the
diff hides the consumers.

WHAT IT CLAIMS TO DO: make the SCM-RCE convection campaign tune each scheme's
parameters to the CRM's TEMPERATURE and HUMIDITY profiles instead of to its
condensate, tune the scheme's physical parameters rather than a registry tier
that both admits numerics knobs and excludes the CAPE trigger, and report a
before/after T and humidity ranking that is not mislabelled. It implements the
findings of a prior strategy review, which are summarised in
docs/atmosphere/scm_rce_tq_tuning_strategy.md and in the commit messages.

FILES:
  packages/ml/legoesm/training/scm_rce_metrics.py    (score_thermo_jax,
      relative_humidity_profile, tropospheric_mass_weights)
  packages/ml/legoesm/training/param_collector.py    (include_tier0)
  scripts/run/run_scm_rce_campaign.py                (objective_value,
      OBJECTIVE_FIELD, _thermo_diagnostics, reference_pressure_profile,
      resolve_param_selection, _frac_from_value, tune_category_winner)
  scripts/run/run_scm_rce_convection_intercomparison.py (_run_signature,
      _signature_compatible, CSV_FIELDS, _row, write_csv, new flags)
  scripts/plot/plot_scm_rce_tq_ranking.py
  scripts/validate/scm_rce_tq_objective_probe.py
  tests/unit/test_scm_rce_thermo_objective.py
  tests/unit/test_plot_scm_rce_tq_ranking.py
  tests/unit/test_scm_rce_tq_objective_probe.py

FIND, with file:line for each:

A. NUMERICS OF THE OBJECTIVE. (1) score_thermo_jax computes RH for the SCM and
   for the CRM on the SAME pressure profile taken from the reference's sigma
   times a fixed surface pressure. Is that right, or does the SCM's own surface
   pressure drift enough in a 100-day RCE integration to make it wrong? Say how
   to check. (2) Is the liquid/ice blend applied identically on both sides, and
   can the blend make the objective non-monotone in q_v anywhere? (3) The
   tropospheric mask uses a fixed 100 hPa bound. Trace whether the mask, the
   weights and the profiles are all on the same index convention -- the
   reference grid DESCENDS in height, index 0 is the model top. A silent
   ordering flip here would score the stratosphere and mask the boundary layer.
   (4) weighted_rmse over renormalised masked weights: confirm the masked-out
   levels contribute exactly zero and cannot leak through a NaN.

B. THE PARAMETER SET. (1) resolve_param_selection admits tier-0 parameters
   whose spec reference contains the string 'AD-unreachable'. Is a substring
   match on a free-text reference field a safe selector? What breaks it, and
   what should it match on instead? (2) Does it drop anything it should keep,
   or keep anything it should drop? Check every convection scheme's
   __param_spec__ in packages/atmosphere/legoesm/atmosphere/physics/convection/
   config.py. (3) build_trainable_params now has include_tier0. Can any EXISTING
   caller reach it accidentally, and does the tier-0 path still seed, transform
   and bound the parameter exactly as a tier 1-3 one does? Sigmoid transforms
   on a parameter whose default sits outside its declared bounds are the
   specific hazard.

C. THE SEARCH. (1) The refinement loop moves one parameter at a time from the
   INCUMBENT, greedily, updating the incumbent inside the loop. Trace the
   interaction with n_refine: can it spend more or fewer evaluations than
   budgeted, loop forever, or re-evaluate the same configuration many times
   (the cache would hide the cost but not the wasted budget accounting)?
   (2) _frac_from_value claims to invert _interp. Check the log branch at the
   bounds and for a default outside the bounds. (3) The budget split
   n_random = tune_evals - n_refine changes what a given --tune-evals buys.
   Does anything downstream still assume tune_evals == number of random draws?

D. LABELLING AND CHECKPOINTS. (1) _row now reads the objective from the ROW's
   signature. What happens for a checkpoint whose signature lacks it, and is
   'unstamped' handled consistently in the sort key, the CSV and the figure?
   (2) _run_signature raises for a field that is neither an argparse dest nor a
   declared constant. Is every existing caller of _run_signature safe?
   (3) _signature_compatible no longer grandfathers unstamped checkpoints. What
   breaks, and is there a test that would have caught the old behaviour?
   (4) Are prior_score/tuned_score/score_improvement_pct still the combined
   score while prior_objective/tuned_objective are the tuned one? If so, is any
   consumer (plotters, summary writer, docs) going to read the wrong pair?

E. TESTS. Name every test in the three new files that CANNOT FAIL as written,
   and every docstring claim the test does not actually establish. In
   particular: does test_humidity_term_sees_the_free_troposphere compare two
   quantities that are legitimately comparable, and does
   test_csv_sorts_and_labels_by_the_objective_that_was_tuned actually
   distinguish the two orderings?

F. Anything in the diff that would produce a WRONG NUMBER rather than an error.

Numbered list, file:line, one-line fix each. If a risk is not a defect, say so
in one line. Do not pad.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN - do not treat this job as a review"
echo ALLDONE
