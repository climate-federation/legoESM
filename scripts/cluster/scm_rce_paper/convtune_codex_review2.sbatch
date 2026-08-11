#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_cdx2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_cdx2_%j.out
set +e
# ---------------------------------------------------------------------------
# Round 2: review the FIXES from round 1 (job 9355983) and the GLM code review,
# not the original diff.  A fix is a change, and three of the round-1 findings
# were fixed by changing a TEST — the class of change most likely to be wrong.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
# CODEX_HOME must be node-local AND must still exist when codex reads it.
# On g059 (job 9361561) the /tmp dir vanished between `cp` and codex start —
# "Error finding codex home: ... that path does not exist" — so the location is
# chosen by TESTING it, and codex's own exit status is checked rather than
# assumed (a bad CODEX_HOME has historically exited 0 having reviewed nothing).
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
echo "CODEX_HOME=$CODEX_HOME (auth staged: $(ls -1 $CODEX_HOME/auth.json 2>/dev/null | wc -l))"
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version || exit 91

cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
FOCUSED ADVERSARIAL REVIEW, ROUND 2. Answer the numbered questions and stop.
Report only defects that would make a PUBLISHED RESULT WRONG or let a broken
implementation pass a gate. For each: file:line, what breaks, smallest fix.

The commits under review are the fixes applied after round 1:
  git log --oneline -8
  git diff HEAD~5..HEAD -- scripts/ tests/ packages/

R1. TESTS CHANGED TO PASS. Round 1 found that
tests/unit/test_rcemip_gsam_oracle_ic.py asserted a gSAM-sounding calibration
through the module defaults, which are now the published RCEMIP protocol. The
fix RETARGETED those assertions onto GSAM_SND_{T_V0,GAMMA,Q_SFC} and added
test_module_defaults_are_the_published_protocol. Changing a test to make it
pass is the most suspect kind of fix there is: check that the retargeted gates
still FAIL if the gSAM calibration is wrong, that the new default-pinning test
would fail if someone swapped the defaults back, and that no gate was
weakened rather than retargeted. Same question for the four theta anchors
re-measured in tests/validation/test_rcemip_plane_smoke.py (job 9356398).

R2. THE PROBE'S NEW CRITERION.
scripts/validate/check_ifs_supersaturation_cap.py replaced a relative-
magnitude threshold for the ICE half with two internal controls: above the
qci gate (q_ice = 1e-7) and above the 235 K ramp cutoff (T = 260 K), the ON
and OFF configurations must be BIT-IDENTICAL. The motivation was measured:
the absolute effect is identical in morrison and thompson (2.389e-10) but is
91 % of thompson's tendency and 0.44 % of morrison's, because morrison's
tendency at that cell is dominated by Cooper nucleation. Questions: can an
implementation that applies the allowance to the WRONG quantity (e.g. the
psychrometric denominator, or the nucleation gate) still pass both controls
plus the below-gate check? Is `!=` on floats the right comparison here, or
could a legitimate implementation differ in the last bit for a reason other
than the flag? Does the warm control at 260 K actually exercise a path where
ice deposition is active at all — if the cell has no ice physics running, the
control is trivially satisfied and proves nothing.

R3. THE SAMPLER CHANGE in run_scm_rce_campaign._candidate_values /
_sample_scale / _interp. Parameters with strictly-positive bounds spanning
>= 2 decades are now sampled geometrically, in both the one-at-a-time sweep
and the joint draws, with one uniform(0,1) draw per parameter so a longer
budget stays a prefix of a shorter one. Check: (a) is the prefix property
actually preserved for a mix of log and linear parameters; (b) does the
DEFAULT candidate still come first and still get evaluated; (c) does the
change alter the meaning of a previously published tuned value from this
sampler, and is that fact recorded anywhere a reader would see it; (d) is
_sample_scale's 2-decade threshold applied to any parameter where a log draw
is physically wrong (e.g. a bounded fraction, an exponent, a temperature)?
Enumerate the convection __param_spec__ entries that now switch to log
sampling and say whether each is appropriate.

R4. THE ACCEPTANCE GUARD. The tuner now maps a non-finite incumbent score to
+inf and requires a finite trial score. Does that change which candidate wins
in any case where the old code was correct? Can a scheme whose default run
FAILS now report a "tuned" result that is silently a different experiment?

R5. THE NEW CSV COLUMNS. _row(res, ref) computes physical-unit RMSE via
camp.physical_profile_rmse and returns NaN when ref is None. Is there a call
path where ref is present but the profiles are stale or from a different
vertical grid than the reference, so the numbers would be silently wrong
rather than NaN? Does write_summary's markdown table and the CSV agree on
what they report?

R6. THE PRODUCTION SBATCH scripts/cluster/scm_rce_paper/convtune_arms.sbatch.
It derives each scheme's tuning budget by running a python snippet and parsing
its last line. What happens if that snippet prints a warning last, or exits
non-zero, or the registry returns 0 parameters? Is the REF guard (refusing a
path under our own CRM results) actually sufficient, and does the array's
shared --outdir have any other write race besides run_meta.json (which is now
written atomically)?

R7. Anything in the last five commits that is simply wrong and that neither
round 1 nor the GLM review would have been looking for.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
