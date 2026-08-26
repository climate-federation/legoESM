#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=subcloud_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/subcloud_codex_%j.out
set +e
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
# Codex needs a NODE-LOCAL CODEX_HOME (the shared FS home has failed before,
# and a bad CODEX_HOME exits 0 having reviewed NOTHING).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version || exit 90

cd $WT || exit 91
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 92; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW. Review the HEAD commit of this repo. Run
`git show --stat HEAD` and `git show HEAD` first, and also read the two
commits before it if they touch the same files.

CONTEXT. A single-column radiative-convective-equilibrium (SCM-RCE) campaign
scores convection schemes against a SAM cloud-resolving-model reference
(RCEMIP RCE_small300, fixed SST 300 K). A measurement found that our columns'
lowest model level sits at relative humidity 0.82-0.99 and within ~1 K of the
SST, while the CRM's sits at RH 0.752 and 3.06 K below it. So the air-sea
humidity difference driving bulk evaporation is 26-59 % of the reference's,
and our evaporation is 1.2-1.9 mm/day against 2.73.

The commit adds (a) a SUB-CLOUD objective so the tuner can see that layer, and
(b) a cross-category tuner that jointly moves CLUBB boundary-layer mixing
coefficients, morrison rain re-evaporation coefficients, and the convection
scheme's downdraft evaporation efficiency.

Be maximally skeptical and CONCRETE (cite file:line). Priorities:

1. IS THE SUB-CLOUD SCORE A SOUND OBJECTIVE?
   `subcloud_mass_weights` restricts the reference's mass weights to z <= top
   and RENORMALIZES to sum 1; `score_subcloud_jax` then normalizes each of T
   and q_v by the reference's own standard deviation ACROSS THAT RESTRICTED
   LAYER (`normalized_profile_rmse` computes the std with the same weights).
   Over a shallow, nearly well-mixed layer that std can be very small.
   QUESTION: does this make the objective ill-conditioned or degenerate —
   e.g. can a column with a slightly wrong LAPSE RATE in the layer score
   better than one with the right lapse rate but a 3 K offset, because the
   normalizer collapses? Work out what the normalizer actually is for a
   ~1 km, ~5-level layer of a tropical sounding, and say whether the floor
   (PROFILE_FLOOR = 1e-12, a numerical guard, not a physical one) can ever
   bind. If the objective is ill-posed, propose the specific alternative.

2. DOES THE FOCUSED TUNER APPLY WHAT IT REPORTS?
   `tune_focused_params` groups overrides by scheme_key using
   `c.name.rsplit(".", 1)[0]`, applies them per category, and records the
   APPLIED value (`applied[c.name] = float(field_values[c.field])`), not the
   sampler's requested value. Check: is the rsplit scheme_key extraction
   correct for EVERY qualified name in the curated sets (including any
   array-shaped parameter)? Can a parameter belong to two active categories
   at once and be applied twice or dropped? Is `_set_active_subconfig` being
   called on `trial_cfg` (accumulating) rather than `base_cfg` (last-wins) —
   verify by reading, and say what would break if it were the other way.

3. THE SUB-CLOUD DIAGNOSTIC IS READ AT INDEX -1 OF THE PROFILE.
   Confirm the SCM's profile and the CRM reference's profile are on the same
   grid with the same ordering at that index, from the code and not from the
   comment. If the ordering is only true by convention, name what would make
   it silently false.

4. CONDENSATE SPLIT. `_condensate_parts_from_tracers` factors the column
   condensate into cloud (q_c+q_i) and precipitating (q_r+q_s+q_g, minus q_g
   for p3) halves, while `_qcond_from_tracers` keeps summing the SAME ordered
   list for the total. The claim in the commit message is that the total is
   therefore bit-identical to before the change. VERIFY that claim by reading
   the code, and say whether the two profiles' independent clipping at zero
   can make cloud+precip differ from the total in a way that matters.

5. IS THE HYPOTHESIS EVEN TESTABLE WITH THESE KNOBS? The curated set is in
   run_scm_rce_convection_intercomparison.py (SUBCLOUD_*_INCLUDE). For each
   family, say whether moving it can plausibly DRY and COOL the sub-cloud
   layer in a single column with no large-scale forcing, or whether some of
   them cannot move the target at all (a parameter with no path to the
   lowest level is wasted search budget). Name any obviously missing knob.

6. Anything that would silently produce a WRONG NUMBER rather than an error.

Report findings as a numbered list, each with file:line, severity
(critical/major/minor/not-a-defect), and a concrete fix. If a concern turns
out not to be a defect, say so explicitly rather than padding the list.
PROMPT
)" 2>&1 | tail -220
echo "CODEX_RC=$?"
echo "COMPLETED"
