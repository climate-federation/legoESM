#!/bin/bash
#SBATCH --job-name=codex_camp
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/codex_campaign_chain_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_wt_campaign
# Token persistence: see _codex_mxl_choice4_r1.sbatch. codex writes a REFRESHED
# token back into CODEX_HOME; a throwaway CODEX_HOME discards it and the next
# run replays a consumed refresh token, which is what produced the days of
# `refresh_token_reused` 401s that were misread as a usage cap.
REAL_CODEX=/burg-archive/home/pg2328/.codex
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp $REAL_CODEX/auth.json "$CODEX_HOME/" 2>/dev/null || true
cp $REAL_CODEX/config.toml "$CODEX_HOME/" 2>/dev/null || true
_persist_token() {
  if [ -s "$CODEX_HOME/auth.json" ] && \
     ! cmp -s "$CODEX_HOME/auth.json" "$REAL_CODEX/auth.json"; then
    cp -f "$CODEX_HOME/auth.json" "$REAL_CODEX/auth.json" \
      && echo "=== token REFRESHED during run -> persisted to $REAL_CODEX ==="
  else
    echo "=== token unchanged (nothing to persist) ==="
  fi
  rm -rf "$CODEX_HOME"
}
trap _persist_token EXIT
echo "=== CODEX_HOME=$CODEX_HOME auth=$(ls -l $CODEX_HOME/auth.json 2>&1) ==="

# A short round-trip probe is a NECESSARY but NOT SUFFICIENT check, and saying
# otherwise cost a retracted claim on 2026-08-06: jobs 9333103 and 9333105 both
# printed "preflight OK (real API call succeeded)" and then died seconds later
# with refresh_token_reused. The probe and the review hit different endpoints --
# the review uses wss://chatgpt.com/backend-api/codex/responses, which forces a
# token refresh the probe never triggers, so a live ACCESS token passes the
# probe while a dead one-shot REFRESH token kills the review.
# Keep the probe to fail FAST on obviously dead auth, but never treat it as
# evidence the review ran. The verdict comes from codex's OWN final output,
# checked after the fact below.
_pf=$(codex exec --sandbox read-only -C $WT \
        "Reply with exactly this token and nothing else: PREFL""IGHT_ALIVE" 2>&1)
if ! printf '%s' "$_pf" | grep -q "PREFL""IGHT_ALIVE"; then
  echo "FATAL: codex probe FAILED -- auth dead or API refusing. Run 'codex login'."
  printf '%s\n' "$_pf" | grep -iE "unauthor|token|error" | head -4
  exit 2
fi
echo "=== codex probe: alive (NOT proof the review will run) ==="
OUTLOG=/burg-archive/glab/users/pg2328/legoESM/logs/codex_campaign_verdict_${SLURM_JOB_ID}.txt
echo "=== worktree $WT at $(git -C $WT rev-parse HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "ADVERSARIAL REVIEW of the top three commits. Run 'git log --oneline -4' and
'git show' each of the three first.

CONTEXT. This is the SLURM chain for legoESM's SCM-RCE convection-ranking
campaign: scripts/cluster/scm_rce_paper/{arms_ab,merge_ab,arm_c_gradient}.sbatch.
Arm A = literature defaults, arm B = derivative-free tuning (both from
arms_ab.sbatch in one pass), arm C = gradient tuning initialised from arm B's
optimum. Each runs twice, once per kernel arm (--subsidence-solve as_shipped
vs implicit_flux). Jobs 9331806/9331807 (arms) -> 9331808/9331809 (merge) ->
9331810/9331811 (arm C) are IN THE QUEUE RIGHT NOW at 48h walltime, so a defect
found here still saves the round.

ATTACK THESE CLAIMS. For each: CONFIRMED or REFUTED, with file:line.

 1. REFERENCE SUBSTITUTION (the claim the whole campaign rests on). The commits
    assert that scoring against results/rcemip1_n128_ocean was invalid because
    that is our own CRM output rather than an oracle, and switch the default to
    results/rcemip_ref_sam300. Check that EVERY stage now resolves the new
    reference -- arms, merge, arm C, and arms_ab_substep_control. Is there any
    remaining path, default, or fallback inside
    scripts/run/run_scm_rce_convection_intercomparison.py or
    scripts/run/run_scm_rce_campaign.py (camp.DEFAULT_REFERENCE_DIR) that can
    still reach the old directory when REF is unset or empty? A stage that
    silently falls back would produce a ranking labelled with the new reference
    and computed against the old one.

 2. IC/REFERENCE COHERENCE. The SCM column is built from wing2018_* profiles
    (run_scm_rce_campaign.py around line 567), whose constants WING_GAMMA,
    WING_T_V0, WING_Q_SFC_DEFAULT were recalibrated against the gSAM oracle on
    this branch. Confirm the branch actually carries BOTH the new reference and
    the recalibrated constants. If the pinned worktree the jobs execute from
    could carry one without the other, that is a blocker: schemes would start
    from a supersaturated column and be scored against a correct one.

 3. EXIT-CODE HONESTY. Earlier arms aborted with rc=134 while SLURM recorded
    COMPLETED/0:0 because each wrapper ended on 'echo ALLDONE' and returned its
    status. Verify the new RC capture in all three files is actually reading the
    payload's status and not some later command's. Pay particular attention to
    merge_ab.sbatch, where an earlier revision of MY OWN fix read RC=\$? after a
    'sed' and therefore still reported OK for a failed merge. Is the current
    version correct, including the redirect into \$OUT/.merge.log and the
    non-empty summary.md guard? Does the guard's exit code propagate?

 4. DEPENDENCY SEMANTICS. merge is chained afterany on the arms; arm C is
    chained afterok on merge. Argue whether that is right. Specifically: with
    afterany, can the merge publish a ranking from a PARTIAL set of schemes and
    report STAGE_STATUS=OK? If 3 of 10 schemes died, does anything in
    run_scm_rce_convection_intercomparison.py --merge-only fail, or does it
    happily rank the 7 survivors with no visible marker that 3 are missing? A
    ranking silently computed over a subset is the failure mode I care about
    most here.

 5. ARM C DEGENERACY GUARD. arm_c_gradient.sbatch now refuses to run when
    \$AB/tuned_parameters.json is missing or empty, on the stated grounds that
    train_scm_rce_params.py otherwise falls back to shipped *Config defaults and
    silently turns arm C into a relabelled copy of arm A. VERIFY THAT FALLBACK
    ACTUALLY EXISTS by reading train_scm_rce_params.py -- if it does not, the
    comment is a false claim about behaviour and should be corrected. Also: does
    the guard catch a file that exists and is non-empty but contains an empty
    JSON object, or tuned values for a DIFFERENT scheme than \$SCHEME?

 6. CONTROLLED COMPARISON. arms_ab.sbatch's header promises every flag is
    'pinned here, never defaulted' so the protocol is reconstructable from the
    log. I added --scm-convection-substeps \${SUBSTEPS:-10} on that basis and
    claimed it is not a behaviour change because argparse already defaulted to
    camp.DEFAULT_SCM_CONVECTION_SUBSTEPS = 10. Verify that equality. Then check
    whether any OTHER flag consumed by the merge or arm C stages is still
    defaulted rather than pinned, such that arms and merge could disagree.

 7. Anything else that would make the ranking WRONG rather than merely make a
    job fail. I would much rather have a job crash than publish a plausible
    table computed against the wrong target, over a subset of schemes, or with
    two arms that differ in more than the one variable under test.

Do not edit anything. End with a prioritized findings list, each BLOCKER /
HIGH / MEDIUM / LOW, and state explicitly if there are NO BLOCKERS." \
  > "$OUTLOG" 2>&1
RC=$?
cat "$OUTLOG"
echo "codex rc=$RC"

# THE decisive check. An exit code, and a passing probe, both proved nothing
# here on 2026-08-06. A review that produced no verdict is UNREVIEWED, and must
# say so in a way that cannot be mistaken for a clean pass.
#
# Do NOT grep for BLOCKER/HIGH/MEDIUM: codex echoes the PROMPT into its own
# transcript, and the prompt asks for exactly those words, so such a grep
# matches its own instructions and can never fail. A local self-test confirmed
# all four sentinels are present in the prompt text; splitting the literal in
# bash source does not help, because it rejoins before grep ever sees it.
# Gate on `tokens used` instead -- codex's own footer, emitted only after a
# completed turn and absent from anything this script writes.
if [ "$RC" -ne 0 ] || ! grep -qiE "tokens used" "$OUTLOG"; then
  echo "REVIEW_STATUS=UNREVIEWED (rc=$RC, no verdict in codex output)"
  grep -iE "token_expired|refresh_token_reused|usage limit" "$OUTLOG" | tail -2
  echo ALLDONE
  exit 3
fi
echo "REVIEW_STATUS=REVIEWED (codex emitted a verdict)"
echo ALLDONE
exit 0
