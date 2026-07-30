#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_codex_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt_final
# Codex needs a NODE-LOCAL CODEX_HOME (shared-FS home has failed before).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version

cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW, FINAL ROUND. Review the WHOLE branch:
`git log --oneline cf/main..HEAD` and `git diff cf/main...HEAD`.

This is a paper campaign that will publish a RANKING of 10 convection schemes
against a CRM under single-column RCE, in 3 parameter arms x 2 transport-kernel
configurations. Rounds 1-3 already fixed: Emanuel's dead selector, a missing
condensation-heating pairing in Tiedtke/ZM/mass_flux, the SAME defect in EDMF
(a real bug in shipped code), Bechtold's unconditional latent term, the
param-spec ratchet, and a wrong conserved quantity in BOTH the unit gate and
the validator's Test-5 sweep.

Your job now is ONLY to find things that would make the PUBLISHED RANKING
WRONG, or that are outright defects. Do not re-litigate style.

Check specifically:

1. Any remaining asymmetry between the 10 schemes that the two kernel arms do
   NOT control for, and that a reader would mistake for scheme physics. E.g. a
   scheme that gets a latent release the others do not, a different condensate
   hand-off convention, a different trigger normalisation, a prognostic carry
   that is seeded differently, or a scheme whose SCM substep count differs
   (see DEFAULT_SCM_CONVECTION_SUBSTEPS / SCM_CONVECTION_SUBSTEP_SCHEMES in
   scripts/run/run_scm_rce_campaign.py -- does it apply to only SOME schemes?
   if so that is an uncontrolled per-scheme difference in the ranking).
2. The anti-confound machinery: is there ANY path by which a cached result
   from one kernel arm, or one parameter arm, can be reused for another?
   Check _run_signature/_SIGNATURE_FIELDS in
   run_scm_rce_convection_intercomparison.py, run_cached in
   run_scm_rce_campaign.py (what is its cache KEY? does it include the
   config?), and the arm (c) trainer.
3. scripts/cluster/scm_rce_paper/*.sbatch -- do the flags actually exist, are
   the two arms identical in EVERYTHING except --subsidence-solve, and does
   arm (c) read the tuned_parameters.json of the MATCHING arm (not the other)?
4. The new anti-confound guard in train_scm_rce_params.py
   _choose_active_scheme_keys (raises SystemExit when --convection names a
   scheme absent from tuned_parameters.json). Is it correct and can it
   false-positive?
5. Any remaining place where a scheme is reported as kernel-matched when it is
   not, or where a conservation residual differs between arms for a reason
   OTHER than the kernel.
6. Genuine bugs introduced anywhere in the diff.

Output a numbered list tagged BLOCKER / MAJOR / MINOR / NIT with file:line and
a concrete fix. If the branch is clean on a point, SAY SO explicitly. Do not
pad the list.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
