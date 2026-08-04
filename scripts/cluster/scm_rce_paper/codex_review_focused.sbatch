#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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
FOCUSED REVIEW. Budget your time: answer the SIX questions below and stop.
Do not read the whole repository.

Context: a paper campaign ranks 10 convection schemes against a CRM in
single-column RCE, in 3 parameter arms x 2 transport-kernel arms. The branch is
`git diff cf/main...HEAD --stat`. I only need defects that would make the
PUBLISHED RANKING WRONG.

Q1. scripts/run/run_scm_rce_campaign.py defines DEFAULT_SCM_CONVECTION_SUBSTEPS
    and SCM_CONVECTION_SUBSTEP_SCHEMES. Print SCM_CONVECTION_SUBSTEP_SCHEMES.
    If the substepping applies to only SOME of the 10 convection schemes, that
    is an UNCONTROLLED per-scheme difference in the ranking (some schemes get a
    60 s effective convective step, others 600 s). Say whether it is
    controlled, and if not, exactly which schemes differ.

Q2. In the same file, find `run_cached`. What is its cache KEY? Does the key
    include the full PhysicsConfig (so two different subsidence_solve values
    cannot collide), or only the label string? Quote the code.

Q3. scripts/run/run_scm_rce_convection_intercomparison.py: quote
    _SIGNATURE_FIELDS and confirm "subsidence_solve" is in it. Then confirm
    that in main() the SAME signature is both compared against an existing
    checkpoint AND stamped into newly saved ones.

Q4. Compare scripts/cluster/scm_rce_paper/arms_ab.sbatch and
    arm_c_gradient.sbatch. Are the two ARM values ("implicit_flux" vs
    "as_shipped") the ONLY difference between the two invocations of each
    script, and does arm_c read tuned_parameters.json from the arms_ab output
    directory of the MATCHING arm (not the other one)? Do all the flags passed
    actually exist in each script's argparse?

Q5. scripts/run/train_scm_rce_params.py `_choose_active_scheme_keys` has a new
    SystemExit guard when --convection names a scheme with no rows in
    tuned_parameters.json. Can it false-positive (abort a legitimate run)?

Q6. Any OTHER uncontrolled per-scheme asymmetry in the SCM harness that the
    two kernel arms do not equalise -- e.g. a scheme whose prognostic carry is
    seeded differently, a per-scheme special case in run_scm_rce, or a scheme
    excluded from the realism/equilibrium gate.

Answer each as Q1..Q6 with a verdict (CLEAN or a finding tagged
BLOCKER/MAJOR/MINOR) plus file:line evidence. Be brief. Stop after Q6.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
