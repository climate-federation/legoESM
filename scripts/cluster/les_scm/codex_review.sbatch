#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_codex
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_codex_%j.out
set +e
# Mandatory adversarial review of the LES -> SCM turbulence-tuning stack.
# CLAUDE.md requires codex review after any substantial numerics/physics change.
# The /codex: slash commands crash, so this drives the CLI directly.
#
# TRAP (recorded from a previous session): a CODEX_HOME pointing somewhere
# unwritable makes codex exit 0 having reviewed NOTHING. Use node-local /tmp
# and check for real output below.
WT=/burg/glab/users/pg2328/les_scm_bridge_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp -r /burg-archive/home/pg2328/.codex/* $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH

OUT=/burg-archive/glab/users/pg2328/legoESM/logs/les_codex_review_$SLURM_JOB_ID.md

read -r -d '' PROMPT <<'EOF'
Adversarial review of a MULTI-CASE LES-to-SCM turbulence tuning stack and a
stratocumulus entrainment investigation. Be hostile: report ONLY defects,
ranked HIGH/MEDIUM/LOW with file:line and a concrete failure scenario.

Files:
  packages/atmosphere/legoesm/atmosphere/forcing/scm/analytic_scm_case.py  (NEW)
  packages/atmosphere/legoesm/atmosphere/forcing/scm/sam_case_scm.py
  packages/ml/legoesm/training/les_reference.py
  packages/ml/legoesm/training/scm_rce_metrics.py
  scripts/run/run_scm_les_turbulence_tuning.py
  scripts/cluster/les_scm/dycoms_entrainment_ab.sbatch

PART A - the new analytic-case bridge. Each case was transcribed from the LES
driver that produced its stored reference, because the same-named cases in
scripts/matrix/scm/ do NOT match (run_spectral_cbl.py integrates Nieuwstadt
CBL_N91, not Wangara Day 33; the archived GABLS1 reference took the
constant-flux branch, not the cooling-T_s branch).
 1. Verify each transcription against its driver: theta profile (including the
    tanh-smoothed GABLS1 inversion), surface flux sign and magnitude,
    Coriolis, geostrophic wind, z0, domain top. Any value that does not match
    the driver is a defect.
 2. les_domain_top_m is set to 0.75*Lz wherever the LES sponges. Is that the
    right cut, and is the sponge really at 0.75*Lz in each driver?
 3. The SCM column is built by a hydrostatic theta->p mapping while these LES
    are BOUSSINESQ with uniform dz. Does that mismatch invalidate the
    comparison, and is the resulting z placement right?
 4. Ekman scores u,v only and CBL scores theta only. Is that justified, or is
    a scored variable being dropped that carries real signal?

PART B - the multi-case joint tuning.
 5. joint = mean over cases of a per-case normalized score. Can one case
    dominate or be silently ignored? Are the per-case scores really
    commensurable given each is normalized by its OWN reference spread?
 6. The window guard now tolerates half an SCM timestep. Can that hide a real
    window mismatch?
 7. A non-finite rollout is mapped to NONFINITE_PENALTY because safe_sqrt(NaN)
    returns 0 and 0 is the perfect score. Is the penalty applied everywhere it
    must be (per-case scores, per-component scores, the joint, the gradient
    path)? Can a NaN still reach a ranking or a tuned parameter? Does the
    penalty break the line search or the preflight freeze?
 8. Per-case surface configs, per-case scored variables, per-case windows: is
    the across-scheme control still valid, i.e. is it still true that within a
    case only PhysicsConfig.turbulence differs?
 9. Anything left over from the single-case refactor: dead code, stale
    references, output fields that no longer mean what they say.

PART C - the DYCOMS entrainment A/B.
10. The experiment concludes: the coarse config burns off worse over 2 h than
    the fine one, so the good 0.25 h smoke was TIME not resolution; removing
    the Lilly SGS suppression makes it worse, so that suppression is active.
    Are those conclusions supported by the arms actually run, or is there a
    confound (arm A changes BOTH nx/ny/nz AND Lx/Ly - is that one variable)?
11. Is 2 h long enough to conclude anything about a 4 h case?
EOF

echo "$PROMPT" > $CODEX_HOME/prompt.txt
codex exec --sandbox read-only -C $WT "$PROMPT" 2>&1 | tee $OUT
rc=$?
echo "codex rc=$rc"
lines=$(wc -l < $OUT)
echo "review output lines: $lines"
if [ "$lines" -lt 20 ]; then
  echo "REVIEW SUSPECT: fewer than 20 lines of output — codex may have run"
  echo "with a bad CODEX_HOME and reviewed nothing. Treat as UNREVIEWED."
  exit 3
fi
echo "review written to $OUT"
echo "DONE $(date +%T)"
