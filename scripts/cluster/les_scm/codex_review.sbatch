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
SECOND-ROUND adversarial review of the LES-to-SCM turbulence-closure tuning
stack. A first round returned 11 findings; they have been fixed. Be hostile:
verify the fixes are REAL and find what round one missed. Rank HIGH/MEDIUM/LOW
with file:line and a concrete failure scenario. Report ONLY defects.

Files:
  packages/atmosphere/legoesm/atmosphere/forcing/scm/sam_case_scm.py
  packages/ml/legoesm/training/les_reference.py
  scripts/run/run_scm_les_turbulence_tuning.py
  scripts/run/les_record.py
  tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py
  tests/unit/test_les_reference.py
  tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py

PART A - are these fixes real, or cosmetic?
 1. Column top is now the LES domain top (sigma_top derived from
    les_domain_top_m). Does that actually bound the column, and is the
    derivation correct given the p(z) reconstruction?
 2. Height mapping and rho_sfc now use virtual potential temperature. Is the
    theta_v formula right, is epsilon used correctly, and does it match what
    spectral_les_moist.make_anelastic_reference does?
 3. surface_mode is declared per case. Is RICO's "T_s" actually what
    run_rico_les.py does, and does the SCM's interactive bulk flux then
    resemble the LES's C_H/C_Q coefficients at all?
 4. Constant surface series collapse to constants. Does that genuinely keep
    T_s off the traced path in the rollout, or does another traced call remain?
 5. u/v dropped from the default score. Does anything still tune against them?
 6. --hours must equal the LES reference end. Can the check be bypassed?
 7. Terminal-frame recording in the six LES drivers: is the `_last_rec_h`
    tracker correct in EVERY driver, in scope, and free of duplicate emission?
    Check the Lagrangian-SDM path in run_bomex_les.py specifically.
 8. Frame case-matching and duplicate-timestamp rejection: can a wrong or
    stale reference still get through (e.g. missing 'case' key, subtly
    different times, a frame outside the analysis window)?

PART B - what round one missed. Prioritise:
 - Any remaining sign/unit error, checked against the CONSUMER not docstrings.
 - Gradient correctness: does the loss actually depend on the turbulence
   parameters for EACH of the nine schemes, or will some silently freeze?
   Is jax.checkpoint applied so the chunked scan is correct, not just cheap?
 - The masked_step lax.cond padding and the [:nsteps] slice: can padding reach
   the analysis mean, the p_s drift assert, or a gradient?
 - Anything that makes the ACROSS-SCHEME ranking not controlled.
 - Tests that would pass even if the feature were removed (assert-nothing
   tests, tests that inspect a delegating wrapper).
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
