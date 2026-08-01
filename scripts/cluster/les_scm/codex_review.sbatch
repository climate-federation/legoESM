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
Adversarial review of a new LES-to-SCM turbulence-closure tuning stack in this
repository. Be hostile: find defects, do not summarise or praise. Rank findings
HIGH / MEDIUM / LOW with file:line and a concrete failure scenario for each.

Files under review:
  packages/atmosphere/legoesm/atmosphere/forcing/scm/sam_case_scm.py
  packages/ml/legoesm/training/les_reference.py
  scripts/run/run_scm_les_turbulence_tuning.py
  scripts/run/les_record.py
  packages/ml/legoesm/training/scm_rce_metrics.py  (only the newly factored
      normalized_profile_rmse and its use by score_profiles_jax)

Focus, in priority order:

1. SIGN AND UNIT CONVENTIONS. sam_case_scm converts a gSAM deck into SCMForcing.
   Verify against the actual consumer, scm_forcing.compute_forcing_tendencies:
   (a) is w_th_s really an ABSOLUTE-temperature kinematic flux there (added to
   dT_dt with no Exner factor) so SHF/(rho*c_pd) is right and the LES's
   SHF/(rho*c_pd*Exner) would be wrong? (b) is theta_adv really a potential-
   temperature tendency so dividing the deck's tls by Exner is right?
   (c) is subsidence_w positive-up on both sides? (d) is rho_sfc computed from
   a defensible temperature? Check every sign against the code that consumes
   it, not against the docstrings.

2. VERTICAL ORDERING. The deck is bottom-to-top in height; the SCM is
   top-to-bottom in sigma. Audit every interpolation, mask and flip in
   sam_case_scm._deck_pressure_profile, _heights_from_pressure, and
   les_reference._interp_to_scm for an inverted array. One reversal bug was
   already found and fixed here; look for others.

3. DIFFERENTIABILITY AND RETRACE. run_scm_les_turbulence_tuning rebuilds the
   SCM inside the loss so config leaves are traced. Does anything in that path
   break the gradient (a Python float cast, a NumPy op on a traced value, a
   .item(), control flow on a tracer)? Is jax.checkpoint applied correctly
   around the chunked scan? Does the masked_step lax.cond padding leak into the
   analysis-window mean?

4. THE CONTROLLED-COMPARISON CLAIM. The driver claims every scheme gets a
   byte-identical column and forcing and differs only in
   PhysicsConfig.turbulence. Find any way that is false: mutable state shared
   across arms, a scheme whose config carries a different surface default, the
   Ch_neutral zeroing missing a scheme, order-dependence, a cached closure.

5. THE LES REFERENCE. Is the trailing analysis window actually the same window
   on both sides (LES frames vs SCM rollout steps)? Can a frame be silently
   dropped or double counted? Is the mass weighting correct for a top-to-bottom
   column? Does the NaN-outside-mask sentinel ever reach arithmetic?

6. les_record._profiles: is wth = mean(w' * theta') computed with the correct
   perturbation (is the theta mean removed on the right axes)? Does the new
   optional-scalars path change any previously written key?

Report ONLY defects. If a claim in a docstring is wrong, say so.
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
