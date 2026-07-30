#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_codex_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt3
# Codex needs a NODE-LOCAL CODEX_HOME (shared-FS home has failed before).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version

cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW, ROUND 2. Review commit 9c0b39836 (HEAD) together with its
parent b47595e22 -- `git show HEAD`, `git show HEAD~1`, and `git diff HEAD~2..HEAD`
give the full two-commit change.

Round 1 raised 2 BLOCKERs and 4 MAJORs. This commit claims to fix them:

 B1 Emanuel selector was dead under the shipped config
    -> subsidence_solve/theta_implicit REMOVED from EmanuelConfig; Emanuel
       declared OUTSIDE the matched-kernel family.
 B2 missing +(L_v/c_p)dq_c condensation heating under implicit_flux
    -> new SHARED helper mass_flux.release_detrained_condensate_latent,
       applied in tiedtke.py, zhang_mcfarlane.py, mass_flux.py; Kain-Fritsch's
       duplicated inline term routed through the same helper.
 M4 theta_implicit violated the __param_spec__ ratchet
    -> added to the `excluded` block of MassFlux/ZM/Tiedtke/KainFritsch.
 M5/M6/M7 weak tests
    -> run(None) default-vs-explicit comparison; relative machine-precision
       water closure + strict improvement; new column-MSE gate; new
       finite-difference VJP check of the implicit kernel itself; new
       test_fixture_actually_convects.
 Also: KainFritschConfig gained subsidence_solve defaulting to "implicit_flux"
 (was hardcoded at the call site); mass_flux_convection_from_closure's `dt`
 is now REQUIRED.

VERIFY EACH FIX ACTUALLY LANDS, and be adversarial about the NEW code:

1. Is `release_detrained_condensate_latent` correct in SIGN and MAGNITUDE, and
   is it applied EXACTLY ONCE per unit of detrained condensate in each of
   tiedtke / zhang_mcfarlane / mass_flux / kain_fritsch? Specifically:
   - Tiedtke pairs against the KERNEL's third return (dq_c_kernel) but hands
     microphysics a DIFFERENT, separately recomputed dq_c_conv_dt. Are those
     two profiles actually equal? If not, is the heating paired with the vapor
     debit (correct) or with the microphysics hand-off (wrong)? Quantify.
   - Does Tiedtke's downdraft block, or ZM's split_convective_rain, or KF's
     CONDLOAD/downdraft chain, now double-count or double-heat?
   - Bechtold still applies its own inline term UNCONDITIONALLY (not routed
     through the helper). Under BechtoldConfig(subsidence_solve="advective")
     does Bechtold now over-heat? Is that a pre-existing defect, a new one, or
     not a defect at all?
2. Is making the helper a NO-OP on "advective" right? Argue from the kernel's
   actual advective branch (does it debit vapor for dq_c or not?).
3. Does the KF change alter SHIPPED behaviour in any way? The default must
   remain exactly the previous hardcoded "implicit_flux" path, and the
   previously-unconditional latent term must still fire on that default.
4. Re-check the __param_spec__ ratchet: does tests/test_param_specs.py now
   pass for this module? Did the excluded entries land in the RIGHT config
   blocks (verify by reading the dict, not by trusting the diff)?
5. Attack the NEW tests. Is test_implicit_flux_closes_column_mse genuinely
   non-vacuous -- would it FAIL if release_detrained_condensate_latent were
   deleted? Is the MSE integrand the right conserved quantity (is including
   +L_v*dq_r correct, or does the rain split already account for it)? Is the
   finite-difference VJP tolerance meaningful? Does the fixture convect for
   ALL FOUR schemes including kain_fritsch?
6. run_scm_rce_campaign.apply_subsidence_solve_override: any way it silently
   reports a scheme as kernel-matched when it is not, or vice versa? Is
   returning `not_applicable` for emanuel the honest outcome?
7. Anything else that would make a published scheme RANKING wrong.

Output a numbered list, each tagged BLOCKER / MAJOR / MINOR / NIT, with
file:line and a concrete fix. Explicitly state which round-1 findings are now
RESOLVED and which are NOT. If a concern is unfounded, say so rather than
padding.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
