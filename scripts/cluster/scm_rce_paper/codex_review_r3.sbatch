#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_codex_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt4
# Codex needs a NODE-LOCAL CODEX_HOME (shared-FS home has failed before).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version

cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW, ROUND 3. Review HEAD (9a2fb6738) and the two commits below
it: `git log --oneline -3`, `git show HEAD`, `git diff HEAD~3..HEAD`.

Rounds 1-2 raised issues that this commit claims to fix:

 R2-1 BLOCKER: EDMF called the kernel with implicit_flux but never added the
      paired condensation heating -> now routed through
      release_detrained_condensate_latent in mass_flux.edmf_convection.
      NOTE this deliberately CHANGES shipped EDMF behaviour (adds
      previously-missing heating).
 R2-2 MAJOR: Bechtold's latent term was unconditional -> now gated through the
      helper. DELIBERATE DEVIATION from your suggestion: it still pairs against
      Bechtold's OWN dq_c_conv_dt (recomputed from UNCLIPPED M_u_new), not the
      kernel's third return (clipped to M_b_max), to avoid changing this
      validated scheme's DEFAULT output. Judge whether that deviation is
      acceptable or whether it leaves a real mispairing on the default path --
      quantify how often M_u_new exceeds M_b_max in practice.
 R2-3 MAJOR: MSE gate used the wrong invariant -> replaced with vapor MSE
      c_p*dT + L_v*dq_v (test_implicit_flux_closes_vapor_mse).
 R2-4 MINOR: VJP tolerance -> assert_allclose(rtol=1e-5, atol=1e-9).
 EDMF and Bechtold added to the threading test matrix (now 6 schemes).

VERIFY EACH, then attack what is NEW:

1. Is the EDMF fix placed correctly RELATIVE TO split_convective_rain? The
   helper is called BEFORE the split. Is that right, or does the split need to
   see the heated/unheated state? Compare with how Bechtold and ZM order the
   two. Does EDMF now double-heat via any other path (its a_u carry, its
   updraft_velocity_from_buoyancy, its own precip)?
2. Quantify the EDMF behaviour change: what is the expected magnitude of the
   added heating in K/day for a typical tropical column, and could it
   destabilise an AMIP run? Is there anywhere in the repo (tests, baselines,
   tuned parameter JSONs, documentation) that encodes the OLD unheated EDMF
   behaviour and will now be wrong or fail?
3. Re-examine the Bechtold deviation in 2 above. Also: does Bechtold's
   sub-cloud/downdraft re-evaporation, or its IFS in-plume precip path
   (use_ifs_inplume_precip), now interact wrongly with the gated term under
   subsidence_solve="advective"?
4. Is test_implicit_flux_closes_vapor_mse now the RIGHT invariant, and is its
   5e-2 tolerance justified rather than tuned to pass? Does it hold for EDMF
   and Bechtold, whose condensate profiles differ from the kernel return?
   Would it FAIL if the helper were deleted, for every one of the 6 schemes?
5. scripts/validate/validate_convection_physics.py grew a `dca` scheme, a
   subsidence_solve arm, kernel_arm_status, and a Test-5 sweep with
   --json-out. Check: does adding `dca` break the existing Test-2/3/4
   assertions or the _NEAR_ZERO_EXEMPT logic? Is kernel_arm_status's hasattr
   policy consistent with run_scm_rce_campaign.apply_subsidence_solve_override
   (they must never disagree about which schemes are matched)? Is `args`
   in scope where Test 5 uses it?
6. Any remaining path where a PUBLISHED SCHEME RANKING could be wrong: a
   scheme silently reported as kernel-matched when it is not, an energy or
   water leak that tuning would compensate for, or a conservation residual
   that differs between the two ranking arms for a reason other than the
   kernel.

Output a numbered list tagged BLOCKER / MAJOR / MINOR / NIT with file:line and
a concrete fix. State explicitly which prior findings are RESOLVED and which
are NOT. If a concern is unfounded, say so rather than padding.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
