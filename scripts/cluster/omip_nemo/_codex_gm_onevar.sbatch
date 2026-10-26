#!/bin/bash
#SBATCH --job-name=codex_gm1v
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_gm1v_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of an uncommitted change plus the empirical claim behind it.

THE CLAIM.  scripts/run/run_omip_core2.py's --gm-treguier branch built its
GM/Redi block from run_omip._DEFAULT_BATHY_GM_REDI.  I claim that constant is
the LAT-LON bathymetry default (kappa_GM=kappa_Redi=800, VisbeckConfig
enabled=True, slope_scheme left at the GMRediConfig default 'triads') and NOT
the block the eORCA1 tripole control actually runs, which comes from
legoesm.ocean.fidelity.nemo_match_recipe (kappa_GM=kappa_Redi=600, Visbeck
disabled, slope_scheme='centered').  Evidence: two run_manifest.json files,
arm results/omip_nemo/nemolev_trp_gmtreg600_d5 vs control
results/omip_nemo/nemolev_trp_gwcorr_d30, differ in kappa_GM, kappa_Redi,
slope_scheme AND treguier.  Consequence claimed: no --gm-treguier arm was ever
a one-variable experiment, and the day-5 non-finite abort of the aei0=900 arm
(job 9379514) cannot be attributed to the Treguier coefficient.

THE CHANGE.  git diff (uncommitted) in:
  scripts/run/run_omip_core2.py                    (new _tripole_treguier_gm_redi)
  packages/ocean/legoesm/ocean/fidelity/nemo_match_recipe.py  (gm_aei0 1800->900)
  tests/unit/test_run_omip_core2_gm_treguier.py    (new TestTreguierBlockIsOneVariable)
  tests/ocean/unit/test_nemo_match_recipe.py       (aei0 expectation 1800->900)

Review, severity-tagged, verdict per item:
 1. Is the claim about _DEFAULT_BATHY_GM_REDI correct?  Read the constant and
    the tripole path in run_omip._create_setup.  Does build_tripole's _ovr
    dict actually override the recipe block, i.e. is the control's block really
    the recipe's?  Quote the lines that decide it.
 2. Does _tripole_treguier_gm_redi reproduce the control block exactly apart
    from `treguier`?  Check every GMRediConfig field, including ones the new
    test does not name.  Is the NamedTuple _fields comparison in the test
    sound, or can a nested config compare equal while differing?
 3. nemo_match_tripole_model_config(cfg) also builds a physics block when
    physics=None.  Is taking only .gm_redi from it safe, or does that path have
    side effects / cost?
 4. The recipe default gm_aei0 changed 1800 -> 900.  gm_treguier defaults to
    False, so I claim no existing default run changes behaviour.  Verify by
    finding every consumer of gm_aei0.  Is 900 right for the EIV coefficient
    (rn_Ue/rn_Le in namtra_eiv, laplacian prefactor zUfac=r1_2*rn_Ud,
    ldftra.F90:290-293)?
 5. NON-VACUITY: scripts/cluster/omip_nemo/_gm_onevar_tests.sbatch copies the
    new test file onto a pinned pre-fix worktree (b9334f9b8) and asserts a
    non-zero rc.  Can that check pass for the wrong reason (import error,
    collection error, missing symbol) rather than a genuine assertion failure?
 6. What ELSE in run_omip_core2.py builds a config block from a run_omip
    default that does not belong to the grid being built?  Same defect class.
 7. Steel-man: is there a legitimate reason --gm-treguier should have carried
    the lat-lon 800/triads block, e.g. Treguier being tuned for it?  What
    measurement would decide?"
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
