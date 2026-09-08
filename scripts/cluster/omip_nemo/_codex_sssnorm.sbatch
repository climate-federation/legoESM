#!/bin/bash
#SBATCH --job-name=codex_sss
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_sss_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of commit 0ff0b0869 ('NEMO-form SSS restoring (live-S
denominator), unblocking real_freshwater') and every claim in its message.

NEMO source:   /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE
NEMO namelist: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/namelist_cfg

Severity-tagged, verdict per item:
 1. VERIFY THE FORTRAN. Read sbcssr.F90 and namelist_cfg. Is ORCA1 really
    nn_sssr=2, and is the nn_sssr=2 branch really a VOLUME flux into emp
    divided by LIVE sss_m? Does it also apply an associated heat flux (qns)
    that we do NOT reproduce? If so, is omitting it a defect or negligible?
 2. THE SIGN. I claim NEMO's erp is positive UPWARD (into emp) and ours is
    positive INTO the ocean, so they differ by exactly a minus sign and both
    freshen a too-salty cell. Verify against how emp and our freshwater_flux
    are each consumed downstream. A sign error here silently doubles the bias.
 3. THE MAGNITUDE CLAIM. I claim z1_m/tau = 10 m / 45.5 d = 219.8 mm/day equals
    NEMO's rn_deds = -220 mm/day, so only the DENOMINATOR differed. Check the
    unit algebra including rho, and whether our regional tau map
    (build_region_masks / DEFAULT_OMIP2_REGIONS) makes tau spatially varying --
    if it does, the single-number comparison is only valid where tau = 45.5.
 4. THE CAP. We clip freshwater_flux to +/- max_flux_kg_m2_s and RE-DERIVE
    dS_dt_top from the capped flux. NEMO clips zerp with ln_sssr_bnd BEFORE it
    enters emp. Under live_s, are those the same operation, or does our
    re-derivation change what the bound means?
 5. THE GUARD. real_freshwater_restoring_conflict() now allows
    real_freshwater + live_s. Is that sound? The original refusal cited BOTH
    the S_target division AND the FIXED z1. live_s fixes the first -- does the
    fixed z1_m still make the relaxation the wrong strength under volume-only
    dilution, i.e. did I remove a guard that was right for two reasons by
    fixing only one?
 6. TEST NON-VACUITY. tests/ocean/unit/test_sss_restoring.py, classes
    TestNEMONormalization and TestRealFreshwaterRestoringGuard. Would each fail
    if the feature were reverted? Is test_live_s_matches_the_nemo_formula
    re-deriving OUR implementation rather than independently encoding NEMO's?
 7. BIT-IDENTITY. I claim the default 's_target' leaves every existing run
    unchanged. Verify no other caller constructs SSSRestoringConfig in a way
    that now takes a different branch.
 8. STEEL-MAN THE NULL: give the strongest reason real_freshwater + live_s will
    NOT reduce the Arctic +1.4 psu bias, and name the measurement that
    discriminates your reason from mine."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
