#!/bin/bash
#SBATCH --job-name=codex_aei0
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_aei0_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of two things, one code change and one empirical claim.

(A) THE aei0 FIX (commits touching scripts/run/run_omip_core2.py
_GM_AEI0_DEFAULT, tests/unit/test_run_omip_core2_gm_treguier.py, and
packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py).
Claim: the --gm-treguier cap default was 1800 but NEMO's is 900, because
ldftra.F90:290-293 sets zUfac = r1_2*rn_Ud for the LAPLACIAN operator
(ORCA1 runs ln_traldf_lap), so aei0 = 1/2*rn_Ue*rn_Le = 900; NEMO's emitted
aeiu_2d maxes at exactly 900 on eORCA1.
 1. Verify the Fortran: is zUfac really the prefactor that lands in aei0 for
    nn_aei_ijk_t=21, or does the 21 branch recompute it (ldftra.F90 CASE 21
    and ldf_eiv)?  Is aei0 the CAP or the reference value?  Quote lines.
 2. Does our TreguierConfig use aei0 the same way NEMO uses it (cap on a
    growth-rate kappa), or is there an off-by-a-factor in how we apply it?
 3. Is 900 right for the EIV coefficient specifically (rn_Ue/rn_Le in
    namtra_eiv) as opposed to the tracer aht (rn_Ud/rn_Ld in namtra_ldf)?
    Both are 0.018/100e3 in ORCA1, so the numbers coincide -- say whether
    the DERIVATION is right or only the number.
 4. The DINO default (config.py aei0=3000) was deliberately NOT changed.
    Is that the correct call, or does DINO also run a laplacian and so
    inherit the same factor-2 error?

(B) THE fct2 A/B INTERPRETATION.  Measured at day 30 vs NEMO month 1,
one variable (--tracer-advection superbee -> fct2), same window:
   superbee: SST rmse 0.9116, SSS 0.8583, MLD 37.8, ArcSST 1.067, deep 161
   fct2    : SST rmse 0.9166, SSS 0.8744, MLD 38.7, ArcSST 1.075, deep 160
I concluded: REFUTED at the state level; matching NEMO's advection OPERATOR
(Stage C: fct2 correlates better with ttrd_totad in every region) does not
improve the 30-day STATE, so superbee stays the default.
 5. Is that conclusion supported, or could the differences be noise?  What
    is the effective sample size / what spread would two identical-physics
    runs show (deterministic model, so is ANY difference meaningful)?
 6. Is there a confound: the fct2 arm dropped --gateway-transports (the
    driver refuses store_salt_flux with fct2).  Could that change the
    scored SST/SSS/MLD at all, or is it purely diagnostic?
 7. Steel-man the opposite reading: what would have to be true for fct2 to
    be the better scheme despite these numbers, and what measurement would
    discriminate?
Severity-tagged findings and a verdict per item."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
