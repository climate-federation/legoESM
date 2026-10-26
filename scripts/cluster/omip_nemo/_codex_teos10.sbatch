#!/bin/bash
#SBATCH --job-name=cdxteos
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxteos_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "HEAD: $(git -C $WT log --oneline -1)"
codex exec --sandbox read-only -C $WT \
  "REFUTE THIS EOS PORT IF IT IS WRONG. Silent wrong numbers are the failure
mode I care about, not style.

WHAT LANDED (see git log for eos.py, last 2 commits):
 * _ROQUET_TEOS10 -- 52 EOS + 35 ALP + 35 BET coefficients, transcribed
   MECHANICALLY from eosbn2.F90:1926-2110 by a parser that was first
   validated by requiring it to reproduce all 52 coefficients of the
   existing, trusted _ROQUET_EOS80 exactly.
 * nemo_roquet_alpha_beta -- NEMO rab_3d Horner (eosbn2.F90:1108-1143),
   alpha = zn_ALP*r1_rho0, beta = zn_BET/zs*r1_rho0, depth is GEOMETRIC
   gdept*r1_Z0.
 * compute_buoyancy_frequency_nemo_bn2(eos_form='teos10') selects it; the
   bn2 assembly is unchanged because NEMO shares bn2_t across EOS branches.

QUESTIONS, quote sources:
 1. Check the coefficient VALUES against eosbn2.F90 -- spot-check at least
    ALP000/ALP500/ALP050, BET000/BET500/BET050, EOS000/EOS600/EOS060, and
    the normalization (rdeltaS=32, r1_S0=0.875/35.16504). Any single wrong
    digit is the whole point of this review.
 2. Is the Horner NESTING right? I collapsed ALP and BET into one _horner(P)
    closure because NEMO's two blocks are structurally identical. Verify
    they really are -- if BET's nesting differs anywhere, the shared closure
    silently computes the wrong polynomial for one of them.
 3. My validation: alpha/beta reproduce a central difference of
    nemo_roquet_eos (same TEOS-10 coefficients) to 2e-4. Is that check
    CIRCULAR? Both use _ROQUET_TEOS10, so it validates internal consistency
    of the ALP/BET tables against the EOS table. Does it actually catch a
    wrong ALP coefficient, or only a wrong /zs factor?
 4. CT/SA. ORCA1 sets ln_teos10 so NEMO's tracers ARE Conservative
    Temperature and Absolute Salinity. Our model carries potential
    temperature and practical salinity, but both models initialise from the
    SAME files. Is feeding PT/SP into a CT/SA polynomial defensible for an
    ORCA1 twin, or does it need a conversion? Quantify the error if it does.
 5. I measured this port as NOT improving agreement with NEMO's own rn2
    (recovered as 2*dissl^2): log10 rms 0.527 vs 0.537 antarctic, 0.303 vs
    0.298 tropics, 0.647 vs 0.639 arctic -- better in 1 of 3 bands. Is that
    inversion (rn2 = 2*dissl^2 from zdftke.F90:651,680,717) valid? Where do
    the sweeps contaminate it, and does that explain the ~0.3-0.65 log rms?
 6. Anything else I have over-read or under-tested."
rc=$?
echo "=== CODEX RC=$rc ==="
