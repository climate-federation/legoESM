#!/bin/bash
#SBATCH --job-name=cdxclip
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxclip_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "HEAD: $(git -C $WT log --oneline -1)"
codex exec --sandbox read-only -C $WT \
  "YOU REFUTED THIS EXACT CHANGE ONCE. Check whether I have actually answered
you or just re-broken it.

HISTORY. In review 9405307 you refuted n2_mode='nemo_bn2' on the ORCA1 card
because legoESM's nemo_bn2 is S-EOS while ORCA1 runs ln_teos10=.true., so it
was a DIFFERENT N2 -- and separately because the MPAS bridge fail-loud
rejects n2_mode != 'insitu', which would have broken the cross-grid arms. I
reverted it. Since then I added the TEOS-10 coefficient set and rab
(9a3a54ce3), and in review 9408213 you confirmed those coefficients and the
Horner nesting but flagged that the TEOS path was NOT production-wired
(compute_N2's nemo_bn2 branch omitted eos_form).

WHAT I HAVE NOW DONE (git diff HEAD~1..HEAD and the working tree):
 (a) TKEConfig.n2_eos_form ('seos'|'teos10'), forwarded tke.py ->
     compute_N2 -> compute_buoyancy_frequency_nemo_bn2(eos_form=).
 (b) ORCA1 card: n2_mode='nemo_bn2', n2_eos_form='teos10'.
 (c) MPAS: guard narrowed from 'anything != insitu' to 'not in (insitu,
     nemo_bn2)', and the bridge now threads the geometric depth ladders via
     a new _bn2_ladder_kwargs helper.
 (d) rho0 pinned to NEMO's 1026 in nemo_roquet_alpha_beta.

WHY I BELIEVE IT IS WORTH DOING: job 9407791 showed the clipped in-situ N2 is
the whole remaining mixing-length deficit -- seeding the length with a signed
N2 moved our zero-step length ratio against NEMO's own from 0.897 to 0.997
(Southern Ocean), 0.914 to 0.998 (tropics), 0.792 to 1.004 (Arctic). And at
day 30 the sqrt(2)+alpha fixes drove the Southern Ocean MLD from +1.5 m to
-19.7 m, i.e. removing two over-mixing errors exposed this under-mixing one.

QUESTIONS, quote sources:
 1. Is (a) actually complete? Trace card -> TKEConfig -> tke.py -> compute_N2
    -> bn2 and find any call site that still drops eos_form or n2_eos_form.
    I forwarded at 2 sites in tke.py; are there others (catke, the
    enhanced_diffusion convection path, the Mode-B branch)?
 2. Is the MPAS narrowing SAFE? Its old message blamed cell-centre
    hydrostatic pressure. Confirm that requirement is specific to
    'adiabatic' and that 'nemo_bn2' needs only the ladders. Then check my
    _bn2_ladder_kwargs actually reaches the kernel on the MPAS path and that
    nemo_bn2_depth_ladders works for an MPAS z_coord (does it have
    t_depth_ref / z_full_ref / z_half_ref?).
 3. DOES THIS CHANGE THE CONVECTION TRIGGER? n2_mode also feeds EVD and the
    Prandtl chain. Going from clipped-nonnegative to SIGNED means N2<0 is now
    reachable in production for the first time. What else keys off the sign?
    Name anything that assumed N2 >= 0 and would now take a different branch
    -- especially the enhanced_diffusion / EVD path and any sqrt of N2.
 4. IS THE DIRECTION RIGHT FOR THE DAY-30 REGRESSION? Signed N2 lengthens
    the mixing length in neutral water, which INCREASES mixing. Our Southern
    Ocean at day 30 is 19.7 m too shallow. Is +mixing the right sign, and is
    the magnitude plausibly ~20 m, or am I about to overshoot the other way
    again?
 5. What have I under-tested? I pin the card values, the compute_N2 forward
    (asserting the two forms DIFFER), and the MPAS guard/ladders by source
    inspection. Name a defect those would miss."
rc=$?
echo "=== CODEX RC=$rc ==="
