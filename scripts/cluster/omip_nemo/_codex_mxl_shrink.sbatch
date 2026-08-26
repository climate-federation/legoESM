#!/bin/bash
#SBATCH --job-name=cdx_mxl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_mxl_%j.out
#
# AUTH: do NOT copy ~/.codex/auth.json into a scratch CODEX_HOME (rotated
# refresh token -> exit 0 reviewing nothing).  unset CODEX_HOME.
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_mxl_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
"Read our TKE mixing-length code against NEMO's zdftke.F90 and find why our
equatorial thermocline viscosity is ~10x NEMO's.  Concrete, cite file:line.

THE MEASUREMENT (5-day-mean K, days 26-30, matched NEMO grid_W rec 5, cold
tongue 200-260E |lat|<=2, on the merged card):
  entrainment 65-105 m:  our avm 2.1e-5   NEMO avm 2.4e-6   (x8.6 too big)
  surface 5-65 m:        our avm ~ NEMO within 2x (intermittent)
Our carried TKE at 65-105 m is AT the floor (tke_background=1e-6); checked a
snapshot: tke 1.7e-5 @11m -> 1.0e-6 @57m and below.  kappaM_min=avmb=1.4e-6
(iwm molecular), so our 2.1e-5 is 15x ABOVE the floor = a real closure value,
not the floor.  So K_M = c_k*l_k*sqrt(e) with e=1e-6 and l_k~0.14m gives
2.1e-5; NEMO's 2.4e-6 implies l~0.01-0.02m, ~10x shorter.

THE CODE (branch fix/omip-gm-treguier-one-variable):
  packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py
    compute_mixing_lengths, tke_mxl_choice==3 branch (~line 690-800): builds
    l_int = sqrt(2e)/N at interior interfaces, then bounds it by lup (downward
    scan l(k)=min(l(k-1)+e3t(k-1), l_int)) and ldown (upward scan), l_k =
    max(min(lup,ldn), mxl_min).  Production runs choice 3, ln_mxl0 anchor on.
    compute_K_from_tke (~line 1552-1668): kappa_convention='veros_sqrte' =>
    K_M = c_k*l_k*sqrt(max(0,e)); prandtl_mode='richardson' path floors K_M at
    kappaM_min AFTER computing K_H.

QUESTIONS, in order:
1. THE MAIN ONE: by the formula both NEMO and legoESM should get l_int =
   sqrt(2*rn_emin)/N ~ 0.14 m at 65-105 m (same floored TKE 1e-6, same N^2 ~
   1e-4), and lup/ldown saturate at l_int far from both boundaries, so both
   should give ~0.14 m and ~2e-5 viscosity.  Yet NEMO's dumped avm is 10x
   smaller.  Read NEMO zdftke.F90's zmxlm construction (nn_mxl=2/3, the
   zmxlm/zmxld lup/ldown loops ~:658-690 upstream, DINO MY_SRC copy) and our
   compute_mixing_lengths side by side.  Where do they DIVERGE at a stratified
   interior interface far from both boundaries?  Specifically:
   (a) Does NEMO compute zmxlm from the RAW en or from MAX(en,rn_emin)?  If
       NEMO uses raw en (which drops BELOW rn_emin below the mixed layer)
       while we floor e before sqrt(2e)/N, our length stays at 0.14 m while
       NEMO's shrinks -- is that the bug?  Confirm from the Fortran whether en
       is floored before OR after the zmxlm computation.
   (b) Is our l_int = sqrt(2e)/N missing a bound NEMO applies to zmxlm (e.g.
       NEMO also caps zmxlm by the distance to the surface/bottom, or by a
       different e3w vs e3t)?
   (c) The amplitude: NEMO avm = rn_ediff*zmxlm*sqrt(en) [sqrt(en), not
       sqrt(2en)].  Our veros_sqrte uses sqrt(e) too -- confirm we are not
       carrying an extra sqrt(2) somewhere between l_int (which HAS sqrt(2e))
       and K_M.
2. Is our avm 2.1e-5 physically the closure at a STRATIFIED thermocline where
   NEMO's closure has correctly shut off to near-molecular?  I.e. is this a
   real over-mixing bug that would over-damp the equatorial undercurrent
   (measured 4x too weak)?
3. Anything in the recent ldown-seed fix (the cfg.mxl_min seed, the extra
   e3t(jpk) row logic) that affects the 65-105 m interior, not just the deep
   near-neutral columns it was written for?

Give a numbered list of concrete findings with file:line and, for the main
question, state whether there IS a code divergence from NEMO or whether the
10x is expected/explained.  If you cannot tell without NEMO's dumped zmxlm,
say exactly what to dump." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
