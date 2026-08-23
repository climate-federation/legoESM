#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_areafix
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_areafix_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_areafix_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
Review a PROPOSED FIX and the reasoning behind it. Not a diff -- a
choice between two repairs. Attack the choice, the evidence for it, and
what it would break. Everything you need is pasted; mark anything you
infer.

THE DEFECT, measured this session with controls passing. A cubed-sphere
dynamical core ported from Fortran to NumPy differed from its reference
by 1.3003e-07 m/s of vertical velocity (6.6116e-04 relative) at cells one
step in from cube-panel corners, open since 19 August. Localisation:

  * the error does not scale with the flow -- geometry, not dynamics;
  * vertical velocity starts at exactly zero and is written only by a
    column-local solver, so the cause is in that solver's inputs;
  * the layer height enters its update routine at the parity floor on
    every halo ring and leaves it wrong in the compute window near the
    panel corners;
  * that update routine is NOT handed the grid structure's area. It is
    handed a cached copy built by a helper that takes the port's area,
    runs its OWN exchange over it, and fills the corner region itself;
  * the grid structure's area agrees with the reference at the parity
    floor EVERYWHERE -- interior, halo edge strips, and corner wedges
    alike (7.2e-02 on values of 3.5e+10, about two parts in a trillion);
  * the cached copy that kernel receives is off by 4.1e+09 in the halo
    edge strips (twelve percent) and 1.4e+09 in the corner wedges (four
    percent), on all six faces, all thirty-six corner cells per face.

Substituting the reference's values at exactly the disagreeing cells:
the height error at its update falls from 3.7e-05 to 1.6e-11 (the parity
floor) on all six faces and both interfaces, and the worst one-step
error over all faces and fields falls from 6.6116e-04 to 1.4778e-06 --
a factor of 447. Both readings were pre-registered.

THE HELPER'S OWN JUSTIFICATION IS VOID. Its docstring says the
single-tile grid structure "leaves BIG_NUMBER sentinels in the
corner-diagonal halo cells of area" and that the kernel reads them, so
the six-face integration "owns supplying the analog". Measured: that
array holds a real value (2.55e+10) at the corner-diagonal cell, against
an interior median of 3.56e+10. The sentinel claim is false today,
whatever it was when written.

MY PROPOSED FIX, which is what you are attacking: DELETE the extra
exchange and hand the kernel the grid structure's area directly. Reasons:
the grid structure's array is already correct in the halo to the parity
floor, so the exchange cannot add information and demonstrably subtracts
it; the stated reason for the exchange is refuted; and deletion is
smaller and cheaper than repairing an exchange whose output must then be
proven equal to what the untouched array already holds.

THE ALTERNATIVE: keep the exchange and repair it so it reproduces the
reference's halo values.

WHAT I WANT:

A. IS THE DELETION RIGHT? What would it break that the measurement does
   not cover? The comparison scored the area against the reference on
   ONE deck at one resolution, with one halo width, on a bounded
   (duogrid) domain. Name the configuration in which the untouched array
   would NOT be correct in the halo and the exchange would be doing real
   work -- and say how to check cheaply whether that configuration
   exists in this repo.

B. WAS THE EXCHANGE EVER RIGHT? A helper does not get written for no
   reason. What is the most likely history that produces this -- and
   does that history imply other arrays got the same treatment? The same
   helper also caches a reciprocal derived from the exchanged values.
   Where else might this pattern be, and how would I find it by grep
   rather than by reasoning?

C. THE RESIDUAL. After the substitution the worst one-step error is
   1.4778e-06, led by vertical velocity, against 1.19e-09 on the
   hydrostatic arm of the same gate. Three orders. Is that consistent
   with "one defect remains", with "the substitution is imperfect at
   cells it did not write", or with something else? What is the cheapest
   measurement that distinguishes them?

D. WHAT WOULD MAKE THE FIX BELIEVABLE beyond the gate number? The
   transplant proves the VALUES are wrong. A deletion is a different
   change and could pass the gate for a different reason. Name the check
   that would catch that.

E. ANYTHING IN MY REASONING THAT IS AN INFERENCE DRESSED AS A
   MEASUREMENT.

Output a recommendation (delete / repair / neither, with the reason),
then MAJOR / MINOR concerns with the concrete failure scenario for each.

================ THE HELPER AND ITS CONSUMER ================
HDR
sed -n '300,440p' packages/core/legoesm/core/fv3_native_dsw_tail_3d.py
echo "================ THE MEASUREMENT ================"
cat scripts/validate/fv3_native/nh_area_compare.py
echo "================ ITS OUTPUT ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|Jax plugin|for d in range|\^\^\^|RuntimeError: jaxlib|Traceback" /burg-archive/glab/users/pg2328/fv3_duo_gaps/area_cmp_9466872.log | tail -24
echo "================ THE FULL-STEP GATE AFTER THE SUBSTITUTION ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|Jax plugin|for d in range|\^\^\^|RuntimeError: jaxlib|Traceback" /burg-archive/glab/users/pg2328/fv3_duo_gaps/corner_area_9466902.log | tail -12
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/areafix_review_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_AREAFIX_DONE
