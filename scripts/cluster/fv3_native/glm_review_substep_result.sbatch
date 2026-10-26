#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_subres
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_subres_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_subres_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are the SECOND adversarial reviewer, and you reviewed this instrument
once already -- your M2 (nothing authenticated the hand-rolled mapping the
series is read through) became control 7 and it passed. THIS review is
about the MEASUREMENT and the MECHANISM I am inferring from it, not the
code. Do not rely on memory of this model; everything needed is pasted.
Mark anything you infer rather than read.

THE SETTING. A cubed-sphere atmospheric dynamical core ported from
Fortran to Python. One discrepancy has been open since 2026-08-19: on the
non-hydrostatic dry configuration the worst one-step relative error is
6.6116e-04, which is 1.3003e-07 m/s of vertical velocity at cells on the
boundaries between cube panels. Already eliminated by their own
measurements: the grid metrics (by transplanting the reference's own,
bitwise), the physical constants, a vertical-velocity limiter, and the
initial condition for layer thickness. Already established: the error is
fully present when the fast-wave (acoustic) loop ends, the vertical remap
that follows changes it by 0.008%, and temperature and layer mass are at
the floor of what double precision can resolve at that point.

The fast-wave loop runs EIGHT sub-steps per model step. The new
measurement reads the error at the end of every one of them.

ALL SEVEN CONTROLS PASSED: the initial condition's face mapping at
3.685e-14; the full-step error reproduced at 6.6116e-04; the instrumented
binary's own restart BITWISE IDENTICAL to the certified one on every
prognostic; the instrumented binary's last sub-step dump bitwise its own
end-of-loop dump; consecutive sub-step dumps differing; the Python side's
last sub-step capture equal to its own end-of-loop capture; and the last
sub-step reproducing the previously established 1.3003e-07.

THE SERIES (top model level; largest absolute difference over all cells
and all six panels; one fixed scale per field):

  sub-step   vertical velocity   temperature    layer mass
     1        1.2952e-07          1.6342e-13     2.5466e-11
     2        1.2980e-07          6.7146e-13     6.5187e-07
     3        1.2965e-07          1.2452e-12     1.2198e-06
     4        1.2970e-07          1.7124e-12     1.6860e-06
     5        1.2977e-07          2.0997e-12     2.0996e-06
     6        1.2986e-07          2.3999e-12     2.4521e-06
     7        1.2995e-07          2.6255e-12     2.7576e-06
     8        1.3003e-07          2.8102e-12     3.0198e-06

Every vertical-velocity difference is NEGATIVE (port minus reference).
Every layer-mass difference is POSITIVE. The worst vertical-velocity cell
moves between panels and between the index pairs (0,46), (46,0), (1,0),
(0,1) -- all of them panel-boundary cells -- while its magnitude barely
changes.

FIXED-CELL HISTORIES (the cell that is worst at sub-step 8, read at every
sub-step, so these are one cell's history rather than a moving maximum):

  vertical velocity: -1.295e-07 -1.298e-07 -1.297e-07 -1.297e-07
                     -1.298e-07 -1.299e-07 -1.299e-07 -1.300e-07
  temperature:       +1.776e-15 -6.715e-13 -1.245e-12 -1.712e-12
                     -2.100e-12 -2.400e-12 -2.625e-12 -2.810e-12
  layer mass:        +3.638e-12 +6.467e-07 +1.220e-06 +1.686e-06
                     +2.100e-06 +2.452e-06 +2.758e-06 +3.020e-06

MY READING, which is what you are attacking:
  (a) the vertical-velocity error is COMPLETE after one sub-step and
      does not grow;
  (b) temperature and layer mass start at nothing and accumulate, so
      they are DOWNSTREAM of the vertical-velocity error rather than
      independent sources;
  (c) therefore something inside a SINGLE sub-step computes vertical
      velocity wrong immediately, with a consistent sign, at panel
      boundaries -- this is not an error accumulating over eight
      applications of a slightly wrong coefficient.

WHAT I WANT FROM YOU:

A. IS (a) THE RIGHT DESCRIPTION? The value rises 0.4% across the eight
   sub-steps, monotonically after sub-step 3. Is "does not grow" a fair
   summary, or is "a large first-step jump PLUS a slow drift" a
   different signature I am flattening into one? What physical or
   numerical process gives exactly that shape, and what would separate
   the two readings?

B. IS (b) SOUND? Layer mass jumps four orders of magnitude between
   sub-steps 1 and 2 while vertical velocity does not move at all. If
   layer mass were simply being driven by a constant vertical-velocity
   error, what should its history look like -- and does the observed one
   match? Consider that at sub-step 1 both temperature and layer mass
   sit at rounding level, which the vertical velocity does not.

C. THE MOVING MAXIMUM. The worst cell changes panel and index between
   sub-steps while the magnitude barely moves. Does that indicate many
   cells carrying nearly identical errors -- a systematic boundary
   effect -- or one error moving? What does the fixed-cell history
   settle, and what does it leave open?

D. THE SIGNS. Vertical velocity is negative at every sub-step and every
   worst cell; layer mass is positive at every sub-step. What class of
   defect produces a fixed sign, and which candidate causes does that
   rule out?

E. GIVEN ALL OF IT, what is the SINGLE most likely mechanism, and what
   is the cheapest measurement that would refute it? The obvious next
   step is capturing between the stages inside one sub-step; say whether
   something cheaper discriminates first, and which stage boundaries
   carry the most information.

F. ANYTHING IN THE NUMBERS INCONSISTENT with all seven controls holding.

Output MAJOR / MINOR with the concrete failure scenario and the smallest
fix, plus a ranked mechanism list with what would falsify each. Where you
find nothing, say so.

================ THE PROBE ================
HDR
cat scripts/validate/fv3_native/nh_substep_w_parity.py
echo "================ THE FULL RUN OUTPUT ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|RuntimeError: jaxlib|Jax plugin|Traceback|File \"|for d in range|\^\^\^" /burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_wsubp_9450591.log
echo "================ THE SUB-STEP BODY (pinned dyn_core.F90 :339-1736, stage calls only) ================"
sed -n '339,1736p' /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90 | grep -nE "^ *call [a-z]|^ *if *\(" | head -120
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/review_subres_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_SUBRES_DONE
