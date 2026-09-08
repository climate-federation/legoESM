#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_nhsub
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_nhsub_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_nhsub_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are the SECOND adversarial reviewer. codex works the diff; you work
ARCHITECTURE, INVARIANTS, and whether an instrument can answer the
question it was built for. Do not rely on memory of this model;
everything needed is pasted below. Mark anything you infer rather than
read.

THE SITUATION. A cubed-sphere atmospheric dynamical core has been ported
from Fortran to Python, and one discrepancy against the Fortran
reference has stayed open since 2026-08-19: on the non-hydrostatic dry
configuration the worst one-step relative error is 6.6116e-04, which is
1.3003e-07 m/s of vertical velocity at model-top cells. Two things are
already established, each by its own measurement:

  * the error is fully present when the acoustic (fast-wave) loop ends,
    and the vertical remap that follows changes it by 0.008%;
  * it is specific to vertical velocity: temperature and layer mass are
    at the floor of what fp64 parity can resolve (2.3e-13, 3.0e-10).

So the error is born inside the acoustic loop, which runs EIGHT
sub-steps per model step. The new instrument reads the error at the end
of every sub-step, to distinguish

    full-size at sub-step 1        -> one STAGE of the sub-step is wrong
    growing across the eight       -> a COEFFICIENT is wrong

WHAT WAS BUILT. On the Fortran side, a second instrumented binary: a
copy of the pinned reference source with a dump inserted at the end of
the sub-step loop, AND the previously used end-of-loop dump kept. On the
Python side, the stepper returns one deep copy of its state per
sub-step, by RETURN rather than by callback -- the codebase forbids a
probe that re-runs the chain itself, since that would measure a
different program. The probe scores them against each other.

FIVE CONTROLS run before any number is reported, and the probe exits
non-zero on each failure:
  1. the initial condition's face-to-tile mapping is re-derived and must
     hit its own ~1e-14 floor;
  2. the Python side's full-step error must reproduce the established
     6.6116e-04, or this is not the configuration under investigation;
  3. the new binary's own restart file must be bitwise the certified
     one -- a freshly compiled binary is a different program until shown
     otherwise;
  4. the new binary's LAST sub-step dump must be bitwise its own
     end-of-loop dump, since nothing between those two points writes
     vertical velocity;
  5. consecutive sub-step dumps must differ, or the series is eight
     copies of one reading.

WHAT I WANT FROM YOU:

A. CAN THIS INSTRUMENT ANSWER ITS QUESTION? Suppose the true defect is a
   single wrong coefficient applied once per sub-step. Suppose instead
   it is one wrong stage. Write down what the eight-number series looks
   like in each case, and say whether the two are actually
   distinguishable from this data. If they are not, what extra reading
   separates them, and is it cheap?

B. THE SERIES IS SCORED AT ONE VERTICAL LEVEL (the one where the
   full-step error peaks) and reduced with a maximum over cells and
   faces. What can a max-over-cells series hide? In particular: the
   argument-maximum can move between sub-steps. Does the probe need to
   track WHERE the worst cell is, and why?

C. THE FIXED SCALE. Relative errors are formed with ONE denominator per
   field, held fixed across the eight sub-steps, because a per-reading
   denominator that itself grows would hide growth. Is that the right
   call? What does it distort?

D. CONTROL 4 IS THE NEW ONE and it is the only thing tying the two
   capture points together. It rests on a claim about the reference
   source: that nothing between the sub-step loop's end and the
   routine's return writes vertical velocity. If that claim were false,
   what happens? Is there a cheaper or stronger way to authenticate a
   newly inserted dump?

E. ARE THE FIVE CONTROLS SUFFICIENT? Name a class of instrument defect
   that passes all five and still yields a wrong reading of the series.

F. THE PYTHON STEPPER NOW COPIES ITS ENTIRE STATE ONCE PER SUB-STEP when
   asked. Any invariant that breaks -- aliasing, mutation-after-capture,
   interaction with the existing end-of-loop capture, behaviour when the
   outer loop runs more than once?

G. ANYTHING ASSERTED THAT NOTHING CHECKS.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so.

================ THE COMMIT ================
HDR
git log --oneline -3
git show $(git log --format=%H --grep="dump w per acoustic sub-step" -1)
echo "================ THE PROBE ================"
cat scripts/validate/fv3_native/nh_substep_w_parity.py
echo "================ THE SHARED READER AND HELPERS ================"
sed -n '/^def read_dump/,/^def main/p' scripts/validate/fv3_native/nh_preremap_w_parity.py
sed -n '/^def rel(/,/^def /p' scripts/validate/fv3_native/full_step_oracle_parity.py | head -60
echo "================ THE ORACLE BUILD ================"
cat scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch
echo "================ FORTRAN: LOOP END TO ROUTINE RETURN ================"
sed -n '1725,1760p' /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90
echo "================ PRIOR MEASUREMENT (the pre-remap probe's run) ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|RuntimeError: jaxlib|Jax plugin|Traceback|File \"" /burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_prew_9448215.log | tail -30
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/review_nhsub_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_NHSUB_DONE
