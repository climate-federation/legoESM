#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_nhac
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_nhac_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_nhac_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are the SECOND adversarial reviewer. codex works the diff; you work
ARCHITECTURE, INVARIANTS and -- here especially -- WHETHER THE
CONCLUSION FOLLOWS FROM THE MEASUREMENT. Do not rely on memory of FV3;
everything needed is pasted. Mark anything you infer.

WHAT HAPPENED. A discrepancy between this port and its Fortran oracle
has been open since 2026-08-19: on the non-hydrostatic dry
configuration the worst one-step relative error is 6.6116e-04, which is
1.3003e-07 m/s of vertical velocity at model-top boundary cells.
Previously ELIMINATED as causes: the grid metrics (by a bitwise
transplant of the oracle's own), the physical constants, a vertical
velocity limiter, and the initial condition for layer thickness. Cause
unknown.

The step has two halves: an acoustic sub-step loop, then a vertical
remap. The new measurement compares the port's vertical velocity
against the oracle's AT THE BOUNDARY BETWEEN THEM -- an instrumented
Fortran build dumps its state there, and the port returns its own:

    pre-remap  relative error   9.3761e-04
    full-step  relative error   6.6116e-04
    ratio                       1.418

    per face, pre-remap: 6.85e-05  9.376e-04  4.23e-05
                         9.376e-04 9.376e-04  3.95e-04

STATED VERDICT: the discrepancy is already present when the acoustic
loop finishes, so it lives there, and the remap is exonerated.

WHAT I WANT FROM YOU:

A. DOES THE VERDICT FOLLOW? The pre-remap error is LARGER than the
   full-step error. Is "already there, and the remap slightly reduces
   it" the only reading of a 1.418 ratio? The two numbers come from the
   same relative-error function, whose denominator is the larger of the
   two fields' peaks -- so the denominators differ between the two
   comparisons. Could that alone produce 1.418? What would separate the
   readings?

B. IS ONE VERTICAL LEVEL ENOUGH? The dump is a single model level (the
   top one, which is where the discrepancy's maximum was previously
   located). The conclusion is about the whole field. What does that
   licence and what does it not?

C. THREE CONTROLS ARE CLAIMED and one of them caught a real error on
   the first attempt (the comparison had dropped a required grid
   transform and read 1.0071e+00 instead of 6.6116e-04). Are the three
   sufficient? What class of instrument defect would pass all three and
   still give a wrong verdict?

D. THE ORACLE SIDE IS A NEWLY BUILT BINARY -- a copy of the pinned
   source with a dump inserted, compiled without a preprocessor define
   that the other instrumented build carries. What has to be true for
   its output to be comparable to the certified oracle's, and what
   would silently break that?

E. GIVEN THE VERDICT, what is the NEXT discriminating measurement
   inside the acoustic loop, and is it cheap? The loop runs 8
   sub-steps; the instrument can dump per sub-step.

F. ANYTHING ASSERTED THAT NOTHING CHECKS.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so.

================ THE COMMITS ================
HDR
git log --oneline -8
git show $(git log --format=%H --grep="instrumented NH build" -1)
git show $(git log --format=%H --grep="probe to split the NH gap" -1)
git show $(git log --format=%H --grep="dropped the face map" -1)
echo "================ THE PROBE ================"
cat scripts/validate/fv3_native/nh_preremap_w_parity.py
echo "================ THE HELPERS IT REUSES ================"
sed -n '/^def port_window/,/^def rel/p' scripts/validate/fv3_native/full_step_oracle_parity.py
sed -n '/^def rel(/,/^def /p' scripts/validate/fv3_native/full_step_oracle_parity.py | head -60
echo "================ THE RUN OUTPUT ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|RuntimeError: jaxlib|Jax plugin|Traceback|File \"" /burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_prew_9448074.log | tail -24
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/review_nhac_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_NHAC_DONE
