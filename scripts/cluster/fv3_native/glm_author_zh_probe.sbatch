#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_zh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_zh_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_zh_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are AUTHORING an instrument. Two reviewers attack it afterwards. Your
last two contributions here each had a defect that made the arm useless
(a missing import; an inverted write polarity), and one had a premise
measurement refuted. Write this one so neither can happen: every name you
call must exist in the file you are editing or in its imports, every
comment must be enforced by code, and every count must be checked rather
than asserted.

WHERE THE INVESTIGATION IS. A cubed-sphere dynamical core ported from
Fortran to NumPy differs from its reference by 1.3003e-07 m/s of vertical
velocity, at cells one step in from cube-panel corners, on a
non-hydrostatic dry one-step case. Established this session, each with
passing controls:

  * the error does not scale with the flow (two panels differing 14x in
    local amplitude carry the identical error) -- a geometry signature;
  * it is already full size at the END of the FIRST acoustic sub-step;
  * the layer height carries the same spatial footprint at that same
    sub-step, and neither field is the time-integral of the other, so
    they are siblings of one cause;
  * every grid-metric family has now been transplanted from the
    reference or shown identical, and the four that disagree do so only
    at cells no operator ever reads (verified by index bounds).

THE REMAINING ARGUMENT, and what this instrument must test. Vertical
velocity starts at EXACTLY zero on both sides. The only routine that can
make it non-zero is the Riemann solver, and that routine is strictly
COLUMN-LOCAL -- every array indexed (i,j) inside its j/i loops, no
horizontal stencil. So a corner-localised error coming out of a
column-local solver is a statement about its INPUTS. Its only input
carrying a horizontal stencil is the layer height, written by the
routine that advances it (``update_dz_d``, called from the acoustic loop
after the D-grid winds are updated), and that height is halo-filled by
the duo exchange before the loop.

That splits the remaining space in two, and one measurement separates
them:

  A. the height is ALREADY different on entry to the height-update
     routine  ->  the defect is in the halo FILL of the height;
  B. the height is identical on entry and different on exit ->  the
     defect is in the height-update OPERATOR itself.

WHAT TO WRITE: a probe that dumps the reference's layer height over the
FULL PADDED BOX -- halo rings included, which no instrument in this
campaign has done for this field -- at two points inside acoustic
sub-step 1: immediately BEFORE the height-update call, and immediately
AFTER it. Then score the port's own height at the same two points against
it, ring by ring, separating side strips from corner wedges.

Deliverables, both:
  1. the patch to the instrumented Fortran build script that adds the two
     dump points (an existing script already patches this tree and dumps
     other fields per sub-step -- extend that mechanism, do not invent a
     second one);
  2. the Python probe that reads them and scores the port.

Requirements:
  * HALOS ARE THE POINT. Every previous comparison in this campaign
    scored the compute window only. The scoring must cover every ring,
    and must report corner wedges separately from side strips, because
    the error under investigation is corner-localised.
  * The port side must come BY RETURN from the stepping code, never from
    a probe that re-runs the chain itself -- that would be a second
    implementation of the thing under test. An existing mechanism in the
    stepper already returns per-sub-step state; extend it rather than
    adding a callback.
  * Controls before any number is reported: the face map at its floor;
    the instrumented binary's own restart bitwise the certified one; and
    a check that the two dump points are NOT the same state (if the
    before and after dumps are identical the probe is not where it
    claims).
  * State what the probe does not cover.
  * Print no verdict. Print the ring-by-ring numbers and let the
    analysis read them.

STYLE: comments explain WHY and cite reference file:line; a degenerate
branch raises rather than returning a plausible number; no scope word
("every", "all", "identical") that the code does not check.

OUTPUT: the exact edits as a unified diff or complete files. No prose
beyond what belongs in the code.

================ THE EXISTING INSTRUMENTED BUILD (extend this) ================
HDR
cat scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch
echo "================ THE EXISTING PER-SUB-STEP PROBE (its controls are the model) ================"
cat scripts/validate/fv3_native/nh_substep_w_parity.py
echo "================ THE PORT'S ACOUSTIC SUB-STEP, showing the return mechanism ================"
sed -n '400,520p' packages/core/legoesm/core/fv3_native_acoustic_3d.py
echo "================ THE ORACLE ACOUSTIC LOOP AROUND THE HEIGHT UPDATE ================"
sed -n '1395,1420p' /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90
echo "================ RING/REGION MASK HELPER ALREADY IN THE REPO ================"
sed -n '/^def region_masks/,/^    return out/p' scripts/validate/fv3_native/compare_gs_metrics.py
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --code --max-tokens 60000 --out $G/zh_patch_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_ZH_DONE
