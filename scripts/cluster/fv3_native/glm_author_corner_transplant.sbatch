#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_corner
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_corner_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_corner_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are AUTHORING a change. Two other reviewers will attack it after, so
write it to survive that: no silent fallbacks, no claim in a comment that
the code does not enforce, and every scope boundary stated.

THE SITUATION. A cubed-sphere atmospheric dynamical core ported from
Fortran to NumPy. One discrepancy against the Fortran reference has been
open since 2026-08-19: on the non-hydrostatic dry one-step case the worst
error is 1.3003e-07 m/s of vertical velocity at cells near cube-panel
corners. A measurement this session established three things:

  * the error does not scale with the flow -- two panels whose local
    vertical-velocity amplitude differs by 14x carry the IDENTICAL error,
    1.2952e-07;
  * the layer-height field carries the same error with the SAME spatial
    footprint at the very first acoustic sub-step (same count of cells
    above half the peak, same four cells above 90%, same sign split);
  * the six panels fall into exactly two classes -- four at 1.2952e-07
    over 52 cells, two at 6.72e-08 over 108 cells.

That is a geometry signature, not a dynamics one.

THE HOLE TO CLOSE. An earlier experiment concluded that the grid metrics
were eliminated as a cause, by transplanting the reference's OWN runtime
metric values into the port and re-running. But the transplant loop masks
a cell if EITHER side exceeds a sentinel magnitude, and the port's
single-tile builder leaves a large sentinel value in exactly the
CORNER-DIAGONAL halo cells. So those cells were masked on the PORT side
and were never transplanted -- the elimination covers everything except
the region where the error lives.

The reference genuinely uses real values there: its dumped areas at the
corner-diagonal cells are finite and mirror-symmetric. And those cells
ARE read on this lane: the sixth-order damping operator at its configured
order fills its work array over the entire padded box including the
corner diagonals, and the corner-rotation repair that would normally fix
them is skipped on this grid variant on BOTH sides.

WHAT TO WRITE. An opt-in arm of the existing transplant script that also
imports the reference's values at the cells the current masking skips --
i.e. cells the port marks with a sentinel but the reference does not.
Requirements, all of them:

  1. OPT-IN, off by default, so the existing arm's numbers stay
     reproducible. A command-line flag.
  2. It must be IMPOSSIBLE for the arm to silently do nothing: count the
     cells it newly transplants, per family and in total, and REFUSE
     (exit non-zero) if that count is zero -- a no-op arm that reports
     "unchanged" would read as an elimination and would be the second
     time this exact mistake is made.
  3. It must refuse cells where the REFERENCE side is itself sentinel or
     non-finite: those carry no information and writing them into the
     port would inject garbage.
  4. The post-transplant check that already exists (the transplanted
     values must equal the reference exactly) must still hold, now over
     the widened set.
  5. Report, before the physics is re-run, what the pre-transplant
     disagreement WAS at the newly covered cells, per family -- if the
     port and the reference already agreed there, the arm cannot move
     anything and that must be visible as a number, not inferred.
  6. State in the docstring what this arm still does NOT cover.

STYLE OF THIS REPO, follow it: comments explain WHY and cite the
reference file:line where a rule comes from; a diagnostic never prints
its own verdict; a degenerate branch returns NaN or raises rather than a
plausible number; scope boundaries are stated rather than implied.

OUTPUT: the exact edits as a unified diff against the file below, or as
complete replacement functions with enough context to apply them
unambiguously. No prose beyond what belongs in the code. If the existing
structure makes a requirement awkward, say so in one line and propose the
smallest restructuring.

================ THE TRANSPLANT SCRIPT ================
HDR
cat scripts/validate/fv3_native/nh_metric_transplant.py
echo "================ THE MASKING AND FAMILY TABLE IT IMPORTS ================"
sed -n '40,200p' scripts/validate/fv3_native/compare_gs_metrics.py
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --code --max-tokens 60000 --out $G/corner_patch_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_CORNER_DONE
