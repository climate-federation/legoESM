#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_nharea
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_nharea_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_nharea_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are AUTHORING the arm that tests the current prime suspect. Two
reviewers attack it afterwards. Your last three contributions each had a
defect that made the arm useless or wrong (a missing import; an inverted
write polarity; a port-side patch aimed at the wrong file reading a key
that does not exist). Every name you call must exist; every comment must
be enforced by code; every count checked, not asserted.

WHERE THE HUNT STANDS. A cubed-sphere dynamical core ported to NumPy
differs from its Fortran reference by 1.3003e-07 m/s of vertical
velocity at cells one step in from cube-panel corners. Measured, with
controls, over this campaign:

  * the error does not scale with the flow -- a geometry signature;
  * vertical velocity starts at exactly zero and is written only by a
    column-local solver, so the cause is in that solver's inputs;
  * the only input with a horizontal stencil is the layer height;
  * MEASURED YESTERDAY (job 9466634): the height enters its update
    routine at the parity floor on EVERY halo ring and in the interior
    (1.5e-11 against a field of 5.8e+03), and leaves it wrong in the
    COMPUTE WINDOW by 3.7e-05, with the halo untouched. So the halo
    exchange is exonerated and the UPDATE ROUTINE is the seat;
  * the maximum sits at compute cells 0-1 in from the panel corners on
    all six faces, in two face classes (3.67e-05 on four faces,
    1.90e-05 on two);
  * the magnitudes close: 3.7e-05 m of height over one 240 s sub-step
    is 1.53e-07 m/s against the 1.30e-07 m/s of vertical velocity.

THE SUSPECT, and it was predicted in writing by the port's own docstring
before any of this was measured. The height update is NOT handed the
gridstruct's area. It is handed a SEPARATE cached array built by
``nh_exchanged_area6`` (pasted below), which takes the port's own area,
whose corner-diagonal halo cells hold a sentinel, and fills those cells
with a Lagrange corner-region fill of its own. Its docstring says the
substitution is PLAUSIBLE and that "any disagreement will localise to
corner-adjacent stencils in the parity map". That is exactly what the
measurement found.

The reference's own area holds real, mirror-symmetric values at those
cells, and they are dumped: tag ``M_AREA`` (and ``M_RAREA``) in
/burg-archive/glab/users/pg2328/fv3_duo_gaps/retro_gsmetrics/run_c48,
one ``extchain_t<tile>.dat`` + ``.mf`` manifest per tile. Note that the
existing metric-transplant experiment CANNOT have tested this: it writes
``gs6[t]["area"]`` and preserves the port's sentinels there, and this
array is then derived from it and filled by the port's own routine.

WHAT TO WRITE: an opt-in arm that replaces the cached array's
corner-region values with the reference's, at exactly the cells the port
fills itself, and re-runs the existing height probe so the post-update
interior number can be compared against the measured 3.7e-05.

Requirements:
 1. Opt-in, its own flag; default behaviour byte-identical.
 2. Write ONLY the cells the port's own fill supplied -- i.e. those the
    unexchanged area marks sentinel. Count them, per face, and REFUSE
    with a non-zero exit if that count is zero, or if none of them lies
    inside the window the update routine actually reads. A cell nothing
    reads cannot move the answer, and an arm that writes only such cells
    returns "unchanged" by construction; that mistake has already been
    made once in this campaign and cost a run.
 3. Also write the reciprocal array consistently -- it is cached
    alongside and derived from the same values.
 4. Report the pre-write disagreement at those cells, per face, before
    anything runs.
 5. PRE-REGISTER the reading against the measured baseline (post-update
    interior 3.67e-05 on faces 1/2/4/5 and 1.90e-05 on 3/6): what value
    confirms this substitution is the seat, what refutes it, and what
    would establish neither.
 6. State what the arm does not cover.

STYLE: comments explain WHY and cite reference file:line; no verdict
printed by a diagnostic; a degenerate branch raises rather than
returning a plausible number.

OUTPUT: exact edits as a unified diff or complete functions. No prose
beyond what belongs in the code.

================ THE SUBSTITUTION AND ITS CONSUMER ================
HDR
sed -n '300,440p' packages/core/legoesm/core/fv3_native_dsw_tail_3d.py
echo "================ THE HEIGHT PROBE THIS ARM MUST RE-RUN ================"
cat scripts/validate/fv3_native/nh_halo_zh_parity.py
echo "================ HOW AN EXISTING ARM READS THE REFERENCE DUMPS ================"
sed -n '260,300p' scripts/validate/fv3_native/nh_metric_transplant.py
echo "================ THE MEASURED BASELINE ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|Jax plugin|for d in range|\^\^\^|RuntimeError: jaxlib" /burg-archive/glab/users/pg2328/fv3_duo_gaps/zh_halo_9466634.log | tail -32
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --code --max-tokens 60000 --out $G/nharea_patch_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_NHAREA_DONE
