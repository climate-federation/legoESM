#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_damp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_damp_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
P=$G/prompt_damp_${SLURM_JOB_ID}.txt
mkdir -p $G
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are AUTHORING a change. Two reviewers will attack it afterwards, and
the last thing you wrote for this campaign had a missing import that made
it dead on arrival plus a premise that measurement refuted. Write this
one to survive that: no claim in a comment the code does not enforce, no
scope word ("every", "all", "eliminated") that is not checked, and every
name you call must exist in the file's imports.

THE STATE OF THE INVESTIGATION. A cubed-sphere dynamical core ported from
Fortran to NumPy differs from its reference by 1.3003e-07 m/s of vertical
velocity, at cells one step in from cube-panel corners. Measured this
session, all with passing instrument controls:

  * the error does NOT scale with the flow: two panels whose local
    vertical-velocity amplitude differs 14x carry the identical error;
  * the layer height carries the same spatial footprint at the FIRST
    acoustic sub-step, so height and vertical velocity are siblings of
    one cause rather than one driving the other;
  * the six panels split into exactly two classes.

That is a geometry signature. Two hypotheses were then tested and both
came back negative in an informative way:

  1. An earlier experiment transplanted the reference's own runtime grid
     metrics into the port and re-ran; the error did not move
     (6.611558e-04 -> 6.611559e-04).
  2. That experiment was suspected of skipping the corner cells, because
     its mask drops a cell when EITHER side looks like a sentinel. An
     opt-in "widen" arm was built to cover cells where the port is
     sentinel and the reference is real. It found ZERO such cells and
     refused rather than report "unchanged". The sentinel population is
     on BOTH sides -- about 14000 cells across the two-dimensional
     families.

So a transplant cannot reach those cells at all, and a COMPARISON was run
instead. Result, and this is what the new work is for:

  * `del6_u`, `del6_v`, `divg_u`, `divg_v` -- the four damping-coefficient
    arrays -- DIFFER between port and reference at 79 to 99 of the 108
    both-sides-sentinel cells per face, 7 to 12 of them inside corner
    regions, with absolute differences of 2.5e-06 to 7.0e-06;
  * every other family (all the trigonometric ones) is BITWISE IDENTICAL
    at those cells;
  * `area_c` and `rarea_c` differ at exactly one cell per face, which is
    the deliberately poisoned slot.

The values there are around 1e8, so the RELATIVE disagreement is about
1e-14. Whether that can produce 1.3e-07 m/s is unknown and must not be
assumed either way.

WHAT TO WRITE. A new opt-in arm of the existing transplant experiment
that imports the reference's values at the BOTH-SIDES-SENTINEL cells, for
those four damping families ONLY, and re-runs the same gate. Requirements:

 1. Opt-in via its own flag, separate from the existing `--widen-corners`.
    The default arms keep their exact behaviour and their JSON tags.
 2. Restricted to the four families by an explicit named set. Widening
    everything would confound the test; say so in a comment.
 3. It must be impossible for the arm to silently do nothing: count the
    cells it writes, per family and per face, split by whether the cell
    is in a corner region, and REFUSE with a non-zero exit if the corner
    count is zero. The previous arm's global counter was the defect that
    review caught -- a nonzero total from an unrelated family let a no-op
    pass.
 4. Report, before the physics re-runs, the pre-transplant disagreement
    at the cells it is about to write, per family, so a null result can
    be read against a known non-zero starting disagreement.
 5. A post-transplant exactness check over the written set.
 6. PRE-REGISTER the reading in the docstring: what value of the gate's
    output confirms that these coefficients are the seat of the error,
    what value refutes it, and what neither would establish. Include the
    baseline number (6.611558e-04 relative, 1.3003e-07 m/s absolute) so
    the comparison is against a stated prior, not a remembered one.
 7. State what the arm still does not cover.

CONTEXT YOU NEED: the existing script is pasted below, including the
`--widen-corners` arm you wrote last round and the corrections that were
applied to it. `region_masks`, `sent_mag`, `sent_lo`, `sentinel_mask`,
`FAMILIES`, `SWAP_PARTNER`, `_ODD`, `resolve_sign` are imported at the
top from compare_gs_metrics; `op_scalar` and the face map come from
compare_extchain_oracle. Note that `del6_u`/`divg_u` swap with their `_v`
partner under a transposing face map -- `SWAP_PARTNER` already encodes
that and the existing loop uses it.

STYLE: comments explain WHY and cite reference file:line where a rule
comes from; a diagnostic never prints its own verdict; a degenerate
branch raises rather than returning a plausible number.

OUTPUT: the exact edits as a unified diff, or complete replacement
functions with enough context to apply unambiguously. No prose beyond
what belongs in the code.

================ THE SCRIPT ================
HDR
cat scripts/validate/fv3_native/nh_metric_transplant.py
echo "================ THE COMPARISON THAT MOTIVATES IT ================"
cat scripts/validate/fv3_native/corner_metric_compare.py
echo "================ ITS MEASURED OUTPUT ================"
grep -vE "cuInit|xla_cuda12|_check_cuda|plugin_module|discover_pjrt|jax/_src|Jax plugin|for d in range|\^\^\^|RuntimeError: jaxlib|Traceback" /burg-archive/glab/users/pg2328/fv3_duo_gaps/corner_mc_9455470.log | head -80
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --code --max-tokens 60000 --out $G/damp_patch_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_DAMP_DONE
