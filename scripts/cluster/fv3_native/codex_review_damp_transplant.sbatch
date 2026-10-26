#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_damp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_damp_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW. The code was written by GLM-5.2 and applied by Claude
with two corrections; you and a third reviewer are the check on both.
Your last review of this same file found a missing import that made the
whole arm dead on arrival, and a premise that measurement then refuted --
so assume nothing here is verified until you verify it.

FIND IT: `git log --oneline -3`; the commit is "diag(fv3): transplant the
damping coefficients at the cells nothing could reach". The function is
`transplant_sentinel_damping` in
scripts/validate/fv3_native/nh_metric_transplant.py. Its motivating
measurement is scripts/validate/fv3_native/corner_metric_compare.py.

WHERE THE INVESTIGATION STANDS, all measured this session with passing
controls. The port differs from the pinned Fortran by 1.3003e-07 m/s of
vertical velocity at cells one step in from cube-panel corners. The error
does not scale with the flow (two panels differing 14x in local amplitude
carry the identical error); the layer height shows the same footprint at
the first acoustic sub-step; the panels split into two classes. A metric
transplant left the error unmoved. A widened arm meant to cover the
corner cells found ZERO cells to cover and refused -- the sentinel values
are on BOTH sides. A comparison then found `del6_u`/`del6_v`/`divg_u`/
`divg_v` disagreeing at 79-99 of 108 both-sides-sentinel cells per face
(7-12 inside corner regions, |d| 2.5e-06 to 7.0e-06 on values ~1e8, i.e.
~1e-14 relative), with every trigonometric family bitwise identical
there. This arm transplants exactly those cells for exactly those four
families.

ATTACK IT. Cite file:line.

1. THE WRITE SET. It is spelled as both-sides sentinel via
   `_sent_mask_single` on each side. Is that the set the comparison
   measured? Check `corner_metric_compare.py`'s own mask and say whether
   the two agree cell-for-cell, including the reciprocal and trig
   threshold classes and the `SWAP_PARTNER` handling under a transposing
   face map. If they differ, the arm is not testing what motivated it.
2. THE WRITE ITSELF. The applier reports fixing an inverted polarity.
   Verify the direction now written is the oracle's value at the target
   cells and the port's elsewhere, and that `_map_to_port` is the right
   inverse for BOTH the data and the boolean mask.
3. ORDERING. The arm runs after `transplant_metrics` inside the same
   context wrapper. Can the base arm have already changed any cell this
   arm then writes, or vice versa -- and does the claim "the only
   difference from the transplanted arm is these four families" survive?
4. CAN IT SILENTLY DO NOTHING, OR SILENTLY DO TOO MUCH? Trace the corner
   counter and the per-face refusals. Is there a path where the arm
   writes zero cells for one family or face and still completes? Is
   there a path where it writes cells OUTSIDE the four families?
5. THE PRE-REGISTERED READING in the docstring. Is it actually
   falsifiable as written, and is the "confirms" criterion (w becomes
   bitwise identical on all six faces) achievable in principle given
   what else is known to differ? If it is not achievable, the arm can
   only ever refute, and the docstring should say so.
6. THE DOWNSTREAM. `da_min`/`da_max` and the ectx dx6/dy6 refresh run
   inside `transplant_metrics`, BEFORE this arm writes. Does anything
   cached from the damping arrays escape the transplant?
7. IS THE EXPERIMENT WORTH BELIEVING AT ALL? A ~1e-14 relative
   difference in a damping coefficient against a 2.7e-05 relative error
   in w. State the smallest argument that would make a null result here
   informative rather than merely unsurprising.
8. ANY COMMENT OR DOCSTRING THE CODE DOES NOT ENFORCE.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_DAMP_DONE
