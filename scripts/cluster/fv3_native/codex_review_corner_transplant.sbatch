#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_corner
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_corner_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW. The code under review was written by a DIFFERENT
model (GLM-5.2) and applied by me with three corrections; you and a third
reviewer are the check on both of us. Attack the code AND the corrections.

FIND IT: `git log --oneline -3`; the commit is "diag(fv3): transplant the
corner-diagonal cells the first elimination skipped". File:
scripts/validate/fv3_native/nh_metric_transplant.py, plus its runner
scripts/cluster/fv3_native/nh_metric_transplant.sbatch. The masking and
family tables it imports live in
scripts/validate/fv3_native/compare_gs_metrics.py.

WHY IT EXISTS. On the non-hydrostatic dry one-step case the port differs
from the pinned Fortran by 1.3003e-07 m/s of vertical velocity near cube
panel corners. Measured this session (job 9451331, seven controls
passed): the error does NOT scale with the flow -- two panels whose local
vertical-velocity amplitude differs by 14x carry the identical error; the
layer height carries the same footprint at the first acoustic sub-step;
and the six panels fall into exactly two classes. That is a geometry
signature.

A 2026-08-19 experiment had concluded the grid metrics were eliminated,
by transplanting the oracle's own runtime metric values into the port and
re-running. But `sentinel_mask` masks a cell when EITHER side is
sentinel, and the port's builder leaves 1e30 in exactly the
corner-diagonal halo cells -- so those were never transplanted. The new
`--widen-corners` arm imports wherever the ORACLE is real.

ATTACK IT. Cite file:line.

1. IS THE WIDENED SET THE RIGHT SET? `_sent_mask_single` decides what
   gets written. Check it against `sentinel_mask`/`sent_mag`/`sent_lo` in
   compare_gs_metrics.py -- in particular the RECIPROCAL families, whose
   sentinel is a TINY value (1e-30), not a large one. Does the widened
   mask handle those correctly, or does it write garbage into, or refuse,
   the wrong cells for that family class?
2. CAN THE ARM SILENTLY DO NOTHING? The refusal is on
   `total_widen_new == 0`. Trace how that counter is accumulated --
   including whether a family that raises, `continue`s, or is `skipped`
   still contributes, and whether the port-side mapping of the
   newly-covered mask (`_map_to_port`) can lose or duplicate cells. A
   nonzero count that comes from the wrong family would satisfy the
   refusal while the corner cells stayed untouched.
3. THE THREE CORRECTIONS I MADE TO THE AUTHORED CODE. Each is a claim:
   (a) the slot-assignment mask for sin_sg/cos_sg was left NARROW while
   only the write widens -- verify that is what the code now does, and
   that the assignment cannot be reached with widened data;
   (b) a running maximum seeded at -inf instead of nan -- check every
   accumulator of that kind in the file, not just the one I changed;
   (c) unknown CLI flags now raise. Did I get any of them wrong, and did
   I miss a fourth?
4. THE POST-TRANSPLANT EXACTNESS CHECK. It is re-evaluated over the
   widened set. Can it pass vacuously now -- e.g. because the mask it
   excuses grew? Is there any family where the check no longer covers
   what its comment says?
5. SIGN RESOLUTION. `resolve_sign` still runs on the both-sides-live set.
   Is that right, and is there a family where the widened cells could
   have needed the opposite sign?
6. DOWNSTREAM CONSUMERS. `da_min`/`da_max` are recomputed from the
   transplanted areas, and `ectx` dx6/dy6 are refreshed "where held".
   Does widening change what those see, and can the "where held"
   comparison now match cells it should not?
7. WOULD A NEGATIVE RESULT MEAN ANYTHING? If the widened arm leaves the
   error at 1.3003e-07, what exactly has been eliminated and what has
   not? Name the smallest additional thing that would still have to be
   true for "the metrics are not the cause" to be a sound conclusion.
8. ANY COMMENT OR DOCSTRING HERE THAT THE CODE DOES NOT ENFORCE.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_CORNER_DONE
