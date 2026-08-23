#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_subres
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_subres_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW OF A CONCLUSION, not of a diff. You reviewed this
instrument once already (job 9450542) and its BLOCKER and two MAJORs were
applied; the probe now runs seven controls and prints no verdict. This
review is about the NUMBER it produced and the reading I am putting on it.

FILES: scripts/validate/fv3_native/nh_substep_w_parity.py,
scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch,
packages/core/legoesm/core/fv3_native_{acoustic_3d,dynamics}.py.
`git log --oneline -6`.

THE RUN (job 9450591, all seven controls passed: IC face map 3.685e-14;
full-step w residual reproduced at 6.6116e-04; instrumented-vs-certified
restart BITWISE IDENTICAL on every field; oracle sub-step 8 == its own
pre-remap dump, |dw|max exactly 0.0; consecutive sub-steps differ; port
sub-step 8 == port pre-remap on all fields; anchor 1.3003e-07 hit).

Level k=0, |d|max over cells and faces, one fixed scale per field:

  it     w              pt             delp
   1   1.2952e-07     1.6342e-13     2.5466e-11
   2   1.2980e-07     6.7146e-13     6.5187e-07
   3   1.2965e-07     1.2452e-12     1.2198e-06
   4   1.2970e-07     1.7124e-12     1.6860e-06
   5   1.2977e-07     2.0997e-12     2.0996e-06
   6   1.2986e-07     2.3999e-12     2.4521e-06
   7   1.2995e-07     2.6255e-12     2.7576e-06
   8   1.3003e-07     2.8102e-12     3.0198e-06

Every w reading is NEGATIVE (port minus oracle). The worst w cell moves
between faces and between (i, j) = (0, 46), (46, 0), (1, 0), (0, 1) --
all panel-boundary cells. Fixed-cell history at the final worst cell
(face2->tile5, i=0, j=1): -1.295e-07, -1.298e-07, -1.297e-07, -1.297e-07,
-1.298e-07, -1.299e-07, -1.299e-07, -1.300e-07.

pt's and delp's fixed-cell histories both start at rounding level
(+1.8e-15, +3.6e-12) and ramp monotonically.

MY READING, which is what you are attacking:
  (a) the w error is COMPLETE after ONE sub-step and does not grow;
  (b) pt and delp start at nothing and accumulate, so they are
      DOWNSTREAM of w rather than independent sources;
  (c) therefore some computation inside a single sub-step produces w
      wrong immediately, with a consistent sign, at panel boundaries --
      it is not an error accumulating over eight applications.

ATTACK IT:

1. IS (a) SOUND? w moves 1.2952e-07 -> 1.3003e-07 across eight
   sub-steps, a 0.4% rise. Is "does not grow" the right description, or
   is a slow drift on top of a large first-step jump a DIFFERENT
   signature that I am flattening? What process would give exactly this
   shape?
2. IS (b) SOUND? Name what would make pt and delp ramp INDEPENDENTLY of
   w -- and say whether their ramp rates are consistent with being
   driven by a constant w error. delp goes from 2.5e-11 to 6.5e-07
   between sub-steps 1 and 2, a jump of four orders; w does not. Does
   that pattern actually support "downstream", or does it need its own
   explanation?
3. THE MOVING ARGMAX. The worst w cell changes face and index between
   sub-steps while the magnitude barely moves. Does that mean many cells
   carry nearly the same error (a systematic boundary effect), or that
   one error is moving? Which does the fixed-cell history settle, and
   what does it leave open?
4. THE SIGN. Every w difference is negative and every delp difference is
   positive. What class of defect produces a fixed sign, and what class
   does not?
5. WHAT CHEAP MEASUREMENT would falsify (c)? The obvious next cut is
   dumping between the stages inside one sub-step. Given the loop's
   structure in the pinned dyn_core.F90, which stage boundaries are
   worth dumping at, in what order of expected information, and is there
   anything cheaper that would discriminate first? Cite file:line.
6. ANYTHING IN THE RUN OUTPUT INCONSISTENT WITH THE CONTROLS PASSING --
   a number that should not be possible if all seven really hold.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_SUBRES_DONE
