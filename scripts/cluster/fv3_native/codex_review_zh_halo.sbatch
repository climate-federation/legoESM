#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_zh
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_zh_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of an instrument BEFORE its number is believed. GLM
authored the design; Claude replaced its port-side half and wrote the
probe. Attack both. Your last two reviews of this campaign each found a
defect that made the arm useless -- assume the same standard.

FIND IT: `git log --oneline -4`. The commits are "diag(fv3): dump the
layer height, halos included, around its own update" and "diag(fv3):
score the layer height ring by ring, halos included". Files:
scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch (third
embedded patch), packages/core/legoesm/core/fv3_native_dsw_tail_3d.py
(the new before-hook), scripts/validate/fv3_native/nh_halo_zh_parity.py.

WHAT IT IS FOR. The port differs from the pinned Fortran by 1.3003e-07
m/s of vertical velocity at cells one step in from cube-panel corners.
Measured this session: the error does not scale with the flow (14x
amplitude spread, identical error), is full size at the end of acoustic
sub-step 1, and shares its footprint with the layer height there. Every
grid-metric family has been transplanted from the reference or shown
identical, and the four that disagree do so only at cells no operator
reads (verified by index bounds). Vertical velocity starts at exactly
zero and is written only by Riem_Solver3, which is strictly column-local,
so the remaining space is: the height is already wrong on entry to
update_dz_d (halo fill), or made wrong by it (operator). This dumps the
reference's height over the FULL PADDED BOX at both points and scores the
port's against it, region by region.

ATTACK IT. Cite file:line.

1. THE FORTRAN CAPTURE POINTS. Read the third patch in the build script.
   Verify both anchors are unique, that the inserted calls sit outside
   any preprocessor guard, that `zh` and `n_map` are in scope with the
   slices valid against the declaration at dyn_core.F90:299, and that
   the patch cannot silently no-op. Does dumping inside this loop change
   what the run computes?
2. THE PORT CAPTURE POINTS. The after-hook already existed; the
   before-hook is new. Are the two symmetric -- same array, same copy
   depth, nothing between the before-hook and the call that mutates the
   height? Does the hook fire once per face per sub-step, and does the
   probe's single sub-step 1 correspond to the reference's n_map=1 dump?
3. THE HOOK ARITY. The stepper calls stage_hook with two arguments for
   some stages and three for others. Check the probe's hook handles
   every call the stepper will make on this path, including any stage
   emitted between entry and update_dz_d.
4. THE REGION SPLIT. `region_masks` is imported from
   compare_dyncore_stages rather than re-derived. Verify its semantics
   match what the probe claims (corner = the diagonal wedges, width =
   NG) and that applying it to a PADDED plane means what the header
   line says. Does the DIHEDRAL applied to the port side commute with
   the region split?
5. THE CONTROLS. Four are claimed. For each, name the defect it catches
   and one it does NOT. In particular: can control 4's threshold
   (1e-4 relative on the before-interior) pass while the mapping is
   still wrong, and can control 3 pass on a dump that is stale rather
   than mislocated?
6. THE READING. The docstring says a difference in the BEFORE halo
   points at the exchange and one appearing only in AFTER points at the
   operator. Is that inference sound, or can the operator produce a
   difference that shows up in the BEFORE reading of a later sub-step?
   The probe only runs sub-step 1 -- say what that buys and what it
   forecloses.
7. ANY COMMENT OR DOCSTRING THE CODE DOES NOT ENFORCE, and any name
   called that is not imported or defined.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_ZH_DONE
