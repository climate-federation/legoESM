#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_nhac
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_nhac_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a DIAGNOSTIC and the physics verdict it produced.
This repo reviews instruments, not just model code, because a bug in an
instrument manufactures a confident wrong conclusion -- and this one
just closed a question that had been open since 2026-08-19.

FIND THE COMMITS: `git log --oneline -8`; the range starts at
"diag(fv3): instrumented NH build to split the 6.6e-4 gap" and runs to
HEAD.

THE CLAIM UNDER REVIEW. The non-hydrostatic dry arm scores 6.6116e-04
against the pinned Fortran, which is 1.3003e-07 m/s of `w` at k=0
boundary cells. Metrics (including a bitwise transplant), constants,
the w_limiter and the delz IC were all previously eliminated. The new
measurement:

    pre-remap  w rel   9.3761e-04   at face2 -> tile5
    full-step  w rel   6.6116e-04
    ratio pre/full     1.418

and the stated VERDICT is: the gap is already present when dyn_core
returns, so it is in the ACOUSTIC LOOP and the vertical remap of w
(kord_wz) is exonerated.

ATTACK THE INSTRUMENT FIRST. Cite file:line.

1. IS THE PORT SIDE THE RIGHT STATE? fv_dynamics_step gained
   `return_pre_remap`. Verify the capture point is where the Fortran
   dump is -- after dyn_core, before Lagrangian_to_Eulerian -- and on
   the SAME n_map iteration. Does anything mutate `state` between the
   capture and the remap? Is the copy deep enough that a later in-place
   remap cannot have altered what was captured?
2. IS THE ORACLE SIDE THE RIGHT STATE? Read
   scripts/cluster/fv3_native/build_nh_wdump_oracle.sbatch. The dump is
   injected by a Python patch into a COPY of the pinned tree. Verify the
   insertion point, that the patch cannot silently no-op, that the build
   really drops -DSW_DYNAMICS, and that the resulting binary is the same
   configuration as the certified NH oracle (constants, defines, deck).
   A dump from a differently-built binary would be a different program.
3. THE LAYOUTS, which is where the first attempt failed. read_dump
   returns (1, j, i) to match load_oracle's storage; the port side gets
   DIHEDRAL[nm]; oracle_ij handles the transpose. Verify the whole chain
   for BOTH the control and the measurement, including the padded-box
   offset arithmetic and which k level is being compared (the dump is
   level 1 of the Fortran, index 0 of the port -- is that the same
   level?).
4. CAN THE CONTROLS FAIL? Three are claimed: the IC face map at
   3.685e-14, the script reproducing 6.6116e-04, and the oracle's own
   pre-remap-vs-restart difference being non-zero. For each, name the
   defect it would catch and the defect it would NOT.
5. DOES THE VERDICT FOLLOW FROM THE NUMBER? pre-remap 9.3761e-04 is
   LARGER than the full-step 6.6116e-04. Is "already there before the
   remap" the only reading? What else produces a ratio of 1.418 --
   could a scale/normalisation difference between the two comparisons
   (rel()'s denominator differs when the fields differ) manufacture it?
   Is comparing ONE k level enough to conclude about a gap whose argmax
   is at k=0?
6. WHAT WOULD FALSIFY THE VERDICT, and is that test cheap?
7. ANY COMMENT, DOCSTRING OR STATE.md CLAIM ADDED HERE THAT IS NOT TRUE.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_NHAC_DONE
