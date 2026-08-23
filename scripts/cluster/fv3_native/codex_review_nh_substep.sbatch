#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_nhsub
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_nhsub_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a DIAGNOSTIC INSTRUMENT, before its number is
believed. In this repo instruments get reviewed like model code, because
a bug in one manufactures a confident wrong physics conclusion -- this
campaign has retracted eight claims that way.

FIND THE COMMIT: `git log --oneline -3`; the one under review is
"diag(fv3): dump w per acoustic sub-step, both sides" plus its runner.

BACKGROUND, established and not under review here. The non-hydrostatic
dry arm scores 6.6116e-04 against the pinned Fortran, which is
1.3003e-07 m/s of `w` at k=0 cells. It was localised twice: it is
already present when dyn_core returns (the remap changes it 0.008% at
k=0), and it is w-specific (pt 2.3e-13, delp 3.0e-10 at the parity
floor). So it is born inside dyn_core's `do it=1,n_split` loop, which
runs 8 times. The new instrument measures the error at the end of EVERY
sub-step, to separate "one stage is wrong" (full-size at sub-step 1)
from "a coefficient is wrong" (accumulates over the eight).

WHAT THE DIFF DOES:
* packages/core/legoesm/core/fv3_native_acoustic_3d.py gains
  `substeps_out`, appended once per sub-step;
* packages/core/legoesm/core/fv3_native_dynamics.py gains
  `return_substeps`, which threads that list for the LAST n_map only and
  returns it;
* scripts/cluster/fv3_native/build_nh_wsubstep_oracle.sbatch is a NEW
  instrumented Fortran build: it copies the pinned oracle tree and
  patches BOTH fv_dynamics.F90 (the existing pre-remap dump) and
  dyn_core.F90 (the new per-sub-step dump);
* scripts/validate/fv3_native/nh_substep_w_parity.py is the probe;
* two tests, in tests/grids/test_fv3_native_acoustic_3d.py and
  tests/grids/test_fv3_native_dynamics.py.

ATTACK IT. Cite file:line.

1. CAPTURE POINT, PORT SIDE. Is `substeps_out.append(...)` at the same
   point in the sub-step as the Fortran dump (immediately before
   `enddo   ! time split loop`)? Is the copy deep enough that later
   in-place mutation of `state` cannot alter it? With k_split > 1, is
   passing the list only on `last_step` right, and can the length check
   `len(substeps) != n_split` ever be reached with a stale list?
2. CAPTURE POINT, ORACLE SIDE. Read the sbatch's embedded Python patch.
   Can it silently no-op? Is the inserted block inside any preprocessor
   guard that could exclude it? Is `it` in scope and is it the loop
   index? Is the appended writer subroutine legal Fortran there (module
   `contains` section, `mpp_pe` in scope, assumed-shape dummy)? Does
   patching dyn_core.F90 change the numerics of the run in any way --
   including by perturbing optimisation or by the I/O itself?
3. THE CROSS-CHECK THAT CARRIES CONTROL 4. The claim is that nothing
   between the loop's end and dyn_core's return writes `w`, so the
   sub-step-8 dump must be bitwise the pre-remap dump. VERIFY THAT
   CLAIM IN THE PINNED SOURCE
   (/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90,
   the region from `enddo   ! time split loop` to `end subroutine
   dyn_core`) -- and say whether it also holds for `pt` and `delp`,
   which the probe deliberately does NOT assert on. If the claim is
   false, control 4 is a guaranteed failure and the probe can never run.
4. LAYOUTS. `read_dump` gained a `prefix` argument and is shared with
   the pre-remap probe. Follow the whole chain for the new probe:
   padded-box offset, `oracle_ij` transpose, `DIHEDRAL[nm]` on the port
   side, and WHICH k level each side is on. The first version of the
   sibling probe got exactly this wrong and read 1.0071e+00.
5. CAN THE FIVE CONTROLS FAIL? For each of the five, name the defect it
   catches and one it does NOT.
6. THE SERIES AND ITS SCALE. The probe fixes one scale per field across
   all sub-steps and prints |d|max plus d(it)/d(last). Is a fixed scale
   the right choice for a growth question? Can the printed ratios be
   read as growth when they are something else? The sibling probe
   published a ratio of 1.418 that was purely its own denominator.
7. THE TESTS. Can either pass with the feature removed? Do they assert
   anything that is true by construction? Note that the default pytest
   configuration deselects `-m slow`.
8. ANY COMMENT OR DOCSTRING ADDED HERE THAT IS NOT TRUE.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_NHSUB_DONE
