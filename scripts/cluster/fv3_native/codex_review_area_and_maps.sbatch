#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_maps
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_maps_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW, two commits in the FV3 duo-grid JAX port. Cite
file:line, classify BLOCKER / MAJOR / MINOR, give the smallest fix, and
say explicitly where you found nothing.

COMMIT A -- `git show 9de677eb3` -- "stop exchanging the area, which is
what broke the NH parity". Files packages/core/legoesm/core/
fv3_native_dsw_tail_3d.py and fv3_dsw_tail_3d.py (the JAX twin). The
port had been running its own halo exchange over the cell area before
handing it to the layer-height update, on the stated grounds that the
gridstruct leaves sentinel values in the corner-diagonal halo. Measured:
the gridstruct's area holds a real value there and matches the reference
at the parity floor everywhere, while the exchanged copy was 12% off in
the halo edge strips. Removing the exchange took the worst one-step
non-hydrostatic residual against the pinned Fortran from 6.6116e-04 to
1.4778e-06.

  A1. Is the deleted exchange truly dead for EVERY caller and every
      configuration on this path, or does some path exist where the
      gridstruct's area is genuinely not filled in the halo? Follow the
      builder, do not trust the commit message.
  A2. The caller now asserts that what it receives is the gridstruct's
      array bitwise. Can that assertion pass while the wrong array is
      used -- aliasing, a stale binding, an array that is bitwise equal
      for an unrelated reason? Can it fire spuriously under jit or on
      the JAX lane, where an identity check on tracers means something
      different?
  A3. Are the numpy lane and the JAX twin actually changed the SAME
      way? Diff them against each other, not against the message.
  A4. Any comment or docstring left in either file that reasons about
      the exchange that no longer exists.

COMMIT B -- `git show 6fff55ffb` -- the maps instrument. Files
scripts/validate/fv3_native/full_step_oracle_parity.py (new
--save-fields), scripts/plot/fv3_duo_face_maps.py,
tests/grids/test_fv3_duo_face_maps.py,
scripts/cluster/fv3_native/full_step_maps.sbatch.

  B1. --save-fields writes the arrays out of the residual loop. Are they
      the arrays that loop SCORES, at the same point, unmutated
      afterwards? A saved array that is one transform away from the
      scored one makes every picture a different claim from the gate.
  B2. THE EDGE METRIC. `high_pass` is a second difference; the ring/
      interior split is applied on the SHRUNK index space it returns.
      Does the ring the code selects correspond to the panel boundary of
      the ORIGINAL plane, or is it off by the shrink? State the exact
      cell indices.
  B3. Is the port/oracle ratio-of-ratios a sound instrument? Name a real
      port defect it would MISS, and a case where it fires with no
      defect. In particular: the compute window here excludes halos --
      does that make "panel boundary" mean what the docstring says?
  B4. The gate value 1.5 in the sbatch is asserted, not derived. Say
      what evidence would fix it and whether the current value can pass
      on a visibly broken field.
  B5. THE TESTS. Do they fail if the metric is broken? Check each of the
      four for the "passes with the feature removed" failure mode, and
      whether the third one (interior noise must LOWER the edge ratio)
      is actually guaranteed by the arithmetic or is an accident of the
      chosen amplitude.
  B6. Any name used and not imported, any docstring claim the code does
      not enforce, any path where the plot script would raise after the
      expensive parity run has already succeeded.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_MAPS_DONE
