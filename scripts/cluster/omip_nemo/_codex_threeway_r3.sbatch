#!/bin/bash
#SBATCH --job-name=codex_3way3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_3way_r3_%j.out

# Round 2 of the adversarial loop on the three-way scorer, after the round-1
# fixes.  Round 1 returned 4 RED / 2 YELLOW / 2 LOW; this asks whether the
# fixes are real fixes and whether they introduced anything new.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Round 3 review of scripts/validate/ocean_fidelity/compare_three_way_nemo.py
and tests/validate/test_compare_three_way_nemo.py.  Review ONLY those two files;
you may read scripts/validate/compare_omip_nemo.py for the imported helpers.

Round 2 left these open.  For EACH, judge whether the round-3 code fixes it,
only appears to, or trades it for something new.  Cite line numbers.

R2-1 'SSH is not defensible; remove it rather than caveat it.'  DONE: SSH, the
  --ssh flag, _lego_ssh and _nemo_ssh are all deleted.  Confirm nothing
  SSH-related survives and that removing it did not orphan a variable, break the
  per-field area plumbing, or leave a stale docstring claim.

R2-2 'the 2-degree rerun is not a diagnostic control for the IDW footprint
  asymmetry; use fixed-radius smoothing or conservative remap.'  FIX CLAIMED:
  a new _smooth_common_footprint applies ONE great-circle top-hat to all three
  fields AFTER regridding, selected by --smooth-radius-deg.
  SCRUTINISE THE INSTRUMENT AND THE INFERENCE:
  (a) is a top-hat mean over a great-circle radius on the TARGET grid actually a
      common physical filter, given the target cells are not equal-area (cells
      shrink toward the poles, so a fixed-radius disc contains more cells at high
      latitude)?  Does that latitude dependence invalidate the control, and if so
      in which direction?
  (b) the chord conversion 2*sin(radius/2) -- correct for a great-circle radius
      on a unit sphere?
  (c) the measured result was: as the radius goes 0 -> 2 -> 4 deg, tripole-vs-NEMO
      and MPAS-vs-NEMO stay within 1-3% of each other (SSS 0.5383 vs 0.5544 at 4
      deg) while the cross-grid RMSE falls faster than either (0.1616 -> 0.0661).
      I concluded (i) the regrid artifact is REFUTED because tripole does not
      degrade relative to MPAS under a common filter, and (ii) the cross-grid
      difference is small-scale while the NEMO gap is large-scale.  Are BOTH
      inferences supported by that table?  What alternative explanation would
      produce the same numbers?  Is the unsmoothed mean preserved (the test only
      pins it to 1e-3)?

R2-3 '_wstats can still return NaN correlation; report.json emits bare NaN.'
  FIX CLAIMED: _json_safe nulls non-finite floats recursively and the report is
  written with allow_nan=False.  Check for a path that bypasses it, and whether
  turning an undefined correlation into null loses information a reader needs.

R2-4 'the 10-cell zonal-row floor is not defensible.'  FIX CLAIMED: rows now need
  at least _MIN_ZONAL_SUPPORT_FRAC (0.25) of the cells the common mask offers in
  that row.  Is a fraction-of-available the right criterion, and is 0.25
  defensible or just a different arbitrary number?

R2-5 'the helper tests would still pass if main reverted to ocean = coverage.'
  FIX CLAIMED: the decision is factored into build_ocean_mask and four tests now
  pin it, including that the two modes must DISAGREE.  Would those tests fail on
  that exact revert?

R2-6 'nearest-centre classification is not a real land/sea classification; it has
  no cell polygons or connectivity.'  NOT fixed in code; the docstring of
  build_ocean_mask now states the limitation and calls the near-land sub-domain a
  screening diagnostic.  Is documenting it sufficient for the use it is put to
  (comparing coastal SSS/SST bias between two model grids and NEMO), or does it
  still produce a wrong REPORTED NUMBER?

THEN: anything NEW introduced by the round-3 edits, and any test that is
trivially true or asserts a wrong constant.  Rank RED (produces a wrong reported
number) / YELLOW / LOW.  No style comments.  Label speculation as speculative."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
