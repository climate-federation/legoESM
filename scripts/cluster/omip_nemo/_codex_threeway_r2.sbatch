#!/bin/bash
#SBATCH --job-name=codex_3way2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_3way_r2_%j.out

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
  "Round 2 review of scripts/validate/ocean_fidelity/compare_three_way_nemo.py
and its new test file tests/validate/test_compare_three_way_nemo.py.  You may
read scripts/validate/compare_omip_nemo.py for the imported helpers.  Review
ONLY those files.

Round 1 of this review returned these findings.  For EACH, judge whether the
current code actually fixes it, only appears to, or has traded it for a new
defect.  Be concrete and cite line numbers.

R1-RED-1 'the ocean mask is a distance-to-wet-data flag, not a land/sea
  classification; it admits land cells within 2.5 deg of ocean and scores them
  with a water-side extrapolation, and this contaminates near_land most'.
  FIX CLAIMED: a new _nn_wet_mask classifies each target cell by whether the
  NEAREST cell of every source is wet, and the common mask is the intersection
  of that with the old coverage flag (--mask-mode nearest, now the default).
  Judge: is nearest-source classification correct here?  Round 1 also noted the
  KD-tree has no coastline topology and can cross narrow land barriers -- does
  that objection apply to the MASK as well as to the interpolated VALUES, and
  if so how wrong is it at 1 degree in, say, the Canadian Archipelago or the
  Danish straits?  Is intersecting with the old coverage flag redundant,
  harmful, or right?

R1-RED-2 'SSH discards all three coverage flags and scores with SST area'.
  FIX CLAIMED: ssh_ok = ocean & the three SSH coverage flags, its own area used
  for BOTH the de-meaning and the scoring, and a per-field finite intersection
  applied to every field in the scoring loop.

R1-RED-3 'de-meaning does not reconcile inverse-barometer / free-surface
  diagnostic / datum differences, so SSH is not defensible'.
  FIX CLAIMED: SSH is now opt-in behind --ssh, off by default, and labelled
  informational in the docstring and the help text.  Is a caveat sufficient, or
  should the field be removed outright?  State plainly which.

R1-RED-4 'the coastal mask pads latitude with land so every ocean cell in the
  top and bottom rows is coastal; and n_cells=2 is a Manhattan halo, not the
  documented 3x3 dilation'.
  FIX CLAIMED: _coastal_mask now dilates with all 8 neighbours, is periodic in
  x, and shifts in y WITHOUT padding (content shifted off an edge contributes
  nothing).  Verify the _shift helper index arithmetic line by line for both
  signs of dy -- I want to know if it is off by one, and whether n_cells=2 now
  gives exactly a Chebyshev halo of half-width 2.

R1-YELLOW-1 'the IDW stencil (k=4) smooths a ~111 km ORCA1 cell and a ~60 km
  MPAS cell unequally, which flatters tripole-vs-NEMO because tripole and NEMO
  share a grid scale'.
  NOT fixed in code; instead the job runs the same pair on a 2-degree target as
  a control.  Is that control sufficient to detect the artifact?  What would
  the numbers look like if the artifact were real vs absent?  If the control is
  not diagnostic, say what is.

R1-YELLOW-2 'missingness is not loud; NaNs can reach report.json as bare NaN;
  the zonal helper can average one cell against a NEMO row of 300'.
  FIX CLAIMED: a per-field finite triple intersection zeroes the area of any
  non-finite cell and prints a warning; the zonal means use ONE row mask common
  to all three curves and drop rows with fewer than _MIN_ZONAL_CELLS (10) cells.
  Check whether _wstats can still return NaN, whether any statistic is computed
  before the finite gate is applied, and whether 10 is a defensible floor.

R1-LOW-1 '--res-deg is only geometrically correct at 1 degree'.  FIX CLAIMED:
  the target grid is now built as -90 + res/2 + res*k with a divisibility check.
R1-LOW-2 'the docstring claims the run manifests are copied into the report but
  nothing reads them'.  FIX CLAIMED: _manifest_summary now records the command
  line, git_dirty flag, version and creation time of each arm.

THEN, independently of the above: review the NEW TEST FILE as an instrument in
its own right.  A test that cannot fail is worse than no test.  For each test,
say whether it would actually FAIL if the behaviour it claims to pin were
reverted -- in particular test_coastal_mask_all_ocean_has_no_coast against the
old land-padding, and test_nn_wet_mask_* against the old coverage flag.  Flag
any assertion that is trivially true, any hard-coded expected count that is
wrong, and anything important that is NOT covered.

Rank findings RED (produces a wrong reported number) / YELLOW (fragile or
misleading) / LOW.  No style comments.  Label speculation as speculative."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
