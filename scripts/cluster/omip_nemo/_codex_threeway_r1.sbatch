#!/bin/bash
#SBATCH --job-name=codex_3way
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_3way_r1_%j.out

# Adversarial review of the THREE-WAY scorer before its numbers are reported.
# This instrument produces the campaign headline "how far are our two grids from
# NEMO, and from each other" table; an unreviewed instrument is exactly the class
# that has produced retractions in this repo.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarially review ONE new file: scripts/validate/ocean_fidelity/compare_three_way_nemo.py.
You may READ scripts/validate/compare_omip_nemo.py (it supplies the regridder,
the NEMO loader, the weighted-stats and the latitude-band helpers) and
scripts/validate/ocean_fidelity/compare_grids_tripole_mpas.py (the two-way
predecessor), but review ONLY the new file.

WHAT IT IS.  It regrids a legoESM tripole (eORCA1 curvilinear, 332x362) snapshot,
a legoESM MPAS (unstructured Voronoi, 163842 cells) snapshot, and a NEMO ORCA1
grid_T record onto one regular lat-lon target, scores every pair on the cells
resolved on ALL THREE, and emits maps + zonal means + a report.json.  The numbers
will be reported as campaign findings.

WHERE I MOST SUSPECT IT IS WRONG:

1. THE COMMON MASK.  ocean = (ocT>0.5)&(ocM>0.5)&(ocN>0.5) where each oc* is the
   regridder's own coverage flag.  Does intersecting three IDW coverage flags
   actually give a set of cells where all three fields are MEANINGFUL, or does a
   cell that is land in NEMO but wet within max_deg=2.5 of a wet legoESM cell
   survive and get scored against an interpolated land value?  Say what the
   regridder does at a coastline and whether the near_land sub-domain -- the
   sub-domain I care most about -- is therefore contaminated by exactly the
   cells it is meant to measure.

2. AREA WEIGHTS.  I use cos(lat) on the TARGET regular lat-lon grid, which is
   the correct cell-area factor there (the sources are already regridded).
   Confirm or refute.  Separately: is an area-weighted RMSE of an IDW-regridded
   field a fair comparison between a 1-degree tripole and an unstructured MPAS
   mesh whose local resolution differs, or does the regrid smooth one arm more
   than the other and thereby flatter it?  Quantify the direction of that bias.

3. MLD.  It is scored on its OWN coverage intersection (mld_ok) with its own
   area (_mld_area), while the sub-domain masks are built from the SST/SSS
   ocean.  Trace whether _mld_area and the sub-masks are consistently intersected
   for the MLD field, and whether the printed band table for MLD uses the right
   area.  Also: is nan_to_num(mld, nan=0.0) with a finite-mask weight the right
   way to regrid a field whose land value is NaN, or does a 0.0 leak into a
   weighted average anywhere?

4. SSH.  Both sides are de-meaned with the area-weighted mean over the COMMON
   mask.  Is removing the mean the right gauge fix for comparing NEMO zos to our
   eta, or does NEMO zos carry an additional reference (e.g. a global volume /
   free-surface convention, or an inverse-barometer term) that a mean removal
   does not reconcile?  If the comparison is not defensible, say so plainly --
   I would rather drop the field than report a wrong number.

5. THE COASTAL DILATION.  _coastal_mask rolls periodically in x and pads with
   LAND in y.  Check the y padding logic line by line (the concatenate of a row
   of ones at the top and bottom) -- does it grow the mask in the intended
   direction, and is a pole-adjacent ocean cell really a boundary cell for the
   purpose I state?  Does n_cells=2 give a 2-cell halo or a 4-cell one?

6. FAILURE MODES MUST BE LOUD.  Point at any path where missing data, a NaN, an
   empty sub-domain or a skipped field could be printed in a way a reader would
   mistake for a measurement of zero difference.  I specifically want to know
   whether the zonal-mean helper zm() (which hand-rolls a nan-safe mean) can
   return a number from a single cell in a latitude row and plot it next to a
   NEMO zonal mean computed from 300 cells.

7. Anything else that would make a reported number wrong.

Be concrete: cite line numbers, say what the value WOULD be vs what it SHOULD
be, and give the fix.  Rank findings RED (produces a wrong reported number) /
YELLOW (fragile or misleading) / LOW.  No style comments.  Label speculation as
speculative -- five sharp findings beat twenty hedged ones."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
