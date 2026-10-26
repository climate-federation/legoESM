#!/bin/bash
#SBATCH --job-name=cdx_fold
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fold_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_fold_body_${SLURM_JOB_ID}.txt
codex exec --sandbox read-only -C $WT \
"CLAIM REVIEW BEFORE CODE (a fold-BC fix for the eORCA025 tripole).  Review
the CLAIM, the planned fix, and name anything that would falsify it.

EVIDENCE (measured, scripts/validate/ocean_fidelity/check_tripole_fold_pairing.py,
exact to <1e-6 deg on the stored mesh coordinates):
eORCA1.2 (works; 332x362; fold row all LAND):
  T: row331 = row330[n_lon-1-i]     U: row331 = row330[n_lon-2-i]
  V: row330 = row330[n_lon-1-i]     F: row330 = row330[n_lon-2-i]  (self)
eORCA025 (NaN by step 23; 1206x1440; fold row 366 WET cells):
  T: row1205 = row1205[(n_lon-i)%n_lon]  (self)
  U: row1205 = row1205[n_lon-1-i]        (self)
  V: row1205 = row1204[(n_lon-i)%n_lon]
  F: row1205 = row1204[n_lon-1-i]
CLAIM: both meshes are NEMO jperio=4 T-PIVOT grids; eORCA1.2 stores the
duplicated HALO row as its top row, eORCA025 is DE-HALOED and stores the
self-symmetric PIVOT row as its top row.  Our fold operators
(packages/core/legoesm/grids/operators_latlon_cgrid.py: apply_north_fold,
pad_ns_vector_v, fold_row; descriptor from packages/core/legoesm/grids/
tripole.py:_detect_fold) implement ONLY the halo-row-stored layout (ghost
row above the stored top = permuted stored top row) and never enforce the
pivot row's half-mirror self-duplication — so on eORCA025 the 366 wet fold
cells evolve as two independent mirror copies with a wrong ghost BC, and
the run NaNs within ~23 steps.
PLANNED FIX: (1) _detect_fold classifies the stored layout by these exact
coincidence tests (halo_row_stored vs pivot_row_stored) instead of the
lat-only symmetry test; (2) for pivot_row_stored, the ghost row above the
top comes from the row BELOW the pivot (row j-1) with the T/U maps derived
from NEMO lbc_nfd T-pivot ihls=1 indices translated to the de-haloed
layout, and the pivot row itself gets a per-step half-mirror enforcement
(tracers +, vectors sign-flipped) exactly as NEMO's 'line ipj-ihls right
half' block; (3) v/F staggered rows use the measured cross-row maps; (4) a
synthetic pivot-row-stored tripole gate mirrors the existing synthetic
tripole test.
ATTACK: read nemo_orca1/nemo_5.0.1/src/OCE/LBC/lbc_nfd_generic.h90 ('T'
pivot section) and our operators; check (a) the claim's layout reading of
lbc_nfd against the measured maps INCLUDING the U-map one-column offsets,
(b) what the correct GHOST-row maps are for the de-haloed layout for each
of T/U/V/F and the C-grid v array we carry at (n_lat+1) faces, (c) whether
the vector SIGN for u/v across this fold is -1 for both components in
NEMO (psgn) and where our vector_sign_v enters, (d) any place our step
reads beyond the ghost row (2-row-deep stencils) that a 1-row fix misses,
(e) whether eORCA1.2's current treatment is also subtly wrong on WET rows
near the fold (row 330 is wet?) and only masked at the very top.
Number findings; end with the CORRECTED mapping table you would implement." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|not logged in" "$LOG" && echo "FATAL: codex auth failed"
cat "$LOG"
echo DONE
