#!/bin/bash
#SBATCH --job-name=cdx_pad
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_pad_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_pad_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Adversarially review an UNCOMMITTED fix (git diff of the working tree vs
HEAD) in packages/core/legoesm/grids/tripole.py and its new tests in
tests/grids/test_tripole_internals.py.  Find defects; do not summarise.

CONTEXT.  pad_tripole_grid_south prepends land rows at the south of a tripole
geometry so the SPMD lat-band decomposition divides evenly (eORCA025 n_lat=
1207).  It missed cos_lat_v — the (n_lat+1,) clamped |cos| profile at v-faces
— which stayed un-grown, so on the padded 1/4-degree grid the band slicer's
[s:e+1] handed the north band one fewer element than the interior bands and
the per-field band stack raised 'All input arrays must have the same shape'
(the observed 4-GPU crash).  The fix grows cos_lat_v (prepending clamped cos
of the new rows' 1-D latitudes) and seam_wall_rows (if present, prepending
1.0 = walled, since the new rows are land).

ATTACK, citing file:line:
1. VALUE CORRECTNESS of the new cos_lat_v entries: cos_lat_v is a v-FACE
   profile; the fix prepends cos of the new CELL-CENTER 1-D latitudes
   (lat_new_1d), not face latitudes.  The new rows are land (values only
   need finite+positive per the pad's contract) — but check every CONSUMER
   of cos_lat_v (grep packages/) to confirm none of them touches the padded
   rows in a way where a center-vs-face latitude off-by-half-a-cell matters
   (e.g. a global reduction without a wet mask, a metric ratio, a
   divergence weight that reaches masked rows before masking).
2. ORDER/ALIGNMENT: after prepending n_pad values, is element [n_pad] of
   the new cos_lat_v the ORIGINAL element [0] (bit-exact preservation of
   the wet rows), matching the other v-face fields' _edge_pad alignment?
   Off-by-one here silently shifts a metric on every wet row.
3. seam_wall_rows: is 1.0 (walled) the right value for land rows given the
   consumers (compute_face_masks, compute_face_masks_3d,
   compute_vertex_mask, barotropic diffusion-mask)?  Could a walled-vs-open
   disagreement with the padded land_mask (0) create an inconsistent face
   mask at the pad seam row?
4. THE TESTS: can the generic stagger audit pass while the pad is still
   broken (vacuity)?  Check the axis-matching rule (d in (n_lat, n_lat+1))
   for collisions at the chosen test sizes (n_lat=15, n_lon=24) — an axis
   of size 15/16 that is NOT a latitude axis would be wrongly required to
   grow.  Any geometry field where that rule mis-fires?
5. COMPLETENESS: diff the geometry's _fields against the pad's _replace —
   is ANY array field still missing after this fix?
Number the findings; say GO/NO-GO for committing." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
