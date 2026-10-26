#!/bin/bash
#SBATCH --job-name=cdx_spmd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_spmd_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_spmd_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Adversarially review commit 1e3d1d5ca (full-card SPMD lat-band support:
band z_coord + iwm + cut faces).  git show 1e3d1d5ca for the diff.  Find
defects, cite file:line.  Three files: ocean_model_latlon_cgrid.py (a
z_coord=None override threaded through 17 step-path methods with a per-method
alias _zc, plus iwm_fields=None on the 5 step drivers),
sharded_ocean_step.py (band stacks for z-coord per-cell fields + iwm pytree,
indexed [0] in-body, threaded into _step_body), latlon_cgrid_operators.py
(compute_face_masks_3d: under the armed spmd backend, overwrite the two
boundary v-face rows with the neighbour band's adjacent cell row via
ppermute).

ATTACK, in order:
1. MISSED self.z_coord SITES: the threading was scripted; grep the CURRENT
   file for any remaining `self.z_coord` read inside a threaded method's span
   (a site the script's span logic missed keeps using the GLOBAL coordinate
   inside the band body — silent wrong-shape or wrong-values under SPMD).
   Also check the aliased methods FORWARD z_coord to every threaded callee
   (a call that omits z_coord=z_coord silently reverts to the global).
2. HALF-THREADED HELPERS: methods that read self.z_coord but were NOT
   threaded (check the full method list) — are ANY of them reachable from
   the SPMD _step_body path (_eke_3d_step? _apply_after_level_reconcile?
   seed_scan_carry?)?  A reachable unthreaded helper is the same bug one
   layer down.
3. THE CUT-FACE FIX: verify the ppermute payload/permutation pairing against
   latlon_band_perms' documented semantics (perm_north: band b receives
   b+1's payload; sources send WHICH row?).  Verify the overwrite rows'
   alignment with the band v-face convention (band face 0 = global face s,
   between global cells s-1,s).  Check the meridionally-periodic early
   return above the fix: under SPMD with y-reentrant, the wrap branch
   returns BEFORE the cut-face overwrite — is that combination guarded
   anywhere, or silently wrong?
4. compute_face_masks_3d is also called from HOST-side diagnostics
   (diagnostics_streamfunction.py:89, diagnostics_sections.py:753) — can
   the spmd backend ever still be armed there (the step arms/restores
   around fn() — check the restore is exception-safe and that no other
   caller leaves it armed)?
5. THE 2-D face masks + vertex mask: compute_face_masks (2-D) has the same
   band-local boundary structure — is IT called in-body anywhere on the
   partial-cell path (or is the 2-D v_mask always the carried state field)?
   If called in-body, the same cut bug exists there untested.
6. iwm banding: the stack replicates non-(n_lat,...) leaves per band and
   slices (n_lat,...) leaves [r*nl:(r+1)*nl] — check IWMForcing's actual
   fields for any lat-DEPENDENT 1-D profile or staggered array that this
   rule mis-handles.
Number findings; GO/NO-GO for the 4-GPU eORCA025 smoke." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
