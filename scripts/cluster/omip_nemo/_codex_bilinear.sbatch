#!/bin/bash
#SBATCH --job-name=cdx_bilin
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_bilin_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_bilin_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Adversarially review commit 5f9c0b1aa (git show 5f9c0b1aa): bilinear
CORE-II forcing sampling on the tripole, replacing nearest-neighbour, in
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py.  ATTACK:
1. _bilinear_point_maps correctness: the latitude bracket uses argsort +
   searchsorted — verify it is right for DESCENDING source lat and for
   targets exactly on a source row; verify the longitude wrap branch
   (at_wrap) picks the true bracketing columns for a source lon array that
   does NOT start at 0, and that dspan can never be 0 -> wlon NaN.
2. Weight cache key: same float-checksum scheme as _NN_INDEX_CACHE — can
   two different target sets collide?  Is the 'bilin' prefix enough to
   never collide with the NN cache?
3. The scan lane: build_core2_forcing_device_stack now returns 5 items and
   compute_omip2_surface_forcing_jax takes nn_w=None (legacy).  Grep for
   EVERY caller of both functions (packages/, scripts/, tests/) and verify
   none still unpacks 4 items or omits nn_w where bilinear is intended —
   ESPECIALLY scripts/run/run_omip_core2.py's scan-block lane and
   scripts/validate/ocean_fidelity/run_scm_column_twins.py (which builds
   its own NN stack — is it still consistent as a COLUMN harness?).
4. Host-vs-scan equivalence: with nn_w passed, is the device gather
   (rec[i4,j4]*w4).sum(0) numerically identical (up to fp) to the host
   (f[i4,j4]*w4).sum(0)?  Any dtype mismatch (host f64, device f32/f64)?
5. Semantics: bilinear samples across the land/sea boundary of the
   FORCING grid — CORE-II fields are defined globally (no land mask), so
   no wet-masking is needed — confirm the loaders never carry masked/NaN
   cells that bilinear would smear (nearest was immune to a single NaN
   only if never selected).
6. The two UNCHANGED nearest paths (cube, MPAS): confirm no call site
   accidentally now gets bilinear, and the method kwarg raises on typos.
Number findings, severity-tagged; GO/NO-GO." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
