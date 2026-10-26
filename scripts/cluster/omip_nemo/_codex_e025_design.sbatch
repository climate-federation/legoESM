#!/bin/bash
#SBATCH --job-name=cdx_e025
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_e025_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_e025_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Review the DESIGN of a proposed 1/4-degree (eORCA025) OMIP run BEFORE it is
launched.  Goal: test whether 1/4-degree resolution fixes an equatorial cold
tongue that is confirmed NOT to be a mixing-closure bug (our TKE closure is
faithful and self-consistent).  At 1 degree our equatorial undercurrent is 4x
too weak; the hypothesis is that resolving the EUC at 1/4 degree cools the
tongue.  Do NOT rubber-stamp -- find what will make this run fail or the test
invalid.

THE PROPOSED SCRIPT: scripts/cluster/omip_nemo/_eorca025_equatorial_test.sbatch
It reuses the faithful 1-degree card (in _kmean_prandtl_merged_d30.sbatch and
_longrun_180d_merged.sbatch) on the eORCA025 mesh (data/grids/
eORCA025_mesh_mask.nc), single GPU, --nemo-vertical (75 levels), --dt 75,
--balanced-init --freeze-floor, full physics (iwm/isf/bbl/mle/tke/prognostic-
ice), 30 days.

THE ONLY PRIOR RUNNABLE eORCA025 CONFIG is
scripts/cluster/omip_nemo/run_e025_rk3_highres.sbatch: it used --nlev 20
(NOT 75), --dt 75, NO iwm/isf/bbl/mle/tke/prognostic-ice, NO tripole-vmix, and
still needed --momentum-rk3 + --balanced-init to survive cold start (an earlier
attempt blew up at day 2 without RK3).  So the faithful full-physics card is
UNPROVEN at 1/4 degree.

QUESTIONS, concrete, cite file:line:
1. MEMORY FIT: will eORCA025 (n_lat~1207, n_lon~1442) at 75 levels with the
   full physics fit ONE GPU (check the state + K-profile + ice + tracer
   allocations against ~40-80 GB)?  Read run_omip_core2.py's fp32 path
   (search --fp32, _FP32) -- is fp32 REQUIRED for eORCA025 to fit, and does the
   faithful card tolerate fp32 (the closure/EOS/PGF in single precision)?  The
   prior config used 20 levels for a reason -- was it memory?
2. COLD-START STABILITY: the 1-degree card uses dt=150; this proposes dt=75 at
   4x finer horizontal grid.  Is dt=75 CFL-safe for eORCA025 with RK3, or does
   the equatorial 1/9-degree cells + full physics need a smaller dt?  Check the
   barotropic sub-stepping and the smag CFL safety.
3. FLAG COMPATIBILITY: does every flag in the proposed script exist and apply on
   the eORCA025 tripole path (iwm/isf/bbl/mle/prognostic-ice/sw-rgb-chl/
   tke-mxl-choice)?  Any that are 1-degree-only or need a different forcing/
   input file at 1/4 degree (the iwm/isf forcing files are on the ORCA1 grid --
   do they regrid to eORCA025 or fail)?
4. FORCING: the corrected CORE-II cache is the default -- does it resolve/
   regrid onto eORCA025, or is it built for eORCA1 only?  A mismatched forcing
   grid would confound the whole comparison.
5. TEST VALIDITY: the comparison is INTERNAL (our 1/4 vs our 1 degree) with the
   same corrected forcing but different mesh/bathymetry/coastline.  Is that a
   valid resolution test, and what confounds (land mask, partial cells,
   different NEMO reference) must be stated?  Is 30 days enough for the EUC to
   spin up?
Answer with a numbered GO / NO-GO per question and the single change most
likely to prevent a wasted 1/4-degree run." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
