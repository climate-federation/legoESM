#!/bin/bash
#SBATCH --job-name=cdx_xg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_xg_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_xg_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review a cross-grid conclusion in an ocean fidelity campaign.
Do NOT summarise. Attack the inference. Rank CRITICAL/MAJOR/MINOR and name the
measurement that settles each.

SETUP. Two ocean models are matched against NEMO ORCA1 forced with CORE-II
normal-year forcing, cold start from a January climatology, scored at day 30
against NEMO's OWN matched cold start (RUN_GATEWAY 5-day grid_T, record 5 =
days 26-30), regridded to a common 1-degree lat-lon.

  tripole  = this repo's eORCA1 tripole, 75 levels, dt 150 s, TKE closure
  FESOM    = fesom-jax on the CORE2 unstructured mesh, dt 1800 s, f32

Both now read the SAME bias-corrected CORE-II cache (verified from each run's
own resolved-path log line).

MEASURED, day 30:
                      SST rmse  SST bias  nino3 bias  SSS rmse  MLD rmse
  tripole corrected      0.515    +0.012      +1.42     0.464     21.5
  FESOM raw winds        0.890    -0.287      +1.55     1.016     42.9
  FESOM corrected        1.016    -0.649      +0.17     1.026     54.1

FESOM band SST bias, corrected: antarctic -0.41, SH midlat -0.60, tropics
-0.57, NH midlat -1.44, arctic -0.46.  FESOM MLD bias: antarctic +10.8,
SH midlat +5.7, tropics +12.9, NH midlat +44.2, arctic +47.7 m.
FESOM Arctic SSS rmse 2.52.

THE CLAIMS I AM ABOUT TO BUILD ON:
 C1. The corrected forcing's cold-tongue fix REPLICATES across two independent
     dycores (tripole +2.56 -> +1.42, FESOM +1.55 -> +0.17), so the forcing
     correction is established rather than a tripole artifact.
 C2. The tripole's REMAINING +1.42 cold-tongue bias is tripole-specific,
     because FESOM on the identical atmosphere reaches +0.17. Therefore it is
     neither the forcing nor 1-degree resolution, and the search should move
     to the tripole's own numerics -- specifically its equatorial vertical
     mixing and lateral viscosity.
 C3. The two grids fail oppositely -- tripole warm at the equator with
     reasonable mixed layers, FESOM cold everywhere with mixed layers 44 to 48
     metres too deep -- so vertical mixing intensity is the common lever,
     too weak on one and too strong on the other.

KNOWN UNMATCHED between the two: FESOM keeps its own PHC initial state, its
own sea-surface-salinity restoring, runoff and chlorophyll files, its own sea
ice, dt 1800 vs 150 s, single vs double precision, and an unstructured mesh
regridded to the scoring grid. The tripole additionally runs diurnal
shortwave, ice-shelf melt, bottom-boundary-layer advection, RGB chlorophyll
and wave-driven interior mixing, none of which FESOM has here.

ATTACK:
 1. Does C2 survive that list of unmatched differences? Name the single most
    likely confound that would let FESOM get the cold tongue right for a
    reason having nothing to do with the tripole's defect, and the cheapest
    measurement that separates it.
 2. FESOM's cold tongue is right while FESOM is 0.65 C COLD globally. Is a
    correct nino3 bias inside a uniformly cold ocean evidence of correct
    equatorial dynamics, or an accident of two errors cancelling? What would
    distinguish them?
 3. C1 says the forcing lever replicates. But on FESOM the same change made
    global SST error and mixed-layer error WORSE (0.890 to 1.016, 42.9 to
    54.1). Is 'replicates' the honest word, and what should be said instead?
 4. Both grids are scored after regridding to 1 degree. FESOM is unstructured
    and the tripole is curvilinear. Can regridding alone produce a difference
    of the size claimed in nino3, and how would I bound that?
 5. Day 30 of a cold start. nino3 bias on the tripole GREW from +0.93 at day
    20 to +1.42 at day 30. FESOM's is +0.17 at day 30 -- do I know its
    trajectory, and does a single day-30 number support a claim about which
    model is right?
 6. Anything in the repo's own FESOM comparison path that would systematically
    favour or penalise one grid (masking, the freeze clamp at -1.8 C, the
    area weighting, the mixed-layer criterion).
" > "$LOG" 2>&1
rc=$?
tail -240 "$LOG"
echo "=== codex rc=$rc ==="
if grep -q "Failed to refresh token\|401 Unauthorized\|token_expired" "$LOG"; then
  echo "FATAL: codex authentication failed -- THIS ITERATION IS UNREVIEWED."
  exit 2
fi
echo DONE
