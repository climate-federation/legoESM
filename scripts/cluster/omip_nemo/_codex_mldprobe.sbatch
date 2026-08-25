#!/bin/bash
#SBATCH --job-name=cdx_mld
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_mld_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_mld_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Adversarially review two things.

A) The new instrument scripts/validate/ocean_fidelity/mld_basin_vs_nemo.py
   (git show 738315c71 for its introduction).  Its output already drove a
   reported verdict, so attack the INSTRUMENT:
   1. The equator-line jump detector: the per-basin function bj() computes
      np.diff over rows then indexes d[rows] with row indices derived from
      tgt_lat[:-1] — check the row/diff alignment is right and that the
      basin masking (np.where(oc_b, field, nan)) cannot make the ratio
      spuriously ~1 by flooding NaNs (nanmean of empty slices, coastal
      cells).  Could a REAL one-row artifact at the equator be washed out
      by the 1-degree regrid before this detector ever sees it (the model
      grid is 292x362 curvilinear; regrid_curv_to_latlon may average the
      artifact row with neighbours)?  If so the 'no artifact' verdict is
      weaker than reported — say so.
   2. The basin stats: BASINS lon windows (pacific 120-290 etc.) on a
      0-360 tgt_lon — check the Indonesian/Atlantic boundaries do not
      double-count or orphan cells, and that corr on a handful of cells
      (min 20) is meaningful.
   3. The MLD computation calls mixed_layer_depth WITHOUT JAX_ENABLE_X64
      (the sbatch sets JAX_PLATFORMS=cpu but not X64; UserWarnings in the
      log show f32 truncation).  Quantify: does f32 sigma (~1025 kg/m3,
      threshold 0.01) resolve the crossing depth to better than ~2 m, or
      could f32 alone produce the reported -5 to -7 m tropical bias?  This
      is the difference between 'physical signal' and 'instrument noise'.
   4. Ours-instantaneous vs NEMO-5-day-mean MLD: for a diurnally cycling
      tropical mixed layer, which SIGN of bias does snapshot-vs-mean
      produce at the snapshot's write time (midnight UTC = varying local
      time across the Pacific)?  Could that mismatch alone explain a -7 m
      tropical Pacific bias?
B) Commit fe3b5f23f (the codex-finding fixes): verify the +0.05-day
   overshoot actually lands snapshot_day0090 (check the driver's snapshot
   cadence arithmetic at dt=150), that 0.0823288 years > 30 days by enough
   at dt=150, and that the standard scorer's octal guard \$((10#\$1)) and
   PIPESTATUS check are correct bash.
Number findings, severity-tagged; end with VERDICT on whether the reported
'no Pacific bug, two physical signals' conclusion stands on this
instrument." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
