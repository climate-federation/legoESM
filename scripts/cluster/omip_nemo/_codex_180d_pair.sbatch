#!/bin/bash
#SBATCH --job-name=cdx_180pair
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_180pair_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_180pair_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review commit 2ada0c850 (two sbatch cards:
scripts/cluster/omip_nemo/_longrun_180d_merged.sbatch and
scripts/cluster/omip_nemo/_mpas_fullharm_180d.sbatch). They launch a matched
180-day tripole + MPAS ocean pair, scored vs NEMO GATEWAY 5-day means at days
30/60/90 (records 5/11/17) plus a nino3 trajectory heredoc. Jobs already
running (9562867/9562868) -- if you find a CRITICAL defect we cancel them.
Attack ONLY: (1) output-dir or snapshot-cadence mistakes that would lose or
overwrite data; (2) the GATEWAY record indices vs snapshot days (rec N = days
5N-4..5N); (3) the nino3 heredoc on MPAS snapshots (keys: T (nCells,75),
lat_T, lon_T, land_mask, no cell_area -> cos(lat) fallback); (4) walltime vs
measured 0.77 steps/s at dt 150 for 180 days on MPAS; (5) any flag in the
MPAS card the driver silently ignores. Rank CRITICAL/MAJOR/MINOR, one line
each, no summary." > "$LOG" 2>&1
rc=$?
tail -100 "$LOG"
echo "=== codex rc=$rc ==="
if grep -q "Failed to refresh token\|401 Unauthorized\|token_expired" "$LOG"; then
  echo "FATAL: codex authentication failed -- THIS ITERATION IS UNREVIEWED."
  exit 2
fi
echo DONE
