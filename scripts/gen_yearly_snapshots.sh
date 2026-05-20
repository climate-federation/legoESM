#!/bin/bash
# Generate yearly snapshots from MPAS OMIP restarts.
# Run on CPU so it doesn't interfere with GPU runs.
# Usage: bash scripts/gen_yearly_snapshots.sh results/mpas_jra55_etopo_100yr_ico6/mpas/ico6 6

RUNDIR=${1:?Usage: $0 RUNDIR SUBDIVISION}
SUB=${2:?Usage: $0 RUNDIR SUBDIVISION}

RESTARTS=$(ls $RUNDIR/restarts/restart_day*.npz 2>/dev/null | while read f; do
  day=$(echo $f | grep -o '[0-9]\{6\}')
  # Yearly: every 360 days
  if [ $((10#$day % 360)) -eq 0 ] && [ $((10#$day)) -gt 0 ]; then
    # Skip if snapshot already exists
    snap=$RUNDIR/snapshots/snapshot_day${day}.png
    if [ ! -f "$snap" ]; then
      echo $f
    fi
  fi
done)

if [ -z "$RESTARTS" ]; then
  echo "No new yearly snapshots to generate."
  exit 0
fi

N=$(echo "$RESTARTS" | wc -w)
echo "Generating $N yearly snapshots on CPU..."
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python scripts/plot_mpas_omip_snapshot.py \
  --sub $SUB $RESTARTS
