#!/bin/bash
# Submit the 2^n strong-scaling ladder (user request 2026-08-18):
# both atmosphere lanes, n_dev = 1,2,4,...,128 (Levante gpu = 59 usable
# nodes -> 256/512 impossible; 128 = 32 nodes is the ceiling power).
# One job per point so small arms clear the queue fast. Multi-GPU
# single-node arms (2,4) share one node; NDEV=1 uses 1 GPU.
set -u
cd "$(dirname "$0")/../../.." || exit 2
for LANE in mpas latlon; do
  for NDEV in 1 2 4 8 16 32 64 128; do
    GPN=4; NODES=$(( (NDEV + GPN - 1) / GPN ))
    sbatch -N "$NODES" -J "lad_${LANE}_${NDEV}" \
      -o "lad_${LANE}_${NDEV}.%j.log" \
      --export=ALL,LANE="$LANE",NDEV="$NDEV" \
      scripts/cluster/scaling_levante/ladder_2n_atm.sbatch
  done
done
