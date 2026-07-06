#!/usr/bin/env bash
# Submit the first link of the self-chaining Ginsburg production AMIP run
# (authoritative C48/L40 config + real ERA5 IC / ETOPO land / observed SST-SIC).
#
#   bash scripts/cluster/amip/submit_amip_ginsburg_production.sh [OUTDIR] [TARGET_DAYS]
#
# The sbatch re-submits itself each link until TARGET_DAYS is reached; the JAX
# compile cache (in OUTDIR) amortizes the ~16 min rrtmg JIT across links.
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUTDIR="${1:-$REPO/results/amip/prod_1979}"
TARGET_DAYS="${2:-365}"
SBATCH="$REPO/scripts/cluster/amip/amip_ginsburg_production_chain.sbatch"

# SLURM opens --output BEFORE the body's mkdir runs, so create the dir first.
mkdir -p "$OUTDIR"
echo "Submitting first chain link:"
echo "  OUTDIR:      $OUTDIR"
echo "  TARGET_DAYS: $TARGET_DAYS"
echo "  Script:      $SBATCH"

J1=$(sbatch --parsable \
     --output="$OUTDIR/slurm_%j.out" \
     --export="ALL,OUTDIR=$OUTDIR,TARGET_DAYS=$TARGET_DAYS" \
     "$SBATCH")
echo "Link 1 submitted: $J1"
echo "Monitor:  squeue -j $J1 -o '%.10i %.18j %.8T %.10M %R'"
echo "Logs:     $OUTDIR/slurm_$J1.out"
echo "Progress: ls $OUTDIR/checkpoint_day_*.npz | tail -1   (latest simulated day)"
echo "Cancel:   scancel $J1   (the next link is only queued after this one succeeds)"
