#!/bin/bash
# ===========================================================================
# Finalize a full-node CPU-vs-A100 campaign: aggregate every job under <outdir>
# into one tidy CSV, then plot per-grid CPU-vs-GPU throughput across resolutions.
# Run AFTER the submit_fullnode.sh jobs have finished (qstat -u $USER is empty).
#
# Any conda env with legoESM installed works (aggregate needs legoesm.constants,
# the plotter needs matplotlib) -- no GPU/MPI env required for this step.
#
# Usage:
#   scripts/cluster/scaling_derecho/finalize_fullnode.sh <outdir>
#     <outdir> : the SAME dir you passed to submit_fullnode.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTDIR="${1:-}"
if [ -z "$OUTDIR" ]; then
    echo "usage: $0 <outdir>   (the dir passed to submit_fullnode.sh)" >&2
    exit 2
fi
if [ ! -d "$OUTDIR" ]; then
    echo "ERROR: outdir '$OUTDIR' does not exist" >&2
    exit 2
fi
PY="${PY:-python}"

echo "=== aggregating all jobs under: $OUTDIR ==="
"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$OUTDIR" --out "$OUTDIR/all_tidy.csv"

echo "=== plotting per-grid CPU vs A100 across resolutions ==="
"$PY" scripts/plot/plot_fullnode_cpu_vs_gpu.py \
    --csv "$OUTDIR/all_tidy.csv" --out "$OUTDIR/plots"

echo "=== done ==="
echo "CSV:   $OUTDIR/all_tidy.csv"
echo "Plots: $OUTDIR/plots/fullnode_cpu_vs_gpu_<grid>.png"
