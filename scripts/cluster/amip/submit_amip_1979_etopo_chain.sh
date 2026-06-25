#!/usr/bin/env bash
# Submit the first link of the self-chaining 1979 AMIP C48 ETOPO run.
#
# Usage:
#   ./scripts/cluster/amip/submit_amip_1979_etopo_chain.sh [OUTDIR]
#
# The sbatch script re-submits itself on success until 365 simulation days
# are reached.  Expected: 3 links × ≤12 h each ≈ 27.5 h total wall time.

set -euo pipefail

OUTDIR="${1:-/scratch/b/b309178/amip_1979_etopo}"
SBATCH="${BASH_SOURCE%/*}/amip_1979_c48_etopo_chain.sbatch"
TARGET_DAYS=365

mkdir -p "${OUTDIR}"
echo "Submitting first chain link:"
echo "  OUTDIR:      ${OUTDIR}"
echo "  TARGET_DAYS: ${TARGET_DAYS}"
echo "  Script:      ${SBATCH}"

J1=$(sbatch --parsable \
     --export="OUTDIR=${OUTDIR},TARGET_DAYS=${TARGET_DAYS}" \
     "${SBATCH}")
echo "Link 1 submitted: ${J1}"
echo "Monitor: squeue -j ${J1} -o '%.10i %.22j %.8T %.10M %R'"
echo "Logs:    ${OUTDIR}/slurm-${J1}.out"
echo "Cancel:  scancel ${J1}  (subsequent links cancel automatically)"
