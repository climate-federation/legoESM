#!/usr/bin/env bash
# Submit one Nature-figure ladder matrix on Derecho at a chosen device count.
#
# The qsub line for this ladder is long and the select string differs between
# the GPU and CPU nodes, which is the only reason this wrapper exists. It does
# nothing the header of nature_ladder.pbs does not already document; it just
# does not let you mistype it.
#
#   ./submit_nature_ladder.sh ocean_gpu 1024
#   ./submit_nature_ladder.sh ocean_cpu 1024 12:00:00
#   ./submit_nature_ladder.sh atm_gpu    512
#
# Four ranks per node either way: on the GPU nodes that is one rank per A100,
# on the CPU nodes it is the rank density Levante measured as the optimum,
# four ranks of thirty-two cores. So the node count is the device count over
# four, and 1024 devices is 256 nodes.
#
# Arms needing more nodes than the allocation are skipped and a valid receipt
# is never re-run, so resubmitting into the same output directory resumes
# rather than repeats. Pass OUTDIR to do that deliberately.
set -euo pipefail

MATRIX="${1:?usage: $0 <atm_gpu|atm_cpu|ocean_gpu|ocean_cpu> <devices> [walltime]}"
DEVICES="${2:?usage: $0 <matrix> <devices> [walltime]}"
WALLTIME="${3:-}"

case "$MATRIX" in atm_gpu|atm_cpu|ocean_gpu|ocean_cpu) ;;
  *) echo "unknown matrix '$MATRIX'" >&2; exit 2 ;; esac
[[ "$DEVICES" =~ ^[0-9]+$ ]] && (( DEVICES > 0 )) || {
  echo "devices must be a positive integer, got '$DEVICES'" >&2; exit 2; }
(( DEVICES % 4 == 0 )) || {
  echo "devices must be a multiple of 4 (four ranks per node), got $DEVICES" >&2
  exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES=$(( DEVICES / 4 ))
BACKEND="${MATRIX##*_}"

if [[ "$BACKEND" == gpu ]]; then
  SELECT="select=${NODES}:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB"
  WALLTIME="${WALLTIME:-04:00:00}"
else
  SELECT="select=${NODES}:ncpus=128:mpiprocs=4:mem=200GB"
  WALLTIME="${WALLTIME:-08:00:00}"
fi

echo "matrix=$MATRIX devices=$DEVICES nodes=$NODES walltime=$WALLTIME"
echo "qsub -l $SELECT -l walltime=$WALLTIME -v MATRIX=$MATRIX${OUTDIR:+,OUTDIR=$OUTDIR} $HERE/nature_ladder.pbs"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

exec qsub -l "$SELECT" -l "walltime=$WALLTIME" \
          -v "MATRIX=$MATRIX${OUTDIR:+,OUTDIR=$OUTDIR}" \
          "$HERE/nature_ladder.pbs"
