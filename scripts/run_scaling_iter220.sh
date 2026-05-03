#!/bin/bash
# Iter-220 scan-steps sweep — find optimal K per (grid, resolution)
# now that --scan-steps is wired in.  Records steps/s for K ∈ {1, 6, 12,
# 24, 48} on the GPU across all three grids at two resolutions each.
#
# Usage:  bash scripts/run_scaling_iter220.sh [--quick]
#
# --quick = drop the K=48 sample and the larger resolutions

set -euo pipefail

QUICK=${1:-}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/results/scaling/iter220"
CSV="${ROOT}/results/scaling/iter220_scan_steps.csv"
mkdir -p "${OUT_DIR}"
echo "tag,backend,grid,resolution,scan_steps,wall_time_s,steps_per_sec" > "${CSV}"

# (grid, resolution, dt) tuples
declare -a CASES=(
  "spectral T21 600"
  "spectral T42 300"
  "cubed-sphere 24 450"
  "cubed-sphere 48 150"
  "icosahedral 4 150"
  "icosahedral 5 75"
)
declare -a SCAN_STEPS=(1 6 12 24 48)
if [ "${QUICK}" = "--quick" ]; then
  CASES=(
    "spectral T21 600"
    "cubed-sphere 24 450"
    "icosahedral 4 150"
  )
  SCAN_STEPS=(1 12 24)
fi

run_one() {
  local grid="$1" res="$2" dt="$3" ss="$4"
  local tag="iter220_gpu_${grid//-/_}_${res}_ss${ss}"
  local outdir="${OUT_DIR}/${tag}"
  mkdir -p "${outdir}"
  local logfile="${outdir}/run.log"

  printf "[iter220] >>> %s ... " "${tag}"
  if env -i HOME="${HOME}" PATH="${PATH}" JAX_ENABLE_X64=1 \
      bash -c "source '${ROOT}/scripts/gpu_env.sh' && \
               PYTHONPATH='${ROOT}' '${ROOT}/.venv/bin/python' \
                 '${ROOT}/scripts/run_baroclinic_wave_benchmark.py' \
                 --grid ${grid} --resolution ${res} --dt ${dt} --days 1 \
                 --scan-steps ${ss} \
                 --output-dir '${outdir}' --tag '${tag}'" \
      > "${logfile}" 2>&1; then
    local sps wall
    sps=$(grep -E "Integration complete:" "${logfile}" | sed -E 's/.* \(([0-9]+\.?[0-9]*) steps\/s\).*/\1/' | head -1)
    wall=$(grep -E "Integration complete:" "${logfile}" | sed -E 's/Integration complete: ([0-9]+)s.*/\1/' | head -1)
    printf "%s sps (wall %ss)\n" "${sps:-?}" "${wall:-?}"
    echo "${tag},gpu,${grid},${res},${ss},${wall:-?},${sps:-?}" >> "${CSV}"
  else
    printf "FAILED\n"
    echo "${tag},gpu,${grid},${res},${ss},?,?" >> "${CSV}"
  fi
}

for c in "${CASES[@]}"; do
  read -r grid res dt <<<"${c}"
  for ss in "${SCAN_STEPS[@]}"; do
    run_one "${grid}" "${res}" "${dt}" "${ss}"
  done
done

echo
echo "[iter220] Sweep CSV: ${CSV}"
column -t -s, "${CSV}"
