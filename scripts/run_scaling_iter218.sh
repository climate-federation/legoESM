#!/bin/bash
# Iteration-218 scaling sweep for the GPU-scaling branch.
#
# Runs run_baroclinic_wave_benchmark.py for each of the three production
# grid types (spectral, cubed-sphere, icosahedral) at two resolutions
# each, on both CPU and GPU.  Records wall time + steps/s in
# results/scaling/iter218_throughput.csv.
#
# Usage:
#   bash scripts/run_scaling_iter218.sh             # full sweep
#   bash scripts/run_scaling_iter218.sh --quick     # only T21/I4/C24
#
# The companion ``scripts/plot_scaling_iter218.py`` consumes the CSV
# and writes ``results/scaling/scaling.png``.

set -euo pipefail

QUICK=${1:-}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/results/scaling/iter218"
CSV="${ROOT}/results/scaling/iter218_throughput.csv"

mkdir -p "${OUT_DIR}"

# header
echo "tag,backend,grid,resolution,nlev,dt,days,wall_time_s,steps_per_sec" > "${CSV}"

declare -a CASES=(
  "spectral T21 600 1"
  "spectral T42 300 1"
  "cubed-sphere 24 450 1"
  "cubed-sphere 48 150 1"
  "icosahedral 4 150 1"
  "icosahedral 5 75 1"
)
if [ "${QUICK}" = "--quick" ]; then
  CASES=(
    "spectral T21 600 1"
    "cubed-sphere 24 450 1"
    "icosahedral 4 150 1"
  )
fi

run_one() {
  local backend="$1" grid="$2" res="$3" dt="$4" days="$5"
  local tag="iter218_${backend}_${grid//-/_}_${res}"
  local outsubdir="${OUT_DIR}/${tag}"
  mkdir -p "${outsubdir}"
  local logfile="${outsubdir}/run.log"

  echo "[iter218] >>> ${tag}"
  if [ "${backend}" = "gpu" ]; then
    env -i HOME="${HOME}" PATH="${PATH}" JAX_ENABLE_X64=1 \
        bash -c "source '${ROOT}/scripts/gpu_env.sh' && \
                 PYTHONPATH='${ROOT}' '${ROOT}/.venv/bin/python' \
                   '${ROOT}/scripts/run_baroclinic_wave_benchmark.py' \
                   --grid ${grid} --resolution ${res} --dt ${dt} --days ${days} \
                   --output-dir '${outsubdir}' --tag '${tag}'" \
        > "${logfile}" 2>&1 || { echo "[iter218] FAILED: ${tag}"; return 1; }
  else
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="${ROOT}" \
        "${ROOT}/.venv/bin/python" \
        "${ROOT}/scripts/run_baroclinic_wave_benchmark.py" \
        --grid "${grid}" --resolution "${res}" --dt "${dt}" --days "${days}" \
        --output-dir "${outsubdir}" --tag "${tag}" \
        > "${logfile}" 2>&1 || { echo "[iter218] FAILED: ${tag}"; return 1; }
  fi

  # Parse the summary block.  The script prints
  # "Wall time:       <s>s (<sps> steps/s)" near the end.
  local wall sps
  wall=$(grep -E "Wall time:" "${logfile}" | sed -E 's/[^0-9.]+([0-9]+\.?[0-9]*)s.*/\1/' | head -1)
  sps=$(grep -E "Wall time:" "${logfile}" | sed -E 's/.*\(([0-9]+\.?[0-9]*) steps\/s\)/\1/' | head -1)
  local nlev=26
  echo "${tag},${backend},${grid},${res},${nlev},${dt},${days},${wall},${sps}" >> "${CSV}"
  printf "[iter218]    wall=%s s, %s sps\n" "${wall:-?}" "${sps:-?}"
}

for c in "${CASES[@]}"; do
  read -r grid res dt days <<<"${c}"
  run_one cpu "${grid}" "${res}" "${dt}" "${days}" || true
done

# Then GPU
for c in "${CASES[@]}"; do
  read -r grid res dt days <<<"${c}"
  run_one gpu "${grid}" "${res}" "${dt}" "${days}" || true
done

echo
echo "[iter218] Throughput CSV: ${CSV}"
column -t -s, "${CSV}"
