#!/bin/bash
# Iter-222 weak-scaling sweep — vary problem size at fixed device count
# (single GPU on this host) to map the cells/s saturation curve.
# Uses --scan-steps auto so each case is at its iter-220 optimum.
#
# Resolutions cover ~3 orders of magnitude in ncells:
#   spectral:     T21 (8K), T42 (32K), T63 (62K)
#   cubed-sphere: C24 (14K), C48 (55K), C96 (221K)
#   icosahedral:  I4  (2.5K), I5  (10K), I6  (41K)
#
# Records ncells, dt, scan_steps, wall_time, steps/s, Mcells/s into
# results/scaling/iter222_weak.csv.
#
# Usage:  bash scripts/run_scaling_iter222_weak.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/results/scaling/iter222"
CSV="${ROOT}/results/scaling/iter222_weak.csv"
mkdir -p "${OUT_DIR}"
echo "tag,backend,grid,resolution,ncells,nlev,dt,scan_steps,wall_time_s,steps_per_sec,Mcells_per_s" > "${CSV}"

# (grid, resolution, dt) — dt scaled with resolution per CFL
declare -a CASES=(
  "spectral T21 600"
  "spectral T42 300"
  "spectral T63 200"
  "cubed-sphere 24 450"
  "cubed-sphere 48 150"
  "cubed-sphere 96 75"
  "icosahedral 4 150"
  "icosahedral 5 75"
  "icosahedral 6 35"
)

ncells_of() {
  local grid="$1" res="$2"
  case "${grid}" in
    spectral)
      local n=${res#T}
      echo $(( (3*n + 1) * 2 * (3*n + 1) ))
      ;;
    icosahedral)
      local lvl=${res#I}
      lvl=${lvl#i}
      python3 -c "print(10*4**${lvl}+2)"
      ;;
    cubed-sphere)
      local n=${res#C}
      echo $(( 6 * n * n ))
      ;;
  esac
}

run_one() {
  local grid="$1" res="$2" dt="$3"
  local ncells; ncells=$(ncells_of "${grid}" "${res}")
  local tag="iter222_gpu_${grid//-/_}_${res}"
  local outdir="${OUT_DIR}/${tag}"
  mkdir -p "${outdir}"
  local logfile="${outdir}/run.log"
  local nlev=26

  printf "[iter222] >>> %s (n=%d) ... " "${tag}" "${ncells}"
  if env -i HOME="${HOME}" PATH="${PATH}" JAX_ENABLE_X64=1 \
      bash -c "source '${ROOT}/scripts/data/gpu_env.sh' && \
               PYTHONPATH='${ROOT}' '${ROOT}/.venv/bin/python' \
                 '${ROOT}/scripts/run_baroclinic_wave_benchmark.py' \
                 --grid ${grid} --resolution ${res} --dt ${dt} --days 1 \
                 --output-dir '${outdir}' --tag '${tag}'" \
      > "${logfile}" 2>&1; then
    local sps wall ss
    sps=$(grep -E "Integration complete:" "${logfile}" | sed -E 's/.* \(([0-9]+\.?[0-9]*) steps\/s\).*/\1/' | head -1)
    wall=$(grep -E "Integration complete:" "${logfile}" | sed -E 's/Integration complete: ([0-9]+)s.*/\1/' | head -1)
    ss=$(grep -E "scan-batch K =" "${logfile}" | sed -E 's/.*K = ([0-9]+).*/\1/' | head -1)
    [ -z "${ss}" ] && ss=1
    local mcells_per_s
    if [ -n "${sps}" ]; then
      mcells_per_s=$(python3 -c "print(${ncells} * ${nlev} * ${sps} / 1e6)")
    else
      mcells_per_s="?"
    fi
    printf "%s sps, %s Mcells/s, K=%s\n" "${sps:-?}" "${mcells_per_s:-?}" "${ss:-?}"
    echo "${tag},gpu,${grid},${res},${ncells},${nlev},${dt},${ss:-?},${wall:-?},${sps:-?},${mcells_per_s:-?}" >> "${CSV}"
  else
    printf "FAILED\n"
    echo "${tag},gpu,${grid},${res},${ncells},${nlev},${dt},?,?,?,?" >> "${CSV}"
  fi
}

for c in "${CASES[@]}"; do
  read -r grid res dt <<<"${c}"
  run_one "${grid}" "${res}" "${dt}"
done

echo
echo "[iter222] Sweep CSV: ${CSV}"
column -t -s, "${CSV}"
