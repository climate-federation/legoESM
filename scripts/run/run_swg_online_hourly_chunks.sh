#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 7 ]] || {
  printf 'USAGE: %s PHYSICAL_GPU FIRST_RESTART_DAY LAST_RESTART_DAY CHUNK_DAYS TRUTH_ROOT OUTPUT_ROOT LABEL\n' "$0" >&2
  exit 64
}
physical_gpu=$1
first_day=$2
last_day=$3
chunk_days=$4
truth_root=$5
output_root=$6
label=$7
[[ $physical_gpu =~ ^[0-9]+$ && $first_day =~ ^[0-9]+$ \
   && $last_day =~ ^[0-9]+$ && $chunk_days =~ ^[1-9][0-9]*$ \
   && $first_day -lt $last_day ]] || {
  printf 'REFUSE: GPU/days must be non-negative integers with FIRST < LAST\n' >&2
  exit 64
}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
mkdir -p "$output_root"
day=$first_day
while (( day < last_day )); do
  remaining=$((last_day - day))
  run_days=$chunk_days
  (( run_days <= remaining )) || run_days=$remaining
  stop=$((day + run_days))
  output=$output_root/$(printf '%s_day%04d_to_day%04d' "$label" "$day" "$stop")
  if [[ -f $output/manifest.json ]] \
      && grep -q '"status": "complete"' "$output/manifest.json"; then
    day=$stop
    continue
  fi
  if [[ -e $output ]]; then
    archive=$output.interrupted
    suffix=0
    while [[ -e $archive ]]; do
      suffix=$((suffix + 1))
      archive=$output.interrupted.$suffix
    done
    mv -- "$output" "$archive"
    printf 'ARCHIVED_INCOMPLETE_CHUNK: %s -> %s\n' "$output" "$archive" >&2
  fi
  restart=$truth_root/$(printf 'day%04d.npz' "$day")
  verify=$truth_root/$(printf 'day%04d.npz' "$((day + 1))")
  "$here/run_swg_double_gyre_gpu.sh" "$physical_gpu" \
    --deck maddison --days "$run_days" --restart "$restart" \
    --hourly-targets --verify-next "$verify" --output "$output"
  day=$stop
done
