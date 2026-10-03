#!/usr/bin/env bash
# Build run dirs for the snow-thermal-node A/B (no submission).
# Usage: setup_arms.sh <name> [<name> ...]   (names from arms.tsv)
# Each dir gets a read-only COPY of its pinned mv3y_vl checkpoint (never a
# pointer to a live run), a .device_count stamp of 4, and arm.tsv
# (name, target, extra, repo) for bundle.sbatch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${ROOT:=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}"
: "${SRC:=${ROOT}/mv3y_vl}"
(( $# )) || { echo "name the run(s) to set up" >&2; exit 2; }
for want in "$@"; do
  found=0
  while IFS=$'\t' read -r name day target repo extra; do
    [[ "${name}" == \#* || "${name}" != "${want}" ]] && continue
    found=1
    ck="${SRC}/checkpoint_day_0${day}.npz"
    [[ -f "${ck}" ]] || { echo "missing ${ck}" >&2; exit 2; }
    [[ "${repo}" == /* && -e "${repo}/.git" ]] || { echo "bad repo ${repo}" >&2; exit 2; }
    d="${ROOT}/${name}"
    [[ -e "${d}" ]] && { echo "REFUSING: ${d} exists" >&2; exit 3; }
    mkdir -p "${d}"
    cp -f "${ck}" "${d}/"
    chmod a-w "${d}/$(basename "${ck}")"
    echo 4 > "${d}/.device_count"
    printf '%s\t%s\t%s\t%s\n' "${name}" "${target}" "${extra}" "${repo}" > "${d}/arm.tsv"
    cp -f "${HERE}/PREREG.md" "${d}/PREREG.md"
    echo "${name}: restart ${day}, target ${target}, repo ${repo} @ $(git -c safe.directory='*' -C "${repo}" rev-parse --short HEAD), extra '${extra}'"
  done < "${HERE}/arms.tsv"
  (( found )) || { echo "no arm ${want} in arms.tsv" >&2; exit 2; }
done
