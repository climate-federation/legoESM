#!/usr/bin/env bash
# Build run dirs for the snow-thermal-node A/B (no submission).
# Usage: setup_arms.sh <name> [<name> ...]   (names from arms.tsv)
# Each dir gets a read-only COPY of its pinned mv3y_vl checkpoint (never a
# pointer to a live run), a .device_count stamp of 4, and arm.tsv
# (name, target, repo, commit, extra -- extra LAST: it may be empty, and bash
# collapses adjacent tab delimiters) for bundle.sbatch.  The repo must be clean
# and at the arms.tsv commit (a full hash, or HEAD = EXPECT_COMMIT from the env);
# bundle.sbatch refuses to launch a repo that has moved since.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${ROOT:=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}"
: "${SRC:=${ROOT}/mv3y_vl}"
(( $# )) || { echo "name the run(s) to set up" >&2; exit 2; }
for want in "$@"; do
  found=0
  while IFS=$'\t' read -r name day target repo want_commit extra; do
    [[ "${name}" == \#* || "${name}" != "${want}" ]] && continue
    found=1
    ck="${SRC}/checkpoint_day_0${day}.npz"
    [[ -f "${ck}" ]] || { echo "missing ${ck}" >&2; exit 2; }
    [[ "${repo}" == /* && -e "${repo}/.git" ]] || { echo "bad repo ${repo}" >&2; exit 2; }
    head=$(git -c safe.directory='*' -C "${repo}" rev-parse HEAD)
    # HEAD rows: the operator names the reviewed commit (EXPECT_COMMIT); a
    # clean tree at any other commit is refused.
    [[ "${want_commit}" == HEAD ]] && want_commit="${EXPECT_COMMIT:?set EXPECT_COMMIT to the reviewed commit}"
    [[ "${want_commit}" == "${head}" ]] \
      || { echo "REFUSING: ${repo} is at ${head}, expected ${want_commit}" >&2; exit 2; }
    [[ -z $(git -c safe.directory='*' -C "${repo}" status --porcelain --untracked-files=no) ]] \
      || { echo "REFUSING: ${repo} has uncommitted changes" >&2; exit 2; }
    d="${ROOT}/${name}"
    [[ -e "${d}" ]] && { echo "REFUSING: ${d} exists" >&2; exit 3; }
    mkdir -p "${d}"
    cp -f "${ck}" "${d}/"
    chmod a-w "${d}/$(basename "${ck}")"
    echo 4 > "${d}/.device_count"
    printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${target}" "${repo}" "${head}" "${extra}" > "${d}/arm.tsv"
    cp -f "${HERE}/PREREG.md" "${d}/PREREG.md"
    echo "${name}: restart ${day}, target ${target}, repo ${repo} @ ${head}, extra '${extra}'"
  done < "${HERE}/arms.tsv"
  (( found )) || { echo "no arm ${want} in arms.tsv" >&2; exit 2; }
done
