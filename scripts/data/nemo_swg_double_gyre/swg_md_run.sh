#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  swg_md_run.sh --nemo-cfg DIR --output-root DIR --python PYTHON --conda-prefix DIR [--mpiexec CMD] [--execute] <label> <nn_SWG> <dt_s> <steps> [ranks] [nn_stock]
  swg_md_run.sh --nemo-cfg DIR --output-root DIR --python PYTHON --conda-prefix DIR [--mpiexec CMD] [--execute] --smoke <label> [ranks]
  swg_md_run.sh --nemo-cfg DIR --output-root DIR --python PYTHON --conda-prefix DIR [--mpiexec CMD] [--execute] --three-year <label> [ranks]
  swg_md_run.sh --nemo-cfg DIR --output-root DIR --python PYTHON --conda-prefix DIR [--mpiexec CMD] [--execute] --short-ladder <label>

Default mode is a read-only preflight. Smoke is 30 days (51,840 steps);
three-year is 1,095 days (1,892,160 steps) with annual restarts. The committed
Maddison namelist patch defines the NEMO-side deck; completed daily runs are
cross-checked against legoESM's independently encoded resolved card.
--conda-prefix is the NEMO build environment: its bin/ is put first on PATH
and its Open MPI mpirun is the default launcher. --mpiexec may name a site
launcher for that same Open MPI (the tool adds --oversubscribe, an Open MPI
option).
EOF
}

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE_UNHANDLED: swg_md_run.sh line %s exited %s\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

nemo_cfg=
output_root=
mpiexec=
python=
conda_prefix=
execute=0
mode=generic
positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo-cfg) nemo_cfg=${2-}; shift ;;
    --output-root) output_root=${2-}; shift ;;
    --mpiexec) mpiexec=${2-}; shift ;;
    --python) python=${2-}; shift ;;
    --conda-prefix) conda_prefix=${2-}; shift ;;
    --execute) execute=1 ;;
    --smoke) mode=smoke ;;
    --three-year) mode=three-year ;;
    --short-ladder) mode=short-ladder ;;
    --help|-h) usage; exit 0 ;;
    --*) printf 'REFUSE_USAGE: unknown option %s\n' "$1" >&2; exit 64 ;;
    *) positionals+=("$1") ;;
  esac
  shift
done
[[ -n "$nemo_cfg" && -n "$output_root" && -n "$python" && -n "$conda_prefix" ]] || {
  printf 'REFUSE_USAGE: --nemo-cfg, --output-root, --python, and --conda-prefix are required\n' >&2; exit 64; }
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: tool is outside a Git checkout\n' >&2; exit 66; }
source "$here/checkout_env.bash"
swg_dg_bind_checkout "$python"
[[ -x "$conda_prefix/bin/mpirun" ]] || {
  printf 'REFUSE_TOOLCHAIN: mpirun missing under %s/bin\n' "$conda_prefix" >&2; exit 66; }
export PATH="$conda_prefix/bin:$PATH"
mpiexec=${mpiexec:-$conda_prefix/bin/mpirun}
[[ -d "$nemo_cfg" && -x "$nemo_cfg/BLD/bin/nemo.exe" \
   && -f "$nemo_cfg/EXP00/namelist_cfg" && -f "$nemo_cfg/EXP00/namelist_ref" ]] || {
  printf 'REFUSE_NEMO_CFG: incomplete built case %s\n' "$nemo_cfg" >&2; exit 66; }

case "$mode" in
  generic)
    [[ ${#positionals[@]} -ge 4 && ${#positionals[@]} -le 6 ]] || {
      printf 'REFUSE_USAGE: generic mode needs label nn_SWG dt_s steps [ranks] [nn_stock]\n' >&2; exit 64; }
    label=${positionals[0]}; nswg=${positionals[1]}; dt=${positionals[2]}; steps=${positionals[3]}
    ranks=${positionals[4]:-32}; stock=${positionals[5]:-$steps}
    ;;
  smoke)
    [[ ${#positionals[@]} -ge 1 && ${#positionals[@]} -le 2 ]] || {
      printf 'REFUSE_USAGE: --smoke needs label [ranks]\n' >&2; exit 64; }
    label=${positionals[0]}; ranks=${positionals[1]:-32}
    nswg=16; dt=50; steps=51840; stock=51840
    ;;
  three-year)
    [[ ${#positionals[@]} -ge 1 && ${#positionals[@]} -le 2 ]] || {
      printf 'REFUSE_USAGE: --three-year needs label [ranks]\n' >&2; exit 64; }
    label=${positionals[0]}; ranks=${positionals[1]:-32}
    nswg=16; dt=50; steps=1892160; stock=630720
    ;;
  short-ladder)
    [[ ${#positionals[@]} -eq 1 ]] || {
      printf 'REFUSE_USAGE: --short-ladder needs label\n' >&2; exit 64; }
    label=${positionals[0]}; ranks=1
    nswg=16; dt=50; steps=10; stock=1
    ;;
esac
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$label" != "." && "$label" != ".." ]] || {
  printf 'REFUSE_LABEL: expected one safe output-directory component, got %s\n' "$label" >&2; exit 64; }
for pair in "nn_SWG:$nswg" "steps:$steps" "ranks:$ranks" "nn_stock:$stock"; do
  name=${pair%%:*}; value=${pair#*:}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    printf 'REFUSE_INTEGER: %s=%s\n' "$name" "$value" >&2; exit 64; }
done
[[ "$dt" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf 'REFUSE_DT: %s\n' "$dt" >&2; exit 64; }
cadence=$("$python" -I -c 'import sys; q=86400/float(sys.argv[1]); assert q.is_integer(); print(int(q))' "$dt") || {
  printf 'REFUSE_DAILY_CLOCK: 86400/dt must be an integer\n' >&2; exit 68; }
dt_render=$("$python" -I -c 'import sys; print(format(float(sys.argv[1]), ".17g"))' "$dt")
[[ "$steps" -ge "$stock" ]] || {
  printf 'REFUSE_STOCK: nn_stock exceeds steps\n' >&2; exit 64; }

deck_patch=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/namelist_cfg_maddison_3yr.patch
[[ -f "$deck_patch" ]] || { printf 'REFUSE_MISSING_DECK: %s\n' "$deck_patch" >&2; exit 67; }
run_dir=$output_root/$label
[[ ! -e "$run_dir" ]] || { printf 'REFUSE_OUTPUT_EXISTS: %s\n' "$run_dir" >&2; exit 69; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$nemo_cfg/EXP00/namelist_cfg" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$deck_patch"
sed -i -E \
  -e "s/^(\s*nn_SWG\s*=\s*)[0-9]+/\1${nswg}/" \
  -e "s/^(\s*rn_Dt\s*=\s*)[0-9.]+/\1${dt_render}/" \
  -e "s/^(\s*nn_itend\s*=\s*)[0-9]+/\1${steps}/" \
  -e "s/^(\s*nn_write\s*=\s*)[0-9]+/\1${cadence}/" \
  -e "s/^(\s*nn_stock\s*=\s*)[0-9]+/\1${stock}/" \
  "$dry/namelist_cfg"
for expected in \
  "rn_domsiz:4000000" "rn_tau:0.10" "rn_modified_grav:9.80665" \
  "rn_rfr:1.e-7" "rn_beta:2.e-11" "rn_f0:0.7e-4" "rn_Uv:2.0" "rn_Lv:10"; do
  key=${expected%%:*}; value=${expected#*:}
  grep -qE "^[[:space:]]*$key[[:space:]]*=[[:space:]]*$value([[:space:],!]|$)" "$dry/namelist_cfg" || {
    printf 'REFUSE_RESOLVED_DECK: %s=%s missing\n' "$key" "$value" >&2; exit 68; }
done
printf 'PREFLIGHT label=%s nn_SWG=%s dt=%s steps=%s ranks=%s nn_stock=%s nn_write=%s\n' \
  "$label" "$nswg" "$dt" "$steps" "$ranks" "$stock" "$cadence"
printf 'WOULD_RUN: %s -np %s --oversubscribe ./nemo\n' "$mpiexec" "$ranks"
if [[ "$execute" -ne 1 ]]; then
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$run_dir"
cp "$dry/namelist_cfg" "$nemo_cfg/EXP00/namelist_ref" "$run_dir/"
cp "$deck_patch" "$run_dir/committed_maddison_deck.patch"
cp "$nemo_cfg/BLD/bin/nemo.exe" "$run_dir/nemo"
for xml in "$nemo_cfg"/EXP00/*.xml; do
  [[ ! -e "$xml" ]] || cp -L "$xml" "$run_dir/"
done
cd "$run_dir"
grep -nE 'nn_SWG|rn_domsiz|rn_tau|rn_modified_grav|rn_rfr|rn_beta|rn_f0|rn_Uv|rn_Lv|rn_Dt|nn_itend|nn_write|nn_stock|ln_meshmask|l_runstat' \
  namelist_cfg | grep -vE '^\S*:\s*!' >resolved_namelist_lines.txt
{
  sha256sum nemo namelist_cfg namelist_ref
  sha256sum committed_maddison_deck.patch
} >manifest.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECONDS=0
printf 'RUN_STARTED_UTC=%s\n' "$started" >run.log
"$mpiexec" -np "$ranks" --oversubscribe ./nemo >>run.log 2>&1
printf 'WALL_SECONDS=%s\nRUN_FINISHED_UTC=%s\n' "$SECONDS" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.log
[[ -s time.step && -s run.stat ]] || {
  printf 'REFUSE_INCOMPLETE_RUN: time.step or run.stat missing\n' >&2; exit 70; }
completed_step=$(tr -d '[:space:]' <time.step)
runstat_step=$(awk '$1 == "it" && $2 == ":" { step=$3 } END { print step }' run.stat)
[[ "$completed_step" == "$steps" && "$runstat_step" == "$steps" ]] || {
  printf 'REFUSE_INCOMPLETE_RUN: requested=%s time.step=%s run.stat=%s\n' \
    "$steps" "${completed_step:-missing}" "${runstat_step:-missing}" >&2
  exit 70
}
if [[ "$mode" == smoke || "$mode" == three-year ]]; then
  [[ -s output.namelist.dyn ]] || {
    printf 'REFUSE_INCOMPLETE_RUN: output.namelist.dyn missing\n' >&2; exit 70; }
  model_days=$((steps / cadence))
  deck_guard=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_deck_guard.py
  "$python" "$deck_guard" --nemo-run-dir . \
    --days "$model_days" --json deck_guard.json
  sha256sum deck_guard.json >>manifest.sha256
fi
printf 'NEMO_RUN_COMPLETE label=%s steps=%s ranks=%s wall_seconds=%s\n' \
  "$label" "$steps" "$ranks" "$SECONDS"
