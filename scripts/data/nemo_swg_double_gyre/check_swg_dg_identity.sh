#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: check_swg_dg_identity.sh --nemo-root DIR --arch-name NAME --output DIR --python PYTHON --conda-prefix DIR [--mpiexec CMD] [--execute]

Default mode is a read-only preflight. --execute builds an unmodified shipped
SWG reference with the selected arch, runs both reference and SWG_DG for ten
steps on one rank, and bit-compares every variable in the four grid outputs.
--conda-prefix is the NEMO build environment: its bin/ is put first on PATH
for makenemo, and its Open MPI mpirun is the default launcher. --mpiexec may
name a site launcher for that same Open MPI (the tool adds --oversubscribe,
an Open MPI option).
EOF
}

nemo_root=
arch_name=
output=
mpiexec=
python=
conda_prefix=
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo-root) nemo_root=${2-}; shift ;;
    --arch-name) arch_name=${2-}; shift ;;
    --output) output=${2-}; shift ;;
    --mpiexec) mpiexec=${2-}; shift ;;
    --python) python=${2-}; shift ;;
    --conda-prefix) conda_prefix=${2-}; shift ;;
    --execute) execute=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'REFUSE_USAGE: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
  shift
done
[[ -n "$nemo_root" && -n "$arch_name" && -n "$output" && -n "$python" && -n "$conda_prefix" ]] || {
  printf 'REFUSE_USAGE: --nemo-root, --arch-name, --output, --python, and --conda-prefix are required\n' >&2; exit 64; }
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: tool is outside a Git checkout\n' >&2; exit 66; }
source "$here/checkout_env.bash"
swg_dg_bind_checkout "$python"
[[ -x "$conda_prefix/bin/mpif90" && -x "$conda_prefix/bin/mpirun" ]] || {
  printf 'REFUSE_TOOLCHAIN: mpif90/mpirun missing under %s/bin\n' "$conda_prefix" >&2; exit 66; }
export PATH="$conda_prefix/bin:$PATH"
archiver=$(command -v ar) || {
  printf 'REFUSE_ARCHIVER: no ar on PATH; install the host GNU binutils\n' >&2; exit 66; }
mpiexec=${mpiexec:-$conda_prefix/bin/mpirun}
variant=$nemo_root/tests/SWG_DG
shipped=$nemo_root/tests/SWG
for path in "$nemo_root/makenemo" "$shipped/EXPREF/namelist_cfg" \
            "$shipped/MY_SRC/usrdef_zgr.F90" "$variant/BLD/bin/nemo.exe" \
            "$nemo_root/arch/arch-$arch_name.fcm"; do
  [[ -e "$path" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
[[ ! -e "$output" ]] || { printf 'REFUSE_OUTPUT_EXISTS: %s\n' "$output" >&2; exit 69; }

deck_patch=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/namelist_cfg_kt10.patch
mesh_patch=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/namelist_meshmask.patch
checker=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/check_records.py
for path in "$deck_patch" "$mesh_patch" "$checker"; do
  [[ -f "$path" ]] || { printf 'REFUSE_MISSING_TOOL: %s\n' "$path" >&2; exit 67; }
done
printf 'PREFLIGHT identity=shipped-SWG-versus-SWG_DG steps=10 ranks=1 arch=%s mpiexec=%s ar=%s\n' \
  "$arch_name" "$mpiexec" "$archiver"
printf 'WOULD_BUILD: ./makenemo -a SWG -n SWG_IDENTITY_REF -m %s del_key key_xios\n' "$arch_name"
if [[ "$execute" -ne 1 ]]; then
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

reference_cfg=$nemo_root/tests/SWG_IDENTITY_REF
[[ ! -e "$reference_cfg" ]] || {
  printf 'REFUSE_REFERENCE_CFG_EXISTS: %s\n' "$reference_cfg" >&2; exit 69; }
mkdir -p "$output/reference" "$output/variant"
cd "$nemo_root"
./makenemo -a SWG -n SWG_IDENTITY_REF -m "$arch_name" del_key key_xios
cp -r "$shipped/EXPREF/." "$reference_cfg/EXP00/"
cp -r "$shipped/MY_SRC/." "$reference_cfg/MY_SRC/"
touch "$reference_cfg"/MY_SRC/*.F90
./makenemo -n SWG_IDENTITY_REF -m "$arch_name"

for arm in reference variant; do
  if [[ "$arm" == reference ]]; then binary=$reference_cfg/BLD/bin/nemo.exe; else binary=$variant/BLD/bin/nemo.exe; fi
  cp -rL "$shipped/EXPREF/." "$output/$arm/"
  patch -s --fuzz=0 "$output/$arm/namelist_cfg" <"$deck_patch"
  patch -s --fuzz=0 "$output/$arm/namelist_cfg" <"$mesh_patch"
  cp "$binary" "$output/$arm/nemo"
  (
    cd "$output/$arm"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    "$mpiexec" -np 1 --oversubscribe ./nemo >run.log 2>&1
  )
done
"$python" -I "$checker" --outputs-only \
  --reference "$output/reference" --evidence "$output/variant" \
  --expected-variable-count 41 --expected-completed-step 10 \
  --json "$output/swg_dg_identity.json"
