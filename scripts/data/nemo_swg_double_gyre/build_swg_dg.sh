#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: build_swg_dg.sh --nemo-root DIR --conda-prefix DIR --manifest-dir DIR --python PYTHON [--arch-name NAME] [--execute]

Default mode is a read-only preflight. --execute creates tests/SWG_DG and
renders arch/arch-NAME.fcm inside the supplied NEMO checkout. No path defaults
to a developer machine.
EOF
}

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE_UNHANDLED: build_swg_dg.sh line %s exited %s\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

nemo_root=
conda_prefix=
manifest_dir=
python=
arch_name=swg-dg-scalarmath
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo-root) nemo_root=${2-}; shift ;;
    --conda-prefix) conda_prefix=${2-}; shift ;;
    --manifest-dir) manifest_dir=${2-}; shift ;;
    --python) python=${2-}; shift ;;
    --arch-name) arch_name=${2-}; shift ;;
    --execute) execute=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'REFUSE_USAGE: unknown argument %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

for value in nemo_root conda_prefix manifest_dir python; do
  [[ -n ${!value} ]] || { printf 'REFUSE_USAGE: --%s is required\n' "${value//_/-}" >&2; exit 64; }
done
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: tool is outside a Git checkout\n' >&2; exit 66; }
source "$here/checkout_env.bash"
swg_dg_bind_checkout "$python"
[[ -d "$nemo_root" && -x "$nemo_root/makenemo" ]] || {
  printf 'REFUSE_NEMO_ROOT: missing executable makenemo under %s\n' "$nemo_root" >&2; exit 66; }
[[ -d "$conda_prefix" ]] || {
  printf 'REFUSE_CONDA_PREFIX: not a directory: %s\n' "$conda_prefix" >&2; exit 66; }
[[ -x "$conda_prefix/bin/mpif90" && -x "$conda_prefix/bin/gfortran" ]] || {
  printf 'REFUSE_TOOLCHAIN: mpif90/gfortran missing under %s/bin\n' "$conda_prefix" >&2; exit 66; }
# The architecture's %AR is a plain `ar`; the conda environment provides only
# prefixed archivers, so the host's GNU binutils `ar` builds the NEMO library.
archiver=$(PATH="$conda_prefix/bin:$PATH" command -v ar) || {
  printf 'REFUSE_ARCHIVER: no ar on PATH; install the host GNU binutils\n' >&2; exit 66; }
[[ "$arch_name" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'REFUSE_ARCH_NAME: %s\n' "$arch_name" >&2; exit 64; }

source_case=$nemo_root/tests/SWG
target_case=$nemo_root/tests/SWG_DG
arch_template=$here/arch-conda-scalarmath.fcm.in
arch_path=$nemo_root/arch/arch-$arch_name.fcm
source_patch=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/usrdef_zgr_domsiz.patch

for path in "$source_case/cpp_SWG.fcm" "$source_case/EXPREF/namelist_cfg" \
            "$source_case/MY_SRC/usrdef_zgr.F90" "$arch_template" "$source_patch"; do
  [[ -f "$path" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
keys=$(sed -E 's/.*fppkeys *//' "$source_case/cpp_SWG.fcm" | xargs)
[[ "$keys" == 'key_xios key_qco key_vco_3d key_RK3' ]] || {
  printf 'REFUSE_SHIPPED_KEYS: %s\n' "$keys" >&2; exit 68; }
[[ "$(sha256sum "$source_patch" | awk '{print $1}')" == \
   c3af0880a1beb44e7ab40027aa8012d11f1e43fc97127d6cc27d50a44fcab08b ]] || {
  printf 'REFUSE_SOURCE_PATCH_HASH: %s\n' "$source_patch" >&2; exit 68; }
[[ ! -e "$target_case" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$target_case" >&2; exit 69; }
[[ ! -e "$arch_path" ]] || {
  printf 'REFUSE_ARCH_EXISTS: %s\n' "$arch_path" >&2; exit 69; }
[[ ! -e "$manifest_dir" ]] || {
  printf 'REFUSE_MANIFEST_EXISTS: %s\n' "$manifest_dir" >&2; exit 69; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$source_case/MY_SRC/usrdef_zgr.F90" "$dry/usrdef_zgr.F90"
patch -s --fuzz=0 "$dry/usrdef_zgr.F90" <"$source_patch"
[[ "$(grep -c 'rn_domsiz + 10000._wp.*SWG_DG' "$dry/usrdef_zgr.F90")" -eq 2 ]] || {
  printf 'REFUSE_PATCH_RESULT: expected two resolved-domain limits\n' >&2; exit 68; }
sed "s|@CONDA_PREFIX@|$conda_prefix|g" "$arch_template" >"$dry/arch-$arch_name.fcm"
[[ "$(grep -c '@CONDA_PREFIX@' "$dry/arch-$arch_name.fcm" || true)" -eq 0 ]] || {
  printf 'REFUSE_ARCH_RENDER: unresolved prefix token\n' >&2; exit 68; }

revision=$(git -C "$nemo_root" rev-parse HEAD 2>/dev/null) || {
  printf 'REFUSE_NEMO_REVISION: %s is not a Git checkout\n' "$nemo_root" >&2; exit 68; }
expected_revision=dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
[[ "$revision" == "$expected_revision" ]] || {
  printf 'REFUSE_NEMO_REVISION: expected %s, got %s\n' "$expected_revision" "$revision" >&2; exit 68; }
[[ -z "$(git -C "$nemo_root" status --porcelain --untracked-files=no)" ]] || {
  printf 'REFUSE_DIRTY_NEMO: tracked changes under %s\n' "$nemo_root" >&2; exit 68; }
printf 'PREFLIGHT NEMO=%s revision=%s source=SWG target=SWG_DG arch=%s ar=%s\n' \
  "$nemo_root" "$revision" "$arch_name" "$archiver"
printf 'WOULD_BUILD: ./makenemo -a SWG -n SWG_DG -m %s del_key key_xios\n' "$arch_name"
if [[ "$execute" -ne 1 ]]; then
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$manifest_dir"
cp "$dry/arch-$arch_name.fcm" "$arch_path"
export PATH="$conda_prefix/bin:$PATH"
cd "$nemo_root"
./makenemo -a SWG -n SWG_DG -m "$arch_name" del_key key_xios
cp -r "$source_case/EXPREF/." "$target_case/EXP00/"
cp -r "$source_case/MY_SRC/." "$target_case/MY_SRC/"
patch -s --fuzz=0 "$target_case/MY_SRC/usrdef_zgr.F90" <"$source_patch"
touch "$target_case"/MY_SRC/*.F90
./makenemo -n SWG_DG -m "$arch_name"
[[ -x "$target_case/BLD/bin/nemo.exe" ]] || {
  printf 'REFUSE_BUILD_BINARY: %s\n' "$target_case/BLD/bin/nemo.exe" >&2; exit 70; }
[[ "$(grep -c 'rn_domsiz + 10000._wp.*SWG_DG' \
       "$target_case/BLD/ppsrc/nemo/usrdef_zgr.f90")" -eq 2 ]] || {
  printf 'REFUSE_COMPILED_PATCH: patched limits absent\n' >&2; exit 70; }

mkdir -p "$manifest_dir/artifacts/MY_SRC"
# The deck guard is exhaustive only for the namelist reads of the committed
# inventory; derive this build's reads and refuse a build that differs.
"$python" -I "$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_namelist_inventory.py" \
  --source-root "$target_case/BLD/ppsrc/nemo" \
  --json "$manifest_dir/artifacts/compiled_namelist_reads.json"
cp "$target_case/BLD/bin/nemo.exe" "$manifest_dir/artifacts/nemo.exe"
cp "$source_patch" "$manifest_dir/artifacts/usrdef_zgr_domsiz.patch"
cp "$arch_path" "$manifest_dir/artifacts/arch-$arch_name.fcm"
cp "$target_case"/MY_SRC/*.F90 "$manifest_dir/artifacts/MY_SRC/"
(
  cd "$manifest_dir"
  find artifacts -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_dir/swg_dg_build_manifest.sha256"
{
  printf 'schema=swg-dg-build-manifest-v1\n'
  printf 'nemo_revision=%s\n' "$revision"
  printf 'nemo_version=5.0.2\n'
  printf 'source_case=SWG\n'
  printf 'target_case=SWG_DG\n'
  printf 'arch_name=%s\n' "$arch_name"
  printf 'hash_inventory_root=%s\n' "$manifest_dir"
  printf 'hash_inventory_check=cd %s && sha256sum -c swg_dg_build_manifest.sha256\n' "$manifest_dir"
  printf 'cpp=cpp -Dkey_nosignedzero\n'
  printf 'fcflags=-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize\n'
  printf 'cflags=-O0\n'
  printf 'gfortran=%s\n' "$(gfortran --version | head -1)"
  printf 'mpi=%s\n' "$(mpirun --version | head -1)"
  printf 'archiver=%s %s\n' "$archiver" "$(ar --version | head -1)"
} >"$manifest_dir/swg_dg_build_manifest.txt"
printf 'BUILD_DONE %s\n' "$target_case"
