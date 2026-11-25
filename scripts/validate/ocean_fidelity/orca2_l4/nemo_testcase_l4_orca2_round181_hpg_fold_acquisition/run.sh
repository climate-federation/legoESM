#!/usr/bin/env bash
# Operator-run acquisition of the actual fold-side operands evaluated by HPG.
set -Eeuo pipefail
trap 's=$?; printf "REFUSE: round-182 acquisition failed at line %s (exit %s)\n" "${BASH_LINENO[0]:-?}" "$s" >&2; exit "$s"' ERR

mode=${1:---run}
case "$mode" in --run|--preflight-only|--admit-existing|--plant-layout|--plant-stage-identity) ;; *)
  printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout|--plant-stage-identity]\n' "$0" >&2; exit 63;; esac
export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly nemo=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly source_cfg=ORCA2_OMIP_L4_R180HPG1 target_cfg=ORCA2_OMIP_L4_R182HPGFOLD
readonly source_root=$nemo/cfgs/$source_cfg target_root=$nemo/cfgs/$target_cfg
readonly source_run=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round180/acquisition/orca2_rung0_hpg1_ranked_10step_np2
readonly evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round182/acquisition
readonly target_run=$evidence/orca2_rung0_hpg_fold_ranked_10step_np2
readonly work=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work
readonly py=/home/dbalwada/legoESM/.venv/bin/python fc=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly patch_file=$here/dynhpg_round181.patch checker=$here/check_record.py prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round182.md

refuse() { printf 'REFUSE: %s\n' "$1" >&2; exit "${2:-64}"; }
pin() { [[ -f "$2" ]] || refuse "missing $3: $2"; [[ "$(sha256sum "$2" | awk '{print $1}')" == "$1" ]] || refuse "$3 hash moved" 65; }
layout() {
  [[ $(grep -Fc "r181_magic = 'NEMO_L4_R181HPF'" "$1") -eq 1 ]] &&
  [[ $(grep -Fc "r181_name = 'north_e3w'" "$1") -eq 1 ]] &&
  [[ $(grep -Fc "r181_name = 'zhpj'" "$1") -eq 1 ]] &&
  [[ $(grep -Fc 'ORCA2_R181_HPG_FOLD_DUMP' "$1") -eq 1 ]]
}
artifact_digest() { sha256sum "$0" "$patch_file" "$checker" "$prereg" | sha256sum | awk '{print $1}'; }

cd "$repo"
[[ -z $(git status --porcelain --untracked-files=all) ]] || refuse 'producer tree is dirty'
for f in "$0" "$patch_file" "$checker" "$prereg"; do git ls-files --error-unmatch "${f#"$repo"/}" >/dev/null || refuse "uncommitted acquisition artifact: $f"; done
pin 0fca1d9b74dec033f96737e4ab6fe883f68d4df10c432ce67daff63bd4d449f6 "$source_root/MY_SRC/dynhpg.F90" 'source dynhpg'
pin 2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67 "$source_root/cpp_$source_cfg.fcm" 'source cpp keys'
pin d8fac9757951d95be7077daa11bbabb19b6b32a08b4bd877f3f3036ad8da6ab9 "$source_root/BLD/bin/nemo.exe" 'source binary'
[[ -x "$source_run/nemo" ]] || refuse 'source run binary missing'
cmp -s "$source_root/BLD/bin/nemo.exe" "$source_run/nemo" || refuse 'source run/build binary mismatch'
[[ $(awk '/^--- /{next} /^-/{n++} END{print n+0}' "$patch_file") -eq 0 ]] || refuse 'writer patch is not additions-only' 66
mkdir -p "$evidence" "$work"
scratch=$(mktemp -d "$work/orca2-r181-hpgfold.XXXXXX")
cp "$source_root/MY_SRC/dynhpg.F90" "$scratch/dynhpg.F90"
git apply --unsafe-paths -p0 --directory="$scratch" "$patch_file"
layout "$scratch/dynhpg.F90" || refuse 'writer layout incomplete' 66
symlink_probe=$(mktemp -d "$work/orca2-r182-symlink-probe.XXXXXX")
mkdir "$symlink_probe/real"
cp "$source_root/MY_SRC/dynhpg.F90" "$symlink_probe/real/dynhpg.F90"
ln -s "$symlink_probe/real" "$symlink_probe/linked"
patch -s --fuzz=0 -p0 -d "$symlink_probe/linked" <"$patch_file"
cmp -s "$scratch/dynhpg.F90" "$symlink_probe/real/dynhpg.F90" || refuse 'symlink-parent patch proof differs from syntax-proved source' 66
printf 'PATCH_SYMLINK_PARENT_PROOF_PASS dynhpg.F90\n'
if [[ "$mode" == --plant-layout ]]; then
  sed -i "/r181_name = 'north_e3w'/d" "$scratch/dynhpg.F90"
  layout "$scratch/dynhpg.F90" && refuse 'layout plant stayed green' 69
  printf 'STATUS PLANT-FIRED layout\n'; exit 69
fi
if [[ "$mode" == --plant-stage-identity ]]; then
  cp "$scratch/dynhpg.F90" "$scratch/staged_dynhpg.F90"
  printf '! planted staged-source delta\n' >>"$scratch/staged_dynhpg.F90"
  cmp -s "$scratch/dynhpg.F90" "$scratch/staged_dynhpg.F90" && refuse 'stage-identity plant stayed green' 69
  printf 'STATUS PLANT-FIRED stage-identity\n'; exit 69
fi
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I "$source_root/WORK" -I "$source_root/BLD/inc" "$scratch/dynhpg.F90" -o "$scratch/dynhpg.f90"
"$fc" -fsyntax-only -ffree-line-length-none -I "$source_root/BLD/inc" -J "$scratch" "$scratch/dynhpg.f90"
printf 'SYNTAX_PROOF_PASS dynhpg.f90\n'
[[ "$mode" != --preflight-only ]] || { printf 'ORCA2_ROUND182_HPG_FOLD_PREFLIGHT_READY %s\n' "$target_run"; exit 0; }

admit() {
  [[ -f "$target_run/producer_content.sha256" ]] || refuse 'target content stamp missing' 70
  [[ $(cat "$target_run/producer_content.sha256") == "$(artifact_digest)" ]] || refuse 'producer content changed' 70
  [[ $(find "$target_run" -maxdepth 1 -name 'oracle_r181_hpgfold_rank????_kt00000001.bin' | wc -l) -eq 2 ]] || refuse 'expected two HPG fold records' 70
  for p in header field-name field-dims truncation restart-byte; do
    "$py" "$checker" --root "$target_run" --baseline "$source_run" --plant "$p" >"$target_run/round181_${p}_plant.log" 2>&1 && refuse "$p plant stayed green" 71
    grep -Fq 'STATUS PLANT-FIRED' "$target_run/round181_${p}_plant.log" || refuse "$p plant lacks marker" 71
  done
  "$py" "$checker" --root "$target_run" --baseline "$source_run" --output "$target_run/round181_hpg_fold_admission.json"
  printf 'ORCA2_ROUND182_HPG_FOLD_ACQUISITION_PASS %s\n' "$target_run"
}
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
[[ ! -e "$target_root" && ! -e "$target_run" ]] || refuse 'target config or run directory already exists' 68
for m in "$work" "$evidence" "$nemo"; do [[ $(df -Pk "$m" | awk 'NR==2{print $4}') -ge 4194304 ]] || refuse "$m has under 4 GiB free" 67; done

cd "$nemo"
./makenemo -r ORCA2_ICE_PISCES -n "$target_cfg" -m conda-scalarmath del_key key_xios
find "$source_root/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | while IFS= read -r -d '' f; do cp -a "$f" "$target_root/EXP00/$(basename "$f")"; done
find "$source_root/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | while IFS= read -r -d '' f; do cp -a "$f" "$target_root/MY_SRC/$(basename "$f")"; done
cp "$source_root/cpp_$source_cfg.fcm" "$target_root/cpp_$target_cfg.fcm"
patch -s --fuzz=0 -p0 -d "$target_root/MY_SRC" <"$patch_file"
cmp -s "$scratch/dynhpg.F90" "$target_root/MY_SRC/dynhpg.F90" || refuse 'staged writer differs from syntax-proved source' 68
touch "$target_root/MY_SRC/"*.F90
./makenemo -n "$target_cfg" -m conda-scalarmath del_key key_xios
layout "$target_root/BLD/ppsrc/nemo/dynhpg.f90" || refuse 'compiled writer incomplete' 69
mkdir "$target_run"
while read -r _ name; do cp -a "$source_run/$name" "$target_run/$name"; done <"$source_run/deck_files.sha256"
while read -r _ name; do cp -a "$source_run/$name" "$target_run/$name"; done <"$source_run/input_files.sha256"
cp "$source_run/deck_files.sha256" "$source_run/input_files.sha256" "$target_run/"
cp "$target_root/BLD/bin/nemo.exe" "$target_run/nemo"
artifact_digest >"$target_run/producer_content.sha256"
(cd "$target_run"; set +e; mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log; rc=${PIPESTATUS[0]}; set -e; [[ $rc -eq 0 ]] || refuse "NEMO failed: $rc" 69; grep -Fxq 'STOP 0' run.user.stdout.log || refuse 'run lacks STOP 0' 69)
admit
