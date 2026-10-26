#!/usr/bin/env bash
# Operator-only extension of the reviewed round-333 acquisition driver.
# The default is a read-only preflight. Codex never invokes --run.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}
run_in_dir() {
  local directory=$1; shift
  ( cd "$directory" && "$@" )
}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PARENT_DIR=$here/../nemo_testcase_l4_orca2_round333_tracer_acquisition
readonly PARENT=$PARENT_DIR/run.sh
readonly SOURCE_CFG=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI
readonly SOURCE_PPSRC=$SOURCE_CFG/BLD/ppsrc/nemo/trazdf.f90
readonly SOURCE_ZDFPHY_PPSRC=$SOURCE_CFG/BLD/ppsrc/nemo/zdfphy.f90
readonly BASE_ZDFPHY_PPSRC=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90
readonly SOURCE_OBJECT=$SOURCE_CFG/BLD/obj/trazdf.o
readonly SOURCE_MODULES=$SOURCE_CFG/BLD/inc
readonly SOURCE_OUTPUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2/ocean.output
readonly SOURCE_NAMELIST=${SOURCE_OUTPUT%/*}/namelist_cfg
readonly NAMELIST_REF=$SOURCE_CFG/EXP00/namelist_ref
readonly TRAZDF_SOURCE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/trazdf.F90
readonly ZDF_OCE_SOURCE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/ZDF/zdf_oce.F90
readonly PATCH=$here/trazdf_round336.patch
readonly ZDFPHY_PATCH=$here/zdfphy_round336.patch
readonly ZDFPHY_PPSRC_PATCH=$here/zdfphy_ppsrc_round336.patch
readonly WRITER=$here/round336_trazdf_writer.F90
readonly FIXTURE_STUBS=$here/round336_fixture_stubs.F90
readonly FIXTURE=$here/round336_writer_fixture.F90
readonly PARENT_EXTENSION=$here/round336_parent_extension.sh.inc
readonly NAMELIST_GATE=$here/round336_namelist_gate.py
readonly GATE=$here/../nemo_testcase_l4_orca2_round336_trazdf_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round336_trazdf_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_trazdf_internal_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly MPIFC=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpif90
readonly NEMO_BIN=/home/dbalwada/miniconda3/envs/nemo-build/bin
readonly FPP_KEYS='-Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3'

for path in "$PARENT" "$SOURCE_PPSRC" "$SOURCE_ZDFPHY_PPSRC" "$BASE_ZDFPHY_PPSRC" "$SOURCE_OBJECT" \
  "$SOURCE_OUTPUT" "$SOURCE_NAMELIST" "$NAMELIST_REF" "$TRAZDF_SOURCE" "$ZDF_OCE_SOURCE" \
  "$PATCH" "$ZDFPHY_PATCH" "$ZDFPHY_PPSRC_PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" \
  "$PARENT_EXTENSION" "$NAMELIST_GATE" "$GATE" "$TEST"; do
  [[ -f "$path" ]] || refuse "missing acquisition input $path"
done
[[ -d "$SOURCE_MODULES" ]] || refuse "missing record-build module directory $SOURCE_MODULES"
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'

mkdir -p "$EVIDENCE"
driver_dir=$(mktemp -d "$EVIDENCE/.r336-driver.XXXXXX")
retention_armed=0
cleanup() {
  local rc=$? refused
  trap - EXIT
  set +e
  if [[ $rc -ne 0 && $retention_armed -ne 0 && -d "$TARGET_RUN" ]]; then
    refused=$EVIDENCE/refused/$(basename "$TARGET_RUN")_admission_refused_$(date -u +%Y%m%dT%H%M%SZ)_$$
    mkdir -p "$EVIDENCE/refused"
    if mv -- "$TARGET_RUN" "$refused"; then
      printf 'R336_FAILED_OUTPUTS_KEPT kind=admission_refused path=%s\n' "$refused" >&2
    else
      printf 'R336_FAILED_OUTPUTS_LEFT_IN_PLACE path=%s\n' "$TARGET_RUN" >&2
    fi
  fi
  [[ ! -d "$driver_dir" ]] || rm -rf -- "$driver_dir"
  exit "$rc"
}
trap cleanup EXIT
readonly driver_dir
readonly ADAPTED_PARENT=$driver_dir/run-parent.sh

check_no_removed_lines() {
  [[ $(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$1") -eq 0 ]]
}
check_real_context() {
  ! grep -Eq '^@@ -[0-9]+,0 ' "$1"
}
echo_has() {
  grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null
}
echo_number_is() {
  awk -v key="$2" -v expected="$3" '
    {
      for(i=1; i<NF-1; i++) if($i==key && $(i+1)=="=") { found++; value=$(i+2) }
    }
    END { exit !(found==1 && value+0==expected+0) }
  ' "$1"
}
check_selectors() {
  echo_has "$1" ln_zad_Aimp F \
    && echo_has "$1" ln_zdfmfc F \
    && echo_has "$1" ln_rnf_mouth T \
    && echo_has "$1" ln_zdftke T \
    && echo_has "$1" ln_zdfevd T \
    && echo_has "$1" ln_zdfddm T \
    && echo_has "$1" ln_zdfswm F \
    && echo_has "$1" ln_zdfiwm T \
    && echo_has "$1" ln_traldf_msc T \
    && echo_number_is "$1" rn_avt_rnf 0.001 \
    && echo_number_is "$1" rn_evd 100.0 \
    && echo_number_is "$1" rn_avts 0.0001 \
    && echo_number_is "$1" rn_hsbfr 1.6
}
check_r336_landing() {
  local source=$1 plant=${2:-none}
  awk -v plant="$plant" '
    /USE round336_trazdf_writer/ {use_count++; use_line=NR}
    /SUBROUTINE tra_zdf\(/ {outer=NR}
    /CALL r336_begin/ {begin_count++; begin=NR}
    /CALL tra_zdf_imp/ {solve_count++; solve=NR}
    /CALL r336_capture/ {capture_count++; capture=NR}
    /SSS control/ {sss=NR}
    /SUBROUTINE tra_zdf_imp/ {inner=NR}
    END {
      if(plant=="use") use_line=outer+1
      if(plant=="begin") begin=solve+1
      if(plant=="capture") capture=solve-1
      if(plant=="inside") inner=solve-1
      ok=(use_count==1 && begin_count==1 && solve_count==1 && capture_count==1)
      ok=ok && use_line<outer && outer<begin && begin<solve && solve<capture && capture<sss && sss<inner
      exit !ok
    }
  ' "$source"
}
export -f check_r336_landing

check_r336_zdfphy_landing() {
  local source=$1 plant=${2:-none}
  awk -v plant="$plant" '
    /SUBROUTINE zdf_phy\(/ {routine=NR}
    /avt\(ji,jj,jk\) = avt_k/ {copy=NR}
    /CALL r336_zdfphy_capture\(kt,1,/ {n1++; p1=NR}
    /IF\( ln_rnf_mouth \) THEN/ {runoff=NR}
    /CALL r336_zdfphy_capture\(kt,2,/ {n2++; p2=NR}
    /CALL zdf_evd/ {evd=NR}
    /CALL r336_zdfphy_capture\(kt,3,/ {n3++; p3=NR}
    /CALL zdf_ddm/ {ddm=NR}
    /CALL r336_zdfphy_capture\(kt,4,/ {n4++; p4=NR}
    /CALL zdf_iwm/ {iwm=NR}
    /CALL r336_zdfphy_capture\(kt,5,/ {n5++; p5=NR}
    /CALL zdf_mxl_turb/ {mxl=NR}
    END {
      if(plant=="tke") p1=copy-1
      if(plant=="runoff") p2=runoff-1
      if(plant=="evd") p3=evd-1
      if(plant=="ddm") p4=ddm-1
      if(plant=="iwm") p5=iwm-1
      ok=(n1==1 && n2==1 && n3==1 && n4==1 && n5==1)
      ok=ok && routine<copy && copy<p1 && p1<runoff && runoff<p2
      ok=ok && p2<evd && evd<p3 && p3<ddm && ddm<p4
      ok=ok && p4<iwm && iwm<p5 && p5<mxl
      exit !ok
    }
  ' "$source"
}
export -f check_r336_zdfphy_landing

refuse_nemo_errors() {
  local output=$1 lines
  [[ -f "$output" ]] || return 1
  lines=$(grep -nF 'E R R O R' "$output" || true)
  [[ -z "$lines" ]] || {
    printf 'NEMO_ERROR_LINES_BEGIN %s\n%s\nNEMO_ERROR_LINES_END\n' "$output" "$lines" >&2
    return 1
  }
}

fp_disassembly_signature() {
  objdump -drwC --no-show-raw-insn "$1" | awk '
    /^[[:xdigit:]]+[[:space:]]+<[^>]+>:/ {
      fn=$2; gsub(/[<>:]/,"",fn); next
    }
    /^[[:space:]]*[[:xdigit:]]+:/ {
      op=$2
      if ((fn=="__trazdf_MOD_tra_zdf" || fn=="__trazdf_MOD_tra_zdf_imp") &&
          op ~ /^(f(add|sub|mul|div|sqrt)|add(s|p)[sd]|sub(s|p)[sd]|mul(s|p)[sd]|div(s|p)[sd]|sqrt(s|p)[sd]|min(s|p)[sd]|max(s|p)[sd])/) print fn,op
    }
  ' | sort | uniq -c >"$2"
  for fn in __trazdf_MOD_tra_zdf __trazdf_MOD_tra_zdf_imp; do
    awk -v fn="$fn" '$2==fn {n+=$1} END {exit !(n>0)}' "$2" || return 1
  done
}
compare_fp_disassembly() {
  local left=$1 right=$2 label=$3
  fp_disassembly_signature "$left" "$driver_dir/${label}.left.fp"
  fp_disassembly_signature "$right" "$driver_dir/${label}.right.fp"
  cmp -s "$driver_dir/${label}.left.fp" "$driver_dir/${label}.right.fp"
}
fp_named_function_signature() {
  local object=$1 output=$2 wanted=$3
  objdump -drwC --no-show-raw-insn "$object" | awk -v wanted="$wanted" '
    /^[[:xdigit:]]+[[:space:]]+<[^>]+>:/ {
      fn=$2; gsub(/[<>:]/,"",fn); next
    }
    /^[[:space:]]*[[:xdigit:]]+:/ {
      op=$2
      if (fn==wanted && op ~ /^(f(add|sub|mul|div|sqrt)|add(s|p)[sd]|sub(s|p)[sd]|mul(s|p)[sd]|div(s|p)[sd]|sqrt(s|p)[sd]|min(s|p)[sd]|max(s|p)[sd])/) print fn,op
    }
  ' | sort | uniq -c >"$output"
  awk -v fn="$wanted" '$2==fn {n+=$1} END {exit !(n>0)}' "$output"
}
check_admission_manifest() {
  local run=$1 current=$2 producer binary_sha
  [[ -f "$run/producer_commit.txt" && $(wc -l <"$run/producer_commit.txt") -eq 1 ]] \
    || return 1
  producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
    "$run/producer_commit.txt")
  [[ $producer =~ ^[0-9a-f]{40}$ ]] || return 1
  git -C "$REPO" cat-file -e "$producer^{commit}" 2>/dev/null || return 1
  git -C "$REPO" merge-base --is-ancestor "$producer" "$current" || return 1
  [[ -f "$run/nemo.sha256" && $(wc -l <"$run/nemo.sha256") -eq 1 ]] \
    || return 1
  binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
    "$run/nemo.sha256")
  [[ $binary_sha =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -f "$run/nemo" && $(sha256sum "$run/nemo" | awk '{print $1}') == "$binary_sha" ]]
}
compile_nemo_object() {
  PATH=$NEMO_BIN:$PATH "$MPIFC" -c -fdefault-real-8 -O3 -funroll-all-loops \
    -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch \
    -fno-tree-vectorize -I "$driver_dir/nemo-mod" -I "$SOURCE_MODULES" \
    -J "$driver_dir/nemo-mod" "$1" -o "$2"
}
preprocess_trazdf() {
  ( cd "$(dirname "$TRAZDF_SOURCE")" && \
    cpp -Dkey_nosignedzero -P -traditional $FPP_KEYS \
      -I "$SOURCE_CFG/WORK" -I "$SOURCE_MODULES" \
      "$(basename "$TRAZDF_SOURCE")" -o "$1" )
}
nonblank_copy() { awk 'NF' "$1" >"$2"; }
check_zdf_oce_interior_allocations() {
  local collapsed
  collapsed=$(tr '\n' ' ' <"$1")
  for name in avs avt avt_k en; do
    grep -Eq "${name}[[:space:]]*\\([[:space:]]*A2D\\(0\\)[[:space:]]*,[[:space:]]*jpk[[:space:]]*\\)" \
      <<<"$collapsed" || return 1
  done
}

for observer_patch in "$PATCH" "$ZDFPHY_PATCH" "$ZDFPHY_PPSRC_PATCH"; do
  check_no_removed_lines "$observer_patch" \
    || refuse "round-336 observer patch removes a source line: $observer_patch"
  check_real_context "$observer_patch" \
    || refuse "round-336 observer patch has a zero-context hunk: $observer_patch"
done
check_selectors "$SOURCE_OUTPUT" || refuse 'record run does not select the registered tra_zdf branches'
for selector in ln_zad_Aimp ln_zdfmfc ln_rnf_mouth ln_zdftke ln_zdfevd \
    ln_zdfddm ln_zdfswm ln_zdfiwm ln_traldf_msc; do
  case "$selector" in
    ln_zad_Aimp|ln_zdfmfc|ln_zdfswm) replacement=T ;;
    *) replacement=F ;;
  esac
  sed -E "s/(${selector}[[:space:]]*=[[:space:]]*)[TF]/\\1$replacement/" \
    "$SOURCE_OUTPUT" >"$driver_dir/${selector}.moved"
  plant_must_refuse "r336-${selector}-moved" check_selectors "$driver_dir/${selector}.moved"
done
for parameter in rn_avt_rnf rn_evd rn_avts rn_hsbfr; do
  sed -E "s/(${parameter}[[:space:]]*=[[:space:]]*)[-+0-9.Ee]+/\1 99/" \
    "$SOURCE_OUTPUT" >"$driver_dir/${parameter}.moved"
  plant_must_refuse "r336-${parameter}-moved" \
    check_selectors "$driver_dir/${parameter}.moved"
done

"$PY" "$NAMELIST_GATE" --staged "$SOURCE_NAMELIST" --reference "$NAMELIST_REF"
plant_must_refuse r336-namelist-misspelled "$PY" "$NAMELIST_GATE" \
  --staged "$SOURCE_NAMELIST" --reference "$NAMELIST_REF" --plant misspelled
plant_must_refuse r336-namelist-misplaced "$PY" "$NAMELIST_GATE" \
  --staged "$SOURCE_NAMELIST" --reference "$NAMELIST_REF" --plant misplaced
printf 'before\nE R R O R staged namelist rejected\nafter\n' >"$driver_dir/ocean.output.error-plant"
plant_must_refuse r336-ocean-output-error-lines \
  refuse_nemo_errors "$driver_dir/ocean.output.error-plant"
check_zdf_oce_interior_allocations "$ZDF_OCE_SOURCE" \
  || refuse 'zdf_oce no longer allocates avt/avs/avt_k/en with A2D(0)'
sed 's/avs(A2D(0),jpk)/avs(jpi,jpj,jpk)/' "$ZDF_OCE_SOURCE" \
  >"$driver_dir/zdf_oce.extent-plant.F90"
plant_must_refuse r336-zdf-oce-a2d-extent \
  check_zdf_oce_interior_allocations "$driver_dir/zdf_oce.extent-plant.F90"

source_check=$driver_dir/source-check
mkdir "$source_check"
cp "$TRAZDF_SOURCE" "$source_check/trazdf.F90"
patch --batch -s --fuzz=0 -p0 -d "$source_check" <"$PATCH"
check_r336_landing "$source_check/trazdf.F90" \
  || refuse 'tra_zdf observers do not bracket the executing solve outside its loops'
for landing in use begin capture inside; do
  plant_must_refuse "r336-${landing}-misplaced" \
    check_r336_landing "$source_check/trazdf.F90" "$landing"
done
[[ $(grep -Fc 'CALL r336_begin' "$source_check/trazdf.F90") -eq 1 \
   && $(grep -Fc 'CALL r336_capture' "$source_check/trazdf.F90") -eq 1 ]] \
  || refuse 'tra_zdf observer call census moved'

cp "$SOURCE_ZDFPHY_PPSRC" "$source_check/zdfphy.F90"
patch --batch -s --fuzz=0 -p0 -d "$source_check" <"$ZDFPHY_PPSRC_PATCH"
check_r336_zdfphy_landing "$source_check/zdfphy.F90" \
  || refuse 'zdf_phy observers do not bracket the five executing contributions'
for landing in tke runoff evd ddm iwm; do
  plant_must_refuse "r336-zdfphy-${landing}-misplaced" \
    check_r336_zdfphy_landing "$source_check/zdfphy.F90" "$landing"
done
[[ $(grep -Fc 'CALL r336_zdfphy_capture' "$source_check/zdfphy.F90") -eq 5 ]] \
  || refuse 'zdf_phy contribution observer call census moved'

# Compile the actual record-build ppsrc with its production arithmetic flags.
# The normalized census ignores the two observer calls and must be identical.
mkdir "$driver_dir/nemo-mod"
preprocess_trazdf "$driver_dir/trazdf-cpp.f90"
nonblank_copy "$driver_dir/trazdf-cpp.f90" "$driver_dir/trazdf-cpp.nonblank"
nonblank_copy "$SOURCE_PPSRC" "$driver_dir/trazdf-record.nonblank"
cmp -s "$driver_dir/trazdf-cpp.nonblank" "$driver_dir/trazdf-record.nonblank" \
  || refuse 'build-cpp trazdf differs from the record-build ppsrc'
sed '0,/SUBROUTINE tra_zdf(/s//SUBROUTINE tra_zdf_planted(/' \
  "$driver_dir/trazdf-cpp.nonblank" >"$driver_dir/trazdf-cpp.plant.nonblank"
plant_must_refuse r336-build-cpp-same-program cmp -s \
  "$driver_dir/trazdf-cpp.plant.nonblank" "$driver_dir/trazdf-record.nonblank"
cp "$driver_dir/trazdf-cpp.f90" "$driver_dir/trazdf-base.f90"
cp "$driver_dir/trazdf-cpp.f90" "$driver_dir/trazdf.F90"
patch --batch -s --fuzz=0 -p0 -d "$driver_dir" <"$PATCH"
check_r336_landing "$driver_dir/trazdf.F90" \
  || refuse 'tra_zdf observer placement moved in record-build ppsrc'
compile_nemo_object "$WRITER" "$driver_dir/round336-writer.o"
compile_nemo_object "$driver_dir/trazdf-base.f90" "$driver_dir/trazdf-base.o"
compile_nemo_object "$driver_dir/trazdf.F90" "$driver_dir/trazdf-observed.o"
compare_fp_disassembly "$driver_dir/trazdf-base.o" "$driver_dir/trazdf-observed.o" trazdf \
  || refuse 'tra_zdf observer changed normalized floating-point instruction census'
printf 'R336_TRAZDF_NORMALIZED_OBJECT_DISASSEMBLY_PASS fp=%s\n' \
  "$(awk '{n+=$1} END {print n+0}' "$driver_dir/trazdf.left.fp")"
for fn in __trazdf_MOD_tra_zdf __trazdf_MOD_tra_zdf_imp; do
  printf 'R336_TRAZDF_FUNCTION_FP_PASS function=%s fp=%s\n' "$fn" \
    "$(awk -v fn="$fn" '$2==fn {n+=$1} END {print n+0}' "$driver_dir/trazdf.left.fp")"
done
sed "0,/CALL tra_zdf_imp( 'TRA', rDt/s//CALL tra_zdf_imp( 'TRA', rDt + 1._wp/" \
  "$driver_dir/trazdf-base.f90" >"$driver_dir/trazdf-arithmetic-plant.f90"
compile_nemo_object "$driver_dir/trazdf-arithmetic-plant.f90" "$driver_dir/trazdf-arithmetic-plant.o"
plant_must_refuse r336-trazdf-arithmetic-disassembly compare_fp_disassembly \
  "$driver_dir/trazdf-base.o" "$driver_dir/trazdf-arithmetic-plant.o" trazdf-plant

# The zdf_phy observer uses the same production flags and an observer-free
# record-build ppsrc baseline. Count only the executing zdf_phy function.
compile_nemo_object "$BASE_ZDFPHY_PPSRC" "$driver_dir/zdfphy-base.o"
compile_nemo_object "$source_check/zdfphy.F90" "$driver_dir/zdfphy-observed.o"
fp_named_function_signature "$driver_dir/zdfphy-base.o" \
  "$driver_dir/zdfphy-base.fp" __zdfphy_MOD_zdf_phy \
  || refuse 'observer-free zdf_phy function has a zero FP census'
fp_named_function_signature "$driver_dir/zdfphy-observed.o" \
  "$driver_dir/zdfphy-observed.fp" __zdfphy_MOD_zdf_phy \
  || refuse 'instrumented zdf_phy function has a zero FP census'
cmp -s "$driver_dir/zdfphy-base.fp" "$driver_dir/zdfphy-observed.fp" \
  || refuse 'zdf_phy contribution observers changed its FP instruction census'
printf 'R336_ZDFPHY_FUNCTION_FP_PASS function=__zdfphy_MOD_zdf_phy fp=%s\n' \
  "$(awk '{n+=$1} END {print n+0}' "$driver_dir/zdfphy-base.fp")"
sed '0,/avt(ji,jj,jk) = avt_k/s//avt(ji,jj,jk) = 1._wp + avt_k/' \
  "$BASE_ZDFPHY_PPSRC" >"$driver_dir/zdfphy-arithmetic-plant.f90"
compile_nemo_object "$driver_dir/zdfphy-arithmetic-plant.f90" \
  "$driver_dir/zdfphy-arithmetic-plant.o"
fp_named_function_signature "$driver_dir/zdfphy-arithmetic-plant.o" \
  "$driver_dir/zdfphy-plant.fp" __zdfphy_MOD_zdf_phy \
  || refuse 'zdf_phy arithmetic plant produced a zero FP census'
plant_must_refuse r336-zdfphy-function-arithmetic-disassembly cmp -s \
  "$driver_dir/zdfphy-base.fp" "$driver_dir/zdfphy-plant.fp"

# Execute the actual patched trazdf source on two rank-distinct fixtures with
# bounds checking and production scalar-math flags.  Its live result must be
# bitwise equal to a separately compiled unpatched trazdf module; the observer
# replay is then checked against that live result.
fixture_dir=$driver_dir/fixture
mkdir "$fixture_dir" "$fixture_dir/mod"
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
fixture_flags=(-O3 -funroll-all-loops -fno-tree-vectorize \
  -fcheck=bounds -ffree-line-length-none)
sed -e 's/^MODULE trazdf$/MODULE trazdf_reference/' \
  -e 's/^END MODULE trazdf$/END MODULE trazdf_reference/' \
  "$driver_dir/trazdf-base.f90" >"$fixture_dir/trazdf-reference.f90"
cp "$driver_dir/trazdf.F90" "$fixture_dir/trazdf-observed.f90"
"$FC" -c "${fixture_flags[@]}" -J "$fixture_dir/mod" \
  "$FIXTURE_STUBS" -o "$fixture_dir/stubs.o"
"$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
  "$WRITER" -o "$fixture_dir/writer.o"
"$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
  "$fixture_dir/trazdf-reference.f90" -o "$fixture_dir/trazdf-reference.o"
"$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
  "$fixture_dir/trazdf-observed.f90" -o "$fixture_dir/trazdf-observed.o"
"$FC" "${fixture_flags[@]}" -I "$fixture_dir/mod" "$FIXTURE" \
  "$fixture_dir/stubs.o" "$fixture_dir/writer.o" \
  "$fixture_dir/trazdf-reference.o" "$fixture_dir/trazdf-observed.o" \
  -o "$fixture_dir/fixture"
for rank in 0 1; do
  run_in_dir "$fixture_dir" ./fixture "$rank" >>"$fixture_dir/fixture.log" 2>&1
  grep -E "R336_PATCHED_TRAZDF_FIXTURE_PASS[[:space:]]+$rank([[:space:]]|$)" \
    "$fixture_dir/fixture.log" >/dev/null \
    || refuse "round-336 bounds fixture failed for rank $rank"
done
"$PY" - "$fixture_dir" <<'PY'
import copy
import sys
from pathlib import Path
import numpy as np
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round331_tke_gate as checksum_gate
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round336_trazdf_gate as gate

root = Path(sys.argv[1])
for rank in (0, 1):
    record = gate.read_record(root / gate.record_name(rank), expected_shape=(7, 6, 5))
    assert record["header"]["rank"] == rank
    report = gate.verify_record(record)
    assert report["wet_seen"] > 0 and report["signed_zeros"] > 0
    assert report["rhs_association_discriminators"] > 0
for plant in ("header", "field-order", "truncation", "selector", "extent", "time-level"):
    try:
        gate.read_record(root / gate.record_name(0), plant=plant, expected_shape=(7, 6, 5))
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-{plant}")
    else:
        raise AssertionError(f"{plant} parser plant stayed green")
record = gate.read_record(root / gate.record_name(0), expected_shape=(7, 6, 5))
header = record["header"]
wet = record["fields"]["tmask"] != 0.0
wet[:header["ntsi"] - 1, :, :] = False
wet[header["ntei"]:, :, :] = False
wet[:, :header["ntsj"] - 1, :] = False
wet[:, header["ntej"]:, :] = False
global_index = tuple(np.argwhere(wet)[0])
index = (global_index[0] - (header["ntsi"] - 1),
         global_index[1] - (header["ntsj"] - 1), global_index[2])
record["fields"]["avt"][index] = np.nextafter(record["fields"]["avt"][index], np.inf)
try:
    checksum_gate._verify_checksums(record)
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r336-checksum")
else:
    raise AssertionError("checksum plant stayed green")
for plant in ("nonfinite", "signed-zero", "wet", "replay", "association", "actual",
              "filled-range"):
    record = gate.read_record(root / gate.record_name(0), expected_shape=(7, 6, 5))
    try:
        gate.verify_record(record, plant=plant)
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-{plant}")
    else:
        raise AssertionError(f"{plant} value plant stayed green")
for plant, expected in gate.ZDFPHY_PLANT_MESSAGES.items():
    record = gate.read_record(root / gate.record_name(0), expected_shape=(7, 6, 5))
    try:
        gate.verify_record(record, plant=plant)
    except gate.GateError as error:
        if str(error) != expected:
            raise AssertionError(
                f"{plant} hit {str(error)!r}, expected phase refusal {expected!r}")
        print(f"PREFLIGHT_PLANT_FIRED r336-{plant}: {error}")
    else:
        raise AssertionError(f"{plant} phase plant stayed green")
print("R336_PATCHED_TRAZDF_BOUNDS_REPLAY_ASSOCIATION_AND_SIGN_PASS ranks=2 fields=36")
PY

# A changed writer statement must disagree with the independent live solve.
mutant_dir=$driver_dir/mutant
mkdir "$mutant_dir" "$mutant_dir/mod"
sed '0,/diagonal(ji,jj,jk,jn)=e3t_3d/s//diagonal(ji,jj,jk,jn)=1._wp+e3t_3d/' \
  "$WRITER" >"$mutant_dir/writer.F90"
"$FC" -c "${fixture_flags[@]}" -J "$mutant_dir/mod" \
  "$FIXTURE_STUBS" -o "$mutant_dir/stubs.o"
"$FC" -c "${fixture_flags[@]}" -I "$mutant_dir/mod" -J "$mutant_dir/mod" \
  "$mutant_dir/writer.F90" -o "$mutant_dir/writer.o"
sed -e 's/^MODULE trazdf$/MODULE trazdf_reference/' \
  -e 's/^END MODULE trazdf$/END MODULE trazdf_reference/' \
  "$driver_dir/trazdf-base.f90" >"$mutant_dir/trazdf-reference.f90"
cp "$driver_dir/trazdf.F90" "$mutant_dir/trazdf-observed.f90"
"$FC" -c "${fixture_flags[@]}" -I "$mutant_dir/mod" -J "$mutant_dir/mod" \
  "$mutant_dir/trazdf-reference.f90" -o "$mutant_dir/trazdf-reference.o"
"$FC" -c "${fixture_flags[@]}" -I "$mutant_dir/mod" -J "$mutant_dir/mod" \
  "$mutant_dir/trazdf-observed.f90" -o "$mutant_dir/trazdf-observed.o"
"$FC" "${fixture_flags[@]}" -I "$mutant_dir/mod" "$FIXTURE" \
  "$mutant_dir/stubs.o" "$mutant_dir/writer.o" \
  "$mutant_dir/trazdf-reference.o" "$mutant_dir/trazdf-observed.o" \
  -o "$mutant_dir/fixture"
run_in_dir "$mutant_dir" ./fixture 0
plant_must_refuse r336-writer-arithmetic "$PY" -c \
  "from pathlib import Path; from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round336_trazdf_gate as g; g.verify_record(g.read_record(Path('$mutant_dir') / g.record_name(0), expected_shape=(7,6,5)))"

"$PY" -m pytest -q -p no:xdist "$TEST"

# Re-check this wrapper's manifest pin on both --admit-existing routes.  The
# imported driver performs the same two green route checks; these plants prove
# that neither route accepts a moved producer commit or binary digest.
identity_root=$driver_dir/admission-identity
identity_head=$(git -C "$REPO" rev-parse HEAD)
for route in in-place retained; do
  if [[ $route == in-place ]]; then
    identity_run=$identity_root/orca1ice_trazdf_internal_kt2_np2
  else
    identity_run=$identity_root/refused/orca1ice_trazdf_internal_kt2_np2_admission_refused_synthetic
  fi
  mkdir -p "$identity_run"
  cp "$PARENT" "$identity_run/nemo"
  printf '%s\n' "$identity_head" >"$identity_run/producer_commit.txt"
  ( cd "$identity_run" && sha256sum nemo >nemo.sha256 )
  check_admission_manifest "$identity_run" "$identity_head" \
    || refuse "valid $route admission manifest was rejected"
  printf '%040d\n' 0 >"$identity_run/producer_commit.txt"
  plant_must_refuse "r336-$route-producer-pin" \
    check_admission_manifest "$identity_run" "$identity_head"
  printf '%s\n' "$identity_head" >"$identity_run/producer_commit.txt"
  printf 'binary digest plant\n' >>"$identity_run/nemo"
  plant_must_refuse "r336-$route-binary-pin" \
    check_admission_manifest "$identity_run" "$identity_head"
done
printf 'R336_ADMIT_EXISTING_IDENTITY_PASS routes=in-place,retained plants=producer,binary\n'

# Import the reviewed driver wholesale. Only pinned identity strings and hook
# additions are changed; its checklist, controls, syntax proof, same-program
# proof, deterministic rerun, admission, and retention logic remain active.
sed \
  -e "s|^here=\$(CDPATH= cd -- \"\$(dirname -- \"\$0\")\" && pwd -P)$|here=$PARENT_DIR|" \
  -e 's|ORCA2_ORCA1ICE_OMIP_L4_R333LDFI|ORCA2_ORCA1ICE_OMIP_L4_R336TZDF|g' \
  -e 's|round333/acquisition|round336/acquisition|g' \
  -e 's|orca1ice_traldf_iso_internal_kt2_np2|orca1ice_trazdf_internal_kt2_np2|g' \
  "$PARENT" >"$ADAPTED_PARENT"

# Superseded direct-edit prototype retained inert so its exact failure mode is
# auditable; the extension fragment below targets the generated inner driver.
if false; then
# Add tra_zdf to the inherited same-program allow-list.
sed -i \
  's|--observer-patch \\"traldf_iso=\\$scratch/traldf_combined.patch\\" \\\\\\|--observer-patch \\"traldf_iso=\\$scratch/traldf_combined.patch\\" \\\\\\\n+      --observer-patch \\"trazdf='"$PATCH"'\\" \\\\\\|' \
  "$ADAPTED_PARENT"

# Install and audit the new unit at the inherited target-build hook points.
sed -i "/# makenemo rewrites shared work_cfgs.txt/i\\
sha256sum $PATCH $WRITER $GATE $TEST >>\"\$manifest/toolchain.sha256\"" \
  "$ADAPTED_PARENT"
sed -i "/touch \"\$TARGET_ROOT\\/MY_SRC\\/\"\\*\.F90/i\\
cp $TRAZDF_SOURCE \"\$TARGET_ROOT/MY_SRC/trazdf.F90\"\\
cp $WRITER \"\$TARGET_ROOT/MY_SRC/round336_trazdf_writer.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <$PATCH" \
  "$ADAPTED_PARENT"
sed -i "/diff <(grep -v/i\\
grep 'NEMO_L4_R336TZ' \"\$TARGET_ROOT/BLD/ppsrc/nemo/round336_trazdf_writer.f90\" >/dev/null \\
  || refuse 'compiled round-336 tra_zdf writer is absent'\\
check_r336_landing \"\$TARGET_ROOT/BLD/ppsrc/nemo/trazdf.f90\" \\
  || refuse 'round-336 tra_zdf placement moved under TARGET_ROOT'" \
  "$ADAPTED_PARENT"
fi

# Expand the small extension fragment and insert it immediately before the
# imported wrapper seals its generated inner driver.
sed \
  -e "s|@PATCH@|$PATCH|g" \
  -e "s|@ZDFPHY_PATCH@|$ZDFPHY_PATCH|g" \
  -e "s|@ZDFPHY_PPSRC_PATCH@|$ZDFPHY_PPSRC_PATCH|g" \
  -e "s|@WRITER@|$WRITER|g" \
  -e "s|@GATE@|$GATE|g" \
  -e "s|@TEST@|$TEST|g" \
  -e "s|@NAMELIST_GATE@|$NAMELIST_GATE|g" \
  -e "s|@TRAZDF_SOURCE@|$TRAZDF_SOURCE|g" \
  -e "s|@SOURCE_PPSRC@|$SOURCE_PPSRC|g" \
  -e "s|@SOURCE_ZDFPHY_PPSRC@|$SOURCE_ZDFPHY_PPSRC|g" \
  "$PARENT_EXTENSION" >"$driver_dir/parent-extension.sh"
awk -v extension="$driver_dir/parent-extension.sh" '
  $0 == "chmod 700 \"$ADAPTED\"" {
    while ((getline line < extension) > 0) print line
    close(extension)
  }
  { print }
' "$ADAPTED_PARENT" >"$driver_dir/run-parent.extended.sh"
mv -- "$driver_dir/run-parent.extended.sh" "$ADAPTED_PARENT"
chmod 700 "$ADAPTED_PARENT"

if diff -u "$PARENT" "$ADAPTED_PARENT" >"$driver_dir/import.diff"; then
  refuse 'round-336 driver import changed no pinned line'
else
  [[ $? -eq 1 ]] || refuse 'could not audit imported driver'
fi
for token in \
  "here=$PARENT_DIR" \
  'TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R336TZDF' \
  'round336/acquisition' \
  'orca1ice_trazdf_internal_kt2_np2' \
  'observer-patch "trazdf=' \
  'zdfphy_combined.patch' \
  'round336_trazdf_writer.F90' \
  'check_r336_landing'; do
  grep -F "$token" "$ADAPTED_PARENT" >/dev/null \
    || refuse "adapted driver lacks substitution: $token"
done
printf 'R336_INHERITED_DRIVER_AUDIT_PASS parent=round333 additions=trazdf\n'

if [[ "$MODE" != --preflight-only ]]; then retention_armed=1; fi
"$ADAPTED_PARENT" "$MODE"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

refuse_nemo_errors "$TARGET_RUN/ocean.output" \
  || refuse 'NEMO ocean.output contains the quoted E R R O R lines above'
current_commit=$(git -C "$REPO" rev-parse HEAD)
check_admission_manifest "$TARGET_RUN" "$current_commit" \
  || refuse 'run manifest producer/ancestry or binary digest pin moved'
producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
  "$TARGET_RUN/producer_commit.txt")
binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
  "$TARGET_RUN/nemo.sha256")

for plant in header field-order truncation checksum stamp selector extent time-level \
    nonfinite signed-zero wet replay association actual filled-range zdfphy \
    zdfphy-avt-k-zdf zdfphy-avt-after-tke zdfphy-avs-after-tke \
    zdfphy-avt-after-runoff zdfphy-avs-after-runoff zdfphy-avt-after-evd \
    zdfphy-avs-after-evd zdfphy-avt-after-ddm zdfphy-avs-after-ddm \
    zdfphy-avt-after-iwm zdfphy-avs-after-iwm zdfphy-final-avs \
    zdfphy-nonfinite zdfphy-negative; do
  log=$TARGET_RUN/admit-r336-trazdf-$plant.log
  if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "round-336 tra_zdf admission plant stayed green: $plant"
  fi
  if [[ $plant == zdfphy* ]]; then
    expected=$("$PY" -c "from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round336_trazdf_gate as g; print(g.ZDFPHY_PLANT_MESSAGES['$plant'])")
    grep -Fx "STATUS PLANT-FIRED: $expected" "$log" >/dev/null \
      || refuse "round-336 zdf_phy plant hit the wrong refusal: $plant"
  else
    grep -F 'STATUS PLANT-FIRED' "$log" >/dev/null \
      || refuse "round-336 tra_zdf admission plant missed refusal: $plant"
  fi
  printf 'ACQUISITION_PLANT_FIRED r336-trazdf-%s\n' "$plant"
done
"$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
  --output "$TARGET_RUN/round336_trazdf_admission.json"
( cd "$TARGET_RUN" && sha256sum oracle_r336_trazdf_kt00000002_r*.bin \
    round336_trazdf_admission.json admit-r336-trazdf-*.log \
    >round336_trazdf_outputs.sha256 )
retention_armed=0
printf 'ORCA2_ROUND336_TRAZDF_ACQUISITION_PASS %s\n' "$TARGET_RUN"
