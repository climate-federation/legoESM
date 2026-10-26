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
readonly SOURCE_OBJECT=$SOURCE_CFG/BLD/obj/trazdf.o
readonly SOURCE_MODULES=$SOURCE_CFG/BLD/inc
readonly SOURCE_OUTPUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2/ocean.output
readonly TRAZDF_SOURCE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/trazdf.F90
readonly PATCH=$here/trazdf_round336.patch
readonly WRITER=$here/round336_trazdf_writer.F90
readonly FIXTURE_STUBS=$here/round336_fixture_stubs.F90
readonly FIXTURE=$here/round336_writer_fixture.F90
readonly PARENT_EXTENSION=$here/round336_parent_extension.sh.inc
readonly GATE=$here/../nemo_testcase_l4_orca2_round336_trazdf_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round336_trazdf_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_trazdf_internal_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly MPIFC=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpif90
readonly NEMO_BIN=/home/dbalwada/miniconda3/envs/nemo-build/bin

for path in "$PARENT" "$SOURCE_PPSRC" "$SOURCE_OBJECT" "$SOURCE_OUTPUT" \
  "$TRAZDF_SOURCE" "$PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" \
  "$PARENT_EXTENSION" "$GATE" "$TEST"; do
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
check_selectors() {
  echo_has "$1" ln_zad_Aimp F \
    && echo_has "$1" ln_zdfmfc F \
    && echo_has "$1" ln_zdfevd T \
    && echo_has "$1" ln_zdfddm T \
    && echo_has "$1" ln_zdfiwm T \
    && echo_has "$1" ln_traldf_msc T
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

fp_disassembly_signature() {
  objdump -drwC --no-show-raw-insn "$1" \
    | sed -nE 's/^[[:space:]]*[0-9a-f]+:[[:space:]]+((f(add|sub|mul|div|sqrt)[a-z0-9]*|add(s|p)[sd]|sub(s|p)[sd]|mul(s|p)[sd]|div(s|p)[sd]|sqrt(s|p)[sd]|min(s|p)[sd]|max(s|p)[sd])).*/\1/p' \
    | sort | uniq -c >"$2"
}
compare_fp_disassembly() {
  local left=$1 right=$2 label=$3
  fp_disassembly_signature "$left" "$driver_dir/${label}.left.fp"
  fp_disassembly_signature "$right" "$driver_dir/${label}.right.fp"
  cmp -s "$driver_dir/${label}.left.fp" "$driver_dir/${label}.right.fp"
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

check_no_removed_lines "$PATCH" || refuse 'tra_zdf observer patch removes a source line'
check_real_context "$PATCH" || refuse 'tra_zdf observer patch has a zero-context hunk'
check_selectors "$SOURCE_OUTPUT" || refuse 'record run does not select the registered tra_zdf branches'
for selector in ln_zad_Aimp ln_zdfmfc ln_zdfevd ln_zdfddm ln_zdfiwm ln_traldf_msc; do
  case "$selector" in
    ln_zad_Aimp|ln_zdfmfc) replacement=T ;;
    *) replacement=F ;;
  esac
  sed -E "s/(${selector}[[:space:]]*=[[:space:]]*)[TF]/\\1$replacement/" \
    "$SOURCE_OUTPUT" >"$driver_dir/${selector}.moved"
  plant_must_refuse "r336-${selector}-moved" check_selectors "$driver_dir/${selector}.moved"
done

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

# Compile the actual record-build ppsrc with its production arithmetic flags.
# The normalized census ignores the two observer calls and must be identical.
mkdir "$driver_dir/nemo-mod"
cp "$SOURCE_PPSRC" "$driver_dir/trazdf-base.f90"
cp "$SOURCE_PPSRC" "$driver_dir/trazdf.F90"
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
sed '0,/zwd(ji,jk) = /s//zwd(ji,jk) = 1._wp + /' \
  "$driver_dir/trazdf-base.f90" >"$driver_dir/trazdf-arithmetic-plant.f90"
compile_nemo_object "$driver_dir/trazdf-arithmetic-plant.f90" "$driver_dir/trazdf-arithmetic-plant.o"
plant_must_refuse r336-trazdf-arithmetic-disassembly compare_fp_disassembly \
  "$driver_dir/trazdf-base.o" "$driver_dir/trazdf-arithmetic-plant.o" trazdf-plant

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
for plant in ("header", "field-order", "truncation", "selector", "extent"):
    try:
        gate.read_record(root / gate.record_name(0), plant=plant, expected_shape=(7, 6, 5))
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-{plant}")
    else:
        raise AssertionError(f"{plant} parser plant stayed green")
record = gate.read_record(root / gate.record_name(0), expected_shape=(7, 6, 5))
record["fields"]["avt"][2, 2, 1] = np.nextafter(record["fields"]["avt"][2, 2, 1], np.inf)
try:
    checksum_gate._verify_checksums(record)
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r336-checksum")
else:
    raise AssertionError("checksum plant stayed green")
for plant in ("nonfinite", "signed-zero", "wet", "replay", "association", "actual"):
    record = gate.read_record(root / gate.record_name(0), expected_shape=(7, 6, 5))
    try:
        gate.verify_record(record, plant=plant)
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-{plant}")
    else:
        raise AssertionError(f"{plant} value plant stayed green")
print("R336_PATCHED_TRAZDF_BOUNDS_REPLAY_ASSOCIATION_AND_SIGN_PASS ranks=2 fields=25")
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
  -e "s|@WRITER@|$WRITER|g" \
  -e "s|@GATE@|$GATE|g" \
  -e "s|@TEST@|$TEST|g" \
  -e "s|@TRAZDF_SOURCE@|$TRAZDF_SOURCE|g" \
  -e "s|@SOURCE_PPSRC@|$SOURCE_PPSRC|g" \
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
  'round336_trazdf_writer.F90' \
  'check_r336_landing'; do
  grep -F "$token" "$ADAPTED_PARENT" >/dev/null \
    || refuse "adapted driver lacks substitution: $token"
done
printf 'R336_INHERITED_DRIVER_AUDIT_PASS parent=round333 additions=trazdf\n'

"$ADAPTED_PARENT" "$MODE"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

retention_armed=1
current_commit=$(git -C "$REPO" rev-parse HEAD)
check_admission_manifest "$TARGET_RUN" "$current_commit" \
  || refuse 'run manifest producer/ancestry or binary digest pin moved'
producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
  "$TARGET_RUN/producer_commit.txt")
binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
  "$TARGET_RUN/nemo.sha256")

for plant in header field-order truncation checksum stamp selector extent \
    nonfinite signed-zero wet replay association actual; do
  log=$TARGET_RUN/admit-r336-trazdf-$plant.log
  if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "round-336 tra_zdf admission plant stayed green: $plant"
  fi
  grep -F 'STATUS PLANT-FIRED' "$log" >/dev/null \
    || refuse "round-336 tra_zdf admission plant missed refusal: $plant"
  printf 'ACQUISITION_PLANT_FIRED r336-trazdf-%s\n' "$plant"
done
"$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
  --output "$TARGET_RUN/round336_trazdf_admission.json"
( cd "$TARGET_RUN" && sha256sum oracle_r336_trazdf_kt00000002_r*.bin \
    round336_trazdf_admission.json admit-r336-trazdf-*.log \
    >round336_trazdf_outputs.sha256 )
retention_armed=0
printf 'ORCA2_ROUND336_TRAZDF_ACQUISITION_PASS %s\n' "$TARGET_RUN"
