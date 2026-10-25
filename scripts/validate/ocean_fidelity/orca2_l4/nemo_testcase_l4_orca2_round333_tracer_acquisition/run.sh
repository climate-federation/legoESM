#!/usr/bin/env bash
# Operator-only extension of the reviewed round-328/332 acquisition driver.
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
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PARENT_DIR=$here/../nemo_testcase_l4_orca2_round328_dynzdf_acquisition
readonly PARENT=$PARENT_DIR/run.sh
readonly PARENT_STAGE_PATCH=$PARENT_DIR/stprk3_stg_round328.patch
readonly SOURCE_OUTPUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2/ocean.output
readonly SOURCE_STAGE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/MY_SRC/stprk3_stg.F90
readonly PATCH=$here/stprk3_stg_round333.patch
readonly WRITER_PATCH=$here/round332_writer_round333.patch
readonly BASE_WRITER=$here/../nemo_testcase_l4_orca2_round332_source_order_acquisition/round332_writer.F90
readonly FIXTURE_STUBS=$here/round333_fixture_stubs.F90
readonly FIXTURE=$here/round333_writer_fixture.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round333_tracer_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round333_tracer_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

for path in "$PARENT" "$PARENT_STAGE_PATCH" "$SOURCE_OUTPUT" "$SOURCE_STAGE" "$PATCH" \
  "$WRITER_PATCH" "$BASE_WRITER" "$FIXTURE_STUBS" "$FIXTURE" "$GATE" "$TEST"; do
  [[ -f "$path" ]] || refuse "missing acquisition input $path"
done
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'

mkdir -p "$EVIDENCE"
driver_dir=$(mktemp -d "$EVIDENCE/.r333-driver.XXXXXX")
retention_armed=0
cleanup() {
  local rc=$? refused
  trap - EXIT
  set +e
  if [[ $rc -ne 0 && $retention_armed -ne 0 && -d "$TARGET_RUN" ]]; then
    refused=$EVIDENCE/refused/$(basename "$TARGET_RUN")_admission_refused_$(date -u +%Y%m%dT%H%M%SZ)_$$
    mkdir -p "$EVIDENCE/refused"
    if mv -- "$TARGET_RUN" "$refused"; then
      printf 'R333_FAILED_OUTPUTS_KEPT kind=admission_refused path=%s\n' "$refused" >&2
    else
      printf 'R333_FAILED_OUTPUTS_LEFT_IN_PLACE path=%s\n' "$TARGET_RUN" >&2
    fi
  fi
  [[ ! -d "$driver_dir" ]] || rm -rf -- "$driver_dir"
  exit "$rc"
}
trap cleanup EXIT
readonly driver_dir
readonly ADAPTED=$driver_dir/run.sh
readonly COMBINED_WRITER=$driver_dir/round332_writer.F90

check_no_removed_lines() {
  [[ $(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$1") -eq 0 ]]
}
check_real_context() {
  ! grep -Eq '^@@ -[0-9]+,0 ' "$1"
}
echo_has() {
  grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null
}
check_tracer_selectors() {
  echo_has "$1" ln_trabbc T && echo_has "$1" ln_trabbl T \
    && echo_has "$1" ln_tradmp T
}
check_r333_stage_landing() {
  local source=$1 plant=${2:-none}
  awk -v plant="$plant" '
    /ts\(:,:,:,jn,Krhs\) = 0\._wp/ { zero=NR }
    /oracle_rktracer_stage3_kt00000001\.bin/ { old_dump_open=NR }
    /CALL tra_adv[[:space:]]*\(/ { adv=NR }
    /CALL tra_sbc_RK3/ { sbc=NR }
    /CALL tra_isf/ { isf=NR }
    /CALL tra_qsr/ { qsr=NR }
    /CALL tra_ldf/ { ldf=NR }
    /CALL tra_bbc/ { bbc=NR }
    /CALL tra_dmp/ { dmp=NR }
    /CALL tra_mfc/ { mfc=NR }
    /CALL tra_zdf/ { zdf=NR }
    /L2_RK_TRACER_STAGE3_DUMP/ { old_dump_finish=NR }
    /CALL tra_npc/ { npc=NR }
    /CALL r333_tracer_open/ { open_count++; open=NR }
    /CALL r333_tracer_rhs/ { rhs_count++; rhs[rhs_count]=NR }
    /CALL r333_tracer_finish/ { finish_count++; finish=NR }
    END {
      if (plant=="open") open=old_dump_open+1
      if (plant=="advection") rhs[1]=sbc+1
      if (plant=="sbc") rhs[2]=isf+1
      if (plant=="qsr") rhs[3]=ldf+1
      if (plant=="ldf") rhs[4]=bbc+1
      if (plant=="damping") rhs[5]=mfc+1
      if (plant=="finish") finish=old_dump_finish-1
      ok=(open_count==1 && rhs_count==5 && finish_count==1)
      ok=ok && zero<open && open<old_dump_open
      ok=ok && adv<rhs[1] && rhs[1]<sbc
      ok=ok && sbc<rhs[2] && rhs[2]<isf
      ok=ok && qsr<rhs[3] && rhs[3]<ldf
      ok=ok && ldf<rhs[4] && rhs[4]<bbc
      ok=ok && dmp<rhs[5] && rhs[5]<mfc
      ok=ok && zdf<old_dump_finish && old_dump_finish<finish && finish<npc
      exit !ok
    }
  ' "$source"
}

cp "$BASE_WRITER" "$COMBINED_WRITER"
patch -s --fuzz=0 -p0 -d "$driver_dir" <"$WRITER_PATCH"
for observer_patch in "$PATCH" "$WRITER_PATCH"; do
  check_no_removed_lines "$observer_patch" \
    || refuse "observer patch removes a source line: $observer_patch"
  check_real_context "$observer_patch" \
    || refuse "observer patch contains a zero-context hunk: $observer_patch"
done

check_tracer_selectors "$SOURCE_OUTPUT" \
  || refuse 'real ocean.output does not enable BBC, BBL, and damping'
for selector in ln_trabbc ln_trabbl ln_tradmp; do
  sed -E "s/(${selector}[[:space:]]*=[[:space:]]*)T/\1F/" "$SOURCE_OUTPUT" \
    >"$driver_dir/${selector}.off"
  plant_must_refuse "r333-${selector}-off" \
    check_tracer_selectors "$driver_dir/${selector}.off"
done

stage_check=$driver_dir/stage-check
mkdir "$stage_check"
cp "$SOURCE_STAGE" "$stage_check/stprk3_stg.F90"
patch -s --fuzz=0 -p0 -d "$stage_check" <"$PARENT_STAGE_PATCH"
patch -s --fuzz=0 -p0 -d "$stage_check" <"$PATCH"
check_r333_stage_landing "$stage_check/stprk3_stg.F90" \
  || refuse 'round-333 tracer observers do not bracket the intended ppsrc order'
landing_plants=(
  r333-open-misplaced:open
  r333-advection-misplaced:advection
  r333-sbc-misplaced:sbc
  r333-qsr-misplaced:qsr
  r333-ldf-misplaced:ldf
  r333-damping-misplaced:damping
  r333-finish-misplaced:finish
)
for planted in "${landing_plants[@]}"; do
  plant_must_refuse "${planted%%:*}" \
    check_r333_stage_landing "$stage_check/stprk3_stg.F90" "${planted#*:}"
done
[[ $(grep -Fc 'CALL r333_tracer_open' "$stage_check/stprk3_stg.F90") -eq 1 ]] \
  || refuse 'round-333 open call census moved'
[[ $(grep -Fc 'CALL r333_tracer_rhs' "$stage_check/stprk3_stg.F90") -eq 5 ]] \
  || refuse 'round-333 RHS call census moved'
[[ $(grep -Fc 'CALL r333_tracer_finish' "$stage_check/stprk3_stg.F90") -eq 1 ]] \
  || refuse 'round-333 finish call census moved'
printf 'R333_STAGE_LANDING_PASS zero=604 adv=637 sbc=645 qsr=717 ldf=734 damping=740 zdf=749\n'

# Compile and execute the actual extended writer twice under bounds checking.
# The fixture feeds rank-distinct data and negative-zero dry/halo cells.
fixture_dir=$driver_dir/fixture
mkdir "$fixture_dir" "$fixture_dir/mod"
"$FC" -c -O0 -fcheck=bounds -ffree-line-length-none -J "$fixture_dir/mod" \
  "$FIXTURE_STUBS" -o "$fixture_dir/stubs.o"
"$FC" -c -O0 -fcheck=bounds -ffree-line-length-none -I "$fixture_dir/mod" \
  -J "$fixture_dir/mod" "$COMBINED_WRITER" -o "$fixture_dir/writer.o"
"$FC" -O0 -fcheck=bounds -ffree-line-length-none -I "$fixture_dir/mod" \
  "$FIXTURE" "$fixture_dir/stubs.o" "$fixture_dir/writer.o" \
  -o "$fixture_dir/round333_writer_fixture"
for rank in 0 1; do
  ( cd "$fixture_dir" && ./round333_writer_fixture "$rank" ) \
    >>"$fixture_dir/fixture.log" 2>&1
  grep -F "R333_WRITER_FIXTURE_PASS $rank T" "$fixture_dir/fixture.log" >/dev/null \
    || refuse "round-333 bounds fixture failed for rank $rank"
done
"$PY" - "$fixture_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round331_tke_gate as r331
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round333_tracer_gate as gate

root = Path(sys.argv[1])
for rank in (0, 1):
    path = root / gate.record_name(rank)
    record = gate.read_stage3(path, expected_shape=(5, 4, 5))
    r331._verify_checksums(record)
    assert record["header"]["rank"] == rank
    assert np.signbit(record["fields"]["zero_T"][0, 0, 0])
    assert np.signbit(record["fields"]["r3t_Kbb"][0, 0])
try:
    gate.read_stage3(root / gate.record_name(0), plant="truncation", expected_shape=(5, 4, 5))
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r333-truncation")
else:
    raise AssertionError("truncation plant stayed green")
record = gate.read_stage3(root / gate.record_name(0), expected_shape=(5, 4, 5))
record["fields"]["after_advection_T"][2, 1, 1] = np.nextafter(
    record["fields"]["after_advection_T"][2, 1, 1], np.inf
)
try:
    r331._verify_checksums(record)
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r333-checksum")
else:
    raise AssertionError("checksum plant stayed green")
print("R333_WRITER_BOUNDS_AND_SIGN_PASS ranks=2")
PY

# Import every reviewed helper, plant, syntax proof, same-program check and
# admission step from the round-328/332 driver. The arithmetic-baseline edit
# removes both r332 and every marked r333 observer line before its FP census.
sed \
  -e "s|^here=\$(CDPATH= cd -- \"\$(dirname -- \"\$0\")\" && pwd -P)$|here=$PARENT_DIR|" \
  -e 's|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R332RAW|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333TR|' \
  -e 's|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition|' \
  -e 's|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_raw_postfold_v_kt1_np2|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2|' \
  -e 's|orca1ice_raw_postfold_v_kt1_np2_\*|orca1ice_tracer_stage3_kt2_np2_*|' \
  -e "s|readonly R332_STAGE_PATCH=\$R332_DIR/stprk3_stg_round332.patch|readonly R332_STAGE_PATCH=$PATCH|" \
  -e "s|readonly R332_WRITER=\$R332_DIR/round332_writer.F90|readonly R332_WRITER=$COMBINED_WRITER|" \
  "$PARENT" >"$ADAPTED"
sed -i \
  "s|-e '/CALL r332_raw_postfold_v/d' \"\$dry/stprk3_stg.F90\"|-e '/CALL r332_raw_postfold_v/d' -e '/R333_OBSERVER/d' \"\$dry/stprk3_stg.F90\"|" \
  "$ADAPTED"
chmod 700 "$ADAPTED"

if diff -u "$PARENT" "$ADAPTED" >"$driver_dir/import.diff"; then
  refuse 'round-333 driver import changed no pinned line'
else
  [[ $? -eq 1 ]] || refuse 'could not audit the imported driver'
fi
changed=$(awk '
  /^--- |^\+\+\+ / { next }
  /^[-+]/ { count++ }
  END { print count+0 }
' "$driver_dir/import.diff")
[[ "$changed" -eq 16 ]] || refuse "imported driver changed $changed lines, expected 16"
for token in \
  "here=$PARENT_DIR" \
  'TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333TR' \
  "EVIDENCE=$EVIDENCE" \
  'DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tracer_stage3_kt2_np2' \
  '$(basename "$TARGET_RUN") == orca1ice_tracer_stage3_kt2_np2_*' \
  "readonly R332_STAGE_PATCH=$PATCH" \
  "readonly R332_WRITER=$COMBINED_WRITER" \
  "R333_OBSERVER"; do
  [[ $(grep -Fc "$token" "$ADAPTED") -eq 1 ]] \
    || refuse "adapted driver lacks unique substitution: $token"
done

# Exercise the inherited manifest/ancestry admission identity on a synthetic
# round-333 run through both supported routes. This invokes the same adapted
# driver used below, but exits before deterministic-output admission.
synthetic_root=$driver_dir/synthetic-admission
synthetic_default=$synthetic_root/orca1ice_tracer_stage3_kt2_np2
mkdir -p "$synthetic_default"
cp "$PARENT" "$synthetic_default/nemo"
chmod 700 "$synthetic_default/nemo"
git -C "$REPO" rev-parse HEAD >"$synthetic_default/producer_commit.txt"
( cd "$synthetic_default" && sha256sum nemo >nemo.sha256 )
R328_ADMISSION_IDENTITY_ONLY=1 R328_ADMISSION_TEST_ROOT=$synthetic_root \
  R328_ACQ_TARGET=tke332 "$ADAPTED" --admit-existing \
  >"$driver_dir/synthetic-in-place.log"
grep -F 'R328_SYNTHETIC_ADMISSION_IDENTITY_PASS route=in-place' \
  "$driver_dir/synthetic-in-place.log" >/dev/null \
  || refuse 'synthetic in-place admission did not pass'
mkdir -p "$synthetic_root/refused"
synthetic_retained=$synthetic_root/refused/orca1ice_tracer_stage3_kt2_np2_admission_refused_synthetic
mv -- "$synthetic_default" "$synthetic_retained"
R328_ADMISSION_IDENTITY_ONLY=1 R328_ADMISSION_TEST_ROOT=$synthetic_root \
  R328_EXISTING_RUN=$synthetic_retained R328_ACQ_TARGET=tke332 \
  "$ADAPTED" --admit-existing >"$driver_dir/synthetic-retained.log"
grep -F 'R328_SYNTHETIC_ADMISSION_IDENTITY_PASS route=retained' \
  "$driver_dir/synthetic-retained.log" >/dev/null \
  || refuse 'synthetic retained admission did not pass'
printf 'R333_SYNTHETIC_RETAINED_ADMISSION_PASS routes=in-place,retained\n'

export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
"$PY" -m pytest -q -p no:xdist "$TEST"

export R328_ACQ_TARGET=tke332
"$ADAPTED" "$MODE"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND333_TRACER_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

# The inherited driver has now completed its admission (including retained-run
# movement). Any r333 refusal below is covered by this trap and retains the
# complete run, including every admit-r333-*.log, under refused/.
retention_armed=1
[[ -f "$TARGET_RUN/producer_commit.txt" && \
   $(wc -l <"$TARGET_RUN/producer_commit.txt") -eq 1 ]] \
  || refuse 'run manifest lacks its single producer commit'
producer=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
  "$TARGET_RUN/producer_commit.txt")
[[ $producer =~ ^[0-9a-f]{40}$ ]] || refuse 'run manifest producer commit is malformed'
[[ -f "$TARGET_RUN/nemo.sha256" && $(wc -l <"$TARGET_RUN/nemo.sha256") -eq 1 ]] \
  || refuse 'run manifest lacks its single binary SHA'
binary_sha=$(awk 'NF == 2 && $2 == "nemo" && $1 ~ /^[0-9a-f]{64}$/ {print $1}' \
  "$TARGET_RUN/nemo.sha256")
[[ $binary_sha =~ ^[0-9a-f]{64}$ ]] || refuse 'run manifest binary SHA is malformed'
[[ $(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}') == "$binary_sha" ]] \
  || refuse 'run binary differs from its manifest SHA'

for plant in header truncation checksum stamp zero kmm time-level; do
  log=$TARGET_RUN/admit-r333-$plant.log
  if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "round-333 admission plant stayed green: $plant"
  fi
  grep -F "STATUS PLANT-FIRED" "$log" >/dev/null \
    || refuse "round-333 admission plant missed its refusal: $plant"
  printf 'ACQUISITION_PLANT_FIRED r333-%s\n' "$plant"
done
"$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
  --output "$TARGET_RUN/round333_tracer_admission.json"
( cd "$TARGET_RUN" && sha256sum oracle_rktracer_stage3_kt00000002_r*.bin \
    round333_tracer_admission.json admit-r333-*.log \
    >round333_tracer_outputs.sha256 )
retention_armed=0
printf 'ORCA2_ROUND333_TRACER_ACQUISITION_PASS %s\n' "$TARGET_RUN"
