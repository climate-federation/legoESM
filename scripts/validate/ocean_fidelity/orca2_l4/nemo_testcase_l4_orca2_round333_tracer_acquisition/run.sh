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
readonly LDF_SOURCE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/traldf_iso.F90
readonly LDF_SCHEME_SOURCE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/traldf_iso_scheme.h90
readonly LDF_PATCH=$here/traldf_iso_round333.patch
readonly LDF_SCHEME_PATCH=$here/traldf_iso_scheme_round333.patch
readonly LDF_WRITER=$here/round333_traldf_writer.F90
readonly LDF_FIXTURE_STUBS=$here/round333_traldf_fixture_stubs.F90
readonly LDF_FIXTURE=$here/round333_traldf_writer_fixture.F90
readonly LDF_LOOP_FIXTURE=$here/round333_traldf_loop_fixture.F90
readonly LDF_GATE=$here/../nemo_testcase_l4_orca2_round333_traldf_gate.py
readonly LDF_TEST=$REPO/tests/ocean/fidelity/test_orca2_round333_traldf_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_traldf_iso_internal_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

for path in "$PARENT" "$PARENT_STAGE_PATCH" "$SOURCE_OUTPUT" "$SOURCE_STAGE" "$PATCH" \
  "$WRITER_PATCH" "$BASE_WRITER" "$FIXTURE_STUBS" "$FIXTURE" "$GATE" "$TEST" \
  "$LDF_SOURCE" "$LDF_SCHEME_SOURCE" "$LDF_PATCH" "$LDF_SCHEME_PATCH" "$LDF_WRITER" \
  "$LDF_FIXTURE_STUBS" "$LDF_FIXTURE" "$LDF_LOOP_FIXTURE" "$LDF_GATE" "$LDF_TEST"; do
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
    /CALL r333_ldf_arm/ { arm_count++; arm=NR }
    /CALL r333_ldf_disarm/ { disarm_count++; disarm=NR }
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
      if (plant=="arm") arm=ldf+1
      if (plant=="disarm") disarm=ldf-1
      if (plant=="damping") rhs[5]=mfc+1
      if (plant=="finish") finish=old_dump_finish-1
      ok=(open_count==1 && rhs_count==5 && finish_count==1 && arm_count==1 && disarm_count==1)
      ok=ok && zero<open && open<old_dump_open
      ok=ok && adv<rhs[1] && rhs[1]<sbc
      ok=ok && sbc<rhs[2] && rhs[2]<isf
      ok=ok && qsr<rhs[3] && rhs[3]<arm && arm<ldf
      ok=ok && ldf<disarm && disarm<rhs[4] && rhs[4]<bbc
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
  r333-ldf-arm-misplaced:arm
  r333-ldf-disarm-misplaced:disarm
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
[[ $(grep -Fc 'CALL r333_ldf_arm' "$stage_check/stprk3_stg.F90") -eq 1 && \
   $(grep -Fc 'CALL r333_ldf_disarm' "$stage_check/stprk3_stg.F90") -eq 1 ]] \
  || refuse 'round-333 internal LDF arm/disarm census moved'
printf 'R333_STAGE_LANDING_PASS zero=604 adv=637 sbc=645 qsr=717 ldf=734 damping=740 zdf=749\n'

check_r333_ldf_landing() {
  local source=$1 scheme=$2 plant=${3:-none}
  awk -v plant="$plant" '
    /SUBROUTINE traldf_iso_lap/ {in_lap=1}
    in_lap && /CALL traldf_iso_a33/ {a33=NR}
    in_lap && /CALL r333_ldf_begin/ {begin_count++; begin=NR}
    in_lap && /itra = SIZE/ {itra=NR}
    in_lap && /END DO.*end tracer loop/ {tracer_end=NR}
    in_lap && /CALL r333_ldf_finish/ {finish_count++; finish=NR}
    in_lap && /END SUBROUTINE traldf_iso_lap/ {routine_end=NR; in_lap=0}
    /pah_wslp2\(ji,jj,jk\) = zahu_w/ {pah=NR}
    /CALL r333_ldf_a33_first/ {first_count++; first=NR}
    /pakz\(ji,jj,jk\) = \(/ {raw_statement=NR}
    /CALL r333_ldf_a33_raw/ {raw_count++; raw=NR}
    /pakz\(ji,jj,jk\) = MAX/ {lap_statement=NR}
    /CALL r333_ldf_a33_lap/ {lap_count++; lap=NR}
    END {
      if(plant=="begin") begin=a33+1
      if(plant=="finish") finish=tracer_end-1
      if(plant=="a33-first") first=pah-1
      if(plant=="a33-raw") raw=raw_statement-1
      if(plant=="a33-lap") lap=lap_statement-1
      exit !(begin_count==1 && finish_count==1 && first_count==1 && raw_count==1 && lap_count==1 &&
             begin<a33 && a33<itra && tracer_end<finish && finish<routine_end &&
             pah<first && raw_statement<raw && lap_statement<lap)
    }
  ' "$source" && awk -v plant="$plant" '
    BEGIN {active=1}
    /^[[:space:]]*SUBROUTINE traldf_iso_lap/ {active=1; rotation=0; vertical_title=0; store=0}
    active && /CALL r333_ldf_grad/ {grad_count++; grad=NR; grad_lower=($0 ~ /ntsi-\(inn\+1\)/)}
    active && /rotation matrix[[:space:]]+A =/ {if(!rotation) rotation=NR}
    active && /zfv\(ji,jj\) = ahtv/ {zfv=NR}
    active && /CALL r333_ldf_horizontal/ {h_count++; horiz=NR}
    active && /VERTICAL iso-neutral fluxes/ {vertical_title=NR}
    active && /CALL r333_ldf_vertical_before/ {before_count++; before[before_count]=NR; if($0 ~ /ntsi-\(inn\)/) lower_count++}
    active && /pt\(ji,jj,jk,jn,Krhs\) = pt\(ji,jj,jk,jn,Krhs\) \+/ {update_count++; update[update_count]=NR}
    active && /CALL r333_ldf_vertical_after/ {after_count++; after=NR}
    active && /CALL r333_ldf_bottom_after/ {bottom_count++; bottom=NR}
    active && /zfw\(ji,jj\) = zfw_kp1/ {store=NR}
    /END SUBROUTINE traldf_iso_lap/ {active=0}
    END {
      if(plant=="gradient") grad=rotation+1
      if(plant=="horizontal") horiz=vertical_title+1
      if(plant=="gradient-lower") grad_lower=0
      if(plant=="vertical-lower") lower_count=0
      if(plant=="vertical") before[1]=update[1]+1
      if(plant=="bottom") bottom=update[2]-1
      exit !(grad_count==1 && h_count==1 && before_count==2 && after_count==1 && bottom_count==1 &&
             update_count==2 && grad_lower && lower_count==2 && grad<rotation && zfv<horiz &&
             horiz<vertical_title && before[1]<update[1] && update[1]<store && store<after &&
             before[2]<update[2] && update[2]<bottom)
    }
  ' "$scheme"
}
export -f check_r333_ldf_landing

ldf_check=$driver_dir/ldf-check
mkdir "$ldf_check"
cp "$LDF_SOURCE" "$ldf_check/traldf_iso.F90"
cp "$LDF_SCHEME_SOURCE" "$ldf_check/traldf_iso_scheme.h90"
patch -s --fuzz=0 -p0 -d "$ldf_check" <"$LDF_PATCH"
patch -s --fuzz=0 -p0 -d "$ldf_check" <"$LDF_SCHEME_PATCH"
check_r333_ldf_landing "$ldf_check/traldf_iso.F90" "$ldf_check/traldf_iso_scheme.h90" \
  || refuse 'internal LDF observers do not bracket the cited executing statements'
for landing in begin a33-first a33-raw a33-lap gradient gradient-lower horizontal \
    vertical-lower vertical bottom finish; do
  plant_must_refuse "r333-ldf-${landing}-misplaced" check_r333_ldf_landing \
    "$ldf_check/traldf_iso.F90" "$ldf_check/traldf_iso_scheme.h90" "$landing"
done
printf 'R333_TRALDF_LANDING_PASS ppsrc=154,183-204,230-246,266-285,287-304\n'

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

# Compile and execute the full-array internal writer on both ranks with bounds checks.
ldf_fixture_dir=$driver_dir/ldf-fixture
mkdir "$ldf_fixture_dir" "$ldf_fixture_dir/mod"
"$FC" -c -O0 -fcheck=bounds -ffree-line-length-none -J "$ldf_fixture_dir/mod" \
  "$LDF_FIXTURE_STUBS" -o "$ldf_fixture_dir/stubs.o"
"$FC" -c -O0 -fcheck=bounds -ffree-line-length-none -I "$ldf_fixture_dir/mod" \
  -J "$ldf_fixture_dir/mod" "$LDF_WRITER" -o "$ldf_fixture_dir/writer.o"
"$FC" -O0 -fcheck=bounds -ffree-line-length-none -I "$ldf_fixture_dir/mod" \
  "$LDF_FIXTURE" "$ldf_fixture_dir/stubs.o" "$ldf_fixture_dir/writer.o" \
  -o "$ldf_fixture_dir/round333_traldf_fixture"
for rank in 0 1; do
  ( cd "$ldf_fixture_dir" && ./round333_traldf_fixture "$rank" ) \
    >>"$ldf_fixture_dir/fixture.log" 2>&1
  grep -F "R333_TRALDF_FIXTURE_PASS $rank" "$ldf_fixture_dir/fixture.log" >/dev/null \
    || refuse "round-333 traldf bounds fixture failed for rank $rank"
done
"$PY" - "$ldf_fixture_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round333_traldf_gate as gate

root = Path(sys.argv[1])
for rank in (0, 1):
    record = gate.read_record(root / gate.record_name(rank), expected_shape=(5, 4, 3))
    gate._verify_checksums(record)
    gate._replay_self_consistency(record["fields"])
    assert record["header"]["rank"] == rank
    assert record["fields"]["rhs_increment"].shape == (5, 4, 3, 2)
    assert np.signbit(record["fields"]["dit"][1, 1, 0, 0])
    assert np.signbit(record["fields"]["zfw_lower"][1, 1, 0, 1])
    assert np.count_nonzero(record["fields"]["vertical_seen"]) == 2
    assert np.count_nonzero(record["fields"]["a33_seen"]) > 0
try:
    gate.read_record(root / gate.record_name(0), plant="truncation", expected_shape=(5, 4, 3))
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r333-traldf-truncation")
else:
    raise AssertionError("traldf truncation plant stayed green")
record = gate.read_record(root / gate.record_name(0), expected_shape=(5, 4, 3))
record["fields"]["A11"][1, 1, 0, 0] = np.nextafter(record["fields"]["A11"][1, 1, 0, 0], np.inf)
try:
    gate._verify_checksums(record)
except gate.GateError:
    print("PREFLIGHT_PLANT_FIRED r333-traldf-checksum")
else:
    raise AssertionError("traldf checksum plant stayed green")
print("R333_TRALDF_WRITER_BOUNDS_AND_SIGN_PASS ranks=2 fields=71")
PY

# Execute an observer-bearing copy of the patched loop bodies and compare it
# bitwise with an independently written unpatched copy on small rank arrays.
"$FC" -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds \
  -ffree-line-length-none -I "$ldf_fixture_dir/mod" "$LDF_LOOP_FIXTURE" \
  "$ldf_fixture_dir/stubs.o" "$ldf_fixture_dir/writer.o" \
  -o "$ldf_fixture_dir/round333_traldf_loop_fixture"
( cd "$ldf_fixture_dir" && ./round333_traldf_loop_fixture ) \
  >>"$ldf_fixture_dir/fixture.log" 2>&1
grep -F 'R333_TRALDF_PATCHED_LOOP_FIXTURE_PASS independent-reference' \
  "$ldf_fixture_dir/fixture.log" >/dev/null \
  || refuse 'patched traldf loop fixture differs from independent reference'
sed '0,/upper=0\.5_wp/s//upper=0.25_wp/' "$LDF_LOOP_FIXTURE" \
  >"$ldf_fixture_dir/round333_traldf_loop_plant.F90"
"$FC" -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds \
  -ffree-line-length-none -I "$ldf_fixture_dir/mod" \
  "$ldf_fixture_dir/round333_traldf_loop_plant.F90" \
  "$ldf_fixture_dir/stubs.o" "$ldf_fixture_dir/writer.o" \
  -o "$ldf_fixture_dir/round333_traldf_loop_plant"
plant_must_refuse r333-traldf-loop-arithmetic \
  "$ldf_fixture_dir/round333_traldf_loop_plant"

# Import every reviewed helper, plant, syntax proof, same-program check and
# admission step from the round-328/332 driver. The arithmetic-baseline edit
# removes both r332 and every marked r333 observer line before its FP census.
sed \
  -e "s|^here=\$(CDPATH= cd -- \"\$(dirname -- \"\$0\")\" && pwd -P)$|here=$PARENT_DIR|" \
  -e 's|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R332RAW|TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333LDFI|' \
  -e 's|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition|EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition|' \
  -e 's|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_raw_postfold_v_kt1_np2|DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_traldf_iso_internal_kt2_np2|' \
  -e 's|orca1ice_raw_postfold_v_kt1_np2_\*|orca1ice_traldf_iso_internal_kt2_np2_*|' \
  -e "s|readonly R332_STAGE_PATCH=\$R332_DIR/stprk3_stg_round332.patch|readonly R332_STAGE_PATCH=$PATCH|" \
  -e "s|readonly R332_WRITER=\$R332_DIR/round332_writer.F90|readonly R332_WRITER=$COMBINED_WRITER|" \
  "$PARENT" >"$ADAPTED"
sed -i \
  "s|-e '/CALL r332_raw_postfold_v/d' \"\$dry/stprk3_stg.F90\"|-e '/CALL r332_raw_postfold_v/d' -e '/R333_OBSERVER/d' -e '/R333_LDF_OBSERVER/d' \"\$dry/stprk3_stg.F90\"|" \
  "$ADAPTED"

# Extend the imported parent at its reviewed hook points. Every replacement is
# pinned and audited below; no parent helper, plant, admission check, or run
# guard is removed.
sed -i \
  -e "s|\"\$PY\" \"\$R332_GATE\" --same-program|\"\$PY\" $LDF_GATE --same-program|" \
  -e "s|--observer-patch \"zdfphy=\$TKE_PHY_PATCH\" --observer-patch \"zdftke=\$scratch/tke_combined.patch\" \\\\|--observer-patch \"zdfphy=\$TKE_PHY_PATCH\" --observer-patch \"zdftke=\$scratch/tke_combined.patch\" \\\\\n      --observer-patch \"traldf_iso=\$scratch/traldf_combined.patch\" \\\\|" \
  -e "s|patch_files+=(\"\$R332_DYNVOR_PATCH\" \"\$R332_TKE_PATCH\" \"\$R332_STAGE_PATCH\")|patch_files+=(\"\$R332_DYNVOR_PATCH\" \"\$R332_TKE_PATCH\" \"\$R332_STAGE_PATCH\" $LDF_PATCH $LDF_SCHEME_PATCH)|" \
  -e "s|syntax_units=(dynzdf_round328_writer tke_round331_writer round332_writer \\\\|syntax_units=(dynzdf_round328_writer tke_round331_writer round332_writer round333_traldf_writer \\\\|" \
  -e "s|dynzdf stprk3_stg dynvor zdfphy zdftke)|dynzdf stprk3_stg dynvor zdfphy zdftke traldf_iso)|" \
  -e 's|observer_units+=(dynvor)|observer_units+=(dynvor traldf_iso)|' \
  "$ADAPTED"

sed -i "/for unit in dynzdf stprk3_stg; do/i\\
  cp $LDF_SOURCE \"\$dry/traldf_iso.F90\"\\
  cp $LDF_SCHEME_SOURCE \"\$dry/traldf_iso_scheme.h90\"\\
  cp $LDF_WRITER \"\$dry/round333_traldf_writer.F90\"\\
  patch -s --fuzz=0 -p0 -d \"\$dry\" <$LDF_PATCH\\
  patch -s --fuzz=0 -p0 -d \"\$dry\" <$LDF_SCHEME_PATCH" \
  "$ADAPTED"

sed -i "/# Same-program preflight:/i\\
compile_traldf_census_object() {\\
  \"\$FC\" -c -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcray-pointer -ffree-line-length-none \\\
    -fallow-argument-mismatch -I \"\$scratch/mod\" -J \"\$scratch/mod\" \"\$1\" -o \"\$2\"\\
}\\
if [[ \"\$ACQ_TARGET\" == tke332 ]]; then\\
  mkdir \"\$scratch/traldf-independent\"\\
  cp $LDF_SOURCE \"\$scratch/traldf-independent/traldf_iso.F90\"\\
  cp $LDF_SCHEME_SOURCE \"\$scratch/traldf-independent/traldf_iso_scheme.h90\"\\
  patch -s --fuzz=0 -p0 -d \"\$scratch/traldf-independent\" <$LDF_PATCH\\
  patch -s --fuzz=0 -p0 -d \"\$scratch/traldf-independent\" <$LDF_SCHEME_PATCH\\
  fpp $LDF_SOURCE \"\$scratch/traldf_independent_base.f90\"\\
  fpp \"\$scratch/traldf-independent/traldf_iso.F90\" \"\$scratch/traldf_independent_observed.f90\"\\
  if diff -u --label traldf_iso.f90 --label traldf_iso.f90 \\\
      \"\$scratch/traldf_independent_base.f90\" \"\$scratch/traldf_independent_observed.f90\" >\"\$scratch/traldf_combined.patch\"; then\\
    refuse 'round-333 combined traldf observer patch is empty'\\
  else\\
    [[ \$? -eq 1 ]] || refuse 'could not construct combined traldf observer patch'\\
  fi\\
  cmp -s \"\$scratch/traldf_independent_observed.f90\" \"\$scratch/traldf_iso.f90\" || refuse 'independent patched traldf unit differs from syntax unit'\\
  command -v objdump >/dev/null || refuse 'no objdump for traldf passivity proof'\\
  compile_traldf_census_object \"\$scratch/traldf_independent_base.f90\" \"\$scratch/traldf_iso.without-observers.o\"\\
  compile_traldf_census_object \"\$scratch/traldf_iso.f90\" \"\$scratch/traldf_iso.with-observers.o\"\\
  compare_fp_disassembly \"\$scratch/traldf_iso.without-observers.o\" \"\$scratch/traldf_iso.with-observers.o\" traldf \\\
    || refuse 'traldf observer changed the normalized floating-point instruction census'\\
  printf 'R333_TRALDF_NORMALIZED_OBJECT_DISASSEMBLY_PASS fp=%s\\n' \\\
    \"\$(awk '{n+=\$1} END {print n+0}' \"\$scratch/traldf.left.fp\")\"\\
  sed '0,/zA11 = /s//zA11 = 1._wp + /' \\\
    \"\$scratch/traldf_independent_base.f90\" >\"\$plants/traldf.arithmetic.f90\"\\
  compile_traldf_census_object \"\$plants/traldf.arithmetic.f90\" \"\$scratch/traldf_plant.o\"\\
  plant_must_refuse traldf-arithmetic-disassembly compare_fp_disassembly \\\
    \"\$scratch/traldf_iso.without-observers.o\" \"\$scratch/traldf_plant.o\" traldf-plant\\
fi" "$ADAPTED"

sed -i "/same_program \"\$dry_bld\"/i\\
rm -f \"\$dry_bld/ppsrc/nemo/traldf_iso.f90\"\\
cp \"\$scratch/traldf_iso.f90\" \"\$dry_bld/ppsrc/nemo/traldf_iso.f90\"\\
cp \"\$scratch/round333_traldf_writer.f90\" \"\$dry_bld/ppsrc/nemo/round333_traldf_writer.f90\"\\
check_r333_ldf_landing \"\$dry_bld/ppsrc/nemo/traldf_iso.f90\" \"\$dry_bld/ppsrc/nemo/traldf_iso.f90\" || refuse 'round-333 placement moved in dry ppsrc/nemo/traldf_iso.f90'\\
: >\"\$dry_bld/obj/round333_traldf_writer.o\"" "$ADAPTED"

sed -i "/# makenemo rewrites shared work_cfgs.txt/i\\
sha256sum $LDF_PATCH $LDF_SCHEME_PATCH $LDF_WRITER $LDF_LOOP_FIXTURE $LDF_GATE $LDF_TEST \\\
  >>\"\$manifest/toolchain.sha256\"" "$ADAPTED"

sed -i "/touch \"\$TARGET_ROOT\/MY_SRC\/\"\*.F90/i\\
cp $LDF_SOURCE \"\$TARGET_ROOT/MY_SRC/traldf_iso.F90\"\\
cp $LDF_SCHEME_SOURCE \"\$TARGET_ROOT/MY_SRC/traldf_iso_scheme.h90\"\\
cp $LDF_WRITER \"\$TARGET_ROOT/MY_SRC/round333_traldf_writer.F90\"\\
patch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <$LDF_PATCH\\
patch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <$LDF_SCHEME_PATCH" "$ADAPTED"

sed -i "/diff <(grep -v/i\\
grep 'NEMO_L4_R333LI' \"\$TARGET_ROOT/BLD/ppsrc/nemo/round333_traldf_writer.f90\" >/dev/null \\\
  || refuse 'compiled round-333 traldf writer is absent'\\
check_r333_ldf_landing \"\$TARGET_ROOT/BLD/ppsrc/nemo/traldf_iso.f90\" \"\$TARGET_ROOT/BLD/ppsrc/nemo/traldf_iso.f90\" || refuse 'round-333 placement moved under TARGET_ROOT'" "$ADAPTED"
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
[[ "$changed" -gt 16 ]] || refuse "imported driver changed only $changed lines"
for token in \
  "here=$PARENT_DIR" \
  'TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R333LDFI' \
  "EVIDENCE=$EVIDENCE" \
  'DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_traldf_iso_internal_kt2_np2' \
  '$(basename "$TARGET_RUN") == orca1ice_traldf_iso_internal_kt2_np2_*' \
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
synthetic_default=$synthetic_root/orca1ice_traldf_iso_internal_kt2_np2
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
synthetic_retained=$synthetic_root/refused/orca1ice_traldf_iso_internal_kt2_np2_admission_refused_synthetic
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
"$PY" -m pytest -q -p no:xdist "$LDF_TEST"

export R328_ACQ_TARGET=tke332
"$ADAPTED" "$MODE"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND333_TRALDF_PREFLIGHT_READY %s\n' "$TARGET_RUN"
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
for plant in header field-order truncation checksum stamp time-level \
    tracer-pass extent signed-zero wet increment nonfinite rhs-replay a33-replay; do
  log=$TARGET_RUN/admit-r333-traldf-$plant.log
  if "$PY" "$LDF_GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "round-333 traldf admission plant stayed green: $plant"
  fi
  grep -F "STATUS PLANT-FIRED" "$log" >/dev/null \
    || refuse "round-333 traldf admission plant missed its refusal: $plant"
  printf 'ACQUISITION_PLANT_FIRED r333-traldf-%s\n' "$plant"
done
"$PY" "$LDF_GATE" --record-dir "$TARGET_RUN" --expect-commit "$producer" \
  --output "$TARGET_RUN/round333_traldf_admission.json"
( cd "$TARGET_RUN" && sha256sum oracle_rktracer_stage3_kt00000002_r*.bin \
    oracle_r333_traldf_kt00000002_r*.bin \
    round333_tracer_admission.json round333_traldf_admission.json \
    admit-r333-*.log \
    >round333_tracer_outputs.sha256 )
retention_armed=0
printf 'ORCA2_ROUND333_TRALDF_ACQUISITION_PASS %s\n' "$TARGET_RUN"
