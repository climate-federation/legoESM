#!/usr/bin/env bash
# Operator-only passive acquisition. Default mode is a read-only preflight;
# this agent never invokes --run because the round contract reserves NEMO runs
# for the operator.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-328 dyn_zdf acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac
readonly ACQ_TARGET=${R328_ACQ_TARGET:-dynzdf}
case "$ACQ_TARGET" in
  dynzdf|zdfphy|tke331|tke332|tke336) ;;
  *) printf 'REFUSE: R328_ACQ_TARGET must be dynzdf, zdfphy, tke331, tke332, or tke336\n' >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
  [[ -d $R336_TKE_STUB_FLOW_ROOT && $R336_TKE_STUB_FLOW_ROOT == /* &&
     -d ${R336_TKE_STUB_BIN:-} && -x ${R336_TKE_STUB_NEMO:-} &&
     -d ${R336_TKE_STUB_RECORD_ROOT:-} && -d ${R336_TKE_STUB_DETERMINISTIC_SOURCE:-} ]] \
    || { printf 'REFUSE: invalid round-336 TKE stub-flow inputs\n' >&2; exit 64; }
  PATH=$R336_TKE_STUB_BIN:$PATH
  export PATH
fi
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R326HMI
TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R328ZDF
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly BASELINE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition
DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_dynzdf_kt1_np2
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R328ZDFPHY
  DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_zdfphy_kt1_np2
fi
if [[ "$ACQ_TARGET" == tke331 ]]; then
  TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R331TKE
  EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round331/acquisition
  DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tke331_kt1_np2
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  TARGET_CFG=${R328_TARGET_CFG_OVERRIDE:-ORCA2_ORCA1ICE_OMIP_L4_R332RAW}
  EVIDENCE=${R328_EVIDENCE_OVERRIDE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition}
  target_run_name=${R328_TARGET_RUN_NAME_OVERRIDE:-orca1ice_raw_postfold_v_kt1_np2}
  DEFAULT_TARGET_RUN=$EVIDENCE/$target_run_name
  if [[ -n ${R328_TARGET_CFG_OVERRIDE:-}${R328_EVIDENCE_OVERRIDE:-}${R328_TARGET_RUN_NAME_OVERRIDE:-} ]]; then
    [[ ${R328_TARGET_CFG_OVERRIDE:-} == ORCA2_ORCA1ICE_OMIP_L4_R336TKE &&
       ${R328_EVIDENCE_OVERRIDE:-} == /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition &&
       ${R328_TARGET_RUN_NAME_OVERRIDE:-} == orca1ice_tke_internal_kt2_np2 ]] \
      || { printf 'REFUSE: the tke332 override is not the pinned round-336 kt=2 extension\n' >&2; exit 64; }
  fi
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R336TKE
  EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition
  DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tke_internal_kt2_np2
  if [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
    EVIDENCE=$R336_TKE_STUB_FLOW_ROOT/evidence
    DEFAULT_TARGET_RUN=$EVIDENCE/orca1ice_tke_internal_kt2_np2
  fi
fi
readonly ADMISSION_IDENTITY_ONLY=${R328_ADMISSION_IDENTITY_ONLY:-0}
case "$ADMISSION_IDENTITY_ONLY" in
  0|1) ;;
  *) printf 'REFUSE: R328_ADMISSION_IDENTITY_ONLY must be 0 or 1\n' >&2; exit 64 ;;
esac
if [[ "$ADMISSION_IDENTITY_ONLY" == 1 ]]; then
  [[ "$MODE" == --admit-existing && -n ${R328_ADMISSION_TEST_ROOT:-} ]] \
    || { printf 'REFUSE: identity-only admission requires --admit-existing and R328_ADMISSION_TEST_ROOT\n' >&2; exit 64; }
  admission_default_name=$(basename "$DEFAULT_TARGET_RUN")
  EVIDENCE=$(realpath -e -- "$R328_ADMISSION_TEST_ROOT")
  DEFAULT_TARGET_RUN=$EVIDENCE/$admission_default_name
fi
TARGET_RUN=$DEFAULT_TARGET_RUN
if [[ "$MODE" == --admit-existing && -n ${R328_EXISTING_RUN:-} ]]; then
  TARGET_RUN=$(realpath -e -- "$R328_EXISTING_RUN")
  if [[ "$ACQ_TARGET" == zdfphy ]]; then
    expected_retained="$EVIDENCE/refused/orca1ice_zdfphy_kt1_np2_admission_refused_20261024T043519Z_4166419"
  elif [[ "$ACQ_TARGET" == tke331 ]]; then
    expected_retained="$EVIDENCE/refused/orca1ice_tke331_kt1_np2_admission_refused_20261024T103431Z_331969"
  elif [[ "$ACQ_TARGET" == tke332 || "$ACQ_TARGET" == tke336 ]]; then
    [[ $(dirname "$TARGET_RUN") == "$EVIDENCE/refused" && \
       $(basename "$TARGET_RUN") == "$(basename "$DEFAULT_TARGET_RUN")"_* ]] \
      || { printf 'REFUSE: retained round-332 run is outside its refused directory\n' >&2; exit 64; }
    expected_retained=$TARGET_RUN
  else
    printf 'REFUSE: retained admission is unsupported for %s\n' "$ACQ_TARGET" >&2; exit 64
  fi
  [[ "$TARGET_RUN" == "$expected_retained" ]] \
    || { printf 'REFUSE: R328_EXISTING_RUN is not the pinned retained run\n' >&2; exit 64; }
fi
readonly TARGET_CFG EVIDENCE DEFAULT_TARGET_RUN TARGET_RUN
TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
if [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
  TARGET_ROOT=$R336_TKE_STUB_FLOW_ROOT/build/$TARGET_CFG
fi
readonly TARGET_ROOT
readonly SOURCE_DYNZDF=$NEMO_ROOT/src/OCE/DYN/dynzdf.F90
readonly SOURCE_DYNVOR_PRISTINE=$NEMO_ROOT/src/OCE/DYN/dynvor.F90
readonly SOURCE_STAGE_PRISTINE=$NEMO_ROOT/src/OCE/stprk3_stg.F90
readonly SOURCE_STPRK3_PRISTINE=$NEMO_ROOT/src/OCE/stprk3.F90
readonly SOURCE_ZDFTKE_PRISTINE=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90
readonly SOURCE_TRALDF_PRISTINE=$NEMO_ROOT/src/OCE/TRA/traldf_iso.F90
readonly SOURCE_DYNVOR=$SOURCE_ROOT/MY_SRC/dynvor.F90
readonly SOURCE_STAGE=$SOURCE_ROOT/MY_SRC/stprk3_stg.F90
readonly SOURCE_ZDFPHY=$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90
readonly SOURCE_ZDFDDM=$NEMO_ROOT/src/OCE/ZDF/zdfddm.F90
readonly SOURCE_ZDFIWM=$NEMO_ROOT/src/OCE/ZDF/zdfiwm.F90
readonly SOURCE_ZDFSH2=$NEMO_ROOT/src/OCE/ZDF/zdfsh2.F90
readonly SOURCE_ZDFDRG=$NEMO_ROOT/src/OCE/ZDF/zdfdrg.F90
readonly SOURCE_ZDFMXL=$NEMO_ROOT/src/OCE/ZDF/zdfmxl.F90
readonly SOURCE_ZDFTKE=$SOURCE_ROOT/MY_SRC/zdftke.F90
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly CONDA=/home/dbalwada/miniconda3/envs/nemo-build
readonly MEM_MIN_GB=150
readonly RUN_TIMEOUT_S=900
readonly FPP_KEYS='-Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3'
readonly SOURCE_DYNZDF_SHA=4489773bb4cafe46123cfd658b1561188215c9ce0db11f6859fb27cb019bb174
readonly SOURCE_DYNVOR_SHA=a9caad920256ca4441e6fbf4a29c54fead26357d03dd363bdd013959ece2d0e1
readonly SOURCE_STAGE_SHA=2b1636336acc5a5e3326291801c217f263f8ccc3ca875dbfec0d3b789d911e5b
readonly SOURCE_ZDFPHY_SHA=76eeec2a280d56c55bec9e59b579e5290623c408c03962d04652e406334b45fd
readonly SOURCE_ZDFDDM_SHA=27f10278c4ae57a393617cf254cb372efecbf0c26d476afa500199cc0652e544
readonly SOURCE_ZDFIWM_SHA=413868ce3ac1344539a501f425eaf9c61ffa92423a043fd320a0ebd614704dc6
readonly SOURCE_ZDFSH2_SHA=c31f062e282b8710a0b565be7aecbc35b8e741b05a96a393402fcd5a737f28f8
readonly SOURCE_ZDFDRG_SHA=f70dcfc7fa072bad6eeb77b24e8689ea1dfce75808fbf10eddb4c339acf255c2
readonly SOURCE_ZDFMXL_SHA=3a9caebc8599e6e432bbd33d23bf5ef63f7a291cf093ad077e884162c7a5f732
readonly SOURCE_ZDFTKE_SHA=2a7f4ce379d5ad48f915a93f35f06ac16da6685df7d502e01b5297ec6738fd31
readonly SOURCE_ZDFTKE_PRISTINE_SHA=2e78c1a215e4a7ee51dab4b3b19ef446cceb1ef8e5ad9380dd7c482d37e10397
readonly SOURCE_BINARY_SHA=20376f40a78cb1a5491838ff112bfbe738178e4da8da1e43d5bd5c3884c72934
readonly DECK_MANIFEST_SHA=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly INPUT_MANIFEST_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly OCEAN_OUTPUT_SHA=b1e6c021d79013f3303582034350574fae2f0b1e68c976caf3af4061ac51a80f
readonly RETAINED_ZDFPHY_PRODUCER=30cb3f2a81866381cbca2a0d8b36d01ac1794d4a
readonly RETAINED_ZDFPHY_BINARY_SHA=aa16dbf74e4e5bae05df0903038ea71e29f1adf9ab5d9be9752f1d3079dfb684
readonly RETAINED_TKE331_PRODUCER=4e20a8af573e5484d0e331200d20284a8b3c8338
readonly RETAINED_TKE331_BINARY_SHA=ad665a96be3b73ebbe83793c09e7996aba69fd191fb6510288be40d82161b0b3

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynzdf_round328.patch
readonly STAGE_PATCH=$here/stprk3_stg_round328.patch
readonly WRITER=$here/dynzdf_round328_writer.F90
readonly FIXTURE_STUBS=$here/dynzdf_round328_fixture_stubs.F90
readonly FIXTURE=$here/dynzdf_round328_layout_fixture.F90
readonly EXCERPT=$here/round326_ocean.output.excerpt
readonly GATE=$here/../nemo_testcase_l4_orca2_round328_dynzdf_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round328_dynzdf_gate.py
readonly R326_TEST=$REPO/tests/ocean/fidelity/test_orca2_round326_hmi_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round328.md
readonly ZP_DIR=$here/../nemo_testcase_l4_orca2_round328_zdf_terms_acquisition
readonly ZP_PATCH=$ZP_DIR/zdfphy_round328.patch
readonly ZP_DDM_PATCH=$ZP_DIR/zdfddm_round328.patch
readonly ZP_IWM_PATCH=$ZP_DIR/zdfiwm_round328.patch
readonly ZP_WRITER=$ZP_DIR/zdfphy_round328_writer.F90
readonly ZP_FIXTURE=$ZP_DIR/zdfphy_round328_layout_fixture.F90
readonly ZP_GATE=$here/../nemo_testcase_l4_orca2_round328_zdf_terms_gate.py
readonly ZP_TEST=$REPO/tests/ocean/fidelity/test_orca2_round328_zdf_terms_gate.py
readonly ADMITTED_DYN_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_dynzdf_kt1_np2
readonly ADMITTED_ZDFPHY_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_zdfphy_kt1_np2
readonly TKE_DIR=$here/../nemo_testcase_l4_orca2_round331_tke_acquisition
readonly TKE_PHY_PATCH=$TKE_DIR/zdfphy_round331.patch
readonly TKE_PATCH=$TKE_DIR/zdftke_round331.patch
readonly TKE_WRITER=$TKE_DIR/tke_round331_writer.F90
readonly TKE_FIXTURE=$TKE_DIR/tke_round331_layout_fixture.F90
readonly TKE_GATE=$here/../nemo_testcase_l4_orca2_round331_tke_gate.py
readonly TKE_TEST=$REPO/tests/ocean/fidelity/test_orca2_round331_tke_gate.py
readonly TKE_PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round331.md
readonly R332_DIR=$here/../nemo_testcase_l4_orca2_round332_source_order_acquisition
readonly R332_DYNVOR_PATCH=$R332_DIR/dynvor_round332.patch
readonly R332_TKE_PATCH=$R332_DIR/zdftke_round332.patch
readonly R332_STAGE_PATCH=$R332_DIR/stprk3_stg_round332.patch
readonly R332_WRITER=$R332_DIR/round332_writer.F90
readonly R332_FIXTURE=$R332_DIR/round332_layout_fixture.F90
readonly R332_VOR_INTERFACE_FIXTURE=$R332_DIR/round332_vor_interface_fixture.F90
readonly R332_GATE=$here/../nemo_testcase_l4_orca2_round332_source_order_gate.py
readonly R332_TEST=$REPO/tests/ocean/fidelity/test_orca2_round332_source_order_gate.py
readonly R336_TKE_DIR=$here/../nemo_testcase_l4_orca2_round336_tke_acquisition
readonly R336_TKE_PATCH=$R336_TKE_DIR/zdftke_round336.patch
readonly R336_TKE_WRITER=$R336_TKE_DIR/tke_round336_writer.F90
readonly R336_TKE_FIXTURE_STUBS=$R336_TKE_DIR/tke_round336_fixture_stubs.F90
readonly R336_TKE_FIXTURE=$R336_TKE_DIR/tke_round336_fixture.F90
readonly R336_TKE_GATE=$here/../nemo_testcase_l4_orca2_round336_tke_gate.py
readonly R336_TKE_TEST=$REPO/tests/ocean/fidelity/test_orca2_round336_tke_gate.py
readonly R336_TRACER_REFERENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2
readonly R336_LDF_REFERENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2
readonly R336_TRACER_R0_SHA=b7a6af26e7e5e4e30e4cf482a0bcf32dc7e2c21741306b88c48e175d9935116b
readonly R336_TRACER_R1_SHA=a2d60b8a2afb80802ac28bfb3e94e633af56fcc9cb28015c23f760165300d5b1
readonly R336_LDF_R0_SHA=8ff5d8e0d5fc13c39dbe3d82cea168f1a98edaa94f05f10b64a3c6a0ca9fb16d
readonly R336_LDF_R1_SHA=f965b671fb4d1f4c6561b6742b6f18dd43ca7999fa8599e7fbade3d4cfb1a9e6
readonly R336_ICE_R0_SHA=4367aeaeacf7f6449fa6f800a3041206b424cb0a988683e61fae2b420bd2c1ee
readonly R336_ICE_R1_SHA=4c69a6733b13a8f60d419e6cd1e42a1e0f15c902f9d7804c9431ef1d6d717243
readonly HIST_R5_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly HIST_R326_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2
readonly HIST_R328_DYN_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_dynzdf_kt1_np2
readonly HIST_R328_ZDF_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_zdfphy_kt1_np2
readonly HIST_R331_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round331/acquisition/orca1ice_tke331_kt1_np2
readonly HIST_R332_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_source_order_kt1_np2
readonly HIST_R332_RAW_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round332/acquisition/orca1ice_raw_postfold_v_kt1_np2
readonly HIST_R333_TR_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2
readonly HIST_R333_LDF_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

scratch=
R336_PRISTINE_BLD=
created_target=0
created_run=0
run_completed=0
cleanup() {
  local rc=$? refused failure_kind
  trap - EXIT
  set +e
  [[ -z "$scratch" || ! -e "$scratch" ]] || rm -rf -- "$scratch"
  if [[ $rc -ne 0 ]]; then
    if [[ $created_run -ne 0 && -e "$TARGET_RUN" ]]; then
      failure_kind=crashed
      [[ $run_completed -eq 0 ]] || failure_kind=admission_refused
      refused=$EVIDENCE/refused/$(basename "$TARGET_RUN")_${failure_kind}_$(date -u +%Y%m%dT%H%M%SZ)_$$
      if mkdir -p -- "$EVIDENCE/refused" && mv -- "$TARGET_RUN" "$refused"; then
        printf 'R328_FAILED_OUTPUTS_KEPT kind=%s path=%s\n' "$failure_kind" "$refused" >&2
      else
        printf 'R328_FAILED_OUTPUTS_LEFT_IN_PLACE kind=%s path=%s\n' "$failure_kind" "$TARGET_RUN" >&2
      fi
    fi
    if [[ $created_target -ne 0 && -e "$TARGET_ROOT" ]]; then
      printf 'R328_FAILED_BUILD_KEPT path=%s rerun=move-this-directory-aside-first\n' \
        "$TARGET_ROOT" >&2
    fi
    [[ $created_run -eq 0 && $created_target -eq 0 ]] || \
      printf 'R328_FAILED_INVOCATION_HANDLED target=%s run=%s completed=%s\n' \
        "$created_target" "$created_run" "$run_completed" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
r336_tke_stub_prepare_target() {
  [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} && $1 == "$R336_TKE_STUB_FLOW_ROOT"/* ]] \
    || refuse 'stub target build escaped its flow root'
  mkdir -p "$1/EXP00" "$1/MY_SRC"
}
r336_tke_stub_finish_target() {
  local target=$1 template=$2 twin_binary=$3
  [[ -d $template/ppsrc && -d $template/obj ]] \
    || refuse 'stub build template is absent'
  mkdir -p "$target/BLD/bin"
  cp -a "$template/." "$target/BLD/"
  cp -- "$SOURCE_ROOT/BLD/ppsrc/nemo/icevar.f90" \
    "$target/BLD/ppsrc/nemo/icevar.f90"
  cp -- "$R336_TKE_STUB_NEMO" "$target/BLD/bin/nemo.exe"
  cp -- "$R336_TKE_STUB_NEMO" "$twin_binary"
  chmod 700 "$target/BLD/bin/nemo.exe"
  chmod 700 "$twin_binary"
}
r336_tke_stub_assert() {
  local producer binary_sha scenario plant_count
  [[ $TARGET_RUN == "$R336_TKE_STUB_FLOW_ROOT"/* && -x $TARGET_RUN/nemo ]] \
    || refuse 'stub admission escaped its flow root or lacks NEMO'
  producer=$(cat "$TARGET_RUN/producer_commit.txt")
  [[ $producer =~ ^[0-9a-f]{40}$ ]] || refuse 'stub producer manifest is malformed'
  git -C "$REPO" cat-file -e "$producer^{commit}" 2>/dev/null \
    || refuse 'stub producer is not a local commit'
  git -C "$REPO" merge-base --is-ancestor "$producer" HEAD \
    || refuse 'stub producer is not an ancestor'
  binary_sha=$(awk 'NF == 2 && $2 == "nemo" {print $1}' "$TARGET_RUN/nemo.sha256")
  [[ $binary_sha =~ ^[0-9a-f]{64}$ &&
     $(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}') == "$binary_sha" ]] \
    || refuse 'stub binary identity changed'
  scenario=$(cat "$TARGET_RUN/.r336-tke-stub-scenario")
  case "$scenario" in success|retention) ;; *) refuse 'stub scenario is not admissible' ;; esac
  [[ -s $TARGET_RUN/round336_tke_kt2_admission.json ]] \
    || refuse 'stub did not execute the real round-336 admission'
  plant_count=$(find "$TARGET_RUN" -maxdepth 1 -type f \
    -name 'round336_tke_*_plant.log' -printf x | wc -c)
  [[ $plant_count -eq 19 ]] || refuse "stub real admission ran $plant_count/19 plants"
  printf 'R336_TKE_STUB_ADMISSION_PASS scenario=%s path=%s\n' "$scenario" "$TARGET_RUN"
}

r336_reference_args() {
  printf '%s\0' \
    --tracer-reference-dir "$R336_TRACER_REFERENCE" \
    --ldf-reference-dir "$R336_LDF_REFERENCE" \
    --ice-reference-dir "$SOURCE_RUN" \
    --reference-sha256 "tracer_r0000=$R336_TRACER_R0_SHA" \
    --reference-sha256 "tracer_r0001=$R336_TRACER_R1_SHA" \
    --reference-sha256 "ldf_r0000=$R336_LDF_R0_SHA" \
    --reference-sha256 "ldf_r0001=$R336_LDF_R1_SHA" \
    --reference-sha256 "ice_r0000=$R336_ICE_R0_SHA" \
    --reference-sha256 "ice_r0001=$R336_ICE_R1_SHA"
}
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}
refuse_nemo_errors() {
  local output=$1 lines
  [[ -f "$output" ]] || return 0
  lines=$(grep -nF 'E R R O R' "$output" || true)
  [[ -z "$lines" ]] || {
    printf 'NEMO_ERROR_LINES_BEGIN %s\n%s\nNEMO_ERROR_LINES_END\n' \
      "$output" "$lines" >&2
    return 1
  }
}
run_nemo_case() {
  local run_dir=$1 label=$2 rc started
  (
    cd "$run_dir"
    mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB before $label run"
    free -g >run.user.memory.log
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    started=$SECONDS
    set +e
    timeout --kill-after=60 "$RUN_TIMEOUT_S" mpirun -np 2 --oversubscribe ./nemo \
      >run.user.stdout.log 2>&1
    rc=$?
    set -e
    if [[ $label == observed ]]; then
      refuse_nemo_errors "$TARGET_RUN/ocean.output" \
        || refuse 'NEMO ocean.output contains the quoted E R R O R lines above'
      [[ $rc -eq 0 ]] || refuse "mpirun exited $rc"
    else
      refuse_nemo_errors "$run_dir/ocean.output" \
        || refuse "NEMO $label ocean.output contains the quoted E R R O R lines above"
      [[ $rc -eq 0 ]] || refuse "$label mpirun exited $rc"
    fi
    grep 'STOP 0' run.user.stdout.log >/dev/null || refuse "$label NEMO did not report STOP 0"
    printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
}
pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || refuse "missing $label: $path"
  [[ $(sha256sum "$path" | awk '{print $1}') == "$digest" ]] \
    || refuse "$label SHA-256 changed: $path"
}
echo_has() {
  grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null
}
check_echo() {
  echo_has "$1" nn_it000 1 && echo_has "$1" nn_itend 10 && echo_has "$1" ln_tile F \
    && echo_has "$1" ln_drgimp T && echo_has "$1" ln_drgice_imp T \
    && echo_has "$1" ln_dynadv_vec T && echo_has "$1" ln_dynspg_ts T \
    && echo_has "$1" ln_zad_Aimp F && echo_has "$1" nn_mxlice 2 \
    && echo_has "$1" jpl 1 \
    && echo_has "$1" ln_rnf_mouth T && echo_has "$1" ln_zdftke T \
    && echo_has "$1" nn_evdm 0 && echo_has "$1" ln_zdfddm T \
    && echo_has "$1" ln_zdfswm F && echo_has "$1" ln_zdfiwm T
}
nml_int() {
  sed -E 's/!.*//' "$1" | grep -E "^[[:space:]]*$2[[:space:]]*=" | tail -n 1 \
    | sed -E "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*([-0-9]+).*/\1/"
}
check_namelist() {
  local itend stock fsbc
  itend=$(nml_int "$1" nn_itend); stock=$(nml_int "$1" nn_stock); fsbc=$(nml_int "$1" nn_fsbc)
  [[ -n "$itend" && -n "$stock" && -n "$fsbc" ]] || return 1
  (( itend == 10 && itend % fsbc == 0 && stock % fsbc == 0 ))
}
check_ref_cfg() { grep -E "^$1[[:space:]]" "$NEMO_ROOT/cfgs/ref_cfgs.txt" >/dev/null; }
check_no_continuation_insert() {
  awk '/^[ +]/ { kind=substr($0,1,1); line=substr($0,2); if (kind=="+" && prev_kind==" " && prev ~ /&[[:space:]]*$/) bad=1; prev=line; prev_kind=kind } END { exit bad }' "$1"
}
check_no_removed_lines() { [[ $(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$1") -eq 0 ]]; }
check_patched_call_predecessors() {
  awk '{ if ($0 ~ /CALL r(328|331|332)_/ && previous ~ /&[[:space:]]*$/) bad=1; previous=$0 }
       END { exit bad }' "$1"
}
check_r332_tke_landing() {
  awk '
    /^[[:space:]]*END SELECT[[:space:]]*$/ { last_select=NR }
    /CALL r332_tke_row\(Kbb,Kmm,jj,zmxlm/ {
      count++; call_line=NR; after_select=(last_select==NR-2)
    }
    /Vertical eddy viscosity and diffusivity/ {
      if (call_line && !viscosity_line) viscosity_line=NR
    }
    END { exit !(count==1 && after_select && viscosity_line==call_line+3) }
  ' "$1"
}
check_r332_raw_landing() {
  awk '
    /CALL lbc_lnk.*uu\(:,:,:,       Kaa\)/ { in_lbc=1; has_avm=0 }
    in_lbc && /avm_k\(:,:,:\)/ { has_avm=1 }
    in_lbc && /ldfull=.TRUE./ {
      if (has_avm) { target++; lbc_end=NR; state=1 }
      in_lbc=0
      next
    }
    state==1 {
      if (/CALL r332_raw_postfold_v\(kstp,kstg,Kaa,vv\(:,:,:,Kaa\)\)/) {
        raw++; if (NR != lbc_end+1) bad=1; state=2
      } else bad=1
    }
    END { exit !(target==1 && raw==1 && state==2 && !bad) }
  ' "$1"
}
free_ok() {
  local p=$1 need=${2:-8388608}
  while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done
  (( $(df -Pk "$p" | awk 'NR == 2 {print $4}') >= need ))
}
check_r336_tke_landing() {
  awk '
    /^[[:space:]]*SUBROUTINE zdf_tke\(/ {outer=NR; routine="outer"}
    /CALL r336_tke_begin/ {begin=NR}
    /CALL tke_tke\(/ && routine=="outer" {call_tke=NR}
    /CALL r336_tke_after_tke/ {after_tke=NR}
    /CALL tke_avn\(/ && routine=="outer" {call_avn=NR}
    /CALL r336_tke_finish/ {finish=NR}
    /^[[:space:]]*SUBROUTINE tke_tke\(/ {inner=NR; routine="tke"}
    /r336_after_bc/ && routine=="tke" {after_bc=NR}
    /IF\([[:space:]]*ln_lc[[:space:]]*\)/ && routine=="tke" {langmuir=NR}
    /r336_after_langmuir/ {after_langmuir=NR}
    /r336_after_rhs/ {after_rhs=NR}
    /r336_matrix_diag/ {diag=NR}
    /r336_matrix_upper/ {upper=NR}
    /r336_matrix_lower/ {lower=NR}
    /r336_solve_diag/ {solve_diag=NR}
    /r336_solve_rhs/ {solve_rhs=NR}
    /r336_solve_bottom/ {solve_bottom=NR}
    /r336_solve_back/ {solve_back=NR}
    /r336_after_solve/ {after_solve=NR}
    /^[[:space:]]*END SUBROUTINE tke_tke/ {end_tke=NR; routine=""}
    /^[[:space:]]*SUBROUTINE tke_avn\(/ {avn=NR; routine="avn"}
    /r336_pre_sweep_mxl/ {pre_sweep=NR}
    /SELECT CASE[[:space:]]*\([[:space:]]*nn_mxl/ && routine=="avn" {mxl_select=NR}
    /r336_after_lup/ {after_lup=NR}
    /r336_after_ldown/ {after_ldown=NR}
    /r336_final_mxl\(/ {final_mxl=NR}
    /r336_final_mxld\(/ {final_mxld=NR}
    /zav[[:space:]]*=[[:space:]]*rn_ediff/ && routine=="avn" {zav=NR}
    /r336_pre_pdlr_avt/ {pre_avt=NR}
    /IF\([[:space:]]*nn_pdl[[:space:]]*==[[:space:]]*1[[:space:]]*\)/ && routine=="avn" {pdlr=NR}
    /^[[:space:]]*END SUBROUTINE tke_avn/ {end_avn=NR; routine=""}
    END {
      ok=(outer<begin && begin<call_tke && call_tke<after_tke &&
          after_tke<call_avn && call_avn<finish && finish<inner)
      ok=ok && inner<after_bc && after_bc<langmuir && langmuir<after_langmuir
      ok=ok && after_langmuir<after_rhs && after_rhs<diag && diag<upper &&
          upper<lower && lower<solve_diag && solve_diag<solve_rhs &&
          solve_rhs<solve_bottom && solve_bottom<solve_back &&
          solve_back<after_solve && after_solve<end_tke
      ok=ok && avn<pre_sweep && pre_sweep<mxl_select && mxl_select<after_lup &&
          after_lup<after_ldown && after_ldown<final_mxl && final_mxl<final_mxld &&
          final_mxld<zav && zav<pre_avt && pre_avt<pdlr && pdlr<end_avn
      exit !ok
    }
  ' "$1"
}
check_absent() { [[ ! -e "$1" ]]; }
mem_ok() { (( $(free -g | awk '$1 == "Mem:" {print $7}') >= $1 )); }
compile_record_object() {
  "$FC" -c -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer \
    -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize \
    -I "$scratch/mod" -I "$CONDA/include" -J "$scratch/mod" "$1" -o "$2"
}
fp_disassembly_signature() {
  objdump -drwC --no-show-raw-insn "$1" \
    | sed -nE 's/^[[:space:]]*[0-9a-f]+:[[:space:]]+((f(add|sub|mul|div|sqrt)[a-z0-9]*|add(s|p)[sd]|sub(s|p)[sd]|mul(s|p)[sd]|div(s|p)[sd]|sqrt(s|p)[sd]|min(s|p)[sd]|max(s|p)[sd])).*/\1/p' \
    | sort | uniq -c >"$2"
}
compare_fp_disassembly() {
  local left=$1 right=$2 label=$3
  fp_disassembly_signature "$left" "$scratch/${label}.left.fp"
  fp_disassembly_signature "$right" "$scratch/${label}.right.fp"
  cmp -s "$scratch/${label}.left.fp" "$scratch/${label}.right.fp"
}
fp_named_signature() {
  local object=$1 output=$2 wanted=$3
  objdump -drwC --no-show-raw-insn --disassemble="$wanted" "$object" | awk -v wanted="$wanted" '
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
fp_whole_executable_signature() {
  local root=$1 output=$2 object rel
  : >"$output"
  for object in "$root"/obj/*.o; do
    [[ -e $object ]] || continue
    rel=$(basename "$object")
    [[ $rel != tke_round336_writer.o ]] || continue
    objdump -drwC --no-show-raw-insn "$object" | awk -v object="$rel" '
      /^[[:xdigit:]]+[[:space:]]+<[^>]+>:/ {
        fn=$2; gsub(/[<>:]/,"",fn); next
      }
      /^[[:space:]]*[[:xdigit:]]+:/ {
        op=$2
        if (op ~ /^(f(add|sub|mul|div|sqrt)|add(s|p)[sd]|sub(s|p)[sd]|mul(s|p)[sd]|div(s|p)[sd]|sqrt(s|p)[sd]|min(s|p)[sd]|max(s|p)[sd])/) {
          print object,fn,op
        }
      }
    ' >>"$output"
  done
  sort "$output" | uniq -c >"$output.sorted"
  mv "$output.sorted" "$output"
}
compare_whole_executable_fp() {
  local pristine=$1 observed=$2 label=$3
  fp_whole_executable_signature "$pristine" "$scratch/${label}.pristine.fp"
  fp_whole_executable_signature "$observed" "$scratch/${label}.observed.fp"
  cmp -s "$scratch/${label}.pristine.fp" "$scratch/${label}.observed.fp"
}
historical_fp_row() {
  local tool=$1 baseline=$2 producer=$3 function=$4 expected=$5 expected_base=$6 expected_observed=$7
  local base_count observed_count verdict
  fp_named_signature "$baseline" "$scratch/historical-${tool}-${function}.base.fp" "$function" \
    || refuse "historical $tool production function has a zero FP census: $function"
  fp_named_signature "$producer" "$scratch/historical-${tool}-${function}.observed.fp" "$function" \
    || refuse "historical $tool producer function has a zero FP census: $function"
  base_count=$(awk '{n+=$1} END {print n+0}' "$scratch/historical-${tool}-${function}.base.fp")
  observed_count=$(awk '{n+=$1} END {print n+0}' "$scratch/historical-${tool}-${function}.observed.fp")
  [[ $base_count -eq $expected_base && $observed_count -eq $expected_observed ]] \
    || refuse "historical $tool $function count $base_count/$observed_count != $expected_base/$expected_observed"
  verdict=different
  cmp -s "$scratch/historical-${tool}-${function}.base.fp" \
    "$scratch/historical-${tool}-${function}.observed.fp" && verdict=equal
  [[ $verdict == "$expected" ]] \
    || refuse "historical $tool $function census unexpectedly $verdict"
  printf 'R336_HISTORICAL_OBSERVER_FP tool=%s function=%s production=%s producer=%s verdict=%s\n' \
    "$tool" "$function" "$base_count" "$observed_count" "$verdict"
}
fpp() {
  ( cd "$(dirname "$1")" && cpp -Dkey_nosignedzero -P -traditional $FPP_KEYS \
      -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$(basename "$1")" -o "$2" )
}
same_program() {
  if [[ "$ACQ_TARGET" == tke336 ]]; then
    "$PY" "$R336_TKE_GATE" --same-program "$1" --reference-bld "$R336_PRISTINE_BLD" \
      --observer-patch "zdftke=$R336_TKE_PATCH" \
      --new-cfg "$TARGET_CFG" --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
  elif [[ "$ACQ_TARGET" == tke332 ]]; then
    "$PY" "$R332_GATE" --same-program "$1" --reference-bld "$SOURCE_ROOT/BLD" \
      --observer-patch "dynzdf=$PATCH" --observer-patch "stprk3_stg=$scratch/stage_combined.patch" \
      --observer-patch "dynvor=$R332_DYNVOR_PATCH" \
      --observer-patch "zdfphy=$TKE_PHY_PATCH" --observer-patch "zdftke=$scratch/tke_combined.patch" \
      --new-cfg "$TARGET_CFG" --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
  elif [[ "$ACQ_TARGET" == tke331 ]]; then
    "$PY" "$TKE_GATE" --same-program "$1" --reference-bld "$SOURCE_ROOT/BLD" \
      --observer-patch "dynzdf=$PATCH" --observer-patch "stprk3_stg=$STAGE_PATCH" \
      --observer-patch "zdfphy=$TKE_PHY_PATCH" --observer-patch "zdftke=$TKE_PATCH" \
      --new-cfg "$TARGET_CFG" --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
  elif [[ "$ACQ_TARGET" == zdfphy ]]; then
    "$PY" "$ZP_GATE" --same-program "$1" --reference-bld "$SOURCE_ROOT/BLD" \
      --observer-patch "dynzdf=$PATCH" --observer-patch "stprk3_stg=$STAGE_PATCH" \
      --observer-patch "zdfphy=$ZP_PATCH" --observer-patch "zdfddm=$ZP_DDM_PATCH" \
      --observer-patch "zdfiwm=$ZP_IWM_PATCH" --new-cfg "$TARGET_CFG" \
      --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
  else
    "$PY" "$GATE" --same-program "$1" --reference-bld "$SOURCE_ROOT/BLD" \
      --dynzdf-patch "$PATCH" --stage-patch "$STAGE_PATCH" \
      --new-cfg "$TARGET_CFG" --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
  fi
}
compare_deterministic() {
  local reference=$1 candidate=$2 source name count=0
  while IFS= read -r -d '' source; do
    name=$(basename "$source")
    [[ -f "$candidate/$name" ]] || return 1
    if [[ "$name" == oracle_transport_*.bin ]]; then
      compare_transport_defined "$source" "$candidate/$name" || return 1
    else
      cmp -s "$source" "$candidate/$name" || return 1
    fi
    count=$((count+1))
  done < <(find "$reference" -maxdepth 1 -type f \
    \( -name 'oracle_*.bin' -o -name 'ORCA2_00000010_restart*.nc' \) -print0)
  (( count > 0 )) || return 1
  printf 'DETERMINISTIC_PASSIVITY_PASS %s files=%s\n' "$candidate" "$count"
}

compare_r336_twin() {
  local observed=$1 twin=$observed/observer_free_twin name count=0
  "$PY" "$R336_TKE_GATE" --check-output-set "$observed" \
    --output-contract "$observed/round336_output_contract.json" \
    --output-role observed || return 1
  "$PY" "$R336_TKE_GATE" --check-output-set "$twin" \
    --output-contract "$observed/round336_output_contract.json" \
    --output-role twin || return 1
  for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
    ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc \
    ocean.output; do
    cmp -s "$observed/$name" "$twin/$name" || return 1
    count=$((count+1))
  done
  printf 'R336_OBSERVER_FREE_TWIN_PASS compared=%s observer_records=2\n' "$count"
}

compare_transport_defined() {
  local reference=$1 candidate=$2 size payload defined
  [[ -f "$reference" && -f "$candidate" ]] || return 1
  size=$(stat -c %s "$reference")
  [[ $(stat -c %s "$candidate") -eq $size && $size -gt 48 ]] || return 1
  payload=$((size - 48))
  (( payload % 24 == 0 )) || return 1
  defined=$((48 + 2 * payload / 3))
  # MY_SRC/stprk3_stg.F90:323-350: with ln_dynadv_vec=T, zFu/zFv are
  # defined here but zFw is only computed later by tra_adv.F90:267-312.
  # Compare the header and the two defined records; any such change refuses.
  cmp -s -n "$defined" "$reference" "$candidate"
}

cd "$REPO"
[[ -z $(git status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'
readonly COMMIT=$(git rev-parse HEAD)
EXPECTED_PRODUCER=$COMMIT
if [[ "$MODE" == --admit-existing ]]; then
  [[ -f "$TARGET_RUN/producer_commit.txt" &&
     $(wc -l <"$TARGET_RUN/producer_commit.txt") -eq 1 ]] \
    || refuse 'existing run lacks its single-entry producer manifest'
  EXPECTED_PRODUCER=$(awk 'NF == 1 && $1 ~ /^[0-9a-f]{40}$/ {print $1}' \
    "$TARGET_RUN/producer_commit.txt")
  [[ $EXPECTED_PRODUCER =~ ^[0-9a-f]{40}$ ]] \
    || refuse 'existing producer manifest is malformed'
  if [[ "$TARGET_RUN" != "$DEFAULT_TARGET_RUN" && "$ACQ_TARGET" == zdfphy ]]; then
    retained_producer=$RETAINED_ZDFPHY_PRODUCER
    retained_binary_sha=$RETAINED_ZDFPHY_BINARY_SHA
  elif [[ "$TARGET_RUN" != "$DEFAULT_TARGET_RUN" && "$ACQ_TARGET" == tke331 ]]; then
    retained_producer=$RETAINED_TKE331_PRODUCER
    retained_binary_sha=$RETAINED_TKE331_BINARY_SHA
  else
    retained_producer=$EXPECTED_PRODUCER
    [[ -f "$TARGET_RUN/nemo.sha256" && $(wc -l <"$TARGET_RUN/nemo.sha256") -eq 1 ]] \
      || refuse 'retained run lacks its single-entry binary digest manifest'
    retained_binary_sha=$(awk 'NF == 2 && $2 == "nemo" && length($1) == 64 {print $1}' \
      "$TARGET_RUN/nemo.sha256")
    [[ $retained_binary_sha =~ ^[0-9a-f]{64}$ ]] \
      || refuse 'retained binary digest manifest is malformed'
  fi
  [[ "$EXPECTED_PRODUCER" == "$retained_producer" ]] \
    || refuse 'existing run has the wrong producer commit'
  git cat-file -e "$EXPECTED_PRODUCER^{commit}" 2>/dev/null \
    || refuse 'existing producer is not a local commit object'
  git merge-base --is-ancestor "$EXPECTED_PRODUCER" "$COMMIT" \
    || refuse 'existing producer is not an ancestor of the admission repair'
  pin "$retained_binary_sha" "$TARGET_RUN/nemo" 'existing acquisition binary'
  if [[ "$ACQ_TARGET" == tke336 ]]; then
    [[ -f "$TARGET_RUN/observer_free_nemo.sha256" &&
       $(wc -l <"$TARGET_RUN/observer_free_nemo.sha256") -eq 1 ]] \
      || refuse 'existing run lacks its observer-free binary digest manifest'
    ( cd "$TARGET_RUN" && sha256sum -c observer_free_nemo.sha256 >/dev/null ) \
      || refuse 'existing observer-free binary digest manifest changed'
  fi
  if [[ "$ACQ_TARGET" == tke332 ]]; then
    [[ -f "$TARGET_RUN/nemo.sha256" ]] \
      || refuse 'existing run lacks its binary digest manifest'
    ( cd "$TARGET_RUN" && sha256sum -c nemo.sha256 >/dev/null ) \
      || refuse 'existing binary digest manifest changed'
  fi
fi
readonly EXPECTED_PRODUCER
if [[ "$ADMISSION_IDENTITY_ONLY" == 1 ]]; then
  route=in-place
  [[ "$TARGET_RUN" == "$DEFAULT_TARGET_RUN" ]] || route=retained
  printf 'R328_SYNTHETIC_ADMISSION_IDENTITY_PASS route=%s run=%s\n' "$route" "$TARGET_RUN"
  exit 0
fi
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$STAGE_PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" "$EXCERPT" \
  "$GATE" "$TEST" "$R326_TEST" "$PREREG" "$SOURCE_DYNZDF" "$SOURCE_STAGE" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/ocean.output" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" \
  "$SOURCE_ROOT/BLD/arch_nemo.fcm" "$SOURCE_ROOT/BLD/ppsrc/nemo/dynzdf.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90"; do
  [[ -f "$path" ]] || refuse "missing input $path"
done
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  for path in "$ZP_PATCH" "$ZP_DDM_PATCH" "$ZP_IWM_PATCH" "$ZP_WRITER" \
    "$ZP_FIXTURE" "$ZP_GATE" "$ZP_TEST" \
    "$SOURCE_ZDFPHY" "$SOURCE_ZDFDDM" "$SOURCE_ZDFIWM" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfphy.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfddm.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfiwm.f90"; do
    [[ -f "$path" ]] || refuse "missing zdf_phy input $path"
  done
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  for path in "$TKE_PHY_PATCH" "$TKE_PATCH" "$TKE_WRITER" "$TKE_FIXTURE" \
    "$TKE_GATE" "$TKE_TEST" "$TKE_PREREG" "$SOURCE_ZDFPHY" "$SOURCE_ZDFSH2" \
    "$SOURCE_ZDFDRG" "$SOURCE_ZDFMXL" "$SOURCE_ZDFTKE" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfphy.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfsh2.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfdrg.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfmxl.f90" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdftke.f90" \
    "$ADMITTED_ZDFPHY_RUN/oracle_zdfphy_kt00000001_r0000.bin" \
    "$ADMITTED_ZDFPHY_RUN/oracle_zdfphy_kt00000001_r0001.bin"; do
    [[ -f "$path" ]] || refuse "missing round-331 TKE input $path"
  done
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  for path in "$R332_DYNVOR_PATCH" "$R332_TKE_PATCH" "$R332_STAGE_PATCH" \
    "$R332_WRITER" "$R332_FIXTURE" \
    "$R332_VOR_INTERFACE_FIXTURE" \
    "$R332_GATE" "$R332_TEST" "$SOURCE_DYNVOR" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/dynvor.f90"; do
    [[ -f "$path" ]] || refuse "missing round-332 source-order input $path"
  done
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  for path in "$R336_TKE_PATCH" "$R336_TKE_WRITER" "$R336_TKE_FIXTURE_STUBS" \
    "$R336_TKE_FIXTURE" "$R336_TKE_GATE" "$R336_TKE_TEST" "$SOURCE_ZDFTKE" \
    "$SOURCE_ROOT/BLD/ppsrc/nemo/zdftke.f90" \
    "$R336_TRACER_REFERENCE/oracle_rktracer_stage3_kt00000002_r0000.bin" \
    "$R336_TRACER_REFERENCE/oracle_rktracer_stage3_kt00000002_r0001.bin" \
    "$R336_LDF_REFERENCE/oracle_r333_traldf_kt00000002_r0000.bin" \
    "$R336_LDF_REFERENCE/oracle_r333_traldf_kt00000002_r0001.bin" \
    "$SOURCE_RUN/oracle_tke_hmi_kt00000002_r0000.bin" \
    "$SOURCE_RUN/oracle_tke_hmi_kt00000002_r0001.bin" \
    "$SOURCE_STPRK3_PRISTINE" \
    "$SOURCE_STAGE_PRISTINE" "$SOURCE_DYNVOR_PRISTINE" "$SOURCE_ZDFTKE_PRISTINE" \
    "$SOURCE_TRALDF_PRISTINE" "$SOURCE_ZDFPHY" "$SOURCE_ZDFDDM" "$SOURCE_ZDFIWM" \
    "$HIST_R5_RUN/nemo" "$HIST_R326_RUN/nemo" \
    "$HIST_R328_DYN_RUN/nemo" "$HIST_R328_ZDF_RUN/nemo" "$HIST_R331_RUN/nemo" \
    "$HIST_R332_RUN/nemo" "$HIST_R332_RAW_RUN/nemo" "$HIST_R333_TR_RUN/nemo" \
    "$HIST_R333_LDF_RUN/nemo"; do
    [[ -f "$path" ]] || refuse "missing round-336 TKE input $path"
  done
fi
pin "$SOURCE_DYNZDF_SHA" "$SOURCE_DYNZDF" 'source dynzdf'
pin "$SOURCE_STAGE_SHA" "$SOURCE_STAGE" 'round-326 record stprk3_stg'
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  pin "$SOURCE_ZDFPHY_SHA" "$SOURCE_ZDFPHY" 'source zdfphy'
  pin "$SOURCE_ZDFDDM_SHA" "$SOURCE_ZDFDDM" 'source zdfddm'
  pin "$SOURCE_ZDFIWM_SHA" "$SOURCE_ZDFIWM" 'source zdfiwm'
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  pin "$SOURCE_ZDFPHY_SHA" "$SOURCE_ZDFPHY" 'source zdfphy'
  pin "$SOURCE_ZDFSH2_SHA" "$SOURCE_ZDFSH2" 'source zdfsh2'
  pin "$SOURCE_ZDFDRG_SHA" "$SOURCE_ZDFDRG" 'source zdfdrg'
  pin "$SOURCE_ZDFMXL_SHA" "$SOURCE_ZDFMXL" 'source zdfmxl'
  pin "$SOURCE_ZDFTKE_SHA" "$SOURCE_ZDFTKE" 'source zdftke'
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  pin "$SOURCE_DYNVOR_SHA" "$SOURCE_DYNVOR" 'source dynvor'
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  pin "$SOURCE_ZDFTKE_SHA" "$SOURCE_ZDFTKE" 'source zdftke'
  pin "$SOURCE_ZDFTKE_PRISTINE_SHA" "$SOURCE_ZDFTKE_PRISTINE" 'pristine source zdftke'
  pin "$R336_TRACER_R0_SHA" "$R336_TRACER_REFERENCE/oracle_rktracer_stage3_kt00000002_r0000.bin" 'round-333 tracer replay rank 0'
  pin "$R336_TRACER_R1_SHA" "$R336_TRACER_REFERENCE/oracle_rktracer_stage3_kt00000002_r0001.bin" 'round-333 tracer replay rank 1'
  pin "$R336_LDF_R0_SHA" "$R336_LDF_REFERENCE/oracle_r333_traldf_kt00000002_r0000.bin" 'round-333 LDF replay rank 0'
  pin "$R336_LDF_R1_SHA" "$R336_LDF_REFERENCE/oracle_r333_traldf_kt00000002_r0001.bin" 'round-333 LDF replay rank 1'
  pin "$R336_ICE_R0_SHA" "$SOURCE_RUN/oracle_tke_hmi_kt00000002_r0000.bin" 'round-326 ice replay rank 0'
  pin "$R336_ICE_R1_SHA" "$SOURCE_RUN/oracle_tke_hmi_kt00000002_r0001.bin" 'round-326 ice replay rank 1'
  pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$HIST_R5_RUN/nemo" 'round-5 producer binary'
  pin 20376f40a78cb1a5491838ff112bfbe738178e4da8da1e43d5bd5c3884c72934 "$HIST_R326_RUN/nemo" 'round-326 producer binary'
  pin eead2832bfe824810d40fb32014d86c9b6b7ece52eba86673c48d5a0a5878bb5 "$HIST_R328_DYN_RUN/nemo" 'round-328 dyn producer binary'
  pin aa16dbf74e4e5bae05df0903038ea71e29f1adf9ab5d9be9752f1d3079dfb684 "$HIST_R328_ZDF_RUN/nemo" 'round-328 zdf producer binary'
  pin ad665a96be3b73ebbe83793c09e7996aba69fd191fb6510288be40d82161b0b3 "$HIST_R331_RUN/nemo" 'round-331 producer binary'
  pin 9b27cdd323e847cd31d892a30af9731d4839a27b01147a42a7100105df6f3503 "$HIST_R332_RUN/nemo" 'round-332 source producer binary'
  pin 5791908de5c6181d0f12235b073e98992f70f92849d38bca6a8328dd2167e192 "$HIST_R332_RAW_RUN/nemo" 'round-332 raw producer binary'
  pin c2afb5b26a0b6c9bff941cf35b68c62d9c21d2ac8c44755498a580bc389757e5 "$HIST_R333_TR_RUN/nemo" 'round-333 tracer producer binary'
  pin 4eb9f3701c647cdee8c8ae2c992c7eceec58e6844de92aa285581aabbe0fd539 "$HIST_R333_LDF_RUN/nemo" 'round-333 LDF producer binary'
fi
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
pin "$OCEAN_OUTPUT_SHA" "$SOURCE_RUN/ocean.output" 'source ocean.output'

command -v timeout >/dev/null || refuse 'no timeout binary'
mkdir -p "$(dirname "$EVIDENCE")"
scratch=$(mktemp -d "$(dirname "$EVIDENCE")/.r328-scratch.XXXXXX")
plants=$scratch/plants; mkdir "$plants" "$scratch/mod"

# Checklist (1): additions-only and no insertion following a continued line.
patch_files=()
[[ "$ACQ_TARGET" == tke336 ]] || patch_files+=("$PATCH" "$STAGE_PATCH")
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  patch_files+=("$ZP_PATCH" "$ZP_DDM_PATCH" "$ZP_IWM_PATCH")
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  patch_files+=("$TKE_PHY_PATCH" "$TKE_PATCH")
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  patch_files+=("$R332_DYNVOR_PATCH" "$R332_TKE_PATCH" "$R332_STAGE_PATCH")
fi
[[ "$ACQ_TARGET" != tke336 ]] || patch_files+=("$R336_TKE_PATCH")
for patch_file in "${patch_files[@]}"; do
  check_no_removed_lines "$patch_file" || refuse "observer patch removes a source line: $patch_file"
  check_no_continuation_insert "$patch_file" || refuse "patch inserts after a continuation: $patch_file"
done
printf ' x = a + &\n+ CALL planted()\n' >"$plants/continuation.patch"
plant_must_refuse continuation check_no_continuation_insert "$plants/continuation.patch"
printf 'x = a + &\nCALL r331_planted()\n' >"$plants/observer-continuation.F90"
plant_must_refuse observer-continuation \
  check_patched_call_predecessors "$plants/observer-continuation.F90"
# Checklist (2): makenemo -r must name a registered reference config.
check_ref_cfg "$REFERENCE_CFG" || refuse "$REFERENCE_CFG not in ref_cfgs.txt"
plant_must_refuse ref-cfg check_ref_cfg ORCA2_OMIP_L4
# Checklist (3): the copied deck must satisfy NEMO's timestep/restart rules.
check_namelist "$SOURCE_RUN/namelist_cfg" || refuse 'source namelist breaks itend/stock/fsbc rules'
sed -E 's/^([[:space:]]*nn_itend[[:space:]]*=[[:space:]]*)[0-9]+/\1 9/' \
  "$SOURCE_RUN/namelist_cfg" >"$plants/namelist_cfg"
plant_must_refuse namelist check_namelist "$plants/namelist_cfg"
# LOG-CHECK: a committed excerpt is pinned to the real round-326 output.
[[ $(sed -nE 's/^# Complete source SHA-256: ([0-9a-f]{64})$/\1/p' "$EXCERPT") == "$OCEAN_OUTPUT_SHA" ]] \
  || refuse 'committed ocean.output excerpt has the wrong source SHA'
check_echo "$EXCERPT" || refuse 'real ocean.output excerpt lacks frozen selectors'
sed -E 's/(ln_zad_Aimp[[:space:]]*=[[:space:]]*)F/\1T/' "$EXCERPT" >"$plants/ocean.output"
plant_must_refuse echo check_echo "$plants/ocean.output"
sed -E 's/(nn_mxlice[[:space:]]*=[[:space:]]*)2/\1 0/' "$EXCERPT" >"$plants/ocean.output.mxlice"
plant_must_refuse echo-mxlice check_echo "$plants/ocean.output.mxlice"
sed -E 's/(jpl[[:space:]]*=[[:space:]]*)1/\1 5/' "$EXCERPT" >"$plants/ocean.output.jpl"
plant_must_refuse echo-jpl check_echo "$plants/ocean.output.jpl"
printf 'before\nE R R O R staged namelist rejected\nafter\n' >"$plants/ocean.output.error"
plant_must_refuse ocean-output-error-lines \
  refuse_nemo_errors "$plants/ocean.output.error"
# Checklist (4)+(5): compiled exact writer layout, every expected frame, A4,
# and one-ULP A5 plants. The imported round-326 test proves ELF root-only normalisation.
tests=("$TEST")
[[ "$ACQ_TARGET" != zdfphy ]] || tests+=("$ZP_TEST")
[[ "$ACQ_TARGET" != tke331 ]] || tests+=("$TKE_TEST")
[[ "$ACQ_TARGET" != tke332 ]] || tests+=("$TKE_TEST" "$R332_TEST")
[[ "$ACQ_TARGET" != tke336 ]] || tests+=("$R336_TKE_TEST")
"$PY" -m pytest -q -p no:cacheprovider "${tests[@]}" \
  "$R326_TEST::test_objects_equal_ignores_only_the_embedded_config_root"
if [[ "$ACQ_TARGET" == tke332 ]]; then
  vor_fixture_exe=$scratch/round332_vor_interface_fixture
  "$FC" -O3 -fcheck=bounds \
    -ffree-line-length-none "$R332_VOR_INTERFACE_FIXTURE" -o "$vor_fixture_exe"
  "$vor_fixture_exe" >"$scratch/vor-interface.clean.log" 2>&1
  grep 'R332_VOR_INTERFACE_PASS 94 152 31 T' "$scratch/vor-interface.clean.log" >/dev/null \
    || refuse 'full-shape vor_een interface fixture did not preserve negative zero'
  if "$vor_fixture_exe" inner \
      >"$scratch/vor-interface.inner.log" 2>&1; then
    refuse 'inner-domain vor_een scratch plant stayed green'
  fi
  grep 'vor_een explicit-shape extent mismatch' "$scratch/vor-interface.inner.log" >/dev/null \
    || refuse 'inner-domain vor_een scratch plant missed extent guard'
  printf 'PREFLIGHT_PLANT_FIRED vor-een-inner-shape\n'
fi
# Checklist (6): disk checks work before the evidence directory exists.
for mount in "$EVIDENCE" "$NEMO_ROOT"; do
  free_ok "$mount" || refuse "$mount has under 8 GB free"
done
free_ok "$EVIDENCE/not/created/yet" || refuse 'disk check fails on a missing parent'
plant_must_refuse disk free_ok "$NEMO_ROOT" 999999999999999
mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB available memory"
plant_must_refuse memory mem_ok 999999
# Checklist (7): both build and run targets must be absent.
if [[ "$MODE" != --admit-existing ]]; then
  check_absent "$TARGET_ROOT" && check_absent "$TARGET_RUN" \
    || refuse 'target config or run directory already exists'
fi
plant_must_refuse target-exists check_absent "$SOURCE_RUN"

# Prove all inherited deterministic frames and terminal restarts already pass
# between round 5 and its admitted round-326 record. Diagnostic grid NetCDFs
# are deliberately excluded because their TimeStamp attributes are wall-clock.
compare_deterministic "$BASELINE_RUN" "$SOURCE_RUN" \
  || refuse 'round-326 deterministic outputs differ from round 5'

# Syntax proof uses the record program's stprk3_stg, not pristine NEMO.
dry=$scratch/dry; mkdir "$dry"
if [[ "$ACQ_TARGET" != tke336 ]]; then
  cp "$SOURCE_DYNZDF" "$dry/dynzdf.F90"
  cp "$SOURCE_STAGE" "$dry/stprk3_stg.F90"
  cp "$WRITER" "$dry/dynzdf_round328_writer.F90"
  patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$STAGE_PATCH"
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  cp "$SOURCE_ZDFPHY" "$dry/zdfphy.F90"
  cp "$SOURCE_ZDFDDM" "$dry/zdfddm.F90"
  cp "$SOURCE_ZDFIWM" "$dry/zdfiwm.F90"
  cp "$ZP_WRITER" "$dry/zdfphy_round328_writer.F90"
  patch -s --fuzz=0 -p0 -d "$dry" <"$ZP_PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$ZP_DDM_PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$ZP_IWM_PATCH"
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  cp "$SOURCE_ZDFPHY" "$dry/zdfphy.F90"
  cp "$SOURCE_ZDFTKE" "$dry/zdftke.F90"
  cp "$TKE_WRITER" "$dry/tke_round331_writer.F90"
  patch -s --fuzz=0 -p0 -d "$dry" <"$TKE_PHY_PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$TKE_PATCH"
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  cp "$SOURCE_DYNVOR" "$dry/dynvor.F90"
  cp "$R332_WRITER" "$dry/round332_writer.F90"
  patch -s --fuzz=0 -p0 -d "$dry" <"$R332_DYNVOR_PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$R332_TKE_PATCH"
  patch -s --fuzz=0 -p0 -d "$dry" <"$R332_STAGE_PATCH"
  if diff -u --label zdftke.F90 --label zdftke.F90 \
      "$SOURCE_ZDFTKE" "$dry/zdftke.F90" >"$scratch/tke_combined.patch"; then
    refuse 'round-332 combined TKE observer patch is empty'
  else
    [[ $? -eq 1 ]] || refuse 'could not construct combined TKE observer patch'
  fi
  if diff -u --label stprk3_stg.F90 --label stprk3_stg.F90 \
      "$SOURCE_STAGE" "$dry/stprk3_stg.F90" >"$scratch/stage_combined.patch"; then
    refuse 'round-332 combined stage observer patch is empty'
  else
    [[ $? -eq 1 ]] || refuse 'could not construct combined stage observer patch'
  fi
  check_r332_tke_landing "$dry/zdftke.F90" \
    || refuse 'round-332 TKE observer did not land between END SELECT and viscosity loop'
  check_r332_raw_landing "$dry/stprk3_stg.F90" \
    || refuse 'round-332 raw V observer is not immediately after the l_zdfsh2 fold'
  awk '
    /CALL r332_tke_row\(Kbb,Kmm,jj,zmxlm/ { skip=1; next }
    skip { skip=0; next }
    /zemlp = SQRT/ {
      print "                  CALL r332_tke_row(Kbb,Kmm,jj,zmxlm(ntsi:ntei,1:jpkm1), &"
      print "                     & zmxld(ntsi:ntei,1:jpkm1),p_pdlr(ntsi:ntei,jj,2:jpkm1))"
    }
    { print }
  ' "$dry/zdftke.F90" >"$plants/zdftke.misplaced.F90"
  plant_must_refuse tke-observer-misplaced \
    check_r332_tke_landing "$plants/zdftke.misplaced.F90"
  awk '
    /CALL r332_raw_postfold_v/ { saved=$0; next }
    /^[[:space:]]*ELSE[[:space:]]*$/ && saved != "" { print; print saved; saved=""; next }
    { print }
  ' "$dry/stprk3_stg.F90" >"$plants/stprk3_stg.raw-misplaced.F90"
  plant_must_refuse raw-observer-misplaced \
    check_r332_raw_landing "$plants/stprk3_stg.raw-misplaced.F90"
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  cp "$SOURCE_ZDFTKE_PRISTINE" "$dry/zdftke.F90"
  cp "$R336_TKE_WRITER" "$dry/tke_round336_writer.F90"
  patch -s --fuzz=0 -p0 -d "$dry" <"$R336_TKE_PATCH"
fi
if [[ "$ACQ_TARGET" != tke336 ]]; then
  for unit in dynzdf stprk3_stg; do
    check_patched_call_predecessors "$dry/$unit.F90" \
      || refuse "observer call inserted after a continuation: $unit.F90"
  done
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  for unit in zdfphy zdfddm zdfiwm; do
    check_patched_call_predecessors "$dry/$unit.F90" \
      || refuse "observer call inserted after a continuation: $unit.F90"
  done
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  for unit in zdfphy zdftke; do
    check_patched_call_predecessors "$dry/$unit.F90" \
      || refuse "observer call inserted after a continuation: $unit.F90"
  done
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  check_patched_call_predecessors "$dry/zdftke.F90" \
    || refuse 'observer call inserted after a continuation: zdftke.F90'
  [[ $(grep -Fc 'CALL r336_tke_' "$dry/zdftke.F90") -eq 3 ]] \
    || refuse 'round-336 outside-loop TKE call census changed'
  [[ $(grep -Fc 'R336_TKE_COPYOUT' "$dry/zdftke.F90") -eq 17 ]] \
    || refuse 'round-336 TKE copy-out census changed'
  ! grep -F 'r336_pre_pdlr_zav' "$dry/zdftke.F90" >/dev/null \
    || refuse 'round-336 still copies zav inside the coefficient loop'
  "$PY" "$R336_TKE_GATE" --read-set-source "$dry/zdftke.F90" \
    --writer-source "$dry/tke_round336_writer.F90"
  sed 's/entry_htau/entry_htau_missing/g' "$dry/tke_round336_writer.F90" \
    >"$plants/tke_round336_writer.read-set-missing.F90"
  plant_must_refuse r336-tke-read-set \
    "$PY" "$R336_TKE_GATE" --read-set-source "$dry/zdftke.F90" \
      --writer-source "$plants/tke_round336_writer.read-set-missing.F90"
fi
if [[ "$ACQ_TARGET" != tke336 ]]; then
  [[ $(grep -Fc 'CALL r328_' "$dry/dynzdf.F90") -eq 17 ]] || refuse 'dynzdf writer call census changed'
  [[ $(grep -Fc 'CALL r328_finish' "$dry/stprk3_stg.F90") -eq 1 ]] || refuse 'stage writer call census changed'
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  [[ $(grep -Fc 'CALL r328_zp_' "$dry/zdfphy.F90") -eq 3 ]] \
    || refuse 'zdfphy writer call census changed'
  [[ $(grep -Fc 'CALL r328_zp_ddm_cell' "$dry/zdfddm.F90") -eq 1 ]] \
    || refuse 'zdfddm writer call census changed'
  [[ $(grep -Fc 'CALL r328_zp_iwm_level' "$dry/zdfiwm.F90") -eq 1 ]] \
    || refuse 'zdfiwm writer call census changed'
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  [[ $(grep -Fc 'CALL r331_' "$dry/zdfphy.F90") -eq 6 ]] \
    || refuse 'round-331 zdfphy writer call census changed'
  [[ $(grep -Fc 'CALL r331_' "$dry/zdftke.F90") -eq 4 ]] \
    || refuse 'round-331 zdftke writer call census changed'
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  check_patched_call_predecessors "$dry/dynvor.F90" \
    || refuse 'observer call inserted after a continuation: dynvor.F90'
  [[ $(grep -Fc 'CALL r332_vor_' "$dry/dynvor.F90") -eq 3 ]] \
    || refuse 'round-332 dynvor observer call census changed'
  [[ $(grep -Fc 'CALL r332_tke_row' "$dry/zdftke.F90") -eq 1 ]] \
    || refuse 'round-332 TKE observer call census changed'
  [[ $(grep -Fc 'CALL r332_tke_begin' "$dry/zdftke.F90") -eq 1 ]] \
    || refuse 'round-336 TKE context observer call census changed'
  [[ $(grep -Fc 'CALL r332_raw_postfold_v' "$dry/stprk3_stg.F90") -eq 1 ]] \
    || refuse 'round-332 raw V observer call census changed'
fi
cp "$SOURCE_ROOT"/BLD/inc/*.mod "$scratch/mod/"
if [[ "$ACQ_TARGET" == tke336 ]]; then
  # The source configuration is an acquisition build with fifteen inherited
  # observer-bearing NEMO units.  Synthesize the pristine executable baseline
  # by replacing every one from NEMO src and removing its private writer.
  R336_PRISTINE_BLD=$scratch/r336_pristine_bld
  mkdir "$R336_PRISTINE_BLD"
  cp -as "$SOURCE_ROOT/BLD/ppsrc" "$R336_PRISTINE_BLD/ppsrc"
  cp -as "$SOURCE_ROOT/BLD/obj" "$R336_PRISTINE_BLD/obj"
  for name in arch_nemo.fcm cpp.fcm bld.cfg; do
    cp "$SOURCE_ROOT/BLD/$name" "$R336_PRISTINE_BLD/$name"
  done
  rm -f "$R336_PRISTINE_BLD/ppsrc/nemo/zdftke_round326_writer.f90" \
    "$R336_PRISTINE_BLD/obj/zdftke_round326_writer.o"
  r336_pristine_specs=(
    "dynhpg:$NEMO_ROOT/src/OCE/DYN/dynhpg.F90"
    "dynspg_ts:$NEMO_ROOT/src/OCE/DYN/dynspg_ts.F90"
    "dynvor:$NEMO_ROOT/src/OCE/DYN/dynvor.F90"
    "eosbn2:$NEMO_ROOT/src/OCE/TRA/eosbn2.F90"
    "icedyn_adv_pra:$NEMO_ROOT/src/ICE/icedyn_adv_pra.F90"
    "icedyn_rhg_evp:$NEMO_ROOT/src/ICE/icedyn_rhg_evp.F90"
    "icestp:$NEMO_ROOT/src/ICE/icestp.F90"
    "icethd:$NEMO_ROOT/src/ICE/icethd.F90"
    "sbcblk:$NEMO_ROOT/src/OCE/SBC/sbcblk.F90"
    "stp2d:$NEMO_ROOT/src/OCE/stp2d.F90"
    "stprk3:$NEMO_ROOT/src/OCE/stprk3.F90"
    "stprk3_stg:$NEMO_ROOT/src/OCE/stprk3_stg.F90"
    "traadv:$NEMO_ROOT/src/OCE/TRA/traadv.F90"
    "traqsr:$NEMO_ROOT/src/OCE/TRA/traqsr.F90"
    "zdftke:$NEMO_ROOT/src/OCE/ZDF/zdftke.F90"
  )
  [[ $(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f -name '*.F90' | wc -l) -eq 16 ]] \
    || refuse 'round-326 inherited Fortran source census moved'
  for spec in "${r336_pristine_specs[@]}"; do
    unit=${spec%%:*}; source=${spec#*:}
    [[ -f $source ]] || refuse "missing pristine NEMO source $source"
    rm -f "$R336_PRISTINE_BLD/ppsrc/nemo/$unit.f90" "$R336_PRISTINE_BLD/obj/$unit.o"
    fpp "$source" "$R336_PRISTINE_BLD/ppsrc/nemo/$unit.f90"
    compile_record_object "$R336_PRISTINE_BLD/ppsrc/nemo/$unit.f90" \
      "$R336_PRISTINE_BLD/obj/$unit.o"
  done
  printf 'R336_PRISTINE_EXECUTABLE_BASELINE_PASS replaced=%s removed_writers=1 objects=%s\n' \
    "${#r336_pristine_specs[@]}" \
    "$(find -L "$R336_PRISTINE_BLD/obj" -maxdepth 1 -type f -name '*.o' | wc -l)"
fi
syntax_units=(dynzdf_round328_writer dynzdf stprk3_stg)
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  syntax_units=(dynzdf_round328_writer zdfphy_round328_writer dynzdf \
    stprk3_stg zdfddm zdfiwm zdfphy)
fi
if [[ "$ACQ_TARGET" == tke331 ]]; then
  syntax_units=(dynzdf_round328_writer tke_round331_writer dynzdf \
    stprk3_stg zdfphy zdftke)
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  syntax_units=(dynzdf_round328_writer tke_round331_writer round332_writer \
    dynzdf stprk3_stg dynvor zdfphy zdftke)
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  syntax_units=(tke_round336_writer zdftke)
fi
for unit in "${syntax_units[@]}"; do
  fpp "$dry/$unit.F90" "$scratch/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch/mod" -J "$scratch/mod" "$scratch/$unit.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$unit"
done

if [[ "$ACQ_TARGET" == tke336 ]]; then
  check_r336_tke_landing "$scratch/zdftke.f90" \
    || refuse 'round-336 TKE observer placement moved in real preprocessed source'
  r336_observers=(r336_tke_begin r336_tke_after_tke r336_tke_finish \
    r336_after_bc r336_after_langmuir r336_after_rhs \
    r336_matrix_diag r336_matrix_upper r336_matrix_lower r336_solve_diag \
    r336_solve_rhs r336_solve_bottom r336_solve_back r336_after_solve \
    r336_pre_sweep_mxl r336_after_lup r336_after_ldown r336_final_mxl \
    r336_final_mxld r336_pre_pdlr_avt)
  for observer in "${r336_observers[@]}"; do
    awk -v token="$observer" '
      index($0,token) && $0 !~ /USE tke_round336_writer/ &&
          $0 !~ /^[[:space:]]*&/ && saved=="" {saved=$0; next}
      {print}
      END {if(saved!="") print saved}
    ' "$scratch/zdftke.f90" >"$plants/zdftke.${observer}.misplaced.f90"
    plant_must_refuse "r336-${observer}-misplaced" check_r336_tke_landing \
      "$plants/zdftke.${observer}.misplaced.f90"
  done
  printf 'R336_TKE_PPSRC_PLACEMENT_PASS observers=%s\n' "${#r336_observers[@]}"
fi

if [[ "$ACQ_TARGET" == tke332 ]]; then
  command -v objdump >/dev/null || refuse 'no objdump for arithmetic passivity proof'
  sed -e '/USE round332_writer, ONLY : r332_raw_postfold_v/d' \
      -e '/round-332 WRITE-only raw post-fold V observer/d' \
      -e '/CALL r332_raw_postfold_v/d' "$dry/stprk3_stg.F90" >"$scratch/stage_arithmetic_base.F90"
  fpp "$scratch/stage_arithmetic_base.F90" "$scratch/stage_arithmetic_base.f90"
  compile_record_object "$scratch/stage_arithmetic_base.f90" "$scratch/stage_arithmetic_base.o"
  compile_record_object "$scratch/stprk3_stg.f90" "$scratch/stage_arithmetic_observed.o"
  compare_fp_disassembly "$scratch/stage_arithmetic_base.o" \
    "$scratch/stage_arithmetic_observed.o" stage \
    || refuse 'stage observer changed the normalized floating-point instruction census'
  printf 'R332_NORMALIZED_OBJECT_DISASSEMBLY_PASS stage_fp=%s\n' \
    "$(awk '{n+=$1} END {print n+0}' "$scratch/stage.left.fp")"
  sed '0,/r1_2 \*/s//(r1_2 + r1_2) */' "$scratch/stage_arithmetic_base.F90" \
    >"$plants/stage.arithmetic.F90"
  fpp "$plants/stage.arithmetic.F90" "$scratch/stage_arithmetic_plant.f90"
  compile_record_object "$scratch/stage_arithmetic_plant.f90" "$scratch/stage_arithmetic_plant.o"
  plant_must_refuse arithmetic-disassembly compare_fp_disassembly \
    "$scratch/stage_arithmetic_base.o" "$scratch/stage_arithmetic_plant.o" stage-plant

  # Checklist item 1: the kt=2 extension activates the already reviewed
  # in-loop record calls, so prove the executing TKE functions retain their
  # nonzero production-flag floating-point instruction multisets.
  fpp "$SOURCE_ZDFTKE" "$scratch/zdftke_arithmetic_base.f90"
  compile_record_object "$scratch/zdftke_arithmetic_base.f90" \
    "$scratch/zdftke_arithmetic_base.o"
  compile_record_object "$scratch/zdftke.f90" "$scratch/zdftke_arithmetic_observed.o"
  for fn in __zdftke_MOD_tke_avn; do
    fp_named_signature "$scratch/zdftke_arithmetic_base.o" \
      "$scratch/${fn}.base.fp" "$fn" \
      || refuse "baseline TKE function has a zero FP census: $fn"
    fp_named_signature "$scratch/zdftke_arithmetic_observed.o" \
      "$scratch/${fn}.observed.fp" "$fn" \
      || refuse "observed TKE function has a zero FP census: $fn"
    cmp -s "$scratch/${fn}.base.fp" "$scratch/${fn}.observed.fp" \
      || refuse "TKE observer changed the production FP census: $fn"
    printf 'R332_TKE_FUNCTION_FP_PASS function=%s instructions=%s\n' "$fn" \
      "$(awk '{n+=$1} END {print n+0}' "$scratch/${fn}.base.fp")"
  done
  sed '0,/zraug = vkarmn \* 2.e5_wp/s//zraug = (vkarmn + vkarmn) * 2.e5_wp/' \
    "$SOURCE_ZDFTKE" >"$plants/zdftke.arithmetic.F90"
  fpp "$plants/zdftke.arithmetic.F90" "$scratch/zdftke_arithmetic_plant.f90"
  compile_record_object "$scratch/zdftke_arithmetic_plant.f90" \
    "$scratch/zdftke_arithmetic_plant.o"
  fp_named_signature "$scratch/zdftke_arithmetic_plant.o" \
    "$scratch/tke_avn.plant.fp" __zdftke_MOD_tke_avn
  plant_must_refuse tke-arithmetic-disassembly cmp -s \
    "$scratch/__zdftke_MOD_tke_avn.base.fp" "$scratch/tke_avn.plant.fp"
fi

if [[ "$ACQ_TARGET" == tke336 ]]; then
  command -v objdump >/dev/null || refuse 'no objdump for TKE arithmetic passivity proof'
  fpp "$SOURCE_ZDFTKE_PRISTINE" "$scratch/zdftke_arithmetic_base.f90"
  compile_record_object "$scratch/zdftke_arithmetic_base.f90" \
    "$scratch/zdftke_arithmetic_base.o"
  compile_record_object "$scratch/zdftke.f90" "$scratch/zdftke_arithmetic_observed.o"
  for fn in __zdftke_MOD_zdf_tke __zdftke_MOD_tke_avn; do
    fp_named_signature "$scratch/zdftke_arithmetic_base.o" \
      "$scratch/${fn}.base.fp" "$fn" \
      || refuse "baseline TKE function has a zero FP census: $fn"
    fp_named_signature "$scratch/zdftke_arithmetic_observed.o" \
      "$scratch/${fn}.observed.fp" "$fn" \
      || refuse "observed TKE function has a zero FP census: $fn"
    cmp -s "$scratch/${fn}.base.fp" "$scratch/${fn}.observed.fp" \
      || refuse "round-336 copy-out changed the production FP census: $fn"
    printf 'R336_TKE_FUNCTION_FP_PASS function=%s instructions=%s\n' "$fn" \
      "$(awk '{n+=$1} END {print n+0}' "$scratch/${fn}.base.fp")"
  done
  if nm "$scratch/zdftke_arithmetic_base.o" | grep '__zdftke_MOD_tke_tke' >/dev/null || \
     nm "$scratch/zdftke_arithmetic_observed.o" | grep '__zdftke_MOD_tke_tke' >/dev/null; then
    refuse 'production tke_tke unexpectedly stopped being fully inlined into zdf_tke'
  fi
  printf 'R336_TKE_INLINE_PASS source_function=tke_tke object_function=__zdftke_MOD_zdf_tke\n'
  sed '0,/zfact2  = 1.5_wp \* rn_Dt \* rn_ediss/s//zfact2  = (1.5_wp + rn_ediff) * rn_Dt * rn_ediss/' \
    "$SOURCE_ZDFTKE_PRISTINE" >"$plants/zdftke.arithmetic.F90"
  fpp "$plants/zdftke.arithmetic.F90" "$scratch/zdftke_arithmetic_plant.f90"
  compile_record_object "$scratch/zdftke_arithmetic_plant.f90" \
    "$scratch/zdftke_arithmetic_plant.o"
  fp_named_signature "$scratch/zdftke_arithmetic_plant.o" \
    "$scratch/zdf_tke.plant.fp" __zdftke_MOD_zdf_tke \
    || refuse 'TKE arithmetic plant produced a zero FP census'
  plant_must_refuse r336-tke-arithmetic-disassembly cmp -s \
    "$scratch/__zdftke_MOD_zdf_tke.base.fp" "$scratch/zdf_tke.plant.fp"

  # Audit the admitted round-331/332 source programs independently.  These
  # are historical measurements, not the source used by the new acquisition.
  cp "$SOURCE_ZDFTKE" "$scratch/historical-r331.F90"
  patch -s --fuzz=0 -p0 "$scratch/historical-r331.F90" <"$TKE_PATCH"
  cp "$scratch/historical-r331.F90" "$scratch/historical-r332.F90"
  patch -s --fuzz=0 -p0 "$scratch/historical-r332.F90" <"$R332_TKE_PATCH"
  # This module is compiled only to provide the checksum interface required by
  # the historical round-331 writer; its dyn_zdf observer patch is not applied
  # to, compiled into, or linked into the round-336 acquisition target.
  fpp "$WRITER" "$scratch/historical_dynzdf_round328_writer.f90"
  compile_record_object "$scratch/historical_dynzdf_round328_writer.f90" \
    "$scratch/historical_dynzdf_round328_writer.o"
  fpp "$TKE_WRITER" "$scratch/tke_round331_writer.f90"
  compile_record_object "$scratch/tke_round331_writer.f90" "$scratch/tke_round331_writer.o"
  fpp "$R332_WRITER" "$scratch/round332_writer.f90"
  compile_record_object "$scratch/round332_writer.f90" "$scratch/round332_writer.o"
  for variant in r331 r332; do
    fpp "$scratch/historical-${variant}.F90" "$scratch/historical-${variant}.f90"
    compile_record_object "$scratch/historical-${variant}.f90" \
      "$scratch/historical-${variant}.o"
  done
  for spec in \
    'base __zdftke_MOD_zdf_tke 1007' \
    'base __zdftke_MOD_tke_avn 783' \
    'r331 __zdftke_MOD_zdf_tke 783' \
    'r331 __zdftke_MOD_tke_tke 1054' \
    'r332 __zdftke_MOD_tke_tke 1054' \
    'r332 __zdftke_MOD_tke_avn 783'; do
    read -r variant fn expected <<<"$spec"
    object=$scratch/zdftke_arithmetic_base.o
    [[ $variant == base ]] || object=$scratch/historical-${variant}.o
    fp_named_signature "$object" "$scratch/historical-${variant}-${fn}.fp" "$fn" \
      || refuse "historical $variant function has zero FP census: $fn"
    actual=$(awk '{n+=$1} END {print n+0}' "$scratch/historical-${variant}-${fn}.fp")
    [[ $actual -eq $expected ]] \
      || refuse "historical $variant $fn FP census $actual != pinned $expected"
    printf 'R336_HISTORICAL_TKE_FP variant=%s function=%s instructions=%s\n' \
      "$variant" "$fn" "$actual"
  done
  if fp_named_signature "$scratch/historical-r332.o" \
      "$scratch/historical-r332-zdf_tke.fp" __zdftke_MOD_zdf_tke; then
    refuse 'round-332 historical zdf_tke unexpectedly retained FP arithmetic'
  fi
  printf 'R336_HISTORICAL_TKE_RESULT admitted_rounds=331,332 arithmetic_differs_from_production=yes\n'

  # Audit every admitted ORCA2 recorder in the requested round set against an
  # observer-free source object at the record flags.  The right-hand side is
  # the pinned binary that actually produced the admitted files, so inherited
  # observers are included in the program verdict rather than reconstructed.
  for source_spec in \
    "stprk3:$SOURCE_STPRK3_PRISTINE" \
    "stprk3_stg:$SOURCE_STAGE_PRISTINE" \
    "dynzdf:$SOURCE_DYNZDF" \
    "zdfphy:$SOURCE_ZDFPHY" \
    "zdfddm:$SOURCE_ZDFDDM" \
    "zdfiwm:$SOURCE_ZDFIWM" \
    "zdftke:$SOURCE_ZDFTKE_PRISTINE" \
    "dynvor:$SOURCE_DYNVOR_PRISTINE" \
    "traldf_iso:$SOURCE_TRALDF_PRISTINE"; do
    unit=${source_spec%%:*}; source=${source_spec#*:}
    fpp "$source" "$scratch/production-${unit}.f90"
    compile_record_object "$scratch/production-${unit}.f90" "$scratch/production-${unit}.o"
  done
  historical_fp_row r5 "$scratch/production-stprk3.o" "$HIST_R5_RUN/nemo" \
    __stprk3_MOD_stp_rk3 equal 30 30
  historical_fp_row r326 "$scratch/production-zdftke.o" "$HIST_R326_RUN/nemo" \
    __zdftke_MOD_zdf_tke equal 1007 1007
  historical_fp_row r326 "$scratch/production-zdftke.o" "$HIST_R326_RUN/nemo" \
    __zdftke_MOD_tke_avn equal 783 783
  historical_fp_row r328-dyn "$scratch/production-dynzdf.o" "$HIST_R328_DYN_RUN/nemo" \
    __dynzdf_MOD_dyn_zdf different 2446 2365
  historical_fp_row r328-dyn "$scratch/production-stprk3_stg.o" "$HIST_R328_DYN_RUN/nemo" \
    __stprk3_stg_MOD_stp_rk3_stg different 1175 1367
  historical_fp_row r328-zdf "$scratch/production-zdfphy.o" "$HIST_R328_ZDF_RUN/nemo" \
    __zdfphy_MOD_zdf_phy equal 121 121
  historical_fp_row r328-zdf "$scratch/production-zdfddm.o" "$HIST_R328_ZDF_RUN/nemo" \
    __zdfddm_MOD_zdf_ddm equal 58 58
  historical_fp_row r328-zdf "$scratch/production-zdfiwm.o" "$HIST_R328_ZDF_RUN/nemo" \
    __zdfiwm_MOD_zdf_iwm equal 423 423
  historical_fp_row r331 "$scratch/production-zdfphy.o" "$HIST_R331_RUN/nemo" \
    __zdfphy_MOD_zdf_phy equal 121 121
  historical_fp_row r331 "$scratch/production-zdftke.o" "$HIST_R331_RUN/nemo" \
    __zdftke_MOD_zdf_tke different 1007 783
  historical_fp_row r332 "$scratch/production-zdfphy.o" "$HIST_R332_RUN/nemo" \
    __zdfphy_MOD_zdf_phy equal 121 121
  historical_fp_row r332 "$scratch/production-dynvor.o" "$HIST_R332_RUN/nemo" \
    __dynvor_MOD_vor_een equal 555 555
  historical_fp_row r332 "$scratch/production-stprk3_stg.o" "$HIST_R332_RAW_RUN/nemo" \
    __stprk3_stg_MOD_stp_rk3_stg different 1175 1367
  historical_fp_row r332 "$scratch/production-zdftke.o" "$HIST_R332_RUN/nemo" \
    __zdftke_MOD_tke_avn equal 783 783
  historical_fp_row r333-tracer "$scratch/production-stprk3_stg.o" "$HIST_R333_TR_RUN/nemo" \
    __stprk3_stg_MOD_stp_rk3_stg different 1175 1367
  historical_fp_row r333-ldf "$scratch/production-stprk3_stg.o" "$HIST_R333_LDF_RUN/nemo" \
    __stprk3_stg_MOD_stp_rk3_stg different 1175 1367
  historical_fp_row r333-ldf "$scratch/production-traldf_iso.o" "$HIST_R333_LDF_RUN/nemo" \
    __traldf_iso_MOD_traldf_iso_lap equal 526 526
  historical_fp_row r333-ldf "$scratch/production-traldf_iso.o" "$HIST_R333_LDF_RUN/nemo" \
    __traldf_iso_MOD_traldf_iso_a33 equal 538 538
  nm "$HIST_R331_RUN/nemo" | grep '__zdftke_MOD_tke_tke' >/dev/null \
    || refuse 'round-331 producer lost its split tke_tke function'
  nm "$HIST_R332_RUN/nemo" | grep '__zdftke_MOD_tke_tke' >/dev/null \
    || refuse 'round-332 producer lost its split tke_tke function'
  nm "$HIST_R332_RUN/nemo" | grep '__dynvor_MOD_dyn_vor' >/dev/null \
    || refuse 'round-332 producer lost its dyn_vor wrapper'
  if nm "$scratch/production-zdftke.o" | grep '__zdftke_MOD_tke_tke' >/dev/null; then
    refuse 'observer-free production unexpectedly split tke_tke'
  fi
  printf 'R336_HISTORICAL_OBSERVER_TOPOLOGY tool=r331 function=tke_tke production=inlined producer=1054 verdict=different\n'
  if nm "$HIST_R331_RUN/nemo" | grep '__zdftke_MOD_tke_avn' >/dev/null; then
    refuse 'round-331 producer unexpectedly retained a separate tke_avn function'
  fi
  printf 'R336_HISTORICAL_OBSERVER_TOPOLOGY tool=r331 function=tke_avn production=783 producer=inlined verdict=different\n'
  if fp_named_signature "$HIST_R332_RUN/nemo" \
      "$scratch/historical-r332-dyn_vor.fp" __dynvor_MOD_dyn_vor; then
    refuse 'round-332 dyn_vor wrapper unexpectedly gained FP arithmetic'
  fi
  printf 'R336_HISTORICAL_OBSERVER_TOPOLOGY tool=r332 function=dyn_vor production=zero-fp producer=zero-fp verdict=equal\n'
  printf 'R336_HISTORICAL_OBSERVER_TOPOLOGY tool=r332 function=zdf_tke production=1007 producer=zero-fp verdict=different\n'
  printf 'R336_HISTORICAL_OBSERVER_TOPOLOGY tool=r332 function=tke_tke production=inlined producer=1054 verdict=different\n'
  printf 'R336_HISTORICAL_OBSERVER_RESULT equal=r5,r326,r328-zdf,r333-ldf-arithmetic different=r328-dyn,r331,r332,r333-stage\n'

  # Checklist items 2, 3, 5 and 14: compile and execute the actual patched
  # NEMO routines with full-grid p_avm/rn2/rn2b/wmask, interior-only A2D(0)
  # en/avt/sh2/dissl, rank-distinct values and signed-zero sentinels.
  fixture_dir=$scratch/r336-tke-fixture
  mkdir "$fixture_dir" "$fixture_dir/mod"
  extract_tke_fixture_source() {
    local source=$1 output=$2
    awk '
      BEGIN {header=1; keep=0}
      header {print; if ($0 ~ /^[[:space:]]*CONTAINS/) header=0; next}
      /^[[:space:]]*SUBROUTINE (zdf_tke|tke_tke|tke_avn)\(/ {keep=1}
      keep {print}
      keep && /^[[:space:]]*END SUBROUTINE (zdf_tke|tke_tke|tke_avn)/ {keep=0}
    ' "$source" | sed -e '/PUBLIC   zdf_tke_init/d' -e '/PUBLIC   tke_rst/d' \
      -e '/PUBLIC   zdf_tke[[:space:]]*!/a\   PUBLIC r336_fixture_config' \
      >"$output.body"
    awk '1; END {
      print "   SUBROUTINE r336_fixture_config"
      print "      ln_mxl0=.TRUE.; ln_mxhsw=.FALSE.; nn_mxlice=2; rn_mxlice=1._wp"
      print "      nn_mxl=3; rn_mxl0=0.001_wp; nn_pdl=1; rn_ediff=0.1_wp; rn_ediss=0.7_wp"
      print "      rn_ebb=67.83_wp; rn_emin=1.e-10_wp; rn_emin0=1.e-4_wp; rn_bshear=1.e-20_wp"
      print "      nn_etau=1; nn_htau=1; nn_bc_surf=0; nn_bc_bot=0; rn_efr=0.05_wp"
      print "      ln_lc=.TRUE.; rn_lc=0.15_wp; nn_eice=1"
      print "      ri_cri=2._wp/(2._wp+rn_ediss/rn_ediff); rmxl_min=0.001_wp"
      print "      ALLOCATE(htau(Nis0:Nie0,Njs0:Nje0),dissl(Nis0:Nie0,Njs0:Nje0,jpk))"
      print "      htau=50._wp; dissl=0.02_wp"
      print "   END SUBROUTINE r336_fixture_config"
      print "END MODULE zdftke"
    }' "$output.body" >"$output"
  }
  extract_tke_fixture_source "$scratch/zdftke.f90" "$fixture_dir/observed.f90"
  extract_tke_fixture_source "$scratch/zdftke_arithmetic_base.f90" \
    "$fixture_dir/reference.tmp.f90"
  sed -e 's/^MODULE zdftke$/MODULE zdftke_reference/' \
    -e 's/^END MODULE zdftke$/END MODULE zdftke_reference/' \
    "$fixture_dir/reference.tmp.f90" >"$fixture_dir/reference.f90"
  fixture_flags=(-O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds \
    -ffree-line-length-none -ffunction-sections)
  "$FC" -c "${fixture_flags[@]}" -J "$fixture_dir/mod" \
    "$R336_TKE_FIXTURE_STUBS" -o "$fixture_dir/stubs.o"
  "$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
    "$scratch/tke_round336_writer.f90" -o "$fixture_dir/writer.o"
  "$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
    "$fixture_dir/reference.f90" -o "$fixture_dir/reference.o"
  "$FC" -c "${fixture_flags[@]}" -I "$fixture_dir/mod" -J "$fixture_dir/mod" \
    "$fixture_dir/observed.f90" -o "$fixture_dir/observed.o"
  "$FC" "${fixture_flags[@]}" -I "$fixture_dir/mod" "$R336_TKE_FIXTURE" \
    "$fixture_dir/stubs.o" "$fixture_dir/writer.o" "$fixture_dir/reference.o" \
    "$fixture_dir/observed.o" -Wl,--gc-sections -o "$fixture_dir/fixture"
  for rank in 0 1; do
    ( cd "$fixture_dir" && ./fixture "$rank" ) >>"$fixture_dir/fixture.log" 2>&1
    grep -E "R336_PATCHED_TKE_FIXTURE_PASS[[:space:]]+$rank([[:space:]]|$)" \
      "$fixture_dir/fixture.log" >/dev/null \
      || refuse "round-336 patched TKE fixture failed for rank $rank"
  done
  "$PY" - "$fixture_dir" <<'PY'
import copy
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round336_tke_gate as gate

root = Path(sys.argv[1])
layout = (7, 6, 5, 3, 5, 3, 4)
for rank in (0, 1):
    record = gate.read_record(root / gate.record_name(rank), expected_layout=layout)
    assert record["header"]["rank"] == rank
    report = gate.verify_record(record)
    assert report["wet_process_interfaces"] > 0
for plant in ("header", "field-order", "truncation", "checksum", "extent", "time-level", "drag-range"):
    try:
        gate.read_record(root / gate.record_name(0), plant=plant, expected_layout=layout)
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-tke-{plant}")
    else:
        raise AssertionError(f"{plant} parser plant stayed green")
for plant in ("entry", "rn2", "langmuir", "solve-steps", "pre-sweep", "sweep-steps", "north-halo-avm-k", "pdlr", "final", "signed-zero", "source-values"):
    record = gate.read_record(root / gate.record_name(0), expected_layout=layout)
    try:
        gate.verify_record(copy.deepcopy(record), plant=plant)
    except gate.GateError:
        print(f"PREFLIGHT_PLANT_FIRED r336-tke-{plant}")
    else:
        raise AssertionError(f"{plant} semantic plant stayed green")
print("R336_PATCHED_TKE_BOUNDS_AND_SOURCE_EXECUTION_PASS ranks=2 fields=51")
PY
fi

# Same-program preflight: for tke336 the baseline is pristine NEMO, with every
# inherited round-326 MY_SRC observer removed.  A source or FP-census plant
# must refuse.
dry_bld=$scratch/dry_bld; mkdir "$dry_bld"
dry_base=$SOURCE_ROOT/BLD
[[ "$ACQ_TARGET" != tke336 ]] || dry_base=$R336_PRISTINE_BLD
cp -as "$dry_base/ppsrc" "$dry_bld/ppsrc"
cp -as "$dry_base/obj" "$dry_bld/obj"
for name in arch_nemo.fcm cpp.fcm bld.cfg; do
  sed "s|$SOURCE_CFG|$TARGET_CFG|g" "$SOURCE_ROOT/BLD/$name" >"$dry_bld/$name"
done
observer_units=()
[[ "$ACQ_TARGET" == tke336 ]] || observer_units=(dynzdf stprk3_stg)
[[ "$ACQ_TARGET" != zdfphy ]] || observer_units+=(zdfphy zdfddm zdfiwm)
[[ "$ACQ_TARGET" != tke331 ]] || observer_units+=(zdfphy zdftke)
[[ "$ACQ_TARGET" != tke332 ]] || observer_units+=(zdfphy zdftke)
[[ "$ACQ_TARGET" != tke332 ]] || observer_units+=(dynvor)
[[ "$ACQ_TARGET" != tke336 ]] || observer_units+=(zdftke)
for unit in "${observer_units[@]}"; do
  rm -f "$dry_bld/ppsrc/nemo/$unit.f90"
done
if [[ "$ACQ_TARGET" != tke336 ]]; then
  cp "$scratch/dynzdf.f90" "$dry_bld/ppsrc/nemo/dynzdf.f90"
  cp "$scratch/stprk3_stg.f90" "$dry_bld/ppsrc/nemo/stprk3_stg.f90"
  cp "$scratch/dynzdf_round328_writer.f90" \
    "$dry_bld/ppsrc/nemo/dynzdf_round328_writer.f90"
  : >"$dry_bld/obj/dynzdf_round328_writer.o"
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  for unit in zdfphy zdfddm zdfiwm; do
    cp "$scratch/$unit.f90" "$dry_bld/ppsrc/nemo/$unit.f90"
  done
  cp "$scratch/zdfphy_round328_writer.f90" \
    "$dry_bld/ppsrc/nemo/zdfphy_round328_writer.f90"
  : >"$dry_bld/obj/zdfphy_round328_writer.o"
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  for unit in zdfphy zdftke; do
    cp "$scratch/$unit.f90" "$dry_bld/ppsrc/nemo/$unit.f90"
  done
  cp "$scratch/tke_round331_writer.f90" \
    "$dry_bld/ppsrc/nemo/tke_round331_writer.f90"
  : >"$dry_bld/obj/tke_round331_writer.o"
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  cp "$scratch/dynvor.f90" "$dry_bld/ppsrc/nemo/dynvor.f90"
  cp "$scratch/round332_writer.f90" \
    "$dry_bld/ppsrc/nemo/round332_writer.f90"
  : >"$dry_bld/obj/round332_writer.o"
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  cp "$scratch/zdftke.f90" "$dry_bld/ppsrc/nemo/zdftke.f90"
  cp "$scratch/tke_round336_writer.f90" \
    "$dry_bld/ppsrc/nemo/tke_round336_writer.f90"
  rm -f "$dry_bld/obj/zdftke.o"
  cp "$scratch/zdftke_arithmetic_observed.o" "$dry_bld/obj/zdftke.o"
  compile_record_object "$scratch/tke_round336_writer.f90" \
    "$dry_bld/obj/tke_round336_writer.o"
fi
same_program "$dry_bld"
if [[ "$ACQ_TARGET" == tke336 ]]; then
  compare_whole_executable_fp "$R336_PRISTINE_BLD" "$dry_bld" r336-whole \
    || refuse 'round-336 whole executable FP census differs from pristine'
  fp_named_signature "$R336_PRISTINE_BLD/obj/stprk3_stg.o" \
    "$scratch/r336-stage.pristine.fp" __stprk3_stg_MOD_stp_rk3_stg
  fp_named_signature "$dry_bld/obj/stprk3_stg.o" \
    "$scratch/r336-stage.observed.fp" __stprk3_stg_MOD_stp_rk3_stg
  r336_stage_count=$(awk '{n+=$1} END {print n+0}' "$scratch/r336-stage.observed.fp")
  [[ $r336_stage_count -eq 1175 ]] \
    || refuse "round-336 pristine stage FP census $r336_stage_count != 1175"
  r336_function_count=$(awk '{print $2,$3}' "$scratch/r336-whole.pristine.fp" | sort -u | wc -l)
  r336_instruction_count=$(awk '{n+=$1} END {print n+0}' "$scratch/r336-whole.pristine.fp")
  printf 'R336_WHOLE_EXECUTABLE_FP_PASS functions=%s instructions=%s stage=%s\n' \
    "$r336_function_count" "$r336_instruction_count" "$r336_stage_count"
  # Recompile the inherited round-326 stage observer for the plant.  The
  # source BLD object may have been refreshed independently and is therefore
  # not evidence that MY_SRC/stprk3_stg.F90 still carries the 1367-FP body.
  fpp "$SOURCE_STAGE" "$scratch/stage.inherited-with-eos.f90"
  sed '/eos_l2_dump_next/d' "$scratch/stage.inherited-with-eos.f90" \
    >"$scratch/stage.inherited.f90"
  compile_record_object "$scratch/stage.inherited.f90" "$scratch/stage.inherited.o"
  fp_named_signature "$scratch/stage.inherited.o" "$scratch/r336-stage.inherited.fp" \
    __stprk3_stg_MOD_stp_rk3_stg
  r336_inherited_stage_count=$(awk '{n+=$1} END {print n+0}' \
    "$scratch/r336-stage.inherited.fp")
  [[ $r336_inherited_stage_count -eq 1367 ]] \
    || refuse "round-336 inherited stage plant FP census $r336_inherited_stage_count != 1367"
  cp "$dry_bld/obj/stprk3_stg.o" "$scratch/stage.clean.o"
  rm -f "$dry_bld/obj/stprk3_stg.o"
  cp "$scratch/stage.inherited.o" "$dry_bld/obj/stprk3_stg.o"
  plant_must_refuse r336-inherited-stage-census compare_whole_executable_fp \
    "$R336_PRISTINE_BLD" "$dry_bld" r336-inherited-stage-plant
  rm -f "$dry_bld/obj/stprk3_stg.o"
  cp "$scratch/stage.clean.o" "$dry_bld/obj/stprk3_stg.o"
fi
rm -f "$dry_bld/ppsrc/nemo/icevar.f90"
{ sed '1s/$/ ! planted/' "$SOURCE_ROOT/BLD/ppsrc/nemo/icevar.f90"; } >"$dry_bld/ppsrc/nemo/icevar.f90"
plant_must_refuse same-program same_program "$dry_bld"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND328_%s_PREFLIGHT_READY %s\n' "${ACQ_TARGET^^}" "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  local -a reference_args=()
  [[ $(cat "$TARGET_RUN/producer_commit.txt") == "$EXPECTED_PRODUCER" ]] || refuse 'producer commit changed'
  if [[ "$ACQ_TARGET" == tke336 ]]; then
    compare_r336_twin "$TARGET_RUN" \
      || refuse 'round-336 output set or observer-free twin differs'
  else
    compare_deterministic "$SOURCE_RUN" "$TARGET_RUN" \
      || refuse 'write-only observer changed an inherited deterministic output'
  fi
  if [[ "$ACQ_TARGET" == tke336 ]]; then
    mapfile -d '' -t reference_args < <(r336_reference_args)
    for plant in header field-order truncation checksum stamp extent entry rn2 \
      langmuir solve-steps pre-sweep sweep-steps north-halo-avm-k \
      pdlr final signed-zero source-values time-level drag-range; do
      if "$PY" "$R336_TKE_GATE" --record-dir "$TARGET_RUN" \
        "${reference_args[@]}" \
        --expect-commit "$EXPECTED_PRODUCER" \
        --plant "$plant" >"$TARGET_RUN/round336_tke_${plant}_plant.log" 2>&1; then
        refuse "$plant round-336 TKE plant stayed green"
      fi
      grep 'REFUSE:' "$TARGET_RUN/round336_tke_${plant}_plant.log" >/dev/null \
        || refuse "$plant round-336 TKE plant missed its refusal"
    done
    "$PY" "$R336_TKE_GATE" --record-dir "$TARGET_RUN" \
      "${reference_args[@]}" \
      --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round336_tke_kt2_admission.json"
    ( cd "$TARGET_RUN" && sha256sum oracle_r336_tke_kt00000002_r*.bin \
        round336_tke_kt2_admission.json round336_tke_*_plant.log \
        >round336_tke_kt2_outputs.sha256 )
    [[ -z ${R336_TKE_STUB_FLOW_ROOT:-} ]] || r336_tke_stub_assert
  elif [[ "$ACQ_TARGET" == tke332 ]]; then
    for name in oracle_rhs_kt00000001.bin oracle_stage_kt00000001_s1.bin; do
      cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" \
        || refuse "round-332 observer changed passivity witness $name"
    done
    printf 'R332_PASSIVITY_WITNESSES_PASS pv_rhs=oracle_rhs_kt00000001.bin ua_va=oracle_stage_kt00000001_s1.bin\n'
    for rank in 0000 0001; do
      cmp -s "$ADMITTED_DYN_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        "$TARGET_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        || refuse "dyn_zdf record rank $rank differs from its admission"
    done
    "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round332_dynzdf_readmission.json"
    for plant in header field-order truncation stamp vor pdlr raw-header raw-time-level raw-truncation raw-checksum; do
      if "$PY" "$R332_GATE" --record-dir "$TARGET_RUN" \
        --zdf-reference-dir "$ADMITTED_ZDFPHY_RUN" --expect-commit "$EXPECTED_PRODUCER" \
        --plant "$plant" >"$TARGET_RUN/round332_source_${plant}_plant.log" 2>&1; then
        refuse "$plant round-332 source-order plant stayed green"
      fi
      grep 'REFUSE:' "$TARGET_RUN/round332_source_${plant}_plant.log" >/dev/null \
        || refuse "$plant round-332 source-order plant did not reach its refusal"
    done
    "$PY" "$R332_GATE" --record-dir "$TARGET_RUN" \
      --zdf-reference-dir "$ADMITTED_ZDFPHY_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round332_source_order_admission.json"
    ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin oracle_tke331_*.bin \
        oracle_r332_source_*.bin oracle_r332_raw_v_*.bin round332_dynzdf_readmission.json \
        round332_source_order_admission.json round332_source_*_plant.log \
        >round332_source_order_outputs.sha256 )
  elif [[ "$ACQ_TARGET" == tke331 ]]; then
    for rank in 0000 0001; do
      cmp -s "$ADMITTED_DYN_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        "$TARGET_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        || refuse "dyn_zdf record rank $rank differs from its admission"
    done
    "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round331_dynzdf_readmission.json"
    for plant in header field-order truncation stamp sh2 drag mxl boundary matrix solve final; do
      if "$PY" "$TKE_GATE" --record-dir "$TARGET_RUN" \
        --zdf-reference-dir "$ADMITTED_ZDFPHY_RUN" --expect-commit "$EXPECTED_PRODUCER" \
        --plant "$plant" >"$TARGET_RUN/round331_tke_${plant}_plant.log" 2>&1; then
        refuse "$plant round-331 TKE plant stayed green"
      fi
      grep 'REFUSE:' "$TARGET_RUN/round331_tke_${plant}_plant.log" >/dev/null \
        || refuse "$plant round-331 TKE plant did not reach its refusal"
    done
    "$PY" "$TKE_GATE" --record-dir "$TARGET_RUN" \
      --zdf-reference-dir "$ADMITTED_ZDFPHY_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round331_tke_admission.json"
    ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin oracle_tke331_*.bin \
        round331_dynzdf_readmission.json round331_tke_admission.json \
        round331_tke_*_plant.log >round331_tke_outputs.sha256 )
  elif [[ "$ACQ_TARGET" == zdfphy ]]; then
    for rank in 0000 0001; do
      cmp -s "$ADMITTED_DYN_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        "$TARGET_RUN/oracle_dynzdf_kt00000001_s3_r${rank}.bin" \
        || refuse "dyn_zdf record rank $rank differs from its admission"
    done
    "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round328_dynzdf_readmission.json"
    for plant in header field-order truncation stamp boundary1 boundary2 boundary3; do
      if "$PY" "$ZP_GATE" --record-dir "$TARGET_RUN" --expect-commit "$EXPECTED_PRODUCER" \
        --plant "$plant" >"$TARGET_RUN/round328_zdfphy_${plant}_plant.log" 2>&1; then
        refuse "$plant zdf_phy plant stayed green"
      fi
      grep 'REFUSE:' "$TARGET_RUN/round328_zdfphy_${plant}_plant.log" >/dev/null \
        || refuse "$plant zdf_phy plant did not reach its refusal"
    done
    "$PY" "$ZP_GATE" --record-dir "$TARGET_RUN" --expect-commit "$EXPECTED_PRODUCER" \
      --output "$TARGET_RUN/round328_zdfphy_admission.json"
    ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin oracle_zdfphy_*.bin \
        round328_dynzdf_readmission.json round328_zdfphy_admission.json \
        round328_zdfphy_*_plant.log >round328_zdfphy_outputs.sha256 )
  else
    for plant in header field-order truncation stamp avm bottom top solve endpoint; do
      if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" --plant "$plant" \
        >"$TARGET_RUN/round328_${plant}_plant.log" 2>&1; then
        refuse "$plant plant stayed green"
      fi
      grep 'REFUSE:' "$TARGET_RUN/round328_${plant}_plant.log" >/dev/null \
        || refuse "$plant plant did not reach its refusal"
    done
    "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" \
      --output "$TARGET_RUN/round328_dynzdf_admission.json"
    ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin round328_dynzdf_admission.json \
        round328_*_plant.log >round328_outputs.sha256 )
  fi
  printf 'ORCA2_ROUND328_%s_ACQUISITION_PASS %s\n' "${ACQ_TARGET^^}" "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  if [[ "$TARGET_RUN" == "$DEFAULT_TARGET_RUN" ]]; then
    [[ -x "$TARGET_RUN/nemo" ]] || refuse 'existing target lacks its binary'
  else
    [[ -x "$TARGET_RUN/nemo" ]] || refuse 'retained run lacks its binary'
    [[ ! -e "$DEFAULT_TARGET_RUN" ]] || refuse 'canonical admission target already exists'
  fi
  admit
  if [[ "$TARGET_RUN" != "$DEFAULT_TARGET_RUN" ]]; then
    mv -- "$TARGET_RUN" "$DEFAULT_TARGET_RUN"
    printf 'R328_RETAINED_OUTPUTS_ADMITTED %s\n' "$DEFAULT_TARGET_RUN"
  fi
  exit 0
fi

manifest=$scratch/manifest; mkdir "$manifest"
r336_twin_binary=$scratch/nemo.observer-free
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$STAGE_PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" \
  "$EXCERPT" "$GATE" "$TEST" "$PREREG" "$SOURCE_DYNZDF" "$SOURCE_STAGE" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" >"$manifest/toolchain.sha256"
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  sha256sum "$ZP_PATCH" "$ZP_DDM_PATCH" "$ZP_IWM_PATCH" "$ZP_WRITER" \
    "$ZP_FIXTURE" "$ZP_GATE" "$ZP_TEST" \
    "$SOURCE_ZDFPHY" "$SOURCE_ZDFDDM" "$SOURCE_ZDFIWM" \
    >>"$manifest/toolchain.sha256"
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  sha256sum "$TKE_PHY_PATCH" "$TKE_PATCH" "$TKE_WRITER" "$TKE_FIXTURE" \
    "$TKE_GATE" "$TKE_TEST" "$TKE_PREREG" "$SOURCE_ZDFPHY" \
    "$SOURCE_ZDFSH2" "$SOURCE_ZDFDRG" "$SOURCE_ZDFMXL" "$SOURCE_ZDFTKE" \
    >>"$manifest/toolchain.sha256"
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  sha256sum "$R332_DYNVOR_PATCH" "$R332_TKE_PATCH" "$R332_STAGE_PATCH" \
    "$R332_WRITER" "$R332_FIXTURE" \
    "$R332_VOR_INTERFACE_FIXTURE" "$R332_GATE" "$R332_TEST" "$SOURCE_DYNVOR" \
    >>"$manifest/toolchain.sha256"
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  sha256sum "$R336_TKE_PATCH" "$R336_TKE_WRITER" "$R336_TKE_FIXTURE_STUBS" \
    "$R336_TKE_FIXTURE" "$R336_TKE_GATE" "$R336_TKE_TEST" "$SOURCE_ZDFTKE" \
    "$SOURCE_ZDFTKE_PRISTINE" \
    >>"$manifest/toolchain.sha256"
  {
    printf '%s  %s\n' "$R336_TRACER_R0_SHA" round333/tracer/oracle_rktracer_stage3_kt00000002_r0000.bin
    printf '%s  %s\n' "$R336_TRACER_R1_SHA" round333/tracer/oracle_rktracer_stage3_kt00000002_r0001.bin
    printf '%s  %s\n' "$R336_LDF_R0_SHA" round333/ldf/oracle_r333_traldf_kt00000002_r0000.bin
    printf '%s  %s\n' "$R336_LDF_R1_SHA" round333/ldf/oracle_r333_traldf_kt00000002_r0001.bin
    printf '%s  %s\n' "$R336_ICE_R0_SHA" round326/ice/oracle_tke_hmi_kt00000002_r0000.bin
    printf '%s  %s\n' "$R336_ICE_R1_SHA" round326/ice/oracle_tke_hmi_kt00000002_r0001.bin
  } >"$manifest/replay_inputs.sha256"
fi

# makenemo rewrites shared work_cfgs.txt; only the target's line may change.
cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.before"
cd "$NEMO_ROOT"
created_target=1
if [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
  r336_tke_stub_prepare_target "$TARGET_ROOT"
else
  ./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
fi
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
if [[ "$ACQ_TARGET" != tke336 ]]; then
  while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
    < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
fi
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
if [[ "$ACQ_TARGET" == tke336 && -z ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
  ./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
  [[ -x $BINARY ]] || refuse 'pristine twin build produced no executable'
  cp "$BINARY" "$r336_twin_binary"
fi
if [[ "$ACQ_TARGET" != tke336 ]]; then
  cp "$SOURCE_DYNZDF" "$TARGET_ROOT/MY_SRC/dynzdf.F90"
  cp "$WRITER" "$TARGET_ROOT/MY_SRC/dynzdf_round328_writer.F90"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$STAGE_PATCH"
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  cp "$SOURCE_ZDFPHY" "$TARGET_ROOT/MY_SRC/zdfphy.F90"
  cp "$SOURCE_ZDFDDM" "$TARGET_ROOT/MY_SRC/zdfddm.F90"
  cp "$SOURCE_ZDFIWM" "$TARGET_ROOT/MY_SRC/zdfiwm.F90"
  cp "$ZP_WRITER" "$TARGET_ROOT/MY_SRC/zdfphy_round328_writer.F90"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$ZP_PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$ZP_DDM_PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$ZP_IWM_PATCH"
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  cp "$SOURCE_ZDFPHY" "$TARGET_ROOT/MY_SRC/zdfphy.F90"
  cp "$SOURCE_ZDFTKE" "$TARGET_ROOT/MY_SRC/zdftke.F90"
  cp "$TKE_WRITER" "$TARGET_ROOT/MY_SRC/tke_round331_writer.F90"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$TKE_PHY_PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$TKE_PATCH"
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  cp "$SOURCE_DYNVOR" "$TARGET_ROOT/MY_SRC/dynvor.F90"
  cp "$R332_WRITER" "$TARGET_ROOT/MY_SRC/round332_writer.F90"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$R332_DYNVOR_PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$R332_TKE_PATCH"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$R332_STAGE_PATCH"
  check_r332_tke_landing "$TARGET_ROOT/MY_SRC/zdftke.F90" \
    || refuse 'built round-332 TKE observer did not land at the intended statement'
  check_r332_raw_landing "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" \
    || refuse 'built round-332 raw V observer did not land at the intended statement'
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  cp "$SOURCE_ZDFTKE_PRISTINE" "$TARGET_ROOT/MY_SRC/zdftke.F90"
  cp "$R336_TKE_WRITER" "$TARGET_ROOT/MY_SRC/tke_round336_writer.F90"
  patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$R336_TKE_PATCH"
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
if [[ -n ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then
  r336_tke_stub_finish_target "$TARGET_ROOT" "$dry_bld" "$r336_twin_binary"
else
  ./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
fi
[[ -x "$BINARY" ]] || refuse 'build produced no executable'
if [[ "$ACQ_TARGET" != tke336 ]]; then
  grep 'NEMO_L4_R328ZDF' "$TARGET_ROOT/BLD/ppsrc/nemo/dynzdf_round328_writer.f90" >/dev/null \
    || refuse 'compiled writer is absent'
fi
if [[ "$ACQ_TARGET" == zdfphy ]]; then
  grep 'NEMO_L4_R328ZP' "$TARGET_ROOT/BLD/ppsrc/nemo/zdfphy_round328_writer.f90" >/dev/null \
    || refuse 'compiled zdf_phy writer is absent'
fi
if [[ "$ACQ_TARGET" == tke331 || "$ACQ_TARGET" == tke332 ]]; then
  grep 'NEMO_L4_R331TK' "$TARGET_ROOT/BLD/ppsrc/nemo/tke_round331_writer.f90" >/dev/null \
    || refuse 'compiled round-331 TKE writer is absent'
fi
if [[ "$ACQ_TARGET" == tke332 ]]; then
  grep 'NEMO_L4_R332SO' "$TARGET_ROOT/BLD/ppsrc/nemo/round332_writer.f90" >/dev/null \
    || refuse 'compiled round-332 source-order writer is absent'
  grep 'NEMO_L4_R332RV' "$TARGET_ROOT/BLD/ppsrc/nemo/round332_writer.f90" >/dev/null \
    || refuse 'compiled round-332 raw V writer is absent'
fi
if [[ "$ACQ_TARGET" == tke336 ]]; then
  grep 'NEMO_L4_R336TKE' "$TARGET_ROOT/BLD/ppsrc/nemo/tke_round336_writer.f90" >/dev/null \
    || refuse 'compiled round-336 TKE writer is absent'
fi
diff <(grep -v "^$TARGET_CFG " "$scratch/work_cfgs.before") \
     <(grep -v "^$TARGET_CFG " "$NEMO_ROOT/cfgs/work_cfgs.txt") >/dev/null \
  || refuse 'work_cfgs.txt changed beyond the target line'
same_program "$TARGET_ROOT/BLD" "$manifest/same_program.json"
if [[ "$ACQ_TARGET" == tke336 ]]; then
  "$PY" "$R336_TKE_GATE" \
    --build-output-contract "$manifest/same_program.json" \
    --compiled-writer "$TARGET_ROOT/BLD/ppsrc/nemo/tke_round336_writer.f90" \
    --output "$manifest/round336_output_contract.json" >/dev/null
  compare_whole_executable_fp "$R336_PRISTINE_BLD" "$TARGET_ROOT/BLD" r336-built-whole \
    || refuse 'built round-336 whole executable FP census differs from pristine'
  printf 'R336_BUILT_WHOLE_EXECUTABLE_FP_PASS reference=%s target=%s\n' \
    "$R336_PRISTINE_BLD" "$TARGET_ROOT/BLD"
fi

mkdir -p "$EVIDENCE"
created_run=1
mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done \
  <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done \
  <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum nemo >nemo.sha256 )
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )
if [[ "$ACQ_TARGET" == tke336 ]]; then
  twin_run=$TARGET_RUN/observer_free_twin
  mkdir "$twin_run"
  while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$twin_run/$name"; done \
    <"$SOURCE_RUN/deck_files.sha256"
  while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$twin_run/$name"; done \
    <"$SOURCE_RUN/input_files.sha256"
  cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$twin_run/"
  cp "$r336_twin_binary" "$twin_run/nemo"
  ( cd "$twin_run" && sha256sum nemo >nemo.sha256 && \
      sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )
  ( cd "$TARGET_RUN" && sha256sum observer_free_twin/nemo >observer_free_nemo.sha256 )
  run_nemo_case "$twin_run" observer-free-twin
fi
run_nemo_case "$TARGET_RUN" observed
run_completed=1
check_echo "$TARGET_RUN/ocean.output" || refuse 'run selectors changed'
[[ "$ACQ_TARGET" != tke336 ]] || check_echo "$TARGET_RUN/observer_free_twin/ocean.output" \
  || refuse 'observer-free twin selectors changed'
admit
