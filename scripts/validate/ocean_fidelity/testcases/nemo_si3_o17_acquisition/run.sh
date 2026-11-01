#!/usr/bin/env bash
# O17 acquisition tool. Default is read-only; only the operator uses --run.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly py=${PYTHON_BIN:-python3}
readonly default_root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step
readonly timeout_seconds=${O17_TIMEOUT_SECONDS:-21600}
readonly min_ram_gb=${O17_MIN_FREE_RAM_GB:-180}
readonly min_disk_kb=${O17_MIN_FREE_DISK_KB:-300000000}
export PYTHONPATH="$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/ice:$repo/packages/atmosphere:$repo/packages/land:$repo/packages/coupler:$repo/packages/driver"

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
resource_gates() {
  local target=$1 ram disk probe=$1
  ram=$(free -g | awk '/^Mem:/ {print $7}')
  while [[ ! -e "$probe" ]]; do probe=$(dirname "$probe"); done
  disk=$(df -Pk "$probe" | awk 'NR==2 {print $4}')
  [[ -n "$ram" && "$ram" -ge "$min_ram_gb" ]] || refuse "available RAM ${ram:-unknown} GB below $min_ram_gb GB"
  [[ -n "$disk" && "$disk" -ge "$min_disk_kb" ]] || refuse "free disk ${disk:-unknown} KiB below $min_disk_kb KiB"
}

preflight() {
  local scratch
  scratch=$(mktemp -d /tmp/si3_o17_component_preflight.XXXXXX)
  "$py" "$here/preflight.py" --repo "$repo"
  if "$py" "$here/preflight.py" --repo "$repo" --plant swap-rheology-advection >/dev/null 2>&1; then
    refuse 'swap-rheology-advection plant survived'
  fi
  if "$py" "$here/preflight.py" --repo "$repo" --plant drop-redistribution >/dev/null 2>&1; then
    refuse 'drop-redistribution plant survived'
  fi
  # Import the admitted shell acquisition controls by executing their
  # read-only modes. Each uses only mktemp scratch and its stub drivers.
  EVIDENCE_ROOT="$scratch/o4_root" EVIDENCE="$scratch/o4" \
    REFUSED_DIR="$scratch/o4_refused" \
    REF_CFG="O17_PREFLIGHT_${BASHPID}_O4_REF" REC_CFG="O17_PREFLIGHT_${BASHPID}_O4_REC" \
    "$here/../nemo_si3_o4_orca2_prather_acquisition/run.sh" --preflight-only
  EVIDENCE_ROOT="$scratch/o5_root" EVIDENCE="$scratch/o5" \
    REFUSED_DIR="$scratch/o5_refused" \
    REF_CFG="O17_PREFLIGHT_${BASHPID}_O5_REF" REC_CFG="O17_PREFLIGHT_${BASHPID}_O5_REC" \
    "$here/../nemo_si3_o5_orca2_sas_acquisition/run.sh" --preflight-only
  EVIDENCE_ROOT="$scratch/o7_root" EVIDENCE="$scratch/o7" \
    REFUSED_DIR="$scratch/o7_refused" \
    REF_CFG="O17_PREFLIGHT_${BASHPID}_O7_REF" REC_CFG="O17_PREFLIGHT_${BASHPID}_O7_REC" \
    "$here/../nemo_si3_o7_orca2_sas_acquisition/run.sh" --preflight-only
  EVIDENCE_ROOT="$scratch/o8_o16_root" EVIDENCE="$scratch/o8_o16" \
    REFUSED_DIR="$scratch/o8_o16_refused" \
    REF_CFG="O17_PREFLIGHT_${BASHPID}_O8O16_REF" \
    REC_CFG="O17_PREFLIGHT_${BASHPID}_O8O16_REC" \
    "$here/../nemo_si3_o8_o16_master_acquisition/run.sh" --preflight-only
  rm -rf -- "$scratch"
  printf 'PASS_O17_COMPONENT_PREFLIGHTS_AND_PLANTS\n'
}

mode=${1:---preflight-only}
evidence=${2:-$default_root}
case "$mode" in
  --preflight-only) preflight ;;
  --run)
    [[ ${O17_OPERATOR_ACK:-} == RUN_UNMODIFIED_ORCA2_SAS_ICE_100X10800 ]] || \
      refuse 'set O17_OPERATOR_ACK=RUN_UNMODIFIED_ORCA2_SAS_ICE_100X10800'
    [[ "$evidence" =~ /si3_rounds/round[0-9]+/ ]] || refuse 'evidence must be under si3_rounds/round<k>/'
    resource_gates "$evidence"
    preflight
    mkdir -p "$evidence/manifest"
    # Checklist 9: immutable intent/route manifest exists before any NEMO run.
    "$py" - "$evidence/manifest/run_manifest.json" "$repo" "$timeout_seconds" "$min_disk_kb" <<'PY'
import hashlib,json,pathlib,subprocess,sys,time
out,repo,timeout,disk=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
paths=[repo/'scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh',
       repo/'scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/preflight.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'created_utc_epoch':int(time.time()),'producer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'deck':'unmodified ORCA2_SAS_ICE','steps':100,'dt_seconds':10800,
 'records_each_step':['complete_step_entry','raw_o7_meteorology','o7_products','named_call_boundaries','exchange_outputs','restart'],
 'labels':['INDEPENDENT','GIVEN-NEMO-ENTRY'],'wall_clock_gate_seconds':timeout,'disk_gate_kib':disk,
 'retention':'failed runs move beneath refused; admitted raw records remain immutable',
 'admission_routes':{'O4':'imported O4 gate','O5':'imported O5 gate','O7':'imported O7 gate','O8-O16':'imported master gate','O17':'composition scorer'},
 'tool_sha256':{str(p.relative_to(repo)):sha(p) for p in paths}}
out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
    mkdir -p "$evidence/components"
    started=$SECONDS
    EVIDENCE="$evidence/components/o4" \
      "$here/../nemo_si3_o4_orca2_prather_acquisition/run.sh" --run "$evidence/components/o4"
    (( SECONDS - started <= timeout_seconds )) || refuse 'global wall-clock gate exceeded after O4'
    RUN_TIMEOUT_SEC="$timeout_seconds" O5_ACQ_STEPS=100 O5_ACQ_STOCK=1 EVIDENCE="$evidence/components/o5" \
      "$here/../nemo_si3_o5_orca2_sas_acquisition/run.sh" --run "$evidence/components/o5"
    (( SECONDS - started <= timeout_seconds )) || refuse 'global wall-clock gate exceeded after O5'
    RUN_TIMEOUT_SEC="$timeout_seconds" O7_ACQ_STEPS=100 O7_ACQ_STOCK=1 EVIDENCE="$evidence/components/o7" \
      "$here/../nemo_si3_o7_orca2_sas_acquisition/run.sh" --run "$evidence/components/o7"
    (( SECONDS - started <= timeout_seconds )) || refuse 'global wall-clock gate exceeded after O7'
    RUN_TIMEOUT_SEC="$timeout_seconds" EVIDENCE="$evidence/components/o8_o16" \
      "$here/../nemo_si3_o8_o16_master_acquisition/run.sh" --run "$evidence/components/o8_o16"
    (( SECONDS - started <= timeout_seconds )) || refuse 'global wall-clock gate exceeded after O8--O16'
    "$py" "$here/../nemo_si3_o17_composition_scorer.py" \
      --evidence "$evidence" --labels INDEPENDENT GIVEN-NEMO-ENTRY \
      > "$evidence/manifest/score.json"
    touch "$evidence/RUN_COMPLETE"
    printf 'ACQUISITION_DONE: %s\n' "$evidence"
    ;;
  *) refuse "usage: $0 [--preflight-only|--run] [si3_rounds/round<k>/path]" ;;
esac
