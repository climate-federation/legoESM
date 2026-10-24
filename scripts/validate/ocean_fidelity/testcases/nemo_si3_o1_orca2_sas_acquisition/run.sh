#!/usr/bin/env bash
# SI3 ladder Rung O1 ACQUISITION (DECISION 159) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  bash run.sh [--preflight-only]
#   acquire (operator):   bash run.sh --run
#   re-admit existing:    bash run.sh --admit-existing
#
# The agent that wrote this tool ran only --preflight-only; it never invoked
# makenemo or mpirun.
set -Eeuo pipefail

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG_SOURCE=ORCA2_SAS_ICE
readonly TARGET_REF_CFG=${TARGET_REF_CFG:-ORCA2_SAS_ICE_O1_REF}
readonly TARGET_REC_CFG=${TARGET_REC_CFG:-ORCA2_SAS_ICE_O1_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4}
readonly EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o1_orca2_sas_ice}
REFUSED_DIR=${REFUSED_DIR:-$EVIDENCE_ROOT/refused}

readonly ORCA2_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz
readonly ORCA2_BUNDLE_SHA256=5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4
readonly SAS_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz
readonly SAS_BUNDLE_SHA256=ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf

readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-8388608}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-1800}

readonly PY=/home/dbalwada/legoESM/.venv/bin/python3
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

readonly WRITER_F90=$here/o1_si3_state_writer.F90
readonly PRA_PATCH=$here/icedyn_adv_pra_o1_public.patch
readonly STP_PATCH=$here/icestp_o1_writer.patch
readonly STUBS_F90=$here/o1_si3_fixture_stubs.F90
readonly FIXTURE_F90=$here/o1_si3_layout_fixture.F90
readonly NML_CFG=$here/namelist_cfg
readonly NML_ICE_CFG=$here/namelist_ice_cfg
readonly LOG_EXCERPT=$here/ocean.output.excerpt
readonly GATE_PY=$here/../nemo_si3_o1_orca2_sas_gate.py

readonly SHIPPED_PRA=$NEMO_ROOT/src/ICE/icedyn_adv_pra.F90
readonly SHIPPED_STP=$NEMO_ROOT/src/ICE/icestp.F90
readonly SHIPPED_NML_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ref
readonly SHIPPED_NML_ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly SHIPPED_EXPREF_NML=$NEMO_ROOT/cfgs/$REF_CFG_SOURCE/EXPREF/namelist_cfg
readonly SHIPPED_EXPREF_ICE=$NEMO_ROOT/cfgs/$REF_CFG_SOURCE/EXPREF/namelist_ice_cfg
readonly NEMO_MOD_DIR=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj
readonly NEMO_INC_DIR=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/inc
readonly NEMO_WORK_DIR=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/WORK

mode=${1:---preflight-only}
preflight_json_out=${2:-}
case "$mode" in
  --preflight-only|--run|--admit-existing) ;;
  *)
    printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing] [json_out]\n' "$0" >&2
    exit 64
    ;;
esac

ACTIVE_RUN_DIR=""
ACTIVE_SCRATCH_DIR=""

cleanup_on_exit() {
  local rc=$?
  if [[ -n "$ACTIVE_SCRATCH_DIR" && -d "$ACTIVE_SCRATCH_DIR" ]]; then
    rm -rf -- "$ACTIVE_SCRATCH_DIR"
  fi
  # Never delete or move a completed run (marked by RUN_COMPLETE).
  # Move any incomplete/refused run directory into REFUSED_DIR.
  if [[ "$rc" -ne 0 && -n "$ACTIVE_RUN_DIR" && -d "$ACTIVE_RUN_DIR" ]]; then
    if [[ ! -f "$ACTIVE_RUN_DIR/RUN_COMPLETE" ]]; then
      mkdir -p "$REFUSED_DIR"
      local ts
      ts=$(date -u +%Y%m%dT%H%M%SZ)
      local dest="$REFUSED_DIR/$(basename "$ACTIVE_RUN_DIR")_${ts}_rc${rc}"
      mv "$ACTIVE_RUN_DIR" "$dest"
      printf 'REFUSE_CLEANUP: moved incomplete/refused run %s -> %s\n' "$ACTIVE_RUN_DIR" "$dest" >&2
    fi
  fi
  exit "$rc"
}
trap cleanup_on_exit EXIT

check_free_ram_gb() {
  local floor_gb=$1
  local avail_gb
  avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [[ -z "$avail_gb" || "$avail_gb" -lt "$floor_gb" ]]; then
    printf 'REFUSE: available RAM %s GB is below floor %s GB\n' "${avail_gb:-unknown}" "$floor_gb" >&2
    return 1
  fi
  return 0
}

check_free_disk_kb() {
  local target_path=$1
  local floor_kb=$2
  local probe=$target_path
  while [[ ! -e "$probe" ]]; do
    probe=$(dirname "$probe")
  done
  local avail_kb
  avail_kb=$(df -Pk "$probe" | awk 'NR==2 {print $4}')
  if [[ -z "$avail_kb" || "$avail_kb" -lt "$floor_kb" ]]; then
    printf 'REFUSE: available disk %s KB on %s (for %s) is below floor %s KB\n' \
      "${avail_kb:-unknown}" "$probe" "$target_path" "$floor_kb" >&2
    return 1
  fi
  return 0
}

check_ref_cfg_in_registry() {
  local cfg_name=$1
  local ref_file=$2
  grep -E "^[[:space:]]*${cfg_name}([[:space:]]|$)" "$ref_file" >/dev/null
}

check_target_absent() {
  local path=$1
  if [[ -e "$path" ]]; then
    printf 'REFUSE: target path already exists: %s\n' "$path" >&2
    return 1
  fi
  return 0
}

check_ascii_only() {
  local file=$1
  if LC_ALL=C grep -n '[^ -~	]' "$file" >/dev/null; then
    printf 'REFUSE: non-ASCII characters found in %s\n' "$file" >&2
    return 1
  fi
  return 0
}

check_intrinsic_use() {
  local file=$1
  if grep -E '^[[:space:]]*USE[[:space:]]+iso_fortran_env' "$file" >/dev/null; then
    printf 'REFUSE: %s uses bare USE iso_fortran_env without USE, INTRINSIC\n' "$file" >&2
    return 1
  fi
  grep -E '^[[:space:]]*USE,[[:space:]]*INTRINSIC[[:space:]]*::[[:space:]]*iso_fortran_env' "$file" >/dev/null
}

verify_namelists_and_stems() {
  local nml_cfg=$1
  local nml_ice=$2
  local toc_file=$3
  local plant=${4:-none}
  "$PY" - "$nml_cfg" "$nml_ice" "$SHIPPED_NML_REF" "$SHIPPED_NML_ICE_REF" \
    "$SHIPPED_EXPREF_NML" "$SHIPPED_EXPREF_ICE" "$toc_file" "$plant" <<'PY'
import pathlib
import re
import sys

(
    nml_cfg_p,
    nml_ice_p,
    ref_oce_p,
    ref_ice_p,
    exp_oce_p,
    exp_ice_p,
    toc_p,
    plant,
) = sys.argv[1:]

nml_cfg = pathlib.Path(nml_cfg_p).read_text()
nml_ice = pathlib.Path(nml_ice_p).read_text()
ref_oce = pathlib.Path(ref_oce_p).read_text()
ref_ice = pathlib.Path(ref_ice_p).read_text()
exp_oce = pathlib.Path(exp_oce_p).read_text()
exp_ice = pathlib.Path(exp_ice_p).read_text()
toc_lines = pathlib.Path(toc_p).read_text().splitlines()
archive_basenames = {pathlib.PurePosixPath(line.strip()).name for line in toc_lines if line.strip()}

if plant == "unknown-group":
    nml_cfg = nml_cfg + "\n&namrun_typo\n/\n"
elif plant == "missing-stem":
    nml_cfg = nml_cfg + "\n   sn_fake = 'missing_forcing_stem', 6., 'FAKE', .false., .true., 'yearly', '', '', ''\n"
elif plant == "bad-fsbc-multiple":
    nml_cfg = re.sub(r"(nn_fsbc\s*=\s*)\d+", r"\g<1>2", nml_cfg)
    nml_cfg = re.sub(r"(nn_itend\s*=\s*)\d+", r"\g<1>3", nml_cfg)
elif plant == "unapproved-deviation":
    nml_ice = re.sub(r"(&nampar)", "&nampar\n   rn_amax_n = 0.5", nml_ice, count=1)

def groups(text: str) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"^\s*&([A-Za-z0-9_]+)", text, flags=re.M)}

oce_groups = groups(nml_cfg)
ice_groups = groups(nml_ice)
ref_oce_groups = groups(ref_oce)
ref_ice_groups = groups(ref_ice)

unknown_oce = sorted(oce_groups - ref_oce_groups)
unknown_ice = sorted(ice_groups - ref_ice_groups)
if unknown_oce or unknown_ice:
    raise SystemExit(f"STATUS PLANT-FIRED: unknown namelist groups oce={unknown_oce} ice={unknown_ice}")

def parse_assignments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("!") or s.startswith("&") or s.startswith("/"):
            continue
        code = s.split("!")[0].strip()
        if "=" in code:
            k, v = code.split("=", 1)
            out[k.strip()] = v.strip()
    return out

cfg_oce_vars = parse_assignments(nml_cfg)
exp_oce_vars = parse_assignments(exp_oce)
cfg_ice_vars = parse_assignments(nml_ice)
exp_ice_vars = parse_assignments(exp_ice)

nn_fsbc = int(cfg_oce_vars.get("nn_fsbc", "1"))
nn_itend = int(cfg_oce_vars.get("nn_itend", "0"))
nn_stock = int(cfg_oce_vars.get("nn_stock", "0"))
if nn_fsbc <= 0 or nn_itend <= 0 or nn_stock <= 0 or (nn_itend % nn_fsbc != 0) or (nn_stock % nn_fsbc != 0):
    raise SystemExit(
        f"STATUS PLANT-FIRED: nn_itend={nn_itend} or nn_stock={nn_stock} not a positive multiple of nn_fsbc={nn_fsbc}"
    )

# Check O1 deck deviation vs shipped ORCA2_SAS_ICE/EXPREF
oce_diff_keys = {
    k for k in (set(cfg_oce_vars) | set(exp_oce_vars))
    if cfg_oce_vars.get(k) != exp_oce_vars.get(k)
}
ice_diff_keys = {
    k for k in (set(cfg_ice_vars) | set(exp_ice_vars))
    if cfg_ice_vars.get(k) != exp_ice_vars.get(k)
}
allowed_oce_diff = {"nn_itend", "nn_stock", "nn_write"}
allowed_ice_diff = {"ln_icedyn", "ln_icethd"}
if oce_diff_keys != allowed_oce_diff or ice_diff_keys != allowed_ice_diff:
    raise SystemExit(
        f"STATUS PLANT-FIRED: deck deviation mismatch oce_diff={sorted(oce_diff_keys)} ice_diff={sorted(ice_diff_keys)}"
    )
if cfg_ice_vars.get("ln_icedyn", "").lower() != ".false." or cfg_ice_vars.get("ln_icethd", "").lower() != ".false.":
    raise SystemExit("STATUS PLANT-FIRED: O1 requires ln_icedyn=.false. and ln_icethd=.false.")

# Verify cn_domcfg and every sn_* file stem + weights file exists in the tarballs
domcfg_stem = cfg_oce_vars["cn_domcfg"].strip("\"'")
if f"{domcfg_stem}.nc" not in archive_basenames:
    raise SystemExit(f"STATUS PLANT-FIRED: cn_domcfg {domcfg_stem}.nc missing from bundles")

for k, v in list(cfg_oce_vars.items()) + list(cfg_ice_vars.items()):
    if k.startswith("sn_"):
        parts = [p.strip().strip("\"'") for p in v.split(",")]
        stem = parts[0]
        if stem and stem != "NOT USED":
            if f"{stem}.nc" not in archive_basenames:
                raise SystemExit(f"STATUS PLANT-FIRED: {k} stem {stem}.nc missing from bundles")
        if len(parts) >= 7:
            weights = parts[6]
            if weights and f"{weights}.nc" not in archive_basenames:
                raise SystemExit(f"STATUS PLANT-FIRED: {k} weights {weights}.nc missing from bundles")
PY
}

run_preflight() {
  ACTIVE_SCRATCH_DIR=$(mktemp -d /tmp/o1_si3_preflight.XXXXXXXX)
  local scratch=$ACTIVE_SCRATCH_DIR

  # Preconditions: required files and NEMO reference tree
  for f in "$WRITER_F90" "$PRA_PATCH" "$STP_PATCH" "$STUBS_F90" "$FIXTURE_F90" \
           "$NML_CFG" "$NML_ICE_CFG" "$LOG_EXCERPT" "$GATE_PY" \
           "$SHIPPED_PRA" "$SHIPPED_STP" "$SHIPPED_NML_REF" "$SHIPPED_NML_ICE_REF" \
           "$SHIPPED_EXPREF_NML" "$SHIPPED_EXPREF_ICE" \
           "$ORCA2_BUNDLE" "$SAS_BUNDLE" "$NEMO_ROOT/makenemo" "$NEMO_ROOT/cfgs/ref_cfgs.txt"; do
    [[ -f "$f" ]] || { printf 'REFUSE: missing required input %s\n' "$f" >&2; exit 65; }
  done

  # Verify input bundle SHA-256 digests and extract table of contents
  local actual_orca2_sha actual_sas_sha
  actual_orca2_sha=$(sha256sum "$ORCA2_BUNDLE" | awk '{print $1}')
  actual_sas_sha=$(sha256sum "$SAS_BUNDLE" | awk '{print $1}')
  [[ "$actual_orca2_sha" == "$ORCA2_BUNDLE_SHA256" ]] || {
    printf 'REFUSE: ORCA2 bundle SHA-256 %s != %s\n' "$actual_orca2_sha" "$ORCA2_BUNDLE_SHA256" >&2
    exit 65
  }
  [[ "$actual_sas_sha" == "$SAS_BUNDLE_SHA256" ]] || {
    printf 'REFUSE: SAS bundle SHA-256 %s != %s\n' "$actual_sas_sha" "$SAS_BUNDLE_SHA256" >&2
    exit 65
  }
  tar -ztf "$ORCA2_BUNDLE" > "$scratch/bundles.toc"
  tar -ztf "$SAS_BUNDLE" >> "$scratch/bundles.toc"

  # ---- Checklist Item 1: gfortran syntax proof of patched NEMO files + plants ----
  if grep -q '^-[^-]' "$PRA_PATCH" "$STP_PATCH"; then
    printf 'REFUSE: source patches must be strictly additive\n' >&2
    exit 66
  fi
  for f90 in "$WRITER_F90" "$STUBS_F90" "$FIXTURE_F90"; do
    check_intrinsic_use "$f90"
  done
  # Plant for bare USE iso_fortran_env without INTRINSIC
  sed 's/USE, INTRINSIC :: iso_fortran_env/USE iso_fortran_env/' "$WRITER_F90" > "$scratch/bad_intrinsic.F90"
  if check_intrinsic_use "$scratch/bad_intrinsic.F90" 2>/dev/null; then
    printf 'REFUSE: bare USE iso_fortran_env plant did not fire\n' >&2
    exit 72
  fi

  cp "$SHIPPED_PRA" "$scratch/icedyn_adv_pra.F90"
  cp "$SHIPPED_STP" "$scratch/icestp.F90"
  patch -s --fuzz=0 "$scratch/icedyn_adv_pra.F90" < "$PRA_PATCH"
  patch -s --fuzz=0 "$scratch/icestp.F90" < "$STP_PATCH"

  preprocess_nemo() {
    cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_linssh -Dkey_vco_1d3d -Dkey_RK3 \
      -P -traditional -I "$NEMO_WORK_DIR" -I "$NEMO_INC_DIR" "$1" -o "$2"
  }
  preprocess_nemo "$scratch/icedyn_adv_pra.F90" "$scratch/icedyn_adv_pra.f90"
  preprocess_nemo "$WRITER_F90" "$scratch/o1_si3_state_writer.f90"
  preprocess_nemo "$scratch/icestp.F90" "$scratch/icestp.f90"

  "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$NEMO_MOD_DIR" -I "$NEMO_INC_DIR" -J "$scratch" "$scratch/icedyn_adv_pra.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$NEMO_MOD_DIR" -I "$NEMO_INC_DIR" -J "$scratch" "$scratch/o1_si3_state_writer.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$NEMO_MOD_DIR" -I "$NEMO_INC_DIR" -J "$scratch" "$scratch/icestp.f90"

  # Plant 1a: insertion immediately after a continuation line ending in '&'
  awk '{print} /IF\( ln_icedyn .AND. .NOT.ln_c1d \)[[:space:]]*&/ {print "         CALL o1_write_si3_state_snapshot( '\''step'\'', kt, lrst_ice )"}' \
    "$scratch/icestp.F90" > "$scratch/icestp_amp_plant.F90"
  preprocess_nemo "$scratch/icestp_amp_plant.F90" "$scratch/icestp_amp_plant.f90"
  if "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$NEMO_MOD_DIR" -I "$NEMO_INC_DIR" -J "$scratch" "$scratch/icestp_amp_plant.f90" 2>/dev/null; then
    printf 'REFUSE: Plant 1a (insertion after continuation &) did not fail gfortran -fsyntax-only\n' >&2
    exit 72
  fi

  # Plant 1b: undeclared identifier in MY_SRC
  sed 's/CALL o1_write_si3_state_snapshot( '\''entry'\'', 0/CALL o1_write_si3_state_snapshot( '\''entry'\'', o1_undeclared_identifier/' "$scratch/icestp.f90" > "$scratch/icestp_undef_plant.f90"
  if "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$NEMO_MOD_DIR" -I "$NEMO_INC_DIR" -J "$scratch" "$scratch/icestp_undef_plant.f90" 2>/dev/null; then
    printf 'REFUSE: Plant 1b (undeclared identifier) did not fail gfortran -fsyntax-only\n' >&2
    exit 72
  fi

  # ---- Checklist Item 2: makenemo -r in cfgs/ref_cfgs.txt + plant ----
  check_ref_cfg_in_registry "$REF_CFG_SOURCE" "$NEMO_ROOT/cfgs/ref_cfgs.txt"
  if check_ref_cfg_in_registry "ORCA2_OMIP_L4" "$NEMO_ROOT/cfgs/ref_cfgs.txt" 2>/dev/null; then
    printf 'REFUSE: Plant 2 (work config ORCA2_OMIP_L4 in ref_cfgs.txt) did not fire\n' >&2
    exit 72
  fi

  # ---- Checklist Item 3: namelist groups, nn_fsbc multiples, sn_* stems, deck deviations + plants ----
  verify_namelists_and_stems "$NML_CFG" "$NML_ICE_CFG" "$scratch/bundles.toc" "none"
  for nml_plant in unknown-group missing-stem bad-fsbc-multiple unapproved-deviation; do
    if verify_namelists_and_stems "$NML_CFG" "$NML_ICE_CFG" "$scratch/bundles.toc" "$nml_plant" >"$scratch/nml_${nml_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 3 (%s) did not fail namelist verification\n' "$nml_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/nml_${nml_plant}.log" >/dev/null
  done

  # ---- Checklist Item 4: compiled gfortran multi-rank layout fixture + plants ----
  mkdir -p "$scratch/fixture_run/o1_si3_records"
  "$FC" -O2 -ffree-line-length-none -J "$scratch" \
    "$STUBS_F90" "$WRITER_F90" "$FIXTURE_F90" \
    -o "$scratch/o1_si3_layout_fixture.exe"
  # Plant 4a: missing o1_si3_records output directory must fail the writer
  mkdir -p "$scratch/fixture_missing_dir"
  if ( cd "$scratch/fixture_missing_dir" && "$scratch/o1_si3_layout_fixture.exe" ) >/dev/null 2>&1; then
    printf 'REFUSE: Plant 4a (missing writer output directory) did not fail\n' >&2
    exit 72
  fi
  ( cd "$scratch/fixture_run" && "$scratch/o1_si3_layout_fixture.exe" )
  "$PY" "$GATE_PY" --verify-layout-fixture "$scratch/fixture_run/o1_si3_records" > "$scratch/layout_fixture.json"

  # ---- Checklist Item 5: all expected frames written + ASCII-only check + plants ----
  for lf_plant in rank-collapse wrong-offset truncation bad-magic missing-frame field-swap; do
    if "$PY" "$GATE_PY" --verify-layout-fixture "$scratch/fixture_run/o1_si3_records" --plant "$lf_plant" >"$scratch/lf_${lf_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 4/5 (%s) did not fail layout fixture verification\n' "$lf_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/lf_${lf_plant}.log" >/dev/null
  done

  for ascii_target in "$WRITER_F90" "$PRA_PATCH" "$STP_PATCH" "$STUBS_F90" "$FIXTURE_F90" "$NML_CFG" "$NML_ICE_CFG"; do
    check_ascii_only "$ascii_target"
  done
  printf '! UTF-8 em-dash plant: —\n' > "$scratch/utf8_plant.F90"
  if check_ascii_only "$scratch/utf8_plant.F90" 2>/dev/null; then
    printf 'REFUSE: Plant 5 (UTF-8 character) did not fail check_ascii_only\n' >&2
    exit 72
  fi

  # ---- Checklist Item 6: disk check when parent does not exist yet + free -g >= 150 GB + plants ----
  check_free_disk_kb "$EVIDENCE/nonexistent_parent_a/nonexistent_parent_b/run" "$MIN_FREE_DISK_KB"
  if check_free_disk_kb "$EVIDENCE/nonexistent_parent_a/run" 999999999999999 2>/dev/null; then
    printf 'REFUSE: Plant 6a (impossible disk floor) did not fail\n' >&2
    exit 72
  fi
  check_free_ram_gb "$MIN_FREE_RAM_GB"
  if check_free_ram_gb 999999 2>/dev/null; then
    printf 'REFUSE: Plant 6b (impossible RAM floor) did not fail\n' >&2
    exit 72
  fi

  # ---- Checklist Item 7: target absence + whitespace-tolerant log check + EXIT trap to refused/ ----
  check_target_absent "$NEMO_ROOT/cfgs/$TARGET_REF_CFG"
  check_target_absent "$NEMO_ROOT/cfgs/$TARGET_REC_CFG"
  check_target_absent "$EVIDENCE"
  mkdir -p "$scratch/existing_target_plant"
  if check_target_absent "$scratch/existing_target_plant" 2>/dev/null; then
    printf 'REFUSE: Plant 7a (existing target directory) did not fail\n' >&2
    exit 72
  fi

  "$PY" "$GATE_PY" --verify-log "$LOG_EXCERPT" --expect-icedyn T --expect-icethd T > "$scratch/log_check.json"
  sed -E 's/=/   =    /g' "$LOG_EXCERPT" > "$scratch/ocean.output.spaced"
  "$PY" "$GATE_PY" --verify-log "$scratch/ocean.output.spaced" --expect-icedyn T --expect-icethd T >/dev/null
  for log_plant in error-token missing-aaa wrong-jpl wrong-flag; do
    if "$PY" "$GATE_PY" --verify-log "$LOG_EXCERPT" --plant "$log_plant" >"$scratch/log_${log_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 7b (log plant %s) did not fail\n' "$log_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/log_${log_plant}.log" >/dev/null
  done

  # Plant 7c: verify EXIT trap moves failed run to refused/ and preserves completed run
  mkdir -p "$scratch/trap_test/incomplete_run" "$scratch/trap_test/completed_run" "$scratch/trap_test/refused"
  touch "$scratch/trap_test/completed_run/RUN_COMPLETE"
  if (
    trap cleanup_on_exit EXIT
    REFUSED_DIR="$scratch/trap_test/refused"
    ACTIVE_RUN_DIR="$scratch/trap_test/incomplete_run"
    ACTIVE_SCRATCH_DIR=""
    exit 91
  ) 2>/dev/null; then
    printf 'REFUSE: Plant 7c subshell did not exit non-zero\n' >&2
    exit 72
  fi
  if [[ -e "$scratch/trap_test/incomplete_run" ]] || [[ -z "$(ls -A "$scratch/trap_test/refused")" ]]; then
    printf 'REFUSE: Plant 7c EXIT trap did not move incomplete run to refused/\n' >&2
    exit 72
  fi
  if (
    trap cleanup_on_exit EXIT
    REFUSED_DIR="$scratch/trap_test/refused"
    ACTIVE_RUN_DIR="$scratch/trap_test/completed_run"
    ACTIVE_SCRATCH_DIR=""
    exit 92
  ) 2>/dev/null; then
    printf 'REFUSE: Plant 7c completed subshell did not exit non-zero\n' >&2
    exit 72
  fi
  if [[ ! -f "$scratch/trap_test/completed_run/RUN_COMPLETE" ]]; then
    printf 'REFUSE: Plant 7c EXIT trap moved or deleted a completed run\n' >&2
    exit 72
  fi

  # Also run the full O1 gate self-test and its plants P2-P6
  "$PY" "$GATE_PY" --self-test > "$scratch/gate_self_test.json"
  for gate_plant in 1ulp-state 1ulp-moment north-fold-sign agg-formula stop-gradient; do
    if "$PY" "$GATE_PY" --self-test --plant "$gate_plant" >"$scratch/gp_${gate_plant}.log" 2>&1; then
      printf 'REFUSE: gate self-test plant %s did not fail\n' "$gate_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/gp_${gate_plant}.log" >/dev/null
  done

  if [[ -n "$preflight_json_out" ]]; then
    mkdir -p "$(dirname "$preflight_json_out")"
    "$PY" - "$preflight_json_out" "$ORCA2_BUNDLE_SHA256" "$SAS_BUNDLE_SHA256" \
      "$scratch/layout_fixture.json" "$scratch/log_check.json" "$scratch/gate_self_test.json" <<'PY'
import json
import pathlib
import sys

out_p, orca2_sha, sas_sha, lf_p, log_p, gst_p = sys.argv[1:]
report = {
    "status": "PREFLIGHT_OK",
    "rung": "O1",
    "reference_config": "ORCA2_SAS_ICE",
    "orca2_bundle_sha256": orca2_sha,
    "sas_bundle_sha256": sas_sha,
    "checklist_items_1_to_7": "PASS",
    "plants_verified": [
        "1a_continuation_ampersand",
        "1b_undeclared_identifier",
        "1c_bare_use_iso_fortran_env",
        "2_work_config_not_in_ref_cfgs",
        "3a_unknown_namelist_group",
        "3b_missing_sn_stem",
        "3c_bad_fsbc_multiple",
        "3d_unapproved_deck_deviation",
        "4a_missing_writer_output_dir",
        "4b_rank_collapse",
        "4c_wrong_offset",
        "4d_truncation",
        "4e_bad_magic",
        "5a_missing_frame",
        "5b_field_swap",
        "5c_utf8_non_ascii",
        "6a_disk_floor_nonexistent_parent",
        "6b_ram_floor_150gb",
        "7a_existing_target_dir",
        "7b_log_error_token",
        "7c_log_missing_aaa",
        "7d_log_wrong_jpl",
        "7e_log_wrong_flag",
        "7f_exit_trap_refused_and_completed_preserved",
        "gate_1ulp_state",
        "gate_1ulp_moment",
        "gate_north_fold_sign",
        "gate_agg_formula",
        "gate_stop_gradient",
    ],
    "layout_fixture": json.loads(pathlib.Path(lf_p).read_text()),
    "log_verifier": json.loads(pathlib.Path(log_p).read_text()),
    "gate_self_test": json.loads(pathlib.Path(gst_p).read_text()),
}
pathlib.Path(out_p).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
  fi

  printf 'PREFLIGHT_OK: checklist items 1-7 and all plants passed\n'
}

if [[ "$mode" == --preflight-only ]]; then
  run_preflight
  exit 0
fi

if [[ "$mode" == --admit-existing ]]; then
  "$PY" "$GATE_PY" --admit-evidence "$EVIDENCE" --json "$EVIDENCE/o1_admission.json"
  printf 'ACQUISITION_ADMITTED %s\n' "$EVIDENCE"
  exit 0
fi

# ---- Operator-only --run mode ----
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: --run requires a clean committed git worktree\n' >&2
  exit 63
}
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)

run_preflight

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/inputs_scratch"
ACTIVE_RUN_DIR="$EVIDENCE"
printf '%s\n' "$COMMIT" > "$EVIDENCE/manifest/producer_commit.txt"
sha256sum "$ORCA2_BUNDLE" "$SAS_BUNDLE" "$WRITER_F90" "$PRA_PATCH" "$STP_PATCH" \
  "$NML_CFG" "$NML_ICE_CFG" "$GATE_PY" > "$EVIDENCE/manifest/inputs.sha256"

tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/inputs_scratch"
tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/inputs_scratch"

build_target() {
  local cfg_name=$1
  local instrumented=$2
  local cfg_root=$NEMO_ROOT/cfgs/$cfg_name
  check_free_ram_gb "$MIN_FREE_RAM_GB"
  cd "$NEMO_ROOT"
  ./makenemo -r "$REF_CFG_SOURCE" -n "$cfg_name" -m conda-scalarmath del_key 'key_xios'
  if [[ "$instrumented" -eq 1 ]]; then
    cp "$SHIPPED_PRA" "$cfg_root/MY_SRC/icedyn_adv_pra.F90"
    cp "$SHIPPED_STP" "$cfg_root/MY_SRC/icestp.F90"
    cp "$WRITER_F90" "$cfg_root/MY_SRC/o1_si3_state_writer.F90"
    patch --fuzz=0 "$cfg_root/MY_SRC/icedyn_adv_pra.F90" < "$PRA_PATCH"
    patch --fuzz=0 "$cfg_root/MY_SRC/icestp.F90" < "$STP_PATCH"
    touch "$cfg_root/MY_SRC/"*.F90
    ./makenemo -n "$cfg_name" -m conda-scalarmath del_key 'key_xios'
  fi
  [[ -x "$cfg_root/BLD/bin/nemo.exe" ]] || {
    printf 'REFUSE: %s did not produce nemo.exe\n' "$cfg_name" >&2
    exit 68
  }
  local dyn_syms
  dyn_syms=$(nm -D "$cfg_root/BLD/bin/nemo.exe")
  if grep '_ZGV' <<<"$dyn_syms" >/dev/null; then
    printf 'REFUSE: vector-math symbol present in %s\n' "$cfg_name" >&2
    exit 68
  fi
}

stage_and_run() {
  local cfg_name=$1
  local run_dir=$2
  local is_rst_run=$3
  local cfg_root=$NEMO_ROOT/cfgs/$cfg_name

  check_free_ram_gb "$MIN_FREE_RAM_GB"
  check_free_disk_kb "$run_dir" "$MIN_FREE_DISK_KB"
  mkdir -p "$run_dir/o1_si3_records"
  cp -L "$SHIPPED_NML_REF" "$run_dir/namelist_ref"
  cp -L "$SHIPPED_NML_ICE_REF" "$run_dir/namelist_ice_ref"
  cp "$NML_CFG" "$run_dir/namelist_cfg"
  cp "$NML_ICE_CFG" "$run_dir/namelist_ice_cfg"
  if [[ "$is_rst_run" -eq 1 ]]; then
    sed -E -i 's/^( *nn_itend *= *).*/\11/' "$run_dir/namelist_cfg"
    sed -E -i 's|^(&namini.*)|\1\n   nn_iceini_file = 2\n   cn_icerst_in = "ORCA2_SAS_00000001_restart_ice"|' "$run_dir/namelist_ice_cfg"
    cp "$EVIDENCE/rec_2step/ORCA2_SAS_00000001_restart_ice.nc" "$run_dir/ORCA2_SAS_00000001_restart_ice.nc"
  fi
  for f in "$EVIDENCE/inputs_scratch"/*.nc; do
    ln -sf "$f" "$run_dir/$(basename "$f")"
  done
  cp "$cfg_root/BLD/bin/nemo.exe" "$run_dir/nemo"
  (
    cd "$run_dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > run.user.time.log
    timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1
    printf 'MPIRUN_RC=0\nRUN_FINISHED_UTC=%s\nRUN DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> run.user.time.log
  )
  "$PY" "$GATE_PY" --verify-log "$run_dir/ocean.output" --expect-icedyn F --expect-icethd F > "$run_dir/log_check.json"
  touch "$run_dir/RUN_COMPLETE"
}

build_target "$TARGET_REF_CFG" 0
build_target "$TARGET_REC_CFG" 1
sha256sum "$NEMO_ROOT/cfgs/$TARGET_REF_CFG/BLD/bin/nemo.exe" \
          "$NEMO_ROOT/cfgs/$TARGET_REC_CFG/BLD/bin/nemo.exe" > "$EVIDENCE/manifest/binaries.sha256"

stage_and_run "$TARGET_REF_CFG" "$EVIDENCE/ref_2step" 0
stage_and_run "$TARGET_REC_CFG" "$EVIDENCE/rec_2step" 0
stage_and_run "$TARGET_REC_CFG" "$EVIDENCE/rst_1step" 1

"$PY" "$GATE_PY" --admit-evidence "$EVIDENCE" --json "$EVIDENCE/o1_admission.json"
touch "$EVIDENCE/RUN_COMPLETE"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
