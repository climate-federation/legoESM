#!/usr/bin/env bash
set -Eeuo pipefail

# OPERATOR-EXECUTED ACQUISITION ONLY.  With no argument this performs a
# read-only patch/deck/compiler preflight.  The two run modes start from the
# SAME vector-math day-180 restart; only the compiler arithmetic flavour
# differs.
mode=preflight
if [[ $# -gt 1 ]]; then
  printf 'REFUSE_USAGE: %s [--preflight-vector|--preflight-scalar|--run-vector|--run-scalar]\n' "$0" >&2
  exit 64
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --preflight-vector) mode=preflight ;;
    --preflight-scalar) mode=preflight-scalar ;;
    --run-vector) mode=vector ;;
    --run-scalar) mode=scalar ;;
    *) printf 'REFUSE_USAGE: %s [--preflight-vector|--preflight-scalar|--run-vector|--run-scalar]\n' "$0" >&2; exit 64 ;;
  esac
fi

refuse_on_error() {
  local status=$?
  printf 'REFUSE: round-17 vorticity acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: acquisition tool is outside git\n' >&2
  exit 65
}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_NAME=GYRE_OMIP_L2_P3
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly COMMON_RESTART=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected/control/GYRE_R4_SMAG_R14_BUDGET_00004320_restart.nc
readonly COMMON_RESTART_SHA256=a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12
readonly INPUT_RESTART=GYRE_R4_SMAG_R14_BUDGET_00004320_restart.nc
readonly OUTPUT_RESTART=GYRE_R17_VORTEX_00004321_restart.nc
readonly STEP=4321
readonly RANKS=${RANKS:-1}
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BASE_NAMELIST=$here/../configs/gyre_omip_l2_phase3_namelist_cfg
readonly SMAG_PATCH=$here/../nemo_testcase_l2_gyre_hires/namelist_cfg_r4_smag_kt10.patch
readonly DECK_SH=$here/../nemo_testcase_l2_gyre_hires/deck.sh
readonly RUN_PATCH=$here/namelist_cfg_round17.patch
readonly STAGE_PATCH=$here/stprk3_stg_round17.patch
readonly VOR_PATCH=$here/dynvor_round17.patch
readonly PREREG=$repo/docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_hires_round17_preregistration.md
readonly EXPECTED_NEMO_COMMIT=dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
readonly STAGE_RECORD_SIZE=10749372
readonly Q_RECORD_SIZE=7801968

if [[ "$mode" == scalar || "$mode" == preflight-scalar ]]; then
  readonly ARCH=conda-scalarmath
  readonly ARCH_SHA256=132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561
  readonly TARGET_NAME=GYRE_OMIP_L2_P3_R17VORS
  readonly FLAVOUR=scalar-math-conda
  readonly DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_scalar
else
  readonly ARCH=conda
  readonly ARCH_SHA256=64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a
  readonly TARGET_NAME=GYRE_OMIP_L2_P3_R17VORV
  readonly FLAVOUR=vector-math-conda
  readonly DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_vector
fi
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly EVIDENCE=${EVIDENCE:-$DEFAULT_EVIDENCE}

[[ "$RANKS" == 1 ]] || {
  printf 'REFUSE_RANKS: the writer is lwp-only and not gathered; RANKS must be 1\n' >&2
  exit 65
}
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact writer and preregistration first\n' >&2
  exit 66
}
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)
readonly NEMO_COMMIT=$(git -C "$NEMO_ROOT" rev-parse HEAD)
[[ "$NEMO_COMMIT" == "$EXPECTED_NEMO_COMMIT" ]] || {
  printf 'REFUSE_NEMO_COMMIT: got %s, expected %s\n' \
    "$NEMO_COMMIT" "$EXPECTED_NEMO_COMMIT" >&2
  exit 66
}
nemo_engine_changes=$(git -C "$NEMO_ROOT" status --porcelain \
  --untracked-files=all -- src ext mk tools makenemo)
[[ -z "$nemo_engine_changes" ]] || {
  printf 'REFUSE_NEMO_ENGINE_TREE: executable source inputs changed:\n%s\n' \
    "$nemo_engine_changes" >&2
  exit 66
}
for path in "$COMMON_RESTART" "$BASE_NAMELIST" "$SMAG_PATCH" "$DECK_SH" \
  "$RUN_PATCH" "$STAGE_PATCH" "$VOR_PATCH" "$PREREG" \
  "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$SOURCE_ROOT/MY_SRC/dynvor.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" "$NEMO_ROOT/arch/arch-$ARCH.fcm"; do
  [[ -f "$path" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    printf 'REFUSE_SOURCE_IDENTITY: %s is %s, expected %s\n' \
      "$path" "$actual" "$expected" >&2
    exit 68
  }
}
check_hash "$COMMON_RESTART" "$COMMON_RESTART_SHA256"
check_hash "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
  4d3493de0e837654e961a30ac4684801d87b4226ed9e8c226922db780d699afe
check_hash "$SOURCE_ROOT/MY_SRC/dynvor.F90" \
  06b1fd76c44e62d926e896e53006153ace4ec87d759b7d17765ad37462c6b8de
check_hash "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" \
  54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7
check_hash "$NEMO_ROOT/arch/arch-$ARCH.fcm" "$ARCH_SHA256"

dry=$(mktemp -d /tmp/gyre-r17-vorticity-source.XXXXXX)
syntax=$(mktemp -d /tmp/gyre-r17-vorticity-syntax.XXXXXX)
trap 'rm -rf -- "$dry" "$syntax"' EXIT
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
cp "$SOURCE_ROOT/MY_SRC/dynvor.F90" "$dry/dynvor.F90"
patch -s --fuzz=0 "$dry/stprk3_stg.F90" <"$STAGE_PATCH"
patch -s --fuzz=0 "$dry/dynvor.F90" <"$VOR_PATCH"
cp "$BASE_NAMELIST" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$SMAG_PATCH"
. "$DECK_SH"
# First certify the exact operator-selected r4_smag deck while it still has
# the ten-step acquisition cadence that the shared resolver validates.  The
# round-17 patch below changes run control and restart only.
gyre_hires_verify_deck r4_smag "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$RUN_PATCH"
for pair in 'cn_exp:"GYRE_R17_VORTEX"' 'nn_it000:4321' \
  'nn_itend:4321' 'nn_stock:4321' 'nn_write:4321' 'ln_rstart:.true.' \
  'nn_rstctl:2' 'cn_ocerst_in:"GYRE_R4_SMAG_R14_BUDGET_00004320_restart"'; do
  key=${pair%%:*}; expected=${pair#*:}
  actual=$(gyre_hires_namelist_value "$dry/namelist_cfg" "$key")
  [[ "$actual" == "$expected" ]] || {
    printf 'REFUSE_RESOLVED_NAMELIST: %s=%s, expected %s\n' \
      "$key" "$actual" "$expected" >&2
    exit 68
  }
done
gyre_hires_write_resolved_namelist \
  "$dry/namelist_cfg" "$dry/resolved_namelist_lines.txt"
for unit in stprk3_stg dynvor; do
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$unit.F90" -o "$syntax/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -J "$syntax" "$syntax/$unit.f90"
done
for needle in GYRE_R17_WRITE NEMO_L2_R17VOR1 NEMO_L2_R17Q__1; do
  grep -Fq "$needle" "$dry/stprk3_stg.F90" "$dry/dynvor.F90" || {
    printf 'REFUSE_WRITER_PROGRAM: dry sources lack %s\n' "$needle" >&2
    exit 68
  }
done
printf 'PREFLIGHT: deck=r4_smag step=%s ranks=1 precision=%s restart_sha256=%s evidence=%s commit=%s\n' \
  "$STEP" "$FLAVOUR" "$COMMON_RESTART_SHA256" "$EVIDENCE" "$COMMIT"
if [[ "$mode" == preflight || "$mode" == preflight-scalar ]]; then
  printf 'WOULD_RUN_VECTOR: EVIDENCE=%s RANKS=1 %s --run-vector\n' \
    '/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_vector' "$0"
  printf 'WOULD_RUN_SCALAR: EVIDENCE=%s RANKS=1 %s --run-scalar\n' \
    '/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round17/oracle_gyre_r4_smag_stage3_vorticity_scalar' "$0"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s or %s\n' "$TARGET_ROOT" "$EVIDENCE" >&2
  exit 69
}
for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 3145728 ]] || {
    printf 'REFUSE_SPACE: %s has under 3 GB free\n' "$mount" >&2
    exit 69
  }
done

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
cd "$NEMO_ROOT"
./makenemo -r "$SOURCE_NAME" -n "$TARGET_NAME" -d 'OCE' -m "$ARCH" -j 0
[[ -d "$TARGET_ROOT/MY_SRC" && -d "$TARGET_ROOT/EXP00" ]] || {
  printf 'REFUSE_BUILD_CLONE: makenemo did not create target config\n' >&2
  exit 70
}
if ! grep -Eq "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" cfgs/work_cfgs.txt; then
  printf '%s OCE\n' "$TARGET_NAME" >>cfgs/work_cfgs.txt
fi
find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f -print0 | sort -z | \
  while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/"; done
cp "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" "$TARGET_ROOT/cpp_$TARGET_NAME.fcm"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STAGE_PATCH"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynvor.F90" <"$VOR_PATCH"
cmp -s "$dry/stprk3_stg.F90" "$TARGET_ROOT/MY_SRC/stprk3_stg.F90"
cmp -s "$dry/dynvor.F90" "$TARGET_ROOT/MY_SRC/dynvor.F90"
touch "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" "$TARGET_ROOT/MY_SRC/dynvor.F90"
./makenemo -n "$TARGET_NAME" -m "$ARCH" -j 8
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE_BUILD_BINARY: missing %s\n' "$BINARY" >&2; exit 70; }
grep -Fq NEMO_L2_R17VOR1 "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90"
grep -Fq NEMO_L2_R17Q__1 "$TARGET_ROOT/BLD/ppsrc/nemo/dynvor.f90"

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/control" "$EVIDENCE/instrumented"
cp "$dry/resolved_namelist_lines.txt" "$EVIDENCE/manifest/"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/producer_commit.txt"
printf '%s\n' "$FLAVOUR" >"$EVIDENCE/manifest/precision_flavour.txt"
printf '%s  %s\n' "$COMMON_RESTART_SHA256" "$COMMON_RESTART" \
  >"$EVIDENCE/manifest/common_restart.sha256"
sha256sum "$STAGE_PATCH" "$VOR_PATCH" "$RUN_PATCH" "$SMAG_PATCH" \
  "$PREREG" "$here/run.sh" >"$EVIDENCE/manifest/writer_source.sha256"
readonly PREPARED='namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml'
for run in control instrumented; do
  cp "$dry/namelist_cfg" "$EVIDENCE/$run/namelist_cfg"
  cp "$COMMON_RESTART" "$EVIDENCE/$run/$INPUT_RESTART"
  for name in $PREPARED; do
    [[ -e "$SOURCE_ROOT/EXP00/$name" ]] || {
      printf 'REFUSE_PREPARED_INPUT: missing %s\n' "$name" >&2
      exit 71
    }
    cp -L "$SOURCE_ROOT/EXP00/$name" "$EVIDENCE/$run/$name"
  done
  cp "$BINARY" "$EVIDENCE/$run/nemo"
done
cmp -s "$EVIDENCE/control/nemo" "$EVIDENCE/instrumented/nemo"

run_nemo() {
  local name=$1 enabled=$2 root=$EVIDENCE/$1
  (
    cd "$root"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export GYRE_R17_WRITE=$enabled
    printf 'RUN_STARTED_UTC=%s GYRE_R17_WRITE=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$enabled" >run.user.time.log
    mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
    printf 'RUN_FINISHED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.time.log
  )
  grep -Fxq 'STOP 0' "$root/run.user.log" || {
    printf 'REFUSE_NEMO_RUN: %s did not terminate with STOP 0\n' "$name" >&2
    exit 72
  }
  [[ -f "$root/$OUTPUT_RESTART" ]] || {
    printf 'REFUSE_RESTART: %s omitted %s\n' "$name" "$OUTPUT_RESTART" >&2
    exit 72
  }
}
run_nemo control 0
run_nemo instrumented 1
cmp -s "$EVIDENCE/control/$OUTPUT_RESTART" \
  "$EVIDENCE/instrumented/$OUTPUT_RESTART" || {
  printf 'REFUSE_WRITER_MOVED_TRAJECTORY: terminal restarts differ\n' >&2
  exit 73
}
if find "$EVIDENCE/control" -maxdepth 1 -type f -name 'oracle_r17_*.bin' | grep -q .; then
  printf 'REFUSE_CONTROL_WRITES: disabled run emitted round-17 records\n' >&2
  exit 73
fi
stage=$EVIDENCE/instrumented/oracle_r17_stage3_vor_kt00004321.bin
[[ -f "$stage" && "$(stat -c %s "$stage")" -eq "$STAGE_RECORD_SIZE" ]] || {
  printf 'REFUSE_STAGE_LAYOUT: %s missing or wrong size\n' "$stage" >&2
  exit 73
}
q_count=$(find "$EVIDENCE/instrumented" -maxdepth 1 -type f \
  -name 'oracle_r17_ene_q_kmm?_kt00004321.bin' | wc -l)
[[ "$q_count" -eq 3 ]] || {
  printf 'REFUSE_Q_RECORD_COUNT: got %s, expected 3\n' "$q_count" >&2
  exit 73
}
while IFS= read -r q_record; do
  [[ "$(stat -c %s "$q_record")" -eq "$Q_RECORD_SIZE" ]] || {
    printf 'REFUSE_Q_LAYOUT: %s has wrong size\n' "$q_record" >&2
    exit 73
  }
done < <(find "$EVIDENCE/instrumented" -maxdepth 1 -type f \
  -name 'oracle_r17_ene_q_kmm?_kt00004321.bin' | sort)
(
  cd "$EVIDENCE"
  find control instrumented manifest -type f ! -name manifest.sha256 \
    -print0 | sort -z | \
    xargs -0 sha256sum >manifest.sha256
)
printf 'ACQUISITION_DONE: %s precision=%s common_restart=%s passivity=BIT_EXACT\n' \
  "$EVIDENCE" "$FLAVOUR" "$COMMON_RESTART_SHA256"
