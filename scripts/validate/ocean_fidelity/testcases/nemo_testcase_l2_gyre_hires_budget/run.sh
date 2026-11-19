#!/usr/bin/env bash
set -Eeuo pipefail

# OPERATOR-EXECUTED ACQUISITION ONLY.  The default action is a read-only
# source/patch/syntax preflight.  --run-short verifies the correction at the
# first registered day; --run performs the corrected vector-math 180-day
# acquisition; --run-scalar repeats that acquisition for the arithmetic twin.
mode=preflight
if [[ $# -gt 1 ]]; then
  printf 'REFUSE_USAGE: %s [--run-short|--run|--preflight-scalar|--run-scalar]\n' "$0" >&2
  exit 64
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --run-short) mode=short ;;
    --run) mode=full ;;
    --preflight-scalar) mode=scalar_preflight ;;
    --run-scalar) mode=scalar ;;
    *) printf 'REFUSE_USAGE: %s [--run-short|--run|--preflight-scalar|--run-scalar]\n' "$0" >&2; exit 64 ;;
  esac
fi

refuse_on_error() {
  local status=$?
  printf 'REFUSE: round-15 NEMO budget failed at line %s (status %s)\n' \
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
if [[ "$mode" == short ]]; then
  readonly TARGET_NAME=GYRE_OMIP_L2_P3_R15BUDV
  readonly LAST_DAY=30
  readonly LAST_STEP=720
  readonly RECORD_COUNT=1
  readonly EXPERIMENT=GYRE_R4_SMAG_R15_VERIFY
  readonly DURATION_PATCH_NAME=namelist_cfg_r4_smag_30d_verify.patch
  readonly DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_verify_day30
elif [[ "$mode" == scalar || "$mode" == scalar_preflight ]]; then
  readonly TARGET_NAME=GYRE_OMIP_L2_P3_R16BUDSM
  readonly LAST_DAY=180
  readonly LAST_STEP=4320
  readonly RECORD_COUNT=151
  readonly EXPERIMENT=GYRE_R4_SMAG_R14_BUDGET
  readonly DURATION_PATCH_NAME=namelist_cfg_r4_smag_180d.patch
  readonly DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/oracle_gyre_r4_smag_budget_180d_scalar
else
  readonly TARGET_NAME=GYRE_OMIP_L2_P3_R15BUD
  readonly LAST_DAY=180
  readonly LAST_STEP=4320
  readonly RECORD_COUNT=151
  readonly EXPERIMENT=GYRE_R4_SMAG_R14_BUDGET
  readonly DURATION_PATCH_NAME=namelist_cfg_r4_smag_180d.patch
  readonly DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected
fi
if [[ "$mode" == scalar || "$mode" == scalar_preflight ]]; then
  readonly ARCH=conda-scalarmath
  readonly PRECISION_FLAVOUR=scalar-math-conda
  readonly ARCH_SHA256=132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561
else
  readonly ARCH=conda
  readonly PRECISION_FLAVOUR=vector-math-conda
  readonly ARCH_SHA256=64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a
fi
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly ENGINE_ROOT=$NEMO_ROOT/src/OCE/DYN
readonly BASE_NAMELIST=$here/../configs/gyre_omip_l2_phase3_namelist_cfg
readonly SMAG_PATCH=$here/../nemo_testcase_l2_gyre_hires/namelist_cfg_r4_smag_kt10.patch
readonly DURATION_PATCH=$here/$DURATION_PATCH_NAME
readonly DECK_SH=$here/../nemo_testcase_l2_gyre_hires/deck.sh
readonly STAGE_PATCH=$here/stprk3_stg_round14.patch
readonly ADV_PATCH=$here/dynadv_round14.patch
readonly ZDF_PATCH=$here/dynzdf_round14.patch
readonly MODULE=$here/gyre_r14_budget.F90
readonly PACKAGER=$here/package_records.py
readonly PREREG=$repo/docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_hires_round14_preregistration.md
readonly REGISTRATION_COMMIT=3293fd5f49c332038cc61735ed1dc7bbc0f7b798
readonly REGISTRATION_SHA256=480a45f3d7f5e02bd5c4771316bae668e59ee33aaeab87a73c47a6212edf5174
readonly EXPECTED_NEMO_COMMIT=dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
readonly EVIDENCE=${EVIDENCE:-$DEFAULT_EVIDENCE}
readonly RANKS=${RANKS:-1}
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly RECORD_SIZE=$((16 + 9 * 4 + 2 * 8 + 126 * 86 * 9 * 8))
printf -v restart_step '%08d' "$LAST_STEP"
readonly RESTART=${EXPERIMENT}_${restart_step}_restart.nc

[[ "$RANKS" == 1 ]] || {
  printf 'REFUSE_RANKS: the writer is lwp-only and not gathered; RANKS must be 1\n' >&2
  exit 65
}
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact writer and preregistration first\n' >&2
  exit 66
}
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" merge-base --is-ancestor "$REGISTRATION_COMMIT" "$COMMIT" || {
  printf 'REFUSE_REGISTRATION_ANCESTRY: %s does not descend from %s\n' \
    "$COMMIT" "$REGISTRATION_COMMIT" >&2
  exit 66
}
readonly NEMO_COMMIT=$(git -C "$NEMO_ROOT" rev-parse HEAD)
[[ "$NEMO_COMMIT" == "$EXPECTED_NEMO_COMMIT" ]] || {
  printf 'REFUSE_NEMO_COMMIT: got %s, expected %s\n' \
    "$NEMO_COMMIT" "$EXPECTED_NEMO_COMMIT" >&2
  exit 66
}
nemo_build_changes=$(git -C "$NEMO_ROOT" status --porcelain \
  --untracked-files=all -- src ext mk tools makenemo)
[[ -z "$nemo_build_changes" ]] || {
  printf 'REFUSE_NEMO_BUILD_TREE: executing source/build inputs changed:\n%s\n' \
    "$nemo_build_changes" >&2
  exit 66
}
for path in "$BASE_NAMELIST" "$SMAG_PATCH" "$DURATION_PATCH" \
  "$STAGE_PATCH" "$ADV_PATCH" "$ZDF_PATCH" "$MODULE" "$PACKAGER" \
  "$PREREG" "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
  "$ENGINE_ROOT/dynadv.F90" "$ENGINE_ROOT/dynzdf.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" "$NEMO_ROOT/arch/arch-$ARCH.fcm"; do
  [[ -f "$path" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
[[ "$RECORD_SIZE" -eq 780260 ]] || {
  printf 'REFUSE_LAYOUT: record layout arithmetic changed\n' >&2
  exit 67
}

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    printf 'REFUSE_SOURCE_IDENTITY: %s is %s, expected %s\n' \
      "$path" "$actual" "$expected" >&2
    exit 68
  }
}
check_hash "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
  4d3493de0e837654e961a30ac4684801d87b4226ed9e8c226922db780d699afe
check_hash "$SOURCE_ROOT/MY_SRC/dynhpg.F90" \
  6c15901bfcb58e768523ef1afab7f8365085afe6a8e86be810a7ee2078e4bd73
check_hash "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
  c50912ba02772fe11eb8a164b05b05077a993ca88058e5ed01aad1d16d6b8b82
check_hash "$SOURCE_ROOT/MY_SRC/dynvor.F90" \
  06b1fd76c44e62d926e896e53006153ace4ec87d759b7d17765ad37462c6b8de
check_hash "$SOURCE_ROOT/MY_SRC/eosbn2.F90" \
  8ffe1b56ce90b8924ecebc37c618c6be67a521aa190c102d48e136533fcf5067
check_hash "$SOURCE_ROOT/MY_SRC/stprk3.F90" \
  11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee
check_hash "$SOURCE_ROOT/MY_SRC/traadv.F90" \
  4807fa59ce15237ac04ef8e84f39c77b3da4080aa9ad1effa517fb79e1a6968d
check_hash "$SOURCE_ROOT/EXP00/namelist_ref" \
  b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd
check_hash "$ENGINE_ROOT/dynadv.F90" \
  d6802c22ce6a3b0307b556cbb20326f04f916711e2288f68e7a67dc7beb94a6a
check_hash "$ENGINE_ROOT/dynzdf.F90" \
  4489773bb4cafe46123cfd658b1561188215c9ce0db11f6859fb27cb019bb174
check_hash "$BASE_NAMELIST" \
  85d0f2b0655d7d89dbe676e1d1a0dd1b73ab387b76111dfabc1627d4ab81271d
check_hash "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" \
  54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7
check_hash "$NEMO_ROOT/arch/arch-$ARCH.fcm" "$ARCH_SHA256"
check_hash "$PREREG" "$REGISTRATION_SHA256"
readonly EXPECTED_MY_SRC='dynhpg.F90
dynspg_ts.F90
dynvor.F90
eosbn2.F90
stprk3.F90
stprk3_stg.F90
traadv.F90'
actual_my_src=$(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f \
  -printf '%f\n' | sort)
[[ "$actual_my_src" == "$EXPECTED_MY_SRC" ]] || {
  printf 'REFUSE_MY_SRC_REGISTRY: expected exactly:\n%s\ngot:\n%s\n' \
    "$EXPECTED_MY_SRC" "$actual_my_src" >&2
  exit 68
}

dry=$(mktemp -d /tmp/gyre-r14-budget-source.XXXXXX)
syntax=$(mktemp -d /tmp/gyre-r14-budget-syntax.XXXXXX)
trap 'rm -rf -- "$dry" "$syntax"' EXIT
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
cp "$ENGINE_ROOT/dynadv.F90" "$dry/dynadv.F90"
cp "$ENGINE_ROOT/dynzdf.F90" "$dry/dynzdf.F90"
cp "$MODULE" "$dry/gyre_r14_budget.F90"
patch -l -s --fuzz=0 "$dry/stprk3_stg.F90" <"$STAGE_PATCH"
patch -l -s --fuzz=0 "$dry/dynadv.F90" <"$ADV_PATCH"
patch -l -s --fuzz=0 "$dry/dynzdf.F90" <"$ZDF_PATCH"
cp "$BASE_NAMELIST" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$SMAG_PATCH"
. "$DECK_SH"
gyre_hires_verify_deck r4_smag "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DURATION_PATCH"
for pair in "cn_exp:\"$EXPERIMENT\"" 'nn_it000:1' "nn_itend:$LAST_STEP" \
            "nn_stock:$LAST_STEP" 'nn_write:24' \
            'nn_GYRE:4' 'rn_Dt:3600.' 'rn_Ld:25000.' \
            'ln_dynldf_lap:.false.' 'ln_dynldf_blp:.true.' \
            'ln_dynldf_lev:.true.' 'nn_ahm_ijk_t:32' 'rn_csmc:3.5' \
            'rn_minfac:1.0' 'rn_maxfac:1.0'; do
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

for unit in gyre_r14_budget dynadv dynzdf stprk3_stg; do
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$unit.F90" -o "$syntax/$unit.f90"
done
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" \
  "$syntax/gyre_r14_budget.f90" -o "$syntax/gyre_r14_budget.o"
for unit in dynadv dynzdf stprk3_stg; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/$unit.f90"
done
"$PY" -m py_compile "$PACKAGER"

for needle in 'GYRE_R14_BUDGET' 'jr14_first_step = 697' \
  'jr14_last_step  = 4320' 'r14_after_hpg' 'r14_after_vor' \
  'r14_after_keg' 'r14_after_zad' 'r14_after_ldf' \
  'r14_zdf_pre_row' 'r14_zdf_row' 'r14_finish'; do
  grep -R -Fq "$needle" "$dry"/*.F90 || {
    printf 'REFUSE_WRITER_PROGRAM: dry source lacks %s\n' "$needle" >&2
    exit 68
  }
done
printf 'PREFLIGHT: deck=r4_smag days=%s window=30:%s ranks=1 ' \
  "$LAST_DAY" "$LAST_DAY"
printf 'record_count=%s record_size=%s precision=%s evidence=%s commit=%s\n' \
  "$RECORD_COUNT" "$RECORD_SIZE" "$PRECISION_FLAVOUR" "$EVIDENCE" "$COMMIT"
if [[ "$mode" == preflight || "$mode" == scalar_preflight ]]; then
  if [[ "$mode" == preflight ]]; then
    printf 'WOULD_RUN_SHORT: EVIDENCE=%s RANKS=1 %s --run-short\n' \
      '/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_verify_day30' "$0"
    printf 'WOULD_RUN_FULL: EVIDENCE=%s RANKS=1 %s --run\n' "$EVIDENCE" "$0"
  fi
  printf 'WOULD_RUN_SCALAR: EVIDENCE=%s RANKS=1 %s --run-scalar\n' \
    '/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/oracle_gyre_r4_smag_budget_180d_scalar' "$0"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s or %s\n' "$TARGET_ROOT" "$EVIDENCE" >&2
  exit 69
}
for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 6291456 ]] || {
    printf 'REFUSE_SPACE: %s has under 6 GB free\n' "$mount" >&2
    exit 69
  }
done

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
cd "$NEMO_ROOT"
./makenemo -r "$SOURCE_NAME" -n "$TARGET_NAME" -d 'OCE' -m "$ARCH" -j 0
[[ -d "$TARGET_ROOT/MY_SRC" && -d "$TARGET_ROOT/EXP00" ]] || {
  printf 'REFUSE_BUILD_CLONE: makenemo did not create the target configuration\n' >&2
  exit 70
}
if ! grep -Eq "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" cfgs/work_cfgs.txt; then
  printf '%s OCE\n' "$TARGET_NAME" >>cfgs/work_cfgs.txt
fi
find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f -print0 | sort -z | \
  while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/"; done
cp "$ENGINE_ROOT/dynadv.F90" "$TARGET_ROOT/MY_SRC/dynadv.F90"
cp "$ENGINE_ROOT/dynzdf.F90" "$TARGET_ROOT/MY_SRC/dynzdf.F90"
cp "$MODULE" "$TARGET_ROOT/MY_SRC/gyre_r14_budget.F90"
cp "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" "$TARGET_ROOT/cpp_$TARGET_NAME.fcm"
patch -l -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STAGE_PATCH"
patch -l -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynadv.F90" <"$ADV_PATCH"
patch -l -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynzdf.F90" <"$ZDF_PATCH"
for unit in stprk3_stg dynadv dynzdf gyre_r14_budget; do
  cmp -s "$dry/$unit.F90" "$TARGET_ROOT/MY_SRC/$unit.F90" || {
    printf 'REFUSE_BUILD_SOURCE: target %s differs from syntax proof\n' "$unit" >&2
    exit 70
  }
done
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_NAME" -m "$ARCH" -j 8
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE_BUILD_BINARY: missing %s\n' "$BINARY" >&2; exit 70; }
for compiled in gyre_r14_budget dynadv dynzdf stprk3_stg; do
  path=$TARGET_ROOT/BLD/ppsrc/nemo/$compiled.f90
  [[ -f "$path" ]] && grep -Fq 'r14_' "$path" || {
    printf 'REFUSE_COMPILED_PROGRAM: %s lacks round-14 calls\n' "$path" >&2
    exit 70
  }
done

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/control" "$EVIDENCE/instrumented"
cp "$dry/resolved_namelist_lines.txt" "$EVIDENCE/manifest/"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/producer_commit.txt"
readonly SOURCE_MANIFEST=$EVIDENCE/manifest/writer_source.sha256
: >"$SOURCE_MANIFEST"
manifest_add() {
  local path=$1 name=$2 digest
  digest=$(sha256sum "$path" | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$name" >>"$SOURCE_MANIFEST"
}
manifest_add "$MODULE" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/gyre_r14_budget.F90'
manifest_add "$STAGE_PATCH" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/stprk3_stg_round14.patch'
manifest_add "$ADV_PATCH" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/dynadv_round14.patch'
manifest_add "$ZDF_PATCH" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/dynzdf_round14.patch'
manifest_add "$DURATION_PATCH" \
  "repo:${DURATION_PATCH#"$repo/"}"
manifest_add "$PACKAGER" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/package_records.py'
manifest_add "$here/run.sh" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh'
manifest_add "$SMAG_PATCH" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/namelist_cfg_r4_smag_kt10.patch'
manifest_add "$DECK_SH" \
  'repo:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/deck.sh'
manifest_add "$BASE_NAMELIST" \
  'repo:scripts/validate/ocean_fidelity/testcases/configs/gyre_omip_l2_phase3_namelist_cfg'
manifest_add "$PREREG" \
  'repo:docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_hires_round14_preregistration.md'
for name in dynhpg.F90 dynspg_ts.F90 dynvor.F90 eosbn2.F90 stprk3.F90 \
  stprk3_stg.F90 traadv.F90; do
  manifest_add "$SOURCE_ROOT/MY_SRC/$name" \
    "oracle:cfgs/GYRE_OMIP_L2_P3/MY_SRC/$name"
done
manifest_add "$SOURCE_ROOT/EXP00/namelist_ref" \
  'oracle:cfgs/GYRE_OMIP_L2_P3/EXP00/namelist_ref'
manifest_add "$SOURCE_ROOT/cpp_$SOURCE_NAME.fcm" \
  'oracle:cfgs/GYRE_OMIP_L2_P3/cpp_GYRE_OMIP_L2_P3.fcm'
manifest_add "$ENGINE_ROOT/dynadv.F90" 'oracle:src/OCE/DYN/dynadv.F90'
manifest_add "$ENGINE_ROOT/dynzdf.F90" 'oracle:src/OCE/DYN/dynzdf.F90'
manifest_add "$NEMO_ROOT/arch/arch-$ARCH.fcm" "oracle:arch/arch-$ARCH.fcm"

readonly PREPARED='namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml'
for run in control instrumented; do
  cp "$dry/namelist_cfg" "$EVIDENCE/$run/namelist_cfg"
  for name in $PREPARED; do
    [[ -e "$SOURCE_ROOT/EXP00/$name" ]] || {
      printf 'REFUSE_PREPARED_INPUT: missing %s\n' "$name" >&2
      exit 71
    }
    cp -L "$SOURCE_ROOT/EXP00/$name" "$EVIDENCE/$run/$name"
  done
  cp "$BINARY" "$EVIDENCE/$run/nemo"
done
cmp -s "$EVIDENCE/control/nemo" "$EVIDENCE/instrumented/nemo" || {
  printf 'REFUSE_CONTROL_BINARY: control and instrumented executables differ\n' >&2
  exit 71
}

run_nemo() {
  local name=$1 enabled=$2 root=$EVIDENCE/$1
  (
    cd "$root"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export GYRE_R14_BUDGET=$enabled
    printf 'RUN_STARTED_UTC=%s GYRE_R14_BUDGET=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$enabled" >run.user.time.log
    mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
    printf 'RUN_FINISHED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.time.log
  )
  grep -Fxq 'STOP 0' "$root/run.user.log" || {
    printf 'REFUSE_NEMO_RUN: %s did not terminate with STOP 0\n' "$name" >&2
    exit 72
  }
  grep -Eq "GYRE_R14_BUDGET_ENABLED[[:space:]]+[TF]" "$root/ocean.output" || {
    printf 'REFUSE_NEMO_RUN: %s lacks activation receipt\n' "$name" >&2
    exit 72
  }
}
run_nemo control 0
run_nemo instrumented 1
cmp -s "$EVIDENCE/control/output.namelist.dyn" \
  "$EVIDENCE/instrumented/output.namelist.dyn" || {
  printf 'REFUSE_RESOLVED_NAMELIST: control/instrumented runtime namelists differ\n' >&2
  exit 72
}

control_count=$(find "$EVIDENCE/control" -maxdepth 1 -type f \
  -name 'oracle_gyre_r14_budget_day*.bin' | wc -l)
[[ "$control_count" -eq 0 ]] || {
  printf 'REFUSE_CONTROL_WRITES: disabled run wrote %s records\n' "$control_count" >&2
  exit 73
}
record_count=$(find "$EVIDENCE/instrumented" -maxdepth 1 -type f \
  -name 'oracle_gyre_r14_budget_day*.bin' | wc -l)
[[ "$record_count" -eq "$RECORD_COUNT" ]] || {
  printf 'REFUSE_RECORD_COUNT: got %s, expected %s\n' \
    "$record_count" "$RECORD_COUNT" >&2
  exit 73
}
for day in $(seq 30 "$LAST_DAY"); do
  path=$(printf '%s/instrumented/oracle_gyre_r14_budget_day%04d.bin' \
    "$EVIDENCE" "$day")
  [[ -f "$path" && "$(stat -c %s "$path")" -eq "$RECORD_SIZE" ]] || {
    printf 'REFUSE_RECORD_LAYOUT: %s is missing or not %s bytes\n' \
      "$path" "$RECORD_SIZE" >&2
    exit 73
  }
done
for run in control instrumented; do
  [[ -f "$EVIDENCE/$run/$RESTART" ]] || {
    printf 'REFUSE_RESTART: %s omitted %s\n' "$run" "$RESTART" >&2
    exit 73
  }
done

package_mode=()
[[ "$mode" != short ]] || package_mode=(--verify-short)
"$PY" "$PACKAGER" \
  --run-root "$EVIDENCE/instrumented" \
  --control-restart "$EVIDENCE/control/$RESTART" \
  --instrumented-restart "$EVIDENCE/instrumented/$RESTART" \
  --preregistration "$PREREG" \
  --registration-commit "$REGISTRATION_COMMIT" \
  --producer-commit "$COMMIT" \
  --resolved-namelist "$EVIDENCE/instrumented/output.namelist.dyn" \
  --writer-source-manifest "$EVIDENCE/manifest/writer_source.sha256" \
  --binary "$BINARY" \
  --nemo-engine-commit "$NEMO_COMMIT" \
  --precision-flavour "$PRECISION_FLAVOUR" \
  "${package_mode[@]}" \
  >"$EVIDENCE/instrumented/package_records.log"
(
  cd "$EVIDENCE"
  find manifest control instrumented -maxdepth 2 -type f -print0 | \
    sort -z | xargs -0 sha256sum >operator_manifest.sha256
)
printf 'ACQUISITION_DONE: %s (r4_smag, %s, days 30..%s, same-binary passive control)\n' \
  "$EVIDENCE" "$PRECISION_FLAVOUR" "$LAST_DAY"
