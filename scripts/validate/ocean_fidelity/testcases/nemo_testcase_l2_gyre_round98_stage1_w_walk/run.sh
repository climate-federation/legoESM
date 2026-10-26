#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected failure at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

# ``--run`` is the user-executed NEMO acquisition. ``--admit-existing`` only
# repairs and admits the already-completed run; it cannot invoke makenemo or
# mpirun. Neither mode modifies canonical NEMO source.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R94STGCLS
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R98WWALK
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round94/oracle_stage_closure
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round98/oracle_stage1_w_walk
readonly ROUND99=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round99
readonly RECORD=oracle_stage1_w_walk_kt00000001.bin
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout|--admit-existing|--plant-size) ;;
  *)
    printf 'REFUSE: invalid mode; usage: %s [--run|--preflight-only|--plant-layout|--admit-existing|--plant-size]\n' "$0" >&2
    exit 64
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/sshwzv.f90
readonly ACQUISITION_COMMIT=3e7a15c1e64e036e2e066dcfceaa663f25406f24
readonly BASE=$NEMO_ROOT/src/OCE/DYN/sshwzv.F90
readonly PATCH=$here/sshwzv_round98.patch
readonly STAGE_GATE=$here/../nemo_testcase_l2_gyre_round46_kt2_stage_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round98.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$BASE" "$PATCH" "$STAGE_GATE" "$ADMISSION" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || {
  printf 'REFUSE: source configuration is incomplete: %s\n' "$SOURCE_ROOT" >&2
  exit 64
}
[[ -f "$SOURCE_RUN/$FINAL_RESTART" && -f "$SOURCE_RUN/mesh_mask.nc" ]] || {
  printf 'REFUSE: source run lacks restart or mesh identity file\n' >&2
  exit 64
}
for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' \
  'ln_dynadv_vec *= *T' 'ocean time step.*rn_Dt *= *14400\.0+' \
  'tide_init : tidal components not used \(ln_tide = F\)' \
  'bdy_init : open boundaries not used \(ln_bdy = F\)'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done

dry=$(mktemp -d /tmp/gyre-r98-source.XXXXXXXX)
cp "$BASE" "$dry/sshwzv.F90"
patch -s "$dry/sshwzv.F90" <"$PATCH"
[[ -z "$(diff "$BASE" "$dry/sshwzv.F90" | grep '^<' || true)" ]] || {
  printf 'REFUSE: patch removes or replaces a source-card line\n' >&2
  exit 66
}
check_layout() {
  local source=$1
  [[ "$(grep -c "NEMO_L2_R98W_1" "$source")" -eq 1 ]] &&
  [[ "$(grep -c "WRITE(l98_unit) hdiv, ze3div" "$source")" -eq 1 ]] &&
  [[ "$(grep -c "WRITE(l98_unit) r3t(:,:,Kaa), r3t(:,:,Kbb)" "$source")" -eq 1 ]] &&
  [[ "$(grep -c "WRITE(l98_unit) pww" "$source")" -eq 1 ]]
}
check_layout "$dry/sshwzv.F90" || {
  printf 'REFUSE: patched source does not contain the registered writes\n' >&2
  exit 66
}
readonly NX=36
readonly NY=26
readonly NZ=31
readonly HLS=2
readonly LOCAL_NX=$((NX - 2 * HLS + 2))
readonly LOCAL_NY=$((NY - 2 * HLS + 2))
# hdiv and ze3div have the compiled local one-halo extent; e3t_3d and pww
# are full 3-D fields and the two r3t slots are full 2-D fields.
readonly EXPECTED_SIZE=$((16 + 9 * 4 + (2 * LOCAL_NX * LOCAL_NY * NZ + 2 * NX * NY * NZ + 2 * NX * NY + 1) * 8))
[[ "$EXPECTED_SIZE" -eq 884028 ]] || {
  printf 'REFUSE: byte-layout arithmetic moved from 884028\n' >&2
  exit 66
}

syntax=$(mktemp -d /tmp/gyre-r98-syntax.XXXXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/sshwzv.F90" -o "$syntax/sshwzv.f90"
if ! "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
     -J "$syntax" "$syntax/sshwzv.f90"; then
  printf 'REFUSE: patched sshwzv failed gfortran syntax proof\n' >&2
  exit 66
fi

if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/sshwzv.planted.F90
  cp "$dry/sshwzv.F90" "$planted"
  sed -i '0,/WRITE(l98_unit) hdiv, ze3div/s//WRITE(l98_unit) hdiv/' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
    exit 2
  fi
  printf 'REFUSE: layout plant removed ze3div\n' >&2
  exit 69
fi
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND98_STAGE1_W_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
  exit 0
fi

admit_existing() {
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$COMPILED" \
      "$TARGET_RUN/$RECORD" "$TARGET_RUN/nemo" \
      "$TARGET_RUN/binary.sha256" "$TARGET_RUN/producer_commit.txt" \
      "$TARGET_RUN/source_cfg.sha256" "$TARGET_RUN/toolchain.sha256" \
      "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.time.log" \
      "$TARGET_RUN/$FINAL_RESTART" "$TARGET_RUN/mesh_mask.nc"; do
    [[ -e "$path" ]] || {
      printf 'REFUSE: existing acquisition lacks %s\n' "$path" >&2
      exit 64
    }
  done
  check_layout "$COMPILED" || {
    printf 'REFUSE: compiled R98 writer does not match the registered layout\n' >&2
    exit 66
  }
  grep -Fq 'pww(ji,jj,jk) = pww(ji,jj,jk+1)' "$COMPILED" || {
    printf 'REFUSE: compiled R98 writer lacks the registered recurrence\n' >&2
    exit 66
  }
  local actual_size
  actual_size=$(stat -c %s "$TARGET_RUN/$RECORD")
  if [[ "$MODE" == --plant-size ]]; then
    local planted_size=$((EXPECTED_SIZE + 8))
    if [[ "$actual_size" -eq "$planted_size" ]]; then
      printf 'REFUSE: size plant stayed green\n' >&2
      exit 2
    fi
    printf 'REFUSE: size plant expected %s bytes but record has %s\n' \
      "$planted_size" "$actual_size" >&2
    exit 69
  fi
  [[ "$actual_size" -eq "$EXPECTED_SIZE" ]] || {
    printf 'REFUSE: R98 record size %s != compiled layout %s\n' \
      "$actual_size" "$EXPECTED_SIZE" >&2
    exit 66
  }

  local producer expected_binary recorded_binary built_binary copied_binary
  producer=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  [[ "$producer" == "$ACQUISITION_COMMIT" ]] || {
    printf 'REFUSE: producer commit %s != registered %s\n' \
      "$producer" "$ACQUISITION_COMMIT" >&2
    exit 66
  }
  mkdir -p "$ROUND99"
  if ! (
    cd "$SOURCE_ROOT"
    sha256sum -c "$TARGET_RUN/source_cfg.sha256"
  ) >"$ROUND99/round98_source_cfg_check.log" 2>&1; then
    printf 'REFUSE: source-card manifest no longer verifies\n' >&2
    exit 66
  fi

  manifest_hash() {
    local suffix=$1
    awk -v suffix="$suffix" '
      $2 ~ (suffix "$") { value=$1; count++ }
      END { if (count != 1) exit 1; print value }
    ' "$TARGET_RUN/toolchain.sha256"
  }
  verify_git_tool() {
    local relative=$1
    local suffix=${2:-/$relative}
    local expected actual
    expected=$(manifest_hash "$suffix") || {
      printf 'REFUSE: toolchain manifest lacks unique %s\n' "$relative" >&2
      exit 66
    }
    actual=$(git show "$producer:$relative" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
      printf 'REFUSE: producer-commit tool %s differs from manifest\n' \
        "$relative" >&2
      exit 66
    }
    printf '%s %s\n' "$actual" "$relative"
  }
  verify_live_tool() {
    local path=$1
    local suffix=$2
    local expected actual
    expected=$(manifest_hash "$suffix") || {
      printf 'REFUSE: toolchain manifest lacks unique %s\n' "$suffix" >&2
      exit 66
    }
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
      printf 'REFUSE: live tool %s differs from acquisition manifest\n' \
        "$path" >&2
      exit 66
    }
    printf '%s %s\n' "$actual" "$path"
  }
  {
    verify_live_tool "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
      '/arch/arch-conda-scalarmath.fcm'
    verify_live_tool "$BASE" '/src/OCE/DYN/sshwzv.F90'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round98_stage1_w_walk/sshwzv_round98.patch'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round46_kt2_stage_gate.py' \
      '/nemo_testcase_l2_gyre_round98_stage1_w_walk/../nemo_testcase_l2_gyre_round46_kt2_stage_gate.py'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round21_admission.py' \
      '/nemo_testcase_l2_gyre_round98_stage1_w_walk/../nemo_testcase_l2_gyre_round21_admission.py'
    verify_git_tool \
      'docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round98.md'
    verify_live_tool "$SOURCE_RUN/ocean.output" \
      '/round94/oracle_stage_closure/ocean.output'
  } >"$ROUND99/round98_toolchain_check.log"
  read -r expected_binary recorded_binary <"$TARGET_RUN/binary.sha256"
  [[ "$recorded_binary" == "$BINARY" ]] || {
    printf 'REFUSE: binary manifest names %s, not %s\n' \
      "$recorded_binary" "$BINARY" >&2
    exit 66
  }
  built_binary=$(sha256sum "$BINARY" | awk '{print $1}')
  copied_binary=$(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}')
  [[ "$built_binary" == "$expected_binary" ]] || {
    printf 'REFUSE: built binary digest differs from binary.sha256\n' >&2
    exit 66
  }
  [[ "$copied_binary" == "$expected_binary" ]] || {
    printf 'REFUSE: executed binary copy differs from binary.sha256\n' >&2
    exit 66
  }
  if nm -D "$BINARY" | grep -q '_ZGV'; then
    printf 'REFUSE: vector-math symbol present in recorded binary\n' >&2
    exit 68
  fi
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || {
    printf 'REFUSE: existing NEMO stdout lacks STOP 0\n' >&2
    exit 66
  }

  local digest
  digest=$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$producer" "$RECORD" \
    >"$TARGET_RUN/$RECORD.stamp"
  if ! grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log"; then
    printf 'RUN_RECOVERED_UTC=%s\nRUN_DONE\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$TARGET_RUN/run.user.time.log"
  fi

  if ! "$PY" -c "import pathlib,sys; sys.path.insert(0,'$here/..'); from nemo_testcase_l2_gyre_round46_kt2_stage_gate import read_stage1_w_walk_record; r=read_stage1_w_walk_record(pathlib.Path('$TARGET_RUN/$RECORD')); assert r['header']['Kaa']==3 and r['header']['local_jpi']==34 and r['header']['local_jpj']==24"; then
    printf 'REFUSE: repaired reader rejected the existing R98 record\n' >&2
    exit 66
  fi
  if ! "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
       --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
       --allowed-new "$RECORD" --output "$TARGET_RUN/round98_admission.json"; then
    printf 'REFUSE: consumed-field admission rejected the existing run\n' >&2
    exit 66
  fi
  local plant_status
  if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
       --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
       --allowed-new "$RECORD" --plant-consumed \
       --output "$TARGET_RUN/round98_admission_plant.json" \
       >"$TARGET_RUN/round98_admission_plant.log" 2>&1; then
    printf 'REFUSE: admission consumed-field plant stayed green\n' >&2
    exit 69
  else
    plant_status=$?
  fi
  [[ "$plant_status" -eq 1 ]] || {
    printf 'REFUSE: admission plant exited %s instead of scientific refusal 1\n' \
      "$plant_status" >&2
    exit 69
  }
  (
    cd "$TARGET_RUN"
    sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
      round98_admission.json round98_admission_plant.json \
      round98_admission_plant.log >round98_outputs.sha256
  )
  printf 'ROUND98_STAGE1_W_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing || "$MODE" == --plant-size ]]; then
  admit_existing
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

manifest=$(mktemp -d /tmp/gyre-r98-manifest.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$BASE" "$PATCH" \
  "$STAGE_GATE" "$ADMISSION" "$PREREG" "$SOURCE_RUN/ocean.output" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$BASE" "$TARGET_ROOT/MY_SRC/sshwzv.F90"
patch -s "$TARGET_ROOT/MY_SRC/sshwzv.F90" <"$PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || {
  printf 'REFUSE: rebuilt NEMO binary is not executable\n' >&2
  exit 66
}
check_layout "$COMPILED" || {
  printf 'REFUSE: rebuilt compiled writer does not match registered layout\n' >&2
  exit 66
}
grep -Fq 'pww(ji,jj,jk) = pww(ji,jj,jk+1)' "$COMPILED" || {
  printf 'REFUSE: rebuilt compiled writer lacks registered recurrence\n' >&2
  exit 66
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"; done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  [[ "$(stat -c %s "$RECORD")" -eq "$EXPECTED_SIZE" ]]
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  digest=$(sha256sum "$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$COMMIT" "$RECORD" >"$RECORD.stamp"
)

"$PY" -c "import pathlib,sys; sys.path.insert(0,'$here/..'); from nemo_testcase_l2_gyre_round46_kt2_stage_gate import read_stage1_w_walk_record; r=read_stage1_w_walk_record(pathlib.Path('$TARGET_RUN/$RECORD')); assert r['header']['Kaa']==3"
"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$RECORD" --output "$TARGET_RUN/round98_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$RECORD" --plant-consumed \
     --output "$TARGET_RUN/round98_admission_plant.json" \
     >"$TARGET_RUN/round98_admission_plant.log" 2>&1; then
  printf 'REFUSE: admission consumed-field plant stayed green\n' >&2
  exit 69
fi
(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
    round98_admission.json round98_admission_plant.json \
    round98_admission_plant.log >round98_outputs.sha256
)
printf 'ROUND98_STAGE1_W_READY %s\n' "$TARGET_RUN"
