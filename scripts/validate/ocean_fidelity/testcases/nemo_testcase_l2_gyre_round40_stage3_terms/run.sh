#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  GYRE's kt=2 first-over-bar is now only u and v,
# and round 33 localised the owner to the STAGE-3 MOMENTUM RHS: the dyn_zdf
# entry residual at kt=1 stage 3 is exactly rDt times the stage-3 RHS residual,
# and the implicit solve and the barotropic correction are bit-exact given
# NEMO's own inputs.  Round 30 then bracketed that RHS as a TOTAL -- the
# pre-dyn_ldf frame, NEMO_L2_RKPLD_1 -- at 2.06e-16 on 17383 of 17400 U faces.
#
# THAT TOTAL CANNOT NAME AN OPERATOR.  Stage 3 accumulates dyn_hpg, dyn_vor and
# dyn_adv into one Krhs slot (stprk3_stg.F90:324, :327, :331), and the only
# per-operator record this campaign has is oracle_rkstage2_terms, whose header
# is locked to kstg == 2.  Every existing momentum-operand dump in the source
# card's MY_SRC is gated the same way.  So the stage-3 split has never been
# scored, and no statement inside it can be named without this record.
#
# WHAT IS ADDED.  Four snapshots of uu/vv(:,:,:,Krhs) at kstg == 3 -- on entry,
# after dyn_hpg, after dyn_vor, after dyn_adv -- plus, in the SAME file, the
# DERIVED stage geometry those operators read and which nothing else in this
# campaign carries: rhd, ww, r3f, e3f_0vor, fe3mask, the Kmm face thicknesses
# and their reference halves, the masks and the horizontal metric.  r3f and
# e3f_0vor matter because key_qco IS defined on this build
# (cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm), so vor_ene divides its potential
# vorticity by e3f_0vor*(1+r3f*fe3mask) at dynvor.F90:490 and the nn_e3f_typ
# SELECT at :493-516 is dead code.
#
# `before_u`/`before_v` are a diagnostic of the INCOMING Krhs slot ONLY.  Under
# key_RK3 dyn_hpg OVERWRITES Krhs (dynhpg.F90:359-363, :383-387), so they are
# not an operand of anything at this stage and must not be read as one.
#
# THE STRONGEST CHECK THIS ACQUISITION HAS is that every record the source card
# already wrote must come back unmoved.  Adding a stage-3 writer must not move
# stage 1, stage 2, or any tracer record; if it does, the delta is not
# WRITE-only and the run is refused.
#
# THE INVARIANT IS CHECKED ON THE FINAL FILE, NOT ON THE PATCH: the built
# MY_SRC/stprk3_stg.F90 must differ from the source card's by ADDITIONS ONLY,
# zero removed lines.  Measured before this file was written: 0.
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have;
# makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  The line below puts the conda build environment first,
# which is the PATH the round-33, round-35, round-37 and round-38 builds used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R40STG3TRM
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
# The direct parent run: its namelists are copied and its records are the twin
# this one is compared against, field by field, through the admission's reader.
readonly SOURCE_RUN=$L2/round38_oracle_trazdf_kt2
# The consumed-field admission ALSO runs against round 37, the run every
# round-35/37/38 tracer arm scores against.
readonly ADMISSION_BASE=$L2/round37_oracle_trazdf_matrix
readonly TARGET_RUN=$L2/round40_oracle_stage3_terms
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin
readonly NEW_RECORD=oracle_rkstage3_terms_kt00000001.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly STG_PATCH=$here/stprk3_stg_round40_stage3_terms.patch
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
[[ -d "$REPO/packages/ocean" && -d "$REPO/src" ]]
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly R29_GATE=$here/../nemo_testcase_l2_gyre_round29_zdf_matrix.py
readonly SOURCE_STG=$NEMO_ROOT/cfgs/$SOURCE_CFG/MY_SRC/stprk3_stg.F90

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$STG_PATCH" && -f "$SOURCE_STG" ]]
[[ -f "$ADMISSION" && -f "$R29_GATE" ]]
[[ -d "$SOURCE_RUN" && -d "$ADMISSION_BASE" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi
# The premises this delta rests on.
if ! grep -q 'NEMO_L2_RKTRM_1' "$SOURCE_STG"; then
  printf 'REFUSE: %s MY_SRC/stprk3_stg.F90 is not the stage-2 term instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_RKPLD_1' "$SOURCE_STG"; then
  printf 'REFUSE: %s MY_SRC/stprk3_stg.F90 lacks the round-29 pre-LDF writer\n' "$SOURCE_CFG" >&2
  exit 66
fi
if grep -q 'NEMO_L2_RKTS3_1' "$SOURCE_STG"; then
  printf 'REFUSE: %s already writes the stage-3 split; nothing to acquire\n' "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'key_qco' "$source_cfg/cpp_${SOURCE_CFG}.fcm"; then
  printf 'REFUSE: key_qco is not defined; e3f_vor is not the divisor this record assumes\n' >&2
  exit 66
fi
dry=$(mktemp -d /tmp/gyre-r40a-dryrun.XXXXXX)
cp "$SOURCE_STG" "$dry/stprk3_stg.F90"
if ! patch -s "$dry/stprk3_stg.F90" <"$STG_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the round-40 stage-3 delta does not apply cleanly\n' >&2
  rm -rf "$dry"; exit 67
fi
# THE WRITE-ONLY INVARIANT, on the RESULT: additions only against the source
# card's own stprk3_stg.F90, which itself is additions-only against NEMO's.
removed=$(diff "$SOURCE_STG" "$dry/stprk3_stg.F90" | grep -c '^<' || true)
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: the patched instrument removes %s line(s) of the source stprk3_stg.F90\n' \
    "$removed" >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

work_manifest=$(mktemp -d /tmp/gyre-r40a-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SOURCE_STG" "$STG_PATCH" \
  >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
# The tracer instrument must be carried across BYTE-IDENTICAL: every round-35
# and round-38 tracer arm is scored against a record this card also writes.
cmp "$source_cfg/MY_SRC/trazdf.F90" "$target_cfg/MY_SRC/trazdf.F90"

patch "$target_cfg/MY_SRC/stprk3_stg.F90" <"$STG_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
cmp "$source_cfg/MY_SRC/trazdf.F90" "$target_cfg/MY_SRC/trazdf.F90"
sha256sum "$target_cfg/MY_SRC/stprk3_stg.F90" "$target_cfg/MY_SRC/trazdf.F90" \
  "$STG_PATCH" >"$work_manifest/round40_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
grep -q 'NEMO_L2_RKTS3_1' "$target_cfg/BLD/ppsrc/nemo/stprk3_stg.f90" \
  || { printf 'REFUSE: the stage-3 term writer is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q 'NEMO_L2_RKTRM_1' "$target_cfg/BLD/ppsrc/nemo/stprk3_stg.f90" \
  || { printf 'REFUSE: the stage-2 term writer is absent from the compiled ppsrc\n' >&2; exit 69; }
grep -q 'NEMO_L2_TRAZD_1' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the tra_zdf writer is absent from the compiled ppsrc\n' >&2; exit 69; }
grep -q 'e3f_0vor' "$target_cfg/BLD/ppsrc/nemo/stprk3_stg.f90" \
  || { printf 'REFUSE: the e3f_vor materialisation is absent from the compiled ppsrc\n' >&2; exit 69; }
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s\n' "$target_binary" >&2
  exit 65
fi
sha256sum "$target_binary" >"$work_manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  for name in $PREPARED; do
    cmp "$ADMISSION_BASE/$name" "$name"
  done
  sha256sum $PREPARED >prepared_files.sha256
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log
)

# TWIN IDENTITY.  RAW byte identity against the SOURCE run is REPORTED for
# every record and REQUIRED for one -- the kt = nit000 trazdf record every
# round-35/37/38 arm scores against.  Round 39 established why the blanket raw
# rule is wrong: NEMO's stream dumps write whole work arrays INCLUDING the
# nn_hls halo, which NEMO neither owns nor initialises, and one record carries
# a zFw slot the executed branch has not defined at the write point.  Those
# bytes are uninitialised memory and differ between two runs of the SAME
# executable, so raw identity is not attainable and a rule that demands it
# refuses runs that changed no model state.  Everything except the one record
# below is decided by the CONSUMED-FIELD admission.
: >"$TARGET_RUN/round40_raw_twin_cmp.log"
for source_record in "$SOURCE_RUN"/oracle_*.bin; do
  record=${source_record##*/}
  if [[ ! -e "$TARGET_RUN/$record" ]]; then
    printf 'REFUSE: the instrumented run did not write %s\n' "$record" >&2
    exit 71
  fi
  if cmp -s "$source_record" "$TARGET_RUN/$record"; then
    printf 'RAW_IDENTICAL %s\n' "$record"
  else
    printf 'RAW_DIFFERS   %s (falls through to the consumed-field admission)\n' \
      "$record"
  fi
done | tee -a "$TARGET_RUN/round40_raw_twin_cmp.log"
if ! grep -q "^RAW_IDENTICAL $KT1_RECORD\$" \
     "$TARGET_RUN/round40_raw_twin_cmp.log"; then
  printf 'REFUSE: adding the stage-3 writer moved the kt=1 trazdf record\n' >&2
  exit 71
fi
# EVERY OTHER record must come back with identical CONSUMED FIELDS, against
# BOTH the direct parent and the run the tracer arms are scored on.
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" \
  --output "$TARGET_RUN/round40_source_admission.json" \
  || { printf 'REFUSE: a round-38 consumed field moved\n' >&2; exit 71; }
if python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     --output "$TARGET_RUN/round40_source_admission_plant.json" \
     >"$TARGET_RUN/round40_source_admission_plant.log" 2>&1; then
  printf 'REFUSE: the source-admission plant did not turn the gate red\n' >&2
  exit 72
fi
# AND IT MUST HAVE LANDED.  The gate also exits non-zero when the plant was
# never applied, so "exited non-zero" alone is satisfied by a control that
# proved nothing.
grep -q '"plant_applied": true' "$TARGET_RUN/round40_source_admission_plant.json" \
  || { printf 'REFUSE: the source-admission plant never landed\n' >&2; exit 72; }

python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" oracle_trazdf_matrix_kt00000002.bin \
  --output "$TARGET_RUN/round40_admission.json" \
  || { printf 'REFUSE: a round-37 consumed field moved\n' >&2; exit 71; }
if python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" oracle_trazdf_matrix_kt00000002.bin \
     --plant-consumed \
     --output "$TARGET_RUN/round40_admission_plant.json" \
     >"$TARGET_RUN/round40_admission_plant.log" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round40_admission_plant.json" \
  || { printf 'REFUSE: the admission plant never landed\n' >&2; exit 72; }

# The record this acquisition exists for must exist, must parse to EOF from its
# own header, and must NOT be a copy of the stage-2 one.
[[ -s "$TARGET_RUN/$NEW_RECORD" ]] \
  || { printf 'REFUSE: the run wrote no stage-3 term record\n' >&2; exit 71; }
if cmp -s "$TARGET_RUN/oracle_rkstage2_terms_kt00000001.bin" \
          "$TARGET_RUN/$NEW_RECORD"; then
  printf 'REFUSE: the stage-3 record is byte-identical to the stage-2 record\n' >&2
  exit 71
fi
python - "$TARGET_RUN/$NEW_RECORD" >"$TARGET_RUN/round40_stage3_terms_header.txt" <<'PYHDR'
import struct, sys
import numpy as np
path = sys.argv[1]
with open(path, "rb") as handle:
    magic = handle.read(16).decode("ascii").rstrip()
    header = struct.unpack("=16i", handle.read(64))
    assert magic == "NEMO_L2_RKTS3_1", f"bad magic {magic!r}"
    version, kt, kstg, kbb, kmm, krhs, kaa, nx, ny, nz, nzm1 = header[:11]
    print("HEADER", header)
    assert (version, kt, kstg) == (1, 1, 3), "wrong version/kt/kstg"
    assert header[15] == 64, "wp is not 64-bit"
    names, seen = [], 0
    while True:
        raw = handle.read(16)
        if not raw:
            break
        name = raw.decode("ascii").rstrip()
        rank, n1, n2, n3 = struct.unpack("=4i", handle.read(16))
        count = 1 if rank == 0 else (n1 if rank == 1 else n1 * n2 * (n3 if rank == 3 else 1))
        block = np.frombuffer(handle.read(8 * count), dtype=np.float64)
        assert block.size == count, f"{name}: truncated payload"
        names.append(name); seen += 1
print("ARRAYS", seen)
print(" ".join(names))
required = {"before_u", "before_v", "after_hpg_u", "after_hpg_v", "after_vor_u",
            "after_vor_v", "after_adv_u", "after_adv_v", "rhd", "ww", "r3f",
            "e3f_0vor", "e3f_vor_Kmm", "fe3mask", "e3u_Kmm", "e3v_Kmm"}
missing = sorted(required - set(names))
print("MISSING", missing)
print("HEADER_VERDICT", "PASS" if not missing else "FAIL")
sys.exit(0 if not missing else 1)
PYHDR
tail -3 "$TARGET_RUN/round40_stage3_terms_header.txt"

# The round-29 momentum-matrix gate: unchanged by this delta, so every row it
# scored must still be AT-BAR.
python "$R29_GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round40_r29_regression.json"

(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin round40_admission.json round40_admission_plant.json \
    round40_source_admission.json round40_source_admission_plant.json \
    round40_r29_regression.json round40_stage3_terms_header.txt \
    round40_raw_twin_cmp.log \
    "$FINAL_RESTART" mesh_mask.nc >round40_outputs.sha256
)
printf 'ROUND40_GYRE_STAGE3_TERMS_ORACLE_READY %s\n' "$TARGET_RUN"
