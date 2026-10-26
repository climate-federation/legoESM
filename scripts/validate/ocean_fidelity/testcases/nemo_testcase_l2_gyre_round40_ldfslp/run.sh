#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  Round 39 moved legoESM's isoneutral slopes to
# NEMO's placement -- once per step, on the BEFORE state -- and GYRE's kt=2
# tracers cleared.  What it could NOT clear is the slope TRANSCRIPTION: given
# NEMO's own before state at kt=2, legoESM's isoneutral fold reaches
# 1.1728064666279615e-10 where NEMO's ah_wslp2 reaches 3.3898494597440722e-08,
# 17400 of 17400 wet faces unequal, 289x apart, with the argmax at the first
# interior face.  That row is DEBT with a named owner and no instrument.
#
# THE EXISTING RECORD CANNOT SCORE A STATEMENT.  The round-38 kt=2 trazdf
# record carries ah_wslp2, i.e. the PRODUCT of the slopes and the diffusivity,
# and none of the operands ldf_slp actually reads: not prd (=rhd at Nbb), not
# pn2 (=rn2b), not nmln, not hmlp, and none of uslp/vslp/wslpi/wslpj.  So every
# hypothesis about which statement differs -- the mixed-layer index, the N2
# denominator, the double limiter, the Shapiro filter, the face thickness -- is
# unfalsifiable against it.
#
# WHAT IS ADDED.  One WRITE-only block immediately after ldf_slp's own
# lbc_lnk (ldfslp.F90:319), carrying every operand, every per-jk intermediate
# the DESCENDING jk loop destroys, and the four output slopes -- at kt = nit000
# AND kt = nit000+1.  nit000 alone is useless: GYRE's analytic initial T and S
# are horizontally uniform on wet cells, so every slope there is exactly zero
# and the record can only say whether a candidate is also exactly zero.
#
# THE INTERMEDIATES ARE CAPTURED IN PLACE.  zau, zbu, zfi, zdepu, zbw, zci,
# zai, zbi, zck and the rest are SCALARS inside NEMO's own DO_2D bodies and the
# next level overwrites them, so each is copied into a zeroed 3-D buffer inside
# the loop.  No NEMO arithmetic is touched, no NEMO line is removed, and the
# loop is not restructured.  Measured before this file was written: the delta
# removes 0 of NEMO's own lines.  A buffer slot the routine never assigns stays
# a deterministic 0._wp -- never uninitialised memory.
#
# THE STRONGEST CHECK is that adding a reader to ldf_slp must move NOTHING.
# ldf_slp is called once per step from stprk3.F90:174 and its outputs feed
# traldf_iso; a delta that changed a slope would change every tracer record the
# card already writes, and the admission below would refuse it.
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have;
# makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  The line below puts the conda build environment first.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R40LDFSLP
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/round38_oracle_trazdf_kt2
readonly ADMISSION_BASE=$L2/round37_oracle_trazdf_matrix
readonly TARGET_RUN=$L2/round40_oracle_ldfslp
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin
readonly KT2_RECORD=oracle_trazdf_matrix_kt00000002.bin
readonly NEW_RECORD=oracle_ldfslp_kt00000001.bin
readonly NEW_RECORD2=oracle_ldfslp_kt00000002.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly SLP_PATCH=$here/ldfslp_round40.patch
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
[[ -d "$REPO/packages/ocean" && -d "$REPO/src" ]]
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly R29_GATE=$here/../nemo_testcase_l2_gyre_round29_zdf_matrix.py
readonly SHIPPED=$NEMO_ROOT/src/OCE/LDF/ldfslp.F90

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$SLP_PATCH" && -f "$SHIPPED" ]]
[[ -f "$ADMISSION" && -f "$R29_GATE" ]]
[[ -d "$SOURCE_RUN" && -d "$ADMISSION_BASE" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi
# The premises this delta rests on.
if [[ -e "$source_cfg/MY_SRC/ldfslp.F90" ]]; then
  printf 'REFUSE: %s already overrides ldfslp.F90; this delta patches the shipped file\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_TRAZD_1' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s MY_SRC/trazdf.F90 is not the round-35/38 instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'kt <= nit000 + 1' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s does not write the kt=2 tracer record this arm is paired with\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'ln_traldf_iso *= *T' "$SOURCE_RUN/ocean.output"; then
  printf 'REFUSE: the source run does not resolve ln_traldf_iso = T; ldf_slp is not called\n' >&2
  exit 66
fi
dry=$(mktemp -d /tmp/gyre-r40b-dryrun.XXXXXX)
cp "$SHIPPED" "$dry/ldfslp.F90"
if ! patch -s "$dry/ldfslp.F90" <"$SLP_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the round-40 ldfslp delta does not apply cleanly\n' >&2
  rm -rf "$dry"; exit 67
fi
# THE WRITE-ONLY INVARIANT, on the RESULT: additions only against NEMO's own
# shipped ldfslp.F90.
removed=$(diff "$SHIPPED" "$dry/ldfslp.F90" | grep -c '^<' || true)
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: the patched instrument removes %s line(s) of the shipped ldfslp.F90\n' \
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

work_manifest=$(mktemp -d /tmp/gyre-r40b-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SHIPPED" "$SLP_PATCH" \
  >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"

cp "$SHIPPED" "$target_cfg/MY_SRC/ldfslp.F90"
patch "$target_cfg/MY_SRC/ldfslp.F90" <"$SLP_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
# Every other MY_SRC file must be carried across BYTE-IDENTICAL: the tracer
# records this card writes are what the new slope record is paired with.
for name in "$source_cfg"/MY_SRC/*.F90; do
  cmp "$name" "$target_cfg/MY_SRC/$(basename "$name")"
done
sha256sum "$target_cfg/MY_SRC/ldfslp.F90" "$target_cfg/MY_SRC/trazdf.F90" \
  "$SLP_PATCH" >"$work_manifest/round40_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
grep -q 'NEMO_L2_LDFSL_1' "$target_cfg/BLD/ppsrc/nemo/ldfslp.f90" \
  || { printf 'REFUSE: the ldf_slp writer is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q "oracle_ldfslp_kt', kt" "$target_cfg/BLD/ppsrc/nemo/ldfslp.f90" \
  || { printf 'REFUSE: the per-kt file name is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q 'NEMO_L2_TRAZD_1' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the tra_zdf writer is absent from the compiled ppsrc\n' >&2; exit 69; }
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

# TWIN IDENTITY.  RAW byte identity is REPORTED for every record and REQUIRED
# for the two the tracer arms score against.  Round 39 established why the
# blanket raw rule is wrong: NEMO's stream dumps write whole work arrays
# INCLUDING the nn_hls halo, which NEMO neither owns nor initialises, and one
# record carries a zFw slot the executed branch has not defined at the write
# point.  Those bytes differ between two runs of the SAME executable.
# Everything else is decided by the CONSUMED-FIELD admission.
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
for must in "$KT1_RECORD" "$KT2_RECORD"; do
  if ! grep -q "^RAW_IDENTICAL $must\$" "$TARGET_RUN/round40_raw_twin_cmp.log"; then
    printf 'REFUSE: adding the ldf_slp reader moved %s\n' "$must" >&2
    exit 71
  fi
done
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" "$NEW_RECORD2" \
  --output "$TARGET_RUN/round40_source_admission.json" \
  || { printf 'REFUSE: a round-38 consumed field moved\n' >&2; exit 71; }
if python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" "$NEW_RECORD2" --plant-consumed \
     --output "$TARGET_RUN/round40_source_admission_plant.json" \
     >"$TARGET_RUN/round40_source_admission_plant.log" 2>&1; then
  printf 'REFUSE: the source-admission plant did not turn the gate red\n' >&2
  exit 72
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round40_source_admission_plant.json" \
  || { printf 'REFUSE: the source-admission plant never landed\n' >&2; exit 72; }

python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" "$NEW_RECORD2" "$KT2_RECORD" \
  --output "$TARGET_RUN/round40_admission.json" \
  || { printf 'REFUSE: a round-37 consumed field moved\n' >&2; exit 71; }
if python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" "$NEW_RECORD2" "$KT2_RECORD" --plant-consumed \
     --output "$TARGET_RUN/round40_admission_plant.json" \
     >"$TARGET_RUN/round40_admission_plant.log" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round40_admission_plant.json" \
  || { printf 'REFUSE: the admission plant never landed\n' >&2; exit 72; }

# THE RECORDS THIS ACQUISITION EXISTS FOR.  Both must parse to EOF from their
# own headers.  And the kt=2 slopes must NOT be identically zero: if they are,
# the acquisition has not closed the discrimination gap, and that is a FINDING
# to print, not a silent pass.
for record in "$NEW_RECORD" "$NEW_RECORD2"; do
  [[ -s "$TARGET_RUN/$record" ]] \
    || { printf 'REFUSE: the run wrote no %s\n' "$record" >&2; exit 71; }
done
if cmp -s "$TARGET_RUN/$NEW_RECORD" "$TARGET_RUN/$NEW_RECORD2"; then
  printf 'REFUSE: the kt=2 slope record is byte-identical to the kt=1 record\n' >&2
  exit 71
fi
python - "$TARGET_RUN/$NEW_RECORD" "$TARGET_RUN/$NEW_RECORD2" \
  >"$TARGET_RUN/round40_ldfslp_header.txt" <<'PYHDR'
import struct, sys
import numpy as np

def read(path, expect_kt):
    with open(path, "rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=14i", handle.read(56))
        assert magic == "NEMO_L2_LDFSL_1", f"{path}: bad magic {magic!r}"
        assert header[0] == 1 and header[1] == expect_kt, f"{path}: bad header {header}"
        assert header[13] == 64, f"{path}: wp is not 64-bit"
        arrays = {}
        while True:
            raw = handle.read(16)
            if not raw:
                break
            name = raw.decode("ascii").rstrip()
            rank, n1, n2, n3 = struct.unpack("=4i", handle.read(16))
            count = 1 if rank == 0 else (n1 if rank == 1 else n1 * n2 * (n3 if rank == 3 else 1))
            block = np.frombuffer(handle.read(8 * count), dtype=np.float64)
            assert block.size == count, f"{path}: {name} truncated"
            arrays[name] = block
    return header, arrays

h1, a1 = read(sys.argv[1], 1)
h2, a2 = read(sys.argv[2], 2)
print("HEADER_KT1", h1)
print("HEADER_KT2", h2)
print("ARRAYS", len(a1), len(a2))
required = {"prd", "pn2", "nmln", "hmlp", "zhmlpt", "r1_hmlu", "r1_hmlv",
            "r1_hmlw", "zdzr", "zau", "zbu", "zai", "zbi", "zck", "zfk",
            "uslp", "vslp", "wslpi", "wslpj", "e3u_Kmm", "e3w_Kmm"}
missing = sorted(required - set(a1))
print("MISSING", missing)
for name in ("uslp", "vslp", "wslpi", "wslpj"):
    print(f"KT1_ABSMAX {name} {float(np.abs(a1[name]).max()):.17g}")
    print(f"KT2_ABSMAX {name} {float(np.abs(a2[name]).max()):.17g}")
nonzero = float(np.abs(a2["wslpi"]).max()) > 0.0
print("KT2_WSLPI_NONZERO", nonzero)
if not nonzero:
    print("FINDING the kt=2 wslpi is ALSO identically zero; this acquisition "
          "did not close the discrimination gap and a later step is needed")
print("HEADER_VERDICT", "PASS" if not missing else "FAIL")
sys.exit(0 if not missing else 1)
PYHDR
tail -4 "$TARGET_RUN/round40_ldfslp_header.txt"

python "$R29_GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round40_r29_regression.json"

(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin round40_admission.json round40_admission_plant.json \
    round40_source_admission.json round40_source_admission_plant.json \
    round40_r29_regression.json round40_ldfslp_header.txt \
    round40_raw_twin_cmp.log \
    "$FINAL_RESTART" mesh_mask.nc >round40_outputs.sha256
)
printf 'ROUND40_GYRE_LDFSLP_ORACLE_READY %s\n' "$TARGET_RUN"
