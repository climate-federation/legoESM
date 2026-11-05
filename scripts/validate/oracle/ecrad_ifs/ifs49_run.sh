#!/bin/bash
# Run the 49r1 IFS-interface oracle (ifs49_build.sh output) on the meridian
# fixture columns: baseline twice (determinism), the unused-input controls
# (nan_unused, big_unused, nan_thf), the ecRad-namelist RAD_CONFIG print, then
# ifs49_compare.py. Serial, one thread.
# usage: ifs49_run.sh <build dir> <ecrad_ifs49.npz> <data dir (A5 ifsdata files)> <run dir>
set -euo pipefail
B=$(readlink -f "$1"); NPZ=$(readlink -f "$2"); DD=$(readlink -f "$3"); R=$(readlink -m "$4")
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}
EXE=$B/ifs/ifs49_driver.x
rm -rf "$R"; mkdir -p "$R"/inputs "$R"/data
"$PY" -I "$HERE"/ifs49_make_inputs.py "$NPZ" "$R"/inputs
ln -s "$DD" "$R"/data/ifsdata          # SU and RRTM read $DATA/ifsdata (su_radiation_scheme.F90:155, suecrad.F90:2112)
export DATA=$R/data OMP_NUM_THREADS=1
( cd "$DD" && sha256sum $(ls | grep -vE "EXTRACTED|SHA256SUMS") ) > "$R"/data_sha256.txt
one() {  # one <name> <control>
  mkdir -p "$R/$1"; sed -E "s/^(CONTROL *= *)'none'/\1'$2'/" "$HERE"/ifs49_harness.nam > "$R/$1.nam"
  grep -q "CONTROL *= *'$2'" "$R/$1.nam" || { echo "control $2 not applied"; exit 1; }
  ( cd "$R/$1" && ECRAD49_STAGE_DUMP=$R/$1 "$EXE" run "$R"/inputs "$R/$1.nam" > "$R/$1.log" 2>&1 ) \
    || { tail -30 "$R/$1.log"; exit 1; }
  grep -q IFS49_DRIVER_OK "$R/$1.log" || { tail -30 "$R/$1.log"; exit 1; }
  echo "RUN_OK $1 ($(ls "$R/$1" | wc -l) dumps)"
}
one base_a none
one base_b none
one nan_unused nan_unused
one big_unused big_unused
one nan_thf nan_thf
sed "s#\"DATA_DIR\"#\"$DD\"#" "$HERE"/config49r1_rrtmg.nam > "$R"/refconfig.nam
export IFS49_REFNAM=$R/refconfig.nam
one refcfg refcfg
( cd "$R" && "$EXE" refconfig "$R"/refconfig.nam > "$R"/refconfig.log 2>&1 ) || { tail -20 "$R"/refconfig.log; exit 1; }
"$PY" -I "$HERE"/ifs49_compare.py "$R"
