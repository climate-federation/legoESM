#!/bin/bash
# Run the 49r1 IFS-interface oracle in its AEROSOL ARM (NAERMACC=1, ifs49_harness_aer.nam)
# on the meridian fixture columns: base twice (determinism), the 20200201 date (vote B3a)
# and the MMR_SCALE=100 control (vote B6a). Serial, one thread; McICA stream dumped.
# usage: ifs49_run_aer.sh <build dir> <ecrad_ifs49.npz> <data dir (ifsdata with the 49r1 optics name)>
#                         <CAMS 2D climatology file> <l137.html> <run dir>
set -euo pipefail
B=$(readlink -f "$1"); NPZ=$(readlink -f "$2"); DD=$(readlink -f "$3"); CLIM=$(readlink -f "$4")
HTML=$(readlink -f "$5"); R=$(readlink -m "$6")
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}
L137_URL=https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
L137_SHA=cc5e1fd86b61e4c54bd536b4de153502cde4bd012486295e34878ca3e3cbc9f0
EXE=$B/ifs/ifs49_driver.x
rm -rf "$R"; mkdir -p "$R"/inputs "$R"/data
"$PY" -I "$HERE"/ifs49_make_inputs.py "$NPZ" "$R"/inputs
"$PY" -I "$HERE"/ifs49_l137_ab.py "$HTML" $L137_SHA "$R"/inputs
echo "L137 source $L137_URL sha256 $L137_SHA" | tee "$R"/l137_source.txt
ln -s "$DD" "$R"/data/ifsdata
export DATA=$R/data OMP_NUM_THREADS=1
( cd "$DD" && sha256sum $(ls | grep -vE "EXTRACTED|SHA256SUMS") ) > "$R"/data_sha256.txt
sha256sum "$CLIM" >> "$R"/data_sha256.txt
one() {  # one <name> <sed expression on the namelist>
  mkdir -p "$R/$1"
  sed -E -e "s#'CAMS_CLIM'#'$CLIM'#" -e "$2" "$HERE"/ifs49_harness_aer.nam > "$R/$1.nam"
  grep -q "$CLIM" "$R/$1.nam" || { echo "climatology path not applied"; exit 1; }
  ( cd "$R/$1" && ECRAD49_STAGE_DUMP=$R/$1 ECRAD49_RNG_DUMP=$R/$1.rng.txt "$EXE" run "$R"/inputs "$R/$1.nam" > "$R/$1.log" 2>&1 ) \
    || { tail -30 "$R/$1.log"; exit 1; }
  grep -aq IFS49_DRIVER_OK "$R/$1.log" || { tail -30 "$R/$1.log"; exit 1; }
  echo "RUN_OK $1 ($(ls "$R/$1" | wc -l) dumps)"
}
one aer_a 's/^$//'
one aer_b 's/^$//'
one d0201 's/^(NINDAT_H *= *)20200115,/\120200201,/'
grep -q "NINDAT_H   = 20200201," "$R/d0201.nam" || { echo "date override not applied"; exit 1; }
one x100 's/^(MMR_SCALE *= *)1.0,/\1100.0,/'
grep -q "MMR_SCALE = 100.0," "$R/x100.nam" || { echo "scale override not applied"; exit 1; }
# determinism: every dump of the two base runs identical, byte for byte
for f in "$R"/aer_a/*.txt; do cmp -s "$f" "$R/aer_b/$(basename "$f")" || { echo "NONDETERMINISTIC $(basename "$f")"; exit 1; }; done
cmp -s "$R"/aer_a.rng.txt "$R"/aer_b.rng.txt || { echo "NONDETERMINISTIC rng"; exit 1; }
echo "DETERMINISTIC aer_a == aer_b ($(ls "$R"/aer_a | wc -l) dumps + rng)"
