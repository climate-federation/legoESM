#!/bin/bash
# Build the OpenIFS 49r1 (8f6f722) IFS-interface oracle: radiation_scheme.F90,
# su_radiation_scheme.F90, liquid_effective_radius.F90, ice_effective_radius.F90,
# cloud_overlap_decorr_len.F90, radheatn.F90 (arpifs/phys_radi), satur.F90
# (arpifs/phys_ec), the arpifs/module type modules they use, and
# POSNAM/POSNAME (arpifs/utility), compiled unchanged on a fresh build of the
# 49r1 ecRad core (build49.sh, unchanged, into <build dir>/core). Harness
# files: ifs49_stubs.F90 (MODEL/SPP/GFL/RIP/MPL stand-ins), ifs49_radintg_post.F90
# (radintg.F90 model-grid statements), ifs49_driver.F90, generated interface
# blocks (ifs49_intfb.py, each proven against its definition, planted control),
# and the write-only dump patch of a copy of radiation_scheme.F90.
# Must run inside sbatch (git cat-file of blobs outside the sparse checkout).
#
# usage: ifs49_build.sh <exported 49r1 dir> <OpenIFS checkout> <ecRad 1.7.2 clone> <build dir>
set -euo pipefail
X=$1; OIFS=$2; E172=$3; B=$4; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=8f6f722f732b92fed2c39af763d8fcd1ab15f573
[ "$(git -C "$OIFS" rev-parse HEAD)" = $SHA ] || { echo "OpenIFS HEAD is not $SHA"; exit 1; }
NC=${NETCDF_PREFIX:?set NETCDF_PREFIX}
P49=$X/ifs-source/arpifs/phys_radi
A49=$OIFS/ifs-source/arpifs
rm -rf "$B"; mkdir -p "$B"
bash "$HERE"/build49.sh "$X" "$OIFS" "$E172" "$B/core" > "$B/core_build.log" 2>&1 || { tail -30 "$B/core_build.log"; exit 1; }
tail -1 "$B/core_build.log" | grep -qx BUILD_OK || { echo "core build failed"; exit 1; }
C=$B/core; L=$B/ifs; mkdir -p "$L"/{src,mod,inc,obj}; cd "$L"
: > MANIFEST.txt
add() { cp "$1" src/; echo "$2 $(basename "$1") $(git hash-object "$1")" >> MANIFEST.txt; }
sparse() {  # sparse <path under ifs-source>: copy from the sparse checkout, proven equal to the blob at SHA
  [ "$(git -C "$OIFS" rev-parse "$SHA:ifs-source/$1")" = "$(git hash-object "$OIFS/ifs-source/$1")" ] \
    || { echo "sparse $1 differs from $SHA"; exit 1; }
  add "$OIFS/ifs-source/$1" oifs49_sparse
}
blob() {    # blob <path under ifs-source>: not in the sparse checkout, read from the commit
  local b; b=$(git -C "$OIFS" rev-parse "$SHA:ifs-source/$1"); mkdir -p fetched
  git -C "$OIFS" cat-file blob "$b" > "fetched/$(basename "$1")"
  [ "$(git hash-object "fetched/$(basename "$1")")" = "$b" ] || { echo "blob hash $1"; exit 1; }
  add "fetched/$(basename "$1")" oifs49_blob
}
exported() { [ "$(git -C "$OIFS" rev-parse "$SHA:ifs-source/arpifs/phys_radi/$1")" = "$(git hash-object "$P49/$1")" ] \
    || { echo "exported $1 differs from $SHA"; exit 1; }; add "$P49/$1" oifs49_phys_radi; }

# --- 49r1 sources ------------------------------------------------------------
for m in yomlun yomdyncore yomcst yoethf yomct3 yomrip0 yomphy2 yomphy3 yoerdu yoecld yoecldp yoephli \
         yoephy yoe_aerodiag yoeaeratm yoeaerraddesc yoe_spectral_planck yoerad yoeradiation yoeaerop spp_gen_mod; do
  sparse arpifs/module/$m.F90
done
sparse arpifs/phys_ec/satur.F90
sparse arpifs/setup/sudyncore.F90
blob arpifs/utility/posnam.F90
blob arpifs/utility/posname.F90
for f in radiation_scheme.F90 su_radiation_scheme.F90 liquid_effective_radius.F90 ice_effective_radius.F90 \
         cloud_overlap_decorr_len.F90 radheatn.F90; do exported $f; done
# yoerdi.F90 and surdi.F90 (49r1) are compiled in the core build (bld49 MANIFEST) and reused;
# surdi.F90 is copied to gen/ only to generate and prove its interface block
mkdir -p gen; cp "$P49"/surdi.F90 gen/; cmp -s gen/surdi.F90 "$C"/src/surdi.F90 || { echo "core surdi differs"; exit 1; }
add "$HERE"/ifs49_stubs.F90 harness
add "$HERE"/ifs49_radintg_post.F90 harness
add "$HERE"/ifs49_driver.F90 harness
M49=$OIFS/ifs-source/arpifs/module
python3 -I "$HERE"/ifs49_poison.py src/ifs49_poison.F90 $M49/yoerad.F90:YOERAD:TERAD $M49/yoeaeratm.F90:YOEAERATM:TEAERATM \
  $C/src/yoerdi.F90:YOERDI:TERDI $M49/yoecldp.F90:YOECLDP:TECLDP $M49/yoecld.F90:YOECLD:TECLD $M49/yoephy.F90:YOEPHY:TEPHY \
  $M49/yoephli.F90:YOEPHLI:TEPHLI $M49/yomphy2.F90:YOMPHY2:TPHY2 $M49/yomphy3.F90:YOMPHY3:TPHY3
echo "harness_generated ifs49_poison.F90 $(git hash-object src/ifs49_poison.F90)" >> MANIFEST.txt
python3 -I "$HERE"/ifs49_config_dump.py "$X"/ifs-source/radiation/module/radiation_config.F90 src/ifs49_config_dump.F90
echo "harness_generated ifs49_config_dump.F90 $(git hash-object src/ifs49_config_dump.F90)" >> MANIFEST.txt
# write-only dumps around the ecRad call (env-gated, ECRAD49_STAGE_DUMP)
python3 -I "$HERE"/ifs49_patch_scheme_dump.py src/radiation_scheme.F90
echo "harness_patch radiation_scheme.F90 $(git hash-object src/radiation_scheme.F90)" >> MANIFEST.txt
# function-statement includes from the sparse checkout
for h in fcttim.func.h fcttre.func.h fcttrm.func.h; do sparse arpifs/function/$h; mv src/$h inc/; done
blob arpifs/namelist/namdyncore.nam.h; mv src/namdyncore.nam.h inc/

# --- interface blocks (generated, then proven) -------------------------------
INTF="satur su_radiation_scheme radiation_scheme liquid_effective_radius ice_effective_radius \
      cloud_overlap_decorr_len radheatn surdi posnam posname"
srcof() { [ -f src/$1.F90 ] && echo src/$1.F90 || echo gen/$1.F90; }
for r in $INTF; do python3 -I "$HERE"/ifs49_intfb.py $(srcof $r) inc/$r.intfb.h; echo "harness_intfb $r.intfb.h $(git hash-object inc/$r.intfb.h)" >> MANIFEST.txt; done
cp "$C"/inc/abor1.intfb.h inc/ 2>/dev/null || cp "$E172"/include/abor1.intfb.h inc/

# --- compile: repeated passes until every unit compiles (module order) ------
FF="${OPT:--O1} -ffp-contract=off -fno-fast-math -cpp -DLITTLE_ENDIAN -fno-range-check -ffree-line-length-none -g -fbacktrace -fopenmp"
INC="-Iinc -Imod -Jmod -I$C/inc -I$C/mod -I$NC/include"
todo=$(cd src && ls *.F90)
for pass in $(seq 1 30); do
  left=""
  for f in $todo; do
    if gfortran $FF $INC -c "src/$f" -o "obj/${f%.F90}.o" > "obj/${f%.F90}.log" 2>&1; then :; else left="$left $f"; fi
  done
  [ -z "$left" ] && break
  [ $(echo $left | wc -w) -lt $(echo $todo | wc -w) ] || { echo "stuck at pass $pass:"; for f in $left; do echo "== $f"; grep -m4 -iE "error" -B2 -A2 "obj/${f%.F90}.log"; done; exit 1; }
  todo=$left
done

# --- interface-vs-definition check (as build49.sh), separate module dir -----
mkdir -p chkmod
CHK="$FF -Iinc -Imod -I$C/inc -I$C/mod -I$NC/include -Jchkmod -c -o /dev/null"
mkchk() {  # mkchk <routine> <interface file> <out>
  { echo "MODULE ZZCHK_$1"; echo "CONTAINS"; cat "$(srcof $1)"; echo "END MODULE ZZCHK_$1"
    echo "SUBROUTINE ZZCHK_CALLER_$1"; echo "USE ZZCHK_$1, ONLY : ZZDEF => $1"
    echo "#include \"$2\""; echo "PROCEDURE($1), POINTER :: ZZP"; echo "ZZP => ZZDEF"
    echo "END SUBROUTINE ZZCHK_CALLER_$1"; } > "$3"
}
for r in $INTF; do
  mkchk "$r" "$r.intfb.h" "inc/chk_$r.F90"
  gfortran $CHK "inc/chk_$r.F90" > "inc/chk_$r.log" 2>&1 || { echo "INTERFACE MISMATCH $r"; grep -m5 -i error -B2 "inc/chk_$r.log"; exit 1; }
done
# the pointer check cannot see a permutation of two same-type dummies: compare the
# ordered dummy names and each dummy's declaration text against the definition too
for r in $INTF; do python3 -I "$HERE"/ifs49_intfb.py --verify $(srcof $r) inc/$r.intfb.h | tail -1; done
# planted control for that check: swap two same-type dummies (PHRSW, PHRLW) in the SUBROUTINE line
sed '0,/PHRSW *, *PHRLW/s//PHRLW  , PHRSW/' inc/radheatn.intfb.h > inc/planted_perm.intfb.h
cmp -s inc/planted_perm.intfb.h inc/radheatn.intfb.h && { echo "permutation plant did not change radheatn"; exit 1; }
if python3 -I "$HERE"/ifs49_intfb.py --verify src/radheatn.F90 inc/planted_perm.intfb.h > /dev/null 2>&1; then
  echo "TEXT CHECK IS VACUOUS: planted PHRSW/PHRLW permutation accepted"; exit 1
fi
echo "planted dummy permutation rejected (radheatn PHRSW/PHRLW)"
# non-vacuity: a planted optional->required change must be rejected
sed '0,/, *OPTIONAL/s///' inc/radheatn.intfb.h > inc/planted.intfb.h
cmp -s inc/planted.intfb.h inc/radheatn.intfb.h && { echo "planted control did not change radheatn"; exit 1; }
mkchk radheatn planted.intfb.h inc/chk_planted.F90
if gfortran $CHK inc/chk_planted.F90 > inc/chk_planted.log 2>&1; then
  echo "INTERFACE CHECK IS VACUOUS: planted OPTIONAL removal accepted"; exit 1
fi
echo "planted interface mismatch rejected (radheatn): $(grep -m1 -iE "mismatch|error" inc/chk_planted.log)"
echo "interface blocks checked: $(echo $INTF | wc -w)"

# --- link: core objects without the ecRad standalone driver -----------------
gfortran $FF -o ifs49_driver.x obj/*.o $(ls "$C"/obj/*.o | grep -v ecrad_driver.o) \
  -L$NC/lib -lnetcdff -lnetcdf -Wl,-rpath,$NC/lib
{ echo "openifs $SHA"; gfortran --version | head -1; echo "FFLAGS $FF"
  sha256sum ifs49_driver.x "$C"/ecrad49.x; } > BUILD_INFO.txt
echo "sources: $(wc -l < MANIFEST.txt) layer + $(wc -l < "$C"/MANIFEST.txt) core"
echo BUILD_OK
