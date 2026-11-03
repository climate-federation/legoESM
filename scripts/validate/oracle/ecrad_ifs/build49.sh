#!/bin/bash
# Build the ecRad oracle bundled with OpenIFS 49r1 (8f6f722): the
# ifs-source/radiation/module sources and the RRTMG gas optics in
# arpifs/phys_radi + arpifs/module, compiled unchanged, driven by the
# ecRad 1.7.2 standalone driver (driver/ecrad_driver*.F90) and its
# ifsaux/drhook stand-ins (parkind1, yomlun, yomcst, abor1, dummy Dr Hook).
#
# usage: build49.sh <exported 49r1 source dir> <OpenIFS checkout> <ecRad 1.7.2 clone> <build dir>
#   exported dir = output of setup.sbatch (git cat-file of every blob under
#   ifs-source/radiation and ifs-source/arpifs/phys_radi at 8f6f722, hashes checked)
set -euo pipefail
X=$1; OIFS=$2; E172=$3; B=$4; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=8f6f722f732b92fed2c39af763d8fcd1ab15f573
[ "$(git -C "$OIFS" rev-parse HEAD)" = $SHA ] || { echo "OpenIFS HEAD is not $SHA"; exit 1; }
NC=${NETCDF_PREFIX:?set NETCDF_PREFIX (netcdf-fortran install with include/netcdf.mod)}
R49=$X/ifs-source/radiation/module
P49=$X/ifs-source/arpifs/phys_radi
M49=$OIFS/ifs-source/arpifs/module
rm -rf "$B"; mkdir -p "$B"/{src,mod,inc,obj}; cd "$B"

# verify every arpifs/module file we take from the sparse checkout equals the blob at SHA
blob_ok() { [ "$(git -C "$OIFS" rev-parse "$SHA:ifs-source/arpifs/module/$1")" = "$(git hash-object "$M49/$1")" ]; }

# --- source set -------------------------------------------------------------
: > MANIFEST.txt
add() { cp "$1" src/; echo "$2 $(basename "$1") $(git hash-object "$1")" >> MANIFEST.txt; }
# serial IFS infrastructure stand-ins from ecRad 1.7.2 (no physics): Dr Hook dummy,
# serial MPL, abor1, unit numbers, MPI rank, constants (yomcst_ecrad values equal 49r1
# yomcst.F90 for every constant the compiled RRTMG code reads: RG, RMD, RMO3, RMCO2,
# RMCH4, RMN2O, RMNO2, RMCFC11, RMCFC12, RMHCFC22, RMCCL4)
for f in "$E172"/ifsaux/*.F90 "$E172"/drhook/*.F90 "$E172"/ifsrrtm/yom_ygfl.F90 "$E172"/ifsrrtm/yomdimv.F90; do
  case $(basename "$f") in parkind1.F90|mpl_module.F90) ;; *) add "$f" ecrad172_aux;; esac
done
# the real 49r1 kind module, IFS random-number generator and unit-number modules
add $X/ifs-source/contrib/fiat/src/parkind/parkind1.F90 oifs49_fiat
add $X/ifs-source/contrib/fiat/src/fiat/util/ec_lun.F90 oifs49_fiat
add $X/ifs-source/algor/module/random_numbers_mix.F90 oifs49_algor
blob_ok_aux() { [ "$(git -C "$OIFS" rev-parse "$SHA:ifs-source/ifsaux/module/$1")" = "$(git hash-object "$OIFS/ifs-source/ifsaux/module/$1")" ]; }
blob_ok_aux yomlun_ifsaux.F90 && add "$OIFS"/ifs-source/ifsaux/module/yomlun_ifsaux.F90 oifs49_ifsaux
# module-name shims: 49r1 RRTMG uses YOMLUN/YOMMP0/YOMCST, ecRad 1.7.2 stand-ins are *_ECRAD/*_IFSAUX
add "$HERE"/shims49.F90 harness
add "$HERE"/stubs49.F90 harness
add $X/ifs-source/contrib/fiat/src/fiat/util/ec_parkind.F90 oifs49_fiat
for f in "$R49"/*; do add "$f" oifs49_radiation; done
for f in "$E172"/ifsrrtm/*.F90; do
  b=$(basename "$f"); case $b in yom_ygfl.F90|yomdimv.F90) continue;; esac
  if [ -f "$P49/$b" ]; then add "$P49/$b" oifs49_phys_radi
  elif [ -f "$M49/$b" ]; then blob_ok "$b" || { echo "sparse $b differs from $SHA"; exit 1; }; add "$M49/$b" oifs49_module
  else echo "no 49r1 copy of $b"; exit 1; fi
done
for f in ecrad_driver.F90 ecrad_driver_config.F90 ecrad_driver_read_input.F90 print_matrix_mod.F90; do
  add "$E172/driver/$f" ecrad172_driver
done
# 1.7.2 driver vs 49r1 API: the only mismatch is save_general_cloud_optics (absent
# at 49r1; general cloud optics is off in the IFS-RRTMG configuration). Drop its
# import and its one call; refuse to build if the driver text is not as expected.
grep -c "save_general_cloud_optics" src/ecrad_driver.F90 | grep -qx 2 || { echo "driver not as expected"; exit 1; }
sed -i -e '/use radiation_general_cloud_optics, only : save_general_cloud_optics/d' \
       -e '/call save_general_cloud_optics(config/d' src/ecrad_driver.F90
grep -q "save_general_cloud_optics" src/ecrad_driver.F90 && { echo "driver patch failed"; exit 1; }
echo "harness_patch ecrad_driver.F90 $(git hash-object src/ecrad_driver.F90)" >> MANIFEST.txt

# harness instrumentation: dump the McICA random-number stream (A16)
python3 -I "$HERE"/patch_rng_dump.py src/random_numbers_mix.F90
echo "harness_patch random_numbers_mix.F90 $(git hash-object src/random_numbers_mix.F90)" >> MANIFEST.txt

# --- interface blocks ------------------------------------------------------
# ecRad's generator (make_intfbl.1.pl) needs a perl module that is not
# shipped, so take the 1.7.2 generated blocks and PROVE each one matches the
# 49r1 definition: compile definition + interface in one file, where gfortran
# rejects any global-procedure interface mismatch.
: > intfb_checked.txt
for h in $(grep -ho '^#include *"[^"]*"' src/*.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  if [ -f "src/$h" ]; then continue
  elif [ -f "$E172/include/$h" ]; then
    cp "$E172/include/$h" inc/; echo "ecrad172_intfb $h $(git hash-object "inc/$h")" >> MANIFEST.txt
    r=${h%.intfb.h}
    if [ -f "src/$r.F90" ]; then
      { cat "src/$r.F90"; printf 'SUBROUTINE ZZ_INTFB_CHECK\n#include "%s"\nEND SUBROUTINE ZZ_INTFB_CHECK\n' "$h"; } > "inc/chk_$r.F90"
      echo "$r" >> intfb_checked.txt
    fi
  else echo "no interface block for $h"; exit 1; fi
done
# --- compile: repeated passes until every unit compiles (module order) ----
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -DLITTLE_ENDIAN -fno-range-check -fallow-invalid-boz -ffree-line-length-none -g -fbacktrace -fopenmp -DNC_NETCDF4"
INC="-Isrc -Iinc -Imod -Jmod -I$NC/include"
todo=$(cd src && ls *.F90)
for pass in $(seq 1 40); do
  left=""
  for f in $todo; do
    if gfortran $FF $INC -c "src/$f" -o "obj/${f%.F90}.o" > "obj/${f%.F90}.log" 2>&1; then :; else left="$left $f"; fi
  done
  [ -z "$left" ] && break
  [ $(echo $left | wc -w) -lt $(echo $todo | wc -w) ] || { echo "stuck at pass $pass:"; for f in $left; do echo "== $f"; grep -m3 -iE "error" -A1 "obj/${f%.F90}.log"; done; exit 1; }
  todo=$left
done
# interface-vs-definition check (modules exist now)
for r in $(cat intfb_checked.txt); do
  gfortran $FF $INC -fsyntax-only "inc/chk_$r.F90" > "inc/chk_$r.log" 2>&1 || { echo "INTERFACE MISMATCH $r"; grep -m5 -i error "inc/chk_$r.log"; exit 1; }
done
echo "interface blocks checked: $(wc -l < intfb_checked.txt)"
gfortran $FF -o ecrad49.x obj/*.o -L$NC/lib -lnetcdff -lnetcdf -Wl,-rpath,$NC/lib
gfortran --version | head -1
echo "sources: $(wc -l < MANIFEST.txt)"
echo BUILD_OK
