#!/bin/bash
# Extract the OpenIFS convection sources this harness compiles from the
# pinned OpenIFS commit (the one the faithful port cites) into $1/src and
# write $1/MD5SUMS.  usage: extract.sh <build dir> [openifs checkout]
set -euo pipefail
B=$1
O=${2:-/work/bd1083/b309178/diffESM/legoesm_pg/oracle_openifs}
SHA=8f6f722
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
FULL=$(git -C "$O" rev-parse "$SHA")
mkdir -p "$B/src"
FILES="
contrib/fiat/src/parkind/parkind1.F90
ifsaux/module/yomlun_ifsaux.F90
arpifs/module/yomlun.F90
arpifs/module/yomdyncore.F90
arpifs/module/yomcst.F90
arpifs/module/yoethf.F90
arpifs/module/yoephli.F90
arpifs/module/yoecumf.F90
arpifs/module/yoecldp.F90
arpifs/module/yoephy.F90
arpifs/module/parphy.F90
arpifs/module/yomsta.F90
arpifs/module/yomdimv.F90
arpifs/module/yomphy2.F90
arpifs/module/yophnc.F90
arpifs/module/yoecumf2.F90
arpifs/module/yomcumfs.F90
arpifs/module/yompertpar.F90
arpifs/module/spp_gen_mod.F90
arpifs/function/fcttre.func.h
arpifs/function/fcttrm.func.h
arpifs/function/cuadjtq.func.h
arpifs/namelist/namcumf.nam.h
arpifs/namelist/namcldp.nam.h
arpifs/namelist/naephli.nam.h
arpifs/phys_ec/suphec.F90
arpifs/phys_ec/suphli.F90
arpifs/phys_ec/sucumf.F90
arpifs/phys_ec/sucldp.F90
arpifs/utility/fcgeneralized_gamma.F90
"
for r in cumastrn cuinin cubasen cuascn cuentr cubasmcn cudlfsn cuddrafn \
         cuflxn cudtdqn cududv cuctracer cubidiag cuadjtq cuadjtqs satur; do
  FILES="$FILES arpifs/phys_ec/$r.F90"
done
: > "$B/MD5SUMS"
for p in $FILES; do
  git -C "$O" show "$FULL:ifs-source/$p" > "$B/src/$(basename "$p")"
  echo "$(md5sum "$B/src/$(basename "$p")" | cut -d' ' -f1)  ifs-source/$p" >> "$B/MD5SUMS"
done
echo "OPENIFS_COMMIT $FULL" >> "$B/MD5SUMS"
echo "EXTRACT_OK $(wc -l < "$B/MD5SUMS") entries"
