#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --job-name=ifs_cumastrn_oracle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Build the single-column OpenIFS convection oracle (CUMASTRN and callees,
# unmodified OpenIFS 8f6f722 sources, native IFS constants, double precision).
# usage (repo root): sbatch --wait scripts/validate/oracle/ifs_cumastrn/build.sh <build dir>
# The convection routines are compiled as procedures of one module so their
# interfaces are explicit (the IFS build generates *.intfb.h instead; here
# those headers are empty).  No source line is changed.
set -euo pipefail
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
HERE=$REPO/scripts/validate/oracle/ifs_cumastrn
B=${1:?build dir}
mkdir -p "$B"
bash "$HERE/extract.sh" "$B"
S=$B/src
mkdir -p "$B/inc" "$B/obj"
for h in $(grep -ohE '#include *"[a-z0-9_]+\.intfb\.h"' "$S"/*.F90 | awk -F'"' '{print $2}' | sort -u); do
  : > "$B/inc/$h"
done
{
  echo "REPO_SHA $(git -C "$REPO" rev-parse HEAD)"
  echo "REPO_DIRTY $(git -C "$REPO" status --porcelain -- "$HERE" | wc -l)"
  gfortran --version | head -1
} >> "$B/MD5SUMS"
for f in stubs_hook.F90 stubs.F90 tdbg.F90 driver.F90 patch_oracle.py extract.sh; do
  [ -f "$HERE/$f" ] && echo "$(md5sum "$HERE/$f" | cut -d' ' -f1)  harness/$f" >> "$B/MD5SUMS"
done
FF="-O0 -g -cpp -ffree-line-length-none -fno-fast-math -fbacktrace -ffpe-trap=invalid,zero,overflow -I$B/inc -I$S -J$B/obj"
echo "FFLAGS $FF" >> "$B/MD5SUMS"
cd "$B/obj"
# module order follows the USE graph
gfortran $FF -c "$S/parkind1.F90"
gfortran $FF -c "$HERE/stubs_hook.F90"
for m in yomlun_ifsaux yomlun yomdyncore yomcst yoethf yoephli yoecumf yoecldp yoephy parphy yomsta yomdimv yomphy2 yophnc yoecumf2 yomcumfs yompertpar spp_gen_mod; do
  gfortran $FF -c "$S/$m.F90"
done
gfortran $FF -c "$HERE/stubs.F90"
gfortran $FF -c "$HERE/tdbg.F90"
python3 "$HERE/patch_oracle.py" "$S/cumastrn.F90" "$B/inc/cumastrn_dbg.F90" | tee -a "$B/MD5SUMS"
python3 - "$S" "$B/inc" <<'PYEOF'
# verbatim setup blocks the driver includes (no line edited)
import sys
S, I = sys.argv[1], sys.argv[2]
L = open(f"{S}/suphec.F90").read().splitlines()
a = L.index("RVTMP2=0.0_JPRB              !neglect cp,moist")
b = next(i for i, l in enumerate(L) if l.startswith("RKOOP2="))
blk = [l for l in L[a:b + 1] if not l.startswith("ISMAX=")]
open(f"{I}/suphec_thf_block.inc", "w").write("\n".join(blk) + "\n")
P = open(f"{S}/suphli.F90").read().splitlines()
blk2 = [l for l in P if l.strip() in ("LPHYLIN   = .FALSE.", "RLPTRC=RTICE+(RTWAT-RTICE)/SQRT(2.0_JPRB)")]
assert len(blk2) == 2, blk2
open(f"{I}/suphli_block.inc", "w").write("\n".join(blk2) + "\n")
print("SETUP_BLOCKS_OK", len(blk), len(blk2))
PYEOF
{
  echo "MODULE CU_ALL"
  echo "IMPLICIT NONE"
  echo "CONTAINS"
  for r in satur cuadjtq cuadjtqs cubidiag cuentr cubasmcn cuinin cubasen cuascn cudlfsn cuddrafn cuflxn cudtdqn cududv cuctracer; do
    echo "#include \"$r.F90\""
  done
  echo "#include \"cumastrn_dbg.F90\""
  echo "END MODULE CU_ALL"
} > cu_all.F90
gfortran $FF -c cu_all.F90
gfortran $FF -c "$S/sucumf.F90"
gfortran $FF -c "$S/fcgeneralized_gamma.F90"
gfortran $FF -c "$S/sucldp.F90"
echo BUILD_OBJECTS_OK
if [ -f "$HERE/driver.F90" ]; then
  gfortran $FF -c "$HERE/driver.F90"
  gfortran -o "$B/ifs_cumastrn.x" *.o
  echo BUILD_OK
fi
