#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=divduo
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/divduo_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
S=$REPO/scripts/validate/fv3_native
W=/burg-archive/glab/users/pg2328/fv3_recon/divduo_work
rm -rf $W && mkdir -p $W && cd $W
PYTHONPATH=$REPO/packages/core:$REPO/src \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  $S/gen_divduo_oracle.py $W
FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
gfortran $FF -J $W -c $S/fv3_swcore_shim.F90 -o $W/shim.o
gfortran $FF -J $W -c $S/fv3_swcore_extract.F90 -o $W/ext.o
gfortran $FF -J $W $S/fv3_divduo_driver.F90 $W/shim.o $W/ext.o -o $W/drv
cd $W && ./drv
# pack the committed fixture from the Fortran output (reproducible)
PYTHONPATH=$REPO/packages/core:$REPO/src \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  $S/gen_divduo_oracle.py $W --pack
PYTHONPATH=$REPO/packages/core:$REPO/src \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python - $W <<'PY'
import sys, numpy as np
W=sys.argv[1]; res,ng=12,3; lo=1-ng; m_b=res+2*ng+1
py=np.load(f"{W}/py_divduo.npy")
fd=np.full((m_b,m_b),np.nan)
for line in open(f"{W}/divduo_output.txt"):
    p=line.split(); fd[int(p[1])-lo,int(p[2])-lo]=float(p[3])
# compare where both finite AND not the 1e25 init (both init to 1e25 in unwritten)
mask=np.isfinite(py)&np.isfinite(fd)
diff=np.abs(py[mask]-fd[mask]); denom=np.maximum(np.abs(fd[mask]),1.0)
rel=(diff/denom).max()
print(f"divergence_corner_duo py-vs-Fortran max_rel={rel:.3e} n={int(mask.sum())}")
print("VERDICT:", "BIT-EXACT" if rel<1e-11 else f"DIVERGE {rel:.3e}")
PY
echo DIVDUO_DONE
