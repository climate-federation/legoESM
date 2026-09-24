#!/bin/bash
#SBATCH --job-name=hs1029lev
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/hs1029lev.%j.out
# #1029: the interval-1 localisation put both bursts at k=18 and k=16 of the
# 40-level hybrid coordinate.  A level INDEX is not a height, and the reading
# rule for this issue is phrased in terms of where in the COLUMN the maximum
# sits ("lower half" => terrain pressure-gradient error; "near the model top"
# => lid/sponge).  So print the pressure those two indices actually carry
# before either branch is claimed.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
"$PY" - <<'PYEOF'
import numpy as np
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm import constants
s = standard_hybrid_levels(40)
p_s = 1000e2
p = np.asarray(s.pressure_at_full(np.asarray([p_s])))[0]
print(f"standard_hybrid_levels(40), p_s = {p_s/100:.0f} hPa")
print(f"  k=0  (model top) : {p[0]/100:9.3f} hPa")
for k in (14, 15, 16, 17, 18, 19, 20):
    frac = float(p[k] / p_s)
    print(f"  k={k:<2d}            : {p[k]/100:9.3f} hPa   p/p_s = {frac:.3f}")
print(f"  k=39 (surface)   : {p[-1]/100:9.3f} hPa")
print()
print("The two bursts: seed 65537 at k=18, seed 42 at k=16.")
print("Column position is p/p_s: > 0.5 is the LOWER half by mass.")
PYEOF
echo DONE
