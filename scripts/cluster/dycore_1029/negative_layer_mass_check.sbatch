#!/bin/bash
#SBATCH --job-name=hs1029negm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/hs1029negm.%j.out
# #1029: is the upper end of the mountain-height sweep even a valid experiment?
#
# The 40-level hybrid coordinate the matrix builds emits, at construction, a
# warning that it carries NEGATIVE layer mass below a threshold surface
# pressure.  The DCMIP mountain lowers p_s over its peak, so a tall enough
# mountain puts the column INSIDE that invalid regime -- which would make the
# h_0 = 4000 m arm (dead at day 0.46 with max|v| only 4 m/s, no jet runaway at
# all) a coordinate defect rather than a physical failure mode.
#
# Measured here, from the code rather than from scale-height arithmetic: the
# minimum p_s each h_0 in the sweep actually produces, against the threshold
# the coordinate itself reports.  Anything at or below it is a confounded arm
# and must be struck from the height axis.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
"$PY" - <<'PYEOF'
import numpy as np
from legoesm.grids.vertical import (standard_hybrid_levels,
                                    hybrid_min_valid_surface_pressure)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.idealized.held_suarez_topo import _phis_latlon
from legoesm import constants

s = standard_hybrid_levels(40)
thr = hybrid_min_valid_surface_pressure(
    np.asarray(s.A_half), np.asarray(s.B_half), constants.p_ref)
print(f"40-level hybrid: NEGATIVE layer mass for p_s < {thr/100:.1f} hPa\n")

# The init routine's own relation, read off held_suarez_init_latlon:
#   p_s = p_s_init * exp(-phis / (R_d * T_init)),  T_init = 300 K default
grid = create_latlon_grid(72, 144)
T_init = 300.0
for h0 in (0.0, 1000.0, 2000.0, 2800.0, 3200.0, 3600.0, 4000.0):
    phis = np.asarray(_phis_latlon(grid, h0))
    p_s = constants.p_ref * np.exp(-phis / (constants.R_d * T_init))
    lo = float(p_s.min())
    bad = int((p_s < thr).sum())
    verdict = "INVALID -- negative layer mass" if lo < thr else "ok"
    print(f"  h_0 = {h0:6.0f} m   min p_s = {lo/100:7.2f} hPa   "
          f"cells below threshold = {bad:5d} / {p_s.size}   {verdict}")
print("\nSweep survival for reference (days): 0 -> 200 (completes), "
      "1000 -> 71.5, 2000 -> 54.6, 2800 -> 5.8, 3200 -> 3.9, 3600 -> 3.5, "
      "4000 -> 0.46")
PYEOF
echo DONE
