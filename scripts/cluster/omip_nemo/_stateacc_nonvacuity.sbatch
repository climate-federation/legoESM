#!/bin/bash
#SBATCH --job-name=stacc_nv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/stacc_nv_%j.out
# Non-vacuity: the state-mean test must FAIL when the state sum is removed.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_merge; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PYEOF'
import numpy as np
from scripts.run import run_omip_core2 as d
import tests.ocean.unit.test_surface_flux_accumulator as t
d._SurfaceFluxAccumulator._STATE_FIELDS = ()   # the "fix reverted" model
try:
    t.test_state_mean_is_the_window_mean_of_T_and_S()
    print("VACUOUS: test passed without the state sum"); raise SystemExit(1)
except (KeyError, AssertionError) as e:
    print("NON-VACUOUS: test fails without the state sum ->", type(e).__name__, e)
PYEOF
echo "=== card exit: $? ==="
