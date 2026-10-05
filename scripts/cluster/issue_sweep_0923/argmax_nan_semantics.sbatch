#!/bin/bash
#SBATCH --job-name=argmax_nan
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/argmax_nan.%j.out
# What does jnp.argmax ACTUALLY do with a partially non-finite field?
#
# A reviewer asserted that NaN never wins a comparison reduction, so argmax
# over a mixed field returns the largest FINITE element and a guard should
# therefore localise the partial case rather than refuse it.  A test written
# from that claim went red, so the claim is measured here instead of argued.
# The three hypotheses are separated by putting the NaN somewhere that is
# NEITHER index 0 NOR the finite maximum:
#     finite-max wins  -> (1, 1)
#     NaN wins         -> (2, 3)
#     first cell wins  -> (0, 0)
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
"$PY" - <<'PYEOF'
import jax, jax.numpy as jnp, numpy as np
print("jax", jax.__version__, jax.devices())
def probe(name, f):
    a = jnp.asarray(f)
    at = tuple(int(x) for x in jnp.unravel_index(jnp.argmax(a), a.shape))
    print(f"  {name:38s} argmax -> {at}   max = {float(jnp.max(a))}")

f = np.zeros((3, 4)); f[1, 1] = 5.0; f[2, 3] = np.nan
probe("NaN at (2,3), finite max at (1,1)", f)
g = np.zeros((3, 4)); g[1, 1] = 5.0; g[0, 0] = np.nan
probe("NaN at (0,0), finite max at (1,1)", g)
h = np.zeros((3, 4)); h[1, 1] = 5.0; h[2, 3] = np.inf
probe("+Inf at (2,3), finite max at (1,1)", h)
k = np.full((3, 4), np.nan)
probe("wholly NaN", k)
print("\nnumpy for contrast:", np.unravel_index(np.argmax(f), f.shape))
PYEOF
echo DONE
