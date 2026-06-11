#!/bin/bash
#SBATCH --job-name=diagnose_jit_cache
#SBATCH --output=results/profile_jit_compile/cache_diag_%j.out
#SBATCH --error=results/profile_jit_compile/cache_diag_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Task #29 diagnostic — figure out why the JAX persistent compilation
# cache misses across smoke runs.
#
# Hypothesis: ``PYTHONHASHSEED`` randomisation makes some part of the
# trace produce a different hash on each fresh process, so the cache
# key (HLO module hash) differs and JAX writes a new cache entry
# every smoke run.  Effect: the 100-y AMIP chain pays full compile
# every job — a multi-hour overhead per restart.
#
# Experiment design:
#   A. Run the profile script TWICE in a FRESH cache dir.
#      - Pass 1: cold cache, ``compile_s`` should be > 0
#      - Pass 2: warm cache, ``compile_s`` should be near zero
#      - If cache hits, pass 2 is fast → confirms the cache CAN hit
#        in principle (rules out a broken cache configuration).
#   B. Then run the SAME profile in a NEW process with the SAME
#      cache dir from A — does the cache hit across processes?
#      - If yes: the original smoke-vs-smoke miss is from a real
#        config difference (run-to-run drift in dt clamp or similar).
#      - If no: PYTHONHASHSEED or some process-specific identity is
#        leaking into the trace; fixable by setting PYTHONHASHSEED=0.
#   C. Repeat (B) with ``PYTHONHASHSEED=0`` set explicitly — does the
#      cache hit now?

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/profile_jit_compile

PY="${HOME}/.venvs/legoesm-mpi/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Isolated cache for this experiment so we can observe hits/misses
# without interference from the production ~/.cache/legoesm/jit_cache.
CACHE_DIR="${TMPDIR:-/tmp}/jit_cache_diag_${SLURM_JOB_ID:-$$}"
rm -rf "${CACHE_DIR}"
export LEGOESM_JIT_CACHE_DIR="${CACHE_DIR}"
# Force everything to be cached (production smokes cross the 1s
# threshold but the small profile compiles do not — set to 0 here).
export LEGOESM_JIT_CACHE_MIN_SECS=0

# Verbose JAX cache logging — every hit/miss + the key.
export JAX_COMPILATION_CACHE_LOGGING=1
export JAX_EXPLAIN_CACHE_MISSES=true

PROFILE="${PY} ${REPO_ROOT}/scripts/bench/jit_profile/profile_step_jit_compile.py
         --n-lat 16 --n-lon 32 --nlev 4 --tracers 3
         --use-polar-filter --integrator ssp_rk3_scan"

echo "========================================"
echo "  JIT cache miss diagnostic (Task #29)"
echo "  Job ID:    ${SLURM_JOB_ID:-<no-slurm>}"
echo "  Cache dir: ${CACHE_DIR}"
echo "  Default PYTHONHASHSEED (random): ${PYTHONHASHSEED:-<unset>}"
echo "========================================"

# ----------------------------------------------------------------------
# Experiment A: same-process cache hit (sanity check)
# ----------------------------------------------------------------------
echo
echo "=== Experiment A: cold + warm in ONE process ==="
${PY} - <<'PYEOF'
import os, sys, time
import jax
sys.path.insert(0, "/burg-archive/glab/users/pg2328/legoESM/src")

# Re-import legoesm with the cache config from env.
from legoesm.runtime.backend import _configure_persistent_jit_cache
_configure_persistent_jit_cache()

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonHydrostaticState,
)
import jax.numpy as jnp
import numpy as np

grid = create_latlon_grid(n_lat=16, radius=constants.R_earth, omega=constants.Omega)
sigma = create_sigma_coordinate(n_levels=4)
cfg = CGridLatLonPrimitiveEquationConfig(
    fix_mass=True, use_polar_filter=True, use_ppm_transport=True,
    time_integrator="ssp_rk3_scan",
)
model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=300.0)

rng = np.random.default_rng(seed=42)
state = CGridLatLonHydrostaticState(
    u=jnp.asarray(rng.standard_normal((16, 33, 4)) * 1e-3),
    v=jnp.asarray(rng.standard_normal((17, 32, 4)) * 1e-3),
    T=jnp.asarray(300.0 + rng.standard_normal((16, 32, 4)) * 1e-3),
    p_s=jnp.asarray(1e5 + rng.standard_normal((16, 32)) * 10),
    phis=jnp.zeros((16, 32)),
    tracers={
        "q_v": jnp.asarray(1e-3 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_c": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_r": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
    },
)

# Pass 1: cold cache, force compile
t0 = time.perf_counter()
out = model._step_cgrid(state, 300.0)
jax.block_until_ready(out.T)
t1 = time.perf_counter()
print(f"Pass A1 (cold):  {t1-t0:.3f} s")

# Pass 2: same process, should be cached
t0 = time.perf_counter()
out = model._step_cgrid(state, 300.0)
jax.block_until_ready(out.T)
t1 = time.perf_counter()
print(f"Pass A2 (warm):  {t1-t0:.3f} s")
PYEOF

# List the cache entries after experiment A
echo
echo "--- Cache entries after experiment A ---"
ls -la "${CACHE_DIR}" 2>/dev/null | head -10 || true
KEY_A=$(ls "${CACHE_DIR}" 2>/dev/null | grep -E "jit_.*step_cgrid" | head -1 || echo "<no step_cgrid key>")
echo "step_cgrid key: ${KEY_A}"

# ----------------------------------------------------------------------
# Experiment B: same config, NEW process, default PYTHONHASHSEED random
# ----------------------------------------------------------------------
echo
echo "=== Experiment B: NEW process, default PYTHONHASHSEED (random) ==="
${PY} - <<'PYEOF'
import os, sys, time
import jax
sys.path.insert(0, "/burg-archive/glab/users/pg2328/legoESM/src")
from legoesm.runtime.backend import _configure_persistent_jit_cache
_configure_persistent_jit_cache()

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonHydrostaticState,
)
import jax.numpy as jnp, numpy as np

grid = create_latlon_grid(n_lat=16, radius=constants.R_earth, omega=constants.Omega)
sigma = create_sigma_coordinate(n_levels=4)
cfg = CGridLatLonPrimitiveEquationConfig(
    fix_mass=True, use_polar_filter=True, use_ppm_transport=True,
    time_integrator="ssp_rk3_scan",
)
model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=300.0)

rng = np.random.default_rng(seed=42)
state = CGridLatLonHydrostaticState(
    u=jnp.asarray(rng.standard_normal((16, 33, 4)) * 1e-3),
    v=jnp.asarray(rng.standard_normal((17, 32, 4)) * 1e-3),
    T=jnp.asarray(300.0 + rng.standard_normal((16, 32, 4)) * 1e-3),
    p_s=jnp.asarray(1e5 + rng.standard_normal((16, 32)) * 10),
    phis=jnp.zeros((16, 32)),
    tracers={
        "q_v": jnp.asarray(1e-3 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_c": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_r": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
    },
)

t0 = time.perf_counter()
out = model._step_cgrid(state, 300.0)
jax.block_until_ready(out.T)
t1 = time.perf_counter()
print(f"Pass B (new process, default seed): {t1-t0:.3f} s")
PYEOF

echo
echo "--- Cache entries after experiment B ---"
ls -la "${CACHE_DIR}" 2>/dev/null | head -10 || true
KEY_B=$(ls "${CACHE_DIR}" 2>/dev/null | grep -E "jit_.*step_cgrid" | head -1 || echo "<no key>")
echo "step_cgrid key: ${KEY_B}"

# ----------------------------------------------------------------------
# Experiment C: same config, NEW process, PYTHONHASHSEED=0
# ----------------------------------------------------------------------
echo
echo "=== Experiment C: NEW process, PYTHONHASHSEED=0 ==="
PYTHONHASHSEED=0 ${PY} - <<'PYEOF'
import os, sys, time
import jax
sys.path.insert(0, "/burg-archive/glab/users/pg2328/legoESM/src")
from legoesm.runtime.backend import _configure_persistent_jit_cache
_configure_persistent_jit_cache()

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonHydrostaticState,
)
import jax.numpy as jnp, numpy as np

print(f"PYTHONHASHSEED in subprocess: {os.environ.get('PYTHONHASHSEED', '<unset>')}")

grid = create_latlon_grid(n_lat=16, radius=constants.R_earth, omega=constants.Omega)
sigma = create_sigma_coordinate(n_levels=4)
cfg = CGridLatLonPrimitiveEquationConfig(
    fix_mass=True, use_polar_filter=True, use_ppm_transport=True,
    time_integrator="ssp_rk3_scan",
)
model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=300.0)

rng = np.random.default_rng(seed=42)
state = CGridLatLonHydrostaticState(
    u=jnp.asarray(rng.standard_normal((16, 33, 4)) * 1e-3),
    v=jnp.asarray(rng.standard_normal((17, 32, 4)) * 1e-3),
    T=jnp.asarray(300.0 + rng.standard_normal((16, 32, 4)) * 1e-3),
    p_s=jnp.asarray(1e5 + rng.standard_normal((16, 32)) * 10),
    phis=jnp.zeros((16, 32)),
    tracers={
        "q_v": jnp.asarray(1e-3 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_c": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
        "q_r": jnp.asarray(1e-5 * (1 + rng.standard_normal((16, 32, 4)) * 0.05)),
    },
)

t0 = time.perf_counter()
out = model._step_cgrid(state, 300.0)
jax.block_until_ready(out.T)
t1 = time.perf_counter()
print(f"Pass C (new process, PYTHONHASHSEED=0): {t1-t0:.3f} s")
PYEOF

echo
echo "--- Cache entries after experiment C ---"
ls -la "${CACHE_DIR}" 2>/dev/null | head -15 || true
N_KEYS=$(ls "${CACHE_DIR}" 2>/dev/null | grep -E "jit_.*step_cgrid" | wc -l || echo 0)
echo
echo "Total step_cgrid cache entries: ${N_KEYS}"
echo "(If 1 entry: cache hits across processes — production smokes should hit too.)"
echo "(If 3 entries: each process gets its own key — non-deterministic trace.)"

rm -rf "${CACHE_DIR}"
echo
echo "========================================"
echo "  Diagnostic complete"
echo "  Date: $(date -u)"
echo "========================================"
