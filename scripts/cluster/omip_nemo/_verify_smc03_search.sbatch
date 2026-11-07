#!/bin/bash
#SBATCH --job-name=smc03search
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/smc03search_%j.out
# The SMC03 enclosing-cell search no longer builds a (..., nlev, n_t) table.
# This checks the index is identical, the PRESSURES are bitwise identical on
# the lat-lon caller, and that every SMC03 suite still passes.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "=== 1. BITWISE: pressures from the new search vs the old dense table ==="
$PY - <<'PY'
import jax, jax.numpy as jnp, numpy as np
from legoesm.ocean.dynamics.pgf_smc03 import (
    compute_pressure_at_target_smc03, reconstruct_harmonic_slopes)

rng = np.random.default_rng(7)
n_col, nlev = 53, 20
h = jnp.asarray(rng.uniform(4.0, 200.0, size=(n_col, nlev)))
rho = jnp.asarray(1020.0 + np.cumsum(rng.uniform(0.0, 0.4, (n_col, nlev)), -1))
z_bot = jnp.cumsum(h, -1); z_top = z_bot - h; z_c = z_bot - 0.5 * h
sigma = reconstruct_harmonic_slopes(rho, z_c, jnp.ones_like(h))
g = 9.80665

def dense(rho, h, z_c, sigma, z_t, g):
    z_b = jnp.cumsum(h, -1); z_tp = z_b - h
    dP = g * h * rho; P_b = jnp.cumsum(dP, -1); P_tp = P_b - dP
    z_tc = jnp.clip(z_t, min=0.0, max=z_b[..., -1:])
    in_cell = (z_tc[..., None, :] >= z_tp[..., :, None]) & (
               z_tc[..., None, :] <= z_b[..., :, None])
    k = jnp.argmax(in_cell.astype(jnp.int32), axis=-2)
    ta = lambda a: jnp.take_along_axis(a, k, axis=-1)
    dz = z_tc - ta(z_tp)
    rho_eff = ta(rho) + 0.5 * ta(sigma) * (z_tc + ta(z_tp) - 2.0 * ta(z_c))
    return ta(P_tp) + g * dz * rho_eff

# GLM's caveat, tested rather than filed: a partial-cell seafloor leaves a
# TRAILING RUN of h=0 cells, so z_bot repeats its last value and the two rules
# could pick different members of that run.  Build columns whose last 6 cells
# are empty and check the pressures anyway.
h_pc = h.at[:, -6:].set(0.0)
z_bot_pc = jnp.cumsum(h_pc, -1); z_c_pc = z_bot_pc - 0.5 * h_pc
sigma_pc = reconstruct_harmonic_slopes(
    rho, z_c_pc, jnp.concatenate([jnp.ones((n_col, nlev - 6)),
                                  jnp.zeros((n_col, 6))], axis=-1))
for name, z_t in (("centroids", z_c), ("interfaces", z_bot),
                  ("cell tops", z_top),
                  ("random", jnp.asarray(rng.uniform(0, 1, (n_col, nlev)))
                             * z_bot[:, -1:])):
    new = compute_pressure_at_target_smc03(rho, h, z_c, sigma, z_t, g)
    old = dense(rho, h, z_c, sigma, z_t, g)
    same = bool(jnp.all(new == old))
    print(f"  {name:11s} bitwise identical: {same}  max|d|={float(jnp.max(jnp.abs(new-old))):.3e}")
    assert same, f"{name} differs"
for name, z_t in (("zero-thick centroids", z_c_pc),
                  ("zero-thick interfaces", z_bot_pc)):
    new_p = compute_pressure_at_target_smc03(rho, h_pc, z_c_pc, sigma_pc, z_t, g)
    old_p = dense(rho, h_pc, z_c_pc, sigma_pc, z_t, g)
    same = bool(jnp.all(new_p == old_p))
    fin = bool(jnp.all(jnp.isfinite(new_p)))
    print(f"  {name:22s} bitwise identical: {same}  finite: {fin}")
    assert same and fin, f"{name} differs or is not finite"
print("all target sets bitwise identical")
PY

echo
echo "=== 2. the SMC03 suites (lat-lon phases + the MPAS wrapper) ==="
$PY -m pytest -q tests/ocean/unit/test_pgf_smc03_phase1.py \
  tests/ocean/unit/test_pgf_smc03_phase2.py tests/ocean/unit/test_pgf_smc03_phase3.py \
  tests/ocean/unit/test_pgf_smc03_phase4.py tests/ocean/unit/test_pgf_smc03_mpas.py 2>&1 | tail -3
echo DONE
