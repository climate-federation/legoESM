"""FV3_3D iter 65: C96 1-day stability smoke test.

iter 63 added user guidance for C96+ but C96 was empirically
UNTESTED.  This probe runs HS C96 with the iter-43 default scale
(LEGOESM_AH_SCALE=10.0 auto-applied at n>=72) for 1 day and reports
whether the integration completes finite.

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/_iter65_c96_smoke.py

The probe lifts the matrix's HS-3D setup verbatim (grid, vertical
coord, init, config) and runs only the time-stepping; expected
wall time ~30-60 s based on C72 scaling.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Allow `import run_atmosphere_test_matrix` when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_c96_smoke(days: float = 1.0, ah_scale: float = 10.0) -> dict:
    """Run HS C96 for ``days`` and return a diagnostic dict."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import (
        held_suarez_init,
        held_suarez_forcing,
    )
    import run_atmosphere_test_matrix as M

    n = 96
    nlev = M.DEFAULT_NLEV
    grid = create_cubed_sphere(n)
    sigma = M._create_vertical(nlev, "hybrid")

    hd = M._hyperdiff_cube(n)
    dd = M._div_damp_cube(n)
    ah_base = M._laplacian_visc_cube(n)
    ah = ah_base * ah_scale

    print(
        f"[iter65] n={n} nlev={nlev} hd={hd:.3e} dd={dd:.3e} "
        f"ah_base={ah_base:.3e} ah_scale={ah_scale} ah={ah:.3e}",
        flush=True,
    )

    dt = float(os.environ.get("ITER65_DT", "200.0"))
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
        div_damp_coeff=dd, A_h=ah,
        use_conservation_fixer=True, fix_mass=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    physics_fn = held_suarez_forcing

    nsteps = int(round(days * 86400.0 / dt))
    print(f"[iter65] running {nsteps} steps × dt={dt}s = {days} days",
          flush=True)

    @jax.jit
    def _step(s):
        return model.step_with_physics(s, dt, physics_fn)

    t0 = time.time()
    for i in range(nsteps):
        state = _step(state)
        if i in (0, nsteps // 4, nsteps // 2, 3 * nsteps // 4):
            mu = float(jnp.max(jnp.abs(state.u.data)))
            print(f"[iter65] step {i}/{nsteps}  max|u|={mu:.3f}",
                  flush=True)
            if not np.isfinite(mu):
                wall = time.time() - t0
                return {"finite": False, "step_blowup": i, "wall": wall}

    state.u.data.block_until_ready()
    wall = time.time() - t0

    max_u = float(jnp.max(jnp.abs(state.u.data)))
    max_v = float(jnp.max(jnp.abs(state.v.data)))
    finite_all = (
        bool(jnp.all(jnp.isfinite(state.u.data)))
        and bool(jnp.all(jnp.isfinite(state.v.data)))
        and bool(jnp.all(jnp.isfinite(state.T.data)))
        and bool(jnp.all(jnp.isfinite(state.p_s.data)))
    )
    return {
        "finite": finite_all,
        "max_u": max_u,
        "max_v": max_v,
        "wall": wall,
        "steps": nsteps,
    }


if __name__ == "__main__":
    days = float(os.environ.get("ITER65_DAYS", "1.0"))
    scale = float(os.environ.get("ITER65_AH_SCALE", "10.0"))
    res = _run_c96_smoke(days=days, ah_scale=scale)
    print(f"\n[iter65] RESULT: {res}", flush=True)
