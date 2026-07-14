"""FV3_3D C96+ stability probe (iter 65, expanded through iter 84).

Originally written for iter 65 to validate C96 1-day stability
under the iter-43 ``LEGOESM_AH_SCALE=10`` default.  Expanded
through iter 70/82/85 to support the full iter 65-85 sweep across
``dt``, ``ah_scale``, ``smag_cs``, and ``n``.

Env vars:
    ITER65_DAYS      run length in days (default 1.0)
    ITER65_AH_SCALE  multiplier on the v1 helper A_h (default 10.0)
    ITER65_SMAG_CS   Smagorinsky c_s (default 0.0)
    ITER65_DT        timestep in seconds (default 200.0)
    ITER65_N         cube face count (default 96)

Diagnostic prints fire at i in (0, nsteps//4, nsteps//2,
3*nsteps//4) plus a final RESULT line; if NaN appears at any
diagnostic, the run halts early and reports step_blowup.

Usage examples::

    # iter-65 1-day smoke at C96 dt=200 default
    .venv/bin/python scripts/tmp/_iter65_c96_smoke.py

    # iter-79 / iter-82 30-day C96 dt=50 long-time validation
    JAX_ENABLE_X64=1 ITER65_DAYS=30.0 ITER65_DT=50.0 \\
      .venv/bin/python scripts/tmp/_iter65_c96_smoke.py

The probe lifts the matrix's HS-3D setup verbatim (grid, vertical
coord, init, config) and runs only the time-stepping with quartile
diagnostics + final RESULT line.
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

# Allow `import run_atmosphere_test_matrix` (now at scripts/matrix/) from this
# file's scripts/tmp/ location: parent.parent is scripts/, + matrix.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "matrix"))


def _run_c96_smoke(
    days: float = 1.0,
    ah_scale: float = 10.0,
    smag_cs: float = 0.0,
    n: int = 96,
) -> dict:
    """Run HS at the cube resolution ``n`` for ``days`` and return
    a diagnostic dict.

    Despite the legacy ``_c96_`` name and ``n=96`` default, the
    probe accepts ``n`` as an explicit argument so the same script
    handles C36, C48, C72, C96, C144, etc.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init,
        held_suarez_forcing,
    )
    import run_atmosphere_test_matrix as M

    nlev = M.DEFAULT_NLEV
    grid = create_cubed_sphere(n)
    sigma = M._create_vertical(nlev, "hybrid")

    hd = M._hyperdiff_cube(n)
    dd = M._div_damp_cube(n)
    ah_base = M._laplacian_visc_cube(n)
    ah = ah_base * ah_scale

    print(
        f"[iter65] n={n} nlev={nlev} hd={hd:.3e} dd={dd:.3e} "
        f"ah_base={ah_base:.3e} ah_scale={ah_scale} ah={ah:.3e} "
        f"smag_cs={smag_cs}",
        flush=True,
    )

    dt = float(os.environ.get("ITER65_DT", "200.0"))
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
        div_damp_coeff=dd, A_h=ah,
        smagorinsky_cs=smag_cs,
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
    smag_cs = float(os.environ.get("ITER65_SMAG_CS", "0.0"))
    n_cube = int(os.environ.get("ITER65_N", "96"))
    res = _run_c96_smoke(
        days=days, ah_scale=scale, smag_cs=smag_cs, n=n_cube,
    )
    print(f"\n[iter65] RESULT: {res}", flush=True)
