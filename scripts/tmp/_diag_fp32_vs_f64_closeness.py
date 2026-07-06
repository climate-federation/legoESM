#!/usr/bin/env python
"""fp32-vs-f64 CLOSENESS check for the OMIP lat-lon C-grid ocean step.

Runs K steps of a small lat-lon C-grid ocean (implicit_cn barotropic PCG +
wright EOS + implicit vertical mixing — the OMIP recipe) and saves the final
state.  Invoked twice — once with ``--fp32`` (JAX x64 OFF, float32 arrays +
PrecisionPolicy.fp32) and once without (f64) — by the companion sbatch, which
then numpy-compares the two .npz files.

The point: confirm fp32 computes the SAME physics as f64 (not a different or
broken code path) to ~fp32 precision over a few steps.  fp32 carries ~7
significant digits, so a relative agreement of ~1e-4 over K steps is the
expected single-precision floor (cf. the SPMD re-association floor of
atol=2e-4/rtol=1e-3 in tests/parallel/test_latlon_ocean_spmd_step.py); a
divergent/broken fp32 path would show O(1) disagreement instead.

Throwaway diagnostic (scripts/tmp/), not a tracked driver.  Run via sbatch:
    JAX_PLATFORMS=cpu python scripts/tmp/_diag_fp32_vs_f64_closeness.py --fp32 --out fp32.npz
    JAX_PLATFORMS=cpu python scripts/tmp/_diag_fp32_vs_f64_closeness.py        --out f64.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# x64 MUST be decided before any JAX op — sniff --fp32 from argv (the same
# pattern run_omip_core2.py uses for the production --fp32 toggle).
_FP32 = "--fp32" in sys.argv[1:]

import jax  # noqa: E402

if not _FP32:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    rest_state_latlon_cgrid_ocean,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)


def _perturbed_state(grid, z_coord):
    """Rest state + small u/v/T/eta perturbations so the step exercises every
    term (advection / Coriolis / PGF / barotropic), not the trivial rest fixed
    point.  Mirrors tests/parallel/test_latlon_ocean_spmd_step.py::_perturbed_state
    so the comparison runs the same well-exercised configuration."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    # Build the perturbations in float64 on the host, then let jnp.asarray cast
    # to the active default dtype (float32 under --fp32, float64 otherwise) so
    # BOTH runs start from the SAME numbers to working precision.
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--fp32", action="store_true",
                   help="Run in float32 (x64 off + PrecisionPolicy.fp32).")
    p.add_argument("--out", type=str, required=True,
                   help="Output .npz path for the final state arrays.")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--n-lat", type=int, default=48)
    p.add_argument("--n-lon", type=int, default=96)
    p.add_argument("--nlev", type=int, default=10)
    args = p.parse_args()

    if bool(args.fp32) != _FP32:
        raise SystemExit("internal: --fp32 argv sniff / argparse mismatch.")

    set_policy(PrecisionPolicy.fp32() if args.fp32 else PrecisionPolicy.fp64())
    print(f"[closeness] precision={'fp32' if args.fp32 else 'fp64'} "
          f"x64={jax.config.jax_enable_x64} steps={args.steps} dt={args.dt}")

    grid = create_latlon_grid(n_lat=args.n_lat, n_lon=args.n_lon)
    z_coord = create_ocean_z_star(n_levels=args.nlev, H_max=4000.0)
    # OMIP recipe: implicit_cn barotropic (fixed-iter PCG via force_pcg so the
    # single-rank run takes the SAME solver the SPMD OMIP path uses) + implicit
    # vertical mixing.  These are exactly the fp32-sensitive blocks under test.
    # from_flat: barotropic_* fields are nested post-#501.
    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="implicit_cn",
        barotropic_implicit_force_pcg=True,
        implicit_vertical_mixing=True,
        eos="wright",
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = _perturbed_state(grid, z_coord)

    # Report the working dtype actually carried by the state (the fp32 proof).
    print(f"[closeness] state dtypes: T={state.T.data.dtype} "
          f"u={state.u.data.dtype} eta={state.eta.data.dtype}")

    for _ in range(int(args.steps)):
        state = model.step(state, args.dt)

    out = {}
    for nm in ("u", "v", "eta", "T", "S"):
        out[nm] = np.asarray(getattr(state, nm).data, dtype=np.float64)
        finite = np.isfinite(out[nm]).all()
        print(f"[closeness] {nm}: shape={out[nm].shape} finite={finite} "
              f"max|.|={np.max(np.abs(out[nm])):.4e}")
        if not finite:
            raise SystemExit(f"[closeness] FAIL: {nm} has non-finite values "
                             f"in {'fp32' if args.fp32 else 'fp64'}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[closeness] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
