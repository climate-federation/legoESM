#!/usr/bin/env python
"""Profile JIT compile time of the lat-lon C-grid PE step.

Task #25 root-cause: the lat-lon C-grid step at multi-rank takes >2.5h
to JIT-compile (smokes 8063948 + 8063949 both TIMED OUT at 3h walltime
with execution never starting).  Hypothesis: ``ssp_rk3_step`` calls
``tendency_fn`` THREE TIMES sequentially (see
``ssp_rk3.py`` ``ssp_rk3_step``), so XLA inlines three
full copies of the entire tendency pipeline (advection + polar filter
+ diffusion + hydrostatic + vertical advection) into a single XLA
module.  XLA compile complexity is super-linear in op count, so
3× more ops → 9–27× more compile time.

This script measures:

* ``len(jaxpr.eqns)``         — jaxpr equation count.  Linear in the
                                pipeline; a scan-based RK3 should drop
                                this ~3× because the scan body is
                                traced once.
* ``compile_time``            — wall-clock for the first ``step_fn`` call
                                (forces JIT compile).
* ``second_call_time``        — wall-clock for the second call (should
                                hit JIT cache and be very fast; measures
                                actual run time).
* ``jaxpr_text_bytes``        — size of the jaxpr text dump.  Another
                                proxy for module complexity.

Usage::

    JAX_ENABLE_X64=1 python scripts/bench/jit_profile/profile_step_jit_compile.py \\
        --n-lat 16 --n-lon 32 --nlev 4 --tracers 0

    JAX_ENABLE_X64=1 python scripts/bench/jit_profile/profile_step_jit_compile.py \\
        --n-lat 16 --n-lon 32 --nlev 4 --tracers 3 \\
        --integrator ssp_rk3_scan

The minimal grid (n_lat=16, n_lon=32, nlev=4) gives a tractable trace
size for serial profiling — small enough to compile in seconds at the
INLINE integrator on a login node, large enough that the operator
selection is realistic.

To compare integrators::

    for INT in ssp_rk3 ssp_rk3_scan; do
        echo "=== $INT ==="
        python scripts/bench/jit_profile/profile_step_jit_compile.py \\
            --n-lat 16 --n-lon 32 --nlev 4 --tracers 3 \\
            --integrator "$INT"
    done

The expected outcome of Task #25 work: ``ssp_rk3_scan`` should report
a jaxpr.eqns count ~3× smaller than ``ssp_rk3``, and compile_time
proportionally smaller (XLA optimizer is super-linear in op count, so
the speedup typically exceeds 3×).
"""
from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp


def _build_model(n_lat: int, n_lon: int, nlev: int, tracers: int,
                 use_polar_filter: bool, integrator: str):
    """Build a minimal CGridLatLonPrimitiveEquationModel for profiling."""
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonHydrostaticState,
    )

    grid = create_latlon_grid(
        n_lat=n_lat, radius=constants.R_earth, omega=constants.Omega,
    )
    sigma = create_sigma_coordinate(n_levels=nlev)
    config = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,
        use_polar_filter=use_polar_filter,
        use_ppm_transport=(tracers > 0),
        time_integrator=integrator,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)

    # Build a perturbed rest state matching the test fixtures.
    import numpy as _np
    rng = _np.random.default_rng(seed=31337)
    eps = 1.0e-3
    u = jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev)))
    T = jnp.asarray(300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev)))
    p_s = jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon)))
    phis = jnp.zeros((n_lat, n_lon))

    tr = {}
    for k in range(tracers):
        name = ["q_v", "q_c", "q_r", "q_i", "q_s", "q_g"][k]
        tr[name] = jnp.asarray(
            1.0e-3 * (1.0 + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
        )

    state = CGridLatLonHydrostaticState(
        u=u, v=v, T=T, p_s=p_s, phis=phis, tracers=tr,
    )
    return model, state


def _jaxpr_stats(model, state, dt):
    """Return (n_eqns, jaxpr_text_bytes) for the step function trace."""
    closed_jaxpr = jax.make_jaxpr(model._step_cgrid)(state, dt)
    text = closed_jaxpr.pretty_print()
    return len(closed_jaxpr.eqns), len(text)


def _time_compile(model, state, dt, n_calls: int = 2):
    """Time JIT compile (first call) + cached execution (second call)."""
    times = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        # `_step_cgrid` returns (state_new, phys_state_out) since the
        # #413 carry threading; no carry threaded in this benchmark.
        out, _ = model._step_cgrid(state, dt)
        # Block until execution completes — `_step_cgrid` is `@jax.jit`
        # so the return is a deferred array.  Block to measure true
        # compile+execute time.
        jax.block_until_ready(out.T)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-lat", type=int, default=16)
    parser.add_argument("--n-lon", type=int, default=32)
    parser.add_argument("--nlev", type=int, default=4)
    parser.add_argument("--tracers", type=int, default=0,
                        help="Number of tracers (0 = dry, 3 = AMIP, 6 = full mp)")
    parser.add_argument("--use-polar-filter", action="store_true",
                        default=False)
    parser.add_argument("--integrator", type=str, default="ssp_rk3",
                        help="Time integrator (ssp_rk3, ssp_rk3_scan, ...)")
    parser.add_argument("--n-calls", type=int, default=2,
                        help="How many times to call _step_cgrid (default 2: "
                             "first call = compile+exec, second = cached exec)")
    parser.add_argument("--dump-jaxpr", action="store_true",
                        help="Print the full jaxpr text dump (LARGE).")
    args = parser.parse_args(argv)

    print(f"--- Profile: lat-lon C-grid PE step JIT compile ---")
    print(f"  Grid:        n_lat={args.n_lat}, n_lon={args.n_lon}, nlev={args.nlev}")
    print(f"  Tracers:     {args.tracers}")
    print(f"  Polar filter: {args.use_polar_filter}")
    print(f"  Integrator:  {args.integrator!r}")
    print(f"  JAX x64:     {jax.config.read('jax_enable_x64')}")
    print()

    model, state = _build_model(
        args.n_lat, args.n_lon, args.nlev, args.tracers,
        args.use_polar_filter, args.integrator,
    )

    # Stage 1: jaxpr stats
    print("Tracing _step_cgrid (jax.make_jaxpr)…")
    t0 = time.perf_counter()
    n_eqns, jaxpr_text_bytes = _jaxpr_stats(model, state, 300.0)
    t1 = time.perf_counter()
    print(f"  Trace time:           {t1 - t0:6.2f} s")
    print(f"  jaxpr.eqns:           {n_eqns:6d}")
    print(f"  jaxpr text size:      {jaxpr_text_bytes:6d} bytes")
    print()

    # Stage 2: compile + cached execution
    print(f"Timing {args.n_calls} _step_cgrid calls (first = compile+exec)…")
    times = _time_compile(model, state, 300.0, n_calls=args.n_calls)
    for i, t in enumerate(times):
        tag = "compile+exec" if i == 0 else f"call #{i+1} (cached)"
        print(f"  {tag:20s} {t:7.2f} s")
    print()

    if args.dump_jaxpr:
        print("--- jaxpr dump ---")
        closed_jaxpr = jax.make_jaxpr(model._step_cgrid)(state, 300.0)
        print(closed_jaxpr.pretty_print())

    # Machine-readable summary (one line CSV for sbatch-side aggregation)
    print()
    print(
        "CSV:"
        f"n_lat={args.n_lat},n_lon={args.n_lon},nlev={args.nlev},"
        f"tracers={args.tracers},polar={args.use_polar_filter},"
        f"integrator={args.integrator},"
        f"jaxpr_eqns={n_eqns},jaxpr_bytes={jaxpr_text_bytes},"
        f"compile_s={times[0]:.2f},"
        + (f"cached_s={times[1]:.4f}" if len(times) > 1 else "cached_s=NA")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
