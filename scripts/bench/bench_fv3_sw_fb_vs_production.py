"""Benchmark FB vs production cube SW step (FV3 single-implementation, Phase-1 M1).

Times the jitted steady-state per-step wall time of

* ``production``      — ``FV3EdgeShallowWaterModel.step`` (A-L RK3, shipped
  non-duogrid grid + ``iter1009_dual_target_config``),
* ``production_duo``  — same model on a DUOGRID grid (isolates the duogrid
  halo cost from the core cost, since the FB chain is duogrid-only),
* ``fb``              — ``FV3FBShallowWaterModel.step`` (faithful FV3
  forward-backward chain, duogrid, ``fb_m1_preset_config``),

on the Williamson-2 IC at the requested resolutions.  AOT pattern (memory
'AOT-vs-jit-cache timing'): ``jit(...).lower(state).compile()`` FIRST, then
time ONLY the compiled executable — never a first ``.step`` call whose wall
time is compile+run.

``--stages`` additionally times the FB phases separately-jitted at the first
resolution (entry/exit covariant conversion, c_sw, p_grad_c, the d_sw1..6
chain, the d_sw3 B-grid KE PPM transport alone, phase-4 PGF) to localize the
overhead.  Separately-jitted stage sums exceed the fused full step (XLA
fusion across phase boundaries), so stage times are attribution, not an
exact decomposition.

Usage::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
        scripts/bench/bench_fv3_sw_fb_vs_production.py \
        --resolutions C36,C48,C96 --steps 30 --stages
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

DT = 300.0
_CORES = ("production", "production_duo", "fb")


def _w2_state(grid, cdgrid):
    """Matrix-runner W2 edge-staggered IC (run_atmosphere_test_matrix.py:2603)."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import FV3EdgeShallowWaterState

    from tests.test_cases.williamson import williamson_test2
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


def _build(core: str, n: int):
    """Build (model, state) for one bench lane."""
    import warnings

    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        FV3FBShallowWaterModel,
        cdgrid_hyperdiff_cube,
        fb_m1_preset_config,
        iter1009_dual_target_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    def _prod_config():
        # EXACT matrix W2/W5/W6 production lane config (codex M1 MEDIUM:
        # the bare iter1009 preset omits the 2x biharmonic hyperdiff the
        # matrix adds, undercharging the production step).
        return iter1009_dual_target_config(
            n, div_damp_factor=8.0,
            hyperdiff_coeff=2.0 * cdgrid_hyperdiff_cube(n))

    with warnings.catch_warnings():
        # iter1009 warns off-C36; irrelevant for a timing bench.
        warnings.simplefilter("ignore", UserWarning)
        if core == "production":
            grid = create_cubed_sphere(n)
            model = FV3EdgeShallowWaterModel(grid, _prod_config())
        elif core == "production_duo":
            grid = create_cubed_sphere(n, use_duogrid=True)
            model = FV3EdgeShallowWaterModel(grid, _prod_config())
        elif core == "fb":
            grid = create_cubed_sphere(n, use_duogrid=True)
            model = FV3FBShallowWaterModel(grid, fb_m1_preset_config())
        else:
            raise ValueError(f"unknown core '{core}'; expected {_CORES}")
    return model, _w2_state(grid, model.cdgrid)


def _time_compiled(compiled, state, steps: int, warmup: int) -> float:
    """Steady-state per-step seconds of an AOT-compiled step executable."""
    for _ in range(warmup):
        state = compiled(state)
    jax.block_until_ready(state)
    t0 = time.perf_counter()
    out = state
    for _ in range(steps):
        out = compiled(out)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / steps


def bench_core(core: str, n: int, steps: int, warmup: int) -> dict:
    from legoesm.core.precision import cast_pytree
    model, state = _build(core, n)
    # step() returns storage precision; cast the IC to storage FIRST so the
    # AOT executable's input signature is the steady-state one (feeding a
    # float64 IC to an executable lowered for float32 storage would raise).
    state = cast_pytree(state, None, "storage")
    fn = jax.jit(lambda s: model.step(s, DT))
    t0 = time.perf_counter()
    compiled = fn.lower(state).compile()
    t_compile = time.perf_counter() - t0
    t_step = _time_compiled(compiled, state, steps, warmup)
    return {"core": core, "n": n, "per_step_s": t_step,
            "compile_s": t_compile}


def bench_fb_stages(n: int, steps: int, warmup: int) -> dict[str, float]:
    """Separately-jitted FB phase timings (attribution, not decomposition)."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import fb_m1_preset_config
    from legoesm.core.fv3_sw_core import (
        _EPS,
        _bgrid_ke_transport,
        _c_sw,
        _d_sw_native,
        _p_grad_c,
        fb_v_d_to_covariant,
        fb_v_d_to_orthogonal,
    )
    from legoesm.core.operators_cdgrid import interp_center_to_corner_a2b_ord4
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    grid = create_cubed_sphere(n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    from legoesm.core.precision import cast_pytree
    # Match the full-step storage dtype so stage times are comparable.
    st = cast_pytree(_w2_state(grid, cd), None, "storage")
    cfg = fb_m1_preset_config()
    h, u_d, h_s = st.h, st.u_d, st.h_s
    from legoesm import constants
    g = constants.g

    v_cov = fb_v_d_to_covariant(u_d, st.v_d, cd)
    h_star, uc, vc, ua, va = _c_sw(h, u_d, v_cov, h_s, cd, DT, g)
    dpx, dpy = _p_grad_c(h_star, h_s, cd, 0.5 * DT, g)
    uc, vc = uc + dpx, vc + dpy

    def dsw(hh, uu, vv, ucc, vcc, uaa, vaa):
        return _d_sw_native(
            hh, uu, vv, h_s, ucc, vcc, uaa, vaa, cd, DT, g,
            d2_bg=cfg.d2_bg, dddmp=cfg.dddmp, d4_bg=cfg.d4_bg,
            nord=cfg.nord, damp_v=cfg.damp_v, nord_v=cfg.nord_v)

    def pgf(h_new):
        gz_b = interp_center_to_corner_a2b_ord4(g * (h_new + h_s), cd)
        rdx_u = 1.0 / jnp.maximum(cd.dx_edge_y, _EPS)
        rdy_v = 1.0 / jnp.maximum(cd.dy_edge_x, _EPS)
        du = DT * rdx_u * (gz_b[:, :-1, :] - gz_b[:, 1:, :])
        dv = DT * rdy_v * (gz_b[:, :, :-1] - gz_b[:, :, 1:])
        return du, dv

    stage_fns = {
        "entry_conv (fb_v_d_to_covariant)":
            (lambda: (u_d, st.v_d),
             jax.jit(lambda u, v: fb_v_d_to_covariant(u, v, cd))),
        "c_sw":
            (lambda: (h, u_d, v_cov),
             jax.jit(lambda hh, uu, vv: _c_sw(hh, uu, vv, h_s, cd, DT, g))),
        "p_grad_c":
            (lambda: (h_star,),
             jax.jit(lambda hs_: _p_grad_c(hs_, h_s, cd, 0.5 * DT, g))),
        "d_sw chain (d_sw1..6)":
            (lambda: (h, u_d, v_cov, uc, vc, ua, va), jax.jit(dsw)),
        "  d_sw3 alone (_bgrid_ke_transport)":
            (lambda: (u_d, v_cov, uc, vc),
             jax.jit(lambda uu, vv, ucc, vcc: _bgrid_ke_transport(
                 uu, vv, ucc, vcc, cd, DT))),
        "phase4 PGF (a2b_ord4 + grad)":
            (lambda: (h,), jax.jit(pgf)),
        "exit_conv (fb_v_d_to_orthogonal)":
            (lambda: (u_d, v_cov),
             jax.jit(lambda u, v: fb_v_d_to_orthogonal(u, v, cd))),
    }

    out = {}
    for name, (argfn, fn) in stage_fns.items():
        args = argfn()
        compiled = fn.lower(*args).compile()
        for _ in range(warmup):
            jax.block_until_ready(compiled(*args))
        t0 = time.perf_counter()
        for _ in range(steps):
            r = compiled(*args)
        jax.block_until_ready(r)
        out[name] = (time.perf_counter() - t0) / steps
    return out


def run_bench(resolutions, steps: int = 30, warmup: int = 5,
              stages: bool = False,
              cores=_CORES) -> dict:
    # x64 sci-bench policy: without this the default fp32 policy leaves the
    # float32 IC uncast while x64 promotions make step() return float64 —
    # an AOT signature mismatch on the second call (cast_pytree never
    # downcasts).  fp64 everywhere is the representative sci-run config.
    # Restored in the finally below (codex M1 MEDIUM: a leaked global
    # policy makes later tests/callers order-dependent).
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    prev_policy = get_policy()
    # set_policy(fp64) also flips process-global jax_enable_x64 (codex M1
    # round-2 MEDIUM) — capture and restore BOTH.
    prev_x64 = bool(jax.config.read("jax_enable_x64"))
    set_policy(PrecisionPolicy.fp64())
    try:
        return _run_bench_inner(resolutions, steps, warmup, stages, cores)
    finally:
        set_policy(prev_policy)
        jax.config.update("jax_enable_x64", prev_x64)


def _run_bench_inner(resolutions, steps, warmup, stages, cores) -> dict:
    results = {"dt": DT, "steps": steps, "rows": [], "stages": {}}
    for res in resolutions:
        n = int(res.lstrip("Cc"))
        rows = {}
        for core in cores:
            r = bench_core(core, n, steps, warmup)
            rows[core] = r
            results["rows"].append(r)
            print(f"  C{n:<3} {core:<15} per-step {r['per_step_s']*1e3:8.2f} ms"
                  f"   (compile {r['compile_s']:.1f} s)")
        if "fb" in rows and "production" in rows:
            ratio = rows["fb"]["per_step_s"] / rows["production"]["per_step_s"]
            print(f"  C{n:<3} FB/production ratio = {ratio:.2f}x")
    if stages:
        n0 = int(resolutions[0].lstrip("Cc"))
        print(f"\nFB stage attribution at C{n0} (separately jitted):")
        st = bench_fb_stages(n0, steps, warmup)
        results["stages"] = st
        for k, v in st.items():
            print(f"  {k:<40} {v*1e3:8.2f} ms")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resolutions", type=str, default="C36,C48,C96",
                   help="Comma-separated cube resolutions (default C36,C48,C96)")
    p.add_argument("--steps", type=int, default=30,
                   help="Timed steps per lane (default 30)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Untimed warmup steps after AOT compile (default 5)")
    p.add_argument("--stages", action="store_true",
                   help="Also time the FB phases separately-jitted at the "
                        "first resolution")
    p.add_argument("--json-out", type=str, default=None,
                   help="Optional path to dump the results dict as JSON")
    args = p.parse_args(argv)
    if args.steps <= 0 or args.warmup < 0:
        p.error("--steps must be positive and --warmup non-negative")
    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    results = run_bench(resolutions, steps=args.steps, warmup=args.warmup,
                        stages=args.stages)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json_out}")
    return results


if __name__ == "__main__":
    main()
