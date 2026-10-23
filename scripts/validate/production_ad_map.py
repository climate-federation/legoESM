#!/usr/bin/env python
"""Gradient map of the PRODUCTION AMIP model, component by component (review F9).

Builds the production deck (``config/amip/amip_production.yaml``: MPAS L32,
CAM6 suite, two-leaf multilayer land) through run_amip's own argument path at
a SMALL mesh, exports the run loop's step closures
(``ModelDriver.export_mpas_step``) and measures, per stage:

  components  each CAM6 module exactly as the combined physics calls it
              (radiation, ZM convection, CLUBB, Morrison, McFarlane GWD),
              recorded by spies on the module factories during one eager
              production physics call
  physics     the combined CAM6 physics (macmic loop, production cadence)
  params      the same physics REBUILT by the run loop's own builder from the
              production config with the active schemes' tier-1 (core) spec'd
              parameters spliced in as traced leaves (raw values at the spec
              defaults): per-parameter |dL/dtheta| and one FD check
  dycore      one MPAS dynamics step (step_pure, no physics)
  step        mpas_segment: N production steps (dycore + physics + post-step)
  land        the two-leaf multilayer land step (inputs: the marshalled
              atmosphere-to-surface forcing of the first step)
  eager       eager run loop vs mpas_segment over N steps (forward max|diff|;
              with land + ice skin on the segment holds their feedback)

Each gradient check: a scalar loss L = sum_leaf <w_leaf, out_leaf> with a fixed
random projection w (a norm would vanish where the output vanishes), the
reverse-mode directional derivative <dL/dx, v> against a central difference
(L(x+hv) - L(x-hv)) / 2h along a fixed random direction v.  Reported:
finite?, |grad|max, AD, FD, FD/AD.  x = (T, q_v) of the state unless noted.

Usage (sbatch; needs the Ginsburg forcing files, see amip_production.ginsburg.sh):
  production_ad_map.py --stage components -- <run_amip path flags>
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DECK = REPO / "config" / "amip" / "amip_production.yaml"


def _run_amip():
    spec = importlib.util.spec_from_file_location(
        "run_amip_ad_map", REPO / "scripts" / "run" / "run_amip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_driver(argv, resolution, n_steps):
    """run_amip.main's argument path (deck -> argv -> postprocess -> config),
    then a ModelDriver with ``days`` set to ``n_steps`` model steps."""
    ra = _run_amip()
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = ra.build_arg_parser()
    keys = load_yaml_config(str(DECK), parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    argv = list(argv) + ["--resolution", str(resolution)]
    args = parser.parse_args(argv)
    ra._resolve_land_model(args, parser, argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    args = ra._apply_aimip_classical_overrides(args)
    ra._require_full_physics_for_amip(args, parser)
    config = ra.build_config_from_args(args)
    dt = config.dycore.dt
    # +0.5 step: int(days*86400/dt) must not round 16 down to 15
    config = config._replace(days=(n_steps + 0.5) * dt / 86400.0)
    driver = ModelDriver(config)
    driver.setup()
    return driver


def _proj_loss(out, seed=0):
    leaves = [x for x in jax.tree_util.tree_leaves(out)
              if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact)]
    key = jax.random.PRNGKey(seed)
    tot = 0.0
    for i, x in enumerate(leaves):
        w = jax.random.normal(jax.random.fold_in(key, i), x.shape, x.dtype)
        tot = tot + jnp.sum(w * x)
    return tot


H_SWEEP = (1.0,)          # multiples of --h; --h-sweep widens it
RAD_FP64 = False          # --rad-fp64


def grad_check(f, x, v, h, name):
    """f: pytree -> scalar.  AD directional derivative vs central FD (at each
    step in ``H_SWEEP * h``: an FD that converges to AD as h shrinks is a
    smooth map; one that wanders is a kink / sampling noise, not a severed
    gradient)."""
    t0 = time.time()
    # jit both: op-by-op dispatch of a full physics graph maps one executable
    # per op and exhausts LLVM's code-section memory on CPU
    f = jax.jit(f)
    g = jax.jit(jax.grad(f))(x)
    t_ad = time.time() - t0
    gl = jax.tree_util.tree_leaves(g)
    finite = all(bool(jnp.all(jnp.isfinite(a))) for a in gl)
    gmax = max(float(jnp.max(jnp.abs(a))) for a in gl)
    ad = float(sum(jnp.vdot(a, b) for a, b in zip(gl, jax.tree_util.tree_leaves(v))))
    fds = {}
    for m in H_SWEEP:
        hh = h * m
        xp = jax.tree_util.tree_map(lambda a, b, _h=hh: a + _h * b, x, v)
        xm = jax.tree_util.tree_map(lambda a, b, _h=hh: a - _h * b, x, v)
        fds[f"{hh:.0e}"] = float((f(xp) - f(xm)) / (2.0 * hh))
    fd = fds[f"{h:.0e}"]
    rec = dict(name=name, finite=finite, grad_absmax=gmax, ad=ad, fd=fd,
               ratio=(fd / ad if ad != 0.0 else float("nan")), t_grad_s=t_ad)
    if len(fds) > 1:
        rec["fd_sweep"] = fds
    print("GRADCHECK " + json.dumps(rec), flush=True)
    return rec


def _state_x(state):
    return {"T": state.T.data, "q_v": state.tracers["q_v"].data}


def _with_x(state, x):
    tr = dict(state.tracers)
    tr["q_v"] = tr["q_v"].replace(data=x["q_v"])
    return state._replace(T=state.T.replace(data=x["T"]), tracers=tr)


_DIRECTION = "both"


def _direction(x, seed=1):
    key = jax.random.PRNGKey(seed)
    # 0.1 K in T, 1e-3 relative in q_v: well inside the linear range
    v = {"T": 0.1 * jax.random.normal(key, x["T"].shape, x["T"].dtype),
         "q_v": 1e-3 * x["q_v"] * jax.random.normal(
             jax.random.fold_in(key, 1), x["q_v"].shape, x["q_v"].dtype)}
    for k in v:
        if _DIRECTION not in ("both", k):
            v[k] = jnp.zeros_like(v[k])
    return v


_SPIED = ("make_radiation_physics", "make_convection_physics",
          "make_turbulence_physics", "make_microphysics_physics",
          "make_gwd_physics")


def install_spies(record):
    """Wrap combined's module factories so the FIRST call of each module
    records (fn, args, kwargs).  functools.wraps keeps every marker the
    combined dispatcher reads (_wants_forcing, ...)."""
    import legoesm.atmosphere.physics.combined as comb
    for name in _SPIED:
        orig = getattr(comb, name)

        def factory(*a, _orig=orig, _name=name, **k):
            if _name == "make_radiation_physics" and RAD_FP64:
                # diagnostic: undo the MPAS driver's fp32 RRTMGP solve so FD
                # is not swamped by single-precision roundoff
                a = (a[0]._replace(rrtmgp=a[0].rrtmgp._replace(
                    compute_fp32=False)),) + a[1:]
            fn = _orig(*a, **k)

            @functools.wraps(fn)
            def spy(*fa, **fk):
                if not any(isinstance(x, jax.core.Tracer)
                           for x in jax.tree_util.tree_leaves((fa, fk))):
                    record.setdefault(_name, (fn, fa, fk))
                return fn(*fa, **fk)
            return spy
        setattr(comb, name, factory)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("components", "physics", "params", "dycore",
                             "step", "land", "eager"))
    ap.add_argument("--resolution", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--h-sweep", action="store_true",
                    help="FD at h x {100, 10, 1, 0.1, 0.01}")
    ap.add_argument("--direction", choices=("both", "T", "q_v"),
                    default="both", help="perturb T, q_v or both")
    ap.add_argument("--rad-fp64", action="store_true",
                    help="RRTMGP solve in fp64 (diagnostic: FD vs fp32 roundoff)")
    ap.add_argument("--per-knob", action="store_true",
                    help="params stage: AD vs FD per knob instead of one direction")
    ap.add_argument("--only", default="",
                    help="components stage: comma list of modules to check")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = [x for x in a.rest if x != "--"]
    global H_SWEEP, _DIRECTION, RAD_FP64
    _DIRECTION = a.direction
    RAD_FP64 = a.rad_fp64
    if a.h_sweep:
        H_SWEEP = (100.0, 10.0, 1.0, 0.1, 0.01)
    print(f"stage={a.stage} res={a.resolution} n_steps={a.n_steps} "
          f"x64={jax.config.jax_enable_x64} backend={jax.default_backend()}",
          flush=True)
    rec = {}
    if a.stage == "components" or a.rad_fp64:
        install_spies(rec)
    t0 = time.time()
    drv = build_driver(rest, a.resolution, a.n_steps)
    exp = drv.export_mpas_step()
    print(f"built+exported in {time.time() - t0:.1f} s; ncol="
          f"{exp['state'].T.data.shape}, physics every "
          f"{exp['phys_update_steps']} steps, radiation every "
          f"{exp['rad_update_steps']}, land_k={exp['land_k']}", flush=True)
    model, state0, ps0 = exp["model"], exp["state"], exp["phys_state"]
    x0 = _state_x(state0)
    v = _direction(x0)
    forcing0 = dict(exp["forcing"])
    doy, sod = exp["calendar"](0)
    forcing0.update(day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(sod))
    mesh, sigma = model.mesh, model.sigma_coord

    if a.stage == "components":
        # one eager production physics call populates the spies
        _ = exp["physics_fn"](state0, mesh, sigma, phys_state=ps0, forcing=forcing0)
        print("recorded modules: " + ", ".join(sorted(rec)), flush=True)
        only = {x.strip() for x in a.only.split(",") if x.strip()}
        for name, (fn, fa, fk) in sorted(rec.items()):
            short = name.replace("make_", "").replace("_physics", "")
            if only and short not in only:
                continue
            st = fa[0]

            def f(x, _fn=fn, _fa=fa, _fk=fk, _st=st):
                return _proj_loss(_fn(_with_x(_st, x), *_fa[1:], **_fk))
            try:
                grad_check(f, _state_x(st), _direction(_state_x(st)), a.h,
                           short)
            except Exception as e:   # report, do not hide: a non-traceable module
                print(f"GRADCHECK_FAIL {name}: {type(e).__name__}: {e}"[:2000],
                      flush=True)
    elif a.stage == "physics":
        for label, pfn in (("physics_full", exp["physics_fn"]),
                           ("physics_norad", exp["physics_fn_norad"])):
            if pfn is None:
                continue

            def f(x, _p=pfn):
                return _proj_loss(_p(_with_x(state0, x), mesh, sigma,
                                     phys_state=ps0, forcing=forcing0))
            grad_check(f, x0, v, a.h, label)
    elif a.stage == "params":
        from legoesm.training.aimip_params import (
            _splice_scheme_overrides,
            aimip_active_scheme_keys,
            resolved_inactive_fields,
        )
        from legoesm.training.inert_params import assert_no_inert
        from legoesm.training.param_collector import build_trainable_params
        keys = aimip_active_scheme_keys(exp["phys_cfg"])
        # the production trainable set: knobs the resolved options never read
        # are frozen out by name, the rest must carry gradient (gate below)
        frozen = resolved_inactive_fields(
            exp["phys_cfg"], subgrid_orography=getattr(
                drv.grid, "subgrid_topo_stddev", None) is not None)
        tp = build_trainable_params(active_scheme_keys=keys, tier="core",
                                    dtype=jnp.float64,
                                    exclude=tuple(sorted(frozen)))
        print(f"active scheme keys: {sorted(keys)}; {len(tp.raw_values)} "
              f"core parameters; frozen out: {sorted(frozen)}", flush=True)

        def f(raw):
            trial = type(tp)(raw_values=raw, constraints=tp.constraints)
            cfg_ = _splice_scheme_overrides(exp["phys_cfg"], trial.to_overrides())
            pfn = exp["build_physics"](cfg_)
            return _proj_loss(pfn(state0, mesh, sigma, phys_state=ps0,
                                  forcing=forcing0))
        raw0 = dict(tp.raw_values)
        g = jax.jit(jax.grad(f))(raw0)
        for k in sorted(g):
            gk = np.asarray(g[k])
            print(f"PARAMGRAD {k} |g|={float(np.max(np.abs(gk))):.3e} "
                  f"finite={bool(np.all(np.isfinite(gk)))}", flush=True)
        dv = {k: jnp.asarray(0.1 * np.random.default_rng(5).standard_normal(
            np.shape(x))) for k, x in raw0.items()}
        if not a.per_knob:
            grad_check(f, raw0, dv, 1e-3, "physics_wrt_core_params")
        # per knob: reverse AD vs central FD at two steps (forward-mode jvp
        # is unavailable: the shared Thomas solver is a custom_vjp).  FD
        # stable in h and != AD -> the graph drops a dependence (stop_gradient,
        # degenerate branch, custom_vjp); FD that moves with h -> a kink.
        fj = jax.jit(f)
        for k in (sorted(raw0) if a.per_knob else ()):
            fds = []
            for hk in (1e-2, 1e-4):
                up = dict(raw0, **{k: raw0[k] + hk})
                dn = dict(raw0, **{k: raw0[k] - hk})
                fds.append(float((fj(up) - fj(dn)) / (2 * hk)))
            ad = float(np.sum(np.asarray(g[k])))
            print(f"PARAMFD {k} ad={ad:.6e} fd1e-2={fds[0]:.6e} "
                  f"fd1e-4={fds[1]:.6e}", flush=True)
        # gate LAST, so the diagnostics above are printed for a dead knob too
        assert_no_inert(g)
        print("NO_INERT_GATE passed", flush=True)
    elif a.stage == "dycore":
        tm = model.mass_target(state0)

        def f(x):
            return _proj_loss(model.step_pure(_with_x(state0, x), exp["dt"],
                                              target_mass=tm)[0])
        grad_check(f, x0, v, a.h, "dycore_step")
    elif a.stage == "step":
        from legoesm.driver.mpas_diff_step import mpas_segment
        for n in (1, a.n_steps):
            def f(x, _n=n):
                s, _ = mpas_segment(exp, _with_x(state0, x), ps0, _n,
                                    checkpoint=True)
                return _proj_loss(s)
            grad_check(f, x0, v, a.h, f"segment_{n}_steps")
    elif a.stage == "land":
        if exp["land_step"] is None:
            print("land: OFF in this config", flush=True)
            return 0
        # one eager production step populates the surface export the land
        # marshalling reads (sfc_diag), exactly as the loop's first land call
        model.step(state0, exp["dt"], physics_fn=exp["physics_fn"],
                   forcing=forcing0, phys_state=ps0)
        a2s = exp["marshal_land"]()
        doy64 = jnp.asarray(doy, dtype=jnp.float64)
        land0 = exp["land_state"]

        def f(z):
            out = exp["land_step"](land0, z, doy64, doy64 - 1.0,
                                   jnp.asarray(1979.0, jnp.float64))
            return _proj_loss(out[1:9])   # skin T, albedo, q_sfc, fluxes, stress
        fl = {k: getattr(a2s, k) for k in ("T_lowest", "q_lowest", "sw_down",
                                           "lw_down", "u_lowest", "v_lowest")}
        key = jax.random.PRNGKey(3)
        dv = {k: 1e-3 * (jnp.abs(x) + 1e-6) * jax.random.normal(
            jax.random.fold_in(key, i), x.shape, x.dtype)
            for i, (k, x) in enumerate(fl.items())}
        grad_check(lambda z: f(a2s._replace(**z)), fl, dv, 1.0, "land_step_wrt_a2s")
    elif a.stage == "eager":
        from legoesm.driver.mpas_diff_step import mpas_segment
        s_seg, _ = mpas_segment(exp, state0, ps0, a.n_steps)
        drv2 = build_driver(rest, a.resolution, a.n_steps)
        print("eager status: " + drv2.run(), flush=True)
        for k in ("T", "u", "p_s"):
            d = float(jnp.max(jnp.abs(getattr(drv2.state, k).data
                                      - getattr(s_seg, k).data)))
            print(f"EAGER_VS_SEGMENT {k} max|diff| = {d:.3e} "
                  f"(scale {float(jnp.max(jnp.abs(getattr(s_seg, k).data))):.3e})",
                  flush=True)
        dq = float(jnp.max(jnp.abs(drv2.state.tracers['q_v'].data
                                   - s_seg.tracers['q_v'].data)))
        print(f"EAGER_VS_SEGMENT q_v max|diff| = {dq:.3e}", flush=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
