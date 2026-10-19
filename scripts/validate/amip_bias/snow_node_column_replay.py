#!/usr/bin/env python3
"""Offline land-column test of the snow thermal node on a run's own winter state.

capture: relaunches the run from ``checkpoint_day_N`` with its own command line
(``mpas_onestep_param_grad.launch_argv``; single CPU process) and records every
call of the coupled land step (``step_multilayer_land_with_diagnostics``) for
``--calls`` land steps: the FULL argument set of the first call (land state,
config, per-column parameters, latitude, ...) and the atmospheric forcing of
every call.  Nothing in the model is changed; the wrapper only reads.

replay: integrates the captured land columns OFFLINE, forced by the captured
forcing cycled for ``--days`` days, twice from the same start state: snow node
off (the captured config) and on (the same config with
``thermal.snow_insulation=True``; the node is initialised by
``init_snow_temperature``, as on a restart).  The atmosphere does not respond:
this isolates what the switch does to the soil under identical forcing.
Prints, for land columns (land fraction > 0.5) in a latitude band that hold
snow at the start: soil enthalpy change (exact freezing-curve integral), mean
heat flux into the soil top, top-soil temperature, snow mass and skin.

Usage (run's checkout on PYTHONPATH, JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1, cwd =
repo root):
  snow_node_column_replay.py capture <run> --day N --calls K --out DIR
  snow_node_column_replay.py replay <capture.pkl> --days D [--lat-band 45 70]
  snow_node_column_replay.py soil-budget <capture.pkl> [--control] [--nq 200]
"""
from __future__ import annotations

import argparse
import functools
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class _Done(BaseException):
    pass


def split_static(tree, is_dynamic):
    """Flatten ``tree``; return (treedef, leaves with the dynamic ones set to
    None, the dynamic leaves in order, their positions).  Positions, not a
    placeholder object: None is itself a legitimate static leaf, and plain
    ints survive pickling."""
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=lambda v: v is None)
    pos = [i for i, v in enumerate(leaves) if is_dynamic(v)]
    stat = [None if i in set(pos) else v for i, v in enumerate(leaves)]
    return treedef, stat, [leaves[i] for i in pos], pos


def join_static(treedef, stat, dyn, pos):
    import jax
    leaves = list(stat)
    for i, v in zip(pos, dyn, strict=True):
        leaves[i] = v
    return jax.tree_util.tree_unflatten(treedef, leaves)


def capture(a):
    import jax
    import mpas_onestep_param_grad as H
    import legoesm.land.multilayer_land as mll
    from legoesm.driver.model_driver import ModelDriver

    t0 = time.time()
    out = Path(a.out)
    scratch = out / f"_launch_{a.run}_d{a.day:04d}"
    scratch.mkdir(parents=True, exist_ok=True)
    argv = H.launch_argv(a.run, a.day, scratch)
    while "--distributed" in argv:
        argv.remove("--distributed")
    # Long enough for the start-up call plus --calls full land steps (the run
    # is stopped as soon as they are recorded).
    argv[argv.index("--days") + 1] = str(int(np.ceil(a.calls * a.land_dt / 86400.0)) + 1)
    if a.config:                       # e.g. another run's state on THIS tree's deck
        argv[argv.index("--config") + 1] = a.config
    for m in a.path_map:               # moved input trees: same files, new place
        src, dst = m.split("=", 1)
        if not (src.startswith("/") and len(src) > 1 and dst.startswith("/")):
            raise SystemExit(f"--path-map needs absolute OLD=NEW paths, got {m!r}")
        argv = [v.replace(src, dst) for v in argv]
    if a.restart_copy:                 # a checkpoint copy away from the run's sidecars
        argv[argv.index("--restart-from") + 1] = a.restart_copy
    rec: dict = {"forcing": [], "cells": None}

    def is_dyn(v):
        return isinstance(v, jax.core.Tracer)

    real = mll.step_multilayer_land_with_diagnostics

    @functools.wraps(real)
    def wrapped(state, forcing, *x, **k):
        treedef, stat, dyn, pos = split_static((state, forcing, x, k), is_dyn)
        stat = [np.asarray(v) if isinstance(v, jax.Array) else v for v in stat]

        dt_call = float(x[2])          # static: the land step length of THIS call

        def cb(*vals):
            # The driver's first land call after a (re)start is a short
            # start-up step (dt = the atmosphere step); only full land steps
            # are kept, from the first one on (its state is the start state).
            if dt_call < a.land_dt:
                print(f"skipped start-up land call (dt {dt_call} s)", flush=True)
                return
            vals = [np.asarray(v) for v in vals]
            if "args" not in rec:
                rec["args"] = (treedef, stat, vals, pos)
            rec["forcing"].append(join_static(treedef, stat, vals, pos)[1])
            print(f"land call {len(rec['forcing'])} dt {dt_call} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        jax.debug.callback(cb, *dyn)
        return real(state, forcing, *x, **k)
    mll.step_multilayer_land_with_diagnostics = wrapped

    real_gather = mll.gather_land_columns

    def gather(tree, idx, ncol):
        if rec["cells"] is None:
            rec["cells"] = np.asarray(idx)
        return real_gather(tree, idx, ncol)
    mll.gather_land_columns = gather

    drv: dict = {}
    real_run = ModelDriver.run

    def run(self, *x, **k):
        drv["d"] = self
        return real_run(self, *x, **k)
    ModelDriver.run = run

    # Stop once K land calls are on the host.
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pe
    real_step = pe.MPASPrimitiveEquationModel.step

    def step(self, *x, **k):
        jax.effects_barrier()
        if len(rec["forcing"]) >= a.calls:
            raise _Done()
        return real_step(self, *x, **k)
    pe.MPASPrimitiveEquationModel.step = step

    sys.path.insert(0, str(Path.cwd() / "scripts" / "run"))
    import run_amip
    try:
        run_amip.main(argv)
    except _Done:
        pass
    jax.effects_barrier()
    if len(rec["forcing"]) < a.calls:
        raise SystemExit(f"FATAL: only {len(rec['forcing'])} land calls recorded")
    d = drv["d"]
    f_land = np.asarray(d._f_land, dtype=np.float64).reshape(-1)
    cells = rec["cells"] if rec["cells"] is not None else np.arange(f_land.size)
    f = out / f"snownode_cap_{a.run}_d{a.day:04d}.pkl"
    with open(f, "wb") as fh:
        pickle.dump({"run": a.run, "day": a.day, "argv": argv, "args": rec["args"],
                     "forcing": rec["forcing"][:a.calls],
                     "f_land_packed": f_land[cells]}, fh)
    print(f"wrote {f} ({time.time() - t0:.0f}s)")


def soil_enthalpy(T, theta, hyd, th, dz, n=200):
    """Soil enthalpy relative to T_freeze [J/m2]: exact integral of the heat
    capacity the solver uses at fixed theta -- apparent (sensible + freezing
    curtain) with freeze/thaw on, sensible only with it off."""
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.soil_thermal import (compute_apparent_heat_capacity,
                                           compute_heat_capacity)
    tf = constants.T_freeze
    q = (jnp.arange(n) + 0.5) / n
    Tq = tf + (T - tf)[..., None] * q
    C = ((lambda Tk: compute_apparent_heat_capacity(Tk, theta, hyd, th))
         if th.enable_freeze_thaw else (lambda Tk: compute_heat_capacity(theta, hyd, th)))
    Cq = jax.vmap(C, in_axes=-1, out_axes=-1)(Tq)
    return jnp.sum(jnp.mean(Cq, -1) * (T - tf) * dz, axis=-1)


def replay(a):
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.multilayer_land import (
        init_snow_temperature, land_skin_temperature,
        step_multilayer_land_with_diagnostics)
    from legoesm.land.soil_grid import make_soil_grid

    cap = pickle.load(open(a.capture, "rb"))
    treedef, stat, vals, pos = cap["args"]
    state0, _, x, k = join_static(treedef, stat, [jnp.asarray(v) for v in vals], pos)
    config, U_min, dt = x[0], x[1], x[2]
    forcings = [jax.tree_util.tree_map(jnp.asarray, f) for f in cap["forcing"]]
    lat = np.asarray(k["lat"])
    lat_deg = np.rad2deg(lat) if np.max(np.abs(lat)) <= np.pi / 2 + 1e-9 else lat
    th = config.thermal
    if abs(len(forcings) * dt - 86400.0) > 1e-6:
        raise SystemExit(f"FATAL: the captured forcing spans {len(forcings) * dt} s, "
                         "not one day; a replay day must be a real day")
    print(f"run {cap['run']} day {cap['day']}: {state0.T_soil.shape[0]} columns, "
          f"{len(forcings)} forcing calls, dt {dt} s, freeze/thaw "
          f"{th.enable_freeze_thaw}, scheme {type(config.surface_scheme).__name__}, "
          f"snow_scheme {config.snow_scheme}")
    if th.snow_insulation:
        raise SystemExit("FATAL: captured run already has the snow node on")
    sel = ((lat_deg >= a.lat_band[0]) & (lat_deg <= a.lat_band[1])
           & (cap["f_land_packed"] > 0.5) & (np.asarray(state0.snow_depth) > 0.0))
    print(f"selected {int(sel.sum())} snow-covered land columns, "
          f"{a.lat_band[0]}-{a.lat_band[1]}N (unweighted means; quasi-uniform mesh)")
    grid = make_soil_grid(config.soil_grid)
    dz = jnp.asarray(grid.dz)
    kw = {kk: v for kk, v in k.items()}

    def make_step(cfg):      # config closed over: it is static in the driver too
        return jax.jit(lambda s, f: step_multilayer_land_with_diagnostics(
            s, f, cfg, U_min, dt, **kw))

    n_steps = a.days * len(forcings)
    res = {}
    for arm, on in (("off", False), ("on", True)):
        cfg = config._replace(thermal=th._replace(snow_insulation=on))
        s = init_snow_temperature(state0) if on else state0
        step = make_step(cfg)
        H0 = soil_enthalpy(s.T_soil, s.theta_soil, cfg.hydraulics, th, dz)
        daily = []
        for i in range(n_steps):
            out = step(s, forcings[i % len(forcings)])
            s = out[0]
            if not np.all(np.isfinite(np.asarray(s.T_soil))):
                raise SystemExit(f"FATAL: {arm} non-finite soil T at step {i}")
            if (i + 1) % len(forcings) == 0:
                H = soil_enthalpy(s.T_soil, s.theta_soil, cfg.hydraulics, th, dz)
                daily.append(dict(
                    H=np.asarray(H - H0), T0=np.asarray(s.T_soil[:, 0]),
                    snow=np.asarray(s.snow_depth),
                    skin=np.asarray(land_skin_temperature(s))))
        res[arm] = daily
    S0 = np.asarray(state0.snow_depth)[sel]
    print("day | soil heat gain MJ/m2 off on on-off | flux into soil W/m2 off on |"
          " top-soil T K off on | snow kg/m2 off on (start %.1f) | skin K off on"
          % S0.mean())
    for d in range(a.days):
        o, n = res["off"][d], res["on"][d]
        secs = (d + 1) * len(forcings) * dt
        m = lambda v: float(np.mean(v[sel]))  # noqa: E731
        print(f"{d + 1:3d} | {m(o['H'])/1e6:7.2f} {m(n['H'])/1e6:7.2f} "
              f"{(m(n['H']) - m(o['H']))/1e6:7.2f} | {m(o['H'])/secs:7.2f} "
              f"{m(n['H'])/secs:7.2f} | {m(o['T0']):7.2f} {m(n['T0']):7.2f} | "
              f"{m(o['snow']):7.2f} {m(n['snow']):7.2f} | {m(o['skin']):7.2f} "
              f"{m(n['skin']):7.2f}")
    last_o, last_n = res["off"][-1], res["on"][-1]
    ratio = np.mean(last_n["snow"][sel]) / max(np.mean(last_o["snow"][sel]), 1e-12)
    print(f"snow mass on/off at end: {ratio:.3f} (validity band 0.8-1.2)")
    tf = constants.T_freeze
    print(f"columns with on-arm skin > T_freeze under snow: "
          f"{int(np.sum((last_n['skin'] > tf + 1e-9) & (last_n['snow'] > 0)))}")


def to_plain(x):
    """Code-version-independent copy: NamedTuples become tagged field dicts."""
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return {"__nt__": f"{type(x).__module__}:{type(x).__qualname__}",
                **{f: to_plain(v) for f, v in zip(x._fields, x)}}
    if isinstance(x, dict):
        return {kk: to_plain(v) for kk, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_plain(v) for v in x)
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return np.asarray(x)
    return x


def from_plain(x, log):
    """Rebuild with THIS checkout's classes; fields this checkout lacks are
    dropped and fields it adds take its defaults, both recorded in ``log``
    (class -> (dropped, defaulted)).  A new field without a default raises."""
    import importlib
    if isinstance(x, dict) and "__nt__" in x:
        mod, name = x["__nt__"].split(":")
        cls = importlib.import_module(mod)
        for part in name.split("."):
            cls = getattr(cls, part)
        given = {f: v for f, v in x.items() if f != "__nt__"}
        known = set(cls._fields)
        log[name] = (sorted(set(given) - known), sorted(known - set(given)))
        return cls(**{f: from_plain(v, log) for f, v in given.items() if f in known})
    if isinstance(x, dict):
        return {kk: from_plain(v, log) for kk, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(from_plain(v, log) for v in x)
    return x


def export(a):
    """Write a capture as plain data that another checkout can rebuild."""
    import jax.numpy as jnp
    cap = pickle.load(open(a.capture, "rb"))
    treedef, stat, vals, pos = cap["args"]
    state0, _, x, k = join_static(treedef, stat, [jnp.asarray(v) for v in vals], pos)
    with open(a.out, "wb") as fh:
        pickle.dump({"run": cap["run"], "day": cap["day"],
                     "f_land_packed": cap["f_land_packed"],
                     "state": to_plain(state0), "forcing": [to_plain(f) for f in cap["forcing"]],
                     "config": to_plain(x[0]), "U_min": float(x[1]), "dt": float(x[2]),
                     "kw": to_plain(dict(k))}, fh)
    print(f"wrote {a.out}")


TRACE_SWE = 1e-6      # kg/m2: "trace" pack, below any physical snow cover
HEALTHY_SWE = 1.0     # kg/m2


def trace_summary(T_snow, T_soil0, skin, swe, land, t_freeze, cohort=None):
    """Per-class extremes on land columns: trace (0 < SWE < TRACE_SWE), healthy
    (SWE > HEALTHY_SWE), and all land; runaway = T_snow > T_freeze + 5 K on a
    column that holds snow."""
    from legoesm import constants
    meltout_K = constants.L_f / constants.c_pi
    out = {}
    classes = [("trace", land & (swe > 0) & (swe < TRACE_SWE)),
               ("healthy", land & (swe > HEALTHY_SWE)), ("land", land)]
    if cohort is not None:      # columns trace at ANY step so far, melted out or not
        classes.insert(1, ("cohort", land & cohort))
    for name, m in classes:
        n = int(m.sum())
        snowy = m & (swe > 0)       # T_snow of a snow-free column mirrors soil
        out[name] = dict(
            n=n,
            runaway=int(np.sum(snowy & (T_snow > t_freeze + 5.0))),
            # trace pack melts out only above L_f/c_ice over freezing
            past_meltout=int(np.sum(snowy & (T_snow > t_freeze + meltout_K))),
            T_snow_max=float(T_snow[snowy].max()) if snowy.any() else float("nan"),
            T_soil0_max=float(T_soil0[m].max()) if n else float("nan"),
            skin_max=float(skin[m].max()) if n else float("nan"))
    return out


def _nanmax(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmax(v)) if (~np.isnan(v)).any() else float("nan")


def worst_of(rows):
    """Day aggregate of per-step summaries: the largest value of every entry
    (a node overheated at one step is melted back at the next, so a day-end
    look alone can miss it); n is the day-end count."""
    agg = {}
    for c in rows[-1]:
        agg[c] = {key: (rows[-1][c][key] if key == "n" else
                        max(r[c][key] for r in rows) if key in ("runaway", "past_meltout") else
                        _nanmax([r[c][key] for r in rows]))
                  for key in rows[-1][c]}
    return agg


def load_plain(path):
    """Rebuild an exported capture with THIS checkout's classes (rebuild log
    printed); refuses a node-on capture and forcing that is not one day."""
    import jax
    import jax.numpy as jnp
    cap = pickle.load(open(path, "rb"))
    log: dict = {}
    to_j = lambda t: jax.tree_util.tree_map(  # noqa: E731
        lambda v: jnp.asarray(v) if isinstance(v, np.ndarray) else v, t)
    state0 = to_j(from_plain(cap["state"], log))
    forcings = [to_j(from_plain(f, log)) for f in cap["forcing"]]
    config = from_plain(cap["config"], log)
    k = to_j(from_plain(cap["kw"], log))
    U_min, dt = cap["U_min"], cap["dt"]
    for name, (dropped, defaulted) in sorted(log.items()):
        if dropped or defaulted:
            print(f"rebuild {name}: dropped {dropped}, defaulted {defaulted}")
    if config.thermal.snow_insulation:
        raise SystemExit("FATAL: captured run already has the snow node on")
    if abs(len(forcings) * dt - 86400.0) > 1e-6:
        raise SystemExit(f"FATAL: the captured forcing spans {len(forcings) * dt} s")
    return cap, state0, forcings, config, k, U_min, dt, cap["f_land_packed"] > 0.5


def trace(a):
    """Snow node ON from an exported (node OFF) capture's start, as a run
    restarted without the node starts it; report runaway on trace-snow land
    columns.  The export is rebuilt with THIS checkout's classes, so the same
    columns, forcing and resolved land config run on either code version."""
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.multilayer_land import (
        init_snow_temperature, land_skin_temperature,
        step_multilayer_land_with_diagnostics)

    cap, state0, forcings, config, k, U_min, dt, land = load_plain(a.plain)
    sw0 = np.asarray(state0.snow_depth)
    start_trace = (sw0 > 0) & (sw0 < TRACE_SWE)     # followed even if perturbed
    if a.trace_swe_to is not None:
        sw = state0.snow_depth
        state0 = state0._replace(snow_depth=jnp.where(
            (sw > 0) & (sw < TRACE_SWE), a.trace_swe_to, sw))
    tf = constants.T_freeze
    kw = dict(k)
    res = {}
    for arm, on in (("off", False), ("on", True)):
        cfg = config._replace(thermal=config.thermal._replace(snow_insulation=on))
        s = init_snow_temperature(state0) if on else state0
        step = jax.jit(lambda s_, f_, c=cfg: step_multilayer_land_with_diagnostics(
            s_, f_, c, U_min, dt, **kw))
        rows, day = [], []
        cohort = start_trace.copy()
        for i in range(a.days * len(forcings)):
            s = step(s, forcings[i % len(forcings)])[0]
            T0 = np.asarray(s.T_soil[:, 0])
            # no snow node OFF: its snow-node measures are NaN / 0, not soil proxies
            Ts = np.asarray(s.T_snow) if on else np.full_like(T0, np.nan)
            sw = np.asarray(s.snow_depth)
            cohort |= (sw > 0) & (sw < TRACE_SWE)
            day.append(trace_summary(Ts, T0, np.asarray(land_skin_temperature(s)),
                                     sw, land, tf, cohort))
            if (i + 1) % len(forcings) == 0:
                rows.append(worst_of(day))
                day = []
        res[arm] = rows
    print(f"run {cap['run']} day {cap['day']}: {int(land.sum())} land columns, "
          f"dt {dt} s, trace SWE < {TRACE_SWE} kg/m2, healthy > {HEALTHY_SWE}"
          + (f", start trace SWE set to {a.trace_swe_to}" if a.trace_swe_to is not None else ""))
    print(f"day arm | class n(day end) | worst over the day's steps: runaway "
          f"past_meltout(>{constants.L_f / constants.c_pi:.0f} K) T_snow_max-Tf (snow columns) "
          "T_soil0_max skin_max (K); cohort = land columns trace at any step")
    for d in range(a.days):
        for arm in ("off", "on"):
            for c, v in res[arm][d].items():
                print(f"{d + 1:3d} {arm:3s} | {c:7s} {v['n']:6d} {v['runaway']:6d} "
                      f"{v['past_meltout']:6d} "
                      f"{v['T_snow_max'] - tf:8.1f} {v['T_soil0_max']:8.1f} {v['skin_max']:8.1f}")
    return res


def soil_budget(a):
    """Energy budget of the FINAL soil-thermal solve of every land step, node
    OFF (main's bulk path), on the captured columns over the captured day.

    Per column, per land step, all in J/m2 (W/m2 = / dt):
      residual = true soil enthalpy change - (G_in + Q_geo + sum(source)) * dt
      true     = sum_layers dz * int_{T0}^{T1} C_app(T; theta1) dT
    with T0 the start-of-step and T1 the solved soil temperature, theta1 the
    post-Richards water, C_app the apparent heat capacity (sensible + latent
    freezing curtain), G_in the ground flux the solve receives (positive INTO
    the soil), Q_geo the bottom flux (positive into the soil) and ``source``
    the latent heat of the ice change Richards made at fixed T0
    (``moisture_fusion_heat_source``, positive = heating).  So the identity
    is H(T1, theta1) - H(T0, theta0) = (G_in + Q_geo) dt, water moving at
    fixed temperature.  SIGN: residual > 0 = the scheme CREATED energy (the
    soil gained more than it was given); < 0 = destroyed.
    Per layer the residual is true_l - lin_l, lin_l = sum_k C_app(T_k) dz
    (T_{k+1} - T_k) over the sub-steps; sum_l lin_l equals the flux exactly
    (printed as an instrument check), so the per-layer split is complete.
    ``--control``: freeze/thaw OFF and no snow -- C is constant, the residual
    must vanish (instrument check)."""
    import jax
    import jax.numpy as jnp
    import legoesm.land.multilayer_land as mll
    from legoesm.land.soil_grid import make_soil_grid
    from legoesm.land.soil_thermal import (compute_apparent_heat_capacity,
                                           compute_heat_capacity)

    cap = pickle.load(open(a.capture, "rb"))
    treedef, stat, vals, pos = cap["args"]
    state0, _, x, k = join_static(treedef, stat, [jnp.asarray(v) for v in vals], pos)
    config, U_min, dt = x[0], x[1], x[2]
    forcings = [jax.tree_util.tree_map(jnp.asarray, f) for f in cap["forcing"]]
    if config.thermal.snow_insulation:
        raise SystemExit("FATAL: captured run has the snow node on")
    if abs(len(forcings) * dt - 86400.0) > 1e-6:
        raise SystemExit(f"FATAL: the captured forcing spans {len(forcings) * dt} s, "
                         "not one day")
    if a.control:
        config = config._replace(thermal=config.thermal._replace(enable_freeze_thaw=False))
        state0 = state0._replace(snow_depth=jnp.zeros_like(state0.snow_depth))
        forcings = [f._replace(precip_snow=jnp.zeros_like(f.precip_snow)) for f in forcings]
    th = config.thermal
    dz = jnp.asarray(make_soil_grid(config.soil_grid).dz)
    lat = np.asarray(k["lat"])
    lat_deg = np.rad2deg(lat) if np.max(np.abs(lat)) <= np.pi / 2 + 1e-9 else lat
    rec = []
    real = mll.solve_soil_thermal

    def cap_C(T, theta, hyd, thc):
        return (compute_apparent_heat_capacity(T, theta, hyd, thc)
                if thc.enable_freeze_thaw else compute_heat_capacity(theta, hyd, thc))

    def wrapped(T0, theta, grid, hyd, thc, G, dt_, **kw):
        T1 = real(T0, theta, grid, hyd, thc, G, dt_, **kw)
        if "layer_source" not in kw:           # Picard tentative solve, not the update
            return T1
        if kw.get("surface_conductance") is not None:
            raise SystemExit("FATAL: semi-implicit surface; budget not defined here")
        src, n = kw["layer_source"], kw.get("n_substeps", 1)
        T, lin = T0, 0.0
        for _ in range(n):                     # the solve's own sub-steps, one by one
            Tn = real(T, theta, grid, hyd, thc, G, dt_ / n, layer_source=src)
            lin = lin + cap_C(T, theta, hyd, thc) * dz * (Tn - T)
            T = Tn
        q = (jnp.arange(a.nq) + 0.5) / a.nq
        Tq = T0[..., None] + (T1 - T0)[..., None] * q
        Cq = jax.vmap(lambda Tk: cap_C(Tk, theta, hyd, thc), in_axes=-1,
                      out_axes=-1)(Tq)
        true = jnp.mean(Cq, -1) * (T1 - T0) * dz
        flux = (G + thc.Q_geothermal
                + (0.0 if src is None else jnp.sum(src, axis=-1))) * dt_
        jax.debug.callback(lambda *v: rec.append([np.asarray(u) for u in v]),
                           true - lin, jnp.sum(lin, -1) - flux,
                           jnp.max(jnp.abs(T - T1)), G)
        return T1

    mll.solve_soil_thermal = wrapped
    n_sub0 = mll.FINAL_THERMAL_SUBSTEPS
    if a.substeps:                  # attribution test: the linearisation error
        mll.FINAL_THERMAL_SUBSTEPS = a.substeps   # must fall with more sub-steps
    try:
        step = jax.jit(lambda s, f: mll.step_multilayer_land_with_diagnostics(
            s, f, config, U_min, dt, **k))
        s = state0
        held = []
        for f in forcings:
            out = step(s, f)
            s = out[0]
            h = out[-1].held
            held.append(np.zeros(s.T_soil.shape[0], bool) if h is None
                        else np.asarray(h).astype(bool))
        jax.effects_barrier()
    finally:
        mll.solve_soil_thermal = real
        mll.FINAL_THERMAL_SUBSTEPS = n_sub0
    if len(rec) != len(forcings):
        raise SystemExit(f"FATAL: {len(rec)} final solves for {len(forcings)} steps")
    res = np.stack([r[0] for r in rec])                 # (steps, ncol, nl) J/m2
    # A held column's update is rejected and reverted: its solve changed nothing.
    res = np.where(np.stack(held)[..., None], 0.0, res)
    chk = np.stack([r[1] for r in rec])
    sub = max(float(r[2]) for r in rec)
    G = np.stack([r[3] for r in rec])
    span = len(forcings) * dt
    print(f"run {cap['run']} day {cap['day']}: {len(forcings)} land steps, dt {dt} s, "
          f"freeze/thaw {th.enable_freeze_thaw}, control {a.control}, quadrature {a.nq}, "
          f"final-solve sub-steps {a.substeps or n_sub0}")
    print(f"instrument: max |sum lin - flux| {np.max(np.abs(chk)) / dt:.2e} W/m2; "
          f"max |sub-stepped by hand - solve| {sub:.2e} K")
    land = cap["f_land_packed"] > 0.5
    snow0 = np.asarray(state0.snow_depth) > 0.0
    band = land & (lat_deg >= a.lat_band[0]) & (lat_deg <= a.lat_band[1])
    tot = res.sum(-1).sum(0) / span                      # (ncol,) W/m2, daily mean
    for name, m in (("all land", land), (f"{a.lat_band[0]:.0f}-{a.lat_band[1]:.0f}N land", band),
                    ("  of which snow", band & snow0), ("  of which no snow", band & ~snow0)):
        if not m.any():
            continue
        print(f"{name:22s} n={int(m.sum()):6d} daily-mean residual {tot[m].mean():+8.3f} W/m2 "
              f"(|.| {np.abs(tot[m]).mean():7.3f}; >0 in {100 * np.mean(tot[m] > 0):5.1f}%) "
              f"mean G_in {G[:, m].mean():+8.2f}")
    if band.any():
        lay = res[:, band, :].sum(0).mean(0) / span
        print("by layer (band, W/m2): " + " ".join(f"{v:+.3f}" for v in lay))
        worst = np.max(np.abs(res[:, band, :].sum(-1)), axis=1) / dt
        print(f"largest single-step |residual| in band: {worst.max():.2f} W/m2")
        step_r = res[:, band, :].sum(-1) / dt                # (steps, ncol) W/m2
        hb = np.stack(held)[:, band]                         # rejected-and-reverted
        acc = ~hb.any(0)                                     # never held this day
        day_r = res[:, band, :].sum(-1).sum(0) / span        # (ncol,) W/m2
        i = int(np.argmax(np.where(acc, np.abs(day_r), -1.0)))
        print(f"held column-steps in band: {int(hb.sum())}; largest 24-h |residual| of "
              f"a never-held column: {abs(day_r[i]):.3f} W/m2 (lat "
              f"{lat_deg[band][i]:.1f}, snow {bool(snow0[band][i])})")
        print("band column-steps with |residual| > 1 / 10 W/m2: "
              f"{100 * np.mean(np.abs(step_r) > 1):.2f}% / "
              f"{100 * np.mean(np.abs(step_r) > 10):.3f}%; >0 share of those "
              f"> 1 W/m2: {100 * np.mean(step_r[np.abs(step_r) > 1] > 0):.0f}%")


def node_budget(dE_node, dH_soil, G, Q_geo, src_sum, dt):
    """Residual [W/m2] of the final node+soil solve: stored - supplied.
    dE_node = c_ice S (T1 - T0) [J/m2]; dH_soil = soil enthalpy change at
    the solve's water [J/m2]; G the flux the solve receives (+ into the
    column), Q_geo the bottom flux, src_sum the layer sources [W/m2].
    > 0 = the solve CREATED energy."""
    return (dE_node + dH_soil) / dt - (G + Q_geo + src_sum)


def pin_melting_node(T_s0, S, T_g0, theta, grid, hyd, thc, G, dt, T_s1, T_g1, src,
                     n_substeps, nq):
    """PROTOTYPE of a fix (replay only): where the solved node ends above
    T_freeze on a pack, redo the soil with the node held at T_freeze (infinite
    node capacity) and give the node the energy it absorbed as melt enthalpy,
    T_node = T_freeze + E_melt / (c_ice S), which the model's next-step
    ``melt_snow_node_excess`` turns into melt (leftover heat to the soil once
    the pack is gone).  Energy: E_melt = supplied - soil enthalpy change -
    c_ice S (T_freeze - T_s0)."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.soil_thermal import snow_thermal_conductivity, solve_snow_soil_thermal
    tf = constants.T_freeze
    rho = thc.snow_bulk_density_kg_m3
    R = S / rho / snow_thermal_conductivity(rho)
    _, Tp_g = solve_snow_soil_thermal(
        jnp.full_like(S, tf)[:, None], jnp.full_like(S, 1e30)[:, None],
        jnp.zeros_like(T_g0[:, :0]), R, jnp.ones_like(S), T_g0, theta, grid,
        hyd, thc, G, dt, layer_source=src, n_substeps=n_substeps)
    dz = jnp.asarray(grid.dz)
    supplied = (G + thc.Q_geothermal + (0.0 if src is None else jnp.sum(src, -1))) * dt
    E_m = (supplied - (soil_enthalpy(Tp_g, theta, hyd, thc, dz, nq)
                       - soil_enthalpy(T_g0, theta, hyd, thc, dz, nq))
           - constants.c_pi * S * (tf - T_s0))
    use = (S > 0) & (T_s1 > tf) & (E_m > 0)
    # A pack that melts entirely this step: the node carries exactly its melt
    # (next-step melt leaves no leftover) and the soil takes the rest of the
    # flux as a bare-soil solve NOW -- routed through the next pack's top it
    # would be absorbed again as melt and cycle.
    melt_E = constants.L_f * S + constants.c_pi * S * (tf - T_s0)
    full = use & (E_m >= constants.L_f * S)
    T_new = jnp.where(full, tf + constants.L_f / constants.c_pi,
                      jnp.where(use, tf + E_m / jnp.where(use, constants.c_pi * S, 1.0), T_s1))
    from legoesm.land.soil_thermal import solve_soil_thermal
    Tb_g = solve_soil_thermal(T_g0, theta, grid, hyd, thc, G - melt_E / dt, dt,
                              layer_source=src, n_substeps=n_substeps)
    Tp_g = jnp.where(full[:, None], Tb_g, Tp_g)
    return T_new, jnp.where(use[:, None], Tp_g, T_g1)


def mechanism(a):
    """Snow node ON vs OFF on an exported capture, with per-step records of
    the FINAL thermal solve: the ground flux it receives, the node+soil
    energy budget (ON) and soil warming, by column class.  ``--uncap``: the
    canopy's Picard loop sees the node's own temperature instead of
    min(node, T_freeze) (replay-only patch of the canopy's soil callback);
    ``--trace-swe-to`` rescales the start SWE of trace columns."""
    import jax
    import jax.numpy as jnp
    import legoesm.land.multilayer_land as mll
    from legoesm import constants
    from legoesm.land.soil_grid import make_soil_grid

    cap, state0, forcings, config, k, U_min, dt, land = load_plain(a.plain)
    sw0 = np.asarray(state0.snow_depth)
    start_trace = land & (sw0 > 0) & (sw0 < TRACE_SWE)
    healthy = land & (sw0 > HEALTHY_SWE)
    if a.trace_swe_to is not None:
        state0 = state0._replace(snow_depth=jnp.where(
            jnp.asarray(start_trace), a.trace_swe_to, state0.snow_depth))
    if a.no_freeze_thaw:        # instrument control: constant heat capacity
        config = config._replace(thermal=config.thermal._replace(enable_freeze_thaw=False))
    th = config.thermal
    dz = jnp.asarray(make_soil_grid(config.soil_grid).dz)
    tf = constants.T_freeze
    real_node, real_soil = mll.solve_snow_node_soil_thermal, mll.solve_soil_thermal
    real_canopy = mll.compute_two_leaf_canopy_fluxes
    rec, last = [], {}

    def node(T_s0, swe, T_g0, theta, grid, hyd, thc, G, dt_, **kw):
        T_s1, T_g1 = real_node(T_s0, swe, T_g0, theta, grid, hyd, thc, G, dt_, **kw)
        if "layer_source" not in kw:            # canopy Picard pass
            last.update(Ts=T_s1, snow=swe)
            return T_s1, T_g1
        src = kw["layer_source"]
        S = jnp.maximum(swe, 0.0)
        if a.pin_melt:
            T_s1, T_g1 = pin_melting_node(T_s0, S, T_g0, theta, grid, hyd, thc, G, dt_,
                                          T_s1, T_g1, src, kw.get("n_substeps", 1), a.nq)
        dE = jnp.where(S > 0, constants.c_pi * S * (T_s1 - T_s0), 0.0)
        dH = (soil_enthalpy(T_g1, theta, hyd, thc, dz, a.nq)
              - soil_enthalpy(T_g0, theta, hyd, thc, dz, a.nq))
        ssum = 0.0 if src is None else jnp.sum(src, -1)
        r = node_budget(dE, dH, G, thc.Q_geothermal, ssum, dt_)
        jax.debug.callback(lambda *v: rec.append([np.asarray(u) for u in v]),
                           G, r, dH / dt_)
        return T_s1, T_g1

    def soil(T0, theta, grid, hyd, thc, G, dt_, **kw):
        T1 = real_soil(T0, theta, grid, hyd, thc, G, dt_, **kw)
        if "layer_source" in kw and not on_arm[0]:   # OFF arm final solve
            src = kw["layer_source"]
            dH = (soil_enthalpy(T1, theta, hyd, thc, dz, a.nq)
                  - soil_enthalpy(T0, theta, hyd, thc, dz, a.nq))
            r = node_budget(0.0, dH, G, thc.Q_geothermal,
                            0.0 if src is None else jnp.sum(src, -1), dt_)
            jax.debug.callback(lambda *v: rec.append([np.asarray(u) for u in v]),
                               G, r, dH / dt_)
        return T1

    def canopy(*args, soil_thermal_fn, **kw):
        if a.uncap and on_arm[0]:
            inner = soil_thermal_fn

            def soil_thermal_fn(G, dt_):         # noqa: F811
                capped = inner(G, dt_)
                return jnp.where(last["snow"] > 0.0, last["Ts"], capped)
        return real_canopy(*args, soil_thermal_fn=soil_thermal_fn, **kw)

    on_arm = [False]
    mll.solve_snow_node_soil_thermal, mll.solve_soil_thermal = node, soil
    mll.compute_two_leaf_canopy_fluxes = canopy
    out = {}
    try:
        for arm, on in (("off", False), ("on", True)):
            on_arm[0] = on
            rec.clear()
            cfg = config._replace(thermal=th._replace(snow_insulation=on))
            s = mll.init_snow_temperature(state0) if on else state0
            step = jax.jit(lambda s_, f_, c=cfg: mll.step_multilayer_land_with_diagnostics(
                s_, f_, c, U_min, dt, **dict(k)))
            T0max, Tsmax = [], []
            for i in range(a.days * len(forcings)):
                s = step(s, forcings[i % len(forcings)])[0]
                T0max.append(np.asarray(s.T_soil[:, 0]))
                if on:
                    Tsmax.append(np.where(np.asarray(s.snow_depth) > 0,
                                          np.asarray(s.T_snow), -np.inf))
            jax.effects_barrier()
            if len(rec) != a.days * len(forcings):
                raise SystemExit(f"FATAL: {arm}: {len(rec)} final solves recorded "
                                 f"for {a.days * len(forcings)} steps")
            G = np.stack([r[0] for r in rec])
            res = np.stack([r[1] for r in rec])
            heat = np.stack([r[2] for r in rec])
            out[arm] = dict(G=G, res=res, heat=heat, T0max=np.max(T0max, 0),
                            Tsmax=np.max(Tsmax, 0) if on else None,
                            snow_end=np.asarray(s.snow_depth))
    finally:
        mll.solve_snow_node_soil_thermal, mll.solve_soil_thermal = real_node, real_soil
        mll.compute_two_leaf_canopy_fluxes = real_canopy
    n_on = out["on"]
    runaway = land & (n_on["Tsmax"] > tf + 5.0)
    print(f"run {cap['run']} day {cap['day']}: {a.days} d, dt {dt} s, uncap {a.uncap}, "
          f"pin_melt {a.pin_melt}, freeze/thaw {th.enable_freeze_thaw}, "
          f"trace SWE set to {a.trace_swe_to}; trace < {TRACE_SWE}, healthy > "
          f"{HEALTHY_SWE} kg/m2 at start; runaway = ON T_snow > T_freeze + 5 K at any step")
    print("class n | mean G into column W/m2 off on | mean soil heating W/m2 off on | "
          "max top-soil T K off on | max T_snow-Tf K | budget residual mean |.| W/m2 off on | "
          "end snow kg/m2 off on" + (" (pin: T_snow carries melt enthalpy)" if a.pin_melt else ""))
    for name, m in (("trace", start_trace), ("healthy", healthy),
                    ("runaway", runaway), ("all land", land)):
        if not m.any():
            continue
        o = out["off"]
        print(f"{name:8s} {int(m.sum()):6d} | {o['G'][:, m].mean():+8.2f} "
              f"{n_on['G'][:, m].mean():+8.2f} | {o['heat'][:, m].mean():+8.2f} "
              f"{n_on['heat'][:, m].mean():+8.2f} | {o['T0max'][m].max():7.1f} "
              f"{n_on['T0max'][m].max():7.1f} | {np.max(n_on['Tsmax'][m]) - tf:7.1f} | "
              f"{np.abs(o['res'][:, m]).mean():.2e} {np.abs(n_on['res'][:, m]).mean():.2e} | "
              f"{o['snow_end'][m].mean():8.3f} {n_on['snow_end'][m].mean():8.3f}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("run")
    c.add_argument("--day", type=int, required=True)
    c.add_argument("--calls", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--restart-copy", default="",
                   help="restart from this copy of the day's checkpoint instead "
                        "(e.g. without the run's CMOR accumulator sidecars)")
    c.add_argument("--config", default="",
                   help="replace the run's --config deck (default: keep it)")
    c.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite this path prefix in the run's command line "
                        "(an input tree that moved)")
    c.add_argument("--land-dt", type=float, default=1800.0,
                   help="full land step [s]; shorter (start-up) calls are skipped")
    r = sp.add_parser("replay")
    r.add_argument("capture")
    r.add_argument("--days", type=int, required=True)
    r.add_argument("--lat-band", type=float, nargs=2, default=(45.0, 70.0))
    e = sp.add_parser("export")
    e.add_argument("capture")
    e.add_argument("--out", required=True)
    t = sp.add_parser("trace")
    t.add_argument("plain", help="output of the export subcommand")
    t.add_argument("--days", type=int, required=True)
    t.add_argument("--trace-swe-to", type=float, default=None,
                   help="set start SWE of trace columns to this value [kg/m2]")
    m = sp.add_parser("mechanism")
    m.add_argument("plain", help="output of the export subcommand")
    m.add_argument("--days", type=int, required=True)
    m.add_argument("--trace-swe-to", type=float, default=None,
                   help="set start SWE of trace columns to this value [kg/m2]")
    m.add_argument("--uncap", action="store_true",
                   help="canopy sees the node temperature, not min(node, T_freeze)")
    m.add_argument("--pin-melt", action="store_true",
                   help="PROTOTYPE fix: node held at T_freeze while melting (replay only)")
    m.add_argument("--no-freeze-thaw", action="store_true",
                   help="control: freeze/thaw off, both budgets must close")
    m.add_argument("--nq", type=int, default=200)
    b = sp.add_parser("soil-budget")
    b.add_argument("capture")
    b.add_argument("--lat-band", type=float, nargs=2, default=(45.0, 70.0))
    b.add_argument("--nq", type=int, default=200)
    b.add_argument("--control", action="store_true")
    b.add_argument("--substeps", type=int, default=0,
                   help="override the final solve's freeze/thaw sub-steps (0 = model's)")
    a = p.parse_args(argv)
    return {"capture": capture, "replay": replay, "export": export, "trace": trace,
            "soil-budget": soil_budget, "mechanism": mechanism}[a.cmd](a)


if __name__ == "__main__":
    main()
