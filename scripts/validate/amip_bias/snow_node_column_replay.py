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
    rec: dict = {"forcing": [], "cells": None}

    def is_dyn(v):
        return isinstance(v, jax.core.Tracer)

    real = mll.step_multilayer_land_with_diagnostics

    @functools.wraps(real)
    def wrapped(state, forcing, *x, **k):
        treedef, stat, dyn, pos = split_static((state, forcing, x, k), is_dyn)
        stat = [np.asarray(v) if isinstance(v, jax.Array) else v for v in stat]

        def cb(*vals):
            vals = [np.asarray(v) for v in vals]
            if "args" not in rec:
                rec["args"] = (treedef, stat, vals, pos)
            rec["forcing"].append(join_static(treedef, stat, vals, pos)[1])
            print(f"land call {len(rec['forcing'])} ({time.time() - t0:.0f}s)",
                  flush=True)
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
    """Soil enthalpy relative to T_freeze [J/m2]: exact integral of the apparent
    heat capacity (sensible + freezing curtain) at fixed theta."""
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.land.soil_thermal import compute_apparent_heat_capacity
    tf = constants.T_freeze
    q = (jnp.arange(n) + 0.5) / n
    Tq = tf + (T - tf)[..., None] * q
    Cq = jax.vmap(lambda Tk: compute_apparent_heat_capacity(Tk, theta, hyd, th),
                  in_axes=-1, out_axes=-1)(Tq)
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
    try:
        step = jax.jit(lambda s, f: mll.step_multilayer_land_with_diagnostics(
            s, f, config, U_min, dt, **k))
        s = state0
        for f in forcings:
            s = step(s, f)[0]
        jax.effects_barrier()
    finally:
        mll.solve_soil_thermal = real
    if len(rec) != len(forcings):
        raise SystemExit(f"FATAL: {len(rec)} final solves for {len(forcings)} steps")
    res = np.stack([r[0] for r in rec])                 # (steps, ncol, nl) J/m2
    chk = np.stack([r[1] for r in rec])
    sub = max(float(r[2]) for r in rec)
    G = np.stack([r[3] for r in rec])
    span = len(forcings) * dt
    print(f"run {cap['run']} day {cap['day']}: {len(forcings)} land steps, dt {dt} s, "
          f"freeze/thaw {th.enable_freeze_thaw}, control {a.control}, quadrature {a.nq}")
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("run")
    c.add_argument("--day", type=int, required=True)
    c.add_argument("--calls", type=int, required=True)
    c.add_argument("--out", required=True)
    r = sp.add_parser("replay")
    r.add_argument("capture")
    r.add_argument("--days", type=int, required=True)
    r.add_argument("--lat-band", type=float, nargs=2, default=(45.0, 70.0))
    b = sp.add_parser("soil-budget")
    b.add_argument("capture")
    b.add_argument("--lat-band", type=float, nargs=2, default=(45.0, 70.0))
    b.add_argument("--nq", type=int, default=200)
    b.add_argument("--control", action="store_true")
    a = p.parse_args(argv)
    return {"capture": capture, "replay": replay, "soil-budget": soil_budget}[a.cmd](a)


if __name__ == "__main__":
    main()
