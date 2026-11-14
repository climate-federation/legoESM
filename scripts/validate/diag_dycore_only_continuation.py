"""Dycore-only continuation of an AMIP run: does a feature grow with physics OFF?

    diag_dycore_only_continuation.py --config <deck> [--restart <checkpoint.npz>] --steps N

Answers "is this the dynamical core or the physics" for a growing local feature
(a hot column, a wind maximum) that a forward run shows.  Used 2026-09-22 to
attribute the CAM6 AMIP arm's stratospheric warming: the peak grew 340.4 -> 359.2 K
in six simulated hours with physics_fn=None, while the production-grid arm under the
same treatment was flat, which put the cause in the dycore rather than in radiation.

Builds the driver EXACTLY as run_amip does (same deck, same mesh, same vertical
coordinate, same dycore config), restores the checkpoint, then integrates with
``model.step(state, dt)`` and physics_fn=None.  Prints, every --stride steps,
the global T max and the temperature of the columns that were already hot in
the restart state.  NUMBERS ONLY -- no verdict.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def global_masses(state, area, sigma_coord):
    """Global mass [kg] of air, total water and each tracer, all in f64.

    Tracer mass is ``sum_cells area * sum_k q_k dp_k / g`` with ``dp`` the
    coordinate's own layer thickness at the state's ``p_s``; air mass is
    ``sum area * p_s / g``.  Water = the mass species the dry-mass fixer
    counts (no number concentrations).
    """
    from legoesm import constants
    from legoesm.core.conservation import WATER_MASS_SPECIES
    ps = np.asarray(state.p_s.data, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    dp = np.asarray(sigma_coord.layer_thickness_dp(ps), dtype=np.float64)
    out = {"air": float(np.sum(area * ps) / constants.g)}
    water = 0.0
    for name, f in (state.tracers or {}).items():
        q = np.asarray(getattr(f, "data", f), dtype=np.float64)
        m = float(np.sum(area[:, None] * q * dp) / constants.g)
        out[name] = m
        if name in WATER_MASS_SPECIES:
            water += m
    out["water"] = water
    return out


def water_tendency_terms(state, model):
    """Instantaneous global total-water tendency [kg/m2/day], split by term.

    Tracer mass per layer is ``q*dp/g``, so its rate is ``(dp*dq + q*ddp)/g``
    with ``ddp = dB*dp_s_dt``.  ``total`` uses the dycore's own tendencies
    (physics off).  ``horizontal`` is the advective horizontal tracer term
    with the matching layer-mass divergence ``-div(u*dp_edge)``; ``ps_del4``
    is the NET change in ``total`` from the surface-pressure hyperdiffusion
    (total minus total with ``nu_del4_ps = 0``, so any compensation in the
    tracer tendency is included); ``rest`` = total - both (vertical
    transport, filters).
    """
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.conservation import WATER_MASS_SPECIES
    from legoesm.core.operators_voronoi import (
        cell_to_edge_avg_3d, divergence_cell_3d)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        mpas_model_tendencies)
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        tracer_horizontal_advection)
    mesh, sc, cfg = model.mesh, model.sigma_coord, model.config
    ps = state.p_s.data
    dp = sc.layer_thickness_dp(ps)
    dB = jax.jvp(sc.layer_thickness_dp, (ps,), (jnp.ones_like(ps),))[1]  # exact d(dp)/d(p_s)
    names = [k for k in WATER_MASS_SPECIES if k in (state.tracers or {})]
    q = jnp.stack([state.tracers[k].data for k in names], axis=-1)
    u = state.u.data
    area = jnp.asarray(mesh.areaCell)
    day_per_area = 86400.0 / float(jnp.sum(area))

    def glob(rate):  # (nCells, nlev, nspecies) kg/kg*Pa/s -> kg/m2/day
        return float(jnp.sum(area[:, None, None] * rate) / constants.g * day_per_area)

    def total_rate(c):
        t = mpas_model_tendencies(state, mesh, sc, c)
        dq = jnp.stack([t.tracer_tendencies[k].data for k in names], axis=-1)
        return glob(dp[..., None] * dq + q * (dB * t.dp_s_dt.data[:, None])[..., None]), t

    total, t1 = total_rate(cfg)
    dqh = tracer_horizontal_advection(q, u, mesh)
    ddp_h = -divergence_cell_3d(u * cell_to_edge_avg_3d(dp, mesh), mesh)
    horizontal = glob(dp[..., None] * dqh + q * ddp_h[..., None])
    ps_del4 = ps_del4_air = 0.0
    if cfg.nu_del4_ps > 0:
        total0, t0 = total_rate(cfg._replace(nu_del4_ps=0.0))
        ps_del4 = total - total0
        # activity of the filter itself: area-mean |dp_s/dt| it adds [Pa/day]
        dps = t1.dp_s_dt.data - t0.dp_s_dt.data
        ps_del4_air = float(jnp.sum(area * jnp.abs(dps)) / jnp.sum(area) * 86400.0)
    return {"total": total, "horizontal": horizontal, "ps_del4": ps_del4,
            "rest": total - horizontal - ps_del4, "ps_del4_air_Pa_day": ps_del4_air}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", default=None)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--hot-level", type=int, default=3)
    ap.add_argument("--hot-threshold", type=float, default=300.0)
    ap.add_argument("--thermo-terms", action="store_true",
                    help="decompose dT/dt at the restart state and exit (no integration)")
    ap.add_argument("--tracer-mass", action="store_true",
                    help="also print global mass of air, total water and every tracer "
                         "(sum area*q*dp/g, f64) at each report")
    ap.add_argument("--inject-qv", type=float, default=0.0,
                    help="instrument self-test: multiply q_v by (1+X) after step "
                         "--inject-step; the tracer-mass report must show a +X jump")
    ap.add_argument("--inject-step", type=int, default=5)
    ap.add_argument("--water-terms", action="store_true",
                    help="print the instantaneous global water tendency split by term "
                         "at the first and last report")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="run_amip path flags (see config/amip/amip_production.sh)")
    return ap


def main():
    a = build_arg_parser().parse_args()

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    _cfg_keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**_cfg_keys)
    parser.set_defaults(_config_keys=frozenset(_cfg_keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    if a.restart:
        step0, day0 = driver.load_checkpoint(a.restart)
        print(f"restored step={step0} day={day0}", flush=True)
    else:
        print("cold start from the deck's initial condition (no --restart)", flush=True)

    state = driver.state
    dt = float(config.dycore.dt)
    T0 = np.asarray(state.T.data)
    hot = np.where(T0[:, a.hot_level] > a.hot_threshold)[0]
    print(f"dt={dt} nlev={T0.shape[1]} n_hot(k={a.hot_level},>{a.hot_threshold})={hot.size}", flush=True)

    def report(n, s):
        T = np.asarray(s.T.data)
        u = np.asarray(s.u.data)
        h = T[hot, a.hot_level] if hot.size else np.array([np.nan])
        print(f"step {n:5d}  t={n*dt/3600:7.3f} h  Tmax={T.max():8.2f}  "
              f"Tmax(k={a.hot_level})={T[:, a.hot_level].max():8.2f}  "
              f"hot_mean={h.mean():8.2f}  |u|max={np.abs(u).max():8.2f}", flush=True)
        if a.tracer_mass:
            m = global_masses(s, driver.model.mesh.areaCell, driver.model.sigma_coord)
            print("  mass " + " ".join(f"{k}={v:.17e}" for k, v in m.items()), flush=True)
        if a.water_terms and n in (0, a.steps):
            w = water_tendency_terms(s, driver.model)
            print("  water_terms kg/m2/day " + " ".join(f"{k}={v:+.6e}" for k, v in w.items()),
                  flush=True)

    report(0, state)

    if a.thermo_terms:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            mpas_model_tendencies,
        )
        m = driver.model
        _, th = mpas_model_tendencies(
            state, m.mesh, m.sigma_coord, m.config, return_thermo_terms=True)
        day = 86400.0
        cols = [("horiz_adv", th.horiz_adv), ("horiz_diff", th.horiz_diff),
                ("vert_adv", th.vert_adv), ("adiabatic_ps", th.adiabatic_ps),
                ("sigma_dot", th.sigma_dot)]
        print("  dT/dt decomposition [K/day], hot columns vs all columns", flush=True)
        for name, arr in cols:
            v = np.asarray(arr)
            sc = 1.0 if name == "sigma_dot" else day
            print(f"   {name:13s} hot(k={a.hot_level}) mean={v[hot, a.hot_level].mean()*sc:11.4f} "
                  f"max={v[hot, a.hot_level].max()*sc:11.4f}   all mean={v[:, a.hot_level].mean()*sc:11.4f}",
                  flush=True)
        return

    for n in range(1, a.steps + 1):
        state = driver.model.step(state, dt)
        if a.inject_qv and n == a.inject_step:
            report(n, state)
            qv = state.tracers["q_v"]
            state = state._replace(tracers={**state.tracers, "q_v": qv.replace(
                data=qv.data * (1.0 + a.inject_qv))})
            print(f"  injected q_v *= 1+{a.inject_qv:g} after step {n}", flush=True)
            report(n, state)
        if n % a.stride == 0 or n == a.steps:
            report(n, state)
            if not np.isfinite(np.asarray(state.T.data)).all():
                print("NON-FINITE", flush=True)
                break


if __name__ == "__main__":
    main()
