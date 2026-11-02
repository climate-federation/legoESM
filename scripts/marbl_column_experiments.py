"""Differentiable-MARBL column experiments (BATS, WOA initial state, KPP column physics).

Three experiments, all starting from the same spun-up column state:

  grad-check   reverse-mode (jax.grad) and forward-mode (jax.jvp) derivatives of
               annual-mean NPP with respect to MARBL parameters, compared with
               central finite differences over a range of step sizes.
  sensitivity  one reverse pass per diagnostic gives the elasticity
               d ln(metric) / d ln(parameter) of NPP, POC export at 100 m, surface
               Chl, surface NO3 and the air-sea CO2 flux for every float
               parameter in MarblConfig; written as a ranked table.
  twin         identical-twin calibration: synthetic daily "observations" from the
               default model, the K most influential parameters perturbed, then
               recovered with Adam on bounded log-scale factors
               (scale = exp(B * tanh(phi))).

Parameters are perturbed as multiplicative log-scale factors theta (value * exp(theta));
tuple-valued (per-PFT) parameters get one factor shared by all PFTs.

Usage (repo root, JAX_ENABLE_X64=1; needs scripts/run_marbl_column.py):
    python scripts/marbl_column_experiments.py spinup      --years 2
    python scripts/marbl_column_experiments.py grad-check  --days 30
    python scripts/marbl_column_experiments.py sensitivity --days 365
    python scripts/marbl_column_experiments.py twin        --days 365 --k 6 --iters 60
Outputs: results/marbl_column_experiments/
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.column.model import ColumnConfig, initial_state, make_column_grid, run_column_raw
from legoesm.ocean.biogeochemistry.marbl import MarblConfig, MarblTracers, TRACER_NAMES

OUT = Path("results/marbl_column_experiments")
DAY = 86400.0
S2Y_GC = DAY * 365 * 12e-3            # mmol C/m2/s -> g C/m2/yr
METRICS = ("NPP", "POC100", "Chl_sfc", "NO3_sfc", "fCO2")

_spec = importlib.util.spec_from_file_location("rmc", Path(__file__).with_name("run_marbl_column.py"))
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


# --------------------------------------------------------------------- set-up
def setup(dt=3600.0, nlev=60):
    grid = make_column_grid(rmc.LAT, rmc.LON, rmc.DEPTH, n_levels=nlev)
    grid = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a, jnp.float64) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a,
        grid)
    zt = np.asarray(grid.zt, np.float64)
    woa = rmc.load_woa(zt, rmc.WOA_DIR)
    forcing, T_clim, S_clim = rmc.make_forcing(zt, woa)
    st0 = initial_state(jnp.asarray(T_clim[0]), jnp.asarray(S_clim[0]), rmc.initial_bgc(zt, woa))
    cfg = ColumnConfig(dt=dt)
    return grid, forcing, cfg, st0, int(round(DAY / dt))


def tunable_params(mcfg):
    """Float or tuple-of-float fields with a non-zero value -> {name: kind}."""
    out = {}
    for k in mcfg._fields:
        v = getattr(mcfg, k)
        if isinstance(v, bool):
            continue
        if isinstance(v, float) and v != 0.0:
            out[k] = "scalar"
        elif isinstance(v, tuple) and v and all(isinstance(x, float) for x in v) and any(x != 0.0 for x in v):
            out[k] = "per_pft"
    return out


def scaled_config(mcfg, names, theta):
    """MarblConfig with each named parameter multiplied by exp(theta_i)."""
    upd = {}
    for i, k in enumerate(names):
        v, f = getattr(mcfg, k), jnp.exp(theta[i])
        upd[k] = tuple(x * f for x in v) if isinstance(v, tuple) else v * f
    return mcfg._replace(**upd)


def metrics_from_run(states, diags):
    """Window means of the five diagnostics."""
    b = states.bgc
    return jnp.stack([
        jnp.mean(jnp.sum(diags.NPP, axis=-1)) * S2Y_GC,           # g C/m2/yr
        jnp.mean(diags.POC_flux_100m) * DAY,                        # mmol C/m2/d
        jnp.mean(b.spChl[:, 0] + b.diatChl[:, 0] + b.diazChl[:, 0]),  # mg/m3
        jnp.mean(b.NO3[:, 0]),                                      # mmol/m3
        jnp.mean(diags.flux_co2) * DAY * 365e-3,                    # mol/m2/yr, + into ocean
    ])


def make_metric_fn(grid, forcing, cfg, st0, steps, n_days, mcfg, names):
    def f(theta):
        mc = scaled_config(mcfg, names, theta)
        _, states, diags = run_column_raw(st0, forcing, grid, cfg, mc, n_days, steps)
        return metrics_from_run(states, diags)
    return jax.jit(f)


def load_spun_state(st0):
    p = OUT / "spinup_state.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run the 'spinup' experiment first")
    d = np.load(p)
    bgc = MarblTracers(**{k: jnp.asarray(d[f"bgc_{k}"]) for k in TRACER_NAMES})
    return st0._replace(T=jnp.asarray(d["T"]), S=jnp.asarray(d["S"]), u=jnp.asarray(d["u"]),
                        v=jnp.asarray(d["v"]), bgc=bgc, h_bl=jnp.asarray(d["h_bl"]),
                        time=jnp.asarray(0.0))


# ---------------------------------------------------------------- experiments
def exp_spinup(a):
    grid, forcing, cfg, st0, steps = setup()
    mcfg = MarblConfig()
    run_year = jax.jit(lambda s: run_column_raw(s, forcing, grid, cfg, mcfg, 365, steps))
    st = st0
    for y in range(a.years):
        t0 = time.time()
        st, states, diags = run_year(st)
        m = np.asarray(metrics_from_run(states, diags))
        print(f"spin-up year {y + 1}: " + "  ".join(f"{k}={v:.3g}" for k, v in zip(METRICS, m))
              + f"   ({time.time() - t0:.0f} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "spinup_state.npz", T=st.T, S=st.S, u=st.u, v=st.v, h_bl=st.h_bl,
             **{f"bgc_{k}": getattr(st.bgc, k) for k in TRACER_NAMES})
    print(f"wrote {OUT / 'spinup_state.npz'}")


def exp_grad_check(a):
    grid, forcing, cfg, st0, steps = setup()
    st0 = load_spun_state(st0)
    mcfg = MarblConfig()
    params = tunable_params(mcfg)
    names = a.params.split(",") if a.params else list(params)[: a.n_params]
    missing = [n for n in names if n not in params]
    if missing:
        raise SystemExit(f"not tunable float parameters: {missing}; available: {list(params)}")
    f = make_metric_fn(grid, forcing, cfg, st0, steps, a.days, mcfg, names)
    npp = lambda th: f(th)[0]                                  # noqa: E731
    th0 = jnp.zeros(len(names))
    t0 = time.time()
    g_rev = np.asarray(jax.grad(npp)(th0))
    print(f"reverse-mode gradient: {time.time() - t0:.0f} s for {len(names)} parameters")
    res = []
    for i, n in enumerate(names):
        e = jnp.zeros(len(names)).at[i].set(1.0)
        _, g_fwd = jax.jvp(npp, (th0,), (e,))
        fd = {}
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            fd[h] = float((npp(th0 + h * e) - npp(th0 - h * e)) / (2 * h))
        best = min(fd.values(), key=lambda x: abs(x - g_rev[i]))
        rel = abs(best - g_rev[i]) / max(abs(g_rev[i]), 1e-30)
        res.append(dict(param=n, reverse=float(g_rev[i]), forward=float(g_fwd), fd=fd, rel_err_best_fd=rel))
        print(f"  {n:28s} rev {g_rev[i]: .6e}  fwd {float(g_fwd): .6e}  "
              + "  ".join(f"fd(h={h:g}) {v: .6e}" for h, v in fd.items()) + f"   rel.err {rel:.1e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gradcheck.json").write_text(json.dumps(dict(days=a.days, metric="NPP [gC/m2/yr]", results=res),
                                                   indent=1, default=str))
    print(f"wrote {OUT / 'gradcheck.json'}")


def exp_sensitivity(a):
    grid, forcing, cfg, st0, steps = setup()
    st0 = load_spun_state(st0)
    mcfg = MarblConfig()
    params = tunable_params(mcfg)
    names = list(params)
    print(f"{len(names)} tunable parameters, window {a.days} days")
    f = make_metric_fn(grid, forcing, cfg, st0, steps, a.days, mcfg, names)
    th0 = jnp.zeros(len(names))
    t0 = time.time()
    m0 = np.asarray(f(th0))
    t_fwd = time.time() - t0
    t0 = time.time()
    J = np.asarray(jax.jacrev(f)(th0))                 # (n_metrics, n_params): d metric / d ln p
    t_rev = time.time() - t0
    E = J / np.where(np.abs(m0[:, None]) > 1e-30, m0[:, None], np.nan)   # elasticities
    print(f"forward {t_fwd:.0f} s, reverse (all {len(METRICS)} metrics x {len(names)} params) {t_rev:.0f} s")
    print("baseline: " + "  ".join(f"{k}={v:.4g}" for k, v in zip(METRICS, m0)))
    order = np.argsort(-np.nansum(np.abs(E[:4]), axis=0))   # rank by NPP+export+Chl+NO3
    print(f"\n{'parameter':28s} " + " ".join(f"{k:>9s}" for k in METRICS))
    for i in order[:25]:
        print(f"{names[i]:28s} " + " ".join(f"{E[j, i]:9.3f}" for j in range(len(METRICS))))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "sensitivity.npz", names=np.array(names), metrics=np.array(METRICS),
             baseline=m0, dmetric_dlnp=J, elasticity=E, rank=order)
    (OUT / "sensitivity_ranking.json").write_text(json.dumps(
        [dict(param=names[i], kind=params[names[i]],
              **{m: float(E[j, i]) for j, m in enumerate(METRICS)}) for i in order], indent=1))
    print(f"\nwrote {OUT / 'sensitivity.npz'} and sensitivity_ranking.json")


def exp_twin(a):
    grid, forcing, cfg, st0, steps = setup()
    st0 = load_spun_state(st0)
    mcfg = MarblConfig()
    if a.params:
        names = a.params.split(",")
    else:
        rk = OUT / "sensitivity_ranking.json"
        if not rk.exists():
            raise SystemExit("run 'sensitivity' first, or pass --params")
        names = [r["param"] for r in json.loads(rk.read_text())[: a.k]]
    K, B = len(names), a.bound
    print(f"twin experiment: {K} parameters {names}, window {a.days} days, bound x{np.exp(B):.2f}")

    def run(theta):
        mc = scaled_config(mcfg, names, theta)
        _, states, diags = run_column_raw(st0, forcing, grid, cfg, mc, a.days, steps)
        b = states.bgc
        return dict(logChl=jnp.log(b.spChl[:, 0] + b.diatChl[:, 0] + b.diazChl[:, 0] + 1e-6),
                    NO3=b.NO3[:, 0], PO4=b.PO4[:, 0],
                    NPP=jnp.sum(diags.NPP, axis=-1) * S2Y_GC,
                    POC100=diags.POC_flux_100m * DAY)

    obs = jax.tree_util.tree_map(np.asarray, jax.jit(run)(jnp.zeros(K)))
    scale = {k: max(float(np.std(v)), 1e-3 * (abs(float(np.mean(v))) + 1e-12)) for k, v in obs.items()}

    def loss(phi):
        sim = run(B * jnp.tanh(phi))
        return sum(jnp.mean(((sim[k] - obs[k]) / scale[k]) ** 2) for k in obs) / len(obs)

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(a.seed)
    theta_init = rng.choice([-1, 1], K) * rng.uniform(0.25, 0.5, K)          # factors ~0.6-0.8 or 1.3-1.65
    phi = jnp.arctanh(jnp.asarray(theta_init) / B)
    m, v = jnp.zeros(K), jnp.zeros(K)
    hist = []
    for it in range(1, a.iters + 1):
        t0 = time.time()
        L, g = vg(phi)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        phi = phi - a.lr * (m / (1 - 0.9 ** it)) / (jnp.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        th = np.asarray(B * jnp.tanh(phi))
        hist.append(dict(iter=it, loss=float(L), theta=th.tolist()))
        print(f"iter {it:3d}  loss {float(L):.4e}  max|theta| {np.max(np.abs(th)):.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    th = np.asarray(B * jnp.tanh(phi))
    print("\nparameter                    true  initial  recovered   (scale factors)")
    for i, n in enumerate(names):
        print(f"{n:28s} 1.000  {np.exp(theta_init[i]):7.3f}  {np.exp(th[i]):9.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"twin_seed{a.seed}.json").write_text(json.dumps(dict(
        params=names, days=a.days, bound=B, theta_init=theta_init.tolist(), theta_final=th.tolist(),
        history=hist), indent=1))
    print(f"wrote {OUT / f'twin_seed{a.seed}.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="exp", required=True)
    s = sub.add_parser("spinup"); s.add_argument("--years", type=int, default=2)
    s = sub.add_parser("grad-check"); s.add_argument("--days", type=int, default=30)
    s.add_argument("--params", default=None); s.add_argument("--n-params", type=int, default=6)
    s = sub.add_parser("sensitivity"); s.add_argument("--days", type=int, default=365)
    s = sub.add_parser("twin"); s.add_argument("--days", type=int, default=365)
    s.add_argument("--k", type=int, default=6); s.add_argument("--params", default=None)
    s.add_argument("--iters", type=int, default=60); s.add_argument("--lr", type=float, default=0.05)
    s.add_argument("--bound", type=float, default=1.0); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")
    dict(spinup=exp_spinup, **{"grad-check": exp_grad_check}, sensitivity=exp_sensitivity, twin=exp_twin)[a.exp](a)


if __name__ == "__main__":
    main()
