"""#1029 probe: anatomy of the lat-lon held_suarez_topo burst.

Rebuilds the matrix model exactly as ``run_held_suarez`` (latlon branch) does,
runs one seed, and records per-sample diagnostics: wind maximum + location,
static stability (theta inversions), thinnest layer, p_s range, T range.
Samples every ``--every`` steps until ``--fine-from``, then every step until the
wind trips 1000 m/s (the matrix threshold) or ``--max-steps``.  The last
``--keep`` clean states before the trip are saved as npz for branch arms.

Control: the trip step must reproduce the matrix's interval-1 trip
(seed 65537 -> 4730), else the probe is not the matrix model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
    CGridLatLonHydrostaticState, hydrostatic_to_cgrid)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_forcing_latlon)
from legoesm.atmosphere.idealized.held_suarez_topo import (
    held_suarez_topo_init_latlon)

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "matrix_runner", ROOT / "scripts/matrix/run_atmosphere_test_matrix.py")
runner = importlib.util.module_from_spec(_spec)
sys.modules["matrix_runner"] = runner
_spec.loader.exec_module(runner)


def build(h0: float, seed: int, dt_scale: float = 1.0, nlev: int = 40,
          a_h_override: float | None = None, nu4: float = 0.0):
    grid = create_latlon_grid(72, 144)
    sigma = runner._create_vertical(nlev, "hybrid")
    dx_pole = float(grid.radius) * grid.dlon * math.cos(math.pi / 2 - grid.dlat / 2)
    dt = runner._latlon_dt(dx_pole, 200.0, "held_suarez_topo") * dt_scale
    ah = min(runner._laplacian_visc_latlon(72), 0.4 * dx_pole**2 / dt)
    if a_h_override is not None:
        ah = a_h_override
    sig0 = float(np.asarray(sigma.sigma_full)[0])
    sponge = max((0.15 - sig0) / 0.15, 0.0) ** 2 / 3600.0
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=ah, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=runner._latlon_polar_filter_on("held_suarez_topo"),
        sponge_coeff=sponge, sponge_width_m=10000.0,
        sponge_scale_height_m=7500.0, sponge_shape="sin2", nu_del4=nu4)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    st = hydrostatic_to_cgrid(
        held_suarez_topo_init_latlon(grid, sigma, h_0=h0, seed=seed), grid)
    return grid, sigma, model, st, dt, ah


def make_diag(grid, sigma, model=None):
    lat = np.degrees(np.asarray(grid.lat))
    lon = np.degrees(np.asarray(grid.lon))
    kappa = constants.R_d / constants.c_pd

    @jax.jit
    def diag(s):
        uc = 0.5 * (s.u[:, :-1, :] + s.u[:, 1:, :])
        vc = 0.5 * (s.v[:-1, :, :] + s.v[1:, :, :])
        spd = jnp.sqrt(uc**2 + vc**2)
        p = sigma.pressure_at_full(s.p_s)
        ph = sigma.pressure_at_half(s.p_s)
        dp = ph[..., 1:] - ph[..., :-1]
        th = s.T * (constants.p_ref / p) ** kappa
        dth = th[..., :-1] - th[..., 1:]          # >0 stable (k=0 is top)
        leak = jnp.zeros(())
        if model is not None and model.config.nu_del4 > 0.0:
            from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
                cgrid_latlon_hydrostatic_tendencies)
            off = model.config._replace(nu_del4=0.0)
            dT_on = cgrid_latlon_hydrostatic_tendencies(
                s, grid, sigma, model.config, dt=model.dt)[2]
            dT_off = cgrid_latlon_hydrostatic_tendencies(s, grid, sigma, off)[2]
            col = jnp.sum(constants.c_pd * (dT_on - dT_off) * dp, axis=-1) / constants.g
            leak = jnp.sum(col * grid.area) / jnp.sum(grid.area)
        ul = s.u[:, :-1, 30:]
        u2dx = jnp.max(jnp.abs(ul - 0.5 * (jnp.roll(ul, 1, 1) + jnp.roll(ul, -1, 1))))
        return dict(umax=jnp.max(jnp.abs(s.u)), spd=jnp.max(spd), heat_leak_Wm2=leak,
                    u2dx_low=u2dx,
                    i_spd=jnp.argmax(spd), dth_min=jnp.min(dth),
                    i_dth=jnp.argmin(dth), n_unst=jnp.sum(dth < 0),
                    n_unst_low=jnp.sum(dth[..., 20:] < 0),
                    dp_min=jnp.min(dp), ps_min=jnp.min(s.p_s),
                    ps_max=jnp.max(s.p_s), T_min=jnp.min(s.T),
                    T_max=jnp.max(s.T))

    def host(s, step, dt):
        d = {k: float(v) for k, v in diag(s).items()}
        sh = (72, 144, 40)
        a, b, c = np.unravel_index(int(d.pop("i_spd")), sh)
        d["spd_at"] = [float(lat[a]), float(lon[b]), int(c)]
        a, b, c = np.unravel_index(int(d.pop("i_dth")), (72, 144, 39))
        d["dth_at"] = [float(lat[a]), float(lon[b]), int(c)]
        d["step"] = step
        d["day"] = step * dt / 86400.0
        return d
    return host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=65537)
    ap.add_argument("--dt-scale", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--fine-from", type=int, default=4400)
    ap.add_argument("--max-steps", type=int, default=4800)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--save-at", type=int, nargs="*", default=[])
    ap.add_argument("--zeta-frac", type=float, default=None)
    ap.add_argument("--a-h", type=float, default=None)
    ap.add_argument("--nu4", type=float, default=0.0)
    ap.add_argument("--nu4-matrix", action="store_true",
                    help="use the matrix runner's lat-lon del-4 coefficient")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.nu4_matrix:
        a.nu4 = runner._biharmonic_visc_latlon(72)
    if a.zeta_frac is not None:
        import functools
        import legoesm.atmosphere.idealized.held_suarez_topo as hst
        hst.dcmip_2_0_0_mountain = functools.partial(
            hst.dcmip_2_0_0_mountain, zeta_frac=a.zeta_frac)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid, sigma, model, st, dt, ah = build(a.h0, a.seed, a.dt_scale, a_h_override=a.a_h,
                                      nu4=a.nu4)
    host = make_diag(grid, sigma, model)
    meta = dict(h0=a.h0, seed=a.seed, dt=dt, A_h=ah, nu_del4=a.nu4,
                sponge=float(model.config.sponge_coeff))
    print(json.dumps(meta), flush=True)
    log = open(out / "diag.jsonl", "w")
    recent = deque(maxlen=a.keep)
    t0 = time.time()
    trip = None
    for i in range(a.max_steps):
        st = model.step_with_physics(st, dt, held_suarez_forcing_latlon)
        step = i + 1
        if step in a.save_at:
            np.savez(out / f"state_{step}.npz", **{f: np.asarray(getattr(st, f))
                     for f in ("u", "v", "T", "p_s", "phis")})
        if step % a.every == 0 or step >= a.fine_from:
            d = host(st, step, dt)
            log.write(json.dumps(d) + "\n")
            if step % a.every == 0:
                print(f"{step} d={d['day']:.2f} umax={d['umax']:.1f} "
                      f"spd@{d['spd_at']} nunst={d['n_unst']:.0f} "
                      f"dthmin={d['dth_min']:.2f}@{d['dth_at']} "
                      f"leak={d['heat_leak_Wm2']:.3e}W/m2 u2dx={d['u2dx_low']:.2f} "
                      f"dpmin={d['dp_min']:.0f} ps=[{d['ps_min']:.0f},"
                      f"{d['ps_max']:.0f}] T=[{d['T_min']:.1f},{d['T_max']:.1f}] "
                      f"{time.time()-t0:.0f}s", flush=True)
            if not math.isfinite(d["umax"]) or d["umax"] > 1000.0:
                trip = step
                print("TRIP", json.dumps(d), flush=True)
                break
            if step >= a.fine_from:
                recent.append((step, st))
    for step, s in recent:
        np.savez(out / f"clean_{step}.npz", **{f: np.asarray(getattr(s, f))
                 for f in ("u", "v", "T", "p_s", "phis")})
    meta["trip_step"] = trip
    (out / "meta.json").write_text(json.dumps(meta))
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
