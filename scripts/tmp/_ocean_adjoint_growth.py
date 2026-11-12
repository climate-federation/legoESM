"""Ocean WB phase 1: adjoint growth vs rollout length on the 1-degree lat-lon C-grid ocean.

Model and initial grid/bathymetry come from run_omip's OWN production setup: the probe calls
``run_omip_single("latlon", args)`` and captures ``model, state, z_coord, dt`` at the entry of
``_run_omip_loop`` (so partial cells, equatorial smoothing, strait handling and the 80N cap are
exactly the driver's).  Flags: 180x360, NLEV tanh z*, dt 1800, ETOPO 1 deg, KPP, implicit
vertical mixing, --north-cap-lat 80 (the driver's documented fix: the open pole blew up at
89.5N within 3 steps from both rest and ORAS5 ICs, jobs 28850719/28850720), fp64.

Stepping mirrors ``model.step``'s restoring-mode path: ``_step_impl`` then the static-gated
post-step projections (polar filter, freeze floor, EW overlap), after ``prime_step_caches``.
No surface forcing (the forcing path is not built): numbers are UNFORCED tangent-linear growth.

IC and target: daily ORAS5 from the Neptune store (14 depths, 181x360 north-to-south, NaN land):
the two bracketing rows averaged to cell-centre latitudes (lon identical), linear in depth onto
the reference level centres (constant beyond the end depths).  Where ORAS5 has no value at a
model-wet cell the IC is filled by neighbour diffusion, and the loss MASKS that cell out.

Loss (Neptune-style L1): per variable, sum over valid wet cells of |pred - target| x cell area x
layer thickness, divided by the summed weights and by that variable's area-and-volume-weighted
std of the day-0 ORAS5 field; T, S, u, v, eta averaged with equal weight.  u faces weighted by
cos(lat), seam face counted once; v faces by cos(face lat).  eta: both sides have their
area-weighted mean over valid wet cells removed.  A smooth twin (squared error) is used only to
validate the adjoint by finite differences, which L1's kinks make unreliable.

Usage: python _ocean_adjoint_growth.py NLEV OUT_JSON [DAYS ...]
"""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/run"))
import run_omip as RO  # noqa: E402

NLEV = int(sys.argv[1])
OUT = Path(sys.argv[2])
DAYS = [int(d) for d in sys.argv[3:]] or [1, 5, 10, 30]
FD_DAYS = (1, 5)
FD_EPS = (1e-1, 1e-2, 1e-3)
FD_TOL = 1e-3                     # |ratio - 1| at the best eps, smooth loss
DT = 1800.0
STEPS_PER_DAY = int(86400 / DT)
STORE = Path("/work/bd1083/b309178/diffESM/S2S_model/data/cmems_daily/neptune")
DEPTHS = np.array([0.50576, 1.5558553, 2.6676817, 8.092519, 26.5583, 53.850636, 108.03028,
                   199.7900, 300.8875, 508.6399, 1045.854, 1945.296, 2955.570, 5089.479])
DATE0 = np.datetime64("2010-01-01")
ARGV = ["--grid", "latlon", "--resolution", "180x360", "--nlev", str(NLEV), "--dt", str(DT),
        "--days", "1", "--bathymetry", str(REPO / "data/bathymetry/etopo_1deg.nc"),
        "--implicit-vertical-mixing", "--north-cap-lat", "80", "--precision", "fp64",
        "--output", str(OUT.parent / f"omip_setup_{NLEV}")]


class _Captured(Exception):
    pass


def _capture(model, state, grid_type, grid, z_coord, dt, *a, **k):
    raise _Captured((model, state, grid, z_coord, dt))


args = RO.parse_args(ARGV)
RO._run_omip_loop = _capture
try:
    RO.run_omip_single("latlon", args)
    raise SystemExit("run_omip_single returned without reaching _run_omip_loop")
except _Captured as c:
    model, state0, grid, z_coord, dt_run = c.args[0]
assert float(dt_run) == DT, dt_run
print("captured model; config dt", dt_run, "T", state0.T.data.shape, "u", state0.u.data.shape,
      "v", state0.v.data.shape, flush=True)
lat_c = np.rad2deg(np.asarray(grid.lat)).ravel()
lon_c = np.rad2deg(np.asarray(grid.lon)).ravel()
assert np.allclose(lon_c, np.arange(360.0)), "model lon is not 0..359"
assert np.allclose(lat_c, np.arange(-89.5, 90.0, 1.0)), "model lat is not -89.5..89.5"
z_full = np.abs(np.asarray(z_coord.z_full_ref))
dz = np.asarray(z_coord.dz_ref)
wet3 = np.asarray(state0.land_mask.data)[..., None] * np.ones(NLEV)
print("levels", NLEV, "centres m", np.round(z_full, 1).tolist(), flush=True)
cfg = model.config


def step_fn(m):
    """model.step's restoring-mode path, traceable: _step_impl + static post-step ops."""
    def f(c):
        n = m._step_impl(c, DT, surface_forcing=None)
        if m.config.polar_filter.use_polar_filter:
            n = m._apply_polar_filter(n, DT)
        if m.config.freeze_floor:
            n = m._apply_freeze_floor(n)
        if getattr(m.config, "ew_cyclic_overlap", False):
            n = m._apply_ew_cyclic_overlap(n)
        return n
    return f


def _fill(f, valid, wet2):
    """Fill wet cells that lack data by repeated 4-neighbour averaging of valid values."""
    f = np.where(valid, f, 0.0)
    v = valid.astype(float)
    for _ in range(200):
        need = (v == 0) & wet2
        if not need.any():
            break
        s = sum(np.roll(np.where(v > 0, f, 0), sh, ax) for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
        n = sum(np.roll(v, sh, ax) for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
        upd = need & (n > 0)
        f = np.where(upd, s / np.maximum(n, 1), f)
        v = np.where(upd, 1.0, v)
    left = (v == 0) & wet2
    if left.any():
        f = np.where(left, f[v > 0].mean(), f)
    return f


def oras5(day):
    """ORAS5 day -> model fields + validity masks."""
    a = np.load(STORE / f"{DATE0 + np.timedelta64(day, 'D')}.npy").astype(np.float64)
    a = a[:, ::-1, :]                                        # south-to-north, row k = -90 + k
    cnt = np.isfinite(a[:, :-1]).astype(float) + np.isfinite(a[:, 1:])
    num = np.nan_to_num(a[:, :-1]) + np.nan_to_num(a[:, 1:])
    c = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)  # centres -89.5..89.5
    idx = np.clip(np.searchsorted(DEPTHS, z_full), 1, 13)
    w = np.clip((z_full - DEPTHS[idx - 1]) / (DEPTHS[idx] - DEPTHS[idx - 1]), 0, 1)
    out, val = {}, {}
    for k, name in enumerate(("T", "S", "u", "v")):
        prof = c[14 * k:14 * (k + 1)]
        lo, hi = prof[idx - 1], prof[idx]
        f = lo * (1 - w)[:, None, None] + hi * w[:, None, None]
        f = np.where(np.isfinite(f), f, np.where(w[:, None, None] < 0.5, lo, hi))
        ok = np.isfinite(f)
        f = np.stack([_fill(f[l], ok[l], wet3[..., l] > 0) for l in range(NLEV)], 0)
        out[name] = np.moveaxis(f, 0, -1)
        val[name] = np.moveaxis(ok, 0, -1) & (wet3 > 0)
    zos = c[56]
    val["eta"] = np.isfinite(zos) & (wet3[..., 0] > 0)
    out["eta"] = _fill(zos, np.isfinite(zos), wet3[..., 0] > 0)
    u, vu = out["u"], val["u"]
    um = np.asarray(state0.u_mask.data)[..., None]
    vm = np.asarray(state0.v_mask.data)[..., None]
    uf = np.concatenate([0.5 * (u[:, -1:] + u[:, :1]), 0.5 * (u[:, :-1] + u[:, 1:]),
                         0.5 * (u[:, -1:] + u[:, :1])], axis=1) * um
    uvf = np.concatenate([vu[:, -1:] & vu[:, :1], vu[:, :-1] & vu[:, 1:], vu[:, -1:] & vu[:, :1]], axis=1)
    v, vv = out["v"], val["v"]
    vf = np.concatenate([v[:1], 0.5 * (v[:-1] + v[1:]), v[-1:]], axis=0) * vm
    vvf = np.concatenate([vv[:1] & False, vv[:-1] & vv[1:], vv[-1:] & False], axis=0)
    out["u"], out["v"] = uf, vf
    val["u"], val["v"] = uvf & (um > 0), vvf & (vm > 0)
    return {k: jnp.asarray(x) for k, x in out.items()}, val


ic, ic_val = oras5(0)
area_c = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, 360))
lat_f = np.arange(-90.0, 90.5, 1.0)
area_v = np.cos(np.deg2rad(lat_f))[:, None] * np.ones((1, 360))
area_u = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, 361))
area_u[:, -1] = 0.0                                            # seam face == face 0
BASE_W = {"T": area_c[..., None] * dz, "S": area_c[..., None] * dz,
          "u": area_u[..., None] * dz, "v": area_v[..., None] * dz, "eta": area_c}


def weights(val):
    return {k: jnp.asarray(BASE_W[k] * val[k]) for k in BASE_W}


W0 = weights(ic_val)
STD = {}
for k in ("T", "S", "u", "v", "eta"):
    x, wk = np.asarray(ic[k]), np.asarray(W0[k])
    mu = (x * wk).sum() / wk.sum()
    STD[k] = float(np.sqrt(((x - mu) ** 2 * wk).sum() / wk.sum()))
print("ORAS5 day-0 weighted std", STD, flush=True)
state = state0._replace(T=state0.T.replace(data=ic["T"]), S=state0.S.replace(data=ic["S"]),
                        u=state0.u.replace(data=ic["u"]), v=state0.v.replace(data=ic["v"]),
                        eta=state0.eta.replace(data=ic["eta"] * jnp.asarray(wet3[..., 0])))
model.prime_step_caches(state)

# Which float knobs can be traced through one step (a Python branch on the value fails).
CANDIDATES = ["A_h", "B_h", "K_h", "A_v", "K_v", "bottom_drag_r", "B_h_barotropic", "C_smag_lap",
              "A_h_floor", "bottom_drag_bg_velocity"]
trace_result = {}
for name in CANDIDATES:
    v0 = float(cfg.flat_get(name))
    m = copy.copy(model)

    def one(x, name=name, m=m):
        m.config = cfg.replace_flat(**{name: x})
        return step_fn(m)(state).T.data
    try:
        jax.eval_shape(one, v0)
        trace_result[name] = "traces"
    except Exception as e:
        import traceback
        tb = [l for l in traceback.format_exception(e) if "packages/" in l]
        trace_result[name] = f"{type(e).__name__} @ {tb[-1].strip().split(', in ')[0] if tb else '?'}"
    print(f"trace {name}={v0:g}: {trace_result[name]}", flush=True)
if os.environ.get("PROBE_MODE") == "trace":
    print("TRACE_DONE", sum(v == "traces" for v in trace_result.values()), "/", len(trace_result), flush=True)
    raise SystemExit(0)


def make_loss(n_days, tg, W, kind):
    f_step = step_fn(model)
    sw = {k: float(jnp.sum(W[k])) for k in W}

    def term(p, t, k):
        e = (p - t) / STD[k]
        e = jnp.abs(e) if kind == "l1" else e * e
        return jnp.sum(W[k] * e) / sw[k]

    def loss(dT0, du0):
        s = state._replace(T=state.T.replace(data=state.T.data + dT0),
                           u=state.u.replace(data=state.u.data + du0))

        @jax.checkpoint
        def stp(c, _):
            return f_step(c), None

        @jax.checkpoint
        def day(c, _):
            return jax.lax.scan(stp, c, None, length=STEPS_PER_DAY)[0], None

        s = jax.lax.scan(day, s, None, length=n_days)[0]
        we = W["eta"]
        pe = s.eta.data - jnp.sum(we * s.eta.data) / sw["eta"]
        te = tg["eta"] - jnp.sum(we * tg["eta"]) / sw["eta"]
        return (term(s.T.data, tg["T"], "T") + term(s.S.data, tg["S"], "S")
                + term(s.u.data, tg["u"], "u") + term(s.v.data, tg["v"], "v")
                + term(pe, te, "eta")) / 5.0
    return loss


if os.environ.get("PROBE_MODE") == "opsweep":
    # Phase-1e: which operator makes the single-step adjoint jumps the stepsweep found (x40 at
    # step 484, x8 at 494, x6 at 495, all at the T surface cell (128,9,0), 38.5N 9E).  Forward on
    # the BASE config to each listed step; there, pull a one-hot T cotangent at (128,9,0) back
    # through ONE step under config variants that each switch one operator off for that step
    # only (same input state).  The variant whose growth collapses names the operator.
    # Control: the same at steps without a jump.
    STEPS = [int(x) for x in os.environ.get("OP_STEPS", "483:484:485:493:494:495").split(":")]
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    leaves, tdef = jax.tree_util.tree_flatten(state)
    isf = [jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) for x in leaves]

    def split(s_):
        lv = jax.tree_util.tree_leaves(s_)
        return [x for x, f in zip(lv, isf) if f], [x for x, f in zip(lv, isf) if not f]

    def merge(fl, ot):
        fi, oi = iter(fl), iter(ot)
        return jax.tree_util.tree_unflatten(tdef, [next(fi) if f else next(oi) for f in isf])

    def step_cfg(c_):
        def f(st):
            n = model._step_impl(st, DT, surface_forcing=None, config=c_)
            if c_.polar_filter.use_polar_filter:
                n = model._apply_polar_filter(n, DT)
            if c_.freeze_floor:
                n = model._apply_freeze_floor(n)
            if getattr(c_, "ew_cyclic_overlap", False):
                n = model._apply_ew_cyclic_overlap(n)
            return n
        return f

    ph = cfg.physics
    variants = {"base": cfg,
                "no_fixer": cfg.replace_flat(use_conservation_fixer=False),
                "no_gm_redi": cfg.replace_flat(gm_redi=None),
                "no_convection": cfg.replace_flat(physics=ph._replace(
                    convection=OceanConvectionConfig(scheme="none"))),
                "no_polar_filter": cfg._replace(
                    polar_filter=cfg.polar_filter._replace(use_polar_filter=False)),
                "no_freeze_floor": cfg.replace_flat(freeze_floor=False),
                "vmix_constant": cfg.replace_flat(physics=ph._replace(
                    vertical_mixing=ph.vertical_mixing._replace(scheme="constant")))}
    if os.environ.get("OP_ONLY"):
        variants = {k: variants[k] for k in os.environ["OP_ONLY"].split(":")}
    print("variants:", {k: (v.use_conservation_fixer, v.gm_redi is not None,
                            v.physics.convection.scheme, v.polar_filter.use_polar_filter,
                            v.freeze_floor, v.physics.vertical_mixing.scheme)
                        for k, v in variants.items()}, flush=True)
    vjps = {name: jax.jit(lambda fl, ot, sd, _c=c_: jax.vjp(
                lambda f_: split(step_cfg(_c)(merge(f_, ot)))[0], fl)[1](sd)[0])
            for name, c_ in variants.items()}
    f_base = jax.jit(step_fn(model))
    s_ = state
    for k in range(max(STEPS) + 1):
        if k in STEPS:
            fl, ot = split(s_)
            seed = merge([jnp.zeros_like(x) for x in fl], ot)
            seed = split(seed._replace(T=seed.T.replace(data=seed.T.data.at[128, 9, 0].set(1.0))))[0]
            row = []
            for name, c_ in variants.items():
                g = vjps[name](fl, ot, seed)
                gf = merge(g, ot)
                nT = float(jnp.sqrt(jnp.sum(gf.T.data ** 2)))
                nS = float(jnp.sqrt(jnp.sum(gf.S.data ** 2)))
                row.append(f"{name} |T| {nT:.3e} |S| {nS:.3e}")
            print(f"step {k}: " + " ; ".join(row), flush=True)
            if os.environ.get("OP_FD"):
                # Is the step's adjoint right?  <seed, J d> by central differences of the
                # nonlinear step along d = the VJP output itself, vs the AD value <g, d> = |g|^2.
                g = vjps["base"](fl, ot, seed)
                gn = jnp.sqrt(sum(jnp.sum(x * x) for x in g))
                d = [x / gn for x in g]
                ad = float(gn)
                for eps in (1e-2, 1e-4, 1e-6):
                    fp = split(f_base(merge([a + eps * b for a, b in zip(fl, d)], ot)))[0]
                    fm = split(f_base(merge([a - eps * b for a, b in zip(fl, d)], ot)))[0]
                    fd = float(sum(jnp.sum(sd * (a - b)) for sd, a, b in zip(seed, fp, fm))) / (2 * eps)
                    print(f"  FD step {k} eps {eps:g}: fd {fd:.6e} ad {ad:.6e} ratio {fd / ad:.6f}",
                          flush=True)
        s_ = f_base(s_)
    print("OPSWEEP_DONE", flush=True)
    raise SystemExit(0)


if os.environ.get("PROBE_MODE") == "stepsweep":
    # Phase-1d: localise the one-day adjoint jump the adjsweep found (day 11 -> 10, 1e-3 -> 5e5,
    # at 38.5N 1-9E).  Forward to the start of day SWEEP_DAY, keep every step's state, seed a
    # unit random cotangent on T (wet cells) at the end of that day and pull it back ONE STEP at
    # a time: per step, cotangent L2/max/argmax per field, plus the forward state at the argmax
    # column (T, S, |u|, eta, bottom level, partial-cell thickness) and the day's max T location.
    SD = int(os.environ.get("SWEEP_DAY", "10"))
    leaves, tdef = jax.tree_util.tree_flatten(state)
    isf = [jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) for x in leaves]

    def split(s_):
        lv = jax.tree_util.tree_leaves(s_)
        return [x for x, f in zip(lv, isf) if f], [x for x, f in zip(lv, isf) if not f]

    def merge(fl, ot):
        fi, oi = iter(fl), iter(ot)
        return jax.tree_util.tree_unflatten(tdef, [next(fi) if f else next(oi) for f in isf])

    f_step = jax.jit(step_fn(model))
    zc = model.z_coord
    s_ = state
    for k in range(SD * STEPS_PER_DAY):
        s_ = f_step(s_)
        if (k + 1) % STEPS_PER_DAY == 0:
            T_ = np.asarray(s_.T.data)
            print(f"fwd day {(k + 1) // STEPS_PER_DAY}: maxT {T_.max():.2f} @"
                  f"{np.unravel_index(int(T_.argmax()), T_.shape)} minT {T_.min():.2f} @"
                  f"{np.unravel_index(int(T_.argmin()), T_.shape)}", flush=True)
    traj = [s_]
    for k in range(STEPS_PER_DAY):
        traj.append(f_step(traj[-1]))
    FIELDS = ("T", "S", "u", "v", "eta")
    rng = np.random.default_rng(1)
    seed_T = rng.standard_normal(state.T.data.shape) * (wet3 > 0)
    zero = merge([jnp.zeros_like(x) for x in split(traj[-1])[0]], split(traj[-1])[1])
    cot = split(zero._replace(T=zero.T.replace(data=jnp.asarray(seed_T / np.linalg.norm(seed_T)))))[0]

    def col(st, i, j):
        bl = int(np.asarray(zc.bottom_level)[i, j]) if hasattr(zc, "bottom_level") else -9
        hp = np.asarray(zc.h_partial)[i, j] if hasattr(zc, "h_partial") else None
        return (f"col({i},{j}) bottom_level {bl} h_bottom "
                f"{(hp[bl] if hp is not None and bl >= 0 else float('nan')):.3g} m H "
                f"{float(np.asarray(st.H_bathy.data)[i, j]):.1f} eta {float(np.asarray(st.eta.data)[i, j]):.3f} "
                f"T[{float(np.asarray(st.T.data)[i, j].min()):.2f},{float(np.asarray(st.T.data)[i, j].max()):.2f}] "
                f"S[{float(np.asarray(st.S.data)[i, j].min()):.2f},{float(np.asarray(st.S.data)[i, j].max()):.2f}] "
                f"|u|max {float(np.abs(np.asarray(st.u.data)[i, j:j + 2]).max()):.3f}")

    vstep = jax.jit(lambda fl, ot, c: jax.vjp(lambda f_: split(f_step(merge(f_, ot)))[0], fl)[1](c)[0])
    for k in range(STEPS_PER_DAY - 1, -1, -1):
        fl, ot = split(traj[k])
        cot = list(vstep(fl, ot, cot))
        c_full = merge(cot, ot)
        x = np.abs(np.asarray(c_full.T.data))
        i, j, l = np.unravel_index(int(x.argmax()), x.shape)
        rows = []
        for nm in FIELDS:
            y = np.asarray(getattr(c_full, nm).data)
            rows.append(f"{nm} {float(np.sqrt((y * y).sum())):.2e}")
        print(f"adj step {SD * STEPS_PER_DAY + k}: " + " ".join(rows)
              + f" | T max {x.max():.2e} @({i},{j},{l}) | " + col(traj[k], i, j), flush=True)
    print("STEPSWEEP_DONE", flush=True)
    raise SystemExit(0)


if os.environ.get("PROBE_MODE") == "adjsweep":
    # Phase-1c (forward-mode is impossible: the step contains custom_vjp ops).  Reverse sweep
    # day by day: cotangent of the day-N smooth loss (squared error vs ORAS5 day N) pulled back
    # one day at a time through the SAME production step; per day and field: L2, max|.|, flat
    # argmax of the cotangent.  Beside it, a nonlinear twin (T0 + eps*d) gives the finite-
    # perturbation growth per day.  Adjoint growing while the twin stays small => the
    # linearisation of some operator is unstable, not the physics.
    N = int(os.environ.get("ADJ_DAYS", "12"))
    EPS = float(os.environ.get("TLM_EPS", "1e-2"))
    leaves, tdef = jax.tree_util.tree_flatten(state)
    isf = [jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) for x in leaves]

    def split(s_):
        lv = jax.tree_util.tree_leaves(s_)
        return [x for x, f in zip(lv, isf) if f], [x for x, f in zip(lv, isf) if not f]

    def merge(fl, ot):
        fi, oi = iter(fl), iter(ot)
        return jax.tree_util.tree_unflatten(tdef, [next(fi) if f else next(oi) for f in isf])

    f_step = step_fn(model)

    @jax.jit
    def day(s_):
        @jax.checkpoint
        def stp(c, _):
            return f_step(c), None
        return jax.lax.scan(stp, s_, None, length=STEPS_PER_DAY)[0]

    FIELDS = ("T", "S", "u", "v", "eta")
    rng = np.random.default_rng(0)
    d = rng.standard_normal(state.T.data.shape) * (wet3 > 0)
    d = jnp.asarray(d / np.linalg.norm(d))
    traj, sp = [state], state._replace(T=state.T.replace(data=state.T.data + EPS * d))
    twin = []
    for k in range(N):
        traj.append(day(traj[-1]))
        sp = day(sp)
        row = []
        for nm in FIELDS:
            x = np.asarray((getattr(sp, nm).data - getattr(traj[-1], nm).data) / EPS)
            row.append([float(np.sqrt((x * x).sum())), float(np.abs(x).max()), int(np.abs(x).argmax())])
        twin.append(row)
        print(f"fwd day {k + 1}: twin T L2 {row[0][0]:.3e} u L2 {row[2][0]:.3e} "
              f"maxT {np.asarray(traj[-1].T.data).max():.2f}", flush=True)
    tg, tv = oras5(N)
    W = weights({kk: tv[kk] & ic_val[kk] for kk in tv})
    sw = {kk: float(jnp.sum(W[kk])) for kk in W}

    def final_loss(fl):
        s_ = merge(fl, split(traj[-1])[1])
        return sum(jnp.sum(W[kk] * ((getattr(s_, kk).data - tg[kk]) / STD[kk]) ** 2) / sw[kk]
                   for kk in ("T", "S", "u", "v")) / 4.0

    cot = jax.grad(final_loss)(split(traj[-1])[0])
    adj = []
    for k in range(N - 1, -1, -1):
        ot = split(traj[k])[1]
        _, vjp = jax.vjp(lambda fl: split(day(merge(fl, ot)))[0], split(traj[k])[0])
        cot = list(vjp(cot)[0])
        c_full = merge(cot, ot)
        row = []
        for nm in FIELDS:
            x = np.asarray(getattr(c_full, nm).data)
            row.append([float(np.sqrt((x * x).sum())), float(np.abs(x).max()), int(np.abs(x).argmax())])
        adj.append([k] + row)
        locs = {nm: np.unravel_index(row[i][2], getattr(state, nm).data.shape) for i, nm in enumerate(FIELDS)}
        print(f"adj at day {k}: " + " ".join(f"{nm} L2 {row[i][0]:.3e} max {row[i][1]:.3e} @{locs[nm]}"
                                            for i, nm in enumerate(FIELDS)), flush=True)
    _sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    OUT.write_text(json.dumps({"mode": "adjsweep", "nlev": NLEV, "dt": DT, "date0": str(DATE0),
                               "days": N, "eps": EPS, "argv": ARGV, "commit": _sha,
                               "fields": FIELDS, "cols": ["L2", "max", "flat_argmax"],
                               "twin": twin, "adjoint_by_day": adj}, indent=1))
    print("ADJSWEEP_DONE", flush=True)
    raise SystemExit(0)

if os.environ.get("PROBE_MODE") == "tlm":
    # Phase-1b: where does the tangent-linear model grow?  Forward-mode JVP carried step by
    # step next to a nonlinear twin (state + eps*d) from the same unit random T0 direction d.
    # TLM growing while the twin stays small => the linearisation of some operator is unstable
    # (non-smooth op), not the physics.  Per step and field: L2, max|.|, flat argmax.
    TLM_DAYS = int(os.environ.get("TLM_DAYS", "12"))
    EPS = float(os.environ.get("TLM_EPS", "1e-2"))
    leaves, tdef = jax.tree_util.tree_flatten(state)
    isf = [jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) for x in leaves]

    def split(s):
        lv = jax.tree_util.tree_leaves(s)
        return [x for x, f in zip(lv, isf) if f], [x for x, f in zip(lv, isf) if not f]

    def merge(fl, ot):
        fi, oi = iter(fl), iter(ot)
        return jax.tree_util.tree_unflatten(tdef, [next(fi) if f else next(oi) for f in isf])

    f_step = step_fn(model)
    rng = np.random.default_rng(0)
    d = rng.standard_normal(state.T.data.shape) * (wet3 > 0)
    d = jnp.asarray(d / np.linalg.norm(d))
    fl0, ot0 = split(state)
    tl0 = [jnp.zeros_like(x) for x in fl0]
    t_full = merge(tl0, ot0)
    t_full = t_full._replace(T=t_full.T.replace(data=d))
    tl0, _ = split(t_full)
    sp = state._replace(T=state.T.replace(data=state.T.data + EPS * d))
    FIELDS = ("T", "S", "u", "v", "eta")

    def stats(x):
        a = jnp.abs(x)
        return jnp.stack([jnp.sqrt(jnp.sum(x * x)), jnp.max(a), jnp.argmax(a).astype(x.dtype)])

    def body(c, _):
        fl, ot, tl, sp = c
        def g(fl_):
            o = f_step(merge(fl_, ot))
            a, b = split(o)
            return a, b
        fl1, tl1, ot1 = jax.jvp(g, (fl,), (tl,), has_aux=True)
        sp1 = f_step(sp)
        s1, t1 = merge(fl1, ot1), merge(tl1, ot1)
        st = jnp.stack([stats(getattr(t1, k).data) for k in FIELDS]
                       + [stats((getattr(sp1, k).data - getattr(s1, k).data) / EPS) for k in FIELDS])
        return (fl1, ot1, tl1, sp1), st

    @jax.jit
    def run_day(c):
        return jax.lax.scan(body, c, None, length=STEPS_PER_DAY)

    c = (fl0, ot0, tl0, sp)
    hist = []
    for dday in range(TLM_DAYS):
        c, st = run_day(c)
        st = np.asarray(st)
        hist.append(st)
        for k, nm in enumerate(FIELDS):
            print(f"day {dday + 1} {nm}: tlm L2 {st[-1, k, 0]:.3e} max {st[-1, k, 1]:.3e} at "
                  f"{np.unravel_index(int(st[-1, k, 2]), getattr(state, nm).data.shape)} | twin L2 "
                  f"{st[-1, 5 + k, 0]:.3e} max {st[-1, 5 + k, 1]:.3e} at "
                  f"{np.unravel_index(int(st[-1, 5 + k, 2]), getattr(state, nm).data.shape)}", flush=True)
    np.save(OUT.with_suffix(".npy"), np.concatenate(hist, 0))
    _sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    _dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                 capture_output=True, text=True).stdout.strip())
    OUT.write_text(json.dumps({"mode": "tlm", "nlev": NLEV, "dt": DT, "date0": str(DATE0),
                               "days": TLM_DAYS, "eps": EPS, "argv": ARGV, "commit": _sha,
                               "dirty": _dirty, "npy": str(OUT.with_suffix(".npy")),
                               "layout": "(step, 10, 3): rows T,S,u,v,eta tangent then the "
                                         "same five for the nonlinear twin / eps; cols "
                                         "L2, max|.|, flat argmax"}, indent=1))
    print("TLM_DONE", flush=True)
    raise SystemExit(0)

commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
zT, zu = jnp.zeros_like(state.T.data), jnp.zeros_like(state.u.data)
rows = []
for n in DAYS:
    tg, tv = oras5(n)
    W = weights({k: tv[k] & ic_val[k] for k in tv})
    r = {"days": n}
    t0 = time.time()
    for kind in ("l1", "mse"):
        loss = make_loss(n, tg, W, kind)
        val, (gT, gu) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(zT, zu)
        gT, gu = np.asarray(gT), np.asarray(gu)
        r[kind] = {"loss": float(val), "gT_max": float(np.abs(gT).max()), "gT_l2": float(np.linalg.norm(gT)),
                   "gu_max": float(np.abs(gu).max()), "gu_l2": float(np.linalg.norm(gu)),
                   "finite": bool(np.isfinite(float(val)) and np.isfinite(gT).all() and np.isfinite(gu).all()),
                   "gT_nonzero_frac_wet": float(np.mean(np.abs(gT[wet3 > 0]) > 0))}
        if kind == "mse" and n in FD_DAYS:
            f = jax.jit(loss)
            rng = np.random.default_rng(n)
            fd_rows = []
            for which, g, shape, mask in (("T0", gT, zT.shape, wet3 > 0),
                                          ("u0", gu, zu.shape, np.asarray(state0.u_mask.data)[..., None] > 0)):
                for j in range(2):
                    d = rng.standard_normal(shape) * np.broadcast_to(mask, shape)
                    d = d / np.linalg.norm(d)
                    ad = float((g * d).sum())
                    ratios = {}
                    for eps in FD_EPS:
                        dd = jnp.asarray(eps * d)
                        if which == "T0":
                            fp, fm = f(zT + dd, zu), f(zT - dd, zu)
                        else:
                            fp, fm = f(zT, zu + dd), f(zT, zu - dd)
                        fd = (float(fp) - float(fm)) / (2 * eps)
                        ratios[f"{eps:g}"] = fd / ad if ad != 0 else float("nan")
                    best = min(ratios.values(), key=lambda x: abs(x - 1) if np.isfinite(x) else 1e300)
                    fd_rows.append({"var": which, "dir": j, "ad": ad, "ratios": ratios, "best": best,
                                    "pass": bool(np.isfinite(best) and abs(best - 1) < FD_TOL)})
            r["fd"] = fd_rows
    mem = None
    try:
        mem = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**30
    except Exception:
        pass
    r["seconds"], r["peak_GiB"] = time.time() - t0, mem
    print(json.dumps(r), flush=True)
    rows.append(r)
    OUT.write_text(json.dumps({"nlev": NLEV, "dt": DT, "date0": str(DATE0), "argv": ARGV,
                               "rows": rows, "traceability": trace_result, "std": STD,
                               "commit": commit, "dirty": bool(dirty)}, indent=1))
print("PROBE_DONE", flush=True)
