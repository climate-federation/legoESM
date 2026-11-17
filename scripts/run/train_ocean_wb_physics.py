"""Ocean-WB arm (c): train the lat-lon ocean's physics knobs against daily ORAS5.

The model is run_omip's OWN 1-degree lat-lon production setup (captured at the entry of
``_run_omip_loop``, as the phase-1 probe does) on the eORCA1 L75 vertical grid, so ORAS5 level k
is model level k.  Trained: ``LATLON_DYNAMICS_TRAINABLE`` (8 knobs, user-approved limits),
started from the run's resolved values.  Loss: :func:`legoesm.training.oras5_ocean.l1_loss`
against every daily target of an N-day rollout, averaged over the days.

Pieces reused from the WB trainer: the no-inert gate (``assert_no_inert_over`` over every sample
before the first update), ``create_optimizer``, ``ema_update``,
and ``training_history.jsonl`` for ``scripts/plot/plot_wb_training_history.py``.

Forcing: only ``--forcing none`` exists so far.  An unforced run is PLUMBING ONLY: its loss is
never comparable with a forced run, and every output is labelled so.

Detail choices (votes author/codex/GLM, 2026-11-11): level k -> k with a 1 m guard (3-0); L75
thicknesses generated from the eORCA1 mesh mask, not committed (2-1); loss over every daily target
(3-0); Adam lr 1e-2, warmup 10, clip 1.0 (3-0); EMA 0.99 (3-0); smoke 2 days x 4 dates x 2
epochs (3-0); gate over all samples (3-0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NATIVE_MASK = Path("/work/bb1596/b309178/diffESM/ocean_wb/oras5_native_mask/thetao_oras_1993-01-01.nc")
MESH_MASK = Path("/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/grids/eORCA1.2_mesh_mask.nc")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True, type=Path, help="ORAS5 L75 day store")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--forcing", required=True, choices=("none", "bulk", "prescribed"),
                   help="'none' = unforced, PLUMBING ONLY; 'bulk' = ERA5 state + coupler bulk "
                        "formulas; 'prescribed' = ERA5's own fluxes")
    p.add_argument("--era5-store", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/era5_daily"))
    p.add_argument("--clim-dir", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/climatology"))
    p.add_argument("--mesh-mask", type=Path, default=MESH_MASK)
    p.add_argument("--native-mask", type=Path, default=NATIVE_MASK,
                   help="one day of native 0.25-degree ORAS5 thetao: its NaNs define the face walls")
    p.add_argument("--dz-file", type=Path,
                   default=Path("/work/bb1596/b309178/diffESM/ocean_wb/eorca1_L75_dz.txt"))
    p.add_argument("--days", type=int, default=2, help="rollout length [days]")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--first-year", type=int, default=1993)
    p.add_argument("--last-year", type=int, default=2017)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--dt", type=float, default=1800.0)
    return p.parse_args(argv)


def write_dz_file(mesh_mask, path):
    """eORCA1 L75 ``e3t_1d`` -> text thicknesses for run_omip's ``--dz-ref-file``."""
    if not path.exists():
        import xarray as xr
        e = np.asarray(xr.open_dataset(mesh_mask)["e3t_1d"].values, np.float64).ravel()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(f"# e3t_1d from {mesh_mask}\n" + "\n".join(f"{x:.9f}" for x in e) + "\n")
        tmp.replace(path)
    return np.loadtxt(path)


def capture_model(argv):
    """run_omip's production model + initial state at ``_run_omip_loop`` entry."""
    sys.path.insert(0, str(REPO / "scripts/run"))
    import run_omip as RO

    class _Captured(Exception):
        pass

    def _capture(model, state, grid_type, grid, z_coord, dt, *a, **k):
        raise _Captured((model, state, grid, z_coord, dt))

    args = RO.parse_args(argv)
    saved, RO._run_omip_loop = RO._run_omip_loop, _capture
    try:
        RO.run_omip_single("latlon", args)
    except _Captured as c:
        return c.args[0]
    finally:
        RO._run_omip_loop = saved
    raise SystemExit("run_omip_single returned without reaching _run_omip_loop")


def make_step(model, dt):
    """``model.step``'s restoring-mode path with a (possibly traced) config override."""
    pc = model.config

    def step(s, cfg, fw=None, sf=None):
        n = model._step_impl(s, dt, freshwater=fw, surface_forcing=sf, config=cfg)
        if pc.polar_filter.use_polar_filter:
            n = model._apply_polar_filter(n, dt)
        if pc.freeze_floor:
            n = model._apply_freeze_floor(n)
        if getattr(pc, "ew_cyclic_overlap", False):
            n = model._apply_ew_cyclic_overlap(n)
        return n
    return step


def gate_failure(loss, grads):
    """Non-finite loss or gradient on ONE sample -> message, else None (the gate's max over
    samples uses fmax, which would hide a NaN sample)."""
    bad = [n for n, g in grads.items() if not np.isfinite(float(g))]
    if np.isfinite(float(loss)) and not bad:
        return None
    return f"loss {float(loss)}, non-finite gradient for {bad}"


def _first_nan_op(step, s, cfg, fw, sf):
    """Re-run ONE step eagerly under jax_debug_nans; the first primitive that makes a NaN
    raises.  May name a NaN in a branch a ``where`` later discards (then look further)."""
    import traceback
    jax.config.update("jax_debug_nans", True)
    try:
        with jax.disable_jit():
            step(s, cfg, fw, sf)
        return "none raised eagerly"
    except FloatingPointError as e:
        fr = [f for f in traceback.extract_tb(e.__traceback__) if "/packages/" in f.filename]
        return f"{e} | " + " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                                      for f in fr[::-1][:6])
    finally:
        jax.config.update("jax_debug_nans", False)


def first_nonfinite(step, cfg, s, x, day_forcing, spd, days, forcing, O, lat_c, wet3=None,
                    s0_T=None, s0_S=None, hfull=None):
    """Step the forward model un-differentiated and name the first non-finite field
    (step, field, count, first index with its latitude); also checks the IC and targets."""
    for k in O.FIELDS:
        for what, arr in (("ic", x["ic"][k]), ("target", x["tgt"][k]), ("weight", x["w"][k])):
            n = int((~np.isfinite(np.asarray(arr))).sum())
            if n:
                return f"{what} {k}: {n} non-finite"
    jstep = jax.jit(lambda s_, fw_, sf_: step(s_, cfg, fw_, sf_))
    print("  locator IC: " + " ".join(f"{k}[{np.min(np.asarray(v)):.3g},{np.max(np.asarray(v)):.3g}]"
                                      for k, v in x["ic"].items()), flush=True)
    if wet3 is not None and forcing == "none":
        # control: same IC, but dry (below-seabed / land) T,S taken from the model's own
        # setup state instead of 0 -- does step 1 still produce the eta shock?
        d3 = jnp.asarray(~wet3)
        s_ctl = s._replace(T=s.T.replace(data=jnp.where(d3, s0_T, s.T.data)),
                           S=s.S.replace(data=jnp.where(d3, s0_S, s.S.data)))
        s_rest = s._replace(u=s.u.replace(data=jnp.zeros_like(s.u.data)),
                            v=s.v.replace(data=jnp.zeros_like(s.v.data)))
        s_flat = s._replace(eta=s.eta.replace(data=jnp.zeros_like(s.eta.data)))
        for lab, st in (("IC as built", s), ("dry T,S from setup state", s_ctl),
                        ("u=v=0", s_rest), ("eta=0", s_flat)):
            o = jstep(st, None, None)
            e = np.asarray(o.eta.data)
            print(f"  locator control [{lab}] step 1 eta[{e.min():.3g},{e.max():.3g}] "
                  f"u[{np.asarray(o.u.data).min():.3g},{np.asarray(o.u.data).max():.3g}]", flush=True)
        r = s_rest
        for i in range(spd):
            r = jstep(r, None, None)
        print(f"  locator control [u=v=0] after {spd} steps: " + " ".join(
            f"{k}[{np.nanmin(np.asarray(v)):.3g},{np.nanmax(np.asarray(v)):.3g}]"
            for k, v in O.state_fields(r).items()), flush=True)
        r2 = s._replace(u=s_rest.u, v=s_rest.v, eta=s_flat.eta)
        T0, S0, w3 = np.asarray(s.T.data), np.asarray(s.S.data), np.asarray(wet3)
        for nm, A in (("S", S0), ("T", T0)):
            for ax, axn in ((1, "lon"), (0, "lat")):
                if ax == 1:
                    d = np.abs(np.roll(A, -1, 1) - A); ok = w3 & np.roll(w3, -1, 1)
                else:
                    d = np.zeros_like(A); d[:-1] = np.abs(A[1:] - A[:-1])
                    ok = np.zeros_like(w3); ok[:-1] = w3[1:] & w3[:-1]
                d = np.where(ok, d, 0.0)
                for f in np.argsort(d.ravel())[-4:][::-1]:
                    j = np.unravel_index(f, d.shape)
                    print(f"  locator IC jump {nm} along {axn} at {tuple(int(q) for q in j)} "
                          f"(lat {lat_c[j[0]]:.1f}): |d{nm}|={d[j]:.3g} value {A[j]:.4g}, "
                          f"wet levels in column {int(w3[j[:2]].sum())}", flush=True)
        def _down_fill(A, ok):
            # PROBE ONLY: invalid wet cells copy the nearest valid value above in the same column
            A = A.copy(); have = ok[..., 0].copy(); last = A[..., 0].copy()
            for l in range(1, A.shape[-1]):
                okl = ok[..., l]
                A[..., l] = np.where(okl, A[..., l], np.where(have & w3[..., l], last, A[..., l]))
                last = np.where(okl, A[..., l], last); have |= okl
            return A
        okT = np.asarray(x["w"]["T"][0]) > 0
        print(f"  locator IC: {int((w3 & ~okT).sum())} of {int(w3.sum())} wet T cells lack ORAS5 data; "
              f"cell (96,121,59) has data: {bool(okT[96, 121, 59])}, (96,122,59): {bool(okT[96, 122, 59])}",
              flush=True)
        r3 = s._replace(T=s.T.replace(data=jnp.asarray(_down_fill(T0, okT))),
                        S=s.S.replace(data=jnp.asarray(_down_fill(S0, np.asarray(x["w"]["S"][0]) > 0))),
                        u=s_rest.u, v=s_rest.v, eta=s_flat.eta)
        for i in range(0):
            r3 = jstep(r3, None, None)
            if i in (0, 1, 5) or i % 12 == 11:
                print(f"  locator control [ORAS5 T,S filled downward, rest] step {i + 1}: " + " ".join(
                    f"{k}[{np.nanmin(np.asarray(v)):.3g},{np.nanmax(np.asarray(v)):.3g}]"
                    for k, v in O.state_fields(r3).items()), flush=True)
        from legoesm import constants as C
        from legoesm.ocean.eos import wright_eos
        zmid = np.cumsum(hfull) - 0.5 * hfull
        rho = np.asarray(wright_eos(jnp.asarray(T0), jnp.asarray(S0), jnp.asarray(C.rho_ocean * C.g * zmid)))
        pk = C.g * (np.cumsum(rho * hfull, -1) - 0.5 * rho * hfull)   # hydrostatic, full cells, eta=0
        dlat = np.deg2rad(abs(lat_c[1] - lat_c[0]))
        for ax, axn in ((1, "lon"), (0, "lat")):
            if ax == 1:
                dp = np.roll(pk, -1, 1) - pk; ok = w3 & np.roll(w3, -1, 1)
                dx = (C.R_earth * np.cos(np.deg2rad(lat_c)) * dlat)[:, None, None]
            else:
                dp = np.zeros_like(pk); dp[:-1] = pk[1:] - pk[:-1]
                ok = np.zeros_like(w3); ok[:-1] = w3[1:] & w3[:-1]
                dx = C.R_earth * dlat
            acc = np.where(ok, np.abs(dp) / C.rho_ocean / dx, 0.0)
            v_ = acc[ok]
            print(f"  locator IC baroclinic accel along {axn} (full cells, eta=0): median {np.median(v_):.2e} "
                  f"p99 {np.percentile(v_, 99):.2e} p99.99 {np.percentile(v_, 99.99):.2e} m/s2", flush=True)
            for f in np.argsort(acc.ravel())[-5:][::-1]:
                j = np.unravel_index(f, acc.shape)
                j2 = (j[0], (j[1] + 1) % acc.shape[1], j[2]) if ax == 1 else (j[0] + 1, j[1], j[2])
                print(f"    {acc[j]:.2e} at {tuple(int(q) for q in j)} (lat {lat_c[j[0]]:.1f}) lev {j[2]}: "
                      f"T {T0[j]:.3g}/{T0[j2]:.3g} S {S0[j]:.3g}/{S0[j2]:.3g} "
                      f"ORAS5 data {bool(okT[j])}/{bool(okT[j2])} wet levels {int(w3[j[:2]].sum())}/{int(w3[j2[:2]].sum())}",
                      flush=True)
        polar = jnp.asarray(np.broadcast_to((np.abs(lat_c) >= 60.0)[:, None, None], w3.shape))
        r4 = s._replace(T=s.T.replace(data=jnp.where(polar, s0_T, s.T.data)),
                        S=s.S.replace(data=jnp.where(polar, s0_S, s.S.data)),
                        u=s_rest.u, v=s_rest.v, eta=s_flat.eta)
        for i in range(0):
            r4 = jstep(r4, None, None)
            if i in (0, 1, 5, 11, 23, 47):
                print(f"  locator control [ORAS5 T,S only |lat|<60, rest] step {i + 1}: " + " ".join(
                    f"{k}[{np.nanmin(np.asarray(v)):.3g},{np.nanmax(np.asarray(v)):.3g}]"
                    for k, v in O.state_fields(r4).items()), flush=True)
        nw = w3.sum(-1)
        bot = np.zeros_like(w3); ii, jj = np.nonzero(nw > 0); bot[ii, jj, nw[ii, jj] - 1] = True
        for i in range(spd):
            r2 = jstep(r2, None, None)
            if i < 4:
                Tn = np.asarray(r2.T.data)
                lo, hi = T0[w3].min() - 0.5, T0[w3].max() + 0.5
                badc = w3 & ((Tn < lo) | (Tn > hi))
                if badc.any():
                    j = np.unravel_index(np.argmin(np.where(w3, Tn, np.inf)), Tn.shape)
                    lv = np.bincount(np.nonzero(badc)[2], minlength=w3.shape[2])
                    print(f"  locator rest step {i + 1} T out of IC range [{lo:.3g},{hi:.3g}]: {int(badc.sum())} cells; "
                          f"levels with most: {[(int(q), int(lv[q])) for q in np.argsort(lv)[-5:][::-1]]}; "
                          f"min T {Tn[j]:.3g} at {tuple(int(q) for q in j)} (lat {lat_c[j[0]]:.1f}), "
                          f"wet levels {int(w3[j[:2]].sum())}, cell thickness {hfull[j[2]]:.3g} m, "
                          f"T column IC lev-1..+1 {[round(float(T0[j[0], j[1], q]), 2) for q in range(max(j[2] - 1, 0), min(j[2] + 2, w3.shape[2]))]}",
                          flush=True)
            if i in (0, 1, 3, 5, 11, 23):
                for k in ("u", "v"):
                    a_ = np.abs(np.asarray(getattr(r2, k).data))
                    sl = (slice(None), slice(0, w3.shape[1])) if k == "u" else (slice(0, w3.shape[0]),)
                    a_ = a_[sl]
                    b_, n_ = np.where(bot & w3, a_, 0), np.where(~bot & w3, a_, 0)
                    top = np.argsort(np.where(w3, a_, 0).ravel())[-1000:]
                    print(f"  locator rest step {i + 1} |{k}|: max bottom-cell {b_.max():.3g}, max interior "
                          f"{n_.max():.3g} at {tuple(int(q) for q in np.unravel_index(np.argmax(n_), n_.shape))}; "
                          f"share of top-1000 that are bottom cells {bot.ravel()[top].mean():.2f} "
                          f"(bottom cells are {bot.sum() / w3.sum():.3f} of wet)", flush=True)
            if i in (0, 1):
                for k in ("eta", "v", "u"):
                    a_ = np.abs(np.asarray(getattr(r2, k).data))
                    j = np.unravel_index(np.argmax(a_), a_.shape)
                    col = tuple(int(q) for q in j[:2])
                    lv = int(j[2]) if a_.ndim == 3 else 0
                    print(f"  locator rest step {i + 1} max|{k}| {a_[j]:.3g} at {tuple(int(q) for q in j)} "
                          f"(lat {lat_c[j[0]]:.1f}), wet levels {int(w3[col].sum())}; "
                          f"T,S there at lev {lv}: {T0[col + (lv,)]:.3g},{S0[col + (lv,)]:.3g}; "
                          f"S neighbours E/W/N/S: " + ",".join(
                              f"{S0[(col[0] + a) % S0.shape[0], (col[1] + b) % S0.shape[1], lv]:.3g}"
                              f"{'' if w3[(col[0] + a) % S0.shape[0], (col[1] + b) % S0.shape[1], lv] else '(dry)'}"
                              for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0))), flush=True)
            if i < 6 or i % 6 == 5:
                print(f"  locator control [ORAS5 T,S, u=v=eta=0] step {i + 1}: " + " ".join(
                    f"{k}[{np.nanmin(np.asarray(v)):.3g},{np.nanmax(np.asarray(v)):.3g}]"
                    for k, v in O.state_fields(r2).items()), flush=True)
        z = s._replace(T=s.T.replace(data=s0_T), S=s.S.replace(data=s0_S),
                       u=s_rest.u, v=s_rest.v, eta=s_flat.eta)
        for i in range(spd):
            z = jstep(z, None, None)
        print(f"  locator control [setup T,S, u=v=eta=0] after {spd} steps: " + " ".join(
            f"{k}[{np.nanmin(np.asarray(v)):.3g},{np.nanmax(np.asarray(v)):.3g}]"
            for k, v in O.state_fields(z).items()), flush=True)
        print(f"  locator setup state dry S range [{float(jnp.where(d3, s0_S, jnp.inf).min()):.3g},"
              f"{float(jnp.where(d3, s0_S, -jnp.inf).max()):.3g}]", flush=True)
    for d in range(days):
        f = jax.tree.map(lambda v: v[d], x["frc"]) if forcing != "none" else None
        for i in range(spd):
            fw, sf = day_forcing(s, f) if f is not None else (None, None)
            prev, s = s, jstep(s, fw, sf)
            sfd = {k: np.asarray(v) for k, v in O.state_fields(s).items()}
            if i < 3:
                for k in ("S", "eta"):
                    j = np.unravel_index(np.argmin(sfd[k]), sfd[k].shape)
                    ic_v = np.asarray(x["ic"][k])[j]
                    print(f"  locator step {i + 1} min {k} at {tuple(int(q) for q in j)} "
                          f"(lat {lat_c[j[0]]:.1f}): {sfd[k][j]:.4g}, IC there {ic_v:.4g}, "
                          f"wet3 there {None if wet3 is None else bool(wet3[j] if wet3.ndim == len(j) else wet3[j + (0,)])}, "
                          f"wet levels in column {None if wet3 is None else int(wet3[j[:2]].sum())}", flush=True)
            print(f"  locator step {i + 1}: " + " ".join(
                f"{k}[{np.nanmin(v):.3g},{np.nanmax(v):.3g}]" for k, v in sfd.items()), flush=True)
            if wet3 is not None:   # PROBE ONLY: where wet T leaves [-3, 35] degC
                Tw = np.where(wet3, sfd["T"], np.nan)
                bad = wet3 & ((sfd["T"] < -3.0) | (sfd["T"] > 35.0))
                if bad.any():
                    j = np.unravel_index(np.nanargmax(np.abs(Tw - 16.0)), Tw.shape)
                    nw = int(wet3[j[:2]].sum())
                    print(f"  locator step {i + 1} wet T outside [-3,35]: {int(bad.sum())} cells, "
                          f"levels {np.unique(np.nonzero(bad)[2])[:8].tolist()}; worst {Tw[j]:.3g} at "
                          f"{tuple(int(q) for q in j)} (lat {lat_c[j[0]]:.1f}), wet levels {nw}, "
                          f"bottom cell {j[2] == nw - 1}, IC T {float(np.asarray(x['ic']['T'])[j]):.3g}, "
                          f"|u|,|v| there {abs(sfd['u'][j]):.3g},{abs(sfd['v'][j]):.3g}", flush=True)
                if hfull is not None:   # PROBE ONLY: vertical Courant |w| dt / h
                    cw = np.where(wet3, np.abs(np.asarray(s.w.data)) * (86400.0 / spd) / hfull, 0.0)
                    jw = np.unravel_index(np.argmax(cw), cw.shape)
                    msg = (f"  locator step {i + 1} vertical Courant: max per level 0-9 "
                           f"{[round(float(cw[..., k].max()), 2) for k in range(10)]}; global max "
                           f"{cw[jw]:.3g} at {tuple(int(q) for q in jw)}; cells >1 {int((cw > 1).sum())}")
                    if bad.any():
                        msg += f"; at worst-T cell {cw[j]:.3g}, column max {cw[j[:2]].max():.3g}"
                    print(msg, flush=True)
            # EVERY carried float leaf, every step (not only the five scored fields)
            leaves = jax.tree_util.tree_flatten_with_path(s)[0]
            for kp, arr in leaves:
                if not jnp.issubdtype(jnp.asarray(arr).dtype, jnp.floating):
                    continue
                k = jax.tree_util.keystr(kp)
                bad = ~np.isfinite(np.asarray(arr))
                if bad.any():
                    idx = tuple(int(v[0]) for v in np.nonzero(bad))
                    return (f"day {d} step {i + 1} leaf {k}: {int(bad.sum())} of {bad.size}, first "
                            f"{idx}; "
                            f"first NaN-producing operation: {_first_nan_op(step, prev, cfg, fw, sf)}")
    return "none: the forward state stays finite"


def d20_mae(model, oras5, keep, cosw):
    """Area-weighted mean |model - ORAS5| D20 per lead over the scored cells; arrays
    (sample, lead, lat, lon), ``cosw`` (lat, lon).  Unscored (incl. dry, NaN) cells drop out."""
    w = cosw[None, None] * keep
    err = np.where(keep, np.abs(model - oras5), 0.0)
    return (err * w).sum(axis=(0, 2, 3)) / w.sum(axis=(0, 2, 3))


def sample_dates(first_year, last_year, days, n, seed):
    lo = np.datetime64(f"{first_year}-01-01")
    hi = np.datetime64(f"{last_year}-12-31") - np.timedelta64(days, "D")
    span = int((hi - lo) / np.timedelta64(1, "D")) + 1
    pick = np.random.default_rng(seed).choice(span, size=n, replace=False)
    return [lo + np.timedelta64(int(i), "D") for i in sorted(pick)]


def noleap_day(date):
    """Noleap day-of-year index (Feb 29 pooled with Feb 28), as the climatologies."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.data.build_ocean_wb_climatologies import noleap_doy
    return noleap_doy(date)


def forcing_arrays(a, date, days):
    """Per rollout day k (ERA5 of date+k): ERA5 channels, ice climatology, runoff climatology."""
    from legoesm.training.era5_ocean_forcing import load_era5_day
    ice = np.load(a.clim_dir / "siconc_oras5_1993_2017_daily.npy", mmap_mode="r")
    ro = np.load(a.clim_dir / "runoff_jra55do_daily.npy", mmap_mode="r")
    ds = [date + np.timedelta64(k, "D") for k in range(days)]
    doy = [noleap_day(d) for d in ds]
    return {"era5": jnp.asarray(np.stack([load_era5_day(a.era5_store, d) for d in ds]), jnp.float64),
            "ice": jnp.asarray(np.stack([ice[k] for k in doy]), jnp.float64),
            "runoff": jnp.asarray(np.stack([ro[k] for k in doy]), jnp.float64),
            "day": jnp.asarray([k + 0.5 for k in doy], jnp.float64)}


def build_sample(O, store, date, days, wet3, um, vm, base, place):
    a0 = O.load_day(store, date)
    ic, ic_val = O.day_to_fields(a0, wet3, um, vm)
    x_ic = dict(ic)
    x_ic["T"], x_ic["S"] = place(a0)
    tgts, ws = [], []
    for k in range(1, days + 1):
        t, tv = O.day_to_fields(O.load_day(store, date + np.timedelta64(k, "D")), wet3, um, vm)
        tgts.append({f: jnp.asarray(t[f]) for f in O.FIELDS})
        ws.append(O.weights(base, {f: tv[f] & ic_val[f] for f in O.FIELDS}))
    stack = lambda xs: {f: jnp.stack([x[f] for x in xs]) for f in O.FIELDS}
    return ({"ic": {f: jnp.asarray(x_ic[f]) for f in O.FIELDS}, "tgt": stack(tgts),
             "w": stack(ws)}, ic, O.weights(base, ic_val))


def main(argv=None):
    a = parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    from legoesm.ml.training import TrainingConfig, create_optimizer
    from legoesm.training import oras5_ocean as O
    from legoesm.training.ema import ema_update, init_ema
    from legoesm.training.inert_params import assert_no_inert_over
    from legoesm.training.trainable_ocean_params import (
        LATLON_DYNAMICS_TRAINABLE, TrainableOceanParams, names)
    import optax

    label = "PLUMBING ONLY (unforced)" if a.forcing == "none" else a.forcing
    a.out.mkdir(parents=True, exist_ok=True)
    dz = write_dz_file(a.mesh_mask, a.dz_file)
    argv_omip = ["--grid", "latlon", "--resolution", "180x360", "--nlev", str(dz.size),
                 "--dz-ref-file", str(a.dz_file), "--H-max", f"{dz.sum():.9f}",
                 "--dt", str(a.dt), "--days", "1",
                 "--bathymetry", str(REPO / "data/bathymetry/etopo_1deg.nc"),
                 "--implicit-vertical-mixing", "--north-cap-lat", "80", "--precision", "fp64",
                 "--output", str(a.out / "omip_setup")]
    import os
    if os.environ.get("OWB_WOA_NC"):   # PROBE ONLY: setup T,S placed by the OMIP driver from this file
        argv_omip += ["--woa-init", "--woa-t", os.environ["OWB_WOA_NC"], "--woa-s", os.environ["OWB_WOA_NC"],
                      "--woa-t-var", "thetao", "--woa-s-var", "so"]
    dates = sample_dates(a.first_year, a.last_year, a.days, a.n_samples, a.seed)
    cap, n_gap = O.depth_cap([O.load_day(a.store, d) for d in dates], dz)
    cap_file = a.dz_file.with_name("oras5_depth_cap.npy")
    np.save(cap_file, cap)
    print(f"[{label}] ORAS5 depth cap -> {cap_file} ({int((cap > 0).sum())} wet columns, "
          f"{n_gap} with valid cells below a gap, ignored)", flush=True)
    argv_omip += ["--depth-cap-file", str(cap_file)]
    import xarray as xr
    nat = xr.open_dataset(a.native_mask)["thetao_oras"].isel(time=0)
    nat = nat.transpose("depth", "latitude", "longitude")
    fu, fv = O.face_walls(np.isfinite(nat.values), nat.latitude.values, nat.longitude.values)
    wall_file = a.dz_file.with_name("oras5_face_walls.npz")
    np.savez(wall_file, face_nlev_u=fu, face_nlev_v=fv, source=str(a.native_mask))
    for name, (j, c, uv) in {"Gibraltar 36N 354.5E": (125, 355, "u"),
                             "Bab-el-Mandeb 12.5N 43.5E": (102, 44, "u"),
                             "Makassar 2S 118E": (88, 118, "v"),
                             "Denmark Strait 66N 333E": (156, 333, "v")}.items():
        print(f"[{label}] face walls: {name} open levels {int((fu if uv == 'u' else fv)[j, c])}",
              flush=True)
    argv_omip += ["--face-wall-file", str(wall_file)]
    model, state0, grid, z_coord, dt_run = capture_model(argv_omip)
    assert float(dt_run) == a.dt, dt_run
    O.check_levels(a.store, z_coord.z_full_ref)
    lat_c = np.rad2deg(np.asarray(grid.lat)).ravel()
    # the model's own wet geometry: partial bottom cells, nothing below its seabed
    zc = model.z_coord
    wet3 = (np.asarray(state0.land_mask.data)[..., None] > 0) & np.asarray(zc.is_active)
    n_bad = int((wet3 & ~np.isfinite(np.moveaxis(np.asarray(O.load_day(a.store, dates[0])[:O.NLEV]), 0, -1))).sum())
    if n_bad:
        raise SystemExit(f"[{label}] {n_bad} model-wet cells where ORAS5 has no T despite the depth cap")
    um, vm = np.asarray(state0.u_mask.data), np.asarray(state0.v_mask.data)
    base = O.base_weights(lat_c, np.asarray(zc.h_partial) * wet3)
    from legoesm.ocean.vertical import compute_centroid_depth
    H = jnp.asarray(state0.H_bathy.data, jnp.float64)
    cell_depths = np.asarray(compute_centroid_depth(jnp.zeros_like(H), jnp.maximum(H, 1e-3), zc))
    meta = O.read_meta(a.store)
    place = lambda a0: O.place_ts(a0, meta, grid, zc, cell_depths, state0.land_mask.data)
    built = [build_sample(O, a.store, d, a.days, wet3, um, vm, base, place) for d in dates]
    std = O.field_std([b[1] for b in built], [b[2] for b in built])    # fixed, pooled IC days
    samples = [dict(b[0], std={f: jnp.asarray(std[f]) for f in O.FIELDS}) for b in built]
    print(f"[{label}] loss normaliser std {std}", flush=True)
    if a.forcing != "none":
        for x, d in zip(samples, dates):
            x["frc"] = forcing_arrays(a, d, a.days)
    model.prime_step_caches(O.set_state_fields(state0, samples[0]["ic"]))
    base_cfg = model.config
    step = make_step(model, a.dt)
    from legoesm.coupler.config import CouplerConfig
    from legoesm.training.era5_ocean_forcing import surface_forcing
    # OMIP's LY09 bulk config, with ERA5's reference heights: wind 10 m, T/q 2 m
    coupler_cfg = CouplerConfig(bulk_scheme="large_yeager", z_ref=10.0, z_t_atm=2.0,
                                z_q_atm=2.0, ocean_albedo=0.06)
    lat_rad = jnp.asarray(np.deg2rad(lat_c))[:, None]
    lon_rad = jnp.asarray(np.deg2rad(np.rad2deg(np.asarray(grid.lon)).ravel()))[None, :]
    ocean2 = jnp.asarray(wet3[..., 0])

    def day_forcing(c, f):
        """(freshwater, surface_forcing) for one model day, or (None, None) unforced."""
        if a.forcing == "none":
            return None, None
        return surface_forcing(a.forcing, f["era5"], c.T.data[..., 0], f["runoff"], f["ice"],
                               ocean2, lat_rad=lat_rad, lon_rad=lon_rad, day=f["day"],
                               ref_year=1993, coupler_cfg=coupler_cfg)   # zenith uses doy only
    spd = int(round(86400.0 / a.dt))

    def run_day(c, cfg, f):
        """One model day; forcing (if any) recomputed every step from the current SST."""
        @jax.checkpoint
        def stp(cc, _):
            fw, sf = day_forcing(cc, f)
            return step(cc, cfg, fw, sf), None
        return jax.lax.scan(stp, c, None, length=spd)[0]

    def day_inputs(x):
        return x["frc"] if a.forcing != "none" else jnp.zeros(a.days)

    def loss_fn(params, x):
        cfg = params.to_latlon_config(base_cfg)

        @jax.checkpoint
        def day(c, tw):
            c = run_day(c, cfg, tw[2])
            return c, O.l1_loss(O.state_fields(c), tw[0], tw[1], x["std"])

        s = O.set_state_fields(state0, x["ic"])
        return jnp.mean(jax.lax.scan(day, s, (x["tgt"], x["w"], day_inputs(x)))[1])

    params = TrainableOceanParams.from_values(
        {n: float(base_cfg.flat_get(n)) for n in names(LATLON_DYNAMICS_TRAINABLE)},
        LATLON_DYNAMICS_TRAINABLE)
    kn = names(LATLON_DYNAMICS_TRAINABLE)

    # gate: every trained knob carries a finite nonzero gradient on the samples (any sample)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    t0 = time.time()
    per = []
    import os
    if os.environ.get("OWB_LOCATE_ONLY"):   # PROBE ONLY: forward locator, no gradient
        x = samples[0]
        print(f"[{label}] first non-finite forward state: "
              f"{first_nonfinite(step, base_cfg, O.set_state_fields(state0, x['ic']), x, day_forcing, spd, a.days, a.forcing, O, lat_c, wet3, state0.T.data, state0.S.data, np.asarray(model.z_coord.h_partial).max(axis=(0, 1)))}",
              flush=True)
        raise SystemExit(0)
    for j, x in enumerate(samples):
        l0, g = vg(params, x)
        per.append({n: g.raw[i] for i, n in enumerate(kn)})
        msg = gate_failure(l0, per[-1])
        if msg:
            if not np.isfinite(float(l0)):
                print(f"[{label}] first non-finite forward state: "
                      f"{first_nonfinite(step, base_cfg, O.set_state_fields(state0, x['ic']), x, day_forcing, spd, a.days, a.forcing, O, lat_c, wet3, state0.T.data, state0.S.data, np.asarray(model.z_coord.h_partial).max(axis=(0, 1)))}",
                      flush=True)
            raise SystemExit(f"[{label}] GATE FAILED: sample {j} ({dates[j]}) {msg}")
    gate = {n: float(max(abs(float(p[n])) for p in per)) for n in kn}
    print(f"[{label}] gate |grad| max over {len(samples)} samples: {gate}", flush=True)
    assert_no_inert_over(per)
    print(f"[{label}] GATE PASSED ({time.time() - t0:.0f} s)", flush=True)

    steps = a.epochs * len(samples)
    opt = create_optimizer(TrainingConfig(lr=a.lr, warmup_steps=min(a.warmup, steps - 1),
                                          total_steps=steps, optimizer="adam",
                                          grad_clip_norm=1.0))
    opt_state = opt.init(params)
    ema = init_ema(params)
    hist_path = a.out / "training_history.jsonl"
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (a.out / "run_meta.json").write_text(json.dumps({
        "label": label, "forcing": a.forcing, "commit": commit, "dates": [str(d) for d in dates],
        "args": {k: str(v) for k, v in vars(a).items()}, "omip_argv": argv_omip,
        "gate_abs_grad_max": gate, "start_values": {n: float(v) for n, v in params.as_dict().items()}},
        indent=1))
    hist_path.write_text("")
    for ep in range(a.epochs):
        losses = []
        for i in np.random.default_rng(a.seed + 1 + ep).permutation(len(samples)):
            loss, g = vg(params, samples[i])         # one sample per update, one device
            upd, opt_state = opt.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            ema = ema_update(ema, params, a.ema_decay)
            losses.append(float(loss))
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "label": label,
               "params": {n: float(v) for n, v in params.as_dict().items()},
               "ema_params": {n: float(v) for n, v in ema.as_dict().items()}}
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{label}] epoch {ep} mean loss {row['train_loss']:.6f}", flush=True)
    # D20 DIAGNOSTIC (not in the loss): ORAS5 vs model, same isotherm_depth on both, scored where
    # ORAS5 crosses 20 C; EMA parameters, every sample and lead day.
    from legoesm.ocean.diagnostics import isotherm_depth
    z_c = jnp.abs(jnp.asarray(z_coord.z_full_ref))
    wet_j = jnp.asarray(wet3)
    cosw = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, wet3.shape[1]))

    @jax.jit
    def rollout_T(p, x):
        cfg = p.to_latlon_config(base_cfg)
        def day(c, f):
            c = run_day(c, cfg, f)
            return c, c.T.data
        return jax.lax.scan(day, O.set_state_fields(state0, x["ic"]), day_inputs(x))[1]

    d20_o, d20_m, keep = [], [], []
    for x in samples:
        Tm = rollout_T(ema, x)
        o = np.asarray(isotherm_depth(x["tgt"]["T"], z_c, wet_mask=wet_j[None]))
        m = np.asarray(isotherm_depth(Tm, z_c, wet_mask=wet_j[None]))
        Tt = np.asarray(x["tgt"]["T"])
        obs = np.all((np.asarray(x["w"]["T"]) > 0) | ~wet3[None], axis=-1)   # no filled level
        k = ((Tt[..., 0] > 20.0) & np.any((Tt <= 20.0) & wet3[None], axis=-1)
             & wet3[None, ..., 0] & obs)                    # ORAS5 crosses 20 C, observed column
        d20_o.append(o); d20_m.append(m); keep.append(k)
    d20_o, d20_m, keep = map(np.stack, (d20_o, d20_m, keep))       # (sample, lead, lat, lon)
    mae = d20_mae(d20_m, d20_o, keep, cosw)
    np.savez(a.out / "d20_diagnostic.npz", oras5=d20_o, model=d20_m, scored=keep,
             mae_by_lead_m=mae, lead_days=np.arange(1, a.days + 1), lat=lat_c,
             dates=np.array([str(d) for d in dates]), label=label, width_K=0.5)
    print(f"[{label}] D20 area-weighted MAE by lead day [m]: {mae.round(2).tolist()}", flush=True)
    print(f"[{label}] TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
