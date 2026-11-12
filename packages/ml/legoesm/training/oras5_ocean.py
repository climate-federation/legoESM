"""Daily ORAS5 (75 levels, 1 degree) -> lat-lon C-grid ocean fields + the ocean-WB loss.

Store: ``scripts/data/download_oras5_cmems_l75.py`` -- one float32 ``(304, 180, 360)`` ``.npy``
per day on the model's tracer grid (lat -89.5..89.5 S->N, lon 0..359), channels thetao, so, uo,
vo at 75 levels then zos, mlotst, siconc, sithick, NaN on land, ``meta.json`` alongside.

Vertical: ORAS5 level k IS model level k (ORAS5 depths equal eORCA1 L75 ``gdept_1d`` to 2e-4 m;
the model runs on eORCA1 ``e3t_1d`` and puts its centres at layer midpoints, <= 0.65 m away).
``check_levels`` refuses a model whose centres sit further than ``LEVEL_TOL_M`` from the store's.

Horizontal: T/S/eta at centres as stored; u, v averaged to C-grid faces (u: seam face from the
two edge columns; v: polar faces carry no target).  Wet model cells with no ORAS5 value are
filled by neighbour averaging for the initial condition and MASKED out of the loss.

Loss (Neptune-style L1, as the phase-1 probe): per variable, sum over valid cells of
|pred - target| x area x layer thickness / summed weights / that variable's weighted std
(one fixed std per variable, pooled over the training samples' initial days);
T, S, u, v, eta averaged with equal weight; eta with its area-weighted mean removed on both sides.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

NLEV, NLAT, NLON = 75, 180, 360
LEVEL_TOL_M = 1.0
FIELDS = ("T", "S", "u", "v", "eta")


def read_meta(store) -> dict:
    return json.loads((Path(store) / "meta.json").read_text())


def check_levels(store, z_full_m) -> None:
    d = np.asarray(read_meta(store)["depths_m"])
    z = np.abs(np.asarray(z_full_m, np.float64))
    if z.shape != d.shape or np.abs(z - d).max() > LEVEL_TOL_M:
        raise ValueError(f"model level centres differ from the ORAS5 store's by more than "
                         f"{LEVEL_TOL_M} m (max {np.abs(z - d).max() if z.shape == d.shape else 'n/a'};"
                         f" model {z.shape}, store {d.shape})")


def load_day(store, date) -> np.ndarray:
    """One stored day, ``(304, 180, 360)`` float32 (memory-mapped)."""
    a = np.load(Path(store) / f"{np.datetime64(date, 'D')}.npy", mmap_mode="r")
    if a.shape != (4 * NLEV + 4, NLAT, NLON):
        raise ValueError(f"{date}: bad ORAS5 day shape {a.shape}")
    return a


def _fill(f, valid, wet2, n_iter=200):
    """Fill wet cells lacking data by repeated 4-neighbour averaging of valid values."""
    f = np.where(valid, f, 0.0)
    v = valid.astype(float)
    for _ in range(n_iter):
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


def day_to_fields(a, wet3, u_mask, v_mask):
    """Stored day -> (fields, valid): model layouts T/S (lat,lon,lev), u (lat,lon+1,lev),
    v (lat+1,lon,lev), eta (lat,lon); ``valid`` marks where ORAS5 itself has data."""
    a = np.asarray(a, np.float64)
    wet3 = np.asarray(wet3) > 0
    out, val = {}, {}
    for k, name in enumerate(("T", "S", "u", "v")):
        f = a[NLEV * k:NLEV * (k + 1)]                       # (lev, lat, lon)
        ok = np.isfinite(f)
        lv = []
        for l in range(NLEV):
            if ok[l].any():
                lv.append(_fill(f[l], ok[l], wet3[..., l]))
            elif l == 0:
                raise ValueError(f"ORAS5 {name} has no valid surface cell")
            else:          # no ORAS5 data on this level at all: IC copies the level above (user 2026-11-12)
                lv.append(np.where(wet3[..., l], lv[-1], 0.0))
        f = np.stack(lv, 0)
        out[name] = np.moveaxis(f, 0, -1)
        val[name] = np.moveaxis(ok, 0, -1) & wet3
    zos = a[4 * NLEV]
    val["eta"] = np.isfinite(zos) & wet3[..., 0]
    out["eta"] = _fill(zos, np.isfinite(zos), wet3[..., 0])
    um = np.asarray(u_mask)[..., None] if np.ndim(u_mask) == 2 else np.asarray(u_mask)
    vm = np.asarray(v_mask)[..., None] if np.ndim(v_mask) == 2 else np.asarray(v_mask)
    u, vu = out["u"], val["u"]
    out["u"] = np.concatenate([0.5 * (u[:, -1:] + u[:, :1]), 0.5 * (u[:, :-1] + u[:, 1:]),
                               0.5 * (u[:, -1:] + u[:, :1])], axis=1) * um
    val["u"] = np.concatenate([vu[:, -1:] & vu[:, :1], vu[:, :-1] & vu[:, 1:],
                               vu[:, -1:] & vu[:, :1]], axis=1) & (um > 0)
    v, vv = out["v"], val["v"]
    out["v"] = np.concatenate([v[:1], 0.5 * (v[:-1] + v[1:]), v[-1:]], axis=0) * vm
    val["v"] = np.concatenate([vv[:1] & False, vv[:-1] & vv[1:], vv[-1:] & False], axis=0) & (vm > 0)
    out["eta"] = out["eta"] * wet3[..., 0]
    return out, val


def base_weights(lat_c_deg, h):
    """Area x thickness weights per variable layout (seam u face counted once).

    ``h``: layer thickness [m], ``(nlev,)`` or the model's per-cell ``(lat, lon, nlev)``
    (partial cells, zero below the seabed).  A face takes the THINNER of its two cells
    (NEMO ``e3u_0``/``e3v_0``); polar v faces carry no target, they take their one cell.
    """
    lat_c = np.asarray(lat_c_deg, np.float64)
    nlon = NLON
    h = np.asarray(h, np.float64)
    if h.ndim == 1:
        h = np.broadcast_to(h, (lat_c.size, nlon, h.size))
    area_c = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, nlon))
    lat_f = np.concatenate([[lat_c[0] - 0.5 * (lat_c[1] - lat_c[0])],
                            0.5 * (lat_c[:-1] + lat_c[1:]),
                            [lat_c[-1] + 0.5 * (lat_c[-1] - lat_c[-2])]])
    area_v = np.cos(np.deg2rad(lat_f))[:, None] * np.ones((1, nlon))
    area_u = np.cos(np.deg2rad(lat_c))[:, None] * np.ones((1, nlon + 1))
    area_u[:, -1] = 0.0
    hw = np.concatenate([h[:, -1:], h], axis=1)                     # west neighbour of face i
    h_u = np.minimum(hw, np.concatenate([h, h[:, :1]], axis=1))     # (lat, lon+1, lev)
    h_v = np.concatenate([h[:1], np.minimum(h[:-1], h[1:]), h[-1:]], axis=0)
    return {"T": area_c[..., None] * h, "S": area_c[..., None] * h,
            "u": area_u[..., None] * h_u, "v": area_v[..., None] * h_v,
            "eta": area_c * (h[..., 0] > 0)}


def weights(base, valid):
    return {k: jnp.asarray(base[k] * valid[k]) for k in FIELDS}


def field_std(fields_list, w_list):
    """Weighted std per variable pooled over several days (the fixed loss normaliser);
    eta with each day's weighted mean removed first, as :func:`l1_loss` does."""
    out = {}
    for k in FIELDS:
        xs, ws = [], []
        for f, w in zip(fields_list, w_list):
            x, wk = np.asarray(f[k], np.float64), np.asarray(w[k], np.float64)
            if k == "eta":
                x = x - (x * wk).sum() / wk.sum()
            xs.append(x.ravel()); ws.append(wk.ravel())
        x, wk = np.concatenate(xs), np.concatenate(ws)
        mu = (x * wk).sum() / wk.sum()
        out[k] = float(np.sqrt(((x - mu) ** 2 * wk).sum() / wk.sum()))
    return out


def l1_loss(pred, target, w, std):
    """Neptune-style L1; ``pred``/``target`` dicts of FIELDS arrays, ``w`` from :func:`weights`."""
    tot = 0.0
    for k in FIELDS:
        p, t, wk = pred[k], target[k], w[k]
        sw = jnp.sum(wk)
        if k == "eta":
            p = p - jnp.sum(wk * p) / sw
            t = t - jnp.sum(wk * t) / sw
        tot = tot + jnp.sum(wk * jnp.abs(p - t)) / sw / std[k]
    return tot / len(FIELDS)


def state_fields(state):
    return {"T": state.T.data, "S": state.S.data, "u": state.u.data, "v": state.v.data,
            "eta": state.eta.data}


def set_state_fields(state, f):
    return state._replace(T=state.T.replace(data=jnp.asarray(f["T"])),
                          S=state.S.replace(data=jnp.asarray(f["S"])),
                          u=state.u.replace(data=jnp.asarray(f["u"])),
                          v=state.v.replace(data=jnp.asarray(f["v"])),
                          eta=state.eta.replace(data=jnp.asarray(f["eta"])))
