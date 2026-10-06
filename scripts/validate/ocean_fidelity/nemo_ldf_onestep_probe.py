"""One-step certification of the tripole NEMO lateral-viscosity port.

Feeds NEMO's own start-of-step velocity (RK3 Kbb = restart un/vn, the state
dyn_ldf reads at stage 3, stprk3_stg.F90:400 / dynldf.F90:70) and its qco
thickness e3t(Kbb) = e3t_0 * (1 + ssh/ht_0) to ``nemo_ldf_lap_viscosity_e3_cgrid``
with the coefficients ``attach_nemo_ldf_fields`` builds, and compares against
NEMO's per-step ``utrd_ldf``/``vtrd_ldf`` for that step (RUN_LDF1TS, step 289).

Geometry (masks, e3t_0) comes from NEMO's domain_cfg, not the model's own
partial cells: this certifies the OPERATOR, not the bathymetry.

Reports per level band: corr, regression ratio ours/NEMO, relative rmse;
separately for coast-adjacent faces, interior faces and the fold rows.
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import namedtuple
from pathlib import Path

import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

import jax.numpy as jnp  # noqa: E402
import netCDF4 as nc4  # noqa: E402

from legoesm.grids.tripole import create_tripole_grid, mesh_file_list  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks_3d, compute_vertex_mask, min_cell_to_vertex, nemo_fmask_shlat_3d,
    nemo_ldf_lap_viscosity_e3_cgrid,
)

ROOT = Path(__file__).resolve().parents[3]


def _driver():
    spec = importlib.util.spec_from_file_location(
        "run_omip_core2", ROOT / "scripts/run/run_omip_core2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path, names):
    with nc4.Dataset(path) as ds:
        return [np.asarray(ds.variables[n][:], np.float64).squeeze() for n in names]


def stats(a, b, sel):
    a, b = a[sel], b[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 10 or not np.any(b):
        return f"n={a.size:7d}  (empty)"
    c = np.corrcoef(a, b)[0, 1]
    r = float(np.dot(a, b) / np.dot(b, b))
    rr = float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
    return f"n={a.size:7d} corr={c:.6f} ratio={r:.6f} rel_rmse={rr:.3e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--domcfg", required=True)
    p.add_argument("--ldf", required=True)
    p.add_argument("--restart-npz", required=True, help="inner-domain un/vn/sshn at step N")
    p.add_argument("--trd-u", required=True)
    p.add_argument("--trd-v", required=True)
    p.add_argument("--rec", type=int, default=0, help="1ts record = first step after restart")
    p.add_argument("--rn-shlat", type=float, default=2.0)
    p.add_argument("--domcfg-metrics", action="store_true",
                   help="replace the grid's e2u/e1v by domain_cfg's (NEMO strait widths)")
    p.add_argument("--dump-at", default=None, help="k:r:c (NEMO inner indices) neighbourhood dump")
    p.add_argument("--debug-shift", default="0,0,0,0",
                   help="diagnostic only: roll ahmf by (dj,di) and ahmt by (dj,di)")
    a = p.parse_args()

    grid = create_tripole_grid(mesh_file_list(a.mesh))
    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    if a.domcfg_metrics:
        e2u_d, e1v_d = _read(a.domcfg, ("e2u", "e1v"))
        dy_u = np.asarray(grid.dy_u).copy(); dx_v = np.asarray(grid.dx_v).copy()
        dy_u[:331] = e2u_d[:, (np.arange(n_lon + 1) - 2) % 360]
        dx_v[1:332] = e1v_d[:, (np.arange(n_lon) - 1) % 360]
        nd = int((np.abs(dy_u - np.asarray(grid.dy_u)) > 1).sum() + (np.abs(dx_v - np.asarray(grid.dx_v)) > 1).sum())
        for nm, new_, old_ in (("dy_u", dy_u, np.asarray(grid.dy_u)), ("dx_v", dx_v, np.asarray(grid.dx_v))):
            dd = np.abs(new_ - old_) > 1.0
            jj, ii = np.nonzero(dd)
            print(f"[domcfg-metrics] {nm}: {dd.sum()} changed; cols hist {np.bincount(np.minimum(ii, 3))[:4]} "
                  f"(0,1,2,>=3); rows min/max {jj.min() if jj.size else -1}/{jj.max() if jj.size else -1}; "
                  f"old/new sample {[(int(a_), int(b_), round(float(old_[a_, b_])), round(float(new_[a_, b_]))) for a_, b_ in list(zip(jj, ii))[:4]]}")
        grid = grid._replace(dy_u=jnp.asarray(dy_u), dx_v=jnp.asarray(dx_v))
        print(f"[domcfg-metrics] e2u/e1v from {a.domcfg}: {nd} face widths changed")
    e3t0_i, top_i, bot_i = _read(a.domcfg, ("e3t_0", "top_level", "bottom_level"))
    nk = e3t0_i.shape[0]
    ci_t = (np.arange(n_lon) - 1) % 360

    def to_T(x):   # (..., 331, 360) inner -> full (332, 362, ...)
        x = x[..., :, ci_t]
        x = np.moveaxis(x, 0, -1) if x.ndim == 3 else x
        return np.concatenate([x, np.zeros_like(x[:1])], axis=0)

    k = np.arange(nk)[:, None, None]
    tmask_i = ((k >= top_i - 1) & (k <= bot_i - 1) & (bot_i > 0))
    act = to_T(tmask_i.astype(float)) > 0.5
    e3t0 = to_T(e3t0_i)
    rs = np.load(a.restart_npz)
    ht0 = np.sum(e3t0 * act, axis=-1)
    ssh = to_T(rs["sshn"])
    r3t = np.where(ht0 > 0, ssh / np.where(ht0 > 0, ht0, 1.0), 0.0)
    h_k = e3t0 * (1.0 + r3t[..., None]) * act

    ci_u = (np.arange(n_lon + 1) - 2) % 360           # our u col i <- NEMO U inner col
    u = np.moveaxis(rs["un"][:, :, ci_u], 0, -1)
    u = np.concatenate([u, np.zeros_like(u[:1])], axis=0)
    ci_v = (np.arange(n_lon) - 1) % 360
    v = np.moveaxis(rs["vn"][:, :, ci_v], 0, -1)       # our v row j <- NEMO V row j-1
    v = np.concatenate([np.zeros_like(v[:1]), v, np.zeros_like(v[:1])], axis=0)

    Zc = namedtuple("Zc", "n_levels is_active h_partial dz_ref nemo_ldf_ahmt nemo_ldf_ahmf nemo_e3f_0")
    zc = Zc(nk, jnp.asarray(act), jnp.asarray(h_k), jnp.asarray(e3t0_i.max(axis=(1, 2))),
            None, None, None)
    zc = _driver().attach_nemo_ldf_fields(zc, grid, a.mesh, a.ldf, a.domcfg, a.rn_shlat)

    fj, fi, tj, ti = (int(x) for x in a.debug_shift.replace(":", ",").split(","))
    if (fj, fi, tj, ti) != (0, 0, 0, 0):
        _af = np.asarray(zc.nemo_ldf_ahmf); _fmk = np.asarray(nemo_fmask_shlat_3d(jnp.asarray(act), grid, a.rn_shlat))
        _raw = np.where(_fmk > 0, _af / np.where(_fmk > 0, _fmk, 1.0), 0.0)
        _raw = np.where(_fmk > 0, _raw, np.roll(_raw, (-fj, -fi), axis=(0, 1)))  # fill land from neighbour before roll
        zc = zc._replace(nemo_ldf_ahmf=jnp.asarray(np.roll(_raw, (fj, fi), axis=(0, 1)) * _fmk),
                         nemo_ldf_ahmt=jnp.roll(zc.nemo_ldf_ahmt, (tj, ti), axis=(0, 1)))
        print(f"[debug-shift] ahmf ({fj},{fi}) ahmt ({tj},{ti})")
    um, vm = compute_face_masks_3d(jnp.asarray(act), grid)
    um = np.asarray(um, float); vm = np.asarray(vm, float)
    fm = np.asarray(zc.nemo_ldf_ahmf) > 0
    vm3 = np.stack([np.asarray(compute_vertex_mask(jnp.asarray(act[..., kk].astype(float)), grid=grid))
                    for kk in range(nk)], axis=-1)
    hk = jnp.asarray(h_k)
    h_vtx = jnp.where(vm3 > 0, min_cell_to_vertex(hk, grid), zc.nemo_e3f_0)
    tu, tv = nemo_ldf_lap_viscosity_e3_cgrid(
        jnp.asarray(u * um), jnp.asarray(v * vm), grid, zc.nemo_ldf_ahmt, zc.nemo_ldf_ahmf, hk,
        mask=jnp.asarray(act[..., 0].astype(float)), u_mask=jnp.asarray(um),
        v_mask=jnp.asarray(vm), vertex_mask=jnp.asarray(fm.astype(float)), h_vtx=h_vtx)
    tu, tv = np.asarray(tu), np.asarray(tv)
    _kw = dict(mask=jnp.asarray(act[..., 0].astype(float)), u_mask=jnp.asarray(um),
               v_mask=jnp.asarray(vm), vertex_mask=jnp.asarray(fm.astype(float)), h_vtx=h_vtx)
    du_div, dv_div = (np.asarray(x) for x in nemo_ldf_lap_viscosity_e3_cgrid(
        jnp.asarray(u * um), jnp.asarray(v * vm), grid, zc.nemo_ldf_ahmt,
        0.0 * zc.nemo_ldf_ahmf, hk, **_kw))
    du_cur, dv_cur = tu - du_div, tv - dv_div

    (nu,) = _read(a.trd_u, ("utrd_ldf",))
    (nv,) = _read(a.trd_v, ("vtrd_ldf",))
    nu = nu[a.rec] if nu.ndim == 4 else nu
    nv = nv[a.rec] if nv.ndim == 4 else nv
    # NEMO U inner (r, c) = our u (r, c+2); V inner (r, c) = our v (r+1, c+1)
    ours_u = tu[:331, 2:362].transpose(2, 0, 1)
    ours_v = tv[1:332, 1:361].transpose(2, 0, 1)
    wu = um[:331, 2:362].transpose(2, 0, 1) > 0
    wv = vm[1:332, 1:361].transpose(2, 0, 1) > 0
    coast_f = (np.asarray(zc.nemo_ldf_ahmf) > 0) & ~(vm3 > 0)
    cu = (coast_f[:-1] | coast_f[1:])[:331, 2:362].transpose(2, 0, 1)   # vertex S/N of u face
    cv = (coast_f[:, :-1] | coast_f[:, 1:])[1:332, 1:361].transpose(2, 0, 1)
    if a.dump_at:
        kq, rq, cq = (int(x) for x in a.dump_at.split(":"))
        sl = (slice(rq - 2, rq + 3), slice(cq - 2, cq + 3))
        np.set_printoptions(precision=3, linewidth=160)
        print("tmask (inner, rows S->N)\n", tmask_i[kq][sl].astype(int))
        print("our v-face mask\n", wv[kq][sl].astype(int)); print("our u-face mask\n", wu[kq][sl].astype(int))
        print("v ours\n", ours_v[kq][sl]); print("v nemo\n", nv[kq][sl])
        print("u ours\n", ours_u[kq][sl]); print("u nemo\n", nu[kq][sl])
        fmk = np.asarray(zc.nemo_ldf_ahmf)[..., kq][1:332, 2:363]   # vertex (r+1, c+2) = NEMO F inner (r, c)
        print("ahmf*fmask at NEMO F inner\n", fmk[sl])
        print("restart vn\n", rs["vn"][kq][sl]); print("restart un\n", rs["un"][kq][sl])
    rows = np.arange(331)[None, :, None]
    fold = rows >= 325
    print(f"[provenance] restart={a.restart_npz} trd={a.trd_u} rec={a.rec} rn_shlat={a.rn_shlat} fp64")
    for lab, ks in (("k 0-9", slice(0, 10)), ("k 10-39", slice(10, 40)), ("k 40-74", slice(40, nk))):
        kk = np.zeros(nk, bool); kk[ks] = True; kk = kk[:, None, None]
        for comp, o, n, w, c in (("u", ours_u, nu, wu, cu), ("v", ours_v, nv, wv, cv)):
            print(f"{lab:8s} {comp} all    ", stats(o, n, w & kk & ~fold))
            print(f"{lab:8s} {comp} coast  ", stats(o, n, w & kk & c & ~fold))
            print(f"{lab:8s} {comp} inter  ", stats(o, n, w & kk & ~c & ~fold))
            print(f"{lab:8s} {comp} fold   ", stats(o, n, w & kk & fold))
    (gphit,) = _read(a.mesh, ("gphit",))
    _lat = gphit[:331, 1:361][None]
    for comp, dd, cc, n, w, c in (("u", du_div[:331, 2:362].transpose(2, 0, 1), du_cur[:331, 2:362].transpose(2, 0, 1), nu, wu, cu),
                                  ("v", dv_div[1:332, 1:361].transpose(2, 0, 1), dv_cur[1:332, 1:361].transpose(2, 0, 1), nv, wv, cv)):
        for lo, hi, k1 in ((-20, 5, 10), (-20, 5, 75), (-90, 90, 75)):
            sel = w & ~c & ~fold & (_lat >= lo) & (_lat < hi) & (np.arange(nk)[:, None, None] < k1)
            A = np.stack([dd[sel], cc[sel]], 1); y = n[sel]
            ok = np.all(np.isfinite(A), 1) & np.isfinite(y)
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            res = y[ok] - A[ok] @ coef
            print(f"fit {comp} lat {lo}..{hi} k<{k1}: nemo = {coef[0]:.5f}*div + {coef[1]:.5f}*curl; "
                  f"rel_rmse fit {np.sqrt(np.mean(res**2)/np.mean(y[ok]**2)):.3e}; "
                  f"rms div/curl {np.sqrt(np.mean(A[ok,0]**2)):.2e}/{np.sqrt(np.mean(A[ok,1]**2)):.2e}")
    lat = gphit[:331, 1:361][None]
    for comp, o, n, w, c in (("u", ours_u, nu, wu, cu), ("v", ours_v, nv, wv, cv)):
        for lo, hi in ((-90, -50), (-50, -20), (-20, -5), (-5, 5), (5, 20), (20, 50), (50, 90)):
            band = (lat >= lo) & (lat < hi)
            print(f"interior {comp} lat {lo:4d}..{hi:3d}", stats(o, n, w & ~c & ~fold & band))
            if -20 <= lo < 5:
                for k0, k1 in ((0, 1), (1, 10), (10, 40), (40, nk)):
                    kk = np.zeros(nk, bool); kk[k0:k1] = True
                    print(f"      k {k0:2d}-{k1-1:2d}", stats(o, n, w & ~c & ~fold & band & kk[:, None, None]))
        (glam,) = _read(a.mesh, ("glamt",))
        lon = glam[:331, 1:361][None]
        sel = w & ~c & ~fold
        d = np.where(sel, np.abs(o - n), 0.0)
        tot = np.sum(d ** 2)
        idx = np.argsort(d.ravel())[::-1][:12]
        print(f"worst interior {comp} faces (share of squared error in top 12: "
              f"{np.sum(d.ravel()[idx] ** 2) / tot:.3f}):")
        for q in idx:
            kq, jq, iq = np.unravel_index(q, d.shape)
            print(f"   k={kq:2d} r={jq:3d} c={iq:3d} lat={lat[0, jq, iq]:7.2f} lon={lon[0, jq, iq]:7.2f} "
                  f"ours={o[kq, jq, iq]: .3e} nemo={n[kq, jq, iq]: .3e}")


if __name__ == "__main__":
    main()
