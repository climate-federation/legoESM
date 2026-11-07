"""Rank horizontal in-situ density jumps between ADJACENT wet cells of an ocean state.

Tests the "initial-condition density wall" class (the MPAS 104 m/s blowup,
PR #1732) on any state: every wet neighbour pair, every level, |rho_A - rho_B|
and the implied geostrophic shear  g*|d rho|*h / (rho0*|f|*dx).  Reports the
global top-N and where a named box (default: Sea of Marmara) ranks.

Two inputs are understood:
  --orca-snapshot  run_omip tripole snapshot_final.npz (T, S (nj, ni, nlev),
                   lat_T, lon_T, land_mask, H_bathy, z_center_ref)
  --fesom-mesh     fesom_jax mesh dir (T_ic/S_ic (nod, nlev), geo_coord_nod2D
                   radians (lon, lat), edges (nedge, 2), zbar, nlevels_nod2D)

Density: Wright (1997) via legoesm.ocean.eos.wright_eos, potential T, sea
pressure rho0*g*depth (hydrostatic reference, the same approximation the
runaway-column probe used).  CPU, JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.eos import wright_eos

RHO0 = 1025.0
OMEGA = constants.Omega
R_EARTH = constants.R_earth


def _rho(T, S, depth_m):
    p = RHO0 * constants.g * np.asarray(depth_m, dtype=np.float64)
    return np.asarray(wright_eos(jnp.asarray(T), jnp.asarray(S), jnp.asarray(p)))


def _gc_dist(lat1, lon1, lat2, lon2):
    """Great-circle distance [m] from degrees."""
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _shear(drho, dx, h, lat):
    f = np.maximum(np.abs(2 * OMEGA * np.sin(np.radians(lat))), 2 * OMEGA * np.sin(np.radians(2.0)))
    return constants.g * drho * h / (RHO0 * f * dx)


def _report(name, recs, box, top):
    """recs: structured array with lat, lon, depth, drho, dx, shear."""
    order = np.argsort(recs["drho"])[::-1]
    print(f"\n=== {name}: {recs.size} wet adjacent pairs x levels; top {top} by |d rho| ===")
    print(f"{'rank':>5} {'lat':>8} {'lon':>8} {'depth':>7} {'drho':>7} {'dx_km':>6} {'geo_shear_m/s':>13}")
    for r, k in enumerate(order[:top], 1):
        e = recs[k]
        print(f"{r:5d} {e['lat']:8.3f} {e['lon']:8.3f} {e['depth']:7.1f} {e['drho']:7.3f} "
              f"{e['dx']/1e3:6.1f} {e['shear']:13.2f}")
    order_s = np.argsort(recs["shear"])[::-1]
    print(f"--- top {top} by implied geostrophic shear over the pair (g*drho*h/(rho0 f dx)) ---")
    for r, k in enumerate(order_s[:top], 1):
        e = recs[k]
        print(f"{r:5d} {e['lat']:8.3f} {e['lon']:8.3f} {e['depth']:7.1f} {e['drho']:7.3f} "
              f"{e['dx']/1e3:6.1f} {e['shear']:13.2f}")
    lat0, lat1, lon0, lon1 = box
    inbox = (recs["lat"] >= lat0) & (recs["lat"] <= lat1) & (recs["lon"] >= lon0) & (recs["lon"] <= lon1)
    print(f"--- box lat[{lat0},{lat1}] lon[{lon0},{lon1}]: {int(inbox.sum())} pairs ---")
    if inbox.any():
        kb = np.flatnonzero(inbox)
        kbest = kb[np.argmax(recs["drho"][kb])]
        rank = int(np.flatnonzero(order == kbest)[0]) + 1
        e = recs[kbest]
        print(f"box max |d rho| {e['drho']:.3f} kg/m3 at ({e['lat']:.3f},{e['lon']:.3f}) depth {e['depth']:.1f} m, "
              f"dx {e['dx']/1e3:.1f} km, shear {e['shear']:.2f} m/s -> GLOBAL RANK {rank} of {recs.size}")
        kbest = kb[np.argmax(recs["shear"][kb])]
        rank = int(np.flatnonzero(order_s == kbest)[0]) + 1
        e = recs[kbest]
        print(f"box max shear {e['shear']:.2f} m/s (|d rho| {e['drho']:.3f}, dx {e['dx']/1e3:.1f} km, "
              f"depth {e['depth']:.1f}) -> GLOBAL RANK {rank} by shear")
    print(f"global: median |d rho| {np.median(recs['drho']):.4f}, p99.9 {np.percentile(recs['drho'], 99.9):.3f}, "
          f"max {recs['drho'].max():.3f}; pairs with |d rho|>1: {int((recs['drho']>1).sum())}, "
          f">2: {int((recs['drho']>2).sum())}")


_DT = np.dtype([("lat", "f8"), ("lon", "f8"), ("depth", "f8"), ("drho", "f8"), ("dx", "f8"), ("shear", "f8")])


def probe_orca(path, box, top, keep_min_drho):
    s = np.load(path, allow_pickle=False)
    T, S = s["T"], s["S"]
    lat, lon = s["lat_T"], s["lon_T"]
    land, H, zc = s["land_mask"] > 0.5, s["H_bathy"], s["z_center_ref"]
    print(f"ORCA snapshot {path}: T {T.shape}, step {int(s['_step'])}, wet columns {int(land.sum())}")
    nj, ni, nl = T.shape
    # spacing between T-points (great circle), E-W pairs (j,i)-(j,i+1), N-S pairs (j,i)-(j+1,i)
    dx_e = _gc_dist(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dx_n = _gc_dist(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    out = []
    for k in range(nl):
        wet = land & (zc[k] < H)
        h = (H - zc[k]) if k == nl - 1 else np.minimum(H - zc[k], zc[k + 1] - zc[k])  # rough cell thickness
        h = np.maximum(h, 1.0)
        rho = _rho(T[:, :, k], S[:, :, k], zc[k])
        for (a, b, dx, la, lo, hh) in (
            (rho[:, :-1], rho[:, 1:], dx_e, lat[:, :-1], lon[:, :-1], np.minimum(h[:, :-1], h[:, 1:])),
            (rho[:-1, :], rho[1:, :], dx_n, lat[:-1, :], lon[:-1, :], np.minimum(h[:-1, :], h[1:, :])),
        ):
            w = (wet[:, :-1] & wet[:, 1:]) if a.shape == rho[:, :-1].shape else (wet[:-1, :] & wet[1:, :])
            d = np.abs(a - b)
            m = w & np.isfinite(d) & (d >= keep_min_drho)
            if not m.any():
                continue
            rec = np.empty(int(m.sum()), dtype=_DT)
            rec["lat"], rec["lon"], rec["depth"], rec["drho"], rec["dx"] = la[m], lo[m], zc[k], d[m], dx[m]
            rec["shear"] = _shear(rec["drho"], rec["dx"], hh[m], rec["lat"])
            out.append(rec)
    recs = np.concatenate(out)
    _report(f"ORCA12 step {int(s['_step'])} (pairs with |d rho| >= {keep_min_drho})", recs, box, top)
    # uniform (filled) columns inside the box
    lat0, lat1, lon0, lon1 = box
    inbox = land & (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
    jj, ii = np.nonzero(inbox)
    n_uniform = 0
    for j, i in zip(jj, ii):
        nw = int((zc < H[j, i]).sum())
        if nw >= 3 and np.ptp(T[j, i, :nw]) < 1e-6 and np.ptp(S[j, i, :nw]) < 1e-6:
            n_uniform += 1
    print(f"box: {jj.size} wet columns, {n_uniform} vertically UNIFORM T and S columns (fill signature)")
    return recs


def probe_fesom(mesh, box, top, keep_min_drho):
    T = np.load(f"{mesh}/T_ic.npy"); S = np.load(f"{mesh}/S_ic.npy")
    coord = np.load(f"{mesh}/geo_coord_nod2D.npy")          # radians (lon, lat)
    lon, lat = np.degrees(coord[:, 0]), np.degrees(coord[:, 1])
    edges = np.load(f"{mesh}/edges.npy")
    zbar = -np.load(f"{mesh}/zbar.npy")                      # positive down, (nl,)
    nlev = np.load(f"{mesh}/nlevels_nod2D.npy")
    print(f"FESOM mesh {mesh}: T_ic {T.shape}, edges {edges.shape}, zbar {zbar.shape}, lon range {lon.min():.1f}..{lon.max():.1f}")
    nl = T.shape[1]
    a, b = edges[:, 0], edges[:, 1]
    dx = _gc_dist(lat[a], lon[a], lat[b], lon[b])
    lat_e = 0.5 * (lat[a] + lat[b]); lon_e = lon[a]
    out = []
    for k in range(nl):
        depth = zbar[k]
        wet = nlev > k + 1                                   # level k is a wet T level if the column has more levels
        rho = _rho(T[:, k], S[:, k], depth)
        w = wet[a] & wet[b]
        d = np.abs(rho[a] - rho[b])
        h = zbar[min(k + 1, nl - 1)] - zbar[k] if k + 1 < nl else zbar[k] - zbar[k - 1]
        m = w & np.isfinite(d) & (d >= keep_min_drho)
        if not m.any():
            continue
        rec = np.empty(int(m.sum()), dtype=_DT)
        rec["lat"], rec["lon"], rec["depth"], rec["drho"], rec["dx"] = lat_e[m], lon_e[m], depth, d[m], dx[m]
        rec["shear"] = _shear(rec["drho"], rec["dx"], h, rec["lat"])
        out.append(rec)
    recs = np.concatenate(out)
    _report(f"FESOM IC (pairs with |d rho| >= {keep_min_drho})", recs, box, top)
    lat0, lat1, lon0, lon1 = box
    inbox = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
    print(f"box: {int(inbox.sum())} FESOM nodes, wet levels min/median/max "
          f"{nlev[inbox].min() if inbox.any() else 0}/{np.median(nlev[inbox]) if inbox.any() else 0}/{nlev[inbox].max() if inbox.any() else 0}")
    return recs


def probe_woa(path, box, top, keep_min_drho, void_fill):
    """Adjacent-cell |d rho| on the observed SOURCE grid after the per-level
    fill run_omip applies (``--woa-void-fill`` selects the harmonic void fill),
    over source OCEAN cells (observed at some depth).  A source-grid wall is a
    wall the model inherits; the converse does not hold (the model samples
    bilinearly at partial-cell depths), so the model-grid probe is the verdict.
    Also counts filled columns that the fill left statically unstable."""
    from legoesm.ocean.init_woa import (_fill_source_levels_nearest_valid, _reject_unstable_donors,
                                        load_woa18)
    T, S, lat, lon, depth = load_woa18(path, path)
    observed = np.isfinite(T) & np.isfinite(S)
    (T, S), n_fill, n_void = _fill_source_levels_nearest_valid((T, S), lat, lon, void_fill=void_fill)
    # Donor-relevant columns: observed at some depth, or an 8-neighbour of one
    # (a never-observed 1-deg row next to the ocean is still a bilinear donor
    # for the model cells beside it -- the Aegean's 40.5N row on ORCA12).
    # Continental interiors are excluded; a source depth with no donor
    # anywhere stays NaN and is skipped level by level below.
    ocean = observed.any(axis=-1)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            ocean = ocean | np.roll(np.roll(observed.any(axis=-1), dj, axis=0), di, axis=1)
    T, S, n_unstable = _reject_unstable_donors(T, S, ~observed & np.isfinite(T), depth)
    print(f"WOA source {path}: T {T.shape}, filled {n_fill}, void-filled {n_void} (void_fill={void_fill}), "
          f"{n_unstable} lighter-than-above donors replaced by the level above")
    # vertical stability of the filled columns: adjacent levels compared at the
    # LOWER level's pressure (compressibility removed); an inversion deeper
    # than 0.01 kg/m3 counts
    rho_col = np.stack([_rho(T[:, :, k], S[:, :, k], depth[k]) for k in range(T.shape[-1])], axis=-1)
    rho_upper_at_lower_p = np.stack([_rho(T[:, :, k], S[:, :, k], depth[k + 1]) for k in range(T.shape[-1] - 1)], axis=-1)
    inv = (rho_col[:, :, 1:] - rho_upper_at_lower_p < -0.01) & ocean[:, :, None]
    filled_col = ocean & ~observed.all(axis=-1)
    print(f"columns with a density inversion: {int(inv.any(axis=-1).sum())} of {int(ocean.sum())} ocean "
          f"({int((inv.any(axis=-1) & filled_col).sum())} of {int(filled_col.sum())} partly filled columns)")
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    dx_e = _gc_dist(lat2, lon2, lat2, np.roll(lon2, -1, axis=1))
    dx_n = _gc_dist(lat2[:-1], lon2[:-1], lat2[1:], lon2[1:])
    h = np.diff(np.concatenate([[0.0], np.asarray(depth, dtype=np.float64)]))
    out, out_filled = [], []
    for k in range(T.shape[-1]):
        rho = rho_col[:, :, k]
        wet = np.isfinite(rho) & ocean
        obs = observed[:, :, k]
        for a, b, w, dx, la, lo, anyfill in (
            (rho, np.roll(rho, -1, axis=1), wet & np.roll(wet, -1, axis=1), dx_e, lat2, lon2,
             ~obs | ~np.roll(obs, -1, axis=1)),
            (rho[:-1], rho[1:], wet[:-1] & wet[1:], dx_n, lat2[:-1], lon2[:-1], ~obs[:-1] | ~obs[1:]),
        ):
            d = np.abs(a - b)
            m = w & np.isfinite(d) & (d >= keep_min_drho)
            if not m.any():
                continue
            rec = np.empty(int(m.sum()), dtype=_DT)
            rec["lat"], rec["lon"], rec["depth"], rec["drho"], rec["dx"] = la[m], lo[m], depth[k], d[m], dx[m]
            rec["shear"] = _shear(rec["drho"], rec["dx"], max(h[k], 1.0), rec["lat"])
            out.append(rec)
            out_filled.append(rec[anyfill[m]])
    if not out:
        print("no wet adjacent pair above the threshold")
        return np.empty(0, dtype=_DT)
    recs = np.concatenate(out)
    _report(f"WOA source grid after fill (void_fill={void_fill}, pairs with |d rho| >= {keep_min_drho})",
            recs, box, top)
    filled = np.concatenate(out_filled)
    if filled.size:
        _report("... of which pairs with at least one FILLED (unobserved) cell -- the fill's own walls",
                filled, box, top)
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orca-snapshot")
    p.add_argument("--fesom-mesh")
    p.add_argument("--woa-source", help="observed T/S file in WOA layout (T and S in one file)")
    p.add_argument("--void-fill", action="store_true", help="with --woa-source: apply the harmonic void fill")
    p.add_argument("--box", type=float, nargs=4, default=(40.3, 41.2, 26.5, 29.9),
                   metavar=("LAT0", "LAT1", "LON0", "LON1"), help="Sea of Marmara by default")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-drho", type=float, default=0.05,
                   help="keep only pairs above this |d rho| [kg/m3] to bound memory (ranking unaffected above it)")
    a = p.parse_args()
    if a.orca_snapshot:
        probe_orca(a.orca_snapshot, a.box, a.top, a.min_drho)
    if a.fesom_mesh:
        probe_fesom(a.fesom_mesh, a.box, a.top, a.min_drho)
    if a.woa_source:
        probe_woa(a.woa_source, a.box, a.top, a.min_drho, a.void_fill)


if __name__ == "__main__":
    main()
