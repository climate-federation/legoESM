"""Where do the three OMIP grids disagree in the equatorial Pacific, and through what?

Window means (``--state-accumulate``) of the tripole / MPAS / FESOM runs
against NEMO RUN_GATEWAY 5-day means, every field put through ONE remap (IDW
k=4 onto a 1-degree grid, ``compare_omip_nemo``'s ``regrid_curv_to_latlon``)
and ONE common wet mask per level, so the lanes are compared on identical
footprints.

Outputs (JSON + PNGs in ``--out``):
  1. nino3 box-mean temperature at the top level and near 50 m, days 5..30,
     each lane and NEMO (record = day/5 - 1): WHEN the spread opens, and
     whether the surface or the subsurface leads.
  2. nino3 box-mean temperature profile at day 30, 0-300 m.
  3. equatorial (|lat|<=1) zonal sections at day 30 of temperature and zonal
     velocity, 140E-280E; every lane minus the tripole and minus NEMO, as
     depth-band means over the nino3 longitudes.
  4. candidate carriers per source (nino3 box means unless named): Z20 and its
     east-west tilt, dT/dz near 50 m, mixed-layer depth, South Equatorial
     Current (0-20 m u, |lat|<=1), EUC core, and upwelling at ~50 m.

Upwelling.  Only NEMO (``wo``) and the tripole (``mass_flux_w_mean``) store a
vertical velocity, so every source also gets the SAME continuity proxy built
on the common 1-degree grid: w(50 m) = integral over 0-50 m of the horizontal
divergence of the remapped (u, v) (rigid lid).  The proxy is checked against
the stored w on NEMO and the tripole before it is read on MPAS and FESOM, and
the check is printed next to it.  FESOM stores no window-mean velocity, so
its velocities are the instantaneous day-30 state; to keep one like-for-like
comparison a second, INSTANTANEOUS set (tripole ``u``/``v``, MPAS
``u_east``/``v_north``, FESOM ``u_east``/``v_north``) is scored as well.

Conventions stated, not converted: NEMO ``to`` is CONSERVATIVE temperature
(ln_teos10; see diaar5.F90:321-334); ours is potential temperature.  The offset
is common to all three lanes, so lane-minus-lane differences are unaffected;
lane-minus-NEMO carries it.  Lane T defaults to the thickness-weighted window
mean (``T_mean_hw``), which is what NEMO's ``to`` is.  Tripole velocities are
grid-relative; ORCA1 is geographic in the tropics.  The tripole w includes the
GM bolus (advecting w); NEMO ``wo`` is Eulerian.  MPAS mean velocity is the
Perot reconstruction of the mean edge-normal velocity on the run's own mesh
(coordinates checked against the snapshot, plus a reconstruction check of the
snapshot velocity against the stored u_east).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/validate
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_omip_nemo import regrid_curv_to_latlon  # noqa: E402
from equatorial_thermocline import z20_from_column  # noqa: E402

NINO3 = (-5.0, 5.0, 210.0, 270.0)
WEST = (-5.0, 5.0, 160.0, 200.0)
CONTROL = {"off_eq_N_5_10": (5.0, 10.0, 210.0, 270.0), "off_eq_S_10_5": (-10.0, -5.0, 210.0, 270.0)}
GLAT = np.arange(-9.5, 10.0, 1.0)
GLON = np.arange(130.5, 290.0, 1.0)
ZMAX = 300.0
Z_UPW = 50.0
DAYS = (5, 10, 15, 20, 25, 30)
BANDS = ((0, 20), (20, 50), (50, 100), (100, 200), (200, 300))


def _f(x):
    a = np.ma.filled(np.ma.asarray(x).astype(np.float64), np.nan)
    a[np.abs(a) > 1e10] = np.nan
    return a


def remap(field, lat, lon, wet):
    """field (npts,) or (ny,nx); wet same shape -> (GLAT, GLON) with NaN off-ocean."""
    lat = np.asarray(lat).ravel(); lon = np.asarray(lon).ravel() % 360.0
    sel = (np.abs(lat) < 15.0) & (lon > 120.0) & (lon < 300.0)
    w = np.asarray(wet).ravel()[sel] & np.isfinite(np.asarray(field).ravel()[sel])
    if not w.any():
        return np.full((GLAT.size, GLON.size), np.nan)
    out, oc = regrid_curv_to_latlon(np.nan_to_num(np.asarray(field).ravel()[sel]), lat[sel], lon[sel],
                                    w.astype(float), GLAT, GLON, max_deg=1.5)
    out[oc < 0.5] = np.nan
    return out


def remap3(f3, lat, lon, wet3):
    """f3 (..., nz) with wet3 same shape -> (nz, GLAT, GLON)."""
    nz = f3.shape[-1]
    return np.stack([remap(f3[..., k], lat, lon, wet3[..., k]) for k in range(nz)])


def box_mean(g2, box=NINO3):
    la0, la1, lo0, lo1 = box
    LA, LO = np.meshgrid(GLAT, GLON, indexing="ij")
    m = (LA >= la0) & (LA <= la1) & (LO >= lo0) & (LO <= lo1) & np.isfinite(g2)
    w = np.cos(np.deg2rad(LA))[m]
    return float(np.sum(g2[m] * w) / np.sum(w)) if m.any() else float("nan")


def wet3_from(H, z, shape):
    """Wet where the level centre is above the bathymetry."""
    return (np.asarray(H)[..., None] > np.abs(z)) & np.ones(shape, bool)


def to_levels(F, zf, zN):
    """(nzf, ...) on depths ``zf`` -> (len(zN), ...) on ``zN``, linear in depth.
    Above the first source centre the source's top-layer value is held (the
    top layer is well mixed and FESOM's first centre sits ~6 mm below NEMO's);
    below the last source centre -> NaN."""
    return np.apply_along_axis(
        lambda c: np.interp(zN, zf, c, left=c[0], right=np.nan), 0, F)


def band_weights(zi, z0, z1):
    """Thickness [m] of each layer (interfaces ``zi``, len nz+1) inside [z0, z1)."""
    zi = np.asarray(zi, dtype=np.float64)
    return np.clip(np.minimum(zi[1:], z1) - np.maximum(zi[:-1], z0), 0.0, None)


def horizontal_divergence(u, v, radius):
    """(nz, GLAT, GLON) geographic u, v [m/s] on the 1-degree grid -> div_h [1/s],
    centred differences, NaN where a neighbour is missing."""
    phi = np.deg2rad(GLAT)[None, :, None]
    dl = np.deg2rad(GLON[1] - GLON[0]); dp = np.deg2rad(GLAT[1] - GLAT[0])
    du = np.full_like(u, np.nan); dv = np.full_like(v, np.nan)
    du[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dl)
    vc = v * np.cos(phi)
    dv[:, 1:-1, :] = (vc[:, 2:, :] - vc[:, :-2, :]) / (2.0 * dp)
    return (du + dv) / (radius * np.cos(phi))


def w_from_divergence(div, zi, z_w=Z_UPW):
    """Upward w [m/s] at depth ``z_w`` from continuity with a rigid lid:
    dw/dz' = -div_h with z' up, w(0) = 0  =>  w(-z_w) = integral_{-z_w}^0 div_h dz.
    ``div`` is (nz, ...); ``zi`` the nz+1 layer interfaces [m, >0, zi[0] = 0].
    NaN where any layer above ``z_w`` is missing.  Rigid lid: the 5-day-mean
    free-surface tendency (~cm per window) is <1% of a ~1 m/day equatorial w."""
    wgt = band_weights(np.asarray(zi)[: div.shape[0] + 1], 0.0, z_w)
    wgt = wgt.reshape((-1,) + (1,) * (div.ndim - 1))
    used = (wgt > 0).ravel()
    return np.sum(np.where(wgt > 0, div, 0.0) * wgt, axis=0) \
        + np.where(np.all(np.isfinite(div[used]), axis=0), 0.0, np.nan)


def z20_map(T, z):
    """(nz, GLAT, GLON) -> (GLAT, GLON) depth of the 20 C isotherm."""
    out = np.full(T.shape[1:], np.nan)
    for j in range(T.shape[1]):
        for i in range(T.shape[2]):
            if np.isfinite(T[0, j, i]):
                out[j, i] = z20_from_column(T[:, j, i], z)
    return out


def carriers(g, zN, zi, w_maps, mld=None):
    """Box-mean candidate carriers for one source on the common grid.  ``w_maps``
    {name: (GLAT, GLON) upward w at Z_UPW [m/s]}, already on a shared mask."""
    eq = np.abs(GLAT) <= 1.0
    nl = (GLON >= NINO3[2]) & (GLON <= NINO3[3])
    z20 = z20_map(g["T"], zN)
    k1 = int(np.argmin(np.abs(zN - Z_UPW))); k0 = k1 - 1
    dtdz = (g["T"][k0] - g["T"][k1 + 1]) / (zN[k1 + 1] - zN[k0])       # K/m, >0 stratified
    bw = band_weights(zi, 0.0, 20.0)[: zN.size]
    ul = np.nanmean(g["u"][:, eq][:, :, nl], axis=(1, 2)); ok = np.isfinite(ul) & (bw > 0)
    sec = float(np.sum(ul[ok] * bw[ok]) / np.sum(bw[ok])) if ok.any() else float("nan")
    w_px = w_maps["proxy"]
    dTdx_s = np.full(GLON.size, np.nan)
    Ts = np.nanmean(g["T"][0][eq], axis=0)
    dTdx_s[1:-1] = (Ts[2:] - Ts[:-2]) / (2.0 * np.deg2rad(1.0) * _radius())
    out = {"z20_nino3_m": box_mean(z20), "z20_west_m": box_mean(z20, WEST),
           "z20_nan_frac_nino3": float(np.mean(~np.isfinite(z20[np.ix_(np.abs(GLAT) <= 5, nl)]))),
           "dTdz_50m_K_per_m": box_mean(dtdz),
           "u_sec_0_20m_eq_nino3": float(sec),
           **{f"w50_{k}_m_per_day": box_mean(wm) * 86400.0 for k, wm in w_maps.items()},
           **{f"w50_{k}_eq_m_per_day": box_mean(wm, (-1.0, 1.0, NINO3[2], NINO3[3])) * 86400.0
              for k, wm in w_maps.items()},
           # PLAUSIBLE rough advective cooling rates [K/day], nino3: w*dT/dz and u*dT/dx (eq)
           "vert_adv_rate_K_per_day": -box_mean(w_px) * box_mean(dtdz) * 86400.0,
           "zonal_adv_rate_eq_K_per_day": -float(sec) * float(np.nanmean(dTdx_s[nl])) * 86400.0}
    out["z20_tilt_m"] = out["z20_west_m"] - out["z20_nino3_m"]
    if "model" in w_maps:
        out["vert_adv_rate_model_w_K_per_day"] = -box_mean(w_maps["model"]) * box_mean(dtdz) * 86400.0
    if mld is not None:
        out["mld_m"] = box_mean(mld)
    return out, {"z20": z20, "w": w_maps.get("model", w_px),
                 "w_kind": "model" if "model" in w_maps else "proxy"}


def _radius():
    from legoesm import constants
    return float(constants.R_earth)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--nemo-prefix", required=True,
                   help=".../RUN_GATEWAY/ORCA1_5d_20000101_20000331_")
    p.add_argument("--trp", default="trp_base1013_d30")
    p.add_argument("--mpas", default="mpas_base1014_d30")
    p.add_argument("--fesom", default="fesom_base1014_d30")
    p.add_argument("--t-field", default="T_mean_hw", choices=("T_mean_hw", "T_mean"),
                   help="lane temperature: thickness-weighted window mean (matches NEMO 'to') or plain")
    p.add_argument("--mpas-level", type=int, default=7)
    p.add_argument("--mpas-lloyd", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    import netCDF4 as nc
    TF = a.t_field

    T = nc.Dataset(a.nemo_prefix + "grid_T.nc"); U = nc.Dataset(a.nemo_prefix + "grid_U.nc")
    V = nc.Dataset(a.nemo_prefix + "grid_V.nc"); W = nc.Dataset(a.nemo_prefix + "grid_W.nc")
    nlat, nlon = _f(T["nav_lat"][:]), _f(T["nav_lon"][:])
    zN = _f(T["deptht"][:]); kz = int(np.searchsorted(zN, ZMAX)) + 1; zN = zN[:kz]
    k50 = int(np.argmin(np.abs(zN - Z_UPW)))
    snap = lambda lane, d: np.load(a.results / lane / (
        "snapshot_final.npz" if d == 30 and not (a.results / lane / "snapshot_day0030.npz").exists()
        else f"snapshot_day{d:04d}.npz"))
    rep = {"provenance": {"argv": sys.argv, "nemo": a.nemo_prefix, "t_field": TF,
                          "lanes": [a.trp, a.mpas, a.fesom], "remap": "IDW k=4 1deg max 1.5deg",
                          "nemo_T_convention": "conservative (to)", "lane_T_convention": "potential"}}

    # ---------- 1. nino3 top-level and ~50 m T, days 5..30 ----------
    labs = ("tripole", "MPAS", "FESOM")
    ts = {k: [] for k in ("NEMO",) + labs}; ts50 = {k: [] for k in ts}; z50 = {}
    for d in DAYS:
        rec = d // 5 - 1
        top, t50 = {}, {}
        toN = _f(T["to"][rec, 0]); top["NEMO"] = remap(toN, nlat, nlon, np.isfinite(toN))
        toN = _f(T["to"][rec, k50]); t50["NEMO"] = remap(toN, nlat, nlon, np.isfinite(toN))
        z50["NEMO"] = float(zN[k50])
        for lab, lane in zip(labs, (a.trp, a.mpas, a.fesom)):
            try:
                s = snap(lane, d)
            except FileNotFoundError:
                continue
            zl = np.abs(s["z_center_ref"]); kl = int(np.argmin(np.abs(zl - Z_UPW))); z50[lab] = float(zl[kl])
            wet = s["land_mask"] > 0.5
            top[lab] = remap(s[TF][..., 0], s["lat_T"], s["lon_T"], wet)
            t50[lab] = remap(s[TF][..., kl], s["lat_T"], s["lon_T"],
                             wet & (np.asarray(s["H_bathy"]) > zl[kl]))
        # one footprint per day: finite in every source present that day
        for src, dst in ((top, ts), (t50, ts50)):
            cm = np.all([np.isfinite(x) for x in src.values()], axis=0)
            for k in dst:
                dst[k].append(box_mean(np.where(cm, src[k], np.nan)) if k in src else float("nan"))
    rep["nino3_top_T_by_day"] = {"days": list(DAYS), **ts}
    rep["nino3_T50_by_day"] = {"days": list(DAYS), "level_depth_m": z50, **ts50}

    # ---------- 2+3+4. day-30 3-D T, u, v (+ instantaneous set) ----------
    toN3 = np.moveaxis(_f(T["to"][5, :kz]), 0, -1)                  # (y,x,z)
    uoN = np.nan_to_num(np.moveaxis(_f(U["uo"][5, :kz]), 0, -1))
    voN = np.nan_to_num(np.moveaxis(_f(V["vo"][5, :kz]), 0, -1))
    uN = 0.5 * (uoN + np.roll(uoN, 1, axis=1)); vN = 0.5 * (voN + np.roll(voN, 1, axis=0))
    wetN = np.isfinite(toN3)
    zW = _f(W["depthw"][:]); kw = int(np.argmin(np.abs(zW - Z_UPW)))
    zi = zW[: kz + 1]                          # NEMO W levels = layer interfaces, zi[0] = 0
    assert abs(zi[0]) < 1e-6 and np.all(np.diff(zi) > 0), zi[:3]
    woN = _f(W["wo"][5, kw])
    G = {"NEMO": {"T": remap3(toN3, nlat, nlon, wetN), "u": remap3(uN, nlat, nlon, wetN),
                  "v": remap3(vN, nlat, nlon, wetN)}}
    mldN = _f(T["mldr10_1"][5]); MLD = {"NEMO": remap(mldN, nlat, nlon, np.isfinite(mldN))}
    WD = {"NEMO": remap(woN, nlat, nlon, np.isfinite(woN))}
    rep["provenance"]["w_direct_depth_m"] = {"NEMO": float(zW[kw])}
    GI = {}

    t = snap(a.trp, 30); zt = np.abs(t["z_center_ref"])[:kz]
    assert np.allclose(zt, zN, atol=0.5), ("tripole levels != NEMO levels", zt[:5], zN[:5])
    wt = wet3_from(t["H_bathy"], zt, t[TF][..., :kz].shape) & (t["land_mask"] > 0.5)[..., None]
    c2t = lambda um, vm: (0.5 * (um[:, :-1] + um[:, 1:]), 0.5 * (vm[:-1] + vm[1:]))
    uT, vT = c2t(t["u_mean"][..., :kz], t["v_mean"][..., :kz])
    G["tripole"] = {k: remap3(f, t["lat_T"], t["lon_T"], wt) for k, f in
                    (("T", t[TF][..., :kz]), ("u", uT), ("v", vT))}
    uI, vI = c2t(t["u"][..., :kz], t["v"][..., :kz])
    GI["tripole"] = {"u": remap3(uI, t["lat_T"], t["lon_T"], wt), "v": remap3(vI, t["lat_T"], t["lon_T"], wt)}
    # z_interface_ref holds the INTERIOR interfaces only; mass_flux_w_mean carries
    # surface + interior + bottom (nlev+1), so interior interface k is w index k+1.
    zit = np.abs(t["z_interface_ref"]); kwt = int(np.argmin(np.abs(zit - Z_UPW)))
    assert t["mass_flux_w_mean"].shape[-1] == zit.size + 2, (t["mass_flux_w_mean"].shape, zit.size)
    rep["provenance"]["w_direct_depth_m"]["tripole"] = float(zit[kwt])
    WD["tripole"] = remap(t["mass_flux_w_mean"][..., kwt + 1], t["lat_T"], t["lon_T"], t["land_mask"] > 0.5)
    MLD["tripole"] = remap(t["mld_mean"], t["lat_T"], t["lon_T"], t["land_mask"] > 0.5)

    m = np.load(a.results / a.mpas / "snapshot_final.npz")
    zm = np.abs(m["z_center_ref"])[:kz]
    assert np.allclose(zm, zN, atol=0.5), ("MPAS levels != NEMO levels", zm[:5])
    wm = wet3_from(m["H_bathy"], zm, m[TF][..., :kz].shape) & (m["land_mask"] > 0.5)[..., None]
    import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    mesh = create_voronoi_mesh(a.mpas_level, lloyd_iterations=a.mpas_lloyd)
    err = float(np.max(np.abs(np.rad2deg(np.asarray(mesh.latCell)) - m["lat_T"])))
    rep["provenance"]["mpas_mesh_lat_err_deg"] = err
    # 1e-5 deg (~1 m) residual seen with a mesh rebuilt by a different code version; the
    # reconstruction check below (snapshot u -> stored u_east) is the real mesh test.
    assert err < 1e-3, f"mesh lloyd {a.mpas_lloyd} does not match the MPAS snapshot ({err} deg)"
    ue_s, vn_s = reconstruct_cell_velocity(jnp.asarray(m["u"][:, :kz]), mesh)
    rep["provenance"]["mpas_recon_check_max_abs"] = float(max(
        np.max(np.abs(np.asarray(ue_s) - m["u_east"][:, :kz])),
        np.max(np.abs(np.asarray(vn_s) - m["v_north"][:, :kz]))))
    ueM, vnM = reconstruct_cell_velocity(jnp.asarray(m["u_mean"][:, :kz]), mesh)
    G["MPAS"] = {k: remap3(np.asarray(f), m["lat_T"], m["lon_T"], wm) for k, f in
                 (("T", m[TF][..., :kz]), ("u", ueM), ("v", vnM))}
    GI["MPAS"] = {k: remap3(m[f][:, :kz], m["lat_T"], m["lon_T"], wm) for k, f in
                  (("u", "u_east"), ("v", "v_north"))}
    MLD["MPAS"] = remap(m["mld_mean"], m["lat_T"], m["lon_T"], m["land_mask"] > 0.5)
    # native continuity on the Voronoi mesh (TRiSK divergence), then remap the scalar w
    from legoesm.core.operators_voronoi import divergence_cell_3d
    r_mesh = float(np.sqrt(np.sum(np.asarray(mesh.areaCell)) / (4.0 * np.pi)))
    div_m = np.asarray(divergence_cell_3d(jnp.asarray(m["u_mean"][:, :kz]), mesh)) * r_mesh / _radius()
    div_m = np.where(wm, div_m, np.nan)
    # same target depth as the stored w (NEMO's W level nearest Z_UPW)
    WD["MPAS"] = remap(w_from_divergence(div_m.T, zi, float(zW[kw])), m["lat_T"], m["lon_T"],
                       m["land_mask"] > 0.5)
    rep["provenance"]["mpas_mesh_radius"] = r_mesh

    fz = snap(a.fesom, 30); zf = np.abs(fz["z_center_ref"])
    nf = zf.size                       # FESOM arrays may carry a pad slot past the last level
    wf = wet3_from(fz["H_bathy"], zf, fz[TF][..., :nf].shape) & (fz["land_mask"] > 0.5)[..., None]
    # FESOM levels -> NEMO level centres (linear in depth, per column)
    rf = lambda f: to_levels(remap3(f[..., :nf], fz["lat_T"], fz["lon_T"], wf), zf, zN)
    G["FESOM"] = {"T": rf(fz[TF]), "u": rf(fz["u_east"]), "v": rf(fz["v_north"])}
    GI["FESOM"] = {"u": G["FESOM"]["u"], "v": G["FESOM"]["v"]}
    MLD["FESOM"] = remap(fz["mld_mean"], fz["lat_T"], fz["lon_T"], fz["land_mask"] > 0.5)
    rep["provenance"]["fesom_velocity"] = "instantaneous day-30 (no window mean stored)"

    # common wet mask per level: finite in every source
    common = np.all([np.isfinite(g["T"]) for g in G.values()], axis=0)
    for g in list(G.values()) + list(GI.values()):
        for k in g:
            g[k] = np.where(common, g[k], np.nan)
    c2 = np.all([np.isfinite(x) for x in MLD.values()], axis=0) & common[0]
    MLD = {k: np.where(c2, x, np.nan) for k, x in MLD.items()}

    rep["nino3_profile_day30"] = {"z_m": zN.tolist(),
                                  **{lab: [box_mean(g["T"][k]) for k in range(kz)] for lab, g in G.items()}}
    rep["nino3_u_profile_day30"] = {"z_m": zN.tolist(),
                                    **{lab: [box_mean(g["u"][k]) for k in range(kz)] for lab, g in G.items()}}

    eq = np.abs(GLAT) <= 1.0
    nl = (GLON >= NINO3[2]) & (GLON <= NINO3[3])
    sec = {lab: {k: np.nanmean(g[k][:, eq, :], axis=1) for k in ("T", "u")} for lab, g in G.items()}

    def _wavg(vals, z0, z1):
        bw = band_weights(zi, z0, z1)[: kz]
        vals = np.asarray(vals, dtype=np.float64); ok = np.isfinite(vals) & (bw > 0)
        return float(np.sum(vals[ok] * bw[ok]) / np.sum(bw[ok])) if ok.any() else float("nan")

    def band_means(d):            # equatorial section, nino3 longitudes, thickness-weighted
        return {f"{z0}-{z1}m": _wavg(np.nanmean(d[:, nl], axis=1), z0, z1) for z0, z1 in BANDS}

    def box_bands(d3, box=NINO3):
        prof = [box_mean(d3[k], box) for k in range(kz)]
        return {f"{z0}-{z1}m": _wavg(prof, z0, z1) for z0, z1 in BANDS}
    for lab in ("MPAS", "FESOM"):
        rep[f"{lab}_minus_tripole"] = {
            "eq_T_bands": band_means(sec[lab]["T"] - sec["tripole"]["T"]),
            "eq_u_bands": band_means(sec[lab]["u"] - sec["tripole"]["u"]),
            "nino3_box_T_bands": box_bands(G[lab]["T"] - G["tripole"]["T"]),
            # off-equator controls: a shared column-mixing difference would show here too
            **{f"{cn}_T_bands": box_bands(G[lab]["T"] - G["tripole"]["T"], cb) for cn, cb in CONTROL.items()}}
    for lab in labs:
        rep[f"{lab}_minus_NEMO"] = {
            "eq_T_bands": band_means(sec[lab]["T"] - sec["NEMO"]["T"]),
            "eq_u_bands": band_means(sec[lab]["u"] - sec["NEMO"]["u"]),
            "nino3_box_T_bands": box_bands(G[lab]["T"] - G["NEMO"]["T"])}
    rep["euc_core"] = {}
    for lab, s in sec.items():
        u = np.where(zN[:, None] > 30, s["u"], -9)
        i = np.nanargmax(u); zk, li = np.unravel_index(i, u.shape)
        rep["euc_core"][lab] = {"u": float(u[zk, li]), "z_m": float(zN[zk]), "lon_E": float(GLON[li])}

    # upwelling at Z_UPW: direct (NEMO wo, tripole mass_flux_w_mean), native continuity
    # (MPAS TRiSK), and the 1-degree proxy for every source and velocity set -- all
    # on ONE footprint (finite in every map) so differences are not footprint.
    R = _radius()
    PX = {lab: w_from_divergence(horizontal_divergence(g["u"], g["v"], R), zi, float(zW[kw])) for lab, g in G.items()}
    PXI = {lab: w_from_divergence(horizontal_divergence(g["u"], g["v"], R), zi, float(zW[kw])) for lab, g in GI.items()}
    cw = np.all([np.isfinite(x) for x in list(PX.values()) + list(PXI.values()) + list(WD.values())], axis=0)
    msk = lambda x: np.where(cw, x, np.nan)
    rep["provenance"]["w_kinds"] = {lab: ("direct" if lab in ("NEMO", "tripole") else "native")
                                    for lab in WD}

    rep["carriers_mean_velocity"] = {}; maps = {}
    for lab, g in G.items():
        wm = {"proxy": msk(PX[lab])}
        if lab in WD:
            wm["model"] = msk(WD[lab])
        rep["carriers_mean_velocity"][lab], maps[lab] = carriers(g, zN, zi, wm, MLD[lab])
    rep["carriers_instantaneous_velocity"] = {}
    for lab, g in GI.items():
        gi = {"T": G[lab]["T"], **g}
        rep["carriers_instantaneous_velocity"][lab] = {
            k: v for k, v in carriers(gi, zN, zi, {"proxy": msk(PXI[lab])})[0].items()
            if k.startswith(("w50", "u_sec", "vert_adv", "zonal_adv"))}

    (a.out / "pacbias_grids.json").write_text(json.dumps(rep, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 14), constrained_layout=True)
    for axi, (fld, lab) in zip(ax, ((sec["tripole"]["T"] - sec["NEMO"]["T"], "T tripole - NEMO [K]"),
                                    (sec["MPAS"]["T"] - sec["tripole"]["T"], "T MPAS - tripole [K]"),
                                    (sec["FESOM"]["T"] - sec["tripole"]["T"], "T FESOM - tripole [K]"),
                                    (sec["MPAS"]["u"] - sec["tripole"]["u"], "u MPAS - tripole [m/s]"))):
        v = np.nanmax(np.abs(fld)); pc = axi.pcolormesh(GLON, zN, fld, cmap="RdBu_r", vmin=-v, vmax=v)
        axi.invert_yaxis(); axi.set_title(lab + "  (|lat|<=1, day 26-30 means)"); fig.colorbar(pc, ax=axi)
        axi.axvline(210, c="k", lw=0.5); axi.axvline(270, c="k", lw=0.5)
    fig.savefig(a.out / "eq_sections_grids.png", dpi=110)

    fig, ax = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    pairs = (("MPAS", "SST"), ("FESOM", "SST"), ("MPAS", "z20"), ("FESOM", "z20"), ("MPAS", "w"), ("FESOM", "w"))
    for axi, (lab, fld) in zip(ax.ravel(), pairs):
        if fld == "SST":
            d = G[lab]["T"][0] - G["tripole"]["T"][0]; unit = "K"
        elif fld == "z20":
            d = maps[lab]["z20"] - maps["tripole"]["z20"]; unit = "m"
        else:
            if maps[lab]["w_kind"] == "model":
                d = (maps[lab]["w"] - maps["tripole"]["w"]) * 86400.0
                unit = "m/day at 50 m, native - tripole direct"
            else:      # FESOM: no window-mean velocity -> instantaneous proxy on both sides
                d = msk(PXI[lab] - PXI["tripole"]) * 86400.0
                unit = "m/day at 50 m, instantaneous proxy both"
        v = np.nanmax(np.abs(d)) or 1.0
        pc = axi.pcolormesh(GLON, GLAT, d, cmap="RdBu_r", vmin=-v, vmax=v)
        axi.add_patch(plt.Rectangle((210, -5), 60, 10, fill=False, ec="k", lw=0.8))
        axi.set_title(f"{fld} {lab} - tripole [{unit}], day 26-30"); fig.colorbar(pc, ax=axi)
    fig.savefig(a.out / "nino3_xgrid_maps.png", dpi=110)
    print(json.dumps({k: rep[k] for k in rep if k != "provenance"}, indent=1)[:12000])
    print("PNG", a.out / "eq_sections_grids.png", a.out / "nino3_xgrid_maps.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
