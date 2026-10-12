"""Where do the three OMIP grids disagree in the equatorial Pacific?

Window means (``--state-accumulate``) of the 2026-10-06 baseline runs
(tripole / MPAS / FESOM) against NEMO RUN_GATEWAY 5-day means, every field put
through ONE remap (IDW k=4 onto a 1-degree grid, ``compare_omip_nemo``'s
``regrid_curv_to_latlon``) and ONE common wet mask per level, so the lanes are
compared on identical footprints.

Outputs (JSON + PNG in ``--out``):
  1. nino3 box-mean top-level temperature, days 5..30, each lane and NEMO
     (record = day/5 - 1).  Shows WHEN the grid spread opens.
  2. nino3 box-mean temperature profile at day 30, 0-300 m.
  3. equatorial (|lat|<=1) zonal sections at day 30 of temperature and zonal
     velocity, 140E-280E, 0-300 m; location and size of the largest
     tripole-minus-MPAS difference, and its depth-band means over nino3.

Conventions stated, not converted: NEMO ``to`` is CONSERVATIVE temperature
(ln_teos10; see diaar5.F90:321-334); ours is potential temperature.  The offset
is common to all three lanes, so lane-minus-lane differences are unaffected;
lane-minus-NEMO carries it.  FESOM stores no window-mean velocity: its u is
the instantaneous day-30 snapshot and is labelled as such.  MPAS mean velocity
is the Perot reconstruction of the mean edge-normal velocity on the run's own
mesh (coordinates checked against the snapshot to 1e-3 deg, plus a
reconstruction check of the snapshot velocity against the stored u_east).
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
from compare_omip_nemo import regrid_curv_to_latlon  # noqa: E402

NINO3 = (-5.0, 5.0, 210.0, 270.0)
GLAT = np.arange(-9.5, 10.0, 1.0)
GLON = np.arange(130.5, 290.0, 1.0)
ZMAX = 300.0
DAYS = (5, 10, 15, 20, 25, 30)


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--nemo-prefix", required=True,
                   help=".../RUN_GATEWAY/ORCA1_5d_20000101_20000331_")
    p.add_argument("--trp", default="trp_base1006_d30")
    p.add_argument("--mpas", default="mpas_base1006_d30")
    p.add_argument("--fesom", default="fesom_base1006_d30")
    p.add_argument("--mpas-level", type=int, default=7)
    p.add_argument("--mpas-lloyd", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    import netCDF4 as nc

    T = nc.Dataset(a.nemo_prefix + "grid_T.nc"); U = nc.Dataset(a.nemo_prefix + "grid_U.nc")
    nlat, nlon = _f(T["nav_lat"][:]), _f(T["nav_lon"][:])
    zN = _f(T["deptht"][:]); kz = int(np.searchsorted(zN, ZMAX)) + 1; zN = zN[:kz]
    snap = lambda lane, d: np.load(a.results / lane / (
        "snapshot_final.npz" if d == 30 and not (a.results / lane / "snapshot_day0030.npz").exists()
        else f"snapshot_day{d:04d}.npz"))
    rep = {"provenance": {"argv": sys.argv, "nemo": a.nemo_prefix,
                          "lanes": [a.trp, a.mpas, a.fesom], "remap": "IDW k=4 1deg max 1.5deg",
                          "nemo_T_convention": "conservative (to)", "lane_T_convention": "potential"}}

    # ---------- 1. nino3 top-level T, days 5..30 ----------
    ts = {k: [] for k in ("NEMO", "tripole", "MPAS", "FESOM")}
    for d in DAYS:
        rec = d // 5 - 1
        toN = _f(T["to"][rec, 0]); ts["NEMO"].append(box_mean(remap(toN, nlat, nlon, np.isfinite(toN))))
        for lab, lane in (("tripole", a.trp), ("MPAS", a.mpas), ("FESOM", a.fesom)):
            try:
                s = snap(lane, d)
            except FileNotFoundError:
                ts[lab].append(float("nan")); continue
            t0 = s["T_mean"][..., 0]
            wet = s["land_mask"] > 0.5
            ts[lab].append(box_mean(remap(t0, s["lat_T"], s["lon_T"], wet)))
    rep["nino3_top_T_by_day"] = {"days": list(DAYS), **ts}

    # ---------- 2+3. day-30 3-D T and u ----------
    toN3 = np.moveaxis(_f(T["to"][5, :kz]), 0, -1)                  # (y,x,z)
    uoN = np.moveaxis(_f(U["uo"][5, :kz]), 0, -1)
    uN = 0.5 * (np.nan_to_num(uoN) + np.roll(np.nan_to_num(uoN), 1, axis=1))
    wetN = np.isfinite(toN3)
    G = {"NEMO": {"T": remap3(toN3, nlat, nlon, wetN), "u": remap3(uN, nlat, nlon, wetN)}}

    t = snap(a.trp, 30); zt = np.abs(t["z_center_ref"])[:kz]
    assert np.allclose(zt, zN, atol=0.5), ("tripole levels != NEMO levels", zt[:5], zN[:5])
    wt = wet3_from(t["H_bathy"], zt, t["T_mean"][..., :kz].shape) & (t["land_mask"] > 0.5)[..., None]
    um = t["u_mean"][..., :kz]; uT = 0.5 * (um[:, :-1] + um[:, 1:])
    G["tripole"] = {"T": remap3(t["T_mean"][..., :kz], t["lat_T"], t["lon_T"], wt),
                    "u": remap3(uT, t["lat_T"], t["lon_T"], wt)}

    m = np.load(a.results / a.mpas / "snapshot_final.npz")
    zm = np.abs(m["z_center_ref"])[:kz]
    assert np.allclose(zm, zN, atol=0.5), ("MPAS levels != NEMO levels", zm[:5])
    wm = wet3_from(m["H_bathy"], zm, m["T_mean"][..., :kz].shape) & (m["land_mask"] > 0.5)[..., None]
    import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    mesh = create_voronoi_mesh(a.mpas_level, lloyd_iterations=a.mpas_lloyd)
    err = float(np.max(np.abs(np.rad2deg(np.asarray(mesh.latCell)) - m["lat_T"])))
    rep["provenance"]["mpas_mesh_lat_err_deg"] = err
    # 1e-5 deg (~1 m) residual seen with a mesh rebuilt by a different code version; the
    # reconstruction check below (snapshot u -> stored u_east) is the real mesh test.
    assert err < 1e-3, f"mesh lloyd {a.mpas_lloyd} does not match the MPAS snapshot ({err} deg)"
    ue_s, _ = reconstruct_cell_velocity(jnp.asarray(m["u"][:, :kz]), mesh)
    rep["provenance"]["mpas_recon_check_max_abs"] = float(np.max(np.abs(np.asarray(ue_s) - m["u_east"][:, :kz])))
    ueM, _ = reconstruct_cell_velocity(jnp.asarray(m["u_mean"][:, :kz]), mesh)
    G["MPAS"] = {"T": remap3(m["T_mean"][..., :kz], m["lat_T"], m["lon_T"], wm),
                 "u": remap3(np.asarray(ueM), m["lat_T"], m["lon_T"], wm)}

    fz = snap(a.fesom, 30); zf = np.abs(fz["z_center_ref"])
    nf = zf.size                       # FESOM arrays may carry a pad slot past the last level
    wf = wet3_from(fz["H_bathy"], zf, fz["T_mean"][..., :nf].shape) & (fz["land_mask"] > 0.5)[..., None]
    Tf = remap3(fz["T_mean"][..., :nf], fz["lat_T"], fz["lon_T"], wf)
    uf = remap3(fz["u_east"][..., :nf], fz["lat_T"], fz["lon_T"], wf)
    # FESOM levels -> NEMO level centres (linear in depth, per column)
    interp = lambda F: np.stack([np.apply_along_axis(lambda c: np.interp(zN, zf, c, left=np.nan, right=np.nan), 0, F)])[0]
    G["FESOM (u instantaneous)"] = {"T": interp(Tf), "u": interp(uf)}

    # common wet mask per level: finite in every lane
    common = np.all([np.isfinite(g["T"]) for g in G.values()], axis=0)
    for g in G.values():
        for k in g:
            g[k] = np.where(common, g[k], np.nan)

    rep["nino3_profile_day30"] = {"z_m": zN.tolist(),
                                  **{lab: [box_mean(g["T"][k]) for k in range(kz)] for lab, g in G.items()}}
    rep["nino3_u_profile_day30"] = {"z_m": zN.tolist(),
                                    **{lab: [box_mean(g["u"][k]) for k in range(kz)] for lab, g in G.items()}}

    eq = np.abs(GLAT) <= 1.0
    sec = {lab: {k: np.nanmean(g[k][:, eq, :], axis=1) for k in ("T", "u")} for lab, g in G.items()}
    d_tm = sec["tripole"]["T"] - sec["MPAS"]["T"]
    d_um = sec["tripole"]["u"] - sec["MPAS"]["u"]
    def argmax_loc(d):
        i = np.nanargmax(np.abs(d)); zi, li = np.unravel_index(i, d.shape)
        return {"value": float(d[zi, li]), "z_m": float(zN[zi]), "lon_E": float(GLON[li])}
    bands = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 300)]
    nl = (GLON >= NINO3[2]) & (GLON <= NINO3[3])
    def band_means(d):
        return {f"{z0}-{z1}m": float(np.nanmean(d[(zN >= z0) & (zN < z1)][:, nl])) for z0, z1 in bands}
    rep["eq_section_trp_minus_mpas"] = {
        "T_argmax": argmax_loc(d_tm), "u_argmax": argmax_loc(d_um),
        "T_nino3_lon_band_means": band_means(d_tm), "u_nino3_lon_band_means": band_means(d_um)}
    for lab in ("tripole", "MPAS"):
        rep[f"eq_section_{lab}_minus_NEMO"] = {
            "T_nino3_lon_band_means": band_means(sec[lab]["T"] - sec["NEMO"]["T"]),
            "u_nino3_lon_band_means": band_means(sec[lab]["u"] - sec["NEMO"]["u"])}
    # EUC core per lane (max u below 30 m, 140-280E)
    rep["euc_core"] = {}
    for lab, s in sec.items():
        u = np.where(zN[:, None] > 30, s["u"], -9)
        i = np.nanargmax(u); zi, li = np.unravel_index(i, u.shape)
        rep["euc_core"][lab] = {"u": float(u[zi, li]), "z_m": float(zN[zi]), "lon_E": float(GLON[li])}

    (a.out / "pacbias_grids.json").write_text(json.dumps(rep, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 14), constrained_layout=True)
    for axi, (fld, lab) in zip(ax, ((sec["tripole"]["T"] - sec["NEMO"]["T"], "T tripole - NEMO [K]"),
                                    (sec["MPAS"]["T"] - sec["NEMO"]["T"], "T MPAS - NEMO [K]"),
                                    (d_tm, "T tripole - MPAS [K]"), (d_um, "u tripole - MPAS [m/s]"))):
        v = np.nanmax(np.abs(fld)); pc = axi.pcolormesh(GLON, zN, fld, cmap="RdBu_r", vmin=-v, vmax=v)
        axi.invert_yaxis(); axi.set_title(lab + "  (|lat|<=1, day 26-30 means)"); fig.colorbar(pc, ax=axi)
        axi.axvline(210, c="k", lw=0.5); axi.axvline(270, c="k", lw=0.5)
    fig.savefig(a.out / "eq_sections_grids.png", dpi=110)
    print(json.dumps({k: rep[k] for k in rep if k != "provenance"}, indent=1)[:6000])
    print("PNG", a.out / "eq_sections_grids.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
