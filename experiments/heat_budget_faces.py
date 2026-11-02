"""Per-face advective heat budget of tropical boxes + Indonesian Throughflow estimate.
Usage: python heat_budget_faces.py <run_dir> [y0 y1]"""
import sys, glob, re, numpy as np
RUN = sys.argv[1].rstrip("/"); Y0, Y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2, 3)
zz = np.load("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_bias_woa/z_levels.npz")
zh, zc = zz["zh"], zz["zc"]; nz = len(zc)
day = lambda f: int(re.search(r"day(\d+)", f).group(1))
fs = sorted([f for f in glob.glob(RUN + "/restart_day*.npz") if 365*(Y0-1) < day(f) <= 365*Y1], key=day)
z0 = np.load(fs[0]); H = z0["H_bathy"]; oc = z0["land_mask"] > 0.5
R = 6.371e6; yr = 365 * 86400.0
lat = np.linspace(-89.5, 89.5, 180); lon = np.arange(360) + 0.5
LAT, LON = np.meshgrid(lat, lon, indexing="ij")
dy = R * np.deg2rad(1.0); dx = R * np.cos(np.deg2rad(LAT)) * np.deg2rad(1.0)
dxv = R * np.cos(np.deg2rad(np.linspace(-90, 90, 181)))[:, None] * np.deg2rad(1.0) * np.ones((1, 360))
dz = np.clip(np.minimum(zh[None, None, 1:], H[..., None]) - zh[None, None, :-1], 0, None) * oc[..., None]
vol = dx[..., None] * dy * dz
kk = (zc > 100) & (zc < 600)
boxes = {"trop Atlantic": (((LON >= 290) | (LON < 20)) & (np.abs(LAT) < 20)),
         "trop Indian":   ((LON >= 40) & (LON < 100) & (np.abs(LAT) < 20))}
for name, hb in boxes.items():
    B = (hb[..., None] & kk[None, None, :] & (dz > 0)).astype(float); V = np.sum(vol * B)
    faces = {k: 0.0 for k in ["west", "east", "south", "north", "top", "bottom"]}
    Tref = None
    for f in fs:
        z = np.load(f); T = z["T"] * oc[..., None]
        if Tref is None: Tref = np.sum(T * vol * B) / V
        Tp = (T - Tref) * oc[..., None]
        Bm1 = np.roll(B, 1, axis=1); dzf = np.minimum(dz, np.roll(dz, 1, axis=1))
        Fu = z["u"][:, :360, :] * 0.5 * (Tp + np.roll(Tp, 1, axis=1)) * dy * dzf
        faces["west"] += np.sum(Fu * ((B == 1) & (Bm1 == 0)));  faces["east"] -= np.sum(Fu * ((B == 0) & (Bm1 == 1)))
        Fv = z["v"][1:180] * 0.5 * (Tp[1:] + Tp[:-1]) * dxv[1:180, :, None] * np.minimum(dz[1:], dz[:-1])
        faces["south"] += np.sum(Fv * ((B[1:] == 1) & (B[:-1] == 0))); faces["north"] -= np.sum(Fv * ((B[1:] == 0) & (B[:-1] == 1)))
        Fw = z["w"][..., 1:nz] * 0.5 * (Tp[..., :-1] + Tp[..., 1:]) * (dx * dy)[..., None] * (dz[..., 1:] > 0)
        faces["top"] += np.sum(Fw * ((B[..., :-1] == 0) & (B[..., 1:] == 1)))     # box cell below interface, w<0 brings heat in
        faces["top"] *= 1.0
        faces["bottom"] += np.sum(Fw * ((B[..., :-1] == 1) & (B[..., 1:] == 0)))
    # sign fix for top: inflow from above = -w*T' (w>0 up)
    faces["top"] = -faces["top"]
    print(f"\n=== {name}, 100-600 m: advective heat INTO box by face [C/yr]")
    for k, val in faces.items(): print(f"  {k:7s} {val / len(fs) * yr / V:+.3f}")
    print(f"  total   {sum(faces.values()) / len(fs) * yr / V:+.3f}")
# Indonesian Throughflow = -(net northward volume transport across 32S, 20E-115E), full depth
j = int(np.argmin(np.abs(np.linspace(-90, 90, 181) - (-32.0))))
sel = (lon >= 20) & (lon <= 115)
itf = []; itf_T = []
for f in fs:
    z = np.load(f); vv = z["v"][j][sel]          # (nlon_sel, nz)
    dzs = np.minimum(dz[j - 1][sel], dz[j][sel]); area = dxv[j][sel][:, None] * dzs
    itf.append(-np.sum(vv * area) / 1e6)
    T = z["T"]; Tf = 0.5 * (T[j - 1][sel] + T[j][sel])
    itf_T.append(-np.sum(vv * area * Tf) / max(-np.sum(vv * area), 1e-9))
print(f"\nIndonesian Throughflow (from 32S closure): {np.mean(itf):+.1f} Sv  (obs ~10-20 Sv westward)")
print(f"  transport-weighted net temperature signal: {np.mean(itf_T):+.1f} C (indicative only)")
