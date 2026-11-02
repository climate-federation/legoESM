"""Offline heat budget of tropical thermocline boxes from restart snapshots.
Usage: python heat_budget_box.py <run_dir> [y0 y1]   (default years 2-3)"""
import sys, glob, re, numpy as np
RUN = sys.argv[1].rstrip("/"); Y0, Y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2, 3)
zz = np.load("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_bias_woa/z_levels.npz")
zh, zc = zz["zh"], zz["zc"]; nz = len(zc)
day = lambda f: int(re.search(r"day(\d+)", f).group(1))
fs = sorted([f for f in glob.glob(RUN + "/restart_day*.npz") if 365*(Y0-1) < day(f) <= 365*Y1], key=day)
print(f"{len(fs)} snapshots, days {day(fs[0])}-{day(fs[-1])}")
z0 = np.load(fs[0]); H = z0["H_bathy"]; oc = z0["land_mask"] > 0.5
print("shapes: T", z0["T"].shape, " u", z0["u"].shape, " v", z0["v"].shape, " w", z0["w"].shape)
R, rho, cp = 6.371e6, 1025.0, 3992.0
lat = np.linspace(-89.5, 89.5, 180); lon = np.arange(360) + 0.5
LAT, LON = np.meshgrid(lat, lon, indexing="ij")
dy = R * np.deg2rad(1.0); dx = R * np.cos(np.deg2rad(LAT)) * np.deg2rad(1.0)
dxv = R * np.cos(np.deg2rad(np.linspace(-90, 90, 181)))[:, None] * np.deg2rad(1.0) * np.ones((1, 360))
dz = np.clip(np.minimum(zh[None, None, 1:], H[..., None]) - zh[None, None, :-1], 0, None) * oc[..., None]  # partial cells
vol = dx[..., None] * dy * dz
kk = (zc > 100) & (zc < 600)
boxes = {"trop Atlantic": (((LON >= 290) | (LON < 20)) & (np.abs(LAT) < 20)),
         "trop Indian":   ((LON >= 40) & (LON < 100) & (np.abs(LAT) < 20))}
w0 = z0["w"]; w_iface = (w0.shape[-1] == nz + 1)
print("w on", "interfaces (nlev+1)" if w_iface else "levels (nlev): assumed at TOP interface of each cell")
for name, hb in boxes.items():
    B = (hb[..., None] & kk[None, None, :] & (dz > 0)).astype(float)
    V = np.sum(vol * B)
    def conv(z, tracer):
        T = tracer
        u, v, w = z["u"], z["v"], z["w"]
        # zonal faces: face i between cell i-1 and i (periodic)
        Tm1 = np.roll(T, 1, axis=1); Bm1 = np.roll(B, 1, axis=1); dzf = np.minimum(dz, np.roll(dz, 1, axis=1))
        Fu = u[:, :360, :] * 0.5 * (T + Tm1) * dy * dzf
        cu = np.sum(Fu * (B - Bm1))
        # meridional faces: face j between cell j-1 and j
        Fv = v[1:180] * 0.5 * (T[1:] + T[:-1]) * dxv[1:180, :, None] * np.minimum(dz[1:], dz[:-1])
        cv = np.sum(Fv * (B[1:] - B[:-1]))
        # vertical: interface k between cell k-1 (above) and k (below), w>0 up
        wk = w[..., 1:nz] if w_iface else w[..., 1:nz]
        Fw = wk * 0.5 * (T[..., :-1] + T[..., 1:]) * (dx * dy)[..., None] * (dz[..., 1:] > 0)
        cw = np.sum(Fw * (B[..., :-1] - B[..., 1:]))
        return cu, cv, cw
    acc = np.zeros(3); accv = np.zeros(3)
    Tref = None
    for f in fs:
        z = np.load(f)
        if Tref is None: Tref = np.sum(z["T"] * vol * B) / V
        acc += np.array(conv(z, (z["T"] - Tref) * oc[..., None]))
        accv += np.array(conv(z, np.ones_like(z["T"]) * oc[..., None]))
    acc /= len(fs); accv /= len(fs)
    yr = 365 * 86400.0
    Ta = np.sum(np.load(fs[0])["T"] * vol * B) / V; Tb = np.sum(np.load(fs[-1])["T"] * vol * B) / V
    obs = (Tb - Ta) / ((day(fs[-1]) - day(fs[0])) * 86400.0) * yr
    adv = acc * yr / V
    print(f"\n=== {name}, 100-600 m  (V = {V:.3e} m3)")
    print(f"  observed warming        {obs:+.3f} C/yr")
    print(f"  advection: zonal {adv[0]:+.3f}  merid {adv[1]:+.3f}  vertical {adv[2]:+.3f}  total {adv.sum():+.3f} C/yr")
    print(f"  residual (mixing+numerics) {obs - adv.sum():+.3f} C/yr")
    print(f"  volume-flux check: net {accv.sum()*yr/V:+.4f} /yr (should be ~0; large = w convention wrong)")
