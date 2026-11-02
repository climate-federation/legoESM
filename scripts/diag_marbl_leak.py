"""Where do MARBL tracer inventories change between two restarts? (conservation debugging)
Usage: python diag_marbl_leak.py RESTART_A RESTART_B GRID_NPZ"""
import sys
import numpy as np

a, b, g = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3])
zh = np.abs(np.asarray(g["z_half"], float))                       # 36 interfaces
lat = np.asarray(g["lat"], float); lat = np.rad2deg(lat) if np.abs(lat).max() < 1.6 else lat
R = 6.371e6; dlat = np.deg2rad(1.0)
area = (R ** 2 * dlat * dlat * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, 360))

def thick(r):
    H, eta, m = r["H_bathy"], r["eta"], r["land_mask"] > 0.5
    top, bot = zh[:-1][None, None, :], zh[1:][None, None, :]
    h0 = np.clip(np.minimum(bot, H[..., None]) - top, 0.0, None)          # partial cells
    return h0 * (1.0 + eta / np.maximum(H, 1.0))[..., None] * m[..., None], h0 > 0

ha, act = thick(a); hb, _ = thick(b)
tr = {"PO4": "biogeo_PO4", "SiO3": "biogeo_SiO3", "DIC": "biogeo_DIC", "Fe": "biogeo_Fe"}
H = a["H_bathy"]; nwet = act.sum(-1); kb = np.maximum(nwet - 1, 0)
bottom = np.zeros_like(act); np.put_along_axis(bottom, kb[..., None], True, axis=-1); bottom &= act
shallow = (H < 100.0)[..., None] & act
Va, Vb = (ha * area[..., None]).sum(), (hb * area[..., None]).sum()
print(f"ocean volume change A->B (from eta): {(Vb - Va) / Va:+.3e}   mean eta {a['eta'].mean():+.4f} -> {b['eta'].mean():+.4f} m")
print(f"T,S: salt inventory change {((b['S'] * hb - a['S'] * ha) * area[..., None]).sum() / (a['S'] * ha * area[..., None]).sum():+.3e}")
for name, key in tr.items():
    ca, cb = a[key], b[key]
    Ia = ca * ha * area[..., None]; Ib = cb * hb * area[..., None]; d = Ib - Ia; tot = Ia.sum()
    dconc = (cb - ca) * ha * area[..., None]                     # change from concentration only
    print(f"\n{name}: total {d.sum() / tot:+.3e}  (from concentration change {dconc.sum() / tot:+.3e}, "
          f"from thickness change {(d.sum() - dconc.sum()) / tot:+.3e})")
    print(f"   bottom cells {d[bottom].sum() / tot:+.3e}   other cells {d[act & ~bottom].sum() / tot:+.3e}   "
          f"shallow (H<100 m) {d[shallow].sum() / tot:+.3e}   land/inactive {d[~act].sum() / tot:+.3e}")
    lv = d.sum((0, 1)) / tot
    print("   by level: " + "  ".join(f"k{k}:{lv[k]:+.1e}" for k in range(0, 35, 3)))
    i, j, k = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    print(f"   largest cell: lat {lat[i]:.1f} lon {j + 0.5:.1f} k {k} H {H[i, j]:.0f} m  wet levels {nwet[i, j]}  "
          f"c {ca[i, j, k]:.4g} -> {cb[i, j, k]:.4g}")
