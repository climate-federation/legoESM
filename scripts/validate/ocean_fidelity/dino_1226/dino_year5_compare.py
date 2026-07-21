"""Controlled comparison: legoESM nemo_dino_kamm year-5 mean vs NEMO DINO year-5 mean.

Same IC, grid, forcing, dt, window — the only difference is the two codes. Reports
SST/T-section/SSH/BSF match; writes a multi-panel PNG. BSF = zonal integral of the
depth-integrated meridional transport (ψ=0 at west wall), NEMO metric for BOTH.
"""
import sys, numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LEGO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-10257/-home-dbalwada-legoESM/ef25578f-fbe6-4472-b331-67177c73fc11/scratchpad/y5_mlf.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-10257/-home-dbalwada-legoESM/ef25578f-fbe6-4472-b331-67177c73fc11/scratchpad/kamm_compare_5y.png"
R = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_5Y"
P = f"{R}/DINO_1y_00050101_00051230"; H = 2  # nn_hls
iy, ix = slice(H, -H), slice(H, -H)   # interior halo strip -> (195,48)

# --- NEMO year-5 mean (interior, to (lat,lon,lev)) ---
gT = nc.Dataset(f"{P}_grid_T.nc")
mm = nc.Dataset("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc")
def llz(a): return np.moveaxis(np.asarray(a).squeeze(), 0, -1)  # (lev,y,x)->(y,x,lev)
# native-iom artifact: votemper/vosaline are toce*e3t UNDIVIDED (no XIOS /e3t op) ->
# recover physical T,S by dividing by the co-written thickness field vovvle3t.
_e3d = llz(gT["vovvle3t"][0])[iy, ix]
nT = (llz(gT["votemper"][0])[iy, ix]) / np.maximum(_e3d, 1e-6)   # (195,48,36) degC
nS_div = (llz(gT["vosaline"][0])[iy, ix]) / np.maximum(_e3d, 1e-6)
nSSH = np.asarray(gT["sossheig"][0])[iy, ix]
nV = llz(nc.Dataset(f"{P}_grid_V.nc")["vomecrty"][0])[iy, ix]
tmask = llz(mm["tmask"][0])[iy, ix] > 0.5
e3t = llz(mm["e3t_0"][0])[iy, ix]          # cell thickness (full-step)
e1v = np.asarray(mm["e1v"][0]).squeeze()[iy, ix]
gphit = np.asarray(mm["gphit"][0]).squeeze()[iy, ix]
gdept = np.asarray(mm["gdept_1d"][:]).squeeze()

# --- legoESM year-5 mean ---
lo = np.load(LEGO)
lT, lSSH = lo["T"], lo["eta"]
lV = 0.5 * (lo["v"][:-1] + lo["v"][1:])    # v-faces (196,48,36) -> cell-centre (195,48,36)
wet2d = (lo["land_mask"] > 0.5) & tmask[:, :, 0]
wet3d = tmask & (lo["land_mask"][:, :, None] > 0.5)

def stat(a, b, m):
    a, b = a[m], b[m]
    return dict(bias=float(np.mean(a - b)), rms=float(np.sqrt(np.mean((a - b) ** 2))),
                corr=float(np.corrcoef(a, b)[0, 1]), lo=(float(a.min()), float(a.max())),
                ne=(float(b.min()), float(b.max())))

def bsf(V):  # depth-integ meridional transport, zonally integrated from west wall -> Sv
    Vbt = np.sum(np.where(wet3d, V * e3t, 0.0), axis=2)      # (195,48) m^2/s
    return np.cumsum(Vbt * e1v, axis=1) / 1e6               # (195,48) Sv

sst = stat(lT[:, :, 0], nT[:, :, 0], wet2d)
ssh = stat(lSSH - lSSH[wet2d].mean(), nSSH - nSSH[wet2d].mean(), wet2d)  # gauge-free
kz = np.argmin(np.abs(gdept - 300.0))                      # ~300 m thermocline level
tmid = stat(lT[:, :, kz], nT[:, :, kz], tmask[:, :, kz] & wet2d)
lBSF, nBSF = bsf(lV), bsf(nV)
print(f"SST   corr={sst['corr']:.3f} bias={sst['bias']:+.2f} rms={sst['rms']:.2f}  lego{sst['lo']} nemo{sst['ne']}")
print(f"T300m corr={tmid['corr']:.3f} bias={tmid['bias']:+.2f} rms={tmid['rms']:.2f}")
print(f"SSH   corr={ssh['corr']:.3f} rms={ssh['rms']:.3f} m")
print(f"BSF   lego[{lBSF[wet2d].min():+.1f},{lBSF[wet2d].max():+.1f}] nemo[{nBSF[wet2d].min():+.1f},{nBSF[wet2d].max():+.1f}] Sv"
      f"  range_ratio={ (lBSF[wet2d].max()-lBSF[wet2d].min()) / (nBSF[wet2d].max()-nBSF[wet2d].min()):.2f}x")

# --- plot ---
fig, ax = plt.subplots(3, 3, figsize=(15, 12))
def show(a, x, ti, cmap, vmin=None, vmax=None, mask=wet2d):
    d = np.where(mask, a, np.nan)
    im = ax[x].pcolormesh(d, cmap=cmap, vmin=vmin, vmax=vmax); ax[x].set_title(ti); fig.colorbar(im, ax=ax[x])
sv = max(abs(nBSF[wet2d].min()), abs(nBSF[wet2d].max())) * 1.2
tv = (nT[:, :, 0][wet2d].min(), nT[:, :, 0][wet2d].max())
show(lT[:, :, 0], (0, 0), "SST legoESM", "turbo", *tv); show(nT[:, :, 0], (0, 1), "SST NEMO", "turbo", *tv)
show(lT[:, :, 0] - nT[:, :, 0], (0, 2), f"SST diff (rms {sst['rms']:.2f})", "RdBu_r", -1.5, 1.5)
show(lBSF, (1, 0), f"BSF legoESM [{lBSF[wet2d].min():.0f},{lBSF[wet2d].max():.0f}]Sv", "RdBu_r", -sv, sv)
show(nBSF, (1, 1), f"BSF NEMO [{nBSF[wet2d].min():.0f},{nBSF[wet2d].max():.0f}]Sv", "RdBu_r", -sv, sv)
show(lBSF - nBSF, (1, 2), "BSF diff", "RdBu_r", -sv, sv)
# zonal-mean T section (lat vs depth)
def zsec(T): return np.nanmean(np.where(wet3d, T, np.nan), axis=1).T  # (lev,lat)
la = gphit[:, 0]
for j, (T, ti) in enumerate([(lT, "T zonal-mean legoESM"), (nT, "T zonal-mean NEMO")]):
    im = ax[2, j].pcolormesh(la, -gdept, zsec(T), cmap="turbo", vmin=0, vmax=20); ax[2, j].set_title(ti); fig.colorbar(im, ax=ax[2, j])
im = ax[2, 2].pcolormesh(la, -gdept, zsec(lT) - zsec(nT), cmap="RdBu_r", vmin=-2, vmax=2); ax[2, 2].set_title("T section diff"); fig.colorbar(im, ax=ax[2, 2])
plt.tight_layout(); plt.savefig(OUT, dpi=90); print("wrote", OUT)
