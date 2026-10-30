"""Build identical test columns for the CAM clubb_intr glue oracle.

Usage: build_inputs.py OUT_DIR RFMIP_INPUTS_NC
(RFMIP_INPUTS_NC = rrtmgp-data v1.8.2 examples/rfmip-clear-sky/inputs/
multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc).
Copied from the tier-1 audit (oracle_audit/clubb_cam_glue_cloud_fraction) with
ONE change: the prescribed momentum increments du, dv are zero, so the fixture
isolates the thermodynamic glue (theta-l map in/out, surface heat flux, energy
fixer, water reset/return) from the local KE-dissipation heating, which is a
separate audit finding not ported here.

Thermodynamic profiles: RFMIP (ERA-Interim) present-day sites (expt 0) from
rrtmgp-data v1.8.2 examples/rfmip-clear-sky/inputs, log-p interpolated onto the
production CAM6 L32 hybrid grid (legoESM make_cam6_l32_levels, p_s = RFMIP bottom
interface).  Cloud liquid/ice, winds, convective mass flux, surface fluxes and the
"CLUBB output" increments are SYNTHETIC (documented in README) and identical on
both sides.  Writes inputs.npz (ours) and oracle_in.txt (Fortran, top-down CAM order).
"""
import os
import sys
import numpy as np
import netCDF4
import jax
jax.config.update("jax_enable_x64", True)
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())   # production deck: precision fp64
import jax.numpy as jnp
from legoesm import constants as C
from legoesm.grids.vertical import make_cam6_l32_levels
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma, compute_rho

OUT = sys.argv[1]
F = sys.argv[2]
nc = netCDF4.Dataset(F)
lat_all = np.array(nc["lat"][:], dtype=np.float64)
sst = np.array(nc["sst"][:], dtype=np.float64)
pl = np.array(nc["pres_layer"][:], dtype=np.float64)       # (site, 60)
pe = np.array(nc["pres_level"][:], dtype=np.float64)       # (site, 61)
Tl = np.array(nc["temp_layer"][0], dtype=np.float64)       # expt 0
wv = np.array(nc["water_vapor"][0], dtype=np.float64)      # mole fraction


def pick(cond, key):
    idx = np.where(cond)[0]
    return int(idx[np.argmax(key[idx])])


qcol = wv[:, :].max(axis=1)
psurf = pe.max(axis=1)
sites = {
    "tropical_deep": pick(np.abs(lat_all) < 10, qcol),
    "subtrop_sc": pick((lat_all < -15) & (lat_all > -30) & (sst > 0), -sst),   # coolest subtropical ocean
    "midlat": pick((np.abs(lat_all) > 40) & (np.abs(lat_all) < 55), psurf),
    "polar_nh": pick(lat_all > 65, lat_all),
    "antarctic_highterrain": pick(lat_all < -70, -psurf),
    "subtrop_nh": pick((lat_all > 15) & (lat_all < 35), qcol),
}
names = list(sites) + ["rhi_ramp", "zero_edge"]
idx = [sites[n] for n in sites] + [sites["tropical_deep"], sites["midlat"]]
ncol = len(names)

coord = make_cam6_l32_levels()
ps = np.array([psurf[i] for i in idx])
p_half = np.asarray(coord.pressure_at_half(jnp.asarray(ps)))
p_full = np.asarray(coord.pressure_at_full(jnp.asarray(ps)))
nlev = p_full.shape[1]

T = np.zeros((ncol, nlev)); qv = np.zeros((ncol, nlev))
eps = C.epsilon
for c, i in enumerate(idx):
    o = np.argsort(pl[i])
    lp = np.log(pl[i][o])
    T[c] = np.interp(np.log(p_full[c]), lp, Tl[i][o])
    r = wv[i][o] * eps                                  # mass mixing ratio from mole fraction
    qv[c] = np.interp(np.log(p_full[c]), lp, r / (1.0 + r))   # specific humidity
lat = np.deg2rad(np.array([lat_all[i] for i in idx]))

hPa = 100.0
def bump(p, pc, w, amp):
    return amp * np.exp(-((p - pc) / w) ** 2)

qc = np.zeros_like(T); qi = np.zeros_like(T)
qc[0] = bump(p_full[0], 800 * hPa, 60 * hPa, 2e-4)
qc[1] = bump(p_full[1], 920 * hPa, 30 * hPa, 4e-4)
qc[2] = bump(p_full[2], 650 * hPa, 80 * hPa, 1e-4)
qc[3] = bump(p_full[3], 900 * hPa, 50 * hPa, 2e-5)
qc[5] = bump(p_full[5], 850 * hPa, 50 * hPa, 1e-4)
qi[0] = bump(p_full[0], 250 * hPa, 80 * hPa, 3e-5)
qi[2] = bump(p_full[2], 350 * hPa, 80 * hPa, 1e-5)
qi[3] = bump(p_full[3], 500 * hPa, 200 * hPa, 5e-6)
qi[4] = bump(p_full[4], 400 * hPa, 150 * hPa, 2e-6)
qi[5] = bump(p_full[5], 200 * hPa, 50 * hPa, 8e-6)
# limiter branches: sub-minice, icimr<qist_min, icimr>qist_max
for c in (0, 2, 5):
    qi[c, 3] = 5e-13; qi[c, 4] = 3e-9
qi[0, 18] = 4e-3
# rhi ramp column: qv scanned so that rh over ICE spans 0.6..1.2 in the cold layers
# (built with the Murphy-Koop ice curve, an INPUT choice only)
Tr = T[6]
esi_mk = np.exp(9.550426 - 5723.265 / Tr + 3.53068 * np.log(Tr) - 0.00728332 * Tr)
target = np.linspace(0.6, 1.2, nlev)
qv[6] = np.where(Tr < 268.0, target * eps * esi_mk / (p_full[6] - (1 - eps) * esi_mk), qv[6])
qi[6] = np.where(Tr < 268.0, 2e-6, 0.0)
# zero/edge column: dry of condensate, no convection, no fluxes
qc[7] = 0.0; qi[7] = 0.0

u = 5.0 + 20.0 * np.sin(np.pi * (1 - p_full / ps[:, None]))
v = -3.0 + 4.0 * np.cos(np.pi * p_full / ps[:, None])
u[7] = 0.0; v[7] = 0.0

z_full, z_half = (np.asarray(a) for a in compute_heights_from_sigma(
    jnp.asarray(T), jnp.asarray(p_half), q_v=jnp.asarray(qv)))
rho = np.asarray(compute_rho(jnp.asarray(T), jnp.asarray(p_full), q_v=jnp.asarray(qv)))
dp = p_half[:, 1:] - p_half[:, :-1]

# convective mass flux (interfaces, top-down) + in-cloud water, tropical & subtropNH only
mf = np.zeros((ncol, nlev + 1)); icw = np.zeros_like(T)
for c, amp in ((0, 0.04), (5, 0.01), (6, 0.03)):
    mf[c] = bump(p_half[c], 500 * hPa, 250 * hPa, amp)
    mf[c, -1] = 0.0
    icw[c] = bump(p_full[c], 500 * hPa, 250 * hPa, 5e-4)
    icw[c, 0:3] = 1e-13     # below ic_limit
mf[0, 20] = 1e-5            # deepcu below frac_limit

# surface fluxes [W/m2, kg/m2/s, Pa], drag-negative stress convention
shf = np.array([10., 8., 40., -20., -10., 15., 10., 0.])
E = np.array([5e-5, 3e-5, 4e-5, 2e-6, 1e-6, 4e-5, 5e-5, 0.0])
lhf = C.L_v * E
taux = np.array([-0.05, -0.10, -0.20, -0.08, -0.12, -0.06, -0.05, 0.0])
tauy = np.array([0.02, 0.03, -0.05, 0.01, 0.04, 0.0, 0.02, 0.0])

# synthetic "CLUBB core output": increments (Delta) of the conserved means and
# absolute rcm / cloud_frac / rtp2.  Identical on both sides.  Zero above the
# CAM top_lev (CLUBB truncated there) is applied per side.
dt = 600.0
shape_bl = np.exp(-((ps[:, None] - p_full) / (100 * hPa)) ** 2)
mass = dp / C.g
# water increment closes the column with the surface flux: sum(drt * dp/g) = E*dt
drt = (E * dt / np.sum(shape_bl * mass, axis=1))[:, None] * shape_bl
# plus a water-conserving dipole in the UPPER troposphere / tropopause region
up = bump(p_full, 140 * hPa, 40 * hPa, 1.0) - bump(p_full, 250 * hPa, 40 * hPa, 1.0) * 0.0
drt_up = 2e-7 * bump(p_full, 140 * hPa, 30 * hPa, 1.0)
drt = drt + drt_up
dthl = (0.3 * shape_bl - 0.05 * bump(p_full, 600 * hPa, 200 * hPa, 1.0)
        + 0.02 * bump(p_full, 140 * hPa, 30 * hPa, 1.0))
# No momentum increment (see module docstring): the KE-dissipation heating
# placement is not part of this fixture.
du = np.zeros_like(T)
dv = np.zeros_like(T)
rcm_out = np.where(qc > 0, 1.1 * qc, 0.0)
rcm_out[0] += bump(p_full[0], 600 * hPa, 30 * hPa, 5e-5)    # new cloud level
cf_out = np.clip(np.where(rcm_out > 1e-8, 0.2 + 2000.0 * rcm_out, 0.0), 0.0, 0.95)
# rtp2 on interfaces (top-down, k = TOP interface of layer k)
pi_ = p_half
rtp2 = np.where(pi_ > 200 * hPa, 1e-8 * (pi_ / ps[:, None]) ** 4, 1e-16)
rtp2[:, 0] = 1e-16
rtp2[0, 9] = 1e-12          # isolated non-quiet level above the tropical tropopause
for a in (drt, dthl, du, dv, rcm_out, cf_out):
    a[7] = 0.0
# col 7 (edge): keep the generic rtp2 so clubbtop sits near 200 hPa, and plant one
# nearly-dry level below it to trigger CAM's apply_to_surface branch (set per case below)
qv[7, 20] = 1e-9
# col 6: move to 60N so the tropical (high, <=125 hPa) tropopause is REJECTED by the
# polar rule and the fallback surface is used (exercises both branches)
lat[6] = np.deg2rad(60.0)


# --- per-case CLUBB-domain mask (CAM ref_pres top_lev) + exact water closure ----
pref_mid = np.asarray(coord.pressure_at_full(jnp.asarray([1.0e5])))[0]
cases = {}
for tctp in (15000.0, 100.0):
    tl = int(np.where(pref_mid > tctp)[0][0])          # 0-based top_lev
    m = (np.arange(nlev) >= tl)[None, :].astype(float)
    c = {k: a * m for k, a in (("drt", drt), ("dthl", dthl), ("du", du), ("dv", dv),
                               ("rcm_out", rcm_out), ("cf_out", cf_out))}
    # column water change = E*dt exactly (CLUBB conserves rt with the surface flux):
    resid = E * dt - np.sum(c["drt"] * mass, axis=1)
    c["drt"] = c["drt"] + (resid / np.sum(shape_bl * m * mass, axis=1))[:, None] * shape_bl * m
    c["drt"][7] = 0.0
    c["drt"][7, tl:16] = -0.5 * (qv[7, tl:16] + qc[7, tl:16])   # removed above clubbtop
    # ... and put back in the boundary layer so CLUBB's net column change stays E*dt = 0
    _r = -np.sum(c["drt"][7] * mass[7])
    c["drt"][7] += _r / np.sum(shape_bl[7] * m[0] * mass[7]) * shape_bl[7] * m[0]
    c["top_lev"] = tl
    cases[int(tctp)] = c


def nosub(a):   # XLA CPU flushes subnormals to zero; keep inputs normal on both sides
    a = np.array(a, dtype=np.float64)
    a[np.abs(a) < 1e-200] = 0.0
    return a


for k in ("qc", "qi", "icw", "mf"):
    globals()[k] = nosub(globals()[k])
for c in cases.values():
    for k in ("drt", "dthl", "du", "dv", "rcm_out", "cf_out"):
        c[k] = nosub(c[k])
extra = {f"{k}_{t}": v for t, c in cases.items() for k, v in c.items()}
np.savez(os.path.join(OUT, "inputs.npz"), names=np.array(names), site_index=np.array(idx), lat=lat,
         ps=ps, p_half=p_half, p_full=p_full, T=T, qv=qv, qc=qc, qi=qi, u=u, v=v,
         z_full=z_full, z_half=z_half, rho=rho, mf=mf, icw=icw, shf=shf, E=E,
         lhf=lhf, taux=taux, tauy=tauy, rtp2=rtp2, dt=dt, pref_mid=pref_mid, **extra)

for t, cc in cases.items():
    with open(os.path.join(OUT, f"oracle_in_{t}.txt"), "w") as f:
        f.write(f"{ncol} {nlev}\n")
        f.write(" ".join(f"{x:.17e}" for x in pref_mid) + "\n")
        for c in range(ncol):
            f.write(f"{lat[c]:.17e} {ps[c]:.17e} {shf[c]:.17e} {E[c]:.17e} {taux[c]:.17e} {tauy[c]:.17e}\n")
            for arr in (p_full[c], dp[c], T[c], qv[c], qc[c], qi[c], u[c], v[c], z_full[c],
                        icw[c], cc["drt"][c], cc["dthl"][c], cc["du"][c], cc["dv"][c],
                        cc["rcm_out"][c], cc["cf_out"][c]):
                f.write(" ".join(f"{x:.17e}" for x in arr) + "\n")
            for arr in (p_half[c], z_half[c], mf[c], rtp2[c]):
                f.write(" ".join(f"{x:.17e}" for x in arr) + "\n")
print("sites", sites)
print("ps", ps)
print("pref_mid", pref_mid)
print("top_lev(0b)", {t: c["top_lev"] for t, c in cases.items()})
