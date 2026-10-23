"""Generate tests/atmosphere/oracle_fixtures/rrtmgp_lw_sfc_face.npz.

Oracle: RTE+RRTMGP v1.8 (compute_Planck_source in mo_gas_optics_rrtmgp_kernels.F90),
given tlev from the CAM6 host convention (rrtmg_state.F90: log-p interpolation between
layer midpoints, lid = top-layer temperature).  The surface interface is set to T_sfc;
CAM6 uses (lwup/stebol)**0.25, which equals T_sfc only for a unit-emissivity surface
(for the emissivity-0.98 RFMIP columns T_b - T_sfc = -0.2..-0.6 K).
The fixture holds the arrays legoESM's production entry ``RRTMGP.solve_columns``
actually hands to the LW solver (pressure/temperature/vmr incl. overhead layer and
halos, captured by a pass-through wrapper of ``two_stream.solve_lw``) and the oracle's
per-g-point surface level source lev_source(:,1,:) and surface source sfc_source.

Columns: RFMIP clear-sky (rrtmgp-data v1.8.2, expt 0) sites as in the tier-1 audit
(/burg/glab/users/pg2328/oracle_audit/rrtmgp_gas_optics_rte/ours/run_ours.py),
truncated at ~225 Pa.

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu):
    bash build.sh <rte-rrtmgp@v1.8> <conda env> <build dir>
    python make_fixture.py <build dir>/oracle_driver <rte-rrtmgp@v1.8> <rrtmgp-data@v1.8.2> <out.npz>
"""
import json
import os
import subprocess
import sys
import tempfile

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as R  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp import constants as RC  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream as TS  # noqa: E402

ORACLE, RTE, DATA, OUT = sys.argv[1:5]
RFMIP = os.path.join(DATA, "examples/rfmip-clear-sky/inputs/"
                     "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc")
P_TOP_TARGET = 225.0  # Pa, CAM L32 model-top interface (tier-1 audit column set)
EXPT = 0

# ------------------------------------------------ columns (tier-1 audit selection)
ds = netCDF4.Dataset(RFMIP)
g = lambda k: np.asarray(ds[k][:], dtype=np.float64)  # noqa: E731
lat = g("lat")
p_lay, p_lev = g("pres_layer"), g("pres_level")
t_lay = g("temp_layer")[EXPT]
tsfc_all = g("surface_temperature")[EXPT]
h2o_all, o3_all = g("water_vapor")[EXPT], g("ozone")[EXPT]
emis_all, alb_all = g("surface_emissivity"), g("surface_albedo")
mu_all = np.cos(np.deg2rad(g("solar_zenith_angle")))
tsi_all = g("total_solar_irradiance")
gm = {k: float(g(f"{k}_GM")[EXPT]) * float(ds[f"{k}_GM"].units)
      for k in ("carbon_dioxide", "methane", "nitrous_oxide", "cfc11", "cfc12")}
h2o_all = h2o_all * float(ds["water_vapor"].units)
o3_all = o3_all * float(ds["ozone"].units)
ds.close()
if p_lev[0, 0] > p_lev[0, -1]:  # legoESM column convention: index 0 = TOA
    p_lay, p_lev, t_lay = p_lay[:, ::-1], p_lev[:, ::-1], t_lay[:, ::-1]
    h2o_all, o3_all = h2o_all[:, ::-1], o3_all[:, ::-1]
dp_all = np.diff(p_lev, axis=1)
pw = np.sum(h2o_all * dp_all, axis=1)
day = mu_all > 0.05
alat = np.abs(lat)
m = (alat < 20) & day
cands = {"tropical_moist": np.where(m)[0][np.argmax(pw[m])]}
m = (alat > 15) & (alat < 35) & day
cands["subtropical_dry"] = np.where(m)[0][np.argmin(pw[m])]
m = (alat > 40) & (alat < 60) & day
cands["midlatitude"] = np.where(m)[0][np.argmin(np.abs(alat[m] - 50))]
cands["polar_cold"] = int(np.argmin(tsfc_all))
cands["high_altitude"] = int(np.where(alat < 60)[0][np.argmin(p_lev[alat < 60, -1])])
names = list(cands) + ["edge_dry_lowsun", "terminator_mu0.005", "terminator_mu0.002"]
sites = [int(v) for v in cands.values()] + [int(cands["polar_cold"])] + [int(cands["midlatitude"])] * 2
k0 = max(int(np.where(p_lev[s] >= P_TOP_TARGET)[0][0]) for s in sites)
cols = []
for name, s in zip(names, sites):
    pf = p_lay[s, k0:]
    h2o = h2o_all[s, k0:].copy()
    mu0 = float(mu_all[s]) if mu_all[s] > 0.05 else 0.5
    if name == "edge_dry_lowsun":
        h2o[pf < 1.0e4] = 0.0
        mu0 = float(np.cos(np.deg2rad(87.0)))
    if name.startswith("terminator_mu0"):
        mu0 = float(name.split("mu0")[1])
    r = RC.R_V / RC.R_D
    cols.append(dict(p_half=p_lev[s, k0:], p_full=pf, T=t_lay[s, k0:], q=h2o / (r + h2o),
                     o3=o3_all[s, k0:],
                     o3_top=float(np.sum(o3_all[s, :k0] * dp_all[s, :k0]) / np.sum(dp_all[s, :k0])),
                     tsfc=float(tsfc_all[s]), emis=float(emis_all[s]), alb=float(alb_all[s]),
                     mu0=mu0, tsi=float(tsi_all[s])))
stack = lambda k: np.stack([c[k] for c in cols])  # noqa: E731
vec = lambda k: np.array([c[k] for c in cols])  # noqa: E731
ghg = {"co2": gm["carbon_dioxide"], "ch4": gm["methane"], "n2o": gm["nitrous_oxide"],
       "cfc11": gm["cfc11"], "cfc12": gm["cfc12"]}
cfg_kw = dict(S_0=float(np.mean(vec("tsi"))), include_clouds=False, overhead_layer=True,
              compute_fp32=False, gpoint_batch_size=0, use_optimal_angle=False,
              co2_ppmv=ghg["co2"] * 1e6, ch4_ppbv=ghg["ch4"] * 1e9, n2o_ppbv=ghg["n2o"] * 1e9)

# ------------------------------------------------ legoESM production entry, captured
CAP = {}
_orig_lw = TS.solve_lw


def _cap_lw(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields=None,
            sfc_temperature=None, **kw):
    CAP.update(p=pressure, T=temperature, mol=molecules, vmr=dict(vmr_fields),
               sfcT=sfc_temperature, optics=optics_lib)
    return _orig_lw(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields,
                    sfc_temperature, **kw)


TS.solve_lw = _cap_lw
solver = R.RRTMGP.from_legoesm_config(RRTMGPConfig(**cfg_kw))
solver.solve_columns(
    T=jnp.asarray(stack("T")), p_full=jnp.asarray(stack("p_full")),
    p_half=jnp.asarray(stack("p_half")), sfc_temperature=jnp.asarray(vec("tsfc")),
    q_v=jnp.asarray(stack("q")), cos_zenith=jnp.asarray(vec("mu0")),
    sfc_albedo=jnp.asarray(vec("alb")), sfc_emissivity=jnp.asarray(vec("emis")),
    o3_vmr=jnp.asarray(stack("o3")), o3_top_vmr=jnp.asarray(vec("o3_top")),
    ghg_vmr_override=ghg)
TS.solve_lw = _orig_lw
opt = CAP["optics"]

# ------------------------------------------------ oracle inputs (surface-first)
interior = lambda a: np.asarray(a, dtype=np.float64)[:, 0, 1:-1]  # noqa: E731
play, tlay = interior(CAP["p"]), interior(CAP["T"])
col_dry = interior(CAP["mol"]) / 1.0e4  # molecules/m2 -> molecules/cm2
ph = stack("p_half")
plev = np.concatenate([np.minimum(ph[:, :1], 1.0), ph], axis=1)[:, ::-1]  # incl. 1 Pa lid
tsfc = np.asarray(CAP["sfcT"], dtype=np.float64).reshape(-1)
glist, vlist = [], []
for gname in (str(x) for x in opt.gas_optics_lw.idx_gases):
    if gname in ("dry_air", "h2o_frgn", "h2o_self"):
        continue
    if gname in CAP["vmr"]:
        v = interior(CAP["vmr"][gname])
    elif gname in opt.vmr_lib.global_means:
        v = np.full(play.shape, float(opt.vmr_lib.global_means[gname]))
    else:
        continue  # gas absent in legoESM (vmr 0)
    glist.append(gname)
    vlist.append(v)
# CAM6 host tint (rrtmg_state.F90:113-119) with the surface level = T_sfc (not
# (lwup/stebol)**0.25), TOA-first then flipped to surface-first
pl, pv, tl = play[:, ::-1], plev[:, ::-1], tlay[:, ::-1]
tint = np.empty_like(pv)
tint[:, 0] = tl[:, 0]
tint[:, -1] = tsfc
for k in range(1, pl.shape[1]):
    dy = (np.log(pv[:, k]) - np.log(pl[:, k])) / (np.log(pl[:, k - 1]) - np.log(pl[:, k]))
    tint[:, k] = tl[:, k] - dy * (tl[:, k] - tl[:, k - 1])
tlev = tint[:, ::-1]
ncol, nlay = play.shape
mu0 = vec("mu0")
toa = np.outer(np.ones(ncol), cfg_kw["S_0"] * np.asarray(opt.solar_fraction_by_gpt, np.float64))

with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "dims.txt"), "w") as f:
        f.write(f"{ncol} {nlay} {len(glist)}\n" + "\n".join(glist) + "\n")
    with open(os.path.join(d, "in.bin"), "wb") as f:
        for a in (play, plev, tlay, tlev, tsfc, col_dry, np.stack(vlist, -1), vec("emis"), mu0,
                  vec("alb"), vec("alb"), toa):
            np.asarray(a, dtype="<f8").ravel(order="F").tofile(f)
    run = subprocess.run([ORACLE, d, os.path.join(DATA, "rrtmgp-gas-lw-g128.nc"),
                          os.path.join(DATA, "rrtmgp-gas-sw-g112.nc")],
                         capture_output=True, text=True, check=True)
    print(run.stdout)
    assert "ORACLE_DONE" in run.stdout
    raw = np.fromfile(os.path.join(d, "out_lw.bin"), dtype="<f8")
ng = opt.n_gpt_lw
o = ncol * nlay * ng * 2  # skip tau, lay_source
lev_src = raw[o:o + ncol * (nlay + 1) * ng].reshape(ncol, nlay + 1, ng, order="F")
o += ncol * (nlay + 1) * ng
sfc_src = raw[o:o + ncol * ng].reshape(ncol, ng, order="F")

sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                               capture_output=True, text=True).stdout.strip()
meta = dict(
    oracle="rte-rrtmgp v1.8 " + sha(RTE),
    oracle_note="lev_source(:,1,:) and sfc_source from mo_gas_optics_rrtmgp_kernels.F90 "
                "compute_Planck_source, tlev = CAM6 rrtmg_state.F90 tint (surface = T_sfc)",
    rrtmgp_data="v1.8.2 " + sha(DATA),
    legoesm=sha(os.path.dirname(os.path.abspath(__file__))),
    generator="scripts/validate/oracle/rrtmgp_lw_sfc_face/{build.sh,make_fixture.py}",
    columns=names, cfg=cfg_kw)
vnames = sorted(CAP["vmr"])
np.savez_compressed(
    OUT, p=np.asarray(CAP["p"], np.float64), T=np.asarray(CAP["T"], np.float64),
    sfcT=np.asarray(CAP["sfcT"], np.float64), vmr_names=np.array(vnames),
    vmr=np.stack([np.asarray(CAP["vmr"][k], np.float64) for k in vnames]),
    tlev_sfc_oracle=tlev[:, 0], lev_src_sfc_oracle=lev_src[:, 0, :], sfc_src_oracle=sfc_src,
    meta=json.dumps(meta))
print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")
