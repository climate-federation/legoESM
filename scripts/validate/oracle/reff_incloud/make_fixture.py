"""Generate tests/atmosphere/oracle_fixtures/reff_incloud.npz.

Oracle: CAM cam_cesm2_1_rel_60 MG2 (micro_mg2_0.F90 + micro_mg_utils.F90), the code
that forms CAM6's radiation droplet/ice effective radius (REL/REI): lcldm = icldm =
max(ast, mincld) (L877-879; micro_mg_cam.F90:1810-1811 point liqcldf/icecldf at ast),
in-cloud dumc = qc/lcldm and, with a specified droplet number (nccons), the in-cloud
dumnc = ncnst/rho (L2739-2768), then effi (L2772-2808) and effc (L2811-2865).  legoESM's
droplet PSD is the WRF v4.6.1 M2005 one (certified to round-off in the tier-1 audit with
legoESM's rho_w and moist density); the asserted reference ``effc_wrf`` is that PSD
evaluated by the oracle driver on MG2's in-cloud (dumc, ncnst).  ``effc_mg2`` (MG2's own
exponential-Martin PSD, 2-50 um mean-diameter bounds) is stored for the PSD gap only.

Inputs: the 5 tier-1 CAM6-L32 columns (scripts/validate/oracle/rrtmgp_cloud_optics/
make_fixture.py ``build_columns``; constructed soundings, NOT a checkpoint) plus three
edge levels where ast < mincld with uncapped in-cloud water, and a deep-convective mass
flux in two columns so the radiative cover (ast + deepcu) differs from ast.  The cloud
fraction handed to the oracle is legoESM's own ast = max(alst, aist) (cam6
``cam6_ice_stratus_fraction``); the droplet number is the specified in-cloud CCN number.

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu, PYTHONPATH incl. packages):
    bash build.sh <CAM@cam_cesm2_1_rel_60> <WRF phys/module_mp_morr_two_moment.F> <build dir>
    python make_fixture.py <build dir>/oracle_driver <CAM checkout> <WRF file> <out.npz>
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import compute_rho  # noqa: E402
from legoesm.atmosphere.physics.clouds.cloud_fraction import (  # noqa: E402
    cam6_ice_stratus_fraction, compute_cloud_properties)

ORACLE, CAM, WRF, OUT = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
_spec = importlib.util.spec_from_file_location(
    "cloud_optics_fixture", os.path.join(HERE, "../rrtmgp_cloud_optics/make_fixture.py"))
cof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cof)
CAM6_DCS = 500.0e-6  # namelist_defaults_cam.xml:1244 micro_mg_dcs (mg2); sets MG2 ice bounds

names, cols = cof.build_columns()
e = cols[4]
k = lambda p: int(np.argmin(np.abs(e["p_full"] - p)))  # noqa: E731
for p, qc, a in ((85000., 3.0e-7, 1.0e-6), (75000., 2.0e-7, 0.0), (65000., 1.0e-6, 0.01)):
    e["q_c"][k(p)], e["alst"][k(p)] = qc, a          # ast < mincld (first two), uncapped
# dead droplet number (n <= 1 -> CloudConfig.Nc_default, in-cloud water over the
# radiative cover floored at mincld): cover between mincld and 1e-3, and below mincld
DEAD_P = (10331., 14299.)
for p, a in zip(DEAD_P, (5.0e-4, 2.0e-5)):
    e["q_c"][k(p)], e["alst"][k(p)], e["n_ccn"][k(p)] = 1.0e-7, a, 0.0
S = lambda key: np.stack([c[key] for c in cols])  # noqa: E731
ncol, nlev = S("T").shape
p_half, p_full = S("p_half"), S("p_full")
mf = np.zeros((ncol, nlev + 1))
icw = np.zeros((ncol, nlev))
for j, (lo, hi, m) in ((0, (20000., 90000., 0.03)), (2, (40000., 92000., 0.01))):
    sel = (p_full[j] >= lo) & (p_full[j] <= hi)
    mf[j, :-1][sel] = m        # face above layer k; deepcu reads the LOWER face k+1
    mf[j, 1:][sel] = m
    icw[j][sel] = 1.0e-3

ccfg = cof.production_cloud_config()
assert ccfg.scheme == "cam6_clubb", ccfg.scheme
lat = np.array([c["lat"] for c in cols])
kw = dict(T=jnp.asarray(S("T")), p_full=jnp.asarray(p_full), q_v=jnp.asarray(S("q_v")),
          dp=jnp.asarray(np.diff(p_half, axis=1)), config=ccfg,
          q_cloud=jnp.asarray(S("q_c")), q_ice=jnp.asarray(S("q_i")),
          n_ice=jnp.asarray(S("n_i")), n_cloud=jnp.asarray(S("n_ccn")),
          cloud_fraction_override=jnp.asarray(S("alst")), lat=jnp.asarray(lat),
          conv_mass_flux_up=jnp.asarray(mf), conv_icwmr=jnp.asarray(icw),
          p_half=jnp.asarray(p_half))
props = compute_cloud_properties(**kw, n_cloud_in_cloud=True)
aist = cam6_ice_stratus_fraction(kw["q_v"], kw["T"], kw["p_full"], kw["q_ice"], kw["lat"],
                                 ccfg, kw["p_half"][:, :-1])
ast = np.asarray(jnp.maximum(jnp.clip(kw["cloud_fraction_override"], 0.0, 1.0), aist))
rho = np.asarray(compute_rho(jnp.maximum(kw["T"], 1.0), kw["p_full"], kw["q_v"]))

with tempfile.TemporaryDirectory() as d:
    fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
    with open(fin, "wb") as f:
        np.asarray([ncol * nlev], "<i4").tofile(f)
        np.asarray([CAM6_DCS, constants.rho_water], "<f8").tofile(f)
        nc_or = np.where(S("n_ccn") > 1.0, S("n_ccn"), ccfg.Nc_default)
        for a in (S("q_c"), S("q_i"), S("n_i"), nc_or, ast, rho):
            np.asarray(a, "<f8").ravel().tofile(f)
    run = subprocess.run([ORACLE, fin, fout], capture_output=True, text=True, check=True)
    assert "ORACLE_DONE" in run.stdout, run.stdout
    raw = np.fromfile(fout, dtype="<f8")
assert raw.size == 7 * ncol * nlev
lcldm, dumc, effc_mg2, effi_mg2, ni_adj, effc_wrf, rel_cam = raw.reshape(7, ncol, nlev)

cf = np.asarray(props.cloud_fraction)
liq = S("q_c") > 1.0e-14
print("deepcu>0 on liquid levels:", int(np.sum(liq & (cf > ast + 1e-12))),
      "| ast<mincld uncapped liquid:", int(np.sum(liq & (ast < 1e-4) & (dumc < 5e-3))))
assert np.sum(liq & (cf > ast + 1e-12)) > 0 and np.sum(liq & (ast < 1e-4) & (dumc < 5e-3)) > 0
dead = S("n_ccn") <= 1.0
# the oracle sees ast; legoESM's dead-number branch divides by the radiative cover
assert dead.sum() == len(DEAD_P) and np.all(cf[dead] == ast[dead]), (cf[dead], ast[dead])
print("max |rel_cam - effc_mg2| / effc_mg2:",
      float(np.max(np.abs(rel_cam - effc_mg2) / effc_mg2)))

sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                               capture_output=True, text=True).stdout.strip()
tagof = lambda p: subprocess.run(["git", "-C", p, "describe", "--tags"],  # noqa: E731
                                 capture_output=True, text=True).stdout.strip()
cfg_d = ccfg._asdict()
assert type(ccfg)(**json.loads(json.dumps(cfg_d))) == ccfg
meta = dict(
    oracle=f"CAM {tagof(CAM)} {sha(CAM)} micro_mg2_0.F90 L877-879,2739-2768,2772-2808,"
           "2811-2865 + micro_mg_cam.F90 L2296,2534-2554 (grid REL) + micro_mg_utils.F90 "
           "(unmodified), dcs=500e-6; dead droplet number -> ncnst = Nc_default",
    oracle_psd=f"WRF module_mp_morr_two_moment.F L3918-3949,4024-4029 (sha256 "
               f"{hashlib.sha256(open(WRF, 'rb').read()).hexdigest()}, identical to v4.6.1), "
               "DUM = legoESM moist density, RHOW = constants.rho_water",
    legoesm=sha(REPO),
    generator="scripts/validate/oracle/reff_incloud/{build.sh,oracle_driver.F90,make_fixture.py}",
    columns=names, cloud_config=cfg_d)
np.savez_compressed(
    OUT, T=S("T"), p_full=p_full, p_half=p_half, q_v=S("q_v"), q_c=S("q_c"), q_i=S("q_i"),
    n_i=S("n_i"), n_ccn=S("n_ccn"), alst=S("alst"), lat=lat, conv_mass_flux_up=mf,
    conv_icwmr=icw, ast=ast, lcldm=lcldm, dumc=dumc, effc_wrf=effc_wrf, effc_mg2=effc_mg2,
    effi_mg2=effi_mg2, ni_adj=ni_adj, rel_cam=rel_cam, meta=json.dumps(meta))
print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")
