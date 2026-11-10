"""Generate tests/atmosphere/oracle_fixtures/rrtmgp_pressure_range.npz.

Gas optics of RUNNING unpatched RTE+RRTMGP v1.8 (value checks off, see
oracle_driver.F90) on layers whose pressure spans the table and leaves it at
both ends: 0.3 and 0.8 Pa above the top (p_ref min 1.005 Pa), 1.12e5 and
1.3e5 Pa below the bottom (p_ref max 1096.6 hPa), the two end nodes exactly,
and log-spaced in-table layers.  Temperature is held inside the table
(200/250/300 K, one per column) so pressure is the only out-of-range input.
Gases: RFMIP global means (column 0 h2o and o3 interpolated in ln p, ends held);
a table gas RFMIP does not carry gets vmr 0 on both sides.

    python make_fixture.py <build>/oracle_driver <rte-rrtmgp@v1.8> <rrtmgp-data@v1.8.2> <out.npz>

Run on a compute node (sbatch).
"""

import json
import os
import subprocess
import sys
import tempfile

import netCDF4
import numpy as np

ORACLE, RTE, DATA, OUT = sys.argv[1:5]
LW = os.path.join(DATA, "rrtmgp-gas-lw-g128.nc")
SW = os.path.join(DATA, "rrtmgp-gas-sw-g112.nc")
RFMIP = os.path.join(
    DATA, "examples/rfmip-clear-sky/inputs/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc")
SKIP = ("dry_air", "h2o_frgn", "h2o_self")
# RRTMGP gas name -> RFMIP global-mean variable stem
RFMIP_GM = {"co2": "carbon_dioxide", "ch4": "methane", "n2o": "nitrous_oxide", "co": "carbon_monoxide",
            "o2": "oxygen", "n2": "nitrogen", "ccl4": "carbon_tetrachloride", "cfc11": "cfc11",
            "cfc12": "cfc12", "cfc22": "hcfc22", "hfc143a": "hfc143a", "hfc125": "hfc125", "hfc23": "hfc23",
            "hfc32": "hfc32", "hfc134a": "hfc134a", "cf4": "cf4", "no2": "no2"}


def table(path):
    with netCDF4.Dataset(path) as d:
        x = np.asarray(d["gas_names"][:])
        names = [str(n).strip() for n in (x if x.ndim == 1 else netCDF4.chartostring(x))]
        return names, np.asarray(d["press_ref"][:], np.float64)


lw_names, p_ref = table(LW)
sw_names, p_ref_sw = table(SW)
assert np.array_equal(p_ref, p_ref_sw)
gases = [g for g in dict.fromkeys(lw_names + sw_names) if g not in SKIP]

p_in = np.exp(np.linspace(np.log(2.0), np.log(1.05e5), 24))
play1 = np.sort(np.concatenate([[0.3, 0.8, p_ref.min(), p_ref.max(), 1.12e5, 1.3e5], p_in]))[::-1]
nlay = play1.size
temps = np.array([200.0, 250.0, 300.0])
ncol = temps.size
play = np.broadcast_to(play1, (ncol, nlay)).copy()                # surface first
plev = np.concatenate([play[:, :1] * 1.01, np.sqrt(play[:, 1:] * play[:, :-1]), play[:, -1:] * 0.99], 1)
tlay = np.broadcast_to(temps[:, None], (ncol, nlay)).copy()
tlev = np.broadcast_to(temps[:, None], (ncol, nlay + 1)).copy()
tsfc = temps.copy()
col_dry = np.full((ncol, nlay), 2.0e19)                            # molecules/cm2

with netCDF4.Dataset(RFMIP) as d:
    vmr = {}
    for g in gases:
        if g in ("h2o", "o3"):
            v = "water_vapor" if g == "h2o" else "ozone"
            prof = np.asarray(d[v][0, 0, :], np.float64) * float(d[v].units)
            pl = np.asarray(d["pres_layer"][0, :], np.float64)
            o = np.argsort(pl)
            vmr[g] = np.broadcast_to(np.interp(np.log(play1), np.log(pl[o]), prof[o]), (ncol, nlay)).copy()
        elif g in RFMIP_GM and f"{RFMIP_GM[g]}_GM" in d.variables:
            gm = d[f"{RFMIP_GM[g]}_GM"]
            vmr[g] = np.full((ncol, nlay), float(gm[0]) * float(gm.units))
        else:
            vmr[g] = np.zeros((ncol, nlay))


def write_f(path, arrays):
    with open(path, "wb") as f:
        for a in arrays:
            np.asarray(a, dtype="<f8").ravel(order="F").tofile(f)


with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "dims.txt"), "w") as f:
        f.write(f"{ncol} {nlay} {len(gases)}\n" + "\n".join(gases) + "\n")
    write_f(os.path.join(d, "in.bin"), [play, plev, tlay, tlev, tsfc, col_dry,
                                        np.stack([vmr[g] for g in gases], -1)])
    run = subprocess.run([ORACLE, d, LW, SW], capture_output=True, text=True)
    print(run.stdout[-800:], run.stderr[-400:])
    assert run.returncode == 0 and "ORACLE_DONE" in run.stdout, "oracle failed"
    raw = np.fromfile(os.path.join(d, "out.bin"), "<f8")

with netCDF4.Dataset(LW) as d:
    ngl = d.dimensions["gpt"].size
with netCDF4.Dataset(SW) as d:
    ngs = d.dimensions["gpt"].size
out, k = {}, 0
for name, ng in (("tau_lw", ngl), ("lay_src", ngl), ("tau_sw", ngs), ("ssa_sw", ngs)):
    n = ncol * nlay * ng
    out[name] = raw[k:k + n].reshape((ncol, nlay, ng), order="F")
    k += n
assert k == raw.size, (k, raw.size)

sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                               capture_output=True, text=True).stdout.strip()
meta = dict(oracle="rte-rrtmgp v1.8 " + sha(RTE) + ", rte_config_checks(.false.)",
            rrtmgp_data=sha(DATA), generator="scripts/validate/oracle/rrtmgp_pressure_range/"
            "{build.sh,oracle_driver.F90,make_fixture.py}", units="col_dry molecules/cm2; lay_src W/m2")
np.savez(OUT, play=play, tlay=tlay, tlev=tlev, tsfc=tsfc, col_dry=col_dry, gases=np.array(gases),
         vmr=np.stack([vmr[g] for g in gases], -1), meta=json.dumps(meta), **out)
print("FIXTURE_OK", OUT, {k: v.shape for k, v in out.items()})
