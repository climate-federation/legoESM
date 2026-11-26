"""Generate tests/atmosphere/oracle_fixtures/gray_isca.npz by RUNNING Isca's gray radiation.

Oracle: ExeClim/Isca tag v1.0 (5868cfd863b41823a3eeb30754ce06feba7528ec, the only
release tag), ``src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90``,
B_FRIERSON branch, compiled by ``build.sh`` with minimal FMS stubs
(``stubs_gray.F90``) and driven by ``drv_gray.F90`` (init -> down -> up, as Isca's
idealised_moist_phys calls it).  Constants (stefan, grav, cp_air) are legoESM's,
written into the oracle's input file.

Cases = the matched subset of the tier-2 audit (oracle_audit/gray_and_simple_radiation),
where ours and the oracle share the functional forms:

* E0 Isca B_FRIERSON namelist defaults (atm_abs = 0: transparent SW);
* E1 Isca defaults + atm_abs = 0.22 (SW absorption exercised);
* E2 legoESM's production GrayRadiationConfig mapped onto the Isca namelist
  (ir_tau_* = lw_diff_factor * tau_*, linear_tau, solar_exponent, atm_abs = sw_tau_0);

each with the oracle's ``pstd_mks`` set to the column surface pressure (ours uses
sigma = p/p_s; Isca code uses p/pstd_mks, the documented departure, not run here).
Dry LW (B_FRIERSON has no water-vapour term).  8 columns: the 6 tier-1 shared columns
(CAM L32 published-sounding shapes) plus a p_s = 600 hPa plateau and a p_s = 101325 Pa
column, T_sfc = T_bottom + dT per column, exactly as the tier-2 audit built them.

Self-checks (known values, every run): the oracle insolation equals Isca's P2 formula
0.25 S0 (1 + del_sol (1 - 3 sin^2 lat)/4) to 1e-14; with atm_abs = 0 the SW heating
is exactly zero; the TOA downward LW is exactly zero; the upward SW is constant.

usage (compute node):
    bash build.sh <Isca@v1.0> <build dir>
    python make_fixture.py <build dir>/drv_gray <Isca@v1.0> <columns.npz> <out.npz>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig

ORACLE, ISCA, COLS, OUT = sys.argv[1:5]

SPEC = [  # name, lat_deg, albedo, dT_sfc (T_sfc = T_bottom + dT); tier-2 harness.gray_columns
    ("trop_deep", 2.0, 0.06, 1.0), ("subtrop_sc", 22.0, 0.06, 1.5),
    ("midlat_cont", 45.0, 0.17, 2.0), ("trade_cu", 15.0, 0.06, 1.0),
    ("edge", -60.0, 0.5, -20.0), ("polar_dry", 80.0, 0.7, -6.0)]
ROWS = ("scal", "lw_up", "lw_down", "sw_up", "sw_down", "tdt_rad", "tdt_solar")
DEL_SOL = 1.4   # Isca B_FRIERSON namelist default


def columns():
    d = np.load(COLS)
    cols = []
    for n, lat, alb, dts in SPEC:
        T = d[n + "__T"]
        cols.append(dict(name=n, lat=np.deg2rad(lat), alb=alb, ph=d[n + "__p_half"], T=T, Ts=T[-1] + dts))
    ph = d["midlat_cont__p_half"] * (60000.0 / d["midlat_cont__p_half"][-1])
    cols.append(dict(name="plateau_ps600", lat=np.deg2rad(32.0), alb=0.2, ph=ph,
                     T=d["midlat_cont__T"] - 15.0, Ts=d["midlat_cont__T"][-1] - 12.0))
    ph = d["trop_deep__p_half"] * (constants.p_atm_std / d["trop_deep__p_half"][-1])
    cols.append(dict(name="ps_eq_pstd", lat=np.deg2rad(30.0), alb=0.1, ph=ph,
                     T=d["trop_deep__T"], Ts=d["trop_deep__T"][-1] + 1.0))
    return cols


def experiments():
    base = GrayRadiationConfig()
    assert base.sfc_emissivity == 1.0     # Isca's black surface; mapping below needs it
    D = base.lw_diff_factor
    # namelist: solar_constant del_sol del_sw sw_diff ir_tau_eq ir_tau_pole linear_tau
    #           wv_exponent solar_exponent atm_abs
    return {
        "E0_isca_defaults": [constants.S_0, DEL_SOL, 0.0, 0.0, 6.0, 1.5, 0.1, 4.0, 4.0, 0.0],
        "E1_isca_atmabs022": [constants.S_0, DEL_SOL, 0.0, 0.0, 6.0, 1.5, 0.1, 4.0, 4.0, 0.22],
        "E2_production_mapped": [constants.S_0, DEL_SOL, 0.0, 0.0, D * base.tau_equator,
                                 D * base.tau_pole, base.linear_frac, 4.0, base.sw_exponent,
                                 base.sw_tau_0],
    }


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def run(cols, nml):
    nlev = len(cols[0]["T"])
    s = [f"{len(cols)} {nlev}", f17(nml), f17([constants.sigma_sb, constants.g, constants.c_pd])]
    for c in cols:
        s += [f17([c["lat"], c["alb"], c["Ts"], c["ph"][-1]]), f17(c["ph"]), f17(c["T"])]
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "gray_in.txt", Path(td) / "gray_out.txt"
        fin.write_text("\n".join(s) + "\n")
        r = subprocess.run([ORACLE, str(fin), str(fout)], cwd=td, capture_output=True,
                           text=True, check=True)
        assert "GRAY_DRV_OK" in r.stdout, r.stdout + r.stderr
        lines = fout.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        assert lines[i].startswith("COL"), lines[i]
        out.append({k: np.array(lines[i + 1 + j].split(), dtype=np.float64)
                    for j, k in enumerate(ROWS)})
        i += 1 + len(ROWS)
    assert len(out) == len(cols)
    return out


cols = columns()
store = dict(names=np.array([c["name"] for c in cols]),
             lat=np.array([c["lat"] for c in cols]), albedo=np.array([c["alb"] for c in cols]),
             T_sfc=np.array([c["Ts"] for c in cols]), p_half=np.stack([c["ph"] for c in cols]),
             T=np.stack([c["T"] for c in cols]))
exps = experiments()
for tag, nml in exps.items():
    res = run(cols, nml)
    for k in ROWS:
        store[f"{tag}/{k}"] = np.stack([r[k] for r in res])
    # known-value self-checks
    ins = store[f"{tag}/scal"][:, 0]
    p2 = (1.0 - 3.0 * np.sin(store["lat"]) ** 2) / 4.0
    want = 0.25 * constants.S_0 * (1.0 + DEL_SOL * p2)
    assert np.max(np.abs(ins - want) / want) < 1e-14, (tag, ins, want)
    assert np.all(store[f"{tag}/lw_down"][:, 0] == 0.0), tag
    su = store[f"{tag}/sw_up"]
    assert np.all(su == su[:, :1]), tag
    if nml[-1] == 0.0:
        assert np.all(store[f"{tag}/tdt_solar"] == 0.0), tag
    store[f"{tag}/nml"] = np.array(nml)

meta = {
    "oracle": "Isca v1.0 src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90 B_FRIERSON "
              "(build copy: visibility line + memory-only allocate; see build.sh)",
    "repo": "https://github.com/ExeClim/Isca",
    "sha": subprocess.run(["git", "-C", ISCA, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "generator": "scripts/validate/oracle/gray_isca/{build.sh,make_fixture.py,drv_gray.F90,stubs_gray.F90}",
    "pstd_mks": "column p_s (matched subset)", "experiments": list(exps),
    "columns_source": str(COLS),
}
np.savez(OUT, meta=json.dumps(meta), **store)
print("FIXTURE_OK", OUT, len(cols), "columns", list(exps), meta["sha"])
