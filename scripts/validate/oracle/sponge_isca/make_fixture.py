"""Generate tests/atmosphere/oracle_fixtures/rayleigh_sponge_isca.npz by RUNNING Isca's
Polvani-Kushner Rayleigh sponge.

Oracle: ExeClim/Isca tag v1.0 (5868cfd863b41823a3eeb30754ce06feba7528ec)
``src/atmos_param/damping_driver/damping_driver.f90`` compiled UNMODIFIED with minimal
FMS stubs (``stubs_damp.F90``), called through its public ``damping_driver_init`` /
``damping_driver`` (``drv_damp.F90``) with ``do_rayleigh = do_conserve_energy = T``
and every other option off.  Compared: ``udt`` / ``vdt`` / ``tdt`` (subroutine
``rayleigh``: ``-rfactr ((p_sb - p)/p_sb)^2 (u, v)`` on levels ``1..nlev_rayfric``
with ``p < p_sb``, and the time-centred dissipative heating).

Namelist: ``trayfric = 1/sponge_k`` [s] (Isca's positive branch), ``sponge_pbottom`` =
the set's value.  ``pref`` = per column ``p_full/p_s * constants.p_atm_std`` then
``p_atm_std`` (Isca's full-level reference profile at PSTD_MKS, one init per column).
Constants CP_AIR, GRAV = legoESM's, written into the oracle input.

Two column sets (vote W5):
  L32: the 13 CAM L32 columns of the tier-2 GWD audit (oracle_audit/
       simple_gwd_hines_lindzen_rayleigh_spectral/inputs/cols.npz), sponge_pbottom
       1000 Pa (fixture value: their top full level is 364 Pa, so the 50 Pa default
       never acts there).
  L60: 6 synthetic log-pressure columns, top interface 1 Pa, at the 50 Pa default.
dt 1800 s, sponge_k = RayleighConfig default.

Self-checks (known values, every run): zero tendency where p >= p_sb or level >
nlev_rayfric; elsewhere |udt + rfactr ((p_sb-p)/p_sb)^2 u| <= 1e-14 rfactr |u| (same
for v); tdt == -((u + dt/2 udt) udt + (v + dt/2 vdt) vdt)/cp to 1e-14; each set has
active and inactive levels.

usage (compute node):
    bash build.sh <Isca@v1.0> <build dir>
    python make_fixture.py <build dir>/drv_damp <Isca@v1.0> <cols.npz> <out.npz>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig

ORACLE, ISCA, COLS, OUT = sys.argv[1:5]
DT = 1800.0
ROWS = ("udt", "vdt", "tdt")
KEYS = ("p_half", "p_full", "u", "v", "T", "z_half", "z_full", "rho", "lat")


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def synthetic_l60():
    nlev, ps = 60, np.array([1.0e5, 1.013e5, 9.5e4, 8.0e4, 7.0e4, 5.5e4])
    k = np.arange(nlev + 1)
    out = {key: [] for key in KEYS}
    for c, p_s in enumerate(ps):
        ph = np.exp(np.log(1.0) + (np.log(p_s) - np.log(1.0)) * k / nlev)
        pf = np.sqrt(ph[:-1] * ph[1:])
        j = np.arange(nlev)
        T = 240.0 + 25.0 * np.sin(0.21 * j + 0.7 * c)
        u = 45.0 * np.sin(0.17 * j + 1.3 * c) + 5.0
        v = 18.0 * np.cos(0.23 * j + 0.4 * c)
        dlnp = np.log(ph[1:]) - np.log(ph[:-1])
        dz = constants.R_d * T * dlnp / constants.g
        zh = np.concatenate([np.cumsum(dz[::-1])[::-1], [0.0]])
        zf = 0.5 * (zh[:-1] + zh[1:])
        for key, val in zip(KEYS, (ph, pf, u, v, T, zh, zf, pf / (constants.R_d * T),
                                    np.deg2rad(-60.0 + 24.0 * c))):
            out[key].append(val)
    return {key: np.stack(v) for key, v in out.items()}


def run_set(d, p_sb, cfg):
    ncol, nlev = d["p_full"].shape
    s = [f"{ncol} {nlev}", f17([1.0 / cfg.sponge_k, p_sb, DT]),
         f17([constants.c_pd, constants.g])]
    for i in range(ncol):
        pref = d["p_full"][i] / max(d["p_half"][i, -1], 1.0) * constants.p_atm_std
        s += [f17(np.append(pref, constants.p_atm_std)), f17(d["p_half"][i])]
        s += [f17(d[k][i]) for k in ("p_full", "u", "v", "T")]
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "damp_in.txt", Path(td) / "damp_out.txt"
        fin.write_text("\n".join(s) + "\n")
        r = subprocess.run([ORACLE, str(fin), str(fout)], cwd=td, capture_output=True,
                           text=True, check=True)
        assert "DAMP_DRV_OK" in r.stdout, r.stdout + r.stderr
        lines = fout.read_text().splitlines()
    out = {k: [] for k in ROWS}
    for i in range(ncol):
        assert lines[i * 4].startswith("COL"), lines[i * 4]
        for j, k in enumerate(ROWS):
            out[k].append(np.array(lines[i * 4 + 1 + j].split(), dtype=np.float64))
    out = {k: np.stack(v) for k, v in out.items()}
    assert out["udt"].shape == (ncol, nlev)
    # known-value self-checks
    rf = cfg.sponge_k
    pref = d["p_full"] / np.maximum(d["p_half"][:, -1:], 1.0) * constants.p_atm_std
    cap = np.argmin(np.abs(pref - 2.0 * p_sb), axis=1)[:, None]
    act = (np.arange(nlev)[None, :] <= cap) & (d["p_full"] < p_sb)
    assert act.any() and (~act).any()
    fac = rf * ((p_sb - d["p_full"]) / p_sb) ** 2
    for k, w in (("udt", d["u"]), ("vdt", d["v"])):
        assert np.all(out[k][~act] == 0.0), k
        nz = act & (w != 0)
        assert np.max(np.abs(out[k][nz] + fac[nz] * w[nz]) / (rf * np.abs(w[nz]))) < 1e-14, k
    want_t = -((d["u"] + 0.5 * DT * out["udt"]) * out["udt"]
               + (d["v"] + 0.5 * DT * out["vdt"]) * out["vdt"]) / constants.c_pd
    assert np.all(out["tdt"][~act] == 0.0)
    assert np.max(np.abs(out["tdt"] - want_t)) <= 1e-14 * np.max(np.abs(want_t))
    return out


cfg = RayleighConfig()
l32 = {k: np.asarray(v, np.float64) for k, v in np.load(COLS, allow_pickle=True).items()
       if k in KEYS}
sets = {"l32": (l32, 1000.0), "l60": (synthetic_l60(), cfg.sponge_pbottom_Pa)}
save = {}
for name, (d, p_sb) in sets.items():
    o = run_set(d, p_sb, cfg)
    for k in KEYS:
        save[f"{name}_{k}"] = d[k]
    for k in ROWS:
        save[f"{name}_{k}"] = o[k]
    save[f"{name}_p_sb"] = np.float64(p_sb)
meta = {
    "oracle": "Isca v1.0 src/atmos_param/damping_driver/damping_driver.f90 (unmodified) "
              "damping_driver -> rayleigh (do_rayleigh, do_conserve_energy)",
    "repo": "https://github.com/ExeClim/Isca",
    "sha": subprocess.run(["git", "-C", ISCA, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "generator": "scripts/validate/oracle/sponge_isca/{build.sh,make_fixture.py,drv_damp.F90,stubs_damp.F90}",
    "sponge_k_per_s": cfg.sponge_k, "dt": DT, "sets": list(sets), "columns_source": str(COLS),
}
np.savez(OUT, meta=json.dumps(meta), **save)
print("FIXTURE_OK", OUT, meta["sha"])
