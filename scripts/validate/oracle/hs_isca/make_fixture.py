"""Generate tests/atmosphere/oracle_fixtures/rayleigh_hs_isca.npz by RUNNING Isca's
Held-Suarez forcing.

Oracle: ExeClim/Isca tag v1.0 (5868cfd863b41823a3eeb30754ce06feba7528ec)
``src/atmos_param/hs_forcing/hs_forcing.F90`` compiled UNMODIFIED with minimal FMS
stubs (``stubs_hs.F90``), called through its public ``hs_forcing_init`` /
``hs_forcing`` (``drv_hs.F90``).  The compared output is the HS94 low-level Rayleigh
friction ``udt`` / ``vdt`` (Isca ``rayleigh_damping``: ``-kf (sigma - sigma_b)/(1 -
sigma_b) u`` for ``sigma_b < sigma <= 1``).  ``tdt`` (Newtonian damping + dissipative
heating) is recorded, not compared.

Namelist: ``sigma_b`` and ``kf`` = legoESM ``RayleighConfig`` defaults, ``kf`` in 1/s
(Isca's positive-kf branch, so ``vkf`` is our number unchanged).  Constants KAPPA,
CP_AIR, GRAV = legoESM's, written into the oracle input.  Columns = the 13 CAM L32
columns of the tier-2 GWD audit (oracle_audit/simple_gwd_hines_lindzen_rayleigh_spectral
/inputs/cols.npz: 12 tier-1 columns incl. 3 real MPAS AMIP + a rest column), dt 1800 s.

Self-checks (known values, every run): udt = vdt = 0 exactly where
p_full/p_s <= sigma_b; elsewhere |udt - (-kf (sigma - sigma_b)/(1 - sigma_b)) u| <= 1e-14 kf |u| (same for v).

usage (compute node):
    bash build.sh <Isca@v1.0> <build dir>
    python make_fixture.py <build dir>/drv_hs <Isca@v1.0> <cols.npz> <out.npz>
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
DT = 1800.0    # tier-2 GWD audit time step (unused by udt / vdt)
ROWS = ("udt", "vdt", "tdt")


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


d = np.load(COLS, allow_pickle=True)
cfg = RayleighConfig()
ncol, nlev = d["p_full"].shape
s = [f"{ncol} {nlev}", f17([cfg.sigma_b, cfg.k_max, DT]),
     f17([constants.kappa, constants.c_pd, constants.g])]
for i in range(ncol):
    s += [f17([d["lat"][i], 0.0]), f17(d["p_half"][i])]
    s += [f17(d[k][i]) for k in ("p_full", "u", "v", "T")]
with tempfile.TemporaryDirectory() as td:
    fin, fout = Path(td) / "hs_in.txt", Path(td) / "hs_out.txt"
    fin.write_text("\n".join(s) + "\n")
    r = subprocess.run([ORACLE, str(fin), str(fout)], cwd=td, capture_output=True,
                       text=True, check=True)
    assert "HS_DRV_OK" in r.stdout, r.stdout + r.stderr
    lines = fout.read_text().splitlines()
out = {k: [] for k in ROWS}
for i in range(ncol):
    assert lines[i * 4].startswith("COL"), lines[i * 4]
    for j, k in enumerate(ROWS):
        out[k].append(np.array(lines[i * 4 + 1 + j].split(), dtype=np.float64))
out = {k: np.stack(v) for k, v in out.items()}
assert out["udt"].shape == (ncol, nlev)

# known-value self-checks
sig = d["p_full"] / d["p_half"][:, -1:]
inside = (sig > cfg.sigma_b) & (sig <= 1.0)
assert inside.any() and (~inside).any()
for k, w in (("udt", d["u"]), ("vdt", d["v"])):
    assert np.all(out[k][~inside] == 0.0), k
    nz = inside & (w != 0)
    want = -cfg.k_max * (sig[nz] - cfg.sigma_b) / (1.0 - cfg.sigma_b)
    assert np.max(np.abs(out[k][nz] - want * w[nz]) / (cfg.k_max * np.abs(w[nz]))) < 1e-14, k

meta = {
    "oracle": "Isca v1.0 src/atmos_param/hs_forcing/hs_forcing.F90 (unmodified) hs_forcing "
              "-> rayleigh_damping",
    "repo": "https://github.com/ExeClim/Isca",
    "sha": subprocess.run(["git", "-C", ISCA, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "generator": "scripts/validate/oracle/hs_isca/{build.sh,make_fixture.py,drv_hs.F90,stubs_hs.F90}",
    "sigma_b": cfg.sigma_b, "kf_per_s": cfg.k_max, "dt": DT, "columns_source": str(COLS),
}
np.savez(OUT, meta=json.dumps(meta), names=d["names"], lat=d["lat"], p_half=d["p_half"],
         p_full=d["p_full"], u=d["u"], v=d["v"], T=d["T"], z_full=d["z_full"],
         z_half=d["z_half"], rho=d["rho"], **out)
print("FIXTURE_OK", OUT, out["udt"].shape, meta["sha"])
