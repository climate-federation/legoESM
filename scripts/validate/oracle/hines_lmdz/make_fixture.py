"""Generate tests/atmosphere/oracle_fixtures/hines_lmdz.npz by RUNNING LMDZ's Hines scheme.

Oracle: LMDZ6 trunk r5891 (tarball 20251128) ``libf/phylmd/hines_gwd.f90`` and
``yoegwd_mod_h.f90`` compiled UNMODIFIED, with ``dimphy`` / ``yomcst_mod_h``
stand-ins (``stubs_hines.F90``), called through ``hines_gwd`` (``drv_hines.F90``)
exactly as ``physiq_mod.F90`` calls it.  Compared outputs: ``d_u_hin / dtime``,
``d_v_hin / dtime`` (stored top-down, legoESM order) and the top fluxes
``zustrhi`` / ``zvstrhi`` (recorded).

Constants R_d, g, c_pd, pi = legoESM's, written into the oracle input.  Columns =
the 13 CAM L32 columns of the tier-2 GWD audit
(oracle_audit/simple_gwd_hines_lindzen_rayleigh_spectral/inputs/cols.npz, lat in
degrees) plus one constructed 60-level column whose top (1e-3 Pa) lies above
LMDZ's 105 km cutoff, so the exponential-decay branch runs (V49).  dt = 1800 s.

Self-checks (known values, every run): on the 13 audit columns the column sum of
rho*du*dz over LMDZ's own layers is zero to 1e-12 of its largest term (the flux
divergence telescopes); the deep column has levels above 105 km.

usage (compute node):
    bash build.sh <LMDZ libf/phylmd> <build dir>
    python make_fixture.py <build dir>/drv_hines <LMDZ libf/phylmd> <cols.npz> <out.npz>
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants

ORACLE, LMDZ, COLS, OUT = sys.argv[1:5]
DT = 1800.0


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def deep_column(nlev=60):
    """Constructed column to 1e-3 Pa (~130 km): T 288 K surface, 6.5 K/km lapse to
    216 K, warming above; a tropospheric jet over a stratospheric easterly."""
    p_half = np.geomspace(1.0e-3, 1.0e5, nlev + 1)
    p_full = np.sqrt(p_half[1:] * p_half[:-1])
    zk = 7.0 * np.log(1.0e5 / p_full)                       # rough height [km]
    T = np.where(zk < 11.0, 288.0 - 6.5 * zk, 216.0 + 1.5 * np.clip(zk - 20.0, 0.0, None))
    u = 25.0 * np.exp(-((zk - 12.0) / 5.0) ** 2) - 30.0 * np.exp(-((zk - 60.0) / 15.0) ** 2)
    v = 5.0 * np.sin(zk / 10.0)
    return dict(p_half=p_half, p_full=p_full, T=T, u=u, v=v, lat=10.0)


d = np.load(COLS, allow_pickle=True)
cols = [dict(p_half=d["p_half"][i], p_full=d["p_full"][i], T=d["T"][i], u=d["u"][i],
             v=d["v"][i], lat=float(d["lat"][i])) for i in range(len(d["names"]))]
cols.append(deep_column())
names = list(d["names"]) + ["deep_130km"]
ncol = len(cols)
nlevs = {c["p_full"].size for c in cols}

out = {k: [] for k in ("du_dt", "dv_dt", "flux_top")}
for c in cols:                    # one oracle call per column (two level counts)
    nlev = c["p_full"].size
    s = ["1 %d" % nlev, f17([DT, constants.R_d, constants.g, constants.c_pd, np.pi]),
         f17([c["lat"]])]
    s += [f17(c["p_half"][::-1]), f17(c["p_full"][::-1])]
    s += [f17(c[k][::-1]) for k in ("T", "u", "v")]
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "hines_in.txt", Path(td) / "hines_out.txt"
        fin.write_text("\n".join(s) + "\n")
        r = subprocess.run([ORACLE, str(fin), str(fout)], cwd=td, capture_output=True,
                           text=True, check=True)
        assert "HINES_DRV_OK" in r.stdout, r.stdout + r.stderr
        lines = fout.read_text().splitlines()
    assert lines[0].startswith("COL"), lines[0]
    out["du_dt"].append(np.array(lines[1].split(), np.float64)[::-1])
    out["dv_dt"].append(np.array(lines[2].split(), np.float64)[::-1])
    out["flux_top"].append(np.array(lines[3].split(), np.float64))

# known-value self-checks: the flux divergence telescopes, so the column sum of
# drag * rho * dz over LMDZ's own layers (hines_flux dendz) is zero where the
# 105 km decay does not apply (the 13 audit columns)
for c, du, dv in zip(cols[:-1], out["du_dt"], out["dv_dt"]):
    ps = c["p_half"][-1]
    sg, dsg = c["p_full"] / ps, np.diff(c["p_half"]) / ps
    hscal = constants.R_d * c["T"] / constants.g
    rho = sg * ps / (constants.g * hscal)
    alt = np.append(np.cumsum((hscal * dsg / sg)[-2::-1])[::-1], 0.0)
    dz = np.append(alt[0] - alt[1], alt[:-1] - alt[1:])
    for x in (du, dv):
        t = rho * dz * x
        assert abs(np.sum(t)) <= 1e-12 * np.max(np.abs(t)), np.sum(t)
    assert np.abs(du).max() > 0.0
alt_top = 7.0e3 * np.log(1.0e5 / cols[-1]["p_full"][0])
assert alt_top > 105.0e3, alt_top

sha = {f: hashlib.sha256((Path(LMDZ) / f).read_bytes()).hexdigest()
       for f in ("hines_gwd.f90", "yoegwd_mod_h.f90")}
meta = {
    "oracle": "LMDZ6 trunk r5891 (tarball 20251128) libf/phylmd/hines_gwd.f90 (unmodified)",
    "source": "https://lmdz.lmd.jussieu.fr (CeCILL)", "sha256": sha,
    "generator": "scripts/validate/oracle/hines_lmdz/{build.sh,make_fixture.py,drv_hines.F90,stubs_hines.F90}",
    "dt": DT, "columns_source": str(COLS), "lat_units": "degrees",
    "k_launch_rule": "nlev - 3 (LMDZ iplev = klev - 2, 1-based bottom-up -> 0-based top-down)",
}
# ragged (32- and 60-level) columns stored flat, split by "nlev" (no pickled objects)
flat = lambda arrs: np.concatenate([np.asarray(a, np.float64) for a in arrs])
np.savez(OUT, meta=json.dumps(meta), names=np.array(names),
         nlev=np.array([c["p_full"].size for c in cols]),
         lat=np.array([c["lat"] for c in cols]),
         **{k: flat(c[k] for c in cols) for k in ("p_half", "p_full", "T", "u", "v")},
         **{k: flat(v) for k, v in out.items()})
print("FIXTURE_OK", OUT, ncol, sorted(nlevs), sha)
