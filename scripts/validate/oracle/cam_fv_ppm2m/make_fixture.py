"""Write the CAM-FV ppm2m oracle fixture
(tests/grids/oracle_fixtures/cam_fv_ppm2m.npz) for the PPM vertical transport
tests (tests/grids/test_vertical_ppm.py).

Columns (24, levels top first):
  * c1: the 12 dumped C1 columns of the MG2 sedimentation fixture
    (tests/atmosphere/microphysics/oracle_fixtures/mg2_specified_nc_sed.npz,
    32 levels): theta, q_v, q_c, q_r;
  * rf02: the 12 saved states of hours 0.5 to 6 of the RF02 CAM-FV
    column-equivalent run (rf02_x24h_l32_dt1800_omega_cam_fv.npz, 9 levels):
    theta, q_v, q_c.
Theta takes kord 4, iv 1 (a kord 4 test field); the water species kord 7,
iv 0 (te_map.F90's constituents).

The model's temperature path is CAM-FV te_map's (te_map.F90:476-569):
``tv_star``, the virtual temperature times the Lagrangian pkz over the
Eulerian pkz (pt times pkz), mapped over log pe with kord 4, iv 1.  Per
column: ``tv_star`` (computed here in NumPy, independently of the model),
``tv_star_sub`` (thirds of each log pe layer) and ``tv_star_face`` (log pe
edges moved upstream to log(pe - F dt)), both through map1_ppm.

Two oracle problems per field, both through the unmodified map1_ppm:
  * ``sub``: each layer split into three equal sub-layers (target edges at
    1/3 and 2/3 of every layer); three sub-layer means fix the parabola, so
    this checks the reconstruction itself;
  * ``face``: the field mapped from its own edges to target edges moved
    upstream by ``F dt`` (``e - F dt``).  The target layer k then holds
    layer k less the mass that leaves through each face, and that mass is
    exactly the donor's parabola over the swept part of the donor, so
    ``q_k dp_k - remap_k (dp_k - dt (F_{k+1/2} - F_{k-1/2}))``, summed from
    the top, is ``dt F_{k+1/2} q_{k+1/2}`` with CAM's own swept-mass face
    value.  ``F`` is seeded, zero at the top and the surface, with face
    Courant numbers up to COURANT_MAX.

usage: python make_fixture.py --oracle <map1_ppm_oracle> --rf02 <npz> --out <npz>
       [--mg2-fixture <npz>] [--mapz <mapz_module.F90 used for the build>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from legoesm import constants

REPO = Path(__file__).resolve().parents[4]
MG2_FIXTURE = REPO / "tests/atmosphere/microphysics/oracle_fixtures/mg2_specified_nc_sed.npz"
DT = 300.0            # dycore step of the C1 decks [s]
COURANT_MAX = 0.05    # proposal section 5, face-value check
SEED = 20261102
WATER = {"c1": ("q_v", "q_c", "q_r"), "rf02": ("q_v", "q_c")}


def oracle(binary, src, dst, q, iv, kord):
    txt = (f"{src.size - 1} {dst.size - 1} {iv} {kord}\n"
           + " ".join(f"{x:.17e}" for x in src) + "\n"
           + " ".join(f"{x:.17e}" for x in dst) + "\n"
           + " ".join(f"{x:.17e}" for x in q) + "\n")
    out = subprocess.run([binary], input=txt, capture_output=True, text=True,
                         check=True).stdout
    return np.array([float(s) for s in out.split()])


def theta(T, p_full):
    return T * (constants.p_ref / p_full) ** constants.kappa


def pkz(pe):
    """CAM-FV's layer mean of p^kappa on edges pe (te_map multiplies pt by it)."""
    k = constants.kappa
    return (pe[1:] ** k - pe[:-1] ** k) / (k * (np.log(pe[1:]) - np.log(pe[:-1])))


def tv_star(T, q_v, pe, F):
    """T_v after the Lagrangian step: pt (= T_v / pkz(pe)) times pkz of the
    Lagrangian edges pe + F DT (te_map.F90, "Converting pt to t")."""
    zvir = 1.0 / constants.epsilon - 1.0
    return T * (1.0 + zvir * q_v) * pkz(pe + F * DT) / pkz(pe)


def columns(mg2_fixture, rf02):
    d = np.load(mg2_fixture)
    cols = []
    for i in range(d["dump_col_T"].shape[0]):
        cols.append(dict(group="c1", edges=d["dump_col_p_half"][i], T=d["dump_col_T"][i],
                         theta=theta(d["dump_col_T"][i], d["dump_col_p_full"][i]),
                         **{k: d[f"dump_col_{k}"][i] for k in WATER["c1"]}))
    r = np.load(rf02)
    for i, t in enumerate(r["time"]):
        if t / 3600.0 > 6.0:
            break
        cols.append(dict(group="rf02", edges=r["p_half"].copy(), T=r["T"][i],
                         theta=theta(r["T"][i], r["p_full"]),
                         **{k: r[k][i] for k in WATER["rf02"]}))
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--rf02", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mg2-fixture", type=Path, default=MG2_FIXTURE)
    ap.add_argument("--mapz", type=Path, default=None)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    fx, meta_cols = {"dt": np.array(DT)}, []
    for j, c in enumerate(columns(a.mg2_fixture, a.rf02)):
        e = c["edges"]
        dp = np.diff(e)
        km = dp.size
        F = np.zeros(km + 1)
        F[1:-1] = COURANT_MAX * rng.uniform(-1.0, 1.0, km - 1) * np.minimum(dp[:-1], dp[1:]) / DT
        sub = np.sort(np.concatenate([e, e[:-1] + dp / 3.0, e[:-1] + 2.0 * dp / 3.0]))
        fx[f"col{j}_edges"] = e
        fx[f"col{j}_F"] = F
        names = ("theta",) + WATER[c["group"]]
        for nm in names:
            iv, kord = (1, 4) if nm == "theta" else (0, 7)
            q = np.asarray(c[nm], dtype=float)
            fx[f"col{j}_{nm}"] = q
            fx[f"col{j}_{nm}_sub"] = oracle(a.oracle, e, sub, q, iv, kord)
            fx[f"col{j}_{nm}_face"] = oracle(a.oracle, e, e - F * DT, q, iv, kord)
        # te_map's temperature: tv_star mapped over log pe with kord 4, iv 1.
        tvs = tv_star(np.asarray(c["T"], float), np.asarray(c["q_v"], float), e, F)
        x = np.log(e)
        dx = np.diff(x)
        xsub = np.sort(np.concatenate([x, x[:-1] + dx / 3.0, x[:-1] + 2.0 * dx / 3.0]))
        fx[f"col{j}_T"] = np.asarray(c["T"], float)
        fx[f"col{j}_tv_star"] = tvs
        fx[f"col{j}_tv_star_sub"] = oracle(a.oracle, x, xsub, tvs, 1, 4)
        fx[f"col{j}_tv_star_face"] = oracle(a.oracle, x, np.log(e - F * DT), tvs, 1, 4)
        meta_cols.append(dict(group=c["group"], fields=list(names)))
    meta = dict(generator="scripts/validate/oracle/cam_fv_ppm2m/make_fixture.py",
                cam_tag="cam_cesm2_1_rel_60", routine="mapz_module.F90 map1_ppm/ppm2m",
                courant_max=COURANT_MAX, seed=SEED, columns=meta_cols,
                rf02=a.rf02.name, rf02_sha256=hashlib.sha256(a.rf02.read_bytes()).hexdigest(),
                mg2_fixture_sha256=hashlib.sha256(a.mg2_fixture.read_bytes()).hexdigest())
    if a.mapz is not None:
        meta["mapz_sha256"] = hashlib.sha256(a.mapz.read_bytes()).hexdigest()
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print(f"wrote {a.out}: {len(meta_cols)} columns")


if __name__ == "__main__":
    main()
