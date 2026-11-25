"""Fixture for CAM6 cirrus ice nucleation (tests/atmosphere/microphysics/oracle_fixtures/cam6_nucleate_ice.npz).

Oracle: ESCOMP/CAM cam_cesm2_1_rel_60 nucleate_ice.F90 and wv_sat_methods.F90 compiled UNMODIFIED (build.sh)
with driver.F90, which transcribes nucleate_ice_cam_calc's per-level loop for the modal aerosol.

Columns (L32, top first), from a legoESM A2 native dump (``--dump``, a ``feed_*.npz`` with T, q_v, q_c, q_i,
rad_in_n_i, rad_ast, rad_in_p_full and lat_deg):
  * 24 dumped columns spread in latitude, as dumped;
  * the same 24 with q_v reset to an ice relative humidity between 0.9 and 1.6 at every level colder than
    -5 C (so the nucleation thresholds, the regime switches and the pre-existing ice branch all occur);
and synthetic inputs the dump does not hold, of distinct vertical shapes: the sub-grid ice updraft wsubi
cycles through 0.001, 0.02, 0.2, 0.8 and 1.5 m/s by column (1.5 exercises the T < -40 C, W > 1 m/s branch);
modal aerosol numbers and coarse-mode masses (every third column without dust, every fifth without coarse
sulfate).  The CAM tropopause layer is legoESM's port of tropopause_findChemTrop on the column; top_lev is
the A2 deck's CLUBB top level.

usage: python make_fixture.py --build <dir with nucleate_ice_driver> --dump <feed_*.npz> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

N_DUMP = 24
WSUBI = (0.001, 0.02, 0.2, 0.8, 1.5)
TOP_LEV = 9          # 0-based; nucleati is not called above it


def build_columns(dump):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import cam_chem_tropopause_level
    from legoesm.thermo import saturation_vapor_pressure_ice_goff_gratch

    z = np.load(dump)
    lat = np.asarray(z["lat_deg"], np.float64)
    pick = np.argsort(lat)[np.linspace(0, lat.size - 1, N_DUMP).astype(int)]
    T = np.asarray(z["T"][pick], np.float64)
    q = np.asarray(z["q_v"][pick], np.float64)
    p = np.asarray(z["rad_in_p_full"][pick], np.float64)
    ph = np.asarray(z["rad_in_p_half"][pick], np.float64)
    qc = np.asarray(z["q_c"][pick], np.float64)
    qi = np.asarray(z["q_i"][pick], np.float64)
    ni = np.asarray(z["rad_in_n_i"][pick], np.float64)
    ast = np.clip(np.asarray(z["rad_ast"][pick], np.float64), 0.0, 1.0)
    esi = np.asarray(saturation_vapor_pressure_ice_goff_gratch(jnp.asarray(T)))
    qsi = constants.epsilon * esi / (p - (1.0 - constants.epsilon) * esi)
    rhi = 0.9 + 0.7 * ((np.arange(T.shape[1])[None, :] * 7 + np.arange(N_DUMP)[:, None] * 3) % 11) / 10.0
    q_sat = np.where(T < constants.T_freeze - 5.0, rhi * qsi, q)
    cols = dict(T=np.concatenate([T, T]), q=np.concatenate([q, q_sat]), p=np.concatenate([p, p]),
                ph=np.concatenate([ph, ph]), qc=np.concatenate([qc, qc]), qi=np.concatenate([qi, qi]),
                ni=np.concatenate([ni, ni]), ast=np.concatenate([ast, ast]),
                lat=np.concatenate([lat[pick], lat[pick]]))
    n, nl = cols["T"].shape
    s = cols["p"] / cols["ph"][:, -1:]
    idx = np.arange(n)[:, None]
    cols["wsubi"] = np.asarray(WSUBI)[idx % len(WSUBI)] * (0.6 + 0.4 * s)
    cols["num_aitken"] = 4.0e8 * (0.3 + s) * (1.0 + 0.1 * (idx % 7))                 # 1/kg
    cols["num_accum"] = 6.0e7 * (0.2 + s ** 2) * (1.0 + 0.05 * (idx % 5))
    cols["num_coarse"] = 2.0e4 * (0.1 + s ** 3)
    cols["coarse_dust"] = np.where(idx % 3 == 0, 0.0, 3.0e-9 * np.exp(-((cols["p"] - 5.0e4) / 2.5e4) ** 2))
    cols["coarse_nacl"] = 5.0e-9 * s ** 4
    cols["coarse_so4"] = np.where(idx % 5 == 0, 0.0, 4.0e-10 * (0.2 + s))
    cols["troplev"] = np.asarray(cam_chem_tropopause_level(
        jnp.asarray(cols["T"]), jnp.asarray(cols["p"]), jnp.asarray(cols["ph"]),
        jnp.asarray(np.radians(cols["lat"]))))
    return cols


FIELDS = ("T", "q", "p", "qc", "qi", "ni", "ast", "wsubi", "num_accum", "num_aitken", "num_coarse",
          "coarse_dust", "coarse_nacl", "coarse_so4")


def write_input(c, path):
    from legoesm import constants
    n, nl = c["T"].shape
    fmt = lambda a: " ".join(f"{x:.17e}" for x in np.ravel(a))  # noqa: E731
    L = [f"{n} {nl} {TOP_LEV + 1}",
         " ".join(f"{v:.17e}" for v in (constants.T_freeze, constants.R_d, constants.epsilon,
                                         constants.T_triple_cam, constants.T_boil_cam, math.pi))]
    for i in range(n):
        L.append(f"{int(c['troplev'][i]) + 1}")
        L += [fmt(c[f][i]) for f in FIELDS]
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = build_columns(a.dump)
    inp, outp = Path(a.build) / "nuc_in.txt", Path(a.build) / "nuc_out.txt"
    write_input(c, inp)
    subprocess.run([str(Path(a.build) / "nucleate_ice_driver"), str(inp), str(outp)], check=True)
    rows = np.loadtxt(outp)
    n = c["T"].shape[0]
    save = {k: c[k] for k in FIELDS + ("troplev", "lat")}
    save["oracle_naai"] = rows[0::3][:n]
    save["oracle_nihf"] = rows[1::3][:n]
    save["oracle_oso4"] = rows[2::3][:n]
    save["top_lev"] = np.asarray(TOP_LEV)
    save["meta"] = np.asarray(
        "oracle=ESCOMP/CAM cam_cesm2_1_rel_60 nucleate_ice.F90 md5 850753776aa8047db12d59053b04a78c, "
        "wv_sat_methods.F90 md5 cfc0f1a00e03fb0a57f6b188ed0d9a88; dump sha256 "
        + hashlib.sha256(Path(a.dump).read_bytes()).hexdigest()[:16])
    np.savez_compressed(a.out, **save)
    print("wrote", a.out, "columns", n, "nonzero naai points", int((save["oracle_naai"] > 0).sum()))


if __name__ == "__main__":
    main()
