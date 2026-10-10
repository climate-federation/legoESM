#!/usr/bin/env python3
"""Offline replay of NEMO's S-EOS density and hpg_sco for the one-level TSUNAMI.

Instrument (oracle-fidelity Rule 5): the literal statements of the compiled
TSUNAMI_OMIP_L1_RK3 ppsrc, written in numpy float64 in the compiled operand
order, fed only NEMO's recorded entry state.  Every constant is parsed from the
reference run's own ocean.output, never from the card.  The instrument is
validated by reproducing NEMO's recorded stp_2D right-hand side at kt = 1
(where u = v = 0, so the right-hand side is the pressure-gradient trend alone)
BIT FOR BIT; only then is its number quoted.

  eosbn2.f90 eos_insitu_t, np_seos:  zh = gdept_1d(jk)*(1+r3t(ji,jj,Knn))
  dynhpg.f90 hpg_sco, jk = 1:        zhpi, zuap (and the j analogues)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round2/"
    "oracle_tsunami_r2")
_KEYS = {"grav": r"grav", "rho0": r"rho0  ", "a0": "rn_a0", "b0": "rn_b0",
         "lambda1": "rn_lambda1", "lambda2": "rn_lambda2", "mu1": "rn_mu1",
         "mu2": "rn_mu2", "nu": "rn_nu", "T0": "rn_T0", "S0": "rn_S0"}


def parse_constants(ocean_output: Path) -> dict:
    text = ocean_output.read_text()
    out = {}
    for key, pat in _KEYS.items():
        m = re.search(pat + r"\s*=\s*([-0-9.EeDd+]+)", text)
        if m is None:
            raise ValueError(f"{ocean_output}: no {key!r} line")
        out[key] = float(m.group(1).replace("D", "E"))
    return out


def seos_rhd(T, S, r3t, c, gdept1):
    """eosbn2.f90 np_seos with the live stretched depth (qco)."""
    zt = T - c["T0"]
    zs = S - c["S0"]
    zh = gdept1 * (1.0 + r3t)
    zn = (-(c["a0"] * (1.0 + 0.5 * c["lambda1"] * zt + c["mu1"] * zh) * zt)
          + c["b0"] * (1.0 - 0.5 * c["lambda2"] * zs - c["mu2"] * zh) * zs
          - c["nu"] * zt * zs)
    return zn * (1.0 / c["rho0"])


def hpg_sco_k1(rhd, r3t, ssh, c, *, e3w1, gdept1, e1u, e2v):
    """dynhpg.f90 hpg_sco, jk = 1, periodic neighbours; returns (puu, pvv)."""
    z0 = -c["grav"] * 0.5
    out = []
    for ax, e in ((1, e1u), (0, e2v)):
        r1 = 1.0 / e
        n = lambda a: np.roll(a, -1, axis=ax)   # noqa: E731 -- (ji+1) / (jj+1)
        zh = z0 * r1 * ((e3w1 * (1.0 + n(r3t))) * n(rhd)
                        - (e3w1 * (1.0 + r3t)) * rhd)
        za = (-z0 * (n(rhd) + rhd)
              * (((gdept1 * (1.0 + n(r3t))) - n(ssh))
                 - ((gdept1 * (1.0 + r3t)) - ssh)) * r1)
        out.append(zh + za)
    return out[0], out[1]


def replay_from_record(g: dict, c: dict, mesh: dict) -> tuple:
    """g: interior record groups of one step; returns (puu, pvv) at Kbb."""
    rhd = seos_rhd(g["e_tn_k1_bb"], g["e_sn_k1_bb"], g["e_r3t_bb"], c,
                   mesh["gdept1"])
    return hpg_sco_k1(rhd, g["e_r3t_bb"], g["e_ssh_bb"], c, e3w1=mesh["e3w1"],
                      gdept1=mesh["gdept1"], e1u=mesh["e1u"], e2v=mesh["e2v"])


def mesh_from_mask(mesh_mask: Path) -> dict:
    import netCDF4
    with netCDF4.Dataset(mesh_mask) as h:
        return {"gdept1": float(h["gdept_1d"][0, 0]),
                "e3w1": float(h["e3w_1d"][0, 0]),
                "e1u": np.asarray(h["e1u"][0], dtype=np.float64),
                "e2v": np.asarray(h["e2v"][0], dtype=np.float64)}


def validate_kt1(root: Path, record_g: dict, *, perturb: str | None = None):
    c = parse_constants(root / "ref" / "ocean.output")
    if perturb:
        c[perturb] = c[perturb] * (1.0 + 1e-9)
    u, v = replay_from_record(record_g, c, mesh_from_mask(root / "p3" / "mesh_mask.nc"))
    ru, rv = record_g["b_uu_rhs_k1"], record_g["b_vv_rhs_k1"]
    return {"u_bit_identical": bool(np.array_equal(u, ru)),
            "v_bit_identical": bool(np.array_equal(v, rv)),
            "u_max_abs": float(np.max(np.abs(u - ru))),
            "v_max_abs": float(np.max(np.abs(v - rv))),
            "ref_max_abs": float(np.max(np.abs(ru))), "constants": c}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--perturb", choices=sorted(_KEYS),
                    help="plant: move one parsed constant by 1e-9 relative")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import nemo_testcase_l1_tsunami_ladder as lad
    cr, _ = lad._tools()
    g = lad.read_step(cr, a.root / "p3", 1)
    res = validate_kt1(a.root, g, perturb=a.perturb)
    print(json.dumps({k: v for k, v in res.items() if k != "constants"}))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    return 0 if res["u_bit_identical"] and res["v_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
