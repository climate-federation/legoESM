"""Below-cloud rain evaporation: shared Bechtold leaf and faithful IFS chain
against the OpenIFS single-column oracle (cuflxn.F90:443-500).

usage: compare_subcloud_evap.py <driver.x> <work dir> <port_deep.npz>

Run after ``run.sh <build> port_deep``.  <port_deep.npz> is make_fixture.py's
output (oracle with the source defaults, LMFDSNOW=.TRUE.); this script reruns
the oracle on <work dir>/port_deep_inputs.txt with ``LMFDSNOW=.FALSE.``
(detrained precipitation off).  Prints, per oracle-evaporating level:
  leaf   : the leaf on the oracle's own production and PQSEN (formula check)
  chain  : the faithful chain's PDMFUP and dT_dt on the same column
Numbers only; provenance (git SHA, input md5) on the first lines.
"""
import hashlib
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    bechtold_convection, ifs_subcloud_rain_evaporation)
from legoesm.atmosphere.physics.convection.config import BechtoldConfig  # noqa: E402
from legoesm.atmosphere.physics.convection import _ifs_faithful as F  # noqa: E402
from tests.unit.test_ifs_faithful import _deep_column, _kwargs  # noqa: E402

RMFCFL = 3.0


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _read_dump(path):
    hdr, rows = None, {}
    for line in open(path):
        s = line.split()
        if s and s[0] == "LEV":
            hdr = s[1:]
        elif s and s[0].isdigit():
            rows[int(s[0])] = [float(x) for x in s[1:]]
    return {h: np.array([rows[k][i] for k in sorted(rows)]) for i, h in enumerate(hdr)}


def leaf_on_oracle(z):
    """Leaf fed the oracle's production and PQSEN; returns PDMFUP increments."""
    q, ph = z["in_q"], z["in_ph"]
    g = constants.g
    dp = ph[1:] - ph[:-1]
    o = z["oracle_dmfup_flx"]
    prod = np.where(o > 0, o, 0.0)
    kcb, kct = int(z["oracle_kcbot"]) - 1, int(z["oracle_kctop"]) - 1
    qs = z["oracle_qsen"]
    lev = np.arange(q.size)
    ev, _, _ = ifs_subcloud_rain_evaporation(
        jnp.asarray(q)[None], jnp.asarray(qs)[None], jnp.asarray(ph)[None],
        jnp.asarray(dp)[None], jnp.asarray(prod * g / dp)[None],
        jnp.asarray((lev >= kcb).astype(float))[None],
        jnp.asarray([q[kcb] / qs[kcb]]), jnp.asarray([q[kct] / qs[kct]]),
        jnp.asarray([float(int(z["oracle_ktype"]) == 1)]), float(z["in_dt"]), rmfcfl=RMFCFL)
    return -np.asarray(ev)[0] * dp / g


def chain(nlev):
    """Faithful chain on the port_deep column in the source's no-supply mode
    (ifs_shallow_nosupply="first_guess", what the oracle runs); returns
    (PDMFUP into CUDTDQN, dT_dt)."""
    T, q, pf, ph = [a.astype(jnp.float64) for a in _deep_column(nlev=nlev)]
    cap = {}
    orig = F.ifs_convective_tendencies

    def spy(*a, **k):
        cap["pdmfup"] = np.asarray(a[-3])[0]
        return orig(*a, **k)

    F.ifs_convective_tendencies = spy
    try:
        out = bechtold_convection(T, q, pf, ph, config=BechtoldConfig(use_ifs_ascent=True, ifs_shallow_nosupply="first_guess"),
                                  **_kwargs(*T.shape))[0]
    finally:
        F.ifs_convective_tendencies = orig
    return cap["pdmfup"], np.asarray(out.dT_dt)[0]


def main(driver, work, npz):
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"repo_sha={sha} npz={npz} md5={_md5(npz)}")
    z = np.load(npz)
    print(f"npz repo_sha={z['repo_sha']}")
    o_up, o_dT = z["oracle_dmfup_flx"], z["oracle_dT_dt"]
    ev_lev = np.where(o_up < 0)[0]
    lf = leaf_on_oracle(z)
    up, dT = chain(int(z["oracle_klev"]))
    fin = os.path.join(work, "port_deep_inputs.txt")
    nodsnow = os.path.join(work, "port_deep_nodsnow_dump.txt")
    r = subprocess.run([driver, fin, nodsnow, "LMFDSNOW=.FALSE."], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0 and "DRIVER_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    print(f"inputs md5={_md5(fin)} nodsnow dump md5={_md5(nodsnow)} driver md5={_md5(driver)}")
    nd = _read_dump(nodsnow)
    print("lev(1b) oracle_dmfup  leaf_dmfup  chain_dmfup  oracle_dT  chain_dT  [nodsnow_dmfup nodsnow_dT] (dT K/day)")
    for k in ev_lev:
        row = (f"{k + 1:4d} {o_up[k]: .10e} {lf[k]: .10e} {up[k]: .10e} "
               f"{o_dT[k] * 86400: .6f} {dT[k] * 86400: .6f}")
        row += f"  {nd['dmfup_flx'][k]: .10e} {nd['dT_dt'][k] * 86400: .6f}"
        print(row)
    for k in ev_lev:
        print(f"lev {k + 1}: leaf/oracle-1 {lf[k] / o_up[k] - 1: .6e}  chain/oracle-1 {up[k] / o_up[k] - 1: .6e}"
              + f"  leaf/nodsnow-1 {lf[k] / nd['dmfup_flx'][k] - 1: .6e}")
    print(f"chain levels with PDMFUP < 0: {(np.where(up < 0)[0] + 1).tolist()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
