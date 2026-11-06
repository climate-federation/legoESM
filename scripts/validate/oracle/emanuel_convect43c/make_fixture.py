"""Generate the Emanuel CONVECT v4.3c oracle fixture by RUNNING convect43c.f.

Oracle: Kerry Emanuel's ``convect43c.f`` (CONVECT 4.3c, 20 May 2002), sha256
3eb50b9cf65817966ab89c56fbbdaf09fa546056d7f58b2ceb8564a4c254cc23, built by
``build.sh`` with legoESM constants (CPD CPV RV RD LV0 G from
``legoesm.constants``, CL = 2500 = ``EmanuelConfig.c_l_emanuel``), the oracle's
COEFFR = 1.0, COEFFS = 0.8 and cold-branch TLIFT, ``IPBL = 0``,
``MINORIG = 1``, ``NL = ND - 1``, DELT = 1800 s.  ``QS`` (a CONVECT input) is
legoESM's ``_oracle_qsat(T, p)``, as in the scheme.

Two runs per carried cloud-base mass flux (CBMF in = 0 and 0.01 kg/m^2/s):
``dd`` is the oracle as shipped (downdraught on); ``nodd`` takes the oracle's
own ``EP(INB) < 1e-4`` branch that skips the downdraught (IDDOFF = 1), which is
what ``enable_unsaturated_downdraft = False`` selects.

Columns: the 15 columns of the 2026-10-19 tier-2 audit (tier-1 ZM columns: 3
real MPAS AMIP day-1 L32 checkpoint columns + 12 constructed soundings on the
CAM6 L32 grid).

Stored per group: FT, FQ and (since #2266) FU, FV, plus the scalars.
``--columns`` may be the previous fixture (it holds the column inputs); #2266
regenerated from it, the audit ``columns.npz`` being lost, and every array
the earlier fixture held came back bit-identical.  A previous fixture's
``columns_sha256``/``columns`` (the original audit checksum) are kept and the
file actually read is recorded separately as ``columns_read`` (path, sha256).

Usage (CPU node; gfortran 8.5.0)::

    bash scripts/validate/oracle/emanuel_convect43c/build.sh <convect43c.f> <build dir>
    python scripts/validate/oracle/emanuel_convect43c/make_fixture.py --build <build dir> \\
        --columns <columns.npz> --oracle-src <convect43c.f> \\
        --out tests/atmosphere/oracle_fixtures/emanuel_convect43c.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DT = 1800.0
CBMF_IN = {"cb0": 0.0, "cbw": 0.01}
VARIANTS = {"nodd": 1, "dd": 0}  # IDDOFF
COEFFR, COEFFS, ICOLD = 1.0, 0.8, 1  # the oracle's own values
SHA256 = "3eb50b9cf65817966ab89c56fbbdaf09fa546056d7f58b2ceb8564a4c254cc23"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_input(path, c, qs, cbmf_in, iddoff, c_l):
    from legoesm import constants as k

    ncol, nd = c["T"].shape
    lines = [
        f"{k.c_pd!r} {k.c_pv!r} {c_l!r} {k.R_v!r} {k.R_d!r} {k.L_v!r} {k.g!r} "
        f"{COEFFR!r} {COEFFS!r} {iddoff} {ICOLD}",
        f"{ncol} {nd} {nd - 1} {DT!r}",
    ]
    for i in range(ncol):
        lines.append(repr(cbmf_in))
        for kk in range(nd - 1, -1, -1):  # surface-first
            lines.append(
                " ".join(
                    repr(float(x))
                    for x in (
                        c["T"][i, kk],
                        c["q"][i, kk],
                        qs[i, kk],
                        c["u"][i, kk],
                        c["v"][i, kk],
                        c["p_full"][i, kk] / 100.0,
                    )
                )
            )
        for kk in range(nd, -1, -1):
            lines.append(repr(float(c["p_half"][i, kk] / 100.0)))
    Path(path).write_text("\n".join(lines) + "\n")


def read_output(path):
    """Parse driver.f90 output (profiles surface-FIRST, indices 1-based)."""
    toks = Path(path).read_text().split("\n")
    ncol, nd = int(toks[0].split()[1]), int(toks[0].split()[2])
    pos, cols = 1, []
    for _ in range(ncol):
        h = toks[pos].split()
        pos += 1
        col = dict(
            zip(("iflag", "nk", "icb", "inb", "inb1", "ihmin", "stage"), (int(x) for x in h[2:]))
        )
        sc = [float(x) for x in toks[pos].split()]
        pos += 1
        col.update(cbmf=sc[0], precip=sc[1], frac=sc[8])
        prof = np.array([[float(x) for x in toks[pos + k].split()] for k in range(nd)])
        pos += nd
        col.update(FT=prof[:, 0], FQ=prof[:, 1], FU=prof[:, 2], FV=prof[:, 3], M=prof[:, 8],
                   MP=prof[:, 17])
        mats = []
        for _m in range(4):  # MENT, QENT, ELIJ, SIJ
            mats.append(np.array([[float(x) for x in toks[pos + k].split()] for k in range(nd)]))
            pos += nd
        col["MENT"] = mats[0]
        cols.append(col)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--oracle-src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert _sha256(a.oracle_src) == SHA256

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.convection._emanuel_mixing import _oracle_qsat
    from legoesm.atmosphere.physics.convection.config import EmanuelConfig

    c = {k: v for k, v in np.load(a.columns).items()}
    assert np.all(np.diff(c["p_full"], axis=1) > 0), "expected surface-LAST"
    qs = np.asarray(_oracle_qsat(jnp.asarray(c["T"]), jnp.asarray(c["p_full"])))
    c_l = EmanuelConfig().c_l_emanuel
    out = {k: c[k] for k in ("T", "q", "p_full", "p_half", "u", "v")}
    out["qs"] = qs
    out["names"] = np.array([str(n) for n in c["names"]])
    ncol, nd = c["T"].shape
    with tempfile.TemporaryDirectory() as d:
        for v, iddoff in VARIANTS.items():
            for tag, cb in CBMF_IN.items():
                fin, fout = Path(d) / f"{v}_{tag}.txt", Path(d) / f"{v}_{tag}.out"
                write_input(fin, c, qs, cb, iddoff, c_l)
                subprocess.run(
                    [str(Path(a.build) / "em_driver_r8.x"), str(fin), str(fout)], check=True
                )
                cols = read_output(fout)
                p = f"{v}_{tag}"
                for k in ("FT", "FQ", "FU", "FV"):  # stored surface-LAST
                    out[f"{p}_{k}"] = np.stack([col[k][::-1] for col in cols])
                for k in ("cbmf", "precip", "frac", "iflag", "nk", "icb", "inb", "inb1"):
                    out[f"{p}_{k}"] = np.array([col[k] for col in cols])
                # coverage: local-detrainment rows and active mixtures per column
                n_loc, n_mix = [], []
                for col in cols:
                    i0, i1 = col["icb"], col["inb"]  # rows ICB+1..INB (1-based)
                    rows = np.arange(i0, i1)  # 0-based ICB..INB-1
                    md = col["MENT"]
                    n_loc.append(int(np.sum(np.diag(md)[rows] > 0)) if col["stage"] >= 4 else 0)
                    off = md - np.diag(np.diag(md))
                    n_mix.append(int(np.sum(off > 0)))
                out[f"{p}_n_local_rows"] = np.array(n_loc)
                out[f"{p}_n_mixtures"] = np.array(n_mix)
                out[f"{p}_max_mp"] = np.array([float(np.max(col["MP"])) for col in cols])
    gf = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split(
        "\n"
    )[0]
    from legoesm import constants as k

    meta = dict(
        oracle="convect43c.f CONVECT v4.3c (Emanuel, 20 May 2002)",
        oracle_sha256=SHA256,
        cross_check="CliMT/climt 3ebe9427 climt/_lib/emanuel/convect43c.f90",
        gfortran=gf,
        flags="-O2 -fno-fast-math -fdefault-real-8 -fdefault-double-8",
        constants=dict(cpd=k.c_pd, cpv=k.c_pv, cl=c_l, rv=k.R_v, rd=k.R_d, lv0=k.L_v, g=k.g),
        coeffr=COEFFR,
        coeffs=COEFFS,
        ipbl=0,
        minorig=1,
        nl=nd - 1,
        dt=DT,
        cbmf_in=CBMF_IN,
        qs="legoesm _oracle_qsat(T, p)",
        columns_sha256=_sha256(a.columns),
        columns="oracle_audit/emanuel_convection/inputs/columns.npz (the audit copy of the tier-1 ZM columns)",
    )
    if "meta" in c:  # --columns is a previous fixture: keep the audit record, point at the file read
        prev = json.loads(str(c["meta"]))
        meta.update(columns_sha256=prev["columns_sha256"], columns=prev["columns"])
        meta["columns_read"] = dict(path=str(a.columns), sha256=_sha256(a.columns))
    out["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(a.out, **out)
    for p in (f"{v}_{t}" for v in VARIANTS for t in CBMF_IN):
        print(
            p,
            "iflag",
            out[f"{p}_iflag"].tolist(),
            "cbmf>0",
            int(np.sum(out[f"{p}_cbmf"] > 0)),
            "local rows",
            out[f"{p}_n_local_rows"].tolist(),
        )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
