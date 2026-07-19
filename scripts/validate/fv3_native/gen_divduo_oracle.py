#!/usr/bin/env python
"""Gen divergence_corner_duo oracle inputs, or pack the fixture.

Two modes (no regeneration on pack — the fixture inputs, the hash, and the
Fortran output all come from ONE generation, so "bit-exact on identical
inputs" is actually enforced; codex p4c r2 P1):

  gen_divduo_oracle.py <W>          build inputs + run the python port;
                                    write divduo_input.txt (the Fortran
                                    driver reads it) + staging.npz.
  gen_divduo_oracle.py <W> --pack   read the SAME staging.npz + the
                                    first-run divduo_input.txt (hash it,
                                    and reject it if it disagrees with the
                                    staged arrays) + the driver's
                                    divduo_output.txt, and write the
                                    committed fixture npz.

The byte layout of ``divduo_input.txt`` is produced by ONE canonical
writer, :func:`serialize_divduo_inputs`, which the phase-4c test imports
and re-runs on the fixture's STORED arrays to prove ``input_sha256`` is the
hash of exactly those bytes (codex p4c r3 P1 — provenance ENFORCED, not
merely recorded).  Importing this module has no side effects.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

RES, NG = 12, 3

# canonical field order the Fortran driver reads AND the fixture stores;
# (UPPER driver token, fixture npz key).  ONE order, ONE writer.
INPUT_FIELDS = (
    ("RAREA_C", "rarea_c"), ("DXC", "dxc"), ("DYC", "dyc"),
    ("SIN_SG", "sin_sg"), ("COS_SG", "cos_sg"),
    ("U", "u"), ("V", "v"), ("UA", "ua"), ("VA", "va"),
)


def _dump_field(name: str, a: np.ndarray, lo: int) -> str:
    ni, nj = a.shape[:2]
    out = []
    for i in range(ni):
        for j in range(nj):
            if a.ndim == 2:
                out.append(f"{name} {i + lo} {j + lo} {a[i, j]:.17e}\n")
            else:
                for k in range(a.shape[2]):
                    out.append(f"{name} {i + lo} {j + lo} {k + 1} "
                               f"{a[i, j, k]:.17e}\n")
    return "".join(out)


def serialize_divduo_inputs(fields: dict, res: int = RES,
                            ng: int = NG) -> bytes:
    """The exact ``divduo_input.txt`` byte stream for ``fields`` (a dict
    keyed by the lowercase npz names in :data:`INPUT_FIELDS`).

    The Fortran driver reads these bytes and ``input_sha256`` is their
    SHA-256, so gen and the test both call THIS function — gen on the
    freshly built arrays, the test on the fixture's stored arrays — and
    get identical bytes iff the stored arrays are the ones hashed.
    """
    lo = 1 - ng
    s = f"# res {res}\n# ng {ng}\n"
    for name, key in INPUT_FIELDS:
        s += _dump_field(name, np.asarray(fields[key]), lo)
    return s.encode()


def _fix_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "tests", "grids", "fixtures", "divduo_oracle_c12.npz")


def _gen(work: str) -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "packages", "core"))
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    gs = build_fv3_native_gridstruct(RES, NG, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(RES, NG)
    csw = c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
               u=st["u"], v=st["v"], gs=gs, bd=bd, npx=RES + 1, npy=RES + 1,
               dt2=112.5, nord=1, hydrostatic=True, dord4=True, grid_type=0)
    fields = {
        "u": np.nan_to_num(np.asarray(st["u"], float), nan=0.0),
        "v": np.nan_to_num(np.asarray(st["v"], float), nan=0.0),
        "ua": np.nan_to_num(np.asarray(csw["ua"], float), nan=0.0),
        "va": np.nan_to_num(np.asarray(csw["va"], float), nan=0.0),
        "rarea_c": np.asarray(gs["rarea_c"]), "dxc": np.asarray(gs["dxc"]),
        "dyc": np.asarray(gs["dyc"]), "sin_sg": np.asarray(gs["sin_sg"]),
        "cos_sg": np.asarray(gs["cos_sg"]),
    }

    with open(f"{work}/divduo_input.txt", "wb") as f:
        f.write(serialize_divduo_inputs(fields))
    # stage the EXACT arrays the fixture will carry (so --pack does NOT
    # regenerate them; the test re-serialises these to check the hash)
    np.savez_compressed(f"{work}/staging.npz", **fields)
    # smoke: the python port runs (its output is re-derived from the
    # STORED inputs at test time, so nothing else needs saving here)
    divergence_corner_duo(fields["u"], fields["v"], fields["ua"],
                          fields["va"], gs, bd, RES + 1, RES + 1, grid_type=0)
    print("gen: inputs + staging written")


def _pack(work: str) -> None:
    stag = dict(np.load(f"{work}/staging.npz"))
    # canonical bytes from the STAGED arrays; must equal the input.txt the
    # Fortran driver actually read (else staging/text drifted) — codex r3
    canon = serialize_divduo_inputs(stag)
    on_disk = open(f"{work}/divduo_input.txt", "rb").read()
    if canon != on_disk:
        raise SystemExit("divduo_input.txt disagrees with staging.npz — "
                         "regenerate; refusing to pack a drifted fixture")
    inp_hash = hashlib.sha256(canon).hexdigest()

    lo = 1 - NG
    m_b = RES + 2 * NG + 1
    fd = np.full((m_b, m_b), np.nan)
    for line in open(f"{work}/divduo_output.txt"):
        pp = line.split()
        fd[int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])

    np.savez_compressed(
        _fix_dir(), divg_d=fd,
        u=stag["u"], v=stag["v"], ua=stag["ua"], va=stag["va"],
        rarea_c=stag["rarea_c"], dxc=stag["dxc"], dyc=stag["dyc"],
        sin_sg=stag["sin_sg"], cos_sg=stag["cos_sg"],
        res=RES, ng=NG, input_sha256=inp_hash,
        input_lineage="plain-c_sw ua/va (ROUTINE-TRANSLATION gate; the "
        "full DUO pipeline needs dg-initialized d2a2c_vect ua/va)")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
