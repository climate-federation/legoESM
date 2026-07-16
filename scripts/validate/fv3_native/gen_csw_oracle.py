#!/usr/bin/env python
"""Gen the duo c_sw one-step oracle, or pack the fixture.

Runs the port c_sw(..., duogrid=True) on identical inputs to the
authoritative symmetryclean c_sw DUO branch and certifies delpc/ptc/uc/vc
bit-exact.  Same canonical-writer + staging + no-regen-on-pack discipline as
the sibling gen scripts.

  gen_csw_oracle.py <W>          build inputs + staging + smoke the port.
  gen_csw_oracle.py <W> --pack   write the committed fixture npz.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

RES, NG = 12, 3

# (UPPER driver token, fixture npz key): state + every metric c_sw and its
# d2a2c_vect/divergence calls read.  hydrostatic run -> w is unused (zeros).
INPUT_FIELDS = (
    ("DELP", "delp"), ("PT", "pt"), ("W", "w"), ("U", "u"), ("V", "v"),
    ("COSA_S", "cosa_s"), ("RSIN2", "rsin2"), ("RAREA", "rarea"),
    ("DXA", "dxa"), ("DYA", "dya"),
    ("COSA_U", "cosa_u"), ("SINA_U", "sina_u"), ("RSIN_U", "rsin_u"),
    ("DY", "dy"), ("DXC", "dxc"), ("RDXC", "rdxc"),
    ("COSA_V", "cosa_v"), ("SINA_V", "sina_v"), ("RSIN_V", "rsin_v"),
    ("DX", "dx"), ("DYC", "dyc"), ("RDYC", "rdyc"),
    ("RAREA_C", "rarea_c"), ("FC", "fC"),
    ("SIN_SG", "sin_sg"), ("COS_SG", "cos_sg"),
)
FIXTURE_INPUTS = tuple(key for _name, key in INPUT_FIELDS)
OUTPUT_FIELDS = ("delpc", "ptc", "uc", "vc")


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


def serialize_csw_inputs(fields: dict, res: int = RES, ng: int = NG) -> bytes:
    lo = 1 - ng
    s = f"# res {res}\n# ng {ng}\n"
    for name, key in INPUT_FIELDS:
        s += _dump_field(name, np.asarray(fields[key]), lo)
    return s.encode()


def _fix_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "tests", "grids", "fixtures", "csw_oracle_c12.npz")


def _gen(work: str) -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "packages", "core"))
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
    fields = {
        "delp": np.nan_to_num(np.asarray(st["delp"], float), nan=0.0),
        "pt": np.nan_to_num(np.asarray(st["pt"], float), nan=0.0),
        "w": np.zeros_like(np.asarray(st["delp"], float)),
        "u": np.nan_to_num(np.asarray(st["u"], float), nan=0.0),
        "v": np.nan_to_num(np.asarray(st["v"], float), nan=0.0),
    }
    for k in ("cosa_s", "rsin2", "rarea", "dxa", "dya", "cosa_u", "sina_u",
              "rsin_u", "dy", "dxc", "rdxc", "cosa_v", "sina_v", "rsin_v",
              "dx", "dyc", "rdyc", "rarea_c", "fC", "sin_sg", "cos_sg"):
        fields[k] = np.asarray(gs[k])

    with open(f"{work}/csw_input.txt", "wb") as f:
        f.write(serialize_csw_inputs(fields))
    np.savez_compressed(f"{work}/staging.npz", **fields)
    c_sw(delp=fields["delp"], pt=fields["pt"], w=fields["w"], u=fields["u"],
         v=fields["v"], gs=gs, bd=bd, npx=RES + 1, npy=RES + 1, dt2=112.5,
         nord=1, hydrostatic=True, dord4=True, grid_type=0, duogrid=True)
    print("gen: inputs + staging written")


def _pack(work: str) -> None:
    stag = dict(np.load(f"{work}/staging.npz"))
    canon = serialize_csw_inputs(stag)
    on_disk = open(f"{work}/csw_input.txt", "rb").read()
    if canon != on_disk:
        raise SystemExit("csw_input.txt disagrees with staging.npz — "
                         "regenerate; refusing to pack a drifted fixture")
    inp_hash = hashlib.sha256(canon).hexdigest()

    lo = 1 - NG
    m_a = RES + 2 * NG
    m_b = m_a + 1
    shapes = {"delpc": (m_a, m_a), "ptc": (m_a, m_a),
              "uc": (m_b, m_a), "vc": (m_a, m_b)}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/csw_output.txt"):
        pp = line.split()
        outs[pp[0].lower()][int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])

    np.savez_compressed(
        _fix_dir(),
        **{k: outs[k] for k in OUTPUT_FIELDS},
        **{k: stag[k] for k in FIXTURE_INPUTS},
        res=RES, ng=NG, input_sha256=inp_hash,
        input_lineage="analytic-swcore-state; DUO c_sw (flagstruct%duogrid + "
        "dg%is_initialized) — d2a2c_vect_duo + divergence_corner_duo + simple "
        "upwind KE/vort + interior vorticity transport, corner fills skipped")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
