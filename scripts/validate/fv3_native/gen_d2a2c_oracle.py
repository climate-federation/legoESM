#!/usr/bin/env python
"""Gen duo d2a2c_vect oracle inputs, or pack the fixture.

Mirrors gen_divduo_oracle.py: ONE canonical writer serialises
``d2a2c_input.txt`` (the Fortran driver reads it AND ``input_sha256`` is its
hash), gen stages the exact arrays, and ``--pack`` reads that SAME staging +
first-run input.txt + the driver's ``d2a2c_output.txt`` — no regeneration
drift.  Importing this module is side-effect-free.

  gen_d2a2c_oracle.py <W>          build inputs + staging + smoke the port.
  gen_d2a2c_oracle.py <W> --pack   write the committed fixture npz.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

RES, NG = 12, 3

# canonical field order (UPPER driver token, fixture npz key).  cosa/rsin +
# u/v are consumed by BOTH the port and the Fortran duo branch; sin_sg /
# dxa / dya are read by the Fortran (allocation + skipped edge cases) but do
# NOT affect the duo outputs.  ALL are serialised into input.txt and hashed;
# ALL are stored in the fixture (see FIXTURE_INPUTS) so the test can
# reconstruct the exact byte stream and enforce input_sha256.
INPUT_FIELDS = (
    ("COSA_S", "cosa_s"), ("RSIN2", "rsin2"),
    ("COSA_U", "cosa_u"), ("RSIN_U", "rsin_u"),
    ("COSA_V", "cosa_v"), ("RSIN_V", "rsin_v"),
    ("DXA", "dxa"), ("DYA", "dya"), ("SIN_SG", "sin_sg"),
    ("U", "u"), ("V", "v"),
)
# ALL inputs are stored in the fixture so the test can reconstruct the full
# input.txt byte stream and enforce input_sha256 (the port itself reads only
# cosa_s/rsin2/cosa_u/v/rsin_u/v + u/v; dxa/dya/sin_sg are stored for the
# hash + the Fortran driver, unused by the duo output).
FIXTURE_INPUTS = tuple(key for _name, key in INPUT_FIELDS)
PORT_INPUTS = ("cosa_s", "rsin2", "cosa_u", "rsin_u", "cosa_v", "rsin_v",
               "u", "v")
OUTPUT_FIELDS = ("ua", "va", "ut", "vt", "uc", "vc")


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


def serialize_d2a2c_inputs(fields: dict, res: int = RES, ng: int = NG) -> bytes:
    """The exact ``d2a2c_input.txt`` byte stream (driver reads it; the
    fixture hash is over it) — one writer for gen and the enforcement test."""
    lo = 1 - ng
    s = f"# res {res}\n# ng {ng}\n"
    for name, key in INPUT_FIELDS:
        s += _dump_field(name, np.asarray(fields[key]), lo)
    return s.encode()


def _fix_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "tests", "grids", "fixtures", "d2a2c_oracle_c12.npz")


def _gen(work: str) -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "packages", "core"))
    from legoesm.core.fv3_native_duo_sw_core import d2a2c_vect_duo
    from legoesm.core.fv3_native_sw_core import Bounds
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
        "u": np.nan_to_num(np.asarray(st["u"], float), nan=0.0),
        "v": np.nan_to_num(np.asarray(st["v"], float), nan=0.0),
        "cosa_s": np.asarray(gs["cosa_s"]), "rsin2": np.asarray(gs["rsin2"]),
        "cosa_u": np.asarray(gs["cosa_u"]), "rsin_u": np.asarray(gs["rsin_u"]),
        "cosa_v": np.asarray(gs["cosa_v"]), "rsin_v": np.asarray(gs["rsin_v"]),
        "dxa": np.asarray(gs["dxa"]), "dya": np.asarray(gs["dya"]),
        "sin_sg": np.asarray(gs["sin_sg"]),
    }
    with open(f"{work}/d2a2c_input.txt", "wb") as f:
        f.write(serialize_d2a2c_inputs(fields))
    np.savez_compressed(f"{work}/staging.npz", **fields)
    # smoke: the port runs on these inputs
    d2a2c_vect_duo(fields["u"], fields["v"], gs, bd, RES + 1, RES + 1,
                   dord4=True, grid_type=0)
    print("gen: inputs + staging written")


def _pack(work: str) -> None:
    stag = dict(np.load(f"{work}/staging.npz"))
    canon = serialize_d2a2c_inputs(stag)
    on_disk = open(f"{work}/d2a2c_input.txt", "rb").read()
    if canon != on_disk:
        raise SystemExit("d2a2c_input.txt disagrees with staging.npz — "
                         "regenerate; refusing to pack a drifted fixture")
    inp_hash = hashlib.sha256(canon).hexdigest()

    lo = 1 - NG
    m_a = RES + 2 * NG
    m_b = m_a + 1
    shapes = {"ua": (m_a, m_a), "va": (m_a, m_a), "ut": (m_a, m_a),
              "vt": (m_a, m_a), "uc": (m_b, m_a), "vc": (m_a, m_b)}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/d2a2c_output.txt"):
        pp = line.split()
        key = pp[0].lower()
        outs[key][int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])

    np.savez_compressed(
        _fix_dir(),
        **{k: outs[k] for k in OUTPUT_FIELDS},
        **{k: stag[k] for k in FIXTURE_INPUTS},
        res=RES, ng=NG, input_sha256=inp_hash,
        input_lineage="analytic-swcore-state D-winds; duo d2a2c_vect "
        "(dg%is_initialized) D->A->C — closes the divergence_corner_duo "
        "plain-c_sw ua/va scope caveat")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
