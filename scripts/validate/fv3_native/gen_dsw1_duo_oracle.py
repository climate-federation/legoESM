#!/usr/bin/env python
"""Gen the duo d_sw1 per-stage oracle, or pack the fixture.

INPUT PROVENANCE: serialises the COMMITTED phase-4b fixture
``tests/grids/fixtures/dswcore_input.npz`` into the ``dswcore_input.txt``
the Fortran driver reads — no regeneration, so the Fortran and the python
port consume byte-identical inputs (the committed arrays).  The txt
SHA-256 is stored in the output fixture and re-derivable from the npz.

  gen_dsw1_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw1_duo_oracle.py <W> --pack   read the driver's dsw1_output.txt and
                                      write tests/grids/fixtures/
                                      dsw1_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
IN_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                      "dswcore_input.npz")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw1_duo_oracle_c12.npz")

# (driver token, npz key, kind) — kind: a=2D cell-origin, sg=3D sin/cos_sg,
# e=1D edge, s=scalar.  EXACTLY the fields the phase-4b exporter writes
# (the driver reads the same file format).
FIELDS_2D = [
    ("RAREA", "rarea"), ("AREA", "area"), ("AREA_C", "area_c"),
    ("RAREA_C", "rarea_c"), ("DXA", "dxa"), ("DYA", "dya"),
    ("RDXA", "rdxa"), ("RDYA", "rdya"), ("COSA_S", "cosa_s"),
    ("RSIN2", "rsin2"), ("DX", "dx"), ("DY", "dy"), ("RDX", "rdx"),
    ("RDY", "rdy"), ("DXC", "dxc"), ("DYC", "dyc"), ("RDXC", "rdxc"),
    ("RDYC", "rdyc"), ("COSA_U", "cosa_u"), ("SINA_U", "sina_u"),
    ("RSIN_U", "rsin_u"), ("COSA_V", "cosa_v"), ("SINA_V", "sina_v"),
    ("RSIN_V", "rsin_v"), ("COSA", "cosa"), ("SINA", "sina"),
    ("RSINA", "rsina"), ("FC", "fC"), ("F0", "f0"),
    ("DIVG_U", "divg_u"), ("DIVG_V", "divg_v"),
    ("DEL6_U", "del6_u"), ("DEL6_V", "del6_v"),
    ("GRID_LON", "grid_lon"), ("GRID_LAT", "grid_lat"),
    ("AGRID_LON", "agrid_lon"), ("AGRID_LAT", "agrid_lat"),
    ("DELP", "delp"), ("PT", "pt"), ("W", "w"), ("U", "u"), ("V", "v"),
    ("UC", "uc"), ("VC", "vc"), ("UA", "ua"), ("VA", "va"),
    ("DIVGD_IN", "divg_d_in"),
]
OUT_KEYS = ("crx", "xfx", "rax", "cry", "yfx", "ray", "ut", "vt",
            "afx1", "afx4", "xflux_out", "afy1", "afy4", "yflux_out",
            "cx_out", "cy_out")
_TOKEN2KEY = {"CRX": "crx", "XFX": "xfx", "RAX": "rax", "CRY": "cry",
              "YFX": "yfx", "RAY": "ray", "UT": "ut", "VT": "vt",
              "AFX1": "afx1", "AFX4": "afx4", "XFLUX": "xflux_out",
              "AFY1": "afy1", "AFY4": "afy4", "YFLUX": "yflux_out",
              "CX": "cx_out", "CY": "cy_out"}


def serialize_inputs() -> tuple[bytes, int, int, float]:
    z = np.load(IN_NPZ, allow_pickle=True)
    res, ng = int(z["res"]), int(z["ng"])
    dt = float(z["dt"])
    lo = 1 - ng
    lines = [f"# res {res}", f"# ng {ng}", f"# dt {dt:.17e}"]

    def dump2(name, a):
        ni, nj = a.shape
        for i in range(ni):
            for j in range(nj):
                lines.append(f"{name} {i + lo} {j + lo} {a[i, j]:.17e}")

    for name, key in FIELDS_2D:
        dump2(name, np.asarray(z[key], dtype=np.float64))
    for name, key in (("SIN_SG", "sin_sg"), ("COS_SG", "cos_sg")):
        a = np.asarray(z[key], dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    lines.append(f"{name} {i + lo} {j + lo} {k + 1} "
                                 f"{a[i, j, k]:.17e}")
    for name, key in (("EDGE_S", "edge_s"), ("EDGE_N", "edge_n"),
                      ("EDGE_W", "edge_w"), ("EDGE_E", "edge_e")):
        a = np.asarray(z[key], dtype=np.float64)
        for i in range(a.shape[0]):
            lines.append(f"{name} {i + 1} {a[i]:.17e}")
    lines.append(f"DA_MIN {float(z['da_min']):.17e}")
    lines.append(f"DA_MIN_C {float(z['da_min_c']):.17e}")
    return ("\n".join(lines) + "\n").encode(), res, ng, dt


def _gen(work: str) -> None:
    blob, res, ng, dt = serialize_inputs()
    with open(f"{work}/dswcore_input.txt", "wb") as f:
        f.write(blob)
    print(f"gen: dswcore_input.txt written (res={res} ng={ng} dt={dt}); "
          f"sha256 {hashlib.sha256(blob).hexdigest()}")


def _pack(work: str) -> None:
    blob, res, ng, _dt = serialize_inputs()
    on_disk = open(f"{work}/dswcore_input.txt", "rb").read()
    if blob != on_disk:
        raise SystemExit("dswcore_input.txt disagrees with the committed "
                         "npz serialisation — refusing to pack")
    inp_hash = hashlib.sha256(blob).hexdigest()
    lo = 1 - ng
    m_a = res + 2 * ng
    shapes = {"crx": (res + 1, m_a), "xfx": (res + 1, m_a),
              "rax": (res, m_a),
              "cry": (m_a, res + 1), "yfx": (m_a, res + 1),
              "ray": (m_a, res),
              "ut": (m_a + 1, m_a), "vt": (m_a, m_a + 1),
              "afx1": (res + 1, res), "afx4": (res + 1, res),
              "xflux_out": (res + 1, res),
              "afy1": (res, res + 1), "afy4": (res, res + 1),
              "yflux_out": (res, res + 1),
              "cx_out": (res + 1, m_a), "cy_out": (m_a, res + 1)}
    # per-token python-array origin (fortran lower bounds of the dump loops)
    origins = {"crx": (1, lo), "xfx": (1, lo), "rax": (1, lo),
               "cry": (lo, 1), "yfx": (lo, 1), "ray": (lo, 1),
               "ut": (lo, lo), "vt": (lo, lo),
               "afx1": (1, 1), "afx4": (1, 1), "xflux_out": (1, 1),
               "afy1": (1, 1), "afy4": (1, 1), "yflux_out": (1, 1),
               "cx_out": (1, lo), "cy_out": (lo, 1)}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/dsw1_output.txt"):
        pp = line.split()
        key = _TOKEN2KEY[pp[0]]
        oi, oj = origins[key]
        outs[key][int(pp[1]) - oi, int(pp[2]) - oj] = float(pp[3])
    np.savez_compressed(
        OUT_NPZ, **outs, res=res, ng=ng, input_sha256=inp_hash,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean d_sw1 DUO branch, fl%duogrid + "
        "dg%is_initialized, ut/vt workspace = 1e30 sentinel on both sides")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
