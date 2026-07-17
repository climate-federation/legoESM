#!/usr/bin/env python
"""Gen the duo d_sw5 full-chain oracle, or pack the fixture.

INPUT PROVENANCE: identical to the d_sw1..4 oracles — the ONE canonical
serializer ``gen_dsw1_duo_oracle.serialize_inputs`` (imported, not
copied).  The fixture records the d_sw5 extract-file sha256 plus the
authoritative-block SHAs (computed at extraction time).

  gen_dsw5_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw5_duo_oracle.py <W> --pack   read the driver's dsw5_output.txt and
                                      write tests/grids/fixtures/
                                      dsw5_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw5_duo_oracle_c12.npz")
EXTRACT = os.path.join(HERE, "fv3_dsw5_duo_extract.F90")

_spec = importlib.util.spec_from_file_location(
    "gen_dsw1_duo_oracle", os.path.join(HERE, "gen_dsw1_duo_oracle.py"))
_gen1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen1)
serialize_inputs = _gen1.serialize_inputs

_TOKEN2KEY = {"DELPC2": "delpc", "PTC2": "ptc", "WK2": "wk",
              "DIVGD2": "divg_d", "KE2": "ke",
              "UC2": "uc", "UT2": "ut", "VC2": "vc", "VT2": "vt",
              "VFX": "vortfluxx", "VFY": "vortfluxy",
              "UB2": "ub", "VB2": "vb"}


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
    ext_hash = hashlib.sha256(open(EXTRACT, "rb").read()).hexdigest()
    lo = 1 - ng
    m_a = res + 2 * ng
    shapes = {"delpc": (m_a, m_a), "ptc": (m_a, m_a), "wk": (m_a, m_a),
              "divg_d": (m_a + 1, m_a + 1), "ke": (m_a + 1, m_a + 1),
              "uc": (m_a + 1, m_a), "ut": (m_a + 1, m_a),
              "vc": (m_a, m_a + 1), "vt": (m_a, m_a + 1),
              "vortfluxx": (res + 1, res), "vortfluxy": (res, res + 1),
              "ub": (res + 1, res + 1), "vb": (res + 1, res + 1)}
    origins = {"delpc": (lo, lo), "ptc": (lo, lo), "wk": (lo, lo),
               "divg_d": (lo, lo), "ke": (lo, lo),
               "uc": (lo, lo), "ut": (lo, lo),
               "vc": (lo, lo), "vt": (lo, lo),
               "vortfluxx": (1, 1), "vortfluxy": (1, 1),
               "ub": (1, 1), "vb": (1, 1)}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/dsw5_output.txt"):
        pp = line.split()
        key = _TOKEN2KEY[pp[0]]
        oi, oj = origins[key]
        outs[key][int(pp[1]) - oi, int(pp[2]) - oj] = float(pp[3])
    for k, a in outs.items():
        if np.isnan(a).any():
            raise SystemExit(f"dump under-writes token {k}")
    auth_shas = (
        "params:36-63:"
        "3e102b7433b88664d8b31d159251e26aa2a4651085a1ab1c606643c1913ab7cf;"
        "a2b_ord4:a2b_edge:50-330:"
        "b1d4038993561f73a5bfb52b35318f560381ecfe04ad5d16346e85434560ab58;"
        "extrap_corner:a2b_edge:455-465:"
        "acb99d531344860b7ada020b8b10392cf15469ff74973caed770f3c19b7dedee;"
        "d_sw5:1474-1869:"
        "e5f311c4402682fff4d00d52e4792b2edba04456c0811edb6e24358f2d65986b;"
        "smag_corner:2451-2537:"
        "5af7d0414112e6fe4f6ec8fefeb16ca888fa06e20b0d1f910e3b2b8120982711")
    np.savez_compressed(
        OUT_NPZ, **outs, res=res, ng=ng, input_sha256=inp_hash,
        dsw5_extract_sha256=ext_hash, auth_block_sha256=auth_shas,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean SINGLE-FACE chain d_sw1 -> d_sw3 "
        "-> raw kee (verbatim dyn_core loops, UNEXCHANGED — the "
        "BGRID_NE averaging is excluded six-face infrastructure) -> "
        "d_sw4 -> d_sw5, DUO branch; d_sw2 skipped (no d_sw5-lane "
        "effect at hydrostatic/damp_w=0); delpc/ptc/ub/vb/ke-halo "
        "1e30 sentinels via the extract intent shims")
    print("fixture packed; input_sha256", inp_hash)
    print("dsw5 extract sha256", ext_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
