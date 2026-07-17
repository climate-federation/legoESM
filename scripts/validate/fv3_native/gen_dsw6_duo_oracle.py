#!/usr/bin/env python
"""Gen the duo d_sw6 chain oracle, or pack the fixture.

INPUT PROVENANCE: identical to the d_sw1/2/3 oracles — the ONE
canonical serializer ``gen_dsw1_duo_oracle.serialize_inputs`` (imported,
not copied).  The fixture records the sha256 of the d_sw6 extract file
so the test can pin it.

  gen_dsw6_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw6_duo_oracle.py <W> --pack   read the driver's dsw6_output.txt and
                                      write tests/grids/fixtures/
                                      dsw6_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw6_duo_oracle_c12.npz")
EXTRACT = os.path.join(HERE, "fv3_dsw6_duo_extract.F90")

_spec = importlib.util.spec_from_file_location(
    "gen_dsw1_duo_oracle", os.path.join(HERE, "gen_dsw1_duo_oracle.py"))
_gen1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen1)
serialize_inputs = _gen1.serialize_inputs


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
    shapes = {"u": (m_a, m_a + 1), "vt": (m_a, m_a + 1),
              "v": (m_a + 1, m_a), "ut": (m_a + 1, m_a),
              "ub": (res + 1, res + 1), "vb": (res + 1, res + 1),
              "heat": (res, res)}
    origins = {"u": (lo, lo), "vt": (lo, lo), "v": (lo, lo),
               "ut": (lo, lo), "ub": (1, 1), "vb": (1, 1),
               "heat": (1, 1)}
    tok = {"U3": "u", "VT3": "vt", "V3": "v", "UT3": "ut",
           "UB3": "ub", "VB3": "vb", "HEAT3": "heat"}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/dsw6_output.txt"):
        pp = line.split()
        key = tok[pp[0]]
        oi, oj = origins[key]
        outs[key][int(pp[1]) - oi, int(pp[2]) - oj] = float(pp[3])
    for k, a in outs.items():
        if np.isnan(a).any():
            raise SystemExit(f"dump under-writes token {k}")
    auth_shas = (
        "params:36-63:"
        "3e102b7433b88664d8b31d159251e26aa2a4651085a1ab1c606643c1913ab7cf;"
        "d_sw6:1871-2006:"
        "d68b8c9dd62e9be6e007c5faf94ce5417a210be976e57e31cca71a3c7244bebf;"
        "del6_vt_flux:2008-2121:"
        "a0c6676195fbac993b5457a3c2187b14ae4e658edbc6808b1b9888bde3f56d4d")
    np.savez_compressed(
        OUT_NPZ, **outs, res=res, ng=ng, input_sha256=inp_hash,
        dsw6_extract_sha256=ext_hash, auth_block_sha256=auth_shas,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean FULL CHAIN d_sw1 -> d_sw3 -> kee "
        "-> d_sw4 -> d_sw5 -> d_sw6, DUO branch; final circulation-form "
        "winds; ub/vb/heat_source 1e30 sentinels (d_con=0; d_sw2 "
        "skipped so heat stays sentinel)")
    print("fixture packed; input_sha256", inp_hash)
    print("dsw6 extract sha256", ext_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
