#!/usr/bin/env python
"""Gen the duo d_sw2 chain oracle, or pack the fixture.

INPUT PROVENANCE: identical to the d_sw1 oracle — the ONE canonical
serializer ``gen_dsw1_duo_oracle.serialize_inputs`` (imported, not
copied) writes ``dswcore_input.txt`` from the COMMITTED
``tests/grids/fixtures/dswcore_input.npz``; the same sha256 is stored
in the output fixture and re-derivable from the npz.

  gen_dsw2_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw2_duo_oracle.py <W> --pack   read the driver's dsw2_output.txt and
                                      write tests/grids/fixtures/
                                      dsw2_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw2_duo_oracle_c12.npz")

_spec = importlib.util.spec_from_file_location(
    "gen_dsw1_duo_oracle", os.path.join(HERE, "gen_dsw1_duo_oracle.py"))
_gen1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen1)
serialize_inputs = _gen1.serialize_inputs

_TOKEN2KEY = {"DELP2": "delp2", "PT2": "pt2", "PTC2": "ptc2",
              "HEAT": "heat", "DW": "dw"}


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
    shapes = {"delp2": (m_a, m_a), "pt2": (m_a, m_a), "ptc2": (m_a, m_a),
              "heat": (res, res), "dw": (res, res)}
    origins = {"delp2": (lo, lo), "pt2": (lo, lo), "ptc2": (lo, lo),
               "heat": (1, 1), "dw": (1, 1)}
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    for line in open(f"{work}/dsw2_output.txt"):
        pp = line.split()
        key = _TOKEN2KEY[pp[0]]
        oi, oj = origins[key]
        outs[key][int(pp[1]) - oi, int(pp[2]) - oj] = float(pp[3])
    for k, a in outs.items():
        if np.isnan(a).any():
            raise SystemExit(f"dump under-writes token {k}")
    np.savez_compressed(
        OUT_NPZ, **outs, res=res, ng=ng, input_sha256=inp_hash,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean d_sw1 -> d_sw2 chain, DUO branch, "
        "fl%duogrid + dg%is_initialized, ut/vt=1e30, ptc/dw/heat=1e30 "
        "pre-call sentinels")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
