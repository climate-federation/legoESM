#!/usr/bin/env python
"""Gen the duo d_sw3 chain oracle, or pack the fixture.

INPUT PROVENANCE: identical to the d_sw1/d_sw2 oracles — the ONE
canonical serializer ``gen_dsw1_duo_oracle.serialize_inputs`` (imported,
not copied) writes ``dswcore_input.txt`` from the COMMITTED
``tests/grids/fixtures/dswcore_input.npz``.  The fixture additionally
records the sha256 of the d_sw3 extract file so the test can pin it.

  gen_dsw3_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw3_duo_oracle.py <W> --pack   read the driver's dsw3_output.txt and
                                      write tests/grids/fixtures/
                                      dsw3_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw3_duo_oracle_c12.npz")
EXTRACT = os.path.join(HERE, "fv3_dsw3_duo_extract.F90")

_spec = importlib.util.spec_from_file_location(
    "gen_dsw1_duo_oracle", os.path.join(HERE, "gen_dsw1_duo_oracle.py"))
_gen1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen1)
serialize_inputs = _gen1.serialize_inputs

_TOKEN2KEY = {"UBBT": "ubbtemp", "VBBT": "vbbtemp",
              "UBB": "ubb", "VBB": "vbb"}


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
    nb = res + 1
    outs = {k: np.full((nb, nb), np.nan) for k in _TOKEN2KEY.values()}
    for line in open(f"{work}/dsw3_output.txt"):
        pp = line.split()
        key = _TOKEN2KEY[pp[0]]
        outs[key][int(pp[1]) - 1, int(pp[2]) - 1] = float(pp[3])
    for k, a in outs.items():
        if np.isnan(a).any():
            raise SystemExit(f"dump under-writes token {k}")
    np.savez_compressed(
        OUT_NPZ, **outs, res=res, ng=ng, input_sha256=inp_hash,
        dsw3_extract_sha256=ext_hash,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean d_sw1 -> d_sw3 chain, DUO branch, "
        "fl%duogrid + dg%is_initialized; d_sw3 duo lane reads no ut/vt "
        "workspace (interior contravariant formulas everywhere)")
    print("fixture packed; input_sha256", inp_hash)
    print("dsw3 extract sha256", ext_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
