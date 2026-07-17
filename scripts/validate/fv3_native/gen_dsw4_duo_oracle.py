#!/usr/bin/env python
"""Gen the duo d_sw4 chain oracle, or pack the fixture.

INPUT PROVENANCE: identical to the d_sw1/2/3 oracles — the ONE
canonical serializer ``gen_dsw1_duo_oracle.serialize_inputs`` (imported,
not copied).  The fixture records the sha256 of the d_sw4 extract file
so the test can pin it.

  gen_dsw4_duo_oracle.py <W>          write dswcore_input.txt from the npz.
  gen_dsw4_duo_oracle.py <W> --pack   read the driver's dsw4_output.txt and
                                      write tests/grids/fixtures/
                                      dsw4_duo_oracle_c12.npz.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
OUT_NPZ = os.path.join(REPO, "tests", "grids", "fixtures",
                       "dsw4_duo_oracle_c12.npz")
EXTRACT = os.path.join(HERE, "fv3_dsw4_duo_extract.F90")

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
    m_b = res + 2 * ng + 1
    ke = np.full((m_b, m_b), np.nan)
    for line in open(f"{work}/dsw4_output.txt"):
        pp = line.split()
        assert pp[0] == "KE", pp[0]
        ke[int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])
    if np.isnan(ke).any():
        raise SystemExit("dump under-writes KE")
    np.savez_compressed(
        OUT_NPZ, ke=ke, res=res, ng=ng, input_sha256=inp_hash,
        dsw4_extract_sha256=ext_hash,
        input_lineage="COMMITTED dswcore_input.npz serialised (no "
        "regeneration); symmetryclean d_sw1 -> d_sw4 chain, DUO branch; "
        "ke=1e30 pre-call on both sides, only the 4 corner B-nodes "
        "written (guard always true at bounded=F)")
    print("fixture packed; input_sha256", inp_hash)
    print("dsw4 extract sha256", ext_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
