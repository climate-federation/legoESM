"""Convert the oracle's tables49.txt into the vendored RRTMG table file (vote V35).

usage: python -I export_tables_npz.py tables49.txt OUT.npz PROVENANCE.json KEY=VALUE...
Each Fortran variable "module.NAME" becomes an array in Fortran shape and
order, plus "lb:module.NAME" holding its Fortran lower bounds. KEY=VALUE pairs
(oracle commit, data sha256, generator commit) go into the provenance JSON with
the sha256 of the text dump and of the npz.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(txt, out, prov, *kv):
    arrays = dumpio.read_arrays(txt)
    payload = {}
    for name, (a, lb) in arrays.items():
        payload[name] = np.asfortranarray(a)
        payload["lb:" + name] = np.asarray(lb, dtype=np.int64)
    np.savez_compressed(out, **payload)
    meta = dict(kv.split("=", 1) for kv in kv)
    meta.update({"source_dump": os.path.basename(txt), "source_dump_sha256": sha256(txt),
                 "npz_sha256": sha256(out), "n_variables": len(arrays)})
    with open(prov, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"wrote {len(arrays)} tables to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
