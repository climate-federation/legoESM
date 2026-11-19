"""Vendor the ecRad cloud-optics coefficient tables as npz (vote CO1, after V35).

usage: python -I export_cloud_optics_npz.py LIQ.nc ICE.nc OUT.npz PROVENANCE.json KEY=VALUE...
Reads the arrays the 49r1 oracle reads in radiation_cloud_optics_data.F90
(coeff_lw, coeff_sw of each file, shape (band, coeff) after the reader's
transpose_matrices), keeps the stored float32 values, and writes
"liq.coeff_lw", "liq.coeff_sw", "ice.coeff_lw", "ice.coeff_sw". The JSON holds
the sha256 of both NetCDF files and of the npz plus the KEY=VALUE pairs.
"""
import hashlib
import json
import os
import sys

import netCDF4
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(liq, ice, out, prov, *kv):
    payload = {}
    for tag, path in (("liq", liq), ("ice", ice)):
        with netCDF4.Dataset(path) as d:
            d.set_auto_mask(False)
            for name in ("coeff_lw", "coeff_sw"):
                payload[f"{tag}.{name}"] = np.asarray(d.variables[name][:])
    np.savez_compressed(out, **payload)
    meta = dict(item.split("=", 1) for item in kv)
    meta.update({"liq_file": os.path.basename(liq), "liq_sha256": sha256(liq),
                 "ice_file": os.path.basename(ice), "ice_sha256": sha256(ice),
                 "npz_sha256": sha256(out),
                 "shapes": {k: list(v.shape) for k, v in payload.items()}})
    with open(prov, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"wrote {len(payload)} arrays to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
