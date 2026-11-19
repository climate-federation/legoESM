"""Vendor the McICA Gamma PDF-sampler table as npz (vote G3, after V35).

usage: python -I export_mcica_gamma_npz.py mcica_gamma.nc OUT.npz PROVENANCE.json KEY=VALUE...
Reads the two arrays radiation_pdf_sampler.F90:setup_pdf_sampler reads
("fsd" and "x"), keeps the stored float32 values and their NetCDF shape
(x: (fsd, cdf), i.e. the Fortran val(ncdf, nfsd) transposed). The JSON holds
the sha256 of the NetCDF file and of the npz plus the KEY=VALUE pairs.
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


def main(src, out, prov, *kv):
    with netCDF4.Dataset(src) as d:
        d.set_auto_mask(False)
        payload = {name: np.asarray(d.variables[name][:]) for name in ("fsd", "x")}
    np.savez_compressed(out, **payload)
    meta = dict(item.split("=", 1) for item in kv)
    meta.update({"file": os.path.basename(src), "sha256": sha256(src), "npz_sha256": sha256(out),
                 "shapes": {k: list(v.shape) for k, v in payload.items()},
                 "dtypes": {k: str(v.dtype) for k, v in payload.items()}})
    with open(prov, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"wrote {len(payload)} arrays to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
