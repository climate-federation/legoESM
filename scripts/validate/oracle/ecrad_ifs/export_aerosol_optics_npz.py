"""Vendor the 49r1 oracle's post-setup aerosol optics tables as npz (vote A3a).

usage: python -I export_aerosol_optics_npz.py <aerosol-arm run dump dir> OPTICS.nc EXE OUT.npz OUT.json KEY=VALUE...
       python -I export_aerosol_optics_npz.py raw OPTICS.nc OUT.npz OUT.json KEY=VALUE...
The second form vendors the wavenumber-resolved file itself (votes G1a, G3a):
the variables setup_general_aerosol_optics reads (radiation_aerosol_optics.F90:
173-191), in Fortran shape ((nwn, ntype), (nwn, nrh, ntype)), for the port's own
band averaging on RRTMGP's bands.
Reads the aot_* dumps that ifs49_driver.F90 (AER_DUMP_TABLES) writes after
SU_RADIATION_SCHEME: RAD_CONFIG%AEROSOL_OPTICS as setup_general_aerosol_optics
left it (aerosol_ifs_49R1_20230119.nc averaged to the RRTMG bands), the 49r1
type map and the background-aerosol settings. Arrays keep their Fortran shape
((nband, ntype), (nband, nrh, ntype)); nothing is derived here. The JSON holds
the sha256 of the optics file, the oracle executable and the npz, plus KEY=VALUE.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dumpio import read_arrays  # noqa: E402

KEYS = {  # npz key -> dump name
    **{f"{v}_{b}_{c}": f"aot_{v}_{b}_{c}" for c in ("phobic", "philic")
       for v, b in (("mass_ext", "sw"), ("ssa", "sw"), ("g", "sw"), ("mass_ext", "lw"), ("ssa", "lw"))},
    "rh_lower": "aot_rh_lower", "type_map": "aot_type_map",
    "trop_bg_mass_ext": "su_trop_bg_aer_mass_ext", "strat_bg_mass_ext": "su_strat_bg_aer_mass_ext",
    "itype_trop_bg": "aot_itype_trop_bg", "itype_strat_bg": "aot_itype_strat_bg",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dump_dir, optics, exe, out, prov, *kv):
    d = {}
    for p in glob.glob(os.path.join(dump_dir, "*.txt")):
        name = os.path.basename(p)[:-4]
        if name in KEYS.values():
            d.update(read_arrays(p))
    z = {}
    for k, name in KEYS.items():
        if name not in d:
            sys.exit(f"missing dump {name}")
        a = np.asarray(d[name][0])
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            sys.exit(f"non-finite values in {name}")
        z[k] = a
    np.savez_compressed(out, **z)
    meta = {"optics_file": os.path.basename(optics), "optics_sha256": sha256(optics),
            "oracle_exe_sha256": sha256(exe), "npz_sha256": sha256(out),
            "departure": ("49r1 reads aerosol_ifs_49R1_20230725.nc (su_radiation_scheme.F90:441); "
                          "the public 48r1 aerosol_ifs_49R1_20230119.nc is used under that name "
                          "(user decision 2026-11-04, vote A1a)"),
            "shapes": {k: list(np.shape(v)) for k, v in z.items()}}
    for item in kv:
        k, v = item.split("=", 1)
        meta[k] = v
    with open(prov, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"EXPORT_OK {len(z)} arrays -> {out}")


RAW_KEYS = {"wavenumber": "wavenumber", "rh_lower": "relative_humidity1",
            **{f"{v}_{c}": f"{n}_hydro{c}" for c in ("phobic", "philic")
               for v, n in (("mass_ext", "mass_ext"), ("ssa", "ssa"), ("g", "asymmetry"))}}


def raw(optics, out, prov, *kv):
    import netCDF4
    with netCDF4.Dataset(optics) as ds:
        # netCDF dimension order is the reverse of the Fortran reader's
        z = {k: np.asarray(ds[name][:], np.float64).T for k, name in RAW_KEYS.items()}
    for k, a in z.items():
        if not np.all(np.isfinite(a)):
            sys.exit(f"non-finite values in {k}")
    if np.any(np.diff(z["wavenumber"]) <= 0):
        sys.exit("wavenumber not strictly increasing")
    np.savez_compressed(out, **z)
    meta = {"optics_file": os.path.basename(optics), "optics_sha256": sha256(optics),
            "npz_sha256": sha256(out), "variables": RAW_KEYS,
            "shapes": {k: list(np.shape(v)) for k, v in z.items()}}
    for item in kv:
        k, v = item.split("=", 1)
        meta[k] = v
    with open(prov, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"EXPORT_RAW_OK {len(z)} arrays -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "raw":
        raw(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
