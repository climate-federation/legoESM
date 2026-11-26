"""Build the ecRad 49r1 Tripleclouds oracle fixture npz from one harness run.

usage: python -I build_tripleclouds_fixture.py RUN_DIR OUT.npz [--all-columns]
RUN_DIR holds out.nc and out_radiative_properties.nc from run49.sh with the
IFS Tripleclouds settings (NLWSOLVER = NSWSOLVER = 3: lw/sw_solver_name =
"Tripleclouds", cloud_fraction_threshold = 2.5e-5, su_radiation_scheme.F90).
Keys: "props:<var>" solver inputs, "flux:<var>" outputs, "columns" the kept
0-based columns (every 4th of 32 by default).
"""
import os
import sys

import netCDF4
import numpy as np

NCOL = 32


def nc_vars(path, cols):
    out = {}
    with netCDF4.Dataset(path) as f:
        f.set_auto_mask(False)
        for v in f.variables:
            a = np.asarray(f[v][:])
            if a.ndim and a.shape[0] == NCOL:
                a = a[cols]
            out[v] = a
    return out


def main(run, out, *flags):
    cols = np.arange(NCOL) if "--all-columns" in flags else np.arange(0, NCOL, 4)
    z = {"columns": cols}
    for k, v in nc_vars(f"{run}/out_radiative_properties.nc", cols).items():
        z["props:" + k] = v
    for k, v in nc_vars(f"{run}/out.nc", cols).items():
        z["flux:" + k] = v
    np.savez_compressed(out, **z)
    print(f"{out}: {len(z)} arrays, columns {cols.tolist()}, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
