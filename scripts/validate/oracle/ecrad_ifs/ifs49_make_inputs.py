"""Export the meridian fixture columns as text inputs for ifs49_driver.x.

Writes one file per array, <out>/<name>.txt: header "name rank n1 [n2 ...]"
(Fortran shape, column first), then the values in Fortran order, one per
line, "%.17e" (float32 fixture values are exact in fp64). No value is
derived here: every conversion the IFS interface needs is done by the
Fortran driver (cited there). NaN anywhere in an exported array is fatal.

usage: python -I ifs49_make_inputs.py <ecrad_ifs49.npz> <out dir>
"""
import os
import sys

import numpy as np

KEYS = ("lat lon skin_temperature cos_solar_zenith_angle sw_albedo sw_albedo_direct lw_emissivity "
        "pressure_hl temperature_hl q o3_mmr co2_vmr n2o_vmr ch4_vmr cfc11_vmr cfc12_vmr hcfc22_vmr "
        "ccl4_vmr cloud_fraction q_liquid q_ice solar_irradiance").split()


def main(npz, out):
    z = np.load(npz)
    os.makedirs(out, exist_ok=True)
    for k in KEYS:
        a = np.asarray(z["in:" + k], dtype=np.float64)
        if not np.all(np.isfinite(a)):
            sys.exit(f"non-finite value in {k}")
        with open(os.path.join(out, k + ".txt"), "w") as f:
            f.write(f"{k} {a.ndim} {' '.join(str(n) for n in a.shape)}\n")
            for v in a.ravel(order="F"):
                f.write("%.17e\n" % v)
    with open(os.path.join(out, "columns.txt"), "w") as f:
        f.write(" ".join(str(int(c)) for c in z["columns"]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
