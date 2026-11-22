"""Fixture for the CAM6 host-omega oracle (tests/atmosphere/oracle_fixtures/clubb_host_omega_cam.npz).

usage: python make_fixture.py <oracle build dir> <A2 smoke dump dir> <out.npz>

Columns: 12 from an A2 C1 smoke dump (feed_00080.5000.npz): the 4 SE Pacific
and 4 warm-pool columns with the strongest omega and the 4 land columns with
the highest surface, with the dumped T, q_v, p_s and wap (the driver's
continuity omega, the field the MPAS lane hands CLUBB).  Heights are legoESM's
own (compute_heights_from_sigma, the MPAS turbulence bridge's), so the
comparison tests the omega -> wm map and the zt2zm interpolation alone.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma  # noqa: E402
from legoesm.grids.vertical import create_hybrid_coordinate  # noqa: E402


def dumped_columns(dump_dir):
    f = np.load(os.path.join(dump_dir, "feed_00080.5000.npz"))
    g = np.load(os.path.join(dump_dir, "grid.npz"))
    lat = np.asarray(f["lat_deg"], float)
    lon = np.degrees(np.asarray(g["lonCell"], float)) % 360.0
    p_s = np.asarray(f["p_s"], float)
    wap = np.asarray(f["wap"], float)
    phis = np.asarray(f["phis"], float)
    wp = (np.abs(lat) < 10) & (lon > 130) & (lon < 170)
    sep = (lat > -30) & (lat < -10) & (lon > 260) & (lon < 290)
    amp = np.abs(wap).max(1)
    pick = list(np.where(sep)[0][np.argsort(-amp[sep])][:4])
    pick += list(np.where(wp)[0][np.argsort(-amp[wp])][:4])
    pick += list(np.argsort(-phis)[:4])
    pick = np.asarray(pick)
    # the MPAS lane's own pressures (the bridge's sigma_coord.pressure_at_*)
    coord = create_hybrid_coordinate(len(g["A_half"]) - 1, jnp.asarray(g["A_half"]),
                                     jnp.asarray(g["B_half"]), constants.p_ref)
    p_half = np.asarray(coord.pressure_at_half(jnp.asarray(p_s[pick])))
    p_full = np.asarray(coord.pressure_at_full(jnp.asarray(p_s[pick])))
    T = np.asarray(f["T"], float)[pick]
    q_v = np.asarray(f["q_v"], float)[pick]
    z_full, z_half = compute_heights_from_sigma(
        jnp.asarray(T), jnp.asarray(p_half), q_v=jnp.asarray(q_v))
    return dict(cell=pick, T=T, q_v=q_v, p_half=p_half, p_full=p_full,
                z_full=np.asarray(z_full), z_half=np.asarray(z_half),
                omega=wap[pick])


def main(build, dump, out):
    c = dumped_columns(dump)
    ncol, nlev = c["T"].shape
    fin = os.path.join(build, "oracle_in.txt")
    fmt = lambda a: " ".join(f"{x:.17e}" for x in a)  # noqa: E731
    with open(fin, "w") as fh:
        fh.write(f"{ncol} {nlev}\n")
        for i in range(ncol):
            for a in (c["z_half"][i], c["z_full"][i], c["p_full"][i],
                      np.diff(c["p_half"][i]), c["omega"][i]):
                fh.write(fmt(a) + "\n")
    res = subprocess.run([os.path.join(build, "wm_oracle"), fin],
                         capture_output=True, text=True, check=True)
    rows = np.array([[float(x) for x in ln.split()] for ln in res.stdout.split("\n") if ln.strip()])
    c["wm_zt_cam"] = rows[0::2]
    c["wm_zm_cam"] = rows[1::2]
    np.savez(out, **c)
    print("wrote", out, {k: v.shape for k, v in c.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
