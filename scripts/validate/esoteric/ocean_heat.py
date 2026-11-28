"""Ocean heat-storage change: day-0 (rebuilt Qobs IC) vs day-N MPAS ocean checkpoint.

Heat per column uses live layer thickness (z-star with SSH), so volume change is
counted. A band's rate is STORAGE change: it includes lateral ocean transport,
not only the surface flux. Usage: ocean_heat.py run_out_dir mesh_level day
"""
import sys

import numpy as np

from _common import BANDS, apply_run_precision, find_key, load_manifest

LEVELS_SHOWN = (0, 1, 2, 3, 5, 8, 12)


def storage_rate(T0, h0, T1, h1, area, days):
    """Mean heat-storage rate [W/m2] over the given cells: rho*c*sum(h1*T1 - h0*T0)/area/time."""
    from legoesm import constants

    dQ = constants.rho_ocean * constants.c_sw * np.sum(h1 * T1 - h0 * T0, axis=1)
    return float(np.sum(dQ * area) / area.sum() / (days * 86400.0))


def check_supported(ocean_ic, H_wet, H_max):
    """The day-0 rebuild reproduces only the Qobs IC on a flat full-depth ocean."""
    if ocean_ic != "qobs":
        raise ValueError(f"day-0 rebuild supports ocean_ic='qobs' only, run used {ocean_ic!r}")
    if not np.allclose(H_wet, H_max):
        raise ValueError(f"wet depths span {H_wet.min():.0f}-{H_wet.max():.0f} m, not flat {H_max:.0f} m "
                         "(partial-cell geometry is not rebuilt here)")


def check_same_instant(atm_ck, ocean_ck, day):
    """The ocean checkpoint must be stamped with the atmosphere step of the same day."""
    if float(atm_ck["day"]) != day or int(ocean_ck["_atm_step"]) != int(atm_ck["step"]):
        raise ValueError(f"checkpoints disagree: atm day {float(atm_ck['day'])} step {int(atm_ck['step'])}, "
                         f"ocean _atm_step {int(ocean_ck['_atm_step'])}, requested day {day}")


def _ckpt_field(ck, name):
    return np.asarray(ck[f"ocean3d_.{name}.[<flat index 0>]"])


def main(argv):
    from legoesm.forcing.analytical import qobs_sst_C
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import REST_T_DEEP_C, rest_state_mpas_ocean
    from legoesm.ocean.vertical import compute_layer_thickness, create_ocean_z_star

    out, lev, day = argv[0], int(argv[1]), int(argv[2])
    man = load_manifest(out)
    prec = apply_run_precision(man)
    nlev, Hmax = int(find_key(man, "ocean_nlev")), float(find_key(man, "ocean_H_max_m"))
    mesh = create_voronoi_mesh(lev)
    z = create_ocean_z_star(nlev, H_max=Hmax)
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell)
    Ts = qobs_sst_C(lat)
    s0 = rest_state_mpas_ocean(mesh, z, T_water_init_C=Ts, T_deep=np.minimum(REST_T_DEEP_C, Ts),
                               H_max=Hmax, land_lat_threshold=90.0)
    ck = np.load(f"{out}/coupled_day_{day:04d}.npz", allow_pickle=True)
    check_same_instant(np.load(f"{out}/checkpoint_day_{day:04d}.npz", allow_pickle=True), ck, day)
    T1, S1, eta1 = _ckpt_field(ck, "T"), _ckpt_field(ck, "S"), _ckpt_field(ck, "eta")
    wet, H = _ckpt_field(ck, "land_mask") > 0.5, _ckpt_field(ck, "H_bathy")
    check_supported(find_key(man, "ocean_ic"), H[wet], Hmax)
    T0, S0, eta0 = (np.asarray(f.data) for f in (s0.T, s0.S, s0.eta))
    if T0.shape != T1.shape or not np.all(np.isfinite(T1[wet])):
        raise ValueError(f"bad ocean T: day0 {T0.shape}, day{day} {T1.shape}")
    h0 = np.asarray(compute_layer_thickness(eta0, H, z))
    h1 = np.asarray(compute_layer_thickness(eta1, H, z))
    aw = area * wet
    print(f"precision {prec}; nlev {nlev} H_max {Hmax}; wet cells {wet.sum()} of {wet.size}")
    print(f"eta day0 mean {np.sum(eta0 * aw) / aw.sum():+.3f} m, day{day} mean {np.sum(eta1 * aw) / aw.sum():+.3f} m")
    print(f"SST area mean day0 {np.sum(T0[:, 0] * aw) / aw.sum():.3f} C, day{day} {np.sum(T1[:, 0] * aw) / aw.sum():.3f} C")
    print(f"global ocean heat-storage rate (live thickness): {storage_rate(T0[wet], h0[wet], T1[wet], h1[wet], area[wet], day):+.1f} W/m2")
    print("band | dSST K | storage rate W/m2 (incl. lateral transport) | dT by level k=" + ",".join(map(str, LEVELS_SHOWN)))
    for a, b in BANDS:
        m = wet & (lat >= a) & (lat < b); w = area[m]
        lv = [np.sum((T1[m, k] - T0[m, k]) * w) / w.sum() for k in LEVELS_SHOWN]
        r = storage_rate(T0[m], h0[m], T1[m], h1[m], w, day)
        print(f"{a:4d}..{b:3d} | {lv[0]:+6.2f} | {r:+8.1f} | " + " ".join(f"{v:+6.2f}" for v in lv))
    vol0, vol1 = np.sum(h0 * aw[:, None]), np.sum(h1 * aw[:, None])
    print(f"SSS area mean day{day} {np.sum(S1[:, 0] * aw) / aw.sum():.4f}; volume-mean S day0 "
          f"{np.sum(S0 * h0 * aw[:, None]) / vol0:.5f} day{day} {np.sum(S1 * h1 * aw[:, None]) / vol1:.5f}")


if __name__ == "__main__":
    main(sys.argv[1:])
