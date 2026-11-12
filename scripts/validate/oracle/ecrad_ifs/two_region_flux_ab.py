"""Offline flux change of ``cloud_optics_inhomogeneity="two_region"`` per deck.

Solves the model's RRTMGP radiation backend (the subcolumn lane every
two_region deck runs) on the 32 ecRad/IFS meridian columns (ecRad test input
``ecrad_meridian.nc``: T, p, q, cloud fraction, liquid and ice), once per
source tree, and diffs the two runs.  Each deck is reduced to the cloud
settings the offline solve can see (cloud scheme, subcolumn count, the two
cloud tunables); convection, turbulence, microphysics and step settings do not
reach a radiation solve on fixed columns, so decks that differ only there
share a row.  Prints column-mean TOA reflected SW, OLR and surface net flux
[W/m2] per setup; numbers only.

    python two_region_flux_ab.py run MERIDIAN.nc OUT.npz      # in each tree
    python two_region_flux_ab.py compare OLD.npz NEW.npz
"""
from __future__ import annotations

import argparse
import os
import subprocess

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

_SUNDQ = dict(rh_crit=0.85, q_c_diagnostic=5.0e-6)
_CAM6_CLUBB_TOP_PA = 15000.0    # spectral_t63_cam6.yaml clubb_top_press_hpa: 150
# setup -> (cloud scheme, n_subcolumns, CloudConfig overrides)
SETUPS = {
    "sundqvist_n8": ("sundqvist", 8, _SUNDQ),
    "sundqvist_n1": ("sundqvist", 1, _SUNDQ),
    "cam6_clubb_n8": ("cam6_clubb", 8, dict(q_c_diagnostic=0.0)),
    "cloud_none": ("none", 8, {}),
}
DECKS = {
    "config/amip/amip_sundqvist_l36.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_base.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_conv_tiedtke.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_gwd_mcf.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_micro_sundq.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_substeps_1.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_turb_louis.yaml": "sundqvist_n8",
    "config/wb/campaign/spectral_t63_amip.yaml": "sundqvist_n8",
    "config/wb/campaign/_mem_subcol_1.yaml": "sundqvist_n1",
    "config/wb/campaign/spectral_t63_cam6.yaml": "cam6_clubb_n8",
    "config/wb/campaign/_mem_cloud_none.yaml": "cloud_none",
}


def load_columns(path):
    import netCDF4
    with netCDF4.Dataset(path) as d:
        v = {k: np.asarray(d[k][:], np.float64) for k in (
            "lat", "skin_temperature", "cos_solar_zenith_angle", "solar_irradiance",
            "pressure_hl", "temperature_hl", "q", "q_liquid", "q_ice", "cloud_fraction")}
    p_hl, t_hl = v["pressure_hl"], v["temperature_hl"]
    return dict(
        T=0.5 * (t_hl[:, 1:] + t_hl[:, :-1]), p_full=0.5 * (p_hl[:, 1:] + p_hl[:, :-1]),
        p_half=p_hl, q_v=v["q"], q_cloud=v["q_liquid"], q_ice=v["q_ice"],
        cloud_fraction=v["cloud_fraction"], lat=np.deg2rad(v["lat"]),
        sfc_temperature=v["skin_temperature"], cos_sza=v["cos_solar_zenith_angle"],
        insolation=v["solar_irradiance"] * np.maximum(v["cos_solar_zenith_angle"], 0.0))


def solve(cols, setup):
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation import integration as rad_int
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

    scheme, n_sub, kw = SETUPS[setup]
    cc = CloudConfig(scheme="sundqvist" if scheme == "none" else scheme,
                     cloud_optics_inhomogeneity="two_region", cloud_fsd=1.0,
                     cloud_vertical_overlap_optics="max_random", cloud_n_subcolumns=n_sub,
                     cloud_partial_coverage_optics="none", **kw)
    cfg = RadiationConfig(scheme="rrtmgp", cloud_scheme=scheme, cloud_config=cc,
                          rrtmgp=RRTMGPConfig(include_clouds=True))
    a = {k: jnp.asarray(v) for k, v in cols.items() if k != "cloud_fraction"}
    extra = {}
    if scheme == "cam6_clubb":    # the CLUBB cover the lane would hand over
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
        extra["cloud_fraction_override"] = jnp.asarray(cols["cloud_fraction"])
        extra["clubb_config"] = CLUBBConfig(trop_cloud_top_press=_CAM6_CLUBB_TOP_PA)
    out = rad_int._call_radiation_backend(
        cfg, rrtmgp_solver=RRTMGP.from_legoesm_config(cfg.rrtmgp), **a, **extra)
    sw_up, sw_dn = np.asarray(out.sw_flux_up), np.asarray(out.sw_flux_down)
    lw_up, lw_dn = np.asarray(out.lw_flux_up), np.asarray(out.lw_flux_down)
    r = np.stack([sw_up[:, 0], lw_up[:, 0],
                  sw_dn[:, -1] - sw_up[:, -1] + lw_dn[:, -1] - lw_up[:, -1]])
    if not np.all(np.isfinite(r)):
        raise FloatingPointError(f"{setup}: non-finite fluxes")
    return r


def run(meridian, out):
    cols = load_columns(meridian)
    res, err = {}, {}
    for s in SETUPS:
        try:
            res[s] = solve(cols, s)
        except Exception as e:  # recorded, never silently dropped
            err[s] = f"{type(e).__name__}: {e}"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    np.savez(out, sha=sha, meridian=meridian, errors=repr(err), **res)
    print(sha, {s: np.round(r.mean(axis=1), 3).tolist() for s, r in res.items()}, err)


def compare(old, new):
    a, b = np.load(old), np.load(new)
    print(f"old {a['sha']}  new {b['sha']}  columns: {a['meridian']}")
    print(f"old errors {a['errors']}  new errors {b['errors']}")
    print(f"{'setup':16s} {'dRSUT':>9s} {'dOLR':>9s} {'dSFCnet':>9s}   (new - old, mean of 32 cols)")
    for s in SETUPS:
        if s in a.files and s in b.files:
            d = (b[s] - a[s]).mean(axis=1)
            print(f"{s:16s} {d[0]:9.3f} {d[1]:9.3f} {d[2]:9.3f}")
        else:
            print(f"{s:16s} not solved in both trees")
    for deck, s in DECKS.items():
        print(f"  {deck} -> {s}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("meridian")
    r.add_argument("out")
    c = sp.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    a = p.parse_args(argv)
    run(a.meridian, a.out) if a.cmd == "run" else compare(a.old, a.new)


if __name__ == "__main__":
    main()
