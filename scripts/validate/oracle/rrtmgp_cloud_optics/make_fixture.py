"""Generate tests/atmosphere/oracle_fixtures/rrtmgp_cloud_optics.npz.

Oracle: RTE+RRTMGP v1.8 ``mo_cloud_optics_rrtmgp%cloud_optics`` (LUT path) run on
the bundled LUTs (cloudysky_lw.nc and cloudysky_sw.nc = rrtmgp-data v1.8.2
rrtmgp-clouds-{lw,sw}.nc), ice roughness 2 (medium):
  * LW: into a ``ty_optical_props_1scl`` -> absorption optical depth (no scattering),
    the RRTMGP reference / CAM6 RRTMG LW cloud treatment;
  * SW: into a ``ty_optical_props_2str``, then ``delta_scale`` (rrtmgp_allsky.F90).
The ice table coordinate is CAM6's ice optics size ``deffi = effi*rhoi/rhows*2``
(CAM cam_cesm2_1_rel_60 micro_mg2_0.F90:2795, rhoi = 500, rhows = 917 from
micro_mg_utils.F90:126,128), computed here from legoESM's PSD radius.

Inputs: 5 CAM6-L32 columns (tier-1 audit set, oracle_audit/radiation_cloud_optics_mcica/
py/run_ours.py; constructed soundings, NOT a checkpoint) through the PRODUCTION
``compute_cloud_properties`` with the CloudConfig resolved from
config/amip/amip_production.yaml by run_amip + model_driver, and the McICA in-cloud
paths (``subcolumns.in_cloud_paths``).  Sizes are pre-clipped to the LUT range for
the oracle (RRTMGP raises out of range; legoESM clips) -- documented contract
difference.

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu, PYTHONPATH incl. packages):
    bash build.sh <rte-rrtmgp@v1.8> <conda env> <build dir>
    python make_fixture.py <build dir>/oracle_driver <rte-rrtmgp@v1.8> <rrtmgp-data@v1.8.2> <out.npz>
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(REPO, "scripts/run"))

from legoesm import constants  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0  # noqa: E402
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties  # noqa: E402
from legoesm.atmosphere.physics.clouds import subcolumns  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_cloud_optics  # noqa: E402

NLEV = 32
LUTDIR = os.path.join(REPO, "packages/atmosphere/legoesm/atmosphere/physics/radiation/"
                      "rrtmgp/optics/rrtmgp_data")
CAM_RHOI, CAM_RHOWS = 500.0, 917.0  # CAM micro_mg_utils.F90:126,128 (oracle constants)


def production_cloud_config():
    import run_amip
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.driver import model_driver as md
    yaml = os.path.join(REPO, "config/amip/amip_production.yaml")
    parser = run_amip.build_arg_parser()
    keys = load_yaml_config(yaml, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    argv = ["--config", yaml]
    for flag in ("--forcing-path", "--sic-path", "--ic-path", "--solar-file", "--ozone-file",
                 "--ghg-file", "--aerosol-file", "--aerosol-ccn-file", "--volcanic-aerosol-file",
                 "--topography", "--surfdata", "--clm-surfdata-path", "--land-mask-file",
                 "--subgrid-orography-file"):
        argv += [flag, "/nonexistent/dummy.nc"]  # machine path flags; nothing is opened
    args = run_amip._postprocess_args(parser.parse_args(argv), parser, argv)
    cfg = run_amip.build_config_from_args(args)
    return md._standalone_cloud_config(cfg, cfg.cloud_scheme, allow_convective_cloud=True)


def column(lat_deg, ps, Ts, T_trop, rh_bl, clouds, n_ccn):
    p_half = np.asarray(CAM6_L32_HYAI) * CAM6_L32_P0 + np.asarray(CAM6_L32_HYBI) * ps
    p_full = 0.5 * (p_half[1:] + p_half[:-1])
    T = np.maximum(Ts * (p_full / ps) ** (constants.R_d * 6.5e-3 / constants.g), T_trop)
    T = np.where(p_full < 1.0e4, T_trop + 8.0 * np.log10(1.0e4 / p_full), T)
    rh = np.clip(rh_bl * (p_full / ps) ** 2, 0.02, 1.0)
    q_c, q_i, n_i, alst = (np.zeros(NLEV) for _ in range(4))
    for (p_lo, p_hi, qc, qi, ni, a, rhc) in clouds:
        m = (p_full >= p_lo) & (p_full <= p_hi)
        q_c[m], q_i[m], n_i[m], alst[m], rh[m] = qc, qi, ni, a, rhc
    q_v = rh * np.asarray(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p_full)))
    return dict(lat=np.deg2rad(lat_deg), p_half=p_half, p_full=p_full, T=T, q_v=q_v, q_c=q_c,
                q_i=q_i, n_i=n_i, alst=alst, n_ccn=np.full(NLEV, n_ccn))


def build_columns():
    """The 5 tier-1 audit columns (names, list of column dicts); also imported by
    scripts/validate/oracle/reff_incloud/make_fixture.py."""
    names = ["trop_deep", "subtrop_sc", "midlat_frontal", "polar", "edge"]
    cols = [
        column(5.0, 101000.0, 301.0, 192.0, 0.85, [
            (60000., 85000., 3.0e-4, 0.0, 0.0, 0.40, 1.0),
            (40000., 55000., 1.0e-4, 5.0e-5, 1.0e5, 0.20, 1.0),
            (15000., 35000., 0.0, 8.0e-5, 2.0e5, 0.0, 1.05),
            (11000., 13000., 0.0, 5.0e-6, 2.0e4, 0.0, 1.0)], 8.0e7),
        column(25.0, 101500.0, 291.0, 200.0, 0.80, [(90000., 95500., 4.0e-4, 0.0, 0.0, 0.95, 1.0)], 1.0e8),
        column(50.0, 99000.0, 283.0, 215.0, 0.85, [
            (80000., 90500., 2.0e-4, 0.0, 0.0, 0.70, 1.0),
            (50000., 70000., 5.0e-5, 6.0e-5, 8.0e4, 0.30, 1.0),
            (25000., 40000., 0.0, 2.0e-5, 5.0e4, 0.0, 1.1)], 2.5e8),
        column(78.0, 100500.0, 252.0, 210.0, 0.90, [
            (90000., 96000., 5.0e-5, 2.0e-6, 3.0e3, 0.60, 1.0),
            (50000., 85000., 0.0, 1.0e-5, 1.0e4, 0.0, 1.0)], 5.0e7),
        column(0.0, 100000.0, 295.0, 195.0, 0.8, [], 1.0e8),
    ]
    e = cols[4]
    k = lambda p: int(np.argmin(np.abs(e["p_full"] - p)))  # noqa: E731
    for p, kq, v, a in ((95000., "q_c", 1.0e-15, 0.5), (92000., "q_c", 1.0e-12, 0.5),
                        (88000., "q_c", 2.0e-3, 1.0), (80000., "q_c", 1.0e-6, 1.0e-90),
                        (70000., "q_c", 5.0e-5, 0.0), (60000., "q_c", 1.0e-4, 0.002)):
        e[kq][k(p)] = v
        e["alst"][k(p)] = a
    e["q_i"][k(30000.)] = 3.0e-5; e["n_i"][k(30000.)] = 0.0    # ice with no number (r above LUT)
    e["q_i"][k(25000.)] = 1.0e-6; e["n_i"][k(25000.)] = 1.0e9  # tiny crystals (r below LUT)
    return names, cols


def main():
    ORACLE, RTE, DATA, OUT = sys.argv[1:5]
    names, cols = build_columns()
    S = lambda key: np.stack([c[key] for c in cols])  # noqa: E731

    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    ccfg = production_cloud_config()
    p_half = S("p_half")
    props = compute_cloud_properties(
        T=jnp.asarray(S("T")), p_full=jnp.asarray(S("p_full")), q_v=jnp.asarray(S("q_v")),
        dp=jnp.asarray(np.diff(p_half, axis=1)), config=ccfg, q_cloud=jnp.asarray(S("q_c")),
        q_ice=jnp.asarray(S("q_i")), n_ice=jnp.asarray(S("n_i")), n_cloud=jnp.asarray(S("n_ccn")),
        conv_precip=None, cloud_fraction_override=jnp.asarray(S("alst")),
        lat=jnp.asarray(np.array([c["lat"] for c in cols])), conv_mass_flux_up=None,
        conv_icwmr=None, p_half=jnp.asarray(p_half), clubb_config=CLUBBConfig())
    cpl, cpi = (np.asarray(a) for a in subcolumns.in_cloud_paths(
        props.cloud_fraction, props.lwp, props.iwp))
    r_liq, r_ice = np.asarray(props.r_eff_liq), np.asarray(props.r_eff_ice)

    out = {}
    for tag in ("lw", "sw"):
        lut_path = os.path.join(LUTDIR, f"cloudysky_{tag}.nc")
        up = os.path.join(DATA, f"rrtmgp-clouds-{tag}.nc")
        h = [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (lut_path, up)]
        assert h[0] == h[1], f"bundled {tag} table is not rrtmgp-data {up}"
        out[f"sha256_{tag}"] = h[0]
        lut = lookup_cloud_optics.from_data_file(lut_path)
        nb = int(lut.ext_liq.shape[0])
        rel = np.clip(r_liq * 1e6, float(lut.radius_liq_lower), float(lut.radius_liq_upper))
        dei = np.clip(r_ice * 1e6 * CAM_RHOI / CAM_RHOWS * 2.0,  # CAM deffi
                      float(lut.diameter_ice_lower), float(lut.diameter_ice_upper))
        ncol, nlay = cpl.shape
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "dims.txt"), "w") as f:
                f.write(f"{ncol} {nlay} {nb}\n")
            with open(os.path.join(d, "in.bin"), "wb") as f:
                for a in (cpl * 1e3, rel, cpi * 1e3, dei):  # kg/m2 -> g/m2
                    np.asarray(a, "<f8").ravel(order="F").tofile(f)
            run = subprocess.run([ORACLE, lut_path, d, "2"], capture_output=True, text=True, check=True)
            assert "ORACLE_DONE" in run.stdout, run.stdout
            raw = np.fromfile(os.path.join(d, "out.bin"), dtype="<f8")
        n = ncol * nlay * nb
        blk = [raw[i * n:(i + 1) * n].reshape(ncol, nlay, nb, order="F") for i in range(7)]
        assert raw.size == 7 * n
        if tag == "lw":
            out["lw_tau_abs"] = blk[0]
        else:
            out.update(sw_tau=blk[1], sw_ssa=blk[2], sw_g=blk[3],
                       sw_tau_ds=blk[4], sw_ssa_ds=blk[5], sw_g_ds=blk[6])

    sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    tagof = lambda p: subprocess.run(["git", "-C", p, "describe", "--tags"],  # noqa: E731
                                     capture_output=True, text=True).stdout.strip()
    meta = dict(
        oracle=f"rte-rrtmgp {tagof(RTE)} {sha(RTE)}",
        oracle_note="mo_cloud_optics_rrtmgp%cloud_optics LUT, ice roughness 2; LW 1scl, SW 2str+delta_scale; "
                    "ice size = CAM6 deffi (micro_mg2_0.F90:2795, cam_cesm2_1_rel_60)",
        rrtmgp_data=f"{tagof(DATA)} {sha(DATA)}",
        legoesm=sha(REPO),
        generator="scripts/validate/oracle/rrtmgp_cloud_optics/{build.sh,oracle_driver.F90,make_fixture.py}",
        columns=names)
    np.savez_compressed(OUT, cpl=cpl, cpi=cpi, r_liq=r_liq, r_ice=r_ice, meta=json.dumps(meta), **out)
    print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
