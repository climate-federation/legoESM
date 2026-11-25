"""Fixture for the CAM6 conv_water_4rad oracle (tests/atmosphere/oracle_fixtures/cam6_conv_water.npz).

usage: python make_fixture.py <oracle build dir> <A2 smoke dump dir> <out.npz>

Columns: 12 from an A2 C1 smoke dump (feed_00080.5000.npz: the 4 warm-pool
and 4 SE Pacific columns with the most deep convective cloud, 4 more spread
over the globe) plus 8 synthetic columns built to hit every branch of the
Fortran (no stratus, stratus under frac_limit, deepcu at the limit, tiny ZM
water, ice-only and liquid-only stratus).  ast and deepcu are legoESM's
cam6_clubb values for the dumped state (they are CAM's pbuf AST and DP_FRAC);
the rest is the dumped state.  The oracle runs CAM's routine on exactly these
inputs, so the comparison tests conv_water_4rad alone.
"""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.clouds.cloud_fraction import (  # noqa: E402
    cam6_deep_convective_fraction, cam6_ice_stratus_fraction)
from legoesm.atmosphere.physics.clouds.config import CloudConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402

FRAC_LIMIT = 0.01


def dumped_columns(dump_dir):
    f = np.load(os.path.join(dump_dir, "feed_00080.5000.npz"))
    g = np.load(os.path.join(dump_dir, "grid.npz"))
    lat = np.asarray(f["lat_deg"], float)
    lon = np.degrees(np.asarray(g["lonCell"], float)) % 360.0
    p_half = (g["A_half"][None] * constants.p_ref
              + g["B_half"][None] * np.asarray(f["p_s"], float)[:, None])
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = np.asarray(f["T"], float)
    q_v = np.asarray(f["q_v"], float)
    q_c = np.maximum(np.asarray(f["q_c"], float), 0.0)
    q_i = np.maximum(np.asarray(f["q_i"], float), 0.0)
    mf = np.asarray(f["conv_mass_flux_up"], float)
    icw = np.asarray(f["conv_icwmr"], float)
    cfg = CloudConfig(scheme="cam6_clubb")
    clubb = CLUBBConfig()
    aist = np.asarray(cam6_ice_stratus_fraction(
        jnp.asarray(q_v), jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(q_i),
        jnp.asarray(np.deg2rad(lat)), cfg, jnp.asarray(p_half[:, :-1]),
        trop_cloud_top_press=clubb.trop_cloud_top_press,
        trop_cloud_taper_lnp_width=clubb.trop_cloud_taper_lnp_width))
    deepcu = np.asarray(cam6_deep_convective_fraction(jnp.asarray(mf), jnp.asarray(icw), cfg))
    ast = np.maximum(np.clip(np.asarray(f["cloud_fraction"], float), 0, 1), aist)
    wp = (np.abs(lat) < 10) & (lon > 130) & (lon < 170)
    sep = (lat > -30) & (lat < -10) & (lon > 260) & (lon < 290)
    score = deepcu.sum(1)
    pick = list(np.where(wp)[0][np.argsort(-score[wp])][:4])
    pick += list(np.where(sep)[0][np.argsort(-score[sep] - 1e-3 * ast[sep].sum(1))][:4])
    rng = np.random.default_rng(3)
    pick += list(rng.choice(np.setdiff1d(np.arange(lat.size), pick), 4, replace=False))
    pick = np.array(pick)
    pdel = p_half[:, 1:] - p_half[:, :-1]
    return dict(ls_liq=q_c[pick], ls_ice=q_i[pick], pdel=pdel[pick], ast=ast[pick],
                dp_frac=deepcu[pick], dp_icwmr=icw[pick]), pick


def synthetic_columns(nlev, n=8, seed=11):
    r = np.random.default_rng(seed)
    sh = (n, nlev)
    ls_liq = r.uniform(0, 3e-4, sh) * (r.random(sh) < 0.6)
    ls_ice = r.uniform(0, 1e-4, sh) * (r.random(sh) < 0.6)
    ast = r.uniform(0, 1, sh) * (r.random(sh) < 0.7)
    dp_frac = r.uniform(0, 0.6, sh) * (r.random(sh) < 0.6)
    dp_icwmr = r.uniform(0, 8e-4, sh) * (r.random(sh) < 0.7)
    ast[0] = 0.0; ls_liq[0] = 0.0; ls_ice[0] = 0.0          # cumulus only, no stratus
    ast[1, ::2] = r.uniform(0, FRAC_LIMIT, nlev // 2)      # stratus under frac_limit
    dp_frac[2, ::3] = FRAC_LIMIT                            # deepcu exactly at the limit
    dp_icwmr[3, ::2] = 5e-13                                # ZM water under ic_limit
    ls_liq[4] = 0.0                                         # ice-only stratus
    ls_ice[5] = 0.0                                         # liquid-only stratus
    dp_frac[6] = 0.0                                        # no cumulus
    pdel = np.tile(np.diff(np.linspace(200.0, 101000.0, nlev + 1)), (n, 1))
    return dict(ls_liq=ls_liq, ls_ice=ls_ice, pdel=pdel, ast=ast,
                dp_frac=dp_frac, dp_icwmr=dp_icwmr)


def main(build_dir, dump_dir, out):
    dumped, pick = dumped_columns(dump_dir)
    nlev = dumped["ast"].shape[1]
    syn = synthetic_columns(nlev)
    keys = ("ls_liq", "ls_ice", "pdel", "ast", "dp_frac", "dp_icwmr")
    inp = {k: np.concatenate([dumped[k], syn[k]]) for k in keys}
    ncol = inp["ast"].shape[0]
    zeros = np.zeros((ncol, nlev))
    rei = np.full((ncol, nlev), 25.0)
    with open(os.path.join(build_dir, "cw_nl.txt"), "w") as fh:
        fh.write(f"&conv_water_nl\n conv_water_in_rad = 1\n conv_water_frac_limit = {FRAC_LIMIT}\n/\n")
    with open(os.path.join(build_dir, "cw_in.bin"), "wb") as fh:
        np.array([ncol, nlev], dtype="<i4").tofile(fh)
        for a in (*[inp[k] for k in keys], zeros, zeros, rei):
            np.ascontiguousarray(a, dtype="<f8").tofile(fh)
    subprocess.run(["./cw_oracle"], cwd=build_dir, check=True)
    o = np.fromfile(os.path.join(build_dir, "cw_out.bin"), dtype="<f8").reshape(2, ncol, nlev)
    np.savez(out, **inp, frac_limit=FRAC_LIMIT, totg_liq=o[0], totg_ice=o[1],
             dump_cells=pick, n_dumped=len(pick))
    print(f"wrote {out}: {ncol} columns x {nlev} levels, "
          f"max totg_liq {o[0].max():.3e}, max totg_ice {o[1].max():.3e}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
