"""Synthetic CAM6 cloud columns shared by the C1 cloud-radiation tests.

Tropical-like soundings with CLUBB liquid fraction, stratiform liquid and ice,
a deep updraft mass flux and ZM in-cloud water, so the cam6_clubb cloud call,
the convective water, snow and droplet number paths all have work to do.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

NCOL, NLEV = 6, 16
CLUBB = CLUBBConfig()


def columns(seed: int = 0, ncol: int = NCOL, nlev: int = NLEV) -> dict:
    r = np.random.default_rng(seed)
    ps = r.uniform(98000.0, 102000.0, ncol)
    eta = np.linspace(0.02, 1.0, nlev + 1)
    p_half = eta[None] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    ts = r.uniform(290.0, 302.0, ncol)
    T = np.maximum(ts[:, None] - 45.0 * np.log(ps[:, None] / p_full), 195.0)
    q_v = 0.016 * (p_full / 1.0e5) ** 3
    cold = T < 255.0
    warm = T > 268.0
    alst = r.uniform(0.0, 0.8, (ncol, nlev)) * (r.random((ncol, nlev)) < 0.5)
    q_c = np.where(warm | (r.random((ncol, nlev)) < 0.3),
                   alst * r.uniform(1e-5, 4e-4, (ncol, nlev)), 0.0)
    q_i = np.where(~warm, r.uniform(0.0, 5e-5, (ncol, nlev)), 0.0)
    q_i[:, 0] = 0.0
    n_i = np.where(q_i > 0, r.uniform(1e4, 5e5, (ncol, nlev)), 0.0)
    mf = np.zeros((ncol, nlev + 1))
    deep = (p_half > 15000.0) & (p_half < 95000.0)
    mf[deep] = r.uniform(0.0, 0.06, deep.sum())
    mf[0] = 0.0                      # column 0: no convection
    icwmr = np.where((p_full > 15000.0) & (p_full < 95000.0),
                     r.uniform(0.0, 6e-4, (ncol, nlev)), 0.0)
    icwmr[0] = 0.0
    q_s = np.where(cold | (r.random((ncol, nlev)) < 0.3),
                   r.uniform(0.0, 3e-4, (ncol, nlev)), 0.0)
    q_s[1] = 0.0                     # column 1: no snow
    n_s = np.where(q_s > 0, r.uniform(1e3, 2e5, (ncol, nlev)), 0.0)
    n_c_tracer = r.uniform(0.0, 5e9, (ncol, nlev))   # per volume [m-3]
    lat = np.deg2rad(r.uniform(-25.0, 25.0, ncol))
    return {k: jnp.asarray(v) for k, v in dict(
        T=T, p_full=p_full, p_half=p_half, q_v=q_v, q_c=q_c, q_i=q_i,
        n_i=n_i, alst=alst, conv_mass_flux_up=mf, conv_icwmr=icwmr,
        q_s=q_s, n_s=n_s, n_c_tracer=n_c_tracer, lat=lat, ts=ts).items()}


def cloud_config(**kw) -> CloudConfig:
    base = dict(scheme="cam6_clubb", cloud_vertical_overlap_optics="mcica")
    base.update(kw)
    return CloudConfig(**base)


RRTMGP = RRTMGPConfig(include_clouds=True, gpoint_batch_size=16)


def radiation_config(cloud_cfg: CloudConfig) -> RadiationConfig:
    return RadiationConfig(scheme="rrtmgp", cloud_scheme="cam6_clubb",
                           cloud_config=cloud_cfg, rrtmgp=RRTMGP,
                           use_clubb_cloud_fraction=True)


def solver():
    """The RRTMGP instance, built outside any jit (its tables load eagerly)."""
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import _get_instance
    return _get_instance(RRTMGP)
