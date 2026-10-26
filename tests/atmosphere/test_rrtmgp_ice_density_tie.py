"""Ice optics size uses the CONFIGURED bulk ice density (oracle-audit follow-up, row 37).

CAM6 uses ONE ``rhoi`` for the ice PSD and the optics size ``deffi = effi*rhoi/rhows*2``
(cam_cesm2_1_rel_60 micro_mg2_0.F90:2795).  legoESM's ice PSD reads
``CloudConfig.rho_cloud_ice``; the RRTMGP cloud optics convert at CAM's 500
(``cloud_optics.RHO_ICE_BULK_CAM_MG2``, certified in test_cloud_optics_faithful), so
``solve_columns(cloud_rho_ice=rho)`` hands the optics ``r*rho/500``: table size
``2 r rho/917``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.radiation import integration as rad_int
from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

NCOL, NLEV = 2, 8


def _inputs():
    p_half = jnp.broadcast_to(jnp.array(
        [225.5, 503.0, 1.5e3, 5e3, 1.5e4, 3.5e4, 6e4, 8.5e4, 1e5]), (NCOL, NLEV + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.broadcast_to(
        jnp.array([255., 240., 225., 215., 220., 250., 275., 290.]), (NCOL, NLEV))
    ice = jnp.zeros((NCOL, NLEV)).at[:, 3:5].set(0.02)
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((NCOL,), 290.0),
                q_v=jnp.full((NCOL, NLEV), 3e-6), cos_zenith=jnp.full((NCOL,), 0.6),
                cloud_path_liq=jnp.zeros((NCOL, NLEV)), cloud_path_ice=ice,
                cloud_r_eff_liq=jnp.full((NCOL, NLEV), 10e-6))


def _flat(out):
    return np.concatenate([np.ravel(np.asarray(x)) for x in
                           (out.sw_flux_up, out.sw_flux_down, out.lw_flux_up,
                            out.lw_flux_down)])


def test_solve_columns_ice_size_uses_configured_density():
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=True))
    inp = _inputs()
    r = jnp.full((NCOL, NLEV), 40e-6)
    base = _flat(solver.solve_columns(**inp, cloud_r_eff_ice=r))
    at500 = _flat(solver.solve_columns(**inp, cloud_r_eff_ice=r, cloud_rho_ice=500.0))
    assert np.array_equal(base, at500)  # CAM rhoi: production bit-identical
    tied = _flat(solver.solve_columns(**inp, cloud_r_eff_ice=r, cloud_rho_ice=700.0))
    ref = _flat(solver.solve_columns(**inp, cloud_r_eff_ice=r * 700.0 / 500.0))
    np.testing.assert_allclose(tied, ref, rtol=1e-12, atol=0.0)
    assert not np.allclose(tied, base, rtol=1e-6)  # the density reaches the optics


class _Reached(Exception):
    pass


def test_backend_passes_cloud_config_density():
    seen = {}

    class _Spy:
        def solve_columns(self, **kw):
            seen.update(kw)
            raise _Reached

    ncol, nlev = 2, 3
    p_half = jnp.broadcast_to(jnp.array([2e4, 5e4, 8e4, 1e5]), (ncol, nlev + 1))
    cfg = RadiationConfig(scheme="rrtmgp", cloud_scheme="sundqvist",
                          cloud_config=CloudConfig(scheme="sundqvist", rho_cloud_ice=700.0))
    with pytest.raises(_Reached):
        rad_int._call_radiation_backend(
            cfg, T=jnp.full((ncol, nlev), 260.0),
            p_full=0.5 * (p_half[:, 1:] + p_half[:, :-1]), p_half=p_half,
            sfc_temperature=jnp.full((ncol,), 290.0), lat=jnp.zeros(ncol),
            q_v=jnp.full((ncol, nlev), 1e-3), insolation=jnp.full((ncol,), 400.0),
            q_cloud=jnp.full((ncol, nlev), 1e-5), q_ice=jnp.full((ncol, nlev), 1e-5),
            rrtmgp_solver=_Spy())
    assert seen["cloud_rho_ice"] == 700.0
