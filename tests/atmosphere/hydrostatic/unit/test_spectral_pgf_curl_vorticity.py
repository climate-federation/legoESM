"""Spectral PE: the pressure-gradient force must enter the VORTICITY equation.

Momentum carries -R_d T' grad(ln p_s) (sigma levels).  Its curl, the
solenoidal torque -R_d grad T' x grad ln p_s, was dropped from the vorticity
tendency (only its divergence reached the divergence equation).

Analytic check at rest (u = v = 0, so every other vorticity source vanishes),
with a level-dependent amplitude A_k so a level mix-up fails:
- "zonal_lnps":  T' = A_k cos(phi) cos(lambda), ln p_s = ln p0 + B sin(phi)
  -> d(zeta)/dt = +R_d A_k B cos(phi) sin(lambda) / a**2   (exercises oc2(Fy))
- "zonal_T":     T' = A_k sin(phi), ln p_s = ln p0 + B cos(phi) cos(lambda)
  -> d(zeta)/dt = -R_d A_k B cos(phi) sin(lambda) / a**2   (exercises dmu(Fx))
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig, isothermal_rest_state_spectral, spectral_pe_tendencies)
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis_3d
from legoesm.grids.vertical import create_sigma_coordinate


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


@pytest.mark.parametrize("case", ["zonal_lnps", "zonal_T"])
def test_pgf_curl_reaches_vorticity_tendency(case):
    grid = create_gaussian_grid(21, radius=constants.R_earth)
    coord = create_sigma_coordinate(8)
    cfg = SpectralPEConfig()
    state = isothermal_rest_state_spectral(
        grid, coord, T_init=cfg.si_T_ref, perturbation_amplitude=0.0)
    lat = np.asarray(grid.lat2d)
    lon = np.asarray(grid.lon2d)
    nlev = state.T_hat.data.shape[-1]
    A = 5.0 * (np.arange(nlev) + 1.0) / nlev  # K, level-dependent
    B = 0.02  # ln p_s amplitude
    if case == "zonal_lnps":
        T_pat, lnps_pat, sign = np.cos(lat) * np.cos(lon), B * np.sin(lat), 1.0
    else:
        T_pat, lnps_pat, sign = np.sin(lat), B * np.cos(lat) * np.cos(lon), -1.0
    T_grid = cfg.si_T_ref + A[None, None, :] * T_pat[..., None]
    T_hat = jnp.stack([sh_analysis(grid, jnp.asarray(T_grid[..., k]))
                       for k in range(nlev)], axis=-1)
    state = state._replace(
        T_hat=state.T_hat.replace(data=T_hat.astype(state.T_hat.data.dtype)),
        lnps_hat=state.lnps_hat.replace(data=sh_analysis(
            grid, jnp.asarray(np.log(constants.p_ref) + lnps_pat)).astype(
                state.lnps_hat.data.dtype)))

    t = spectral_pe_tendencies(state, grid, coord, cfg, None)
    dvor = np.asarray(jnp.real(sh_synthesis_3d(grid, t.vor_hat.data)))
    shape = constants.R_d * B * np.cos(lat) * np.sin(lon) / grid.radius**2
    for k in range(nlev):
        expected = sign * A[k] * shape
        np.testing.assert_allclose(
            dvor[..., k], expected, atol=1e-6 * np.abs(expected).max())
