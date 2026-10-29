"""Physics wind tendencies reach the spectral state through the exact inverse.

The plain Bourke oc2/dmu analysis is not a left-inverse of the dycore's wind
synthesis on the pole rows: a pole-row tendency comes back ~6x larger each
pass, and the CAM macmic sub-cycle compounds that into a NaN (WB eval,
2-Apr init, step 39). The turbulence, gravity-wave-drag and convective
momentum bridges must therefore use ``vordiv_from_uv_exact_3d``; its own
round-trip property is pinned in ``test_spectral_wind_roundtrip.py``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

_N_MAX, _N_LEV, _DT = 8, 6, 1800.0


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=285.0, p_s_init=1.0e5, perturbation_amplitude=0.5,
        tracers={"q_v": jnp.full(shp, 6.0e-3), "q_c": jnp.zeros(shp)})
    return grid, sigma, state


def _spy(monkeypatch, module):
    from legoesm.grids.gaussian import vordiv_from_uv_exact_3d as real
    calls = []

    def spy(grid, u, v):
        calls.append(u.shape)
        return real(grid, u, v)
    monkeypatch.setattr(module, "vordiv_from_uv_exact_3d", spy, raising=False)
    return calls


def _build(kind):
    if kind == "turbulence":
        from legoesm.atmosphere.physics.turbulence import integration as m
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        return m, lambda: m.make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), "spectral_pe", _DT)
    if kind == "gravity_wave_drag":
        from legoesm.atmosphere.physics.gravity_wave_drag import integration as m
        from legoesm.atmosphere.physics.combined import PhysicsConfig
        cfg = type(PhysicsConfig().gravity_wave_drag)(scheme="rayleigh")
        return m, lambda: m.make_gwd_physics(cfg, "spectral_pe", _DT)
    from legoesm.atmosphere.physics.convection import integration as m
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    cc = ConvectionConfig(scheme="zhang_mcfarlane")
    cc = cc._replace(zhang_mcfarlane=cc.zhang_mcfarlane._replace(land_fraction="none"))
    return m, lambda: m.make_convection_physics(cc, "spectral_pe", _DT)


@pytest.mark.parametrize("kind", ["turbulence", "gravity_wave_drag",
                                  "convection"])
def test_wind_tendency_goes_through_the_exact_inverse(setup, monkeypatch, kind):
    grid, sigma, state = setup
    module, make = _build(kind)
    calls = _spy(monkeypatch, module)
    out = make()(state, grid, sigma)
    tend = out[0] if isinstance(out, tuple) else out
    assert calls == [(grid.n_lat, grid.n_lon, _N_LEV)], (kind, calls)
    assert np.all(np.isfinite(np.asarray(tend.vor_hat.data)))
