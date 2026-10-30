"""The lat-lon model's vertical-mixing EOS uses the run's reference density
exactly as its dynamics does (#2243): ``rho0 = config.rho_0`` on the
geometric-depth path, the module default otherwise.

``rho0`` only enters the NEMO EOS80/TEOS-10 polynomials (their depth
reconstruction ``zh = p/(rho0*g)``), so the probe evaluates every captured EOS
at a deep pressure with ``eos="nemo_eos80"`` and a pinned ``rho_0 = 1026``.
Four builds are covered: the closures' EOS handed to the physics pipeline, the
implicit K-profile fallback, the internal-wave add-on and double diffusion.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_RHO0_RUN = 1026.0
_T, _S, _P = 10.0, 35.0, 4.0e7          # ~4 km: the depth term is large


def _rho(eos_fn):
    return float(eos_fn(jnp.asarray(_T), jnp.asarray(_S), jnp.asarray(_P)))


def _expected(eos_depth):
    from legoesm.ocean.eos import make_eos_fn
    kw = {"rho0": _RHO0_RUN} if eos_depth == "geometric" else {}
    return _rho(make_eos_fn(eos="nemo_eos80", **kw))


def test_probe_sees_the_reference_density():
    """Instrument control: the probe point separates the two rho0 values."""
    from legoesm.ocean.eos import make_eos_fn
    assert abs(_expected("geometric") - _rho(make_eos_fn(eos="nemo_eos80"))) > 1e-6


def _model(eos_depth, physics, monkeypatch=None, spy_physics=None):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as mod
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    if spy_physics is not None:
        real = mod.make_ocean_physics

        def spy(*a, **kw):
            spy_physics.append(kw.get("eos_fn"))
            return real(*a, **kw)

        monkeypatch.setattr(mod, "make_ocean_physics", spy)
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        eos="nemo_eos80", eos_depth=eos_depth, rho_0=_RHO0_RUN,
        implicit_vertical_mixing=True, enable_runtime_checks=False,
        physics=physics)
    model = mod.LatLonCGridOceanModel(grid, z, cfg)
    state = rest_state_latlon_cgrid_ocean(grid, z, H_max=4000.0,
                                          land_lat_threshold=80.0)
    return model, state


def _physics(vm):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import (
        LateralMixingConfig,
    )
    return OceanPhysicsConfig(
        vertical_mixing=vm,
        lateral_mixing=LateralMixingConfig(scheme="none"))


@pytest.mark.parametrize("eos_depth", ["geometric", "insitu"])
def test_closure_eos_reference_density(monkeypatch, eos_depth):
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    seen = []
    _model(eos_depth, _physics(VerticalMixingConfig(scheme="richardson")),
           monkeypatch, seen)
    assert len(seen) == 1
    assert _rho(seen[0]) == _expected(eos_depth)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("eos_depth", ["geometric", "insitu"])
def test_fallback_eos_reference_density(monkeypatch, eos_depth):
    import legoesm.ocean.physics.vertical_mixing as vmix
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    model, state = _model(
        eos_depth, _physics(VerticalMixingConfig(scheme="richardson")))
    seen = []

    def spy(*a, eos_fn=None, **kw):
        seen.append(eos_fn)
        raise _Stop

    monkeypatch.setattr(vmix, "compute_vertical_K_profiles", spy)
    with pytest.raises(_Stop):
        model._apply_implicit_vertical_mixing(state, 1800.0, None)
    assert len(seen) == 1
    assert _rho(seen[0]) == _expected(eos_depth)


@pytest.mark.parametrize("eos_depth", ["geometric", "insitu"])
def test_iwm_and_ddm_eos_reference_density(monkeypatch, eos_depth):
    """Fast path (physics-provided K): the internal-wave add-on and double
    diffusion each build an EOS."""
    from legoesm.ocean.physics.vertical_mixing import k_profiles
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        DoubleDiffusionConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig,
    )
    vm = VerticalMixingConfig(scheme="richardson")._replace(
        iwm=IWMConfig(enabled=True),
        ddm=DoubleDiffusionConfig()._replace(enabled=True))
    model, state = _model(eos_depth, _physics(vm))
    seen = {}

    def iwm_spy(*a, eos_fn=None, **kw):
        seen["iwm"] = eos_fn
        return jnp.zeros(state.T.data.shape[:2] + (state.T.data.shape[2] - 1,),
                         dtype=state.T.data.dtype)

    def ddm_spy(*a, eos_fn=None, **kw):
        seen["ddm"] = eos_fn
        raise _Stop

    monkeypatch.setattr(k_profiles, "iwm_K_profile", iwm_spy)
    monkeypatch.setattr(k_profiles, "ddm_K_profile", ddm_spy)
    shape = state.T.data.shape[:2] + (state.T.data.shape[2] - 1,)
    K = jnp.full(shape, 1e-5, dtype=state.T.data.dtype)
    with pytest.raises(_Stop):
        model._apply_implicit_vertical_mixing(
            state, 1800.0, None, K_v_phys=K, A_v_phys=K)
    assert set(seen) == {"iwm", "ddm"}
    want = _expected(eos_depth)
    assert _rho(seen["iwm"]) == want
    assert _rho(seen["ddm"]) == want
