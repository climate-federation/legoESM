"""Fast, input-free controls: each ice-ocean top-drag consumer is live.

The synthetic VORTEX card (no deck files) hosts the G5 top-drag path with a
traced ``jax.jit`` coefficient.  Removing any consumer makes its test fail:
the coefficient map is checked against NEMO's ``dynspg_ts.f90:1431-1432``;
each downstream consumer (top baroclinic residual, external-substep drag,
``dyn_zdf`` top RHS/diagonal) must move the step when omitted.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

CASE = "VORTEX-zco"


@pytest.fixture(scope="module")
def fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def card(fp64):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    return build_nemo_testcase_card(CASE)


def _rcdu(card):
    ny, nx = card.recipe.initial_state.eta.data.shape
    rng = np.random.default_rng(5)
    return jnp.asarray(-1.5e-3 * rng.random((ny, nx)))


def _step(card, rcdu, **omit):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    cfg = card.recipe.model_config._replace(implicit_ice_ocean_top_drag=True)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**omit) if omit else None)
    s0 = card.recipe.initial_state
    zero = jnp.zeros_like(s0.eta.data)
    run = jax.jit(lambda r: model.step(s0, dt=card.dt_s, surface_forcing=OceanSurfaceForcing(
        tau_x=zero, tau_y=zero, ice_ocean_top_drag_rCdu=r)))
    return jax.device_get(run(rcdu))


def _max_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(getattr(a, f).data)
                                   - np.asarray(getattr(b, f).data))))
               for f in ("u", "v", "eta"))


@pytest.fixture(scope="module")
def base(card):
    return _step(card, _rcdu(card))


def test_coefficient_faces_are_nemo_under_traced_jit(card):
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_ice_ocean_top_drag_coeff_faces,
    )
    r = _rcdu(card)
    cu, cv, *_ = jax.jit(lambda x: nemo_ice_ocean_top_drag_coeff_faces(
        x, card.recipe.grid))(r)
    r = np.asarray(r)
    # dynspg_ts.f90:1431-1432 with rCdU_bot = 0: r1_2*((0+0) + (r(i+1)+r(i)))
    want_u = 0.5 * (0.0 + (r[:, 1:] + r[:, :-1]))
    want_v = 0.5 * (0.0 + (r[1:, :] + r[:-1, :]))
    np.testing.assert_array_equal(np.asarray(cu)[:, 1:-1], want_u)
    np.testing.assert_array_equal(np.asarray(cv)[1:-1, :], want_v)
    assert np.all(np.asarray(cu)[:, 1:-1] < 0.0)


def test_coefficient_reaches_the_step(card, base):
    zero = _step(card, jnp.zeros_like(_rcdu(card)))
    assert _max_diff(base, zero) > 1.0e-6


@pytest.mark.parametrize("consumer", ["residual", "substeps", "zdf"])
def test_each_consumer_moves_the_step_when_omitted(card, base, consumer):
    omitted = _step(card, _rcdu(card), **{f"omit_ice_top_drag_{consumer}": True})
    assert _max_diff(omitted, base) > 1.0e-6


def test_stage3_salt_option_refuses_missing_sfx(card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    cfg = card.recipe.model_config._replace(rk3_stage3_salt_nemo_written=True)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    with pytest.raises(ValueError, match="salt_flux_sfx"):
        model._step_impl(card.recipe.initial_state, card.dt_s)
