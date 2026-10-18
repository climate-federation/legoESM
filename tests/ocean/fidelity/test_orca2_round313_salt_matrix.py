"""Round 313: rung 10's literal salt matrix (trazdf.f90:175, :185, :203), one production-JIT step."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as m
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.physics import vertical_mixing as vmix
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round103_rung0_ladder_gate import (  # noqa: E501
    _zero_forcing,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round313_iwm_ddm_d122_gate import (  # noqa: E501
    BOTH,
    IWM,
    _with,
    carried,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")


@pytest.fixture(scope="module")
def card():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    assert jax.numpy.ones(1).dtype == jax.numpy.float64
    yield build_orca2_zps_card(DECK)
    set_policy(before)


def _one_step(card, monkeypatch):
    """Production-JIT step; returns the ZDF trace and every literal avs the K profiles produced."""
    avs = []
    real = vmix.compute_vertical_K_profiles

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        if kwargs.get("return_salt_K"):
            jax.debug.callback(lambda k: avs.append(np.asarray(k)), out[-1])
        return out

    monkeypatch.setattr(vmix, "compute_vertical_K_profiles", spy)
    r = card.recipe
    model = m.LatLonCGridOceanModel(
        r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing,
        _nemo_ws_test_hooks=m._NEMOWSRK3TestHooks(tracer_zdf_trace=True))
    shape = tuple(r.initial_state.eta.data.shape)
    freshwater, surface = _zero_forcing(shape)
    # Zero fluxes; the card's surface statements need these channels present.
    zero = surface.q_net
    surface = surface._replace(
        chl=jax.numpy.full(shape, 0.1), q_nonsolar=zero, salt_flux_sfx=zero,
        ice_concentration=zero, runoff_tracer_content=(zero, zero),
        ice_ocean_top_drag_rCdu=zero)
    trace = jax.device_get(model.step(
        r.initial_state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    assert isinstance(trace, m._NEMOWSTracerZDFTrace)
    wet = np.asarray(r.z_coord.is_active, dtype=np.float64)[..., 1:]
    return trace.solve, avs, wet


def test_ddm_on_salt_takes_literal_avs_plus_isoneutral_term(card, monkeypatch):
    assert carried(card) == BOTH
    solve, avs, wet = _one_step(card, monkeypatch)
    assert avs and all(np.array_equal(a, avs[0]) for a in avs)  # one avs per step
    iso = np.asarray(solve.isoneutral_K)
    assert np.abs(iso * wet).max() > 0.0
    assert not np.array_equal(avs[0], np.asarray(solve.heat_K))
    np.testing.assert_array_equal(np.asarray(solve.salt_K), (avs[0] + iso) * wet)


def test_ddm_off_salt_reuses_heat_matrix(card, monkeypatch):
    off = _with(card, IWM)
    solve, avs, _ = _one_step(off, monkeypatch)
    assert avs  # zdf_iwm's avs is still built (zdfiwm.f90:314) ...
    assert solve.salt_K is None  # ... and never read (trazdf.f90:175)


def test_hybrid_ddm_with_literal_iwm_refuses_at_configuration(card):
    hybrid = _with(card, ("nemo_literal", True, True, "hybrid"))
    r = hybrid.recipe
    with pytest.raises(ValueError, match="hybrid double diffusion"):
        m.LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                iwm_forcing=r.iwm_forcing)
