"""NEMO's river-runoff tracer source, as the WS-RK3 stage ladder deposits it.

``trasbc.F90``'s river-runoff block adds ``rnf_tsc * zdep`` to the tracer
right-hand side on levels ``1..nk_rnf``, with ``zdep = 1/h_rnf`` formed FIRST.
The block sits OUTSIDE ``SELECT CASE( kstg )``, so unlike the EMP/QNS block
above it, it runs at ALL THREE Runge-Kutta stages.  With neither
``ln_rnf_depth`` nor ``ln_rnf_depth_ini`` selected -- ORCA2's resolved case --
``sbcrnf.F90``'s surface arm sets ``nk_rnf = 1`` and ``h_rnf`` to the LIVE
top-cell thickness, so the whole content lands in the top cell.

These are the controls for that statement, on the small lock-exchange card:
the channel moves the step, it deposits in the top cell only, it is carried at
every stage, it uses the reciprocal-first form NEMO writes, and -- the scoping
claim the GYRE landing rests on -- withholding it leaves the step BIT-identical
rather than merely close.
"""

import numpy as np
import pytest

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.state import OceanSurfaceForcing

from tests.ocean.unit.test_nemo_ws_tracer_rk3 import _lock_model
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_lock_exchange_zco_card,
)


def _content(shape, seed=3):
    """A runoff content field with a few active river mouths."""
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float64)
    flat = field.reshape(-1)
    picks = rng.choice(flat.size, size=max(3, flat.size // 7), replace=False)
    flat[picks] = rng.uniform(1.0e-6, 4.0e-5, size=picks.size)
    return field


def _stage_sources(content_pair):
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    state = card.recipe.initial_state
    model = _lock_model(model_module._NEMOWSRK3TestHooks(
        expose_live_stage_operands=True))
    surface = (None if content_pair is None
               else OceanSurfaceForcing(runoff_tracer_content=content_pair))
    trace = model.step(state, dt=card.dt_s, surface_forcing=surface)
    rates, thicknesses = trace.stage_tracer_sources
    return card, rates, thicknesses, trace


@pytest.fixture(scope="module")
def arms():
    card = build_lock_exchange_zco_card()
    shape = np.asarray(card.recipe.initial_state.eta.data).shape
    content = (_content(shape), _content(shape, seed=11))
    return {
        "content": content,
        "on": _stage_sources(content),
        "off": _stage_sources(None),
    }


def test_the_channel_moves_every_stage(arms):
    """Non-vacuity: reverting the deposit makes the two arms identical."""
    _, on, _, _ = arms["on"]
    _, off, _, _ = arms["off"]
    for stage in range(3):
        for tracer in range(2):
            delta = np.abs(np.asarray(on[stage][tracer])
                           - np.asarray(off[stage][tracer]))
            assert float(delta.max()) > 0.0, (
                f"stage {stage + 1} tracer {tracer} did not move")


def test_the_deposit_is_the_top_cell_only(arms):
    """nk_rnf = 1: no level below the surface may receive the runoff."""
    _, on, _, _ = arms["on"]
    _, off, _, _ = arms["off"]
    for stage in range(3):
        for tracer in range(2):
            delta = (np.asarray(on[stage][tracer])
                     - np.asarray(off[stage][tracer]))
            np.testing.assert_array_equal(
                delta[..., 1:], np.zeros_like(delta[..., 1:]))


def test_the_deposit_is_the_reciprocal_first_form(arms):
    """NEMO forms ``zdep = 1/h_rnf`` and multiplies.

    ``content * (1/h)`` and ``content / h`` are different fp64 values on real
    data, so a transcription that divides instead would fail this.
    """
    card, on, thicknesses, _ = arms["on"]
    _, off, _, _ = arms["off"]
    active = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64) \
        if hasattr(card.recipe.z_coord, "is_active") else None
    for stage in range(3):
        top = np.asarray(thicknesses[stage], dtype=np.float64)[..., 0]
        zdep = 1.0 / np.maximum(top, 1.0e-10)
        for tracer in range(2):
            deposit = arms["content"][tracer] * zdep
            if active is not None:
                deposit = deposit * active[..., 0]
            got = (np.asarray(on[stage][tracer])[..., 0]
                   - np.asarray(off[stage][tracer])[..., 0])
            np.testing.assert_allclose(got, deposit, rtol=0.0, atol=1.0e-18)
    # And the divided form is genuinely a different number somewhere, so the
    # check above is not satisfied by both spellings at once.
    top = np.asarray(thicknesses[0], dtype=np.float64)[..., 0]
    a = arms["content"][0] * (1.0 / np.maximum(top, 1.0e-10))
    b = arms["content"][0] / np.maximum(top, 1.0e-10)
    assert not np.array_equal(a, b), (
        "this card's thicknesses cannot distinguish the two spellings; the "
        "reciprocal-first control is vacuous here")


def test_no_content_leaves_the_step_bit_identical():
    """The scoping claim: a card that supplies no runoff content is untouched.

    Not "close" -- BIT-identical, including on cells holding a negative zero,
    which is why the deposit is skipped rather than added as a zero array.
    """
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    state = card.recipe.initial_state
    plain = _lock_model().step(state, dt=card.dt_s)
    with_channel_none = _lock_model().step(
        state, dt=card.dt_s,
        surface_forcing=OceanSurfaceForcing(runoff_tracer_content=None))
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(plain, name).data),
            np.asarray(getattr(with_channel_none, name).data))


def test_a_malformed_content_pair_raises():
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    state = card.recipe.initial_state
    shape = np.asarray(state.eta.data).shape
    with pytest.raises(ValueError, match="runoff_tracer_content"):
        _lock_model().step(
            state, dt=card.dt_s,
            surface_forcing=OceanSurfaceForcing(
                runoff_tracer_content=(np.zeros(shape),)))
