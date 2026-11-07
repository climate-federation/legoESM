"""NEMO's river-runoff tracer source, as the WS-RK3 stage ladder deposits it.

``trasbc.F90``'s river-runoff block adds ``rnf_tsc * zdep`` to the tracer
right-hand side on levels ``1..nk_rnf``, with ``zdep = 1/h_rnf`` formed FIRST.
The block sits OUTSIDE ``SELECT CASE( kstg )``, so unlike the EMP/QNS block
above it, it runs at ALL THREE Runge-Kutta stages.  With neither
``ln_rnf_depth`` nor ``ln_rnf_depth_ini`` selected -- ORCA2's resolved case --
``sbcrnf.F90``'s surface arm sets ``nk_rnf = 1`` and ``h_rnf`` to the LIVE
top-cell thickness, so the whole content lands in the top cell.

These are the controls for that statement, on the small lock-exchange card,
read through the per-stage tracer exposure hook.  The reciprocal-first
spelling is NOT checked here: the stage helper maps the source rate into the
state through the quasi-Eulerian weights, so a one-representable-value
difference in the rate is not recoverable from the stage state.  What carries
that claim is the round-12 runoff gate, which compares the production RATE
bitwise against ``rnf_tsc * (1/h_rnf)`` on every cell and reports how many
cells can tell the two spellings apart at all.
"""

import numpy as np
import pytest

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_lock_exchange_zco_card,
)
from legoesm.ocean.state import OceanSurfaceForcing

from tests.ocean.unit.test_nemo_ws_tracer_rk3 import _lock_model


def _content(shape, seed=3):
    """A runoff content field with a few active river mouths."""
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float64)
    flat = field.reshape(-1)
    flat[rng.choice(flat.size, size=max(3, flat.size // 7), replace=False)] = (
        rng.uniform(1.0e-6, 4.0e-5, size=max(3, flat.size // 7)))
    return field


def _stage_state(stage, content_pair):
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    model = _lock_model(model_module._NEMOWSRK3TestHooks(
        expose_tracer_stage=stage))
    surface = (None if content_pair is None
               else OceanSurfaceForcing(runoff_tracer_content=content_pair))
    return card, model.step(card.recipe.initial_state, dt=card.dt_s,
                            surface_forcing=surface)


@pytest.fixture(scope="module")
def arms():
    card = build_lock_exchange_zco_card()
    shape = np.asarray(card.recipe.initial_state.eta.data).shape
    content = (_content(shape), _content(shape, seed=11))
    return {
        "content": content,
        **{stage: (_stage_state(stage, content)[1],
                   _stage_state(stage, None)[1])
           for stage in (1, 2, 3)},
    }


@pytest.mark.parametrize("stage", (1, 2, 3))
def test_the_channel_moves_every_stage(arms, stage):
    """Non-vacuity: reverting the deposit makes the two arms identical.

    The deposit runs at every stage because NEMO's runoff block is outside
    the stage switch -- a transcription that put it in the stage-3 arm alone
    would leave stages 1 and 2 unmoved here.
    """
    on, off = arms[stage]
    for name in ("T", "S"):
        delta = np.abs(np.asarray(getattr(on, name).data)
                       - np.asarray(getattr(off, name).data))
        assert float(delta.max()) > 0.0, f"stage {stage} {name} did not move"


def test_the_deposit_reaches_only_the_top_cell(arms):
    """nk_rnf = 1: the deposit itself enters the top cell and no other.

    Scored at stage 1 ONLY, and that is not a weakening.  By stage 2 the
    stage-1 tracer field has already been advected, so the deposit has
    legitimately reached level 1 (measured: 1.3e-12 on this card) and by
    stage 3 ``tra_zdf`` mixes it down as well.  Testing a later stage would
    be testing transport, not this statement.
    """
    stage = 1
    on, off = arms[stage]
    for name in ("T", "S"):
        delta = (np.asarray(getattr(on, name).data)
                 - np.asarray(getattr(off, name).data))
        np.testing.assert_array_equal(
            delta[..., 1:], np.zeros_like(delta[..., 1:]))
        assert float(np.abs(delta[..., 0]).max()) > 0.0


def test_no_content_leaves_the_step_bit_identical():
    """The scoping claim GYRE's landing rests on.

    Not "close" -- BIT-identical, including on a cell holding a negative zero,
    which is why the deposit is skipped rather than added as a zero array.
    """
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    state = card.recipe.initial_state
    plain = _lock_model().step(state, dt=card.dt_s)
    channel_none = _lock_model().step(
        state, dt=card.dt_s,
        surface_forcing=OceanSurfaceForcing(runoff_tracer_content=None))
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(plain, name).data),
            np.asarray(getattr(channel_none, name).data))


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
