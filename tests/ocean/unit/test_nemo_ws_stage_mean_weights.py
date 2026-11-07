"""The WS-RK3 stage depth mean uses NEMO's REFERENCE weights, not live ones.

NEMO removes a reference-weighted depth mean from every stage velocity::

    zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)
                                                        stprk3_stg.F90:440
    hu_0(:,:)  = hu_0(:,:) + e3u_0(:,:,jk) * umask(:,:,jk)      domain.F90:145

legoESM used the LIVE face thickness ``h_u_pre = min_cell_to_uface(h_k_pre)``
and its column sum.  Those are not a rescale of the reference pair, because the
minimum is taken over two columns whose free-surface Jacobians differ, so its
per-level argmin can switch sides.  Where the two neighbouring columns have
EQUAL depth the two weightings coincide, which is why this is a staircase-only
rule.

The two assertions below are the rule's two halves, and reverting the fix
(making the live weighting the default again) collapses the arms and fails the
first one.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card


def _run(case: str, legacy: bool, steps: int = 5) -> np.ndarray:
    set_policy(PrecisionPolicy.fp64())
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            legacy_live_stage_mean_weights=legacy))
    # At kt=1 the free surface is flat, so the live and reference weights are
    # EQUAL on every card and the arms coincide by construction.  The rule only
    # bites once ssh has stretched the two sides of a staircase face by
    # different factors, which the committed probe first resolves at kt=2.
    state = card.recipe.initial_state
    for _ in range(steps):
        state = model.step(state, dt=card.dt_s)
    return np.asarray(state.u.data, dtype=np.float64)


@pytest.mark.parametrize("case", ["OVERFLOW-zps"])
def test_staircase_card_distinguishes_reference_from_live_stage_mean_weights(case):
    """On a staircase the two weightings are different operators."""
    # five steps: the difference is exactly 0.0 at kt=1 (flat ssh) by
    # construction, so a one-step test here would pass vacuously
    faithful = _run(case, legacy=False)
    legacy = _run(case, legacy=True)
    moved = float(np.max(np.abs(faithful - legacy)))
    # Reverting the fix makes the default live-weighted, the two arms identical
    # and this assertion fail with moved == 0.0.
    assert moved > 0.0, "reference and live stage-mean weights gave identical states"
    assert np.all(np.isfinite(faithful)) and np.all(np.isfinite(legacy))


def test_flat_bottom_card_is_insensitive_to_the_weighting():
    """The control: with equal-depth neighbours the two weightings coincide.

    LOCK_EXCHANGE has a flat bottom, so ``min_cell_to_uface`` picks one uniform
    Jacobian per column and only the ARITHMETIC differs -- the residual must sit
    at re-association level, not at operator level.
    """
    faithful = _run("LOCK_EXCHANGE-zco", legacy=False)
    legacy = _run("LOCK_EXCHANGE-zco", legacy=True)
    scale = max(float(np.max(np.abs(faithful))), 1.0)
    assert float(np.max(np.abs(faithful - legacy))) / scale < 1e-14
