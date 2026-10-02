"""Fail-closed controls for round 204's stage-1 right-hand-side SPLIT.

Round 201 proved the VORTEX flux card's stage-1 velocity error is carried by
the completed stage-1 momentum right-hand side, and could go no further:
NEMO's three-dimensional array at that boundary holds HPG + LDF + COR/MET
only -- on the ``np_FLX_up3`` branch ``stp_2D`` calls ``dyn_adv_up3`` with
``pUe``/``pVe`` (``stp2d.f90:169-170``), which writes the TWO-dimensional RHS
alone -- while legoESM's own pre-stage array already carries its advection.
The two were not like-for-like, so the record's ``base`` row could not be
scored.

``expose_stage1_momentum_rhs_split`` publishes legoESM's stage-1 right-hand
side with its advection content removed (``"pre_advection"``), and the
completed right-hand side from the SAME evaluation (``"completed"``).  Four
things have to hold before either frame means anything:

* at its default the hook changes NOTHING;
* ``"completed"`` is BIT-identical to what the production-configured
  stage-1 exposure publishes -- asking ``tendencies()`` for its per-term
  decomposition must not perturb the step, or the split is a decomposition
  of a different number;
* what ``"pre_advection"`` removes has DEPTH STRUCTURE, so the flux-form
  horizontal trend (which shares a diagnostic slot with the rotation terms,
  and was missing from the first draft of this seam) is really in it;
* perturbing an ADVECTION-ONLY operand moves the completed frame with depth
  structure while moving the pre-advection frame by at most one number per
  column -- the only channel left open to it, because removing advection
  also moves the depth mean the stage array has subtracted.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

CASE = "VORTEX-zco"


def _card_and_model(hooks=None):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    return card, model


@pytest.fixture(scope="module")
def fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def steps(fp64):
    """Every step this module needs, run once: each one is ~40 s of CPU."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, plain = _card_and_model()
    initial = card.recipe.initial_state
    out = {"plain": plain.step(initial, dt=card.dt_s)}
    for name, hooks in (
            ("defaults", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="")),
            ("rhs", _NEMOWSRK3TestHooks(expose_stage1_momentum_rhs=True)),
            ("completed", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="completed")),
            ("pre_advection", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="pre_advection")),
            # An ADVECTION-ONLY private arm: it selects the legacy transport
            # sign inside the UP3 upwind curvature and touches nothing else.
            ("bent_completed", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="completed",
                legacy_up3_transport_sign_selector=True)),
            ("bent_pre_advection", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="pre_advection",
                legacy_up3_transport_sign_selector=True)),
    ):
        _, model = _card_and_model(hooks)
        out[name] = model.step(initial, dt=card.dt_s)
    return out


def _leaves(state):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]


def _assert_identical(got, want):
    a, b = _leaves(got), _leaves(want)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _depth_split(delta):
    """Largest column mean of ``delta`` and largest departure from it."""
    mean = delta.mean(axis=-1, keepdims=True)
    return float(np.abs(mean).max()), float(np.abs(delta - mean).max())


def test_the_split_hook_at_its_default_changes_nothing(steps):
    """Constructing the hook bundle with the new field at its default is
    identical to passing no hook bundle at all.

    SCOPE, said rather than implied: this covers the field's presence, not
    the new code behind it -- at the default neither the extra operator
    component nor the widened decomposition request runs.  That the round
    changed no production number is carried by the external certified
    registries (both VORTEX cards 0/50, the GYRE ladder and the 360-day
    year), not by this test.
    """
    _assert_identical(steps["defaults"], steps["plain"])


def test_asking_for_the_decomposition_does_not_perturb_the_step(steps):
    """The P3 control, committed.

    ``"completed"`` runs the step with ``diagnose_momentum`` turned on
    inside ``tendencies()`` so the per-term decomposition exists.  If that
    moved a single bit of the completed right-hand side, the pre-advection
    frame would be a half of a DIFFERENT array than the one round 201
    scored.
    """
    for face in ("u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(steps["completed"], face).data),
            np.asarray(getattr(steps["rhs"], face).data))
    # Only the velocity slots are substituted; the rest of the returned
    # state is the production step's.
    np.testing.assert_array_equal(np.asarray(steps["completed"].T.data),
                                  np.asarray(steps["plain"].T.data))
    np.testing.assert_array_equal(np.asarray(steps["completed"].eta.data),
                                  np.asarray(steps["plain"].eta.data))


def test_the_pre_advection_frame_is_live_and_not_the_completed_one(steps):
    for face in ("u", "v"):
        pre = np.asarray(getattr(steps["pre_advection"], face).data)
        done = np.asarray(getattr(steps["completed"], face).data)
        plain = np.asarray(getattr(steps["plain"], face).data)
        assert not np.array_equal(pre, done)
        assert not np.array_equal(pre, plain)


def test_what_the_split_removes_has_depth_structure(steps):
    """The flux-form horizontal trend is really inside the removed half.

    It shares ``diag_vortcor`` with the rotation terms, and the first draft
    of this seam subtracted only ``advection_u`` (the D-term plus the
    vertical flux) -- which left the horizontal trend in the "pre-advection"
    frame and made the scored row an order of magnitude WORSE.  A removed
    half that were only the vertical flux would be far smaller than the
    full trend; require depth structure comparable to the removed half's own
    size, which a column-constant cannot fake.
    """
    for face in ("u", "v"):
        removed = (np.asarray(getattr(steps["completed"], face).data)
                   - np.asarray(getattr(steps["pre_advection"], face).data))
        uniform, varying = _depth_split(removed)
        assert varying > 0.0
        assert varying > 0.1 * max(uniform, 1.0e-300)


def test_an_advection_only_perturbation_misses_the_pre_advection_frame(steps):
    """The decisive control FOR THE HORIZONTAL TREND.

    Scope: ``legacy_up3_transport_sign_selector`` reaches only the flux-form
    horizontal advection, so this proves the published frame excludes THAT
    term -- the one the first draft of the seam left in.  The vertical flux,
    the D-term and the two stage increments are covered by the depth-structure
    control above and by the walk's own scored rows, not here.


    Bending an advection-only operand must move the completed right-hand
    side WITH depth structure.  It also moves the depth mean that the stage
    array has already subtracted (``du_dt_pert = du_dt - F_slow``), so the
    pre-advection frame may move by ONE NUMBER PER COLUMN -- but by nothing
    more.  A frame that still carried advection would move with depth
    structure too.
    """
    for face in ("u", "v"):
        done = (np.asarray(getattr(steps["bent_completed"], face).data)
                - np.asarray(getattr(steps["completed"], face).data))
        pre = (np.asarray(getattr(steps["bent_pre_advection"], face).data)
               - np.asarray(getattr(steps["pre_advection"], face).data))
        _, done_varying = _depth_split(done)
        pre_uniform, pre_varying = _depth_split(pre)
        # the perturbation is real and has depth structure
        assert done_varying > 0.0
        # and the pre-advection frame keeps none of it
        assert pre_varying < 1.0e-9 * max(pre_uniform, done_varying)


@pytest.mark.parametrize("hooks_kwargs,match", [
    ({"expose_stage1_momentum_rhs_split": "advection"},
     "must be '', 'pre_advection', or 'completed'"),
    ({"expose_stage1_momentum_rhs_split": True},
     "must be '', 'pre_advection', or 'completed'"),
    ({"expose_stage1_momentum_rhs_split": "completed",
      "expose_stage1_momentum_rhs": True}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs_split": "pre_advection",
      "expose_stage1_raw_momentum": True}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs_split": "pre_advection",
      "expose_momentum_stage": 1}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs_split": "pre_advection",
      "expose_tracer_transport_stage": 1}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs_split": "completed",
      "stage1_momentum_rhs_override": (1.0, 2.0)},
     "stage1_momentum_rhs_override"),
])
def test_the_construction_guard_refuses_an_ambiguous_split(
        fp64, hooks_kwargs, match):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    with pytest.raises(ValueError, match=match):
        _card_and_model(_NEMOWSRK3TestHooks(**hooks_kwargs))


def test_both_split_arms_alone_are_accepted(fp64):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    for arm in ("pre_advection", "completed"):
        _card_and_model(_NEMOWSRK3TestHooks(
            expose_stage1_momentum_rhs_split=arm))
