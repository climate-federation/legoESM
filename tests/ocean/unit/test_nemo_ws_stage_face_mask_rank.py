"""The WS-RK3 stage velocity carries NEMO's 3-D ``umask``, not a 2-D one.

``stprk3_stg.F90:367`` (``ln_dynadv_vec .OR. lk_linssh``), ``:375`` (the
compiled ``key_qco`` branch) and ``:382`` (the ``#else``) multiply the stage-1
and stage-2 velocity update by ``umask(ji,jj,jk)`` (stage 3's update lives in
``dyn_zdf``, masked there); ``:444`` adds the barotropic correction as
``zub(ji,jj)*umask(ji,jj,jk)``; and ``:273-274`` masks the SAME correction
inside the advective transport ``dyn_adv_up3`` consumes.  So NEMO's ``uu``
AND the transport built from it are EXACTLY zero below the seabed.  ``dyn_adv_up3`` then READS that zero: its k-slab stencil does not
skip a dry neighbour (``dynadv_up3.F90:142-143`` ``zlu_uu``, ``:160``
``zFu``, ``:166-176`` ``zFu_t``).

legoESM masked the stage update with the 2-D ``state.u_mask`` broadcast over
levels, so on a staircase a face wet at ANY level kept the barotropic
depth-mean increment at EVERY level below its own seabed, and the deeper
neighbour's wet bottom level read it as a stencil neighbour.  The private
``legacy_2d_stage_face_mask`` hook is the one-variable control (NEMO has no
such switch).
"""

import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card


def _step(case, **hooks):
    """One step of the named certified card, plus its 3-D live u-face mask."""
    set_policy(PrecisionPolicy.fp64())
    card = build_nemo_testcase_card(case)
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(**hooks))
    live_u = np.asarray(compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)[0]).astype(float)
    return np.asarray(model.step(card.recipe.initial_state, dt=card.dt_s).u.data), live_u


def test_staircase_stage_velocity_is_exactly_zero_below_the_live_seabed():
    """OVERFLOW's shelf is the staircase: the 3-D live u-face mask differs
    from the 2-D broadcast at 3000 points.  The faithful arm leaves EXACTLY
    zero there; the 2-D arm (the pre-fix code) leaves a finite velocity, and
    the two arms disagree on the WET faces too -- so the mask rank changes
    the answer, not just the values nobody scores."""
    faithful, live_u = _step("OVERFLOW-zps")
    legacy, _ = _step("OVERFLOW-zps", legacy_2d_stage_face_mask=True)
    assert faithful.dtype == np.float64
    dry = live_u == 0.0
    # the geometry this test needs actually exists
    assert dry.sum() > 0
    assert float(np.abs(faithful[dry]).max()) == 0.0
    # non-vacuity: the control arm IS the pre-fix behaviour and is nonzero
    assert float(np.abs(legacy[dry]).max()) > 1.0e-3
    # and the wet state moves, so this is physics, not bookkeeping
    wet = live_u > 0.0
    assert float(np.abs(faithful[wet] - legacy[wet]).max()) > 0.0


def test_flat_bottom_card_is_bit_identical_under_the_mask_rank_arm():
    """LOCK_EXCHANGE has no staircase, so its 3-D live u-face mask IS the 2-D
    broadcast and the two rules coincide by construction."""
    faithful, live_u = _step("LOCK_EXCHANGE-zco")
    legacy, _ = _step("LOCK_EXCHANGE-zco", legacy_2d_stage_face_mask=True)
    card = build_nemo_testcase_card("LOCK_EXCHANGE-zco")
    flat = np.broadcast_to(
        np.asarray(card.recipe.initial_state.u_mask.data)[..., None], live_u.shape)
    np.testing.assert_array_equal(live_u, np.asarray(flat, dtype=float))
    np.testing.assert_array_equal(faithful, legacy)
