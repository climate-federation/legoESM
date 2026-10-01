"""NEMO's RK3 after-SSH slot, carried across steps (VORTEX round 6).

At the END of every RK3 step, after the ``Nbb <==> Naa`` rotation, NEMO leaves
the NEXT step's after-SSH guess in the slot the step entered with::

    Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs
    ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)
        -- stprk3.F90:221,225

``Nbb`` is untouched across the three stages, so after the swap ``Nbb`` holds
the height the step PRODUCED and ``Naa`` the height it ENTERED with.  The next
step turns that slot into ``r3t(:,:,Kaa)`` (stp2d.F90:149) immediately before
its first ``CALL wzv`` (stp2d.F90:153), and NEMO persists it across a restart
as ``ssha`` (restart.F90:184, read back at restart.F90:362-370).

legoESM carries it only when the card STATES
``nemo_first_wzv_after_ssh="rk3_extrapolated_carried"``.  No card states that
yet; these tests drive the measurement arm directly.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    NEMO_FIRST_WZV_AFTER_SSH_FORMS,
    nemo_rk3_after_ssh_is_carried,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_nemo_testcase_card,
    with_first_wzv_after_ssh,
)

CASE = "VORTEX_VEC-zco"


@pytest.fixture(scope="module")
def arms():
    """One card, two arms, two steps each -- compiled once per arm."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    if not jax.config.jax_enable_x64:
        pytest.skip("this card is an fp64 NEMO transcription; run with "
                    "JAX_ENABLE_X64=1")
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    out = {}
    for name, form in (("card", None), ("carried", "rk3_extrapolated_carried")):
        cfg = with_first_wzv_after_ssh(card.recipe.model_config, form)
        model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
        s0 = card.recipe.initial_state
        s1 = model.step(s0, dt=card.dt_s)
        s2 = model.step(s1, dt=card.dt_s)
        out[name] = {"model": model, "cfg": cfg, "s0": s0, "s1": s1, "s2": s2}
    out["dt"] = card.dt_s
    try:
        yield out
    finally:
        set_policy(previous)


def _eta(state):
    return np.asarray(state.eta.data)


def test_the_predicate_is_the_one_both_sides_read():
    """The writer (the step) and the reader (the wzv call) share ONE
    predicate, so a card can never write the slot without reading it."""
    card = build_nemo_testcase_card(CASE).recipe.model_config
    assert "rk3_extrapolated_carried" in NEMO_FIRST_WZV_AFTER_SSH_FORMS
    assert not nemo_rk3_after_ssh_is_carried(card)
    assert nemo_rk3_after_ssh_is_carried(
        card._replace(nemo_first_wzv_after_ssh="rk3_extrapolated_carried"))
    # A card that never reaches NEMO's own first wzv call never carries it,
    # whatever it states.
    assert not nemo_rk3_after_ssh_is_carried(
        card._replace(zad_qco_evaluation="generic",
                      nemo_first_wzv_after_ssh="rk3_extrapolated_carried"))


def test_no_card_states_the_carried_form_yet():
    """Round 6 BUILDS the carry and does not switch any card to it; the
    switch is the operator's decision.  This test is the record of that, and
    it goes red the moment a card is switched -- which is when the receipt's
    numbers have to be re-measured."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card as build,
    )
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    # ORCA2-zps needs an explicit deck_root and is not constructible here;
    # its stated form is covered by the card census in
    # tests/ocean/unit/test_nemo_vortex_card.py.
    for case in ("GYRE-zco", "VORTEX_VEC-zco", "VORTEX-zco",
                 "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        cfg = build(case).recipe.model_config
        assert cfg.nemo_first_wzv_after_ssh != "rk3_extrapolated_carried", case
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        assert (dino_config_for_recipe(recipe).nemo_first_wzv_after_ssh
                != "rk3_extrapolated_carried")


def test_the_carried_slot_is_nemo_s_own_extrapolation(arms):
    """``2*end-of-step - step-entry``, to the last bit."""
    arm = arms["carried"]
    assert arm["s0"].eta_rk3_after is None, (
        "the initial state carries no previous step; NEMO's own fallback is "
        "ssh(:,:,Kaa) = ssh(:,:,Kbb) (restart.F90:370)")
    want = 2.0 * _eta(arm["s1"]) - _eta(arm["s0"])
    np.testing.assert_array_equal(
        np.asarray(arm["s1"].eta_rk3_after.data), want)
    want2 = 2.0 * _eta(arm["s2"]) - _eta(arm["s1"])
    np.testing.assert_array_equal(
        np.asarray(arm["s2"].eta_rk3_after.data), want2)


def test_the_uncarried_arm_leaves_the_slot_empty(arms):
    assert arms["card"]["s1"].eta_rk3_after is None
    assert arms["card"]["s2"].eta_rk3_after is None


def test_the_first_step_is_bit_identical_between_the_two_arms(arms):
    """At ``nit000`` NEMO's slot holds the step-entry height either way
    (restart.F90:370), so the carry CANNOT move the first step."""
    for field in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(
            np.asarray(getattr(arms["card"]["s1"], field).data),
            np.asarray(getattr(arms["carried"]["s1"], field).data),
            err_msg=f"the carry moved the FIRST step's {field}")


def test_the_second_step_differs_between_the_two_arms(arms):
    """Non-vacuity: if the two arms agreed at step 2 the carry is not wired."""
    moved = [
        field for field in ("eta", "u", "v", "T")
        if not np.array_equal(
            np.asarray(getattr(arms["card"]["s2"], field).data),
            np.asarray(getattr(arms["carried"]["s2"], field).data))
    ]
    assert moved, "the carried slot changed nothing on the second step"


def test_substituting_the_step_entry_height_reproduces_the_other_arm(arms):
    """THE non-vacuity control, and it is EXACT.

    NEMO's own no-previous-step case is ``ssh(:,:,Kaa) = ssh(:,:,Kbb)``
    (restart.F90:370).  Writing that value into the carried slot must
    reproduce the uncarried arm BIT-FOR-BIT, which says two things at once:
    the slot is genuinely read, and the slot's VALUE is the only difference
    between the two arms -- no second variable came along with the branch.
    """
    arm = arms["carried"]
    substituted = arm["s1"]._replace(
        eta_rk3_after=arm["s1"].eta_rk3_after.replace(data=arm["s1"].eta.data))
    rerun = arm["model"].step(substituted, dt=arms["dt"])
    for field in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(
            np.asarray(getattr(rerun, field).data),
            np.asarray(getattr(arms["card"]["s2"], field).data),
            err_msg=f"{field}: the carried arm fed the step-entry height is "
                    "not the uncarried arm, so the two arms differ in more "
                    "than the slot")


def test_a_perturbation_of_the_carried_slot_moves_the_next_step(arms):
    """The slot is a LIVE operand of the next step, not storage.

    A ONE-ULP change is absorbed and that is MEASURED, not a defect: the slot
    enters as ``r3t = ssh/h_0`` against a ~4 km column and is then scaled by
    ``e3t/dt`` (sshwzv.F90:295-298), so its last bit lands far below the
    vertical velocity's own.  The plant therefore perturbs by 1e-9 m, which
    is still a billionth of the sea surface it sits on.
    """
    arm = arms["carried"]
    slot = np.asarray(arm["s1"].eta_rk3_after.data)
    cell = int(np.argmax(np.abs(slot)))
    planted = slot.copy().ravel()
    planted[cell] += 1.0e-9
    bumped = arm["s1"]._replace(
        eta_rk3_after=arm["s1"].eta_rk3_after.replace(
            data=jnp.asarray(planted.reshape(slot.shape))))
    moved = arm["model"].step(bumped, dt=arms["dt"])
    assert any(
        not np.array_equal(np.asarray(getattr(moved, f).data),
                           np.asarray(getattr(arm["s2"], f).data))
        for f in ("eta", "u", "v", "T")), (
        "a 1e-9 m change to NEMO's after-SSH slot left every prognostic "
        "field untouched; the slot is not reaching the wzv call")
    # ... and the SAME perturbation is inert on the arm that does not read it.
    inert = arms["card"]["model"].step(
        arms["card"]["s1"]._replace(eta_rk3_after=bumped.eta_rk3_after),
        dt=arms["dt"])
    for field in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(
            np.asarray(getattr(inert, field).data),
            np.asarray(getattr(arms["card"]["s2"], field).data),
            err_msg=f"{field} moved on the arm that does not read the slot")


def test_the_measurement_arm_refuses_a_card_that_never_reaches_the_branch():
    """An arm that silently does nothing would report 'no change' as a
    measurement."""
    flux = build_nemo_testcase_card("VORTEX-zco").recipe.model_config
    assert flux.zad_qco_evaluation != "nemo_literal"
    with pytest.raises(ValueError, match="measure nothing"):
        with_first_wzv_after_ssh(flux, "rk3_extrapolated_carried")
    vec = build_nemo_testcase_card(CASE).recipe.model_config
    with pytest.raises(ValueError, match="not one of"):
        with_first_wzv_after_ssh(vec, "rk3_guess")
    assert with_first_wzv_after_ssh(vec, None) is vec


def test_pre_filling_the_slot_with_nemo_s_nit000_value_moves_no_number(arms):
    """What the scan-carry seeding does, and why it is free.

    A ``None -> Field`` transition on the first iteration changes a
    ``lax.scan`` carry's tree structure and the scan rejects it, so the
    scan-carry preparation fills the slot before the loop.  It fills it with
    NEMO's own pre-first-step value -- ``ssh(:,:,Kaa) = ssh(:,:,Kbb)``
    (restart.F90:370) -- and THAT is why seeding cannot move a number: the
    branch reads the step-entry height whether the slot is absent or holds
    exactly that height.  Measured here rather than argued.

    ``seed_scan_carry`` itself is NOT exercised on this card, and that is
    PRE-EXISTING: the card selects NEMO's AB3/AM4 barotropic filter, whose
    six-array history the seeder cannot pre-fill (the module docstring records
    that such runs are step-1-eager and only then scan).  It raises the same
    way on BOTH arms, so this round neither uses nor breaks it; the test
    below pins that rather than leaving it to a reader.
    """
    arm = arms["carried"]
    prefilled = arm["s0"]._replace(eta_rk3_after=arm["s0"].eta)
    stepped = arm["model"].step(prefilled, dt=arms["dt"])
    for field in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(
            np.asarray(getattr(stepped, field).data),
            np.asarray(getattr(arm["s1"], field).data),
            err_msg=f"pre-filling the slot moved {field}")
    assert np.array_equal(np.asarray(stepped.eta_rk3_after.data),
                          2.0 * _eta(stepped) - _eta(arm["s0"]))


def test_the_scan_seeder_is_unreachable_on_this_card_on_both_arms(arms):
    """Pre-existing, and identical on both arms: the AB3/AM4 barotropic
    history cannot be pre-seeded, so ``seed_scan_carry`` refuses this card
    whatever the after-SSH form is."""
    messages = []
    for name in ("card", "carried"):
        with pytest.raises(ValueError) as caught:
            arms[name]["model"].seed_scan_carry(arms[name]["s0"], arms["dt"])
        messages.append(str(caught.value).splitlines()[0])
    assert messages[0] == messages[1], (
        "the two arms fail the seeder differently, so this round changed it")


def test_the_carried_form_is_refused_on_a_program_that_cannot_write_the_slot():
    """The slot is written by NEMO's RK3 end-of-step rotation, and only by it.

    A card that states the carried form with a different time-stepping program
    would READ the slot on every step and never WRITE one, which is the silent
    fallback this campaign keeps being bitten by.  It raises instead.  This is
    a consistency check between two fields the card STATES, not an inference
    that picks a form for it.
    """
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        nemo_first_wzv_after_ssh="rk3_extrapolated_carried",
        momentum_time_integrator="euler")
    with pytest.raises(ValueError, match="only the RK3 stage program"):
        LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def test_the_carried_form_is_refused_with_a_post_step_sea_surface_rewrite():
    """The slot is built from the sea surface the step produced.  A projection
    that rewrites that sea surface AFTER the step would leave the next step
    reading a slot that does not match the height it enters with.  NEMO
    applies none of these, so the combination is refused rather than silently
    producing a slot built on a height nothing ever used."""
    card = build_nemo_testcase_card(CASE)
    base = card.recipe.model_config._replace(
        nemo_first_wzv_after_ssh="rk3_extrapolated_carried")
    with pytest.raises(ValueError, match="freeze_floor"):
        LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                              base._replace(freeze_floor=True))
    # ... and the same card without the projection constructs.
    LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, base)
