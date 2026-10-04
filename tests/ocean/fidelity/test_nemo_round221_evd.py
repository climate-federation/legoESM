"""Round 221 (decision 94) -- NEMO's enhanced vertical diffusion, repaired.

Two defects, both pinned here so neither can come back:

1. **The trigger ran on the wrong fluid.**  ``zdf_evd`` tests the model's own
   ``rn2`` (zdfevd.f90:108 of the VORTEX_SMT1 build's ppsrc:
   ``IF( MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )``), and ``rn2``
   comes from ``bn2`` with the DECK's equation of state.  legoESM's
   ``n2_mode="nemo_bn2"`` trigger built alpha/beta from ``NemoSEOSConfig()``'s
   DINO defaults whatever fluid the card ran, because no caller threaded the
   card's own ``&nameos`` coefficients.  On the seamount SMT-1 card (rn_a0 =
   0.28, every other coefficient zero) that fires on 61 interfaces of a
   pristine, stably stratified initial state where NEMO fires on none.

2. **The coefficient was added, not replaced.**  zdfevd.f90:109 writes
   ``p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)`` INSIDE the IF, and
   zdfphy.f90:359 calls it AFTER the background/closure copy at :348-351 --
   so a fired interface carries rn_evd alone.  legoESM summed the scheme's
   coefficient onto the background and the closure.

Reverting either repair turns the matching test red: test 1 by passing the
defaulted coefficients (``seos_cfg=None``) the old code always passed, test 2
by selecting the ``"additive"`` composition the old code always used.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.eos import (  # noqa: E402
    NemoSEOSConfig,
    nemo_bn2_live_geometry,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_nemo_testcase_card,
)
from legoesm.ocean.physics.convection.config import (  # noqa: E402
    EnhancedDiffusionConfig,
)
from legoesm.ocean.physics.convection.enhanced_diffusion import (  # noqa: E402
    compose_evd_coefficient,
    convective_K_A_flag,
    resolve_evd_composition,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (  # noqa: E402
    _compute_rho,
)
from legoesm.ocean.vertical import compute_ocean_jacobian  # noqa: E402

CASE = "VORTEX_SMT1_VEC-zps"
# zdfevd.f90:108 -- the literal threshold.
EVD_THRESHOLD = -1.0e-12


@pytest.fixture(scope="module")
def smt1():
    return build_nemo_testcase_card(CASE)


def _fired(card, state, seos_cfg):
    """zdfevd's trigger mask through the model's own helper."""
    z = card.recipe.z_coord
    cc = card.recipe.model_config.physics.constants
    cfg = card.recipe.model_config.physics.convection.enhanced_diffusion
    jac = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z)
    rho = _compute_rho(state, z, jac, eos_fn=None, g=cc.g, rho0=cc.rho_0)
    t_depth, w_depth, e3w = nemo_bn2_live_geometry(
        z, state.eta.data, state.H_bathy.data)
    _, _, flag = convective_K_A_flag(
        rho, z.dz_ref, jac, cfg, T=state.T.data, S=state.S.data,
        t_depth=t_depth, w_depth=w_depth, e3w_int=e3w, seos_cfg=seos_cfg,
        g=cc.g, rho_ref=cc.rho_0)
    return np.asarray(flag)


def test_trigger_reads_the_cards_own_equation_of_state(smt1):
    """0 cells fire with the card's S-EOS; 61 with the defaulted one."""
    cfg = smt1.recipe.model_config
    assert cfg.eos == "nemo_seos" and cfg.eos_nemo_seos is not None
    assert cfg.eos_nemo_seos != NemoSEOSConfig(), (
        "the fixture needs a card whose coefficients are NOT the defaults")
    state = smt1.recipe.initial_state
    with_card = int(_fired(smt1, state, cfg.eos_nemo_seos).sum())
    with_defaults = int(_fired(smt1, state, None).sum())
    assert with_card == 0, (
        "NEMO fires on no interface of this pristine state (min N2 = "
        "+9.0e-06 1/s2, round 220 section 3); the card's own S-EOS must "
        f"agree, got {with_card} firing interfaces")
    # The reverted code: every caller passed the defaults.
    assert with_defaults == 61, (
        "the defaulted (DINO) coefficients are expected to fire on 61 "
        f"interfaces of this card, got {with_defaults} -- if this number "
        "moved, the fixture changed, not the fix")


def test_the_defaulted_trigger_fires_only_below_the_seafloor(smt1):
    """Why the round-220 plant read 0.0: all 61 are rock, not ocean."""
    z = smt1.recipe.z_coord
    flag = _fired(smt1, smt1.recipe.initial_state, None).astype(bool)
    active = np.asarray(z.is_active)
    wet_interface = active[..., 1:] & active[..., :-1]
    assert int((flag & wet_interface).sum()) == 0
    assert int((flag & ~wet_interface).sum()) == 61


def _step(card, state, *, k_conv, composition=None):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    cfg = card.recipe.model_config
    ed = cfg.physics.convection.enhanced_diffusion._replace(K_conv=k_conv)
    if composition is not None:
        ed = ed._replace(evd_composition=composition)
    cfg2 = cfg._replace(
        physics=cfg.physics._replace(
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=ed)))
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg2)
    return np.asarray(model.step(state, dt=card.dt_s).T.data)


def _unstable(card):
    """One interior column made statically unstable for the card's fluid."""
    state = card.recipe.initial_state
    active = np.asarray(card.recipe.z_coord.is_active)
    j = i = 31
    assert active[j, i, 0] and active[j, i, 1]
    T = np.array(state.T.data)
    T[j, i, 1] += 2.0          # warm water UNDER cold water
    return state._replace(T=state.T.replace(data=jnp.asarray(T)))


def test_the_replace_wiring_reaches_the_solve(smt1):
    """The composition the card states must reach the tridiagonal solve.

    Reverting either integration hunk (``k_profiles.py``'s composition or the
    physics-provided-K path in ``ocean_model_latlon_cgrid.py``) to ``+``
    makes this difference exactly zero.  What it measures is the background
    ``rn_avt0`` that used to ride on top of ``rn_evd`` on the fired
    interface: NEMO's avt there is 100 m2/s, legoESM's was 100.000012.
    """
    unstable = _unstable(smt1)
    additive = _step(smt1, unstable, k_conv=100.0, composition="additive")
    replace = _step(smt1, unstable, k_conv=100.0, composition="nemo_replace")
    moved = float(np.max(np.abs(replace - additive)))
    assert moved > 0.0, (
        "the stated composition does not reach the solve: replacing and "
        "adding the convective coefficient gave the same temperature")
    assert moved == pytest.approx(2.735e-09, rel=0.05), moved
    # And it is inert where the trigger selects nothing -- the pristine card.
    assert np.array_equal(
        _step(smt1, smt1.recipe.initial_state, k_conv=100.0,
              composition="additive"),
        _step(smt1, smt1.recipe.initial_state, k_conv=100.0,
              composition="nemo_replace"))


def test_the_explicit_branch_refuses_the_replacement():
    """It adds a tendency; zdfevd overwrites a coefficient."""
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics,
    )
    conv = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(
            n2_mode="nemo_bn2", smooth_transition=False, K_bg=0.0, nu_bg=0.0,
            nu_conv=0.0, K_conv=100.0, evd_composition="nemo_replace"))
    with pytest.raises(ValueError, match="cannot express it"):
        make_convection_physics(conv, apply_diffusion=True)
    # The implicit branch surfaces the coefficient and is fine.
    make_convection_physics(conv, apply_diffusion=False)
    # And an unstated NEMO-trigger card raises on this path too.
    unstated = conv._replace(
        enhanced_diffusion=conv.enhanced_diffusion._replace(
            evd_composition=""))
    with pytest.raises(ValueError, match="evd_composition is unset"):
        make_convection_physics(unstated, apply_diffusion=False)


def test_the_guards_survive_tracing_of_the_tunable_coefficient():
    """K_conv is a declared tunable; the guards must not break jit+grad."""
    import jax

    cfg = EnhancedDiffusionConfig(
        n2_mode="nemo_bn2", smooth_transition=False, K_bg=0.0, nu_bg=0.0,
        nu_conv=0.0, K_conv=100.0, evd_composition="nemo_replace")
    other = jnp.asarray([1.0e-5, 1.0e-5, 7.0])
    fired = jnp.asarray([0.0, 1.0, 1.0])

    def loss(k_conv):
        mode = resolve_evd_composition(cfg._replace(K_conv=k_conv))
        evd = jnp.where(fired > 0, k_conv, 0.0)
        return jnp.sum(compose_evd_coefficient(
            other, evd, mode, convective=k_conv))

    assert float(jax.jit(jax.grad(loss))(100.0)) == 2.0


@pytest.mark.slow
def test_the_rn_evd_plant_moves_temperature_where_the_trigger_fires(smt1):
    """rn_evd x 1e4 must MOVE T on a column the trigger selects."""
    unstable = _unstable(smt1)
    assert int(_fired(smt1, unstable,
                      smt1.recipe.model_config.eos_nemo_seos).sum()) == 1
    moved = np.max(np.abs(
        _step(smt1, unstable, k_conv=1.0e6)
        - _step(smt1, unstable, k_conv=100.0)))
    assert moved > 1.0e-3, (
        "the card states an enhanced vertical diffusion that does not reach "
        f"the solve: the plant moved temperature by {moved!r} K")


@pytest.mark.slow
def test_the_plant_is_inert_on_the_pristine_card(smt1):
    """And exactly inert where NEMO's trigger selects nothing."""
    state = smt1.recipe.initial_state
    moved = np.max(np.abs(
        _step(smt1, state, k_conv=1.0e6) - _step(smt1, state, k_conv=100.0)))
    assert moved == 0.0


def test_composition_is_a_replacement_not_a_sum():
    other = jnp.asarray([1.0e-5, 1.0e-5, 7.0])
    evd = jnp.asarray([0.0, 100.0, 100.0])
    add = np.asarray(compose_evd_coefficient(
        other, evd, "additive", convective=100.0))
    rep = np.asarray(compose_evd_coefficient(
        other, evd, "nemo_replace", convective=100.0))
    np.testing.assert_allclose(add, [1.0e-5, 100.00001, 107.0])
    # zdfevd.f90:109 -- the fired interfaces carry rn_evd and nothing else,
    # including where a closure had already written a LARGER coefficient.
    np.testing.assert_array_equal(rep, [1.0e-5, 100.0, 100.0])
    # nn_evdm = 0: the arm never runs, so the operand passes through.
    np.testing.assert_array_equal(
        np.asarray(compose_evd_coefficient(
            other, jnp.zeros_like(evd), "nemo_replace", convective=0.0)),
        np.asarray(other))


def test_a_nemo_trigger_card_must_state_its_composition():
    nemo_trigger = EnhancedDiffusionConfig(
        n2_mode="nemo_bn2", smooth_transition=False, K_bg=0.0, nu_bg=0.0,
        nu_conv=0.0, K_conv=100.0)
    with pytest.raises(ValueError, match="evd_composition is unset"):
        resolve_evd_composition(nemo_trigger)
    assert resolve_evd_composition(
        nemo_trigger._replace(evd_composition="nemo_replace")) == "nemo_replace"
    assert resolve_evd_composition(
        nemo_trigger._replace(evd_composition="additive")) == "additive"
    with pytest.raises(ValueError, match="unknown"):
        resolve_evd_composition(nemo_trigger._replace(evd_composition="max"))
    # Every other card keeps the composition it has always had.
    assert resolve_evd_composition(EnhancedDiffusionConfig()) == "additive"


def test_nn_evdm_is_stated_not_implied():
    base = EnhancedDiffusionConfig(
        n2_mode="nemo_bn2", smooth_transition=False, K_bg=0.0, nu_bg=0.0,
        K_conv=100.0, evd_composition="nemo_replace")
    assert resolve_evd_composition(base._replace(nu_conv=0.0))        # nn_evdm=0
    assert resolve_evd_composition(base._replace(nu_conv=100.0))      # nn_evdm=1
    with pytest.raises(ValueError, match="nn_evdm"):
        resolve_evd_composition(base._replace(nu_conv=7.0))
    with pytest.raises(ValueError, match="nu_bg"):
        resolve_evd_composition(base._replace(nu_bg=1.0e-5))
    with pytest.raises(ValueError, match="smooth_transition"):
        resolve_evd_composition(base._replace(smooth_transition=True))
    with pytest.raises(ValueError, match="K_bg < K_conv"):
        resolve_evd_composition(base._replace(K_bg=100.0))


def test_every_nemo_card_running_evd_states_nemos_composition():
    for case, nn_evdm in (("GYRE-zco", 1), ("VORTEX_SMT1_VEC-zps", 0)):
        card = build_nemo_testcase_card(case)
        ed = card.recipe.model_config.physics.convection.enhanced_diffusion
        assert ed.evd_composition == "nemo_replace", case
        assert resolve_evd_composition(ed) == "nemo_replace", case
        # nn_evdm, read back from the card rather than from the deck comment.
        assert ed.nu_conv == (ed.K_conv if nn_evdm else 0.0), case
