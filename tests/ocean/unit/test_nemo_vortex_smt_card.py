"""VORTEX_SMT: the seamount cards and the zps bottom-level rule they need.

Decision 88 (user, 2026-10-03).  Every test here is written so that it FAILS
when the statement it guards is reverted; the reversion is named in each
docstring.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_nemo_testcase_card,
    build_vortex_smt_zps_card,
    vortex_horizontal_coordinates,
    vortex_smt_bathymetry,
    vortex_smt_partial_cell_geometry,
    _VORTEX_RESOLUTIONS,
)
from legoesm.ocean.vertical import (
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
)


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _uniform_ref(nlev=10, dz=500.0):
    return create_z_star_from_thicknesses(
        jnp.full((nlev,), dz),
        t_depth_ref_m=(np.arange(nlev, dtype=np.float64) + 0.5) * dz,
    )


# --------------------------------------------------------------- the rule ---
def test_zps_e3min_rule_is_not_the_t_point_rule():
    """The two shipped zps rules disagree, and this card needs the OTHER one.

    OVERFLOW's ze3min floor is a TENTH of the reference thickness; the
    T-point rule's implied floor is a HALF.  A column 4560 m deep on a 500 m
    ladder therefore reaches level 10 under ze3min and stops at 9 under the
    T-point rule.  Reverting the card to ``bottom_index_rule="nemo_tpoint"``
    makes this test fail.
    """
    z = _uniform_ref()
    H = jnp.asarray([[4560.0, 4530.0, 4472.708, 5000.0]])
    e3min = create_partial_cell_coordinate(
        z, H, bottom_index_rule="nemo_zps_e3min", min_partial_thickness=50.0)
    tpoint = create_partial_cell_coordinate(
        z, H, bottom_index_rule="nemo_tpoint")
    assert list(np.asarray(e3min.bottom_level)[0]) == [9, 8, 8, 9]
    assert list(np.asarray(tpoint.bottom_level)[0]) == [8, 8, 8, 9]
    # the 4560 m column is the discriminating one
    assert np.asarray(e3min.bottom_level)[0, 0] != (
        np.asarray(tpoint.bottom_level)[0, 0])


def test_zps_e3min_partial_thickness_is_nemos_association():
    """``MIN(zht, pdepw(ik+1)) - pdepw(ik)``, OVERFLOW:222 -- a VALUE pin.

    Round 2's review checked, and this test then confirmed, that on every
    ladder tried -- uniform and stretched -- the source's association and the
    clipped ``MIN(H - pdepw(k), dz_ref[k])`` give the SAME BITS.  So this is
    a pin on the resolved thickness, NOT a discriminator between the two
    forms, and it is labelled as one rather than left to look like proof the
    association matters.  What makes the rule itself non-vacuous is the
    bottom-LEVEL test above, which does separate the two shipped rules.
    """
    z = _uniform_ref()
    H = jnp.asarray([[4560.0, 4763.0722]])
    c = create_partial_cell_coordinate(
        z, H, bottom_index_rule="nemo_zps_e3min", min_partial_thickness=50.0)
    h = np.asarray(c.h_partial)[0]
    assert h[0, 9] == min(4560.0, 5000.0) - 4500.0
    assert h[1, 9] == min(4763.0722, 5000.0) - 4500.0
    # the floor really is a tenth of a cell, not a half: a 4551 m column keeps
    # a 51 m bottom cell instead of dropping a level.
    thin = create_partial_cell_coordinate(
        z, jnp.asarray([[4551.0]]), bottom_index_rule="nemo_zps_e3min",
        min_partial_thickness=50.0)
    assert int(np.asarray(thin.bottom_level)[0, 0]) == 9
    assert np.asarray(thin.h_partial)[0, 0, 9] == 51.0


def test_zps_e3min_rule_requires_its_floor_and_only_it():
    z = _uniform_ref()
    H = jnp.asarray([[4560.0]])
    with pytest.raises(ValueError, match="min_partial_thickness"):
        create_partial_cell_coordinate(
            z, H, bottom_index_rule="nemo_zps_e3min")
    with pytest.raises(ValueError, match="min_partial_thickness"):
        create_partial_cell_coordinate(
            z, H, bottom_index_rule="nemo_tpoint", min_partial_thickness=50.0)
    with pytest.raises(ValueError, match="bottom_index_rule"):
        create_partial_cell_coordinate(z, H, bottom_index_rule="zps")


# -------------------------------------------------------- the bathymetry ---
def test_seamount_constants_are_the_users():
    """h = 5000 - 1000 exp(-r^2/(150 km)^2) centred 300 km WEST of (0,0)."""
    assert vortex_smt_bathymetry(-300.0, 0.0) == pytest.approx(4000.0, abs=0)
    assert vortex_smt_bathymetry(-300.0 + 150.0, 0.0) == pytest.approx(
        5000.0 - 1000.0 * math.exp(-1.0), rel=1e-15)
    # far field, and the sign of "west": +300 km is NOT the summit
    assert vortex_smt_bathymetry(300.0, 0.0) > 4999.0


def test_resolved_geometry_has_a_full_summit_and_real_partial_faces():
    """The summit sits on a w-level, and e3u is genuinely MIN-of-neighbours.

    OVERFLOW's ``pe3u = pe3t`` shortcut, if it had been ported, makes the
    last assertion fail.
    """
    res = _VORTEX_RESOLUTIONS["30km"]
    source = vortex_horizontal_coordinates(res)
    zht, k_bot, e3t, e3u, e3v, e3f = vortex_smt_partial_cell_geometry(source)
    assert sorted(set(np.unique(k_bot).tolist())) == [8, 9, 10]
    j, i = np.unravel_index(np.argmin(zht), zht.shape)
    assert zht[j, i] == pytest.approx(4000.0, abs=1e-9)
    assert e3t[j, i, 7] == 500.0                       # the summit cell is FULL
    assert np.any(e3u < e3t) and np.any(e3f < e3v)     # not the shortcut
    assert np.all(e3u <= e3t) and np.all(e3f <= e3v)


# -------------------------------------------------------------- the cards ---
def test_initial_temperature_reads_the_cards_own_column_depth():
    """gdept = gdept_1d*(1+ssh/ht_0), and ht_0 is the SEAMOUNT's, not 5000 m.

    This is round 212's statement.  Reverting ``vortex_initial_state_fields``
    to the flat ``_VORTEX_H_M`` makes the seamount card's initial T equal the
    flat card's on every commonly-wet cell, and this test fail.
    """
    smt = build_nemo_testcase_card("VORTEX_SMT-zps")
    flat = build_nemo_testcase_card("VORTEX-zco")
    t_smt = np.asarray(smt.recipe.initial_state.T.data)
    t_flat = np.asarray(flat.recipe.initial_state.T.data)
    both = (np.asarray(smt.recipe.z_coord.is_active)
            & np.asarray(flat.recipe.z_coord.is_active))
    delta = np.abs(t_smt - t_flat)[both]
    # NEMO's own two runs differ by 2.7873925644072983e-05 K here.
    assert delta.max() == pytest.approx(2.7873925644072983e-05, rel=1e-9)
    assert (delta > 0).sum() > 3000


def test_both_momentum_decks_build_and_the_dispatch_is_fail_closed():
    for case, vector in (("VORTEX_SMT-zps", False),
                         ("VORTEX_SMT_VEC-zps", True)):
        card = build_nemo_testcase_card(case)
        assert card.case == case
        cfg = card.recipe.model_config
        assert cfg.momentum_advection == (
            "vector_invariant" if vector else "flux_form"), cfg.momentum_advection
        assert cfg.vorticity_scheme == (
            "een_total" if vector else "een_planetary")
    with pytest.raises(ValueError, match="momentum"):
        build_vortex_smt_zps_card("ens")
    with pytest.raises(ValueError, match="unknown NEMO testcase"):
        build_nemo_testcase_card("VORTEX_SMT-zco")


def test_card_column_depth_is_the_summed_partial_thickness():
    card = build_nemo_testcase_card("VORTEX_SMT-zps")
    z = card.recipe.z_coord
    ht = np.asarray(jnp.sum(z.h_partial, axis=-1))
    wet = np.asarray(card.recipe.land_mask) > 0.0
    assert ht[wet].min() == pytest.approx(4000.0, abs=1e-9)
    assert ht[wet].max() == pytest.approx(5000.0, rel=1e-12)
    ops = z.nemo_een_barotropic
    assert np.asarray(ops.hu_0).max() <= 5000.0 + 1e-9


# --------------------------------------------------------------------------
# Decision 93, rung SMT-1: ORCA2 rung 0's background vertical mixing and its
# enhanced vertical diffusion, on the seamount vector deck and nothing else.
# Every number below is the rung-0 namelist's, cited in the card.
# --------------------------------------------------------------------------

def test_smt1_card_carries_orca2_rung0_namzdf_and_nothing_else():
    base = build_nemo_testcase_card("VORTEX_SMT_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    cfg, base_cfg = card.recipe.model_config, base.recipe.model_config

    # rung-0 namelist_cfg:417 / :418
    assert (cfg.A_v, cfg.K_v) == (1.2e-4, 1.2e-5)
    assert (base_cfg.A_v, base_cfg.K_v) == (1.0e-4, 0.0)

    # rung-0 namelist_cfg:409 ln_zdfevd, :411 rn_evd, :410 nn_evdm = 0
    ed = cfg.physics.convection.enhanced_diffusion
    assert cfg.physics.convection.scheme == "enhanced_diffusion"
    assert (ed.K_conv, ed.nu_conv, ed.K_bg, ed.nu_bg) == (100.0, 0.0, 0.0, 0.0)
    # zdfevd.F90:93  MIN( rn2, rn2b ) <= -1.e-12
    assert (ed.n2_threshold, ed.two_level_trigger) == (-1.0e-12, True)
    assert ed.n2_eos_form == "seos"          # decision 69, this deck's fluid
    # ln_zdfcst with a uniform background is the two scalars above, not a
    # closure; and base carries no physics block at all.
    assert cfg.physics.vertical_mixing.scheme == "none"
    assert base_cfg.physics is None

    # ONE module moved: everything the ladder scores is otherwise the same
    # card.  Compare the two configs field by field and require that the
    # only differences are the three namzdf rows.
    moved = {name for name in cfg._fields
             if getattr(cfg, name) != getattr(base_cfg, name)}
    assert moved == {"A_v", "K_v", "physics"}, moved


def test_smt1_geometry_and_initial_state_are_the_smt0_ones():
    base = build_nemo_testcase_card("VORTEX_SMT_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    z0, z1 = base.recipe.z_coord, card.recipe.z_coord
    for name in ("h_partial", "bottom_level", "is_active"):
        np.testing.assert_array_equal(
            np.asarray(getattr(z0, name)), np.asarray(getattr(z1, name)))
    for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(base.recipe.initial_state, name).data),
            np.asarray(getattr(card.recipe.initial_state, name).data))


def test_smt1_is_vector_only_and_the_rung_dispatch_is_fail_closed():
    with pytest.raises(ValueError, match="mini-ladder rung"):
        build_vortex_smt_zps_card("vector", "smt5")
    with pytest.raises(ValueError, match="VECTOR deck only"):
        build_vortex_smt_zps_card("flux", "smt1")
    with pytest.raises(ValueError, match="VECTOR deck only"):
        build_vortex_smt_zps_card("flux", "smt2")
    with pytest.raises(ValueError, match="VECTOR deck only"):
        build_vortex_smt_zps_card("flux", "smt3")
    with pytest.raises(ValueError, match="VECTOR deck only"):
        build_vortex_smt_zps_card("flux", "smt4")


def test_smt1_validator_refuses_a_card_that_drops_a_rung0_value():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )
    card = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    cfg = card.recipe.model_config
    # the background diffusivity back to the shipped 0.0
    bad = card._replace(recipe=card.recipe._replace(
        model_config=cfg._replace(K_v=0.0)))
    with pytest.raises(ValueError, match="rung 0's rn_avm0"):
        validate_nemo_testcase_card(bad)
    # enhanced vertical diffusion dropped
    ed = cfg.physics.convection.enhanced_diffusion
    bad2 = card._replace(recipe=card.recipe._replace(
        model_config=cfg._replace(physics=cfg.physics._replace(
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=ed._replace(K_conv=0.0))))))
    with pytest.raises(ValueError, match="rn_evd=100"):
        validate_nemo_testcase_card(bad2)
    # nn_evdm silently promoted to 1
    bad3 = card._replace(recipe=card.recipe._replace(
        model_config=cfg._replace(physics=cfg.physics._replace(
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=ed._replace(nu_conv=100.0))))))
    with pytest.raises(ValueError, match="nn_evdm=0"):
        validate_nemo_testcase_card(bad3)


def test_smt1_stated_evd_is_executed_and_measured_inert_on_this_rung():
    """Two MEASURED facts, pinned so neither can rot silently.

    1.  Multiplying ``rn_evd`` by ten thousand changes nothing in a full
        step -- because NEMO's own ``zdfevd`` never fires on this rung:
        with the deck's S-EOS (decision 69) the minimum N^2 over the whole
        seamount run is +9.0e-06 s^-2, ten orders the wrong side of
        ``zdfevd.f90:108``'s -1.e-12 threshold.  This is the card being
        FAITHFULLY inert, not the selection being dropped.
    2.  CORRECTION to round 220, which read this same 0.0 as "the stated
        selection is not reaching the solve".  It reaches it: round 221
        measured the same plant moving temperature by 3.268e-02 K once one
        column is made unstable for the card's own fluid
        (``tests/ocean/fidelity/test_nemo_round221_evd.py``).  The 61 cells
        the trigger used to fire on came from the DEFAULTED S-EOS
        coefficients and were all below the seafloor, where the
        wet-interface mask removes them before the solve.
    """
    import numpy as _np
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    card = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")

    def one_step(k_conv):
        cfg = card.recipe.model_config
        ed = cfg.physics.convection.enhanced_diffusion._replace(K_conv=k_conv)
        cfg = cfg._replace(physics=cfg.physics._replace(
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=ed)))
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg)
        return model.step(card.recipe.initial_state, dt=card.dt_s)

    a, b = one_step(100.0), one_step(1.0e6)
    assert float(_np.max(_np.abs(
        _np.asarray(a.T.data) - _np.asarray(b.T.data)))) == 0.0
    assert float(_np.max(_np.abs(
        _np.asarray(a.u.data) - _np.asarray(b.u.data)))) == 0.0

    # And the fluid itself: NEMO's bn2 with the DECK's coefficients.
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_nemo_bn2, nemo_bn2_live_geometry,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import _VORTEX_SEOS
    st = card.recipe.initial_state
    z = card.recipe.z_coord
    t_depth, w_depth, e3w = nemo_bn2_live_geometry(
        z, st.eta.data, st.H_bathy.data)
    n2 = _np.asarray(compute_buoyancy_frequency_nemo_bn2(
        st.T.data, st.S.data, t_depth, w_depth, _VORTEX_SEOS,
        g=card.recipe.model_config.physics.constants.g,
        eos_form="seos", e3w_int=e3w))
    act = _np.asarray(z.is_active)
    wet = act[..., 1:] & act[..., :-1]
    assert float(_np.min(n2[..., :wet.shape[-1]][wet])) > 1.0e-6


# --- rung SMT-2 (decision 93): ORCA2 rung 0's linear bottom drag -----------


def test_smt2_card_carries_rung0_linear_drag_and_nothing_else():
    """SMT-2 is SMT-1 plus namdrg, and namdrg alone."""
    base = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT2_VEC-zps")
    cfg, base_cfg = card.recipe.model_config, base.recipe.model_config

    # rung-0 namelist_cfg:270 ln_lin; namelist_ref:834 rn_Cd0, :835 rn_Uc0
    assert cfg.bottom_drag.bottom_drag_scheme == "nemo_linear"
    assert cfg.bottom_drag.bottom_drag_cd0 == 1.0e-3
    assert cfg.bottom_drag.bottom_drag_uc0 == 0.4
    # namelist_ref:839 ln_boost = .false., and NEMO drags the bottom cell only
    assert cfg.bottom_drag.bottom_drag_bbl_thickness == 0.0
    assert cfg.bottom_drag.bottom_drag_bg_velocity == 0.0
    assert cfg.bottom_drag.bottom_drag_r == 0.0
    # namelist_ref:817 ln_drgimp = .true. with ln_dynspg_ts = .true.
    assert cfg.zdf_drag_in_matrix is True
    assert cfg.barotropic_drag_substep is True
    assert cfg.zdf_baroclinic_only is True
    # SMT-1 resolves ln_drg_OFF and must stay there
    assert base_cfg.bottom_drag.bottom_drag_scheme == "legacy"
    assert base_cfg.zdf_drag_in_matrix is False
    assert base_cfg.barotropic_drag_substep is False

    moved = {name for name in cfg._fields
             if getattr(cfg, name) != getattr(base_cfg, name)}
    assert moved == {"bottom_drag", "zdf_drag_in_matrix",
                     "barotropic_drag_substep"}, moved


def test_smt2_linear_rate_is_nemos_constant_and_reads_no_velocity():
    """zdfdrg np_lin: rCdU_bot = -rn_Cd0*rn_Uc0, the SAME at every speed."""
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        nemo_effective_bottom_drag_r,
    )
    card = build_nemo_testcase_card("VORTEX_SMT2_VEC-zps")
    bd = card.recipe.model_config.bottom_drag
    kw = dict(scheme=bd.bottom_drag_scheme, cd0=bd.bottom_drag_cd0,
              cd_max=bd.bottom_drag_cdmax, z0=bd.bottom_drag_z0,
              ke0=bd.bottom_drag_ke0, uc0=bd.bottom_drag_uc0,
              von_karman=0.4)
    rest = nemo_effective_bottom_drag_r(
        jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.full((3, 3), 500.0), **kw)
    fast = nemo_effective_bottom_drag_r(
        jnp.full((3, 3), 3.0), jnp.full((3, 3), -2.0),
        jnp.full((3, 3), 17.0), **kw)
    want = 1.0e-3 * 0.4
    np.testing.assert_array_equal(np.asarray(rest), np.full((3, 3), want))
    np.testing.assert_array_equal(np.asarray(fast), np.full((3, 3), want))


def test_nemo_linear_refuses_a_missing_reference_velocity():
    """rn_Uc0 has no default in the law: an unstated one is an error."""
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        nemo_drag_r_from_speed_sq,
    )
    with pytest.raises(ValueError, match="rn_Uc0"):
        nemo_drag_r_from_speed_sq(
            jnp.zeros((2, 2)), jnp.ones((2, 2)), scheme="nemo_linear",
            cd0=1.0e-3, cd_max=0.1, z0=3.0e-3, ke0=2.5e-3, von_karman=0.4)


def test_smt2_validator_refuses_a_card_that_drops_a_drag_value():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )
    card = build_nemo_testcase_card("VORTEX_SMT2_VEC-zps")
    cfg = card.recipe.model_config
    for bad_cfg, match in (
        (cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_scheme="nemo_quadratic")), "linear"),
        (cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_uc0=0.5)), "rn_Uc0"),
        (cfg._replace(zdf_drag_in_matrix=False), "ln_drgimp"),
        (cfg._replace(barotropic_drag_substep=False), "ln_drgimp"),
        (cfg._replace(zdf_baroclinic_only=False), "ln_drgimp"),
        (cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_cd0=2.0e-3)), "rn_Cd0"),
        (cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_bbl_thickness=10.0)), "bottom "),
        (cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_r=1.0e-4)), "bottom "),
    ):
        with pytest.raises(ValueError, match=match):
            validate_nemo_testcase_card(card._replace(
                recipe=card.recipe._replace(model_config=bad_cfg)))


def test_smt1_still_refuses_a_drag_law_it_does_not_resolve():
    """ln_drg_OFF on every rung but SMT-2 -- and the gate says so."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )
    card = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    cfg = card.recipe.model_config
    bad = cfg._replace(bottom_drag=cfg.bottom_drag._replace(
        bottom_drag_scheme="nemo_linear"))
    with pytest.raises(ValueError, match="ln_drg_OFF"):
        validate_nemo_testcase_card(card._replace(
            recipe=card.recipe._replace(model_config=bad)))
    # Reviewer R-3: the guard must reach the cards OUTSIDE the mini-ladder
    # too -- they resolve ln_drg_OFF just as hard, and the older
    # bottom_drag_r test lets a NEMO scheme through at r = 0.
    for other in ("VORTEX_SMT_VEC-zps", "VORTEX_VEC-zco"):
        c2 = build_nemo_testcase_card(other)
        g2 = c2.recipe.model_config
        with pytest.raises(ValueError, match="ln_drg_OFF"):
            validate_nemo_testcase_card(c2._replace(
                recipe=c2.recipe._replace(model_config=g2._replace(
                    bottom_drag=g2.bottom_drag._replace(
                        bottom_drag_scheme="nemo_linear"),
                    zdf_drag_in_matrix=True))))


def test_smt2_geometry_and_initial_state_are_the_smt1_ones():
    base = build_nemo_testcase_card("VORTEX_SMT1_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT2_VEC-zps")
    z0, z1 = base.recipe.z_coord, card.recipe.z_coord
    for name in ("h_partial", "bottom_level", "is_active"):
        np.testing.assert_array_equal(
            np.asarray(getattr(z0, name)), np.asarray(getattr(z1, name)))
    for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(base.recipe.initial_state, name).data),
            np.asarray(getattr(card.recipe.initial_state, name).data))


def test_smt3_card_adds_only_rung0_lateral_tracer_diffusion():
    """Decision 93: SMT-3 is SMT-2 plus namtra_ldf, and nothing else."""
    base = build_nemo_testcase_card("VORTEX_SMT2_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT3_VEC-zps")
    cfg, base_cfg = card.recipe.model_config, base.recipe.model_config

    moved = {name for name in cfg._fields
             if getattr(cfg, name) != getattr(base_cfg, name)}
    assert moved == {"gm_redi"}, moved
    gm = cfg.gm_redi
    assert gm is not None
    assert (gm.kappa_GM, gm.kappa_Redi, gm.S_max) == (0.0, 270.0, 0.01)
    assert gm.slope_scheme == "nemo_iso_lap"
    assert gm.slope_positions == "nemo_native"
    assert gm.slope_limit == "nemo_cap"
    assert gm.msc_stabilize is True
    assert gm.implicit_K33 is True
    assert gm.kappa_redi_horizontal_evaluation == "nemo_metric_literal"
    assert gm.kappa_redi_diffusive_velocity == 0.018

    z0, z1 = base.recipe.z_coord, card.recipe.z_coord
    for name in ("h_partial", "bottom_level", "is_active"):
        np.testing.assert_array_equal(
            np.asarray(getattr(z0, name)), np.asarray(getattr(z1, name)))
    for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(base.recipe.initial_state, name).data),
            np.asarray(getattr(card.recipe.initial_state, name).data))


def test_smt3_validator_refuses_a_dropped_rung0_ldf_value():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )
    card = build_nemo_testcase_card("VORTEX_SMT3_VEC-zps")
    cfg = card.recipe.model_config
    bad = cfg._replace(gm_redi=cfg.gm_redi._replace(msc_stabilize=False))
    with pytest.raises(ValueError, match="msc_stabilize"):
        validate_nemo_testcase_card(card._replace(
            recipe=card.recipe._replace(model_config=bad)))


def test_smt4_card_adds_only_rung0_lateral_momentum_diffusion():
    """Decision 93: SMT-4 is SMT-3 plus namdyn_ldf, and nothing else."""
    base = build_nemo_testcase_card("VORTEX_SMT3_VEC-zps")
    card = build_nemo_testcase_card("VORTEX_SMT4_VEC-zps")
    cfg, base_cfg = card.recipe.model_config, base.recipe.model_config

    moved = {name for name in cfg._fields
             if getattr(cfg, name) != getattr(base_cfg, name)}
    assert moved == {
        "lateral_viscosity", "lateral_viscosity_operator",
        "lateral_viscosity_e3_weighting",
    }, moved
    lv = cfg.lateral_viscosity
    assert lv.A_h == 1500.0
    assert lv.A_h_lat_scaling is False
    assert lv.A_h_floor == 0.0
    assert lv.B_h == lv.C_smag == lv.C_smag_lap == lv.C_leith == 0.0
    assert cfg.lateral_viscosity_operator == "nemo_div_curl"
    assert cfg.lateral_viscosity_e3_weighting == "nemo_e3"
    assert cfg.lateral_viscosity_coefficient_source == "nemo_ldf_c2d"
    assert cfg.lateral_side_bc == "free_slip"
    assert cfg.gm_redi == base_cfg.gm_redi

    z0, z1 = base.recipe.z_coord, card.recipe.z_coord
    for name in ("h_partial", "bottom_level", "is_active"):
        np.testing.assert_array_equal(
            np.asarray(getattr(z0, name)), np.asarray(getattr(z1, name)))
    for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(base.recipe.initial_state, name).data),
            np.asarray(getattr(card.recipe.initial_state, name).data))


def test_smt4_validator_refuses_a_dropped_momentum_ldf_value():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card,
    )
    card = build_nemo_testcase_card("VORTEX_SMT4_VEC-zps")
    cfg = card.recipe.model_config
    bad = cfg._replace(
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0))
    with pytest.raises(ValueError, match="mode-20 rn_Uv=0.1"):
        validate_nemo_testcase_card(card._replace(
            recipe=card.recipe._replace(model_config=bad)))


def test_smt2_implicit_drag_divides_by_nemos_bottom_face_thickness():
    """The divisor the step RUNS is NEMO's, and the old rule was not inert.

    ``dynzdf.f90:306`` (and its V twin ``:473``, and the barotropic
    bottom-stress re-add ``:166``/``:168``) divide the semi-implicit drag term
    by ``e3u_3d(ji,jj,iku)*(1+r3u(ji,jj,Kaa)*umask(ji,jj,iku))`` -- the
    REFERENCE face thickness, loaded from ``e3u_0`` at ``domzgr.f90:186``,
    which over z partial steps is the MINIMUM of the two neighbouring
    reference T thicknesses.  legoESM used to divide by the simple two-cell
    AVERAGE of the live thicknesses.

    PLANT, so this test cannot pass vacuously: put the old average rule back
    in place of the one the drag block now calls, and one step on the
    seamount card must MOVE the velocity.

    The SMT-1 arm pins a DIFFERENT fact, and the reason matters: SMT-1 is the
    SAME seamount geometry (see
    ``test_smt2_geometry_and_initial_state_are_the_smt1_ones``), so the two
    divisor rules differ there just as much.  It resolves no bottom drag, so
    the card validator forbids ``zdf_drag_in_matrix`` on it
    (``nemo_testcase_recipe.py``) and the planted symbol is never reached --
    which is why the eleven other certified registries are inert here.
    """
    import numpy as _np
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as _mod
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_uface
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        interp_to_v_points,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    def _one_step(case, plant):
        card = build_nemo_testcase_card(case)
        cfg, zc, grid = (card.recipe.model_config, card.recipe.z_coord,
                         card.recipe.grid)
        st = card.recipe.initial_state
        real = _mod._nemo_dynzdf_drag_face_thickness

        def _old_average_rule(eta, hb, z, c, g, dtype):
            dz = compute_layer_thickness(
                eta, hb, z, min_water_column_m=c.min_water_column_m)
            return (interp_cell_to_uface(dz).astype(dtype),
                    interp_to_v_points(dz, g).astype(dtype))

        if plant:
            _mod._nemo_dynzdf_drag_face_thickness = _old_average_rule
        try:
            model = _mod.LatLonCGridOceanModel(grid, zc, cfg)
            out = model.step(st, dt=card.dt_s)
        finally:
            _mod._nemo_dynzdf_drag_face_thickness = real
        return _np.asarray(out.u.data), _np.asarray(out.v.data)

    # The drag block runs on SMT-2 (zdf_drag_in_matrix), and the seamount's
    # bottom faces are partial, so the plant must be visible.
    u_nemo, v_nemo = _one_step("VORTEX_SMT2_VEC-zps", plant=False)
    u_old, v_old = _one_step("VORTEX_SMT2_VEC-zps", plant=True)
    du = float(_np.max(_np.abs(u_nemo - u_old)))
    dv = float(_np.max(_np.abs(v_nemo - v_old)))
    assert du > 1.0e-6, f"the plant is inert on the seamount: du={du!r}"
    assert dv > 1.0e-7, f"the plant is inert on the seamount: dv={dv!r}"

    # SMT-1 resolves no bottom drag at all, so the block does not execute,
    # the planted symbol is never called, and the step is unchanged.  This
    # pins REACHABILITY, not the arithmetic: it is the unit-level form of the
    # eleven inert registries.
    u1_a, v1_a = _one_step("VORTEX_SMT1_VEC-zps", plant=False)
    u1_b, v1_b = _one_step("VORTEX_SMT1_VEC-zps", plant=True)
    assert float(_np.max(_np.abs(u1_a - u1_b))) == 0.0
    assert float(_np.max(_np.abs(v1_a - v1_b))) == 0.0
