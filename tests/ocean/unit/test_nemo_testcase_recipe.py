from types import SimpleNamespace

import jax.numpy as jnp
import legoesm.ocean.fidelity.nemo_testcase_recipe as testcase_recipe
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import nemo_potential_temperature_from_conservative
from legoesm.ocean.fidelity.nemo_recipe import (
    nemo_gyre_emp,
    nemo_gyre_seasonal_cosines,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    _orca2_depth_ladder,
    _orca2_masks,
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_nemo_testcase_card,
    build_overflow_zps_card,
    gyre_horizontal_coordinates,
    gyre_surface_boundary_condition,
    gyre_vertical_ladder,
    validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)
from legoesm.ocean.vertical import nemo_fe3mask_from_tmask


@pytest.fixture(autouse=True)
def _restore_precision():
    old = get_policy()
    yield
    set_policy(old)


def test_orca2_execution_guard_rejects_registered_unmeasured_arms():
    card = SimpleNamespace(
        case="ORCA2-zps",
        unmeasured_features=("si3_jpl5_layered_prather_state",),
    )
    # Isolate the execution guard from the structural validator: the complete
    # deck-backed card is exercised in the phase-2 gate, not this unit test.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            testcase_recipe, "validate_nemo_testcase_card", lambda _: None
        )
        with pytest.raises(ValueError, match="not execution-ready"):
            validate_nemo_testcase_card_for_execution(card)


def test_orca2_structural_guard_rejects_iceberg_option_drift():
    """The comparison card cannot silently revert to the shipped icb arm."""
    set_policy(PrecisionPolicy.fp64())
    # Reach the ORCA2 option check without constructing the external
    # deck-backed state in this small unit test.
    good = build_gyre_zco_card()
    tke = good.recipe.model_config.physics.vertical_mixing.tke._replace(
        tke_shear_production="nemo_face_native_nbb2",
        tke_shear_avm_weighting="nemo_face",
        tke_shear_evaluation_stage="step_entry",
        tke_shear_metric_source="nemo_qco_live_face",
        tke_langmuir_evaluation="vectorized",
        bottom_tke_bc=True,
        eice=1,
    )
    card = good._replace(
        case="ORCA2-zps",
        surface_boundary_condition="ncar_core_sbcblk",
        surface_input_operator="nemo_fld_read",
        icebergs_enabled=True,
        iceberg_inputs=("icebergs_restart.nc",),
        unmeasured_features=("si3_jpl5_layered_prather_state",),
        recipe=good.recipe._replace(
            model_config=good.recipe.model_config._replace(
                eos="nemo_eos80",
                vorticity_scheme="een_total",
                barotropic=good.recipe.model_config.barotropic._replace(
                    barotropic_coriolis="een_metric",
                    n_barotropic_substeps=65,
                    nemo_barotropic_filter_alpha=0.09,
                ),
                physics=good.recipe.model_config.physics._replace(
                    vertical_mixing=(good.recipe.model_config.physics
                                     .vertical_mixing._replace(tke=tke))
                ),
                bbl_diffusive_option=1,
                bbl_aht_m2_s=1000.0,
            )
        ),
        bbl_diffusive_option=1,
        bbl_aht_m2_s=1000.0,
    )
    with pytest.raises(ValueError, match="ln_icebergs=F"):
        validate_nemo_testcase_card(card)


@pytest.mark.parametrize(
    ("builder", "shape", "nlev", "dt", "cold", "warm"),
    [
        (build_lock_exchange_zco_card, (3, 130), 20, 1.0, 5.0, 30.0),
        (build_overflow_zps_card, (3, 202), 100, 10.0, 10.0, 20.0),
    ],
)
def test_nemo_testcase_cards_are_fp64_source_pinned(
    builder, shape, nlev, dt, cold, warm
):
    # The harness owns policy selection; cards merely honor the active policy.
    set_policy(PrecisionPolicy.fp64())
    card = builder()
    recipe = card.recipe
    state = recipe.initial_state

    assert get_policy() == PrecisionPolicy.fp64()
    assert (recipe.grid.n_lat, recipe.grid.n_lon) == shape
    assert recipe.z_coord.n_levels == nlev
    assert card.dt_s == dt
    assert card.transcendentals == "libm"
    assert card.dummy_bottom_records == 1
    for array in (
        recipe.grid.dx_T,
        recipe.z_coord.dz_ref,
        recipe.z_coord.h_partial,
        state.T.data,
        state.S.data,
        state.u.data,
        state.v.data,
        state.eta.data,
        state.H_bathy.data,
    ):
        assert jnp.asarray(array).dtype == jnp.float64

    wet3 = np.asarray(recipe.z_coord.is_active) & (
        np.asarray(state.land_mask.data) > 0.5
    )[..., None]
    assert set(np.unique(np.asarray(state.T.data)[wet3])) == {cold, warm}
    assert np.array_equal(np.asarray(state.S.data)[wet3], np.full(wet3.sum(), 35.0))
    assert not np.asarray(state.u.data).any()
    assert not np.asarray(state.v.data).any()
    assert not np.asarray(state.eta.data).any()

    cfg = recipe.model_config
    assert cfg.eos == "nemo_teos10"
    assert cfg.eos_depth == "geometric"
    assert cfg.tracer_advection == "fct2"
    assert cfg.momentum_advection == "flux_form"
    assert cfg.momentum_flux_scheme == "nemo_up3"
    assert cfg.momentum_time_integrator == "rk3_ws"
    assert cfg.tracer_time_integrator == "rk3_ws"
    assert "tracer_fct_low_order_predictor" not in cfg._fields
    assert "tracer_rk3_transport_time_levels" not in cfg._fields
    assert "rk3_ws_stage_barotropic_correction" not in cfg._fields
    assert "rk3_ws_momentum_transport_reconcile" not in cfg._fields
    assert cfg.vertical_momentum_scheme == "nemo_up3"
    assert cfg.pgf_scheme == "nemo_sco"
    assert cfg.pgf_quadrature == "nemo_trapezoid"
    assert cfg.adaptive_implicit_vertadv
    assert cfg.barotropic.barotropic_diffusion_alpha == 0.0
    assert cfg.barotropic.barotropic_reconcile_target == "velocity_avg"


def test_lane1_cards_carry_oracle_w_grid_thicknesses():
    """The NEMO SCO PGF consumes mesh ``e3w_0``; no midpoint fallback."""
    set_policy(PrecisionPolicy.fp64())
    lock = build_lock_exchange_zco_card().recipe.z_coord
    overflow = build_overflow_zps_card().recipe.z_coord
    assert lock.nemo_e3w_mesh_reference is True
    assert overflow.nemo_e3w_mesh_reference is True
    np.testing.assert_array_equal(
        np.asarray(lock.nemo_e3w_0), np.ones((3, 130, 20)))
    # OVERFLOW carries usrdef_zgr.F90:157-168's 1-D reference ladder (the
    # same 20 m in every column, partial-bottom included), which is also the
    # layout the raw-W ZDF/wAimp consumers read.  Broadcast-identical to the
    # 3-D mesh_mask field the SCO pressure gradient needs.
    np.testing.assert_array_equal(
        np.asarray(overflow.nemo_e3w_0), np.full((100,), 20.0))
    np.testing.assert_array_equal(
        np.broadcast_to(np.asarray(overflow.nemo_e3w_0), (3, 202, 100)),
        np.full((3, 202, 100), 20.0))


def test_gyre_card_selects_complete_resolved_operator_program():
    set_policy(PrecisionPolicy.fp64())
    cfg = build_gyre_zco_card().recipe.model_config
    assert cfg.momentum_advection == "vector_invariant"
    assert cfg.vorticity_scheme == "ene_total"
    assert cfg.ke_gradient_scheme == "c2"
    assert cfg.vertical_momentum_scheme == "nemo_advective"
    assert cfg.adaptive_implicit_vertadv is False
    assert cfg.lateral_viscosity_operator == "nemo_div_curl"
    assert cfg.lateral_viscosity_e3_weighting == "nemo_e3"
    assert cfg.lateral_viscosity.A_h == 1.0e5
    assert cfg.gm_redi.kappa_GM == 0.0
    assert cfg.gm_redi.kappa_Redi == 1000.0
    assert cfg.gm_redi.slope_scheme == "nemo_iso_lap"
    assert cfg.A_v == 0.0 and cfg.K_v == 0.0
    assert cfg.physics.vertical_mixing.scheme == "tke"
    assert cfg.physics.vertical_mixing.tke.prognostic is True
    assert cfg.physics.vertical_mixing.tke.kappaM_min == 1.2e-4
    assert cfg.physics.vertical_mixing.tke.kappaH_min == 1.2e-5
    assert cfg.physics.vertical_mixing.tke.n2_eos_form == "teos10"
    assert cfg.physics.vertical_mixing.tke.tke_n2_time_level == "nemo_before"
    assert (
        cfg.physics.vertical_mixing.tke.tke_langmuir_evaluation
        == "nemo_literal"
    )
    assert cfg.physics.convection.scheme == "enhanced_diffusion"
    assert cfg.physics.convection.enhanced_diffusion.K_conv == 100.0
    assert cfg.physics.convection.enhanced_diffusion.nu_conv == 100.0
    assert (
        cfg.physics.convection.enhanced_diffusion.evd_n2_time_level
        == "nemo_now_before"
    )
    assert cfg.physics.shortwave_penetration.scheme == "nemo_qsr_2bd"
    assert cfg.physics.shortwave_penetration.water_type == "I"
    assert cfg.barotropic.barotropic_coriolis == "ene_metric"
    assert cfg.barotropic.barotropic_een_coefficient_evaluation == "nemo_literal"
    assert cfg.bottom_drag.bottom_drag_scheme == "nemo_quadratic"
    assert cfg.bottom_drag.bottom_drag_cd0 == 1.0e-3
    assert cfg.bottom_drag.bottom_drag_ke0 == 2.5e-3
    assert cfg.zdf_drag_in_matrix is True
    assert cfg.zdf_baroclinic_only is True
    assert cfg.barotropic_drag_substep is True


def test_gyre_rk3_evd_uses_step_entry_for_both_n2_arms():
    set_policy(PrecisionPolicy.fp64())
    card = build_gyre_zco_card()
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    T_before, S_before = model._n2_nemo_before_tracers(
        card.recipe.initial_state)
    assert T_before is card.recipe.initial_state.T.data
    assert S_before is card.recipe.initial_state.S.data


def test_gyre_whole_step_identity_rejects_hybrid_and_staged_gm():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    set_policy(PrecisionPolicy.fp64())
    recipe = build_gyre_zco_card().recipe
    LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    with pytest.raises(ValueError, match="one complete momentum program"):
        LatLonCGridOceanModel(
            recipe.grid,
            recipe.z_coord,
            recipe.model_config._replace(ke_gradient_scheme="centered"),
        )
    with pytest.raises(ValueError, match="staged GM bolus"):
        LatLonCGridOceanModel(
            recipe.grid,
            recipe.z_coord,
            recipe.model_config._replace(
                gm_redi=recipe.model_config.gm_redi._replace(kappa_GM=1.0)
            ),
        )


def test_orca2_shared_ws_program_admits_een_without_admitting_a_hybrid():
    """ORCA2 changes dynvor ENE -> EEN, not the surrounding WS program."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    set_policy(PrecisionPolicy.fp64())
    recipe = build_gyre_zco_card().recipe
    cfg = recipe.model_config._replace(
        vorticity_scheme="een_total",
        barotropic=recipe.model_config.barotropic._replace(
            barotropic_coriolis="een_metric"
        ),
    )
    LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    with pytest.raises(ValueError, match="incompatible with adaptive_implicit"):
        LatLonCGridOceanModel(
            recipe.grid,
            recipe.z_coord,
            cfg._replace(adaptive_implicit_vertadv=True),
        )


def test_orca2_depth_ladder_keeps_source_scalar_recurrence():
    e3t = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    e3w = np.array([0.5, 1.5, 3.0], dtype=np.float64)
    gdept, gdepw = _orca2_depth_ladder(e3t, e3w)
    np.testing.assert_array_equal(gdepw, [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(gdept, [0.25, 1.75, 4.75])


def test_orca2_mask_builder_applies_t_fold_and_strait_override():
    bottom = np.array(
        [[2, 2, 1, 1], [2, 1, 2, 1], [1, 2, 2, 1]], dtype=np.int32
    )
    strait = np.full((3, 4), -1.0)
    strait[1, 1] = 0.5
    tmask, umask, vmask, fmask = _orca2_masks(bottom, strait)
    assert tmask.shape == umask.shape == vmask.shape == fmask.shape == (3, 4, 2)
    np.testing.assert_array_equal(vmask[-1], vmask[-2, [0, 3, 2, 1]])
    np.testing.assert_array_equal(fmask[-1], fmask[-2, [3, 2, 1, 0]])
    np.testing.assert_array_equal(fmask[1, 1], [0.5, 0.5])


def test_nemo_fe3mask_precedes_lateral_slip_changes():
    bottom = np.array(
        [[2, 2, 1, 1], [2, 1, 2, 1], [1, 2, 2, 1]], dtype=np.int32
    )
    strait = np.full((3, 4), -1.0)
    strait[1, 1] = 0.5
    tmask, _, _, fmask = _orca2_masks(bottom, strait)
    fe3mask = np.asarray(nemo_fe3mask_from_tmask(tmask.astype(np.float64)))
    assert fmask[1, 1, 0] == 0.5
    assert fe3mask[1, 1, 0] == 1.0
    assert fe3mask[1, 1, 0] != fmask[1, 1, 0]
    assert set(np.unique(fe3mask)).issubset({0.0, 1.0})


def test_testcase_cards_select_their_resolved_barotropic_filters():
    lock_baro = build_lock_exchange_zco_card().recipe.model_config.barotropic
    overflow_baro = build_overflow_zps_card().recipe.model_config.barotropic
    gyre_baro = build_gyre_zco_card().recipe.model_config.barotropic
    assert lock_baro.barotropic_time_filter == "nemo_ab3am4"
    assert overflow_baro.barotropic_time_filter == "nemo_boxcar1_ab3"
    assert gyre_baro.barotropic_time_filter == "nemo_ab3am4"
    assert lock_baro.nemo_barotropic_filter_alpha == 0.07
    assert overflow_baro.nemo_barotropic_filter_alpha == 0.0
    assert gyre_baro.nemo_barotropic_filter_alpha == 0.07
    # Resolved ln_bt_auto counts, independently printed by the pinned NEMO
    # runs: LOCK ocean.output:763 and OVERFLOW ocean.output:879.
    assert lock_baro.n_barotropic_substeps == 1
    assert overflow_baro.n_barotropic_substeps == 3
    assert gyre_baro.n_barotropic_substeps == 50

    missing = build_gyre_zco_card()
    missing = missing._replace(
        recipe=missing.recipe._replace(
            model_config=missing.recipe.model_config._replace(
                barotropic=missing.recipe.model_config.barotropic._replace(
                    nemo_barotropic_filter_alpha=None))))
    with pytest.raises(ValueError, match="must state its deck's rn_bt_alpha"):
        validate_nemo_testcase_card(missing)


def test_gyre_card_pins_rotated_grid_mi96_ic_and_seasonal_sbc():
    set_policy(PrecisionPolicy.fp64())
    card = build_gyre_zco_card()
    recipe = card.recipe
    source = gyre_horizontal_coordinates()
    ladder = gyre_vertical_ladder()

    assert (recipe.grid.n_lat, recipe.grid.n_lon) == (22, 32)
    assert recipe.z_coord.n_levels == 30
    assert card.dt_s == 14400.0
    assert card.n_steps == 4320
    assert card.surface_boundary_condition == "gyre_usrdef_sbc"
    assert card.transcendentals == "libm"
    assert testcase_recipe._resolved_auto_substeps(
        recipe.grid, recipe.initial_state.H_bathy.data, card.dt_s
    ) == 50
    assert np.array_equal(np.asarray(recipe.grid.native_lat_T_deg), source["gphit"])
    assert np.array_equal(np.asarray(recipe.grid.f_T), source["ff_t"])
    expected_f_v = np.concatenate(
        [
            source["ff_f"][:1]
            - (source["ff_f"][1:2] - source["ff_f"][:1]),
            source["ff_f"],
        ],
        axis=0,
    )
    # Pin the certified GYRE generic f_v bytes while carrying native NEMO ff_f
    # independently for the ENE arm.
    np.testing.assert_array_equal(np.asarray(recipe.grid.f_v), expected_f_v)
    np.testing.assert_array_equal(np.asarray(recipe.grid.ff_f), source["ff_f"])
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_een_ene_vertex_coriolis,
        vertex_coriolis,
    )
    generic_vertex = np.concatenate(
        [expected_f_v, expected_f_v[:, :1]], axis=1
    )
    native_with_south = np.concatenate(
        [source["ff_f"][:1], source["ff_f"]], axis=0
    )
    nemo_vertex = np.concatenate(
        [native_with_south[:, -1:], native_with_south], axis=1
    )
    np.testing.assert_array_equal(
        np.asarray(vertex_coriolis(recipe.grid)), generic_vertex
    )
    np.testing.assert_array_equal(
        np.asarray(nemo_een_ene_vertex_coriolis(recipe.grid)), nemo_vertex
    )
    planted = recipe.grid._replace(ff_f=recipe.grid.f_T)
    assert not np.array_equal(
        np.asarray(nemo_een_ene_vertex_coriolis(planted)), nemo_vertex
    )
    assert np.array_equal(np.asarray(recipe.z_coord.dz_ref), ladder["e3t_1d"][:30])
    assert np.array_equal(
        np.asarray(recipe.z_coord.nemo_e3w_0)[1, 1], ladder["e3w_1d"][:30]
    )
    assert np.count_nonzero(np.asarray(recipe.z_coord.nemo_hu_0)) == 580
    assert np.count_nonzero(np.asarray(recipe.z_coord.nemo_hv_0)) == 570
    assert np.array_equal(
        np.asarray(recipe.z_coord.nemo_e1e2t),
        np.asarray(recipe.grid.dx_T) * np.asarray(recipe.grid.dy_T),
    )
    assert np.array_equal(
        np.asarray(recipe.z_coord.t_depth_ref), ladder["gdept_1d"][:30]
    )
    assert np.count_nonzero(np.asarray(recipe.land_mask)) == 600
    assert not np.asarray(recipe.initial_state.u.data).any()
    assert not np.asarray(recipe.initial_state.v.data).any()
    assert not np.asarray(recipe.initial_state.eta.data).any()
    assert np.ptp(np.asarray(recipe.initial_state.T.data)[1:-1, 1:-1]) > 0.0
    assert np.ptp(np.asarray(recipe.initial_state.S.data)[1:-1, 1:-1]) > 0.0

    at_kt1 = gyre_surface_boundary_condition(card, card.dt_s)
    at_half_year = gyre_surface_boundary_condition(
        card, card.dt_s + 180.0 * 86400.0
    )
    cos1, cos2 = nemo_gyre_seasonal_cosines(card.dt_s + 90.0 * 86400.0)
    assert float(cos1) == pytest.approx(0.159, abs=5.0e-4)
    assert float(cos2) == pytest.approx(-0.356, abs=5.0e-4)
    assert at_kt1.qsr_w_m2.dtype == jnp.float64
    assert not np.array_equal(at_kt1.qsr_w_m2, at_half_year.qsr_w_m2)
    wet = np.asarray(recipe.land_mask) > 0.5
    lat = np.asarray(recipe.grid.native_lat_T_deg)
    raw_emp = np.asarray(nemo_gyre_emp(lat, card.dt_s))
    final_emp = np.asarray(at_kt1.emp_kg_m2_s)
    land_contribution = float(np.sum(raw_emp[~wet]))
    assert land_contribution != 0.0
    assert np.array_equal(final_emp[~wet], raw_emp[~wet])
    # glob_2Dsum's unmasked source argument is still ownership-masked: the
    # cropped card's 104 boundary-ring cells are not owned contributions.
    assert float(np.sum(final_emp[wet])) == pytest.approx(0.0, abs=1.0e-17)
    assert card.recipe.model_config.barotropic_coriolis_split == "live"
    assert card.recipe.model_config.barotropic.barotropic_coriolis == "ene_metric"


@pytest.mark.parametrize(
    ("builder", "field", "value"),
    [
        (build_lock_exchange_zco_card, "barotropic_time_filter",
         "nemo_boxcar1_ab3"),
        (build_overflow_zps_card, "n_barotropic_substeps", 1),
        (build_overflow_zps_card, "barotropic_diffusion_alpha", 0.01),
    ],
)
def test_card_validation_rejects_unattested_compositions(builder, field, value):
    card = builder()
    baro = card.recipe.model_config.barotropic._replace(**{field: value})
    cfg = card.recipe.model_config._replace(barotropic=baro)
    mutated = card._replace(recipe=card.recipe._replace(model_config=cfg))
    with pytest.raises(ValueError):
        validate_nemo_testcase_card(mutated)


def test_global_validation_rejects_frankenstein_rk3_and_eos_pgf_pairs():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    card = build_lock_exchange_zco_card()
    base = card.recipe.model_config
    bad = (
        base._replace(momentum_time_integrator="euler"),
        base._replace(tracer_advection="ppm_fct"),
        base._replace(eos_depth="insitu"),
    )
    for cfg in bad:
        with pytest.raises(ValueError):
            LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def test_card_validation_rejects_a_swapped_pressure_gradient():
    """`pgf_scheme` moved from the MODEL guard to the CARD validator.

    The model-level trapezoid allow-list was widened to
    {"nemo_sco", "adcroft"} on the oracle's own evidence -- hpg_zco
    (dynhpg.F90:270-296) accumulates the same e3w(Kmm) trapezoid as hpg_sco
    (:343-374) -- so `adcroft` + `nemo_trapezoid` builds now.  It is still a
    frankenstein for THESE cards, whose certified identity is hpg_sco, and the
    card validator is what owns that.  This is the same assertion, relocated,
    not a dropped one.
    """
    card = build_lock_exchange_zco_card()
    cfg = card.recipe.model_config._replace(pgf_scheme="adcroft")
    mutated = card._replace(recipe=card.recipe._replace(model_config=cfg))
    with pytest.raises(ValueError, match="hpg_sco pressure gradient"):
        validate_nemo_testcase_card(mutated)


def test_testcase_cards_construct_the_shared_canonical_model():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    for builder in (build_lock_exchange_zco_card, build_overflow_zps_card):
        card = builder()
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
        )
    assert (
        build_overflow_zps_card().recipe.model_config.barotropic.barotropic_time_filter
        == "nemo_boxcar1_ab3"
    )


def test_card_builder_does_not_mutate_precision_policy():
    set_policy(PrecisionPolicy.fp32())
    build_lock_exchange_zco_card()
    assert get_policy() == PrecisionPolicy.fp32()


def test_testcase_cards_pin_the_certified_bbl_selectors():
    lock = build_lock_exchange_zco_card()
    overflow = build_overflow_zps_card()
    assert (lock.bbl_adv_option, lock.bbl_diffusive_option) == (0, 0)
    assert (overflow.bbl_adv_option, overflow.bbl_diffusive_option) == (2, 0)
    assert overflow.bbl_aht_m2_s == 1000.0
    assert overflow.bbl_gamma_s == 20.0
    assert lock.recipe.model_config.bbl_aht_m2_s == 0.0
    assert overflow.recipe.model_config.bbl_aht_m2_s == 1000.0


def test_overflow_card_carries_source_exact_unmasked_bbl_mesh_operands():
    """usrdef_zgr.F90:157-186 + trabbl.F90:517-533, without mesh I/O."""
    from legoesm.ocean.physics.bbl_adv import nemo_bbl_static_geometry

    recipe = build_overflow_zps_card().recipe
    z = recipe.z_coord
    assert np.array_equal(
        np.asarray(z.nemo_gdept_0),
        10.0 + 20.0 * np.arange(100, dtype=np.float64),
    )
    assert np.asarray(z.nemo_bbl_e3u_0).shape == (3, 201, 100)
    assert np.asarray(z.nemo_bbl_e3v_0).shape == (2, 202, 100)
    geom = nemo_bbl_static_geometry(
        z.h_partial,
        recipe.land_mask,
        z.nemo_gdept_0,
        z.nemo_bbl_e3u_0,
        z.nemo_bbl_e3v_0,
    )
    # One wet physical row; exactly 29 reference-bottom transitions in x.
    assert int(np.sum(np.asarray(geom.u_active)[1])) == 29
    assert int(np.sum(np.asarray(geom.u_active)[0])) == 0
    assert int(np.sum(np.asarray(geom.u_active)[2])) == 0
    # Same-bottom partial-depth faces are exactly off in NEMO geometry.
    bottom = np.asarray(z.bottom_level)[1]
    same_bottom = bottom[:-1] == bottom[1:]
    assert not np.any(np.asarray(geom.u_active)[1][same_bottom])


def test_overflow_nemo_w_consumers_use_the_raw_reference_ladder():
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w_kmm

    card = build_overflow_zps_card()
    z = card.recipe.z_coord
    h = np.asarray(z.h_partial)
    stretch = np.full(h.shape[:-1], 1.00025)
    got = np.asarray(nemo_e3w_kmm(z, h, stretch))
    expected = np.broadcast_to(20.0 * stretch[..., None], got.shape)
    np.testing.assert_array_equal(got, expected)
    midpoint = 0.5 * (h[..., :-1] + h[..., 1:]) * stretch[..., None]
    wet = np.asarray(z.is_active[..., :-1] & z.is_active[..., 1:])
    assert np.max(np.abs(got[wet] - midpoint[wet])) > 4.9
    np.testing.assert_array_equal(
        np.asarray(z.nemo_e3w_0), np.full((100,), 20.0))
    assert z.nemo_e3w_mesh_reference is True


def test_overflow_card_uses_partial_cells_and_minimum_face_rule():
    recipe = build_overflow_zps_card().recipe
    hp = np.asarray(recipe.z_coord.h_partial)
    wet = np.asarray(recipe.z_coord.is_active)
    assert np.any((hp > 0.0) & (hp < 20.0))
    assert np.allclose(hp.sum(axis=-1), np.asarray(recipe.initial_state.H_bathy.data))
    assert np.array_equal(wet.sum(axis=-1) - 1, np.asarray(recipe.z_coord.bottom_level))


def test_nemo_testcase_dispatch_rejects_unknown_case():
    with pytest.raises(ValueError, match="unknown NEMO testcase"):
        build_nemo_testcase_card("OVERFLOW-sco")


def test_nemo_testcase_dispatch_does_not_hide_builder_keyerror(monkeypatch):
    def broken_builder():
        raise KeyError("internal-card-defect")

    monkeypatch.setattr(testcase_recipe, "build_overflow_zps_card", broken_builder)
    with pytest.raises(KeyError, match="internal-card-defect"):
        testcase_recipe.build_nemo_testcase_card("OVERFLOW-zps")


def test_gyre_teos_surface_operand_and_full_two_band_identity():
    """Pin eosbn2.F90:1500 and the source-named qsr selector."""
    set_policy(PrecisionPolicy.fp64())
    pt = nemo_potential_temperature_from_conservative(
        jnp.asarray(20.0, dtype=jnp.float64),
        jnp.asarray(35.7, dtype=jnp.float64),
    )
    assert float(pt) == pytest.approx(20.02391895, abs=5.0e-7)
    gyre = build_gyre_zco_card().recipe.model_config
    lane1 = build_lock_exchange_zco_card().recipe.model_config
    overflow = build_overflow_zps_card().recipe.model_config
    assert gyre.physics.shortwave_penetration.scheme == "nemo_qsr_2bd"
    assert lane1.physics is None
    assert overflow.physics is None
    assert "nemo_two_band_full_shortwave" not in gyre._fields


def test_nemo_tpoint_bottom_rule_is_selectable_and_unsnapped():
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    z_ref = create_z_star_from_thicknesses(
        jnp.full((3,), 20.0), t_depth_ref_m=jnp.asarray([10.0, 30.0, 50.0])
    )
    depth = jnp.asarray([[39.9999, 40.0001, 50.0]])
    legacy = create_partial_cell_coordinate(z_ref, depth)
    nemo = create_partial_cell_coordinate(
        z_ref, depth, bottom_index_rule="nemo_tpoint"
    )
    assert np.array_equal(np.asarray(legacy.bottom_level), [[1, 2, 2]])
    assert np.array_equal(np.asarray(nemo.bottom_level), [[1, 1, 1]])
    assert float(nemo.h_partial[0, 0, 1]) == pytest.approx(19.9999)
    assert float(nemo.h_partial[0, 1, 1]) == 20.0
    assert float(nemo.h_partial[0, 2, 1]) == 20.0
    with pytest.raises(ValueError, match="bottom_index_rule"):
        create_partial_cell_coordinate(z_ref, depth, bottom_index_rule="average")


def test_nemo_tpoint_requires_explicit_t_depth_on_stretched_grid():
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    # Midpoints are [5, 20, 45] m, whereas the external T points are
    # [5, 25, 50] m.  At H=22 m the midpoint fallback would silently choose
    # bottom level 1 while NEMO's actual T-point rule chooses level 0.
    depth = jnp.asarray([[22.0]])
    missing = create_z_star_from_thicknesses([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="requires an explicit t_depth_ref"):
        create_partial_cell_coordinate(
            missing, depth, bottom_index_rule="nemo_tpoint"
        )

    pinned = create_z_star_from_thicknesses(
        [10.0, 20.0, 30.0], t_depth_ref_m=[5.0, 25.0, 50.0]
    )
    nemo = create_partial_cell_coordinate(
        pinned, depth, bottom_index_rule="nemo_tpoint"
    )
    assert int(nemo.bottom_level[0, 0]) == 0
