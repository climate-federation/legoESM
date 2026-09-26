import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
import legoesm.ocean.fidelity.nemo_testcase_recipe as testcase_recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_lock_exchange_zco_card,
    build_nemo_testcase_card,
    build_overflow_zps_card,
    validate_nemo_testcase_card,
)


@pytest.fixture(autouse=True)
def _restore_precision():
    old = get_policy()
    yield
    set_policy(old)


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


def test_testcase_cards_select_their_resolved_barotropic_filters():
    lock_baro = build_lock_exchange_zco_card().recipe.model_config.barotropic
    overflow_baro = build_overflow_zps_card().recipe.model_config.barotropic
    assert lock_baro.barotropic_time_filter == "nemo_ab3am4"
    assert overflow_baro.barotropic_time_filter == "nemo_boxcar1_ab3"
    # Resolved ln_bt_auto counts, independently printed by the pinned NEMO
    # runs: LOCK ocean.output:763 and OVERFLOW ocean.output:879.
    assert lock_baro.n_barotropic_substeps == 1
    assert overflow_baro.n_barotropic_substeps == 3


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
        base._replace(pgf_scheme="adcroft"),
    )
    for cfg in bad:
        with pytest.raises(ValueError):
            LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


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
