import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_lock_exchange_zco_card,
    build_nemo_testcase_card,
    build_overflow_zps_card,
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
    set_policy(PrecisionPolicy.fp32())
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
    assert cfg.eos == "veros_gsw"
    assert cfg.tracer_advection == "fct2"
    assert cfg.momentum_advection == "flux_form"
    assert cfg.momentum_flux_scheme == "upwind3"
    assert cfg.momentum_time_integrator == "rk3_ws"
    assert cfg.adaptive_implicit_vertadv


def test_testcase_cards_pin_the_certified_bbl_selectors():
    lock = build_lock_exchange_zco_card()
    overflow = build_overflow_zps_card()
    assert (lock.bbl_adv_option, lock.bbl_diffusive_option) == (0, 0)
    assert (overflow.bbl_adv_option, overflow.bbl_diffusive_option) == (2, 0)
    assert overflow.bbl_aht_m2_s == 1000.0
    assert overflow.bbl_gamma_s == 20.0


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


def test_nemo_tpoint_bottom_rule_is_selectable_and_unsnapped():
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    z_ref = create_z_star_from_thicknesses(jnp.full((3,), 20.0))
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
