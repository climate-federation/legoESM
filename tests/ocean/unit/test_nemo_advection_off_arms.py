"""NEMO ln_dynadv_OFF / ln_traadv_OFF arms of the shared dispatchers.

dynadv.f90:185 (np_LIN_dyn) and traadv.f90:444 (np_NO_adv): the dispatchers
add nothing.  Each "unchanged" assertion is paired with a plant that flips
the selection back and must move the same output.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _compute_advection_flux_div,
    _nemo_ws_rk3_tracer_pair_step,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_horizontal_momentum_advection_flux_form,
    _bc_vertical_momentum_advection,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_tsunami_zco_card,
    build_vortex_smt_zps_card,
)


@pytest.fixture(scope="module")
def card():
    base = build_vortex_smt_zps_card()
    config = base.recipe.model_config._replace(
        momentum_flux_scheme="none",
        vertical_momentum_scheme="none",
        tracer_advection="none",
        adaptive_implicit_vertadv=False,
    )
    return base._replace(recipe=base.recipe._replace(model_config=config))


@pytest.fixture(scope="module")
def tsunami_card():
    return build_tsunami_zco_card()


@pytest.fixture(scope="module")
def vortex_off_card():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_vortex_smt_zps_card,
    )
    base = build_vortex_smt_zps_card()
    config = base.recipe.model_config._replace(
        momentum_flux_scheme="none",
        vertical_momentum_scheme="none",
        tracer_advection="none",
        adaptive_implicit_vertadv=False,
    )
    return base._replace(recipe=base.recipe._replace(model_config=config))


def _field(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def test_explicit_card_fixture_config_selects_every_off_arm(card):
    m = card.recipe.model_config
    assert (m.momentum_advection, m.momentum_flux_scheme,
            m.vertical_momentum_scheme, m.tracer_advection) == (
        "flux_form", "none", "none", "none")


def test_explicit_card_config_selects_every_off_arm(vortex_off_card):
    m = vortex_off_card.recipe.model_config
    assert (m.momentum_advection, m.momentum_flux_scheme,
            m.vertical_momentum_scheme, m.tracer_advection) == (
        "flux_form", "none", "none", "none")


def test_horizontal_none_leaves_rhs_unchanged_and_up3_plant_moves_it(card):
    g, cfg = card.recipe.grid, card.recipe.model_config
    nj, ni = g.f_T.shape
    u, v = _field((nj, ni + 1, 1), 0), _field((nj + 1, ni, 1), 1)
    du, dv = _field(u.shape, 2), _field(v.shape, 3)
    hu, hv = jnp.full(u.shape, 100.0), jnp.full(v.shape, 100.0)
    args = (du, dv, u, v, hu, hv, jnp.ones(u.shape), jnp.ones(v.shape),
            jnp.ones((nj, ni)), g)
    out = _bc_horizontal_momentum_advection_flux_form(*args, cfg)
    assert np.array_equal(out[0], du) and np.array_equal(out[1], dv)
    assert not np.any(out[2]) and not np.any(out[3])
    plant = _bc_horizontal_momentum_advection_flux_form(
        *args, cfg._replace(momentum_flux_scheme="nemo_up3"))
    assert not np.array_equal(plant[0], du)


def test_vertical_none_leaves_rhs_unchanged_and_up3_plant_moves_it(card):
    g, cfg = card.recipe.grid, card.recipe.model_config
    nj, ni = g.f_T.shape
    nlev = 3
    u, v = _field((nj, ni + 1, nlev), 4), _field((nj + 1, ni, nlev), 5)
    du, dv = _field(u.shape, 6), _field(v.shape, 7)
    w = _field((nj, ni, nlev + 1), 8).at[..., 0].set(0.0).at[..., -1].set(0.0)
    args = (du, dv, u, v, w, jnp.full(u.shape, 10.0), jnp.full(v.shape, 10.0),
            jnp.ones(u.shape), jnp.ones(v.shape), g, "flux_form", 5)
    out = _bc_vertical_momentum_advection(*args, cfg, u_full=u, v_full=v)
    assert np.array_equal(out[0], du) and np.array_equal(out[1], dv)
    plant = _bc_vertical_momentum_advection(
        *args, cfg._replace(vertical_momentum_scheme="nemo_up3"),
        u_full=u, v_full=v)
    assert not np.array_equal(plant[0], du)


def test_nemo_zad_written_association_changes_compiled_operator_bits(
        vortex_off_card):
    """Run both config branches through the compiled dynzad operator."""
    g = vortex_off_card.recipe.grid
    cfg = vortex_off_card.recipe.model_config._replace(
        vertical_momentum_scheme="nemo_advective")
    nj, ni = g.f_T.shape
    nlev = 3
    rng = np.random.default_rng(1226)

    def values(shape):
        return jnp.asarray(
            rng.standard_normal(shape) * np.exp(rng.uniform(-20, 20, shape)))

    u = values((nj, ni + 1, nlev))
    v = values((nj + 1, ni, nlev))
    du, dv = jnp.zeros_like(u), jnp.zeros_like(v)
    w = values((nj, ni, nlev + 1)).at[..., 0].set(0.0).at[..., -1].set(0.0)
    hu, hv = jnp.full(u.shape, 10.0), jnp.full(v.shape, 10.0)
    umask, vmask = jnp.ones(u.shape), jnp.ones(v.shape)

    def selected(flag):
        selected_cfg = cfg._replace(nemo_zad_written_association=flag)
        call = jax.jit(lambda du, dv, u, v, w, hu, hv, um, vm:
            _bc_vertical_momentum_advection(
                du, dv, u, v, w, hu, hv, um, vm, g, "flux_form", 5,
                selected_cfg, u_full=u, v_full=v))
        return tuple(np.asarray(item) for item in call(
            du, dv, u, v, w, hu, hv, umask, vmask))

    on, off = selected(True), selected(False)
    assert np.count_nonzero(
        on[0].view(np.uint64) != off[0].view(np.uint64)) == 1478
    assert on[0].view(np.uint64)[0, 6, 1] == np.uint64(13881778523158873640)
    assert off[0].view(np.uint64)[0, 6, 1] == np.uint64(13881778523158873639)


@pytest.mark.parametrize("flux, vert", [("none", "upwind_perturbation"),
                                        ("centered", "none")])
def test_half_off_selection_is_refused(card, flux, vert):
    cfg = card.recipe.model_config._replace(
        momentum_flux_scheme=flux, vertical_momentum_scheme=vert)
    with pytest.raises(ValueError, match="one selection"):
        LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def test_tracer_none_dispatch_is_zero(vortex_off_card):
    g = vortex_off_card.recipe.grid
    nj, ni = g.f_T.shape
    tr = _field((nj, ni, 1), 9)
    z = jnp.zeros((nj, ni, 1))
    dh, dv = _compute_advection_flux_div(
        tr, "none", jnp.zeros((nj, ni + 1, 1)), jnp.zeros((nj + 1, ni, 1)),
        jnp.zeros((nj, ni, 2)), z + 100.0, jnp.full((nj, ni + 1, 1), 100.0),
        jnp.full((nj + 1, ni, 1), 100.0), g, 1000.0)
    assert not np.any(dh) and not np.any(dv)


def test_card_fixture_tracer_none_dispatch_is_zero(card):
    g = card.recipe.grid
    nj, ni = g.f_T.shape
    tr = _field((nj, ni, 1), 9)
    z = jnp.zeros((nj, ni, 1))
    dh, dv = _compute_advection_flux_div(
        tr, "none", jnp.zeros((nj, ni + 1, 1)), jnp.zeros((nj + 1, ni, 1)),
        jnp.zeros((nj, ni, 2)), z + 100.0, jnp.full((nj, ni + 1, 1), 100.0),
        jnp.full((nj + 1, ni, 1), 100.0), g, 1000.0)
    assert not np.any(dh) and not np.any(dv)


def test_tsunami_card_selects_every_off_arm(tsunami_card):
    m = tsunami_card.recipe.model_config
    assert (m.momentum_advection, m.momentum_flux_scheme,
            m.vertical_momentum_scheme, m.tracer_advection) == (
        "flux_form", "none", "none", "none")


def test_tracer_none_dispatch_is_zero_and_upwind_plant_is_not(tsunami_card):
    g = tsunami_card.recipe.grid
    nj, ni = g.f_T.shape
    tr = _field((nj, ni, 1), 9)
    args = (0.1 * _field((nj, ni + 1, 1), 17), 0.1 * _field((nj + 1, ni, 1), 18),
            jnp.zeros((nj, ni, 2)), jnp.full((nj, ni, 1), 100.0),
            jnp.full((nj, ni + 1, 1), 100.0), jnp.full((nj + 1, ni, 1), 100.0),
            g, 1000.0)
    dh, dv = _compute_advection_flux_div(tr, "none", *args)
    assert not np.any(dh) and not np.any(dv)
    assert np.any(_compute_advection_flux_div(tr, "upwind", *args)[0])


@pytest.mark.parametrize("field, value", [
    ("eos_depth", "insitu"), ("tracer_advection", "fct2"),
    ("momentum_flux_scheme", "nemo_up3")])
def test_tsunami_validator_refuses_reverting_a_selection(
        tsunami_card, field, value):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        validate_nemo_testcase_card)
    validate_nemo_testcase_card(tsunami_card)
    cfg = tsunami_card.recipe.model_config._replace(**{field: value})
    bad = tsunami_card._replace(
        recipe=tsunami_card.recipe._replace(model_config=cfg))
    with pytest.raises(ValueError, match="TSUNAMI-zco"):
        validate_nemo_testcase_card(bad)


def test_store_salt_flux_is_refused_with_tracer_advection_none(tsunami_card):
    cfg = tsunami_card.recipe.model_config._replace(
        store_salt_flux=True, tracer_time_integrator="euler",
        momentum_time_integrator="euler")
    with pytest.raises(ValueError, match="tracer_advection='none'"):
        LatLonCGridOceanModel(
            tsunami_card.recipe.grid, tsunami_card.recipe.z_coord, cfg)


def test_tracer_none_stage1_is_the_qco_ratio_and_fct2_plant_moves_it(card):
    # stprk3_stg.f90:503-505 with ts(Krhs) = 0 (:467)
    g = card.recipe.grid
    nj, ni = g.f_T.shape
    shape = (nj, ni, 1)
    a, b = 20.0 + _field(shape, 10), 30.0 + _field(shape, 11)
    r3b, r3m, r3a = (1e-3 * _field((nj, ni), s) for s in (12, 13, 14))
    qb, qm, qa = 1.0 + r3b, 1.0 + r3m, 1.0 + r3a
    mfu = 0.1 * _field((nj, ni + 1, 1), 15)
    mfv = 0.1 * _field((nj + 1, ni, 1), 16)
    w = jnp.zeros((nj, ni, 2))
    h = jnp.full(shape, 100.0)

    def step(scheme):
        return _nemo_ws_rk3_tracer_pair_step(
            a, b, scheme, mfu, mfv, w, h, h,
            jnp.full((nj, ni + 1, 1), 100.0), jnp.full((nj + 1, ni, 1), 100.0),
            g, 1000.0, jnp.ones(shape),
            stage_qco_weights=((qb, qm, qa),) * 3, stop_after_stage=1)

    a1, b1 = step("none")
    want_a = (qb[..., None] * a + (1000.0 / 3.0) * qm[..., None] * 0.0) / qa[..., None]
    assert np.array_equal(a1, want_a)
    assert np.array_equal(a1, qb[..., None] * a / qa[..., None])
    assert np.array_equal(b1, qb[..., None] * b / qa[..., None])
    assert not np.array_equal(step("fct2")[0], a1)
