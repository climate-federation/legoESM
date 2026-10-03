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
    """``MIN(zht, pdepw(ik+1)) - pdepw(ik)``, OVERFLOW:222."""
    z = _uniform_ref()
    H = jnp.asarray([[4560.0, 4763.0722]])
    c = create_partial_cell_coordinate(
        z, H, bottom_index_rule="nemo_zps_e3min", min_partial_thickness=50.0)
    h = np.asarray(c.h_partial)[0]
    assert h[0, 9] == min(4560.0, 5000.0) - 4500.0
    assert h[1, 9] == min(4763.0722, 5000.0) - 4500.0


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
    zht, k_bot, e3t, e3u, e3v, e3f = vortex_smt_partial_cell_geometry(
        source, res)
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
