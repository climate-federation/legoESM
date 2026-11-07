"""NEMO's frozen barotropic vorticity thickness, and who it moves.

``dyn_vor_init`` builds ``e3f_0vor`` for every curl-point scheme in one
``SELECT CASE`` arm (``VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890``)
and ``dyn_cor_2D_init`` divides ``ff_f`` by it in every branch -- EEN at
``dynspg_ts.f90:960``, ENE/MIX at ``dynspg_ts.f90:1016``, ENS at
``dynspg_ts.f90:1046`` of the same build.  These tests pin the arithmetic of
that array and the certified consequence: GYRE's ENE coefficients do not move
and both VORTEX cards' EEN coefficients do.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


def _e3f(tmask, e3t, fill, nn_e3f_typ=0):
    from legoesm.ocean.vertical import nemo_e3f_0vor_from_tmask

    return np.asarray(nemo_e3f_0vor_from_tmask(
        jnp.asarray(e3t, dtype=jnp.float64),
        jnp.asarray(tmask, dtype=jnp.float64),
        jnp.asarray(fill, dtype=jnp.float64),
        nn_e3f_typ=nn_e3f_typ))


def _stencil():
    """A 3x3 single-level box whose vertex [0, 0] has one dry corner."""
    tmask = np.ones((3, 3, 1))
    tmask[1, 1, 0] = 0.0           # NEMO (ji+1, jj+1) seen from vertex (0, 0)
    e3t = np.full((3, 3, 1), 8.0)
    return tmask, e3t, np.full((3, 3, 1), 8.0)


def test_nn_e3f_typ_zero_divides_by_four_not_by_the_wet_count():
    """dynvor.f90:897 -- masked sum over FOUR; :905 is the other branch."""
    tmask, e3t, fill = _stencil()
    assert _e3f(tmask, e3t, fill, nn_e3f_typ=0)[0, 0, 0] == 6.0
    assert _e3f(tmask, e3t, fill, nn_e3f_typ=1)[0, 0, 0] == 8.0


def test_a_fully_dry_vertex_is_restored_to_the_cards_own_operand():
    """dynvor.f90:918-920 -- the "insure e3f_0vor /= 0" sweep."""
    tmask = np.zeros((3, 3, 1))
    e3t = np.full((3, 3, 1), 8.0)
    fill = np.full((3, 3, 1), 3.5)
    assert np.array_equal(_e3f(tmask, e3t, fill), np.full((3, 3, 1), 3.5))


def test_a_fully_wet_vertex_is_the_plain_thickness():
    tmask = np.ones((3, 3, 1))
    e3t = np.full((3, 3, 1), 8.0)
    assert _e3f(tmask, e3t, e3t)[0, 0, 0] == 8.0


def test_an_unknown_nn_e3f_typ_raises():
    tmask, e3t, fill = _stencil()
    with pytest.raises(ValueError):
        _e3f(tmask, e3t, fill, nn_e3f_typ=2)


def test_the_builder_refuses_without_the_mesh_operands():
    """No silent fallback to the plain thickness."""
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    z_coord = build_nemo_testcase_card("VORTEX_VEC-zco").recipe.z_coord
    stripped = z_coord._replace(nemo_e3t_0=None)
    raw = z_coord.nemo_een_barotropic
    eta = jnp.zeros(np.asarray(raw.ff_f).shape, dtype=get_policy().control)
    with pytest.raises(ValueError, match="e3f_0vor"):
        bt._nemo_literal_een_coefficients(
            eta, stripped, get_policy().control, scheme="een")


def test_the_certified_cards_move_exactly_where_nemos_branches_say():
    """GYRE is ENE and does not move; both VORTEX cards are EEN and do.

    This is the certified-registry pin for round 198's landing: the SAME
    probe asserts the statement is not vacuous (the old plain-thickness
    coefficients differ on the cards that NEMO's EEN branch couples across
    rows) and that it is inert where NEMO's ENE branch pairs each vertex
    with a mask from its own row (dynspg_ts.f90:1016-1019).
    """
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
            / "ocean_fidelity" / "testcases"
            / "nemo_testcase_l1_vortex_round198_e3f_0vor_scope.py")
    spec = importlib.util.spec_from_file_location("_r198_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cards = module.run(allow_dirty=True)["cards"]

    gyre = cards["GYRE-zco"]
    assert gyre["branch"] == "ene"
    assert gyre["n_vertices_e3f_0vor_differs"] == 3000
    assert gyre["coefficients_bit_identical"] is True
    assert gyre["max_abs_coefficient_change"] == 0.0

    for name in ("VORTEX-zco", "VORTEX_VEC-zco"):
        card = cards[name]
        assert card["branch"] == "een"
        assert card["n_vertices_e3f_0vor_differs"] == 2440
        assert card["coefficients_bit_identical"] is False
        assert card["max_abs_coefficient_change"] > 1e-5
