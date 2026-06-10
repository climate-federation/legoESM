"""Unit tests for the CLUBB staggered-grid operators (``clubb_grid.py``).

Analytic checks (the strongest available for linear interpolation/derivative
operators): linear fields are reproduced/differentiated exactly, round-trip
smoothers preserve linear fields, and shapes/pytree behaviour are correct.

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_grid import (  # noqa: E402
    CLUBBGrid,
    ddzm,
    ddzt,
    make_clubb_grid,
    zm2zt,
    zm2zt2zm,
    zt2zm,
    zt2zm2zt,
)


def _build_grid(ngrdcol=3, nzm=12, stretched=True):
    """Ascending staggered grid: zt are midpoints of consecutive zm levels."""
    if stretched:
        # Geometrically stretched ascending momentum levels.
        idx = np.arange(nzm, dtype=np.float64)
        zm_1d = np.cumsum(np.concatenate([[0.0], 20.0 * 1.15 ** idx[:-1]]))
    else:
        zm_1d = np.linspace(0.0, 3000.0, nzm)
    zm = jnp.asarray(np.tile(zm_1d, (ngrdcol, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), zm, zt


def _linear(z, a=3.0, b=-0.0072):
    return a + b * z


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def test_make_clubb_grid_shapes_and_pytree():
    gr, zm, zt = _build_grid()
    assert isinstance(gr, CLUBBGrid)
    assert gr.zm.shape == zm.shape
    assert gr.zt.shape == zt.shape
    assert gr.invrs_dzt.shape == zt.shape
    assert gr.invrs_dzm.shape == zm.shape
    # NamedTuple flattens as a pytree.
    leaves = jax.tree_util.tree_leaves(gr)
    assert len(leaves) == 4
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves)


def test_invrs_spacings_positive_and_consistent():
    gr, zm, zt = _build_grid()
    # invrs_dzt[k] = 1/(zm[k+1]-zm[k]).
    np.testing.assert_allclose(
        np.asarray(gr.invrs_dzt), np.asarray(1.0 / (zm[:, 1:] - zm[:, :-1]))
    )
    assert jnp.all(gr.invrs_dzt > 0)
    assert jnp.all(gr.invrs_dzm > 0)


def test_invrs_dzm_lower_boundary_matches_reference_nonmidpoint():
    """Lower boundary dzm = 2*(zt[0]-zm[0]), NOT a copy of interior spacing.

    Reference: grid_class.F90 setup_grid_heights / _calc_grid_spacings. Use a
    grid where zt[0] is deliberately NOT the midpoint of zm[0],zm[1] so the two
    formulas disagree.
    """
    zm_1d = np.array([0.0, 100.0, 250.0, 450.0, 700.0], dtype=np.float64)
    zt_1d = np.array([30.0, 175.0, 350.0, 575.0], dtype=np.float64)  # zt[0] != 50
    zm = jnp.asarray(np.tile(zm_1d, (2, 1)))
    zt = jnp.asarray(np.tile(zt_1d, (2, 1)))
    gr = make_clubb_grid(zm, zt)
    # interior dzm[1:-1] = zt[1:]-zt[:-1]; lower dzm[0]=2*(zt[0]-zm[0]); upper copy.
    dzm_int = zt_1d[1:] - zt_1d[:-1]
    dzm_expected = np.concatenate(
        [[2.0 * (zt_1d[0] - zm_1d[0])], dzm_int, dzm_int[-1:]]
    )
    np.testing.assert_allclose(
        np.asarray(gr.invrs_dzm[0]), 1.0 / dzm_expected, rtol=1e-12
    )
    # Confirm it is NOT the (wrong) interior-copy value at the lower boundary.
    assert not np.isclose(np.asarray(gr.invrs_dzm[0, 0]), 1.0 / dzm_int[0])


def test_make_clubb_grid_safe_invrs_on_zero_spacing():
    """Duplicate adjacent levels -> 0 inverse spacing (reference guard), not inf."""
    zm_1d = np.array([0.0, 100.0, 100.0, 300.0], dtype=np.float64)  # dup zm[1]==zm[2]
    zt_1d = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    zm = jnp.asarray(zm_1d[None, :])
    zt = jnp.asarray(zt_1d[None, :])
    gr = make_clubb_grid(zm, zt)
    assert jnp.all(jnp.isfinite(gr.invrs_dzt))
    assert jnp.all(jnp.isfinite(gr.invrs_dzm))
    # The duplicated zm pair gives a zero-spacing -> 0 reciprocal.
    assert float(gr.invrs_dzt[0, 1]) == 0.0


@pytest.mark.parametrize(
    "zm_shape, zt_shape",
    [
        ((2, 5), (2, 5)),   # nzt != nzm - 1
        ((2, 3), (2, 1)),   # nzt < 2
        ((2, 4), (3, 3)),   # ngrdcol mismatch
    ],
)
def test_make_clubb_grid_rejects_bad_shapes(zm_shape, zt_shape):
    zm = jnp.zeros(zm_shape)
    zt = jnp.zeros(zt_shape)
    with pytest.raises(ValueError):
        make_clubb_grid(zm, zt)


def test_make_clubb_grid_rejects_non_2d():
    with pytest.raises(ValueError):
        make_clubb_grid(jnp.zeros(5), jnp.zeros(4))


# ---------------------------------------------------------------------------
# Interpolation: linear fields are exact
# ---------------------------------------------------------------------------

def test_zm2zt_linear_exact():
    gr, zm, zt = _build_grid()
    f_zm = _linear(zm)
    out = zm2zt(f_zm, gr)
    assert out.shape == zt.shape
    np.testing.assert_allclose(np.asarray(out), np.asarray(_linear(zt)), rtol=1e-12)


def test_zt2zm_linear_exact_interior_and_top():
    gr, zm, zt = _build_grid()
    f_zt = _linear(zt)
    out = zt2zm(f_zt, gr)
    assert out.shape == zm.shape
    # Interior + top are exact for a linear field (top uses linear extension).
    np.testing.assert_allclose(
        np.asarray(out[:, 1:]), np.asarray(_linear(zm[:, 1:])), rtol=1e-12
    )
    # Bottom boundary copies azt[0] by construction (documented), not exact.
    np.testing.assert_allclose(np.asarray(out[:, 0]), np.asarray(f_zt[:, 0]))


def test_zt2zm_zm_min_clamp():
    gr, _, zt = _build_grid()
    f_zt = _linear(zt)
    out = zt2zm(f_zt, gr, zm_min=0.0)
    assert jnp.all(out >= 0.0)


# ---------------------------------------------------------------------------
# Derivatives: constant slope of a linear field
# ---------------------------------------------------------------------------

def test_ddzm_linear_constant_slope():
    gr, zm, zt = _build_grid()
    b = -0.0072
    f_zm = _linear(zm, b=b)
    out = ddzm(f_zm, gr)
    assert out.shape == zt.shape
    np.testing.assert_allclose(np.asarray(out), b, rtol=1e-10)


def test_ddzt_linear_constant_slope():
    gr, zm, zt = _build_grid()
    b = 0.0041
    f_zt = _linear(zt, b=b)
    out = ddzt(f_zt, gr)
    assert out.shape == zm.shape
    np.testing.assert_allclose(np.asarray(out), b, rtol=1e-10)


# ---------------------------------------------------------------------------
# Round-trip smoothers preserve linear fields
# ---------------------------------------------------------------------------

def test_zm2zt2zm_preserves_linear_interior():
    gr, zm, _ = _build_grid()
    f_zm = _linear(zm)
    out = zm2zt2zm(f_zm, gr)
    assert out.shape == zm.shape
    np.testing.assert_allclose(
        np.asarray(out[:, 1:]), np.asarray(_linear(zm[:, 1:])), rtol=1e-10
    )


def test_zt2zm2zt_preserves_linear_interior():
    gr, _, zt = _build_grid()
    f_zt = _linear(zt)
    out = zt2zm2zt(f_zt, gr)
    assert out.shape == zt.shape
    # zt2zm's lower boundary sets azm[0]=azt[0] (documented non-linear-exact
    # BC), so the round trip is linear-exact only in the interior (k>=1).
    np.testing.assert_allclose(
        np.asarray(out[:, 1:]), np.asarray(_linear(zt[:, 1:])), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# JIT + grad smoke (operators must stay autodiff/JIT clean)
# ---------------------------------------------------------------------------

def test_operators_jit_and_grad_clean():
    gr, zm, zt = _build_grid(stretched=False)
    f_zt = _linear(zt)

    @jax.jit
    def fwd(field):
        return jnp.sum(zt2zm(field, gr) ** 2) + jnp.sum(ddzt(field, gr) ** 2)

    val = fwd(f_zt)
    assert jnp.isfinite(val)
    g = jax.grad(fwd)(f_zt)
    assert g.shape == f_zt.shape
    assert jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
