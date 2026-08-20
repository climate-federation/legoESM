"""Semi-implicit acoustic tridiagonal on a NON-UNIFORM vertical grid.

The SI acoustic solve is implicit in the vertical, so it must carry NO vertical
sound-wave CFL limit. It did, on stretched grids only: the coupling coefficient
was built as ``dt^2 c_s^2 / dz_half^2`` when the two spacings in that product
are physically distinct — the implicit pressure gradient at an interface uses
the CENTRE spacing ``dz_half[k]``, the continuity divergence feeding back into
it uses the LAYER thickness of the level between the interfaces (``dz[k]``
above, ``dz[k+1]`` below).

``z_full`` is the midpoint of ``z_half``, so ``0.5*(dz[k]+dz[k+1]) == dz_half[k]``
identically, and on a UNIFORM grid ``dz_half == dz`` as well — all three
coincide and the bug is invisible. On the gSAM RCEMIP1 grd the coefficients
were 10-12 % off, leaving that fraction of the acoustic term effectively
explicit and reimposing ``dt <= ~0.04*dz_min`` (non-finite at step 1 at
dt=6 s / dt=12 s; gSAM runs the same grid at dt=12 s).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    precompute_si_tridiag_bands,
)
from legoesm.grids.vertical import create_height_coordinate_from_z_half


def _stretched_hc(n=40, H=33_000.0, dz_sfc=75.0):
    """Geometrically stretched z_half, top-to-bottom (the failing shape)."""
    r = (H / dz_sfc) ** (1.0 / (n - 1))
    dz = dz_sfc * r ** np.arange(n)
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    zi = zi * (H / zi[-1])                      # normalise so the top is H
    return create_height_coordinate_from_z_half(jnp.asarray(zi[::-1].copy()))


def _uniform_hc(n=40, H=33_000.0):
    return create_height_coordinate_from_z_half(
        jnp.asarray(np.linspace(H, 0.0, n + 1)))


def _expected_bands(hc, dt_s, J=1.0):
    """The coefficients derived from the discretisation, independently of the
    implementation: interface k couples to k-1 through layer dz[k] and to k+1
    through layer dz[k+1], both scaled by the centre spacing dz_half[k]."""
    c_p, R_d, c_v = constants.c_pd, constants.R_d, constants.c_vd
    dz = np.asarray(hc.dz)
    dz_half = np.asarray(hc.dz_half)
    T_ref = np.asarray(hc.theta_ref) * np.asarray(hc.exner_ref)
    cs2 = (c_p / c_v) * R_d * T_ref
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])
    a_up = dt_s ** 2 * cs2_half / (dz_half * dz[:-1] * J ** 2)
    a_dn = dt_s ** 2 * cs2_half / (dz_half * dz[1:] * J ** 2)
    a_tri = np.concatenate([[0.0], -a_up[1:]])
    b_tri = 1.0 + a_up + a_dn
    c_tri = np.concatenate([-a_dn[:-1], [0.0]])
    return a_tri, b_tri, c_tri


def test_bands_use_layer_thickness_not_squared_centre_spacing():
    hc = _stretched_hc()
    dt_s = 2.0
    J = jnp.ones((1,))
    a, b, c = precompute_si_tridiag_bands(hc, J, dt_s, constants.g)
    ea, eb, ec = _expected_bands(hc, dt_s)
    np.testing.assert_allclose(np.asarray(a)[0], ea, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(b)[0], eb, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(c)[0], ec, rtol=1e-12)


def test_old_squared_form_would_differ_on_a_stretched_grid():
    """Guard the guard: confirm this grid actually discriminates the two forms
    (on a uniform grid they coincide, which is why the bug survived)."""
    hc = _stretched_hc()
    dz = np.asarray(hc.dz)
    dz_half = np.asarray(hc.dz_half)
    ratio_above = dz_half ** 2 / (dz_half * dz[:-1])
    ratio_below = dz_half ** 2 / (dz_half * dz[1:])
    assert abs(ratio_above - 1.0).max() > 0.05, "grid too mild to discriminate"
    assert abs(ratio_below - 1.0).max() > 0.05


def test_uniform_grid_is_where_the_two_forms_coincide():
    hc = _uniform_hc()
    dz = np.asarray(hc.dz)
    dz_half = np.asarray(hc.dz_half)
    np.testing.assert_allclose(dz_half, dz[:-1], rtol=1e-12)
    np.testing.assert_allclose(dz_half, dz[1:], rtol=1e-12)


def test_diagonal_keeps_both_couplings_at_the_end_interfaces():
    """The rigid lid zeroes the neighbouring w, which removes the OFF-diagonal
    entry but not the layer's own compression term — the previous
    ``alpha_interior`` padding dropped it at the first/last interior interface.
    """
    hc = _uniform_hc(n=12)
    a, b, c = precompute_si_tridiag_bands(hc, jnp.ones((1,)), 2.0, constants.g)
    a, b, c = np.asarray(a)[0], np.asarray(b)[0], np.asarray(c)[0]
    _, eb, _ = _expected_bands(hc, 2.0)
    # The end diagonals must carry BOTH couplings (the old form carried one),
    # while the corresponding off-diagonal entries are the ones that drop out.
    np.testing.assert_allclose(b[0], eb[0], rtol=1e-12)
    np.testing.assert_allclose(b[-1], eb[-1], rtol=1e-12)
    assert a[0] == 0.0 and c[-1] == 0.0
    # ... and "both" is measurably more than "one": b-1 is ~2x a single term.
    single = eb[0] - 1.0 - (eb[0] - 1.0) / 2.0
    assert (b[0] - 1.0) > 1.9 * single


def test_diagonally_dominant_on_the_stretched_grid_at_large_dt():
    """A well-posed implicit operator stays diagonally dominant no matter how
    large dt is — that is what makes the vertical solve unconditionally stable.
    """
    hc = _stretched_hc()
    for dt_s in (2.0, 12.0, 120.0):
        a, b, c = precompute_si_tridiag_bands(hc, jnp.ones((1,)), dt_s,
                                              constants.g)
        a, b, c = np.asarray(a)[0], np.asarray(b)[0], np.asarray(c)[0]
        assert np.all(b >= np.abs(a) + np.abs(c)), (
            f"not diagonally dominant at dt={dt_s}")
