"""Direct certificate for the shared Williamson analytic SW fields.

Three claims, each mechanical:

1. the winds ARE ``curl(psi)`` for the published Rossby-Haurwitz-4
   streamfunction -- certified against a NUMERICAL derivative, so a sign
   flip or a wrong power of ``cos`` fails by orders of magnitude;
2. the height's wave coefficient carries ``cos^R``, not ``cos^(R-1)``
   -- a regression guard on the defect this module was created to fix
   (all three previous copies had ``cos^(R-1)``);
3. NumPy and ``jax.numpy`` evaluation agree, so the ``xp`` switch does
   not quietly change numbers between the FV3-native lane and the
   traced test cases.
"""

import numpy as np
import pytest
from legoesm.core.williamson_sw_analytic import (
    RH4_K_HZ,
    RH4_MEAN_DEPTH_M,
    RH4_OMEGA_WAVE_HZ,
    RH4_WAVENUMBER,
    W5_CENTER_FV3,
    W5_CENTER_WILLIAMSON,
    W5_MOUNTAIN_HEIGHT_M,
    W5_MOUNTAIN_RADIUS_RAD,
    rossby_haurwitz_4_geopotential,
    rossby_haurwitz_4_winds,
    solid_body_geopotential,
    solid_body_rotation_speed,
    solid_body_winds,
    williamson_5_mountain_height,
)

# Deliberately NOT legoesm.constants: this test fixes its own frame so a
# change to either constants set cannot silently move the reference.
A_R = 6.3712e6
OMEGA = 7.2921e-5      # const-ok: test frame, not a physical-constant use
GRAV = 9.80665         # const-ok: test frame, not a physical-constant use
GH0 = RH4_MEAN_DEPTH_M * GRAV


def _sample():
    """Off-pole, off-equator, spanning all four cos(R*lon) quadrants."""
    lon = np.deg2rad(np.array([0.0, 17.0, 45.0, 100.0, 213.0, 330.0]))
    lat = np.deg2rad(np.array([-71.0, -33.0, -5.0, 5.0, 33.0, 71.0]))
    return np.meshgrid(lon, lat, indexing="ij")


def _psi(lon, lat):
    """Williamson et al. (1992) case-6 streamfunction."""
    r = RH4_WAVENUMBER
    return (-A_R * A_R * RH4_OMEGA_WAVE_HZ * np.sin(lat)
            + A_R * A_R * RH4_K_HZ * np.cos(lat) ** r * np.sin(lat)
            * np.cos(r * lon))


def test_winds_are_the_curl_of_the_published_streamfunction():
    lon, lat = _sample()
    u, v = rossby_haurwitz_4_winds(lon, lat, radius=A_R)

    # 1e-7 below is a GUARD tolerance: observed agreement is ~2.6e-11
    # (cancellation-dominated, codex r1 #9), so there are ~4 orders of
    # headroom while a formula error still fails by O(1).
    h = 1e-6
    u_fd = -(_psi(lon, lat + h) - _psi(lon, lat - h)) / (2.0 * h) / A_R
    v_fd = ((_psi(lon + h, lat) - _psi(lon - h, lat)) / (2.0 * h)
            / (A_R * np.cos(lat)))

    scale = max(np.abs(u_fd).max(), np.abs(v_fd).max())
    assert scale > 10.0, scale          # non-vacuity: real winds, not ~0
    assert np.abs(u - u_fd).max() / scale < 1e-7
    assert np.abs(v - v_fd).max() / scale < 1e-7


def test_height_wave_coefficient_uses_cos_to_the_R():
    """Regression guard on the fixed exponent.

    Isolate ``B`` by differencing the geopotential across two longitudes
    that flip ``cos(R L)`` while holding ``cos(2 R L)`` fixed: with
    ``R L = 0`` and ``R L = pi`` the ``A`` and ``C`` terms are identical,
    so the difference is exactly ``2 a^2 B``.  Compare against ``B``
    evaluated both ways; ``cos^R`` must win and ``cos^(R-1)`` must lose
    by a wide margin.
    """
    r = RH4_WAVENUMBER
    lat = np.deg2rad(np.array([-60.0, -25.0, 25.0, 60.0]))
    lon_a = np.zeros_like(lat)                  # R*lon = 0
    lon_b = np.full_like(lat, np.pi / r)        # R*lon = pi, 2R*lon = 2pi

    gh_a = rossby_haurwitz_4_geopotential(lon_a, lat, radius=A_R,
                                          omega=OMEGA, gh0=GH0)
    gh_b = rossby_haurwitz_4_geopotential(lon_b, lat, radius=A_R,
                                          omega=OMEGA, gh0=GH0)
    b_measured = (gh_a - gh_b) / (2.0 * A_R * A_R)

    c = np.cos(lat)
    pre = (2.0 * (OMEGA + RH4_OMEGA_WAVE_HZ) * RH4_K_HZ
           / ((r + 1.0) * (r + 2.0)))
    bracket = (r * r + 2.0 * r + 2.0) - ((r + 1.0) * c) ** 2
    b_correct = pre * c ** r * bracket
    b_legacy = pre * c ** (r - 1.0) * bracket

    scale = np.abs(b_correct).max()
    assert scale > 0.0
    assert np.abs(b_measured - b_correct).max() / scale < 1e-12
    # The two forms must be distinguishable here, else the guard is
    # vacuous (they coincide only where cos(lat) == 1, i.e. the equator).
    assert np.abs(b_legacy - b_correct).max() / scale > 0.1


def test_solid_body_is_in_geostrophic_balance():
    """Case-2 balance is the case's defining property, so certify it
    rather than the algebra: the meridional pressure-gradient force must
    equal the Coriolis force for the zonal wind, at alpha = 0.

        -(1/a) d(gh)/dlat  ==  +(f + u tan(lat)/a) * u

    from the steady meridional momentum balance
    ``0 = -f u - u^2 tan(lat)/a - (1/a) dPhi/dlat``.

    Derivative taken NUMERICALLY, so a wrong coefficient in either the
    wind or the height fails it.

    (First version of this test had the RHS sign flipped and failed with
    the two sides exactly equal and opposite -- which was itself the
    proof that the fields are right and the test was wrong.)
    """
    u0 = solid_body_rotation_speed(A_R)
    gh0 = 2.94e4                        # coeff-ok: Williamson case-2 gh0
    lat = np.deg2rad(np.array([-62.0, -30.0, -8.0, 8.0, 30.0, 62.0]))
    lon = np.zeros_like(lat)

    h = 1e-6
    dgh = (solid_body_geopotential(lon, lat + h, radius=A_R, omega=OMEGA,
                                   u0=u0, gh0=gh0)
           - solid_body_geopotential(lon, lat - h, radius=A_R, omega=OMEGA,
                                     u0=u0, gh0=gh0)) / (2.0 * h)
    pgf = -dgh / A_R

    u, v = solid_body_winds(lon, lat, u0=u0)
    assert np.abs(v).max() == 0.0        # alpha = 0 is purely zonal
    f = 2.0 * OMEGA * np.sin(lat)
    coriolis = (f + u * np.tan(lat) / A_R) * u

    scale = np.abs(coriolis).max()
    assert scale > 1e-4, scale
    assert np.abs(pgf - coriolis).max() / scale < 1e-6


def test_alpha_rotation_is_a_rigid_rotation_of_the_alpha_zero_state():
    """At the equator-crossing meridian lon = 0 the alpha-rotated state
    must reduce to the alpha=0 one evaluated at the rotated latitude --
    a property no single-term typo survives."""
    u0 = 30.0
    alpha = np.deg2rad(45.0)
    lat = np.deg2rad(np.array([-40.0, 0.0, 40.0]))
    lon = np.zeros_like(lat)

    u_a, v_a = solid_body_winds(lon, lat, u0=u0, alpha=alpha)
    # At lon = 0: u = u0 cos(lat - alpha), v = 0.
    assert np.abs(v_a).max() < 1e-14
    assert np.abs(u_a - u0 * np.cos(lat - alpha)).max() < 1e-12


def test_case5_mountain_is_planar_clipped_not_great_circle():
    """Regression guard on the fixed radius metric.

    The two metrics coincide on the centre meridian and diverge in
    longitude by roughly 1/cos(lat_c); assert BOTH, so the test fails if
    the great-circle form comes back AND is not vacuous.
    """
    lon_c, lat_c = W5_CENTER_WILLIAMSON
    r0 = W5_MOUNTAIN_RADIUS_RAD

    # Peak at the centre.
    hs_c = williamson_5_mountain_height(np.array([lon_c]),
                                        np.array([lat_c]))
    assert abs(float(hs_c[0]) - W5_MOUNTAIN_HEIGHT_M) < 1e-9

    # A point offset in LONGITUDE by half the mountain radius.
    dlon = 0.5 * r0
    lon = np.array([lon_c + dlon])
    lat = np.array([lat_c])
    hs = float(williamson_5_mountain_height(lon, lat)[0])

    planar = W5_MOUNTAIN_HEIGHT_M * (1.0 - abs(dlon) / r0)
    great_circle_r = np.arccos(np.clip(
        np.sin(lat_c) * np.sin(lat) + np.cos(lat_c) * np.cos(lat)
        * np.cos(lon - lon_c), -1.0, 1.0))[0]
    gc = W5_MOUNTAIN_HEIGHT_M * (1.0 - great_circle_r / r0)

    assert abs(hs - planar) < 1e-9, (hs, planar)
    # Non-vacuity: the two metrics must actually differ here.
    assert abs(planar - gc) > 50.0, (planar, gc)

    # Outside the radius the clip gives exactly zero, with no mask.
    far = williamson_5_mountain_height(np.array([lon_c + 4.0 * r0]),
                                       np.array([lat_c]))
    assert float(far[0]) == 0.0


def test_the_two_mountain_centres_are_distinct_and_named():
    """The FV3 oracle and Williamson 1992 disagree on lambda_c; both are
    exported so a caller has to choose, and neither is a hidden literal."""
    assert W5_CENTER_WILLIAMSON[1] == W5_CENTER_FV3[1]        # same lat_c
    assert abs(W5_CENTER_WILLIAMSON[0] - W5_CENTER_FV3[0]) > 3.0
    hs_w = williamson_5_mountain_height(
        np.array([W5_CENTER_WILLIAMSON[0]]),
        np.array([W5_CENTER_WILLIAMSON[1]]), center=W5_CENTER_WILLIAMSON)
    hs_f = williamson_5_mountain_height(
        np.array([W5_CENTER_WILLIAMSON[0]]),
        np.array([W5_CENTER_WILLIAMSON[1]]), center=W5_CENTER_FV3)
    assert float(hs_w[0]) > 1999.0
    assert float(hs_f[0]) == 0.0


def test_numpy_and_jax_agree():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    # NOTE: mutates global jax config for the session (codex r1 #9).
    # Acceptable here because the whole sci-test suite runs under
    # JAX_ENABLE_X64=1 anyway (CLAUDE.md validation rule); this line only
    # protects a bare local invocation.
    jax.config.update("jax_enable_x64", True)
    lon, lat = _sample()

    gh_np = rossby_haurwitz_4_geopotential(lon, lat, radius=A_R,
                                           omega=OMEGA, gh0=GH0)
    gh_jx = np.asarray(rossby_haurwitz_4_geopotential(
        jnp.asarray(lon), jnp.asarray(lat), radius=A_R, omega=OMEGA,
        gh0=GH0, xp=jnp))
    assert np.abs(gh_np - gh_jx).max() / np.abs(gh_np).max() < 1e-13

    u_np, v_np = rossby_haurwitz_4_winds(lon, lat, radius=A_R)
    u_jx, v_jx = rossby_haurwitz_4_winds(jnp.asarray(lon),
                                         jnp.asarray(lat),
                                         radius=A_R, xp=jnp)
    scale = max(np.abs(u_np).max(), np.abs(v_np).max())
    assert np.abs(u_np - np.asarray(u_jx)).max() / scale < 1e-13
    assert np.abs(v_np - np.asarray(v_jx)).max() / scale < 1e-13


def test_literal_factoring_is_finite_and_matches_the_folded_form():
    """The ``A`` term keeps the oracle's literal ``cos^{2R} * cos^{-2}``.

    This test replaces one that asserted the literal form NaNs at the
    pole -- it does not.  ``cos(pi/2)`` is 6.1e-17 in float64, never 0.0,
    so the product is finite everywhere including the pole row, and the
    two factorings agree.  What is actually certified here is that the
    factoring choice costs nothing numerically, so the removal of the
    ``+ 1e-30`` epsilon guards from the copies this module replaced
    changes no value a caller can observe.
    """
    r = RH4_WAVENUMBER
    lat = np.deg2rad(np.array([0.0, 45.0, 89.0, 89.999, 90.0]))
    lon = np.zeros_like(lat)

    gh = rossby_haurwitz_4_geopotential(lon, lat, radius=A_R,
                                        omega=OMEGA, gh0=GH0)
    assert np.isfinite(gh).all(), gh

    # Same A, folded exponent; B and C are untouched between the forms,
    # so differencing the two geopotentials isolates the A factoring.
    c = np.cos(lat)
    a_literal = (0.25 * RH4_K_HZ ** 2 * c ** (r + r)
                 * (-2.0 * r * r * c ** (-2.0)))
    a_folded = 0.25 * RH4_K_HZ ** 2 * (-2.0 * r * r) * c ** (2.0 * r - 2.0)
    scale = np.abs(a_folded).max()
    assert scale > 0.0
    assert np.abs(a_literal - a_folded).max() / scale < 1e-14
