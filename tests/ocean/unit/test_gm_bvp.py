"""Ferrari et al. (2010) GM streamfunction boundary-value problem.

The closure is only worth having if it reduces to the thing it replaces where
that thing is valid, and differs from it where it is not. So the anchor test
is the STRONG-STRATIFICATION LIMIT: with the elliptic term negligible the
solution must collapse to the diagnostic ``Gamma = kappa * S``. A wrong sign
on the right-hand side, a wrong factor of g/rho_0, or a transposed slope all
break that while leaving the solve looking healthy.

The rest cover what the limit cannot: the boundary rows, the assembly itself
(checked against a dense matrix solve, which shares no code with the Thomas
sweep), the zero-net-transport property the boundary conditions buy, dry
columns, and jit/grad/vmap.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.gm_bvp import (  # noqa: E402
    GMBVPConfig,
    _assemble,
    baroclinic_wave_speed,
    bolus_velocity,
    solve_gm_streamfunction,
)


def _column(nlev=40, H=4000.0, N2=1e-4, kappa=1000.0, sigma_x=-1e-6,
            sigma_y=0.0):
    dz = np.full(nlev, H / nlev)
    N2_i = np.full(nlev + 1, N2)
    kap = np.full(nlev + 1, kappa)
    sx = np.full(nlev, sigma_x)
    sy = np.full(nlev, sigma_y)
    return dz, N2_i, kap, sx, sy


# ---------------------------------------------------------------------------
# the limit that makes the closure recognisable
# ---------------------------------------------------------------------------

def test_strong_stratification_recovers_the_diagnostic_streamfunction():
    """With c^2 Gamma_zz negligible the BVP must give Gamma = kappa * S.

    ``S = -grad_h(rho)/d(rho)/dz`` and ``N^2 = -(g/rho_0) d(rho)/dz``, so
    ``kappa*S = -(g/rho_0) grad_h(rho) kappa / N^2``. Checked in the column
    INTERIOR, away from the boundary layers the BVP necessarily imposes.
    """
    dz, N2_i, kap, sx, sy = _column(nlev=200, H=4000.0, N2=1e-3,
                                    kappa=1000.0, sigma_x=-1e-6)
    cfg = GMBVPConfig(c_min=1e-4, mode_number=3.0)
    Gx, _ = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i), jnp.asarray(kap),
        jnp.asarray(dz), cfg=cfg, c_wave=jnp.asarray(1e-3),
    )
    expected = -(constants.g / constants.rho_ocean) * sx[0] * kap[0] / N2_i[0]
    mid = np.asarray(Gx)[len(N2_i) // 2]
    assert mid == pytest.approx(expected, rel=1e-3), (
        f"interior Gamma {mid:.6e} vs diagnostic kappa*S {expected:.6e}")


def test_the_elliptic_term_bites_when_its_scale_reaches_the_column():
    """The BVP departs from kappa*S only when the vertical scale c/N becomes
    comparable to the column depth -- that IS the physics, and it is what
    makes the closure more than an expensive way to write kappa*S.

    Measured on H = 4000 m, N = 3.16e-3 /s: c/N = 158 m leaves the interior
    maximum untouched, c/N = 949 m cuts it by ~24 %. So the assertion is
    monotone decrease in c, plus a MEASURABLE reduction once the scale is a
    good fraction of the depth -- not a fixed factor, which would only pin
    this particular geometry.
    """
    H = 4000.0
    dz, N2_i, kap, sx, sy = _column(nlev=100, H=H, N2=1e-5)
    args = (jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i),
            jnp.asarray(kap), jnp.asarray(dz))
    peaks = [float(np.max(np.abs(np.asarray(
        solve_gm_streamfunction(*args, c_wave=jnp.asarray(c))[0]))))
        for c in (1e-4, 0.5, 1.0, 3.0)]
    assert all(b <= a + 1e-12 for a, b in zip(peaks, peaks[1:])), peaks

    # c/N ~ 0.04 H: indistinguishable from the diagnostic limit.
    assert peaks[1] == pytest.approx(peaks[0], rel=1e-3)
    # c/N ~ 0.24 H: a real reduction.
    assert peaks[3] < 0.9 * peaks[0], peaks


# ---------------------------------------------------------------------------
# boundary conditions and the conservation they buy
# ---------------------------------------------------------------------------

def test_streamfunction_vanishes_at_both_boundaries():
    dz, N2_i, kap, sx, sy = _column()
    Gx, Gy = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i), jnp.asarray(kap),
        jnp.asarray(dz))
    for G in (np.asarray(Gx), np.asarray(Gy)):
        assert G[0] == 0.0
        assert G[-1] == 0.0


def test_bolus_transport_over_the_column_is_zero():
    """Gamma = 0 at both ends makes sum(u* dz) telescope to exactly zero: the
    parameterisation moves no net volume. This is the property the diagnostic
    form only approximates, and it is why the boundary rows matter."""
    dz, N2_i, kap, sx, sy = _column(sigma_x=-2e-6, sigma_y=1e-6)
    Gx, Gy = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i), jnp.asarray(kap),
        jnp.asarray(dz))
    u, v = bolus_velocity(Gx, Gy, jnp.asarray(dz))
    assert float(np.sum(np.asarray(u) * dz)) == pytest.approx(0.0, abs=1e-18)
    assert float(np.sum(np.asarray(v) * dz)) == pytest.approx(0.0, abs=1e-18)


# ---------------------------------------------------------------------------
# partial-depth columns -- the seafloor is a BOUNDARY, not an interior row
#
# Both cases below are codex's adversarial counterexamples.  The full-depth
# test above cannot catch either: with every interface wet, the last array
# index IS the seafloor, so padding identity onto index ``nlev`` happens to be
# right.  On a partial column it is not, and the all-depth integral stays
# trivially zero while the physical wet-column integral is wrong -- which is
# exactly why these assert on the WET sub-column.
# ---------------------------------------------------------------------------

def test_seafloor_interface_is_a_boundary_row_on_a_partial_column():
    """Before the fix this gave Gamma[2] = 3.83e-2 (not 0), a spurious bolus
    velocity in the first DRY layer, and a wet-column integral of -3.83e-2."""
    nlev = 4
    dz = jnp.full(nlev, 100.0)
    n2 = jnp.full(nlev + 1, 1e-4)
    kap = jnp.full(nlev + 1, 800.0)
    sx = jnp.full(nlev, -1e-6)
    # 3 wet interfaces: 0 (surface), 1 (interior), 2 (SEAFLOOR).
    wet = jnp.array([True, True, True, False, False])
    k_floor = 2

    Gx, Gy = solve_gm_streamfunction(
        sx, jnp.zeros(nlev), n2, kap, dz,
        wet_interface=wet, c_wave=jnp.asarray(1.0))
    u, _ = bolus_velocity(Gx, Gy, dz)

    # The bottom boundary condition, at the SEAFLOOR rather than at the
    # bottom of the array.
    assert float(Gx[k_floor]) == pytest.approx(0.0, abs=1e-15)
    assert float(Gx[0]) == pytest.approx(0.0, abs=1e-15)
    # ... and the interior interface is still doing something.
    assert abs(float(Gx[1])) > 1e-4

    # No bolus velocity in a dry layer.
    assert np.allclose(np.asarray(u)[k_floor:], 0.0, atol=1e-18)

    # The telescoping property must hold over the WATER COLUMN.  Summing over
    # all levels is vacuous here -- it is zero either way.
    wet_layers = slice(0, k_floor)
    assert float(np.sum(np.asarray(u)[wet_layers] * np.asarray(dz)[wet_layers])
                 ) == pytest.approx(0.0, abs=1e-18)


def test_zero_thickness_dry_layers_do_not_produce_nan():
    """A real partial column carries dz = 0 in its dry layers.  The assembly
    divides by dz, so before the denominator floor this produced b = -inf,
    c = +inf and an all-NaN Gamma."""
    nlev = 4
    dz = jnp.array([100.0, 100.0, 0.0, 0.0])
    n2 = jnp.full(nlev + 1, 1e-4)
    kap = jnp.full(nlev + 1, 800.0)
    sx = jnp.array([-1e-6, -1e-6, 0.0, 0.0])
    wet = jnp.array([True, True, True, False, False])
    cfg = GMBVPConfig()

    c_w = baroclinic_wave_speed(n2, dz, cfg)
    a, b, c, d = _assemble(sx, n2, kap, dz, c_w, wet, cfg)
    for name, arr in (("a", a), ("b", b), ("c", c), ("d", d)):
        assert bool(jnp.all(jnp.isfinite(arr))), f"{name} has inf/NaN: {arr}"

    Gx, Gy = solve_gm_streamfunction(
        sx, jnp.zeros(nlev), n2, kap, dz, wet_interface=wet, cfg=cfg)
    assert bool(jnp.all(jnp.isfinite(Gx)))
    assert float(Gx[2]) == pytest.approx(0.0, abs=1e-15)

    u, v = bolus_velocity(Gx, Gy, dz)
    assert bool(jnp.all(jnp.isfinite(u))) and bool(jnp.all(jnp.isfinite(v)))


def test_bolus_velocity_is_the_vertical_derivative():
    dz = np.array([10.0, 20.0, 40.0])
    G = jnp.asarray([0.0, 3.0, 5.0, 0.0])
    u, _ = bolus_velocity(G, G, jnp.asarray(dz))
    np.testing.assert_allclose(np.asarray(u),
                               [(0.0 - 3.0) / 10.0, (3.0 - 5.0) / 20.0,
                                (5.0 - 0.0) / 40.0], rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# the assembly, against a construction that shares no code with it
# ---------------------------------------------------------------------------

def test_assembled_system_matches_a_dense_solve():
    """Build the tridiagonal matrix explicitly from (a, b, c) and solve with
    a dense LU. Agreement pins the Thomas sweep AND the assembly; a swapped
    sub/super-diagonal would pass a residual check on its own system but not
    this one."""
    rng = np.random.default_rng(0)
    nlev = 24
    dz = rng.uniform(5.0, 200.0, nlev)
    N2_i = rng.uniform(1e-6, 1e-3, nlev + 1)
    kap = rng.uniform(100.0, 1500.0, nlev + 1)
    sx = rng.uniform(-2e-6, 2e-6, nlev)
    cfg = GMBVPConfig()
    c_w = baroclinic_wave_speed(jnp.asarray(N2_i), jnp.asarray(dz), cfg)

    a, b, c, d = _assemble(jnp.asarray(sx), jnp.asarray(N2_i),
                           jnp.asarray(kap), jnp.asarray(dz), c_w,
                           jnp.ones(nlev + 1, dtype=bool), cfg)
    a, b, c, d = (np.asarray(x) for x in (a, b, c, d))
    n = nlev + 1
    M = np.zeros((n, n))
    for k in range(n):
        M[k, k] = b[k]
        if k > 0:
            M[k, k - 1] = a[k]
        if k < n - 1:
            M[k, k + 1] = c[k]
    dense = np.linalg.solve(M, d)

    Gx, _ = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(np.zeros(nlev)), jnp.asarray(N2_i),
        jnp.asarray(kap), jnp.asarray(dz), cfg=cfg, c_wave=c_w)
    np.testing.assert_allclose(np.asarray(Gx), dense, rtol=1e-9, atol=1e-14)


def test_boundary_rows_are_identity():
    dz, N2_i, kap, sx, _ = _column(nlev=8)
    a, b, c, d = _assemble(jnp.asarray(sx), jnp.asarray(N2_i),
                           jnp.asarray(kap), jnp.asarray(dz),
                           jnp.asarray(0.5), jnp.ones(9, dtype=bool),
                           GMBVPConfig())
    for k in (0, -1):
        assert float(np.asarray(a)[k]) == 0.0
        assert float(np.asarray(c)[k]) == 0.0
        assert float(np.asarray(b)[k]) == 1.0
        assert float(np.asarray(d)[k]) == 0.0


# ---------------------------------------------------------------------------
# wave speed
# ---------------------------------------------------------------------------

def test_wave_speed_matches_the_wkb_integral():
    """Constant N over depth H gives c = N*H/(pi*m)."""
    nlev, H, N2 = 50, 4000.0, 1e-4
    dz = np.full(nlev, H / nlev)
    N2_i = np.full(nlev + 1, N2)
    cfg = GMBVPConfig(c_min=1e-6, mode_number=3.0)
    got = float(baroclinic_wave_speed(jnp.asarray(N2_i), jnp.asarray(dz), cfg))
    assert got == pytest.approx(np.sqrt(N2) * H / (np.pi * 3.0), rel=1e-12)


def test_wave_speed_is_floored_and_ignores_unstable_stratification():
    nlev = 10
    dz = np.full(nlev, 10.0)
    cfg = GMBVPConfig(c_min=0.1, mode_number=3.0)
    # A statically UNSTABLE column must contribute nothing, not a NaN.
    got = float(baroclinic_wave_speed(jnp.full(nlev + 1, -1e-4),
                                      jnp.asarray(dz), cfg))
    assert got == pytest.approx(0.1)
    assert np.isfinite(got)


def test_wave_speed_matches_the_fesom_default_parameters():
    """FESOM2 namelist.oce ships K_GM_cmin = 0.1 and K_GM_cm = 3.0."""
    cfg = GMBVPConfig()
    assert cfg.c_min == 0.1
    assert cfg.mode_number == 3.0


# ---------------------------------------------------------------------------
# masking, batching, AD
# ---------------------------------------------------------------------------

def test_dry_interfaces_return_zero_and_do_not_poison_wet_ones():
    nlev = 12
    dz = np.full(nlev, 100.0)
    N2_i = np.full(nlev + 1, 1e-4)
    kap = np.full(nlev + 1, 800.0)
    sx = np.full(nlev, -1e-6)
    wet = np.ones(nlev + 1, dtype=bool)
    wet[8:] = False                      # seafloor part-way up the column
    Gx, _ = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(np.zeros(nlev)), jnp.asarray(N2_i),
        jnp.asarray(kap), jnp.asarray(dz), wet_interface=jnp.asarray(wet))
    G = np.asarray(Gx)
    assert np.all(G[8:] == 0.0)
    assert np.all(np.isfinite(G))
    assert np.any(np.abs(G[1:8]) > 0.0)


def test_batched_columns_match_one_at_a_time():
    rng = np.random.default_rng(1)
    ncol, nlev = 5, 16
    dz = rng.uniform(10.0, 100.0, (ncol, nlev))
    N2_i = rng.uniform(1e-6, 1e-3, (ncol, nlev + 1))
    kap = rng.uniform(100.0, 1000.0, (ncol, nlev + 1))
    sx = rng.uniform(-1e-6, 1e-6, (ncol, nlev))
    sy = rng.uniform(-1e-6, 1e-6, (ncol, nlev))
    Gx, _ = solve_gm_streamfunction(jnp.asarray(sx), jnp.asarray(sy),
                                    jnp.asarray(N2_i), jnp.asarray(kap),
                                    jnp.asarray(dz))
    for k in range(ncol):
        one, _ = solve_gm_streamfunction(
            jnp.asarray(sx[k]), jnp.asarray(sy[k]), jnp.asarray(N2_i[k]),
            jnp.asarray(kap[k]), jnp.asarray(dz[k]))
        np.testing.assert_allclose(np.asarray(Gx)[k], np.asarray(one),
                                   rtol=1e-10, atol=1e-16)


def test_jit_parity_and_finite_gradients():
    dz, N2_i, kap, sx, sy = _column(nlev=20)
    args = (jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i),
            jnp.asarray(kap), jnp.asarray(dz))
    eager, _ = solve_gm_streamfunction(*args)
    jitted, _ = jax.jit(solve_gm_streamfunction)(*args)
    # Not bit-identical: XLA is free to fuse and reassociate the Thomas sweep
    # under jit, and the measured spread is ~1e-15 relative. Requiring bitwise
    # equality here would be testing the compiler, not the solver.
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                               rtol=1e-12, atol=1e-18)

    def loss(kappa):
        Gx, _ = solve_gm_streamfunction(
            args[0], args[1], args[2], kappa, args[4])
        return jnp.sum(Gx ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(kap)))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_sign_flattens_the_density_surface():
    """A density surface tilted one way must drive a bolus flow of the sign
    that flattens it. Reversing grad_h(rho) must reverse the transport."""
    dz, N2_i, kap, sx, sy = _column(nlev=40, sigma_x=-1e-6)
    pos, _ = solve_gm_streamfunction(
        jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(N2_i), jnp.asarray(kap),
        jnp.asarray(dz))
    neg, _ = solve_gm_streamfunction(
        jnp.asarray(-sx), jnp.asarray(sy), jnp.asarray(N2_i),
        jnp.asarray(kap), jnp.asarray(dz))
    np.testing.assert_allclose(np.asarray(neg), -np.asarray(pos),
                               rtol=1e-12, atol=1e-18)
    # And the diagnostic limit fixes the sign: Gamma has the sign of
    # -grad_h(rho), so sigma_x < 0 gives Gamma_x > 0.
    assert np.max(np.asarray(pos)) > 0.0
