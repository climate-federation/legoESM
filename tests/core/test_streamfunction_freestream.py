"""Free-stream / GCL preservation of the streamfunction transport (issue 504).

The cube cosine-bell fragmented because the contravariant d2a2c flux chain is
not geometric-conservation-law consistent at the cube face-boundary seam: a
non-divergent wind acquired spurious area-flux divergence (27x at vertices),
which a constant field exposes (h=1 -> max|h-1|=0.52 over 2 days).

The fix builds the mass flux as the exact discrete curl of a corner
streamfunction, so the divergence telescopes to machine zero for ANY psi.
These tests pin that invariant.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
import jax

from legoesm.core.fv_tp_2d import (
    streamfunction_mass_fluxes, streamfunction_transport_step)
from tests.test_cases.cosine_bell import (
    rotation_streamfunction, _rotation_winds_geo,
    cosine_bell_cubesphere, cosine_bell_exact, cosine_bell_error_norms)


@pytest.fixture(scope="module")
def cube():
    grid = create_cubed_sphere(24)
    return grid, create_cubed_sphere_cdgrid(grid)


def test_massflux_divergence_is_machine_zero(cube):
    """xfx[i]-xfx[i+1] + yfx[j]-yfx[j+1] telescopes to ~0 for ANY psi."""
    grid, cd = cube
    # An arbitrary smooth streamfunction at corners (not the bell one).
    # The divergence telescopes to machine zero in EXACT arithmetic; build psi
    # in float64 so the test sees the true GCL property (the cdgrid metric is
    # float32 by default, which would only show fp32-eps roundoff ~1e-7).
    lonc = np.asarray(cd.lon_corner, dtype=np.float64)
    latc = np.asarray(cd.lat_corner, dtype=np.float64)
    psi = jnp.asarray(np.sin(3 * lonc) * np.cos(2 * latc) + 0.5 * latc,
                      dtype=jnp.float64)
    crx, cry, xfx, yfx, ra_x, ra_y = streamfunction_mass_fluxes(cd, psi, 300.0)
    div = (xfx[:, :-1, :] - xfx[:, 1:, :]
           + yfx[:, :, :-1] - yfx[:, :, 1:])
    scale = float(jnp.max(jnp.abs(xfx)))
    # Telescoping is exact in exact arithmetic; the residual is pure roundoff,
    # so the tolerance must track the WORKING precision (the cdgrid metric is
    # float32 by default and JAX truncates psi to float32 when x64 is off ->
    # ~1e-7 relative, not 1e-12).  Tie the bound to eps of the actual dtype so
    # the test is correct under both x64 on (fp64) and off (fp32).
    eps = float(np.finfo(np.asarray(xfx).dtype).eps)
    assert float(jnp.max(jnp.abs(div))) / scale < 50.0 * eps, "GCL telescoping broken"


def test_flux_sign_convention(cube):
    """Direct sign pin (codex 506b review): for psi increasing in +j, the
    u-face flux xfx = dt*(psi[i,j]-psi[i,j+1]) must be NEGATIVE, and for psi
    increasing in +i the v-face flux yfx = dt*(psi[i+1,j]-psi[i,j]) must be
    POSITIVE.  A reversed convention preserves free-stream (telescopes) but
    advects backwards, so the divergence tests alone cannot catch it."""
    grid, cd = cube
    n = grid.lon.shape[1]
    dt = 300.0
    # psi = j  (monotone increasing along axis 2) -> xfx < 0 everywhere.
    psi_j = jnp.broadcast_to(jnp.arange(n + 1, dtype=jnp.float64)[None, None, :],
                             (6, n + 1, n + 1))
    _, _, xfx_j, yfx_j, _, _ = streamfunction_mass_fluxes(cd, psi_j, dt)
    assert float(jnp.max(xfx_j)) < 0.0, "xfx sign wrong for psi increasing in +j"
    assert float(jnp.max(jnp.abs(yfx_j))) == 0.0, "yfx must vanish for psi=psi(j)"
    # psi = i  (monotone increasing along axis 1) -> yfx > 0 everywhere.
    psi_i = jnp.broadcast_to(jnp.arange(n + 1, dtype=jnp.float64)[None, :, None],
                             (6, n + 1, n + 1))
    _, _, xfx_i, yfx_i, _, _ = streamfunction_mass_fluxes(cd, psi_i, dt)
    assert float(jnp.min(yfx_i)) > 0.0, "yfx sign wrong for psi increasing in +i"
    assert float(jnp.max(jnp.abs(xfx_i))) == 0.0, "xfx must vanish for psi=psi(i)"


def test_constant_field_is_preserved(cube):
    """Free-stream: a constant h stays constant under the transport step
    (was max|h-1|~0.5 with the old d2a2c flux)."""
    grid, cd = cube
    n = grid.lon.shape[1]
    psi = rotation_streamfunction(cd.lon_corner, cd.lat_corner,
                                  grid.radius, math.pi / 4)
    fluxes = streamfunction_mass_fluxes(cd, psi, 300.0)
    h = jnp.ones((6, n, n))
    for _ in range(50):
        h = streamfunction_transport_step(h, fluxes, cd, mass_target=None,
                                          hord=10)
    assert float(jnp.max(jnp.abs(h - 1.0))) < 1e-6, "free-stream not preserved"


def test_freestream_bounded_over_many_revolutions(cube):
    """LONG-RUN edge-artifact guard (issue 521 / FV3 cube faithfulness).

    The divergence-free curl flux must preserve free-stream not just for 50
    substeps but for arbitrarily long integrations: any slow seam/corner GCL
    leak would accumulate over revolutions.  Advect h=1 for 2 full 12-day
    revolutions and assert it stays flat.  Measured behaviour is a constant
    ~1.5e-6 with NO growth rev-over-rev (C48 diag: 1.55e-6 flat to 120 days);
    the pre-fix d2a2c flux gave max|h-1|~0.5 in 2 days."""
    grid, cd = cube
    n = grid.lon.shape[1]
    dt, n_sub = 1800.0, 6
    psi = rotation_streamfunction(cd.lon_corner, cd.lat_corner,
                                  grid.radius, math.pi / 4)
    fluxes = streamfunction_mass_fluxes(cd, psi, dt / n_sub)

    @jax.jit
    def day_block(h):  # one day = 48 dt-steps * 6 substeps
        def body(c, _):
            return streamfunction_transport_step(
                c, fluxes, cd, mass_target=None, hord=10), None
        out, _ = jax.lax.scan(body, h, None, length=48 * n_sub)
        return out

    h = jnp.ones((6, n, n))
    worst = 0.0
    for _ in range(24):                      # 24 days = 2 revolutions
        h = day_block(h)
        worst = max(worst, float(jnp.max(jnp.abs(h - 1.0))))
    # bound well below any artifact (pre-fix 0.5) yet above fp32 roundoff floor
    assert worst < 1e-5, f"free-stream drifted over 2 revolutions: max|h-1|={worst}"


def test_streamfunction_curl_matches_geographic_wind(cube):
    """The analytic curl of ``rotation_streamfunction`` reproduces
    ``_rotation_winds_geo`` (so psi and the wind are the same flow)."""
    grid, cd = cube
    R = float(grid.radius)
    lat = np.deg2rad(np.linspace(-60, 60, 25))
    lon = np.deg2rad(np.linspace(10, 350, 25))
    lo, la = np.meshgrid(lon, lat)
    beta = math.pi / 4
    # Central-difference step: 2e-4 keeps truncation error ~eps^2 negligible
    # while staying well above float32 roundoff of the large psi (~1e8), so the
    # check is robust whether x64 is on or off.
    eps = 2e-4
    psi = lambda LO, LA: np.asarray(
        rotation_streamfunction(jnp.asarray(LO), jnp.asarray(LA), R, beta),
        dtype=np.float64)
    # u_east = -1/R dpsi/dlat ; v_north = 1/(R cos lat) dpsi/dlon
    u_fd = -(psi(lo, la + eps) - psi(lo, la - eps)) / (2 * eps) / R
    v_fd = (psi(lo + eps, la) - psi(lo - eps, la)) / (2 * eps) / (R * np.cos(la))
    u_an, v_an = _rotation_winds_geo(jnp.asarray(lo), jnp.asarray(la), R, beta)
    np.testing.assert_allclose(u_fd, np.asarray(u_an, dtype=np.float64), rtol=5e-3, atol=0.1)
    np.testing.assert_allclose(v_fd, np.asarray(v_an, dtype=np.float64), rtol=5e-3, atol=0.1)


def test_short_time_trajectory_matches_exact(cube):
    """Advecting the bell ~1 day must track the exact solution (L2 small).

    A flux SIGN error preserves free-stream and the full-revolution shape but
    advects the bell BACKWARDS -> ~zero overlap at intermediate times
    (L1=2.0, L2=1.4).  This integration check pins the flux orientation so
    that regression cannot recur."""
    grid, cd = cube
    beta = math.pi / 4
    R = grid.radius
    state = cosine_bell_cubesphere(grid, cd, beta=beta)
    h = state.h
    mass = float(jnp.sum(h * grid.area))
    n_sub = 6
    dt = 1800.0
    psi = rotation_streamfunction(cd.lon_corner, cd.lat_corner, R, beta)
    fluxes = streamfunction_mass_fluxes(cd, psi, dt / n_sub)

    @jax.jit
    def outer(hh):
        def body(c, _):
            return streamfunction_transport_step(
                c, fluxes, cd, mass_target=mass, hord=10), None
        o, _ = jax.lax.scan(body, hh, None, length=n_sub)
        return o

    t_end = 1.0 * 86400.0
    for _ in range(int(round(t_end / dt))):
        h = outer(h)
    h_exact = cosine_bell_exact(grid.lon, grid.lat, R, t_end, beta)
    norms = cosine_bell_error_norms(h, h_exact, grid.area)
    # Correct orientation -> ~0.06 at C24; a reversed flux gives ~1.4.
    assert norms["l2"] < 0.3, f"bell off-trajectory (flux sign?): L2={norms['l2']}"
