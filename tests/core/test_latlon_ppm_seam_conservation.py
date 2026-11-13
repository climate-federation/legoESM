"""Lat-lon limited PPM transport conserves mass at the periodic seam.

The duplicated longitude face (index 0 and n_lon) must carry bit-identical
upwind states, otherwise the flux divergence does not telescope and the
global integral of the tendency is not zero.  With a halo=2 pad the CW84
limiter on the outermost halo cell read a 2nd-order edge, so the two copies
differed wherever the limiter was active (measured: up to 1e-9 relative mass
loss per step in the lat-lon shallow-water model without the mass fixer).

Fields are sharp across the seam so the limiter is active there (checked by
the ``limiter_active`` control), with zonal winds of both signs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm import shallow_water_latlon_cgrid as ll
from legoesm.core.operators_fv_latlon import (
    _ppm_reconstruct_lon,
    cgrid_fv_flux_divergence_latlon,
    fv_flux_divergence_latlon,
)
from legoesm.core.operators_fv_latlon_3d import (
    cgrid_fv_flux_divergence_latlon_3d,
    fv_flux_divergence_latlon_3d,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.grids.latlon import create_latlon_grid

CLOSURE_RTOL = 1e-13
N_LAT, N_LON = 16, 32


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def grid(_x64):
    return create_latlon_grid(N_LAT, N_LON)


def _sharp(grid, shift=0):
    """Step in longitude straddling the seam plus a lat step: limiter active."""
    i = (np.arange(N_LON) + shift) % N_LON
    zonal = np.where((i < 3) | (i >= N_LON - 2), 2000.0, 1000.0)
    merid = np.where(np.arange(N_LAT) < N_LAT // 2, 0.0, 300.0)
    return jnp.asarray(zonal[None, :] + merid[:, None] + 7.0 * np.sin(3.0 * i)[None, :])


def _winds(sign):
    rng = np.random.default_rng(0)
    u = sign * (5.0 + 10.0 * rng.random((N_LAT, N_LON)))
    u_face = np.concatenate([u, u[:, :1]], axis=1)
    v_face = 10.0 * rng.standard_normal((N_LAT + 1, N_LON))
    v_face[0] = v_face[-1] = 0.0
    return u, jnp.asarray(u_face), jnp.asarray(v_face)


def _closure(dq, area):
    terms = dq * area
    return abs(float(jnp.sum(terms))) / float(jnp.sum(jnp.abs(terms)))


def test_limiter_active_at_seam(grid):
    """Control: the limiter changes the seam face states of the test field."""
    qp = pad_halo_latlon(_sharp(grid), halo=3)
    lim = _ppm_reconstruct_lon(qp, True)
    raw = _ppm_reconstruct_lon(qp, False)
    for a, b in zip(lim, raw):
        assert float(jnp.max(jnp.abs(a[:, 0] - b[:, 0]))) > 1.0
        assert float(jnp.max(jnp.abs(a[:, -1] - b[:, -1]))) > 1.0


@pytest.mark.parametrize("shift", [0, 1, N_LON - 1])
def test_seam_face_states_identical(grid, shift):
    q_l, q_r = _ppm_reconstruct_lon(pad_halo_latlon(_sharp(grid, shift), halo=3), True)
    np.testing.assert_array_equal(np.asarray(q_l[:, 0]), np.asarray(q_l[:, -1]))
    np.testing.assert_array_equal(np.asarray(q_r[:, 0]), np.asarray(q_r[:, -1]))


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("shift", [0, 1, N_LON - 1])
def test_2d_closure(grid, sign, shift):
    q = _sharp(grid, shift)
    u_c, u_face, v_face = _winds(sign)
    dq = cgrid_fv_flux_divergence_latlon(q, u_face, v_face, grid)
    assert _closure(dq, grid.area) <= CLOSURE_RTOL
    # A-grid twin (cell-centred winds; polar-row v zero so no polar flux).
    v_c = jnp.asarray(np.asarray(v_face[1:]))
    v_c = v_c.at[0].set(0.0).at[-1].set(0.0)
    dq_a = fv_flux_divergence_latlon(q, jnp.asarray(u_c), v_c, grid)
    assert _closure(dq_a, grid.area) <= CLOSURE_RTOL


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_3d_closure_per_level(grid, sign):
    q = jnp.stack([_sharp(grid, s) for s in (0, 1, N_LON - 1)], axis=-1)
    u_c, u_face, v_face = _winds(sign)
    nlev = q.shape[-1]
    u_face3 = jnp.repeat(u_face[..., None], nlev, axis=-1)
    v_face3 = jnp.repeat(v_face[..., None], nlev, axis=-1)
    dq = cgrid_fv_flux_divergence_latlon_3d(q, u_face3, v_face3, grid)
    v_c = jnp.asarray(np.asarray(v_face[1:])).at[0].set(0.0).at[-1].set(0.0)
    dq_a = fv_flux_divergence_latlon_3d(
        q, jnp.repeat(jnp.asarray(u_c)[..., None], nlev, -1),
        jnp.repeat(v_c[..., None], nlev, -1), grid)
    for k in range(nlev):
        assert _closure(dq[..., k], grid.area) <= CLOSURE_RTOL, k
        assert _closure(dq_a[..., k], grid.area) <= CLOSURE_RTOL, k


def test_shallow_water_no_fixer_mass(grid):
    """10 steps of the PPM shallow-water model without the mass fixer."""
    state = ll.williamson_test2_cgrid(grid)
    lat, lon = grid.lat[:, None], grid.lon[None, :]
    du = 3.0 * jnp.cos(lat) * jnp.sin(lon - 0.5 * grid.dlon)
    state = state._replace(
        h=state.h + 40.0 * jnp.cos(lat) ** 2 * jnp.cos(2 * lon),
        u=state.u + jnp.concatenate([du, du[:, :1]], axis=1),
        v=(state.v + 2.0 * jnp.cos(grid.lat_v[:, None]) * jnp.sin(3 * lon))
        .at[0].set(0.0).at[-1].set(0.0))
    cfg = ll.CGridLatLonShallowWaterConfig(fix_mass=False)
    model = ll.CGridLatLonShallowWaterModel(grid, cfg, dt=300.0)
    mass = lambda s: float(jnp.sum(s.h * grid.area))  # noqa: E731
    m0, s = mass(state), state
    for _ in range(10):
        s = model.step(s, 300.0)
    assert abs(mass(s) - m0) <= CLOSURE_RTOL * m0


def test_mpi_step_refuses_bands_thinner_than_halo():
    """Every rank refuses a thin smallest band before any halo exchange."""
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout, make_latlon_mpi_step

    for rank in range(4):  # 10 rows on 4 ranks -> bands (3, 3, 2, 2)
        layout = make_latlon_band_layout(rank, 4, 10, 20)
        with pytest.raises(ValueError, match="halo=3"):
            make_latlon_mpi_step(None, layout)


def test_refuses_fewer_than_three_lat_rows(_x64):
    g = create_latlon_grid(2, 8)
    q2 = jnp.ones((2, 8))
    u_face = jnp.ones((2, 9))
    v_face = jnp.zeros((3, 8))
    fns = [
        lambda: cgrid_fv_flux_divergence_latlon(q2, u_face, v_face, g),
        lambda: fv_flux_divergence_latlon(q2, jnp.ones((2, 8)), jnp.zeros((2, 8)), g),
        lambda: cgrid_fv_flux_divergence_latlon_3d(
            q2[..., None], u_face[..., None], v_face[..., None], g),
        lambda: fv_flux_divergence_latlon_3d(
            q2[..., None], jnp.ones((2, 8, 1)), jnp.zeros((2, 8, 1)), g),
    ]
    for fn in fns:
        with pytest.raises(ValueError, match="n_lat >= 3"):
            fn()
